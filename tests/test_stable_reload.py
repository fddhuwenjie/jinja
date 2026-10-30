"""Tests for the ``stable_reload`` option of :class:`~jinja2.Environment`.

With ``stable_reload=True`` a single render output pins every template to
the version first resolved during that output. Independent outputs still
pick up new valid versions, interleaved outputs do not influence each
other, and an invalid update of a template that was loaded successfully
before keeps the last valid version in use until the source is fixed.
"""

import asyncio
import typing as t

import pytest

from jinja2 import DictLoader
from jinja2 import Environment
from jinja2 import TemplateNotFound
from jinja2 import TemplateSyntaxError
from jinja2 import UndefinedError
from jinja2.environment import _stable_render_pins

TWICE_MAIN = "{% include 'inc' %}|{{ step() }}|{% include 'inc' %}"
PARK_MAIN = "{% include 'inc' %}|{{ park() }}|{% include 'inc' %}"
BASE_TEMPLATE = "BASE-{{ self.content() }}|{% block content %}default{% endblock %}"
CHILD_TEMPLATE = "{% extends 'base' %}{% block content %}CHILD{% endblock %}"
IMPORT_MAIN = (
    "{% from 'lib' import value %}{{ value }}|{{ step() }}|"
    "{% from 'lib' import value %}{{ value }}"
)
BROKEN_SOURCE = "{% for oops in broken unclosed"


def make_env(
    stable: bool,
    *,
    enable_async: bool = False,
    cache_size: int = 400,
    templates: dict[str, str] | None = None,
) -> tuple[Environment, dict[str, str]]:
    mapping = dict(
        templates if templates is not None else {"main": TWICE_MAIN, "inc": "V1"}
    )
    env = Environment(
        loader=DictLoader(mapping),
        auto_reload=True,
        stable_reload=stable,
        enable_async=enable_async,
        cache_size=cache_size,
    )
    return env, mapping


def install_step(
    env: Environment,
    mapping: dict[str, str],
    updates: dict[str, str],
) -> t.Callable[[], str]:
    """Register a ``step`` global that changes template sources."""

    def step() -> str:
        mapping.update(updates)
        return ""

    env.globals["step"] = step
    return step


def install_park(env: Environment, state: dict[str, bool]) -> t.Callable[[], str]:
    """Register a ``park`` global that flags that it was called."""

    def park() -> str:
        state["parked"] = True
        return ""

    env.globals["park"] = park
    return park


# ---------------------------------------------------------------------------
# Synchronous rendering
# ---------------------------------------------------------------------------


def test_render_pins_template_for_one_output() -> None:
    env, mapping = make_env(True)
    install_step(env, mapping, {"inc": "V2"})
    template = env.get_template("main")

    assert template.render() == "V1||V1"


def test_generate_and_stream_pin_template_for_one_output() -> None:
    def drain(use_stream: bool) -> str:
        env, mapping = make_env(True)
        install_step(env, mapping, {"inc": "V2"})
        output = env.get_template("main")
        return "".join(output.stream() if use_stream else output.generate())

    assert drain(use_stream=False) == "V1||V1"
    assert drain(use_stream=True) == "V1||V1"


def test_next_output_picks_up_new_version() -> None:
    env, mapping = make_env(True)
    install_step(env, mapping, {"inc": "V2"})
    template = env.get_template("main")

    assert template.render() == "V1||V1"
    # A fresh output starts without pins and sees the valid new version.
    assert env.get_template("main").render() == "V2||V2"


def test_interleaved_streams_are_independent() -> None:
    state = {"parked": False}
    env, mapping = make_env(True, templates={"main": PARK_MAIN, "inc": "V1"})
    install_park(env, state)

    first_stream = env.get_template("main").stream()
    first: list[str] = []

    for event in first_stream:
        first.append(event)

        # Stop right after the park point; the second include has not
        # resolved the template yet.
        if state["parked"]:
            break

    mapping["inc"] = "V2"

    second = "".join(env.get_template("main").stream())
    first.extend(first_stream)

    assert "".join(first) == "V1||V1"
    assert second == "V2||V2"


def test_invalid_update_falls_back_to_last_loaded_version() -> None:
    mapping = {"main": "{% include 'inc' %}", "inc": "V1"}
    env, mapping = make_env(True, templates=mapping)
    template = env.get_template("main")

    assert template.render() == "V1"

    mapping["inc"] = BROKEN_SOURCE

    # Ongoing and new outputs keep using the last valid version.
    assert template.render() == "V1"
    fallback = env.get_template("inc")
    assert fallback.render() == "V1"


def test_fixed_source_is_adopted_automatically() -> None:
    mapping = {"main": "{% include 'inc' %}", "inc": "V1"}
    env, mapping = make_env(True, templates=mapping)
    template = env.get_template("main")

    assert template.render() == "V1"

    mapping["inc"] = BROKEN_SOURCE
    assert template.render() == "V1"

    mapping["inc"] = "V2"
    assert env.get_template("main").render() == "V2"


def test_removed_template_still_raises_template_not_found() -> None:
    mapping = {"main": "{% include 'inc' %}", "inc": "V1"}
    env, mapping = make_env(True, templates=mapping)
    template = env.get_template("main")

    assert template.render() == "V1"

    # Deleting a template is not an invalid update: the missing template
    # error must propagate instead of silently serving stale content.
    del mapping["inc"]

    with pytest.raises(TemplateNotFound):
        env.get_template("main").render()


def test_never_loaded_invalid_template_still_raises() -> None:
    env, _ = make_env(True, templates={"bad": BROKEN_SOURCE})

    with pytest.raises(TemplateSyntaxError):
        env.get_template("bad")

    main = make_env(
        True,
        templates={"main": "{% include 'bad' %}", "bad": BROKEN_SOURCE},
    )[0]

    with pytest.raises(TemplateSyntaxError):
        main.get_template("main").render()


def test_inheritance_pins_base_and_recovers_from_broken_update() -> None:
    mapping = {"main": CHILD_TEMPLATE, "base": BASE_TEMPLATE}
    env, mapping = make_env(True, templates=mapping)
    template = env.get_template("main")

    assert template.render() == "BASE-CHILD|CHILD"

    mapping["base"] = BASE_TEMPLATE.replace("BASE", "NEW")
    assert env.get_template("main").render() == "NEW-CHILD|CHILD"

    mapping["base"] = BROKEN_SOURCE
    assert env.get_template("main").render() == "NEW-CHILD|CHILD"

    mapping["base"] = BASE_TEMPLATE.replace("BASE", "FIXED")
    assert env.get_template("main").render() == "FIXED-CHILD|CHILD"


def test_interleaved_inheritance_streams_use_own_base_version() -> None:
    state = {"parked": False}
    mapping = {
        "main": "{% extends 'base' %}{% block content %}CHILD{% endblock %}",
        # The parent renders the block output and only reaches the park
        # point afterwards, after the parent itself has been resolved.
        "base": "{{ self.content() }}|{{ park() }}",
    }
    env, mapping = make_env(True, templates=mapping)
    install_park(env, state)

    first_stream = env.get_template("main").stream()
    first: list[str] = []

    for event in first_stream:
        first.append(event)

        if state["parked"]:
            break

    mapping["base"] = "v2:{{ self.content() }}|{{ park() }}"
    second = "".join(env.get_template("main").stream())
    first.extend(first_stream)

    assert "".join(first) == "CHILD|"
    assert second == "v2:CHILD|"


def test_from_import_pins_template_for_one_output() -> None:
    env, mapping = make_env(
        True,
        templates={"main": IMPORT_MAIN, "lib": "{% set value = 'L1' %}"},
    )
    install_step(env, mapping, {"lib": "{% set value = 'L2' %}"})

    assert env.get_template("main").render() == "L1||L1"
    assert env.get_template("main").render() == "L2||L2"


def test_closing_generator_early_removes_pins() -> None:
    state = {"parked": False}
    env, mapping = make_env(True, templates={"main": PARK_MAIN, "inc": "V1"})
    install_park(env, state)

    # ``stream`` drives this same generator when iterated.
    generator = env.get_template("main").generate()

    for _event in generator:
        if state["parked"]:
            break

    # Closing the generator early (as a WSGI server does when a client
    # disconnects) must release the output's pin map.
    generator.close()

    assert _stable_render_pins.get() is None

    mapping["inc"] = "V2"
    assert env.get_template("main").render() == "V2||V2"


def test_render_error_removes_pins() -> None:
    env, _ = make_env(True, templates={"main": "{{ undefined_var.attr }}"})
    template = env.get_template("main")

    with pytest.raises(UndefinedError):
        template.render()

    assert _stable_render_pins.get() is None


def test_cache_size_zero_still_pins_within_output() -> None:
    env, mapping = make_env(True, cache_size=0)
    install_step(env, mapping, {"inc": "V2"})

    assert env.get_template("main").render() == "V1||V1"


def test_cache_size_zero_broken_update_has_no_last_version() -> None:
    mapping = {"main": "{% include 'inc' %}", "inc": "V1"}
    env, mapping = make_env(True, cache_size=0, templates=mapping)

    assert env.get_template("main").render() == "V1"

    mapping["inc"] = BROKEN_SOURCE

    with pytest.raises(TemplateSyntaxError):
        env.get_template("main").render()


# ---------------------------------------------------------------------------
# Disabled mode keeps the historical behavior
# ---------------------------------------------------------------------------


def test_stable_reload_defaults_to_false() -> None:
    env = Environment(loader=DictLoader({}))

    assert env.stable_reload is False


def test_disabled_mode_reloads_template_mid_output() -> None:
    env, mapping = make_env(False)
    install_step(env, mapping, {"inc": "V2"})

    assert env.get_template("main").render() == "V1||V2"


def test_disabled_mode_invalid_update_raises() -> None:
    mapping = {"main": "{% include 'inc' %}", "inc": "V1"}
    env, mapping = make_env(False, templates=mapping)
    template = env.get_template("main")

    assert template.render() == "V1"

    mapping["inc"] = BROKEN_SOURCE

    with pytest.raises(TemplateSyntaxError):
        env.get_template("main").render()


def test_overlay_inherits_and_overrides_stable_reload() -> None:
    env = Environment(loader=DictLoader({}), stable_reload=True)

    assert env.overlay().stable_reload is True
    assert env.overlay(stable_reload=False).stable_reload is False

    disabled = Environment(loader=DictLoader({}))
    assert disabled.overlay(stable_reload=True).stable_reload is True


# ---------------------------------------------------------------------------
# Asynchronous rendering
# ---------------------------------------------------------------------------


async def _async_render_pins_and_reloads() -> None:
    env, mapping = make_env(True, enable_async=True)
    install_step(env, mapping, {"inc": "V2"})
    template = env.get_template("main")

    assert await template.render_async() == "V1||V1"

    chunks = [chunk async for chunk in template.generate_async()]
    assert "".join(chunks) == "V2||V2"


def test_async_render_pins_and_reloads(run_async_fn: t.Any) -> None:
    run_async_fn(_async_render_pins_and_reloads)


def test_async_env_sync_generate_keeps_pins() -> None:
    # An async environment's synchronous ``generate``/``stream`` collects
    # the async output through ``asyncio.run``; pins must stay active for
    # that nested run and be cleared afterwards. This has to execute
    # outside of a running event loop.
    env, mapping = make_env(True, enable_async=True)
    install_step(env, mapping, {"inc": "V2"})
    template = env.get_template("main")

    assert "".join(template.generate()) == "V1||V1"
    assert _stable_render_pins.get() is None


async def _async_invalid_update_fallback() -> None:
    mapping = {"main": "{% include 'inc' %}", "inc": "V1"}
    env, mapping = make_env(True, enable_async=True, templates=mapping)
    template = env.get_template("main")

    assert await template.render_async() == "V1"

    mapping["inc"] = BROKEN_SOURCE
    assert await env.get_template("main").render_async() == "V1"

    mapping["inc"] = "V2"
    assert await env.get_template("main").render_async() == "V2"


def test_async_invalid_update_fallback(run_async_fn: t.Any) -> None:
    run_async_fn(_async_invalid_update_fallback)


async def _async_never_loaded_invalid_raises() -> None:
    env, _ = make_env(True, enable_async=True, templates={"bad": BROKEN_SOURCE})

    with pytest.raises(TemplateSyntaxError):
        await env.get_template("bad").render_async()


def test_async_never_loaded_invalid_raises(run_async_fn: t.Any) -> None:
    run_async_fn(_async_never_loaded_invalid_raises)


async def _async_inheritance_and_import() -> None:
    mapping = {
        "main": CHILD_TEMPLATE,
        "base": BASE_TEMPLATE,
        "imp": IMPORT_MAIN,
        "lib": "{% set value = 'L1' %}",
    }
    env, mapping = make_env(True, enable_async=True, templates=mapping)
    install_step(env, mapping, {"lib": "{% set value = 'L2' %}"})

    assert await env.get_template("main").render_async() == "BASE-CHILD|CHILD"
    assert await env.get_template("imp").render_async() == "L1||L1"
    assert await env.get_template("imp").render_async() == "L2||L2"


def test_async_inheritance_and_import(run_async_fn: t.Any) -> None:
    run_async_fn(_async_inheritance_and_import)


async def _async_early_aclose_removes_pins() -> None:
    state = {"parked": False}

    async def park() -> str:
        state["parked"] = True
        return ""

    env, mapping = make_env(
        True,
        enable_async=True,
        templates={"main": PARK_MAIN, "inc": "V1"},
    )
    env.globals["park"] = park
    template = env.get_template("main")

    generator = template.generate_async()

    while not state["parked"]:
        await generator.__anext__()

    await generator.aclose()

    assert _stable_render_pins.get() is None

    mapping["inc"] = "V2"
    assert await env.get_template("main").render_async() == "V2||V2"


def test_async_early_aclose_removes_pins(run_async_fn: t.Any) -> None:
    run_async_fn(_async_early_aclose_removes_pins)


async def _async_interleaved_outputs() -> None:
    mapping = {"main": PARK_MAIN, "inc": "W1"}
    env = Environment(
        loader=DictLoader(mapping),
        auto_reload=True,
        stable_reload=True,
        enable_async=True,
    )

    first_parked = asyncio.Event()
    release_first = asyncio.Event()
    second_parked = asyncio.Event()

    async def park_first() -> str:
        first_parked.set()
        await release_first.wait()
        return ""

    async def park_second() -> str:
        second_parked.set()
        await release_first.wait()
        return ""

    env.globals["park"] = park_first
    first_task = asyncio.create_task(env.get_template("main").render_async())

    await first_parked.wait()

    mapping["inc"] = "W2"
    env.globals["park"] = park_second
    second_task = asyncio.create_task(env.get_template("main").render_async())

    # Let the second output resolve its first include with "W2" and park.
    while not second_parked.is_set():
        await asyncio.sleep(0)

    release_first.set()

    assert await first_task == "W1||W1"
    assert await second_task == "W2||W2"
    assert _stable_render_pins.get() is None


def test_async_interleaved_outputs_are_independent() -> None:
    asyncio.run(_async_interleaved_outputs())
