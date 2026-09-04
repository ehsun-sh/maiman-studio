"""The component registry: names to classes.

A project file names the components it contains. Resolving those names by
importing a dotted path out of the file would make opening someone else's
project equivalent to running their code — importing a module executes it. So
the file is only ever allowed to *look up* a name that is already registered,
and a name that is not registered is an error telling the user which package to
install and import.

Registration happens automatically when a :class:`~maiman.component.Component`
subclass is defined, so a third-party component is available as soon as its
package is imported.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .component import Component

_REGISTRY: dict[str, type[Component]] = {}


class UnknownComponentError(LookupError):
    """A project file names a component that nothing has registered."""


def register(component_class: type[Component]) -> None:
    """Register a component class under its type name.

    Re-registering the identical class is allowed, so reimporting a module is
    harmless. Two *different* classes claiming one name is an error: it would
    make a project file ambiguous, and silently picking one would mean the same
    file simulates differently depending on import order.
    """
    name = component_class.type_name()
    existing = _REGISTRY.get(name)
    if existing is not None and existing is not component_class:
        raise ValueError(
            f"component name {name!r} is already registered to "
            f"{existing.__module__}.{existing.__qualname__}; set a distinct "
            f"`registry_name` on {component_class.__module__}."
            f"{component_class.__qualname__}"
        )
    _REGISTRY[name] = component_class


def lookup(name: str) -> type[Component]:
    """The component class registered under ``name``."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise UnknownComponentError(
            f"no component registered as {name!r}. If it comes from a plugin "
            f"package, import that package before loading the project. "
            f"Registered: {sorted(_REGISTRY)}"
        ) from None


def registered_names() -> tuple[str, ...]:
    """Every registered component name, sorted."""
    return tuple(sorted(_REGISTRY))


def manifests() -> dict[str, dict[str, Any]]:
    """Generated manifests for every registered component.

    This is the component palette the GUI is built from — derived from the
    classes themselves, so it cannot drift away from what the engine actually
    does.

    The values are ``Any`` rather than ``object`` because a manifest is a nested
    structure and every real caller reaches into it: ``["ports"]["inputs"]`` is
    the whole point of having one. Annotating the outside as ``object`` did not
    make anything safer, it only meant nobody could read a manifest without a
    cast — which is a worse guarantee than none, because the cast is where the
    real mistakes get made.
    """
    return {name: cls.manifest() for name, cls in sorted(_REGISTRY.items())}
