"""Component base class, typed ports, and the parameter/unit system."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, overload

from .context import SimulationContext
from .signals import Signal
from .units import from_si, known_units, to_si


class PortType(Enum):
    """What kind of signal a port carries.

    Typing ports is what lets the editor reject invalid wiring at edit time
    instead of failing halfway through a run. An MZM has an electrical input as
    well as an optical one; without port types that distinction cannot be
    expressed at all.
    """

    OPTICAL = "optical"
    ELECTRICAL = "electrical"
    BINARY = "binary"
    SYMBOL = "symbol"
    #: Log-likelihood ratios, one real number per *bit*. Not electrical — those
    #: are a sampled waveform with a unit and a sample rate — and emphatically
    #: not binary, because the whole content of a soft decision is the part that
    #: is not a bit yet. Giving it its own type is what stops a soft output being
    #: wired into a hard input, which would silently throw away the two to three
    #: decibels the soft decoder exists to recover.
    SOFT = "soft"
    METRIC = "metric"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class Port:
    """A reference to one named port on one component instance."""

    component: Component
    name: str
    type: PortType

    def __repr__(self) -> str:
        return f"{self.component.label}.{self.name}"


@dataclass(frozen=True)
class PortGroup:
    """A set of outputs and the inputs they are the only function of.

    **This exists because a three-port circulator is not a cycle.** The
    scheduler orders whole components, so a reflective device reached through a
    circulator -- signal into port 1, out of port 2, back into port 2 from the
    grating, out of port 3 -- reads as ``circulator -> grating -> circulator``
    and is refused as a feedback loop. At the level of *ports* there is no loop
    at all: ``out3`` depends on ``in2`` and on nothing else, ``out2`` depends on
    ``in1`` and on nothing else, and the light goes strictly forward. The
    dependency is real, the cycle is an artefact of the granularity.

    So a component may declare that its ports fall into independent groups, and
    the scheduler treats each group as its own node. :meth:`Component.run` is
    then called once per group with that group's inputs, and must return that
    group's outputs and no others.

    The default is one group holding everything, which is the honest answer for
    almost every block -- a fibre's output depends on its input, and an
    equaliser's diagnostics depend on the same samples its symbols do. Splitting
    a component whose outputs are *not* independent would let the scheduler run
    half of it before its other half's input exists.
    """

    inputs: frozenset[str]
    outputs: frozenset[str]

    def __repr__(self) -> str:
        return f"PortGroup({sorted(self.inputs)} -> {sorted(self.outputs)})"


class Param:
    """A component parameter, declared once, with its unit and valid values.

    The unit is part of the declaration rather than a comment, so that
    :meth:`Component.si` can convert consistently and the GUI manifest can be
    generated from the same source. Declaring parameters twice — once in code
    and once in a manifest file — guarantees they drift apart.

    ``min``/``max`` bound a continuous quantity. ``choices`` is for the other
    kind: a parameter whose legal values are a *set*, not an interval. A PRBS
    order may be 7, 9, 11, 15, 23 or 31 and nothing between; a QAM format is 1,
    2, 4, 6 or 8 bits per symbol and never 3. Declaring those as a range was a
    quiet lie — the field said "1 … 8", the editor accepted 3, and only the
    engine knew that 8-QAM is a cross constellation nobody has implemented. An
    interface can offer a set as a list to choose from and cannot offer an
    interval as anything but a box to type in, so the distinction earns its
    keep on screen as well as in validation.

    ``applies_when`` names a :class:`BoolParam` on the same component that this
    parameter only takes effect under — ``"saturate"``, or ``"!auto_span"`` for
    one that applies when the flag is *off*. It changes nothing about how the
    parameter behaves; what it does is let an interface grey out a box that
    cannot do anything, instead of offering a control that is quietly ignored.

    It lives here rather than in the interface for the reason every other part of
    a manifest does: the condition is a fact about the model, the code that reads
    the parameter is three files away from the code that draws the box, and a
    table of these maintained beside the editor is a table that goes stale the
    first time a flag is renamed.
    """

    def __init__(
        self,
        default: float,
        *,
        unit: str = "",
        min: float | None = None,
        max: float | None = None,
        choices: Sequence[float] | None = None,
        doc: str = "",
        applies_when: str | None = None,
    ) -> None:
        if unit not in known_units():
            raise ValueError(f"unknown unit {unit!r}; known units: {sorted(known_units())}")
        self.default = default
        self.unit = unit
        self.min = min
        self.max = max
        self.choices = tuple(choices) if choices is not None else None
        self.doc = doc
        self.applies_when = applies_when
        self.name = "<unbound>"
        if self.choices is not None and default not in self.choices:
            raise ValueError(
                f"default {default} is not one of the declared choices {list(self.choices)}"
            )

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name

    @overload
    def __get__(self, obj: None, owner: type | None = None) -> Param: ...

    @overload
    def __get__(self, obj: Component, owner: type | None = None) -> float: ...

    def __get__(self, obj: Component | None, owner: type | None = None) -> Param | float:
        if obj is None:
            return self
        return obj._values.get(self.name, self.default)

    def validate(self, value: float) -> float:
        if not isinstance(value, int | float) or isinstance(value, bool):
            raise TypeError(f"{self.name} must be a number, got {value!r}")
        value = float(value)
        if math.isnan(value):
            raise ValueError(f"{self.name} must not be NaN")
        if self.choices is not None and value not in self.choices:
            raise ValueError(
                f"{self.name}={value:g} is not one of {', '.join(f'{c:g}' for c in self.choices)}"
            )
        if self.min is not None and value < self.min:
            raise ValueError(f"{self.name}={value} {self.unit} is below the minimum {self.min}")
        if self.max is not None and value > self.max:
            raise ValueError(f"{self.name}={value} {self.unit} is above the maximum {self.max}")
        return value

    def to_dict(self) -> dict[str, Any]:
        """The JSON-serialisable description used to generate GUI manifests."""
        d: dict[str, Any] = {"type": "float", "default": self.default, "unit": self.unit}
        if self.min is not None:
            d["min"] = self.min
        if self.max is not None:
            d["max"] = self.max
        if self.choices is not None:
            d["choices"] = list(self.choices)
        if self.doc:
            d["doc"] = self.doc
        if self.applies_when:
            d["applies_when"] = self.applies_when
        return d


class BoolParam(Param):
    """A component parameter that is a flag rather than a quantity.

    A separate class rather than a mode on :class:`Param` so that a float
    parameter still types as ``float`` at every use site. Flags are read as plain
    attributes; :meth:`Component.si` refuses them, because converting a switch to
    an SI unit is meaningless and asking for it means something is confused.
    """

    def __init__(self, default: bool, *, doc: str = "") -> None:
        super().__init__(0.0, doc=doc)
        self.default = default

    @overload
    def __get__(self, obj: None, owner: type | None = None) -> BoolParam: ...

    @overload
    def __get__(self, obj: Component, owner: type | None = None) -> bool: ...

    def __get__(self, obj: Component | None, owner: type | None = None) -> BoolParam | bool:
        if obj is None:
            return self
        return bool(obj._values.get(self.name, self.default))

    def validate(self, value: object) -> bool:
        if not isinstance(value, bool):
            raise TypeError(f"{self.name} must be True or False, got {value!r}")
        return value

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"type": "bool", "default": self.default}
        if self.doc:
            d["doc"] = self.doc
        return d


class Component:
    """Base class for every block in a simulation graph.

    A component is a pure function of its inputs and its parameters: it receives
    read-only signals and returns new ones. Nothing is mutated in place, so the
    scheduler is free to order, cache, or parallelise execution.
    """

    #: Human-readable name, shown in the GUI palette.
    display_name: ClassVar[str] = ""

    #: Palette grouping.
    category: ClassVar[str] = "Uncategorised"

    #: Model version, bumped when numerical behaviour changes.
    version: ClassVar[str] = "0.1.0"

    #: Name this component is stored under in project files. Defaults to the
    #: class name; set it explicitly if a plugin would otherwise collide with
    #: a built-in.
    registry_name: ClassVar[str] = ""

    #: A class that exists to *share* parameters rather than to be placed in a
    #: graph. It is not registered and never reaches the palette. Set on the base
    #: itself and read from ``cls.__dict__``, so it is not inherited: a subclass
    #: of an abstract base is a real block unless it also declares itself
    #: abstract. Without this, a family of components with a dozen parameters in
    #: common has to choose between declaring them a dozen times and shipping a
    #: half-built block to the palette.
    abstract: ClassVar[bool] = False

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        from .registry import register

        if cls.__dict__.get("abstract", False):
            return
        cls._check_conditions()
        register(cls)

    @classmethod
    def _check_conditions(cls) -> None:
        """Refuse an ``applies_when`` that names nothing, at import time.

        A condition naming a flag that does not exist -- or one that was renamed
        out from under it -- would leave the interface unable to decide whether
        the box applies, and the obvious fallback (assume it does) is exactly the
        stale control this feature exists to remove. Checking at class creation
        means the failure arrives when the component is defined rather than when
        somebody opens its inspector.
        """
        specs = cls.param_specs()
        for name, spec in specs.items():
            condition = getattr(spec, "applies_when", None)
            if not condition:
                continue
            flag = condition[1:] if condition.startswith("!") else condition
            gate = specs.get(flag)
            if gate is None:
                raise TypeError(
                    f"{cls.__name__}.{name} applies_when={condition!r}, and this component "
                    f"has no parameter {flag!r}; it has {sorted(specs)}"
                )
            if not isinstance(gate, BoolParam):
                raise TypeError(
                    f"{cls.__name__}.{name} applies_when={condition!r}, but {flag!r} is a "
                    f"quantity rather than a flag. A condition has to be true or false."
                )

    @classmethod
    def type_name(cls) -> str:
        """The name this component is stored under in project files."""
        return cls.registry_name or cls.__name__

    def structural_config(self) -> dict[str, Any]:
        """Constructor arguments that are not parameters.

        A parameter changes a number; this changes the shape of the component —
        an N-way combiner's input count, for instance. The two are kept apart
        because a GUI has to treat them differently: a parameter can be edited
        in place, while changing the port set invalidates the connections drawn
        to it.
        """
        return {}

    # Port declarations. These are class-level defaults that `__init__` copies
    # onto the instance, so a component with a configurable port count (an N-way
    # combiner, say) can rebind its own without touching the class.
    inputs: dict[str, PortType] = {}
    outputs: dict[str, PortType] = {}

    def __init__(self, *, label: str | None = None, **params: float | bool) -> None:
        self.inputs = dict(type(self).inputs)
        self.outputs = dict(type(self).outputs)
        declared = self.param_specs()
        unknown = set(params) - set(declared)
        if unknown:
            raise TypeError(
                f"{type(self).__name__} has no parameter(s) {sorted(unknown)}; "
                f"declared: {sorted(declared)}"
            )
        self._values: dict[str, float | bool] = {
            name: declared[name].validate(value) for name, value in params.items()
        }

        # A label identifies the component to the RNG, so it must depend only on
        # the graph, never on how many components happened to be constructed
        # earlier in the process. Graph.add() assigns the automatic form.
        self.has_explicit_label = label is not None
        self.label = label if label is not None else type(self).__name__

    # -- introspection ----------------------------------------------------

    @classmethod
    def param_specs(cls) -> dict[str, Param]:
        """Every :class:`Param` declared on this class or a base class."""
        specs: dict[str, Param] = {}
        for klass in reversed(cls.__mro__):
            for name, value in vars(klass).items():
                if isinstance(value, Param):
                    specs[name] = value
        return specs

    @classmethod
    def manifest(cls) -> dict[str, Any]:
        """The component description consumed by the GUI.

        Generated from the class, never hand-written, so the schema cannot drift
        away from the implementation.

        **The ports are a default instance's, not the class's.** A component
        whose port count is configurable binds its own in ``__init__`` — an
        N-way combiner has as many inputs as it was asked for — so the class
        attribute is empty for exactly the components an editor most needs to
        draw. Reading them off a default instance reports two inputs for a
        combiner rather than none, which is what it will have if someone drops
        one on a canvas.

        ``structural`` carries the arguments that decide that shape, with the
        values the default instance was built with. They are listed separately
        from ``parameters`` because they are not the same kind of thing: a
        parameter changes a number and can be edited in place, while changing
        the port count invalidates every wire already drawn to it.

        Instantiating is guarded. A component that cannot be built without
        arguments still gets a manifest, described from its class as before,
        because a palette missing an entry is worse than one describing it
        incompletely.
        """
        try:
            probe: Component | None = cls()
        except Exception:  # a manifest must not fail to exist
            probe = None
        inputs = probe.inputs if probe is not None else cls.inputs
        outputs = probe.outputs if probe is not None else cls.outputs

        return {
            "name": cls.display_name or cls.__name__,
            "type": cls.type_name(),
            "class": f"{cls.__module__}.{cls.__qualname__}",
            "category": cls.category,
            "version": cls.version,
            "parameters": {name: spec.to_dict() for name, spec in cls.param_specs().items()},
            "structural": probe.structural_config() if probe is not None else {},
            "ports": {
                "inputs": {name: str(t) for name, t in inputs.items()},
                "outputs": {name: str(t) for name, t in outputs.items()},
            },
        }

    def validate(self) -> None:
        """Refuse a combination of this component's own parameters.

        Called once for every block before any of them runs, so a settings
        problem is reported instead of discovered part way through a
        simulation. Each parameter is already checked on its own where it is
        declared — a range, a set of choices — and this is for the rarer case
        where two of them are each fine and disagree with each other.

        Rare enough to be worth naming: at the time of writing exactly one
        component needs it. Everything else a block can object to is about its
        *input* rather than its settings, and cannot be known until the input
        arrives. That is why this is a method to override and not a schema in
        the manifest: one constraint does not need a language.
        """

    def si(self, name: str) -> float:
        """The value of parameter ``name`` converted to its SI base unit."""
        spec = self.param_specs().get(name)
        if spec is None:
            raise KeyError(f"{type(self).__name__} has no parameter {name!r}")
        if isinstance(spec, BoolParam):
            raise TypeError(f"{name} is a flag, not a quantity; read it as an attribute")
        return to_si(getattr(self, name), spec.unit)

    def display(self, name: str) -> tuple[float, str]:
        """The value of parameter ``name`` in its declared unit, with the unit."""
        spec = self.param_specs().get(name)
        if spec is None:
            raise KeyError(f"{type(self).__name__} has no parameter {name!r}")
        return from_si(self.si(name), spec.unit), spec.unit

    # -- ports ------------------------------------------------------------

    def port_groups(self) -> tuple[PortGroup, ...]:
        """Independent slices of this component, for the scheduler.

        One group holding every port, unless a subclass says otherwise, which is
        the truthful answer for almost everything: a block computes all of its
        outputs from all of its inputs in one call and there is nothing to
        separate. Overriding it is how a device whose routes genuinely do not
        touch -- a circulator's three -- becomes three scheduler nodes instead of
        one, so that a reflective device hanging off one of them is not mistaken
        for a feedback loop. See :class:`PortGroup`.

        It is an instance method rather than a class attribute because the port
        set is per-instance: a circulator built with two of its three ports
        driven has two groups, not three.
        """
        return (PortGroup(frozenset(self.inputs), frozenset(self.outputs)),)

    def check_port_groups(self) -> None:
        """Refuse groups that are not a partition of this component's ports.

        A port left out of every group never runs and a port in two of them runs
        twice, and both failures look like a wiring mistake somewhere else
        entirely -- an output that is mysteriously absent from the results, or a
        block that reports twice through the progress callback. Checked before
        the run rather than discovered during it.

        A group with no outputs is refused as well: it can never be scheduled to
        any purpose, and the inputs inside it would be silently unreachable.
        """
        groups = self.port_groups()
        if not groups:
            raise ValueError(f"{self.label}: port_groups() returned nothing")

        for group in groups:
            if not group.outputs:
                raise ValueError(f"{self.label}: {group!r} produces no output")

        for kind, declared in (("input", self.inputs), ("output", self.outputs)):
            seen: dict[str, int] = {}
            for index, group in enumerate(groups):
                for name in getattr(group, f"{kind}s"):
                    if name not in declared:
                        raise ValueError(
                            f"{self.label}: port_groups() names {kind} {name!r}, which is "
                            f"not one of its {kind}s {sorted(declared)}"
                        )
                    if name in seen:
                        raise ValueError(
                            f"{self.label}: {kind} {name!r} is in port group {seen[name]} "
                            f"and in {index}; a port belongs to exactly one"
                        )
                    seen[name] = index
            missing = sorted(set(declared) - set(seen))
            if missing:
                raise ValueError(
                    f"{self.label}: port_groups() leaves {kind}(s) {missing} out; "
                    f"every port belongs to exactly one group"
                )

    def __getitem__(self, port_name: str) -> Port:
        if port_name in self.outputs:
            return Port(self, port_name, self.outputs[port_name])
        if port_name in self.inputs:
            return Port(self, port_name, self.inputs[port_name])
        raise KeyError(
            f"{type(self).__name__} has no port {port_name!r}; "
            f"inputs={sorted(self.inputs)} outputs={sorted(self.outputs)}"
        )

    def sole_output(self) -> Port:
        """The only signal-carrying output port, for chaining.

        Metric ports are ignored: a measurement a component emits alongside its
        output — propagation diagnostics, say — is not part of the signal path,
        and having one should not force every chain through it to name ports
        explicitly.
        """
        signal_ports = [name for name, t in self.outputs.items() if t is not PortType.METRIC]
        if len(signal_ports) != 1:
            raise ValueError(
                f"{self.label} has {len(signal_ports)} signal outputs; name one explicitly, "
                f"e.g. {self.label}['{next(iter(self.outputs), 'out')}']"
            )
        return self[signal_ports[0]]

    def sole_input(self) -> Port:
        """The only input port, for chaining. Raises if there is not exactly one."""
        if len(self.inputs) != 1:
            raise ValueError(
                f"{self.label} has {len(self.inputs)} inputs; name one explicitly, "
                f"e.g. {self.label}['{next(iter(self.inputs), 'in')}']"
            )
        return self[next(iter(self.inputs))]

    # -- execution --------------------------------------------------------

    #: Parameters this component used to declare and no longer does.
    #:
    #: A project saved before the change carries them, and refusing to open it
    #: would punish somebody for a decision made here. They are dropped on load
    #: instead — which is safe precisely because a retired parameter is one that
    #: had stopped meaning anything, and is *not* a licence to rename a live one
    #: quietly. Renaming a parameter that still does something needs a migration
    #: that carries its value across, not this.
    retired_parameters: ClassVar[frozenset[str]] = frozenset()

    def run(self, ctx: SimulationContext, inputs: dict[str, Signal]) -> dict[str, Signal]:
        """Process the whole time window and return one signal per output port.

        Called exactly once per run. Implementations must not mutate ``inputs``
        and must not hold state between calls.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement run()")

    #: Where :meth:`report` sends its fraction, installed by the graph around a
    #: call to :meth:`run` and taken away again afterwards. Not a parameter and
    #: not state the component may read: it exists so that the *only* thing a
    #: component knows about progress is how to emit it.
    _reporter: Callable[[float], None] | None = None

    def report(self, fraction: float) -> None:
        """Say how far through this component's own work the run has got, 0 to 1.

        Optional, and almost every component ignores it: the ones that finish in
        a millisecond have nothing to report that anybody could read. It is worth
        calling from a loop whose length is set by a *parameter* rather than by
        the window — a span solved in ten thousand split steps is the case this
        exists for, and it is the difference between an interface that says how
        far along a run is and one that says only that it has not crashed.

        Cheap when nobody is listening, which is what lets a hot loop call it
        without guarding: the graph installs a sink only when a caller asked
        for progress.
        """
        reporter = self._reporter
        if reporter is not None:
            reporter(0.0 if fraction < 0.0 else 1.0 if fraction > 1.0 else float(fraction))

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in sorted(self._values.items()))
        return f"{type(self).__name__}({params})" if params else f"{type(self).__name__}()"
