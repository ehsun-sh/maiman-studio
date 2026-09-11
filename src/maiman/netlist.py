"""Layout netlists: a drawn circuit, read as data and solved here.

A layout tool — gdsfactory is the one this format comes from — knows where every
device sits and what is wired to what. It does not know what any of it *does*
optically: a ``CrossSection`` there carries width, offset, layer and bend radius,
and its model forbids extra fields, so there is no effective index anywhere in
it. This library is the other half. It knows what a directional coupler is and
has nothing to say about where one was placed.

So the join is: **the netlist supplies the topology and the geometry, the PDK
supplies the numbers, and the reduction in :mod:`maiman.circuit` supplies the
answer.** An instance says it is a ``straight`` drawn in cross-section ``strip``
and 16.637 um long; the kit says what ``straight`` is modelled by in this process
and what ``strip`` measures; the solver does the rest.

**Nothing here imports the layout tool.** A netlist is a document, the same way a
``.maiman`` project and a ``.pdk`` are, and reading one runs nothing. That is
partly the rule this library already applies to every file it opens, and partly a
licence: ``gdsfactory`` is MIT, but it requires ``kfactory``, which requires
``klayout``, which is **GPL-3.0-or-later** — 86 packages resolved, measured
rather than guessed. This project already refuses FFTW (GPL-2.0-or-later) and
``klujax`` (LGPL-2.0-only) on that ground, and the note in ``pyproject.toml``
says licences are checked before adoption. Reading the YAML it writes costs none
of that.

**The format.** Taken from gdsfactory's own shipped samples rather than from
memory::

    instances:
      s1:
        component: straight          # the cell, which the PDK maps to a model
        info:   {length: 16.637}     # geometry the router worked out
        settings: {cross_section: strip, length: 10}
    nets:
      - p1: s1,o2
        p2: b1,o1
    ports:
      o1: s1,o1
    name: my_circuit

``placements`` is read and ignored: where a device sits on the die does not
change its scattering matrix, and the lengths that *do* matter are already in
``info``. Older files write ``connections`` as a mapping and schematic files
write ``routes: {name: {links: {...}}}``; both are accepted, because a format
that has three spellings for one idea is not improved by a reader that knows one.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .circuit import Circuit, SMatrix
from .components.photonic import ScatteringDevice
from .pdk import PDK, PDKError

#: Separator between an instance name and a port name in a netlist reference:
#: ``"cp1,o3"``. gdsfactory's, not a choice made here.
PORT_SEPARATOR = ","


class NetlistError(ValueError):
    """A netlist that cannot be read, or that describes a circuit this cannot build."""


@dataclass(frozen=True)
class Instance:
    """One placed device: which cell it is, and what the layout knows about it."""

    name: str
    cell: str
    settings: dict[str, Any] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)
    """Derived geometry the layout tool computed, and the more useful of the two.

    ``settings`` is what the cell was *asked* for and ``info`` is what came out:
    a 90-degree Euler bend asked for ``radius: 10`` reports ``length: 16.637``,
    which is the arc length the light actually travels and is not recoverable
    from the radius alone without knowing the Euler parameter. When a kit sources
    a length from the netlist, this is nearly always where it should come from.
    """

    def lookup(self, path: str) -> Any:
        """Follow a dotted path like ``info.length`` into this instance."""
        head, _, tail = path.partition(".")
        table = {"info": self.info, "settings": self.settings}
        if head not in table or not tail:
            raise NetlistError(
                f"{self.name}: {path!r} is not a netlist path; it must start with "
                f"'info.' or 'settings.'"
            )
        value: Any = table[head]
        for part in tail.split("."):
            if not isinstance(value, Mapping) or part not in value:
                raise NetlistError(
                    f"instance {self.name!r} ({self.cell}) has no {path!r}; its {head} "
                    f"carries {sorted(table[head])}"
                )
            value = value[part]
        return value


@dataclass(frozen=True)
class Netlist:
    """A drawn circuit: what is on it, what is wired, and what faces outward."""

    name: str
    instances: dict[str, Instance]
    nets: tuple[tuple[tuple[str, str], tuple[str, str]], ...]
    ports: dict[str, tuple[str, str]]

    def __str__(self) -> str:
        return (
            f"Netlist({self.name!r}, {len(self.instances)} instances, "
            f"{len(self.nets)} nets, {len(self.ports)} ports)"
        )

    def cells(self) -> list[str]:
        """Every distinct cell used, which is what a kit has to cover."""
        return sorted({instance.cell for instance in self.instances.values()})


def _split(reference: Any, *, where: str) -> tuple[str, str]:
    if not isinstance(reference, str) or PORT_SEPARATOR not in reference:
        raise NetlistError(
            f"{where}: expected a port reference like 'instance{PORT_SEPARATOR}port', "
            f"got {reference!r}"
        )
    instance, _, port = reference.partition(PORT_SEPARATOR)
    return instance.strip(), port.strip()


def parse_netlist(data: Mapping[str, Any], *, where: str = "netlist") -> Netlist:
    """Validate a netlist document and turn it into a :class:`Netlist`.

    Separate from :func:`load_netlist` so that a netlist built in memory goes
    through the same checks a file does — and so that a caller who has already
    parsed the YAML with their own reader does not have to write it back out.
    """
    if not isinstance(data, Mapping):
        raise NetlistError(f"{where} does not contain a netlist object")

    raw_instances = data.get("instances")
    if not isinstance(raw_instances, Mapping) or not raw_instances:
        raise NetlistError(f"{where} has no instances, so there is no circuit in it")

    instances: dict[str, Instance] = {}
    for name, entry in raw_instances.items():
        if not isinstance(entry, Mapping):
            raise NetlistError(f"{where}: instance {name!r} is not an object")
        cell = entry.get("component")
        if not isinstance(cell, str):
            raise NetlistError(
                f"{where}: instance {name!r} does not say which cell it is (no 'component')"
            )
        instances[str(name)] = Instance(
            name=str(name),
            cell=cell,
            settings=dict(entry.get("settings") or {}),
            info=dict(entry.get("info") or {}),
        )

    nets: list[tuple[tuple[str, str], tuple[str, str]]] = []
    seen: dict[tuple[str, str], str] = {}

    def record(a: tuple[str, str], b: tuple[str, str], origin: str) -> None:
        for end in (a, b):
            if end[0] not in instances:
                raise NetlistError(
                    f"{where}: {origin} wires {end[0]}{PORT_SEPARATOR}{end[1]}, but there "
                    f"is no instance {end[0]!r}; there are {sorted(instances)}"
                )
            if end in seen:
                raise NetlistError(
                    f"{where}: {end[0]}{PORT_SEPARATOR}{end[1]} is wired twice, by "
                    f"{seen[end]} and by {origin}. A port carries one waveguide."
                )
            seen[end] = origin
        nets.append((a, b))

    # The modern spelling: a list of {p1, p2}.
    for index, net in enumerate(data.get("nets") or []):
        if not isinstance(net, Mapping) or "p1" not in net or "p2" not in net:
            raise NetlistError(f"{where}: nets[{index}] is not a {{p1, p2}} pair")
        record(
            _split(net["p1"], where=f"{where}: nets[{index}].p1"),
            _split(net["p2"], where=f"{where}: nets[{index}].p2"),
            f"nets[{index}]",
        )

    # The older spelling, and the one a hand-written schematic uses.
    for left, right in (data.get("connections") or {}).items():
        record(
            _split(left, where=f"{where}: connections key"),
            _split(right, where=f"{where}: connections[{left!r}]"),
            "connections",
        )

    # A schematic's routes. The router's own bends and straights are *not* here —
    # they exist only once the route is built — so a routed schematic read at this
    # level is the circuit without its interconnect, and its lengths are missing.
    for route_name, route in (data.get("routes") or {}).items():
        if not isinstance(route, Mapping):
            raise NetlistError(f"{where}: route {route_name!r} is not an object")
        for left, right in (route.get("links") or {}).items():
            record(
                _split(left, where=f"{where}: routes.{route_name}.links key"),
                _split(right, where=f"{where}: routes.{route_name}.links[{left!r}]"),
                f"routes.{route_name}",
            )

    ports: dict[str, tuple[str, str]] = {}
    for exposed, reference in (data.get("ports") or {}).items():
        end = _split(reference, where=f"{where}: ports.{exposed}")
        if end[0] not in instances:
            raise NetlistError(
                f"{where}: port {exposed!r} is on instance {end[0]!r}, which does not exist"
            )
        if end in seen:
            raise NetlistError(
                f"{where}: {end[0]}{PORT_SEPARATOR}{end[1]} is both wired by "
                f"{seen[end]} and exposed as {exposed!r}; it can be one or the other"
            )
        ports[str(exposed)] = end

    return Netlist(
        name=str(data.get("name") or where),
        instances=instances,
        nets=tuple(nets),
        ports=ports,
    )


def load_netlist(path: str | Path) -> Netlist:
    """Read a netlist file. JSON always; YAML when a YAML reader is installed.

    JSON needs nothing — it is in the standard library, and a netlist is a plain
    document. YAML is what the layout tool actually writes, and reading it needs
    ``PyYAML``, which is **not** a dependency of this package: one line of the
    caller's own code turns a netlist into JSON, and a library that made everyone
    install a parser for a file most of them will not have is the wrong trade.
    When it is missing the error says so rather than reporting a syntax error in
    a file that is perfectly valid.
    """
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise NetlistError(f"no such netlist file: {source}") from None

    if source.suffix.lower() in {".yml", ".yaml"}:
        try:
            import yaml  # type: ignore[import-untyped]
        except ModuleNotFoundError:
            raise NetlistError(
                f"{source} is YAML and no YAML reader is installed. Either "
                f"`pip install pyyaml`, or convert it once: "
                f"`json.dump(yaml.safe_load(open({source.name!r})), open('netlist.json','w'))`. "
                f"This package depends on numpy and nothing else, which is why it does "
                f"not carry one."
            ) from None
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as error:
            raise NetlistError(f"{source} is not valid YAML: {error}") from None
    else:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as error:
            raise NetlistError(f"{source} is not valid JSON: {error}") from None

    return parse_netlist(data, where=str(source))


def circuit_from_netlist(
    netlist: Netlist,
    pdk: PDK,
    frequencies: Sequence[float] | np.ndarray,
    *,
    wavelength: float | None = None,
    polarization: str = "te",
) -> Circuit:
    """Build the circuit a netlist describes, with a kit's numbers in it.

    Returns an unsolved :class:`~maiman.circuit.Circuit` rather than its matrix,
    because the caller may want to add to it — a source, a facet, a device the
    layout does not carry — and because a function that both assembles and solves
    gives no place to stand in between when the answer looks wrong.

    ``wavelength`` evaluates the kit's fits somewhere other than its reference,
    the same way :meth:`PDK.make` does. It is *not* the grid: the models here are
    frequency-flat by construction and a fit is resolved once, at build time, so
    a C-band sweep of a kit fitted at 1550 uses the 1550 numbers across the sweep
    unless told otherwise. Sweeping the fit itself means building the circuit
    again at each wavelength, which is a loop the caller writes and can see.
    """
    grid = np.asarray(frequencies, dtype=np.float64)
    circuit = Circuit()
    matrices: dict[str, SMatrix] = {}

    for name, instance in netlist.instances.items():
        entry = pdk.device_for_cell(instance.cell)
        overrides = {
            parameter: float(instance.lookup(path))
            for parameter, path in entry.from_netlist.items()
        }
        component = pdk.make(entry.name, wavelength=wavelength, label=name, **overrides)
        if not isinstance(component, ScatteringDevice):
            raise NetlistError(
                f"instance {name!r} is a {type(component).__name__}, which is not a "
                f"scattering device and cannot go in a circuit. The kit maps the cell "
                f"{instance.cell!r} onto it; a circuit can only hold devices with a "
                f"scattering matrix."
            )
        matrix = component.scattering_matrix(grid, polarization=polarization)
        matrices[name] = matrix
        circuit.add(name, matrix)

    def model_port(instance_name: str, port: str) -> str:
        instance = netlist.instances[instance_name]
        entry = pdk.device_for_cell(instance.cell)
        if port not in entry.ports:
            raise NetlistError(
                f"instance {instance_name!r} ({instance.cell}) uses port {port!r}, which "
                f"the kit's device {entry.name!r} does not map; it maps "
                f"{sorted(entry.ports) or 'nothing'}. Add it to that device's 'ports', "
                f"because guessing which model port a layout port is would wire a drop "
                f"to a through and still solve."
            )
        mapped = entry.ports[port]
        available = matrices[instance_name].ports
        if mapped not in available:
            raise NetlistError(
                f"the kit maps {instance.cell}.{port} onto {mapped!r}, which "
                f"{entry.component} has no such port on; it has {sorted(available)}. "
                f"These are the scattering matrix's ports, which are not always the "
                f"ports a link wires -- an undriven input exists here even when a graph "
                f"cannot connect to it."
            )
        return mapped

    for (left_instance, left_port), (right_instance, right_port) in netlist.nets:
        circuit.link(
            left_instance,
            model_port(left_instance, left_port),
            right_instance,
            model_port(right_instance, right_port),
        )

    for exposed, (instance_name, port) in netlist.ports.items():
        circuit.expose(exposed, instance_name, model_port(instance_name, port))

    if not netlist.ports:
        raise NetlistError(
            f"{netlist.name} exposes no ports, so there is nothing to solve for. A "
            f"layout tool writes them under 'ports'; a hand-written netlist has to too."
        )
    return circuit


def missing_cells(netlist: Netlist, pdk: PDK) -> list[str]:
    """Cells the netlist uses that the kit does not model, in order.

    Worth having before building rather than after: a design of any size uses a
    handful of cells many times over, and being told all of them at once is the
    difference between one edit to a kit and five rounds of trial and error.
    """
    absent = []
    for cell in netlist.cells():
        try:
            pdk.device_for_cell(cell)
        except PDKError:
            absent.append(cell)
    return absent
