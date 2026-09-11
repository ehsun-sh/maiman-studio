"""Process design kits: a foundry's measured numbers, fed to the models here.

A PDK in this library is **not** a component library. It does not add devices,
it does not carry layout, and importing one cannot execute anything — a `.pdk`
file is JSON, read the same way a `.maiman` project is, and for the same reason:
opening a file somebody sent you must not be equivalent to running their code.

What it carries is the half of a real design that this engine cannot derive.
:mod:`maiman.photonics` knows what a directional coupler *is*; it has no way to
know that this foundry's nominal 3 dB coupler measures 0.48 at 1550 nm and
0.41 at 1310, or that their strip waveguide is 2.1 dB/cm and their nitride is
0.08. Those numbers come off a wafer, and a PDK is the file they arrive in.

**Fits, not constants.** Any value here may be a number or a list of polynomial
coefficients in ``(lambda - lambda_ref)``, in nanometres, which is the form a
foundry quotes a fit in::

    "n_eff": [2.44, -1.13e-3]        # 2.44 at reference, sloping with wavelength
    "coupling": [0.48, 4.2e-4, -1e-7]

They are evaluated when a component is built, at the reference wavelength unless
another is asked for. The device models are frequency-flat by construction — see
:func:`maiman.photonics.directional_coupler` — so what a PDK buys is the right
constant for the band you are designing in, chosen by the file rather than
guessed. Asking for a coupler at 1530 and at 1565 gives two different couplers,
and that is the whole of the wavelength dependence: it is resolved at build time,
not inside a run.

**A fit has a window, and it is part of the file.** ``valid_wavelengths`` says
where the polynomial was fitted, and evaluating outside it is refused rather than
extrapolated. This is not fussiness. A first-order C-band fit of a coupler's
ratio, run out to 1310 nm, returns *minus* 0.46 — a negative power fraction, from
arithmetic that never complained. The component's own range happens to catch that
one, and blames the coupling for being negative rather than the caller for asking
240 nm outside the data. Anything without a declared range would have gone
through silently.

**What it refuses.** A device naming a component that is not registered, a
parameter that component does not declare, a cross-section that is not defined,
or a cross-section attached to a device that has no waveguide to apply it to.
All at load time, all naming the file and the entry, because a PDK is read once
and used a hundred times and a typo in it should not surface as a wrong number
three circuits later.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .component import Component
from .registry import UnknownComponentError, lookup

#: The schema this build reads. A PDK is a document like a project is, and the
#: same rule applies: refuse a version you do not understand rather than
#: guessing which half of it still means what it used to.
SCHEMA_VERSION = 1

#: Parameters that describe the waveguide a device is drawn in, as opposed to the
#: device itself. A cross-section may set these and nothing else; the split is
#: what lets one ``strip`` definition serve every block on the die.
CROSS_SECTION_PARAMETERS = frozenset({"n_eff", "n_group", "propagation_loss", "dispersion"})


class PDKError(ValueError):
    """A process design kit that cannot be read, or that describes something impossible."""


@dataclass(frozen=True)
class Fit:
    """One parameter as a polynomial in wavelength.

    ``coefficients[0]`` is the value at the reference wavelength and the rest are
    the slope, curvature and so on in ``(lambda - lambda_ref)`` measured in
    **nanometres**. Nanometres because that is the unit a foundry's fit is quoted
    in, and converting on the way in would mean every coefficient in every PDK
    file had to be rescaled by whoever wrote it.

    A bare number is a zero-order fit, which is the common case and should not
    have to be written as a list of one.
    """

    coefficients: tuple[float, ...]

    @classmethod
    def parse(cls, value: Any, *, where: str) -> Fit:
        if isinstance(value, bool):
            raise PDKError(f"{where}: expected a number or a list of coefficients, got a boolean")
        if isinstance(value, int | float):
            return cls((float(value),))
        if isinstance(value, list) and value and all(isinstance(v, int | float) for v in value):
            return cls(tuple(float(v) for v in value))
        raise PDKError(
            f"{where}: expected a number or a non-empty list of polynomial "
            f"coefficients, got {value!r}"
        )

    def at(self, wavelength_nm: float, reference_nm: float) -> float:
        """The fitted value at ``wavelength_nm`` [same unit as it was quoted in]."""
        delta = wavelength_nm - reference_nm
        # Horner, which is not about speed here — a fit is three coefficients —
        # but about not writing ``delta ** n`` and inviting a large power of a
        # large detuning to lose the constant term in rounding.
        total = 0.0
        for coefficient in reversed(self.coefficients):
            total = total * delta + coefficient
        return total

    @property
    def is_constant(self) -> bool:
        return len(self.coefficients) == 1

    def __str__(self) -> str:
        if self.is_constant:
            return f"{self.coefficients[0]:g}"
        terms = [f"{self.coefficients[0]:g}"]
        terms += [
            f"{c:+.3g}·d{'' if n == 1 else f'^{n}'}"
            for n, c in enumerate(self.coefficients[1:], start=1)
        ]
        return " ".join(terms)


@dataclass(frozen=True)
class CrossSection:
    """A waveguide the devices are drawn in: its indices, its loss, its dispersion."""

    name: str
    fits: dict[str, Fit]

    def at(self, wavelength_nm: float, reference_nm: float) -> dict[str, float]:
        return {name: fit.at(wavelength_nm, reference_nm) for name, fit in self.fits.items()}


@dataclass(frozen=True)
class Device:
    """One entry in a kit: which model, drawn in which waveguide, with which numbers."""

    name: str
    component: str
    cross_section: str | None
    fits: dict[str, Fit]
    structural: dict[str, Any]
    doc: str = ""
    valid_wavelengths: tuple[float, float] | None = None
    """Where this device's fits hold [nm], if narrower than the kit's."""

    cell: str | None = None
    """The layout cell this device is the model of, if a layout tool draws one.

    A netlist from a layout tool names cells, not models: it says ``straight``
    and ``mmi1x2`` because those are what the foundry's library calls them. This
    is the one place that knows ``straight`` means :class:`Waveguide` *in this
    process* — which is exactly a PDK's job, and is why the mapping lives in the
    file rather than in a table inside this library. Another process may call the
    same geometry something else.
    """

    ports: dict[str, str] = field(default_factory=dict)
    """Layout port name to model port name, e.g. ``{"o1": "in", "o2": "out"}``.

    Needed because the two vocabularies do not agree and neither is wrong. A
    layout tool numbers ports going round the boundary, ``o1`` to ``o4``; a model
    names them for what they do, ``in``/``through``/``drop``. Guessing the
    correspondence from the order would silently wire a drop port to a through
    port on any cell whose numbering ran the other way.
    """

    from_netlist: dict[str, str] = field(default_factory=dict)
    """Model parameter to a dotted path in the instance's netlist entry.

    ``{"length": "info.length"}`` says this device's length is whatever the
    layout says it is, per instance, rather than the kit's nominal value. That
    matters most for the cells a router generates: every bend in a route has its
    own arc length and every connecting straight its own, and a kit that gave all
    of them one nominal length would be describing a different circuit from the
    one that was drawn.
    """


@dataclass(frozen=True)
class PDK:
    """A foundry's numbers, and the components they build.

    Read with :func:`load_pdk`. The two things worth doing with one are
    :meth:`parameters`, which says what a device's numbers *are* at a wavelength,
    and :meth:`make`, which hands back a component carrying them.
    """

    name: str
    reference_wavelength: float
    cross_sections: dict[str, CrossSection]
    devices: dict[str, Device]
    description: str = ""
    source: Path | None = None
    valid_wavelengths: tuple[float, float] | None = None
    """Where the kit's fits hold [nm]. Outside it, evaluation is refused."""

    def __post_init__(self) -> None:
        if self.reference_wavelength <= 0.0:
            raise PDKError(f"{self.name}: reference_wavelength must be positive")
        window = self.valid_wavelengths
        if window is not None:
            if window[0] >= window[1]:
                raise PDKError(f"{self.name}: valid_wavelengths must be [low, high], got {window}")
            if not window[0] <= self.reference_wavelength <= window[1]:
                raise PDKError(
                    f"{self.name}: reference_wavelength {self.reference_wavelength:g} nm is "
                    f"outside the kit's own valid_wavelengths {window[0]:g}-{window[1]:g} nm"
                )

    def _check_window(self, entry: Device, wavelength: float) -> None:
        """Refuse a wavelength the fits were never fitted at.

        The device's window wins where it has one, because a kit whose
        waveguides are characterised across the C band may still carry one
        device measured over ten nanometres of it.
        """
        window = entry.valid_wavelengths or self.valid_wavelengths
        if window is None or window[0] <= wavelength <= window[1]:
            return
        raise PDKError(
            f"{self.name}: {entry.name!r} is fitted over "
            f"{window[0]:g}-{window[1]:g} nm and was asked for {wavelength:g} nm. "
            f"Outside that the polynomial is arithmetic, not data — a C-band "
            f"coupler fit run out to the O band returns a negative power "
            f"fraction. Widen valid_wavelengths only if the fit really holds there."
        )

    # -- reading ----------------------------------------------------------

    def device(self, name: str) -> Device:
        try:
            return self.devices[name]
        except KeyError:
            raise PDKError(
                f"{self.name} has no device {name!r}; it has {sorted(self.devices) or 'none'}"
            ) from None

    def device_for_cell(self, cell: str) -> Device:
        """The device modelling layout cell ``cell``.

        Refused rather than skipped when there is none. A netlist importer that
        quietly dropped the cells it did not recognise would return a circuit
        that solves, looks like a spectrum, and is not the circuit on the mask —
        which is the worst of the three possible outcomes.
        """
        for entry in self.devices.values():
            if entry.cell == cell:
                return entry
        known = sorted(e.cell for e in self.devices.values() if e.cell)
        raise PDKError(
            f"{self.name} has no device modelling the layout cell {cell!r}; it models "
            f'{known or "no cells at all"}. Add a device with "cell": {cell!r}, or '
            f"the circuit this builds is not the circuit that was drawn."
        )

    def parameters(self, name: str, *, wavelength: float | None = None) -> dict[str, float]:
        """Every parameter this device would be built with, evaluated [display units].

        Worth having separately from :meth:`make` because it is what you print in
        a report or diff between two kits, and because seeing the numbers is how
        anyone notices that a "3 dB" coupler is 0.41 in the O band.
        """
        entry = self.device(name)
        at = self.reference_wavelength if wavelength is None else wavelength
        self._check_window(entry, at)
        values: dict[str, float] = {}
        if entry.cross_section is not None:
            section = self.cross_sections[entry.cross_section]
            values.update(section.at(at, self.reference_wavelength))
        values.update(
            {key: fit.at(at, self.reference_wavelength) for key, fit in entry.fits.items()}
        )
        # The component is told where its indices were quoted, so that its own
        # frequency expansion is anchored to the same place the fit was.
        component_class = lookup(entry.component)
        if "reference_wavelength" in component_class.param_specs():
            values.setdefault("reference_wavelength", at)
        return values

    def make(
        self,
        name: str,
        *,
        wavelength: float | None = None,
        label: str | None = None,
        **overrides: float,
    ) -> Component:
        """Build the component this device describes.

        ``wavelength`` evaluates the fits somewhere other than the kit's
        reference, which is how one file serves the O band and the C band.
        ``overrides`` win over the kit, because a PDK is where a design starts
        and not a cage: sweeping a foundry's nominal coupling to see what a
        process corner would cost is exactly the thing this should make easy.
        """
        entry = self.device(name)
        values = self.parameters(name, wavelength=wavelength)
        values.update(overrides)
        component_class = lookup(entry.component)
        try:
            return component_class(label=label, **entry.structural, **values)
        except (TypeError, ValueError) as error:
            raise PDKError(f"{self.name}: cannot build {name!r}: {error}") from None

    def __str__(self) -> str:
        return (
            f"PDK({self.name!r}, {len(self.devices)} devices, "
            f"{len(self.cross_sections)} cross-sections, "
            f"at {self.reference_wavelength:g} nm)"
        )


def pdk_from_dict(data: dict[str, Any], *, source: Path | None = None) -> PDK:
    """Validate a PDK document and turn it into a :class:`PDK`.

    Separate from :func:`load_pdk` so that a kit built in memory — a test, a
    generated corner — goes through exactly the same checks a file does.
    """
    where = str(source) if source else "pdk"
    if not isinstance(data, dict):
        raise PDKError(f"{where} does not contain a PDK object")

    version = data.get("schema_version")
    if version is None:
        raise PDKError(f"{where} is not a PDK: no schema_version")
    if version != SCHEMA_VERSION:
        raise PDKError(
            f"{where} uses schema version {version}, this build reads version {SCHEMA_VERSION}"
        )

    name = data.get("name") or (source.stem if source else "unnamed")
    reference = data.get("reference_wavelength")
    if not isinstance(reference, int | float):
        raise PDKError(f"{where}: reference_wavelength is required, in nm")

    def window(value: Any, what: str) -> tuple[float, float] | None:
        if value is None:
            return None
        if (
            not isinstance(value, list)
            or len(value) != 2
            or not all(isinstance(v, int | float) for v in value)
        ):
            raise PDKError(f"{where}: {what} must be [low, high] in nm, got {value!r}")
        return (float(value[0]), float(value[1]))

    kit_window = window(data.get("valid_wavelengths"), "valid_wavelengths")

    cross_sections: dict[str, CrossSection] = {}
    for section_name, section in (data.get("cross_sections") or {}).items():
        if not isinstance(section, dict):
            raise PDKError(f"{where}: cross-section {section_name!r} is not an object")
        unknown = set(section) - CROSS_SECTION_PARAMETERS
        if unknown:
            raise PDKError(
                f"{where}: cross-section {section_name!r} sets {sorted(unknown)}, which is not "
                f"a waveguide property; a cross-section may set "
                f"{sorted(CROSS_SECTION_PARAMETERS)}. Device-specific numbers "
                f"belong on the device."
            )
        cross_sections[section_name] = CrossSection(
            name=section_name,
            fits={
                key: Fit.parse(value, where=f"{where}: {section_name}.{key}")
                for key, value in section.items()
            },
        )

    devices: dict[str, Device] = {}
    for device_name, entry in (data.get("devices") or {}).items():
        if not isinstance(entry, dict):
            raise PDKError(f"{where}: device {device_name!r} is not an object")
        component = entry.get("component")
        if not isinstance(component, str):
            raise PDKError(f"{where}: device {device_name!r} has no 'component'")
        try:
            component_class = lookup(component)
        except UnknownComponentError as error:
            raise PDKError(f"{where}: device {device_name!r} names {error}") from None

        declared = component_class.param_specs()
        fits = {
            key: Fit.parse(value, where=f"{where}: {device_name}.{key}")
            for key, value in (entry.get("params") or {}).items()
        }
        unknown = set(fits) - set(declared)
        if unknown:
            raise PDKError(
                f"{where}: device {device_name!r} sets {sorted(unknown)}, which "
                f"{component} does not declare; it has {sorted(declared)}"
            )

        section = entry.get("cross_section")
        if section is not None:
            if section not in cross_sections:
                raise PDKError(
                    f"{where}: device {device_name!r} is drawn in cross-section {section!r}, "
                    f"which this kit does not define; it has {sorted(cross_sections) or 'none'}"
                )
            # A cross-section on a device that has no waveguide is a statement
            # the engine cannot honour. Dropping it quietly would be worse: the
            # kit would look like it had specified something it had not.
            missing = set(cross_sections[section].fits) - set(declared)
            if missing:
                raise PDKError(
                    f"{where}: device {device_name!r} is drawn in cross-section {section!r}, "
                    f"but {component} has no {sorted(missing)} to apply it to — it is a lumped "
                    f"model with no waveguide in it. Leave the cross-section off."
                )

        ports = entry.get("ports") or {}
        if not isinstance(ports, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in ports.items()
        ):
            raise PDKError(
                f"{where}: device {device_name!r} has a 'ports' that is not a mapping of "
                f"layout port name to model port name, got {ports!r}"
            )
        # The names are not checked here, and that is deliberate rather than
        # lax. A circuit wires the *scattering matrix's* ports, which are not
        # always the graph ports a link wires: a 2x2 MMI driven on one side has
        # three graph ports and a four-port matrix, because the undriven input
        # still exists physically and still reflects nothing. Only the built
        # device knows the real set, so the check is where the device is built,
        # in `maiman.netlist`, and it names the ports it actually has.

        sourced = entry.get("from_netlist") or {}
        if not isinstance(sourced, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in sourced.items()
        ):
            raise PDKError(
                f"{where}: device {device_name!r} has a 'from_netlist' that is not a "
                f"mapping of parameter name to netlist path, got {sourced!r}"
            )
        unknown_sourced = set(sourced) - set(declared)
        if unknown_sourced:
            raise PDKError(
                f"{where}: device {device_name!r} sources {sorted(unknown_sourced)} from the "
                f"netlist, which {component} does not declare; it has {sorted(declared)}"
            )

        cell = entry.get("cell")
        if cell is not None and not isinstance(cell, str):
            raise PDKError(f"{where}: device {device_name!r} has a 'cell' that is not a name")

        devices[device_name] = Device(
            name=device_name,
            component=component,
            cross_section=section,
            fits=fits,
            structural=dict(entry.get("structural") or {}),
            doc=str(entry.get("doc") or ""),
            valid_wavelengths=window(
                entry.get("valid_wavelengths"), f"device {device_name!r} valid_wavelengths"
            ),
            cell=cell,
            ports=dict(ports),
            from_netlist=dict(sourced),
        )

    claimed: dict[str, str] = {}
    for entry_name, entry_device in devices.items():
        if entry_device.cell is None:
            continue
        if entry_device.cell in claimed:
            raise PDKError(
                f"{where}: devices {claimed[entry_device.cell]!r} and {entry_name!r} both "
                f"model the layout cell {entry_device.cell!r}. A netlist naming it could "
                f"mean either, and picking one would be a coin toss nothing reports."
            )
        claimed[entry_device.cell] = entry_name

    return PDK(
        name=str(name),
        description=str(data.get("description") or ""),
        reference_wavelength=float(reference),
        cross_sections=cross_sections,
        devices=devices,
        source=source,
        valid_wavelengths=kit_window,
    )


def load_pdk(path: str | Path) -> PDK:
    """Read a ``.pdk`` file. JSON in, validated :class:`PDK` out, nothing executed."""
    source = Path(path)
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise PDKError(f"no such PDK file: {source}") from None
    except json.JSONDecodeError as error:
        raise PDKError(f"{source} is not valid JSON: {error}") from None
    return pdk_from_dict(data, source=source)
