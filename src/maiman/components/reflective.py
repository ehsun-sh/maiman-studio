"""Devices whose light comes back: the grating, and the circulator that reaches it.

Every other optical block in this library is a one-way street — light enters one
port and leaves another, and the dataflow graph draws that as an arrow. A Bragg
grating is the first one that is not. Its useful output leaves through the fibre
it arrived on, and in hardware the only way to get at that is a circulator.

**Both halves of that are here, and neither of them cheats.** The grating's
scattering matrix has its reflection on the diagonal, where it belongs, and the
circulator is genuinely non-reciprocal —
:meth:`maiman.circuit.Circuit.solve` has always handled both and until now
nothing outside a test made it. What the graph needed was the other half: a
circulator's three routes do not depend on each other, so it declares them as
separate :class:`~maiman.component.PortGroup` s and the scheduler stops reading
``circulator -> grating -> circulator`` as a feedback loop it has to refuse.

The models themselves are in :mod:`maiman.photonics`, with the literature they
come from; this module is what turns them into blocks a link can contain.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np

from ..circuit import SMatrix
from ..component import BoolParam, Param, PortGroup, PortType
from ..context import SimulationContext
from ..photonics import (
    APODIZATIONS,
    DEFAULT_GRATING_SECTIONS,
    SILICA_FIBER_NEFF,
    SILICA_PHOTOELASTIC,
    SILICA_THERMAL_SENSITIVITY,
    bragg_shift,
    circulator,
    fiber_bragg_grating,
    phase_shift_profile,
    sampled_profile,
    section_positions,
)
from ..signals import OpticalSignal, Signal
from ..units import C_LIGHT, wavelength_to_frequency
from .photonic import ScatteringDevice, _sum_signals, apply_response, port_response, solve_once


class FiberBraggGrating(ScatteringDevice):
    """A written grating in a fibre core: a mirror for one channel and a window for the rest.

    The one component in this library that hands back more light than it passes
    on. ``reflected`` is the band the grating turns round; ``transmitted`` is
    everything else, which continues down the fibre with a notch cut in it. Both
    come out of one run because they are two entries of one scattering matrix,
    not two devices.

    **``reflected`` leaves through the input fibre.** That is the physical truth
    and the port name is the only warning the graph can give: wiring it straight
    onward models an ideal circulator with no loss and perfect isolation. Put a
    :class:`Circulator` in the path to pay for it, which is what the hardware
    does.

    **Three things it is worth using for.**

    *A channel filter that drops rather than passes.* A uniform grating a
    centimetre long with ``index_modulation = 1e-4`` reflects 93 % at its Bragg
    wavelength over 0.2 nm, and its sidelobes are 7.6 dB down — far too much
    crosstalk for a WDM drop, which is what ``apodization`` is for: a
    raised-cosine profile takes them to 31 dB down and a Gaussian to 41, at the
    cost of reflecting less for the same length.

    *Dispersion compensation in the line.* A ``chirp`` makes different
    wavelengths turn round at different depths, so the grating has a group delay
    that is linear in wavelength — 966 ps/nm for 10 cm chirped over a nanometre,
    which undoes 57 km of standard fibre in a part the length of a finger. Unlike
    :class:`~maiman.components.DispersionCompensator`, this is an optical device
    and works on a direct-detection link, where there is no recovered field for
    DSP to operate on and never will be.

    **The sign.** A positive ``chirp`` puts the short wavelengths at the near end
    where they turn round early, so the long ones arrive later:
    ``dtau/dlambda > 0``, which is ``D > 0``, the sign standard fibre has. A
    compensator is therefore a **negative** chirp — or the same grating entered
    from its far end, which is why ``reflected`` is computed from the input face
    rather than from whichever end is convenient.

    For a while this device disagreed with :class:`~maiman.components.Fiber`
    about that, and it was the fibre that was wrong:
    :func:`maiman.kernels.propagate_dispersion` carried the opposite quadratic
    sign from :func:`maiman.photonics.propagation_constant`, which the latter's
    docstring had said in as many words since before this device existed. The
    grating was the first component to put both in one graph where it showed, and
    the kernel has since been corrected.

    *A gain-flattening or ASE-blocking element*, since what it reflects it
    removes from the transmitted path exactly.

    See :func:`maiman.photonics.fiber_bragg_grating` for the model, the
    transfer-matrix construction, and what it deliberately leaves out.
    """

    display_name = "Fiber Bragg Grating"
    category = "Passive"

    #: Not a :class:`~maiman.components.photonic._Photonic`, and the reason is
    #: the same one the MMI gives: those parameters are a silicon strip's. A
    #: grating is written into drawn fibre, where the index is 1.4475 rather than
    #: 2.44 and the loss is 0.2 dB/km rather than 2 dB/cm -- a default of the
    #: wrong one would put the reflection 40 % away in wavelength and absorb four
    #: orders of magnitude too much. It carries its own ``n_eff`` and no loss at
    #: all, because over centimetres of fibre there is none to carry.

    length = Param(10.0, unit="mm", min=0.01, doc="Physical length of the written region")
    index_modulation = Param(
        1.0e-4,
        unit="",
        min=0.0,
        max=1.0e-2,
        doc="Amplitude of the index fringe; 1e-4 is strong, 1e-5 weak",
    )
    bragg_wavelength = Param(
        1550.0, unit="nm", min=1200.0, max=1700.0, doc="Wavelength it reflects"
    )
    n_eff = Param(
        SILICA_FIBER_NEFF, unit="", min=1.0, max=2.0, doc="Effective index of the fibre mode"
    )
    chirp = Param(
        0.0,
        unit="nm",
        doc="Total change in Bragg wavelength end to end; what makes it a compensator",
    )
    sections = Param(
        float(DEFAULT_GRATING_SECTIONS),
        unit="",
        min=1.0,
        max=20000.0,
        doc="Pieces the transfer-matrix model cuts the grating into",
    )

    #: A break in the periodicity, which puts a transmission window in the middle
    #: of the stop band -- the DFB laser's cavity, and the narrowest thing a
    #: grating of a given length can make. At pi, dead centre, a 2 cm grating's
    #: window is 0.7 pm wide and reaches full transmission; moving it off centre
    #: makes the two halves unequal mirrors and the resonance dies, measured at
    #: 1.000, 0.705 and 0.099 for positions 0.5, 0.45 and 0.35.
    phase_shifted = BoolParam(False, doc="Break the periodicity once, partway along")
    phase_shift = Param(
        math.pi,
        unit="rad",
        doc="Size of the jump; pi puts the window on the Bragg wavelength",
        applies_when="phase_shifted",
    )
    phase_shift_position = Param(
        0.5,
        unit="",
        min=0.0,
        max=1.0,
        doc="Where along the length it happens, 0 at the input face",
        applies_when="phase_shifted",
    )

    #: **What makes it a sensor.** A grating's period is a length, a length
    #: responds to being stretched and to being warmed, and the reflection reports
    #: it -- so the same device that drops a channel is a strain gauge and a
    #: thermometer, with nothing added to the fibre. See
    #: :func:`maiman.photonics.bragg_shift`.
    strain = Param(
        0.0, unit="ustrain", doc="Fractional elongation in millionths, positive in tension"
    )
    temperature_change = Param(
        0.0, unit="K", doc="Departure from the temperature bragg_wavelength was quoted at"
    )
    photoelastic_constant = Param(
        SILICA_PHOTOELASTIC,
        unit="",
        min=0.0,
        max=1.0,
        doc="p_e: how much the index change cancels the stretch",
    )
    thermal_sensitivity = Param(
        SILICA_THERMAL_SENSITIVITY,
        unit="1/K",
        min=0.0,
        doc="alpha + xi; a coating raises it, often two- or threefold",
    )

    #: Writing switched on and off along the length, which turns the one peak
    #: into a comb of them spaced by ``lambda**2 / (2 n_eff * sampling period)``.
    #: The superstructure grating: one device addressing a whole band.
    sampled = BoolParam(False, doc="Erase the grating periodically, for a comb of peaks")
    sample_periods = Param(
        10.0,
        unit="",
        min=1.0,
        doc="Sampling periods along the length; sets the comb spacing",
        applies_when="sampled",
    )
    sample_duty = Param(
        0.5,
        unit="",
        min=0.01,
        max=1.0,
        doc="Written fraction of each period; 1 is not sampled at all",
        applies_when="sampled",
    )

    inputs = {"in": PortType.OPTICAL}
    outputs = {"reflected": PortType.OPTICAL, "transmitted": PortType.OPTICAL}

    def __init__(
        self, apodization: str = "uniform", *, label: str | None = None, **params: float
    ) -> None:
        if apodization not in APODIZATIONS:
            raise ValueError(f"unknown apodization {apodization!r}; have {sorted(APODIZATIONS)}")
        super().__init__(label=label, **params)
        self.apodization = apodization

    def structural_config(self) -> dict[str, Any]:
        # Structural rather than a Param because a Param is a number, and this is
        # one of three names. It changes no port, which puts it beside the MZI's
        # ``use_mmi`` rather than beside a combiner's input count.
        return {"apodization": self.apodization}

    def sensed_bragg_wavelength(self) -> float:
        """Where it reflects once stretched and warmed [m].

        ``bragg_wavelength`` is the unstrained value at the temperature it was
        quoted at, and this is what an instrument would actually read. Every
        other method here reports against this rather than against the declared
        number, because a sensor that reported its own nameplate would be no use.
        """
        return bragg_shift(
            self.si("bragg_wavelength"),
            strain=self.si("strain"),
            temperature_change=self.si("temperature_change"),
            photoelastic=self.photoelastic_constant,
            thermal_sensitivity=self.si("thermal_sensitivity"),
        )

    def strain_sensitivity(self) -> float:
        """Wavelength shift per unit strain [m], ``(1 - p_e) * lambda_B``.

        SI, so it is metres per unit strain: 1.209e-6 at 1550 nm, which is the
        **1.21 pm per microstrain** a datasheet quotes.
        """
        return (1.0 - self.photoelastic_constant) * self.si("bragg_wavelength")

    def temperature_sensitivity(self) -> float:
        """Wavelength shift per kelvin [m/K], ``(alpha + xi) * lambda_B``.

        11.2 pm/K at 1550 nm for bare fibre, and the thermo-optic term is twelve
        thirteenths of it -- so this is a thermometer made of glass rather than
        one made of geometry, and a coating moves it.
        """
        return self.si("thermal_sensitivity") * self.si("bragg_wavelength")

    def cross_sensitivity(self) -> float:
        """Strain a kelvin of drift imitates, ``(alpha + xi) / (1 - p_e)``.

        **9.26 microstrain per kelvin** for bare silica, and it is the number the
        whole field is organised around: one grating gives one wavelength and
        there are two unknowns behind it, so an uncompensated strain reading is
        a strain reading plus an unknown temperature. Independent of the Bragg
        wavelength, because both sensitivities scale with it -- which is also why
        two gratings of the same fibre at different wavelengths make a matrix too
        close to singular to invert.
        """
        return self.si("thermal_sensitivity") / (1.0 - self.photoelastic_constant)

    def bragg_frequency(self) -> float:
        """The frequency it reflects [Hz], as sensed."""
        return wavelength_to_frequency(self.sensed_bragg_wavelength())

    def coupling(self) -> float:
        """Coupling coefficient ``kappa`` [1/m]; ``kappa * L`` decides the strength."""
        return float(np.pi * self.index_modulation / self.si("bragg_wavelength"))

    def peak_reflectivity(self) -> float:
        """``tanh**2(kappa L)``: what a *uniform* grating returns at its Bragg wavelength.

        Exact for the uniform case and an upper bound for the others, since
        apodization lowers the average coupling for the same length -- which is
        what the sidelobe suppression is bought with.
        """
        return float(np.tanh(self.coupling() * self.si("length")) ** 2)

    def bandwidth(self) -> float:
        """Full width between the first nulls either side of the peak [m of wavelength].

        ``lambda**2 / (pi n_eff L) * sqrt((kappa L)**2 + pi**2)``, the standard
        result. The two terms are the two regimes: a strong grating's width is
        set by its coupling and a weak one's by its length, and the square root
        crosses over between them at ``kappa L = pi``.
        """
        length = self.si("length")
        wavelength = self.si("bragg_wavelength")
        kappa_l = self.coupling() * length
        return float(wavelength**2 / (np.pi * self.n_eff * length) * np.sqrt(kappa_l**2 + np.pi**2))

    def dispersion(self) -> float:
        """Group-delay slope of the reflection [s/m], or zero when unchirped.

        ``2 n_eff L / (c * chirp)``, which is the round trip through the grating
        spread over the band it is chirped across. Geometric, and so an estimate:
        it holds while the chirp is well clear of the grating's own uniform
        bandwidth and degrades as the two approach, which
        :func:`maiman.photonics.fiber_bragg_grating` gives measured numbers for.
        """
        chirp = self.si("chirp")
        if chirp == 0.0:
            return 0.0
        return float(2.0 * self.n_eff * self.si("length") / (C_LIGHT * chirp))

    def run(self, ctx: SimulationContext, inputs: dict[str, Signal]) -> dict[str, Signal]:
        signal: OpticalSignal = inputs["in"]
        matrix_for = self._matrix_factory()
        # The narrowest thing in the response is the sidelobe ripple, whose
        # period is c / (2 n_eff L) -- 10 GHz for a centimetre of fibre. An ASE
        # bin averaged without knowing that strides over a reflection band it
        # only partly overlaps, and the answer depends on where the stride
        # happened to land. No ``period``: a grating's spectrum does not repeat.
        resolution = C_LIGHT / (2.0 * self.n_eff * self.si("length"))
        return {
            "reflected": apply_response(
                signal, port_response(matrix_for, "in", "in"), resolution=resolution
            ),
            "transmitted": apply_response(
                signal, port_response(matrix_for, "out", "in"), resolution=resolution
            ),
        }

    def _matrix_factory(self, polarization: str = "te") -> Callable[[np.ndarray], SMatrix]:
        # ``polarization`` is accepted and ignored. A fibre grating is weakly
        # birefringent -- it splits the two axes by picometres, where a silicon
        # strip splits them by tens of nanometres -- and the number is per draw
        # and per writing setup. See the note in ``_Photonic``.
        length = self.si("length")
        modulation = self.index_modulation
        # Strain and temperature move the whole profile, not only its centre: a
        # stretched grating's local period is longer everywhere, so a chirped one
        # keeps its shape and scales with it. The correction to the chirp is
        # parts per million of a nanometre and changes nothing anyone measures --
        # it is here because leaving it out would be a claim that a stretched
        # grating is chirped differently from an unstretched one, which is not
        # what stretching does.
        wavelength = self.sensed_bragg_wavelength()
        scale = wavelength / self.si("bragg_wavelength")
        index = self.n_eff
        chirp = self.si("chirp") * scale
        pieces = self._sections()

        # Sampling and apodization both shape the coupling, and a grating can
        # carry both -- a sampled device still wants its skirts quietened. The
        # physics function refuses to take a named window *and* an array, so the
        # window is evaluated here and the two are multiplied, which is what a
        # mask over an apodized exposure physically does.
        coupling = None
        if self.sampled:
            coupling = APODIZATIONS[self.apodization](section_positions(pieces))
            coupling = coupling * sampled_profile(
                pieces, periods=self.sample_periods, duty=self.sample_duty
            )
        phase = (
            phase_shift_profile(
                pieces, shift=self.si("phase_shift"), position=self.phase_shift_position
            )
            if self.phase_shifted
            else None
        )
        window = "uniform" if coupling is not None else self.apodization

        return solve_once(
            lambda f: fiber_bragg_grating(
                f,
                length=length,
                index_modulation=modulation,
                bragg_wavelength=wavelength,
                n_eff=index,
                chirp=chirp,
                apodization=window,
                coupling_profile=coupling,
                phase_profile=phase,
                sections=pieces,
            )
        )

    #: Sections per sampling period below which the comb stops being resolved.
    #: Measured: the peak spacing is unchanged from ten sections per period all
    #: the way to four hundred, so ten is the floor and twenty is the margin.
    SECTIONS_PER_SAMPLE = 20

    def _sections(self) -> int:
        """How finely to cut, which sampling can demand more of than the default.

        A sampled grating's structure is the mask, not the envelope, so the
        section count has to resolve *that*: at fewer than ten sections per
        sampling period the comb is no longer the device's. Raised rather than
        refused, because the number is derivable and a component that declines to
        run over an arithmetic it could have done itself is a bad component.
        """
        declared = int(self.sections)
        if not self.sampled:
            return declared
        return max(declared, int(self.SECTIONS_PER_SAMPLE * self.sample_periods))


class Circulator(ScatteringDevice):
    """Three ports, one direction: 1 to 2, 2 to 3, 3 to 1. How a link reaches a mirror.

    A reflective device sends its answer back out of the fibre it arrived on, and
    a dataflow graph has no arrow for that. This is the part that does in the
    model what it does on a bench: it turns the return path into a forward one.

    **Its routes are independent, and that is what makes it wirable.** ``out2`` is
    a function of ``in1`` alone and ``out3`` of ``in2`` alone, so the three are
    declared as separate :class:`~maiman.component.PortGroup` s and scheduled as
    separate nodes. Without that, the canonical drop —

    ::

        signal -> circ.in1 ;  circ.out2 -> fbg.in
        fbg.reflected -> circ.in2 ;  circ.out3 -> receiver

    reads as ``circ -> fbg -> circ`` and is refused as a feedback loop, even
    though no light in it ever travels backwards in time. The cycle is an
    artefact of scheduling whole components; the physics has no loop in it.

    **With isolation the routes overlap, and there is still no loop.** A real
    circulator leaks backwards on every hop, so ``out3`` hears the direct leak from
    ``in1`` as well as the reflection coming back into ``in2``. Solved exactly, that
    leak never returns to the grating; the drop port is the feed-forward sum and
    nothing more. What changes is only that ``in1`` is read by two groups, which
    :class:`~maiman.component.PortGroup` allows.

    **Return loss is the one route that is a cycle.** Light reflected back out of
    the port it entered by goes straight back into whatever is on that port. Next
    to a grating that is a cavity, and the graph refuses it until a
    :class:`~maiman.components.Feedback` on one wire of the loop runs it for a
    declared number of passes.

    ``driven`` says which physical ports light enters, and the outputs follow
    from the routing: driving 1 and 2 — the reflective configuration this exists
    for — gives inputs ``in1``, ``in2`` and outputs ``out2``, ``out3``. Add 3 to
    close the ring. It is structural rather than a parameter for the usual
    reason: it decides the port set, and changing it invalidates wires already
    drawn.

    ``insertion_loss`` is **per hop**, which is how a datasheet quotes it, so
    light that goes in at 1 and comes back out at 3 has paid it twice. That is
    not a detail — it is the reason a grating-based drop costs more than a
    filter-based one, and it is why the number belongs on the circulator rather
    than being folded into the grating.

    See :func:`maiman.photonics.circulator` for the matrix, for why isolation
    turned out not to be a loop, and for the return loss that is one.
    """

    display_name = "Circulator"
    category = "Passive"

    insertion_loss = Param(
        0.7, unit="dB", min=0.0, doc="Loss of one hop; paid again on the next hop"
    )
    isolation = Param(
        0.0,
        unit="dB",
        min=0.0,
        doc="Reverse leak on every hop; 0 is an ideal circulator, a real one 40 to 60",
    )
    return_loss = Param(
        0.0,
        unit="dB",
        min=0.0,
        doc="Reflection back out of the port light entered; 0 is none. Makes a cavity",
    )

    def __init__(
        self,
        driven: tuple[int, ...] | list[int] = (1, 2),
        *,
        label: str | None = None,
        **params: float,
    ) -> None:
        ports = tuple(int(p) for p in driven)
        if not ports:
            raise ValueError("a circulator with no driven port carries nothing")
        if len(set(ports)) != len(ports):
            raise ValueError(f"driven lists a port twice: {list(driven)}")
        if not all(1 <= p <= 3 for p in ports):
            raise ValueError(f"a circulator has ports 1, 2 and 3; got {list(driven)}")
        super().__init__(label=label, **params)
        self.driven = tuple(sorted(ports))
        self.inputs = {f"in{p}": PortType.OPTICAL for p in self.driven}
        self.outputs = {f"out{self._next(p)}": PortType.OPTICAL for p in self.driven}

    @staticmethod
    def _next(port: int) -> int:
        """Where light entering ``port`` comes out: 1 to 2, 2 to 3, 3 back to 1."""
        return port % 3 + 1

    def structural_config(self) -> dict[str, Any]:
        return {"driven": list(self.driven)}

    def _heard(self) -> dict[str, frozenset[str]]:
        """Which inputs each output hears: its forward hop, and the leak and echo if set.

        With neither, an output hears exactly one input and each is its own route,
        which is the whole point of this component. Isolation adds the input one
        port further round -- port 3 hears port 1's leak as well as port 2's
        forward path. Return loss adds the output's own port -- port 2 hears light
        coming back into port 2 -- and next to a reflective device that one is a
        genuine cavity.
        """
        heard: dict[str, frozenset[str]] = {}
        for port in self.driven:
            out = self._next(port)
            names = {f"in{port}"}
            leaker = self._next(out)
            if self.isolation > 0.0 and leaker in self.driven:
                names.add(f"in{leaker}")
            if self.return_loss > 0.0 and out in self.driven:
                names.add(f"in{out}")
            heard[f"out{out}"] = frozenset(names)
        return heard

    def port_groups(self) -> tuple[PortGroup, ...]:
        """One group per distinct set of inputs heard.

        Outputs hearing exactly the same inputs are one node, since nothing could
        order them apart. With return loss and isolation both set, ports 2 and 3
        each hear ports 1 and 2 and share a group -- and next to a grating on port 2
        that group sits on a real cycle, which only a Feedback can close.
        """
        by_inputs: dict[frozenset[str], set[str]] = {}
        for output, names in self._heard().items():
            by_inputs.setdefault(names, set()).add(output)
        return tuple(PortGroup(names, frozenset(outputs)) for names, outputs in by_inputs.items())

    def run(self, ctx: SimulationContext, inputs: dict[str, Signal]) -> dict[str, Signal]:
        # The scheduler calls this once per group, identified by exactly which
        # inputs arrived; port_groups guarantees no two groups read the same set.
        # Called directly with everything at once, it answers for every group those
        # inputs can feed.
        arrived = frozenset(inputs)
        groups = self.port_groups()
        chosen = [g for g in groups if g.inputs == arrived] or [
            g for g in groups if g.inputs <= arrived
        ]
        heard = self._heard()
        matrix_for = self._matrix_factory()
        produced: dict[str, Signal] = {}
        for group in chosen:
            for output in sorted(group.outputs):
                destination = output.removeprefix("out")
                # Everything arriving at one port adds as fields where it shares a
                # band: a reflected channel, the same channel's leak, and its echo.
                contributions = [
                    apply_response(
                        inputs[name],
                        port_response(matrix_for, "p" + destination, "p" + name.removeprefix("in")),
                    )
                    for name in sorted(heard[output])
                ]
                produced[output] = _sum_signals(contributions, where=f"{self.label}.{output}")
        return produced

    def _matrix_factory(self, polarization: str = "te") -> Callable[[np.ndarray], SMatrix]:
        # ``polarization`` is accepted and ignored: the model is one loss figure
        # and a routing, with nothing in it for an index to act on. A real
        # circulator's loss *is* slightly polarization dependent, and that number
        # is per part rather than per physics.
        loss = self.insertion_loss
        isolation = self.isolation
        echo = self.return_loss
        return solve_once(
            lambda f: circulator(
                f, insertion_loss_db=loss, isolation_db=isolation, return_loss_db=echo
            )
        )
