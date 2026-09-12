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

from collections.abc import Callable
from typing import Any

import numpy as np

from ..circuit import SMatrix
from ..component import Param, PortGroup, PortType
from ..context import SimulationContext
from ..photonics import (
    APODIZATIONS,
    DEFAULT_GRATING_SECTIONS,
    SILICA_FIBER_NEFF,
    circulator,
    fiber_bragg_grating,
)
from ..signals import OpticalSignal, Signal
from ..units import C_LIGHT, wavelength_to_frequency
from .photonic import ScatteringDevice, apply_response, port_response, solve_once


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

    **The sign, and a warning about it.** A positive ``chirp`` puts the short
    wavelengths at the near end where they turn round early, so the long ones
    arrive later: ``dtau/dlambda > 0``, which is ``D > 0``, the sign standard
    fibre has. By that reading a compensator is a *negative* chirp, and entering
    the same grating from its far end gives the other sign — which is why
    ``reflected`` is computed from the input face rather than from whichever end
    is convenient.

    **Against this engine's** :class:`~maiman.components.Fiber` **it is the other
    way round, and that is not this block's doing.** A positive chirp is what
    compensates a span here, measured: a 30 ps pulse broadened from 21.2 ps rms
    to 46.1 ps over 80 km comes back to 21.3 ps through ``chirp = +0.71`` nm. The
    two blocks disagree because :func:`maiman.kernels.propagate_dispersion`
    carries the opposite quadratic sign from
    :func:`maiman.photonics.propagation_constant`, which that function's docstring
    has said in as many words since before this device existed — the grating is
    simply the first component to put the two in one graph where it shows.
    ``test_the_compensating_chirp_sign_is_pinned`` holds the behaviour as it
    stands, so that reconciling the kernel reports this as one of the things it
    changed rather than silently inverting every compensator built on it.

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

    def bragg_frequency(self) -> float:
        """The frequency it reflects [Hz]."""
        return wavelength_to_frequency(self.si("bragg_wavelength"))

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
        wavelength = self.si("bragg_wavelength")
        index = self.n_eff
        chirp = self.si("chirp")
        profile = self.apodization
        pieces = int(self.sections)
        return solve_once(
            lambda f: fiber_bragg_grating(
                f,
                length=length,
                index_modulation=modulation,
                bragg_wavelength=wavelength,
                n_eff=index,
                chirp=chirp,
                apodization=profile,
                sections=pieces,
            )
        )


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

    See :func:`maiman.photonics.circulator` for the matrix, and for why isolation
    is deliberately not a parameter here.
    """

    display_name = "Circulator"
    category = "Passive"

    insertion_loss = Param(
        0.7, unit="dB", min=0.0, doc="Loss of one hop; paid again on the next hop"
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

    def port_groups(self) -> tuple[PortGroup, ...]:
        """One group per hop. The whole point of this component."""
        return tuple(
            PortGroup(frozenset({f"in{p}"}), frozenset({f"out{self._next(p)}"}))
            for p in self.driven
        )

    def run(self, ctx: SimulationContext, inputs: dict[str, Signal]) -> dict[str, Signal]:
        # Called once per group, so ``inputs`` holds one port and the answer
        # holds the one it routes to. Written over whatever arrived rather than
        # over ``self.driven``, because the group decides what this call is for.
        matrix_for = self._matrix_factory()
        produced: dict[str, Signal] = {}
        for name in inputs:
            source = int(name.removeprefix("in"))
            destination = self._next(source)
            produced[f"out{destination}"] = apply_response(
                inputs[name], port_response(matrix_for, f"p{destination}", f"p{source}")
            )
        return produced

    def _matrix_factory(self, polarization: str = "te") -> Callable[[np.ndarray], SMatrix]:
        # ``polarization`` is accepted and ignored: the model is one loss figure
        # and a routing, with nothing in it for an index to act on. A real
        # circulator's loss *is* slightly polarization dependent, and that number
        # is per part rather than per physics.
        loss = self.insertion_loss
        return solve_once(lambda f: circulator(f, insertion_loss_db=loss))
