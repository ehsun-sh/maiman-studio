"""Optical fiber.

Attenuation, chromatic dispersion, the Kerr nonlinearity — self-phase, cross-phase
and four-wave mixing — and polarization-mode dispersion.

Model references: G. P. Agrawal, *Nonlinear Fiber Optics*, ch. 2-4 and 10 (NLSE,
GVD-induced broadening, SPM, solitons, XPM and FWM); ITU-T G.652 for typical
values.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from ..component import BoolParam, Component, Param, PortType
from ..context import SimulationContext
from ..kernels import (
    PMDSection,
    PropagationDiagnostics,
    apply_pmd,
    attenuation_db_per_m_to_alpha,
    differential_group_delay,
    dispersion_slope_to_beta3,
    dispersion_to_beta2,
    effective_length,
    fwm_accumulated_phase,
    fwm_mixing_integral,
    fwm_phase_mismatch,
    fwm_product_power,
    propagate_coupled_ssfm,
    propagate_dispersion,
    raman_tilt,
    random_pmd_sections,
    walkoff_from_dispersion,
)
from ..signals import Band, OpticalSignal, Signal
from ..units import db_to_linear

#: Two mixing products this close in frequency are the same wave [Hz]. Real
#: products are separated by channel spacings — gigahertz — so any tolerance
#: between floating-point noise on a 193 THz sum and a gigahertz would do.
MIXING_MERGE_TOLERANCE = 1e3


class Fiber(Component):
    """Single-mode fiber: attenuation, chromatic dispersion, and the Kerr effect.

    With ``nonlinearity`` left at zero the propagation is linear and is solved
    exactly in one frequency-domain step — no stepping error at all. Give it a
    nonzero value and the same span is solved by split-step Fourier instead,
    which is approximate, so the block reports what it did on its
    ``diagnostics`` port.

    Dispersion is applied per band, using each band's *own* centre wavelength to
    compute beta2. Two channels a few nanometres apart really do see different
    dispersion, and because every band carries its own centre frequency the model
    gets that right for free — a single-carrier signal model could not express it.

    **The slope is what makes that difference real.** ``dispersion`` and
    ``dispersion_slope`` are both quoted at ``reference_wavelength``, and D at
    any other wavelength is the first of them plus the second times the offset.
    Left at zero the slope changes nothing: D is flat, beta3 is switched off, and
    every number this block produced before the slope existed still comes out.
    Turn it on and two things happen at once, because they are the same
    coefficient seen from two directions — channels across a comb stop sharing
    one dispersion, which is why a single compensator cannot flatten a C-band,
    and each channel's own spectrum picks up a cubic phase, which broadens a
    pulse *asymmetrically* where beta2 broadens it evenly.

    **Channels interact.** Bands do not propagate independently: each one's phase
    is rotated by every other one's power, at twice the rate its own power
    rotates it, and triplets of them mix to generate light at new frequencies.
    Both fall out of the same ``|A|**2 A`` term that produces self-phase
    modulation — see :func:`maiman.kernels.propagate_coupled_ssfm` for the count
    that gives cross-phase modulation its factor of two, and
    :func:`maiman.kernels.fwm_product_power` for the one that gives four-wave
    mixing its degeneracy factor.

    The two are computed differently, and the difference is worth understanding
    before reading a number off this block. Cross-phase modulation is solved on
    the waveform, inside the split-step loop, because it depends on the
    neighbour's instantaneous power sliding past under walk-off; it therefore
    shows up as a real distortion of the constellation. Four-wave mixing is
    solved in closed form from the band powers and injected as tones, because
    the products land at frequencies no band is sampled at — the whole reason
    this simulator can afford a WDM comb at all is that it never puts the
    channels on one grid, and that choice has to be paid for somewhere.

    **Walk-off is not a parameter.** It is the group-delay term of the same
    expansion of ``beta(omega)`` that gives the dispersion, so setting
    ``dispersion`` to zero both removes the pulse spreading and stops the
    channels sliding past one another. That is the correct coupling and it is
    the reason a dispersion-shifted fiber operated at its zero is the worst
    place to put a WDM comb: nothing averages the cross-phase modulation away
    and nothing dephases the mixing products.

    PMD is drawn as a random realisation, not applied as a fixed impairment,
    because that is what it is: birefringence varies along real fiber and drifts
    with temperature, so the differential group delay is a random variable and a
    link is designed against an outage probability rather than a worst case. The
    phase a mixing product arrives with is treated the same way and for the same
    reason — it is set by fiber details nobody measures — so it is drawn from
    the run's generator, which makes a single run reproducible and a sweep with
    repeats an exploration of the distribution.

    **The polarizations couple only if asked.** By default each is propagated as
    its own scalar problem, which is what every result in this project was taken
    with. Set ``cross_polarization`` and orthogonal power enters the Kerr term at
    two thirds the co-polarized weight — so a neighbour polarized across the
    channel modulates it at exactly one third of the rate a co-polarized one
    does, and the two axes of one channel, no longer accumulating the same phase,
    rotate the state of polarization as the power moves. With all the light on
    one axis the setting changes nothing, which is why it is safe to leave on for
    a dual-polarization link and pointless for a single-polarization one.

    **Stimulated Raman scattering tilts the comb.** A photon can scatter off a
    silica vibration and come out at a lower frequency, and the process is
    stimulated, so the short-wavelength channels pump the long-wavelength ones
    and a flat launch does not arrive flat. Set ``raman_gain_slope`` and a filled
    C band — 80 channels at 0 dBm on a 50 GHz grid — comes out of one 80 km span
    with 0.81 dB between its ends, which is a large fraction of the margin a link
    is designed with. Power is moved, not lost: the sum over channels is
    unchanged to floating point.

    **Mixing products add coherently from span to span.** The signal carries the
    dispersion the path has accumulated, which is all it takes to say how far a
    product generated in one span has rotated away from its pumps by the time the
    next span generates another — four lossless spans give sixteen times one
    span's product where adding in power would give four. What is *not* tracked
    is the pumps' own nonlinear phase, so the interference between spans is
    computed from the linear mismatch alone.

    Not yet modelled: the coherent polarization term
    ``A_x* A_y**2``, which exchanges power between the axes rather than only
    dephasing them, is left out — it is the part that averages away first under
    real birefringence. The Raman gain is taken as rising linearly
    with separation, which it does up to about 13 THz and not past it, so a comb
    spanning the C and L bands together has its far pairs over the peak and the
    transfer between them over-predicted. Pump depletion is not modelled, which
    matters only at powers no link is operated at.
    """

    display_name = "Optical Fiber"
    category = "Fiber"

    length = Param(80.0, unit="km", min=0.0, doc="Fiber span length")
    attenuation = Param(0.2, unit="dB/km", min=0.0, doc="Attenuation coefficient")
    dispersion = Param(
        0.0, unit="ps/nm/km", doc="Dispersion parameter D at the reference wavelength (0 disables)"
    )
    dispersion_slope = Param(
        0.0,
        unit="ps/nm^2/km",
        doc="Slope dD/dlambda at the reference wavelength (0.058 is typical for SSMF)",
    )
    reference_wavelength = Param(
        1550.0, unit="nm", min=1.0, doc="Wavelength the dispersion and its slope are quoted at"
    )
    nonlinearity = Param(
        0.0, unit="1/W/km", min=0.0, doc="Kerr coefficient gamma (0 disables, and is exact)"
    )
    max_nonlinear_phase = Param(
        0.005,
        unit="",
        min=1e-6,
        doc="Largest nonlinear phase rotation allowed per split-step [rad]",
    )
    cross_phase_modulation = BoolParam(
        True, doc="Couple the bands: each is phase-modulated by the others' power"
    )
    cross_polarization = BoolParam(
        False, doc="Couple the two polarizations: orthogonal power modulates at two thirds"
    )
    max_walkoff_slip = Param(
        0.5,
        unit="",
        min=1e-3,
        doc="Largest relative slip between bands allowed per split-step [samples]",
    )
    four_wave_mixing = BoolParam(True, doc="Generate mixing products between bands")
    mixing_floor = Param(
        70.0,
        unit="dB",
        min=0.0,
        doc="Discard mixing products this far below the strongest band",
    )
    raman_gain_slope = Param(
        0.0,
        unit="1/W/km/THz",
        min=0.0,
        doc="Raman gain slope C_R (0 disables; 0.028 is typical for SSMF at 1550 nm)",
    )
    pmd_coefficient = Param(0.0, unit="ps/sqrt(km)", min=0.0, doc="PMD coefficient (0 disables)")
    pmd_sections = Param(60.0, unit="", min=1.0, doc="Waveplates used to build the PMD realisation")

    inputs = {"in": PortType.OPTICAL}
    outputs = {"out": PortType.OPTICAL, "diagnostics": PortType.METRIC}

    def loss_db(self) -> float:
        """Total span loss [dB]."""
        # si() gives dB/m and metres, so their product is dB.
        return self.si("attenuation") * self.si("length")

    def dispersion_at(self, wavelength: float) -> float:
        """Dispersion parameter D [s/m²] at ``wavelength`` [m].

        ``D(lambda) = D_ref + S * (lambda - lambda_ref)``, linear because that is
        all one slope can say. With the slope left at zero D is the same at every
        wavelength, which is what this block used to assume without a reference
        wavelength to hang the assumption on.
        """
        return self.si("dispersion") + self.si("dispersion_slope") * (
            wavelength - self.si("reference_wavelength")
        )

    def beta2_at(self, wavelength: float) -> float:
        """Group-velocity dispersion beta2 [s^2/m] at ``wavelength`` [m]."""
        return dispersion_to_beta2(self.dispersion_at(wavelength), wavelength)

    def beta3_at(self, wavelength: float) -> float:
        """Third-order dispersion beta3 [s^3/m] at ``wavelength`` [m].

        Nonzero even at zero slope, because holding D flat across wavelength is
        itself a statement about how beta2 varies — see
        :func:`maiman.kernels.dispersion_slope_to_beta3`. The block reproduces
        its own history anyway: with ``dispersion_slope`` at its default the
        cubic term is switched off entirely rather than set to that residue, so
        every result taken before the slope existed still holds exactly.
        """
        if self.si("dispersion_slope") == 0.0:
            return 0.0
        return dispersion_slope_to_beta3(
            self.dispersion_at(wavelength), self.si("dispersion_slope"), wavelength
        )

    def mean_dgd(self) -> float:
        """Expected differential group delay over this span [s].

        ``<DGD> = PMD_coefficient * sqrt(L)`` — the square root, not the length,
        because birefringence axes reorient randomly and the delay accumulates
        as a random walk rather than a sum.
        """
        return self.si("pmd_coefficient") * math.sqrt(self.si("length"))

    def reference_beta2(self, signal: OpticalSignal) -> float:
        """The beta2 the four-wave mixing bookkeeping is written against [s²/m].

        One value for the whole comb, taken at the first band's wavelength — the
        same band the mismatch offsets are measured from, so the two cannot
        disagree. It has to be the *first* rather than, say, the lowest in
        frequency, because mixing products are appended to the band list and some
        of them land below the comb: a rule that looked at the extremes would
        pick a different reference in the second span than in the first, and the
        accumulated phase would be measured against a moving post.
        """
        if not signal.bands:
            return 0.0
        return self.beta2_at(signal.bands[0].wavelength)

    def walkoff_of(self, band: Band, reference: Band) -> float:
        """Inverse-group-velocity offset of ``band`` against ``reference`` [s/m].

        Evaluated with beta2 averaged over the two wavelengths, which is the
        trapezoidal value of ``integral beta2 domega`` and so is symmetric in
        the pair — using either end's beta2 alone would make the delay from A to
        B differ from the delay from B to A.

        With a dispersion slope beta2 varies linearly with frequency, and the
        trapezoidal rule is *exact* for a linear integrand. So the averaging that
        was a symmetry argument when there was no slope to justify it is now the
        correct answer rather than a defensible one.
        """
        if band.f0 == reference.f0:
            return 0.0
        beta2 = 0.5 * (self.beta2_at(band.wavelength) + self.beta2_at(reference.wavelength))
        return walkoff_from_dispersion(beta2, band.f0 - reference.f0)

    def run(self, ctx: SimulationContext, inputs: dict[str, Signal]) -> dict[str, Signal]:
        signal: OpticalSignal = inputs["in"]
        distance = self.si("length")
        gamma = self.si("nonlinearity")
        alpha = attenuation_db_per_m_to_alpha(self.si("attenuation"))
        power_factor = db_to_linear(-self.loss_db())

        mean_dgd = self.mean_dgd()
        sections: tuple[PMDSection, ...] = ()
        realised_dgd = 0.0
        if mean_dgd > 0.0:
            sections = random_pmd_sections(
                mean_dgd,
                int(self.pmd_sections),
                ctx.rng("Fiber", self.label, "pmd"),
            )
            realised_dgd = differential_group_delay(sections)

        if gamma == 0.0:
            fields, diagnostics = self._propagate_linear(signal, distance, power_factor)
        else:
            fields, diagnostics = self._propagate_kerr(signal, distance, gamma, alpha)

        if sections:
            # PMD is applied after dispersion and the Kerr effect rather than
            # interleaved with them. That neglects the interaction between
            # nonlinearity and a rotating polarization state, which matters at
            # high power over long spans and does not at the powers and
            # distances this is usually pointed at.
            fields = [
                apply_pmd(ex, ey, band.fs, sections)
                for (ex, ey), band in zip(fields, signal.bands, strict=True)
            ]
        diagnostics = replace(diagnostics, differential_group_delay=realised_dgd)

        bands = [
            Band(Ex=ex.astype(ctx.complex_dtype), Ey=ey.astype(ctx.complex_dtype), f0=b.f0, fs=b.fs)
            for (ex, ey), b in zip(fields, signal.bands, strict=True)
        ]

        bands, tilt = self._raman(signal, bands, alpha=alpha, distance=distance)
        diagnostics = replace(diagnostics, raman_tilt=tilt)

        if gamma != 0.0 and self.four_wave_mixing:
            bands, emitted = self._mix(ctx, signal, bands, gamma=gamma, alpha=alpha)
            diagnostics = replace(diagnostics, mixing_products=emitted)

        return {
            "out": OpticalSignal(
                bands=tuple(bands),
                noise=tuple(n.scale_power(power_factor) for n in signal.noise),
                # The span's own contribution to the path history, which is what
                # tells the *next* span how far these products have already
                # rotated away from their pumps.
                accumulated_gvd=signal.accumulated_gvd + self.reference_beta2(signal) * distance,
            ),
            "diagnostics": diagnostics,
        }

    # -- propagation ------------------------------------------------------

    def _propagate_linear(
        self, signal: OpticalSignal, distance: float, power_factor: float
    ) -> tuple[list[tuple[np.ndarray, np.ndarray]], PropagationDiagnostics]:
        """Closed-form propagation. Linear means exact; do not approximate it.

        Walk-off is a linear operator too, but it is a constant group delay per
        band and every band is reported in its own retarded frame, so it has no
        observable effect without a nonlinearity to interact with. Leaving it
        out here is not an omission; applying it would be work that cancels.
        """
        amplitude = power_factor**0.5
        fields = [
            (
                propagate_dispersion(
                    band.Ex,
                    band.fs,
                    self.beta2_at(band.wavelength),
                    distance,
                    self.beta3_at(band.wavelength),
                )
                * amplitude,
                propagate_dispersion(
                    band.Ey,
                    band.fs,
                    self.beta2_at(band.wavelength),
                    distance,
                    self.beta3_at(band.wavelength),
                )
                * amplitude,
            )
            for band in signal.bands
        ]
        return fields, PropagationDiagnostics(0, distance, 0.0, 0.0, 0.0)

    def _propagate_kerr(
        self, signal: OpticalSignal, distance: float, gamma: float, alpha: float
    ) -> tuple[list[tuple[np.ndarray, np.ndarray]], PropagationDiagnostics]:
        """Split-step propagation, with the bands coupled unless told otherwise.

        By default the two polarizations are propagated as two separate coupled
        systems, which is what makes the model scalar per polarization: a band's
        X field is modulated by every other band's X power and by nothing else.
        Set ``cross_polarization`` and both axes go into one call, labelled, so
        that orthogonal power enters at :data:`maiman.kernels.ORTHOGONAL_KERR_WEIGHT`.

        The two axes of one band share a linear operator — the same beta2, the
        same slope, the same walk-off — because birefringence is this block's
        other business and is applied as a separate element afterwards. What the
        coupling adds is entirely in the nonlinear step.
        """
        bands = signal.bands
        coupled = self.cross_phase_modulation and len(bands) > 1
        if coupled:
            grids = {(band.fs, band.num_samples) for band in bands}
            if len(grids) != 1:
                raise ValueError(
                    f"{self.label or 'Fiber'}: cross-phase modulation couples bands sample by "
                    f"sample and needs one common time grid, but the input carries {len(grids)} "
                    "different ones. Give the channels a common symbol rate and oversampling, "
                    "or set cross_phase_modulation=False to propagate them independently."
                )

        groups = [bands] if coupled else [(band,) for band in bands]
        fields: list[tuple[np.ndarray, np.ndarray]] = []
        diagnostics = PropagationDiagnostics(0, distance, 0.0, 0.0, 0.0)
        for group in groups:
            reference = group[0]
            beta2 = [self.beta2_at(band.wavelength) for band in group]
            beta3 = [self.beta3_at(band.wavelength) for band in group]
            walkoff = [self.walkoff_of(band, reference) for band in group]

            if self.cross_polarization:
                # One system, both axes, labelled. The per-field lists are
                # doubled rather than special-cased in the solver: X and Y of the
                # same band see the same linear operator.
                out, diagnostics = self._solve(
                    [band.Ex for band in group] + [band.Ey for band in group],
                    reference.fs,
                    beta2=beta2 * 2,
                    beta3=beta3 * 2,
                    walkoff=walkoff * 2,
                    polarization=[0] * len(group) + [1] * len(group),
                    gamma=gamma,
                    alpha=alpha,
                    distance=distance,
                    best=diagnostics,
                )
                fields.extend(zip(out[: len(group)], out[len(group) :], strict=True))
                continue

            solved = []
            for axis in ("Ex", "Ey"):
                out, diagnostics = self._solve(
                    [getattr(band, axis) for band in group],
                    reference.fs,
                    beta2=beta2,
                    beta3=beta3,
                    walkoff=walkoff,
                    polarization=None,
                    gamma=gamma,
                    alpha=alpha,
                    distance=distance,
                    best=diagnostics,
                )
                solved.append(out)
            fields.extend(zip(solved[0], solved[1], strict=True))
        return fields, diagnostics

    def _solve(
        self,
        fields: list[np.ndarray],
        sample_rate: float,
        *,
        beta2: list[float],
        beta3: list[float],
        walkoff: list[float],
        polarization: list[int] | None,
        gamma: float,
        alpha: float,
        distance: float,
        best: PropagationDiagnostics,
    ) -> tuple[list[np.ndarray], PropagationDiagnostics]:
        """One call into the solver, keeping whichever diagnostics took more steps."""
        out, diag = propagate_coupled_ssfm(
            fields,
            sample_rate,
            beta2=beta2,
            walkoff=walkoff,
            gamma=gamma,
            beta3=beta3,
            polarization=polarization,
            alpha=alpha,
            distance=distance,
            max_nonlinear_phase=self.max_nonlinear_phase,
            max_walkoff_slip=self.max_walkoff_slip,
        )
        return out, diag if diag.steps > best.steps else best

    # -- stimulated Raman scattering --------------------------------------

    def _raman(
        self,
        signal: OpticalSignal,
        bands: list[Band],
        *,
        alpha: float,
        distance: float,
    ) -> tuple[list[Band], float]:
        """Move power from the short wavelengths to the long ones.

        Applied to the propagated bands as one redistribution rather than
        integrated along the span, which is what the closed form in
        :func:`maiman.kernels.raman_tilt` is for. The ratios come from the
        *launched* powers, because that is what the formula is written in terms
        of and because the loss is common to every channel and cancels out of it.

        Mixing products are added afterwards and are not tilted. They are forty
        decibels down and a decibel of tilt on them is a hundredth of a decibel
        anywhere it could be measured, but it is an approximation and not an
        oversight.
        """
        slope = self.si("raman_gain_slope")
        if slope <= 0.0 or len(bands) < 2 or distance <= 0.0:
            return bands, 0.0

        ratios = raman_tilt(
            [band.f0 for band in signal.bands],
            [band.average_power() for band in signal.bands],
            gain_slope=slope,
            effective_length=effective_length(alpha, distance),
        )
        order = sorted(range(len(ratios)), key=lambda index: signal.bands[index].f0)
        lowest, highest = ratios[order[0]], ratios[order[-1]]
        tilt = 10.0 * math.log10(lowest / highest) if highest > 0.0 else 0.0

        scaled = [
            replace(band, Ex=band.Ex * math.sqrt(ratio), Ey=band.Ey * math.sqrt(ratio))
            for band, ratio in zip(bands, ratios, strict=True)
        ]
        return scaled, tilt

    # -- four-wave mixing -------------------------------------------------

    def _mix(
        self,
        ctx: SimulationContext,
        signal: OpticalSignal,
        bands: list[Band],
        *,
        gamma: float,
        alpha: float,
    ) -> tuple[list[Band], int]:
        """Add the mixing products this span generated to the propagated bands.

        Products are accumulated as complex amplitudes rather than as powers, so
        that several triplets landing on one frequency add the way waves do. A
        product whose frequency coincides with a band already present is added
        *into* that band, at DC in its own rotating frame, which is what makes
        in-band mixing the unfilterable crosstalk it is: on a uniform grid every
        product lands on a channel, and no receiver can separate it afterwards.

        **Across spans the addition is coherent, and that is not a detail.** A
        product's phase here is three things multiplied together. The argument of
        the mixing integral, which is physics and differs between triplets
        because their mismatches do. The accumulated mismatch over the fibre
        already behind this span, from :func:`maiman.kernels.fwm_accumulated_phase`
        — this is what makes span two's contribution add to span one's rather
        than beside it, and over four lossless spans it is the difference between
        four times one span's product and sixteen times it. And a phase drawn per
        *triplet*, standing in for the pump phase combination that the power-only
        treatment of the pumps has thrown away.

        That draw is keyed on the three pump frequencies and on nothing else — not
        on the block's label, not on which span it is. It has to be: the same
        three pumps produce the same product wherever they are, and a phase that
        was redrawn every span would make the spans add in power, which is
        exactly the thing this is not doing any more.

        What is still missing is the pumps' *nonlinear* phase: self- and
        cross-phase modulation rotate the pump combination span by span, and only
        the linear mismatch is tracked here. At a tenth of a radian per span it
        moves the interference between spans a little; it does not change which
        regime the link is in.
        """
        sources = signal.bands
        distance = self.si("length")
        if len(sources) < 2 or distance <= 0.0:
            return bands, 0

        strongest = max((band.average_power() for band in sources), default=0.0)
        if strongest <= 0.0:
            return bands, 0
        # Raw dB: the unit machinery already turns a dB parameter into a linear
        # ratio, and converting a second time would silently move the floor.
        floor = strongest * db_to_linear(-self.mixing_floor)

        reference = sources[0]
        beta2 = self.reference_beta2(signal)
        travelled = signal.accumulated_gvd
        powers = [
            (float(np.mean(np.abs(b.Ex) ** 2)), float(np.mean(np.abs(b.Ey) ** 2))) for b in sources
        ]

        found: list[tuple[float, complex, complex]] = []
        for i in range(len(sources)):
            for j in range(i, len(sources)):
                for k in range(len(sources)):
                    if k in (i, j):
                        # The product frequency is one of the pumps: this term
                        # is cross-phase modulation, and the split-step already
                        # applied it. Counting it here would double it.
                        continue
                    frequency = sources[i].f0 + sources[j].f0 - sources[k].f0
                    if frequency <= 0.0:
                        continue
                    mismatch = fwm_phase_mismatch(
                        beta2,
                        sources[i].f0 - reference.f0,
                        sources[j].f0 - reference.f0,
                        sources[k].f0 - reference.f0,
                    )
                    generated = [
                        fwm_product_power(
                            powers[i][axis],
                            powers[j][axis],
                            powers[k][axis],
                            gamma=gamma,
                            alpha=alpha,
                            distance=distance,
                            phase_mismatch=mismatch,
                            degenerate=i == j,
                        )
                        for axis in (0, 1)
                    ]
                    if sum(generated) < floor:
                        continue
                    offsets = (
                        sources[i].f0 - reference.f0,
                        sources[j].f0 - reference.f0,
                        sources[k].f0 - reference.f0,
                    )
                    # Drawn on the pumps, not on the fibre: the same triplet gets
                    # the same phase in every span, which is what lets the spans
                    # add rather than average.
                    drawn = float(ctx.rng("Fiber", "fwm", *offsets).random())
                    phase = (
                        2.0 * np.pi * drawn
                        + float(np.angle(fwm_mixing_integral(mismatch, alpha, distance)))
                        + fwm_accumulated_phase(travelled, *offsets)
                    )
                    phasor = np.exp(1j * phase)
                    found.append(
                        (
                            frequency,
                            math.sqrt(generated[0]) * phasor,
                            math.sqrt(generated[1]) * phasor,
                        )
                    )

        if not found:
            return bands, 0

        frequencies: list[float] = []
        amplitudes: list[list[complex]] = []
        for frequency, amp_x, amp_y in found:
            for slot, existing in enumerate(frequencies):
                if abs(existing - frequency) <= MIXING_MERGE_TOLERANCE:
                    amplitudes[slot][0] += amp_x
                    amplitudes[slot][1] += amp_y
                    break
            else:
                frequencies.append(frequency)
                amplitudes.append([amp_x, amp_y])

        output = list(bands)
        for frequency, (amp_x, amp_y) in zip(frequencies, amplitudes, strict=True):
            for index, band in enumerate(output):
                if abs(band.f0 - frequency) <= MIXING_MERGE_TOLERANCE:
                    output[index] = replace(
                        band,
                        Ex=(band.Ex + amp_x).astype(band.Ex.dtype),
                        Ey=(band.Ey + amp_y).astype(band.Ey.dtype),
                    )
                    break
            else:
                shape = reference.num_samples
                output.append(
                    Band(
                        Ex=np.full(shape, amp_x, dtype=ctx.complex_dtype),
                        Ey=np.full(shape, amp_y, dtype=ctx.complex_dtype),
                        f0=frequency,
                        fs=reference.fs,
                    )
                )
        return output, len(frequencies)
