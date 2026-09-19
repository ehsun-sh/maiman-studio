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
    ORTHOGONAL_KERR_WEIGHT,
    RAMAN_TRIANGLE_LIMIT,
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
    fwm_nonlinear_rate,
    fwm_phase_mismatch,
    fwm_power_transfer,
    fwm_product_power,
    kerr_rate,
    propagate_coupled_ssfm,
    propagate_dispersion,
    raman_tilt,
    raman_transfer,
    random_pmd_sections,
    walkoff_from_dispersion,
)
from ..signals import Band, KerrHistory, OpticalSignal, Signal
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
    span's product where adding in power would give four.

    **And, with** ``pump_phase``, **the pumps' own nonlinear phase.** Self- and
    cross-phase modulation turn a product's drive and the product itself at
    different rates, ``-gamma (P_i + P_j - P_k)`` apart with cross-phase on, and
    that shifts the mismatch inside a span and the interference between spans.
    Every carrier carries its share of it on the signal, as the dispersion is
    carried. Against the split-step solution of a strong pump and a weak signal,
    a quarter of a radian of nonlinear phase takes the linear-mismatch product
    12--29 % off, depending on the sign of the dispersion; with it, 2 %.

    **The coherent polarization term, and PMD along the span.** With
    ``cross_polarization`` on, ``coherent_polarization`` adds the ``A_x* A_y**2``
    term that moves power between the axes rather than only dephasing them:
    circularly polarized light then picks up two thirds of the nonlinear phase a
    linearly polarized beam does, where the phase-only form gives it five sixths,
    and an elliptical state's axes rotate at ``(2/3) gamma S3`` per metre.
    ``interleave_pmd`` applies the PMD waveplates along the span, between Kerr
    steps, instead of all after it.

    **What neither changes is the Manakov average**, and that is worth knowing
    before expecting it to. With enough waveplates to scramble the state quickly
    the averaged nonlinearity lands on 8/9 of ``gamma P`` with or without the
    coherent term: averaged over every polarization state, the two forms have the
    same invariant part, and the coherent term's contribution is only to how the
    state evolves along the way. Both are off by default, so no earlier result
    moves.

    **The pumps pay for what they make.** Every mixing product takes its photons
    from the two pumps that made it and gives one to the idler, so a span with no
    loss comes out with exactly the power it was given. The product's own power
    is still the undepleted-pump formula, which is right while it is small against
    the pumps; ``diagnostics.fwm_depletion`` says how small.

    **Raman past the peak.** Up to 13.2 THz the gain rises linearly and the
    closed form is exact. A wider comb — the S, C and L bands together; C and L
    alone are 9.7 THz and stay inside — is integrated instead, with silica's
    measured gain shape and photons rather than watts conserved.
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
    coherent_polarization = BoolParam(
        False, doc="Keep the coherent A_x* A_y^2 term, which moves power between the axes"
    )
    interleave_pmd = BoolParam(
        False, doc="Apply PMD along the span between Kerr steps, rather than after it"
    )
    max_walkoff_slip = Param(
        0.5,
        unit="",
        min=1e-3,
        doc="Largest relative slip between bands allowed per split-step [samples]",
    )
    four_wave_mixing = BoolParam(True, doc="Generate mixing products between bands")
    pump_phase = BoolParam(
        False,
        doc="Let the pumps' own SPM and XPM phase shift the mixing, within and between spans",
        applies_when="four_wave_mixing",
    )
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

        # Along the span only where it can make a difference: a linear span's
        # operators commute, so there the chain is applied afterwards, identically.
        interleaved = bool(sections) and gamma != 0.0 and self.interleave_pmd
        if gamma == 0.0:
            fields, diagnostics = self._propagate_linear(signal, distance, power_factor)
        else:
            fields, diagnostics = self._propagate_kerr(
                signal, distance, gamma, alpha, sections if interleaved else ()
            )

        if sections and not interleaved:
            # Applied after dispersion and the Kerr effect. That neglects the Kerr
            # effect acting on a state of polarization that is still rotating;
            # set interleave_pmd to apply the waveplates along the span instead.
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

        history = self._nonlinear_history(signal, gamma=gamma, alpha=alpha)
        if gamma != 0.0 and self.four_wave_mixing:
            bands, emitted, depleted = self._mix(
                ctx, signal, bands, gamma=gamma, alpha=alpha, after=history
            )
            diagnostics = replace(diagnostics, mixing_products=emitted, fwm_depletion=depleted)

        return {
            "out": OpticalSignal(
                bands=tuple(bands),
                noise=tuple(n.scale_power(power_factor) for n in signal.noise),
                # The span's own contribution to the path history, which is what
                # tells the *next* span how far these products have already
                # rotated away from their pumps.
                accumulated_gvd=signal.accumulated_gvd + self.reference_beta2(signal) * distance,
                nonlinear_history=history,
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
        self,
        signal: OpticalSignal,
        distance: float,
        gamma: float,
        alpha: float,
        pmd: tuple[PMDSection, ...] = (),
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
        if (self.coherent_polarization or pmd) and not self.cross_polarization:
            raise ValueError(
                f"{self.label or 'Fiber'}: the coherent polarization term and PMD along the "
                "span both act on the two axes together, so they need cross_polarization on; "
                "with the axes propagated as independent problems neither has anything to act on"
            )
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
                    pairs=[(index, index + len(group)) for index in range(len(group))],
                    coherent_polarization=self.coherent_polarization,
                    pmd=pmd,
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
        pairs: list[tuple[int, int]] | None = None,
        coherent_polarization: bool = False,
        pmd: tuple[PMDSection, ...] = (),
    ) -> tuple[list[np.ndarray], PropagationDiagnostics]:
        """One call into the solver, keeping whichever diagnostics took more steps."""
        out, diag = propagate_coupled_ssfm(
            fields,
            sample_rate,
            on_progress=self.report,
            beta2=beta2,
            walkoff=walkoff,
            gamma=gamma,
            beta3=beta3,
            polarization=polarization,
            pairs=pairs,
            coherent_polarization=coherent_polarization,
            pmd=pmd or None,
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

        A comb no wider than the gain peak uses the closed form, which is exact
        there and is what every result taken before the integrated model existed
        was taken with. A wider one is integrated with silica's measured shape,
        because a straight line past the peak over-predicts the far pairs several
        times over.
        """
        slope = self.si("raman_gain_slope")
        if slope <= 0.0 or len(bands) < 2 or distance <= 0.0:
            return bands, 0.0

        frequencies = [band.f0 for band in signal.bands]
        launched = [band.average_power() for band in signal.bands]
        length = effective_length(alpha, distance)
        if max(frequencies) - min(frequencies) <= RAMAN_TRIANGLE_LIMIT:
            ratios = raman_tilt(frequencies, launched, gain_slope=slope, effective_length=length)
        else:
            ratios = raman_transfer(
                frequencies, launched, gain_slope=slope, effective_length=length
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

    def _couples_bands(self, signal: OpticalSignal) -> bool:
        """Whether the bands share one split-step solve, and so cross-phase modulate."""
        return self.cross_phase_modulation and len(signal.bands) > 1

    def _nonlinear_history(
        self, signal: OpticalSignal, *, gamma: float, alpha: float
    ) -> KerrHistory:
        """The path's Kerr history with this span added: each carrier's ``rate * L_eff``.

        Every carrier present is turned by :func:`maiman.kernels.kerr_rate` at the
        span's start, carried down it by the loss as ``L_eff``; a carrier the
        history has not seen starts from the weak carrier's angle, because that is
        what it was turned by on the way here. The weak carrier itself -- where a
        new mixing product sits -- is turned by the cross-phase of everything
        else, and by nothing when each band is propagated alone. Kept whether or
        not this span mixes: the Kerr phase is there either way.
        """
        before = signal.nonlinear_history
        if gamma == 0.0 or not signal.bands:
            return before
        if self.four_wave_mixing and self.pump_phase and before.conflict:
            raise ValueError(
                f"{self.label or 'Fiber'}: {before.conflict}, and pump_phase needs one history "
                "to say how far a product has turned from its pumps. Combine the channels "
                "before the fibre, or set pump_phase=False on the spans downstream."
            )
        length = effective_length(alpha, self.si("length"))
        weight = ORTHOGONAL_KERR_WEIGHT if self.cross_polarization else 0.0
        coupled = self._couples_bands(signal)
        powers = [
            (float(np.mean(np.abs(b.Ex) ** 2)), float(np.mean(np.abs(b.Ey) ** 2)))
            for b in signal.bands
        ]
        total = (sum(p[0] for p in powers), sum(p[1] for p in powers))
        weak_rate = (
            kerr_rate(gamma, (0.0, 0.0), total, cross_phase=True, orthogonal_weight=weight)
            if coupled
            else (0.0, 0.0)
        )
        carriers = {f0: (x, y) for f0, x, y in before.carriers}
        for band, power in zip(signal.bands, powers, strict=True):
            rate = kerr_rate(gamma, power, total, cross_phase=coupled, orthogonal_weight=weight)
            start = before.at(band.f0)
            for known in [f for f in carriers if abs(f - band.f0) <= 1e-9 * band.f0]:
                del carriers[known]
            carriers[band.f0] = (start[0] + rate[0] * length, start[1] + rate[1] * length)
        return KerrHistory(
            carriers=tuple((f0, x, y) for f0, (x, y) in sorted(carriers.items())),
            weak=(before.weak[0] + weak_rate[0] * length, before.weak[1] + weak_rate[1] * length),
            conflict=before.conflict,
        )

    def _mix(
        self,
        ctx: SimulationContext,
        signal: OpticalSignal,
        bands: list[Band],
        *,
        gamma: float,
        alpha: float,
        after: KerrHistory,
    ) -> tuple[list[Band], int, float]:
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

        **With** ``pump_phase`` **the Kerr phase joins it**, in three places. Inside
        the span, as :func:`~maiman.kernels.fwm_nonlinear_rate`, which the mixing
        integral carries down the span with the pumps' power. Between spans, as
        how far the drive ``A_i A_j A_k*`` has been turned against the carrier the
        product lands on, from the path's
        :class:`~maiman.signals.KerrHistory`. And in the frame the product is added
        in: the split-step goes on turning the band a product was added to, so
        what earlier spans put there has been turned by that carrier's own Kerr
        phase since, and this span's share has to arrive turned the same way or
        the two do not add as the fibre adds them. Without the last, four
        amplified spans of one strong pump and one weak signal were off from the
        split-step solution by a factor of five.

        Per axis, because each axis is mixed as its own scalar problem. The
        coherent polarization term's exchange of power between the axes is not
        part of it. And the phases are those the split-step applies, so the
        product's own phase runs the way its fields do: the flag changes which
        way the drawn phase and the mismatch combine, which is why it is off by
        default and moves nothing until set.

        **And the pumps are depleted.** Every product's photons are taken from
        the pumps that made it and one is given to its idler, per
        :func:`maiman.kernels.fwm_power_transfer`, so a lossless span conserves
        energy. Should the formula ask a pump for more than it holds, the span is
        refused: that happens only far outside the regime the formula is valid
        in, and scaling every transfer down to fit — which was tried — balances
        the books while emptying the pumps into products, a fiction that looks
        like a result. The third value returned is the fraction moved.
        """
        sources = signal.bands
        distance = self.si("length")
        if len(sources) < 2 or distance <= 0.0:
            return bands, 0, 0.0

        strongest = max((band.average_power() for band in sources), default=0.0)
        if strongest <= 0.0:
            return bands, 0, 0.0
        # Raw dB: the unit machinery already turns a dB parameter into a linear
        # ratio, and converting a second time would silently move the floor.
        floor = strongest * db_to_linear(-self.mixing_floor)

        reference = sources[0]
        beta2 = self.reference_beta2(signal)
        travelled = signal.accumulated_gvd
        powers = [
            (float(np.mean(np.abs(b.Ex) ** 2)), float(np.mean(np.abs(b.Ey) ** 2))) for b in sources
        ]
        coupled = self._couples_bands(signal)
        weight = ORTHOGONAL_KERR_WEIGHT if self.cross_polarization else 0.0
        histories = [signal.history_at(b.f0) for b in sources]

        def power_at(frequency: float) -> tuple[float, float]:
            for band, power in zip(sources, powers, strict=True):
                if abs(band.f0 - frequency) <= MIXING_MERGE_TOLERANCE:
                    return power
            return 0.0, 0.0

        found: list[tuple[float, complex, complex]] = []
        # Power each pump gives up per axis; negative for an idler that gains.
        drained = np.zeros((len(sources), 2))
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
                    if self.pump_phase:
                        rates = fwm_nonlinear_rate(
                            gamma,
                            powers[i],
                            powers[j],
                            powers[k],
                            power_at(frequency),
                            cross_phase=coupled,
                            orthogonal_weight=weight,
                        )
                        landing = signal.history_at(frequency)
                        walked = tuple(
                            histories[i][axis]
                            + histories[j][axis]
                            - histories[k][axis]
                            - landing[axis]
                            for axis in (0, 1)
                        )
                        frame = after.at(frequency)
                    else:
                        rates, walked, frame = (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)
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
                            nonlinear_rate=rates[axis],
                        )
                        for axis in (0, 1)
                    ]
                    if sum(generated) < floor:
                        continue
                    for axis in (0, 1):
                        lost_i, lost_j, gained_k = fwm_power_transfer(
                            sources[i].f0, sources[j].f0, sources[k].f0, generated[axis]
                        )
                        drained[i, axis] += lost_i
                        drained[j, axis] += lost_j
                        drained[k, axis] -= gained_k
                    offsets = (
                        sources[i].f0 - reference.f0,
                        sources[j].f0 - reference.f0,
                        sources[k].f0 - reference.f0,
                    )
                    # Drawn on the pumps, not on the fibre: the same triplet gets
                    # the same phase in every span, which is what lets the spans
                    # add rather than average.
                    drawn = float(ctx.rng("Fiber", "fwm", *offsets).random())
                    linear = fwm_accumulated_phase(travelled, *offsets)
                    if self.pump_phase:
                        # In the split-step's own sense, exp(-i angle): the drive's
                        # walk against the landing carrier, this span's integral, and
                        # the frame that carrier has been turned into since.
                        phasors = [
                            np.exp(
                                2j * np.pi * drawn
                                - 1j
                                * (
                                    linear
                                    + walked[axis]
                                    + frame[axis]
                                    + float(
                                        np.angle(
                                            fwm_mixing_integral(
                                                mismatch, alpha, distance, rates[axis]
                                            )
                                        )
                                    )
                                )
                            )
                            for axis in (0, 1)
                        ]
                    else:
                        phase = (
                            2.0 * np.pi * drawn
                            + float(np.angle(fwm_mixing_integral(mismatch, alpha, distance)))
                            + linear
                        )
                        phasors = [np.exp(1j * phase)] * 2
                    found.append(
                        (
                            frequency,
                            math.sqrt(generated[0]) * phasors[0],
                            math.sqrt(generated[1]) * phasors[1],
                        )
                    )

        if not found:
            return bands, 0, 0.0

        held = np.array(
            [
                (float(np.mean(np.abs(b.Ex) ** 2)), float(np.mean(np.abs(b.Ey) ** 2)))
                for b in bands[: len(sources)]
            ]
        )
        if np.any(drained > held):
            raise ValueError(
                f"{self.label or 'Fiber'}: four-wave mixing here is outside the undepleted-pump "
                "regime the multi-band model is built on -- a product would take more than its "
                "pump holds. Lower the launch power or shorten the span."
            )
        output = list(bands)
        for index in range(len(sources)):
            band = output[index]
            factors = []
            for axis in (0, 1):
                before = held[index, axis]
                after = before - drained[index, axis]
                factors.append(math.sqrt(after / before) if before > 0.0 else 1.0)
            output[index] = replace(
                band,
                Ex=(band.Ex * factors[0]).astype(band.Ex.dtype),
                Ey=(band.Ey * factors[1]).astype(band.Ey.dtype),
            )
        made = float(sum(abs(ax) ** 2 + abs(ay) ** 2 for _, ax, ay in found))
        leaving = float(held.sum())
        depleted = made / leaving if leaving > 0.0 else 0.0

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
        return output, len(frequencies), depleted
