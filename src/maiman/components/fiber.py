"""Optical fiber.

Attenuation, chromatic dispersion, the Kerr nonlinearity — self-phase, cross-phase
and four-wave mixing — and polarization-mode dispersion.

Model references: G. P. Agrawal, *Nonlinear Fiber Optics*, ch. 2-4 and 10 (NLSE,
GVD-induced broadening, SPM, solitons, XPM and FWM); ITU-T G.652 for typical
values.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import replace

import numpy as np

from ..component import BoolParam, Component, Param, PortType
from ..context import SimulationContext
from ..kernels import (
    ORTHOGONAL_KERR_WEIGHT,
    POLARIZED,
    RAMAN_TRIANGLE_LIMIT,
    TONE_TOLERANCE,
    PMDSection,
    PropagationDiagnostics,
    apply_group_delay,
    apply_pmd,
    attenuation_db_per_m_to_alpha,
    band_to_grid,
    coherency,
    cross_kerr_matrix,
    degree_of_polarization,
    differential_group_delay,
    dispersion_slope_to_beta3,
    dispersion_to_beta2,
    effective_length,
    fwm_accumulated_phase,
    fwm_cascade_amplitude,
    fwm_drive_phase,
    fwm_mixing_integral,
    fwm_nonlinear_rate,
    fwm_phase_mismatch,
    fwm_power_transfer,
    fwm_product_power,
    fwm_tone_solve,
    fwm_tone_triples,
    fwm_vector_drive,
    geometric_phase,
    grid_to_band,
    kerr_rate,
    kerr_rate_in_state,
    phase_reference,
    pmd_sections_from,
    principal_state,
    propagate_coupled_ssfm,
    propagate_dispersion,
    raman_coupling,
    raman_tilt,
    raman_transfer,
    random_pmd_sections,
    walkoff_from_dispersion,
)
from ..signals import Band, KerrHistory, OpticalSignal, Signal, WalkoffHistory
from ..units import C_LIGHT, db_to_linear

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
    rotate the state of polarization as the power moves. A neighbour whose own
    axes are correlated -- at 45 degrees, say -- turns the channel's state
    outright, which is inter-channel cross-polarization modulation. With all the
    light on one axis the setting changes nothing, which is why it is safe to
    leave on for a dual-polarization link and pointless for a single-polarization
    one.

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

    **Raman on the composite grid.** With ``composite_fwm`` the gain slope enters
    the split-step itself, as the delayed response's ``T_R`` term, so it acts on
    the modulation and on the mixing products the grid carries rather than on mean
    powers afterwards. It is the same straight line, so a comb past the gain peak is
    refused there, and it is driven by both axes' power, so light on both needs
    ``cross_polarization`` on.
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
    carry_walkoff = BoolParam(
        False,
        doc="Carry each band's own group delay onward, so a detector sees the bands walk apart",
    )
    carry_carrier_phase = BoolParam(
        False,
        doc="Carry each band's carrier phase beta(omega) L, which a detector's beat between "
        "bands that took different paths depends on",
    )
    phase_index = Param(
        1.444,
        unit="",
        min=1.0,
        max=2.0,
        doc="Phase index of the guided mode at the reference wavelength",
        applies_when="carry_carrier_phase",
    )
    group_index = Param(
        1.468,
        unit="",
        min=1.0,
        max=2.0,
        doc="Group index at the reference wavelength: sets the phase's slope in frequency",
        applies_when="carry_carrier_phase",
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
    carry_phase = BoolParam(
        False,
        doc="Mix a constant band or a product in the phase it has, not one drawn for its triplet",
        applies_when="cascaded_fwm",
    )
    cascaded_fwm = BoolParam(
        False,
        doc="Let a first-order mixing product drive a second one within the same span",
        applies_when="four_wave_mixing",
    )
    mixing_steps = Param(
        1.0,
        unit="",
        min=1.0,
        max=64.0,
        doc="Cut the span into this many equal pieces, each mixed on its own",
        applies_when="four_wave_mixing",
    )
    tone_solver = BoolParam(
        False,
        doc="Integrate the coupled equations of every constant tone across the span, all orders",
        applies_when="four_wave_mixing",
    )
    tone_order = Param(
        3.0,
        unit="",
        min=1.0,
        max=8.0,
        doc="Highest order of mixing product the tone solver keeps, when it is above the floor",
        applies_when="tone_solver",
    )
    composite_fwm = BoolParam(
        False,
        doc="Split-step every band on one wide grid, modulation and mixing included",
        applies_when="four_wave_mixing",
    )
    composite_order = Param(
        1.0,
        unit="",
        min=1.0,
        max=3.0,
        doc="Highest order of mixing product the composite grid has room for",
        applies_when="composite_fwm",
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

    def validate(self) -> None:
        if self.carry_phase and not (self.pump_phase and self.cascaded_fwm):
            raise ValueError(
                f"{self.label or 'Fiber'}: carry_phase carries a product's own phase from span "
                "to span, and the second-order term it feeds has an in-span companion only "
                "with cascaded_fwm -- alone it overshoots by several times -- and is written "
                "for pump_phase's bookkeeping. Turn on cascaded_fwm and pump_phase, or turn "
                "carry_phase off."
            )
        if self.composite_fwm:
            for name, on in (
                ("tone_solver", self.tone_solver),
                ("pump_phase", self.pump_phase),
                ("carry_phase", self.carry_phase),
                ("cascaded_fwm", self.cascaded_fwm),
                ("mixing_steps", self.mixing_steps > 1.0),
            ):
                if on:
                    raise ValueError(
                        f"{self.label or 'Fiber'}: composite_fwm split-steps the whole comb on one "
                        f"grid itself and has nothing for {name} to do or no way to carry it. "
                        "Turn one off."
                    )
            if self.pmd_coefficient > 0.0 and not self.cross_polarization:
                raise ValueError(
                    f"{self.label or 'Fiber'}: PMD rotates the two polarizations together, so "
                    "composite_fwm needs cross_polarization on to carry it"
                )
        if self.tone_solver:
            for name, on in (
                ("pump_phase", self.pump_phase),
                ("carry_phase", self.carry_phase),
                ("cascaded_fwm", self.cascaded_fwm),
                ("mixing_steps", self.mixing_steps > 1.0),
            ):
                if on:
                    raise ValueError(
                        f"{self.label or 'Fiber'}: tone_solver integrates every tone's coupled "
                        f"equations itself -- cross-phase, mixing to every order, depletion -- "
                        f"and has nothing for {name} to do or no way to carry it. Turn one off."
                    )
        if self.tone_solver and self.pmd_coefficient > 0.0 and not self.cross_polarization:
            raise ValueError(
                f"{self.label or 'Fiber'}: PMD rotates the two polarizations together, so the "
                "tone solver needs cross_polarization on to carry it"
            )
        if self.mixing_steps > 1.0 and self.pmd_coefficient > 0.0:
            raise ValueError(
                f"{self.label or 'Fiber'}: mixing_steps cuts the span into pieces, and each "
                "piece would draw its own PMD realisation -- a different statistic from one "
                "span's. Set pmd_coefficient to 0, or mixing_steps to 1."
            )

    def _run_in_pieces(
        self, ctx: SimulationContext, inputs: dict[str, Signal]
    ) -> dict[str, Signal]:
        """The span as ``mixing_steps`` equal fibres in a row, each mixed on its own.

        The mixing integral holds each pump's state of polarization from the
        start of the span it is taken over. Light that stays in its state -- on
        an axis, on a diagonal, circular -- loses nothing by that. An ellipse
        turns with its own power (Maker, Terhune and Savage, Phys. Rev. Lett.
        12, 507 (1964)), and over 80 km the held state was up to 9 % out where
        the split-step followed it. A shorter piece holds it over less; the
        history carries each carrier's phase from piece to piece exactly as it
        does from span to span, so the answer converges as the pieces shorten
        (maiman-me9) instead of drifting the way it did before the geometric
        phase was kept. It costs ``mixing_steps`` times the work, which is why
        it is a setting and not the default.
        """
        pieces = round(self.mixing_steps)
        length, _ = self.display("length")
        params = {
            **self._values,
            "length": length / pieces,
            "mixing_steps": 1.0,
        }
        signal = inputs["in"]
        out: dict[str, Signal] = {}
        parts: list[PropagationDiagnostics] = []
        for piece in range(pieces):
            fibre = Fiber(label=f"{self.label}.{piece}", **params)
            out = fibre.run(ctx, {"in": signal})
            signal = out["out"]
            diagnostics = out["diagnostics"]
            assert isinstance(diagnostics, PropagationDiagnostics)
            parts.append(diagnostics)
        moved = 1.0
        for part in parts:
            moved *= 1.0 - part.fwm_depletion
        out["diagnostics"] = PropagationDiagnostics(
            steps=sum(p.steps for p in parts),
            distance=sum(p.distance for p in parts),
            shortest_step=min(p.shortest_step for p in parts),
            longest_step=max(p.longest_step for p in parts),
            peak_nonlinear_phase=max(p.peak_nonlinear_phase for p in parts),
            walkoff_span=sum(p.walkoff_span for p in parts),
            peak_walkoff_slip=max(p.peak_walkoff_slip for p in parts),
            mixing_products=parts[-1].mixing_products,
            fwm_depletion=1.0 - moved,
            raman_tilt=sum(p.raman_tilt for p in parts),
        )
        return out

    def _tone_set(
        self,
        f0: np.ndarray,
        amplitude: np.ndarray,
        *,
        beta2: float,
        gamma: float,
        alpha: float,
        distance: float,
        reference: float,
        rounds: int | None = None,
    ) -> np.ndarray:
        """The launched tones and every product above the floor, ``tone_order`` rounds deep.

        A round takes every ordered ``(i, j, k)`` of the tones so far and estimates
        the product at ``f_i + f_j - f_k`` from the first-order build-up
        (:func:`fwm_mixing_integral`'s magnitude, ``|(e^{rL} - 1) / r|`` for
        ``r = i delta_beta - alpha``): the tones so far drive it, so a product of a
        product is found in the next round. One whose power estimate is under the
        floor -- ``mixing_floor`` dB below the strongest tone -- is left out, and so
        is one at or below zero frequency. The estimate ignores what the other
        tones do to it, and the solver then finds the true amplitude, so the floor
        is a bound on what is kept, not on what is computed.
        """
        f = f0.copy()
        a = np.abs(amplitude)
        floor = float(np.max(a) ** 2) * db_to_linear(-self.mixing_floor)
        for _ in range(round(self.tone_order) if rounds is None else rounds):
            i, j, k = (
                index.ravel() for index in np.meshgrid(*(np.arange(f.size),) * 3, indexing="ij")
            )
            q = f[i] + f[j] - f[k]
            fresh = (q > 0.0) & (np.abs(q - f[k]) > TONE_TOLERANCE)
            fresh &= (np.abs(q - f[i]) > TONE_TOLERANCE) & (np.abs(q - f[j]) > TONE_TOLERANCE)
            i, j, k, q = i[fresh], j[fresh], k[fresh], q[fresh]
            if q.size == 0:
                break
            rate = np.array(
                [
                    complex(
                        -alpha,
                        fwm_phase_mismatch(
                            beta2, f[x] - reference, f[y] - reference, f[w] - reference
                        ),
                    )
                    for x, y, w in zip(i, j, k, strict=True)
                ]
            )
            build = np.where(
                np.abs(rate) * distance < 1e-9,
                distance,
                np.abs(np.expm1(rate * distance) / np.where(rate == 0, 1.0, rate)),
            )
            estimate = 2.0 * gamma * a[i] * a[j] * a[k] * build
            keep = estimate**2 >= floor
            known = {round(v / TONE_TOLERANCE) for v in f}
            added: dict[int, tuple[float, float]] = {}
            for value, size in zip(q[keep], estimate[keep], strict=True):
                key = round(value / TONE_TOLERANCE)
                if key in known:
                    continue
                if key not in added or size > added[key][1]:
                    added[key] = (float(value), float(size))
            if not added:
                break
            new = sorted(added.values())
            f = np.concatenate([f, [v for v, _ in new]])
            a = np.concatenate([a, [size for _, size in new]])
        return np.sort(f)

    def _run_tones(self, ctx: SimulationContext, inputs: dict[str, Signal]) -> dict[str, Signal]:
        """One span with every constant tone's coupled equations integrated, mixing to all orders.

        The series :meth:`_mix` sums stops at second order and assumes undepleted
        pumps; what it leaves out grows as ``gamma P L`` (0.89 of the split-step at
        10 mW over 10 km, 0.75 at 30 mW over 20 km). Here the tones are the
        unknowns -- the launched bands and the products that clear the floor, found
        by :meth:`_tone_set` -- and :func:`~maiman.kernels.fwm_tone_solve` carries
        them through the span, cross-phase modulation, depletion and every order
        included, matching the split-step to the digits its own steps allow. It
        replaces the split-step's Kerr step on these bands, so cross-phase is not
        applied twice.

        Only for bands that are one complex number per axis -- a launched tone, an
        unmodulated carrier, a product an earlier span made -- because the
        equations are for amplitudes, and a modulated channel's phase is not one.
        Such a band is refused, by name, rather than treated as if it were. With
        ``cross_polarization`` off each polarization is its own scalar problem, the
        same model that gives the perturbative path; with it on the tones are Jones
        vectors and the two axes are coupled by silica's isotropic Kerr tensor
        (``coherent_polarization`` off drops its coherent term, as the split-step's
        default does). The bands are
        stored conjugated against the equations' convention -- this library turns
        a field by ``exp(-i angle)`` -- so amplitudes are conjugated on the way in
        and out. The history the perturbative path keeps for its pumps' phase is
        left as it was: the amplitudes carry their own.
        """
        signal: OpticalSignal = inputs["in"]
        distance = self.si("length")
        gamma = self.si("nonlinearity")
        alpha = attenuation_db_per_m_to_alpha(self.si("attenuation"))
        power_factor = db_to_linear(-self.loss_db())
        beta2 = self.reference_beta2(signal)
        bands = signal.bands
        if not bands:
            raise ValueError(f"{self.label or 'Fiber'}: the tone solver needs at least one band")
        launched = np.zeros((len(bands), 2), dtype=np.complex128)
        for index, band in enumerate(bands):
            for axis, field in enumerate((band.Ex, band.Ey)):
                mean = complex(np.mean(field))
                power = float(np.mean(np.abs(field) ** 2))
                if power > 0.0 and abs(mean) ** 2 < (1.0 - 1e-9) * power:
                    raise ValueError(
                        f"{self.label or 'Fiber'}: tone_solver needs every band to be one complex "
                        f"amplitude per polarization, and the band at {band.f0 / 1e12:.4f} THz is "
                        "modulated. Its phase is not one number; use the perturbative path "
                        "(tone_solver off) for it."
                    )
                launched[index, axis] = np.conj(mean)
        reference = bands[0].f0
        f0 = np.array([band.f0 for band in bands])
        strength = np.max(np.abs(launched), axis=1)
        if float(np.max(strength)) <= 0.0:
            raise ValueError(f"{self.label or 'Fiber'}: every band is dark; nothing to solve")
        frequencies = self._tone_set(
            f0,
            strength,
            beta2=beta2,
            gamma=gamma,
            alpha=alpha,
            distance=distance,
            reference=reference,
        )
        offsets = frequencies - reference
        start = np.zeros((frequencies.size, 2), dtype=np.complex128)
        for index in range(len(bands)):
            start[int(np.argmin(np.abs(frequencies - f0[index])))] = launched[index]
        triples = fwm_tone_triples(offsets)
        finish = np.zeros_like(start)
        steps = 0
        # Stimulated Raman scattering is a power exchange, so it joins the amplitudes
        # as a gain each tone feels from the others' power. The shape follows
        # :meth:`_raman`: a comb inside the gain peak's straight line is the triangle
        # the closed form is written in, a wider one silica's measured shape with the
        # quantum defect taken.
        coupling = None
        slope = self.si("raman_gain_slope")
        if slope > 0.0 and frequencies.size > 1:
            narrow = max(f0) - min(f0) <= RAMAN_TRIANGLE_LIMIT
            coupling = raman_coupling(
                [float(value) for value in frequencies],
                gain_slope=slope,
                profile="triangle" if narrow else "silica",
                photon_conserving=not narrow,
            )
        sections: tuple[PMDSection, ...] = ()
        realised_dgd = 0.0
        if self.mean_dgd() > 0.0:
            # The chain the split-step draws, from the same stream: the same fibre.
            sections = random_pmd_sections(
                self.mean_dgd(), int(self.pmd_sections), ctx.rng("Fiber", self.label, "pmd")
            )
            realised_dgd = differential_group_delay(sections)
        if self.cross_polarization:
            # One problem: each tone a Jones vector, the two axes coupled by the
            # isotropic Kerr tensor -- or its phase-only form without the coherent
            # term, as the split-step's own default has it.
            def solve(
                amplitudes: np.ndarray, length: float, behind: float
            ) -> tuple[np.ndarray, int]:
                return fwm_tone_solve(
                    offsets,
                    amplitudes,
                    beta2=beta2,
                    gamma=gamma,
                    alpha=alpha,
                    distance=length,
                    accumulated_gvd=behind,
                    triples=triples,
                    coherent=self.coherent_polarization,
                    raman=coupling,
                )

            # PMD meets each tone at its own frequency, measured from the first band's
            # carrier as the composite grid measures it: there a section is its Jones
            # rotation after the delay's phase ``+-omega dgd / 2`` for the tone's offset
            # ``omega``, so the first band is turned as :func:`~maiman.kernels.apply_pmd`
            # turns a band at rest and the others each as they differ from it. The
            # amplitudes are conjugated against this library's fields, so it is the
            # conjugate matrix that turns them. Where the sections are interleaved the
            # span is cut into as many pieces and each is solved before its rotation;
            # otherwise the whole chain comes after the Kerr effect, as it does in the
            # split-step.
            omega = 2.0 * math.pi * offsets

            def rotate(amplitudes: np.ndarray, section: PMDSection) -> np.ndarray:
                phase = np.exp(0.5j * omega * section.dgd)
                delayed = amplitudes * np.stack([np.conj(phase), phase], axis=1)
                return np.asarray(delayed @ np.conj(section.unitary).T)

            if sections and self.interleave_pmd:
                # Each waveplate at the midpoint of its piece, where the split-step puts
                # it: half a piece, then a plate and a whole piece in turn, then the last
                # half after the last plate.
                piece = distance / len(sections)
                finish, steps = solve(start, 0.5 * piece, signal.accumulated_gvd)
                for index, section in enumerate(sections):
                    finish = rotate(finish, section)
                    length = piece if index + 1 < len(sections) else 0.5 * piece
                    finish, taken = solve(
                        finish, length, signal.accumulated_gvd + beta2 * piece * (index + 0.5)
                    )
                    steps += taken
            else:
                finish, steps = solve(start, distance, signal.accumulated_gvd)
                for section in sections:
                    finish = rotate(finish, section)
        else:
            for axis in (0, 1):
                if np.any(start[:, axis] != 0.0):
                    finish[:, axis], taken = fwm_tone_solve(
                        offsets,
                        start[:, axis],
                        beta2=beta2,
                        gamma=gamma,
                        alpha=alpha,
                        distance=distance,
                        accumulated_gvd=signal.accumulated_gvd,
                        triples=triples,
                        raman=coupling,
                    )
                    steps = max(steps, taken)
        finish = np.conj(finish)
        present = [band.f0 for band in bands]

        def is_launched(frequency: float) -> bool:
            return any(abs(frequency - f) <= TONE_TOLERANCE for f in present)

        dtype = ctx.complex_dtype
        launched_bands: list[Band] = []
        for band in bands:
            slot = int(np.argmin(np.abs(frequencies - band.f0)))
            launched_bands.append(
                replace(
                    band,
                    Ex=np.full(band.num_samples, finish[slot, 0], dtype=dtype),
                    Ey=np.full(band.num_samples, finish[slot, 1], dtype=dtype),
                )
            )
        products = [
            Band(
                Ex=np.full(bands[0].num_samples, finish[slot, 0], dtype=dtype),
                Ey=np.full(bands[0].num_samples, finish[slot, 1], dtype=dtype),
                f0=float(frequency),
                fs=bands[0].fs,
            )
            for slot, frequency in enumerate(frequencies)
            if not is_launched(float(frequency))
        ]
        total = float(np.sum(np.abs(finish) ** 2))
        made = float(
            sum(
                np.sum(np.abs(finish[slot]) ** 2)
                for slot, frequency in enumerate(frequencies)
                if not is_launched(float(frequency))
            )
        )
        diagnostics = PropagationDiagnostics(
            steps=steps,
            distance=distance,
            shortest_step=distance / max(steps, 1),
            longest_step=distance / max(steps, 1),
            peak_nonlinear_phase=float(
                gamma * np.max(np.abs(launched)) ** 2 * effective_length(alpha, distance)
            ),
            mixing_products=len(products),
            fwm_depletion=made / total if total > 0.0 else 0.0,
            differential_group_delay=realised_dgd,
        )
        return {
            "out": OpticalSignal(
                bands=tuple(launched_bands + products),
                noise=tuple(n.scale_power(power_factor) for n in signal.noise),
                accumulated_gvd=signal.accumulated_gvd + beta2 * distance,
                nonlinear_history=signal.nonlinear_history,
                walkoff=self._walkoff(signal, distance, products),
            ),
            "diagnostics": diagnostics,
        }

    def _run_composite(
        self, ctx: SimulationContext, inputs: dict[str, Signal]
    ) -> dict[str, Signal]:
        """One span with every band, modulation and all, split-stepped on one wide grid.

        The perturbative series :meth:`_mix` sums works from each band's mean power
        and stops at second order, and the tone solver needs constants. Here the
        bands are put on a single grid wide enough for the whole comb and the
        products it makes, at their carriers' offsets, and the scalar split-step is
        run on the sum -- so mixing, cross-phase, depletion, walk-off and dispersion
        of each band's own modulation are all in the propagation, which is the
        one-band split-step the tests already take as the reference. Each band, and
        each product above the floor, is then cut back out into its own window and
        retarded frame.

        The grid's window is the bands' own, so the same span of time at
        ``factor`` times the samples; carriers must sit on its frequency bins,
        ``1 / window`` apart, or the window's ends would be a discontinuity to the
        Fourier transform (refused otherwise, with the sequence length that would
        do). The grid is cut back into slots that tile it: each band's reaches the
        midpoints to its neighbours, and the outermost two reach out to a band's own
        window, so nothing is dropped between or beyond them. A product, or the skirt
        of one wider than its slot, that lands on a channel is that channel's
        crosstalk and joins it, as it would a receiver's filter. The axes are independent scalar
        problems, as ``cross_polarization`` off makes them.
        """
        signal: OpticalSignal = inputs["in"]
        distance = self.si("length")
        gamma = self.si("nonlinearity")
        alpha = attenuation_db_per_m_to_alpha(self.si("attenuation"))
        power_factor = db_to_linear(-self.loss_db())
        beta2 = self.reference_beta2(signal)
        bands = signal.bands
        if not bands:
            raise ValueError(
                f"{self.label or 'Fiber'}: the composite solver needs at least one band"
            )
        samples, rate = bands[0].num_samples, bands[0].fs
        if any(b.num_samples != samples or b.fs != rate for b in bands):
            raise ValueError(
                f"{self.label or 'Fiber'}: composite_fwm needs every band on one window and rate"
            )
        window = samples / rate
        f0 = np.array([band.f0 for band in bands])
        amplitude = np.array(
            [math.sqrt(float(np.mean(np.abs(b.Ex) ** 2 + np.abs(b.Ey) ** 2))) for b in bands]
        )
        if float(amplitude.max()) <= 0.0:
            raise ValueError(f"{self.label or 'Fiber'}: every band is dark; nothing to solve")
        centre_of = 0.5 * (float(f0.min()) + float(f0.max()))
        frequencies = self._tone_set(
            f0,
            amplitude,
            beta2=beta2,
            gamma=gamma,
            alpha=alpha,
            distance=distance,
            reference=bands[0].f0,
            rounds=round(self.composite_order),
        )
        offsets = frequencies - centre_of
        misfit = np.abs(offsets * window - np.round(offsets * window))
        if float(misfit.max()) > 1e-6:
            raise ValueError(
                f"{self.label or 'Fiber'}: composite_fwm needs every carrier on the window's "
                f"frequency bins, {1.0 / window:.4g} Hz apart, and one is "
                f"{float(misfit.max()) / window:.4g} Hz off; choose a sequence length whose "
                "window holds a whole number of cycles of the carrier spacings"
            )
        ordered = np.sort(frequencies)
        gaps = np.diff(ordered)
        half_width = min(0.5 * rate, 0.5 * float(gaps.min())) if gaps.size else 0.5 * rate
        # The slots the grid is cut back into tile it: each reaches the midpoints to its
        # neighbours, and the two at the ends reach as far as a band's window does, so a
        # product wider than its slot hands its skirt to the next slot instead of dropping
        # it, and nothing past the outer carriers falls between slots.
        reach = float(np.abs(offsets).max()) + 0.5 * rate
        factor = max(1, math.ceil(2.0 * reach / rate))
        grid_rate = factor * rate
        grid = np.arange(samples * factor) / grid_rate

        def carrier(offset: float) -> np.ndarray:
            return np.asarray(np.exp(2j * np.pi * offset * grid))

        raman = self.si("raman_gain_slope")
        if raman > 0.0 and float(ordered[-1] - ordered[0]) > RAMAN_TRIANGLE_LIMIT:
            raise ValueError(
                f"{self.label or 'Fiber'}: Raman on the composite grid is the gain slope's "
                f"straight line, good to the {RAMAN_TRIANGLE_LIMIT / 1e12:.1f} THz gain peak, "
                f"and this comb and its products span "
                f"{float(ordered[-1] - ordered[0]) / 1e12:.1f} THz; turn composite_fwm off to "
                "integrate it with silica's measured shape"
            )
        mismatch = abs(fwm_phase_mismatch(beta2, 0.0, 0.0, float(gaps.min()) if gaps.size else 0.0))
        step = 0.05 / mismatch if mismatch > 0.0 else None
        finished: list[np.ndarray] = []
        steps = 0
        peak = 0.0
        sections: tuple[PMDSection, ...] = ()
        realised_dgd = 0.0
        if self.mean_dgd() > 0.0:
            # The chain the split-step draws, from the same stream: the same fibre.
            sections = random_pmd_sections(
                self.mean_dgd(), int(self.pmd_sections), ctx.rng("Fiber", self.label, "pmd")
            )
            realised_dgd = differential_group_delay(sections)
            # Measured from the first band's carrier, as the walk-off and the reference
            # beta2 are, and not from the grid's centre: that moves whenever a band is
            # added anywhere in the comb, and would turn every other band's state of
            # polarization with it. The first band meets the chain as ``apply_pmd``
            # gives it alone, and each other carrier as it differs from that one.
            sections = pmd_sections_from(sections, -2.0 * math.pi * (bands[0].f0 - centre_of))

        def combine(axis: int) -> np.ndarray:
            fields = [np.asarray(b.Ex if axis == 0 else b.Ey, dtype=np.complex128) for b in bands]
            combined = np.zeros(samples * factor, dtype=np.complex128)
            for band, field in zip(bands, fields, strict=True):
                # A band is stored without the phase its carrier has turned through --
                # ``beta(omega) z`` -- so the phases the four-wave sums need are put
                # back for the propagation and taken out again after it. The part of
                # ``beta`` that matters is the curvature, ``G d^2 / 2`` for the
                # dispersion ``G`` the path has accumulated: the rest is a delay, or
                # cancels in any four-wave combination.
                behind = np.exp(
                    -0.5j * signal.accumulated_gvd * (2.0 * math.pi * (band.f0 - centre_of)) ** 2
                )
                filtered = grid_to_band(
                    band_to_grid(field, factor) * carrier(band.f0 - centre_of),
                    samples,
                    sample_rate=grid_rate,
                    offset=band.f0 - centre_of,
                    half_width=half_width,
                )
                combined += band_to_grid(filtered * behind, factor) * carrier(band.f0 - centre_of)
            return combined

        if self.cross_polarization:
            # One problem: the two axes of the sum, coupled as the split-step couples them
            # -- the phase-only form, or with ``coherent_polarization`` the isotropic
            # tensor's coherent term too -- and the waveplate chain, where there is one,
            # acting across the whole comb on the grid, where each carrier meets it at
            # its own frequency: between the pieces of the span with ``interleave_pmd``,
            # after the Kerr effect otherwise.
            interleaved = bool(sections) and self.interleave_pmd
            (out_x, out_y), diagnostics = propagate_coupled_ssfm(
                (combine(0), combine(1)),
                grid_rate,
                beta2=(beta2, beta2),
                walkoff=(0.0, 0.0),
                gamma=gamma,
                alpha=alpha,
                distance=distance,
                polarization=(0, 1),
                pairs=((0, 1),) if (self.coherent_polarization or interleaved) else None,
                coherent_polarization=self.coherent_polarization,
                pmd=sections if interleaved else None,
                max_step=step,
                raman_slope=raman,
            )
            if sections and not interleaved:
                out_x, out_y = apply_pmd(out_x, out_y, grid_rate, sections)
            finished = [out_x, out_y]
            steps, peak = diagnostics.steps, diagnostics.peak_nonlinear_phase
        else:
            lit = [
                any(np.any(np.asarray(b.Ey if axis else b.Ex) != 0.0) for b in bands)
                for axis in (0, 1)
            ]
            if raman > 0.0 and all(lit):
                raise ValueError(
                    f"{self.label or 'Fiber'}: Raman is driven by the power on both axes, and "
                    "with cross_polarization off composite_fwm solves them apart; turn "
                    "cross_polarization on, or put the light on one axis"
                )
            for axis in (0, 1):
                combined = combine(axis)
                if not np.any(combined != 0.0):
                    finished.append(np.zeros(samples * factor, dtype=np.complex128))
                    continue
                (out,), diagnostics = propagate_coupled_ssfm(
                    (combined,),
                    grid_rate,
                    beta2=(beta2,),
                    walkoff=(0.0,),
                    gamma=gamma,
                    alpha=alpha,
                    distance=distance,
                    max_step=step,
                    raman_slope=raman,
                )
                finished.append(out)
                steps = max(steps, diagnostics.steps)
                peak = max(peak, diagnostics.peak_nonlinear_phase)

        dtype = ctx.complex_dtype
        present = [band.f0 for band in bands]

        def is_launched(frequency: float) -> bool:
            return any(abs(frequency - f) <= TONE_TOLERANCE for f in present)

        def slot(frequency: float) -> tuple[float, float]:
            index = int(np.argmin(np.abs(ordered - frequency)))
            below = 0.5 * float(gaps[index - 1]) if index > 0 else 0.5 * rate
            above = 0.5 * float(gaps[index]) if index < gaps.size else 0.5 * rate
            return min(below, 0.5 * rate), min(above, 0.5 * rate)

        def cut(frequency: float) -> tuple[np.ndarray, np.ndarray]:
            offset = frequency - centre_of
            below, above = slot(frequency)
            delay = beta2 * 2.0 * math.pi * offset * distance
            turned = np.exp(
                0.5j * (signal.accumulated_gvd + beta2 * distance) * (2.0 * math.pi * offset) ** 2
            )
            cut_out = []
            for axis_field in finished:
                piece = grid_to_band(
                    axis_field,
                    samples,
                    sample_rate=grid_rate,
                    offset=offset,
                    half_width=half_width,
                    below=below,
                    above=above,
                )
                cut_out.append(apply_group_delay(piece, rate, -delay) * turned)
            return cut_out[0], cut_out[1]

        launched_bands = []
        for band in bands:
            ex, ey = cut(band.f0)
            launched_bands.append(replace(band, Ex=ex.astype(dtype), Ey=ey.astype(dtype)))
        products = []
        for frequency in frequencies:
            if is_launched(float(frequency)):
                continue
            ex, ey = cut(float(frequency))
            products.append(
                Band(Ex=ex.astype(dtype), Ey=ey.astype(dtype), f0=float(frequency), fs=rate)
            )
        made = sum(float(np.mean(np.abs(b.Ex) ** 2 + np.abs(b.Ey) ** 2)) for b in products)
        total = made + sum(
            float(np.mean(np.abs(b.Ex) ** 2 + np.abs(b.Ey) ** 2)) for b in launched_bands
        )
        tilt = 0.0
        if raman > 0.0 and len(bands) > 1:
            # Measured, not computed: what the lowest and highest carriers kept of what
            # they were given, as ``_raman`` reports it -- with whatever the mixing took
            # from them in it too, because on the grid the two are one propagation.
            order = sorted(range(len(bands)), key=lambda index: bands[index].f0)
            kept = [
                launched_bands[index].average_power() / bands[index].average_power()
                if bands[index].average_power() > 0.0
                else 1.0
                for index in (order[0], order[-1])
            ]
            tilt = 10.0 * math.log10(kept[0] / kept[1]) if kept[1] > 0.0 else 0.0
        diagnostics_out = PropagationDiagnostics(
            steps=steps,
            distance=distance,
            shortest_step=distance / max(steps, 1),
            longest_step=distance / max(steps, 1),
            peak_nonlinear_phase=peak,
            mixing_products=len(products),
            fwm_depletion=made / total if total > 0.0 else 0.0,
            differential_group_delay=realised_dgd,
            raman_tilt=tilt,
        )
        return {
            "out": OpticalSignal(
                bands=tuple(launched_bands + products),
                noise=tuple(n.scale_power(power_factor) for n in signal.noise),
                accumulated_gvd=signal.accumulated_gvd + beta2 * distance,
                nonlinear_history=signal.nonlinear_history,
                walkoff=self._walkoff(signal, distance, products),
            ),
            "diagnostics": diagnostics_out,
        }

    def run(self, ctx: SimulationContext, inputs: dict[str, Signal]) -> dict[str, Signal]:
        if self.tone_solver and self.four_wave_mixing and self.si("nonlinearity") != 0.0:
            return self._run_tones(ctx, inputs)
        if self.composite_fwm and self.four_wave_mixing and self.si("nonlinearity") != 0.0:
            return self._run_composite(ctx, inputs)
        if self.mixing_steps > 1.0 and self.four_wave_mixing and self.si("nonlinearity") != 0.0:
            return self._run_in_pieces(ctx, inputs)
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

        history = self._nonlinear_history(signal, gamma=gamma, alpha=alpha, propagated=bands)
        if gamma != 0.0 and self.four_wave_mixing:
            bands, emitted, depleted = self._mix(
                ctx, signal, bands, gamma=gamma, alpha=alpha, after=history
            )
            diagnostics = replace(diagnostics, mixing_products=emitted, fwm_depletion=depleted)
        walkoff = self._walkoff(signal, distance, bands)

        return {
            "out": OpticalSignal(
                bands=tuple(bands),
                noise=tuple(n.scale_power(power_factor) for n in signal.noise),
                # The span's own contribution to the path history, which is what
                # tells the *next* span how far these products have already
                # rotated away from their pumps.
                accumulated_gvd=signal.accumulated_gvd + self.reference_beta2(signal) * distance,
                nonlinear_history=history,
                walkoff=walkoff,
            ),
            "diagnostics": diagnostics,
        }

    @staticmethod
    def _carrier_phase(band: Band) -> tuple[float | None, float | None]:
        """Each axis's phase if the band is a constant amplitude there, else ``None``.

        A launched tone or a mixing product is one complex number per axis and has a
        phase; a modulated channel's mean says nothing about it, and stays drawn.
        """
        out: list[float | None] = []
        for field in (band.Ex, band.Ey):
            mean = complex(np.mean(field))
            power = float(np.mean(np.abs(field) ** 2))
            out.append(
                float(np.angle(mean))
                if power > 0.0 and abs(mean) ** 2 >= (1.0 - 1e-9) * power
                else None
            )
        return out[0], out[1]

    @staticmethod
    def _product_phase_before(signal: OpticalSignal, frequency: float) -> float | None:
        """The carrier phase a product new at ``frequency`` arrives at this span with [rad].

        A product's amplitude is formed with the curvature the path has accumulated
        put back, so it stands as a carrier at its own frequency that had run the
        whole path -- and carries what such a carrier would have, checked against a
        split-step with the whole ``beta(omega)`` to 1e-4 rad. Along one path the
        phase is a quadratic in frequency whose curvature is ``accumulated_gvd``, so
        two carried carriers ``a`` and ``b`` fix it at ``f = f_a + k (f_b - f_a)``::

            phi = (1 - k) phi_a + k phi_b - (G / 2) k (1 - k) (omega_b - omega_a)^2

        For whole ``k`` only, which keeps it exact modulo ``2 pi``: a product of a comb
        lies a whole number of some pair's spacings from it. ``None`` where no pair
        does, and the product is then left with no entry, as before.
        """
        walkoff = signal.walkoff
        bands = signal.bands
        for index, a in enumerate(bands):
            for b in bands[index + 1 :]:
                k = (frequency - a.f0) / (b.f0 - a.f0)
                if abs(k - round(k)) > 1e-6:
                    continue
                k = round(k)
                omega = 2.0 * math.pi * (b.f0 - a.f0)
                return (
                    (1 - k) * walkoff.phase_at(a.f0)
                    + k * walkoff.phase_at(b.f0)
                    - 0.5 * signal.accumulated_gvd * k * (1 - k) * omega**2
                )
        return None

    def _walkoff(
        self, signal: OpticalSignal, distance: float, made: Sequence[Band] = ()
    ) -> WalkoffHistory:
        """Each band's group delay after this span, against the first band's [s].

        Walk-off is a linear operator like dispersion, but it is a *constant*
        delay per band, and :meth:`_propagate_linear` says why it is not applied
        to the samples: every band is reported in its own retarded frame, where
        a constant delay cancels. What does not cancel is the delay *between*
        bands, and with ``carry_walkoff`` that number rides along on the signal
        instead of being thrown away, for a detector summing several bands to
        apply once, at the end.

        Written against ``bands[0]``, which is the frame :meth:`reference_beta2`
        is written against too -- and since only differences are observable, any
        band would do.

        With ``carry_carrier_phase``, the bands in ``made`` that this span created --
        mixing products -- are given a carrier phase too, from
        :meth:`_product_phase_before` and this span's ``beta L`` at their frequency.
        """
        if not (self.carry_walkoff or self.carry_carrier_phase) or not signal.bands:
            return signal.walkoff
        before = signal.walkoff
        if before.conflict:
            what = (
                "carry_walkoff needs one set of arrival delays"
                if self.carry_walkoff
                else "carry_carrier_phase needs one set of carrier phases"
            )
            raise ValueError(f"{self.label}: {what}, and {before.conflict}")
        delays = dict(before.carriers)
        if self.carry_walkoff:
            reference = signal.bands[0]
            for band in signal.bands:
                delays[band.f0] = before.at(band.f0) + self.walkoff_of(band, reference) * distance
        phases = dict(before.phases)
        if self.carry_carrier_phase:
            for band in signal.bands:
                phases[band.f0] = math.remainder(
                    before.phase_at(band.f0)
                    + self.carrier_phase_of(band, signal.bands[0]) * distance,
                    2.0 * math.pi,
                )
            present = [band.f0 for band in signal.bands]
            for band in made:
                if any(abs(band.f0 - f) <= TONE_TOLERANCE for f in present):
                    continue
                arrived = self._product_phase_before(signal, band.f0)
                if arrived is not None:
                    phases[band.f0] = math.remainder(
                        arrived + self.carrier_phase_of(band, signal.bands[0]) * distance,
                        2.0 * math.pi,
                    )
        return WalkoffHistory(
            carriers=tuple(sorted(delays.items())), phases=tuple(sorted(phases.items()))
        )

    def carrier_phase_of(self, band: Band, about: Band | None = None) -> float:
        """The phase constant ``beta(omega)`` of the carrier at ``band.f0`` [rad/m].

        The mode's propagation constant, absolute -- ``n_p omega_0 / c`` at the
        reference wavelength, ``n_g / c`` for its slope in frequency, and
        ``beta2``, ``beta3`` for the curvature, the same dispersion
        :meth:`beta2_at` and :meth:`beta3_at` give::

            beta(omega) = n_p omega_0 / c + (n_g / c) d + beta2 d^2 / 2 + beta3 d^3 / 6,
            d = omega - omega_0

        Two carriers' *difference* is what a beat sees, and it is mostly
        ``(n_g / c) (omega_a - omega_b)``, which is why the group index matters as
        much as the phase index; ``phase_index`` only sets where the fast phase of
        one carrier over a length difference lands.

        **About a band, for a comb.** Given ``about``, the expansion is taken again
        at that carrier: its value and slope there from the one above, and its
        curvature from :meth:`beta2_at` and :meth:`beta3_at` at that band's own
        wavelength -- the ``beta2`` :meth:`reference_beta2` gives four-wave mixing.
        Holding ``beta2`` at the reference wavelength across a comb sitting
        elsewhere disagrees with that by ``(lambda / lambda_ref)^2``, 0.33 % at
        1552.5 against 1550 nm, which over 20 km of D = 2 put a product's carried
        phase 0.03 rad off the field. A span carries every phase about its first
        band, as it writes its walk-off and its mixing against it.
        """
        reference = self.si("reference_wavelength")
        omega_0 = 2.0 * math.pi * C_LIGHT / reference
        beta2, beta3 = self.beta2_at(reference), self.beta3_at(reference)

        def expanded(omega: float) -> tuple[float, float]:
            d = omega - omega_0
            value = (
                self.phase_index * omega_0 / C_LIGHT
                + self.group_index * d / C_LIGHT
                + 0.5 * beta2 * d**2
                + beta3 * d**3 / 6.0
            )
            return value, self.group_index / C_LIGHT + beta2 * d + 0.5 * beta3 * d**2

        omega = 2.0 * math.pi * band.f0
        if about is None:
            return expanded(omega)[0]
        anchor = 2.0 * math.pi * about.f0
        value, slope = expanded(anchor)
        d = omega - anchor
        return (
            value
            + slope * d
            + 0.5 * self.beta2_at(about.wavelength) * d**2
            + self.beta3_at(about.wavelength) * d**3 / 6.0
        )

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
        self,
        signal: OpticalSignal,
        *,
        gamma: float,
        alpha: float,
        propagated: Sequence[Band] = (),
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
        states = self._polarized_states(signal)
        if states is not None:
            return self._history_in_states(
                signal, states, gamma=gamma, length=length, propagated=propagated
            )
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

    def _polarized_states(self, signal: OpticalSignal) -> list[np.ndarray] | None:
        """Each band's coherency, when the axes are coupled and every band is in one state.

        ``None`` otherwise: with the axes independent each is its own scalar
        problem, and a band of two independent tributaries has no one state for
        its Kerr phase to be taken in, so both keep the per-axis form.
        """
        if not self.cross_polarization or not signal.bands:
            return None
        states = [coherency(b.Ex, b.Ey) for b in signal.bands]
        for state in states:
            if float(np.trace(state).real) > 0.0 and degree_of_polarization(state) < POLARIZED:
                return None
        return states

    def _history_in_states(
        self,
        signal: OpticalSignal,
        states: list[np.ndarray],
        *,
        gamma: float,
        length: float,
        propagated: Sequence[Band] = (),
    ) -> KerrHistory:
        """:meth:`_nonlinear_history` with each carrier turned in its own state.

        A carrier's angle is its rate in its own state times ``L_eff``, the same on
        both axes because it is one wave; the weak carrier's is kept as the matrix
        it accumulates, because its angle depends on the state a product lands in.
        A carrier the history has not seen starts from the weak carrier's angle in
        that carrier's state.

        The angle is a phase on x, the carrier's Jones vector held real there
        (:func:`~maiman.kernels.principal_state`). A state that moves over the
        span -- an ellipse the Kerr effect turns -- adds a geometric phase in that
        gauge as well, :func:`~maiman.kernels.geometric_phase` from the state at
        the start to the one ``propagated`` ends in. Light on an axis, on a
        diagonal or circular keeps its state and adds none.
        """
        before = signal.nonlinear_history
        coupled = self._couples_bands(signal)
        coherent = self.coherent_polarization
        added = np.zeros((2, 2), dtype=np.complex128)
        if coupled:
            for state in states:
                added = added + cross_kerr_matrix(state, coherent=coherent)
        added = gamma * length * added
        if before.weak_matrix:
            wxx, wyy, re, im = before.weak_matrix
            start = np.array([[wxx, complex(re, im)], [complex(re, -im), wyy]])
        else:
            start = np.diag(np.array(before.weak, dtype=np.complex128))
        weak = start + added
        carriers = {f0: (x, y) for f0, x, y in before.carriers}
        for index, band in enumerate(signal.bands):
            if float(np.trace(states[index]).real) <= 0.0:
                continue
            state = principal_state(states[index])
            known = [f for f in carriers if abs(f - band.f0) <= 1e-9 * band.f0]
            begun = carriers[known[0]][0] if known else before.weak_in(state)
            for f in known:
                del carriers[f]
            rate = kerr_rate_in_state(
                gamma, states, state, own=index, coherent=coherent, cross_phase=coupled
            )
            angle = begun + rate * length
            if index < len(propagated):
                ended = coherency(propagated[index].Ex, propagated[index].Ey)
                if degree_of_polarization(ended) >= POLARIZED:
                    angle += geometric_phase(state, principal_state(ended))
            carriers[band.f0] = (angle, angle)
        return KerrHistory(
            carriers=tuple((f0, x, y) for f0, (x, y) in sorted(carriers.items())),
            weak=(float(weak[0, 0].real), float(weak[1, 1].real)),
            conflict=before.conflict,
            weak_matrix=(
                float(weak[0, 0].real),
                float(weak[1, 1].real),
                float(weak[0, 1].real),
                float(weak[0, 1].imag),
            ),
        )

    def _drive_drift(
        self,
        states: list[np.ndarray],
        propagated: Sequence[Band],
        triplet: tuple[int, int, int],
        landed: int | None,
        *,
        length: float,
    ) -> float:
        """How much faster than their Kerr rates a moving drive turns off its carrier [rad/m].

        In the history's gauge, each Jones vector real on x: the geometric phase
        of ``i`` and ``j`` less ``k``'s, less the change in the tensor's own phase
        on x, less the geometric phase of the channel the product lands on --
        each from the state at the span's start to the one it ends in, over
        ``L_eff``. Zero for pumps whose states stay put.
        """
        if length <= 0.0:
            return 0.0

        def ended(index: int) -> np.ndarray | None:
            if index >= len(propagated):
                return None
            state = coherency(propagated[index].Ex, propagated[index].Ey)
            return principal_state(state) if degree_of_polarization(state) >= POLARIZED else None

        start = [principal_state(states[n]) for n in triplet]
        end = [ended(n) for n in triplet]
        drift = 0.0
        for sign, first, last in zip((1.0, 1.0, -1.0), start, end, strict=True):
            if last is not None:
                drift += sign * geometric_phase(first, last)
        if all(vector is not None for vector in end):
            coherent = self.coherent_polarization
            moved = fwm_drive_phase(*end, coherent=coherent) - fwm_drive_phase(  # type: ignore[arg-type]
                *start, coherent=coherent
            )
            drift -= math.remainder(moved, 2.0 * math.pi)
        if landed is not None:
            last = ended(landed)
            if last is not None:
                drift -= geometric_phase(principal_state(states[landed]), last)
        return drift / length

    def _mix_cascade(
        self,
        *,
        sources: Sequence[Band],
        reference: Band,
        beta2: float,
        powers: list[tuple[float, float]],
        power_at: Callable[[float], tuple[float, float]],
        coupled: bool,
        weight: float,
        gamma: float,
        alpha: float,
        distance: float,
        floor: float,
        occupied: Sequence[float],
        phases: Sequence[tuple[float | None, float | None]],
        travelled: float,
        history: KerrHistory,
        after: KerrHistory,
    ) -> list[tuple[float, complex, complex]]:
        """Second-order products: a first-order one driving a further one, within this span.

        The outer two loops are :meth:`_mix`'s own triad loop again -- the same
        ``i``, ``j``, ``k`` over the *launched* bands, the same exclusion of an
        idler that is really cross-phase modulation, the same mismatch. What is
        new is the pair of loops inside it: ``p``, ``q`` range over the launched
        bands again, standing in for the first product's *own* triad partners --
        it drives a further product at ``f_i + f_j - f_k + f_p - f_q`` exactly as
        a launched pump would, via :func:`~maiman.kernels.fwm_cascade_amplitude`,
        which integrates both triads' build-up in one closed form
        (:func:`~maiman.kernels.fwm_cascade_integral`). A result landing on
        ``occupied`` -- a launched band, or a frequency the first-order pass
        already put a product at -- is dropped. A launched band is cross-phase
        modulation the split-step already applied; a first-order product's own
        frequency is a *correction to that product*, not a new tone -- the
        first product feeding back into itself through the same two pumps that
        made it, one order further in ``gamma``, which this pass is not scoped
        to get right (see below) and so leaves alone rather than risk half a
        correction.

        Restricted to launched bands driving both triads -- not another
        second-order product, and not a first-order product appearing twice --
        because those are third order and higher in ``gamma``, a further factor
        of the same smallness that makes this correction worth adding at all.
        ``pump_phase``'s own nonlinear-rate correction is folded into each
        triad's mismatch only at ``alpha = 0``, where the fold is exact (see
        :func:`~maiman.kernels.fwm_cascade_integral`); at nonzero loss it is left
        out rather than approximated silently, so a lossy span gets the
        dispersive mismatch corrected and not the pump phase.

        Needs ``cross_polarization`` off -- the drive here is the scalar,
        per-axis one, and :meth:`_mix` refuses the combination before this is
        called. Runs the launched-band triads twice (outer for the first
        product, inner for its own), so cost is ``O(N**5)`` in the number of
        launched bands; fine for the handful of pumps this was written against,
        not for a filled comb.
        """
        found: list[tuple[float, complex, complex]] = []
        count = len(sources)
        for i in range(count):
            for j in range(i, count):
                for k in range(count):
                    if k in (i, j):
                        continue
                    frequency1 = sources[i].f0 + sources[j].f0 - sources[k].f0
                    if frequency1 <= 0.0:
                        continue
                    mismatch1 = fwm_phase_mismatch(
                        beta2,
                        sources[i].f0 - reference.f0,
                        sources[j].f0 - reference.f0,
                        sources[k].f0 - reference.f0,
                    )
                    rate1 = (0.0, 0.0)
                    if self.pump_phase and alpha == 0.0:
                        rate1 = fwm_nonlinear_rate(
                            gamma,
                            powers[i],
                            powers[j],
                            powers[k],
                            power_at(frequency1),
                            cross_phase=coupled,
                            orthogonal_weight=weight,
                        )
                    for p in range(count):
                        for q in range(count):
                            if p == q:
                                continue
                            frequency2 = frequency1 + sources[p].f0 - sources[q].f0
                            if frequency2 <= 0.0:
                                continue
                            if any(
                                abs(frequency2 - existing) <= MIXING_MERGE_TOLERANCE
                                for existing in (
                                    *occupied,
                                    *(
                                        self._own_products([sources[n].f0 for n in (i, j, k, p, q)])
                                        if self.carry_phase
                                        else ()
                                    ),
                                )
                            ):
                                continue
                            mismatch2 = fwm_phase_mismatch(
                                beta2,
                                frequency1 - reference.f0,
                                sources[p].f0 - reference.f0,
                                sources[q].f0 - reference.f0,
                            )
                            rate2 = (0.0, 0.0)
                            if self.pump_phase and alpha == 0.0:
                                rate2 = fwm_nonlinear_rate(
                                    gamma,
                                    power_at(frequency1),
                                    powers[p],
                                    powers[q],
                                    power_at(frequency2),
                                    cross_phase=coupled,
                                    orthogonal_weight=weight,
                                )
                            amplitudes = [
                                fwm_cascade_amplitude(
                                    powers[i][axis],
                                    powers[j][axis],
                                    powers[k][axis],
                                    powers[p][axis],
                                    powers[q][axis],
                                    gamma=gamma,
                                    alpha=alpha,
                                    distance=distance,
                                    phase_mismatch_1=mismatch1 + rate1[axis],
                                    phase_mismatch_2=mismatch2 + rate2[axis],
                                    degenerate_1=i == j,
                                    degenerate_2=False,
                                )
                                for axis in (0, 1)
                            ]
                            if sum(abs(value) ** 2 for value in amplitudes) < floor:
                                continue
                            if self.carry_phase and self.pump_phase:
                                amplitudes = self._carried_cascade(
                                    amplitudes,
                                    phases,
                                    legs=((i, 1.0), (j, 1.0), (k, -1.0), (p, 1.0), (q, -1.0)),
                                    linear=fwm_accumulated_phase(
                                        travelled,
                                        sources[i].f0 - reference.f0,
                                        sources[j].f0 - reference.f0,
                                        sources[k].f0 - reference.f0,
                                    )
                                    + fwm_accumulated_phase(
                                        travelled,
                                        frequency1 - reference.f0,
                                        sources[p].f0 - reference.f0,
                                        sources[q].f0 - reference.f0,
                                    ),
                                    frame=(
                                        after.at(frequency2)[0] - history.at(frequency2)[0],
                                        after.at(frequency2)[1] - history.at(frequency2)[1],
                                    ),
                                )
                            found.append((frequency2, amplitudes[0], amplitudes[1]))

                    # The first product can also sit in the *conjugated* role,
                    # beside two launched pumps taken un-conjugated: landing at
                    # ``f_p + f_q - f_F`` rather than ``f_F + f_p - f_q``. Both
                    # are genuine second-order terms -- the drive is built from
                    # two un-conjugated factors and one conjugated one, and the
                    # first-order product can be either -- and dropping this one
                    # left a two-pump second-order product a third high in the
                    # low-power limit (maiman-z8j). ``p <= q``, matching the
                    # outer ``i <= j``: the two launched legs here are the pair
                    # that is un-conjugated together.
                    for p in range(count):
                        for q in range(p, count):
                            frequency2 = sources[p].f0 + sources[q].f0 - frequency1
                            if frequency2 <= 0.0:
                                continue
                            if any(
                                abs(frequency2 - existing) <= MIXING_MERGE_TOLERANCE
                                for existing in (
                                    *occupied,
                                    *(
                                        self._own_products([sources[n].f0 for n in (i, j, k, p, q)])
                                        if self.carry_phase
                                        else ()
                                    ),
                                )
                            ):
                                continue
                            mismatch2 = fwm_phase_mismatch(
                                beta2,
                                sources[p].f0 - reference.f0,
                                sources[q].f0 - reference.f0,
                                frequency1 - reference.f0,
                            )
                            rate2 = (0.0, 0.0)
                            if self.pump_phase and alpha == 0.0:
                                rate2 = fwm_nonlinear_rate(
                                    gamma,
                                    powers[p],
                                    powers[q],
                                    power_at(frequency1),
                                    power_at(frequency2),
                                    cross_phase=coupled,
                                    orthogonal_weight=weight,
                                )
                            amplitudes = [
                                fwm_cascade_amplitude(
                                    powers[i][axis],
                                    powers[j][axis],
                                    powers[k][axis],
                                    powers[p][axis],
                                    powers[q][axis],
                                    gamma=gamma,
                                    alpha=alpha,
                                    distance=distance,
                                    phase_mismatch_1=mismatch1 + rate1[axis],
                                    phase_mismatch_2=mismatch2 + rate2[axis],
                                    degenerate_1=i == j,
                                    degenerate_2=p == q,
                                    conjugate_first=True,
                                )
                                for axis in (0, 1)
                            ]
                            if sum(abs(value) ** 2 for value in amplitudes) < floor:
                                continue
                            if self.carry_phase and self.pump_phase:
                                amplitudes = self._carried_cascade(
                                    amplitudes,
                                    phases,
                                    legs=((p, 1.0), (q, 1.0), (i, -1.0), (j, -1.0), (k, 1.0)),
                                    linear=-fwm_accumulated_phase(
                                        travelled,
                                        sources[i].f0 - reference.f0,
                                        sources[j].f0 - reference.f0,
                                        sources[k].f0 - reference.f0,
                                    )
                                    + fwm_accumulated_phase(
                                        travelled,
                                        sources[p].f0 - reference.f0,
                                        sources[q].f0 - reference.f0,
                                        frequency1 - reference.f0,
                                    ),
                                    frame=(
                                        after.at(frequency2)[0] - history.at(frequency2)[0],
                                        after.at(frequency2)[1] - history.at(frequency2)[1],
                                    ),
                                )
                            found.append((frequency2, amplitudes[0], amplitudes[1]))
        return found

    @staticmethod
    def _own_products(legs: Sequence[float]) -> set[float]:
        """Every ``f_a + f_b - f_c`` the frequencies ``legs`` can make between themselves.

        Where a second-order term lands on one of these it is a correction to a
        first-order product of its own pumps -- the first product feeding back
        into itself, or into the other one, one order further in ``gamma`` -- and
        the pass does not carry the other terms of that order (the product's own
        depletion of its pumps, its cross-phase on itself), so adding this one
        alone moves the first order away from the split-step rather than toward
        it: 10 mW over 20 km went from 0.988 to 0.962 of it. It includes the legs
        themselves (``c = b``), which are cross-phase modulation the split-step
        already applied.
        """
        return {a + b - c for a in legs for b in legs for c in legs}

    @staticmethod
    def _carried_cascade(
        amplitudes: list[complex],
        phases: Sequence[tuple[float | None, float | None]],
        *,
        legs: tuple[tuple[int, float], ...],
        linear: float,
        frame: tuple[float, float],
    ) -> list[complex]:
        """A second-order amplitude in the phase its five launched legs give it.

        :func:`~maiman.kernels.fwm_cascade_amplitude` is written for legs at zero
        phase and a span that starts at zero: its field is the physical one. The
        bands here are stored conjugated -- a first-order product leaves
        :meth:`_mix` as ``exp(-i (...))`` -- so the amplitude is conjugated, turned
        by the legs' own phases, turned back by what both triads have
        accumulated over the fibre behind this span, and turned on by the
        landing carrier's own Kerr phase over this one -- ``frame``, its history's
        change across the span, as the
        ordinary triads add it -- which the mismatch the integral folds in has
        taken out of the drive. A leg with no phase of its
        own (a modulated band) leaves the amplitude as it was.
        """
        out = []
        for axis, amplitude in enumerate(amplitudes):
            own = [(phases[index][axis], sign) for index, sign in legs]
            if any(leg is None for leg, _ in own):
                out.append(amplitude)
                continue
            total = sum(sign * leg for leg, sign in own if leg is not None)
            out.append(complex(np.conj(amplitude) * np.exp(1j * (total - linear - frame[axis]))))
        return out

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

        **With** ``cross_polarization`` **the drive is a vector.** Without it
        each axis is mixed as its own scalar problem, which is the model the
        split-step is then running too. With it, a product on x is driven by the
        pumps' y components as well -- :func:`~maiman.kernels.fwm_vector_drive`,
        from each band's coherency and the Kerr tensor the split-step is using,
        isotropic with ``coherent_polarization`` and phase-only without. The two
        axes of a product keep the drive's relative phase, so circular pumps make
        a circular product. Before, a beam at 45 degrees mixed at a quarter of the
        strength it does on the axis.

        With ``pump_phase`` too, and every band in one state, the pumps' Kerr
        phase is taken in each carrier's own state rather than per axis --
        :func:`~maiman.kernels.kerr_rate_in_state`, from the same matrices the
        split-step steps each band through -- and the signal's history keeps the
        weak carrier's accumulated matrix, because the angle a new product's
        carrier has reached depends on the state it lands in. Linear light at any
        angle, and circular light, then add from span to span as light on the
        axis does. An ellipse the Kerr effect turns is followed between spans --
        the geometric phase its moving state adds, and the tensor's phase moving
        with it -- and a span cut finely lands on the split-step; taken whole,
        with each state held from its start, four 80 km spans are up to 9 % out
        (maiman-me9) -- ``mixing_steps`` cuts the span and closes that. A band of two independent
        tributaries has no one state, and keeps the per-axis form. And the phases are those the
        split-step applies, so the product's own phase runs the way its fields
        do: the flag changes which way the drawn phase and the mismatch combine,
        which is why it is off by default and moves nothing until set.

        **With** ``carry_phase`` **a band that has a phase is mixed in it.** A launched tone,
        or a product an earlier span made, is one complex amplitude per axis, and its
        phase is used where the drawn one stood -- and where the history's angle did,
        since a constant band's phase already holds the Kerr angle it has turned
        through. A product leaves the field equation's ``i`` behind it, so a second-order
        one has it twice, and the cascade term is turned by its five legs' phases and by
        the mismatch both its triads have accumulated over the fibre behind the span.
        That is what makes the second order add across spans the way the split-step's
        does: with the mismatch nil it grows as the length squared, ``N (N - 1) / 2``
        units from the products carried across the spans and ``N / 2`` from those made
        inside them. A modulated channel has no phase to carry and stays drawn.
        Scalar drive only, and paired with ``cascaded_fwm``: without the in-span half the
        carried terms overshoot by several times (maiman-z8j).

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
        if self.cascaded_fwm and self.cross_polarization:
            raise ValueError(
                f"{self.label or 'Fiber'}: cascaded_fwm's second-order term is written for the "
                "scalar, per-axis drive; cross_polarization's vector one would need the tensor "
                "carried through a second mixing stage, which it is not. Turn one of them off."
            )

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
        # With the axes coupled, the drive is a vector: each band's coherency,
        # not only its two powers, decides what it mixes into on each axis.
        states = [coherency(b.Ex, b.Ey) for b in sources] if self.cross_polarization else []
        # A band of two independent tributaries has no one state for its Kerr
        # phase to be taken in; then every rate stays per axis.
        polarized = self._polarized_states(signal) is not None
        phases = [self._carrier_phase(b) for b in sources]

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
                    if states:
                        drive = fwm_vector_drive(
                            states[i], states[j], states[k], coherent=self.coherent_polarization
                        )
                        strengths = [(float(drive[axis, axis].real), 1.0, 1.0) for axis in (0, 1)]
                        # The product's two axes are one wave: their relative
                        # phase is the drive's, taken against x -- the same axis
                        # every span, because what the drawn phase and the
                        # history supply is the phase *on that axis*, and a
                        # product that changed axes between spans would change
                        # its phase by the drive's skew. Taken against the
                        # stronger axis it did, on every tie: a circular drive's
                        # two axes are equal to the last bit, the choice flipped
                        # span to span, and four spans added up to 1.7 times the
                        # split-step's product -- 6.4 phase-only (maiman-sew).
                        # Only a drive with nothing on x is taken against y.
                        lead = phase_reference(np.sqrt(np.abs(np.diag(drive).real)))
                        skew = [float(np.angle(drive[axis, lead])) for axis in (0, 1)]
                        if self.pump_phase and polarized:
                            # In each carrier's own state rather than per axis: the
                            # drive turns at rate_i + rate_j - rate_k, the product in
                            # the state the drive gives it, and every neighbour is
                            # read in the state it acts on. One number, since the
                            # product is one wave.
                            landed = next(
                                (
                                    index
                                    for index, band in enumerate(sources)
                                    if abs(band.f0 - frequency) <= MIXING_MERGE_TOLERANCE
                                ),
                                None,
                            )
                            carriers = [
                                kerr_rate_in_state(
                                    gamma,
                                    states,
                                    principal_state(states[n]),
                                    own=n,
                                    coherent=self.coherent_polarization,
                                    cross_phase=coupled,
                                )
                                for n in (i, j, k)
                            ]
                            product = kerr_rate_in_state(
                                gamma,
                                states,
                                principal_state(drive),
                                own=landed,
                                coherent=self.coherent_polarization,
                                cross_phase=coupled,
                            )
                            rate = carriers[0] + carriers[1] - carriers[2] - product
                            # Pumps whose states move turn the drive by more than
                            # their Kerr rates: a geometric phase each, and the
                            # tensor's phase on x changing with them. Both follow
                            # the power down the span as the Kerr rate does, so
                            # they join it as their change over the span / L_eff.
                            rate += self._drive_drift(
                                states,
                                bands,
                                (i, j, k),
                                landed,
                                length=effective_length(alpha, distance),
                            )
                            rates = (rate, rate)
                            # And between spans, from the history kept in the same
                            # terms: the drive's walk against the angle the product's
                            # carrier -- a channel, or the weak one in the product's
                            # state -- had reached, and the frame it has been turned to.
                            landing_state = principal_state(drive)
                            history = signal.nonlinear_history
                            reached = history.angle_in(frequency, landing_state)
                            turned = after.angle_in(frequency, landing_state)
                            pumped = [
                                history.angle_in(sources[n].f0, principal_state(states[n]))
                                for n in (i, j, k)
                            ]
                            step = pumped[0] + pumped[1] - pumped[2] - reached
                            walked, frame = (step, step), (turned, turned)
                            # The history's angles are phases on x, each pump's
                            # Jones vector held real there; the tensor then gives
                            # the drive a phase on x of its own, which moves when
                            # the pumps' states do -- an ellipse turning, where
                            # dropping it cost 6 to 9 % over four spans (maiman-me9).
                            turn = fwm_drive_phase(
                                *(principal_state(states[n]) for n in (i, j, k)),
                                coherent=self.coherent_polarization,
                            )
                            skew = [value + turn for value in skew]
                            # What this span makes is added at its end, where the
                            # pumps' states have moved and taken the drive's with
                            # them: its share goes in the drive's orientation there.
                            # A product on a channel has that channel's geometric
                            # phase in the history already; a new one has none, and
                            # is given the drive's.
                            if all(n < len(bands) for n in (i, j, k)):
                                final = fwm_vector_drive(
                                    *(coherency(bands[n].Ex, bands[n].Ey) for n in (i, j, k)),
                                    coherent=self.coherent_polarization,
                                )
                                ahead = phase_reference(np.sqrt(np.abs(np.diag(final).real)))
                                skew = [
                                    float(np.angle(final[axis, ahead])) + turn for axis in (0, 1)
                                ]
                                if landed is None:
                                    moved = geometric_phase(
                                        principal_state(drive), principal_state(final)
                                    )
                                    frame = (frame[0] + moved, frame[1] + moved)
                    else:
                        strengths = [
                            (powers[i][axis], powers[j][axis], powers[k][axis]) for axis in (0, 1)
                        ]
                        skew = [0.0, 0.0]
                    generated = [
                        fwm_product_power(
                            *strengths[axis],
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
                    draws = [drawn, drawn]
                    if self.carry_phase and self.pump_phase and not states:
                        turns = list(walked)
                        for axis in (0, 1):
                            first, second, third = (phases[n][axis] for n in (i, j, k))
                            if first is None or second is None or third is None:
                                continue
                            # A phase the bands have is used, not drawn: a constant
                            # band's phase already holds the Kerr angle it has turned
                            # through, so it replaces the history's angle for the
                            # three carriers as well as the draw. And the product
                            # leaves the field equation's ``i`` behind it, a quarter
                            # turn a first-order product has and a second-order one
                            # has twice, which the cascade's amplitude carries too.
                            kerr = histories[i][axis] + histories[j][axis] - histories[k][axis]
                            turns[axis] += -kerr - (first + second - third) - math.pi / 2.0
                            draws[axis] = 0.0
                        walked = (turns[0], turns[1])
                    linear = fwm_accumulated_phase(travelled, *offsets)
                    if self.pump_phase:
                        # In the split-step's own sense, exp(-i angle): the drive's
                        # walk against the landing carrier, this span's integral, and
                        # the frame that carrier has been turned into since.
                        phasors = [
                            np.exp(
                                2j * np.pi * draws[axis]
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
                            math.sqrt(generated[0]) * phasors[0] * np.exp(1j * skew[0]),
                            math.sqrt(generated[1]) * phasors[1] * np.exp(1j * skew[1]),
                        )
                    )

        if self.cascaded_fwm:
            # With carry_phase a term is only refused where it lands on a first-order
            # product of its own five legs (:meth:`_own_products`) -- the cross-phase
            # and self-feedback terms the split-step already carries, and
            # corrections to the first order this pass does not carry whole.
            # Landing on a band an earlier span made, or on one the ordinary pass
            # reached from a carried product, is a second-order term the reference
            # has, and adds to it like any other (maiman-z8j).
            occupied = [] if self.carry_phase else [b.f0 for b in sources] + [e[0] for e in found]
            found.extend(
                self._mix_cascade(
                    sources=sources,
                    reference=reference,
                    beta2=beta2,
                    powers=powers,
                    power_at=power_at,
                    coupled=coupled,
                    weight=weight,
                    gamma=gamma,
                    alpha=alpha,
                    distance=distance,
                    floor=floor,
                    occupied=occupied,
                    phases=phases,
                    travelled=travelled,
                    history=signal.nonlinear_history,
                    after=after,
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
