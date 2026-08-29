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
    dispersion_to_beta2,
    fwm_phase_mismatch,
    fwm_product_power,
    propagate_coupled_ssfm,
    propagate_dispersion,
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

    **Channels interact.** Bands do not propagate independently: each one's phase
    is rotated by every other one's power, at twice the rate its own power
    rotates it, and triplets of them mix to generate light at new frequencies.
    Both fall out of the same ``|A|**2 A`` term that produces self-phase
    modulation — see :func:`oosim.kernels.propagate_coupled_ssfm` for the count
    that gives cross-phase modulation its factor of two, and
    :func:`oosim.kernels.fwm_product_power` for the one that gives four-wave
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

    Not yet modelled: the dispersion slope (beta3) and Raman scattering. The
    Kerr coupling is scalar per polarization throughout, so a channel is
    modulated by its neighbours' co-polarized power and not by their orthogonal
    power, which really contributes a third as much; cross-polarization
    modulation is absent for the same reason. Mixing products accumulate from
    span to span in power rather than in field, which understates the coherent
    build-up a dispersion-managed link produces. Pump depletion is not modelled,
    which matters only at powers no link is operated at.
    """

    display_name = "Optical Fiber"
    category = "Fiber"

    length = Param(80.0, unit="km", min=0.0, doc="Fiber span length")
    attenuation = Param(0.2, unit="dB/km", min=0.0, doc="Attenuation coefficient")
    dispersion = Param(
        0.0, unit="ps/nm/km", doc="Dispersion parameter D at the band wavelength (0 disables)"
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
    pmd_coefficient = Param(0.0, unit="ps/sqrt(km)", min=0.0, doc="PMD coefficient (0 disables)")
    pmd_sections = Param(60.0, unit="", min=1.0, doc="Waveplates used to build the PMD realisation")

    inputs = {"in": PortType.OPTICAL}
    outputs = {"out": PortType.OPTICAL, "diagnostics": PortType.METRIC}

    def loss_db(self) -> float:
        """Total span loss [dB]."""
        # si() gives dB/m and metres, so their product is dB.
        return self.si("attenuation") * self.si("length")

    def beta2_at(self, wavelength: float) -> float:
        """Group-velocity dispersion beta2 [s^2/m] at ``wavelength`` [m]."""
        return dispersion_to_beta2(self.si("dispersion"), wavelength)

    def mean_dgd(self) -> float:
        """Expected differential group delay over this span [s].

        ``<DGD> = PMD_coefficient * sqrt(L)`` — the square root, not the length,
        because birefringence axes reorient randomly and the delay accumulates
        as a random walk rather than a sum.
        """
        return self.si("pmd_coefficient") * math.sqrt(self.si("length"))

    def walkoff_of(self, band: Band, reference: Band) -> float:
        """Inverse-group-velocity offset of ``band`` against ``reference`` [s/m].

        Evaluated with beta2 averaged over the two wavelengths, which is the
        trapezoidal value of ``integral beta2 domega`` and so is symmetric in
        the pair — using either end's beta2 alone would make the delay from A to
        B differ from the delay from B to A by the dispersion slope, an
        asymmetry this model does not have a beta3 to justify.
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

        if gamma != 0.0 and self.four_wave_mixing:
            bands, emitted = self._mix(ctx, signal, bands, gamma=gamma, alpha=alpha)
            diagnostics = replace(diagnostics, mixing_products=emitted)

        return {
            "out": OpticalSignal(
                bands=tuple(bands),
                noise=tuple(n.scale_power(power_factor) for n in signal.noise),
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
                propagate_dispersion(band.Ex, band.fs, self.beta2_at(band.wavelength), distance)
                * amplitude,
                propagate_dispersion(band.Ey, band.fs, self.beta2_at(band.wavelength), distance)
                * amplitude,
            )
            for band in signal.bands
        ]
        return fields, PropagationDiagnostics(0, distance, 0.0, 0.0, 0.0)

    def _propagate_kerr(
        self, signal: OpticalSignal, distance: float, gamma: float, alpha: float
    ) -> tuple[list[tuple[np.ndarray, np.ndarray]], PropagationDiagnostics]:
        """Split-step propagation, with the bands coupled unless told otherwise.

        The two polarizations are propagated as two separate coupled systems,
        which is what makes the model scalar per polarization: a band's X field
        is modulated by every other band's X power and by nothing else.
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
            walkoff = [self.walkoff_of(band, reference) for band in group]
            solved = []
            for axis in ("Ex", "Ey"):
                out, diag = propagate_coupled_ssfm(
                    [getattr(band, axis) for band in group],
                    reference.fs,
                    beta2=beta2,
                    walkoff=walkoff,
                    gamma=gamma,
                    alpha=alpha,
                    distance=distance,
                    max_nonlinear_phase=self.max_nonlinear_phase,
                    max_walkoff_slip=self.max_walkoff_slip,
                )
                solved.append(out)
                if diag.steps > diagnostics.steps:
                    diagnostics = diag
            fields.extend(zip(solved[0], solved[1], strict=True))
        return fields, diagnostics

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

        Products are accumulated as complex amplitudes rather than as powers,
        each with its own drawn phase, so that several triplets landing on one
        frequency add the way independent waves do — on average in power, but
        with the spread that a link actually sees. A product whose frequency
        coincides with a band already present is added *into* that band, at DC
        in its own rotating frame, which is what makes in-band mixing the
        unfilterable crosstalk it is: on a uniform grid every product lands on a
        channel, and no receiver can separate it afterwards.
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
        beta2 = self.beta2_at(reference.wavelength)
        powers = [
            (float(np.mean(np.abs(b.Ex) ** 2)), float(np.mean(np.abs(b.Ey) ** 2))) for b in sources
        ]
        rng = ctx.rng("Fiber", self.label, "fwm")

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
                    phasor = np.exp(2j * np.pi * float(rng.random()))
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
