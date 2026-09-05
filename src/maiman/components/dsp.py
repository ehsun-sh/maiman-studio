"""Receiver DSP blocks.

Ordered the way a coherent receiver orders them: the static, deterministic
impairment is removed first at the sample rate, and only then does an adaptive
filter run on what is left. See :mod:`maiman.dsp` for why that split is not a
matter of taste.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..component import BoolParam, Component, Param, PortType
from ..context import SimulationContext
from ..dsp import (
    butterfly_equalize,
    compensate_dispersion,
    dispersive_spread,
    estimate_dispersion,
)
from ..signals import ElectricalSignal, Signal, SymbolSignal


@dataclass(frozen=True)
class DispersionDiagnostics:
    """What the compensator was asked to remove.

    A filter this aggressive should say what it did. The spread in symbol periods
    is the number worth reading: it is what decides whether the adaptive stage
    downstream had any chance, and it makes a misplaced decimal point in the
    dispersion parameter obvious at a glance.
    """

    accumulated_dispersion: float
    """Dispersion removed [s/m]."""

    removed_symbols: float
    """Symbol periods of smearing that corresponds to."""

    estimated: bool
    """Whether that value was searched for rather than declared."""

    declared: float
    """The parameter as set [s/m].

    Reported even when the value was estimated, and *especially* then: it is what
    turns the blind search from a number to be trusted into a number to be
    checked. Set the parameter to what the span should be, switch the search on,
    and the two sit side by side.
    """

    contrast: float
    """Sharpness of the acquisition peak, or 0 when nothing was searched for.

    See :class:`maiman.dsp.DispersionEstimate` for why a high value is not a
    guarantee.
    """

    def __repr__(self) -> str:
        how = "estimated" if self.estimated else "declared"
        return (
            f"DispersionDiagnostics({self.accumulated_dispersion * 1e3:.1f} ps/nm removed, "
            f"{how}, {self.removed_symbols:.1f} symbols of spread)"
        )


class DispersionCompensator(Component):
    """Removes accumulated chromatic dispersion from the received baseband.

    Coherent detection hands back the complex field, so the dispersion the fibre
    applied is still there in full and still invertible — one static all-pass
    filter undoes it. This is what makes a coherent receiver worth building. A
    direct-detection receiver squares the field at the photodiode, destroys the
    phase, and can never do this at all; it has to carry dispersion-compensating
    fibre in the line, with the loss and the nonlinearity that come with it.

    **Why it is not the butterfly's job.** Both are linear filters, so in
    principle one adaptive filter could do both. In practice they have opposite
    requirements: dispersion is static and *long* — 80 km at 32 GBd smears a
    symbol over about thirteen of its neighbours — while polarization mixing is
    short and *fast*, drifting on a millisecond timescale. One filter serving both
    would have to be hundreds of taps and adapt quickly, which converges slowly
    and costs enormously. Splitting them makes each easy.

    ``accumulated_dispersion`` is D·L for the whole path, in ps/nm, and is
    positive for standard fibre. Getting it wrong is not a soft failure: the
    filter then adds dispersion over part of the range instead of removing it,
    so an error of a given size is roughly as damaging as that much uncompensated
    span — over 80 km at 32 GBd, 20 ps/nm out of 1360 costs eight decibels of
    SNR. The ``diagnostics`` port reports how many symbol periods of smearing the
    value corresponds to, so a figure wrong by an order of magnitude is visible
    as a number rather than only as a bad EVM.

    **Set ``estimate`` and it finds the value itself.** No deployed receiver is
    told how long its fibre is; it measures the dispersion from the signal during
    acquisition, before anything downstream has converged. Switching this on does
    the same thing — a clock-tone scan over ``±search_range`` followed by a
    modulus refinement, described in :func:`maiman.dsp.estimate_dispersion`,
    which also gives the measured accuracy and the conditions it needs. The
    declared value is then ignored as an input and reported in the diagnostics
    beside the estimate, which is the useful way to use both: declare what the
    span should be, and read off how close a blind receiver would have got.

    ``wavelength`` must be the *signal* wavelength, since β₂ scales as λ².
    """

    display_name = "CD Compensator"
    category = "DSP"

    accumulated_dispersion = Param(
        0.0, unit="ps/nm", doc="Total D*L to remove; positive for standard fiber (0 disables)"
    )
    wavelength = Param(1550.0, unit="nm", min=1.0, doc="Signal wavelength, for the lambda^2 in b2")
    estimate = BoolParam(
        False, doc="Measure the dispersion from the signal instead of using the declared value"
    )
    # Wide enough for a thousand kilometres of standard fibre, which is where the
    # rest of the coherent chain has been validated. Costs one dot product per
    # grid point, so the default is generous rather than tuned.
    search_range = Param(
        20000.0,
        unit="ps/nm",
        min=0.0,
        doc="Half-width of the blind search; only used when estimate is on",
    )

    inputs = {"i": PortType.ELECTRICAL, "q": PortType.ELECTRICAL}
    outputs = {
        "i": PortType.ELECTRICAL,
        "q": PortType.ELECTRICAL,
        "diagnostics": PortType.METRIC,
    }

    def run(self, ctx: SimulationContext, inputs: dict[str, Signal]) -> dict[str, Signal]:
        in_phase: ElectricalSignal = inputs["i"]
        quadrature: ElectricalSignal = inputs["q"]

        if in_phase.num_samples != quadrature.num_samples:
            raise ValueError(
                f"{self.label}: I and Q differ in length, "
                f"{in_phase.num_samples} and {quadrature.num_samples}"
            )
        if in_phase.fs != quadrature.fs:
            raise ValueError(
                f"{self.label}: I and Q are sampled at different rates, "
                f"{in_phase.fs} and {quadrature.fs} Hz"
            )

        declared = self.si("accumulated_dispersion")
        wavelength = self.si("wavelength")

        baseband = np.asarray(in_phase.samples).astype(np.complex128)
        baseband = baseband + 1j * np.asarray(quadrature.samples)

        contrast = 0.0
        accumulated = declared
        if self.estimate:
            found = estimate_dispersion(
                baseband,
                in_phase.fs,
                symbol_rate=ctx.bit_rate,
                wavelength=wavelength,
                search_range=self.si("search_range"),
            )
            accumulated, contrast = found.accumulated_dispersion, found.contrast

        if accumulated != 0.0:
            baseband = compensate_dispersion(
                baseband,
                in_phase.fs,
                accumulated_dispersion=accumulated,
                wavelength=wavelength,
            )

        # Reported against the occupied bandwidth rather than the sample rate: the
        # signal only has energy out to (1 + roll_off) * Rs, and quoting the spread
        # over the empty rest of the band would inflate it several times over.
        occupied = ctx.bit_rate * 1.2
        diagnostics = DispersionDiagnostics(
            accumulated_dispersion=accumulated,
            removed_symbols=dispersive_spread(accumulated, occupied, wavelength, ctx.bit_rate),
            estimated=self.estimate,
            declared=declared,
            contrast=contrast,
        )

        return {
            "i": ElectricalSignal(
                samples=baseband.real.astype(ctx.real_dtype), fs=in_phase.fs, unit=in_phase.unit
            ),
            "q": ElectricalSignal(
                samples=baseband.imag.astype(ctx.real_dtype), fs=quadrature.fs, unit=quadrature.unit
            ),
            "diagnostics": diagnostics,
        }


class ButterflyEqualizer(Component):
    """Separates two polarization tributaries that the channel has mixed.

    A dual-polarization receiver measures the field on its own axes, and a fibre
    rotates the launched state arbitrarily before it gets there. What arrives is
    two *mixtures*, not two channels. This 2x2 adaptive filter is what turns them
    back into channels, and without it a dual-polarization link recovers nothing
    at all past a small rotation — not a degraded version of the data, nothing.

    It is blind: no training sequence, no reference. The filters are adapted to
    drive each output onto a modulus the constellation actually uses, which a
    clean tributary satisfies and a mixture of two independent ones does not.
    See :func:`maiman.dsp.butterfly_equalize` for the two-stage scheme and for the
    45-degree saddle that the initialisation is tilted to avoid.

    **Which output is which is not determined.** Nothing in a blind cost function
    labels the tributaries, so the filter may deliver them swapped, and each
    carries its own arbitrary phase from the same quadrant ambiguity that
    :class:`~maiman.components.coherent.CarrierRecovery` has. A deployed link
    resolves both by framing and differential encoding; here the measurement
    block resolves them because it holds the reference.
    """

    display_name = "Butterfly Equalizer"
    category = "DSP"

    taps = Param(7.0, unit="", min=1.0, max=65.0, doc="Filter length; must be odd")
    step = Param(3e-3, unit="", min=1e-6, doc="Adaptation step size")
    passes = Param(2.0, unit="", min=1.0, max=8.0, doc="Times the sequence is run through")

    inputs = {"x": PortType.SYMBOL, "y": PortType.SYMBOL}
    outputs = {"x_out": PortType.SYMBOL, "y_out": PortType.SYMBOL}

    def run(self, ctx: SimulationContext, inputs: dict[str, Signal]) -> dict[str, Signal]:
        tributary_x: SymbolSignal = inputs["x"]
        tributary_y: SymbolSignal = inputs["y"]

        if tributary_x.num_symbols != tributary_y.num_symbols:
            raise ValueError(
                f"{self.label}: tributaries differ in length, "
                f"{tributary_x.num_symbols} and {tributary_y.num_symbols}"
            )
        if tributary_x.order != tributary_y.order:
            raise ValueError(
                f"{self.label}: tributaries carry different constellations, "
                f"{tributary_x.order} and {tributary_y.order} points"
            )

        taps = int(self.taps)
        if taps % 2 == 0:
            raise ValueError(f"{self.label}: taps must be odd, got {taps}")

        constellation = np.asarray(tributary_x.constellation)
        out_x, out_y, _ = butterfly_equalize(
            np.asarray(tributary_x.symbols),
            np.asarray(tributary_y.symbols),
            constellation,
            taps=taps,
            step=self.step,
            passes=int(self.passes),
        )
        return {
            "x_out": SymbolSignal(
                symbols=out_x, symbol_rate=tributary_x.symbol_rate, constellation=constellation
            ),
            "y_out": SymbolSignal(
                symbols=out_y, symbol_rate=tributary_y.symbol_rate, constellation=constellation
            ),
        }
