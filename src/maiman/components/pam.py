"""Four-level pulse amplitude modulation, and the equaliser a short-reach link needs.

Data-centre optics carry 100 and 200 Gb/s per lane on PAM4: two bits a symbol,
sent as four intensities, detected on a single photodiode. It halves the symbol
rate NRZ would need, and pays for it three ways -- the eye is a third the height,
the modulator's curve compresses the outer eyes, and a receiver bandwidth that
was generous for NRZ smears one symbol into the next. The first is the price;
the second is what a driver's linearity correction is for; the third is what a
feed-forward equaliser, and after it a decision-feedback one, are for.
"""

from __future__ import annotations

import math

import numpy as np

from ..analysis import blind_sample_offset
from ..component import BoolParam, Component, Param, PortType
from ..context import SimulationContext
from ..dsp import ffe_dfe_equalize
from ..modulation import bits_to_indices, gray_pam_levels, indices_to_bits, ser_pam
from ..signals import BinarySignal, ElectricalSignal, PAMMeasurement, Signal

#: Two bits a symbol, four levels. The module is PAM4 by name because that is the
#: format the parts around it exist for; the arithmetic below is not specific to it.
BITS_PER_SYMBOL = 2
LEVELS = 1 << BITS_PER_SYMBOL


class PAM4Driver(Component):
    """Two bits in, one of four voltages out, held for the symbol.

    The bits are taken in pairs, most significant first, and Gray coded, so the
    two middle levels differ from each other and from their outer neighbours in
    one bit each: a symbol error between adjacent levels costs one bit, not two.
    The source must emit two bits per symbol -- a :class:`PRBSGenerator` with
    ``bits_per_symbol`` set to 2.

    **Without** ``predistort`` the four voltages are evenly spaced between
    ``v_low`` and ``v_high``, and a Mach-Zehnder's ``cos^2`` turns them into
    optical powers that are not: the eyes nearest the curve's knees close first.
    **With** it, the voltages are chosen so that a modulator with this ``v_pi``,
    biased at full transmission as :class:`MachZehnderModulator` is, emits four
    *powers* evenly spaced between the ones ``v_low`` and ``v_high`` give -- the
    linearity correction a PAM4 transmitter's DSP applies. Both voltages must then
    lie within one ``v_pi`` of the bias, on one side of the curve.
    """

    display_name = "PAM4 Driver"
    category = "Electrical"

    v_low = Param(0.0, unit="V", doc="Voltage of the lowest level")
    v_high = Param(1.0, unit="V", doc="Voltage of the highest level")
    predistort = BoolParam(False, doc="Space the modulator's output powers evenly, not the volts")
    v_pi = Param(
        4.0,
        unit="V",
        min=0.0,
        doc="The modulator's V_pi, as the correction believes it",
        applies_when="predistort",
    )

    inputs = {"in": PortType.BINARY}
    outputs = {"out": PortType.ELECTRICAL}

    def level_voltages(self) -> np.ndarray:
        """The voltage of each level, lowest amplitude first."""
        fractions = np.arange(LEVELS) / (LEVELS - 1)
        low, high = self.si("v_low"), self.si("v_high")
        if not self.predistort:
            return low + fractions * (high - low)
        v_pi = self.si("v_pi")
        if not (0.0 <= low <= v_pi and 0.0 <= high <= v_pi):
            raise ValueError(
                f"{self.label}: predistortion inverts one side of the modulator's curve, so "
                f"both levels must lie in [0, v_pi] = [0, {v_pi:g}] V; got {low:g} and {high:g}"
            )
        transmission = np.cos(math.pi * np.array([low, high]) / (2.0 * v_pi)) ** 2
        targets = transmission[0] + fractions * (transmission[1] - transmission[0])
        return (2.0 * v_pi / math.pi) * np.arccos(np.sqrt(np.clip(targets, 0.0, 1.0)))

    def run(self, ctx: SimulationContext, inputs: dict[str, Signal]) -> dict[str, Signal]:
        binary: BinarySignal = inputs["in"]
        expected = ctx.sequence_length * BITS_PER_SYMBOL
        if binary.num_bits != expected:
            raise ValueError(
                f"{self.label}: PAM4 takes two bits a symbol, so a {ctx.sequence_length}-symbol "
                f"window needs {expected} bits and got {binary.num_bits}. Set the source's "
                "bits_per_symbol to 2."
            )
        amplitudes = gray_pam_levels(BITS_PER_SYMBOL)[bits_to_indices(binary.bits, BITS_PER_SYMBOL)]
        order = ((amplitudes + (LEVELS - 1)) / 2.0).astype(np.int64)
        samples = np.repeat(self.level_voltages()[order], ctx.samples_per_symbol)
        return {
            "out": ElectricalSignal(
                samples=samples.astype(ctx.real_dtype), fs=ctx.sample_rate, unit="V"
            )
        }


class FFEDFEEqualizer(Component):
    """Equalises a received PAM4 waveform, decides it, and counts what it got wrong.

    One sample a symbol is taken, at the instant where the symbols are most spread
    -- :func:`maiman.analysis.blind_sample_offset`, unless ``sample_offset`` fixes
    one -- and passed through ``ffe_taps`` of feed-forward equalisation and
    ``dfe_taps`` of decision feedback (:func:`maiman.dsp.ffe_dfe_equalize`).

    **Trained on the reference, measured on its own decisions.** The taps are
    adapted against the transmitted symbols, as a reference receiver in a
    compliance test is; the errors are then counted with the taps frozen and the
    feedback fed the equaliser's own decisions, so a DFE's error propagation is in
    the count. ``ffe_taps = 1`` and ``dfe_taps = 0`` is no equalisation at all --
    a gain, an offset and a slicer -- which is the baseline the others improve on.

    The result also carries :func:`maiman.modulation.ser_pam` at the equalised
    SNR, so a count far above it says the errors are not Gaussian noise: residual
    ISI, or a DFE propagating its own mistakes.
    """

    display_name = "FFE/DFE Equalizer"
    category = "DSP"

    ffe_taps = Param(7.0, unit="", min=1.0, max=63.0, doc="Feed-forward taps; odd, centred")
    dfe_taps = Param(0.0, unit="", min=0.0, max=16.0, doc="Decision-feedback taps; 0 disables")
    step = Param(0.05, unit="", min=1e-4, max=1.0, doc="NLMS adaptation step")
    training_passes = Param(4.0, unit="", min=1.0, max=50.0, doc="Sweeps of training")
    sample_offset = Param(
        -1.0, unit="", doc="Samples into the symbol to take; negative finds it blind"
    )

    inputs = {"in": PortType.ELECTRICAL, "reference": PortType.BINARY}
    outputs = {"out": PortType.METRIC}

    def run(self, ctx: SimulationContext, inputs: dict[str, Signal]) -> dict[str, Signal]:
        waveform: ElectricalSignal = inputs["in"]
        reference: BinarySignal = inputs["reference"]
        taps = int(self.ffe_taps)
        if taps % 2 == 0:
            raise ValueError(f"{self.label}: ffe_taps must be odd, got {taps}")
        expected_bits = ctx.sequence_length * BITS_PER_SYMBOL
        if reference.num_bits != expected_bits:
            raise ValueError(
                f"{self.label}: the reference carries {reference.num_bits} bits and a PAM4 "
                f"window of {ctx.sequence_length} symbols needs {expected_bits}"
            )

        grid = np.asarray(waveform.samples, dtype=np.float64).reshape(
            ctx.sequence_length, ctx.samples_per_symbol
        )
        offset = blind_sample_offset(grid) if self.sample_offset < 0 else int(self.sample_offset)
        if not 0 <= offset < ctx.samples_per_symbol:
            raise ValueError(
                f"{self.label}: sample_offset must be in "
                f"[0, {ctx.samples_per_symbol}), got {offset}"
            )

        pattern = bits_to_indices(reference.bits, BITS_PER_SYMBOL)
        gray = gray_pam_levels(BITS_PER_SYMBOL)
        sent = gray[pattern]
        result = ffe_dfe_equalize(
            grid[:, offset],
            sent,
            levels=gray,
            ffe_taps=taps,
            dfe_taps=int(self.dfe_taps),
            step=self.step,
            passes=int(self.training_passes),
        )

        alphabet = np.sort(gray)
        decided_amplitude = alphabet[result.decisions]
        # Back from amplitude to the bit pattern that Gray code maps to it.
        pattern_of = {float(level): index for index, level in enumerate(gray)}
        decided_pattern = np.array([pattern_of[float(a)] for a in decided_amplitude])
        decided_bits = indices_to_bits(decided_pattern, BITS_PER_SYMBOL)

        mean_power = float(np.mean(sent**2))
        snr = mean_power / result.mse if result.mse > 0.0 else math.inf
        return {
            "out": PAMMeasurement(
                levels=LEVELS,
                symbols_evaluated=ctx.sequence_length,
                symbol_errors=int(np.count_nonzero(decided_amplitude != sent)),
                bits_evaluated=expected_bits,
                bit_errors=int(np.count_nonzero(decided_bits != reference.bits)),
                snr_db=10.0 * math.log10(snr) if math.isfinite(snr) else math.inf,
                ser_expected=ser_pam(snr, LEVELS) if math.isfinite(snr) else 0.0,
                sample_offset=offset,
                ffe_taps=tuple(float(t) for t in result.ffe),
                dfe_taps=tuple(float(t) for t in result.dfe),
            )
        }
