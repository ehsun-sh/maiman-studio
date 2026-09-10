"""Electrical sources and drivers — the transmitter's input side."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..component import BoolParam, Component, Param, PortType
from ..context import SimulationContext
from ..dsp import root_raised_cosine, shape_symbols
from ..fec import (
    CODEWORD_SYMBOLS,
    MESSAGE_SYMBOLS,
    decode_blocks,
    encode_blocks,
    overhead,
)
from ..modulation import QAM_FORMATS
from ..signals import BinarySignal, ElectricalSignal, Signal, SoftSignal, SymbolSignal
from ..softfec import StaircaseCode, staircase_code, staircase_decode, staircase_encode

#: Maximal-length LFSR feedback taps, as (order, tap) exponent pairs.
#: These are the standard polynomials used across optical test equipment;
#: see ITU-T O.150 and IEEE 802.3 for where each order is specified.
PRBS_TAPS: dict[int, tuple[int, int]] = {
    7: (7, 6),
    9: (9, 5),
    11: (11, 9),
    15: (15, 14),
    23: (23, 18),
    31: (31, 28),
}


class PRBSGenerator(Component):
    """Pseudo-random binary sequence from a maximal-length LFSR.

    A PRBS of order n has period ``2**n - 1`` and every n-bit window appears
    exactly once per period except all-zeros. Those properties are what make it a
    test pattern rather than merely random bits, and they are asserted directly
    in the test suite.

    If the requested sequence is longer than one period the pattern repeats,
    which is the intended behaviour — a receiver measuring BER over many periods
    is the normal case.
    """

    display_name = "PRBS Generator"
    category = "Electrical Sources"

    #: Offered as the set it is, taken from the tap table above rather than
    #: restated: an order between two of these has no maximal-length polynomial
    #: here, and declaring 7..31 meant the interface accepted 12.
    order = Param(
        7.0,
        unit="",
        choices=tuple(float(n) for n in sorted(PRBS_TAPS)),
        doc="LFSR order n; period is 2**n - 1. Only these have a standard "
        "maximal-length polynomial",
    )
    bits_per_symbol = Param(
        1.0,
        unit="",
        choices=QAM_FORMATS,
        doc="Bits each downstream symbol carries; scales how many bits are emitted",
    )

    outputs = {"out": PortType.BINARY}

    def run(self, ctx: SimulationContext, inputs: dict[str, Signal]) -> dict[str, Signal]:
        # No order check here any more. It used to live in this method and fire
        # at run time; now `order` declares the tap table as its choices, so
        # every supported way of setting it -- construction, loading a project,
        # a sweep override -- is refused by Param.validate when it is set. A
        # guard that cannot fire is worse than no guard: it reads as protection
        # and tests nothing.
        order = int(self.order)
        n, tap = PRBS_TAPS[order]

        # Seed the register from the run seed so the pattern is reproducible but
        # not identical across differently-seeded runs. An all-zero state is a
        # fixed point of the LFSR, so it is excluded.
        rng = ctx.rng("PRBSGenerator", self.label, order)
        state = int(rng.integers(1, 1 << n))

        # The window is measured in symbols, not bits. A binary format consumes
        # one bit per symbol; a higher-order one consumes several, and the
        # generator has to emit enough of them to fill the same window.
        per_symbol = int(self.bits_per_symbol)
        count = ctx.sequence_length * per_symbol

        bits = np.empty(count, dtype=np.uint8)
        for i in range(count):
            feedback = ((state >> (n - 1)) ^ (state >> (tap - 1))) & 1
            bits[i] = state & 1
            state = ((state << 1) | feedback) & ((1 << n) - 1)

        return {"out": BinarySignal(bits=bits, symbol_rate=ctx.bit_rate * per_symbol)}


#: Bits in one RS(255, 239) codeword on the line. The run window has to hold a
#: whole number of these: a codeword split across two runs cannot be decoded in
#: either, and silently padding one would report a corrected error rate for data
#: that was never sent.
CODED_BITS_PER_BLOCK = CODEWORD_SYMBOLS * 8
PAYLOAD_BITS_PER_BLOCK = MESSAGE_SYMBOLS * 8


@dataclass(frozen=True)
class FECReport:
    """What the decoder found, and what it did about it."""

    corrected_symbols: int
    """Byte errors repaired across the window."""

    failed_blocks: int
    """Codewords the decoder could not correct."""

    blocks: int
    """Codewords in the window."""

    pre_fec_errors: int
    """Bit errors on the line, before decoding."""

    post_fec_errors: int
    """Bit errors left in the payload, after decoding."""

    pre_fec_ber: float
    post_fec_ber: float

    def __repr__(self) -> str:
        return (
            f"FECReport(pre {self.pre_fec_ber:.2e} -> post {self.post_fec_ber:.2e}, "
            f"{self.corrected_symbols} symbols fixed, {self.failed_blocks}/{self.blocks} failed)"
        )


class FECEncoder(Component):
    """PRBS payload, Reed-Solomon coded, filling the line window.

    **Why this is a source rather than a filter.** A coder makes more bits than
    it is given — 255 for every 239 — and the run window here is a fixed number
    of symbols *on the line*. Those two facts settle the shape of the block
    between them: the window is the line rate, so coding does not make the line
    faster, it makes the payload smaller. An OTN framer does exactly this. A
    block that took bits in and handed 6.69 % more back would be refused by the
    mapper downstream, and rightly.

    So this generates its own payload, codes it, and emits a window that is
    exactly full. The uncoded payload leaves on a second port, which is what a
    :class:`BERAnalyzer` downstream of the decoder compares against — the same
    arrangement the shipped coherent project already uses to keep a differential
    and a non-differential copy of its symbols.

    The window must hold a whole number of codewords: ``sequence_length *
    bits_per_symbol`` has to be a multiple of 2040. A codeword split across two
    runs cannot be decoded in either.
    """

    display_name = "FEC Encoder"
    category = "Electrical Sources"

    order = Param(
        23.0,
        unit="",
        choices=tuple(float(n) for n in sorted(PRBS_TAPS)),
        doc="LFSR order for the payload pattern",
    )
    bits_per_symbol = Param(
        1.0,
        unit="",
        choices=QAM_FORMATS,
        doc="Bits each downstream symbol carries; sets the size of the window to fill",
    )

    outputs = {"out": PortType.BINARY, "payload": PortType.BINARY}

    def blocks_in_window(self, ctx: SimulationContext) -> int:
        """Codewords that fit the line window, or a refusal naming a size that fits."""
        per_symbol = int(self.bits_per_symbol)
        coded_bits = ctx.sequence_length * per_symbol
        blocks, remainder = divmod(coded_bits, CODED_BITS_PER_BLOCK)
        if blocks < 1 or remainder:
            symbols_per_block = CODED_BITS_PER_BLOCK // per_symbol
            nearest = max(round(ctx.sequence_length / symbols_per_block), 1) * symbols_per_block
            raise ValueError(
                f"{self.label}: a window of {ctx.sequence_length} symbols at {per_symbol} "
                f"bits/symbol is {coded_bits} bits, which is not a whole number of "
                f"{CODED_BITS_PER_BLOCK}-bit codewords. Use sequence_length={nearest}."
            )
        return blocks

    def run(self, ctx: SimulationContext, inputs: dict[str, Signal]) -> dict[str, Signal]:
        blocks = self.blocks_in_window(ctx)
        per_symbol = int(self.bits_per_symbol)

        order = int(self.order)
        n, tap = PRBS_TAPS[order]
        rng = ctx.rng("FECEncoder", self.label, order)
        state = int(rng.integers(1, 1 << n))

        count = blocks * PAYLOAD_BITS_PER_BLOCK
        payload = np.empty(count, dtype=np.uint8)
        for i in range(count):
            feedback = ((state >> (n - 1)) ^ (state >> (tap - 1))) & 1
            payload[i] = state & 1
            state = ((state << 1) | feedback) & ((1 << n) - 1)

        symbols = np.packbits(payload).reshape(blocks, MESSAGE_SYMBOLS)
        coded = encode_blocks(symbols.astype(np.int32))
        bits = np.unpackbits(coded.astype(np.uint8).reshape(-1))

        rate = ctx.bit_rate * per_symbol
        return {
            "out": BinarySignal(bits=bits, symbol_rate=rate),
            # The payload leaves at the rate it is actually carried at, which is
            # the line rate divided by the coding overhead. Quoting it at the
            # line rate would make a coded link look like free capacity.
            "payload": BinarySignal(bits=payload, symbol_rate=rate / (1.0 + overhead())),
        }


class FECDecoder(Component):
    """Reed-Solomon decoding, and an honest account of what it could not fix.

    Takes the received line bits, hands back the payload, and reports both error
    rates on its diagnostics port. Both, because the interesting number is
    neither one alone: a pre-FEC rate says what the optics did and a post-FEC
    rate says what the customer sees, and the whole design of a coded link is
    the distance between them.

    **A failed codeword is passed through, not dropped.** Its payload bytes go
    out as received, which is what a real decoder does and what makes the
    post-FEC error count meaningful — dropping them would report a perfect link
    that had silently lost data. ``failed_blocks`` says how many.

    **Miscorrection is not detectable from in here.** Past ``t`` errors the
    decoder usually finds no consistent error pattern, but it can instead find a
    wrong one and return a codeword that is confidently incorrect. Those bits
    are counted in the post-FEC rate like any others, because that is where they
    show up in reality; there is no flag for them because a real decoder has
    none either.
    """

    display_name = "FEC Decoder"
    category = "Measurements"

    inputs = {"in": PortType.BINARY, "payload": PortType.BINARY}
    outputs = {"out": PortType.BINARY, "diagnostics": PortType.METRIC}

    def run(self, ctx: SimulationContext, inputs: dict[str, Signal]) -> dict[str, Signal]:
        received: BinarySignal = inputs["in"]
        reference: BinarySignal = inputs["payload"]

        blocks, remainder = divmod(received.num_bits, CODED_BITS_PER_BLOCK)
        if blocks < 1 or remainder:
            raise ValueError(
                f"{self.label}: got {received.num_bits} bits, which is not a whole number "
                f"of {CODED_BITS_PER_BLOCK}-bit codewords"
            )
        if reference.num_bits != blocks * PAYLOAD_BITS_PER_BLOCK:
            raise ValueError(
                f"{self.label}: {blocks} codewords carry "
                f"{blocks * PAYLOAD_BITS_PER_BLOCK} payload bits but the reference "
                f"has {reference.num_bits}"
            )

        line = np.asarray(received.bits, dtype=np.uint8)
        symbols = np.packbits(line).reshape(blocks, CODEWORD_SYMBOLS).astype(np.int32)

        # The pre-FEC rate is measured against the codeword the encoder built,
        # not against the payload — otherwise the parity bits, which are a third
        # of what the optics carried at these sizes, would not be counted.
        transmitted = encode_blocks(
            np.packbits(np.asarray(reference.bits, dtype=np.uint8))
            .reshape(blocks, MESSAGE_SYMBOLS)
            .astype(np.int32)
        )
        pre_errors = int(
            np.count_nonzero(np.unpackbits(np.bitwise_xor(symbols, transmitted).astype(np.uint8)))
        )

        result = decode_blocks(symbols)
        payload_bits = np.unpackbits(result.messages.astype(np.uint8).reshape(-1))
        post_errors = int(
            np.count_nonzero(payload_bits ^ np.asarray(reference.bits, dtype=np.uint8))
        )

        coded_bits = blocks * CODED_BITS_PER_BLOCK
        payload_count = blocks * PAYLOAD_BITS_PER_BLOCK
        report = FECReport(
            corrected_symbols=result.corrected,
            failed_blocks=result.failed,
            blocks=blocks,
            pre_fec_errors=pre_errors,
            post_fec_errors=post_errors,
            pre_fec_ber=pre_errors / coded_bits,
            post_fec_ber=post_errors / payload_count,
        )
        return {
            "out": BinarySignal(bits=payload_bits, symbol_rate=reference.symbol_rate),
            "diagnostics": report,
        }


#: The staircase this project ships, built once. A block is 128x128 bits, so a
#: run window has to hold a whole number of 16384-bit blocks.
SOFT_CODE: StaircaseCode = staircase_code()
SOFT_BLOCK_BITS = SOFT_CODE.half * SOFT_CODE.half


@dataclass(frozen=True)
class SoftFECReport:
    """What the soft decoder found, and what it did about it."""

    blocks: int
    corrections: int
    """Bit decisions the component decoders moved, summed over every pass."""

    pre_fec_errors: int
    post_fec_errors: int
    pre_fec_ber: float
    post_fec_ber: float

    def __repr__(self) -> str:
        return (
            f"SoftFECReport(pre {self.pre_fec_ber:.2e} -> post {self.post_fec_ber:.2e}, "
            f"{self.blocks} blocks)"
        )


class SoftFECEncoder(Component):
    """PRBS payload, staircase-coded, filling the line window.

    A source, for the same reason :class:`FECEncoder` is one: a coder makes more
    bits than it is given and the run window is a fixed number of symbols *on the
    line*, so coding shrinks the payload rather than speeding the line. See that
    class for the argument in full.

    The code is the braided BCH staircase in :mod:`maiman.softfec` at 16.4 %
    overhead — **not** OIF's oFEC, which is normative about an interleaver and a
    framing this does not reproduce. What it is is the family oFEC belongs to,
    built from the open literature, and the thing that makes it worth having in
    a link is that it decodes on log-likelihood ratios rather than on bits.

    The window must hold a whole number of 16384-bit blocks, and it wants
    **several**: the last block of a stream has no successor stripe, so half its
    protection is missing until the next one arrives. Two blocks is the minimum
    that runs; more is a better measurement.
    """

    display_name = "Soft FEC Encoder"
    category = "Electrical Sources"

    order = Param(
        23.0,
        unit="",
        choices=tuple(float(n) for n in sorted(PRBS_TAPS)),
        doc="LFSR order for the payload pattern",
    )
    bits_per_symbol = Param(
        1.0,
        unit="",
        choices=QAM_FORMATS,
        doc="Bits each downstream symbol carries; sets the size of the window to fill",
    )

    outputs = {"out": PortType.BINARY, "payload": PortType.BINARY}

    def blocks_in_window(self, ctx: SimulationContext) -> int:
        """Staircase blocks that fit the line window, or a refusal naming one that does."""
        per_symbol = int(self.bits_per_symbol)
        coded_bits = ctx.sequence_length * per_symbol
        blocks, remainder = divmod(coded_bits, SOFT_BLOCK_BITS)
        if blocks < 1 or remainder:
            symbols_per_block = SOFT_BLOCK_BITS // per_symbol
            nearest = max(round(ctx.sequence_length / symbols_per_block), 1) * symbols_per_block
            raise ValueError(
                f"{self.label}: a window of {ctx.sequence_length} symbols at {per_symbol} "
                f"bits/symbol is {coded_bits} bits, which is not a whole number of "
                f"{SOFT_BLOCK_BITS}-bit staircase blocks. Use sequence_length={nearest}."
            )
        return blocks

    def run(self, ctx: SimulationContext, inputs: dict[str, Signal]) -> dict[str, Signal]:
        blocks = self.blocks_in_window(ctx)
        per_symbol = int(self.bits_per_symbol)
        half, columns = SOFT_CODE.half, SOFT_CODE.information_columns

        order = int(self.order)
        n, tap = PRBS_TAPS[order]
        rng = ctx.rng("SoftFECEncoder", self.label, order)
        state = int(rng.integers(1, 1 << n))

        count = blocks * half * columns
        payload = np.empty(count, dtype=np.uint8)
        for i in range(count):
            feedback = ((state >> (n - 1)) ^ (state >> (tap - 1))) & 1
            payload[i] = state & 1
            state = ((state << 1) | feedback) & ((1 << n) - 1)

        coded = staircase_encode(SOFT_CODE, payload.reshape(blocks, half, columns))

        rate = ctx.bit_rate * per_symbol
        return {
            "out": BinarySignal(bits=coded.reshape(-1), symbol_rate=rate),
            "payload": BinarySignal(bits=payload, symbol_rate=rate * SOFT_CODE.rate),
        }


class SoftFECDecoder(Component):
    """Iterative staircase decoding, on log-likelihood ratios.

    Takes the demapper's soft output rather than a slicer's bits, which is the
    entire point: the two to three decibels a soft code earns come from knowing
    *how* near each symbol was to a decision boundary, and a hard input has
    thrown that away before this block sees it. The port type is what makes
    wiring a hard signal in here impossible rather than merely wrong.

    Reports both error rates for the reason :class:`FECDecoder` does — the design
    of a coded link is the distance between them — and the number of decisions
    the component decoders moved, which is what says the iteration did anything
    at all.

    ``iterations`` is a real cost. Each pass Chase-decodes every row of every
    stripe, so doubling it doubles the run; the shipped default stops early when
    a pass moves nothing, which on a working link is usually after three or four.
    """

    display_name = "Soft FEC Decoder"
    category = "Measurements"

    iterations = Param(
        8.0, unit="", min=1.0, max=32.0, doc="Maximum decoding passes; it stops early"
    )
    test_bits = Param(
        4.0,
        unit="",
        min=0.0,
        max=6.0,
        doc="Least-reliable positions Chase flips; cost is 2**this per codeword",
    )

    inputs = {"in": PortType.SOFT, "payload": PortType.BINARY}
    outputs = {"out": PortType.BINARY, "diagnostics": PortType.METRIC}

    def run(self, ctx: SimulationContext, inputs: dict[str, Signal]) -> dict[str, Signal]:
        received: SoftSignal = inputs["in"]
        reference: BinarySignal = inputs["payload"]
        half, columns = SOFT_CODE.half, SOFT_CODE.information_columns

        blocks, remainder = divmod(received.num_bits, SOFT_BLOCK_BITS)
        if blocks < 1 or remainder:
            raise ValueError(
                f"{self.label}: got {received.num_bits} log-likelihood ratios, which is "
                f"not a whole number of {SOFT_BLOCK_BITS}-bit staircase blocks"
            )
        if reference.num_bits != blocks * half * columns:
            raise ValueError(
                f"{self.label}: {blocks} blocks carry {blocks * half * columns} payload "
                f"bits but the reference has {reference.num_bits}"
            )

        llr = np.asarray(received.llr, dtype=np.float64).reshape(blocks, half, half)
        payload = np.asarray(reference.bits, dtype=np.uint8).reshape(blocks, half, columns)

        # The pre-FEC rate is measured against the transmitted *codeword*, not the
        # payload, so the parity bits the optics also carried are counted.
        transmitted = staircase_encode(SOFT_CODE, payload)
        pre_errors = int(np.count_nonzero((llr < 0.0).astype(np.uint8) != transmitted))

        decoded, corrections = staircase_decode(
            SOFT_CODE,
            llr,
            iterations=int(self.iterations),
            test_bits=int(self.test_bits),
        )
        post_errors = int(np.count_nonzero(decoded != payload))

        report = SoftFECReport(
            blocks=blocks,
            corrections=corrections,
            pre_fec_errors=pre_errors,
            post_fec_errors=post_errors,
            pre_fec_ber=pre_errors / (blocks * SOFT_BLOCK_BITS),
            post_fec_ber=post_errors / payload.size,
        )
        return {
            "out": BinarySignal(bits=decoded.reshape(-1), symbol_rate=reference.symbol_rate),
            "diagnostics": report,
        }


class NRZDriver(Component):
    """Non-return-to-zero driver: bits in, a rectangular voltage waveform out.

    Each bit is held for a full symbol period. The transition is instantaneous —
    a finite rise time and driver bandwidth are a later refinement, and adding
    them changes only this block.
    """

    display_name = "NRZ Driver"
    category = "Electrical"

    v_low = Param(0.0, unit="V", doc="Voltage representing a 0")
    v_high = Param(1.0, unit="V", doc="Voltage representing a 1")

    inputs = {"in": PortType.BINARY}
    outputs = {"out": PortType.ELECTRICAL}

    def run(self, ctx: SimulationContext, inputs: dict[str, Signal]) -> dict[str, Signal]:
        binary: BinarySignal = inputs["in"]
        if binary.num_bits != ctx.sequence_length:
            raise ValueError(
                f"{self.label}: got {binary.num_bits} bits but the run window holds "
                f"{ctx.sequence_length} symbols"
            )

        levels = np.where(binary.bits.astype(bool), self.si("v_high"), self.si("v_low"))
        samples = np.repeat(levels, ctx.samples_per_symbol)
        return {
            "out": ElectricalSignal(
                samples=samples.astype(ctx.real_dtype), fs=ctx.sample_rate, unit="V"
            )
        }


class IQDriver(Component):
    """Turns complex symbols into the two drive waveforms an IQ modulator wants.

    The real and imaginary parts are held for a full symbol, exactly as
    :class:`NRZDriver` holds a bit, and scaled so that the constellation's
    outermost quadrature level reaches ``drive_ratio * v_pi``.

    **Why pre-distortion lives here.** A child MZM biased at its null has *field*
    transmission ``sin(pi*V/(2*V_pi))`` — linear only for small drives. Driven at
    full swing it compresses the outer levels of a 16-QAM constellation while
    leaving QPSK untouched, because QPSK only ever uses the extremes. Real
    transmitters correct this in the DSP by pre-applying the inverse, and so does
    this block: with ``predistort`` on, the drive is
    ``(2*V_pi/pi) * arcsin(drive_ratio * f)`` and the field that emerges is
    proportional to the symbol. Turning it off leaves the compression visible,
    which is the point of being able to turn it off.

    ``v_pi`` is declared here as well as on the modulator because the correction
    genuinely needs it: a transmitter DSP that does not know the modulator's V_pi
    cannot linearise it. Setting the two to different values models exactly that
    mismatch.

    **Pulse shaping raises the peak, and the peak is what clips.** A held symbol
    never leaves the constellation's own levels, so a full-swing drive is exactly
    full swing. A root-raised-cosine waveform overshoots between symbols — that
    is what the tails of the pulse are — so its peak-to-average ratio is higher,
    and driving it at ``drive_ratio = 1`` runs the correction past ``arcsin(1)``
    and clips. The clipping is real and is left visible: at 16-QAM with a 0.2
    roll-off it costs about 7% EVM at full swing and about 1% backed off to 0.4.
    Backing the drive off is what a real transmitter does about it, and the
    residual 1% is the shaping filter's own truncation, set by ``filter_span``.
    """

    display_name = "IQ Driver"
    category = "Electrical"

    v_pi = Param(4.0, unit="V", min=0.0, doc="The modulator's V_pi, as the DSP believes it")
    drive_ratio = Param(
        1.0,
        unit="",
        min=0.01,
        max=1.0,
        doc="Peak drive as a fraction of V_pi; back off to linearise",
    )
    predistort = BoolParam(True, doc="Pre-invert the modulator's sine so the field is linear")
    pulse_shaping = BoolParam(
        False, doc="Root-raised-cosine shaping instead of holding each symbol flat"
    )
    roll_off = Param(0.2, unit="", min=0.0, max=1.0, doc="RRC excess bandwidth factor")
    filter_span = Param(
        16.0,
        unit="",
        min=2.0,
        max=64.0,
        doc="RRC length in symbols; longer leaves less residual ISI",
    )

    inputs = {"in": PortType.SYMBOL}
    outputs = {"i": PortType.ELECTRICAL, "q": PortType.ELECTRICAL}

    def _drive(self, quadrature: np.ndarray, peak: float) -> np.ndarray:
        """Map a normalised quadrature in [-1, 1] to a drive voltage [V]."""
        v_pi = self.si("v_pi")
        ratio = self.drive_ratio
        if self.predistort:
            return (2.0 * v_pi / np.pi) * np.arcsin(np.clip(ratio * quadrature / peak, -1.0, 1.0))
        return ratio * v_pi * quadrature / peak

    def run(self, ctx: SimulationContext, inputs: dict[str, Signal]) -> dict[str, Signal]:
        symbols: SymbolSignal = inputs["in"]
        if symbols.num_symbols != ctx.sequence_length:
            raise ValueError(
                f"{self.label}: got {symbols.num_symbols} symbols but the run window "
                f"holds {ctx.sequence_length}"
            )
        if self.si("v_pi") <= 0.0:
            raise ValueError(f"{self.label}: v_pi must be positive, got {self.v_pi}")

        # Normalise against the *alphabet*, not the symbols that happened to be
        # sent, so a short sequence that never uses an outer point still lands on
        # the same voltages as a long one.
        points = np.asarray(symbols.constellation).astype(np.complex128)
        peak = float(max(np.abs(points.real).max(), np.abs(points.imag).max()))
        if peak <= 0.0:
            raise ValueError(f"{self.label}: the constellation collapses to the origin")

        values = np.asarray(symbols.symbols).astype(np.complex128)

        # Shaping happens on the complex symbol, before the quadratures are taken
        # and before pre-distortion: the pulse shape is what the DSP sends to the
        # converters, and the arcsin correction is applied to the waveform that
        # results. Doing it the other way round would pre-distort a rectangle and
        # then smear the correction.
        if self.pulse_shaping:
            taps = root_raised_cosine(self.roll_off, int(self.filter_span), ctx.samples_per_symbol)
            waveform = shape_symbols(values, ctx.samples_per_symbol, taps)
        else:
            waveform = np.repeat(values, ctx.samples_per_symbol)

        out = {}
        for name, quadrature in (("i", waveform.real), ("q", waveform.imag)):
            samples = self._drive(quadrature, peak)
            out[name] = ElectricalSignal(
                samples=samples.astype(ctx.real_dtype), fs=ctx.sample_rate, unit="V"
            )
        return out


class DCVoltage(Component):
    """Constant voltage source.

    Exists mainly to characterise a modulator: holding the drive at a fixed
    voltage and sweeping it is how a transfer curve is measured on a bench, and
    it is how the MZM model is validated here.
    """

    display_name = "DC Voltage"
    category = "Electrical Sources"

    voltage = Param(0.0, unit="V", doc="Output voltage")

    outputs = {"out": PortType.ELECTRICAL}

    def run(self, ctx: SimulationContext, inputs: dict[str, Signal]) -> dict[str, Signal]:
        samples = np.full(ctx.num_samples, self.si("voltage"), dtype=ctx.real_dtype)
        return {"out": ElectricalSignal(samples=samples, fs=ctx.sample_rate, unit="V")}
