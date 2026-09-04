"""Bits to symbols and back — the digital edge of a coherent transceiver."""

from __future__ import annotations

import numpy as np

from ..component import BoolParam, Component, Param, PortType
from ..context import SimulationContext
from ..modulation import (
    QAM_FORMATS,
    bits_to_indices,
    differential_decode,
    differential_encode,
    nearest_indices,
    qam_constellation,
    quadrant_constellation,
)
from ..signals import BinarySignal, Signal, SymbolSignal


class QAMMapper(Component):
    """Groups bits into Gray-coded QAM symbols.

    ``bits_per_symbol`` selects the format: 1 is BPSK, 2 QPSK, 4 16-QAM, 6
    64-QAM, 8 256-QAM. The constellation is normalised to unit mean power, so
    changing format changes the information rate without changing the average
    optical power the laser is asked for.

    The run window holds :attr:`SimulationContext.sequence_length` *symbols*, so
    a source feeding this must supply that many times ``bits_per_symbol`` bits —
    which is what :class:`~maiman.components.electrical.PRBSGenerator`'s own
    ``bits_per_symbol`` is for. Mismatched lengths raise here rather than being
    silently truncated, because a truncated sequence still produces a BER.
    """

    display_name = "QAM Mapper"
    category = "Modulation"

    bits_per_symbol = Param(
        2.0,
        unit="",
        choices=QAM_FORMATS,
        doc="1 BPSK, 2 QPSK, 4 16-QAM, 6 64-QAM, 8 256-QAM",
    )
    differential = BoolParam(
        False,
        doc="Encode the quadrant differentially, so a quarter-turn ambiguity costs "
        "nothing. Needs at least 2 bits per symbol — BPSK has no quadrants",
    )

    inputs = {"in": PortType.BINARY}
    outputs = {"out": PortType.SYMBOL}

    def validate(self) -> None:
        """The one pair of settings in this library that can disagree.

        Differential encoding here works by relabelling the alphabet as
        (quadrant, position), so a quarter-turn ambiguity becomes a constant
        offset the decoder removes. BPSK has two points on a line and no
        quadrants at all, so there is nothing to relabel. Both settings are
        perfectly legal on their own, which is why neither the range nor the
        choices on either one can catch the combination.

        Checked here rather than in ``run``: this is knowable before the
        simulation starts, and finding out halfway through one wastes the work
        already done and reports the problem further from its cause.
        """
        if self.differential and int(self.bits_per_symbol) < 2:
            raise ValueError(
                f"{self.label}: differential quadrant encoding needs at least 2 bits "
                f"per symbol, got {int(self.bits_per_symbol)}. BPSK has no quadrants "
                f"to difference — turn differential off, and take any Differential "
                f"Decoder out of the chain with it."
            )

    def constellation(self) -> np.ndarray:
        """The alphabet, relabelled by quadrant when differential encoding is on."""
        if self.differential:
            return quadrant_constellation(int(self.bits_per_symbol))
        return qam_constellation(int(self.bits_per_symbol))

    def run(self, ctx: SimulationContext, inputs: dict[str, Signal]) -> dict[str, Signal]:
        binary: BinarySignal = inputs["in"]
        per_symbol = int(self.bits_per_symbol)

        expected = ctx.sequence_length * per_symbol
        if binary.num_bits != expected:
            raise ValueError(
                f"{self.label}: got {binary.num_bits} bits, but a window of "
                f"{ctx.sequence_length} symbols at {per_symbol} bits/symbol needs {expected}"
            )

        points = self.constellation()
        indices = bits_to_indices(np.asarray(binary.bits), per_symbol)
        if self.differential:
            indices = differential_encode(indices, points.shape[0] // 4)

        return {
            "out": SymbolSignal(
                symbols=points[indices],
                symbol_rate=ctx.bit_rate,
                constellation=points,
            )
        }


class DifferentialDecoder(Component):
    """Undoes :class:`QAMMapper`'s differential quadrant encoding.

    Slices the received symbols against the quadrant-labelled alphabet, differences
    the quadrant back out, and re-emits them under the ordinary Gray labelling — so
    what comes out can be compared directly against a plain mapper fed the same
    bits, with no special handling anywhere downstream.

    **What this buys.** Every blind stage upstream leaves a quarter-turn
    ambiguity: the phase search cannot resolve it, and the butterfly equaliser has
    it too. Absolute labelling turns that into total data loss. Differencing turns
    it into a constant that cancels, so a receiver that settles a quarter turn
    away decodes correctly anyway.

    **What it costs.** A symbol error that crosses a quadrant boundary corrupts
    two consecutive symbols instead of one, because each quadrant is decoded
    relative to its predecessor. Roughly a factor of two in error rate near
    threshold — the standard price of differential coding, and much the better
    trade against losing everything.

    The first symbol has no predecessor and carries no data; discard it, as
    ``ignore_edges`` on the analyser already does.

    **This block emits decisions, so do not measure EVM after it.** Differencing
    the quadrant requires slicing first, and what comes out is ideal constellation
    points — an EVM taken here is exactly zero however bad the link is. Modulation
    quality is a soft measurement and belongs upstream, on the recovered symbols;
    the error count is a hard one and belongs here. A link that wants both needs
    an analyser in each place, which is also how a bench does it.
    """

    display_name = "Differential Decoder"
    category = "DSP"

    inputs = {"in": PortType.SYMBOL}
    outputs = {"out": PortType.SYMBOL}

    def run(self, ctx: SimulationContext, inputs: dict[str, Signal]) -> dict[str, Signal]:
        received: SymbolSignal = inputs["in"]
        encoded = np.asarray(received.constellation)
        quarter = encoded.shape[0] // 4
        if quarter < 1:
            # A two-point alphabet passes straight through, and that is the
            # correct operation rather than a concession.
            #
            # There are no quadrants on a line, so QAMMapper refuses to
            # differentially encode BPSK at all — which means nothing was
            # encoded, and undoing nothing is the identity. This block is not
            # quietly declining to do its job; its job here is genuinely to do
            # nothing, and a link built for a higher format still works when
            # someone drops it to BPSK to see what happens.
            #
            # What it cannot do is claim the benefit. BPSK carries the same
            # quarter-turn ambiguity every blind stage upstream leaves, and the
            # differential trick is unavailable to remove it — so a receiver
            # that settles a quarter turn away is still wrong, and that is a
            # property of the format, not of this block.
            return {"out": received}

        indices = nearest_indices(np.asarray(received.symbols), encoded)
        decoded = differential_decode(indices, quarter)

        plain = qam_constellation(received.bits_per_symbol)
        return {
            "out": SymbolSignal(
                symbols=plain[decoded],
                symbol_rate=received.symbol_rate,
                constellation=plain,
            )
        }
