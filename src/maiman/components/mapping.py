"""Bits to symbols and back — the digital edge of a coherent transceiver."""

from __future__ import annotations

from dataclasses import dataclass

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
        doc=(
            "1 BPSK, then 2^n-QAM. Odd orders are rectangular rather than square: "
            "3 is 8-QAM (4x2), 5 is 32-QAM (8x4), 7 is 128-QAM (16x8)"
        ),
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
        bits = int(self.bits_per_symbol)
        if self.differential and bits < 2:
            raise ValueError(
                f"{self.label}: differential quadrant encoding needs at least 2 bits "
                f"per symbol, got {bits}. BPSK has no quadrants "
                f"to difference — turn differential off, and take any Differential "
                f"Decoder out of the chain with it."
            )
        if self.differential and bits % 2:
            # An odd order is a rectangle, and a rectangle turned a quarter maps
            # onto a different alphabet. Its blind ambiguity is a half turn,
            # which quadrant differencing does not address — encoding it anyway
            # would look like it worked and lose data on every other run.
            raise ValueError(
                f"{self.label}: {1 << bits}-QAM is rectangular, so it has no "
                f"quadrant symmetry to difference — a quarter turn takes it to a "
                f"different alphabet. Its blind phase ambiguity is a *half* turn. "
                f"Turn differential off for odd bits per symbol, and take any "
                f"Differential Decoder out of the chain with it."
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


@dataclass(frozen=True)
class PilotEstimate:
    """The rotation the pilots found, and how much it looked like an ambiguity."""

    rotation: float
    """Constant phase removed [rad]."""

    quarter_turns: int
    """Which multiple of pi/2 that was nearest to: 0, 1, 2 or 3."""

    pilots: int
    """Known symbols the estimate was formed from."""

    residual: float
    """How far the rotation sat from an exact quarter turn [rad].

    The number that says whether the stage upstream did its job. Everything
    before this is supposed to leave *only* a quarter-turn ambiguity: the phase
    search tracks the walk and the frequency stage removes the ramp, so what is
    left should be a multiple of pi/2 and nothing else. A residual of a few
    milliradians means that held. A large one means it did not, and the constant
    removed here was covering for something.
    """

    def __repr__(self) -> str:
        return (
            f"PilotEstimate({self.quarter_turns} quarter turns, "
            f"residual {self.residual * 1e3:.1f} mrad, {self.pilots} pilots)"
        )


class PilotInserter(Component):
    """Overwrite every ``spacing``-th symbol with one the receiver already knows.

    The alternative to :class:`DifferentialDecoder`, and the one a link carrying
    soft-decision FEC has to use. Differencing the quadrant resolves the
    ambiguity by *slicing*, and a slice destroys exactly the information a soft
    decoder runs on — see :class:`PilotPhaseRecovery` for the measurement. A
    known symbol resolves the same ambiguity without deciding anything about the
    data.

    **Pilots replace payload, they do not add to it.** The run window is a fixed
    number of symbols on the line, so a pilot costs a data symbol: the overhead
    is ``1/spacing``, 1.6 % at the default. That is the same accounting
    :class:`~maiman.components.FECEncoder` uses and for the same reason — coding
    and pilots both buy reliability with rate, never with bandwidth nobody has.

    **Drawn from the alphabet's own outermost points**, for two reasons. They are
    valid constellation points, so a pilot is not itself a symbol error the way
    an off-grid one would be — using unit-power QPSK pilots inside a 16-QAM link
    puts every pilot exactly between four legal points, and the error count
    measures that rather than the channel. And they carry the most energy, which
    is what a phase estimate is made of.

    The sequence is drawn from the run's seeded generator, so it is reproducible
    and it is *not* constant — a repeated symbol would put a line in the spectrum
    at the pilot rate.
    """

    display_name = "Pilot Inserter"
    category = "Modulation"

    spacing = Param(
        64.0,
        unit="",
        min=2.0,
        doc="Symbols between pilots; the overhead is one over this",
    )

    inputs = {"in": PortType.SYMBOL}
    outputs = {"out": PortType.SYMBOL}

    def run(self, ctx: SimulationContext, inputs: dict[str, Signal]) -> dict[str, Signal]:
        received: SymbolSignal = inputs["in"]
        constellation = np.asarray(received.constellation)
        symbols = np.asarray(received.symbols).astype(np.complex128).copy()

        spacing = int(self.spacing)
        positions = np.arange(0, symbols.shape[0], spacing)
        if positions.size < 2:
            raise ValueError(
                f"{self.label}: a spacing of {spacing} leaves {positions.size} pilots in a "
                f"window of {symbols.shape[0]} symbols, which is not enough to estimate "
                f"anything from"
            )

        magnitude = np.abs(constellation)
        corners = constellation[np.isclose(magnitude, magnitude.max())]
        rng = ctx.rng("PilotInserter", self.label, spacing)
        symbols[positions] = corners[rng.integers(0, corners.size, positions.size)]

        return {
            "out": SymbolSignal(
                symbols=symbols,
                symbol_rate=received.symbol_rate,
                constellation=constellation,
            )
        }


class PilotPhaseRecovery(Component):
    """Resolve the quarter-turn ambiguity from the pilots, without slicing.

    Blind carrier recovery gets the phase right modulo ``2*pi/n`` and cannot do
    better — rotate a symmetric alphabet by one of its own symmetries and nothing
    about the received samples changes. Something has to break the tie, and there
    are exactly two ways: difference the quadrant out, or know some of the
    symbols.

    **Why this one, when the link carries soft FEC.**
    :class:`DifferentialDecoder` has to slice in order to difference, so what
    leaves it is ideal constellation points — measured on the shipped graph, the
    distance from its output to the nearest point is *exactly zero* and a
    demapper reading it returns log-likelihood ratios of order 1e29. Soft
    information does not survive a block that emits decisions. This block moves
    every symbol by one constant angle and decides nothing, so it does.

    The estimate is ``angle(sum(r * conj(p)))`` over the pilot positions — the
    maximum-likelihood constant phase for additive Gaussian noise, and one line.
    It reads the reference **only** at those positions; a test corrupts every
    other symbol of it and requires the answer not to move, because a block handed
    the whole transmitted sequence could quietly do much better than a receiver
    can and nothing would look wrong.

    What it cannot do is track: this removes one constant for the whole window.
    Phase *noise* is :class:`~maiman.components.CarrierRecovery`'s job and has to
    have been done already, which is what ``residual`` on the diagnostics reports.
    """

    display_name = "Pilot Phase Recovery"
    category = "DSP"

    spacing = Param(64.0, unit="", min=2.0, doc="Must match the inserter; pilots are where it says")

    inputs = {"in": PortType.SYMBOL, "reference": PortType.SYMBOL}
    outputs = {"out": PortType.SYMBOL, "diagnostics": PortType.METRIC}

    def run(self, ctx: SimulationContext, inputs: dict[str, Signal]) -> dict[str, Signal]:
        received: SymbolSignal = inputs["in"]
        reference: SymbolSignal = inputs["reference"]
        if received.num_symbols != reference.num_symbols:
            raise ValueError(
                f"{self.label}: received {received.num_symbols} symbols against a "
                f"reference of {reference.num_symbols}"
            )

        symbols = np.asarray(received.symbols).astype(np.complex128)
        known = np.asarray(reference.symbols).astype(np.complex128)
        spacing = int(self.spacing)
        positions = np.arange(0, symbols.shape[0], spacing)

        correlation = complex(np.sum(symbols[positions] * np.conj(known[positions])))
        rotation = float(np.angle(correlation)) if correlation != 0 else 0.0

        turns = round(rotation / (np.pi / 2.0))
        quarter = turns % 4
        residual = abs(rotation - turns * (np.pi / 2.0))

        return {
            "out": SymbolSignal(
                symbols=symbols * np.exp(-1j * rotation),
                symbol_rate=received.symbol_rate,
                constellation=np.asarray(received.constellation),
            ),
            "diagnostics": PilotEstimate(
                rotation=rotation,
                quarter_turns=quarter,
                pilots=int(positions.size),
                residual=residual,
            ),
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
