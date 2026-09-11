"""Carrier acquisition: what the coarse stage reaches and the fine stage cannot.

Builds one coherent link with a matched filter in it, detunes the local
oscillator across four decades, and runs it twice — once with
``FrequencyRecovery`` alone and once with ``CoarseFrequencyRecovery`` in front of
it. Nothing is injected: the LO is an ordinary ``CWLaser`` at a different
wavelength and the beat comes out of the receiver's own mixing.

The first table is the argument for having two stages. The M-th power estimator
is exact below ``symbol_rate / (2M)`` and past that does not degrade but
*aliases*, returning a wrong offset with a right-looking confidence. The second
table is why the coarse stage reads the waveform rather than the symbols: the
alias is a rotation the alphabet is symmetric under, so nothing measured on
symbols can see it.

Run: ``python examples/acquisition_link.py``
"""

from __future__ import annotations

from maiman import Graph, SimulationContext
from maiman.components import (
    CoarseFrequencyRecovery,
    CoherentReceiver,
    ConstellationAnalyzer,
    CWLaser,
    FrequencyRecovery,
    IQDriver,
    IQModulator,
    IQSampler,
    PRBSGenerator,
    QAMMapper,
    TimingRecovery,
)
from maiman.units import C_LIGHT

V_PI = 4.0
SYMBOL_RATE = 32e9
BITS_PER_SYMBOL = 4
ROLL_OFF = 0.2

#: Where the M-th power estimate folds: symbol_rate / (2 * M), M = 4 for square
#: QAM. Printed beside the table rather than stated in prose, because it is the
#: line every row below is either side of.
FINE_LIMIT = SYMBOL_RATE / (2.0 * 4.0)

#: Four decades. 0.2 GHz is what a good tunable holds and the fine stage handles
#: alone; 4.1 is just past where it folds; 20 and 100 are the state a receiver is
#: in before it has acquired -- an LO still settling, or one left on a
#: neighbouring channel of a 50 GHz grid.
OFFSETS = [0.2e9, 4.1e9, -4.1e9, 20e9, 100e9]


def link(offset: float, *, acquire: bool) -> tuple[Graph, dict[str, object]]:
    """The same link twice, differing only by whether acquisition is in the path."""
    # ``bit_rate`` is the symbol rate -- ``sample_rate`` is this times
    # ``samples_per_symbol``, not times the bits carried in each one. The name is
    # from the OOK links this context was first written for, where the two
    # coincide. Passing 128e9 here runs the link at 128 GBd, which moves the fold
    # this whole example is about from 4 GHz to 16.
    ctx = SimulationContext(
        bit_rate=SYMBOL_RATE,
        samples_per_symbol=16,
        sequence_length=2048,
        seed=7,
    )
    graph = Graph(ctx)
    bits = float(BITS_PER_SYMBOL)
    prbs = graph.add(PRBSGenerator(order=15.0, bits_per_symbol=bits, label="prbs"))
    mapper = graph.add(QAMMapper(bits_per_symbol=bits, label="map"))
    driver = graph.add(
        IQDriver(v_pi=V_PI, predistort=True, pulse_shaping=True, roll_off=ROLL_OFF, label="drv")
    )
    laser = graph.add(CWLaser(power=2.0, wavelength=1550.0, label="tx"))
    modulator = graph.add(IQModulator(v_pi=V_PI, label="mod"))
    # Tuned below the signal by the offset, so the beat comes out positive.
    detuned = C_LIGHT / (C_LIGHT / 1550e-9 - offset) * 1e9
    lo = graph.add(CWLaser(power=10.0, wavelength=detuned, label="lo"))
    receiver = graph.add(CoherentReceiver(responsivity=0.8, label="rx"))
    timing = graph.add(TimingRecovery(label="tr"))
    # The matched filter is what makes a large offset matter: it is a
    # root-raised-cosine centred on baseband, so a band 20 GHz away is filtered
    # off rather than measured. That is also why acquisition has to precede it.
    sampler = graph.add(IQSampler(matched_filter=True, roll_off=ROLL_OFF, label="smp"))
    frequency = graph.add(FrequencyRecovery(label="fo"))
    analyzer = graph.add(ConstellationAnalyzer(ignore_edges=64.0, label="vsa"))

    graph.connect(prbs["out"], mapper["in"])
    graph.connect(mapper["out"], driver["in"])
    graph.connect(laser, modulator["optical_in"])
    graph.connect(driver["i"], modulator["i"])
    graph.connect(driver["q"], modulator["q"])
    graph.connect(modulator, receiver["in"])
    graph.connect(lo, receiver["lo"])

    watched: dict[str, object] = {"fine": frequency, "vsa": analyzer}
    if acquire:
        acquisition = graph.add(CoarseFrequencyRecovery(label="acq"))
        graph.connect(receiver["i"], acquisition["i"])
        graph.connect(receiver["q"], acquisition["q"])
        graph.connect(acquisition["i"], timing["i"])
        graph.connect(acquisition["q"], timing["q"])
        watched["coarse"] = acquisition
    else:
        graph.connect(receiver["i"], timing["i"])
        graph.connect(receiver["q"], timing["q"])
    graph.connect(timing["i"], sampler["i"])
    graph.connect(timing["q"], sampler["q"])
    graph.connect(mapper["out"], sampler["reference"])
    graph.connect(sampler["out"], frequency["in"])
    graph.connect(frequency["out"], analyzer["in"])
    graph.connect(mapper["out"], analyzer["reference"])
    return graph, watched


def main() -> None:
    print(f"16-QAM at {SYMBOL_RATE / 1e9:g} GBd, matched filter, 2048 symbols")
    print(f"the M-th power estimate is unambiguous below {FINE_LIMIT / 1e9:.1f} GHz\n")

    print("Fine stage alone")
    print(f"{'LO offset':>12} {'it reports':>13} {'confidence':>11} {'EVM':>9} {'errors':>8}")
    for offset in OFFSETS:
        graph, watched = link(offset, acquire=False)
        results = graph.run(keep=list(watched.values()))
        estimate = results.port(watched["fine"], "diagnostics")
        measured = results[watched["vsa"]]
        print(
            f"{offset / 1e9:+11.2f}G {estimate.offset / 1e9:+12.3f}G "
            f"{estimate.confidence:11.1f} {measured.evm * 100:8.2f}% {measured.symbol_errors:8d}"
        )

    print("\nWith acquisition in front")
    print(
        f"{'LO offset':>12} {'band found at':>15} {'conc.':>7} "
        f"{'residual':>11} {'EVM':>9} {'errors':>8}"
    )
    for offset in OFFSETS:
        graph, watched = link(offset, acquire=True)
        results = graph.run(keep=list(watched.values()))
        coarse = results.port(watched["coarse"], "diagnostics")
        residual = results.port(watched["fine"], "diagnostics")
        measured = results[watched["vsa"]]
        print(
            f"{offset / 1e9:+11.2f}G {coarse.offset / 1e9:+14.3f}G {coarse.concentration:7.3f} "
            f"{residual.offset / 1e6:+10.1f}M {measured.evm * 100:8.2f}% "
            f"{measured.symbol_errors:8d}"
        )

    print(
        "\nThe fine stage's wrong rows carry the same confidence as its right ones,"
        "\nwhich is why the coarse stage reads the waveform instead: past its fold the"
        "\nalias is a rotation this alphabet is symmetric under, and no measurement"
        "\nmade on these symbols can distinguish it from the truth."
    )


if __name__ == "__main__":
    main()
