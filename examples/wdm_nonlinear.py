"""What neighbouring channels cost each other in a nonlinear fiber.

Until now bands propagated independently, which made this simulator a good model
of one channel and an optimistic model of a comb. They no longer do: the same
``|A|**2 A`` term that gives a channel its own self-phase modulation also lets
every other channel rotate its phase, and lets triplets of them generate light
at frequencies nobody launched.

Four things are worth reading off the output, and the last two are the ones that
decide how real links are built.

**A neighbour counts twice.** Expanding ``|A|**2 A`` for a sum of carriers, the
term that rotates channel k by its own power appears once and the term that
rotates it by channel j's power appears twice. So two equal channels see three
times the nonlinear phase one channel sees alone — not two — and the table below
measures exactly that.

**Dispersion is the cure, not the disease.** Chromatic dispersion is usually
introduced as an impairment to be compensated. Between channels it is the only
thing keeping them apart: it makes them travel at different speeds, so a
neighbour's bit pattern slides past instead of sitting on top of the channel it
is modulating. Set the dispersion to zero and nothing averages it away.

**Walk-off does not remove the cross-phase modulation; it converts it.** The
*mean* phase shift is fixed by the neighbour's average power and no amount of
sliding changes it. What collapses is the *variation* around that mean — and a
constant phase offset costs nothing, because carrier recovery absorbs it, while
the variation is what closes an eye. That split is the whole mechanism.

**Four-wave mixing lands on the channels, and that is why the grid is uniform.**
Products appear at ``f_i + f_j - f_k``. On an equally spaced grid those
frequencies *are* channel frequencies, so the crosstalk arrives in band where no
filter can reach it. Dispersion suppresses it too, by dephasing the mixing, and
the last table is the 45 dB that buys.
"""

from __future__ import annotations

import math

import numpy as np

from maiman import Graph, SimulationContext
from maiman.component import Component
from maiman.components import (
    EDFA,
    CarrierRecovery,
    CoherentReceiver,
    Combiner,
    ConstellationAnalyzer,
    CWLaser,
    DispersionCompensator,
    Fiber,
    IQDriver,
    IQModulator,
    IQSampler,
    OpticalFilter,
    PowerMeter,
    PRBSGenerator,
    QAMMapper,
)
from maiman.kernels import (
    attenuation_db_per_m_to_alpha,
    dispersion_to_beta2,
    effective_length,
    fwm_efficiency,
    fwm_phase_mismatch,
    walkoff_from_dispersion,
)
from maiman.signals import Band, OpticalSignal
from maiman.units import C_LIGHT, dbm_to_w, w_to_dbm, wavelength_to_frequency

ANCHOR = 1550.0  # nm — channel 0
SPACING = 100e9  # Hz
CHANNELS = 4
SYMBOL_RATE = 32e9
GAMMA = 1.3  # 1/W/km
ATTENUATION = 0.2  # dB/km
LENGTH = 80.0  # km
SPANS = 4
#: Distinct maximal-length orders, so no two channels carry the same pattern.
PRBS_ORDERS = (7, 9, 11, 15)

ALPHA = attenuation_db_per_m_to_alpha(ATTENUATION * 1e-3)
L_EFF = effective_length(ALPHA, LENGTH * 1e3)


def channel_wavelength(index: int) -> float:
    """Wavelength of channel ``index`` on the grid [nm]."""
    return C_LIGHT / (wavelength_to_frequency(ANCHOR * 1e-9) + index * SPACING) * 1e9


def build(
    *,
    channels: int,
    launch_dbm: float,
    dispersion: float,
    select: int = 1,
    coupled: bool = True,
    mixing: bool = True,
) -> tuple[Graph, dict[str, Component]]:
    """A QPSK comb over four amplified spans, received on one channel.

    Every channel carries its own data — unlike the demultiplexer example, where
    the neighbours were unmodulated carriers on purpose. Here the neighbours'
    patterns are the whole point: it is their power fluctuating that modulates
    the channel under test.
    """
    ctx = SimulationContext(
        bit_rate=SYMBOL_RATE, samples_per_symbol=8, sequence_length=1024, seed=23
    )
    graph = Graph(ctx)
    combiner = graph.add(Combiner(channels, label="mux"))

    for index in range(channels):
        # A different PRBS order per channel, so no two carry the same pattern
        # and the interference is not an artefact of them being synchronised.
        prbs = graph.add(
            PRBSGenerator(
                order=float(PRBS_ORDERS[index % len(PRBS_ORDERS)]),
                bits_per_symbol=2.0,
                label=f"prbs{index}",
            )
        )
        mapper = graph.add(QAMMapper(bits_per_symbol=2.0, label=f"qam{index}"))
        driver = graph.add(IQDriver(label=f"drv{index}"))
        laser = graph.add(
            CWLaser(power=launch_dbm, wavelength=channel_wavelength(index), label=f"ch{index}")
        )
        modulator = graph.add(IQModulator(label=f"iq{index}"))
        if index == select:
            reference = mapper
        graph.connect(prbs, mapper["in"])
        graph.connect(mapper, driver["in"])
        graph.connect(laser, modulator["optical_in"])
        graph.connect(driver["i"], modulator["i"])
        graph.connect(driver["q"], modulator["q"])
        graph.connect(modulator, combiner[f"in{index}"])

    node: Component = combiner
    first_fiber: Component | None = None
    for span in range(SPANS):
        fiber = graph.add(
            Fiber(
                length=LENGTH,
                attenuation=ATTENUATION,
                dispersion=dispersion,
                nonlinearity=GAMMA,
                cross_phase_modulation=coupled,
                four_wave_mixing=mixing,
                label=f"f{span}",
            )
        )
        amplifier = graph.add(
            EDFA(gain=ATTENUATION * LENGTH, noise_figure=5.0, label=f"edfa{span}")
        )
        first_fiber = first_fiber or fiber
        graph.connect(node, fiber["in"])
        graph.connect(fiber, amplifier["in"])
        node = amplifier
    assert first_fiber is not None

    demux = graph.add(
        OpticalFilter(
            center_wavelength=channel_wavelength(select), bandwidth=50.0, order=3.0, label="demux"
        )
    )
    meter = graph.add(PowerMeter(label="pm"))
    graph.connect(node, demux["in"])
    graph.connect(node, meter["in"])

    # Coherent reception of the selected channel. Carrier recovery is not
    # decoration here: cross-phase modulation delivers a large constant phase
    # shift and a small varying one, and it is only the varying part that costs
    # anything. Leaving the constant in would swamp the measurement with a
    # rotation every real receiver removes for free.
    lo = graph.add(CWLaser(power=10.0, wavelength=channel_wavelength(select), label="lo"))
    receiver = graph.add(CoherentReceiver(responsivity=0.8, label="rx"))
    # 320 km of standard fiber is 5440 ps/nm, which closes a 32 GBd constellation
    # completely. Removing it is what makes the *nonlinear* penalty visible; the
    # compensator is told the accumulated value because a receiver that had to
    # estimate it blindly would add a second variable to a one-variable table.
    compensator = graph.add(
        DispersionCompensator(
            accumulated_dispersion=dispersion * LENGTH * SPANS,
            wavelength=channel_wavelength(select),
            label="cdc",
        )
    )
    sampler = graph.add(IQSampler(label="smp"))
    recovery = graph.add(CarrierRecovery(label="cpr"))
    analyzer = graph.add(ConstellationAnalyzer(label="vsa"))
    graph.connect(demux, receiver["in"])
    graph.connect(lo, receiver["lo"])
    graph.connect(receiver["i"], compensator["i"])
    graph.connect(receiver["q"], compensator["q"])
    graph.connect(compensator["i"], sampler["i"])
    graph.connect(compensator["q"], sampler["q"])
    graph.connect(reference["out"], sampler["reference"])
    graph.connect(sampler, recovery["in"])
    graph.connect(recovery, analyzer["in"])
    graph.connect(reference["out"], analyzer["reference"])
    return graph, {"pm": meter, "fiber": first_fiber, "vsa": analyzer}


def main() -> None:
    print(
        f"{CHANNELS} QPSK channels, {SPACING / 1e9:.0f} GHz apart, "
        f"{SPANS} x {LENGTH:.0f} km at gamma = {GAMMA} /W/km\n"
    )

    # ------------------------------------------------------------------
    print("  1. A neighbour counts twice: gamma*(P + 2P + ...)*L_eff")
    print(f"     {'channels':>10}  {'mean phase':>12}  {'vs one channel':>16}  {'hand':>8}")
    print("     " + "-" * 54)
    from maiman.kernels import propagate_coupled_ssfm

    power = 1e-3
    solo = None
    for count in (1, 2, 3, 4):
        fields = [np.full(512, math.sqrt(power), dtype=np.complex128) for _ in range(count)]
        out, _ = propagate_coupled_ssfm(
            fields,
            8 * SYMBOL_RATE,
            beta2=[0.0] * count,
            walkoff=[0.0] * count,
            gamma=GAMMA * 1e-3,
            alpha=ALPHA,
            distance=LENGTH * 1e3,
            max_nonlinear_phase=2e-4,
        )
        phase = float(np.mean(np.angle(out[0] / fields[0])))
        solo = phase if solo is None else solo
        print(
            f"     {count:10d}  {phase:10.6f} r  {phase / solo:15.3f}x  "
            f"{float(2 * count - 1):7.0f}x"
        )
    print("     one channel counts once, every other counts twice: 2n - 1.")

    # ------------------------------------------------------------------
    print("\n  2. Walk-off: how fast a neighbour slides past, per kilometre")
    print(f"     {'spacing':>10}  {'D':>12}  {'walk-off':>14}  {'over 320 km':>13}")
    print("     " + "-" * 56)
    for spacing_ghz in (50.0, 100.0, 200.0):
        for dispersion in (0.0, 17.0):
            beta2 = dispersion_to_beta2(dispersion * 1e-6, ANCHOR * 1e-9)
            rate = abs(walkoff_from_dispersion(beta2, spacing_ghz * 1e9))
            symbols = rate * SPANS * LENGTH * 1e3 * SYMBOL_RATE
            print(
                f"     {spacing_ghz:7.0f}GHz  {dispersion:6.0f}ps/nm/km  "
                f"{rate * 1e12 * 1e3:9.2f} ps/km  {symbols:9.0f} sym"
            )
    print("     at 17 ps/nm/km a 100 GHz neighbour has slid 140 symbols by the end.")
    print("     at zero it has slid none, and sits on top of the channel throughout.")

    # ------------------------------------------------------------------
    print("\n  3. Four-wave mixing efficiency, and what dispersion does to it")
    print(f"     {'spacing':>10}  {'D = 0':>12}  {'D = 2':>12}  {'D = 17':>12}")
    print("     " + "-" * 52)
    for spacing_ghz in (25.0, 50.0, 100.0, 200.0):
        row = []
        for dispersion in (0.0, 2.0, 17.0):
            beta2 = dispersion_to_beta2(dispersion * 1e-6, ANCHOR * 1e-9)
            mismatch = fwm_phase_mismatch(beta2, 0.0, 0.0, spacing_ghz * 1e9)
            eta = fwm_efficiency(mismatch, ALPHA, LENGTH * 1e3)
            row.append(f"{10.0 * math.log10(max(eta, 1e-30)):9.1f} dB")
        print(f"     {spacing_ghz:7.0f}GHz  " + "  ".join(row))
    print("     zero-dispersion fiber is perfectly phase matched at every spacing —")
    print("     which is why dispersion-shifted fiber was abandoned for WDM.")

    # ------------------------------------------------------------------
    print("\n  4. What the coupling costs channel 1, measured on its constellation")
    print("     EVM after carrier recovery, so the constant phase shift is already gone")
    print(f"     {'launch':>8}  {'D':>13}  {'independent':>12}  {'coupled':>9}  {'penalty':>9}")
    print("     " + "-" * 60)
    for launch_dbm in (-3.0, 0.0, 3.0):
        for dispersion in (0.0, 17.0):
            evms = []
            for coupled in (False, True):
                graph, ports = build(
                    channels=CHANNELS,
                    launch_dbm=launch_dbm,
                    dispersion=dispersion,
                    coupled=coupled,
                    mixing=coupled,
                )
                evms.append(graph.run()[ports["vsa"]].evm * 100.0)
            print(
                f"     {launch_dbm:5.0f}dBm  {dispersion:6.0f}ps/nm/km  "
                f"{evms[0]:10.2f} %  {evms[1]:7.2f} %  {evms[1] - evms[0]:+8.2f} %"
            )
    print("     the independent column is what this simulator reported before today.")
    print("     at zero dispersion the neighbours never slide past, so the penalty is")
    print("     the whole impairment rather than a correction to it.")

    # ------------------------------------------------------------------
    print("\n  5. Where the mixing products land, on a uniform grid")
    graph, ports = build(channels=CHANNELS, launch_dbm=0.0, dispersion=0.0)
    reading = graph.run()[ports["pm"]]
    anchor = wavelength_to_frequency(ANCHOR * 1e-9)
    comb_top = (CHANNELS - 1) * 100
    print(f"     {'offset':>10}  {'power':>11}   what it is")
    print("     " + "-" * 54)
    for band in sorted(reading.bands, key=lambda b: b.f0):
        offset = round((band.f0 - anchor) / 1e9)
        kind = "channel" if 0 <= offset <= comb_top and offset % 100 == 0 else "product"
        print(f"     {offset:+7d}GHz  {w_to_dbm(band.power_w):8.2f} dBm   {kind}")
    print("     each of the four channels also carries a product folded in at DC,")
    print("     where no filter downstream can reach it. That is what a uniform grid")
    print("     costs, and why unequal channel spacing was once a serious proposal.")

    # ------------------------------------------------------------------
    print("\n  6. Polarization: an orthogonal neighbour is not an absent one")
    print("     Two unmodulated channels, 100 km, no loss, no dispersion.")
    print("     The phase channel 0 picks up beyond its own self-phase modulation:\n")
    ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=16, sequence_length=64, seed=3)
    power = dbm_to_w(0.0)
    unit = GAMMA * 1e-3 * power * 100e3  # gamma * P * L, radians

    def carrier(px: float, py: float, index: int) -> Band:
        samples = ctx.num_samples
        return Band(
            Ex=np.full(samples, np.sqrt(px), dtype=np.complex128),
            Ey=np.full(samples, np.sqrt(py), dtype=np.complex128),
            f0=wavelength_to_frequency(channel_wavelength(index) * 1e-9),
            fs=ctx.sample_rate,
        )

    def turned(bands: tuple[Band, ...], *, coupled: bool) -> float:
        span = Fiber(
            length=100.0,
            attenuation=0.0,
            dispersion=0.0,
            nonlinearity=GAMMA,
            four_wave_mixing=False,
            cross_polarization=coupled,
            label="fib",
        )
        out = span.run(ctx, {"in": OpticalSignal(bands=bands, noise=())})["out"]
        return float(np.angle(out.bands[0].Ex[0])) / unit

    print(f"     {'neighbour':>22}  {'axes uncoupled':>15}  {'axes coupled':>13}")
    print("     " + "-" * 54)
    for label, neighbour in (
        ("co-polarized", carrier(power, 0.0, 1)),
        ("orthogonal", carrier(0.0, power, 1)),
    ):
        alone_off = turned((carrier(power, 0.0, 0),), coupled=False)
        alone_on = turned((carrier(power, 0.0, 0),), coupled=True)
        off = turned((carrier(power, 0.0, 0), neighbour), coupled=False) - alone_off
        on = turned((carrier(power, 0.0, 0), neighbour), coupled=True) - alone_on
        print(f"     {label:>22}  {off:12.3f} gPL  {on:10.3f} gPL")
    print("     Two and two thirds: orthogonal cross-phase modulation is exactly a")
    print("     third of co-polarized. Uncoupled it is zero, which is the model")
    print("     saying a channel across the polarization is not there at all.")


if __name__ == "__main__":
    main()
