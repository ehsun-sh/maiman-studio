"""Export real engine output for the GUI mockup.

Everything the interface shows should come from a real run. This writes the
component manifests, a real constellation histogram, a real eye, and a real
per-format sweep to JSON so the mockup can be built against true data — which
also proves the data the session server will eventually serve is the data the
engine already produces.

The link is the coherent one, because it is the one that exercises the whole
port-type system: binary into symbols, symbols into two electrical drives, an
optical field, two photocurrents back, and symbols out again.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from maiman import Graph, SimulationContext, manifests, sweep
from maiman.component import Component
from maiman.components import (
    EDFA,
    BERAnalyzer,
    CarrierRecovery,
    CoherentReceiver,
    Combiner,
    ConstellationAnalyzer,
    ConstellationDiagram,
    CWLaser,
    DifferentialDecoder,
    DispersionCompensator,
    ElectricalFilter,
    EyeDiagram,
    Fiber,
    FrequencyRecovery,
    IQDriver,
    IQModulator,
    IQSampler,
    MachZehnderModulator,
    NRZDriver,
    OpticalSpectrumAnalyzer,
    OSNRMeter,
    PINPhotodiode,
    PowerMeter,
    PRBSGenerator,
    QAMMapper,
    TimingRecovery,
)
from maiman.project import graph_to_dict, save
from maiman.units import C_LIGHT, wavelength_to_frequency

V_PI = 4.0
SYMBOL_RATE = 32e9
BITS_PER_SYMBOL = 4
SPAN_KM = 80.0
DISPERSION = 17.0  # ps/nm/km, standard single-mode fiber at 1550 nm
# BPSK is absent on purpose: differential *quadrant* encoding needs a quadrant to
# difference, and a two-point constellation has none. A binary format resolves its
# ambiguity by other means, and the mapper says so rather than silently coping.
FORMATS = {2: "QPSK", 4: "16-QAM", 6: "64-QAM", 8: "256-QAM"}


#: Where the direct-detection project is written, so the studio has a link with
#: an eye in it to open.
OOK_PROJECT = Path(__file__).parent / "ook_eye.maiman"

#: Positions for it, so it opens laid out rather than stacked on a grid. The
#: format carries them and the editor reads them back.
OOK_LAYOUT = {
    "prbs": {"x": 30.0, "y": 220.0},
    "drv": {"x": 168.0, "y": 220.0},
    "tx": {"x": 168.0, "y": 60.0},
    "mzm": {"x": 306.0, "y": 220.0},
    "fib": {"x": 444.0, "y": 220.0},
    "pin": {"x": 582.0, "y": 220.0},
    "lpf": {"x": 720.0, "y": 220.0},
    "eye": {"x": 858.0, "y": 130.0},
    "ber": {"x": 858.0, "y": 310.0},
}


def ook_link() -> Graph:
    """On-off keying into a photodiode: the link an eye diagram is *for*.

    Direct detection squares the field, so what reaches the eye is intensity
    against time and one threshold decides the bit. That is the whole premise of
    the instrument, and it is why the coherent link beside this one has no eye
    to show: there, the rails only mean something relative to a carrier phase
    nothing has recovered yet.

    Deliberately short at 20 km. The point of the shipped project is that the
    eye is *open* when it opens, so someone can see what one looks like before
    they start closing it — and stretching the span is the first thing to try.
    """
    ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=32, sequence_length=512, seed=4)
    graph = Graph(ctx)
    prbs = graph.add(PRBSGenerator(order=7.0, label="prbs"))
    driver = graph.add(NRZDriver(v_low=4.0, v_high=0.0, label="drv"))
    laser = graph.add(CWLaser(power=0.0, wavelength=1550.0, label="tx"))
    modulator = graph.add(MachZehnderModulator(v_pi=4.0, extinction_ratio=30.0, label="mzm"))
    fiber = graph.add(Fiber(length=20.0, attenuation=0.2, dispersion=17.0, label="fib"))
    diode = graph.add(PINPhotodiode(responsivity=0.8, label="pin"))
    filter_ = graph.add(ElectricalFilter(bandwidth=7.0, label="lpf"))
    eye = graph.add(EyeDiagram(span_symbols=2.0, time_bins=64.0, amplitude_bins=72.0, label="eye"))
    analyzer = graph.add(BERAnalyzer(label="ber"))

    graph.chain(prbs, driver)
    graph.connect(laser, modulator["optical_in"])
    graph.connect(driver, modulator["electrical_in"])
    graph.chain(modulator, fiber, diode, filter_)
    graph.connect(filter_, eye["in"])
    graph.connect(filter_, analyzer["in"])
    graph.connect(prbs["out"], analyzer["reference"])
    return graph


def ook_eye() -> Any:
    """The eye the studio ships, and the project it came from, written to disk."""
    graph = ook_link()
    eye_block = of_type(graph, EyeDiagram)
    histogram = graph.run()[eye_block]
    save(graph, OOK_PROJECT, ui=OOK_LAYOUT)
    return histogram


#: Where the WDM project is written, so the studio has a link with a spectrum in
#: it to open. The eye has one of these and the spectrum needs its own: neither
#: instrument has anything to show on the coherent link the page draws by default.
WDM_PROJECT = Path(__file__).parent / "wdm_osa.maiman"

#: Channels on the ITU grid, and the amplified line that gives the trace a floor.
WDM_CHANNELS = 4
WDM_SPACING = 100e9  # Hz — G.694.1, the spacing every one of these plots is read on
WDM_ANCHOR = 1550.0  # nm — channel 0
WDM_SPANS = 2

WDM_LAYOUT = {
    "prbs": {"x": 30.0, "y": 470.0},
    "drv": {"x": 168.0, "y": 470.0},
    **{f"ch{i}": {"x": 30.0, "y": 40.0 + i * 100.0} for i in range(WDM_CHANNELS)},
    **{f"mzm{i}": {"x": 306.0, "y": 40.0 + i * 100.0} for i in range(WDM_CHANNELS)},
    "mux": {"x": 444.0, "y": 190.0},
    "f0": {"x": 582.0, "y": 190.0},
    "edfa0": {"x": 720.0, "y": 190.0},
    "f1": {"x": 858.0, "y": 190.0},
    "edfa1": {"x": 996.0, "y": 190.0},
    "osa": {"x": 1134.0, "y": 60.0},
    "pm": {"x": 1134.0, "y": 200.0},
    "osnr": {"x": 1134.0, "y": 320.0},
}


def wdm_wavelength(index: int) -> float:
    """Wavelength of channel ``index`` on the 100 GHz grid [nm].

    Derived from the anchor's *frequency*, because the grid is defined in
    frequency and channels spaced evenly in wavelength would not land on it.
    """
    return C_LIGHT / (wavelength_to_frequency(WDM_ANCHOR * 1e-9) + index * WDM_SPACING) * 1e9


def wdm_link() -> Graph:
    """Four channels, amplified twice, into an optical spectrum analyser.

    The link a spectrum is *for*. Everything the OSA pane is meant to teach is
    on this one trace: four modulated channels sitting on their grid, each one
    broadened by its own data rather than by an assumption, and underneath them
    the ASE floor the two amplifiers put there. Widen the analyser's resolution
    and the floor climbs decibel for decibel while the channels do not move,
    which is the whole reason an OSNR figure is meaningless without the
    bandwidth it was quoted in.

    One pattern drives all four modulators. Independent data per channel would
    cost four more blocks on a schematic that is already the largest the studio
    ships, and would not change the trace: what sets a channel's width here is
    the bit rate, not which bits.
    """
    ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=16, sequence_length=1024, seed=11)
    graph = Graph(ctx)
    prbs = graph.add(PRBSGenerator(order=11.0, label="prbs"))
    driver = graph.add(NRZDriver(v_low=4.0, v_high=0.0, label="drv"))
    graph.connect(prbs["out"], driver["in"])

    combiner = graph.add(Combiner(WDM_CHANNELS, label="mux"))
    for index in range(WDM_CHANNELS):
        laser = graph.add(CWLaser(power=0.0, wavelength=wdm_wavelength(index), label=f"ch{index}"))
        modulator = graph.add(MachZehnderModulator(v_pi=4.0, label=f"mzm{index}"))
        graph.connect(laser, modulator["optical_in"])
        graph.connect(driver, modulator["electrical_in"])
        graph.connect(modulator, combiner[f"in{index}"])

    # Amplified back to transparency each span. Without real loss to make up
    # there is no ASE worth speaking of and the trace has no floor to show.
    node: Component = combiner
    for span in range(WDM_SPANS):
        fiber = graph.add(Fiber(length=80.0, attenuation=0.2, dispersion=17.0, label=f"f{span}"))
        amplifier = graph.add(EDFA(gain=16.0, noise_figure=6.0, label=f"edfa{span}"))
        graph.connect(node, fiber["in"])
        graph.connect(fiber, amplifier["in"])
        node = amplifier

    # Centred between channels 1 and 2 with a span wide enough to show all four
    # and some clean floor either side of the comb.
    osa = graph.add(
        OpticalSpectrumAnalyzer(
            center_wavelength=(wdm_wavelength(1) + wdm_wavelength(2)) / 2.0,
            span=800.0,
            points=1024,
            resolution_bandwidth=12.5,
            label="osa",
        )
    )
    meter = graph.add(PowerMeter(label="pm"))
    osnr = graph.add(OSNRMeter(label="osnr"))
    graph.connect(node, osa["in"])
    graph.connect(node, meter["in"])
    graph.connect(node, osnr["in"])
    return graph


def wdm_spectrum() -> dict[str, Any]:
    """The trace the studio ships, and the project it came from, written to disk.

    Reduced to what the plot actually draws — wavelength against the level an
    instrument displays — rather than shipped raw. A live run sends four arrays
    because a caller might want any of them; the page only ever reads two, and
    baking the other two into every copy of the file buys nothing.
    """
    graph = wdm_link()
    block = of_type(graph, OpticalSpectrumAnalyzer)
    results = graph.run()
    spectrum = results[block]
    save(graph, WDM_PROJECT, ui=WDM_LAYOUT)

    peak_frequency, peak_power = spectrum.peak()
    return {
        "wavelengths_nm": np.asarray(spectrum.wavelengths_nm).round(4).tolist(),
        # Floored at -90 dBm so an empty display bin plots on the axis instead
        # of running off it; the engine's own floor is far lower and would set
        # the y-range from a bin holding nothing.
        "dbm": np.maximum(spectrum.power_dbm(), -90.0).round(3).tolist(),
        "resolution_bandwidth_ghz": spectrum.resolution_bandwidth / 1e9,
        "total_dbm": round(10.0 * float(np.log10(spectrum.total_power() * 1e3)), 3),
        "peak_nm": round(C_LIGHT / peak_frequency * 1e9, 4),
        "peak_dbm": round(10.0 * float(np.log10(max(peak_power, 1e-18) * 1e3)), 3),
        "osnr_db": round(float(results[of_type(graph, OSNRMeter)]), 3),
    }


#: How far the local oscillator sits below the transmitter [Hz], and the
#: wavelength that puts it there. Quoted as a frequency because that is what a
#: laser is specified in and what the receiver has to remove; the parameter is a
#: wavelength because that is what the block takes.
LO_OFFSET_HZ = 200e6
LO_WAVELENGTH_NM = C_LIGHT / (C_LIGHT / 1550e-9 - LO_OFFSET_HZ) * 1e9


def build(sequence_length: int = 4096) -> Graph:
    ctx = SimulationContext(
        bit_rate=SYMBOL_RATE,
        samples_per_symbol=16,
        sequence_length=sequence_length,
        seed=2026,
        precision="double",
    )
    graph = Graph(ctx)
    prbs = graph.add(
        PRBSGenerator(order=23.0, bits_per_symbol=float(BITS_PER_SYMBOL), label="prbs")
    )
    mapper = graph.add(
        QAMMapper(bits_per_symbol=float(BITS_PER_SYMBOL), differential=True, label="map")
    )
    reference = graph.add(
        QAMMapper(bits_per_symbol=float(BITS_PER_SYMBOL), differential=False, label="ref")
    )
    # Backed off because a root-raised-cosine waveform overshoots between symbols
    # and a full-swing drive would run the pre-distortion past arcsin(1).
    driver = graph.add(
        IQDriver(
            v_pi=V_PI,
            predistort=True,
            pulse_shaping=True,
            roll_off=0.2,
            drive_ratio=0.4,
            label="drv",
        )
    )
    # 100 kHz, which is an ordinary coherent-transmitter laser rather than a
    # specially quiet one. An earlier version of this export had to run at 10 kHz
    # because there was no carrier recovery in the chain and the accumulated
    # phase walk put a hard floor near 18 dB SNR however much power was
    # launched. The CarrierRecovery stage below is what makes the realistic
    # number usable again.
    laser = graph.add(CWLaser(power=2.0, wavelength=1550.0, linewidth=100.0, label="tx"))
    # A real span, not a patch cord. 80 km of standard fiber costs 16 dB and
    # smears each symbol over thirteen of its neighbours; the compensator below
    # is what makes the second of those survivable and nothing makes it optional.
    fiber = graph.add(
        Fiber(
            length=SPAN_KM,
            attenuation=0.2,
            dispersion=DISPERSION,
            nonlinearity=0.0,
            label="fib",
        )
    )
    modulator = graph.add(IQModulator(v_pi=V_PI, label="mod"))
    meter = graph.add(PowerMeter(label="pm"))
    # Detuned by 200 MHz, which is not an injected impairment but the removal of
    # an idealisation: two free-running lasers are never on the same frequency to
    # fifteen decimal places, and a good tunable holds about this much. The beat
    # comes out of the receiver's own mixing, and 200 MHz is six thousandths of
    # one per cent of the symbol rate and destroys the link on its own. The
    # FrequencyRecovery stage below is what makes the realistic number usable, in
    # exactly the way CarrierRecovery makes the realistic linewidth usable.
    lo = graph.add(CWLaser(power=10.0, wavelength=LO_WAVELENGTH_NM, linewidth=100.0, label="lo"))
    receiver = graph.add(CoherentReceiver(responsivity=0.8, label="rx"))
    compensator = graph.add(
        DispersionCompensator(
            accumulated_dispersion=DISPERSION * SPAN_KM, wavelength=1550.0, label="cdc"
        )
    )
    # The receiver's two blind front-end corrections, in the order every
    # deployed receiver applies them: the sampling instant at the sample rate,
    # then the carrier frequency on the symbols, then the carrier phase. The
    # order is not a preference — a phase search covers a quarter turn and
    # averages over a window, so a frequency ramp steep enough to cross that
    # quarter turn inside the window makes it slip rather than track.
    timing = graph.add(TimingRecovery(label="tr"))
    sampler = graph.add(IQSampler(matched_filter=True, roll_off=0.2, label="smp"))
    frequency = graph.add(FrequencyRecovery(label="fo"))
    recovery = graph.add(CarrierRecovery(window=64.0, test_phases=32.0, label="cr"))
    decoder = graph.add(DifferentialDecoder(label="dec"))
    # Two analysers, because the two numbers come from different places. EVM is a
    # soft measurement and has to be taken on the recovered symbols; the decoder
    # emits decisions, so an EVM after it would read zero however bad the link is.
    # The error count is the opposite: it only exists once the data is decoded.
    analyzer = graph.add(ConstellationAnalyzer(ignore_edges=64.0, label="vsa"))
    errors = graph.add(ConstellationAnalyzer(ignore_edges=64.0, label="ber"))
    diagram = graph.add(ConstellationDiagram(bins=96.0, extent=1.5, label="cd"))

    graph.connect(prbs["out"], mapper["in"])
    graph.connect(prbs["out"], reference["in"])
    graph.connect(mapper["out"], driver["in"])
    graph.connect(laser, modulator["optical_in"])
    graph.connect(driver["i"], modulator["i"])
    graph.connect(driver["q"], modulator["q"])
    graph.connect(modulator, fiber["in"])
    # The meter sits after the span, so "received" means received.
    graph.connect(fiber, meter["in"])
    graph.connect(fiber, receiver["in"])
    graph.connect(lo, receiver["lo"])
    graph.connect(receiver["i"], compensator["i"])
    graph.connect(receiver["q"], compensator["q"])
    graph.connect(compensator["i"], timing["i"])
    graph.connect(compensator["q"], timing["q"])
    graph.connect(timing["i"], sampler["i"])
    graph.connect(timing["q"], sampler["q"])
    graph.connect(mapper["out"], sampler["reference"])
    graph.connect(sampler["out"], frequency["in"])
    graph.connect(frequency["out"], recovery["in"])
    graph.connect(recovery["out"], decoder["in"])
    graph.connect(recovery["out"], analyzer["in"])
    graph.connect(mapper["out"], analyzer["reference"])
    graph.connect(decoder["out"], errors["in"])
    graph.connect(reference["out"], errors["reference"])
    graph.connect(recovery["out"], diagram["in"])
    return graph


def of_type(graph: Graph, kind: type) -> Any:
    return next(c for c in graph.components if isinstance(c, kind))


def main() -> None:
    graph = build()
    analyzer = next(c for c in graph.components if c.label == "vsa")
    errors = next(c for c in graph.components if c.label == "ber")
    diagram = of_type(graph, ConstellationDiagram)
    meter = of_type(graph, PowerMeter)
    receiver = of_type(graph, CoherentReceiver)
    laser = next(c for c in graph.components if c.label == "tx")

    results = graph.run(keep=[receiver])
    measurement = results[analyzer]
    counted = results[errors]
    histogram = results[diagram]

    # No eye from this link, and none is possible.
    #
    # An eye diagram is a direct-detection instrument. A coherent receiver's I
    # and Q rails mean nothing until a carrier phase has been recovered — the
    # constellation arrives rotated by whatever the two free-running lasers
    # happen to differ by, measured here at 35 to 45 degrees and drifting — so
    # folding either rail gives a smear rather than four levels. Recovery, when
    # it comes, outputs one sample per symbol, and there is no waveform left to
    # fold.
    #
    # So the eye the studio ships comes from a direct-detection link instead,
    # written beside this one as a project anyone can open.
    eye = ook_eye()

    # And no spectrum either, for a plainer reason: this link has one channel and
    # nothing to look at it with. The trace the studio ships comes from the WDM
    # project written beside it, which is also the one to open to see the OSA's
    # resolution setting move the noise floor and leave the channels alone.
    spectrum = wdm_spectrum()

    # Required received power per format, from the same graph re-run.
    sensitivity: list[dict[str, Any]] = []
    prbs = of_type(graph, PRBSGenerator)
    # Both mappers, not just the first: the reference arm has to be told the
    # format too, or it would encode against a different alphabet than the one
    # being measured.
    mappers = [c for c in graph.components if isinstance(c, QAMMapper)]
    for bits_per_symbol, name in FORMATS.items():
        points = [float(p) for p in range(-28, 22, 3)]
        overrides: dict[Any, list[float]] = {
            (laser, "power"): points,
            (prbs, "bits_per_symbol"): [float(bits_per_symbol)],
        }
        for mapper in mappers:
            overrides[(mapper, "bits_per_symbol")] = [float(bits_per_symbol)]
        curve = sweep(graph, overrides)
        sensitivity.append(
            {
                "name": name,
                "bits_per_symbol": bits_per_symbol,
                "gbps": SYMBOL_RATE * bits_per_symbol / 1e9,
                "points": [
                    {
                        "received_dbm": point.runs[0][meter].power_dbm,
                        "snr_db": point.runs[0][analyzer].snr_db,
                        "ber": point.runs[0][analyzer].ber_estimated,
                    }
                    for point in curve
                ],
            }
        )

    payload: dict[str, Any] = {
        "manifests": manifests(),
        "measurement": {
            "evm": measurement.evm,
            "snr_db": measurement.snr_db,
            "mer_db": measurement.mer_db,
            "ber": measurement.ber_estimated,
            "ber_counted": counted.ber_counted,
            "symbol_errors": counted.symbol_errors,
            "symbols": measurement.symbols_evaluated,
            "bit_errors": counted.bit_errors,
            "bits": measurement.bits_evaluated,
            "frequency_offset_mhz": measurement.frequency_offset / 1e6,
            "bits_per_symbol": measurement.bits_per_symbol,
            "received_dbm": results[meter].power_dbm,
        },
        "constellation": {
            "counts": np.asarray(histogram.counts).astype(int).tolist(),
            "inphase_edges": np.asarray(histogram.inphase_edges).round(5).tolist(),
            "quadrature_edges": np.asarray(histogram.quadrature_edges).round(5).tolist(),
            "reference": [[float(p.real), float(p.imag)] for p in np.asarray(histogram.reference)],
        },
        "eye": {
            "counts": np.asarray(eye.counts).astype(int).tolist(),
            "time_ps": (np.asarray(eye.time_edges) * 1e12).round(3).tolist(),
            "amplitude_ua": (np.asarray(eye.amplitude_edges) * 1e6).round(4).tolist(),
            "unit": eye.unit,
        },
        "spectrum": spectrum,
        "sensitivity": sensitivity,
        # The graph itself, in the same `.maiman` document format the session
        # server accepts. The interface draws its schematic from this and posts
        # it back to run it, so the blocks on the canvas, the values in the
        # inspector and the graph the engine executes cannot drift apart —
        # there is only the one description of them.
        "project": graph_to_dict(graph),
        "context": {
            "symbol_rate": graph.ctx.bit_rate,
            "samples_per_symbol": graph.ctx.samples_per_symbol,
            "sequence_length": graph.ctx.sequence_length,
            "num_samples": graph.ctx.num_samples,
            "seed": graph.ctx.seed,
            "format": FORMATS[BITS_PER_SYMBOL],
            "span_km": SPAN_KM,
            "gbps": SYMBOL_RATE * BITS_PER_SYMBOL / 1e9,
        },
    }

    destination = Path(__file__).parent / "ui_data.json"
    destination.write_text(json.dumps(payload, indent=1), encoding="utf-8")

    print(
        f"{FORMATS[BITS_PER_SYMBOL]} at {SYMBOL_RATE * BITS_PER_SYMBOL / 1e9:.0f} Gb/s: "
        f"EVM = {measurement.evm * 100:.2f}%, SNR = {measurement.snr_db:.2f} dB, "
        f"BER = {measurement.ber_estimated:.3e}, "
        f"{counted.symbol_errors} symbol errors in {counted.symbols_evaluated}"
    )
    print(f"{len(payload['manifests'])} component manifests")
    counts = payload["constellation"]["counts"]
    print(f"constellation histogram {len(counts)}x{len(counts[0])}")
    print(
        f"spectrum {len(spectrum['wavelengths_nm'])} points, peak "
        f"{spectrum['peak_nm']:.3f} nm at {spectrum['peak_dbm']:.2f} dBm per "
        f"{spectrum['resolution_bandwidth_ghz']:.1f} GHz, "
        f"OSNR {spectrum['osnr_db']:.2f} dB"
    )
    print(f"wrote {destination.name} ({destination.stat().st_size / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
