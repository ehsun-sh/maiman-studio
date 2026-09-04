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
from maiman.components import (
    CarrierRecovery,
    CoherentReceiver,
    ConstellationAnalyzer,
    ConstellationDiagram,
    CWLaser,
    DifferentialDecoder,
    DispersionCompensator,
    EyeDiagram,
    Fiber,
    IQDriver,
    IQModulator,
    IQSampler,
    PowerMeter,
    PRBSGenerator,
    QAMMapper,
)
from maiman.project import graph_to_dict

V_PI = 4.0
SYMBOL_RATE = 32e9
BITS_PER_SYMBOL = 4
SPAN_KM = 80.0
DISPERSION = 17.0  # ps/nm/km, standard single-mode fiber at 1550 nm
# BPSK is absent on purpose: differential *quadrant* encoding needs a quadrant to
# difference, and a two-point constellation has none. A binary format resolves its
# ambiguity by other means, and the mapper says so rather than silently coping.
FORMATS = {2: "QPSK", 4: "16-QAM", 6: "64-QAM", 8: "256-QAM"}


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
    lo = graph.add(CWLaser(power=10.0, wavelength=1550.0, linewidth=100.0, label="lo"))
    receiver = graph.add(CoherentReceiver(responsivity=0.8, label="rx"))
    # The eye is a block on the canvas now, not a call beside it. It was
    # computed here with eye_histogram and shipped as data, which meant the
    # interface drew an eye no graph on screen produced — and kept drawing it
    # after a live run, under a badge that said "live". A measurement the
    # picture shows should be a measurement the schematic contains.
    eye_block = graph.add(
        EyeDiagram(span_symbols=2.0, time_bins=64.0, amplitude_bins=72.0, label="eye")
    )
    compensator = graph.add(
        DispersionCompensator(
            accumulated_dispersion=DISPERSION * SPAN_KM, wavelength=1550.0, label="cdc"
        )
    )
    sampler = graph.add(IQSampler(matched_filter=True, roll_off=0.2, label="smp"))
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
    graph.connect(receiver["i"], eye_block["in"])
    graph.connect(receiver["i"], compensator["i"])
    graph.connect(receiver["q"], compensator["q"])
    graph.connect(compensator["i"], sampler["i"])
    graph.connect(compensator["q"], sampler["q"])
    graph.connect(mapper["out"], sampler["reference"])
    graph.connect(sampler["out"], recovery["in"])
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

    # The I-quadrature eye of a coherent receiver: a real thing to look at, and
    # for 16-QAM it shows the four levels the format actually carries. Taken off
    # the block in the graph, so this file and the session server produce it the
    # same way.
    eye = results[of_type(graph, EyeDiagram)]

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
    print(f"wrote {destination.name} ({destination.stat().st_size / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
