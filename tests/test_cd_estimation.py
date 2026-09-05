"""Blind chromatic dispersion estimation: finding D·L without being told it.

:mod:`test_cd_compensation` checks that a declared value is removed correctly.
This checks the harder claim — that the value can be *measured* from the signal,
which is what a deployed receiver has to do at acquisition and what the
compensator's docstring used to say it did not.

Two things are worth guarding, and they need different kinds of test. That the
fast scan is really the slow scan, which is an identity and is checked as one to
floating-point. And that the estimate is close enough to be useful, which is not
a theorem and is checked by measuring the EVM the link actually delivers under
the estimated value against the EVM it delivers under the exact one.
"""

from __future__ import annotations

import numpy as np
import pytest

from maiman import Graph, SimulationContext
from maiman.component import Component
from maiman.components import (
    EDFA,
    CarrierRecovery,
    CoherentReceiver,
    ConstellationAnalyzer,
    CWLaser,
    DispersionCompensator,
    Fiber,
    IQDriver,
    IQModulator,
    IQSampler,
    PRBSGenerator,
    QAMMapper,
)
from maiman.dsp import (
    clock_tone,
    clock_tone_lobe,
    compensate_dispersion,
    estimate_dispersion,
    intensity_cost,
    scan_clock_tone,
)
from maiman.signals import ElectricalSignal
from maiman.units import C_LIGHT

#: s/m per ps/nm. The parameter is typed in ps/nm and the engine holds s/m.
PS_NM = 1e-3

WAVELENGTH_M = 1550e-9
SYMBOL_RATE = 32e9
DISPERSION = 17.0


def link(
    length_km: float,
    *,
    samples_per_symbol: int = 16,
    roll_off: float = 0.2,
    shaping: bool = True,
    bits_per_symbol: float = 4.0,
    dispersion: float = DISPERSION,
    sequence_length: int = 4096,
    noise_figure: float | None = None,
    symbol_rate: float = SYMBOL_RATE,
) -> tuple[Graph, ConstellationAnalyzer, DispersionCompensator, CoherentReceiver]:
    """A coherent link over dispersive fibre, with the compensator set exactly right.

    Deliberately the same shape as ``examples/dispersion_link.py``: one mechanism
    at a time, loss off unless an amplifier is asked for, so that what the
    estimator is being scored against is dispersion and not a power budget.
    """
    ctx = SimulationContext(
        bit_rate=symbol_rate,
        samples_per_symbol=samples_per_symbol,
        sequence_length=sequence_length,
        seed=2026,
        precision="double",
    )
    graph = Graph(ctx)
    prbs = graph.add(PRBSGenerator(order=23.0, bits_per_symbol=bits_per_symbol, label="prbs"))
    mapper = graph.add(QAMMapper(bits_per_symbol=bits_per_symbol, label="map"))
    driver = graph.add(
        IQDriver(
            v_pi=4.0,
            predistort=True,
            pulse_shaping=shaping,
            roll_off=roll_off,
            drive_ratio=0.4,
            label="drv",
        )
    )
    laser = graph.add(CWLaser(power=0.0, wavelength=1550.0, linewidth=0.0, label="tx"))
    modulator = graph.add(IQModulator(v_pi=4.0, label="mod"))
    lo = graph.add(CWLaser(power=13.0, wavelength=1550.0, linewidth=0.0, label="lo"))
    receiver = graph.add(CoherentReceiver(responsivity=0.8, label="rx"))
    compensator = graph.add(
        DispersionCompensator(
            accumulated_dispersion=dispersion * length_km, wavelength=1550.0, label="cdc"
        )
    )
    sampler = graph.add(IQSampler(matched_filter=shaping, roll_off=roll_off, label="smp"))
    recovery = graph.add(CarrierRecovery(label="cr"))
    analyzer = graph.add(ConstellationAnalyzer(ignore_edges=64.0, label="vsa"))

    graph.connect(prbs["out"], mapper["in"])
    graph.connect(mapper["out"], driver["in"])
    graph.connect(laser, modulator["optical_in"])
    graph.connect(driver["i"], modulator["i"])
    graph.connect(driver["q"], modulator["q"])

    tail: Component = modulator
    if length_km:
        fiber = graph.add(
            Fiber(
                length=length_km,
                attenuation=0.2 if noise_figure is not None else 0.0,
                dispersion=dispersion,
                nonlinearity=0.0,
                label="fib",
            )
        )
        graph.connect(tail, fiber["in"])
        tail = fiber
    if noise_figure is not None:
        amplifier = graph.add(EDFA(gain=0.2 * length_km, noise_figure=noise_figure, label="amp"))
        graph.connect(tail, amplifier["in"])
        tail = amplifier

    graph.connect(tail, receiver["in"])
    graph.connect(lo, receiver["lo"])
    graph.connect(receiver["i"], compensator["i"])
    graph.connect(receiver["q"], compensator["q"])
    graph.connect(compensator["i"], sampler["i"])
    graph.connect(compensator["q"], sampler["q"])
    graph.connect(mapper["out"], sampler["reference"])
    graph.connect(sampler["out"], recovery["in"])
    graph.connect(recovery["out"], analyzer["in"])
    graph.connect(mapper["out"], analyzer["reference"])
    return graph, analyzer, compensator, receiver


def received(graph: Graph, receiver: CoherentReceiver) -> tuple[np.ndarray, float]:
    """The complex baseband the receiver hands the compensator, and its rate."""
    results = graph.run(keep=[receiver])
    in_phase: ElectricalSignal = results.port(receiver, "i")
    quadrature: ElectricalSignal = results.port(receiver, "q")
    baseband = np.asarray(in_phase.samples, dtype=np.float64) + 1j * np.asarray(
        quadrature.samples, dtype=np.float64
    )
    return baseband, in_phase.fs


# ---------------------------------------------------------------------------
# the clock tone itself


def test_a_shaped_signal_carries_a_line_at_the_symbol_rate() -> None:
    """The whole method rests on the line being there. Measure it before using it."""
    graph, _, _, receiver = link(0.0)
    baseband, fs = received(graph, receiver)
    tone = clock_tone(baseband, fs, symbol_rate=SYMBOL_RATE)

    spectrum = np.abs(np.fft.fft(np.abs(baseband) ** 2))
    # Against the neighbourhood it sits in, not against nothing: a line has to
    # stand out of the intensity spectrum around it to be a line at all.
    index = round(SYMBOL_RATE / fs * len(baseband))
    neighbourhood = np.median(spectrum[index - 200 : index + 200])
    assert abs(tone) > 20.0 * neighbourhood


def test_the_symbol_rate_must_land_on_a_bin() -> None:
    """Rather than being rounded to the nearest one behind the caller's back."""
    with pytest.raises(ValueError, match="does not land on an FFT bin"):
        clock_tone(np.zeros(1000, dtype=complex), 1e9, symbol_rate=333.5e6)


def test_the_narrowest_lobe_is_the_one_the_grid_has_to_fit_inside() -> None:
    """``c / (lambda^2 * Rs^2)`` — 122 ps/nm at 32 GBd, and four times that at 16."""
    assert clock_tone_lobe(WAVELENGTH_M, SYMBOL_RATE) == pytest.approx(
        C_LIGHT / (WAVELENGTH_M**2 * SYMBOL_RATE**2), abs=0.0, rel=1e-12
    )
    assert clock_tone_lobe(WAVELENGTH_M, SYMBOL_RATE) / PS_NM == pytest.approx(121.9, rel=1e-3)
    # Quadratic in the symbol rate, which is why a 10 GBd link resolves ten
    # times worse than a 32 GBd one rather than three times.
    assert clock_tone_lobe(WAVELENGTH_M, SYMBOL_RATE / 2) == pytest.approx(
        4.0 * clock_tone_lobe(WAVELENGTH_M, SYMBOL_RATE), abs=0.0, rel=1e-12
    )


# ---------------------------------------------------------------------------
# the fast scan is the slow scan


def test_the_scan_agrees_with_compensating_every_candidate() -> None:
    """The identity the acquisition stage is built on, checked as an identity.

    ``scan_clock_tone`` never calls ``compensate_dispersion``; it computes what
    the tone *would* be from one forward transform. If the two ever disagree the
    fast path has stopped being the honest path, and nothing else in the suite
    would notice, because every other test only ever sees the fast one.
    """
    graph, _, _, receiver = link(80.0)
    baseband, fs = received(graph, receiver)
    candidates = np.arange(-2000.0, 2000.1, 137.0) * PS_NM

    fast = scan_clock_tone(
        baseband,
        fs,
        symbol_rate=SYMBOL_RATE,
        wavelength=WAVELENGTH_M,
        candidates=candidates,
        energy_kept=1.0,
    )
    slow = np.array(
        [
            abs(
                clock_tone(
                    compensate_dispersion(
                        baseband, fs, accumulated_dispersion=value, wavelength=WAVELENGTH_M
                    ),
                    fs,
                    symbol_rate=SYMBOL_RATE,
                )
            )
            for value in candidates
        ]
    )
    assert fast == pytest.approx(slow, abs=0.0, rel=1e-12)


def test_dropping_the_quiet_bins_does_not_move_the_peak() -> None:
    """The one approximation in the acquisition stage, bounded by measurement.

    Only bins where the spectrum overlaps its own symbol-rate-shifted copy carry
    any beat; the rest hold noise, and summing them costs a factor of twelve.
    Dropping them is not free — away from the peak, where the tone has cancelled,
    what is dropped is most of what remains — so what has to hold is that the
    peak survives, on the grid acquisition really uses.
    """
    graph, _, _, receiver = link(80.0)
    baseband, fs = received(graph, receiver)
    step = clock_tone_lobe(WAVELENGTH_M, SYMBOL_RATE) / 2.0
    candidates = np.arange(-2000.0 * PS_NM, 2000.0 * PS_NM, step)

    def scan(energy_kept: float) -> np.ndarray:
        return scan_clock_tone(
            baseband,
            fs,
            symbol_rate=SYMBOL_RATE,
            wavelength=WAVELENGTH_M,
            candidates=candidates,
            energy_kept=energy_kept,
        )

    exact = scan(1.0)
    truncated = scan_clock_tone(
        baseband,
        fs,
        symbol_rate=SYMBOL_RATE,
        wavelength=WAVELENGTH_M,
        candidates=candidates,
    )
    assert candidates[int(truncated.argmax())] == candidates[int(exact.argmax())]
    assert np.abs(truncated - exact).max() / exact.max() < 0.01


def test_the_scan_peaks_where_the_dispersion_is() -> None:
    """Coarse, but on a grid fine enough to be checked against the truth."""
    graph, _, _, receiver = link(80.0)
    baseband, fs = received(graph, receiver)
    candidates = np.arange(-3000.0, 3000.1, 25.0) * PS_NM

    strength = scan_clock_tone(
        baseband, fs, symbol_rate=SYMBOL_RATE, wavelength=WAVELENGTH_M, candidates=candidates
    )
    peak = float(candidates[int(strength.argmax())]) / PS_NM
    assert peak == pytest.approx(DISPERSION * 80.0, abs=100.0)
    assert strength.max() / np.median(strength) > 5.0


# ---------------------------------------------------------------------------
# the refinement cost


def test_the_intensity_cost_is_lowest_at_the_true_value() -> None:
    """And is a bowl around it, which is what the parabola fit assumes."""
    graph, _, _, receiver = link(80.0)
    baseband, fs = received(graph, receiver)
    truth = DISPERSION * 80.0 * PS_NM

    offsets = np.array([-120.0, -60.0, -20.0, 0.0, 20.0, 60.0, 120.0]) * PS_NM
    cost = np.array(
        [
            intensity_cost(baseband, fs, wavelength=WAVELENGTH_M, accumulated=truth + offset)
            for offset in offsets
        ]
    )
    assert int(cost.argmin()) == 3, cost
    # Monotone away from the minimum on each side: a bowl, not a plateau with a
    # dip somewhere in it.
    assert np.all(np.diff(cost[3:]) > 0.0), cost
    assert np.all(np.diff(cost[:4]) < 0.0), cost


def test_dispersion_drives_the_intensity_towards_a_gaussian() -> None:
    """Which is the mechanism the cost measures, stated as a number.

    Left uncompensated, the fourth moment of a heavily dispersed intensity
    approaches the value 2 that a complex Gaussian has; compensated, it is well
    below it.
    """
    graph, _, _, receiver = link(1000.0)
    baseband, fs = received(graph, receiver)

    dispersed = intensity_cost(baseband, fs, wavelength=WAVELENGTH_M, accumulated=0.0)
    restored = intensity_cost(
        baseband, fs, wavelength=WAVELENGTH_M, accumulated=DISPERSION * 1000.0 * PS_NM
    )
    # Within a few percent of 2, the value a complex Gaussian has: a thousand
    # kilometres of dispersion has left the field indistinguishable from noise.
    assert dispersed == pytest.approx(2.0, abs=0.06)
    # And a quarter below it once the dispersion is taken back out.
    assert restored < 0.8 * dispersed


# ---------------------------------------------------------------------------
# the estimate, scored the only way that matters


@pytest.mark.parametrize("length_km", [0.0, 20.0, 80.0, 400.0, 1000.0])
def test_the_span_is_found_without_being_declared(length_km: float) -> None:
    """From back to back to a thousand kilometres, on one set of settings.

    Scored twice: against the truth in ps/nm, and against the EVM the link
    delivers when told the exact value. The second is the one that matters — an
    estimate is only as good as the link it leaves behind — and it is why the
    tolerance here is not tighter. 20 ps/nm of residual costs 8 dB of SNR at this
    baud rate, so a test that allowed 50 would be allowing a broken link.
    """
    graph, analyzer, compensator, receiver = link(length_km)
    truth = DISPERSION * length_km * PS_NM

    baseband, fs = received(graph, receiver)
    found = estimate_dispersion(
        baseband,
        fs,
        symbol_rate=SYMBOL_RATE,
        wavelength=WAVELENGTH_M,
        search_range=20000.0 * PS_NM,
    )
    assert found.accumulated_dispersion == pytest.approx(truth, abs=15.0 * PS_NM)

    exact = graph.run()[analyzer]
    blind = graph.run(overrides={(compensator, "estimate"): True})[analyzer]
    assert blind.evm < exact.evm + 0.015


@pytest.mark.parametrize(
    ("name", "settings"),
    [
        ("two samples per symbol", {"samples_per_symbol": 2}),
        ("four samples per symbol", {"samples_per_symbol": 4}),
        ("QPSK", {"bits_per_symbol": 2.0}),
        ("64-QAM", {"bits_per_symbol": 6.0}),
        ("negative dispersion", {"dispersion": -DISPERSION}),
        ("roll-off 0.05", {"roll_off": 0.05}),
        ("roll-off 0.35", {"roll_off": 0.35}),
        ("a short window", {"sequence_length": 512}),
        ("an amplifier", {"noise_figure": 5.0}),
    ],
)
def test_the_estimate_survives(name: str, settings: dict[str, float]) -> None:
    """The span is 80 km throughout; only the thing named changes.

    Every one of these was a guess about what might break it before it was run,
    and two of them did: roll-off 0.05 acquires 950 ps/nm low, and the estimate
    only lands because refinement walks its window in. Losing that walk is a
    change these cases will report and none of the others will.
    """
    graph, _, _, receiver = link(80.0, **settings)  # type: ignore[arg-type]
    truth = settings.get("dispersion", DISPERSION) * 80.0 * PS_NM
    rate = settings.get("symbol_rate", SYMBOL_RATE)

    baseband, fs = received(graph, receiver)
    found = estimate_dispersion(
        baseband, fs, symbol_rate=rate, wavelength=WAVELENGTH_M, search_range=20000.0 * PS_NM
    )
    assert found.accumulated_dispersion == pytest.approx(truth, abs=50.0 * PS_NM), name


def test_resolution_scales_with_the_square_of_the_baud_rate() -> None:
    """The documented limit, measured rather than asserted from the formula.

    The beat's phase lever is proportional to the symbol rate and so is the band
    it acts over, so a link at a third of the baud rate resolves about ten times
    worse. It is worth pinning because it is the reason the estimator is offered
    for a coherent transceiver and not as a general-purpose tool.
    """
    errors = {}
    for rate in (SYMBOL_RATE, 10e9):
        graph, _, _, receiver = link(80.0, symbol_rate=rate)
        baseband, fs = received(graph, receiver)
        found = estimate_dispersion(
            baseband, fs, symbol_rate=rate, wavelength=WAVELENGTH_M, search_range=20000.0 * PS_NM
        )
        errors[rate] = abs(found.accumulated_dispersion - DISPERSION * 80.0 * PS_NM) / PS_NM

    assert errors[SYMBOL_RATE] < 15.0
    assert 25.0 < errors[10e9] < 150.0


# ---------------------------------------------------------------------------
# the component


def test_the_block_reports_what_it_estimated_and_what_was_declared() -> None:
    """Both, so the estimate arrives as something to check rather than to trust."""
    graph, _, compensator, _ = link(80.0)
    results = graph.run(overrides={(compensator, "estimate"): True}, keep=[compensator])
    diagnostics = results.port(compensator, "diagnostics")

    assert diagnostics.estimated is True
    assert diagnostics.declared == pytest.approx(DISPERSION * 80.0 * PS_NM, abs=0.0, rel=1e-12)
    assert diagnostics.accumulated_dispersion == pytest.approx(
        DISPERSION * 80.0 * PS_NM, abs=15.0 * PS_NM
    )
    # Not the declared value passed through: the two agree because the estimate
    # is good, not because the parameter was copied into the answer.
    assert diagnostics.accumulated_dispersion != diagnostics.declared
    assert diagnostics.contrast > 5.0


def test_the_declared_value_is_ignored_when_the_search_is_on() -> None:
    """Set the parameter to nonsense and the link still works.

    This is the claim that the search is a search. A block that quietly fell back
    to the parameter would pass every other test in this file, because everywhere
    else the parameter happens to be right.
    """
    graph, analyzer, compensator, _ = link(80.0)
    overrides = {(compensator, "accumulated_dispersion"): 0.0, (compensator, "estimate"): True}

    exact = graph.run()[analyzer]
    blind = graph.run(overrides=overrides)[analyzer]  # type: ignore[arg-type]
    assert blind.evm < exact.evm + 0.015

    uncompensated = graph.run(overrides={(compensator, "accumulated_dispersion"): 0.0})[analyzer]
    assert uncompensated.evm > 0.5


def test_the_search_is_off_by_default() -> None:
    """It costs half a second a run, and a declared value is the common case."""
    graph, _, compensator, _ = link(80.0)
    results = graph.run(keep=[compensator])
    diagnostics = results.port(compensator, "diagnostics")

    assert compensator.estimate is False
    assert diagnostics.estimated is False
    assert diagnostics.contrast == 0.0
    assert diagnostics.accumulated_dispersion == pytest.approx(
        DISPERSION * 80.0 * PS_NM, abs=0.0, rel=1e-12
    )


def test_the_search_range_bounds_what_can_be_found() -> None:
    """A span outside the range cannot be acquired, and the block does not pretend.

    Worth a test because the failure is silent otherwise: the scan returns the
    best candidate it was given, and the best of a set that excludes the answer
    is still a number.
    """
    graph, _, _, receiver = link(400.0)
    baseband, fs = received(graph, receiver)
    truth = DISPERSION * 400.0 * PS_NM

    wide = estimate_dispersion(
        baseband, fs, symbol_rate=SYMBOL_RATE, wavelength=WAVELENGTH_M, search_range=20000.0 * PS_NM
    )
    assert wide.accumulated_dispersion == pytest.approx(truth, abs=15.0 * PS_NM)

    narrow = estimate_dispersion(
        baseband, fs, symbol_rate=SYMBOL_RATE, wavelength=WAVELENGTH_M, search_range=1000.0 * PS_NM
    )
    assert abs(narrow.accumulated_dispersion - truth) > 1000.0 * PS_NM


def test_the_acquisition_grid_is_half_a_lobe() -> None:
    """Reported, so the resolution the search started from is not a hidden constant."""
    graph, _, _, receiver = link(80.0)
    baseband, fs = received(graph, receiver)
    found = estimate_dispersion(
        baseband, fs, symbol_rate=SYMBOL_RATE, wavelength=WAVELENGTH_M, search_range=2000.0 * PS_NM
    )
    assert found.step == pytest.approx(
        clock_tone_lobe(WAVELENGTH_M, SYMBOL_RATE) / 2.0, abs=0.0, rel=1e-12
    )
    # Acquisition alone lands within a couple of grid steps; refinement is what
    # closes the rest, and the two are reported separately so which one moved is
    # visible when one of them stops working.
    assert abs(found.acquired - DISPERSION * 80.0 * PS_NM) < 3.0 * found.step
