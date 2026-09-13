"""The delay line, and the recirculating loop it turns a pass count into.

A Feedback on its own is a fixed-point iteration: every lap lands on top of the
last, which is right for a cavity short against the window and wrong for a fibre
loop. With a delay in the loop every lap arrives one loop-time after the one
before, at the amplitude the coupler and the loop's loss leave it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from maiman import Graph, SimulationContext
from maiman.components import (
    Attenuator,
    DelayLine,
    DirectionalCoupler,
    Feedback,
    GaussianPulse,
)
from maiman.components.delay import delay_band, delay_signal
from maiman.signals import Band, NoiseBin, OpticalSignal

F0 = 193.4e12
FS = 160e9
N = 1024


def pulse_band(centre: float, width: float = 10e-12) -> Band:
    t = np.arange(N) / FS
    field = np.exp(-(((t - centre) / width) ** 2) / 2.0).astype(np.complex128)
    return Band(Ex=field, Ey=np.zeros(N, dtype=np.complex128), f0=F0, fs=FS)


def centroid(band: Band) -> float:
    power = np.abs(band.Ex) ** 2 + np.abs(band.Ey) ** 2
    t = np.arange(band.num_samples) / band.fs
    return float(np.sum(power * t) / np.sum(power))


# --------------------------------------------------------------------------
# The block on its own
# --------------------------------------------------------------------------


def test_a_whole_number_of_samples_is_a_shift_and_a_carrier_phase() -> None:
    """Exact: no phase ramp, no ringing, just the samples moved and the carrier turned."""
    band = pulse_band(2e-9)
    delay = 37 / FS
    out = delay_band(band, delay)
    carrier = np.exp(-2j * np.pi * math.fmod(F0 * delay, 1.0))
    assert np.array_equal(out.Ex, np.roll(band.Ex, 37) * carrier)


def test_a_fractional_delay_moves_the_pulse_by_exactly_that() -> None:
    """A tenth of a sample and then some, read off the centroid of a band-limited pulse."""
    band = pulse_band(2e-9)
    delay = 123.4 / FS
    out = delay_band(band, delay)
    assert centroid(out) - centroid(band) == pytest.approx(delay, abs=1e-6 / FS)
    assert out.average_power() == pytest.approx(band.average_power(), rel=1e-12, abs=0.0)


def test_the_carrier_turns_by_the_optical_phase_of_the_delay() -> None:
    """What makes two arms of different length interfere: -2 pi f0 delay, modulo a cycle."""
    cw = Band(Ex=np.ones(N, dtype=np.complex128), Ey=np.zeros(N, dtype=np.complex128), f0=F0, fs=FS)
    quarter_cycle = 0.25 / F0
    out = delay_band(cw, 40 / FS + quarter_cycle)
    expected = np.exp(-2j * np.pi * math.fmod(F0 * (40 / FS + quarter_cycle), 1.0))
    assert np.allclose(out.Ex, expected, rtol=0.0, atol=1e-9)


def test_noise_and_path_history_pass_through_untouched() -> None:
    """A delay moves when noise arrives, not how much of it there is."""
    noise = NoiseBin(f_start=F0 - 50e9, f_end=F0 + 50e9, psd_x=1e-18, psd_y=1e-18)
    signal = OpticalSignal(bands=(pulse_band(2e-9),), noise=(noise,), accumulated_gvd=-3e-24)
    out = delay_signal(signal, 5e-12)
    assert out.noise == signal.noise
    assert out.accumulated_gvd == signal.accumulated_gvd


def test_no_delay_is_the_same_signal_and_a_negative_one_is_refused() -> None:
    signal = OpticalSignal(bands=(pulse_band(2e-9),))
    assert delay_signal(signal, 0.0) is signal
    with pytest.raises(ValueError, match="cannot advance"):
        delay_signal(signal, -1e-12)


def test_the_block_keeps_the_single_precision_it_was_given() -> None:
    ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=16, sequence_length=64)
    pulse = GaussianPulse(width=10.0).run(ctx, {})["out"]
    assert isinstance(pulse, OpticalSignal)
    out = DelayLine(delay=12.3).run(ctx, {"in": pulse})["out"]
    assert isinstance(out, OpticalSignal)
    assert out.bands[0].Ex.dtype == pulse.bands[0].Ex.dtype


# --------------------------------------------------------------------------
# A recirculating loop
# --------------------------------------------------------------------------

LOOP_PS = 800.0
LOOP_LOSS_DB = 1.0


def loop_graph(*, passes: int, delay_ps: float) -> tuple[Graph, DirectionalCoupler]:
    """A pulse into a 3 dB coupler whose second pair of ports is joined by a delayed loop."""
    ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=16, sequence_length=64)
    graph = Graph(ctx)
    pulse = graph.add(GaussianPulse(width=10.0, label="pulse"))
    coupler = graph.add(DirectionalCoupler(True, coupling=0.5, label="coupler"))
    delay = graph.add(DelayLine(delay=delay_ps, label="delay"))
    loss = graph.add(Attenuator(attenuation=LOOP_LOSS_DB, label="loss"))
    loop = graph.add(Feedback(passes=float(passes), label="loop"))
    graph.connect(pulse, coupler["in1"])
    graph.connect(coupler["out2"], delay["in"])
    graph.connect(delay["out"], loss["in"])
    graph.connect(loss["out"], loop["in"])
    graph.connect(loop["out"], coupler["in2"])
    return graph, coupler


def lap_powers(graph: Graph, coupler: DirectionalCoupler, laps: int) -> list[float]:
    """Power on the coupler's output at the instant each lap's peak should arrive."""
    ctx = graph.ctx
    signal = graph.run(keep=[coupler]).port(coupler, "out1")
    assert isinstance(signal, OpticalSignal)
    (band,) = signal.bands
    power = np.abs(band.Ex.astype(np.complex128)) ** 2
    start = ctx.num_samples // 2
    step = round(LOOP_PS * 1e-12 * ctx.sample_rate)
    return [float(power[start + lap * step]) for lap in range(laps)]


def test_each_lap_arrives_one_loop_time_later_at_the_loop_s_gain() -> None:
    """Lap n is the through path, then two crossings, then n - 1 more straight laps.

    In power: the first lap is the coupler's straight half; the second is both
    crossings and one pass of loss, a quarter of the loop's power gain; every lap
    after that is another half for the straight path and another pass of loss.
    """
    graph, coupler = loop_graph(passes=4, delay_ps=LOOP_PS)
    powers = lap_powers(graph, coupler, laps=4)
    gain = 10.0 ** (-LOOP_LOSS_DB / 10.0)
    peak = 1e-3  # GaussianPulse's default 0 dBm peak
    expected = [peak * 0.5, peak * 0.25 * gain]
    expected += [expected[-1] * (0.5 * gain) ** n for n in (1, 2)]
    for got, want in zip(powers, expected, strict=True):
        assert got == pytest.approx(want, rel=1e-4, abs=0.0)


def test_the_pass_count_is_the_number_of_laps() -> None:
    """Three passes put three pulses on the output, and nothing where a fourth would go."""
    graph, coupler = loop_graph(passes=3, delay_ps=LOOP_PS)
    powers = lap_powers(graph, coupler, laps=4)
    assert all(p > 1e-6 for p in powers[:3])
    assert powers[3] < 1e-12


def test_without_the_delay_every_lap_lands_on_the_first() -> None:
    """The contrast that is the reason this block exists: no second pulse anywhere."""
    graph, coupler = loop_graph(passes=4, delay_ps=0.0)
    powers = lap_powers(graph, coupler, laps=2)
    assert powers[1] < 1e-12
    assert powers[0] != pytest.approx(1e-3 * 0.5, rel=1e-3)


def test_the_example_prints_the_table_the_readme_quotes(capsys: pytest.CaptureFixture[str]) -> None:
    """``examples/recirculating_loop.py`` is where the README's lap table comes from."""
    import recirculating_loop

    recirculating_loop.main()
    printed = capsys.readouterr().out
    for row in ("0.5000mW", "0.1986mW", "0.0789mW", "0.0313mW"):
        assert printed.count(row) == 2, row  # measured, and the arithmetic beside it
    assert " 0.0000mW   and nothing where the second should be" in printed
