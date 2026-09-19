"""An amplifier's own spontaneous emission, emptying the inversion it comes from.

Saleh's equation is the reservoir's energy balance: ``P_sat ln(G_0 / G)`` is
everything the inversion hands out, over every field in the fibre. The ASE the
amplifier generates is one of those fields -- with no input, leaving by both
ends -- and with ``self_saturation`` it is counted. What holds that in place:

* with no input the gain has a closed form, ``W(beta G_0) / beta`` in Lambert's
  W, solved here by Halley's iteration on ``W e^W`` and sharing nothing with the
  component's Newton on ``ln G``;
* the load the reservoir is charged is exactly the ASE the block emits from one
  end, doubled for the other -- two code paths, one through ``n_sp``;
* the transient integrated to rest lands on the static solve, and relaxes at the
  rate the linearisation says, its own ASE included;
* and with the flag off nothing moves.
"""

from __future__ import annotations

import math

import pytest

from maiman import effective_time_constant, gain_transient, step_schedule
from maiman.components import EDFA
from maiman.units import C_LIGHT, H_PLANCK, db_to_linear

CHANNEL = 10 ** (-6 / 10) / 1e3


def amplifier(gain: float = 30.0, *, own: bool = True, **extra: float) -> EDFA:
    return EDFA(
        gain=gain, saturate=True, saturation_power=17.0, self_saturation=own, label="e", **extra
    )


def lambert_w(x: float) -> float:
    """The principal branch of ``W e^W = x`` for ``x >= 0``, by Halley's iteration."""
    w = math.log1p(x)
    for _ in range(100):
        e = math.exp(w)
        f = w * e - x
        step = f / (e * (w + 1.0) - (w + 2.0) * f / (2.0 * w + 2.0))
        w -= step
        if abs(step) < 1e-15 * max(1.0, abs(w)):
            break
    return w


def test_it_is_off_by_default_and_moves_nothing_until_asked() -> None:
    assert not EDFA().self_saturation
    assert EDFA.param_specs()["self_saturation"].applies_when == "saturate"
    plain = EDFA(gain=30.0, saturate=True, saturation_power=17.0)
    assert plain.self_saturation_load() == 0.0
    assert plain.effective_gain(0.0) == db_to_linear(30.0)
    # Asked for without saturation it has nothing to act on.
    assert EDFA(gain=30.0, self_saturation=True).effective_gain(0.0) == db_to_linear(30.0)


@pytest.mark.parametrize("gain_db", [20.0, 30.0, 35.0, 40.0])
def test_in_the_dark_the_gain_is_lambert_w(gain_db: float) -> None:
    edfa = amplifier(gain_db)
    small_signal = db_to_linear(gain_db)
    beta = edfa.self_saturation_load() / edfa.intrinsic_saturation_power(small_signal)
    expected = lambert_w(beta * small_signal) / beta
    assert edfa.effective_gain(0.0) == pytest.approx(expected, rel=1e-12)
    assert edfa.effective_gain(0.0) < small_signal


def test_the_load_is_the_ase_the_block_emits_from_each_end() -> None:
    """``2 S_ASE B`` per end over both polarizations, through ``n_sp``: the same power."""
    edfa = amplifier(30.0, noise_figure=5.5)
    gain = edfa.effective_gain(1e-6)
    one_end = 2.0 * edfa.ase_psd(gain) * edfa.si("bandwidth")
    assert edfa.self_saturation_load() * gain == pytest.approx(2.0 * one_end, rel=1e-12)
    frequency = C_LIGHT / edfa.si("center_wavelength")
    by_hand = 2.0 * db_to_linear(5.5) * H_PLANCK * frequency * edfa.si("bandwidth")
    assert edfa.self_saturation_load() == pytest.approx(by_hand, rel=1e-12)


def test_the_balance_holds_with_signal_and_noise_together() -> None:
    edfa = amplifier(35.0)
    small_signal = db_to_linear(35.0)
    saturation = edfa.intrinsic_saturation_power(small_signal)
    for dbm in (-40.0, -25.0, -10.0, 0.0):
        power = 10 ** (dbm / 10) / 1e3
        gain = edfa.effective_gain(power)
        handed_out = (gain - 1.0) * power + edfa.self_saturation_load() * gain
        assert saturation * math.log(small_signal / gain) == pytest.approx(handed_out, rel=1e-10)


def test_it_matters_where_the_amplifier_is_lightly_loaded() -> None:
    """The lighter the load and the higher the gain, the more its own noise costs it."""
    costs = {}
    for gain_db in (20.0, 30.0, 40.0):
        for dbm in (-40.0, 0.0):
            power = 10 ** (dbm / 10) / 1e3
            own = amplifier(gain_db).effective_gain(power)
            plain = amplifier(gain_db, own=False).effective_gain(power)
            costs[gain_db, dbm] = 10 * math.log10(plain / own)
    assert costs[20.0, -40.0] == pytest.approx(0.020, abs=0.002)
    assert costs[30.0, -40.0] == pytest.approx(0.186, abs=0.002)
    assert costs[40.0, -40.0] == pytest.approx(1.382, abs=0.005)
    for gain_db in (20.0, 30.0, 40.0):
        assert costs[gain_db, 0.0] < 0.02, "a loaded amplifier hardly notices"


def test_a_coil_whose_noise_would_empty_it_reports_unity_gain() -> None:
    edfa = amplifier(10.0, bandwidth=1e9)  # an absurd bandwidth, in THz
    assert edfa.effective_gain(0.0) == 1.0


def test_the_transient_comes_to_rest_on_the_static_solve() -> None:
    """Dropping every channel lands on Lambert's gain, not on ``G_0``."""
    times, power = step_schedule([(10e-3, 8 * CHANNEL), (400e-3, 0.0)], points_per_segment=800)
    own = gain_transient(amplifier(), times, power)
    plain = gain_transient(amplifier(own=False), times, power)
    assert own.gain[0] == pytest.approx(amplifier().effective_gain(8 * CHANNEL), rel=1e-12)
    assert own.gain[-1] == pytest.approx(amplifier().effective_gain(0.0), rel=1e-9)
    assert plain.gain[-1] == pytest.approx(db_to_linear(30.0), rel=1e-6)
    assert own.gain_db[-1] < plain.gain_db[-1] - 0.15


@pytest.mark.parametrize("dbm", [-40.0, -20.0])
def test_it_relaxes_at_the_rate_its_own_noise_adds(dbm: float) -> None:
    """``tau / (1 + (P_out + P_ASE) / P_sat)``, measured from a small displacement."""
    edfa = amplifier(35.0)
    power = 10 ** (dbm / 10) / 1e3
    rest = edfa.effective_gain(power)
    constant = effective_time_constant(edfa, power)
    assert constant < effective_time_constant(amplifier(35.0, own=False), power)
    times, drive = step_schedule([(0.5 * constant, power)], points_per_segment=400)
    run = gain_transient(edfa, times, drive, initial_gain=rest * 1.0001)
    start = math.log(run.gain[0] / rest)
    end = math.log(run.gain[-1] / rest)
    measured = float(times[-1]) / math.log(start / end)
    assert measured == pytest.approx(constant, rel=1e-3)
