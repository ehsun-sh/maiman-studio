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
* given an erbium spectrum, each slice of that ASE drains at its own cross
  section and photon energy, and the mean over the band has a closed form for a
  flat and for a linearly tilted ``A + G*``;
* and with the flag off nothing moves.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from maiman import effective_time_constant, gain_transient, step_schedule
from maiman.components import EDFA
from maiman.transient import ErbiumSpectrum, self_saturation_weight, spectral_gain_transient
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


# ---------------------------------------------------------------------------
# Across the band: each slice of the ASE at its own cross section
# ---------------------------------------------------------------------------

#: 1500 to 1600 nm, which covers the 4 THz band about 1550 with room either side.
GRID = np.linspace(1500e-9, 1600e-9, 201)


def band_edges(edfa: EDFA) -> tuple[float, float, float, float]:
    """``(nu_c, B, nu_1, nu_2)`` for the amplifier's ASE band [Hz]."""
    centre = C_LIGHT / edfa.si("center_wavelength")
    width = edfa.si("bandwidth")
    return centre, width, centre - width / 2.0, centre + width / 2.0


def test_a_flat_cross_section_leaves_only_the_photon_energies() -> None:
    """``(nu_c / B) ln(nu_2 / nu_1)``: a watt of short-wavelength ASE is fewer photons."""
    edfa = amplifier()
    flat = ErbiumSpectrum(GRID, np.full(GRID.shape, 30.0), np.full(GRID.shape, 40.0))
    centre, width, low, high = band_edges(edfa)
    expected = centre / width * math.log(high / low)
    assert self_saturation_weight(edfa, flat) == pytest.approx(expected, rel=1e-9)
    assert 1.0 < expected < 1.0002, "a part in ten thousand, and above one"


@pytest.mark.parametrize("slope", [-0.2, 0.2])
def test_a_tilted_cross_section_weighs_the_band_by_its_integral(slope: float) -> None:
    """``A + G* = a + b lambda``: the weight integrates to a log and a reciprocal.

    ``w = [a c ln(nu_2/nu_1) + b c^2 (1/nu_1 - 1/nu_2)] / [B lambda_c (a + b lambda_c)]``.
    And it barely moves: across a band centred on the reference, a linear tilt
    gains on one side what it loses on the other, and only the second-order
    cross term with the photon energies survives -- 3e-4 at 0.2 dB/nm.
    """
    edfa = amplifier()
    absorption = 30.0 + slope * (GRID - 1550e-9) * 1e9  # dB, linear in lambda
    tilted = ErbiumSpectrum(GRID, absorption, np.full(GRID.shape, 40.0))
    a = 30.0 - slope * 1550.0 + 40.0  # A + G* = a + b lambda, lambda in nm
    b = slope
    _, width, low, high = band_edges(edfa)
    c_nm = C_LIGHT * 1e9
    expected = (a * c_nm * math.log(high / low) + b * c_nm**2 * (1.0 / low - 1.0 / high)) / (
        width * 1550.0 * (a + b * 1550.0)
    )
    weight = self_saturation_weight(edfa, tilted)
    assert weight == pytest.approx(expected, rel=1e-9)
    assert abs(weight - 1.0) < 1e-3, "a linear tilt cancels to first order"


def test_a_short_wavelength_peak_is_what_moves_it() -> None:
    """Curvature, not tilt: erbium's absorption peaks short of a C-band centre.

    Absorption rising 2 dB/nm below 1550 and flat above it -- the shape of the
    1530 nm peak, crudely -- makes the short half of the ASE drain hard and
    nothing cancels it: eleven percent more load than the centre alone says.
    The exact figure is this spectrum's; that it is well above one is the claim.
    """
    edfa = amplifier()
    absorption = 30.0 + np.where(GRID < 1550e-9, 2.0 * (1550e-9 - GRID) * 1e9, 0.0)
    peaked = ErbiumSpectrum(GRID, absorption, np.full(GRID.shape, 40.0))
    assert self_saturation_weight(edfa, peaked) == pytest.approx(1.113, abs=0.001)


def test_weighed_by_a_spectrum_the_dark_coil_rests_on_lambert_w_with_the_weight() -> None:
    """``W(beta w G_0) / (beta w)``, and a run started there does not move."""
    edfa = amplifier(30.0)
    absorption = 30.0 + np.where(GRID < 1550e-9, 2.0 * (1550e-9 - GRID) * 1e9, 0.0)
    tilted = ErbiumSpectrum(GRID, absorption, np.full(GRID.shape, 40.0))
    weight = self_saturation_weight(edfa, tilted)
    small_signal = db_to_linear(30.0)
    beta = weight * edfa.self_saturation_load() / edfa.intrinsic_saturation_power(small_signal)
    expected = lambert_w(beta * small_signal) / beta

    times, power = step_schedule([(20e-3, 0.0)], points_per_segment=200)
    run = spectral_gain_transient(edfa, tilted, times, power, [1550e-9])
    assert run.reference.gain[0] == pytest.approx(expected, rel=1e-12)
    assert run.reference.gain[-1] == pytest.approx(expected, rel=1e-9), "at rest, and stays"
    unweighted = gain_transient(edfa, times, power)
    assert run.reference.gain[0] < unweighted.gain[0], "this spectrum drains harder"


def test_a_spectrum_narrower_than_the_ase_is_refused_only_when_it_is_needed() -> None:
    narrow = np.linspace(1545e-9, 1555e-9, 21)
    spectrum = ErbiumSpectrum(narrow, np.full(narrow.shape, 30.0), np.full(narrow.shape, 40.0))
    times, power = step_schedule([(1e-3, 1e-5)], points_per_segment=16)
    with pytest.raises(ValueError, match="will not extrapolate"):
        spectral_gain_transient(amplifier(), spectrum, times, power, [1550e-9])
    # Without its own noise as a load there is nothing across the band to weigh.
    spectral_gain_transient(amplifier(own=False), spectrum, times, power, [1550e-9])
