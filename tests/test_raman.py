"""Stimulated Raman scattering between channels.

A photon can scatter off a silica vibration and come out at a lower frequency,
and the process is stimulated: light already there at the lower frequency makes
it more likely. So the short-wavelength end of a comb pumps the long-wavelength
end, and a flat launch does not arrive flat.

The model is a closed form, so most of these are exact statements rather than
trends: power is conserved to floating point, the reference frequency cancels,
and the tilt is proportional to the total power, the effective length and the
separation. The one number that comes from outside is the scale, and it is
anchored against what a filled C band actually does over one span.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from maiman.components import Fiber
from maiman.context import SimulationContext
from maiman.kernels import (
    RAMAN_TRIANGLE_LIMIT,
    attenuation_db_per_m_to_alpha,
    effective_length,
    raman_tilt,
)
from maiman.signals import Band, OpticalSignal
from maiman.units import dbm_to_w, from_si, to_si

ANCHOR = 193.1e12
SPACING = 100e9
#: 0.028 1/(W*km*THz), standard fibre at 1550 nm, in SI.
C_R = to_si(0.028, "1/W/km/THz")

ALPHA = attenuation_db_per_m_to_alpha(0.2e-3)
L_EFF = effective_length(ALPHA, 80e3)

SAMPLES = 256
FS = 160e9
CTX = SimulationContext(
    bit_rate=10e9, samples_per_symbol=16, sequence_length=16, seed=3, precision="double"
)


def comb(
    channels: int, power: float = 1e-3, spacing: float = SPACING
) -> tuple[list[float], list[float]]:
    return (
        [ANCHOR + index * spacing for index in range(channels)],
        [power] * channels,
    )


def tilt_db(ratios: list[float]) -> float:
    """Decibels the long-wavelength end gained over the short one.

    Channel 0 sits at the lowest frequency and so the longest wavelength, which
    is the end Raman scattering feeds.
    """
    return 10.0 * math.log10(ratios[0] / ratios[-1])


# ---------------------------------------------------------------------------
# the closed form


def test_the_transfer_conserves_power() -> None:
    """It moves power between channels; it does not remove any.

    The quantum defect — the energy the lattice keeps — is a part in ten
    thousand at these separations and is not modelled, which this asserts rather
    than only says. It is also what makes a sign error impossible to hide: the
    two ends must move in opposite directions or the sum cannot come out right.
    """
    frequencies, powers = comb(40)
    ratios = raman_tilt(frequencies, powers, gain_slope=C_R, effective_length=L_EFF)
    moved = sum(p * r for p, r in zip(powers, ratios, strict=True))
    assert moved == pytest.approx(sum(powers), abs=0.0, rel=1e-12)


def test_the_long_wavelengths_gain_and_the_short_ones_pay() -> None:
    """The direction, which is the whole of what the effect is."""
    frequencies, powers = comb(8)
    ratios = raman_tilt(frequencies, powers, gain_slope=C_R, effective_length=L_EFF)

    assert ratios[0] > 1.0, "the longest wavelength should end up above what it launched"
    assert ratios[-1] < 1.0, "the shortest should end up below it"
    assert list(ratios) == sorted(ratios, reverse=True), "monotone across the comb"


def test_the_reference_frequency_cancels() -> None:
    """Shifting the whole comb changes the ratios not at all.

    It has to: the reference appears in the numerator and the denominator of the
    same expression. Worth a test because the implementation subtracts the mean
    for numerical reasons, and a subtraction done for numerical reasons is
    exactly the kind that quietly becomes physics.
    """
    frequencies, powers = comb(12)
    here = raman_tilt(frequencies, powers, gain_slope=C_R, effective_length=L_EFF)
    elsewhere = raman_tilt(
        [f + 7.3e12 for f in frequencies], powers, gain_slope=C_R, effective_length=L_EFF
    )
    assert here == pytest.approx(elsewhere, abs=0.0, rel=1e-12)


@pytest.mark.parametrize("factor", [0.5, 2.0, 4.0])
def test_the_tilt_scales_with_power_length_and_separation(factor: float) -> None:
    """All three enter as one product, so scaling any of them does the same thing.

    Checked in the small-tilt limit, where ``exp(-x) ~ 1 - x`` makes the tilt in
    dB proportional rather than merely monotone.
    """
    frequencies, powers = comb(4, power=1e-6)
    base = tilt_db(raman_tilt(frequencies, powers, gain_slope=C_R, effective_length=L_EFF))

    stretched = [ANCHOR + index * SPACING * factor for index in range(4)]
    assert tilt_db(
        raman_tilt(
            frequencies, [p * factor for p in powers], gain_slope=C_R, effective_length=L_EFF
        )
    ) == pytest.approx(base * factor, rel=1e-3)
    assert tilt_db(
        raman_tilt(frequencies, powers, gain_slope=C_R, effective_length=L_EFF * factor)
    ) == pytest.approx(base * factor, rel=1e-3)
    assert tilt_db(
        raman_tilt(stretched, powers, gain_slope=C_R, effective_length=L_EFF)
    ) == pytest.approx(base * factor, rel=1e-3)


def test_a_filled_c_band_tilts_by_most_of_a_decibel_in_one_span() -> None:
    """The scale anchor, against the number the literature reports.

    Eighty channels at 0 dBm on a 50 GHz grid — 19 dBm of total power over four
    terahertz — through one 80 km span of standard fibre. Half to one decibel per
    span is what is measured on real systems, and it is why a C-band line system
    carries a gain-flattening design at all.
    """
    frequencies, powers = comb(80, power=dbm_to_w(0.0), spacing=50e9)
    ratios = raman_tilt(frequencies, powers, gain_slope=C_R, effective_length=L_EFF)
    assert tilt_db(ratios) == pytest.approx(0.81, abs=0.05)
    assert frequencies[-1] - frequencies[0] < RAMAN_TRIANGLE_LIMIT, (
        "the C band alone stays inside the range where the linear gain holds"
    )


def test_nothing_happens_without_something_to_happen_to() -> None:
    """Zero slope, one channel, no power: all of them the identity, exactly."""
    frequencies, powers = comb(6)
    assert raman_tilt(frequencies, powers, gain_slope=0.0, effective_length=L_EFF) == [1.0] * 6
    assert raman_tilt([ANCHOR], [1e-3], gain_slope=C_R, effective_length=L_EFF) == [1.0]
    assert raman_tilt(frequencies, [0.0] * 6, gain_slope=C_R, effective_length=L_EFF) == [1.0] * 6


def test_a_power_and_a_frequency_for_every_channel() -> None:
    with pytest.raises(ValueError, match="one power per frequency"):
        raman_tilt([ANCHOR, ANCHOR + SPACING], [1e-3], gain_slope=C_R, effective_length=L_EFF)


# ---------------------------------------------------------------------------
# through the block


def carrier(power: float, index: int, spacing: float = SPACING) -> Band:
    return Band(
        Ex=np.full(SAMPLES, np.sqrt(power), dtype=np.complex128),
        Ey=np.zeros(SAMPLES, dtype=np.complex128),
        f0=ANCHOR + index * spacing,
        fs=FS,
    )


def propagate(channels: int, *, slope: float, power: float = 1e-3, **settings: float) -> dict:
    settings.setdefault("nonlinearity", 0.0)
    span = Fiber(
        length=80.0,
        attenuation=0.2,
        dispersion=0.0,
        raman_gain_slope=slope,
        label="fib",
        **settings,
    )
    bands = tuple(carrier(power, index) for index in range(channels))
    return span.run(CTX, {"in": OpticalSignal(bands=bands, noise=())})


def test_the_block_tilts_the_comb_and_says_by_how_much() -> None:
    """And the diagnostic is the tilt, not a flag saying it happened."""
    out = propagate(8, slope=0.028)
    flat = propagate(8, slope=0.0)

    tilted = [band.average_power() for band in out["out"].bands]
    untilted = [band.average_power() for band in flat["out"].bands]

    assert tilted[0] > untilted[0], "the longest wavelength gained"
    assert tilted[-1] < untilted[-1], "the shortest paid for it"
    assert all(a == pytest.approx(untilted[0], rel=1e-12) for a in untilted), (
        "with the effect off the comb should still be flat"
    )

    measured = 10.0 * math.log10(tilted[0] / tilted[-1])
    assert out["diagnostics"].raman_tilt == pytest.approx(measured, rel=1e-9)
    assert flat["diagnostics"].raman_tilt == 0.0


def test_the_block_moves_power_rather_than_making_it() -> None:
    """Summed over the comb, a Raman span loses exactly its attenuation and no more."""
    out = propagate(8, slope=0.028)
    flat = propagate(8, slope=0.0)
    assert sum(b.average_power() for b in out["out"].bands) == pytest.approx(
        sum(b.average_power() for b in flat["out"].bands), rel=1e-12
    )


def test_a_single_channel_has_nobody_to_trade_with() -> None:
    """Self-Raman — a channel pumping its own long-wavelength edge — is not this model."""
    out = propagate(1, slope=0.028)
    flat = propagate(1, slope=0.0)
    assert out["out"].bands[0].average_power() == pytest.approx(
        flat["out"].bands[0].average_power(), rel=1e-12
    )
    assert out["diagnostics"].raman_tilt == 0.0


def test_the_effect_is_off_by_default() -> None:
    """Every WDM result in this project was taken without it."""
    span = Fiber(length=80.0, attenuation=0.2, dispersion=0.0, label="fib")
    assert span.raman_gain_slope == 0.0
    assert span.si("raman_gain_slope") == 0.0


def test_the_gain_slope_converts_the_way_its_unit_reads() -> None:
    """Two denominators grow at once, so the factor is 1e-15 and not 1e-3.

    Pinned because it is the sort of conversion that is wrong by twelve orders of
    magnitude and still produces a plausible-looking small number.
    """
    assert to_si(0.028, "1/W/km/THz") == pytest.approx(2.8e-17, abs=0.0, rel=1e-12)
    # And back: the round trip is the pair of tables agreeing with each other.
    assert from_si(C_R, "1/W/km/THz") == pytest.approx(0.028, abs=0.0, rel=1e-12)


def test_raman_needs_no_kerr_coefficient() -> None:
    """They are different nonlinearities and are switched independently.

    Worth stating because both are third-order effects of the same medium and it
    would be a natural mistake to gate one on the other's parameter.
    """
    without = propagate(8, slope=0.028, nonlinearity=0.0)
    with_kerr = propagate(8, slope=0.028, nonlinearity=1.3)
    assert without["diagnostics"].raman_tilt > 0.0
    assert with_kerr["diagnostics"].raman_tilt == pytest.approx(
        without["diagnostics"].raman_tilt, rel=1e-6
    )


def test_a_pump_sized_comb_does_not_underflow_the_exponentials() -> None:
    """Which is what centring the offsets is for, in the one case where it shows.

    The reference frequency cancels, so measuring the offsets from zero is
    algebraically the same answer. It is not the same computation: a comb sitting
    at 193 THz with ten watts in it puts the exponent past 1100, and
    ``exp(-1100)`` is zero — leaving the ratio of two zeros. Centred, the same
    case is ordinary arithmetic. Ten watts across four channels is a
    Raman-pumped span, not a contrived number.
    """
    frequencies, powers = comb(4, power=2.5)
    ratios = raman_tilt(frequencies, powers, gain_slope=C_R, effective_length=L_EFF)

    assert all(math.isfinite(r) and r > 0.0 for r in ratios), ratios
    assert sum(p * r for p, r in zip(powers, ratios, strict=True)) == pytest.approx(
        sum(powers), abs=0.0, rel=1e-12
    )
    # A transfer this strong is well outside the small-tilt regime; the point is
    # that it is computed at all, and monotonically.
    assert tilt_db(ratios) > 5.0
    assert list(ratios) == sorted(ratios, reverse=True)


def test_the_block_tilts_over_the_effective_length_not_the_span() -> None:
    """Where the pump is still bright, which over 80 km is a quarter of it.

    Pinned against the kernel called with ``effective_length`` explicitly,
    because the block is the only place that choice is made and using the span
    length instead would inflate the tilt by 3.8x while leaving every direction
    and conservation test happy.
    """
    out = propagate(8, slope=0.028)
    expected = raman_tilt(
        [ANCHOR + index * SPACING for index in range(8)],
        [1e-3] * 8,
        gain_slope=C_R,
        effective_length=effective_length(ALPHA, 80e3),
    )
    assert out["diagnostics"].raman_tilt == pytest.approx(tilt_db(expected), rel=1e-9)

    over_the_whole_span = raman_tilt(
        [ANCHOR + index * SPACING for index in range(8)],
        [1e-3] * 8,
        gain_slope=C_R,
        effective_length=80e3,
    )
    assert tilt_db(over_the_whole_span) / tilt_db(expected) == pytest.approx(3.78, rel=0.01)
