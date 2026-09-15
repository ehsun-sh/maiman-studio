"""Coupling onto a chip: two Gaussian beams, two facets, and a grating's phase matching.

The edge coupler's overlap is a closed form, so it is checked against the
integral it closes -- a source beam propagated across the gap by FFT, shifted,
tilted and overlapped numerically -- and against the textbook limits. The
Gaussian a fibre's mode is taken to be is checked against the mode itself, from
the solver. The grating coupler's centre is phase matching, checked against its
own finite difference; its passband is a compact model, checked for being the
model it says it is.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from maiman.components import EdgeCoupler, GratingCoupler
from maiman.context import SimulationContext
from maiman.modes import StepIndexFibre, core_modes
from maiman.photonics import (
    fresnel_reflectance,
    gaussian_overlap,
    grating_coupler_centre,
    marcuse_mode_field_radius,
)
from maiman.signals import Band, OpticalSignal
from maiman.units import C_LIGHT

WAVELENGTH = 1.55e-6


def db(fraction: float) -> float:
    return 10.0 * math.log10(fraction)


# ---------------------------------------------------------------------------
# The overlap
# ---------------------------------------------------------------------------


def brute_force(
    source: float, target: float, *, offset: float, tilt: float, gap: float, index: float
) -> float:
    """The overlap integral done numerically: FFT across the gap, then sum."""
    x = np.linspace(-200e-6, 200e-6, 2**17)
    dx = x[1] - x[0]
    k = 2.0 * np.pi * index / WAVELENGTH
    kx = 2.0 * np.pi * np.fft.fftfreq(x.size, dx)
    launched = np.exp(-(x**2) / source**2).astype(complex)
    arrived = np.fft.ifft(np.fft.fft(launched) * np.exp(1j * kx**2 * gap / (2.0 * k)))
    shifted = np.interp(x - offset, x, arrived.real) + 1j * np.interp(x - offset, x, arrived.imag)
    field = shifted * np.exp(-1j * k * math.sin(tilt) * x)
    mode = np.exp(-(x**2) / target**2)
    coupled = abs(np.sum(field * mode) * dx) ** 2
    return float(coupled / (np.sum(abs(field) ** 2) * dx * np.sum(mode**2) * dx))


@pytest.mark.parametrize(
    ("source", "target", "offset", "tilt_deg", "gap", "index"),
    [
        (5.2e-6, 1.5e-6, 0.0, 0.0, 0.0, 1.0),
        (5.2e-6, 1.5e-6, 0.8e-6, 0.0, 0.0, 1.0),
        (5.2e-6, 3.0e-6, 0.5e-6, 2.0, 20e-6, 1.0),
        (5.2e-6, 5.2e-6, 1.0e-6, 1.0, 50e-6, 1.45),
        (3.0e-6, 6.0e-6, -2.0e-6, -3.0, 10e-6, 1.0),
    ],
)
def test_the_closed_form_is_the_integral_it_closes(
    source: float, target: float, offset: float, tilt_deg: float, gap: float, index: float
) -> None:
    tilt = math.radians(tilt_deg)
    closed = float(
        gaussian_overlap(
            source, target, wavelength=WAVELENGTH, offset=offset, tilt=tilt, gap=gap, index=index
        )
    )
    numeric = brute_force(source, target, offset=offset, tilt=tilt, gap=gap, index=index)
    assert closed == pytest.approx(numeric, rel=0.0, abs=2e-7)


def test_the_textbook_limits_hold_exactly() -> None:
    w, d, tilt = 5.2e-6, 2e-6, math.radians(2.0)
    assert float(gaussian_overlap(w, w, wavelength=WAVELENGTH, offset=d)) == pytest.approx(
        math.exp(-(d**2) / w**2), rel=1e-13, abs=0.0
    )
    assert float(gaussian_overlap(w, 1.5e-6, wavelength=WAVELENGTH)) == pytest.approx(
        2 * w * 1.5e-6 / (w**2 + 1.5e-6**2), rel=1e-13, abs=0.0
    )
    assert float(gaussian_overlap(w, w, wavelength=WAVELENGTH, tilt=tilt)) == pytest.approx(
        math.exp(-((math.pi * w * math.sin(tilt) / WAVELENGTH) ** 2)), rel=1e-13, abs=0.0
    )


def test_marcuse_is_the_best_gaussian_to_a_percent() -> None:
    """Against the LP01 field the mode solver computes, which is what it approximates."""
    fibre = StepIndexFibre()
    (mode,) = core_modes(fibre, WAVELENGTH)
    r = np.linspace(1e-9, 60e-6, 200_001)
    dr = r[1] - r[0]
    field = mode.field(r)

    def overlap(radius: float) -> float:
        gaussian = np.exp(-(r**2) / radius**2)
        shared = (np.sum(field * gaussian * r) * dr) ** 2
        return float(shared / (np.sum(field**2 * r) * dr * np.sum(gaussian**2 * r) * dr))

    trial = np.linspace(4.8e-6, 5.4e-6, 601)
    best = float(trial[int(np.argmax([overlap(w) for w in trial]))])
    marcuse = marcuse_mode_field_radius(fibre, WAVELENGTH)
    assert marcuse / best - 1.0 == pytest.approx(0.0071, abs=0.001)
    assert overlap(best) == pytest.approx(0.9924, abs=5e-4)
    assert overlap(marcuse) > 0.99


# ---------------------------------------------------------------------------
# The edge coupler
# ---------------------------------------------------------------------------


def test_a_standard_fibre_onto_a_three_micron_mode() -> None:
    """5.47 dB of mismatch and 0.31 dB of facets: the budget line a PIC starts from."""
    block = EdgeCoupler()
    assert db(block.mode_overlap()) == pytest.approx(-5.472, abs=0.002)
    assert db(1.0 - block.facet_loss()) == pytest.approx(-0.308, abs=0.002)
    assert db(block.coupled_fraction()) == pytest.approx(-5.780, abs=0.002)


def test_matched_modes_cost_only_their_misalignment() -> None:
    def overlap_db(**settings: Any) -> float:
        return db(EdgeCoupler(chip_mfd_x=10.4, chip_mfd_y=10.4, **settings).mode_overlap())

    assert overlap_db() == pytest.approx(0.0, abs=1e-12)
    assert overlap_db(offset_x=1.0) == pytest.approx(-0.161, abs=0.002)
    assert overlap_db(offset_y=2.0) == pytest.approx(-0.642, abs=0.002)
    assert overlap_db(gap=10.0) == pytest.approx(-0.036, abs=0.002)
    assert overlap_db(gap=50.0) == pytest.approx(-0.821, abs=0.002)
    assert overlap_db(gap=10.0, gap_index=1.45) == pytest.approx(-0.017, abs=0.002), (
        "a denser gap diffracts less"
    )


def test_facets_reflect_what_fresnel_says() -> None:
    assert fresnel_reflectance(3.48, 1.0) == pytest.approx((2.48 / 4.48) ** 2, rel=1e-15)
    assert fresnel_reflectance(3.48, 1.0) == pytest.approx(0.30644, abs=1e-5)
    assert EdgeCoupler(gap_index=1.4682, mode_index=1.4682).facet_loss() == 0.0
    assert db(1.0 - EdgeCoupler(mode_index=3.48).facet_loss()) == pytest.approx(-1.748, abs=0.002)


def test_the_joint_makes_no_light() -> None:
    block = EdgeCoupler(offset_x=1.5, tilt=2.0, gap=30.0, mode_index=3.48)
    grid = C_LIGHT / np.linspace(1.5e-6, 1.6e-6, 11)
    s = block.scattering_matrix(grid).s
    for port in (0, 1):
        assert np.all(np.abs(s[:, 1 - port, port]) ** 2 + np.abs(s[:, port, port]) ** 2 <= 1.0)


def test_the_edge_coupler_treats_both_polarizations_alike() -> None:
    grid = np.array([C_LIGHT / WAVELENGTH])
    block = EdgeCoupler(offset_x=1.0)
    assert np.array_equal(
        block.scattering_matrix(grid).s, block.scattering_matrix(grid, polarization="tm").s
    )


# ---------------------------------------------------------------------------
# The grating coupler
# ---------------------------------------------------------------------------


def test_the_centre_is_where_phase_matching_puts_it() -> None:
    assert GratingCoupler().centre_wavelength() * 1e9 == pytest.approx(1549.808, abs=0.005)
    flat = grating_coupler_centre(
        period=611e-9,
        effective_index=2.71,
        group_index=2.71,
        reference_wavelength=1550e-9,
        angle=math.radians(10.0),
    )
    assert flat == pytest.approx(611e-9 * (2.71 - math.sin(math.radians(10.0))), rel=1e-12)


def test_dispersion_slows_the_angle_tuning() -> None:
    """7.0 nm per degree with the grating's dispersion, 10.5 without it."""
    block = GratingCoupler()
    per_degree = block.angle_tuning() * math.pi / 180.0
    step = 0.01
    numeric = (
        GratingCoupler(fibre_angle=10.0 + step).centre_wavelength()
        - GratingCoupler(fibre_angle=10.0 - step).centre_wavelength()
    ) / (2 * step)
    assert per_degree == pytest.approx(numeric, rel=1e-4)
    assert per_degree * 1e9 == pytest.approx(-6.962, abs=0.005)
    flat = GratingCoupler(group_index=2.71).angle_tuning() * math.pi / 180.0
    assert flat * 1e9 == pytest.approx(-10.502, abs=0.005)


def test_the_passband_is_the_compact_model_it_declares() -> None:
    block = GratingCoupler()
    centre = block.centre_wavelength()
    grid = C_LIGHT / np.array([centre, centre + 17.5e-9, centre - 17.5e-9])
    te = [db(abs(t) ** 2) for t in block.scattering_matrix(grid).s[:, 1, 0]]
    tm = [db(abs(t) ** 2) for t in block.scattering_matrix(grid, polarization="tm").s[:, 1, 0]]
    assert te == pytest.approx([-3.0, -4.0, -4.0], abs=1e-9)
    assert tm == pytest.approx([-28.0, -29.0, -29.0], abs=1e-9)


@pytest.mark.parametrize(
    ("rotation", "te_share"), [(0.0, 1.0), (90.0, 0.0), (45.0, 0.5), (-30.0, 0.75)]
)
def test_the_die_s_rotation_decides_how_much_of_a_signal_is_te(
    rotation: float, te_share: float
) -> None:
    """An x-polarized carrier: ``cos^2`` of the rotation lands on TE, the rest on TM."""
    block = GratingCoupler(rotation=rotation)
    ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=8, sequence_length=16)
    carrier = Band(
        Ex=np.full(ctx.num_samples, 1e-3**0.5, dtype=ctx.complex_dtype),
        Ey=np.zeros(ctx.num_samples, dtype=ctx.complex_dtype),
        f0=C_LIGHT / block.centre_wavelength(),
        fs=ctx.sample_rate,
    )
    (band,) = block.run(ctx, {"in": OpticalSignal(bands=(carrier,))})["out"].bands
    te = float(np.mean(np.abs(band.Ex) ** 2))
    tm = float(np.mean(np.abs(band.Ey) ** 2))
    assert te == pytest.approx(1e-3 * te_share * 10 ** (-0.3), rel=1e-5, abs=1e-15)
    assert tm == pytest.approx(1e-3 * (1.0 - te_share) * 10 ** (-2.8), rel=1e-5, abs=1e-15)
