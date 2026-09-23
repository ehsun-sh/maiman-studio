"""Inter-channel cross-polarization modulation: a neighbour turns a channel's state.

A band ``s`` beside a band ``p`` at another frequency is driven, at its own
frequency, by terms like ``(s . p*) p`` -- quadratic in the neighbour, linear in
itself, and not oscillating at all. They are what makes a pump at 45 degrees
rotate a probe's state of polarization, and the coupled split-step used to drop
them as though they beat at the channel spacing.

Two things hold them in place. A CW pump's matrix is constant, so the probe's
Stokes vector precesses about a fixed axis by an angle that has a closed form;
and the one-band vector split-step, which carries every term of ``|E|^2 E``
without being told which are which.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from maiman.kernels import (
    _cross_polarization_coupling,
    dispersion_to_beta2,
    propagate_coupled_ssfm,
)

GAMMA = 1.3e-3  # 1/W/m
LENGTH = 10e3
PUMP = 20e-3
PROBE = 20e-9  # weak enough that its own SPM, which would turn it too, is below 1e-7
SPACING = 100e9
ROOT = 1 / math.sqrt(2)


def stokes(x: complex, y: complex) -> np.ndarray:
    """Normalised ``(S1, S2, S3)`` in the convention the rotation is checked in."""
    power = abs(x) ** 2 + abs(y) ** 2
    return (
        np.array([abs(x) ** 2 - abs(y) ** 2, 2 * (x * np.conj(y)).real, -2 * (x * np.conj(y)).imag])
        / power
    )


def precess(vector: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues: ``vector`` turned by ``angle`` about the unit ``axis``, right-handed."""
    axis = axis / np.linalg.norm(axis)
    return (
        vector * math.cos(angle)
        + np.cross(axis, vector) * math.sin(angle)
        + axis * np.dot(axis, vector) * (1 - math.cos(angle))
    )


def two_bands(
    pump: tuple[complex, complex],
    probe: tuple[complex, complex],
    *,
    coherent: bool,
    beta2: float = 0.0,
) -> tuple[complex, complex]:
    """Pump and probe as two bands, both axes of each: the probe's field after the span."""
    samples = 64
    fields = [
        np.full(samples, pump[0] * math.sqrt(PUMP), dtype=np.complex128),
        np.full(samples, pump[1] * math.sqrt(PUMP), dtype=np.complex128),
        np.full(samples, probe[0] * math.sqrt(PROBE), dtype=np.complex128),
        np.full(samples, probe[1] * math.sqrt(PROBE), dtype=np.complex128),
    ]
    out, _ = propagate_coupled_ssfm(
        fields,
        160e9,
        beta2=[beta2] * 4,
        walkoff=[0.0] * 4,
        gamma=GAMMA,
        polarization=[0, 1, 0, 1],
        pairs=[(0, 1), (2, 3)],
        coherent_polarization=coherent,
        alpha=0.0,
        distance=LENGTH,
        max_nonlinear_phase=1e-3,
    )
    return complex(out[2][0]), complex(out[3][0])


def pump_stokes(pump: tuple[complex, complex]) -> np.ndarray:
    return stokes(*pump)


# ---------------------------------------------------------------------------
# A CW pump: precession about a fixed axis, by a closed-form angle
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("pump", "probe"),
    [
        ((ROOT, ROOT), (1.0, 0.0)),
        ((math.cos(math.pi / 6), math.sin(math.pi / 6)), (ROOT, ROOT)),
        ((1.0, 0.0), (math.cos(0.4), 1j * math.sin(0.4))),
    ],
)
def test_a_linear_pump_turns_the_probe_about_its_own_stokes_vector(
    pump: tuple[complex, complex], probe: tuple[complex, complex]
) -> None:
    """Isotropic tensor: ``(4/3) gamma P L`` about the pump's Stokes vector.

    The traceless part of ``(2/3)[(p^H p) + p p^H + p* p^T]`` for a linear pump is
    ``(2/3) P`` along its own axis, which precesses a Stokes vector at twice that.
    """
    angle = (4.0 / 3.0) * GAMMA * PUMP * LENGTH
    expected = precess(stokes(*probe), pump_stokes(pump), angle)
    turned = stokes(*two_bands(pump, probe, coherent=True))
    assert turned == pytest.approx(expected, abs=1e-6)


def test_a_circular_pump_does_not_turn_a_probe_in_an_isotropic_fibre() -> None:
    """``p p^H + p* p^T = P`` for circular ``p``: the matrix is a phase, nothing precesses."""
    probe = (math.cos(0.4), math.sin(0.4))
    turned = stokes(*two_bands((ROOT, 1j * ROOT), probe, coherent=True))
    # The probe's own SPM, 2.6e-7 rad, is all that is left; at 45 degrees a
    # linear pump would have turned it by 0.35 rad.
    assert turned == pytest.approx(stokes(*probe), abs=1e-6)


def test_the_phase_only_form_turns_it_at_half_the_rate_off_the_axis() -> None:
    """Phase-only: ``(2/3) p_x p_y*`` alone off the diagonal, ``(2/3) gamma P L`` at 45 degrees.

    Its diagonal is the per-axis phase, equal on both axes for a 45-degree pump,
    so what turns the probe is the off-diagonal only, about S2.
    """
    angle = (2.0 / 3.0) * GAMMA * PUMP * LENGTH
    expected = precess(np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), angle)
    turned = stokes(*two_bands((ROOT, ROOT), (1.0, 0.0), coherent=False))
    assert turned == pytest.approx(expected, abs=1e-6)


def test_neighbours_on_an_axis_leave_the_step_as_it_was() -> None:
    """No correlated neighbour, no off-diagonal: the old per-axis step, untouched."""
    fields = [np.ones(8, dtype=np.complex128), np.zeros(8, dtype=np.complex128)] * 2
    assert _cross_polarization_coupling(fields, [(0, 1), (2, 3)], coherent=True) == [None, None]
    fields[1] = np.ones(8, dtype=np.complex128)
    coupling = _cross_polarization_coupling(fields, [(0, 1), (2, 3)], coherent=True)
    assert coupling[0] is None, "a band is not turned by itself here"
    assert coupling[1] is not None


# ---------------------------------------------------------------------------
# Against the one-band vector split-step
# ---------------------------------------------------------------------------


def one_band(
    pump: tuple[complex, complex], probe: tuple[complex, complex], *, coherent: bool, beta2: float
) -> tuple[complex, complex]:
    """Both tones in one band, every term of ``|E|^2 E``: the probe's field after the span."""
    n, fs = 2048, 1.6e12
    t = np.arange(n) / fs
    tone = np.exp(2j * np.pi * SPACING * t)
    x = pump[0] * math.sqrt(PUMP) + probe[0] * math.sqrt(PROBE) * tone
    y = pump[1] * math.sqrt(PUMP) + probe[1] * math.sqrt(PROBE) * tone
    (ax, ay), _ = propagate_coupled_ssfm(
        [x, y],
        fs,
        beta2=[beta2, beta2],
        walkoff=[0.0, 0.0],
        gamma=GAMMA,
        polarization=[0, 1],
        pairs=[(0, 1)],
        coherent_polarization=coherent,
        alpha=0.0,
        distance=LENGTH,
        max_nonlinear_phase=1e-3,
        max_step=2.0,
    )
    k = round(SPACING / (fs / n))
    return complex(np.fft.fft(ax)[k] / n), complex(np.fft.fft(ay)[k] / n)


@pytest.mark.parametrize("coherent", [True, False])
@pytest.mark.parametrize(
    ("pump", "probe"),
    [
        ((ROOT, ROOT), (1.0, 0.0)),
        ((ROOT, 1j * ROOT), (1.0, 0.0)),
        ((math.cos(0.3), 1j * math.sin(0.3)), (math.cos(0.5), math.sin(0.5))),
    ],
)
def test_the_probe_s_state_is_the_one_band_split_step_s(
    pump: tuple[complex, complex], probe: tuple[complex, complex], coherent: bool
) -> None:
    """Standard fibre's dispersion dephases the mixing between them: XPolM is what is compared.

    Measured within 7.5e-4 on every Stokes component. Before, the multi-band
    solver left the probe where it started: S3 = 0 where the reference turned it
    to -0.334.
    """
    beta2 = dispersion_to_beta2(17e-6, 1550e-9)
    reference = stokes(*one_band(pump, probe, coherent=coherent, beta2=beta2))
    solved = stokes(*two_bands(pump, probe, coherent=coherent, beta2=beta2))
    assert solved == pytest.approx(reference, abs=1.5e-3)
