"""Four-wave mixing between pumps in any state of polarization.

Taking each axis as its own scalar problem builds the x product from the pumps'
x components alone. That is right for light on an axis, and a quarter of the
answer for the same light at 45 degrees -- where an isotropic fibre mixes exactly
as it does on the axis. With ``cross_polarization`` on, the drive is a vector,
from the Kerr tensor the split-step is running.

What holds it in place is that split-step, with both tones and both axes in one
band, so every term of the vector ``|A|^2 A`` is there and there is no mixing
model at all. At a milliwatt the Kerr phase is small enough that the drive is the
only thing being compared.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from maiman.components import Fiber
from maiman.context import SimulationContext
from maiman.kernels import coherency, fwm_phase_mismatch, fwm_vector_drive, propagate_coupled_ssfm
from maiman.signals import Band, OpticalSignal

ANCHOR = 193.1e12
SPACING = 100e9
GAMMA = 1.3  # 1/W/km
LENGTH = 10e3
CTX = SimulationContext(
    bit_rate=10e9, samples_per_symbol=16, sequence_length=16, seed=3, precision="double"
)

STATES = {
    "x": (1.0, 0.0),
    "45 degrees": (1 / math.sqrt(2), 1 / math.sqrt(2)),
    "circular": (1 / math.sqrt(2), 1j / math.sqrt(2)),
    "elliptical": (math.cos(0.3), 1j * math.sin(0.3)),
}


def state(jones: tuple[complex, complex], power: float = 1.0) -> np.ndarray:
    field = np.sqrt(power) * np.array(jones, dtype=np.complex128)
    return np.outer(field, np.conj(field))


# ---------------------------------------------------------------------------
# The drive, against the tensor's closed forms
# ---------------------------------------------------------------------------


def test_on_an_axis_the_drive_is_the_scalar_product_of_powers() -> None:
    p_i, p_j, p_k = 3e-3, 2e-3, 0.5e-3
    for coherent in (True, False):
        drive = fwm_vector_drive(
            state((1, 0), p_i), state((1, 0), p_j), state((1, 0), p_k), coherent=coherent
        )
        assert drive[0, 0].real == pytest.approx(p_i * p_j * p_k, rel=1e-15)
        assert abs(drive[1, 1]) == 0.0 and abs(drive[0, 1]) == 0.0


@pytest.mark.parametrize(
    ("name", "coherent", "fraction"),
    [
        # Isotropic: a linear state mixes as it does on the axis, whichever way it points.
        ("45 degrees", True, 1.0),
        # A . A vanishes for circular light, and (2/3)^2 is what is left.
        ("circular", True, 4.0 / 9.0),
        # Phase-only drops the coherent term, which is not isotropic: (5/6)^2 at 45 degrees ...
        ("45 degrees", False, 25.0 / 36.0),
        # ... and the same for circular, since it cannot tell the axes' phase apart.
        ("circular", False, 25.0 / 36.0),
    ],
)
def test_the_drive_off_the_axis_is_the_tensor_s(name: str, coherent: bool, fraction: float) -> None:
    pump = state(STATES[name])
    drive = fwm_vector_drive(pump, pump, pump, coherent=coherent)
    assert float(np.trace(drive).real) == pytest.approx(fraction, rel=1e-12)


def test_the_coherency_of_a_band_is_its_jones_vector_squared() -> None:
    jones = STATES["elliptical"]
    ex = np.full(64, jones[0] * math.sqrt(2e-3), dtype=np.complex128)
    ey = np.full(64, jones[1] * math.sqrt(2e-3), dtype=np.complex128)
    assert coherency(ex, ey) == pytest.approx(state(jones, 2e-3), rel=1e-12)


# ---------------------------------------------------------------------------
# Through the fibre block, against the one-band vector split-step
# ---------------------------------------------------------------------------


def beta2() -> float:
    probe = Band(
        Ex=np.ones(256, dtype=np.complex128),
        Ey=np.zeros(256, dtype=np.complex128),
        f0=ANCHOR,
        fs=160e9,
    )
    return Fiber(dispersion=8.0).reference_beta2(OpticalSignal(bands=(probe,)))


def reference(
    jones: tuple[complex, complex], *, coherent: bool, pump: float, signal: float
) -> float:
    """Pump and signal in one band, both axes, every term: the product at ``pump - spacing`` [W]."""
    n, fs = 2048, 1.6e12
    t = np.arange(n) / fs
    tone = np.sqrt(pump) + np.sqrt(signal) * np.exp(2j * np.pi * SPACING * t)
    mismatch = abs(fwm_phase_mismatch(beta2(), 0.0, 0.0, SPACING))
    (ax, ay), _ = propagate_coupled_ssfm(
        [jones[0] * tone, jones[1] * tone],
        fs,
        beta2=[beta2()] * 2,
        walkoff=[0.0, 0.0],
        gamma=GAMMA * 1e-3,
        polarization=[0, 1],
        pairs=[(0, 1)],
        coherent_polarization=coherent,
        alpha=0.0,
        distance=LENGTH,
        max_nonlinear_phase=1e-3,
        max_step=0.05 / mismatch,
    )
    bin_ = -round(SPACING / (fs / n))
    return float(abs(np.fft.fft(ax)[bin_] / n) ** 2 + abs(np.fft.fft(ay)[bin_] / n) ** 2)


def modelled(
    jones: tuple[complex, complex], *, coherent: bool, pump: float, signal: float, **flags: bool
) -> float:
    """The same two tones as two bands, through the fibre block [W]."""
    bands = tuple(
        Band(
            Ex=np.full(256, jones[0] * np.sqrt(power), dtype=np.complex128),
            Ey=np.full(256, jones[1] * np.sqrt(power), dtype=np.complex128),
            f0=ANCHOR + index * SPACING,
            fs=160e9,
        )
        for index, power in enumerate((pump, signal))
    )
    out = Fiber(
        length=LENGTH / 1e3,
        attenuation=0.0,
        dispersion=8.0,
        nonlinearity=GAMMA,
        mixing_floor=250.0,
        cross_polarization=True,
        coherent_polarization=coherent,
        label="span",
        **flags,
    ).run(CTX, {"in": OpticalSignal(bands=bands)})["out"]
    product = next(b for b in out.bands if abs(b.f0 - (ANCHOR - SPACING)) < 1e3)
    return float(abs(np.mean(product.Ex)) ** 2 + abs(np.mean(product.Ey)) ** 2)


@pytest.mark.parametrize("coherent", [True, False])
@pytest.mark.parametrize("name", list(STATES))
def test_the_product_is_the_split_step_s_in_every_state(name: str, coherent: bool) -> None:
    """A milliwatt, so the Kerr phase is a hundredth of a radian and the drive is what differs.

    Measured 1.003 to 1.004 across all eight, the residual being that hundredth
    of a radian. Before the drive was a vector the 45-degree beam came out at
    0.28 and the circular one at 0.58.
    """
    ratio = modelled(STATES[name], coherent=coherent, pump=1e-3, signal=1e-6) / reference(
        STATES[name], coherent=coherent, pump=1e-3, signal=1e-6
    )
    assert ratio == pytest.approx(1.0, abs=0.01)


def test_the_product_keeps_the_pumps_handedness() -> None:
    """The two axes of a product are one wave: circular pumps make a circular product."""
    jones = STATES["circular"]
    bands = tuple(
        Band(
            Ex=np.full(256, jones[0] * np.sqrt(power), dtype=np.complex128),
            Ey=np.full(256, jones[1] * np.sqrt(power), dtype=np.complex128),
            f0=ANCHOR + index * SPACING,
            fs=160e9,
        )
        for index, power in enumerate((1e-3, 1e-6))
    )
    out = Fiber(
        length=10.0,
        attenuation=0.0,
        dispersion=8.0,
        nonlinearity=GAMMA,
        mixing_floor=250.0,
        cross_polarization=True,
        coherent_polarization=True,
        label="span",
    ).run(CTX, {"in": OpticalSignal(bands=bands)})["out"]
    product = next(b for b in out.bands if abs(b.f0 - (ANCHOR - SPACING)) < 1e3)
    x, y = complex(np.mean(product.Ex)), complex(np.mean(product.Ey))
    assert abs(x) == pytest.approx(abs(y), rel=1e-9)
    assert np.angle(y / x) == pytest.approx(math.pi / 2, abs=1e-9)


def test_without_cross_polarization_each_axis_stays_its_own_scalar_problem() -> None:
    """The declared default: two independent problems, and a 45-degree beam a quarter as strong.

    That is not what a fibre does, and it is what the flag is for; the default
    moves nothing that was measured with it.
    """
    jones = STATES["45 degrees"]
    bands = tuple(
        Band(
            Ex=np.full(256, jones[0] * np.sqrt(power), dtype=np.complex128),
            Ey=np.full(256, jones[1] * np.sqrt(power), dtype=np.complex128),
            f0=ANCHOR + index * SPACING,
            fs=160e9,
        )
        for index, power in enumerate((1e-3, 1e-6))
    )
    fibre = Fiber(
        length=10.0, attenuation=0.0, dispersion=8.0, nonlinearity=GAMMA, mixing_floor=250.0
    )
    out = fibre.run(CTX, {"in": OpticalSignal(bands=bands)})["out"]
    product = next(b for b in out.bands if abs(b.f0 - (ANCHOR - SPACING)) < 1e3)
    diagonal = float(abs(np.mean(product.Ex)) ** 2 + abs(np.mean(product.Ey)) ** 2)
    axial_bands = tuple(
        Band(
            Ex=np.full(256, np.sqrt(power), dtype=np.complex128),
            Ey=np.zeros(256, dtype=np.complex128),
            f0=ANCHOR + index * SPACING,
            fs=160e9,
        )
        for index, power in enumerate((1e-3, 1e-6))
    )
    out = fibre.run(CTX, {"in": OpticalSignal(bands=axial_bands)})["out"]
    product = next(b for b in out.bands if abs(b.f0 - (ANCHOR - SPACING)) < 1e3)
    on_axis = float(abs(np.mean(product.Ex)) ** 2)
    assert diagonal / on_axis == pytest.approx(0.25, rel=0.01)


@pytest.mark.parametrize(
    ("name", "measured"),
    [("x", 1.001), ("45 degrees", 1.106), ("circular", 1.030), ("elliptical", 1.004)],
)
def test_what_the_pumps_phase_still_leaves_off_the_axis(name: str, measured: float) -> None:
    """Pinned, not tuned: at 20 mW the Kerr phase is taken per axis, as a phase only.

    On an axis that is exact and the product lands within 0.1 %; off it, the
    coherent term turns a state's own phase at a rate the per-axis form does not
    see, and the product is 10.6 % high at 45 degrees. That is the part of the
    vector problem still open (maiman-l43), and this is where it stands: a change
    that moves these numbers should say why.
    """
    ratio = modelled(
        STATES[name], coherent=True, pump=20e-3, signal=20e-6, pump_phase=True
    ) / reference(STATES[name], coherent=True, pump=20e-3, signal=20e-6)
    assert ratio == pytest.approx(measured, abs=0.003)
