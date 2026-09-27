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

from maiman.components import EDFA, Fiber
from maiman.context import SimulationContext
from maiman.kernels import (
    attenuation_db_per_m_to_alpha,
    coherency,
    fwm_drive_phase,
    fwm_phase_mismatch,
    fwm_vector_drive,
    geometric_phase,
    principal_state,
    propagate_coupled_ssfm,
)
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
# The phase a state carries, in the gauge the history keeps
# ---------------------------------------------------------------------------


def ellipse(a: float, delta: float) -> np.ndarray:
    return np.array([math.cos(a), math.sin(a) * np.exp(1j * delta)])


def test_a_principal_state_is_real_on_x_and_on_y_only_with_nothing_on_x() -> None:
    for jones in (*STATES.values(), (0.6, -0.8j), (-0.3, 0.2 + 0.9j)):
        vector = principal_state(state(np.exp(1.1j) * np.array(jones)))
        assert vector[0].imag == pytest.approx(0.0, abs=1e-15) and vector[0].real > 0.0
    on_y = principal_state(state((0.0, 1j)))
    assert on_y[1].imag == pytest.approx(0.0, abs=1e-15) and on_y[1].real > 0.0


def test_the_geometric_phase_is_pancharatnam_s() -> None:
    """``sin^2(a) d(delta)`` for ``u = (cos a, sin a e^(i delta))``, and nothing if it stays put.

    Over a finite move the shortest path gives ``arg(cos^2 a + sin^2 a e^(i Delta))``;
    a small one reduces to the differential form.
    """
    for a in (0.0, 0.3, math.pi / 4, 1.2):
        step = 1e-4
        assert geometric_phase(ellipse(a, 0.4), ellipse(a, 0.4 + step)) == pytest.approx(
            math.sin(a) ** 2 * step, rel=1e-4, abs=1e-15
        )
        big = 0.8
        assert geometric_phase(ellipse(a, 0.4), ellipse(a, 0.4 + big)) == pytest.approx(
            float(np.angle(math.cos(a) ** 2 + math.sin(a) ** 2 * np.exp(1j * big))), abs=1e-14
        )
        assert geometric_phase(ellipse(a, 0.4), ellipse(a, 0.4)) == 0.0
    # Turning the ellipse's major axis moves ``a`` alone, and adds nothing.
    assert geometric_phase(ellipse(0.3, 0.0), ellipse(0.5, 0.0)) == 0.0


@pytest.mark.parametrize("coherent", [True, False])
def test_the_drive_s_phase_is_the_tensor_s_on_jones_vectors(coherent: bool) -> None:
    """Against the tensor written out: ``[(u_i.u_k*) u_j + (u_j.u_k*) u_i + (u_i.u_j) u_k*] / 3``.

    Phase-only drops the entries where both pumps sit on the other axis. For
    three pumps in one linear or circular state the phase on x is zero, and for
    one ellipse it is not -- the ``(u . u) u*`` term carries its handedness --
    but a state that stays put keeps it constant, which is why only light whose
    states move ever saw this term go missing.
    """
    u_i, u_j, u_k = ellipse(0.3, 1.0), ellipse(1.1, -0.4), ellipse(0.7, 2.2)

    def written(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> complex:
        drive = (np.dot(p, np.conj(r)) * q + np.dot(q, np.conj(r)) * p) / 3.0
        drive = drive + np.dot(p, q) * np.conj(r) / 3.0
        if not coherent:
            drive[0] = drive[0] - p[1] * q[1] * np.conj(r[0]) / 3.0
        return complex(drive[0])

    assert fwm_drive_phase(u_i, u_j, u_k, coherent=coherent) == pytest.approx(
        float(np.angle(written(u_i, u_j, u_k))), abs=1e-14
    )
    for jones in ((1.0, 0.0), STATES["45 degrees"], (0.6, -0.8), STATES["circular"]):
        u = principal_state(state(jones))
        assert fwm_drive_phase(u, u, u, coherent=coherent) == pytest.approx(0.0, abs=1e-15)
    # An ellipse on the axes has ``u . u`` real and no phase either; once it has
    # turned off them -- as the Kerr effect turns it -- it has one.
    upright = principal_state(state(STATES["elliptical"]))
    assert fwm_drive_phase(upright, upright, upright, coherent=True) == pytest.approx(
        0.0, abs=1e-15
    )
    tilted = ellipse(0.3, 1.0)
    assert abs(fwm_drive_phase(tilted, tilted, tilted, coherent=True)) > 1e-2


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


@pytest.mark.parametrize("coherent", [True, False])
@pytest.mark.parametrize("name", list(STATES))
def test_with_the_pumps_phase_the_product_is_the_split_step_s_in_every_state(
    name: str, coherent: bool
) -> None:
    """20 mW, a quarter of a radian: the Kerr phase taken in each carrier's own state.

    Measured 0.997 to 1.001 across all eight. Taken per axis, as it was, the
    45-degree beam came out 10.6 % high with the coherent term and 6.5 % without:
    a linear state is turned by all of its power, where each axis alone is turned
    by five sixths of it.
    """
    ratio = modelled(
        STATES[name], coherent=coherent, pump=20e-3, signal=20e-6, pump_phase=True
    ) / reference(STATES[name], coherent=coherent, pump=20e-3, signal=20e-6)
    assert ratio == pytest.approx(1.0, abs=0.005)


# ---------------------------------------------------------------------------
# Four amplified spans: the history the signal carries between them
# ---------------------------------------------------------------------------


def reference_link(jones: tuple[complex, complex], dispersion: float, *, coherent: bool) -> float:
    """Four 80 km spans and 16 dB amplifiers, every tone in one band [W]."""
    n, fs = 2048, 1.6e12
    t = np.arange(n) / fs
    alpha = attenuation_db_per_m_to_alpha(0.2e-3)
    b2 = Fiber(dispersion=dispersion).reference_beta2(
        OpticalSignal(
            bands=(
                Band(
                    Ex=np.ones(256, dtype=np.complex128),
                    Ey=np.zeros(256, dtype=np.complex128),
                    f0=ANCHOR,
                    fs=160e9,
                ),
            )
        )
    )
    tone = np.sqrt(20e-3) + np.sqrt(20e-6) * np.exp(2j * np.pi * SPACING * t)
    fields = [jones[0] * tone, jones[1] * tone]
    mismatch = abs(fwm_phase_mismatch(b2, 0.0, 0.0, SPACING))
    for _ in range(4):
        fields, _ = propagate_coupled_ssfm(
            fields,
            fs,
            beta2=[b2, b2],
            walkoff=[0.0, 0.0],
            gamma=GAMMA * 1e-3,
            polarization=[0, 1],
            pairs=[(0, 1)],
            coherent_polarization=coherent,
            alpha=alpha,
            distance=80e3,
            max_nonlinear_phase=1e-3,
            max_step=0.05 / mismatch,
        )
        fields = [field * math.exp(alpha * 80e3 / 2.0) for field in fields]
    bin_ = -round(SPACING / (fs / n))
    return float(sum(abs(np.fft.fft(field)[bin_] / n) ** 2 for field in fields))


def modelled_link(
    jones: tuple[complex, complex], dispersion: float, *, coherent: bool, pieces: int = 1
) -> float:
    """The same link through the fibre block, as two bands, each span in ``pieces`` [W]."""
    signal = OpticalSignal(
        bands=tuple(
            Band(
                Ex=np.full(256, jones[0] * np.sqrt(power), dtype=np.complex128),
                Ey=np.full(256, jones[1] * np.sqrt(power), dtype=np.complex128),
                f0=ANCHOR + index * SPACING,
                fs=160e9,
            )
            for index, power in enumerate((20e-3, 20e-6))
        )
    )
    for index in range(4):
        for piece in range(pieces):
            signal = Fiber(
                length=80.0 / pieces,
                attenuation=0.2,
                dispersion=dispersion,
                nonlinearity=GAMMA,
                mixing_floor=250.0,
                pump_phase=True,
                cross_polarization=True,
                coherent_polarization=coherent,
                label=f"span{index}" if pieces == 1 else f"span{index}.{piece}",
            ).run(CTX, {"in": signal})["out"]
        signal = EDFA(gain=16.0, noise_figure=0.0, label=f"amp{index}").run(CTX, {"in": signal})[
            "out"
        ]
    product = next(b for b in signal.bands if abs(b.f0 - (ANCHOR - SPACING)) < 1e3)
    return float(np.mean(np.abs(product.Ex) ** 2) + np.mean(np.abs(product.Ey) ** 2))


#: The states whose two axes carry equal power, where a product's phase used to
#: be taken against whichever axis came out a bit stronger (maiman-sew).
EQUAL_AXES = {
    "45 degrees": STATES["45 degrees"],
    "-45 degrees": (1 / math.sqrt(2), -1 / math.sqrt(2)),
    "circular": STATES["circular"],
}


@pytest.mark.parametrize("coherent", [True, False])
@pytest.mark.parametrize("dispersion", [4.0, -4.0])
@pytest.mark.parametrize("name", ["x", *EQUAL_AXES])
def test_four_spans_add_as_the_split_step_adds_them_in_any_fixed_state(
    name: str, dispersion: float, coherent: bool
) -> None:
    """Diagonal, anti-diagonal and circular light land where light on the axis does.

    1.003--1.040 at D = +4 and 1.010--1.011 at D = -4, within the 5 % the scalar
    four-span test holds. The history carries each carrier's angle in its own
    state and the weak carrier's as a matrix, because the angle a product's
    carrier has reached depends on the state the product lands in. Taken per
    axis, the 45-degree link was off by up to 2.6 times.

    And the product's phase is taken against the same axis every span. These
    are the states whose two axes are equal, where it used to be taken against
    the stronger one -- which a rounding error chose, and chose differently from
    span to span. A circular product then changed its phase by a quarter turn
    between spans and an anti-diagonal one by a half: 1.7 and 6.4 times the
    split-step circular, 0.12 and 3.0 at -45 degrees.
    """
    jones = STATES.get(name) or EQUAL_AXES[name]
    ratio = modelled_link(jones, dispersion, coherent=coherent) / reference_link(
        jones, dispersion, coherent=coherent
    )
    assert ratio == pytest.approx(1.0, abs=0.05)


def test_a_turning_ellipse_converges_on_the_split_step_as_the_span_is_cut() -> None:
    """Cut each span into eight and elliptical light lands where light on the axis does.

    Light on an axis, on a diagonal or circular keeps its state in a Kerr fibre;
    elliptical light does not -- its ellipse rotates with its own power (Maker,
    Terhune and Savage, Phys. Rev. Lett. 12, 507 (1964)), here 5.95 degrees a
    span. A state that moves adds a geometric phase in the gauge the history
    keeps, and moves the tensor's own phase on x; without either, cutting a span
    into pieces *changed* the answer -- 0.94, then 0.89, 0.89, 0.83 at 2, 4 and
    8 pieces. With both it converges: 1.09, 1.06, 1.04, 1.04, beside the 1.040
    of light on the axis. Cutting a span must not change what it does.
    """
    reference = reference_link(STATES["elliptical"], 4.0, coherent=True)
    fine = modelled_link(STATES["elliptical"], 4.0, coherent=True, pieces=8) / reference
    axial = modelled_link(STATES["x"], 4.0, coherent=True) / reference_link(
        STATES["x"], 4.0, coherent=True
    )
    assert fine == pytest.approx(axial, abs=0.01)


@pytest.mark.parametrize(
    ("dispersion", "coherent", "measured"),
    [(4.0, True, 1.090), (-4.0, True, 1.002), (4.0, False, 1.030), (-4.0, False, 1.009)],
)
def test_what_elliptical_light_still_gets_wrong_in_one_long_span(
    dispersion: float, coherent: bool, measured: float
) -> None:
    """Pinned, not tuned (maiman-me9): the state is still held fixed along a span.

    Between spans the turning ellipse is followed now, and a span cut finely
    lands on the split-step (the test above). Taken whole, what is left is
    inside the span: the mixing integral holds each pump's state from the span's
    start, and the product made there is added in the drive's orientation at
    its end. 9 % over four 80 km spans at D = +4 isotropic, within 3 %
    otherwise; it was 6 % low and 9 % high before the geometric phase was kept.
    A change that moves these numbers should say why.
    """
    ratio = modelled_link(STATES["elliptical"], dispersion, coherent=coherent) / reference_link(
        STATES["elliptical"], dispersion, coherent=coherent
    )
    assert ratio == pytest.approx(measured, rel=0.02)
