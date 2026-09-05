"""The dispersion slope, and the third-order term that comes with it.

A fibre's dispersion is not one number. D changes across the band, and the rate
it changes at is the slope — which shows up twice, in two places that look
unrelated until you notice they are the same coefficient. Across a comb it means
channels do not share a dispersion, so one compensator setting cannot serve all
of them. Within one channel it means a cubic phase, which broadens a pulse
*asymmetrically* where beta2 broadens it evenly.

Both are tested here, and the asymmetry is tested by its sign: a cubic term with
the wrong sign produces exactly the right width and exactly the wrong skew, so a
test that measures only broadening would pass through it.
"""

from __future__ import annotations

import numpy as np
import pytest

from maiman import Graph, SimulationContext
from maiman.component import Component
from maiman.components import (
    CarrierRecovery,
    CoherentReceiver,
    ConstellationAnalyzer,
    CWLaser,
    DispersionCompensator,
    Fiber,
    IQDriver,
    IQModulator,
    IQSampler,
    PowerMeter,
    PRBSGenerator,
    QAMMapper,
)
from maiman.dsp import compensate_dispersion
from maiman.kernels import (
    dispersion_slope_to_beta3,
    dispersion_to_beta2,
    propagate_coupled_ssfm,
    propagate_dispersion,
)
from maiman.signals import Band, OpticalSignal
from maiman.units import C_LIGHT

WAVELENGTH = 1550e-9
#: 17 ps/nm/km in s/m².
DISPERSION = 17e-6
#: 0.058 ps/nm²/km in s/m³ — the slope of standard fibre *at 1550 nm*, which is
#: not the 0.09 a datasheet quotes at the zero-dispersion wavelength.
SLOPE = 0.058e3


def moments(times: np.ndarray, power: np.ndarray) -> tuple[float, float]:
    """RMS width and skewness of an intensity profile."""
    total = power.sum()
    mean = float((times * power).sum() / total)
    variance = float(((times - mean) ** 2 * power).sum() / total)
    skew = float(((times - mean) ** 3 * power).sum() / total) / variance**1.5
    return float(np.sqrt(variance)), skew


def gaussian(width: float, num_samples: int, sample_rate: float) -> tuple[np.ndarray, np.ndarray]:
    """``exp(-T^2 / 2 T0^2)`` and the time axis it lives on."""
    times = (np.arange(num_samples) - num_samples // 2) / sample_rate
    return times, np.exp(-(times**2) / (2.0 * width**2)).astype(complex)


# ---------------------------------------------------------------------------
# the conversion


def test_beta3_inverts_the_relation_that_defines_the_slope() -> None:
    """``S = (2*pi*c/lambda^2)^2 * beta3 + (4*pi*c/lambda^3) * beta2``, run backwards.

    The conversion is a solve, not a formula to be remembered, so it is checked
    against the equation it solves rather than against a table.
    """
    beta2 = dispersion_to_beta2(DISPERSION, WAVELENGTH)
    beta3 = dispersion_slope_to_beta3(DISPERSION, SLOPE, WAVELENGTH)

    recovered = (2.0 * np.pi * C_LIGHT / WAVELENGTH**2) ** 2 * beta3 + (
        4.0 * np.pi * C_LIGHT / WAVELENGTH**3
    ) * beta2
    assert recovered == pytest.approx(SLOPE, abs=0.0, rel=1e-12)


def test_standard_fibre_lands_on_the_published_numbers() -> None:
    """-21.7 ps²/km and 0.13 ps³/km at 1550 nm, from D = 17 and S = 0.058."""
    beta2 = dispersion_to_beta2(DISPERSION, WAVELENGTH) * 1e27  # ps²/km
    beta3 = dispersion_slope_to_beta3(DISPERSION, SLOPE, WAVELENGTH) * 1e39  # ps³/km
    assert beta2 == pytest.approx(-21.7, rel=0.01)
    assert beta3 == pytest.approx(0.13, rel=0.05)


def test_a_flat_dispersion_still_has_a_third_order_term() -> None:
    """Which is the trap the conversion exists to avoid.

    Zero slope does not mean zero beta3: holding D constant across wavelength is
    itself a statement about how beta2 varies, and the ``2*D/lambda`` term is
    what it costs. A conversion that took only the slope would return zero here
    and be wrong by a quarter of the real value everywhere else.
    """
    assert dispersion_slope_to_beta3(DISPERSION, 0.0, WAVELENGTH) * 1e39 == pytest.approx(
        0.036, rel=0.05
    )
    assert dispersion_slope_to_beta3(0.0, 0.0, WAVELENGTH) == 0.0


# ---------------------------------------------------------------------------
# what the cubic term does to a pulse


@pytest.mark.parametrize("distance_km", [10.0, 30.0, 60.0, 120.0])
def test_third_order_broadening_matches_the_closed_form(distance_km: float) -> None:
    """``(sigma/sigma0)^2 = 1 + beta3^2 z^2 / (4 T0^6)`` for an unchirped Gaussian.

    Derived by the moment method: a pure phase filter adds the variance of its
    own group delay to the pulse, and for ``tau = beta3 z omega^2 / 2`` over a
    Gaussian spectrum that variance is ``beta3^2 z^2 / (8 T0^4)``. The same
    method gives ``1 + (beta2 z / T0^2)^2`` for second order, which is the
    textbook result, so the machinery is checked before it is trusted.
    """
    width, num_samples, sample_rate = 1e-12, 1 << 16, 4e12
    beta3 = dispersion_slope_to_beta3(DISPERSION, SLOPE, WAVELENGTH)
    distance = distance_km * 1e3

    times, pulse = gaussian(width, num_samples, sample_rate)
    sigma0, _ = moments(times, np.abs(pulse) ** 2)
    out = propagate_dispersion(pulse, sample_rate, 0.0, distance, beta3)
    sigma, _ = moments(times, np.abs(out) ** 2)

    predicted = np.sqrt(1.0 + beta3**2 * distance**2 / (4.0 * width**6))
    assert sigma / sigma0 == pytest.approx(predicted, abs=0.0, rel=1e-4)


def test_second_order_broadening_matches_its_own_closed_form() -> None:
    """``(sigma/sigma0)^2 = 1 + (beta2 z / T0^2)^2``, so the moment method is not on trial."""
    width, num_samples, sample_rate = 1e-12, 1 << 16, 4e12
    beta2 = dispersion_to_beta2(DISPERSION, WAVELENGTH)
    distance = 3e3

    times, pulse = gaussian(width, num_samples, sample_rate)
    sigma0, _ = moments(times, np.abs(pulse) ** 2)
    out = propagate_dispersion(pulse, sample_rate, beta2, distance, 0.0)
    sigma, _ = moments(times, np.abs(out) ** 2)

    predicted = np.sqrt(1.0 + (beta2 * distance / width**2) ** 2)
    assert sigma / sigma0 == pytest.approx(predicted, abs=0.0, rel=1e-4)


def test_the_sign_of_beta3_shows_in_the_skew_and_nowhere_else() -> None:
    """The test the width cannot do.

    Broadening is even in beta3, so flipping its sign leaves every width
    identical to the last digit and reverses which side of the pulse the tail
    goes to. That is the whole reason the cubic term's sign was derived from the
    same expansion of ``beta(omega)`` as the group delay rather than written
    down: nothing about the amount of spreading could have caught it.
    """
    width, num_samples, sample_rate = 1e-12, 1 << 16, 4e12
    beta3 = dispersion_slope_to_beta3(DISPERSION, SLOPE, WAVELENGTH)
    times, pulse = gaussian(width, num_samples, sample_rate)

    forward = propagate_dispersion(pulse, sample_rate, 0.0, 30e3, beta3)
    reversed_ = propagate_dispersion(pulse, sample_rate, 0.0, 30e3, -beta3)
    sigma_f, skew_f = moments(times, np.abs(forward) ** 2)
    sigma_r, skew_r = moments(times, np.abs(reversed_) ** 2)

    assert sigma_f == pytest.approx(sigma_r, abs=0.0, rel=1e-9)
    assert skew_f == pytest.approx(-skew_r, abs=0.0, rel=1e-6)
    assert abs(skew_f) > 1.0, "the cubic phase should visibly skew the pulse"


def test_second_order_alone_leaves_the_pulse_symmetric() -> None:
    """The other half of the same claim: beta2 spreads both sides alike."""
    width, num_samples, sample_rate = 1e-12, 1 << 16, 4e12
    times, pulse = gaussian(width, num_samples, sample_rate)
    beta2 = dispersion_to_beta2(DISPERSION, WAVELENGTH)

    out = propagate_dispersion(pulse, sample_rate, beta2, 30e3, 0.0)
    sigma, skew = moments(times, np.abs(out) ** 2)
    assert sigma > 10.0 * width, "the pulse should actually have spread"
    assert abs(skew) < 1e-9


def test_the_cubic_term_is_still_exactly_invertible() -> None:
    """All-pass, like the quadratic one, so ``+z`` then ``-z`` is the identity."""
    width, num_samples, sample_rate = 1e-12, 1 << 16, 4e12
    _, pulse = gaussian(width, num_samples, sample_rate)
    beta2 = dispersion_to_beta2(DISPERSION, WAVELENGTH)
    beta3 = dispersion_slope_to_beta3(DISPERSION, SLOPE, WAVELENGTH)

    there = propagate_dispersion(pulse, sample_rate, beta2, 80e3, beta3)
    back = propagate_dispersion(there, sample_rate, beta2, -80e3, beta3)
    assert np.linalg.norm(back - pulse) / np.linalg.norm(pulse) < 1e-14
    assert np.abs(there).sum() != pytest.approx(np.abs(pulse).sum(), rel=1e-6)


def test_the_split_step_solver_carries_the_same_cubic_term() -> None:
    """With the nonlinearity off it must agree with the closed-form propagator.

    The coupled solver has its own copy of the linear operator, and a term added
    to one and not the other is the classic way two code paths drift.
    """
    width, num_samples, sample_rate = 1e-12, 1 << 14, 4e12
    _, pulse = gaussian(width, num_samples, sample_rate)
    beta2 = dispersion_to_beta2(DISPERSION, WAVELENGTH)
    beta3 = dispersion_slope_to_beta3(DISPERSION, SLOPE, WAVELENGTH)

    stepped, _ = propagate_coupled_ssfm(
        [pulse],
        sample_rate,
        beta2=[beta2],
        walkoff=[0.0],
        gamma=0.0,
        beta3=[beta3],
        alpha=0.0,
        distance=30e3,
    )
    exact = propagate_dispersion(pulse, sample_rate, beta2, 30e3, beta3)
    assert np.allclose(stepped[0], exact, rtol=0.0, atol=1e-12 * np.abs(exact).max())


def test_beta3_must_have_one_entry_per_field() -> None:
    _, pulse = gaussian(1e-12, 256, 4e12)
    with pytest.raises(ValueError, match="one entry per field"):
        propagate_coupled_ssfm(
            [pulse, pulse],
            4e12,
            beta2=[0.0, 0.0],
            walkoff=[0.0, 0.0],
            gamma=0.0,
            beta3=[0.0],
            alpha=0.0,
            distance=1e3,
        )


# ---------------------------------------------------------------------------
# the fibre block


def fiber(**settings: float) -> Fiber:
    return Fiber(length=80.0, attenuation=0.0, dispersion=17.0, label="fib", **settings)


def test_dispersion_is_flat_until_a_slope_says_otherwise() -> None:
    """The default has to reproduce every result taken before the slope existed."""
    flat = fiber()
    for wavelength in (1530e-9, 1550e-9, 1570e-9):
        assert flat.dispersion_at(wavelength) == pytest.approx(17e-6, abs=0.0, rel=1e-12)
        assert flat.beta3_at(wavelength) == 0.0


def test_the_slope_tilts_the_dispersion_about_the_reference() -> None:
    """Linear in wavelength, and pinned at the wavelength it is quoted at."""
    sloped = fiber(dispersion_slope=0.058, reference_wavelength=1550.0)
    assert sloped.dispersion_at(1550e-9) == pytest.approx(17e-6, abs=0.0, rel=1e-12)
    # 16 nm up the band, D is 17 + 0.058*16 ps/nm/km.
    assert sloped.dispersion_at(1566e-9) * 1e6 == pytest.approx(17.928, abs=0.0, rel=1e-9)
    assert sloped.dispersion_at(1534e-9) * 1e6 == pytest.approx(16.072, abs=0.0, rel=1e-9)
    # And it crosses zero where D/S says it does, far outside the C band.
    zero = 1550e-9 - 17e-6 / 0.058e3
    assert sloped.dispersion_at(zero) == pytest.approx(0.0, abs=1e-18)


def test_a_slope_gives_the_block_a_third_order_term() -> None:
    """Which the flat case deliberately does not have — see beta3_at."""
    sloped = fiber(dispersion_slope=0.058)
    assert sloped.beta3_at(1550e-9) * 1e39 == pytest.approx(0.13, rel=0.05)
    assert fiber().beta3_at(1550e-9) == 0.0


def link(wavelength: float) -> tuple[Graph, PowerMeter, Fiber]:
    ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=8, sequence_length=64, seed=1)
    graph = Graph(ctx)
    laser = graph.add(CWLaser(power=0.0, wavelength=wavelength, label="tx"))
    span = graph.add(fiber())
    meter = graph.add(PowerMeter(label="pm"))
    graph.chain(laser, span, meter)
    return graph, meter, span


def test_the_slope_changes_nothing_a_flat_fibre_did() -> None:
    """Not approximately: the same samples.

    Every validated number in this project was taken with a flat D, so the
    default has to be the identity rather than merely close to it.
    """
    graph, _, span = link(1550.0)
    before = graph.run(keep=[span]).port(span, "out")
    after = graph.run(overrides={(span, "dispersion_slope"): 0.0}, keep=[span]).port(span, "out")
    assert np.array_equal(before.bands[0].Ex, after.bands[0].Ex)


# ---------------------------------------------------------------------------
# what it costs a link


def coherent(
    length_km: float, wavelength_nm: float = 1550.0
) -> tuple[Graph, ConstellationAnalyzer, DispersionCompensator, Fiber]:
    """A coherent link over dispersive fibre, the shape ``examples/dispersion_link``
    has: one mechanism at a time, loss and nonlinearity off, so that what moves
    when the slope is switched on is the slope.
    """
    ctx = SimulationContext(
        bit_rate=32e9, samples_per_symbol=16, sequence_length=4096, seed=2026, precision="double"
    )
    graph = Graph(ctx)
    prbs = graph.add(PRBSGenerator(order=23.0, bits_per_symbol=4.0, label="prbs"))
    mapper = graph.add(QAMMapper(bits_per_symbol=4.0, label="map"))
    driver = graph.add(
        IQDriver(
            v_pi=4.0,
            predistort=True,
            pulse_shaping=True,
            roll_off=0.2,
            drive_ratio=0.4,
            label="drv",
        )
    )
    laser = graph.add(CWLaser(power=0.0, wavelength=wavelength_nm, linewidth=0.0, label="tx"))
    modulator = graph.add(IQModulator(v_pi=4.0, label="mod"))
    lo = graph.add(CWLaser(power=13.0, wavelength=wavelength_nm, linewidth=0.0, label="lo"))
    span = graph.add(
        Fiber(length=length_km, attenuation=0.0, dispersion=17.0, nonlinearity=0.0, label="fib")
    )
    receiver = graph.add(CoherentReceiver(responsivity=0.8, label="rx"))
    compensator = graph.add(
        DispersionCompensator(
            accumulated_dispersion=17.0 * length_km, wavelength=wavelength_nm, label="cdc"
        )
    )
    sampler = graph.add(IQSampler(matched_filter=True, roll_off=0.2, label="smp"))
    recovery = graph.add(CarrierRecovery(label="cr"))
    analyzer = graph.add(ConstellationAnalyzer(ignore_edges=64.0, label="vsa"))

    graph.connect(prbs["out"], mapper["in"])
    graph.connect(mapper["out"], driver["in"])
    graph.connect(laser, modulator["optical_in"])
    graph.connect(driver["i"], modulator["i"])
    graph.connect(driver["q"], modulator["q"])
    graph.connect(modulator, span["in"])
    graph.connect(span, receiver["in"])
    graph.connect(lo, receiver["lo"])
    graph.connect(receiver["i"], compensator["i"])
    graph.connect(receiver["q"], compensator["q"])
    graph.connect(compensator["i"], sampler["i"])
    graph.connect(compensator["q"], sampler["q"])
    graph.connect(mapper["out"], sampler["reference"])
    graph.connect(sampler["out"], recovery["in"])
    graph.connect(recovery["out"], analyzer["in"])
    graph.connect(mapper["out"], analyzer["reference"])
    return graph, analyzer, compensator, span


def test_one_compensator_setting_cannot_serve_a_whole_band() -> None:
    """The slope's real bite, and the reason it is worth modelling at all.

    Sixteen nanometres up the C band, standard fibre has 0.93 ps/nm/km more
    dispersion; over 80 km that is 74 ps/nm the compensator is not removing, and
    the link that ran at 1.7 % EVM at the centre runs at 15 %. Give that channel
    its own accumulated value and it comes back exactly.
    """
    graph, analyzer, compensator, span = coherent(80.0, 1566.0)
    # Set on the block rather than overridden for the run, so the value
    # dispersion_at() reports is the value the fibre propagated with.
    span._values = {**span._values, "dispersion_slope": 0.058}

    centre_setting = graph.run()[analyzer]
    here = span.dispersion_at(1566e-9) * 1e6 * 80.0  # ps/nm at this channel
    own_setting = graph.run(overrides={(compensator, "accumulated_dispersion"): here})[analyzer]

    assert here == pytest.approx(1434.2, rel=1e-3)
    assert centre_setting.evm > 0.10
    assert own_setting.evm < 0.02
    assert own_setting.evm < 0.3 * centre_setting.evm


def test_the_cubic_phase_is_small_within_one_channel_and_removable_anyway() -> None:
    """Honest about the size of it, and still able to take it out.

    Within a single 32 GBd channel the slope is nearly harmless — 1000 km of it
    costs a tenth of a point of EVM, because the cubic phase across the occupied
    band is only about 0.04 radians. The compensator can still cancel it, and the
    test is worth having because a slope compensation with the wrong sign would
    *double* that residue rather than remove it, which no eyeball on a
    constellation would ever catch.
    """
    graph, analyzer, compensator, span = coherent(1000.0)
    slope: dict[tuple[Component | str, str], float | bool] = {(span, "dispersion_slope"): 0.058}

    flat = graph.run()[analyzer]
    sloped = graph.run(overrides=slope)[analyzer]
    fixed = graph.run(overrides={**slope, (compensator, "accumulated_slope"): 58.0})[analyzer]
    wrong_sign = graph.run(overrides={**slope, (compensator, "accumulated_slope"): -58.0})[analyzer]

    assert sloped.evm > flat.evm
    assert fixed.evm == pytest.approx(flat.evm, rel=0.02)
    assert wrong_sign.evm > sloped.evm


def test_the_compensator_leaves_the_baseband_alone_at_zero_slope() -> None:
    """A default that is the identity, checked on the samples."""
    rng = np.random.default_rng(7)
    baseband = rng.normal(size=4096) + 1j * rng.normal(size=4096)
    without = compensate_dispersion(
        baseband, 512e9, accumulated_dispersion=1.36, wavelength=WAVELENGTH
    )
    explicit = compensate_dispersion(
        baseband, 512e9, accumulated_dispersion=1.36, wavelength=WAVELENGTH, accumulated_slope=0.0
    )
    assert np.array_equal(without, explicit)


def test_the_compensator_inverts_a_sloped_span_exactly() -> None:
    """Accumulated D and accumulated S in, the launched field back out."""
    width, num_samples, sample_rate = 4e-12, 1 << 13, 2e12
    _, pulse = gaussian(width, num_samples, sample_rate)
    length = 80e3
    beta2 = dispersion_to_beta2(DISPERSION, WAVELENGTH)
    beta3 = dispersion_slope_to_beta3(DISPERSION, SLOPE, WAVELENGTH)

    received = propagate_dispersion(pulse, sample_rate, beta2, length, beta3)
    restored = compensate_dispersion(
        received,
        sample_rate,
        accumulated_dispersion=DISPERSION * length,
        wavelength=WAVELENGTH,
        accumulated_slope=SLOPE * length,
    )
    assert np.linalg.norm(restored - pulse) / np.linalg.norm(pulse) < 1e-12

    # Without the slope term the same call leaves the cubic phase behind.
    partial = compensate_dispersion(
        received, sample_rate, accumulated_dispersion=DISPERSION * length, wavelength=WAVELENGTH
    )
    assert np.linalg.norm(partial - pulse) / np.linalg.norm(pulse) > 1e-3


def test_the_nonlinear_path_carries_the_slope_too() -> None:
    """The split-step branch, which the linear one never exercises.

    A fibre with ``nonlinearity`` set is solved by split-step rather than in
    closed form, and that solver holds its own copy of the linear operator. With
    the Kerr coefficient small enough to be irrelevant, the answer has to be the
    third-order broadening the closed form predicts — which it is not, to any
    accuracy, if the block simply never hands the solver a beta3.

    Written with ``dispersion`` at zero so the prediction is the pure cubic one:
    ``beta3`` is then ``(lambda^2/2 pi c)^2 * S`` and nothing else.
    """
    num_samples, sample_rate, width = 1 << 13, 2e12, 4e-12
    ctx = SimulationContext(
        bit_rate=sample_rate / 16, samples_per_symbol=16, sequence_length=num_samples // 16, seed=1
    )
    times, pulse = gaussian(width, num_samples, sample_rate)
    frequency = C_LIGHT / WAVELENGTH
    signal = OpticalSignal(
        bands=(
            Band(
                Ex=pulse * 1e-6,  # microwatts, so the Kerr term cannot matter
                Ey=np.zeros(num_samples, dtype=np.complex128),
                f0=frequency,
                fs=sample_rate,
            ),
        ),
        noise=(),
    )
    span = Fiber(
        length=120.0,
        attenuation=0.0,
        dispersion=0.0,
        dispersion_slope=6.0,
        nonlinearity=1.3,
        label="fib",
    )
    out = span.run(ctx, {"in": signal})["out"]
    assert out.bands[0].Ex.dtype is not None

    sigma0, _ = moments(times, np.abs(pulse) ** 2)
    sigma, skew = moments(times, np.abs(out.bands[0].Ex) ** 2)

    beta3 = span.beta3_at(WAVELENGTH)
    predicted = np.sqrt(1.0 + beta3**2 * 120e3**2 / (4.0 * width**6))
    assert predicted > 1.5, "the case has to actually broaden, or it proves nothing"
    assert sigma / sigma0 == pytest.approx(predicted, rel=1e-3)
    assert abs(skew) > 0.5
