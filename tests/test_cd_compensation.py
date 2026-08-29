"""Chromatic dispersion compensation in the receiver.

Until this existed the coherent transceiver had only ever been run back to back,
which meant the one impairment that dominates every real span had never been in
the loop. Five kilometres of ordinary fibre cost twenty decibels.

The tests come in two layers. The kernel layer asserts that the filter is the
exact inverse of the propagator — not close to it, equal to it — because a pure
phase with unit magnitude is genuinely invertible and anything less would mean a
mistake somewhere. The link layer asserts the thing that matters: an eighty
kilometre span recovers to back-to-back quality, and the same link without the
block recovers nothing at all. Both halves are needed. The first alone would pass
just as well if the fibre were doing nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from oosim import Graph, SimulationContext
from oosim.components import (
    ButterflyEqualizer,
    CarrierRecovery,
    CoherentReceiver,
    ConstellationAnalyzer,
    CWLaser,
    DispersionCompensator,
    DualPolarizationReceiver,
    Fiber,
    IQDriver,
    IQModulator,
    IQSampler,
    PolarizationCombiner,
    PolarizationRotator,
    PRBSGenerator,
    QAMMapper,
    Splitter,
)
from oosim.dsp import compensate_dispersion, dispersive_spread
from oosim.kernels import dispersion_to_beta2, propagate_dispersion
from oosim.signals import ConstellationMeasurement
from oosim.units import C_LIGHT

DISPERSION = 17.0  # ps/nm/km, standard single-mode fibre at 1550 nm
V_PI = 4.0
SYMBOL_RATE = 32e9
BITS_PER_SYMBOL = 4


# --------------------------------------------------------------------------
# The filter itself
# --------------------------------------------------------------------------


def _chirped_burst(num_samples: int) -> np.ndarray:
    rng = np.random.default_rng(11)
    return rng.normal(size=num_samples) + 1j * rng.normal(size=num_samples)


def test_compensation_exactly_inverts_propagation() -> None:
    """Round-tripping a field through fibre and the compensator returns it unchanged.

    Dispersion is an all-pass phase, so this is not a tolerance question: the
    composition is the identity up to floating point. A test written with a loose
    tolerance here would accept a filter that was merely approximately right,
    which is exactly the failure mode that matters — a small residual over one
    span becomes a large one over ten.
    """
    field = _chirped_burst(4096)
    sample_rate = 512e9
    length = 80e3
    beta2 = dispersion_to_beta2(DISPERSION * 1e-6, 1550e-9)

    spread = propagate_dispersion(field, sample_rate, beta2, length)
    assert not np.allclose(spread, field, atol=1e-6)  # the fibre did something

    recovered = compensate_dispersion(
        spread,
        sample_rate,
        accumulated_dispersion=DISPERSION * 1e-6 * length,
        wavelength=1550e-9,
    )
    assert np.allclose(recovered, field, atol=1e-9)


def test_compensation_is_all_pass() -> None:
    """Energy is conserved, so nothing is amplified and nothing is thrown away."""
    field = _chirped_burst(2048)
    out = compensate_dispersion(field, 512e9, accumulated_dispersion=1.36, wavelength=1550e-9)
    assert float(np.sum(np.abs(out) ** 2)) == pytest.approx(
        float(np.sum(np.abs(field) ** 2)), rel=1e-12
    )


def test_compensation_sign_is_not_free() -> None:
    """Compensating with the sign flipped doubles the impairment instead of removing it.

    The most plausible single mistake in this block, and the one a round-trip
    test on its own would still catch only by accident. Pinning it directly means
    the assertion says what it is protecting.
    """
    field = _chirped_burst(4096)
    sample_rate = 512e9
    accumulated = DISPERSION * 1e-6 * 80e3
    beta2 = dispersion_to_beta2(DISPERSION * 1e-6, 1550e-9)

    spread = propagate_dispersion(field, sample_rate, beta2, 80e3)
    doubled = propagate_dispersion(field, sample_rate, beta2, 160e3)

    wrong_way = compensate_dispersion(
        spread, sample_rate, accumulated_dispersion=-accumulated, wavelength=1550e-9
    )
    assert np.allclose(wrong_way, doubled, atol=1e-9)


def test_wavelength_scales_the_compensation_quadratically() -> None:
    """Beta2 goes as lambda squared, so the wavelength is a real parameter.

    Compensating a 1550 nm span as though it were at 1310 nm leaves a residual,
    and the residual is not arbitrary: it is what a span of
    ``(1 - 1310**2/1550**2)`` of the original length would have produced. Asserting
    the *value* rather than merely that it degraded is what makes this catch a
    hardcoded wavelength rather than only a missing one.
    """
    field = _chirped_burst(4096)
    sample_rate = 512e9
    accumulated = DISPERSION * 1e-6 * 80e3
    beta2 = dispersion_to_beta2(DISPERSION * 1e-6, 1550e-9)
    spread = propagate_dispersion(field, sample_rate, beta2, 80e3)

    mis_compensated = compensate_dispersion(
        spread, sample_rate, accumulated_dispersion=accumulated, wavelength=1310e-9
    )

    residual_fraction = 1.0 - (1310e-9**2) / (1550e-9**2)
    expected = propagate_dispersion(field, sample_rate, beta2, 80e3 * residual_fraction)
    assert np.allclose(mis_compensated, expected, atol=1e-9)


def test_dispersive_spread_matches_the_hand_calculation() -> None:
    """Delta-tau = D*L*Delta-lambda, with Delta-lambda = lambda^2 * Delta-f / c."""
    accumulated = DISPERSION * 1e-6 * 80e3  # 1360 ps/nm
    bandwidth = 1.2 * SYMBOL_RATE
    wavelength = 1550e-9

    wavelength_span = wavelength**2 * bandwidth / C_LIGHT
    expected_seconds = accumulated * wavelength_span

    spread = dispersive_spread(accumulated, bandwidth, wavelength, SYMBOL_RATE)
    assert spread == pytest.approx(expected_seconds * SYMBOL_RATE, rel=1e-12)
    # And it is a number a seven-tap equaliser plainly cannot reach.
    assert spread > 10.0


# --------------------------------------------------------------------------
# The link
# --------------------------------------------------------------------------


def coherent_span(
    length_km: float,
    *,
    compensate_km: float | None = None,
    compensator_wavelength: float = 1550.0,
    sequence_length: int = 1024,
) -> tuple[Graph, ConstellationAnalyzer]:
    """A 16-QAM coherent link over ``length_km`` of dispersive fibre.

    Loss and nonlinearity are switched off and the lasers are ideal, so
    dispersion is the *only* thing acting. That isolation is the point: any
    change in the recovered quality has exactly one possible cause.

    ``compensate_km`` defaults to the true span. Passing something else is how
    the mis-set cases are built.
    """
    removed = length_km if compensate_km is None else compensate_km

    ctx = SimulationContext(
        bit_rate=SYMBOL_RATE,
        samples_per_symbol=16,
        sequence_length=sequence_length,
        seed=7,
        precision="double",
    )
    graph = Graph(ctx)
    prbs = graph.add(PRBSGenerator(order=23.0, bits_per_symbol=float(BITS_PER_SYMBOL)))
    mapper = graph.add(QAMMapper(bits_per_symbol=float(BITS_PER_SYMBOL)))
    driver = graph.add(
        IQDriver(v_pi=V_PI, predistort=True, pulse_shaping=True, roll_off=0.2, drive_ratio=0.4)
    )
    laser = graph.add(CWLaser(power=0.0, wavelength=1550.0, linewidth=0.0))
    modulator = graph.add(IQModulator(v_pi=V_PI))
    lo = graph.add(CWLaser(power=13.0, wavelength=1550.0, linewidth=0.0))
    receiver = graph.add(CoherentReceiver(responsivity=0.8))
    compensator = graph.add(
        DispersionCompensator(
            accumulated_dispersion=DISPERSION * removed, wavelength=compensator_wavelength
        )
    )
    sampler = graph.add(IQSampler(matched_filter=True, roll_off=0.2))
    recovery = graph.add(CarrierRecovery())
    analyzer = graph.add(ConstellationAnalyzer(ignore_edges=64.0))

    graph.connect(prbs["out"], mapper["in"])
    graph.connect(mapper["out"], driver["in"])
    graph.connect(laser, modulator["optical_in"])
    graph.connect(driver["i"], modulator["i"])
    graph.connect(driver["q"], modulator["q"])
    if length_km:
        fiber = graph.add(
            Fiber(length=length_km, attenuation=0.0, dispersion=DISPERSION, nonlinearity=0.0)
        )
        graph.connect(modulator, fiber["in"])
        graph.connect(fiber, receiver["in"])
    else:
        graph.connect(modulator, receiver["in"])
    graph.connect(lo, receiver["lo"])
    graph.connect(receiver["i"], compensator["i"])
    graph.connect(receiver["q"], compensator["q"])
    graph.connect(compensator["i"], sampler["i"])
    graph.connect(compensator["q"], sampler["q"])
    graph.connect(mapper["out"], sampler["reference"])
    graph.connect(sampler["out"], recovery["in"])
    graph.connect(recovery["out"], analyzer["in"])
    graph.connect(mapper["out"], analyzer["reference"])
    return graph, analyzer


def measure(length_km: float, **kwargs: object) -> ConstellationMeasurement:
    graph, analyzer = coherent_span(length_km, **kwargs)  # type: ignore[arg-type]
    return graph.run()[analyzer]


def test_an_uncompensated_span_destroys_the_link() -> None:
    """The premise. Without this, everything below would be measuring nothing.

    Five kilometres is the interesting number: it is a metro hop, not a haul, and
    it is already fatal. The eighty kilometre case is past the point where EVM
    means anything, so it is asserted as a symbol error rate at chance instead —
    fifteen sixteenths of symbols wrong is what guessing looks like on 16-QAM.
    """
    short = measure(5.0, compensate_km=0.0)
    assert short.evm > 0.10  # back to back is 0.017

    long = measure(80.0, compensate_km=0.0)
    assert long.snr_db < 0.0
    assert long.symbol_errors / long.symbols_evaluated > 0.7


@pytest.mark.parametrize("length_km", [5.0, 20.0, 80.0, 400.0, 1000.0])
def test_a_compensated_span_recovers_back_to_back_quality(length_km: float) -> None:
    """The headline claim, and it holds to a thousand kilometres.

    A frequency-domain filter has no tap budget, so the span length does not
    appear in its cost the way it would for a time-domain FIR. Testing at 1000 km
    — 167 symbol periods of smearing — is what makes that concrete rather than
    asserted.
    """
    reference = measure(0.0)
    assert reference.evm < 0.02

    result = measure(length_km)
    assert result.evm == pytest.approx(reference.evm, abs=0.002)
    assert result.symbol_errors == 0


def test_mis_setting_the_compensator_is_symmetric_in_sign() -> None:
    """Over- and under-compensating by the same amount cost the same.

    This is the sign check at link level, and it is sharper than it looks. If the
    block applied its correction with the wrong sign, the *nominally correct*
    setting would land at twice the span and the two flanking cases would be
    wildly unequal. Symmetry can only happen around a true zero.
    """
    under = measure(80.0, compensate_km=76.0)
    over = measure(80.0, compensate_km=84.0)

    assert under.evm > 0.05  # four kilometres of error is already expensive
    assert under.evm == pytest.approx(over.evm, rel=0.05)


def test_the_compensator_wavelength_is_used() -> None:
    """Declaring the wrong wavelength degrades the link.

    Beta2 scales as lambda squared, so a compensator told 1310 nm removes about
    71% of what it should. Nothing else in the block would notice.
    """
    correct = measure(80.0)
    mismatched = measure(80.0, compensator_wavelength=1310.0)
    assert mismatched.evm > 10.0 * correct.evm


def test_diagnostics_report_the_spread_that_was_removed() -> None:
    """The number is on the port, so a misplaced decimal point is visible."""
    graph, _ = coherent_span(80.0)
    compensator = next(c for c in graph.components if isinstance(c, DispersionCompensator))
    diagnostics = graph.run(keep=[compensator]).port(compensator, "diagnostics")

    assert diagnostics.accumulated_dispersion == pytest.approx(1.36, rel=1e-9)  # 1360 ps/nm
    assert diagnostics.removed_symbols == pytest.approx(13.4, abs=0.2)


# --------------------------------------------------------------------------
# Why the order is static first, adaptive second
# --------------------------------------------------------------------------


def dual_polarization_span(
    length_km: float, *, compensate: bool, taps: float = 7.0
) -> tuple[Graph, dict[str, ConstellationAnalyzer]]:
    """Two 16-QAM tributaries through a rotator and a dispersive span.

    Four analysers, because a blind butterfly does not label its outputs and may
    deliver the tributaries swapped. Both pairings are resolved in one run rather
    than assuming an ordering the algorithm never promised.
    """
    ctx = SimulationContext(
        bit_rate=SYMBOL_RATE,
        samples_per_symbol=4,
        sequence_length=4096,
        seed=2026,
        precision="double",
    )
    graph = Graph(ctx)
    laser = graph.add(CWLaser(power=0.0, linewidth=100.0))
    splitter = graph.add(Splitter(2))
    graph.connect(laser, splitter["in"])

    mappers, modulators = {}, {}
    for index, axis in enumerate(("x", "y")):
        prbs = graph.add(
            PRBSGenerator(
                order=23.0 if axis == "x" else 15.0, bits_per_symbol=float(BITS_PER_SYMBOL)
            )
        )
        mapper = graph.add(QAMMapper(bits_per_symbol=float(BITS_PER_SYMBOL)))
        driver = graph.add(IQDriver())
        modulator = graph.add(IQModulator())
        graph.chain(prbs, mapper, driver)
        graph.connect(splitter[f"out{index}"], modulator["optical_in"])
        graph.connect(driver["i"], modulator["i"])
        graph.connect(driver["q"], modulator["q"])
        mappers[axis], modulators[axis] = mapper, modulator

    combiner = graph.add(PolarizationCombiner())
    graph.connect(modulators["x"], combiner["x"])
    graph.connect(modulators["y"], combiner["y"])
    rotator = graph.add(PolarizationRotator(angle=32.0, phase=25.0))
    graph.connect(combiner, rotator["in"])

    lo = graph.add(CWLaser(power=13.0, linewidth=100.0))
    receiver = graph.add(DualPolarizationReceiver())
    if length_km:
        fiber = graph.add(
            Fiber(length=length_km, attenuation=0.0, dispersion=DISPERSION, nonlinearity=0.0)
        )
        graph.connect(rotator, fiber["in"])
        graph.connect(fiber, receiver["in"])
    else:
        graph.connect(rotator, receiver["in"])
    graph.connect(lo, receiver["lo"])

    samplers = {}
    for axis in ("x", "y"):
        sampler = graph.add(IQSampler())
        if compensate:
            block = graph.add(
                DispersionCompensator(
                    accumulated_dispersion=DISPERSION * length_km, wavelength=1550.0
                )
            )
            graph.connect(receiver[f"{axis}i"], block["i"])
            graph.connect(receiver[f"{axis}q"], block["q"])
            graph.connect(block["i"], sampler["i"])
            graph.connect(block["q"], sampler["q"])
        else:
            graph.connect(receiver[f"{axis}i"], sampler["i"])
            graph.connect(receiver[f"{axis}q"], sampler["q"])
        graph.connect(mappers[axis]["out"], sampler["reference"])
        samplers[axis] = sampler

    equalizer = graph.add(ButterflyEqualizer(taps=taps))
    graph.connect(samplers["x"]["out"], equalizer["x"])
    graph.connect(samplers["y"]["out"], equalizer["y"])

    analyzers = {}
    for axis, port in (("x", "x_out"), ("y", "y_out")):
        recovery = graph.add(CarrierRecovery())
        graph.connect(equalizer[port], recovery["in"])
        for reference in ("x", "y"):
            analyzer = graph.add(ConstellationAnalyzer(ignore_edges=128.0))
            graph.connect(recovery["out"], analyzer["in"])
            graph.connect(mappers[reference]["out"], analyzer["reference"])
            analyzers[axis + reference] = analyzer
    return graph, analyzers


def dual_polarization_evm(length_km: float, *, compensate: bool, taps: float = 7.0) -> float:
    """Worst tributary EVM under whichever pairing the equaliser actually produced."""
    graph, analyzers = dual_polarization_span(length_km, compensate=compensate, taps=taps)
    results = graph.run(keep=[])
    taken = {key: results[a].evm for key, a in analyzers.items()}
    return min(max(taken["xx"], taken["yy"]), max(taken["xy"], taken["yx"]))


def test_a_longer_adaptive_filter_is_not_a_substitute() -> None:
    """The claim that dispersion belongs in a *separate*, static stage, measured.

    Both blocks are linear filters, so in principle the butterfly could absorb the
    dispersion. In practice it cannot: growing it from 7 taps to 65 — the longest
    the block allows, and already an order of magnitude more expensive — leaves
    the link just as dead, because a blind modulus criterion has no gradient to
    follow when the constellation has been smeared into a Gaussian blob. The
    static block ahead of a 7-tap filter restores back-to-back quality outright.

    Without this test the ordering in :class:`DispersionCompensator`'s docstring
    would be received wisdom repeated rather than a property of this code.
    """
    reference = dual_polarization_evm(0.0, compensate=False)
    assert reference < 0.05

    short_blind = dual_polarization_evm(80.0, compensate=False, taps=7.0)
    long_blind = dual_polarization_evm(80.0, compensate=False, taps=65.0)
    assert short_blind > 1.0
    assert long_blind > 1.0  # nine times the taps, still nothing

    staged = dual_polarization_evm(80.0, compensate=True, taps=7.0)
    assert staged == pytest.approx(reference, abs=0.005)
