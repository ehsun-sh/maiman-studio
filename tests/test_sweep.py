"""Validation of parameter sweeps and repeated runs."""

from __future__ import annotations

import os
from itertools import pairwise

import numpy as np
import pytest

from maiman import Component, Graph, GraphError, SimulationContext, sweep
from maiman.components import (
    BERAnalyzer,
    CWLaser,
    ElectricalFilter,
    Fiber,
    MachZehnderModulator,
    NRZDriver,
    PINPhotodiode,
    PowerMeter,
    PRBSGenerator,
)
from maiman.sweep import _lane_count, derive_run_seed


def _link(sequence_length: int = 2048) -> tuple[Graph, CWLaser, Fiber, BERAnalyzer]:
    ctx = SimulationContext(
        bit_rate=10e9, samples_per_symbol=8, sequence_length=sequence_length, seed=31
    )
    g = Graph(ctx)
    prbs = g.add(PRBSGenerator(order=15.0, label="prbs"))
    driver = g.add(NRZDriver(v_low=4.0, v_high=0.0, label="driver"))
    laser = g.add(CWLaser(power=-18.0, wavelength=1550.0, label="laser"))
    mzm = g.add(MachZehnderModulator(v_pi=4.0, extinction_ratio=30.0, label="mzm"))
    fiber = g.add(Fiber(length=0.0, attenuation=0.2, dispersion=0.0, label="fiber"))
    pin = g.add(PINPhotodiode(responsivity=0.8, shot_noise=False, label="pin"))
    lpf = g.add(ElectricalFilter(bandwidth=7.0, label="lpf"))
    analyzer = g.add(BERAnalyzer(label="ber"))

    g.chain(prbs, driver)
    g.connect(laser, mzm["optical_in"])
    g.connect(driver, mzm["electrical_in"])
    g.chain(mzm, fiber, pin, lpf)
    g.connect(lpf, analyzer["in"])
    g.connect(prbs["out"], analyzer["reference"])
    return g, laser, fiber, analyzer


# --------------------------------------------------------------------------
# The graph must come back unchanged
# --------------------------------------------------------------------------


def test_a_sweep_leaves_the_graph_exactly_as_it_found_it() -> None:
    """The failure mode this design exists to prevent: a leaked override would
    silently contaminate every later run, and nothing would report it."""
    g, laser, _, analyzer = _link()
    before = g.run()[analyzer].q_factor

    sweep(g, {(laser, "power"): [-24.0, -22.0, -20.0]})

    assert laser.power == -18.0
    assert g.run()[analyzer].q_factor == before


def test_overrides_are_restored_even_when_a_run_fails() -> None:
    g, laser, _, _ = _link()
    with pytest.raises(ValueError, match="below the minimum"):
        sweep(g, {(laser, "wavelength"): [1550.0, 10.0]})
    assert laser.wavelength == 1550.0


def test_overriding_an_unknown_parameter_is_rejected() -> None:
    g, laser, _, _ = _link()
    with pytest.raises(GraphError, match="has no parameter"):
        g.run(overrides={(laser, "wattage"): 1.0})


def test_overriding_a_component_outside_the_graph_is_rejected() -> None:
    g, _, _, _ = _link()
    with pytest.raises(GraphError, match="no component labelled"):
        g.run(overrides={("nowhere", "power"): 1.0})


# --------------------------------------------------------------------------
# A sweep must agree with doing it by hand
# --------------------------------------------------------------------------


def test_a_sweep_matches_a_manual_loop() -> None:
    """A sweep is a convenience, not a different calculation."""
    powers = [-24.0, -22.0, -20.0, -18.0]

    manual = []
    for power in powers:
        g, laser, _, analyzer = _link()
        laser._values["power"] = power
        manual.append(g.run()[analyzer].q_factor)

    g, laser, _, analyzer = _link()
    swept = sweep(g, {(laser, "power"): powers}).metric(analyzer, lambda m: m.q_factor)

    np.testing.assert_allclose(swept[:, 0], manual, rtol=1e-12)


def test_sweeping_launch_power_reproduces_the_sensitivity_curve() -> None:
    """Q is proportional to received power in a thermal-limited receiver, so the
    curve a sweep produces has to be a straight line of slope 1 in dB."""
    g, laser, _, analyzer = _link()
    powers = [-24.0, -21.0, -18.0, -15.0]
    result = sweep(g, {(laser, "power"): powers})

    q = result.metric(analyzer, lambda m: m.q_factor)[:, 0]
    for lower, higher in pairwise(q):
        assert higher / lower == pytest.approx(10 ** (3 / 10), rel=0.05)

    np.testing.assert_allclose(result.axis(laser, "power"), powers)


def test_two_axes_sweep_the_cartesian_product() -> None:
    g, laser, fiber, analyzer = _link(sequence_length=512)
    result = sweep(
        g,
        {(laser, "power"): [-18.0, -12.0], (fiber, "length"): [0.0, 20.0, 40.0]},
    )

    assert len(result) == 6
    # The last axis varies fastest.
    np.testing.assert_allclose(result.axis(fiber, "length"), [0, 20, 40, 0, 20, 40])
    np.testing.assert_allclose(result.axis(laser, "power"), [-18, -18, -18, -12, -12, -12])

    q = result.metric(analyzer, lambda m: m.q_factor)[:, 0].reshape(2, 3)
    # More launch power is better; more fiber is worse.
    assert (q[1] > q[0]).all()
    assert (np.diff(q, axis=1) < 0).all()


def test_fixed_overrides_apply_at_every_point() -> None:
    g, laser, fiber, _ = _link(sequence_length=512)
    result = sweep(
        g,
        {(laser, "power"): [-18.0, -15.0]},
        fixed={(fiber, "length"): 40.0, (fiber, "dispersion"): 17.0},
    )
    assert len(result) == 2
    assert fiber.length == 0.0  # still restored
    assert all(point.runs for point in result)


def test_labels_may_be_used_instead_of_component_objects() -> None:
    g, laser, _, analyzer = _link(sequence_length=512)
    by_object = sweep(g, {(laser, "power"): [-20.0]}).metric(analyzer, lambda m: m.q_factor)
    by_label = sweep(g, {("laser", "power"): [-20.0]}).metric(analyzer, lambda m: m.q_factor)
    np.testing.assert_allclose(by_object, by_label)


# --------------------------------------------------------------------------
# Repeated runs
# --------------------------------------------------------------------------


def test_repeated_runs_draw_independent_noise() -> None:
    """A single BER estimate at a marginal operating point is one sample, not an
    answer. Repeats are what turn it into a distribution."""
    g, laser, _, analyzer = _link(sequence_length=4096)
    result = sweep(g, {(laser, "power"): [-21.0]}, runs=8)

    errors = result.metric(analyzer, lambda m: float(m.errors))
    assert errors.shape == (1, 8)
    assert len(set(errors[0])) > 1, "repeated runs produced identical noise"

    # They should still be the same measurement: spread, not disagreement.
    assert errors.std() < 0.5 * errors.mean()


def test_repeated_runs_are_reproducible() -> None:
    g, laser, _, analyzer = _link(sequence_length=1024)
    first = sweep(g, {(laser, "power"): [-21.0]}, runs=4).metric(analyzer, lambda m: m.q_factor)
    second = sweep(g, {(laser, "power"): [-21.0]}, runs=4).metric(analyzer, lambda m: m.q_factor)
    np.testing.assert_array_equal(first, second)


def test_run_seeds_are_derived_not_incremented() -> None:
    seeds = {derive_run_seed(42, i) for i in range(64)}
    assert len(seeds) == 64
    assert derive_run_seed(42, 0) == derive_run_seed(42, 0)
    assert derive_run_seed(42, 0) != derive_run_seed(43, 0)
    # Adjacent run indices must not give adjacent seeds.
    assert abs(derive_run_seed(42, 0) - derive_run_seed(42, 1)) > 1


def test_a_seed_override_changes_the_noise_but_not_the_signal() -> None:
    ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=8, sequence_length=256, seed=1)
    g = Graph(ctx)
    laser = g.add(CWLaser(power=0.0, linewidth=1000.0, label="laser"))
    meter = g.add(PowerMeter(label="meter"))
    g.chain(laser, meter)

    default = g.run(keep=[laser]).port(laser, "out").bands[0]
    reseeded = g.run(keep=[laser], seed=999).port(laser, "out").bands[0]

    assert not np.array_equal(np.asarray(default.Ex), np.asarray(reseeded.Ex))
    # Phase noise moves the phase, never the power.
    assert default.average_power() == pytest.approx(reseeded.average_power(), rel=1e-6)


# --------------------------------------------------------------------------
# Argument checking
# --------------------------------------------------------------------------


def test_a_sweep_needs_at_least_one_axis() -> None:
    g, _, _, _ = _link(sequence_length=256)
    with pytest.raises(ValueError, match="at least one axis"):
        sweep(g, {})


def test_an_empty_axis_is_rejected() -> None:
    g, laser, _, _ = _link(sequence_length=256)
    with pytest.raises(ValueError, match="no values"):
        sweep(g, {(laser, "power"): []})


def test_runs_must_be_positive() -> None:
    g, laser, _, _ = _link(sequence_length=256)
    with pytest.raises(ValueError, match="runs must be"):
        sweep(g, {(laser, "power"): [-20.0]}, runs=0)


def test_asking_for_an_axis_that_was_not_swept() -> None:
    g, laser, fiber, _ = _link(sequence_length=256)
    result = sweep(g, {(laser, "power"): [-20.0]})
    with pytest.raises(KeyError, match="was not swept"):
        result.axis(fiber, "length")


# ---------------------------------------------------------------------------
# workers
#
# The whole claim being defended is that adding workers changes the wall clock
# and nothing else. A sweep whose numbers moved with the thread count would be
# useless as a measurement, and the failure would not look like a crash — it
# would look like a curve with the right shape and the wrong values.


@pytest.mark.parametrize("workers", [1, 2, 3, 4, None])
def test_the_answer_does_not_depend_on_the_worker_count(workers: int | None) -> None:
    """Bit-identical, not close. Every point is a function of its own overrides.

    ``None`` asks for one per core, so this also covers however many that
    happens to be on the machine running it — which is the configuration nobody
    tests by hand.
    """
    graph, laser, _fiber, analyzer = _link()
    powers = [-14.0, -12.0, -10.0, -8.0, -6.0]

    reference = sweep(graph, {(laser, "power"): powers})
    result = sweep(graph, {(laser, "power"): powers}, workers=workers)

    assert [point.index for point in result.points] == list(range(len(powers)))
    assert np.array_equal(reference.axis(laser, "power"), result.axis(laser, "power")), (
        "the points came back in a different order"
    )
    assert np.array_equal(
        reference.metric(analyzer, lambda m: m.q_factor),
        result.metric(analyzer, lambda m: m.q_factor),
    ), "the numbers moved with the worker count"


def test_repeated_runs_keep_their_seeds_when_parallelised() -> None:
    """The derived seeds come from the original context, not from a worker's copy.

    They are the same object, so this could only break by someone deriving from
    ``on.ctx``. It is the kind of change that looks like a tidy-up and silently
    reseeds every repeated run.
    """
    graph, laser, _fiber, analyzer = _link(sequence_length=512)
    axis: dict[tuple[Component | str, str], list[float]] = {(laser, "power"): [-12.0, -10.0]}

    sequential = sweep(graph, axis, runs=3)
    parallel = sweep(graph, axis, runs=3, workers=4)

    assert np.array_equal(
        sequential.metric(analyzer, lambda m: m.q_factor),
        parallel.metric(analyzer, lambda m: m.q_factor),
    )
    # And the repeats really are different from one another, or the check above
    # would pass on a sweep that had quietly stopped varying its seed.
    spread = parallel.metric(analyzer, lambda m: m.q_factor)
    assert spread.shape == (2, 3)
    assert len(set(spread[0])) == 3


def test_a_parallel_sweep_leaves_the_original_graph_alone() -> None:
    """It is never run at all above one worker — only copied.

    A stronger guarantee than the sequential path's, which relies on overrides
    being rolled back. Worth asserting because the copies are what make it true
    and a future change to hand out the original would pass every other test.
    """
    graph, laser, _fiber, _analyzer = _link(sequence_length=512)
    before = dict(laser._values)

    sweep(graph, {(laser, "power"): [-20.0, -10.0, 0.0, 5.0]}, workers=4)

    assert laser._values == before


def test_a_failing_point_still_raises_and_does_not_hang() -> None:
    """One bad point must not strand the workers queued behind it.

    The graph a thread borrowed is returned in a ``finally``; without that the
    queue drains and the pool deadlocks on a sweep that should have taken a
    second to fail.
    """
    graph, laser, _fiber, _analyzer = _link(sequence_length=512)
    # More points than lanes on purpose: with a lane each, nothing ever waits
    # for a graph to come back and a missing `finally` would go unnoticed.
    with pytest.raises(GraphError):
        sweep(graph, {(laser, "nonexistent"): [float(v) for v in range(8)]}, workers=2)


def test_more_workers_than_points_is_not_an_error() -> None:
    """It just makes fewer copies. Asking for sixteen lanes for three points
    should not allocate sixteen graphs, and should certainly not fail."""
    graph, laser, _fiber, analyzer = _link(sequence_length=512)
    result = sweep(graph, {(laser, "power"): [-10.0, -8.0, -6.0]}, workers=16)
    assert len(result.points) == 3
    assert np.all(np.isfinite(result.metric(analyzer, lambda m: m.q_factor)))


def test_the_lane_count_never_exceeds_the_work() -> None:
    """Checked directly, because the observable effect is only a wasted deepcopy."""
    assert _lane_count(16, 3) == 3
    assert _lane_count(2, 9) == 2
    assert _lane_count(1, 9) == 1
    assert _lane_count(None, 1) == 1
    assert _lane_count(None, 10_000) == (os.cpu_count() or 1)


@pytest.mark.parametrize("workers", [0, -1])
def test_a_worker_count_below_one_is_rejected(workers: int) -> None:
    graph, laser, _fiber, _analyzer = _link(sequence_length=512)
    with pytest.raises(ValueError, match="workers must be"):
        sweep(graph, {(laser, "power"): [-10.0]}, workers=workers)
