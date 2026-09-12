"""The loop control, and the one real cavity it was built for.

A circulator's routes are not a cycle, and neither is its isolation leak; both
schedule without any loop control. Its return loss is a cycle -- light reflected
back out of port 2 goes into the grating again -- and the steady state is a
geometric series. These check that Feedback sums it, against Circuit.solve,
which sums it exactly, and that a graph without one still refuses a loop.
"""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np
import pytest

from maiman import CycleError, Graph, SimulationContext
from maiman.circuit import Circuit
from maiman.components import (
    Circulator,
    Combiner,
    CWLaser,
    Feedback,
    FiberBraggGrating,
    PowerMeter,
)
from maiman.graph import Progress
from maiman.photonics import circulator, fiber_bragg_grating
from maiman.units import C_LIGHT

BRAGG = 1550e-9
LOSS = 0.7
ISOLATION = 40.0
RETURN_LOSS = 20.0

#: Storage is complex64, so two independent routes to the same power agree to a
#: few micro-decibels and no better. Every comparison with the exact solve is at
#: this tolerance, which is far below any effect being claimed.
FLOOR_DB = 1e-4


def drop_graph(
    *, return_loss: float, passes: int | None
) -> tuple[Graph, PowerMeter, Feedback | None]:
    """A channel dropped through a circulator; ``passes=None`` wires the loop directly."""
    ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=8, sequence_length=64)
    graph = Graph(ctx)
    dropped = graph.add(CWLaser(power=0.0, wavelength=1550.0, label="ch1550"))
    express = graph.add(CWLaser(power=0.0, wavelength=1552.0, label="ch1552"))
    mux = graph.add(Combiner(2, label="mux"))
    circ = graph.add(
        Circulator(insertion_loss=LOSS, isolation=ISOLATION, return_loss=return_loss, label="circ")
    )
    grating = graph.add(
        FiberBraggGrating(bragg_wavelength=1550.0, length=10.0, index_modulation=1e-4, label="fbg")
    )
    meter = graph.add(PowerMeter(label="drop"))
    graph.connect(dropped, mux["in0"])
    graph.connect(express, mux["in1"])
    graph.connect(mux, circ["in1"])
    graph.connect(circ["out2"], grating["in"])

    loop = None
    if passes is None:
        graph.connect(grating["reflected"], circ["in2"])
    else:
        loop = graph.add(Feedback(passes=float(passes), label="loop"))
        graph.connect(grating["reflected"], loop["in"])
        graph.connect(loop["out"], circ["in2"])
    graph.connect(circ["out3"], meter["in"])
    return graph, meter, loop


def drop_dbm(return_loss: float, passes: int) -> dict[float, float]:
    graph, meter, _ = drop_graph(return_loss=return_loss, passes=passes)
    reading = graph.run()[meter]
    return {round(band.wavelength_nm, 2): band.power_dbm for band in reading.bands}


def exact_dbm(return_loss: float, nm: float) -> float:
    """The drop port from Circuit.solve, which sums every bounce the circuit has."""
    f = np.array([C_LIGHT / (nm * 1e-9)])
    solved = Circuit()
    solved.add(
        "c",
        circulator(f, insertion_loss_db=LOSS, isolation_db=ISOLATION, return_loss_db=return_loss),
    )
    solved.add(
        "g", fiber_bragg_grating(f, length=0.01, index_modulation=1e-4, bragg_wavelength=BRAGG)
    )
    solved.link("c", "p2", "g", "in")
    solved.expose("input", "c", "p1")
    solved.expose("drop", "c", "p3")
    solved.expose("through", "g", "out")
    return float(10.0 * np.log10(abs(solved.solve().transmission("drop", "input")[0]) ** 2))


# --------------------------------------------------------------------------
# The cycle, refused and then run
# --------------------------------------------------------------------------


def test_return_loss_next_to_a_grating_is_a_cycle_and_is_refused_without_a_loop() -> None:
    """The one route of a circulator that really is a loop. The message names the fix."""
    graph, _, _ = drop_graph(return_loss=RETURN_LOSS, passes=None)
    with pytest.raises(CycleError, match="Feedback"):
        graph.run()


def test_without_return_loss_there_is_no_cycle_to_close() -> None:
    """The same wiring with an ideal port runs with no loop control at all."""
    graph, meter, _ = drop_graph(return_loss=0.0, passes=None)
    reading = graph.run()[meter]
    by_nm = {round(band.wavelength_nm, 2): band.power_dbm for band in reading.bands}
    assert by_nm[1550.0] == pytest.approx(exact_dbm(0.0, 1550.0), abs=FLOOR_DB)


def test_one_pass_is_the_leak_alone() -> None:
    """Nothing has been round yet, so the drop port hears only port 1's leak."""
    got = drop_dbm(RETURN_LOSS, passes=1)
    assert got[1550.0] == pytest.approx(-ISOLATION, abs=FLOOR_DB)
    assert got[1552.0] == pytest.approx(-ISOLATION, abs=FLOOR_DB)


def test_two_passes_are_the_feed_forward_sum() -> None:
    """One trip to the grating and back, and no echo yet: the circuit without return loss."""
    got = drop_dbm(RETURN_LOSS, passes=2)
    for nm in (1550.0, 1552.0):
        assert got[nm] == pytest.approx(exact_dbm(0.0, nm), abs=FLOOR_DB)


def test_enough_passes_converge_to_the_exact_cavity() -> None:
    """The geometric series summed, against the exact solve that sums it in one step."""
    got = drop_dbm(RETURN_LOSS, passes=16)
    for nm in (1550.0, 1552.0):
        assert got[nm] == pytest.approx(exact_dbm(RETURN_LOSS, nm), abs=FLOOR_DB)

    # And the cavity is really there: the answer is not the feed-forward one.
    assert abs(exact_dbm(RETURN_LOSS, 1550.0) - exact_dbm(0.0, 1550.0)) > 10 * FLOOR_DB


def test_the_residual_falls_by_the_round_trip_gain_on_every_pass() -> None:
    """The field converges geometrically, and the ratio is the physics.

    Each pass multiplies the change in the returning field by the loop's round-trip
    gain: the echo's amplitude ``10**(-return_loss/20)`` times the grating's own
    reflection ``|r|``. Measured pass on pass, that is the residual's ratio.

    The *power* on the drop port does not fall so tidily, and that is not a flaw.
    Partial sums of a series whose ratio carries a phase overshoot and undershoot
    the limit in turn, so the error in decibels rises and falls while the field
    closes in: three passes are further off than two here, and five than four.
    Watch the residual, not the power.
    """
    residuals = []
    for passes in (3, 4, 5, 6):
        graph, _, loop = drop_graph(return_loss=RETURN_LOSS, passes=passes)
        assert loop is not None
        residuals.append(graph.run()[loop])

    grating = FiberBraggGrating(bragg_wavelength=1550.0, length=10.0, index_modulation=1e-4)
    gain = 10.0 ** (-RETURN_LOSS / 20.0) * math.sqrt(grating.peak_reflectivity())
    for before, after in pairwise(residuals):
        assert after / before == pytest.approx(gain, rel=0.02)


def test_the_residual_says_when_it_has_converged() -> None:
    """Infinite after one pass, falling with every pass after, and small once converged."""
    residuals = {}
    for passes in (1, 3, 6, 16):
        graph, _, loop = drop_graph(return_loss=RETURN_LOSS, passes=passes)
        assert loop is not None
        residuals[passes] = graph.run()[loop]

    assert math.isinf(residuals[1])
    assert residuals[3] > residuals[6] > residuals[16]
    assert residuals[16] < 1e-5


def test_running_twice_gives_the_same_answer() -> None:
    """What a loop carries between passes is forgotten between runs."""
    graph, meter, _ = drop_graph(return_loss=RETURN_LOSS, passes=6)
    first = graph.run()[meter]
    second = graph.run()[meter]
    assert [b.power_dbm for b in first.bands] == [b.power_dbm for b in second.bands]


def test_progress_counts_every_pass() -> None:
    """A run with a loop does passes times the work, and the bar has to know it."""
    totals = []
    for passes in (1, 3):
        graph, _, _ = drop_graph(return_loss=RETURN_LOSS, passes=passes)
        seen: list[Progress] = []
        graph.run(progress=seen.append)
        totals.append(seen[-1].total)
        assert seen[-1].fraction == 1.0
    assert totals[1] == 3 * totals[0]


def test_a_graph_without_a_loop_control_runs_exactly_once() -> None:
    """Every other block reports no passes, so nothing that existed moves."""
    graph, _, _ = drop_graph(return_loss=0.0, passes=None)
    assert all(component.feedback_passes() == 0 for component in graph.components)


def test_the_example_prints_the_round_trip_gain(capsys: pytest.CaptureFixture[str]) -> None:
    """The table in ``examples/fbg_circulator.py`` says what these tests assert."""
    import fbg_circulator

    fbg_circulator.an_echo()
    printed = capsys.readouterr().out
    assert "refused" in printed
    assert "= 0.0966" in printed
    assert "-1.751dBm" in printed
