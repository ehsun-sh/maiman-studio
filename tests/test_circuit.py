"""The scattering-matrix reduction: does eliminating internal ports give the right answer?

Nothing here is about photonics. What is under test is the linear identity in
:meth:`maiman.circuit.Circuit.solve` — that a circuit with feedback can be
reduced to a matrix over its outward-facing ports, exactly, in one step.

The strongest test in this file is
``test_the_reduction_agrees_with_summing_round_trips``, because summing round
trips is a *different algorithm* for the same answer: it walks the light around
the loop one lap at a time and adds the laps up, which converges to what the
reduction produces in closed form. Two methods that share no code agreeing to
thirteen digits is worth more than any number of assertions about one of them.
"""

from __future__ import annotations

import numpy as np
import pytest

from maiman.circuit import Circuit, SMatrix

GRID = np.linspace(193.0e12, 193.4e12, 64)


def constant(ports: tuple[str, ...], entries: dict[tuple[str, str], complex]) -> SMatrix:
    """A frequency-flat device, written as ``{(out, in): amplitude}``."""
    order = {name: i for i, name in enumerate(ports)}
    s = np.zeros((GRID.shape[0], len(ports), len(ports)), dtype=np.complex128)
    for (out, in_), value in entries.items():
        s[:, order[out], order[in_]] = value
    return SMatrix(ports=ports, frequencies=GRID, s=s)


def line(transfer: np.ndarray | complex, ports: tuple[str, str] = ("a", "b")) -> SMatrix:
    """A matched, reciprocal two-port with the given transmission."""
    s = np.zeros((GRID.shape[0], 2, 2), dtype=np.complex128)
    s[:, 0, 1] = transfer
    s[:, 1, 0] = transfer
    return SMatrix(ports=ports, frequencies=GRID, s=s)


def coupler(kappa: float) -> SMatrix:
    """The standard unitary 2x2, ports (i1, i2, o1, o2)."""
    t, k = np.sqrt(1.0 - kappa), 1j * np.sqrt(kappa)
    return constant(
        ("i1", "i2", "o1", "o2"),
        {
            ("o1", "i1"): t,
            ("o2", "i1"): k,
            ("o1", "i2"): k,
            ("o2", "i2"): t,
            ("i1", "o1"): t,
            ("i1", "o2"): k,
            ("i2", "o1"): k,
            ("i2", "o2"): t,
        },
    )


# ---------------------------------------------------------------------------
# the reduction itself


def test_a_circuit_with_nothing_wired_returns_what_was_put_in() -> None:
    """The degenerate case has to be right before the interesting one can be."""
    device = line(0.5 + 0.25j)
    circuit = Circuit().add("only", device).expose("x", "only", "a").expose("y", "only", "b")
    solved = circuit.solve()

    assert solved.ports == ("x", "y")
    assert np.allclose(solved.s, device.s)


def test_two_devices_in_series_multiply() -> None:
    """One internal connection, and the answer is the product. It has to be."""
    first, second = line(0.6 - 0.3j), line(-0.2 + 0.8j)
    solved = (
        Circuit()
        .add("one", first)
        .add("two", second)
        .link("one", "b", "two", "a")
        .expose("in", "one", "a")
        .expose("out", "two", "b")
        .solve()
    )
    expected = (0.6 - 0.3j) * (-0.2 + 0.8j)
    assert np.allclose(solved.transmission("out", "in"), expected)
    assert np.allclose(solved.transmission("in", "out"), expected)


def test_the_reduction_agrees_with_summing_round_trips() -> None:
    """A ring, solved twice by methods that share no code.

    The reduction inverts ``I - C S_ii`` once. The series walks the light round
    the loop lap by lap — coupled in, round n times, coupled back out — and adds
    the laps. They are the same number, and the series is what makes the closed
    form mean something rather than merely be self-consistent.
    """
    kappa = 0.12
    t = np.sqrt(1.0 - kappa)
    loop = 0.97 * np.exp(-1j * np.linspace(0.0, 6.0 * np.pi, GRID.shape[0]))

    solved = (
        Circuit()
        .add("c", coupler(kappa))
        .add("loop", line(loop))
        .link("c", "o2", "loop", "a")
        .link("loop", "b", "c", "i2")
        .expose("in", "c", "i1")
        .expose("through", "c", "o1")
        .solve()
    )

    # Lap by lap: straight through, plus every path that couples in, goes round
    # n times, and couples back out.
    across = 1j * np.sqrt(kappa)
    series = np.full(GRID.shape[0], t, dtype=np.complex128)
    for laps in range(1, 400):
        series = series + across * loop**laps * t ** (laps - 1) * across

    assert np.max(np.abs(solved.transmission("through", "in") - series)) < 1e-13


def test_the_loop_gain_shows_up_as_resonance() -> None:
    """The feedback has to actually do something, or the test above proves nothing.

    Matched to the round trip, so the light coupled back out of the loop cancels
    the light that stayed on the bus exactly. The grid starts at zero phase and
    therefore lands on the resonance rather than near it.
    """
    survives = 0.99
    kappa = 1.0 - survives**2  # the coupling that cancels: t == a
    phases = np.linspace(0.0, 2.0 * np.pi, GRID.shape[0])
    solved = (
        Circuit()
        .add("c", coupler(kappa))
        .add("loop", line(survives * np.exp(-1j * phases)))
        .link("c", "o2", "loop", "a")
        .link("loop", "b", "c", "i2")
        .expose("in", "c", "i1")
        .expose("through", "c", "o1")
        .solve()
    )
    power = solved.power("through", "in")
    # Extinguished on resonance, near unity between: a ring, not an attenuator.
    assert power.min() < 1e-25
    assert power.max() > 0.99


def test_a_lossless_circuit_stays_lossless() -> None:
    """Unitarity in, unitarity out — including through the feedback."""
    solved = (
        Circuit()
        .add("a", coupler(0.3))
        .add("b", coupler(0.7))
        .add("arm", line(np.exp(-1j * np.linspace(0.0, 4.0, GRID.shape[0]))))
        .link("a", "o1", "arm", "a")
        .link("arm", "b", "b", "i1")
        .expose("in1", "a", "i1")
        .expose("in2", "a", "i2")
        .expose("cross", "a", "o2")
        .expose("out1", "b", "o1")
        .expose("out2", "b", "o2")
        .expose("back", "b", "i2")
        .solve()
    )
    assert solved.is_unitary(tol=1e-12)


# ---------------------------------------------------------------------------
# what the reduction does not assume


def test_a_dangling_port_swallows_light_instead_of_reflecting_it() -> None:
    """An unwired port is a facet nothing drives, and ``a = 0`` is its condition.

    Half the light entering the coupler leaves by the port nobody connected. If
    the reduction treated that port as a short or a mirror it would come back and
    show up at the output, and the through path would not be the bare coupler's.
    """
    solved = (
        Circuit().add("c", coupler(0.5)).expose("in", "c", "i1").expose("out", "c", "o1").solve()
    )
    assert np.allclose(solved.power("out", "in"), 0.5)
    assert np.allclose(solved.power("in", "in"), 0.0)


def test_a_non_reciprocal_device_solves_non_reciprocally() -> None:
    """An isolator in a loop. Nothing in the reduction symmetrises anything."""
    isolator = constant(("a", "b"), {("b", "a"): 1.0})  # forward only
    solved = (
        Circuit()
        .add("iso", isolator)
        .add("wire", line(0.8))
        .link("iso", "b", "wire", "a")
        .expose("in", "iso", "a")
        .expose("out", "wire", "b")
        .solve()
    )
    assert np.allclose(solved.transmission("out", "in"), 0.8)
    assert np.allclose(solved.transmission("in", "out"), 0.0)


def test_a_reflection_inside_the_circuit_comes_back_out() -> None:
    """A mirror at the far end of a delay: the round trip is the square.

    Diagonal terms are what a reflecting device has, and a reduction that quietly
    assumed matched ports would return zero here.
    """
    delay = np.exp(-1j * np.linspace(0.0, 3.0, GRID.shape[0]))
    mirror = constant(("face",), {("face", "face"): 0.9})
    solved = (
        Circuit()
        .add("wg", line(delay))
        .add("m", mirror)
        .link("wg", "b", "m", "face")
        .expose("in", "wg", "a")
        .solve()
    )
    assert np.allclose(solved.transmission("in", "in"), 0.9 * delay**2)


def test_the_answer_does_not_depend_on_the_order_things_were_added() -> None:
    """Port bookkeeping is internal, so it must not be observable."""
    kappa = 0.2
    loop = 0.95 * np.exp(-1j * np.linspace(0.0, 5.0, GRID.shape[0]))

    def build(reverse: bool) -> SMatrix:
        circuit = Circuit()
        parts = [("loop", line(loop)), ("c", coupler(kappa))]
        for name, matrix in reversed(parts) if reverse else parts:
            circuit.add(name, matrix)
        if reverse:
            circuit.link("loop", "b", "c", "i2").link("c", "o2", "loop", "a")
        else:
            circuit.link("c", "o2", "loop", "a").link("loop", "b", "c", "i2")
        return circuit.expose("in", "c", "i1").expose("out", "c", "o1").solve()

    assert np.allclose(build(False).s, build(True).s)


# ---------------------------------------------------------------------------
# refusals


def test_devices_on_different_grids_are_refused() -> None:
    """Two answers to different questions must not be silently combined."""
    other = SMatrix(
        ports=("a", "b"),
        frequencies=GRID + 1e9,
        s=np.zeros((GRID.shape[0], 2, 2), dtype=np.complex128),
    )
    circuit = Circuit().add("one", line(0.5)).add("two", other)
    circuit.expose("in", "one", "a")
    with pytest.raises(ValueError, match="same frequency grid"):
        circuit.solve()


def test_a_circuit_with_no_exposed_ports_is_refused() -> None:
    circuit = Circuit().add("only", line(0.5))
    with pytest.raises(ValueError, match="nothing to solve for"):
        circuit.solve()


@pytest.mark.parametrize(
    ("action", "message"),
    [
        (lambda c: c.link("one", "b", "two", "a"), "already linked"),
        (lambda c: c.expose("x", "one", "b"), "cannot also be exposed"),
        (lambda c: c.link("one", "a", "one", "a"), "to itself"),
        (lambda c: c.expose("in", "two", "b"), "already an exposed port"),
        (lambda c: c.link("one", "a", "two", "b"), "cannot also be linked"),
    ],
)
def test_wiring_mistakes_are_refused(action: object, message: str) -> None:
    """Every one of these would otherwise produce a circuit that is not the drawn one."""
    circuit = Circuit().add("one", line(0.5)).add("two", line(0.5))
    circuit.link("one", "b", "two", "a")
    circuit.expose("in", "one", "a")
    with pytest.raises(ValueError, match=message):
        action(circuit)  # type: ignore[operator]


def test_naming_a_port_that_does_not_exist_is_refused() -> None:
    circuit = Circuit().add("one", line(0.5))
    with pytest.raises(KeyError, match="no port"):
        circuit.expose("in", "one", "nowhere")
    with pytest.raises(KeyError, match="no instance"):
        circuit.expose("in", "elsewhere", "a")


def test_a_matrix_of_the_wrong_shape_is_refused() -> None:
    with pytest.raises(ValueError, match="must have shape"):
        SMatrix(ports=("a", "b"), frequencies=GRID, s=np.zeros((GRID.shape[0], 3, 3)))
    with pytest.raises(ValueError, match="unique"):
        SMatrix(
            ports=("a", "a"),
            frequencies=GRID,
            s=np.zeros((GRID.shape[0], 2, 2), dtype=np.complex128),
        )


def test_transmission_names_the_ports_it_has() -> None:
    solved = Circuit().add("one", line(0.5)).expose("in", "one", "a").solve()
    with pytest.raises(KeyError, match="have \\['in'\\]"):
        solved.transmission("out", "in")
