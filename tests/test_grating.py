"""The fibre Bragg grating, the circulator, and the scheduler change that wires them.

The grating is *assembled* — a product of per-section transfer matrices — for the
same reason the ring is, so Erdogan's closed form for a uniform grating lives
here, on the other side of the comparison. A uniform grating cut into two hundred
sections has to reproduce a formula that appears nowhere in the model.

The rest is the part that is not physics: a reflective device reached through a
circulator is a cycle in the component graph and is not a cycle in the light, and
these check that the scheduler now tells the two apart.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar

import numpy as np
import pytest

from maiman import CycleError, Graph, GraphError, PortGroup, SimulationContext
from maiman.component import Component, PortType
from maiman.components import (
    Attenuator,
    Circulator,
    Combiner,
    CWLaser,
    Fiber,
    FiberBraggGrating,
    GaussianPulse,
    PowerMeter,
)
from maiman.photonics import (
    APODIZATIONS,
    SILICA_FIBER_NEFF,
    circulator,
    fiber_bragg_grating,
    phase_shift_profile,
    sampled_profile,
    section_positions,
)
from maiman.signals import Signal
from maiman.units import C_LIGHT

BRAGG = 1550e-9


def erdogan_uniform(
    wavelengths: np.ndarray, *, length: float, index_modulation: float, n_eff: float
) -> np.ndarray:
    """Reflectivity of a uniform grating, from the literature rather than the model.

    Erdogan, *Fiber Grating Spectra*, J. Lightwave Technol. 15(8), 1997, eq. 16,
    with the average index taken as compensated so the self-coupling term is the
    detuning alone. Written out in full here precisely so that it shares no
    arithmetic with :func:`maiman.photonics.fiber_bragg_grating`, which builds
    the same answer by multiplying two hundred matrices together.
    """
    detuning = 2.0 * np.pi * n_eff * (1.0 / wavelengths - 1.0 / BRAGG)
    kappa = np.pi * index_modulation / wavelengths
    gamma = np.sqrt((kappa**2 - detuning**2).astype(np.complex128))
    numerator = -kappa * np.sinh(gamma * length)
    denominator = detuning * np.sinh(gamma * length) + 1j * gamma * np.cosh(gamma * length)
    return np.abs(numerator / denominator) ** 2


def group_delay(matrix, port: str) -> np.ndarray:
    """Group delay of one reflection [s], from the phase the model returns.

    ``tau = -dphi/domega`` is the sign this library's ``exp(-i beta z)``
    convention gives, and it is checked against a plain waveguide in
    ``test_the_group_delay_convention_is_the_one_a_waveguide_sets``.
    """
    response = matrix.transmission(port, port)
    return -np.gradient(np.unwrap(np.angle(response)), 2.0 * np.pi * matrix.frequencies)


# --------------------------------------------------------------------------
# The grating, against the closed form it is not built from
# --------------------------------------------------------------------------


@pytest.mark.parametrize("index_modulation", [1e-5, 5e-5, 1e-4, 5e-4])
def test_a_uniform_grating_reproduces_the_coupled_mode_closed_form(
    index_modulation: float,
) -> None:
    """Two hundred matrices multiplied together, against one formula.

    Across two decades of grating strength, so it is the construction being
    checked and not one lucky operating point.
    """
    wavelengths = np.linspace(BRAGG - 1e-9, BRAGG + 1e-9, 2001)
    matrix = fiber_bragg_grating(
        C_LIGHT / wavelengths,
        length=0.01,
        index_modulation=index_modulation,
        bragg_wavelength=BRAGG,
    )
    predicted = erdogan_uniform(
        wavelengths, length=0.01, index_modulation=index_modulation, n_eff=SILICA_FIBER_NEFF
    )
    # Two hundred complex 2x2 products accumulate rounding; the agreement is at
    # 1e-10 relative, which on a quantity bounded by one is the double-precision
    # floor rather than a modelling difference.
    assert matrix.power("in", "in") == pytest.approx(predicted, abs=1e-9)


def test_peak_reflectivity_is_tanh_squared_of_kappa_l() -> None:
    """The one number that decides how much a grating returns."""
    for length, index_modulation in ((0.005, 1e-4), (0.01, 1e-4), (0.02, 5e-5)):
        kappa = np.pi * index_modulation / BRAGG
        matrix = fiber_bragg_grating(
            np.array([C_LIGHT / BRAGG]),
            length=length,
            index_modulation=index_modulation,
            bragg_wavelength=BRAGG,
        )
        assert matrix.power("in", "in")[0] == pytest.approx(np.tanh(kappa * length) ** 2, rel=1e-12)


def test_the_grating_conserves_energy_everywhere() -> None:
    """``R + T = 1``. There is no loss in the model and there had better be none
    in the arithmetic either — a transfer-matrix product that drifts off unity is
    the classic way this method fails, and it fails quietly."""
    wavelengths = np.linspace(BRAGG - 2e-9, BRAGG + 2e-9, 4001)
    matrix = fiber_bragg_grating(
        C_LIGHT / wavelengths, length=0.02, index_modulation=2e-4, bragg_wavelength=BRAGG
    )
    total = matrix.power("in", "in") + matrix.power("out", "in")
    assert total == pytest.approx(np.ones_like(total), abs=1e-10)


def test_the_first_nulls_sit_where_the_bandwidth_formula_says() -> None:
    """``lambda**2 / (pi n L) sqrt((kappa L)**2 + pi**2)``, the standard width.

    Measured off the model by finding the minima either side of the peak, which
    is what an instrument would do.
    """
    length, index_modulation = 0.01, 1e-4
    kappa_l = np.pi * index_modulation / BRAGG * length
    predicted = BRAGG**2 / (np.pi * SILICA_FIBER_NEFF * length) * np.sqrt(kappa_l**2 + np.pi**2)

    wavelengths = np.linspace(BRAGG - 2e-9, BRAGG + 2e-9, 20001)
    reflectivity = fiber_bragg_grating(
        C_LIGHT / wavelengths,
        length=length,
        index_modulation=index_modulation,
        bragg_wavelength=BRAGG,
    ).power("in", "in")

    peak = int(np.argmax(reflectivity))
    assert wavelengths[peak] == pytest.approx(BRAGG, abs=2e-13)

    minima = [
        i
        for i in range(1, len(reflectivity) - 1)
        if reflectivity[i] < reflectivity[i - 1] and reflectivity[i] <= reflectivity[i + 1]
    ]
    width = (
        wavelengths[min(i for i in minima if i > peak)]
        - wavelengths[max(i for i in minima if i < peak)]
    )
    # The grid is 0.2 pm and the width 198 pm, so a per-mille agreement is the
    # resolution of the measurement rather than of the model.
    assert width == pytest.approx(predicted, rel=2e-3)


def test_apodization_buys_sidelobe_suppression_with_reflectivity() -> None:
    """Both halves of the trade, as numbers.

    A uniform grating's spectrum is a rectangle's transform and has a rectangle's
    sidelobes, which is far too much crosstalk to drop a channel with. Shaping
    the coupling removes them and lowers the average coupling at the same time,
    so the peak comes down as well — there is no free version of this.
    """
    wavelengths = np.linspace(BRAGG - 2e-9, BRAGG + 2e-9, 40001)
    measured = {}
    for profile in APODIZATIONS:
        reflectivity = fiber_bragg_grating(
            C_LIGHT / wavelengths,
            length=0.01,
            index_modulation=1e-4,
            bragg_wavelength=BRAGG,
            apodization=profile,
        ).power("in", "in")
        peak = int(np.argmax(reflectivity))
        sidelobes = [
            i
            for i in range(peak + 1, len(reflectivity) - 1)
            if reflectivity[i] > reflectivity[i - 1] and reflectivity[i] >= reflectivity[i + 1]
        ]
        first = 10.0 * np.log10(reflectivity[sidelobes[0]] / reflectivity[peak])
        measured[profile] = (reflectivity[peak], float(first))

    uniform_peak, uniform_sidelobe = measured["uniform"]
    cosine_peak, cosine_sidelobe = measured["raised-cosine"]
    gaussian_peak, gaussian_sidelobe = measured["gaussian"]

    assert uniform_sidelobe == pytest.approx(-7.6, abs=0.5)
    assert cosine_sidelobe == pytest.approx(-31.4, abs=1.0)
    assert gaussian_sidelobe == pytest.approx(-41.4, abs=1.0)

    # Monotone in both directions: quieter skirts, weaker peak.
    assert gaussian_sidelobe < cosine_sidelobe < uniform_sidelobe
    assert gaussian_peak < uniform_peak
    assert cosine_peak < uniform_peak


def test_a_grating_reflects_where_it_is_told_to() -> None:
    """The Bragg wavelength is a parameter and not an approximation of one."""
    for target in (1530e-9, 1550e-9, 1565e-9):
        wavelengths = np.linspace(target - 1e-9, target + 1e-9, 4001)
        reflectivity = fiber_bragg_grating(
            C_LIGHT / wavelengths,
            length=0.01,
            index_modulation=1e-4,
            bragg_wavelength=target,
        ).power("in", "in")
        assert wavelengths[int(np.argmax(reflectivity))] == pytest.approx(target, abs=1e-12)


# --------------------------------------------------------------------------
# Chirp: the part with no closed form, against the geometric estimate
# --------------------------------------------------------------------------


def test_the_group_delay_convention_is_the_one_a_waveguide_sets() -> None:
    """``tau = -dphi/domega``, pinned on a device whose delay is known exactly.

    This exists because getting it wrong is *not* obvious: the coupled-mode
    literature carries the opposite sign convention, and a grating built with
    Erdogan's sign returns correct reflectivities and negative delays. A mirror
    that answers before it is asked is the symptom, and this is the test that
    names the convention it is measured against.
    """
    from maiman.photonics import straight_waveguide

    frequencies = C_LIGHT / np.linspace(1549.9e-9, 1550.1e-9, 2001)
    guide = straight_waveguide(
        frequencies,
        length=0.10,
        n_eff=SILICA_FIBER_NEFF,
        n_group=SILICA_FIBER_NEFF,
        reference_frequency=C_LIGHT / BRAGG,
    )
    phase = np.unwrap(np.angle(guide.transmission("out", "in")))
    delay = -np.gradient(phase, 2.0 * np.pi * frequencies)
    assert delay.mean() == pytest.approx(SILICA_FIBER_NEFF * 0.10 / C_LIGHT, rel=1e-6)
    assert delay.min() > 0.0


def test_a_chirped_grating_has_the_geometric_dispersion() -> None:
    """``2 n L / (c * chirp)``: the round trip spread over the band it is chirped
    across, which is the whole idea of the device.

    Measured over the flat middle of the reflection band, well inside the regime
    where the chirp dominates the grating's own bandwidth.
    """
    # The two requirements pull against each other and this is the balance
    # point. Reflecting everything wants a strong grating; a *geometric* delay
    # wants the chirp to dominate the grating's own bandwidth, which a strong one
    # widens. At 3e-4 over 10 cm chirped across 4 nm the chirp is twelve times the
    # bandwidth, the band is full, and the slope is 0.3 % off geometric.
    length, chirp = 0.10, 4e-9
    predicted = 2.0 * SILICA_FIBER_NEFF * length / (C_LIGHT * chirp)

    wavelengths = np.linspace(BRAGG - 0.3 * chirp, BRAGG + 0.3 * chirp, 4001)
    matrix = fiber_bragg_grating(
        C_LIGHT / wavelengths,
        length=length,
        index_modulation=3e-4,
        bragg_wavelength=BRAGG,
        chirp=chirp,
    )
    assert matrix.power("in", "in").min() > 0.99

    slope = np.polyfit(wavelengths, group_delay(matrix, "in"), 1)[0]
    assert slope == pytest.approx(predicted, rel=5e-3)
    assert slope > 0.0


def test_the_two_ends_of_a_chirped_grating_are_different_devices() -> None:
    """Same magnitude, opposite dispersion — which is why the matrix carries both
    diagonal entries rather than assuming a mirror is symmetric.

    It is also the practical point: a compensator is this grating turned round,
    and if the model could not tell the ends apart it could not say so.
    """
    length, chirp = 0.10, 4e-9
    wavelengths = np.linspace(BRAGG - 0.3 * chirp, BRAGG + 0.3 * chirp, 4001)
    matrix = fiber_bragg_grating(
        C_LIGHT / wavelengths,
        length=length,
        index_modulation=3e-4,
        bragg_wavelength=BRAGG,
        chirp=chirp,
    )
    near = np.polyfit(wavelengths, group_delay(matrix, "in"), 1)[0]
    far = np.polyfit(wavelengths, group_delay(matrix, "out"), 1)[0]
    assert far == pytest.approx(-near, rel=1e-3)

    # Reciprocity survives it: the *transmission* is one number either way, even
    # though the reflections are not.
    assert matrix.transmission("out", "in") == pytest.approx(
        matrix.transmission("in", "out"), abs=1e-12
    )


def test_a_uniform_grating_is_the_same_device_from_both_ends() -> None:
    """The case where the asymmetry above must vanish."""
    wavelengths = np.linspace(BRAGG - 1e-9, BRAGG + 1e-9, 2001)
    matrix = fiber_bragg_grating(
        C_LIGHT / wavelengths, length=0.01, index_modulation=1e-4, bragg_wavelength=BRAGG
    )
    assert matrix.transmission("in", "in") == pytest.approx(
        matrix.transmission("out", "out"), abs=1e-12
    )


def test_the_section_count_has_converged_by_the_default() -> None:
    """The default is a measurement, so this is the measurement.

    Fifty sections are visibly wrong on a long chirped grating and two hundred
    are not, and nothing above two hundred moves.
    """
    length, chirp = 1.0, 40e-9
    predicted = 2.0 * SILICA_FIBER_NEFF * length / (C_LIGHT * chirp)
    wavelengths = np.linspace(BRAGG - 0.05 * chirp, BRAGG + 0.05 * chirp, 40001)

    def slope(sections: int) -> float:
        matrix = fiber_bragg_grating(
            C_LIGHT / wavelengths,
            length=length,
            index_modulation=2e-4,
            bragg_wavelength=BRAGG,
            chirp=chirp,
            sections=sections,
        )
        return float(np.polyfit(wavelengths, group_delay(matrix, "in"), 1)[0])

    assert abs(slope(50) / predicted - 1.0) > 5e-3
    assert slope(200) == pytest.approx(predicted, rel=2e-3)
    assert slope(2000) == pytest.approx(predicted, rel=2e-3)


def test_an_unmodulated_grating_is_a_piece_of_fibre() -> None:
    """Zero index modulation is the degenerate case, and it must not be a nan.

    ``sinh(gamma dz) / gamma`` is 0/0 exactly on resonance when there is nothing
    to couple, and the limit is ``dz``.
    """
    wavelengths = np.linspace(BRAGG - 1e-12, BRAGG + 1e-12, 5)
    matrix = fiber_bragg_grating(
        C_LIGHT / wavelengths, length=0.01, index_modulation=0.0, bragg_wavelength=BRAGG
    )
    assert np.all(np.isfinite(matrix.s))
    assert matrix.power("in", "in") == pytest.approx(np.zeros(5), abs=1e-12)
    assert matrix.power("out", "in") == pytest.approx(np.ones(5), abs=1e-12)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"length": 0.0}, "length must be positive"),
        ({"bragg_wavelength": 0.0}, "bragg_wavelength must be positive"),
        ({"sections": 0}, "sections must be at least 1"),
        ({"apodization": "hamming"}, "unknown apodization"),
    ],
)
def test_the_model_refuses_what_it_cannot_describe(kwargs: dict, message: str) -> None:
    settings = {
        "length": 0.01,
        "index_modulation": 1e-4,
        "bragg_wavelength": BRAGG,
        **kwargs,
    }
    with pytest.raises(ValueError, match=message):
        fiber_bragg_grating(np.array([C_LIGHT / BRAGG]), **settings)


# --------------------------------------------------------------------------
# The circulator
# --------------------------------------------------------------------------


def test_a_lossless_circulator_is_unitary_and_is_not_symmetric() -> None:
    """The two properties that make it a circulator rather than a splitter.

    Unitary says it conserves energy; asymmetric says it has a direction. A
    device with the first and not the second is a coupler, and the whole reason
    this part exists is the second.
    """
    matrix = circulator(np.array([C_LIGHT / BRAGG]))
    assert matrix.is_unitary()
    assert not np.allclose(matrix.s, np.swapaxes(matrix.s, -1, -2))

    assert matrix.power("p2", "p1")[0] == pytest.approx(1.0)
    assert matrix.power("p3", "p2")[0] == pytest.approx(1.0)
    assert matrix.power("p1", "p3")[0] == pytest.approx(1.0)
    # And nothing at all the other way round.
    assert matrix.power("p1", "p2")[0] == 0.0
    assert matrix.power("p2", "p3")[0] == 0.0
    assert matrix.power("p3", "p1")[0] == 0.0


def test_circulator_loss_is_per_hop() -> None:
    """Which is how a datasheet quotes it, and it means a round trip pays twice."""
    matrix = circulator(np.array([C_LIGHT / BRAGG]), insertion_loss_db=0.7)
    assert 10.0 * np.log10(matrix.power("p2", "p1")[0]) == pytest.approx(-0.7, abs=1e-9)
    assert not matrix.is_unitary()


@pytest.mark.parametrize("driven", [(1,), (1, 2), (1, 2, 3)])
def test_the_circulator_block_declares_one_port_group_per_hop(
    driven: tuple[int, ...],
) -> None:
    """Each route is its own scheduler node, and the groups partition the ports."""
    block = Circulator(driven=driven)
    groups = block.port_groups()
    assert len(groups) == len(driven)
    block.check_port_groups()
    for group in groups:
        assert len(group.inputs) == 1
        assert len(group.outputs) == 1


@pytest.mark.parametrize(
    ("driven", "message"),
    [
        ((), "no driven port"),
        ((1, 1), "lists a port twice"),
        ((0, 1), "has ports 1, 2 and 3"),
        ((1, 4), "has ports 1, 2 and 3"),
    ],
)
def test_the_circulator_refuses_an_impossible_port_set(
    driven: tuple[int, ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        Circulator(driven=driven)


# --------------------------------------------------------------------------
# The scheduler: a reflective device is not a feedback loop
# --------------------------------------------------------------------------


def test_a_grating_reached_through_a_circulator_is_not_a_cycle() -> None:
    """The configuration this whole piece of work exists for.

    ``circ -> fbg -> circ`` is a cycle in the component graph and is not a cycle
    in the light: ``out3`` depends on ``in2`` and on nothing else. Before port
    groups this raised :class:`CycleError`.

    Every number below is arithmetic anyone can check: the drop port is two
    circulator hops and the grating's peak reflectivity, and the express port is
    one hop and its transmission.
    """
    ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=8, sequence_length=64)
    graph = Graph(ctx)
    dropped = graph.add(CWLaser(power=0.0, wavelength=1550.0, label="dropped"))
    express = graph.add(CWLaser(power=0.0, wavelength=1552.0, label="express"))
    mux = graph.add(Combiner(2, label="mux"))
    circ = graph.add(Circulator(insertion_loss=0.7, label="circ"))
    grating = graph.add(
        FiberBraggGrating(bragg_wavelength=1550.0, length=10.0, index_modulation=1e-4)
    )
    drop_meter = graph.add(PowerMeter(label="drop"))
    thru_meter = graph.add(PowerMeter(label="thru"))

    graph.connect(dropped, mux["in0"])
    graph.connect(express, mux["in1"])
    graph.connect(mux, circ["in1"])
    graph.connect(circ["out2"], grating["in"])
    graph.connect(grating["reflected"], circ["in2"])
    graph.connect(circ["out3"], drop_meter["in"])
    graph.connect(grating["transmitted"], thru_meter["in"])

    results = graph.run()
    drop = results[drop_meter]
    thru = results[thru_meter]

    def at(reading, wavelength_nm: float) -> float:
        (band,) = [b for b in reading.bands if round(b.wavelength_nm, 2) == wavelength_nm]
        return band.power_dbm

    reflectivity = grating.peak_reflectivity()
    assert reflectivity == pytest.approx(0.9329, abs=1e-4)

    # Two hops of 0.7 dB, then what the grating returns.
    assert at(drop, 1550.0) == pytest.approx(-1.4 + 10.0 * np.log10(reflectivity), abs=1e-3)
    # One hop, then what it lets past — and the two must still add to one.
    assert at(thru, 1550.0) == pytest.approx(-0.7 + 10.0 * np.log10(1.0 - reflectivity), abs=1e-3)
    # The neighbour goes straight past, having paid one hop and nothing else.
    assert at(thru, 1552.0) == pytest.approx(-0.7, abs=1e-3)
    # And is 40 dB down on the drop port, which is the grating's own sidelobe
    # floor two nanometres out rather than an isolation figure anyone declared.
    assert at(drop, 1552.0) < -40.0


def test_a_real_feedback_loop_is_still_refused() -> None:
    """Port groups must not have turned the cycle check off.

    Two attenuators wired to each other have no independent groups to split into
    and are a genuine loop, which the scheduler has always refused and still does.
    """
    ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=8, sequence_length=16)
    graph = Graph(ctx)
    first = graph.add(Attenuator(label="a"))
    second = graph.add(Attenuator(label="b"))
    graph.connect(first, second)
    graph.connect(second, first)
    with pytest.raises(CycleError, match="feedback loop"):
        graph.run()


def test_a_loop_through_one_circulator_hop_is_still_refused() -> None:
    """And the harder case: a cycle that goes through a *single* port group.

    Splitting a component into independent nodes cannot launder a loop that is
    inside one of them. Light out of port 2 comes back into port 2, which is the
    same route, and there is nothing to order it against.
    """
    ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=8, sequence_length=16)
    graph = Graph(ctx)
    laser = graph.add(CWLaser(label="laser"))
    mux = graph.add(Combiner(2, label="mux"))
    circ = graph.add(Circulator(driven=(1, 2, 3), label="circ"))
    graph.connect(laser, mux["in0"])
    graph.connect(mux, circ["in1"])
    # Round the ring and back into the combiner that feeds port 1: light leaves
    # port 2, returns at port 2, goes out of 3, back in at 3, out of 1 and into
    # the mux. Every hop is its own group and the loop still closes through all
    # of them, which is what a recirculating cavity is.
    graph.connect(circ["out2"], circ["in2"])
    graph.connect(circ["out3"], circ["in3"])
    graph.connect(circ["out1"], mux["in1"])
    with pytest.raises(CycleError, match="feedback loop"):
        graph.run()


# --------------------------------------------------------------------------
# The port-group contract itself
# --------------------------------------------------------------------------


class _Split(Component):
    """A block whose two halves genuinely do not touch, for testing the contract.

    ``abstract`` because it is never placed in a graph — the partition checks
    below call :meth:`~maiman.component.Component.check_port_groups` directly.
    Without it, defining this class would register it, and every test that walks
    the whole library would see a component that only exists in this file.
    """

    abstract = True
    display_name = "Split"
    category = "Test"
    inputs: ClassVar[dict[str, PortType]] = {"a": PortType.OPTICAL, "b": PortType.OPTICAL}
    outputs: ClassVar[dict[str, PortType]] = {"x": PortType.OPTICAL, "y": PortType.OPTICAL}

    #: Overridden per instance by the tests below, to produce a bad partition.
    groups: tuple[PortGroup, ...] | None = None

    def port_groups(self) -> tuple[PortGroup, ...]:
        if self.groups is not None:
            return self.groups
        return (
            PortGroup(frozenset({"a"}), frozenset({"x"})),
            PortGroup(frozenset({"b"}), frozenset({"y"})),
        )

    def run(self, ctx: SimulationContext, inputs: dict[str, Signal]) -> dict[str, Signal]:
        return {"x" if name == "a" else "y": signal for name, signal in inputs.items()}


@pytest.mark.parametrize(
    ("groups", "message"),
    [
        ((), "returned nothing"),
        (
            (PortGroup(frozenset({"a", "b"}), frozenset()),),
            "produces no output",
        ),
        (
            (PortGroup(frozenset({"a"}), frozenset({"x", "y"})),),
            r"leaves input\(s\) \['b'\] out",
        ),
        (
            (
                PortGroup(frozenset({"a", "b"}), frozenset({"x"})),
                PortGroup(frozenset({"a"}), frozenset({"y"})),
            ),
            "belongs to exactly one",
        ),
        (
            (PortGroup(frozenset({"a", "b"}), frozenset({"x", "y", "z"})),),
            "which is not one of its outputs",
        ),
    ],
)
def test_port_groups_must_partition_the_ports(groups: tuple[PortGroup, ...], message: str) -> None:
    """A port in no group never runs and a port in two runs twice, and both look
    like a wiring bug somewhere else entirely. Caught before the run."""
    block = _Split()
    block.groups = groups
    with pytest.raises(ValueError, match=message):
        block.check_port_groups()


def test_the_default_is_one_group_holding_everything() -> None:
    """Which is the honest answer for every block that computes all its outputs
    from all its inputs in one call — that is, almost all of them."""
    grating = FiberBraggGrating()
    (group,) = grating.port_groups()
    assert group.inputs == frozenset(grating.inputs)
    assert group.outputs == frozenset(grating.outputs)


def test_a_group_that_returns_the_wrong_ports_is_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The per-group completeness check. Returning the other group's output here
    would overwrite a result computed from inputs this call was never given."""
    ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=8, sequence_length=16)
    graph = Graph(ctx)
    laser = graph.add(CWLaser(label="laser"))
    circ = graph.add(Circulator(driven=(1,), label="circ"))
    meter = graph.add(PowerMeter(label="meter"))
    graph.connect(laser, circ["in1"])
    graph.connect(circ["out2"], meter["in"])

    monkeypatch.setattr(Circulator, "run", lambda self, ctx, inputs: {})
    with pytest.raises(GraphError, match=r"did not return output\(s\) \['out2'\]"):
        graph.run()


def test_the_compensating_chirp_sign_is_pinned() -> None:
    """Which sign of chirp undoes a span, measured rather than reasoned about.

    The grating's own physics says a positive chirp gives ``D > 0`` — short
    wavelengths at the near end turn round first, so the long ones arrive later,
    which is the sign standard fibre has. By that reading a compensator is a
    *negative* chirp. Against this engine's :class:`~maiman.components.Fiber` it
    is the positive one, because
    :func:`maiman.kernels.propagate_dispersion` carries the opposite quadratic
    sign from :func:`maiman.photonics.propagation_constant` — a disagreement that
    function's docstring has described since before this device existed, and
    which the grating is merely the first component to make visible in one graph.

    This test is a pin, not an endorsement. Reconciling the kernel will invert
    the answer, and when it does this fails and names what else moved.
    """
    ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=64, sequence_length=256, seed=1)

    def rms_width(chirp_nm: float | None, length_km: float) -> float:
        """RMS width of the pulse [ps]."""
        graph = Graph(ctx)
        source = graph.add(GaussianPulse(width=30.0, peak_power=1.0, wavelength=1550.0))
        fiber = graph.add(Fiber(length=length_km, attenuation=0.0, dispersion=17.0))
        graph.connect(source, fiber["in"])
        if chirp_nm is None:
            meter = graph.add(PowerMeter(label="meter"))
            graph.connect(fiber, meter["in"])
            signal = graph.run(keep=[fiber]).port(fiber, "out")
        else:
            circ = graph.add(Circulator(insertion_loss=0.0, label="circ"))
            grating = graph.add(
                FiberBraggGrating(
                    "raised-cosine",
                    length=100.0,
                    index_modulation=2e-4,
                    bragg_wavelength=1550.0,
                    chirp=chirp_nm,
                )
            )
            meter = graph.add(PowerMeter(label="meter"))
            graph.connect(fiber, circ["in1"])
            graph.connect(circ["out2"], grating["in"])
            graph.connect(grating["reflected"], circ["in2"])
            graph.connect(circ["out3"], meter["in"])
            signal = graph.run(keep=[circ]).port(circ, "out3")

        (band,) = signal.bands
        power = np.abs(band.Ex) ** 2 + np.abs(band.Ey) ** 2
        # Centred first: the grating's own half-nanosecond of latency is a
        # circular shift here, and it is not what is being measured.
        power = np.roll(power, band.num_samples // 2 - int(np.argmax(power)))
        times = np.arange(band.num_samples) / band.fs
        centre = (power * times).sum() / power.sum()
        return float(np.sqrt(((times - centre) ** 2 * power).sum() / power.sum())) * 1e12

    launched = rms_width(None, 0.0)
    spread = rms_width(None, 80.0)
    restored = rms_width(+0.71, 80.0)
    worsened = rms_width(-0.71, 80.0)

    assert launched == pytest.approx(21.2, abs=0.3)
    assert spread == pytest.approx(46.1, abs=0.5)
    # Back to the launched width, which is what compensation means.
    assert restored == pytest.approx(launched, rel=0.02)
    # And the other sign adds what the fibre added, rather than removing it.
    assert worsened > spread


# --------------------------------------------------------------------------
# Arbitrary profiles: the devices the named windows cannot describe
# --------------------------------------------------------------------------


@pytest.mark.parametrize("profile", sorted(APODIZATIONS))
def test_a_named_window_and_its_array_are_the_same_device(profile: str) -> None:
    """Bit for bit, not approximately.

    The names have to *be* the arrays rather than resemble them, or the general
    path and the convenient one are two models and only one of them is tested.
    """
    wavelengths = np.linspace(BRAGG - 2e-9, BRAGG + 2e-9, 3001)
    settings = {"length": 0.01, "index_modulation": 1e-4, "bragg_wavelength": BRAGG}
    named = fiber_bragg_grating(C_LIGHT / wavelengths, apodization=profile, **settings)
    spelled = fiber_bragg_grating(
        C_LIGHT / wavelengths,
        coupling_profile=APODIZATIONS[profile](section_positions(200)),
        **settings,
    )
    assert np.array_equal(named.s, spelled.s)


def test_a_linear_chirp_and_its_array_are_the_same_device() -> None:
    """The same claim for the other profile, including which end is which.

    A chirp written as an array has to run the same way round as one written as
    a number, because on this device that is the difference between a compensator
    and something that makes the span worse.
    """
    wavelengths = np.linspace(BRAGG - 2e-9, BRAGG + 2e-9, 3001)
    settings = {"length": 0.10, "index_modulation": 2e-4, "bragg_wavelength": BRAGG}
    named = fiber_bragg_grating(C_LIGHT / wavelengths, chirp=1e-9, **settings)
    spelled = fiber_bragg_grating(
        C_LIGHT / wavelengths,
        bragg_profile=BRAGG + 1e-9 * section_positions(200),
        **settings,
    )
    assert np.array_equal(named.s, spelled.s)


def comb_peaks(wavelengths: np.ndarray, reflectivity: np.ndarray) -> list[float]:
    """Wavelengths of the reflection peaks, the way an instrument would find them."""
    return [
        float(wavelengths[i])
        for i in range(1, len(reflectivity) - 1)
        if reflectivity[i] > reflectivity[i - 1]
        and reflectivity[i] >= reflectivity[i + 1]
        and reflectivity[i] > 0.05
    ]


def test_a_sampled_grating_combs_at_the_sampling_period() -> None:
    """``lambda**2 / (2 n_eff * Lambda_s)`` -- the sampling period is a cavity.

    The superstructure grating, and the closed form is the same Fabry-Perot
    arithmetic as any other cavity of that length. It is here rather than in the
    model, and the model has nothing in it that knows about combs: only a
    coupling that is switched on and off.
    """
    length, periods = 0.02, 10
    predicted = BRAGG**2 / (2.0 * SILICA_FIBER_NEFF * (length / periods))

    wavelengths = np.linspace(BRAGG - 4e-9, BRAGG + 4e-9, 12001)
    reflectivity = fiber_bragg_grating(
        C_LIGHT / wavelengths,
        length=length,
        index_modulation=6e-5,
        bragg_wavelength=BRAGG,
        coupling_profile=sampled_profile(4000, periods=periods, duty=0.5),
    ).power("in", "in")

    peaks = comb_peaks(wavelengths, reflectivity)
    assert len(peaks) == 7
    # The grid is 0.67 pm and the spacing 415 pm, so per-mille is the
    # measurement's resolution rather than the model's.
    assert float(np.diff(peaks).mean()) == pytest.approx(predicted, rel=1e-3)


def test_a_sampled_grating_needs_sections_per_sampling_period_not_per_length() -> None:
    """And has enough by ten, which is where the component's floor comes from."""
    length, periods = 0.02, 10
    wavelengths = np.linspace(BRAGG - 4e-9, BRAGG + 4e-9, 12001)

    def spacing(sections: int) -> float:
        reflectivity = fiber_bragg_grating(
            C_LIGHT / wavelengths,
            length=length,
            index_modulation=6e-5,
            bragg_wavelength=BRAGG,
            coupling_profile=sampled_profile(sections, periods=periods, duty=0.5),
        ).power("in", "in")
        return float(np.diff(comb_peaks(wavelengths, reflectivity)).mean())

    converged = spacing(4000)
    for sections in (100, 200, 1000):
        assert spacing(sections) == pytest.approx(converged, rel=1e-9)


def test_a_full_duty_sample_is_not_a_sampled_grating() -> None:
    """The degenerate case answers as the uniform grating rather than specially."""
    assert np.array_equal(sampled_profile(200, periods=10, duty=1.0), np.ones(200))


def test_a_phase_shift_opens_a_window_in_the_stop_band() -> None:
    """At pi, dead centre: full transmission at the Bragg wavelength itself.

    Two mirrors facing each other with a broken period between them, which is a
    cavity -- the distributed-feedback laser's, and the narrowest feature a
    grating of a given length can produce. 0.7 pm here is 87 MHz.
    """
    wavelengths = np.linspace(BRAGG - 0.4e-9, BRAGG + 0.4e-9, 32001)
    settings = {"length": 0.02, "index_modulation": 1.5e-4, "bragg_wavelength": BRAGG}

    shifted = fiber_bragg_grating(
        C_LIGHT / wavelengths,
        phase_profile=phase_shift_profile(4000, shift=np.pi),
        **settings,
    )
    plain = fiber_bragg_grating(C_LIGHT / wavelengths, **settings)

    # Inside the stop band, where an unbroken grating transmits nothing.
    core = np.abs(wavelengths - BRAGG) < 0.05e-9
    assert plain.power("out", "in")[core].max() < 1e-3

    transmitted = shifted.power("out", "in")[core]
    peak = int(np.argmax(transmitted))
    assert transmitted[peak] == pytest.approx(1.0, abs=1e-4)
    assert wavelengths[core][peak] == pytest.approx(BRAGG, abs=1e-14)

    width = wavelengths[core][transmitted > 0.5 * transmitted[peak]]
    assert (width.max() - width.min()) == pytest.approx(0.7e-12, abs=0.2e-12)

    # And it is still lossless, which a phase riding on the coupling must be.
    total = shifted.power("in", "in") + shifted.power("out", "in")
    assert total == pytest.approx(np.ones_like(total), abs=1e-9)


def test_an_off_centre_phase_shift_loses_the_resonance() -> None:
    """Because the two halves stop being equal mirrors.

    A real design sensitivity rather than a modelling artefact, and the reason
    ``position`` is offered at all.
    """
    wavelengths = np.linspace(BRAGG - 0.05e-9, BRAGG + 0.05e-9, 8001)
    peaks = []
    for position in (0.5, 0.45, 0.35):
        matrix = fiber_bragg_grating(
            C_LIGHT / wavelengths,
            length=0.02,
            index_modulation=1.5e-4,
            bragg_wavelength=BRAGG,
            phase_profile=phase_shift_profile(4000, shift=np.pi, position=position),
        )
        peaks.append(float(matrix.power("out", "in").max()))

    assert peaks[0] == pytest.approx(1.00, abs=0.01)
    assert peaks[1] == pytest.approx(0.71, abs=0.03)
    assert peaks[2] == pytest.approx(0.10, abs=0.03)
    assert peaks[0] > peaks[1] > peaks[2]


def test_a_phase_constant_along_the_length_is_not_observable() -> None:
    """Where a grating's fringes start is not a measurable thing, and the model
    had better agree -- a phase that rode on the detuning instead of the coupling
    would shift the whole spectrum and look perfectly plausible doing it."""
    wavelengths = np.linspace(BRAGG - 1e-9, BRAGG + 1e-9, 2001)
    settings = {"length": 0.02, "index_modulation": 1.5e-4, "bragg_wavelength": BRAGG}
    plain = fiber_bragg_grating(C_LIGHT / wavelengths, **settings)
    turned = fiber_bragg_grating(C_LIGHT / wavelengths, phase_profile=np.full(200, 0.7), **settings)
    assert np.abs(turned.s) == pytest.approx(np.abs(plain.s), abs=1e-12)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"coupling_profile": np.ones(50), "apodization": "gaussian"},
            "both shape the coupling",
        ),
        ({"bragg_profile": np.full(50, BRAGG), "chirp": 1e-9}, "both set the local Bragg"),
        (
            {"coupling_profile": np.ones(50), "phase_profile": np.zeros(40)},
            "different lengths",
        ),
        ({"bragg_profile": np.full(50, -1.0)}, "local Bragg wavelength must be positive"),
    ],
)
def test_profiles_that_contradict_something_else_are_refused(kwargs: dict, message: str) -> None:
    """Two descriptions of one thing is not a preference to resolve quietly."""
    with pytest.raises(ValueError, match=message):
        fiber_bragg_grating(
            np.array([C_LIGHT / BRAGG]),
            length=0.01,
            index_modulation=1e-4,
            bragg_wavelength=BRAGG,
            **kwargs,
        )


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: sampled_profile(100, periods=0.0), "periods must be positive"),
        (lambda: sampled_profile(100, periods=10, duty=0.0), r"duty must be in \(0, 1\]"),
        (lambda: sampled_profile(100, periods=10, duty=1.5), r"duty must be in \(0, 1\]"),
        (lambda: phase_shift_profile(100, position=1.5), "must be within the grating"),
        (lambda: section_positions(0), "sections must be at least 1"),
    ],
)
def test_the_profile_builders_refuse_what_they_cannot_describe(
    call: Callable[[], object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        call()


def test_the_block_reaches_both_new_devices() -> None:
    """The point of putting them on the component and not only in the engine: a
    project file and an inspector can carry numbers and not arrays, so the two
    devices worth having are declared as knobs that build the arrays."""
    combed = FiberBraggGrating(
        length=20.0,
        index_modulation=6e-5,
        bragg_wavelength=1550.0,
        sampled=True,
        sample_periods=10.0,
    )
    wavelengths = np.linspace(BRAGG - 4e-9, BRAGG + 4e-9, 12001)
    reflectivity = combed.scattering_matrix(C_LIGHT / wavelengths).power("in", "in")
    predicted = BRAGG**2 / (2.0 * SILICA_FIBER_NEFF * 0.002)
    assert float(np.diff(comb_peaks(wavelengths, reflectivity)).mean()) == pytest.approx(
        predicted, rel=1e-3
    )

    # Sampling raises the section count on its own, because the structure that
    # has to be resolved is the mask rather than the envelope.
    assert combed._sections() == 200
    assert FiberBraggGrating(sampled=True, sample_periods=40.0)._sections() == 800
    assert FiberBraggGrating(sample_periods=40.0)._sections() == 200

    shifted = FiberBraggGrating(
        length=20.0, index_modulation=1.5e-4, bragg_wavelength=1550.0, phase_shifted=True
    )
    near = np.linspace(BRAGG - 0.05e-9, BRAGG + 0.05e-9, 8001)
    transmitted = shifted.scattering_matrix(C_LIGHT / near).power("out", "in")
    assert transmitted.max() == pytest.approx(1.0, abs=1e-4)
    assert near[int(np.argmax(transmitted))] == pytest.approx(BRAGG, abs=1e-14)


def test_a_sampled_grating_can_also_be_apodized() -> None:
    """Both shape the coupling and a real device carries both: a mask over an
    apodized exposure. The component multiplies them rather than making anyone
    choose, which is what the writing process does."""
    settings = {
        "length": 20.0,
        "index_modulation": 6e-5,
        "bragg_wavelength": 1550.0,
        "sampled": True,
        "sample_periods": 10.0,
    }
    soft = FiberBraggGrating("raised-cosine", **settings)
    hard = FiberBraggGrating(**settings)

    wavelengths = np.linspace(BRAGG - 4e-9, BRAGG + 4e-9, 8001)
    quiet = soft.scattering_matrix(C_LIGHT / wavelengths).power("in", "in")
    loud = hard.scattering_matrix(C_LIGHT / wavelengths).power("in", "in")

    # Apodizing lowers the average coupling, so the whole comb comes down.
    assert quiet.max() < loud.max()
    # And it quietens the skirts of each tooth relative to its own peak, which
    # is the thing apodization is for.
    assert quiet[:500].max() / quiet.max() < loud[:500].max() / loud.max()
