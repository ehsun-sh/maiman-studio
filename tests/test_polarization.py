"""Birefringence: two modes in one waveguide, and a ring that resonates twice.

Until this existed, every photonic block applied one response to both field
components — which says the die treats TE and TM alike, and no strip waveguide
ever has. What is asserted here is mostly that the two polarizations are
*independent and different*: same solver, same models, two sets of indices.
"""

from __future__ import annotations

import numpy as np
import pytest

from maiman import Graph, SimulationContext
from maiman.circuit import Circuit, SMatrix
from maiman.components import CWLaser, PolarizationRotator, RingResonator, Waveguide
from maiman.photonics import (
    POLARIZATIONS,
    SILICON_STRIP_NEFF,
    SILICON_STRIP_NEFF_TM,
    SILICON_STRIP_NGROUP,
    SILICON_STRIP_NGROUP_TM,
    directional_coupler,
    dual_polarization,
    expose_polarizations,
    link_polarizations,
    polarized_port,
    straight_waveguide,
)
from maiman.units import C_LIGHT

REFERENCE = C_LIGHT / 1550e-9
RING_LENGTH = 100e-6


def grid(points: int = 60001, span: float = 1.5e12) -> np.ndarray:
    return np.linspace(REFERENCE - span, REFERENCE + span, points)


def guide(frequencies: np.ndarray, *, n_eff: float, n_group: float) -> SMatrix:
    return straight_waveguide(
        frequencies,
        length=RING_LENGTH,
        n_eff=n_eff,
        n_group=n_group,
        reference_frequency=REFERENCE,
        loss_db_per_m=200.0,
    )


def dual_ring(
    frequencies: np.ndarray,
    *,
    n_eff_tm: float = SILICON_STRIP_NEFF_TM,
    n_group_tm: float = SILICON_STRIP_NGROUP_TM,
    coupling: float = 0.05,
) -> SMatrix:
    """An all-pass ring assembled from dual-polarization devices.

    Assembled rather than written down, exactly as the single-polarization ring
    is: what is under test is that the reduction handles a four-port device with
    two independent modes in it, not a transcription of a formula.
    """
    coupler = dual_polarization(
        dict.fromkeys(POLARIZATIONS, directional_coupler(frequencies, coupling=coupling))
    )
    loop = dual_polarization(
        {
            "te": guide(frequencies, n_eff=SILICON_STRIP_NEFF, n_group=SILICON_STRIP_NGROUP),
            "tm": guide(frequencies, n_eff=n_eff_tm, n_group=n_group_tm),
        }
    )
    circuit = Circuit().add("c", coupler).add("w", loop)
    link_polarizations(circuit, "c", "out2", "w", "in")
    link_polarizations(circuit, "w", "out", "c", "in2")
    expose_polarizations(circuit, "in", "c", "in1")
    expose_polarizations(circuit, "through", "c", "out1")
    return circuit.solve()


def resonances(frequencies: np.ndarray, power: np.ndarray) -> np.ndarray:
    """Frequencies of the notches, by simple local minimum below half depth."""
    threshold = power.min() + 0.5 * (power.max() - power.min())
    interior = np.arange(1, power.size - 1)
    dips = interior[
        (power[1:-1] < power[:-2]) & (power[1:-1] <= power[2:]) & (power[1:-1] < threshold)
    ]
    return frequencies[dips]


# ---------------------------------------------------------------------------
# the combinator


def test_stacking_two_identical_polarizations_reproduces_the_single_one_exactly() -> None:
    """The reduction to the old case, which is what makes this safe to add.

    A birefringent model that did not collapse onto the non-birefringent one when
    handed the same indices twice would mean every existing number in this
    repository was about to move for a reason nobody could name.
    """
    frequencies = grid(257)
    single = guide(frequencies, n_eff=SILICON_STRIP_NEFF, n_group=SILICON_STRIP_NGROUP)
    stacked = dual_polarization(dict.fromkeys(POLARIZATIONS, single))

    assert stacked.ports == ("in@te", "out@te", "in@tm", "out@tm")
    for polarization in POLARIZATIONS:
        assert np.array_equal(
            stacked.transmission(f"out@{polarization}", f"in@{polarization}"),
            single.transmission("out", "in"),
        )


def test_the_two_polarizations_do_not_talk_to_each_other() -> None:
    """Block diagonal, which is right for a straight guide and a symmetric coupler.

    Exactly zero rather than small: there is no mechanism in the model at all, so
    a non-zero here would be an indexing mistake rather than a physical coupling.
    """
    frequencies = grid(129)
    stacked = dual_polarization(
        {
            "te": guide(frequencies, n_eff=SILICON_STRIP_NEFF, n_group=SILICON_STRIP_NGROUP),
            "tm": guide(frequencies, n_eff=SILICON_STRIP_NEFF_TM, n_group=SILICON_STRIP_NGROUP_TM),
        }
    )
    assert np.all(stacked.transmission("out@tm", "in@te") == 0.0)
    assert np.all(stacked.transmission("out@te", "in@tm") == 0.0)


def test_a_cross_term_is_placed_where_the_solver_will_find_it() -> None:
    """Nothing here makes one, and the solver has never needed the assumption.

    A bend, a sidewall that is not vertical, and a mode converter placed there on
    purpose all couple TE to TM. The argument exists so that the block-diagonal
    case is a *choice* the models make rather than a shape the matrix cannot
    hold — and this is what says it can.
    """
    frequencies = grid(65)
    single = guide(frequencies, n_eff=SILICON_STRIP_NEFF, n_group=SILICON_STRIP_NGROUP)
    conversion = np.zeros((frequencies.size, 2, 2), dtype=np.complex128)
    conversion[:, 1, 0] = 0.1  # a tenth of the amplitude, in to out

    stacked = dual_polarization(
        dict.fromkeys(POLARIZATIONS, single), cross={("te", "tm"): conversion}
    )
    assert np.allclose(stacked.transmission("out@tm", "in@te"), 0.1)
    assert np.all(stacked.transmission("out@te", "in@tm") == 0.0), "one direction only"


def test_stacking_preserves_unitarity() -> None:
    """A lossless pair of modes is still lossless taken together."""
    frequencies = grid(33)
    coupler = directional_coupler(frequencies, coupling=0.3)
    assert coupler.is_unitary()
    assert dual_polarization(dict.fromkeys(POLARIZATIONS, coupler)).is_unitary()


def test_a_stack_that_is_not_one_device_is_refused() -> None:
    frequencies = grid(17)
    te = guide(frequencies, n_eff=SILICON_STRIP_NEFF, n_group=SILICON_STRIP_NGROUP)
    other_ports = straight_waveguide(
        frequencies, length=RING_LENGTH, reference_frequency=REFERENCE, ports=("a", "b")
    )
    other_grid = guide(grid(19), n_eff=SILICON_STRIP_NEFF, n_group=SILICON_STRIP_NGROUP)

    with pytest.raises(ValueError, match="at least one polarization"):
        dual_polarization({})
    with pytest.raises(ValueError, match="same device"):
        dual_polarization({"te": te, "tm": other_ports})
    with pytest.raises(ValueError, match="same frequency grid"):
        dual_polarization({"te": te, "tm": other_grid})
    with pytest.raises(KeyError, match="not a polarization"):
        dual_polarization(dict.fromkeys(POLARIZATIONS, te), cross={("te", "xx"): np.zeros(1)})
    with pytest.raises(ValueError, match="to itself"):
        dual_polarization(dict.fromkeys(POLARIZATIONS, te), cross={("te", "te"): np.zeros(1)})
    with pytest.raises(ValueError, match="shaped like one device"):
        dual_polarization(dict.fromkeys(POLARIZATIONS, te), cross={("te", "tm"): np.zeros(3)})
    with pytest.raises(ValueError, match="already contains"):
        polarized_port("in@te", "tm")


# ---------------------------------------------------------------------------
# the ring, which is the effect


def test_a_birefringent_ring_resonates_at_two_sets_of_wavelengths() -> None:
    """The thing that was missing, stated as the README stated it.

    Two combs, on two different free spectral ranges, each matching ``c/(n_g L)``
    for its own group index. Nothing about the solver changed to allow this: the
    ports are named, so two modes are four ports, and the reduction already
    solved those.
    """
    frequencies = grid()
    solved = dual_ring(frequencies)

    spacings = {}
    for polarization, n_group in (("te", SILICON_STRIP_NGROUP), ("tm", SILICON_STRIP_NGROUP_TM)):
        found = resonances(
            frequencies, solved.power(f"through@{polarization}", f"in@{polarization}")
        )
        assert found.size >= 3, f"{polarization} should resonate more than once in 3 THz"
        spacing = float(np.mean(np.diff(found)))
        assert spacing == pytest.approx(C_LIGHT / (n_group * RING_LENGTH), rel=1e-3)
        spacings[polarization] = spacing

    assert spacings["tm"] > spacings["te"], "the lower group index spaces them further apart"
    assert spacings["tm"] / spacings["te"] == pytest.approx(
        SILICON_STRIP_NGROUP / SILICON_STRIP_NGROUP_TM, rel=1e-3
    )


def test_the_two_combs_land_on_top_of_each_other_when_the_indices_agree() -> None:
    """Which is the control for the test above.

    If the two combs were separated by anything other than the indices — an
    off-by-one in the stacking, a port wired to the wrong block — they would
    still be separated here, where there is nothing to separate them.
    """
    frequencies = grid()
    solved = dual_ring(frequencies, n_eff_tm=SILICON_STRIP_NEFF, n_group_tm=SILICON_STRIP_NGROUP)
    te = resonances(frequencies, solved.power("through@te", "in@te"))
    tm = resonances(frequencies, solved.power("through@tm", "in@tm"))

    assert te.size == tm.size
    assert np.allclose(te, tm)


def test_light_launched_on_one_mode_stays_on_it_through_a_resonance() -> None:
    """A resonance is where a coupling fault would be loudest, so look there.

    At a TE resonance the light circulates many times before it leaves. If any
    of the stacking leaked between blocks, that is the frequency at which a
    tenth of a per cent would be amplified into something visible.
    """
    frequencies = grid()
    solved = dual_ring(frequencies)
    te_power = solved.power("through@te", "in@te")
    at_resonance = int(np.argmin(te_power))

    # Against the off-resonance level, not an absolute depth: this ring is
    # under-coupled, so its notch bottoms at 0.70 rather than anywhere near zero,
    # and a constant here would be asserting the coupling rather than the leakage.
    assert te_power[at_resonance] < 0.75 * te_power.max(), "this is a real notch"
    assert solved.power("through@tm", "in@te")[at_resonance] == 0.0
    assert solved.power("through@te", "in@tm")[at_resonance] == 0.0


# ---------------------------------------------------------------------------
# the block, on a graph


def birefringent_ring(wavelength_nm: float, *, birefringent: bool) -> tuple[float, float]:
    """Mean power on each field component out of a ring's through port."""
    ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=8, sequence_length=64, seed=1)
    graph = Graph(ctx)
    laser = graph.add(CWLaser(power=0.0, wavelength=wavelength_nm, label="tx"))
    # 45 degrees, so there is equal light on both axes to be treated differently.
    rotator = graph.add(PolarizationRotator(angle=45.0, label="rot"))
    ring = graph.add(
        RingResonator(length=100.0, coupling=0.05, birefringent=birefringent, label="ring")
    )
    graph.connect(laser, rotator["in"])
    graph.connect(rotator["out"], ring["in"])

    band = graph.run(keep=[ring]).port(ring, "through").bands[0]
    return (
        float(np.mean(np.abs(np.asarray(band.Ex)) ** 2)),
        float(np.mean(np.abs(np.asarray(band.Ey)) ** 2)),
    )


def test_the_block_notches_one_field_component_and_not_the_other() -> None:
    """End to end on a graph, which is the only place this has to be true.

    Measured: the TE comb notches at 1558.16 nm and the TM comb at 1561.71, and
    at each of those the *other* component passes. A ring like this is a
    polarization-selective filter, which is what a real one is and what this
    library said it could not show.
    """
    te_resonance, tm_resonance = 1558.16, 1561.71

    ex, ey = birefringent_ring(te_resonance, birefringent=True)
    assert ex < 0.75 * ey, "TE is in the notch and TM is not"

    ex, ey = birefringent_ring(tm_resonance, birefringent=True)
    assert ey < 0.75 * ex, "and the other way round one comb over"


def test_without_the_flag_both_components_see_the_same_device() -> None:
    """The idealisation, and it has to be exactly the old behaviour.

    Not approximately: ``birefringent`` false must take the same path the code
    took before there was a flag, or every project in this repository moves.
    """
    ex, ey = birefringent_ring(1558.16, birefringent=False)
    assert ex == pytest.approx(ey, rel=1e-12)
    assert ex < 0.75 * birefringent_ring(1554.0, birefringent=False)[0], "still a resonance"


def test_a_waveguide_delays_the_two_modes_by_different_amounts() -> None:
    """The simplest case, and the one every other device is built out of.

    A group index of 4.20 against 3.80 is a 10 % difference in delay, which over
    a millimetre is 1.3 ps — small, and exactly the sort of small that turns into
    a resonance offset once the light goes round a loop.
    """
    straight = Waveguide(length=1000.0, birefringent=True, label="wg")
    assert straight._waveguide_kwargs("te")["n_group"] == SILICON_STRIP_NGROUP
    assert straight._waveguide_kwargs("tm")["n_group"] == SILICON_STRIP_NGROUP_TM

    ideal = Waveguide(length=1000.0, birefringent=False, label="wg")
    assert ideal._waveguide_kwargs("tm")["n_group"] == SILICON_STRIP_NGROUP


def test_loss_and_dispersion_are_shared_between_the_modes_and_that_is_stated() -> None:
    """A limit worth a test rather than only a sentence.

    A real strip has a different loss for TM — it overlaps the sidewalls less and
    the substrate more — and a different dispersion with it. Those are
    per-process numbers and this library will not invent one, so the two indices
    are what is offered and the rest is shared. If a fitted pair ever arrives it
    belongs in a PDK, and this test is what will fail to say so.
    """
    straight = Waveguide(length=1000.0, birefringent=True, propagation_loss=3.0, label="wg")
    te = straight._waveguide_kwargs("te")
    tm = straight._waveguide_kwargs("tm")

    assert te["loss_db_per_m"] == tm["loss_db_per_m"]
    assert te["dispersion"] == tm["dispersion"]
    assert te["n_eff"] != tm["n_eff"], "what does differ is the pair that splits the comb"


def test_wiring_one_polarization_and_not_the_other_is_what_the_helpers_prevent() -> None:
    """The failure they exist for: a plausible spectrum on TE and silence on TM."""
    frequencies = grid(33)
    single = guide(frequencies, n_eff=SILICON_STRIP_NEFF, n_group=SILICON_STRIP_NGROUP)
    stacked = dual_polarization(dict.fromkeys(POLARIZATIONS, single))

    circuit = Circuit().add("a", stacked).add("b", stacked)
    link_polarizations(circuit, "a", "out", "b", "in")
    expose_polarizations(circuit, "in", "a", "in")
    expose_polarizations(circuit, "out", "b", "out")
    solved = circuit.solve()

    assert set(solved.ports) == {"in@te", "in@tm", "out@te", "out@tm"}
    through = single.transmission("out", "in") ** 2
    for polarization in POLARIZATIONS:
        assert np.allclose(
            solved.transmission(f"out@{polarization}", f"in@{polarization}"), through
        )
