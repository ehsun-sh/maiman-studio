"""Integrated-photonic device models, and the blocks that put them in a link.

The models are checked against the closed forms in the literature, which is the
whole reason :func:`maiman.photonics.ring_resonator` *assembles* a ring out of
couplers and waveguides rather than writing one down: the formula is on the other
side of the comparison, where it can disagree.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import cache

import numpy as np
import pytest

from maiman.circuit import Circuit, SMatrix
from maiman.component import Component
from maiman.components import (
    EDFA,
    CWLaser,
    OSNRMeter,
    PolarizationRotator,
    PowerMeter,
    RingResonator,
    Waveguide,
)
from maiman.components.photonic import _Photonic, average_power_response
from maiman.context import SimulationContext
from maiman.graph import Graph
from maiman.kernels import dispersion_to_beta2
from maiman.photonics import (
    SILICON_STRIP_NEFF,
    SILICON_STRIP_NGROUP,
    directional_coupler,
    free_spectral_range,
    propagation_constant,
    resonance_linewidth,
    ring_resonator,
    round_trip_amplitude,
    straight_waveguide,
)
from maiman.registry import registered_names
from maiman.signals import NoiseBin
from maiman.units import C_LIGHT, frequency_to_wavelength, w_to_dbm, wavelength_to_frequency

F0 = wavelength_to_frequency(1550e-9)
LENGTH = 100e-6
LOSS = 300.0  # dB/m, i.e. 3 dB/cm
FSR = free_spectral_range(LENGTH, SILICON_STRIP_NGROUP)
CRITICAL = 1.0 - round_trip_amplitude(LENGTH, LOSS) ** 2


def yariv(
    frequencies: np.ndarray, kappa1: float, kappa2: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Through and drop, straight out of the literature.

    Yariv, Electron. Lett. 36(4), 2000, and Bogaerts et al., Laser Photonics
    Rev. 6(1), 2012, eqs. 15 and 16. Nothing here knows there is a solver.
    """
    beta = propagation_constant(
        frequencies,
        n_eff=SILICON_STRIP_NEFF,
        n_group=SILICON_STRIP_NGROUP,
        reference_frequency=F0,
    )
    half = round_trip_amplitude(LENGTH / 2, LOSS) * np.exp(-1j * beta * LENGTH / 2)
    loop = half**2
    t1, t2 = np.sqrt(1 - kappa1), np.sqrt(1 - kappa2)
    k1, k2 = np.sqrt(kappa1), np.sqrt(kappa2)
    return (
        (t1 - t2 * loop) / (1 - t1 * t2 * loop),
        -k1 * k2 * half / (1 - t1 * t2 * loop),
    )


def solved(frequencies: np.ndarray, kappa1: float, kappa2: float = 0.0) -> SMatrix:
    return ring_resonator(
        frequencies,
        length=LENGTH,
        coupling=kappa1,
        drop_coupling=kappa2,
        reference_frequency=F0,
        loss_db_per_m=LOSS,
    )


def notches(grid: np.ndarray, power: np.ndarray, *, below: float) -> np.ndarray:
    """Frequencies of the local minima that go below ``below``."""
    interior = (power[1:-1] < power[:-2]) & (power[1:-1] < power[2:]) & (power[1:-1] < below)
    return grid[1:-1][interior]


@cache
def resonance(coupling: float, drop: float = 0.0) -> float:
    """The frequency of the deepest notch within one free spectral range of 1550 nm.

    Found in two passes rather than one: a coarse scan of the whole free spectral
    range to locate the notch, then a fine one around it. A resonance can be a
    megahertz wide inside a 714 GHz period, and resolving that in a single sweep
    would need a hundred million points.
    """
    coarse = F0 + np.linspace(-FSR / 2, FSR / 2, 200001)
    near = float(coarse[int(np.argmin(solved(coarse, coupling, drop).power("through", "in")))])
    fine = near + np.linspace(-FSR / 2e5, FSR / 2e5, 20001)
    return float(fine[int(np.argmin(solved(fine, coupling, drop).power("through", "in")))])


# ---------------------------------------------------------------------------
# the models against the closed forms


@pytest.mark.parametrize(
    ("kappa1", "kappa2"),
    [(0.10, 0.02), (0.05, 0.05), (0.30, 0.00), (CRITICAL, 0.0), (0.9, 0.4)],
)
def test_the_assembled_ring_is_the_ring_in_the_literature(kappa1: float, kappa2: float) -> None:
    """Two couplers and two arcs, wired and solved, reproduce Yariv's transfer functions.

    To machine precision, over two terahertz — which is roughly three free
    spectral ranges, so the comparison covers resonances, the flat between them,
    and both slopes.
    """
    grid = F0 + np.linspace(-1e12, 1e12, 20001)
    through, drop = yariv(grid, kappa1, kappa2)
    matrix = solved(grid, kappa1, kappa2)

    assert np.max(np.abs(matrix.transmission("through", "in") - through)) < 1e-13
    assert np.max(np.abs(matrix.transmission("drop", "in") - drop)) < 1e-13


def test_the_free_spectral_range_is_set_by_the_group_index() -> None:
    """And not by the effective index, which is the classic way to be wrong by 1.7x.

    Measured as the spacing between resonances, against ``c / (n_g * L)``.
    """
    # A weak resonance on purpose: broad enough for a tractable grid to resolve,
    # and the spacing is what is under test, not the width.
    grid = F0 + np.linspace(-1.5e12, 1.5e12, 300001)
    dips = notches(grid, solved(grid, 0.3).power("through", "in"), below=0.97)
    assert len(dips) >= 4
    assert float(np.mean(np.diff(dips))) == pytest.approx(FSR, rel=1e-4)

    # The effective index moves the resonances; it does not move their spacing.
    shifted = ring_resonator(
        grid,
        length=LENGTH,
        coupling=0.3,
        n_eff=SILICON_STRIP_NEFF + 0.1,
        reference_frequency=F0,
        loss_db_per_m=LOSS,
    )
    moved = notches(grid, shifted.power("through", "in"), below=0.97)
    assert float(np.mean(np.diff(moved))) == pytest.approx(FSR, rel=1e-4)
    assert abs(moved[0] - dips[0]) > 1e9  # they did move

    # The group index moves the spacing, which is the point.
    slower = ring_resonator(
        grid,
        length=LENGTH,
        coupling=0.3,
        n_group=SILICON_STRIP_NGROUP * 2,
        reference_frequency=F0,
        loss_db_per_m=LOSS,
    )
    apart = notches(grid, slower.power("through", "in"), below=0.97)
    assert float(np.mean(np.diff(apart))) == pytest.approx(FSR / 2, rel=1e-4)


def test_critical_coupling_extinguishes_the_through_port() -> None:
    """The case worth knowing, and the two ways of missing it.

    At the coupling that matches the round-trip loss the field coupled back out
    of the ring cancels the field that stayed on the bus, and the notch has no
    floor. Under-coupling and over-coupling both fill it in.
    """
    peak = resonance(CRITICAL)
    fine = peak + np.linspace(-1e6, 1e6, 20001)

    depths = {
        kappa: float(solved(fine, kappa).power("through", "in").min())
        for kappa in (CRITICAL / 4, CRITICAL, CRITICAL * 4)
    }
    assert depths[CRITICAL] < 1e-12
    assert depths[CRITICAL / 4] > 0.2
    assert depths[CRITICAL * 4] > 0.2


def test_a_measured_notch_depth_does_not_identify_a_ring() -> None:
    """Under- and over-coupling give the same extinction, which is why the drop port exists.

    The depth ``|t - a|**2 / |1 - t a|**2`` is *symmetric in t and a*, so for
    every t there is a second one giving the same notch. Setting the two equal
    and solving gives the partner directly:

        ``t2 = (2a - t1 (1 + a**2)) / (1 + a**2 - 2 a t1)``

    which is its own inverse and has ``t = a``, critical coupling, as its only
    fixed point. Writing an offset symmetric in *t* instead does not work, and
    getting that wrong is how this test was first written.
    """
    a = round_trip_amplitude(LENGTH, LOSS)
    t1 = 0.99
    t2 = (2 * a - t1 * (1 + a**2)) / (1 + a**2 - 2 * a * t1)
    under, over = 1.0 - t1**2, 1.0 - t2**2
    fine = resonance(CRITICAL) + np.linspace(-2e6, 2e6, 40001)

    shallow = float(solved(fine, under).power("through", "in").min())
    deep = float(solved(fine, over).power("through", "in").min())
    assert shallow == pytest.approx(deep, rel=1e-6)
    # And they are genuinely different couplings, one either side of critical.
    assert under / over > 5.0
    assert over < CRITICAL < under


@pytest.mark.parametrize("kappa", [CRITICAL, 0.05, 0.5])
def test_the_linewidth_formula_matches_a_measured_width(kappa: float) -> None:
    """It is what sets the sampling step everything else integrates on.

    Three couplings on purpose. The Lorentzian approximation runs a steady 1.5 %
    wide from critical coupling up, and 2.6 % at kappa = 0.5 where a resonance is
    eighty gigahertz across and barely a resonance — so 3 % is the tolerance the
    measurement supports rather than one chosen to pass.

    The strongly coupled case is also the only one that can see the square root
    in the denominator: dropping it costs 0.1 % at kappa = 0.05 and 14 % here.
    """
    predicted = resonance_linewidth(
        LENGTH, SILICON_STRIP_NGROUP, coupling=kappa, loss_db_per_m=LOSS
    )
    # A whole free spectral range, so that the maximum the half-depth level is
    # taken from is the real one. A window a few linewidths wide never reaches it
    # when the resonance is eighty gigahertz across, and the width comes out 3.5 %
    # narrow for that reason alone.
    peak = resonance(kappa)
    grid = peak + np.linspace(-FSR / 2, FSR / 2, 200001)
    power = solved(grid, kappa).power("through", "in")

    half = 0.5 * (power.max() + power.min())
    below = grid[power <= half]
    assert float(below[-1] - below[0]) == pytest.approx(predicted, rel=0.03)


# ---------------------------------------------------------------------------
# the coupler


@pytest.mark.parametrize("kappa", [0.0, 0.1, 0.5, 0.9, 1.0])
def test_a_lossless_coupler_is_unitary(kappa: float) -> None:
    """Which is the whole reason the cross path carries a factor of j."""
    assert directional_coupler(np.array([F0]), coupling=kappa).is_unitary()


def test_dropping_the_quadrature_would_break_energy_conservation() -> None:
    """Not a convention: a real coupler without it gains energy at one phase and loses at another.

    Built here rather than asserted, so the claim in the docstring is the thing
    being measured. A Mach-Zehnder of two 3 dB couplers conserves power at every
    arm phase; give the couplers a real cross term instead and it does not.
    """
    phases = np.linspace(0.0, 2 * np.pi, 33)
    grid = np.full(phases.shape, F0)

    def interferometer(quadrature: bool) -> np.ndarray:
        coupler = directional_coupler(grid, coupling=0.5)
        if not quadrature:
            coupler = type(coupler)(
                ports=coupler.ports, frequencies=grid, s=np.abs(coupler.s).astype(np.complex128)
            )
        arm_top = np.exp(-1j * phases)
        arm_bottom = np.ones_like(phases)
        circuit = Circuit()
        circuit.add("first", coupler).add("second", coupler)
        for name, transfer in (("top", arm_top), ("bottom", arm_bottom)):
            s = np.zeros((phases.shape[0], 2, 2), dtype=np.complex128)
            s[:, 0, 1] = transfer
            s[:, 1, 0] = transfer
            circuit.add(name, type(coupler)(ports=("a", "b"), frequencies=grid, s=s))
        circuit.link("first", "out1", "top", "a").link("top", "b", "second", "in1")
        circuit.link("first", "out2", "bottom", "a").link("bottom", "b", "second", "in2")
        circuit.expose("in", "first", "in1")
        circuit.expose("out1", "second", "out1").expose("out2", "second", "out2")
        matrix = circuit.solve()
        return matrix.power("out1", "in") + matrix.power("out2", "in")

    assert np.allclose(interferometer(quadrature=True), 1.0, atol=1e-12)
    assert np.max(interferometer(quadrature=False)) > 1.5


def test_the_interferometer_actually_interferes() -> None:
    """A power balance passes trivially if nothing splits, so check that it does."""
    phases = np.linspace(0.0, 2 * np.pi, 33)
    grid = np.full(phases.shape, F0)
    coupler = directional_coupler(grid, coupling=0.5)
    arm = np.zeros((phases.shape[0], 2, 2), dtype=np.complex128)
    arm[:, 0, 1] = arm[:, 1, 0] = np.exp(-1j * phases)
    flat = np.zeros_like(arm)
    flat[:, 0, 1] = flat[:, 1, 0] = 1.0

    circuit = Circuit().add("first", coupler).add("second", coupler)
    circuit.add("top", type(coupler)(ports=("a", "b"), frequencies=grid, s=arm))
    circuit.add("bottom", type(coupler)(ports=("a", "b"), frequencies=grid, s=flat))
    circuit.link("first", "out1", "top", "a").link("top", "b", "second", "in1")
    circuit.link("first", "out2", "bottom", "a").link("bottom", "b", "second", "in2")
    circuit.expose("in", "first", "in1").expose("out1", "second", "out1")
    power = circuit.solve().power("out1", "in")

    assert power.min() < 1e-12
    assert power.max() > 1.0 - 1e-12


# ---------------------------------------------------------------------------
# the waveguide, and how far the dispersion convention reaches


def test_the_waveguide_delays_by_its_group_index() -> None:
    """Measured from the transfer function's phase slope, not from the parameter."""
    length = 1e-3
    grid = F0 + np.linspace(-50e9, 50e9, 4001)
    transfer = straight_waveguide(grid, length=length, reference_frequency=F0).transmission(
        "out", "in"
    )

    # A delay tau is exp(-i * 2 pi f * tau), so the slope of the unwrapped phase
    # against frequency is -2 pi tau. A negative slope, i.e. a delay, not an
    # advance -- which is the half of the sign convention that is not in doubt.
    slope = np.polyfit(grid - F0, np.unwrap(np.angle(transfer)), 1)[0]
    assert -slope / (2 * np.pi) == pytest.approx(SILICON_STRIP_NGROUP * length / C_LIGHT, rel=1e-9)


def test_the_quadratic_term_is_negligible_on_a_ring_and_not_on_a_spiral() -> None:
    """The numbers the sign-convention note in ``propagation_constant`` rests on.

    Measured rather than asserted, so that reconciling the kernel's beta2 sign
    later arrives at a test that already says what it changes.
    """
    edge = 2 * np.pi * 4.4e12  # half a C band, in rad/s
    beta2 = dispersion_to_beta2(-1000e-6, 1550e-9)  # a very large silicon D

    ring = abs(0.5 * beta2 * edge**2 * 100e-6)
    spiral = abs(0.5 * beta2 * edge**2 * 0.1)
    assert ring == pytest.approx(0.0487, abs=0.001)
    assert spiral == pytest.approx(48.7, abs=1.0)
    assert spiral / ring == pytest.approx(1000.0)


def test_a_lossless_waveguide_is_unitary_and_a_lossy_one_is_not() -> None:
    grid = np.array([F0])
    assert straight_waveguide(grid, length=1e-3, reference_frequency=F0).is_unitary()
    assert not straight_waveguide(
        grid, length=1e-3, reference_frequency=F0, loss_db_per_m=LOSS
    ).is_unitary()


# ---------------------------------------------------------------------------
# the blocks in a graph


def context() -> SimulationContext:
    return SimulationContext(bit_rate=10e9, samples_per_symbol=8, sequence_length=128, seed=3)


def test_the_waveguide_block_loses_exactly_what_it_says() -> None:
    graph = Graph(context())
    laser = graph.add(CWLaser(power=0.0, wavelength=1550.0))
    guide = graph.add(Waveguide(length=1000.0, propagation_loss=2.0))
    meter = graph.add(PowerMeter())
    graph.chain(laser, guide, meter)

    # 2 dB/cm over 1 mm is 0.2 dB, and 1 mm at n_g = 4.20 is 14.01 ps.
    # 1e-5 rather than exact: the transfer is applied in double precision and
    # the band is stored in single, which is this context's default.
    assert w_to_dbm(graph.run()[meter].power_w) == pytest.approx(-0.2, abs=1e-5)
    assert guide.group_delay() == pytest.approx(14.010e-12, rel=1e-3)


def test_the_waveguide_carries_its_dispersion_into_the_bookkeeping() -> None:
    """``accumulated_gvd`` is what four-wave mixing across spans is tracked with."""
    graph = Graph(context())
    laser = graph.add(CWLaser(power=0.0, wavelength=1550.0))
    guide = graph.add(Waveguide(length=100000.0, dispersion=-1000.0, propagation_loss=0.0))
    meter = graph.add(PowerMeter())
    graph.chain(laser, guide, meter)
    out = graph.run(keep=[guide]).port(guide, "out")

    expected = dispersion_to_beta2(-1000e-6, frequency_to_wavelength(guide.reference_frequency()))
    # abs=0.0 on purpose: the quantity is 1e-25 s^2 and approx's default absolute
    # tolerance of 1e-12 would accept zero, which is exactly the failure mode.
    assert out.accumulated_gvd == pytest.approx(expected * 0.1, rel=1e-9, abs=0.0)
    assert out.accumulated_gvd != 0.0


def ring_link(
    coupling: float, drop_coupling: float = 0.0
) -> tuple[Graph, RingResonator, dict[str, Component]]:
    graph = Graph(context())
    peak = resonance(coupling, drop_coupling)
    laser = graph.add(CWLaser(power=0.0, wavelength=frequency_to_wavelength(peak) * 1e9))
    ring = graph.add(
        RingResonator(
            length=LENGTH * 1e6,
            propagation_loss=LOSS / 100.0,
            coupling=coupling,
            drop_coupling=drop_coupling,
        )
    )
    through = graph.add(PowerMeter(label="th"))
    drop = graph.add(PowerMeter(label="dr"))
    graph.connect(laser, ring)
    graph.connect(ring["through"], through)
    graph.connect(ring["drop"], drop)
    return graph, ring, {"through": through, "drop": drop}


def test_the_ring_block_notches_a_carrier_sitting_on_resonance() -> None:
    """The device doing its job, through the dataflow engine rather than in isolation."""
    graph, _, meters = ring_link(coupling=CRITICAL)
    result = graph.run()
    assert w_to_dbm(result[meters["through"]].power_w) < -60.0

    # A carrier one linewidth off is not notched.
    off = resonance(CRITICAL) + resonance_linewidth(
        LENGTH, SILICON_STRIP_NGROUP, coupling=CRITICAL, loss_db_per_m=LOSS
    )
    graph2 = Graph(context())
    laser = graph2.add(CWLaser(power=0.0, wavelength=frequency_to_wavelength(off) * 1e9))
    ring = graph2.add(
        RingResonator(length=LENGTH * 1e6, coupling=CRITICAL, propagation_loss=LOSS / 100.0)
    )
    meter = graph2.add(PowerMeter())
    graph2.connect(laser, ring)
    graph2.connect(ring["through"], meter)
    assert w_to_dbm(graph2.run()[meter].power_w) > -4.0


def test_the_drop_port_passes_the_channel_and_the_through_port_does_not() -> None:
    graph, _, meters = ring_link(coupling=0.05, drop_coupling=0.05)
    result = graph.run()
    drop = w_to_dbm(result[meters["drop"]].power_w)
    through = w_to_dbm(result[meters["through"]].power_w)
    assert drop > -1.0
    assert through < -20.0


def test_the_ring_gates_ase_the_way_a_filter_does() -> None:
    """An amplifier's noise spans terahertz; the ring passes it one linewidth at a time.

    This is the second job every filter in this library has, and a resonator that
    got it wrong would look right on every carrier measurement.
    """
    graph = Graph(context())
    peak = resonance(0.05, 0.05)
    laser = graph.add(CWLaser(power=0.0, wavelength=frequency_to_wavelength(peak) * 1e9))
    amp = graph.add(EDFA(gain=20.0, noise_figure=5.0))
    ring = graph.add(
        RingResonator(
            length=LENGTH * 1e6, coupling=0.05, drop_coupling=0.05, propagation_loss=LOSS / 100.0
        )
    )
    before = graph.add(OSNRMeter(label="before"))
    after = graph.add(OSNRMeter(label="after"))
    graph.connect(laser, amp)
    graph.connect(amp, ring)
    graph.connect(amp, before)
    graph.connect(ring["drop"], after)
    result = graph.run()

    # The signal loses under a decibel; the noise loses fifteen.
    assert result[after] - result[before] > 12.0


def drop_response(coupling: float, drop: float, loss: float) -> Callable[[np.ndarray], np.ndarray]:
    def response(frequencies: np.ndarray) -> np.ndarray:
        return ring_resonator(
            frequencies,
            length=LENGTH,
            coupling=coupling,
            drop_coupling=drop,
            reference_frequency=F0,
            loss_db_per_m=loss,
        ).transmission("drop", "in")

    return response


def test_the_noise_average_needs_the_linewidth_and_a_high_q_ring_proves_it() -> None:
    """The two ways of getting an ASE figure wrong, measured against a converged one.

    A ring 116 MHz wide inside a 714 GHz free spectral range is a loaded Q of
    1.7 million — good, and buildable. Averaging its drop response over a 4 THz
    amplifier bin at a fixed four thousand points steps 977 MHz at a time and
    strides straight over the resonance; restricting the average to one period
    helps by a factor of six and is still not enough. Only a point count set from
    the linewidth converges.

    This is the test the first version of this file did not have, and the reason
    the code takes a ``resolution`` at all: without it, a sabotage that sampled
    the whole bin passed everything.
    """
    coupling, loss = 0.0005, 1.0
    response = drop_response(coupling, coupling, loss)
    width = resonance_linewidth(
        LENGTH,
        SILICON_STRIP_NGROUP,
        coupling=coupling,
        drop_coupling=coupling,
        loss_db_per_m=loss,
    )
    assert width == pytest.approx(116e6, rel=0.05)

    peak = resonance(coupling, coupling)
    bin_ = NoiseBin(f_start=peak - 2e12, f_end=peak + 2e12, psd_x=1.0, psd_y=1.0)
    converged = average_power_response(bin_, response, period=FSR, resolution=width)

    whole_bin = average_power_response(bin_, response)
    one_period = average_power_response(bin_, response, period=FSR)

    assert converged == pytest.approx(2.444e-4, rel=0.01)
    assert whole_bin / converged > 1.8  # 97 % high
    assert one_period / converged > 1.2  # 28 % high

    # And it really is converged: twice the resolution changes nothing.
    finer = average_power_response(bin_, response, period=FSR, resolution=width / 2)
    assert finer == pytest.approx(converged, rel=1e-4)


def test_an_ordinary_ring_does_not_need_the_ceiling() -> None:
    """The expensive path is only taken when the device asks for it."""
    coupling, loss = 0.05, LOSS
    response = drop_response(coupling, coupling, loss)
    width = resonance_linewidth(
        LENGTH,
        SILICON_STRIP_NGROUP,
        coupling=coupling,
        drop_coupling=coupling,
        loss_db_per_m=loss,
    )
    peak = resonance(coupling, coupling)
    bin_ = NoiseBin(f_start=peak - 2e12, f_end=peak + 2e12, psd_x=1.0, psd_y=1.0)

    coarse = average_power_response(bin_, response, period=FSR, resolution=width)
    fine = average_power_response(bin_, response, period=FSR, resolution=width / 8)
    assert coarse == pytest.approx(fine, rel=1e-3)

    # The comb passes about one linewidth in every free spectral range.
    assert coarse == pytest.approx(width / FSR, rel=0.5)


def test_both_polarizations_meet_the_same_device() -> None:
    """These are single-mode TE models, so the two axes must at least be treated alike.

    A block that filtered one axis and passed the other would be invisible to
    every measurement that launches into x, which is most of them.
    """
    graph = Graph(context())
    peak = resonance(CRITICAL)
    laser = graph.add(CWLaser(power=0.0, wavelength=frequency_to_wavelength(peak) * 1e9))
    turned = graph.add(PolarizationRotator(angle=30.0))
    ring = graph.add(
        RingResonator(length=LENGTH * 1e6, coupling=CRITICAL, propagation_loss=LOSS / 100.0)
    )
    meter = graph.add(PowerMeter())
    graph.connect(laser, turned)
    graph.connect(turned, ring)
    graph.connect(ring["through"], meter)
    results = graph.run(keep=[turned, ring])
    launched = results.port(turned, "out").bands[0]
    out = results.port(ring, "through").bands[0]

    # 30 degrees puts a quarter of the power on y. Both axes are extinguished.
    on_x = float(np.mean(np.abs(launched.Ex) ** 2))
    on_y = float(np.mean(np.abs(launched.Ey) ** 2))
    assert on_y > 0.2 * on_x
    assert float(np.mean(np.abs(out.Ex) ** 2)) < 1e-6 * on_x
    assert float(np.mean(np.abs(out.Ey) ** 2)) < 1e-6 * on_y


def test_the_ring_tells_the_noise_average_how_fine_to_look() -> None:
    """The block has to hand its linewidth down, and only a graph can check that it does.

    Everything above tests :func:`average_power_response` by calling it. This
    tests the wire between the ring and it: run a high-Q ring on a real amplifier
    bin and compare the ASE that survives against the converged mean of the same
    response. A block that forgot to pass its linewidth would still filter the
    carrier correctly and get the noise wrong by a tenth or more — here low rather
    than high, because an amplifier's band is not centred on a resonance, and
    which way an under-resolved grid misses is a matter of where it landed.
    """
    coupling, loss = 0.0005, 1.0
    graph = Graph(context())
    laser = graph.add(CWLaser(power=0.0, wavelength=1550.0))
    amp = graph.add(EDFA(gain=20.0, noise_figure=5.0))
    ring = graph.add(
        RingResonator(
            length=LENGTH * 1e6,
            coupling=coupling,
            drop_coupling=coupling,
            propagation_loss=loss / 100.0,
        )
    )
    meter = graph.add(PowerMeter())
    graph.connect(laser, amp)
    graph.connect(amp, ring)
    graph.connect(ring["drop"], meter)
    results = graph.run(keep=[amp, ring])

    before = results.port(amp, "out").noise[0]
    after = results.port(ring, "drop").noise[0]
    survived = after.psd_x / before.psd_x

    response = drop_response(coupling, coupling, loss)
    width = resonance_linewidth(
        LENGTH,
        SILICON_STRIP_NGROUP,
        coupling=coupling,
        drop_coupling=coupling,
        loss_db_per_m=loss,
    )
    converged = average_power_response(before, response, period=FSR, resolution=width)
    careless = average_power_response(before, response, period=FSR)

    assert survived == pytest.approx(converged, rel=1e-9)
    assert abs(careless / converged - 1.0) > 0.1


def test_the_add_port_is_off_unless_asked_for() -> None:
    """A block whose wiring becomes invalid when a number is edited is a bad block."""
    assert set(RingResonator().inputs) == {"in"}
    assert set(RingResonator(add_port=True).inputs) == {"in", "add"}
    assert set(RingResonator().outputs) == {"through", "drop"}
    assert RingResonator(add_port=True).structural_config() == {"add_port": True}


def test_the_add_port_reaches_the_drop_port() -> None:
    """Which is what makes it an add-drop multiplexer rather than a drop filter."""
    graph = Graph(context())
    peak = resonance(0.05, 0.05)
    signal = graph.add(
        CWLaser(power=0.0, wavelength=frequency_to_wavelength(peak) * 1e9, label="s")
    )
    added = graph.add(CWLaser(power=0.0, wavelength=1560.0, label="a"))
    ring = graph.add(
        RingResonator(
            add_port=True,
            length=LENGTH * 1e6,
            coupling=0.05,
            drop_coupling=0.05,
            propagation_loss=LOSS / 100.0,
        )
    )
    meter = graph.add(PowerMeter())
    graph.connect(signal, ring["in"])
    graph.connect(added, ring["add"])
    graph.connect(ring["drop"], meter)
    reading = graph.run()[meter]

    # Two carriers arrive: the dropped one and the added one, on separate bands.
    assert len(reading.bands) == 2


def test_the_shared_parameter_base_is_not_a_block() -> None:
    """It exists to declare parameters once; it has no ports and no place in a palette."""
    assert _Photonic.__dict__["abstract"] is True
    assert "_Photonic" not in registered_names()
    for name in ("Waveguide", "DirectionalCoupler", "RingResonator"):
        assert name in registered_names()
    # And its parameters really do reach the blocks that mix it in.
    assert "n_group" in Waveguide.param_specs()
    assert "n_group" in RingResonator.param_specs()
