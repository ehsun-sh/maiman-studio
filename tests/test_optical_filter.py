"""Wavelength-selective filtering, and the spectrum analyser that shows it.

Two components that only make sense against the signal model this project chose
on day one. Because every band carries its own centre frequency, a filter is a
real wavelength-selective operation rather than a choice of array index: it can
be tuned between two channels and attenuate both, or slightly off centre and
clip the edge of what it meant to pass. A single-carrier model could express
neither.

The assertions are arithmetic wherever they can be. A super-Gaussian's
equivalent noise bandwidth has a closed form, so the ASE power a filter passes is
a number to be checked rather than a shape to be eyeballed, and the OSA's trace
must integrate back to the power a power meter reports independently.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from oosim import Graph, SimulationContext
from oosim.component import Component
from oosim.components import (
    EDFA,
    BERAnalyzer,
    Combiner,
    CWLaser,
    ElectricalFilter,
    Fiber,
    MachZehnderModulator,
    NRZDriver,
    OpticalFilter,
    OpticalSpectrumAnalyzer,
    OSNRMeter,
    PINPhotodiode,
    PowerMeter,
    PRBSGenerator,
)
from oosim.kernels import (
    gaussian_noise_bandwidth,
    super_gaussian_noise_bandwidth,
    super_gaussian_response,
)
from oosim.units import C_LIGHT, wavelength_to_frequency

CHANNEL = 1550.0  # nm
SPACING = 100e9  # Hz


# --------------------------------------------------------------------------
# The shape
# --------------------------------------------------------------------------


@pytest.mark.parametrize("order", [1, 2, 3, 5, 10])
def test_the_declared_width_is_the_three_db_width_at_every_order(order: int) -> None:
    """A filter's stated bandwidth has to mean one thing whatever its shape.

    Otherwise raising the order to flatten the top would quietly narrow or widen
    the channel, and a comparison between orders would be measuring the
    definition rather than the filter.
    """
    width = 50e9
    edge = super_gaussian_response(np.array([width / 2.0]), width, order)[0] ** 2
    assert edge == pytest.approx(0.5, rel=1e-12)
    assert super_gaussian_response(np.array([0.0]), width, order)[0] == pytest.approx(1.0)


def test_order_one_is_the_gaussian_the_electrical_filter_already_uses() -> None:
    """The two closed forms must agree where they overlap, or one of them is wrong."""
    assert super_gaussian_noise_bandwidth(50e9, 1) == pytest.approx(
        gaussian_noise_bandwidth(50e9), rel=1e-12
    )


def test_noise_bandwidth_matches_numerical_integration() -> None:
    """``B_n = integral |H(f)|**2 df``, checked against the integral itself.

    The closed form uses a gamma function; the integral uses nothing but the
    response. If they agree at several orders the analytic expression is right,
    and every filtered-ASE figure below rests on it.
    """
    width = 50e9
    grid = np.linspace(-10.0 * width, 10.0 * width, 2_000_001)
    for order in (1, 2, 3, 5):
        numeric = float(np.trapezoid(super_gaussian_response(grid, width, order) ** 2, grid))
        assert numeric == pytest.approx(super_gaussian_noise_bandwidth(width, order), rel=1e-4)


def test_a_steeper_filter_passes_less_out_of_band_light() -> None:
    """The point of raising the order: the skirts, not the top."""
    width = 50e9
    far = np.array([width])  # one full width off centre
    gaussian = super_gaussian_response(far, width, 1)[0] ** 2
    flat_top = super_gaussian_response(far, width, 5)[0] ** 2
    assert flat_top < gaussian / 1000.0


# --------------------------------------------------------------------------
# Selecting a channel
# --------------------------------------------------------------------------


def two_channel_graph(
    filter_wavelength: float | None, *, bandwidth: float = 50.0, order: float = 3.0
) -> tuple[Graph, PowerMeter, OpticalSpectrumAnalyzer]:
    """Two carriers 100 GHz apart, optionally through a filter tuned to one."""
    ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=16, sequence_length=256, seed=2)
    graph = Graph(ctx)
    lower = wavelength_to_frequency(CHANNEL * 1e-9)
    upper_nm = C_LIGHT / (lower + SPACING) * 1e9

    first = graph.add(CWLaser(power=0.0, wavelength=CHANNEL, label="ch1"))
    second = graph.add(CWLaser(power=0.0, wavelength=upper_nm, label="ch2"))
    combiner = graph.add(Combiner(2))
    graph.connect(first, combiner["in0"])
    graph.connect(second, combiner["in1"])

    node: Component = combiner
    if filter_wavelength is not None:
        channel = graph.add(
            OpticalFilter(center_wavelength=filter_wavelength, bandwidth=bandwidth, order=order)
        )
        graph.connect(node, channel["in"])
        node = channel

    meter = graph.add(PowerMeter())
    osa = graph.add(OpticalSpectrumAnalyzer(center_wavelength=CHANNEL, span=400.0, points=2048))
    graph.connect(node, meter["in"])
    graph.connect(node, osa["in"])
    return graph, meter, osa


def test_a_filter_selects_one_channel_and_rejects_its_neighbour() -> None:
    """The demultiplexer. Both carriers arrive; one leaves.

    Rejection is asserted in decibels rather than as "smaller", because a
    demultiplexer that passes a neighbour at -6 dB would satisfy the weaker claim
    and be useless. A 50 GHz third-order filter is a full channel spacing away
    from the neighbour and must bury it.
    """
    graph, meter, _ = two_channel_graph(CHANNEL)
    reading = graph.run()[meter]

    powers = sorted(band.power_dbm for band in reading.bands)
    rejected, passed = powers[0], powers[1]
    assert passed == pytest.approx(0.0, abs=0.1)  # untouched
    assert passed - rejected > 30.0


def test_a_filter_tuned_between_two_channels_rejects_both() -> None:
    """Wavelength selectivity is real, not an index lookup.

    A model that picked bands by position in a list would pass the first channel
    here. This one is tuned 50 GHz from each and must attenuate them equally —
    which is also a check that the response is symmetric about its centre.
    """
    lower = wavelength_to_frequency(CHANNEL * 1e-9)
    midpoint_nm = C_LIGHT / (lower + SPACING / 2.0) * 1e9
    graph, meter, _ = two_channel_graph(midpoint_nm)
    reading = graph.run()[meter]

    powers = [band.power_dbm for band in reading.bands]
    assert max(powers) < -10.0
    assert powers[0] == pytest.approx(powers[1], abs=0.5)


def test_insertion_loss_applies_at_the_peak() -> None:
    """A 3 dB filter costs 3 dB to the channel it passes."""
    ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=8, sequence_length=128, seed=1)
    graph = Graph(ctx)
    laser = graph.add(CWLaser(power=0.0, wavelength=CHANNEL))
    lossy = graph.add(OpticalFilter(center_wavelength=CHANNEL, bandwidth=50.0, insertion_loss=3.0))
    meter = graph.add(PowerMeter())
    graph.chain(laser, lossy, meter)
    assert graph.run()[meter].power_dbm == pytest.approx(-3.0, abs=0.02)


# --------------------------------------------------------------------------
# Gating the ASE
# --------------------------------------------------------------------------


def amplified_spectrum(
    bandwidth: float | None, *, order: float = 3.0
) -> tuple[float, float, float]:
    """Returns (OSNR, ASE power, noise bin width) after an EDFA and a filter."""
    ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=8, sequence_length=128, seed=4)
    graph = Graph(ctx)
    laser = graph.add(CWLaser(power=-20.0, wavelength=CHANNEL))
    amplifier = graph.add(EDFA(gain=20.0, noise_figure=5.0))
    graph.connect(laser, amplifier["in"])

    node: Component = amplifier
    if bandwidth is not None:
        channel = graph.add(
            OpticalFilter(center_wavelength=CHANNEL, bandwidth=bandwidth, order=order)
        )
        graph.connect(node, channel["in"])
        node = channel

    meter = graph.add(PowerMeter())
    osnr = graph.add(OSNRMeter())
    graph.connect(node, meter["in"])
    graph.connect(node, osnr["in"])
    results = graph.run(keep=[node])
    signal = results.port(node, "out")
    width = max((b.bandwidth for b in signal.noise), default=0.0)
    return float(results[osnr]), signal.noise_power(), width


def test_the_filter_passes_exactly_its_equivalent_noise_bandwidth() -> None:
    """ASE power out is density times ``B_n``, and ``B_n`` has a closed form.

    This is the assertion the whole ASE story rests on, and it is arithmetic
    rather than a comparison: an EDFA's 4 THz of spontaneous emission must come
    out reduced by exactly the ratio the gamma function predicts.
    """
    _, unfiltered_power, unfiltered_width = amplified_spectrum(None)
    _, filtered_power, filtered_width = amplified_spectrum(50.0, order=3.0)

    expected_width = super_gaussian_noise_bandwidth(50e9, 3)
    assert filtered_width == pytest.approx(expected_width, rel=1e-9)
    assert filtered_power == pytest.approx(
        unfiltered_power * expected_width / unfiltered_width, rel=1e-9
    )


def test_filtering_ase_does_not_change_osnr() -> None:
    """OSNR is quoted in a fixed reference bandwidth, so cutting ASE outside it
    changes the noise *power* by orders of magnitude and the OSNR not at all.

    This is the distinction that makes OSNR a useful figure and a misleading one
    at the same time, and it is worth pinning: a filter improves a receiver
    enormously while leaving the number most people quote untouched.
    """
    wide_osnr, wide_power, _ = amplified_spectrum(400.0)
    narrow_osnr, narrow_power, _ = amplified_spectrum(25.0)

    assert wide_power > 10.0 * narrow_power
    assert wide_osnr == pytest.approx(narrow_osnr, abs=0.2)


# --------------------------------------------------------------------------
# The spectrum analyser
# --------------------------------------------------------------------------


def test_the_trace_integrates_back_to_the_measured_power() -> None:
    """An OSA and a power meter looking at the same point must agree.

    Two independent reductions of one signal — a periodogram summed across a
    display grid, and a time-domain average — so agreement is evidence that the
    OSA's normalisation is right rather than merely self-consistent.
    """
    graph, meter, osa = two_channel_graph(None)
    results = graph.run()
    spectrum = results[osa]
    reading = results[meter]

    assert spectrum.total_power() == pytest.approx(reading.power_w, rel=0.01)


def test_the_trace_puts_each_carrier_at_its_own_wavelength() -> None:
    """Bands are placed by their own centre frequency, not by array order."""
    graph, _, osa = two_channel_graph(None)
    spectrum = graph.run()[osa]

    peak_frequency, _ = spectrum.peak()
    lower = wavelength_to_frequency(CHANNEL * 1e-9)
    step = float(spectrum.frequencies[1] - spectrum.frequencies[0])
    assert min(abs(peak_frequency - lower), abs(peak_frequency - lower - SPACING)) <= step

    # Both carriers are present, a channel spacing apart.
    trace = np.asarray(spectrum.power_w)
    tall = np.asarray(spectrum.frequencies)[trace > trace.max() / 100.0]
    assert float(tall.max() - tall.min()) == pytest.approx(SPACING, rel=0.05)


def test_resolution_bandwidth_moves_ase_and_leaves_the_signal() -> None:
    """The reason OSNR needs a stated reference bandwidth, made visible.

    An OSA reports power per resolution bandwidth. Widening it by 6 dB raises a
    noise floor by 6 dB, because more of a flat density falls inside — while a
    carrier narrower than either setting does not move at all. A trace that
    scaled both together would be measuring nothing.
    """
    ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=8, sequence_length=128, seed=6)
    graph = Graph(ctx)
    laser = graph.add(CWLaser(power=0.0, wavelength=CHANNEL))
    amplifier = graph.add(EDFA(gain=20.0, noise_figure=5.0))
    narrow = graph.add(
        OpticalSpectrumAnalyzer(
            center_wavelength=CHANNEL, span=2000.0, points=2048, resolution_bandwidth=12.5
        )
    )
    wide = graph.add(
        OpticalSpectrumAnalyzer(
            center_wavelength=CHANNEL, span=2000.0, points=2048, resolution_bandwidth=50.0
        )
    )
    graph.connect(laser, amplifier["in"])
    graph.connect(amplifier, narrow["in"])
    graph.connect(amplifier, wide["in"])
    results = graph.run()

    fine, coarse = results[narrow], results[wide]
    ratio = 10.0 * math.log10(50.0 / 12.5)

    # A point well away from the carrier is pure ASE and must rise by the ratio.
    trace = np.asarray(fine.power_w)
    carrier = int(np.argmax(trace))
    off = (carrier + len(trace) // 4) % len(trace)
    # 0.01 dB, not machine precision: the resolution filter is a sampled Gaussian,
    # so its area is the declared bandwidth only to within the display grid.
    assert fine.power_dbm()[off] + ratio == pytest.approx(coarse.power_dbm()[off], abs=0.01)

    # The carrier is already inside either window, so it barely moves. It does not
    # hold perfectly still: a wider window also sweeps in more of the ASE sitting
    # around the carrier. What matters is that it moves by a fraction of what the
    # noise floor moves, which is the asymmetry the whole measurement rests on.
    carrier_shift = coarse.power_dbm()[carrier] - fine.power_dbm()[carrier]
    assert carrier_shift < ratio / 10.0


def test_the_ase_floor_reads_density_times_resolution_bandwidth() -> None:
    """The trace's absolute level against ASE, not just its ratio between settings.

    Every other spectrum assertion here compares two traces or two regions, and
    all of those survive a trace that is wrong by a constant. Dropping the display
    step when accumulating a noise density is exactly such an error, and it passed
    the resolution-bandwidth test because both traces carried it equally. The
    check that bites is arithmetic against the noise model itself: a flat density
    ``S`` must display as ``S * RBW``, in watts, wherever there is no carrier.
    """
    ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=8, sequence_length=128, seed=8)
    graph = Graph(ctx)
    laser = graph.add(CWLaser(power=-10.0, wavelength=CHANNEL))
    amplifier = graph.add(EDFA(gain=20.0, noise_figure=5.0))
    osa = graph.add(
        OpticalSpectrumAnalyzer(
            center_wavelength=CHANNEL, span=2000.0, points=4096, resolution_bandwidth=12.5
        )
    )
    graph.connect(laser, amplifier["in"])
    graph.connect(amplifier, osa["in"])
    results = graph.run(keep=[amplifier])

    spectrum = results[osa]
    amplified = results.port(amplifier, "out")
    psd_x, psd_y = amplified.noise_psd_at(wavelength_to_frequency(CHANNEL * 1e-9))
    expected = (psd_x + psd_y) * spectrum.resolution_bandwidth

    # A point far from the carrier but still inside the amplifier's noise bin.
    trace = np.asarray(spectrum.power_w)
    carrier = int(np.argmax(trace))
    off = (carrier + len(trace) // 3) % len(trace)
    assert spectrum.power_per_resolution()[off] == pytest.approx(expected, rel=0.02)


# --------------------------------------------------------------------------
# What a second channel changes
# --------------------------------------------------------------------------


def _comb(select: int | None) -> tuple[Graph, OSNRMeter, PowerMeter, BERAnalyzer]:
    """Four carriers, one modulated, amplified, optionally demultiplexed."""
    ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=16, sequence_length=1024, seed=17)
    graph = Graph(ctx)
    combiner = graph.add(Combiner(4))
    prbs = graph.add(PRBSGenerator(order=15.0))
    driver = graph.add(NRZDriver(v_low=4.0, v_high=0.0))
    graph.connect(prbs["out"], driver["in"])

    anchor = wavelength_to_frequency(CHANNEL * 1e-9)
    for index in range(4):
        nm = C_LIGHT / (anchor + index * SPACING) * 1e9
        laser = graph.add(CWLaser(power=0.0, wavelength=nm))
        if index == 1:
            modulator = graph.add(MachZehnderModulator(v_pi=4.0))
            graph.connect(laser, modulator["optical_in"])
            graph.connect(driver, modulator["electrical_in"])
            graph.connect(modulator, combiner[f"in{index}"])
        else:
            graph.connect(laser, combiner[f"in{index}"])

    node: Component = combiner
    for _ in range(4):
        fiber = graph.add(Fiber(length=80.0, attenuation=0.2, dispersion=0.0))
        amplifier = graph.add(EDFA(gain=16.0, noise_figure=6.0))
        graph.connect(node, fiber["in"])
        graph.connect(fiber, amplifier["in"])
        node = amplifier

    if select is not None:
        nm = C_LIGHT / (anchor + select * SPACING) * 1e9
        demux = graph.add(OpticalFilter(center_wavelength=nm, bandwidth=50.0, order=3.0))
        graph.connect(node, demux["in"])
        node = demux

    osnr = graph.add(OSNRMeter())
    meter = graph.add(PowerMeter())
    detector = graph.add(PINPhotodiode())
    receiver = graph.add(ElectricalFilter(bandwidth=7.0))
    analyzer = graph.add(BERAnalyzer())
    graph.connect(node, osnr["in"])
    graph.connect(node, meter["in"])
    graph.connect(node, detector["in"])
    graph.connect(detector, receiver["in"])
    graph.connect(receiver, analyzer["in"])
    graph.connect(prbs["out"], analyzer["reference"])
    return graph, osnr, meter, analyzer


def test_osnr_is_per_channel_after_a_demultiplexer() -> None:
    """OSNR must follow the surviving channel, not average the suppressed ones in.

    Summing every band's power over every band's noise is right only while the
    channels are equal. After a demultiplexer it puts one channel in the numerator
    and four channels' worth of noise in the denominator, and the survivor's OSNR
    reads about 6 dB low — low enough to look like a filter penalty rather than an
    arithmetic error.

    The check is that the demultiplexed OSNR sits within a modulator's insertion
    loss of the comb's, since that is the only real difference between the
    channel measured in each case.
    """
    comb_graph, comb_meter, _, _ = _comb(None)
    comb_value = float(comb_graph.run()[comb_meter])

    demux_graph, demux_meter, _, _ = _comb(1)
    demux_value = float(demux_graph.run()[demux_meter])

    # The comb's strongest channel is an unmodulated carrier; the demultiplexed
    # one is the modulated channel, 3 dB down through its MZM. Nothing else.
    assert comb_value - demux_value == pytest.approx(3.0, abs=0.5)


def test_the_detector_reads_ase_at_the_surviving_channel() -> None:
    """A demultiplexed link must still be ASE-limited.

    The detector has no wavelength selectivity, so it has to decide which band it
    is looking at, and taking the first one in the list is wrong the moment there
    is more than one: after a demultiplexer that band is a suppressed neighbour a
    full channel spacing away, its frequency falls outside the noise bins the
    filter has just clipped, and the ASE density there reads zero. The beat terms
    then contribute nothing and the link looks about four times better than its
    own OSNR permits — silently, because every other number stays plausible.
    """
    graph, meter, _, analyzer = _comb(1)
    results = graph.run()
    osnr_db = float(results[meter])
    q = results[analyzer].q_factor

    noise_bandwidth = 7e9 * math.sqrt(math.pi / (4.0 * math.log(2.0)))
    ratio = 10.0 ** (osnr_db / 10.0)
    limit = 2.0 * math.sqrt(12.5e9 / noise_bandwidth) * ratio / (1.0 + math.sqrt(1.0 + 4.0 * ratio))
    assert q < limit  # ASE-limited, on the correct side of its own limit
    assert q > 0.6 * limit  # and not by some unrelated margin


def test_out_of_band_rejection_stops_at_the_declared_extinction() -> None:
    """A real filter's skirts do not fall away forever, and neither may these.

    A third-order 50 GHz passband is ``exp(-2838)`` one channel spacing off
    centre. That is not a small number, it is zero in double precision — so a
    neighbour's power becomes exactly 0 W, its rejection reports as infinite, and
    a chain of such filters accumulates no crosstalk at all. Real hardware
    specifies 30 to 50 dB and it is that floor, not the skirt, that decides what
    leaks through a long line of them.
    """
    graph, meter, _ = two_channel_graph(CHANNEL, bandwidth=50.0, order=3.0)
    powers = sorted(band.power_dbm for band in graph.run()[meter].bands)
    rejected, passed = powers[0], powers[1]

    assert math.isfinite(rejected)
    assert passed - rejected == pytest.approx(40.0, abs=0.5)  # the declared default


def test_the_floor_can_be_switched_off() -> None:
    """Zero extinction means an ideal filter, for anyone who wants one."""
    ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=8, sequence_length=128, seed=9)
    graph = Graph(ctx)
    laser = graph.add(CWLaser(power=0.0, wavelength=CHANNEL))
    anchor = wavelength_to_frequency(CHANNEL * 1e-9)
    far_nm = C_LIGHT / (anchor + SPACING) * 1e9
    ideal = graph.add(
        OpticalFilter(center_wavelength=far_nm, bandwidth=50.0, order=3.0, extinction=0.0)
    )
    meter = graph.add(PowerMeter())
    graph.chain(laser, ideal, meter)
    assert graph.run()[meter].power_w == 0.0
