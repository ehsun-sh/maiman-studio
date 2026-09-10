"""Validation against closed-form results.

Every physics block is checked against an analytical expression, not against
another simulator. Comparing with a commercial tool needs a licence and cannot
run in CI, so it can confirm a model once but cannot keep it correct.

Each test names the relation it verifies.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from maiman import Graph, SimulationContext
from maiman.components import (
    Attenuator,
    BERAnalyzer,
    Combiner,
    CWLaser,
    ElectricalFilter,
    Fiber,
    MachZehnderModulator,
    NRZDriver,
    PINPhotodiode,
    PowerMeter,
    PRBSGenerator,
)
from maiman.units import dbm_to_w, w_to_dbm

# dBm tolerance. Fields are complex64 by default (~1e-7 relative), so a power
# error of 1e-6 relative is ~4e-6 dB; 1e-4 dB is comfortably above the noise
# floor of the representation and well below anything physically meaningful.
DB_TOL = 1e-4


@pytest.fixture
def ctx() -> SimulationContext:
    return SimulationContext(bit_rate=10e9, samples_per_symbol=16, sequence_length=64, seed=1234)


# --------------------------------------------------------------------------
# Fiber attenuation:  P_out = P_in * 10 ** (-alpha_dB_per_km * L_km / 10)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("p_in_dbm", "length_km", "alpha_db_per_km"),
    [
        (0.0, 80.0, 0.2),  # 16 dB — the canonical C-band span
        (10.0, 100.0, 0.2),  # 20 dB
        (-3.0, 40.0, 0.25),  # 10 dB
        (0.0, 0.0, 0.2),  # zero length: lossless
        (5.0, 25.0, 0.35),  # O-band-ish loss coefficient
    ],
)
def test_fiber_attenuation_matches_beer_lambert(
    ctx: SimulationContext, p_in_dbm: float, length_km: float, alpha_db_per_km: float
) -> None:
    g = Graph(ctx)
    laser = g.add(CWLaser(power=p_in_dbm, wavelength=1550.0))
    fiber = g.add(Fiber(length=length_km, attenuation=alpha_db_per_km))
    meter = g.add(PowerMeter())
    g.chain(laser, fiber, meter)

    reading = g.run()[meter]

    expected_dbm = p_in_dbm - alpha_db_per_km * length_km
    assert reading.power_dbm == pytest.approx(expected_dbm, abs=DB_TOL)

    # And in linear units, against the exponential form directly.
    expected_w = dbm_to_w(p_in_dbm) * 10 ** (-alpha_db_per_km * length_km / 10)
    assert reading.power_w == pytest.approx(expected_w, rel=1e-5)


def test_attenuation_is_multiplicative_across_cascaded_spans(ctx: SimulationContext) -> None:
    """Two 40 km spans must equal one 80 km span. Catches per-span offset errors."""
    g = Graph(ctx)
    laser = g.add(CWLaser(power=0.0))
    span_a = g.add(Fiber(length=40.0, attenuation=0.2, label="span_a"))
    span_b = g.add(Fiber(length=40.0, attenuation=0.2, label="span_b"))
    meter = g.add(PowerMeter())
    g.chain(laser, span_a, span_b, meter)

    assert g.run()[meter].power_dbm == pytest.approx(-16.0, abs=DB_TOL)


def test_attenuator_and_fiber_agree_for_equal_loss(ctx: SimulationContext) -> None:
    """A 16 dB attenuator and an 80 km / 0.2 dB/km span must give the same power."""
    readings = []
    for element in (Attenuator(attenuation=16.0), Fiber(length=80.0, attenuation=0.2)):
        g = Graph(ctx)
        laser = g.add(CWLaser(power=0.0))
        loss = g.add(element)
        meter = g.add(PowerMeter())
        g.chain(laser, loss, meter)
        readings.append(g.run()[meter].power_w)

    assert readings[0] == pytest.approx(readings[1], rel=1e-6)


# --------------------------------------------------------------------------
# Source
# --------------------------------------------------------------------------


def test_laser_power_is_independent_of_time_window() -> None:
    """Average power must not depend on how many samples we happen to simulate.

    A normalisation that divides by the wrong length passes with one window and
    fails with another, so this is checked explicitly rather than assumed.
    """
    powers = []
    for sequence_length in (16, 64, 256):
        ctx = SimulationContext(
            bit_rate=10e9, samples_per_symbol=8, sequence_length=sequence_length
        )
        g = Graph(ctx)
        laser = g.add(CWLaser(power=3.0))
        meter = g.add(PowerMeter())
        g.chain(laser, meter)
        powers.append(g.run()[meter].power_dbm)

    assert powers == pytest.approx([3.0] * 3, abs=DB_TOL)


def test_phase_noise_changes_phase_but_not_average_power() -> None:
    """Linewidth broadens the line; it must not add or remove power."""
    ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=16, sequence_length=256, seed=7)

    g = Graph(ctx)
    laser = g.add(CWLaser(power=0.0, linewidth=1000.0))  # 1 MHz
    meter = g.add(PowerMeter())
    g.chain(laser, meter)
    results = g.run(keep=[laser])

    assert results[meter].power_dbm == pytest.approx(0.0, abs=DB_TOL)

    band = results.port(laser, "out").bands[0]
    phase = np.angle(band.Ex)
    assert np.std(phase) > 0.0, "linewidth was declared but the phase is constant"


#: Noise bandwidth of the Gaussian receiver filter as a multiple of its 3 dB
#: cutoff: ``sqrt(pi / (4 ln2))``. Written here rather than imported so that a
#: change to the filter has to be argued with the closed form rather than
#: silently agreeing with itself.
GAUSSIAN_NOISE_BANDWIDTH = math.sqrt(math.pi / (4.0 * math.log(2.0)))


def detected_rin_snr(rin_db: float, power_dbm: float, cutoff_ghz: float = 7.0) -> float:
    """Photocurrent SNR of a CW laser detected with every other noise turned off."""
    ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=16, sequence_length=8192, seed=3)
    g = Graph(ctx)
    laser = g.add(CWLaser(power=power_dbm, rin=rin_db, label="tx"))
    # Shot and thermal off, so what is left in the current is the laser's own
    # intensity noise and nothing else. This is the one measurement where that
    # is the honest configuration rather than an optimistic one.
    pin = g.add(PINPhotodiode(responsivity=0.8, shot_noise=False, thermal_noise=False, label="pin"))
    lpf = g.add(ElectricalFilter(bandwidth=cutoff_ghz, label="lpf"))
    g.chain(laser, pin, lpf)

    current = np.asarray(g.run(keep=[lpf])[lpf].samples)
    return float(current.mean() ** 2 / current.var())


@pytest.mark.parametrize("rin_db", [-155.0, -145.0, -135.0])
def test_rin_limited_snr_is_one_over_rin_times_bandwidth(rin_db: float) -> None:
    """The closed form every RIN figure on a datasheet is quoted against.

    ``SNR = 1 / (RIN * B_n)``, with ``B_n`` the receiver's *noise* bandwidth —
    which for the Gaussian filter here is 1.0645 times its 3 dB cutoff and is
    exact rather than approximate, which is why this can be an equality and not
    an order of magnitude.
    """
    noise_bandwidth = 7.0e9 * GAUSSIAN_NOISE_BANDWIDTH
    expected = 1.0 / (10.0 ** (rin_db / 10.0) * noise_bandwidth)
    assert detected_rin_snr(rin_db, 0.0) == pytest.approx(expected, rel=0.01)


def test_the_rin_floor_does_not_move_when_power_is_raised() -> None:
    """The property that makes RIN worth modelling at all.

    Shot and thermal noise fall behind the signal as power rises, which is why
    every sensitivity curve in this project keeps improving. Intensity noise
    scales *with* the signal, so the ratio is fixed and ten times the power buys
    nothing. A model that got this wrong would still pass a variance check and
    would be useless for the one question RIN is asked.
    """
    low = detected_rin_snr(-145.0, 0.0)
    high = detected_rin_snr(-145.0, 10.0)
    assert high == pytest.approx(low, rel=1e-9), "ten times the power must change nothing"


def test_rin_scales_with_the_declared_density() -> None:
    """Ten dB more noise is ten dB less signal-to-noise, exactly."""
    quiet = detected_rin_snr(-155.0, 0.0)
    noisy = detected_rin_snr(-145.0, 0.0)
    assert quiet / noisy == pytest.approx(10.0, rel=0.02)


def test_intensity_noise_changes_power_sample_to_sample_but_not_on_average() -> None:
    """The counterpart of the linewidth invariant above, and the same argument.

    Phase noise must not move the average power; intensity noise must not either.
    It moves the *variance*, and a model that quietly added or removed power
    would show up here rather than as a link budget that never quite closes.
    """
    ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=16, sequence_length=8192, seed=11)
    g = Graph(ctx)
    laser = g.add(CWLaser(power=0.0, rin=-140.0))
    meter = g.add(PowerMeter())
    g.chain(laser, meter)
    results = g.run(keep=[laser])

    assert results[meter].power_dbm == pytest.approx(0.0, abs=0.01)

    power = np.abs(results.port(laser, "out").bands[0].Ex) ** 2
    relative_variance = float(power.var() / power.mean() ** 2)
    # RIN * fs / 2, the one-sided bandwidth a sampled window carries.
    expected = 10.0 ** (-140.0 / 10.0) * ctx.sample_rate / 2.0
    assert relative_variance == pytest.approx(expected, rel=0.05)


def test_an_ideal_laser_is_the_default_and_stays_exact() -> None:
    """``rin = 0`` is a sentinel for "no intensity noise", not 0 dB/Hz of it.

    Worth asserting because the sentinel is the one place this parameter is not
    its own logarithm, and because every result in this repository was measured
    with it and would move if it ever started meaning what it says.
    """
    ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=16, sequence_length=256, seed=5)
    g = Graph(ctx)
    laser = g.add(CWLaser(power=0.0))
    g.add(PowerMeter())
    g.chain(laser, g.components[-1])

    power = np.abs(g.run(keep=[laser]).port(laser, "out").bands[0].Ex) ** 2
    assert np.ptp(power) == 0.0, "the default laser must be exactly constant"


def test_the_two_laser_noises_are_drawn_from_separate_streams() -> None:
    """Changing the linewidth must not move the intensity samples, or vice versa.

    Otherwise an experiment that sweeps one and reads the other back is measuring
    both, and the plot looks like a physical coupling that is not there.
    """
    ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=16, sequence_length=512, seed=9)

    def intensity(linewidth: float) -> np.ndarray:
        g = Graph(ctx)
        laser = g.add(CWLaser(power=0.0, rin=-140.0, linewidth=linewidth, label="tx"))
        g.add(PowerMeter())
        g.chain(laser, g.components[-1])
        return np.abs(g.run(keep=[laser]).port(laser, "out").bands[0].Ex) ** 2

    assert np.allclose(intensity(0.0), intensity(1000.0), rtol=1e-12)


def ook_q_db(rin_db: float, power_dbm: float) -> float:
    """Q of a short OOK link, in dB, with everything but the laser held fixed."""
    ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=16, sequence_length=8192, seed=7)
    g = Graph(ctx)
    prbs = g.add(PRBSGenerator(order=15.0, label="prbs"))
    driver = g.add(NRZDriver(v_low=4.0, v_high=0.0, label="drv"))
    laser = g.add(CWLaser(power=power_dbm, rin=rin_db, label="tx"))
    modulator = g.add(MachZehnderModulator(v_pi=4.0, extinction_ratio=30.0, label="mzm"))
    detector = g.add(PINPhotodiode(responsivity=0.8, label="pin"))
    lowpass = g.add(ElectricalFilter(bandwidth=7.0, label="lpf"))
    analyzer = g.add(BERAnalyzer(label="ber"))

    g.chain(prbs, driver)
    g.connect(laser, modulator["optical_in"])
    g.connect(driver, modulator["electrical_in"])
    g.chain(modulator, detector, lowpass)
    g.connect(lowpass, analyzer["in"])
    g.connect(prbs["out"], analyzer["reference"])
    return float(20.0 * math.log10(g.run()[analyzer].q_factor))


def test_the_rin_penalty_grows_with_launch_power() -> None:
    """The link-level consequence, and the one that inverts the usual intuition.

    Every other noise here falls behind the signal as power rises. This one does
    not, so the penalty it costs is *larger* at high power than at low — which is
    the opposite of how a reader trained on shot and thermal noise expects a
    noise term to behave, and the reason the README quotes a penalty rather than
    a sensitivity.

    Quoted as a difference at equal power on purpose: it does not then depend on
    whatever else limits the ideal link at the top of the sweep.
    """
    penalties = [ook_q_db(0.0, p) - ook_q_db(-135.0, p) for p in (-20.0, -10.0, 0.0)]

    assert penalties[0] < 0.1, "at -20 dBm thermal noise should bury it entirely"
    assert penalties[1] > 0.5, "by -10 dBm it should be visible"
    assert penalties[2] > 5.0, "and by 0 dBm it should dominate"
    assert penalties == sorted(penalties), "the penalty must grow with power, not shrink"


def test_laser_wavelength_maps_to_the_expected_optical_frequency(ctx: SimulationContext) -> None:
    g = Graph(ctx)
    laser = g.add(CWLaser(wavelength=1550.0))
    meter = g.add(PowerMeter())
    g.chain(laser, meter)

    (band,) = g.run()[meter].bands
    # c / 1550 nm = 193.4145 THz, the ITU-T G.694.1 grid anchor region.
    assert band.f0 == pytest.approx(193.414489e12, rel=1e-6)
    assert band.wavelength_nm == pytest.approx(1550.0, rel=1e-9)


# --------------------------------------------------------------------------
# Multi-band (WDM) — proves the signal model is real, not decorative
# --------------------------------------------------------------------------


def test_two_carriers_propagate_as_separate_bands(ctx: SimulationContext) -> None:
    """Two wavelengths through one fiber must stay two independently sampled bands.

    This is the test that keeps `OpticalSignal.bands` honest. A single-carrier
    model passes every attenuation test above while being fundamentally unable
    to represent a WDM system; only exercising a second band shows the
    difference, which is why this is here in the first week and not in Phase 3.
    """
    g = Graph(ctx)
    ch1 = g.add(CWLaser(power=0.0, wavelength=1550.0, label="ch1"))
    ch2 = g.add(CWLaser(power=-3.0, wavelength=1551.0, label="ch2"))
    mux = g.add(Combiner(2))
    fiber = g.add(Fiber(length=80.0, attenuation=0.2))
    meter = g.add(PowerMeter())

    g.connect(ch1, mux["in0"])
    g.connect(ch2, mux["in1"])
    g.chain(mux, fiber, meter)

    reading = g.run()[meter]

    assert len(reading.bands) == 2, "the two carriers did not survive as separate bands"

    lower, upper = reading.bands  # sorted by frequency: 1551 nm is the lower one
    assert upper.wavelength_nm == pytest.approx(1550.0, rel=1e-6)
    assert lower.wavelength_nm == pytest.approx(1551.0, rel=1e-6)

    # Each channel is attenuated by its own 16 dB, independently.
    assert upper.power_dbm == pytest.approx(0.0 - 16.0, abs=DB_TOL)
    assert lower.power_dbm == pytest.approx(-3.0 - 16.0, abs=DB_TOL)

    # And the total is their incoherent sum.
    expected_total = w_to_dbm(dbm_to_w(-16.0) + dbm_to_w(-19.0))
    assert reading.power_dbm == pytest.approx(expected_total, abs=DB_TOL)


def test_bands_stay_narrow_regardless_of_channel_spacing(ctx: SimulationContext) -> None:
    """Channel spacing must not drive the sample rate.

    1550 nm and 1500 nm are ~6.4 THz apart. If bands were forced onto one grid,
    representing both would need a sample rate in the terahertz. Each band keeps
    the context sample rate instead — which is the entire reason for the design.
    """
    g = Graph(ctx)
    ch1 = g.add(CWLaser(wavelength=1550.0, label="ch1"))
    ch2 = g.add(CWLaser(wavelength=1500.0, label="ch2"))
    mux = g.add(Combiner(2))
    meter = g.add(PowerMeter())

    g.connect(ch1, mux["in0"])
    g.connect(ch2, mux["in1"])
    g.chain(mux, meter)
    results = g.run(keep=[mux])

    signal = results.port(mux, "out")
    spacing = abs(signal.bands[0].f0 - signal.bands[1].f0)
    assert spacing > 6e12, "expected multi-THz spacing for this test to mean anything"
    for band in signal.bands:
        assert band.fs == ctx.sample_rate
        assert band.num_samples == ctx.num_samples


def test_combining_co_located_carriers_is_rejected(ctx: SimulationContext) -> None:
    """Identical centre frequencies interfere; multiplexing them is not the same
    operation as adding them, so the engine refuses rather than guessing."""
    g = Graph(ctx)
    ch1 = g.add(CWLaser(wavelength=1550.0, label="ch1"))
    ch2 = g.add(CWLaser(wavelength=1550.0, label="ch2"))
    mux = g.add(Combiner(2))
    meter = g.add(PowerMeter())

    g.connect(ch1, mux["in0"])
    g.connect(ch2, mux["in1"])
    g.chain(mux, meter)

    with pytest.raises(ValueError, match="coherently"):
        g.run()


def test_combiner_insertion_loss_applies_to_every_channel(ctx: SimulationContext) -> None:
    g = Graph(ctx)
    ch1 = g.add(CWLaser(power=0.0, wavelength=1550.0, label="ch1"))
    ch2 = g.add(CWLaser(power=0.0, wavelength=1551.0, label="ch2"))
    mux = g.add(Combiner(2, insertion_loss=3.0))
    meter = g.add(PowerMeter())

    g.connect(ch1, mux["in0"])
    g.connect(ch2, mux["in1"])
    g.chain(mux, meter)

    for band in g.run()[meter].bands:
        assert band.power_dbm == pytest.approx(-3.0, abs=DB_TOL)
