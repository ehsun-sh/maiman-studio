"""Signal-ASE beat noise, in both detector families.

Before this existed the project could compute OSNR on an amplified chain to a
hundredth of a decibel and then report a Q-factor that had almost nothing to do
with it. A photodiode squares the field, so ASE arriving with the signal beats
against it rather than merely adding its power; leaving that out made an eight
span link look 12x better than its own OSNR allowed.

The measurements here are anchored to results derived outside this code. For
direct detection that is the standard OSNR-to-Q relation for NRZ-OOK,

    Q = 2 sqrt(B_ref/B_e) OSNR / (1 + sqrt(1 + 4 OSNR))

which is checked against a known point before it is trusted: at 14.5 dB it must
give Q near 6, the industry figure for a 10 Gb/s link at BER 1e-9. For coherent
detection it is ``SNR = 2 OSNR B_ref / R_s``, which follows from the same OSNR
definition once the matched filter has confined the noise to the symbol band.

Neither is expected to be matched exactly. The model carries shot, thermal and
transmitter impairments that the closed forms omit, so it should sit slightly
*below* them and converge as ASE grows to dominate. That convergence is the
assertion — a single number agreeing at one operating point would prove much
less than a fixed offset shrinking to nothing across a sweep.
"""

from __future__ import annotations

import math

import pytest

from maiman import Graph, SimulationContext
from maiman.component import Component
from maiman.components import (
    EDFA,
    BERAnalyzer,
    CoherentReceiver,
    ConstellationAnalyzer,
    CWLaser,
    ElectricalFilter,
    Fiber,
    IQDriver,
    IQModulator,
    IQSampler,
    MachZehnderModulator,
    NRZDriver,
    OpticalFilter,
    OSNRMeter,
    PINPhotodiode,
    PRBSGenerator,
    QAMMapper,
)
from maiman.signals import NoiseBin, OpticalSignal

OSNR_REFERENCE = 12.5e9
FILTER_BANDWIDTH = 7e9
ELECTRICAL_NOISE_BANDWIDTH = 7.4513e9  # gaussian_noise_bandwidth(7 GHz)


def q_from_osnr(osnr_db: float, noise_bandwidth: float = ELECTRICAL_NOISE_BANDWIDTH) -> float:
    """Textbook Q for NRZ-OOK limited by ASE beat noise."""
    ratio = 10.0 ** (osnr_db / 10.0)
    return (
        2.0
        * math.sqrt(OSNR_REFERENCE / noise_bandwidth)
        * ratio
        / (1.0 + math.sqrt(1.0 + 4.0 * ratio))
    )


def test_the_reference_relation_reproduces_its_own_known_point() -> None:
    """Before using a formula as an oracle, check it where the answer is known.

    A 10 Gb/s NRZ link needs about 14.5 dB OSNR for a BER of 1e-9, which is a
    Q of 6. If this did not hold, every comparison below would be measuring the
    formula rather than the model.
    """
    assert q_from_osnr(14.5) == pytest.approx(6.0, abs=0.6)


# --------------------------------------------------------------------------
# Direct detection
# --------------------------------------------------------------------------


def amplified_ook(
    noise_figure: float,
    *,
    spans: int = 8,
    beat: bool = True,
    optical_bandwidth: float = 50.0,
) -> tuple[float, float]:
    """A 10 Gb/s OOK link over ``spans`` amplified 80 km sections.

    Dispersion is off so that the only thing degrading the eye is noise, and the
    amplifiers exactly undo each span's loss so that OSNR is set by the noise
    figure alone.
    """
    ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=16, sequence_length=2048, seed=5)
    graph = Graph(ctx)
    prbs = graph.add(PRBSGenerator(order=15.0))
    driver = graph.add(NRZDriver(v_low=4.0, v_high=0.0))  # 1 -> 0 V -> peak transmission
    laser = graph.add(CWLaser(power=0.0, wavelength=1550.0))
    modulator = graph.add(MachZehnderModulator(v_pi=4.0))
    graph.connect(prbs["out"], driver["in"])
    graph.connect(driver, modulator["electrical_in"])
    graph.connect(laser, modulator["optical_in"])

    node: Component = modulator
    for _ in range(spans):
        fiber = graph.add(Fiber(length=80.0, attenuation=0.2, dispersion=0.0))
        amplifier = graph.add(EDFA(gain=16.0, noise_figure=noise_figure))
        graph.connect(node, fiber["in"])
        graph.connect(fiber, amplifier["in"])
        node = amplifier

    if optical_bandwidth > 0.0:
        channel_filter = graph.add(
            OpticalFilter(center_wavelength=1550.0, bandwidth=optical_bandwidth)
        )
        graph.connect(node, channel_filter["in"])
        node = channel_filter

    meter = graph.add(OSNRMeter())
    detector = graph.add(PINPhotodiode(ase_beat_noise=beat))
    receiver_filter = graph.add(ElectricalFilter(bandwidth=FILTER_BANDWIDTH / 1e9))
    analyzer = graph.add(BERAnalyzer())
    graph.connect(node, meter["in"])
    graph.connect(node, detector["in"])
    graph.connect(detector, receiver_filter["in"])
    graph.connect(receiver_filter, analyzer["in"])
    graph.connect(prbs["out"], analyzer["reference"])

    results = graph.run()
    return float(results[meter]), results[analyzer].q_factor


@pytest.mark.parametrize("noise_figure", [4.0, 8.0, 14.0])
def test_amplified_q_tracks_the_osnr_relation(noise_figure: float) -> None:
    """The headline. Q now follows from OSNR instead of ignoring it.

    Fifteen percent is a wide tolerance and deliberately so: the closed form
    assumes an ideal extinction ratio and no receiver noise, and this link has
    both. What it may not do is be *better* than the ideal, which is what the
    missing beat term used to allow.
    """
    osnr_db, q = amplified_ook(noise_figure)
    expected = q_from_osnr(osnr_db)
    assert q == pytest.approx(expected, rel=0.15)
    assert q < expected  # the model carries noise the closed form does not


def test_without_the_beat_term_the_link_looks_an_order_of_magnitude_better() -> None:
    """The other half, without which the test above proves nothing.

    Both are needed: agreement with a formula means little unless removing the
    mechanism breaks it. At 16 dB OSNR the old model gave Q = 95 where the OSNR
    allows about 8.
    """
    osnr_db, honest = amplified_ook(14.0)
    _, optimistic = amplified_ook(14.0, beat=False)

    assert honest == pytest.approx(q_from_osnr(osnr_db), rel=0.15)
    assert optimistic > 10.0 * honest


def test_q_responds_to_the_amplifier_noise_figure() -> None:
    """Ten dB of OSNR has to cost something. It used to cost 0.0."""
    quiet_osnr, quiet_q = amplified_ook(4.0)
    noisy_osnr, noisy_q = amplified_ook(14.0)

    assert quiet_osnr - noisy_osnr == pytest.approx(10.0, abs=0.2)
    assert quiet_q > 2.5 * noisy_q


def test_the_optical_filter_cuts_power_and_beat_together() -> None:
    """Narrowing the channel filter must improve the link.

    It is the one component that reduces ASE-ASE beating without touching the
    signal much, and it does so by cutting the noise bins the detector then
    integrates — the diode has no wavelength selectivity of its own. An earlier
    version of the detector applied a passband to the beat term but let the whole
    amplifier band's ASE power reach it anyway, which is two different receivers
    averaged together rather than a conservative approximation.
    """
    _, narrow = amplified_ook(14.0, optical_bandwidth=25.0)
    _, wide = amplified_ook(14.0, optical_bandwidth=400.0)
    assert narrow > wide


def test_unamplified_links_are_untouched() -> None:
    """No noise bins, no beat term, no change to anything already validated."""
    _, with_beat = amplified_ook(4.0, spans=0, beat=True)
    _, without = amplified_ook(4.0, spans=0, beat=False)
    assert with_beat == pytest.approx(without, rel=1e-12)


def _detector_noise(ctx: SimulationContext, signal_x: bool, ase_x: bool) -> float:
    """Std of the photocurrent for a signal on one axis and ASE on one axis."""
    import numpy as np

    from maiman.signals import Band

    detector = PINPhotodiode(shot_noise=False, thermal_noise=False)
    detector.label = "pin"
    n = ctx.num_samples
    field = np.full(n, math.sqrt(1e-3), dtype=np.complex128)
    zero = np.zeros(n, dtype=np.complex128)
    f0 = 193.4e12
    band = Band(
        f0=f0,
        Ex=field if signal_x else zero,
        Ey=zero if signal_x else field,
        fs=ctx.sample_rate,
    )
    density = 2e-17
    noise = NoiseBin(f0 - 50e9, f0 + 50e9, density if ase_x else 0.0, 0.0 if ase_x else density)
    signal = OpticalSignal(bands=(band,), noise=(noise,))
    return float(np.std(detector.run(ctx, {"in": signal})["out"].samples))


def test_signal_spontaneous_beat_is_polarization_selective() -> None:
    """ASE orthogonal to the signal cannot beat with it.

    **All four pairings, not just one.** With the signal pinned to X, swapping
    the two polarization terms in the beat expression changes nothing — the Y
    contribution is zero either way — so a test built only on an X signal passes
    against code that has thrown the distinction away. That exact sabotage
    survived the first version of this test. The signal has to be put on Y as
    well for the assertion to have any content.

    Co-polarized pairs must be loud, orthogonal pairs quiet, and the two
    co-polarized cases must agree with each other: nothing in the physics
    prefers an axis.
    """
    ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=8, sequence_length=512, seed=3)

    xx = _detector_noise(ctx, signal_x=True, ase_x=True)
    xy = _detector_noise(ctx, signal_x=True, ase_x=False)
    yy = _detector_noise(ctx, signal_x=False, ase_x=False)
    yx = _detector_noise(ctx, signal_x=False, ase_x=True)

    assert xx > 5.0 * xy
    assert yy > 5.0 * yx
    assert xx == pytest.approx(yy, rel=0.2)
    assert xy == pytest.approx(yx, rel=0.2)


def test_overlapping_noise_bins_are_summed() -> None:
    """Each amplifier appends its own bin, and they cover the same band.

    Reading only the first is an undercount by the number of amplifiers — a
    factor of eight on the link above, silent, and consistent enough across a
    sweep to look like a modelling choice rather than a bug. It was one.
    """
    f0 = 193.4e12
    one = NoiseBin(f0 - 1e12, f0 + 1e12, 3e-17, 1e-17)
    stacked = OpticalSignal(noise=(one,) * 8)
    assert stacked.noise_psd_at(f0) == pytest.approx((2.4e-16, 8e-17))
    assert stacked.noise_psd_at(f0 - 5e12) == (0.0, 0.0)


# --------------------------------------------------------------------------
# Coherent detection
# --------------------------------------------------------------------------

SYMBOL_RATE = 32e9


def amplified_coherent(
    noise_figure: float, *, matched: bool = True, samples_per_symbol: int = 8
) -> tuple[float, float]:
    """QPSK over six amplified spans, measured against the same OSNR."""
    ctx = SimulationContext(
        bit_rate=SYMBOL_RATE,
        samples_per_symbol=samples_per_symbol,
        sequence_length=4096,
        seed=11,
    )
    graph = Graph(ctx)
    prbs = graph.add(PRBSGenerator(order=23.0, bits_per_symbol=2.0))
    mapper = graph.add(QAMMapper(bits_per_symbol=2.0))
    driver = graph.add(
        IQDriver(v_pi=4.0, predistort=True, drive_ratio=0.4, pulse_shaping=matched, roll_off=0.05)
    )
    laser = graph.add(CWLaser(power=3.0, wavelength=1550.0, linewidth=0.0))
    modulator = graph.add(IQModulator(v_pi=4.0))
    graph.connect(prbs["out"], mapper["in"])
    graph.connect(mapper["out"], driver["in"])
    graph.connect(laser, modulator["optical_in"])
    graph.connect(driver["i"], modulator["i"])
    graph.connect(driver["q"], modulator["q"])

    node: Component = modulator
    for _ in range(6):
        fiber = graph.add(Fiber(length=80.0, attenuation=0.2, dispersion=0.0))
        amplifier = graph.add(EDFA(gain=16.0, noise_figure=noise_figure))
        graph.connect(node, fiber["in"])
        graph.connect(fiber, amplifier["in"])
        node = amplifier

    meter = graph.add(OSNRMeter())
    lo = graph.add(CWLaser(power=13.0, wavelength=1550.0, linewidth=0.0))
    receiver = graph.add(CoherentReceiver(responsivity=0.8))
    sampler = graph.add(IQSampler(matched_filter=matched, roll_off=0.05))
    analyzer = graph.add(ConstellationAnalyzer(ignore_edges=64.0))
    graph.connect(node, meter["in"])
    graph.connect(node, receiver["in"])
    graph.connect(lo, receiver["lo"])
    graph.connect(receiver["i"], sampler["i"])
    graph.connect(receiver["q"], sampler["q"])
    graph.connect(mapper["out"], sampler["reference"])
    graph.connect(sampler["out"], analyzer["in"])
    graph.connect(mapper["out"], analyzer["reference"])

    results = graph.run()
    return float(results[meter]), results[analyzer].snr_db


def coherent_snr_target(osnr_db: float) -> float:
    """``SNR = 2 OSNR B_ref / R_s`` once the matched filter has done its job."""
    return osnr_db + 10.0 * math.log10(2.0 * OSNR_REFERENCE / SYMBOL_RATE)


def test_coherent_snr_converges_on_the_osnr_relation() -> None:
    """As ASE comes to dominate, electrical SNR must approach its optical limit.

    The gap is the receiver's own shot and thermal noise plus the transmitter's
    imperfections — a fixed floor. It should shrink monotonically as ASE grows,
    and end close to zero. Asserting the *convergence* rather than a single
    figure is what makes this a test of the beat term rather than of one
    operating point.
    """
    gaps = []
    for noise_figure in (4.0, 10.0, 16.0):
        osnr_db, snr_db = amplified_coherent(noise_figure)
        gaps.append(coherent_snr_target(osnr_db) - snr_db)

    assert gaps[0] > gaps[1] > gaps[2]  # monotonically closing
    assert gaps[2] < 0.5  # and essentially closed once ASE dominates
    assert all(gap > 0.0 for gap in gaps)  # never better than the optical limit


def test_matched_filtering_is_worth_the_sampled_bandwidth_ratio() -> None:
    """Without it the receiver collects ASE across the whole sampled band.

    The penalty is ``10 log10(fs / R_s)`` — 9.0 dB at eight samples per symbol —
    and it is a real result rather than an artefact: a sampler with no filter in
    front of it genuinely integrates noise it has no use for. Getting this
    backwards is what made the first version of the coherent check look like a
    modelling error rather than a missing filter.
    """
    _, unfiltered = amplified_coherent(16.0, matched=False)
    _, filtered = amplified_coherent(16.0, matched=True)
    expected = 10.0 * math.log10(8.0)
    assert filtered - unfiltered == pytest.approx(expected, abs=1.0)


def test_coherent_noise_responds_to_the_amplifier() -> None:
    """The LO beats with ASE, so the amplifier's noise figure has to show up."""
    quiet_osnr, quiet = amplified_coherent(4.0)
    noisy_osnr, noisy = amplified_coherent(16.0)
    assert quiet_osnr - noisy_osnr == pytest.approx(12.0, abs=0.3)
    assert quiet - noisy > 8.0


def _coherent_beat_noise(ctx: SimulationContext, lo_x: bool, ase_x: bool) -> float:
    """Std of the I photocurrent for an LO on one axis and ASE on one axis."""
    import numpy as np

    from maiman.signals import Band

    receiver = CoherentReceiver(shot_noise=False, thermal_noise=False, responsivity=0.8)
    receiver.label = "rx"
    n = ctx.num_samples
    zero = np.zeros(n, dtype=np.complex128)
    f0 = 193.4e12
    lo_field = np.full(n, math.sqrt(20e-3), dtype=np.complex128)
    lo = OpticalSignal(
        bands=(
            Band(
                f0=f0,
                Ex=lo_field if lo_x else zero,
                Ey=zero if lo_x else lo_field,
                fs=ctx.sample_rate,
            ),
        )
    )
    signal_field = np.full(n, math.sqrt(1e-6), dtype=np.complex128)
    density = 5e-17
    signal = OpticalSignal(
        bands=(Band(f0=f0, Ex=signal_field, Ey=zero, fs=ctx.sample_rate),),
        noise=(
            NoiseBin(f0 - 1e12, f0 + 1e12, density if ase_x else 0.0, 0.0 if ase_x else density),
        ),
    )
    out = receiver.run(ctx, {"in": signal, "lo": lo})
    return float(np.std(np.asarray(out["i"].samples)))


def test_lo_ase_beat_is_polarization_selective() -> None:
    """Only ASE co-polarized with the local oscillator reaches baseband.

    The amplified-chain tests above cannot see this: an EDFA emits ASE equally
    into both polarizations, so swapping the two terms in the beat expression
    changes nothing there. That sabotage survived until this test existed — the
    same blind spot, in the same shape, as the one the direct-detection
    polarization test had to be rewritten to close.

    Feeding the receiver ASE on one axis at a time is what makes the distinction
    observable, and the two co-polarized cases must agree because no axis is
    special.
    """
    ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=8, sequence_length=512, seed=7)

    xx = _coherent_beat_noise(ctx, lo_x=True, ase_x=True)
    xy = _coherent_beat_noise(ctx, lo_x=True, ase_x=False)
    yy = _coherent_beat_noise(ctx, lo_x=False, ase_x=False)
    yx = _coherent_beat_noise(ctx, lo_x=False, ase_x=True)

    assert xx > 20.0 * xy
    assert yy > 20.0 * yx
    assert xx == pytest.approx(yy, rel=0.2)
