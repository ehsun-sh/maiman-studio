"""Validation of the EDFA and the OSNR that comes out of it.

The headline check is the OSNR formula every link engineer carries around:

    OSNR [dB] = 58 + P_launch [dBm] - NF [dB] - 10*log10(spans)

Nothing in the implementation is written in those terms — it computes an
inversion factor, an ASE spectral density, and a power ratio in a reference
bandwidth. That the arithmetic lands on the engineering rule of thumb is the
result.

Reference: G. P. Agrawal, *Fiber-Optic Communication Systems*, ch. 6.
"""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np
import pytest

from maiman import Component, Graph, OpticalSignal, SimulationContext
from maiman.analysis import OSNR_REFERENCE_BANDWIDTH, noise_psd_at, osnr
from maiman.components import EDFA, Combiner, CWLaser, Fiber, OSNRMeter, PowerMeter
from maiman.units import C_LIGHT, H_PLANCK, db_to_linear, dbm_to_w, w_to_dbm

CTX = SimulationContext(bit_rate=10e9, samples_per_symbol=8, sequence_length=64)
NU_1550 = C_LIGHT / 1550e-9

#: 10*log10(h*nu*B_ref / 1 mW) at 1550 nm in 12.5 GHz — the "58" in the rule of
#: thumb, spelled out so the test does not smuggle in a magic number.
QUANTUM_FLOOR_DBM = w_to_dbm(H_PLANCK * NU_1550 * OSNR_REFERENCE_BANDWIDTH)


def _amplified(
    launch_dbm: float,
    *,
    gain_db: float = 20.0,
    noise_figure_db: float = 5.0,
    span_km: float = 0.0,
    stages: int = 1,
) -> tuple[Graph, PowerMeter, OSNRMeter]:
    g = Graph(CTX)
    laser = g.add(CWLaser(power=launch_dbm, wavelength=1550.0, label="laser"))
    previous: Component = laser
    for stage in range(stages):
        fiber = g.add(Fiber(length=span_km, attenuation=0.2, label=f"span{stage}"))
        amp = g.add(EDFA(gain=gain_db, noise_figure=noise_figure_db, label=f"edfa{stage}"))
        g.connect(previous, fiber)
        g.connect(fiber, amp)
        previous = amp

    power = g.add(PowerMeter(label="power"))
    osnr_meter = g.add(OSNRMeter(label="osnr"))
    g.connect(previous, power)
    g.connect(previous, osnr_meter)
    return g, power, osnr_meter


def _amplifier_output(graph: Graph, label: str = "edfa0") -> OpticalSignal:
    """The optical signal leaving a named amplifier."""
    amp = next(c for c in graph.components if c.label == label)
    return graph.run(keep=[amp]).port(amp, "out")


# --------------------------------------------------------------------------
# Gain
# --------------------------------------------------------------------------


@pytest.mark.parametrize("gain_db", [0.0, 10.0, 20.0, 30.0])
def test_signal_power_is_multiplied_by_the_gain(gain_db: float) -> None:
    g, power, _ = _amplified(-20.0, gain_db=gain_db)
    reading = g.run()[power]
    assert w_to_dbm(reading.signal_power_w) == pytest.approx(-20.0 + gain_db, abs=1e-4)


def test_an_amplifier_that_is_not_asked_to_saturate_does_not() -> None:
    """The default is an ideal amplifier, and it has to be exactly ideal.

    Declared as an idealisation rather than delivered as one by accident: every
    result in this repository was measured with ``saturate`` false, and a model
    that compressed a little anyway would move all of them by amounts too small
    to notice and too large to be nothing.
    """
    # Asserted on the gain itself as well as through the link, because the
    # round trip through dBm costs a few parts in 1e7 on its own and would hide
    # a compression a hundred times larger than that.
    assert EDFA(gain=20.0).effective_gain(dbm_to_w(0.0)) == db_to_linear(20.0)

    g, power, _ = _amplified(-20.0, gain_db=20.0)
    assert w_to_dbm(g.run()[power].signal_power_w) == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------
# Saturation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("gain_db", "saturation_dbm"), [(20.0, 17.0), (30.0, 20.0), (10.0, 13.0)])
def test_the_gain_compresses_by_three_db_at_the_declared_saturation_power(
    gain_db: float, saturation_dbm: float
) -> None:
    """The definition of the parameter, checked as a definition.

    ``saturation_power`` is the *output* at 3 dB of compression, which is how a
    datasheet quotes it. Drive the amplifier so that its output lands there and
    the gain must be exactly 3 dB down — and the output must be exactly the
    declared number, which is the half of it that catches a conversion applied
    in the wrong direction.

    The 10 dB row is not padding: the conversion from the datasheet number to
    the model's own ``P_sat`` carries a factor ``(G_0 - 2) / G_0``, which is 0.98
    at 20 dB and 0.8 at 10. A version that dropped it as "large gain" passes the
    first two rows.
    """
    amplifier = EDFA(gain=gain_db, saturate=True, saturation_power=saturation_dbm)
    small_signal = db_to_linear(gain_db)
    # The input whose output would be the 3 dB point, if the gain there is G_0/2.
    input_power = 2.0 * dbm_to_w(saturation_dbm) / small_signal

    gain = amplifier.effective_gain(input_power)
    assert 10.0 * math.log10(gain) == pytest.approx(gain_db - 10.0 * math.log10(2.0), abs=1e-6)
    assert w_to_dbm(gain * input_power) == pytest.approx(saturation_dbm, abs=1e-6)


def test_the_solved_gain_satisfies_the_equation_it_came_from() -> None:
    """Newton either converged or it did not, and this is what says which.

    ``G = G_0 exp(-(G-1) P_in / P_sat)`` is implicit, so the only honest check on
    the solver is to put its answer back in. An iteration cap that quietly
    returned a half-converged gain would pass every other test in this file by a
    hair and fail here by orders of magnitude.
    """
    amplifier = EDFA(gain=25.0, saturate=True, saturation_power=15.0)
    small_signal = db_to_linear(25.0)
    p_sat = amplifier.intrinsic_saturation_power(small_signal)

    for input_dbm in (-40.0, -20.0, -10.0, 0.0, 10.0):
        input_power = dbm_to_w(input_dbm)
        gain = amplifier.effective_gain(input_power)
        residual = gain - small_signal * math.exp(-(gain - 1.0) * input_power / p_sat)
        assert abs(residual) < 1e-12 * small_signal


def test_compression_is_smooth_and_has_no_ceiling_to_hit() -> None:
    """What the clamp got qualitatively wrong, and the reason for replacing it.

    A clamp has a kink: below it the amplifier is perfectly ideal and above it
    the gain falls as ``1/P_in``, which means an amplifier that is exactly right
    until it is suddenly wrong. Real compression starts immediately and never
    stops. So: the gain must fall at *every* step, including ones far below the
    saturation power, and the output must keep rising — a saturated amplifier
    delivers less gain, not less power.
    """
    amplifier = EDFA(gain=20.0, saturate=True, saturation_power=17.0)
    inputs = [dbm_to_w(p) for p in (-40.0, -30.0, -20.0, -10.0, -5.0, 0.0, 5.0, 10.0)]
    gains = [amplifier.effective_gain(p) for p in inputs]
    outputs = [g * p for g, p in zip(gains, inputs, strict=True)]

    assert all(later < earlier for earlier, later in pairwise(gains)), (
        "compression must begin immediately, not at a threshold"
    )
    assert all(later > earlier for earlier, later in pairwise(outputs)), (
        "more in must still mean more out; a clamp is what stops being true"
    )
    # Ten dB more input, well past the knee, must buy substantially less than
    # ten dB more output — which is the whole content of the word saturation.
    assert w_to_dbm(outputs[-1]) - w_to_dbm(outputs[-3]) < 5.0


def test_saturation_is_driven_by_total_power_so_channels_share_the_gain() -> None:
    """Two channels compress an amplifier that neither would compress alone.

    The physical prediction that makes this a model of an amplifier rather than
    of a channel: erbium has one inversion and everything in the band draws on
    it. It is also the mechanism behind every gain-flattening argument in a WDM
    system, so a version that saturated per band would be wrong in a way that
    only shows up once someone adds channels.
    """
    amplifier = EDFA(gain=20.0, saturate=True, saturation_power=17.0)
    one = dbm_to_w(-5.0)

    alone = amplifier.effective_gain(one)
    together = amplifier.effective_gain(2.0 * one)
    assert together < alone, "a second channel must cost the first some gain"

    # And the two must share: doubling the load is exactly the same to the
    # amplifier as one channel of twice the power.
    assert together == pytest.approx(
        amplifier.effective_gain(dbm_to_w(-5.0 + 10.0 * math.log10(2.0)))
    )


def test_adding_channels_takes_gain_from_the_ones_already_there() -> None:
    """The same sharing, through a real graph rather than through one function.

    Worth having separately: ``effective_gain`` is driven by
    :meth:`OpticalSignal.total_power`, and a version that summed one band, or
    that saturated each band against its own copy of the amplifier, would give
    the right answer to the unit test above and the wrong one here.

    The per-channel numbers are the physics a WDM operator actually plans
    around: every doubling of the channel count costs the survivors about a
    decibel, and the amplifier's *total* output barely moves — an EDFA near
    saturation is closer to a fixed power source than to a fixed gain.
    """

    def per_channel_dbm(channels: int) -> tuple[float, float]:
        g = Graph(CTX)
        amplifier = g.add(
            EDFA(gain=20.0, saturate=True, saturation_power=17.0, noise_figure=5.0, label="edfa")
        )
        meter = g.add(PowerMeter(label="pm"))
        if channels == 1:
            laser = g.add(CWLaser(power=-5.0, wavelength=1550.0, label="ch0"))
            g.connect(laser, amplifier["in"])
        else:
            combiner = g.add(Combiner(num_inputs=channels, insertion_loss=0.0, label="mux"))
            for index in range(channels):
                # Spaced on a real grid, so these are separate bands rather than
                # one band with more power in it.
                laser = g.add(
                    CWLaser(power=-5.0, wavelength=1550.0 + index * 0.8, label=f"ch{index}")
                )
                g.connect(laser, combiner[f"in{index}"])
            g.connect(combiner, amplifier["in"])
        g.connect(amplifier, meter["in"])
        total = g.run()[meter].signal_power_w
        return w_to_dbm(total), w_to_dbm(total / channels)

    readings = {n: per_channel_dbm(n) for n in (1, 2, 4, 8)}

    each = [readings[n][1] for n in (1, 2, 4, 8)]
    assert each == pytest.approx([13.61, 12.74, 11.56, 10.12], abs=0.02)
    assert all(later < earlier for earlier, later in pairwise(each)), (
        "every added channel must cost the ones already there"
    )

    # Eight times the input, and under 6 dB more out of the amplifier.
    totals = [readings[n][0] for n in (1, 8)]
    assert totals[1] - totals[0] < 6.0


def test_a_project_carrying_the_retired_clamp_still_loads() -> None:
    """``max_output_power`` is gone, and an old document must not fail to open.

    It loads with the parameter dropped, which — unlike the other retirements in
    this library — genuinely changes the link it describes. That is the trade
    taken deliberately: the value described a clamp this class's own docstring
    called a fiction, and refusing to open the file would be worse than opening
    it with the physics in place.
    """
    from maiman.project import graph_from_dict

    document = {
        "schema_version": 1,
        "context": {
            "bit_rate": 10e9,
            "samples_per_symbol": 8,
            "sequence_length": 64,
            "seed": 1,
        },
        "nodes": [
            {"id": "laser", "type": "CWLaser", "params": {"power": 0.0}},
            {
                "id": "edfa",
                "type": "EDFA",
                "params": {"gain": 20.0, "max_output_power": 10.0},
            },
        ],
        "edges": [{"from": ["laser", "out"], "to": ["edfa", "in"]}],
    }
    graph = graph_from_dict(document)
    amplifier = next(c for c in graph.components if isinstance(c, EDFA))
    assert not hasattr(amplifier, "max_output_power")
    assert amplifier.gain == 20.0
    assert not amplifier.saturate, "and it loads as the ideal amplifier, not a compressed one"


# --------------------------------------------------------------------------
# ASE
# --------------------------------------------------------------------------


def test_ase_psd_matches_the_spontaneous_emission_relation() -> None:
    """``S_ASE = n_sp * h * nu * (G - 1)`` per polarization."""
    gain_db, noise_figure_db = 20.0, 5.0
    gain = db_to_linear(gain_db)
    n_sp = db_to_linear(noise_figure_db) * gain / (2.0 * (gain - 1.0))
    expected = n_sp * H_PLANCK * NU_1550 * (gain - 1.0)

    amp = EDFA(gain=gain_db, noise_figure=noise_figure_db)
    # abs=0.0, because an ASE density is ~1e-17 W/Hz and pytest.approx applies
    # a default absolute tolerance of 1e-12 alongside rel. Without this the
    # assertion accepted every value within 1e-12 of the answer — five orders of
    # magnitude either side, zero included — and tested nothing at all.
    assert amp.ase_psd(gain) == pytest.approx(expected, rel=1e-9, abs=0.0)
    assert amp.spontaneous_emission_factor(gain) == pytest.approx(n_sp, rel=1e-9)


def test_total_ase_power_matches_the_textbook_expression() -> None:
    """``P_ASE = 2 * n_sp * h * nu * (G - 1) * B_o``, the factor of two being the
    two polarizations."""
    gain_db, noise_figure_db, bandwidth = 20.0, 5.0, 4e12
    gain = db_to_linear(gain_db)
    n_sp = db_to_linear(noise_figure_db) * gain / (2.0 * (gain - 1.0))
    expected = 2.0 * n_sp * H_PLANCK * NU_1550 * (gain - 1.0) * bandwidth

    g = Graph(CTX)
    laser = g.add(CWLaser(power=-20.0, label="laser"))
    amp = g.add(EDFA(gain=gain_db, noise_figure=noise_figure_db, bandwidth=4.0, label="edfa"))
    meter = g.add(PowerMeter(label="meter"))
    g.chain(laser, amp, meter)

    assert g.run()[meter].noise_power_w == pytest.approx(expected, rel=1e-6)


def test_ase_lives_in_a_noise_bin_not_in_the_sampled_band() -> None:
    """The reason the noise-bin representation exists.

    ASE spans terahertz while the signal occupies a few tens of gigahertz.
    Sampling both together would need a sample rate no machine can afford, so
    the noise is carried as a spectral density — and the sampled band must come
    out of the amplifier carrying signal only.
    """
    g = Graph(CTX)
    laser = g.add(CWLaser(power=-20.0, label="laser"))
    amp = g.add(EDFA(gain=20.0, bandwidth=4.0, label="edfa"))
    meter = g.add(PowerMeter(label="meter"))
    g.chain(laser, amp, meter)

    signal = g.run(keep=[amp]).port(amp, "out")

    assert len(signal.noise) == 1
    assert signal.noise[0].bandwidth == pytest.approx(4e12)
    # The ASE band is tens of times wider than the whole simulated bandwidth.
    assert signal.noise[0].bandwidth > 40 * CTX.sample_rate
    # And the sampled band is exactly the amplified carrier, with nothing added.
    assert signal.signal_power() == pytest.approx(dbm_to_w(0.0), rel=1e-4)


def test_a_unity_gain_amplifier_emits_no_ase() -> None:
    """Spontaneous emission is proportional to (G - 1): an amplifier that does
    not amplify cannot spontaneously emit either."""
    amp = EDFA(gain=0.0, noise_figure=5.0)
    assert amp.ase_psd(1.0) == 0.0


def test_the_noise_figure_cannot_go_below_the_quantum_limit() -> None:
    """n_sp has a floor of 1 — full inversion, a 3 dB noise figure. A lower noise
    figure would describe an amplifier quieter than quantum mechanics allows."""
    amp = EDFA(gain=20.0, noise_figure=0.0)
    assert amp.spontaneous_emission_factor(db_to_linear(20.0)) >= 0.5


# --------------------------------------------------------------------------
# OSNR — the engineering rule of thumb
# --------------------------------------------------------------------------


def test_the_quantum_floor_is_the_58_in_the_rule_of_thumb() -> None:
    assert pytest.approx(58.0, abs=0.1) == -QUANTUM_FLOOR_DBM


@pytest.mark.parametrize("launch_dbm", [-6.0, -3.0, 0.0, 3.0])
@pytest.mark.parametrize("noise_figure_db", [4.0, 5.0, 6.0])
def test_single_stage_osnr_matches_the_engineering_formula(
    launch_dbm: float, noise_figure_db: float
) -> None:
    """``OSNR = 58 + P_launch - NF`` for one amplifier.

    Nothing in the model is expressed this way; it computes an inversion factor,
    a spectral density, and a ratio in a 0.1 nm slice. Landing on the formula is
    the check.
    """
    g, _, osnr_meter = _amplified(launch_dbm, gain_db=20.0, noise_figure_db=noise_figure_db)
    measured = g.run()[osnr_meter]

    expected = launch_dbm - noise_figure_db - QUANTUM_FLOOR_DBM
    assert measured == pytest.approx(expected, abs=0.1)


def test_osnr_does_not_depend_on_the_gain() -> None:
    """Both the signal and its own ASE scale with G, so once past a modest gain
    the ratio settles. An OSNR that improved with gain would be a free lunch."""
    values = [
        _amplified(-3.0, gain_db=gain)[0].run()[_amplified(-3.0, gain_db=gain)[2]]
        for gain in (15.0, 20.0, 25.0, 30.0)
    ]
    assert max(values) - min(values) < 0.15


def test_each_identical_span_costs_three_db_of_osnr() -> None:
    """N spans divide OSNR by N: doubling the count costs 3 dB. This is why long
    haul is hard, and it falls out of the noise bins accumulating."""
    single = _amplified(0.0, span_km=100.0, gain_db=20.0, stages=1)
    quad = _amplified(0.0, span_km=100.0, gain_db=20.0, stages=4)

    osnr_1 = single[0].run()[single[2]]
    osnr_4 = quad[0].run()[quad[2]]

    assert osnr_1 - osnr_4 == pytest.approx(10 * math.log10(4), abs=0.3)


def test_osnr_degrades_one_for_one_with_launch_power() -> None:
    low = _amplified(-6.0)
    high = _amplified(0.0)
    assert high[0].run()[high[2]] - low[0].run()[low[2]] == pytest.approx(6.0, abs=0.05)


def test_a_link_without_amplifiers_has_infinite_osnr() -> None:
    """No amplifier, no ASE. The signal is attenuated but never made noisier —
    which is exactly why a passive link's reach is a loss budget, not an OSNR
    budget."""
    g = Graph(CTX)
    laser = g.add(CWLaser(power=0.0, label="laser"))
    fiber = g.add(Fiber(length=100.0, attenuation=0.2, label="fiber"))
    meter = g.add(OSNRMeter(label="osnr"))
    g.chain(laser, fiber, meter)

    assert g.run()[meter] == math.inf


def test_fiber_attenuates_ase_along_with_the_signal() -> None:
    """Loss after an amplifier cannot improve OSNR: both fall together."""
    g = Graph(CTX)
    laser = g.add(CWLaser(power=0.0, label="laser"))
    amp = g.add(EDFA(gain=20.0, label="edfa"))
    fiber = g.add(Fiber(length=80.0, attenuation=0.2, label="fiber"))
    before = g.add(OSNRMeter(label="before"))
    after = g.add(OSNRMeter(label="after"))
    g.chain(laser, amp)
    g.connect(amp, before)
    g.connect(amp, fiber)
    g.connect(fiber, after)

    results = g.run()
    # Tolerance set by complex64 storage of the band, not by the physics:
    # the ratio is exactly preserved, the stored samples are not.
    assert results[after] == pytest.approx(results[before], abs=1e-4)


def test_osnr_uses_the_noise_density_at_the_carrier() -> None:
    g = Graph(CTX)
    laser = g.add(CWLaser(power=-10.0, wavelength=1550.0, label="laser"))
    amp = g.add(EDFA(gain=20.0, noise_figure=5.0, label="edfa"))
    meter = g.add(PowerMeter(label="meter"))
    g.chain(laser, amp, meter)

    signal = g.run(keep=[amp]).port(amp, "out")
    psd = noise_psd_at(signal, signal.bands[0].f0)
    manual = 10 * math.log10(signal.signal_power() / (psd * OSNR_REFERENCE_BANDWIDTH))
    assert osnr(signal) == pytest.approx(manual, rel=1e-12)


def test_reference_bandwidth_scales_the_result() -> None:
    """Halving the reference bandwidth halves the counted noise: +3 dB."""
    signal = _amplifier_output(_amplified(0.0)[0])

    wide = osnr(signal, reference_bandwidth=12.5e9)
    narrow = osnr(signal, reference_bandwidth=6.25e9)
    assert narrow - wide == pytest.approx(3.0103, abs=1e-6)


def test_a_nonsensical_reference_bandwidth_is_rejected() -> None:
    signal = _amplifier_output(_amplified(0.0)[0])
    with pytest.raises(ValueError, match="reference_bandwidth must be positive"):
        osnr(signal, reference_bandwidth=0.0)


def test_noise_outside_the_carrier_does_not_count_against_it() -> None:
    """A noise bin sitting somewhere else in the spectrum is not this channel's
    problem — the band-resolved lookup is what makes that distinction possible."""
    from maiman.signals import Band, NoiseBin

    band = Band(
        Ex=np.full(16, 0.1, dtype=np.complex64),
        Ey=np.zeros(16, dtype=np.complex64),
        f0=NU_1550,
        fs=80e9,
    )
    elsewhere = NoiseBin(f_start=150e12, f_end=160e12, psd_x=1e-15, psd_y=1e-15)
    assert osnr(OpticalSignal(bands=(band,), noise=(elsewhere,))) == math.inf
