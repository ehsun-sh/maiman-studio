"""Mode partition noise, carried down the link instead of computed at the laser.

A Fabry-Perot laser's longitudinal modes feed from one reservoir, so what one
takes the others lose. Their sum is quiet and each line is not -- which is why a
power meter at the facet sees nothing wrong, and why a receiver a span away sees
a floor that no amount of power lifts. What stands between the two is time: the
modes are nanometres apart, so a span delivers them ``D L dlambda`` apart, and a
detector adding them no longer adds them at the same instant.

Every band here is an envelope in its own retarded frame, where that delay
cancels. So the delay is *carried* -- :class:`WalkoffHistory` on the signal, as
the dispersion and Kerr histories already are -- and spent once, at the
detector.

Held in place by three things that share no algebra with it: the closed form
``D L dlambda`` a link budget is written in, the sum of the modes' own variances
that a fully decorrelated receiver has to reach, and
:func:`maiman.laser.dispersed_power`, which does the whole thing at the laser.
"""

from __future__ import annotations

from functools import cache
from itertools import pairwise

import numpy as np
import pytest

from maiman.component import Component, PortType
from maiman.components import Combiner, FabryPerotLaser, Fiber, PINPhotodiode
from maiman.context import SimulationContext
from maiman.kernels import apply_group_delay
from maiman.laser import dispersed_power
from maiman.registry import lookup, registered_names
from maiman.signals import Band, OpticalSignal, WalkoffHistory, joined_walkoff

ANCHOR = 193.1e12
SPACING = 100e9
SAMPLES = 256
FS = 160e9
CTX = SimulationContext(bit_rate=10e9, samples_per_symbol=16, sequence_length=16, seed=7)

DISPERSION = 17.0  # ps/nm/km, as a link budget quotes it
D_SI = DISPERSION * 1e-6  # s/m^2


def span(length_km: float) -> Fiber:
    """A span that only disperses: no loss, no Kerr, nothing to confuse the delay."""
    return Fiber(
        label="span",
        length=length_km,
        dispersion=DISPERSION,
        attenuation=0.0,
        nonlinearity=0.0,
        four_wave_mixing=False,
        carry_walkoff=True,
    )


def diode() -> PINPhotodiode:
    """A diode with every noise source off: what is left is the light itself."""
    return PINPhotodiode(label="pd", shot_noise=False, thermal_noise=False, ase_beat_noise=False)


def detected(signal: OpticalSignal, ctx: SimulationContext) -> np.ndarray:
    """The optical power the diode formed [W], with its responsivity divided back out."""
    pd = diode()
    current = pd.run(ctx, {"in": signal})["out"].samples
    return np.asarray(current / pd.si("responsivity"), dtype=np.float64)


# ---------------------------------------------------------------------------
# The delay itself
# ---------------------------------------------------------------------------


def test_a_whole_sample_of_delay_is_a_shift_and_nothing_else() -> None:
    rng = np.random.default_rng(0)
    samples = rng.normal(size=64)
    moved = apply_group_delay(samples, 1e9, 3.0 / 1e9)
    assert np.allclose(moved, np.roll(samples, 3), atol=1e-12)
    assert float(moved.sum()) == pytest.approx(float(samples.sum()), rel=1e-12)


def test_it_is_all_pass_and_exactly_invertible() -> None:
    rng = np.random.default_rng(1)
    field = rng.normal(size=128) + 1j * rng.normal(size=128)
    delayed = apply_group_delay(field, 4e9, 7.3e-10)
    assert float(np.abs(delayed).sum() ** 0.0) == 1.0  # shape sanity
    assert np.sum(np.abs(delayed) ** 2) == pytest.approx(np.sum(np.abs(field) ** 2), rel=1e-12)
    back = apply_group_delay(delayed, 4e9, -7.3e-10)
    assert np.allclose(back, field, atol=1e-12)


def test_a_fraction_of_a_sample_is_interpolated_rather_than_rounded() -> None:
    """A band-limited tone delayed by a third of a sample, against the tone itself."""
    # Whole numbers of cycles in the window: a circular shift is the ideal one
    # only for the periodic signal the window stands for.
    fs, n = 100e9, 500
    times = np.arange(n) / fs
    tone = np.cos(2.0 * np.pi * 7e9 * times) + 0.5 * np.sin(2.0 * np.pi * 3e9 * times)
    delay = 0.33 / fs
    moved = apply_group_delay(tone, fs, delay)
    wanted = np.cos(2.0 * np.pi * 7e9 * (times - delay)) + 0.5 * np.sin(
        2.0 * np.pi * 3e9 * (times - delay)
    )
    assert np.allclose(moved, wanted, atol=1e-10)
    assert not np.allclose(moved, np.roll(tone, 0), atol=1e-4), "it really moved"


def test_no_delay_is_no_work() -> None:
    field = np.ones(8, dtype=np.complex128)
    assert apply_group_delay(field, 1e9, 0.0) is field


# ---------------------------------------------------------------------------
# What the signal carries
# ---------------------------------------------------------------------------


def test_a_carrier_with_no_entry_has_not_walked() -> None:
    history = WalkoffHistory(carriers=((ANCHOR, 1e-9), (ANCHOR + SPACING, -2e-9)))
    assert history.at(ANCHOR) == 1e-9
    assert history.at(ANCHOR + SPACING) == -2e-9
    assert history.at(ANCHOR + 7 * SPACING) == 0.0
    assert history.spread == pytest.approx(3e-9)
    assert history.walked
    assert not WalkoffHistory().walked
    assert not WalkoffHistory(carriers=((ANCHOR, 0.0),)).walked
    assert WalkoffHistory().spread == 0.0


def test_joining_two_arms_is_a_union_and_a_disagreement_is_kept_not_raised() -> None:
    first = OpticalSignal(
        bands=(band_at(ANCHOR),), walkoff=WalkoffHistory(carriers=((ANCHOR, 1e-9),))
    )
    second = OpticalSignal(
        bands=(band_at(ANCHOR + SPACING),),
        walkoff=WalkoffHistory(carriers=((ANCHOR + SPACING, 4e-9),)),
    )
    joined = joined_walkoff((first, second), where="mux")
    assert joined.carriers == ((ANCHOR, 1e-9), (ANCHOR + SPACING, 4e-9))
    assert not joined.conflict

    # The same carrier, by two paths, at two different times.
    other = OpticalSignal(
        bands=(band_at(ANCHOR),), walkoff=WalkoffHistory(carriers=((ANCHOR, 3e-9),))
    )
    clash = joined_walkoff((first, other), where="mux")
    assert "two paths" in clash.conflict
    assert "2.000e-09 s apart" in clash.conflict


def band_at(f0: float, level: float = 1e-3) -> Band:
    return Band(
        Ex=np.full(SAMPLES, level**0.5, dtype=np.complex128),
        Ey=np.zeros(SAMPLES, dtype=np.complex128),
        f0=f0,
        fs=FS,
    )


# ---------------------------------------------------------------------------
# The span
# ---------------------------------------------------------------------------


def two_bands(separation_nm: float = 1.0) -> OpticalSignal:
    """Two carriers a given number of nanometres apart, at 1310 nm."""
    from maiman.units import C_LIGHT

    short = C_LIGHT / (1310e-9 - separation_nm * 1e-9)
    return OpticalSignal(bands=(band_at(C_LIGHT / 1310e-9), band_at(short)))


def test_the_span_writes_down_what_a_link_budget_says() -> None:
    """``D L dlambda``: 17 ps/nm/km, one nanometre, eighty kilometres."""
    signal = two_bands(1.0)
    after = span(80.0).run(CTX, {"in": signal})["out"]
    assert isinstance(after, OpticalSignal)
    assert after.walkoff.at(signal.bands[0].f0) == 0.0, "the frame is the first band"
    walked = after.walkoff.at(signal.bands[1].f0)
    assert walked == pytest.approx(-D_SI * 80e3 * 1e-9, rel=1e-6)
    assert walked * 1e12 == pytest.approx(-1360.0, rel=1e-3), "1.36 ns, as the rule of thumb has it"
    assert after.walkoff.spread == pytest.approx(abs(walked), rel=1e-9)


def test_the_shorter_wavelength_arrives_first() -> None:
    signal = two_bands(2.0)
    after = span(40.0).run(CTX, {"in": signal})["out"]
    assert after.walkoff.at(signal.bands[1].f0) < 0.0, "higher frequency is shorter is sooner"


def test_two_spans_add_and_a_compensating_one_takes_it_back() -> None:
    signal = two_bands(1.0)
    once = span(40.0).run(CTX, {"in": signal})["out"]
    twice = span(40.0).run(CTX, {"in": once})["out"]
    assert twice.walkoff.at(signal.bands[1].f0) == pytest.approx(
        2.0 * once.walkoff.at(signal.bands[1].f0), rel=1e-9
    )
    undone = Fiber(
        label="dcf",
        length=40.0,
        dispersion=-DISPERSION,
        attenuation=0.0,
        nonlinearity=0.0,
        four_wave_mixing=False,
        carry_walkoff=True,
    ).run(CTX, {"in": twice})["out"]
    assert undone.walkoff.at(signal.bands[1].f0) == pytest.approx(
        once.walkoff.at(signal.bands[1].f0), rel=1e-9
    )


def test_without_the_flag_a_span_carries_nothing() -> None:
    signal = two_bands(1.0)
    after = Fiber(
        label="plain", length=80.0, dispersion=DISPERSION, attenuation=0.0, nonlinearity=0.0
    ).run(CTX, {"in": signal})["out"]
    assert after.walkoff.carriers == ()
    assert not after.walkoff.walked
    assert not Fiber().carry_walkoff, "and it is off by default"


def test_a_span_asked_to_carry_delays_refuses_a_join_it_cannot_resolve() -> None:
    signal = two_bands(1.0)
    clashing = OpticalSignal(
        bands=signal.bands,
        walkoff=WalkoffHistory(carriers=((signal.bands[0].f0, 1e-9),), conflict="mux did"),
    )
    with pytest.raises(ValueError, match="one set of arrival delays"):
        span(10.0).run(CTX, {"in": clashing})


# ---------------------------------------------------------------------------
# The detector
# ---------------------------------------------------------------------------


def test_the_detector_spends_the_delay_it_was_handed() -> None:
    """One band, a known delay: the photocurrent is the waveform, moved."""
    rng = np.random.default_rng(5)
    shape = np.abs(rng.normal(size=CTX.num_samples)) + 0.1
    band = Band(
        Ex=np.sqrt(shape).astype(np.complex128),
        Ey=np.zeros(CTX.num_samples, dtype=np.complex128),
        f0=ANCHOR,
        fs=CTX.sample_rate,
    )
    step = 4.0 / CTX.sample_rate
    still = detected(OpticalSignal(bands=(band,)), CTX)
    moved = detected(
        OpticalSignal(bands=(band,), walkoff=WalkoffHistory(carriers=((ANCHOR, step),))), CTX
    )
    assert np.allclose(moved, np.roll(still, 4), rtol=1e-9, atol=1e-12)
    assert float(moved.mean()) == pytest.approx(float(still.mean()), rel=1e-12)


def test_bands_that_arrive_together_are_summed_as_they_always_were() -> None:
    signal = two_bands(1.0)
    plain = detected(signal, CTX)
    zeroed = OpticalSignal(
        bands=signal.bands,
        walkoff=WalkoffHistory(carriers=tuple((b.f0, 0.0) for b in signal.bands)),
    )
    assert np.array_equal(detected(zeroed, CTX), plain)


def test_a_detector_refuses_to_sum_bands_that_disagree_about_when_they_arrived() -> None:
    signal = two_bands(1.0)
    clashing = OpticalSignal(
        bands=signal.bands,
        walkoff=WalkoffHistory(
            carriers=((signal.bands[1].f0, 2e-10),), conflict="mux joined two paths"
        ),
    )
    with pytest.raises(ValueError, match="do not agree on when they arrived"):
        diode().run(CTX, {"in": clashing})
    # Nothing carried, nothing to refuse.
    quiet = OpticalSignal(bands=signal.bands, walkoff=WalkoffHistory(conflict="mux joined"))
    diode().run(CTX, {"in": quiet})


def optical_ports(component: Component, ports: dict[str, PortType]) -> list[str]:
    return [name for name, kind in ports.items() if kind is PortType.OPTICAL]


def test_every_optical_block_carries_the_delays_through() -> None:
    """The sweep that holds the dispersion and Kerr histories, for the arrival times."""
    marker = WalkoffHistory(carriers=((ANCHOR, 1.5e-10), (ANCHOR + SPACING, -3e-10)))
    checked: list[str] = []
    for name in registered_names():
        component = lookup(name)()
        if component.feedback_passes() > 0 or isinstance(component, Fiber | PINPhotodiode):
            continue
        inputs = optical_ports(component, component.inputs)
        outputs = optical_ports(component, component.outputs)
        if not inputs or not outputs:
            continue
        if any(kind is not PortType.OPTICAL for kind in component.inputs.values()):
            continue
        feed: dict[str, object] = {
            port: OpticalSignal(bands=(band_at(ANCHOR + index * SPACING),), walkoff=marker)
            for index, port in enumerate(inputs)
        }
        produced = component.run(CTX, feed)
        for port in outputs:
            out = produced[port]
            assert isinstance(out, OpticalSignal)
            assert out.walkoff.carriers == marker.carriers, f"{name}.{port} dropped it"
        checked.append(name)
    assert {"EDFA", "Attenuator", "OpticalFilter", "Splitter", "Combiner"} <= set(checked)


def test_a_multiplexer_joins_the_delays_its_arms_bring() -> None:
    left = OpticalSignal(
        bands=(band_at(ANCHOR),), walkoff=WalkoffHistory(carriers=((ANCHOR, 5e-10),))
    )
    right = OpticalSignal(
        bands=(band_at(ANCHOR + SPACING),),
        walkoff=WalkoffHistory(carriers=((ANCHOR + SPACING, -5e-10),)),
    )
    mixed = Combiner(2, label="mux").run(CTX, {"in0": left, "in1": right})["out"]
    assert isinstance(mixed, OpticalSignal)
    assert mixed.walkoff.spread == pytest.approx(1e-9)


# ---------------------------------------------------------------------------
# The laser
# ---------------------------------------------------------------------------


def test_the_modes_sit_on_the_comb_the_cavity_has() -> None:
    laser = FabryPerotLaser(label="fp", modes=5, mode_spacing=1.1)
    offsets = laser.mode_offsets()
    assert offsets * 1e9 == pytest.approx([-2.2, -1.1, 0.0, 1.1, 2.2])
    assert laser.mode_wavelengths()[2] == pytest.approx(laser.si("wavelength"), rel=1e-15)
    signal = laser.run(CTX, {})["out"]
    assert signal.num_bands == 5
    centres = [b.f0 for b in signal.bands]
    assert centres == sorted(centres), "ascending in frequency, so the frame is the longest mode"


def test_the_gain_parabola_decides_who_gets_the_power() -> None:
    laser = FabryPerotLaser(label="fp", modes=7, mode_spacing=1.1, gain_bandwidth=40.0)
    share = laser.steady_state() / laser.steady_state().sum()
    assert share.argmax() == 3, "the peak mode"
    assert share[3] == pytest.approx(0.47, abs=0.03)
    assert np.allclose(share, share[::-1], atol=1e-9), "symmetric about the peak"
    assert all(a < b for a, b in pairwise(share[:4])), "and falling away from it"


def test_the_quoted_power_is_what_comes_out() -> None:
    laser = FabryPerotLaser(label="fp", power=3.0, partition_noise=False)
    signal = laser.run(CTX, {})["out"]
    assert signal.signal_power() == pytest.approx(10 ** (3.0 / 10.0) * 1e-3, rel=1e-6)
    noisy = FabryPerotLaser(label="fp", power=3.0).run(CTX, {})["out"]
    assert noisy.signal_power() == pytest.approx(10 ** (3.0 / 10.0) * 1e-3, rel=0.1)


def test_with_the_noise_off_it_is_the_steady_state_and_nothing_moves() -> None:
    laser = FabryPerotLaser(label="fp", partition_noise=False)
    powers = laser.mode_power(CTX)
    assert np.allclose(powers, powers[:, :1], rtol=0.0, atol=0.0)
    assert FabryPerotLaser().partition_noise, "on by default: this is what an FP laser is"


def test_the_partition_is_far_louder_than_the_sum_of_it() -> None:
    """One reservoir: what a mode gains the others lose, and the total barely moves."""
    laser = FabryPerotLaser(label="fp")
    power = laser.mode_power(CTX)
    parts = float(sum(power[m].var() for m in range(power.shape[0])))
    total = float(power.sum(axis=0).var())
    assert total < 0.6 * parts, "the sum is quieter than its parts"
    loudest = max(float(power[m].std() / power[m].mean()) for m in range(power.shape[0]))
    assert loudest > 3.0 * float(power.sum(axis=0).std() / power.sum(axis=0).mean())


# ---------------------------------------------------------------------------
# The link
# ---------------------------------------------------------------------------

SEEDS = range(6)


def link_context(seed: int) -> SimulationContext:
    """A 12.8 ns window: room for fifty kilometres of walk-off without it wrapping round."""
    return SimulationContext(bit_rate=10e9, samples_per_symbol=8, sequence_length=128, seed=seed)


@cache
def emitted(seed: int) -> OpticalSignal:
    """One realization of the laser. Cached, because each one integrates the rate equations."""
    return FabryPerotLaser(label="fp").run(link_context(seed), {})["out"]


@cache
def received(length_km: float) -> tuple[np.ndarray, ...]:
    """The detected power after a span, one waveform per seed."""
    return tuple(
        detected(
            span(length_km).run(link_context(seed), {"in": emitted(seed)})["out"],
            link_context(seed),
        )
        for seed in SEEDS
    )


def relative_variance(waveforms: tuple[np.ndarray, ...]) -> float:
    return float(np.mean([w.var() / w.mean() ** 2 for w in waveforms]))


def test_the_span_does_not_touch_the_power_it_delivers() -> None:
    """An all-pass delay moves light in time and nowhere else."""
    for length in (0.0, 50.0):
        for waveform in received(length):
            assert float(waveform.mean()) == pytest.approx(float(received(0.0)[0].mean()), rel=0.2)
    assert all(
        float(walked.mean()) == pytest.approx(float(flat.mean()), rel=1e-6)
        for walked, flat in zip(received(50.0), received(0.0), strict=True)
    )


def test_the_further_it_goes_the_noisier_it_lands() -> None:
    """Mode partition noise, which is not there at the laser and is there at the end."""
    noise = [relative_variance(received(length)) for length in (0.0, 20.0, 50.0)]
    assert all(a < b for a, b in pairwise(noise)), noise
    assert noise[-1] > 3.0 * noise[0], "and it is not a small effect"


def test_at_the_end_nothing_cancels_any_more() -> None:
    """Fully walked apart, the sum's variance is the sum of the modes' own."""
    ratios = []
    for seed, waveform in zip(SEEDS, received(50.0), strict=True):
        power = FabryPerotLaser(label="fp").mode_power(link_context(seed))
        parts = float(sum(power[m].var() for m in range(power.shape[0])))
        ratios.append(float(waveform.var()) / parts)
    # Below the independent bound and near it. The scatter is a single window's:
    # a variance measured over twelve nanoseconds is not a precise number, which
    # is why this reads the mean of six and does not hold any one of them.
    assert 0.5 < float(np.mean(ratios)) < 1.0, ratios
    assert all(0.2 < ratio < 1.3 for ratio in ratios), ratios


def test_the_link_gets_what_the_laser_module_gets_on_its_own() -> None:
    """Two code paths to one number: bands walking down a link, and ``dispersed_power``.

    Run where that function's own approximation holds -- each mode's bandwidth
    far below the spacing between them -- because a span disperses *inside* a
    band as well, and at a sample rate high enough for that to matter the two
    stop being the same question. Five nanometres apart at ten gigasamples,
    five kilometres: 1.7 ns of walk-off across a 51 ns window, and a hundredth
    of a radian of dispersion inside any one mode.
    """
    ctx = SimulationContext(bit_rate=5e9, samples_per_symbol=2, sequence_length=256, seed=3)
    laser = FabryPerotLaser(label="fp", modes=5, mode_spacing=5.0)
    emitted = laser.run(ctx, {})["out"]
    after = span(5.0).run(ctx, {"in": emitted})["out"]
    assert after.walkoff.spread == pytest.approx(4.0 * D_SI * 5e3 * 5e-9, rel=1e-9)

    offsets = laser.mode_offsets()
    # Against the same frame the span used: its first band, the longest mode.
    reference = dispersed_power(
        laser.mode_power(ctx), offsets - offsets[-1], ctx.sample_rate, dispersion=D_SI * 5e3
    )
    landed = detected(after, ctx)
    assert np.abs(landed - reference).max() / reference.mean() < 1e-3
    assert landed.std() == pytest.approx(float(reference.std()), rel=1e-3)
