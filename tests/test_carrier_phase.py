"""The carrier phase a span divides out, carried for the beat that needs it (maiman-681).

Every band is an envelope in its own retarded frame: a span removes the group delay
and the carrier phase ``beta(omega) L`` from it. Inside one band both are constants
and cancel; between carriers they do not. The delay rides on the signal as
:class:`~maiman.signals.WalkoffHistory`, and with ``carry_carrier_phase`` the phase
rides beside it, so that two tones from one source sent down fibre beat at the phase
the fibre gave them -- ``phase_b - phase_a`` -- rather than at the frame's.

The closed form written out here is the propagation constant expanded about the
reference wavelength, ``beta(omega) = n_p omega_0 / c + (n_g / c) d + beta2 d^2 / 2``
with ``d = omega - omega_0``, evaluated independently of :class:`Fiber`.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from maiman.components import Combiner, Fiber, PINPhotodiode
from maiman.context import SimulationContext
from maiman.signals import Band, OpticalSignal, WalkoffHistory, joined_walkoff
from maiman.units import C_LIGHT

ANCHOR = 193.1e12
SPACING = 20e9  # inside half the sample rate, so the diode beats them
SAMPLES = 256
CTX = SimulationContext(bit_rate=10e9, samples_per_symbol=16, sequence_length=16, seed=7)
DISPERSION = 17.0  # ps/nm/km
PHASE_INDEX, GROUP_INDEX = 1.444, 1.468


def tone(f0: float, level: float = 1e-3) -> Band:
    return Band(
        Ex=np.full(SAMPLES, level**0.5, dtype=np.complex128),
        Ey=np.zeros(SAMPLES, dtype=np.complex128),
        f0=f0,
        fs=160e9,
    )


def fibre(length_km: float, dispersion: float = DISPERSION, **settings: object) -> Fiber:
    """A span that only disperses, carrying the phase."""
    return Fiber(
        label=f"span{length_km}",
        length=length_km,
        dispersion=dispersion,
        attenuation=0.0,
        nonlinearity=0.0,
        four_wave_mixing=False,
        carry_carrier_phase=True,
        phase_index=PHASE_INDEX,
        group_index=GROUP_INDEX,
        **settings,  # type: ignore[arg-type]
    )


def beta(frequency: float, dispersion: float = DISPERSION) -> float:
    """``beta(omega)`` [rad/m], expanded about the fibre's reference wavelength."""
    reference = Fiber().si("reference_wavelength")
    omega_0 = 2.0 * math.pi * C_LIGHT / reference
    detuning = 2.0 * math.pi * frequency - omega_0
    beta2 = -dispersion * 1e-6 * reference**2 / (2.0 * math.pi * C_LIGHT)
    return (
        PHASE_INDEX * omega_0 / C_LIGHT
        + GROUP_INDEX * detuning / C_LIGHT
        + 0.5 * beta2 * detuning**2
    )


def beat_phase(signal: OpticalSignal, low: float, high: float) -> float:
    """The phase the photocurrent's tone at ``high - low`` is at, ``cos(w t + phase)``."""
    pd = PINPhotodiode(label="pd", shot_noise=False, thermal_noise=False, ase_beat_noise=False)
    current = pd.run(CTX, {"in": signal})["out"].samples / pd.si("responsivity")
    time = CTX.time_axis()
    tone_ = np.exp(-2j * np.pi * (high - low) * time)
    return float(np.angle(np.sum(current * tone_)))


def wrapped(value: float) -> float:
    return math.remainder(value, 2.0 * math.pi)


A, B = ANCHOR + SPACING, ANCHOR


def test_off_by_default_the_beat_sits_at_the_frames_phase() -> None:
    """Without the flag nothing is carried and the tone is a plain cosine, phase zero."""
    signal = OpticalSignal(bands=(tone(A), tone(B)))
    plain = Fiber(
        label="plain",
        length=10.0,
        dispersion=DISPERSION,
        attenuation=0.0,
        nonlinearity=0.0,
        four_wave_mixing=False,
    ).run(CTX, {"in": signal})["out"]
    assert isinstance(plain, OpticalSignal)
    assert plain.walkoff.phases == ()
    assert not Fiber().carry_carrier_phase
    assert abs(wrapped(beat_phase(plain, B, A))) < 1e-9


@pytest.mark.parametrize("length", [1.0, 10.0, 10.3, 80.0])
def test_two_tones_down_one_span_beat_at_the_difference_of_their_phases(length: float) -> None:
    """The fringe is ``cos(w t - (beta_a - beta_b) L)``: the tone's phase is minus that difference.

    Independent of the block's own code: the closed form is written from the
    indices and the dispersion. The lengths include one a third of a fringe past
    another, so it is the slope that is checked and not only the value.
    """
    signal = OpticalSignal(bands=(tone(A), tone(B)))
    after = fibre(length).run(CTX, {"in": signal})["out"]
    assert isinstance(after, OpticalSignal)
    expected = -(beta(A) - beta(B)) * length * 1e3
    assert wrapped(beat_phase(after, B, A) - expected) == pytest.approx(0.0, abs=1e-3)


def test_the_carried_phase_is_the_propagation_constant_times_the_length() -> None:
    signal = OpticalSignal(bands=(tone(A), tone(B)))
    after = fibre(10.0).run(CTX, {"in": signal})["out"]
    assert isinstance(after, OpticalSignal)
    for f0 in (A, B):
        assert wrapped(after.walkoff.phase_at(f0) - beta(f0) * 10e3) == pytest.approx(0.0, abs=2e-3)


def test_spans_add_and_a_reversed_span_undoes_it() -> None:
    signal = OpticalSignal(bands=(tone(A), tone(B)))
    once = fibre(7.0).run(CTX, {"in": signal})["out"]
    twice = fibre(7.0).run(CTX, {"in": once})["out"]
    assert isinstance(twice, OpticalSignal)
    for f0 in (A, B):
        assert wrapped(twice.walkoff.phase_at(f0) - 2.0 * beta(f0) * 7e3) == pytest.approx(
            0.0, abs=4e-3
        )


def test_tones_that_took_different_lengths_beat_at_the_difference_of_their_own() -> None:
    """One source, two arms of fibre of different lengths, then a multiplexer and a diode.

    ``phase_b L_b - phase_a L_a`` -- here the phase index matters, because the
    lengths differ, where down one span it drops out of the difference. The arms
    are dispersion-free: the multiplexer refuses paths that have accumulated
    different dispersion, which is a rule about four-wave mixing and not about this.
    """
    length_a, length_b = 12.0, 12.7
    arm_a = fibre(length_a, 0.0).run(CTX, {"in": OpticalSignal(bands=(tone(A),))})["out"]
    arm_b = fibre(length_b, 0.0).run(CTX, {"in": OpticalSignal(bands=(tone(B),))})["out"]
    mixed = Combiner(2, label="mux").run(CTX, {"in0": arm_a, "in1": arm_b})["out"]
    assert isinstance(mixed, OpticalSignal)
    expected = -(beta(A, 0.0) * length_a * 1e3 - beta(B, 0.0) * length_b * 1e3)
    assert wrapped(beat_phase(mixed, B, A) - expected) == pytest.approx(0.0, abs=2e-3)


def test_a_carrier_that_arrives_by_two_paths_at_two_phases_is_a_conflict() -> None:
    """One carrier down two different lengths is interference a sum of envelopes cannot express."""
    first = OpticalSignal(bands=(tone(A),), walkoff=WalkoffHistory(phases=((A, 0.3),)))
    other = A + 1e9
    second = OpticalSignal(bands=(tone(other),), walkoff=WalkoffHistory(phases=((other, 0.1),)))
    assert joined_walkoff((first, second), where="mux").conflict == ""
    third = OpticalSignal(bands=(tone(A),), walkoff=WalkoffHistory(phases=((A, 0.9),)))
    joined = joined_walkoff((first, third), where="mux")
    assert "carrier phases differ" in joined.conflict
    # and a phase that differs only by whole turns is the same phase
    turned = OpticalSignal(
        bands=(tone(A),), walkoff=WalkoffHistory(phases=((A, 0.3 + 2.0 * math.pi),))
    )
    assert joined_walkoff((first, turned), where="mux").conflict == ""
    with pytest.raises(ValueError, match="one set"):
        fibre(1.0).run(
            CTX,
            {"in": OpticalSignal(bands=(tone(A),), walkoff=WalkoffHistory(conflict="two paths"))},
        )


def test_a_common_length_shifts_both_tones_and_the_beat_moves_by_their_difference() -> None:
    """Doubling the span doubles the beat's phase, ``(beta_a - beta_b) L``, modulo a turn."""
    signal = OpticalSignal(bands=(tone(A), tone(B)))
    one = beat_phase(fibre(20.0).run(CTX, {"in": signal})["out"], B, A)
    two = beat_phase(fibre(40.0).run(CTX, {"in": signal})["out"], B, A)
    assert wrapped(two - 2.0 * one) == pytest.approx(0.0, abs=5e-3)
