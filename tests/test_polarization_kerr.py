"""The Kerr effect across polarizations.

Light does not only modulate light that shares its axis. In an isotropic medium
the chi(3) tensor gives power in the orthogonal component exactly two thirds the
weight of power in its own — which, because co-polarized cross-phase modulation
already carries a factor of two, makes cross-phase modulation between orthogonal
channels exactly one third of that between co-polarized ones.

That ratio of three is the whole content of this file, and it is checked as an
exact number rather than as a trend, because it is one: unmodulated carriers over
a lossless dispersionless span have a closed-form nonlinear phase and there is
nothing to approximate.

The same coefficient also halves-and-then-some the nonlinear birefringence: the
two axes of one channel used to differ by ``gamma*(Px - Py)*L`` and now differ by
a third of it. Cross-polarization modulation is that difference varying with a
neighbour's power, and it arrives with the coupling rather than being added
beside it.
"""

from __future__ import annotations

import numpy as np
import pytest

from maiman.components import Fiber
from maiman.context import SimulationContext
from maiman.kernels import ORTHOGONAL_KERR_WEIGHT, propagate_coupled_ssfm
from maiman.signals import Band, OpticalSignal

SAMPLES = 1024
FS = 160e9
ANCHOR = 193.1e12
SPACING = 100e9

GAMMA = 1.3  # 1/W/km
LENGTH = 100.0  # km
POWER = 1e-3  # W

#: ``gamma * P * L`` in radians, the unit every phase here is quoted in.
UNIT = GAMMA * 1e-3 * POWER * LENGTH * 1e3

# Double precision because the claims here are exact ratios, and single
# precision rounds 2/3 at the seventh digit.
CTX = SimulationContext(
    bit_rate=10e9, samples_per_symbol=16, sequence_length=64, seed=3, precision="double"
)


def cw(power: float) -> np.ndarray:
    """An unmodulated carrier, so the nonlinear phase is one number."""
    return np.full(SAMPLES, np.sqrt(power), dtype=np.complex128)


def band(power_x: float, power_y: float, index: int = 0) -> Band:
    return Band(Ex=cw(power_x), Ey=cw(power_y), f0=ANCHOR + index * SPACING, fs=FS)


def span(*, coupled: bool, **settings: float | bool) -> Fiber:
    return Fiber(
        length=LENGTH,
        attenuation=0.0,
        dispersion=0.0,
        nonlinearity=GAMMA,
        four_wave_mixing=False,
        cross_polarization=coupled,
        label="fib",
        **settings,
    )


def propagated(
    bands: tuple[Band, ...], *, coupled: bool, **settings: float | bool
) -> OpticalSignal:
    signal = OpticalSignal(bands=bands, noise=())
    out = span(coupled=coupled, **settings).run(CTX, {"in": signal})["out"]
    assert isinstance(out, OpticalSignal)
    return out


def phase(
    bands: tuple[Band, ...], *, coupled: bool, axis: str = "Ex", channel: int = 0, **settings: float
) -> float:
    """Nonlinear phase in units of ``gamma*P*L``."""
    field = getattr(propagated(bands, coupled=coupled, **settings).bands[channel], axis)
    return float(np.angle(field[SAMPLES // 2])) / UNIT


# ---------------------------------------------------------------------------
# the default is the identity


def test_one_populated_axis_is_untouched_by_the_setting() -> None:
    """With all the light on X there is no orthogonal power to weight.

    Every nonlinear result in this project was taken with the axes uncoupled, so
    the setting has to be free where it should be free — and on the samples, not
    to within a tolerance.
    """
    bands = (band(POWER, 0.0),)
    uncoupled = propagated(bands, coupled=False).bands[0]
    coupled = propagated(bands, coupled=True).bands[0]
    assert np.array_equal(uncoupled.Ex, coupled.Ex)
    assert np.array_equal(uncoupled.Ey, coupled.Ey)
    assert phase(bands, coupled=True) == pytest.approx(1.0, rel=1e-9)


def test_the_coupling_is_off_unless_asked_for() -> None:
    """Which is the promise the compatibility argument rests on.

    A dual-polarization signal through a fibre nobody configured has to give the
    scalar answer — half the power on each axis, each axis seeing only its own
    half. Turning the coupling on by default would be defensible physics and a
    silent change to every nonlinear number this project has published.
    """
    assert span(coupled=False).cross_polarization is False
    default = Fiber(length=LENGTH, attenuation=0.0, dispersion=0.0, nonlinearity=GAMMA, label="fib")
    assert default.cross_polarization is False

    out = default.run(CTX, {"in": OpticalSignal(bands=(band(POWER / 2, POWER / 2),), noise=())})
    assert isinstance(out["out"], OpticalSignal)
    turned = float(np.angle(out["out"].bands[0].Ex[SAMPLES // 2])) / UNIT
    assert turned == pytest.approx(0.5, rel=1e-9)


def test_labelling_every_field_the_same_is_not_labelling_them() -> None:
    """The solver's own version of the same claim."""
    fields = [cw(POWER), cw(POWER)]
    common = {
        "beta2": [0.0, 0.0],
        "walkoff": [0.0, 0.0],
        "gamma": GAMMA * 1e-3,
        "alpha": 0.0,
        "distance": LENGTH * 1e3,
    }
    plain, _ = propagate_coupled_ssfm(fields, FS, **common)  # type: ignore[arg-type]
    labelled, _ = propagate_coupled_ssfm(fields, FS, polarization=[0, 0], **common)  # type: ignore[arg-type]
    assert np.array_equal(plain[0], labelled[0])
    assert np.array_equal(plain[1], labelled[1])


# ---------------------------------------------------------------------------
# the weight


def test_the_orthogonal_weight_is_two_thirds() -> None:
    """Split one channel's power evenly and read the self-phase off.

    Uncoupled, X sees only its own half: ``0.5 * gamma*P*L``. Coupled it also
    sees two thirds of the other half, which is ``0.5 + (2/3)*0.5``. Nothing here
    is fitted; both are closed forms.
    """
    bands = (band(POWER / 2, POWER / 2),)
    assert phase(bands, coupled=False) == pytest.approx(0.5, rel=1e-9)
    assert phase(bands, coupled=True) == pytest.approx(0.5 + ORTHOGONAL_KERR_WEIGHT * 0.5, rel=1e-9)


def test_orthogonal_cross_phase_modulation_is_a_third_of_co_polarized() -> None:
    """The textbook ratio, measured as an exact three.

    A co-polarized neighbour turns the channel by ``2*gamma*P*L`` — the factor of
    two that falls out of counting the terms in ``|A|**2 A``. An orthogonal one
    turns it by ``(2/3)*gamma*P*L``. Three is what a link designer uses to decide
    whether interleaving the polarizations of adjacent channels is worth the
    trouble, so it is worth pinning to the digit.
    """
    alone = phase((band(POWER, 0.0),), coupled=True)
    co_polarized = phase((band(POWER, 0.0), band(POWER, 0.0, 1)), coupled=True) - alone
    orthogonal = phase((band(POWER, 0.0), band(0.0, POWER, 1)), coupled=True) - alone

    assert co_polarized == pytest.approx(2.0, rel=1e-9)
    assert orthogonal == pytest.approx(ORTHOGONAL_KERR_WEIGHT, rel=1e-9)
    assert co_polarized / orthogonal == pytest.approx(3.0, rel=1e-9)


def test_an_orthogonal_neighbour_was_invisible_before() -> None:
    """Which is what the coupling exists to fix, stated as the old answer.

    Uncoupled, a neighbour polarized across the channel contributes nothing at
    all — not a third, zero. That is the gap the block's docstring used to
    declare.
    """
    alone = phase((band(POWER, 0.0),), coupled=False)
    orthogonal = phase((band(POWER, 0.0), band(0.0, POWER, 1)), coupled=False) - alone
    assert orthogonal == pytest.approx(0.0, abs=1e-12)


def test_the_weight_applies_between_bands_and_within_one() -> None:
    """Same two thirds in both places, which is why there is one constant.

    A channel's own orthogonal component and a neighbour's orthogonal component
    enter with the same weight; what differs is the factor of two the co-polarized
    sum carries and the orthogonal sum does not.
    """
    own = phase((band(POWER / 2, POWER / 2),), coupled=True) - 0.5
    alone = phase((band(POWER, 0.0),), coupled=True)
    neighbour = phase((band(POWER, 0.0), band(0.0, POWER / 2, 1)), coupled=True) - alone
    assert own == pytest.approx(ORTHOGONAL_KERR_WEIGHT * 0.5, rel=1e-9)
    assert neighbour == pytest.approx(ORTHOGONAL_KERR_WEIGHT * 0.5, rel=1e-9)


# ---------------------------------------------------------------------------
# what it does to the polarization state


def test_the_coupling_cuts_the_nonlinear_birefringence_by_exactly_three() -> None:
    """Cross-polarization modulation, in the only form an unmodulated pair can show it.

    Unequal power on the two axes makes them accumulate unequal phase, which
    rotates the state of polarization. Uncoupled that difference is
    ``gamma*(Px - Py)*L``; coupled, each axis also picks up two thirds of the
    other's power, and the difference falls to ``(1 - 2/3)`` of it. A factor of
    three, again, and from the same coefficient.
    """
    bands = (band(0.8 * POWER, 0.2 * POWER),)
    for coupled, expected in ((False, 0.6), (True, 0.6 * (1.0 - ORTHOGONAL_KERR_WEIGHT))):
        difference = phase(bands, coupled=coupled, axis="Ex") - phase(
            bands, coupled=coupled, axis="Ey"
        )
        assert difference == pytest.approx(expected, rel=1e-9)


def test_a_neighbours_power_moves_the_polarization_state() -> None:
    """Which is cross-polarization modulation proper: the rotation is not the
    channel's own business any more.

    A neighbour with all its power on Y adds two thirds of it to the channel's Y
    phase and nothing to its X phase, so the differential phase moves — by an
    amount set by a *different* channel's power, which is what makes XPolM a
    crosstalk mechanism rather than a self-effect.
    """
    channel = band(0.5 * POWER, 0.5 * POWER)
    alone = phase((channel,), coupled=True, axis="Ex") - phase((channel,), coupled=True, axis="Ey")
    with_neighbour = phase((channel, band(0.0, POWER, 1)), coupled=True, axis="Ex") - phase(
        (channel, band(0.0, POWER, 1)), coupled=True, axis="Ey"
    )

    assert alone == pytest.approx(0.0, abs=1e-12), "even axes start unrotated"
    # The neighbour is on Y, so it enters the channel's Y phase at the
    # co-polarized weight of 2 and its X phase at 2/3. What rotates the state is
    # the *difference* between the two weights, and that is the whole of XPolM.
    assert with_neighbour == pytest.approx(ORTHOGONAL_KERR_WEIGHT - 2.0, rel=1e-9)


def test_the_coupling_is_still_a_pure_phase() -> None:
    """It rotates; it must not move energy between the axes.

    The term that would — the coherent ``A_x* A_y**2`` exchange — is deliberately
    not modelled, and this is the assertion that says so in numbers rather than
    in the docstring.
    """
    bands = (band(0.7 * POWER, 0.3 * POWER), band(0.2 * POWER, 0.8 * POWER, 1))
    out = propagated(bands, coupled=True)
    for before, after in zip(bands, out.bands, strict=True):
        for axis in ("Ex", "Ey"):
            assert float(np.mean(np.abs(getattr(after, axis)) ** 2)) == pytest.approx(
                float(np.mean(np.abs(getattr(before, axis)) ** 2)), rel=1e-12
            )


def test_the_axes_couple_even_with_the_bands_left_independent() -> None:
    """Two settings, two questions: which bands see each other, and which axes do."""
    bands = (band(POWER / 2, POWER / 2), band(POWER, 0.0, 1))
    independent = phase(bands, coupled=True, cross_phase_modulation=False)
    together = phase(bands, coupled=True, cross_phase_modulation=True)

    # Its own axes still couple, so the self-phase is the split-power answer.
    assert independent == pytest.approx(0.5 + ORTHOGONAL_KERR_WEIGHT * 0.5, rel=1e-9)
    # And the neighbour adds 2 * its co-polarized power on top of that.
    assert together == pytest.approx(independent + 2.0, rel=1e-9)


# ---------------------------------------------------------------------------
# the solver's own guards


def test_polarization_labels_must_cover_every_field() -> None:
    with pytest.raises(ValueError, match="one entry per field"):
        propagate_coupled_ssfm(
            [cw(POWER), cw(POWER)],
            FS,
            beta2=[0.0, 0.0],
            walkoff=[0.0, 0.0],
            gamma=0.0,
            polarization=[0],
            alpha=0.0,
            distance=1e3,
        )


def test_polarization_labels_must_be_zero_or_one() -> None:
    with pytest.raises(ValueError, match="must be 0 or 1"):
        propagate_coupled_ssfm(
            [cw(POWER)],
            FS,
            beta2=[0.0],
            walkoff=[0.0],
            gamma=0.0,
            polarization=[2],
            alpha=0.0,
            distance=1e3,
        )
