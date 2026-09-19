"""A tilted grating's comb, as each input polarization sees it.

The scalar model has one comb for both polarizations. The true modes of the
cladding are HE, EH, TE and TM, an LP mode is a family of them split by the
glass-air boundary, and an input polarized in the plane of the tilt (p) weights
each family differently from one polarized across it (s). The checks:

* untilted, the coupling is exactly the long-period grating's vector coupling;
* p reaches TM0m and never TE0m, s the reverse, to the last bit;
* where the cladding guides weakly, the vector modes of one LP family together
  carry what the scalar LP mode does, for either polarization, and converge on
  it as the step at the cladding closes;
* and the block routes the field's ``x`` through p's comb and ``y`` through s's.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from maiman.components.tilted import TiltedFiberBraggGrating
from maiman.context import SimulationContext
from maiman.modes import StepIndexFibre, cladding_modes, core_modes
from maiman.photonics import tilted_grating_coupling, vector_tilted_coupling
from maiman.signals import Band, OpticalSignal
from maiman.vector_modes import vector_cladding_modes, vector_core_modes, vector_coupling

WAVELENGTH = 1.55e-6
PERIOD = 535e-9
TILT = math.radians(4.0)


def coupling(core, partner, polarization: str, tilt: float = TILT) -> float:  # type: ignore[no-untyped-def]
    return vector_tilted_coupling(
        core, partner, period=PERIOD, tilt=tilt, index_modulation=5e-4, polarization=polarization
    )


def test_untilted_it_is_the_long_period_grating_s_coupling() -> None:
    fibre = StepIndexFibre()
    core = vector_core_modes(fibre, WAVELENGTH, order=1)[0]
    for mode in vector_cladding_modes(fibre, WAVELENGTH, order=1, count=4):
        expected = abs(vector_coupling(core, mode, 5e-4))
        for polarization in ("p", "s"):
            assert coupling(core, mode, polarization, tilt=0.0) == pytest.approx(
                expected, rel=1e-12
            )
    for mode in vector_cladding_modes(fibre, WAVELENGTH, order=2, count=2):
        assert coupling(core, mode, "p", tilt=0.0) == 0.0


def test_p_reaches_tm_and_s_reaches_te_and_neither_the_other() -> None:
    fibre = StepIndexFibre()
    core = vector_core_modes(fibre, WAVELENGTH, order=1)[0]
    for mode in vector_cladding_modes(fibre, WAVELENGTH, order=0, count=6):
        reached, missed = ("p", "s") if mode.family == "TM" else ("s", "p")
        assert coupling(core, mode, reached) > 10.0, mode.name
        assert coupling(core, mode, missed) == 0.0, mode.name


def test_the_core_reflects_into_itself_as_the_scalar_core_does() -> None:
    fibre = StepIndexFibre()
    vector = vector_core_modes(fibre, WAVELENGTH, order=1)[0]
    scalar = core_modes(fibre, WAVELENGTH)[0]
    expected = tilted_grating_coupling(
        scalar, scalar, period=PERIOD, tilt=TILT, index_modulation=5e-4
    )
    for polarization in ("p", "s"):
        assert coupling(vector, vector, polarization) == pytest.approx(expected, rel=5e-3)


def family(fibre: StepIndexFibre, order: int, rank: int) -> list:  # type: ignore[type-arg]
    """The vector modes an ``LP_{order, rank}`` stands for, by family and rank."""
    members: list[tuple[int, str]]
    if order == 0:
        members = [(1, "HE")]
    elif order == 1:
        members = [(0, "TE"), (0, "TM"), (2, "HE")]
    else:
        members = [(order - 1, "EH"), (order + 1, "HE")]
    found = []
    for nu, name in members:
        modes = vector_cladding_modes(fibre, WAVELENGTH, order=nu, count=2 * rank + 4)
        found.append([m for m in modes if m.family == name][rank - 1])
    return found


def lp_and_family(
    surrounding: float, order: int, rank: int, polarization: str
) -> tuple[float, float]:
    """The scalar ``LP_{order, rank}``'s coupling, and its vector family's together."""
    fibre = StepIndexFibre(surrounding_index=surrounding)
    core = vector_core_modes(fibre, WAVELENGTH, order=1)[0]
    scalar_core = core_modes(fibre, WAVELENGTH)[0]
    scalar = cladding_modes(fibre, WAVELENGTH, order=order, count=rank)[rank - 1]
    expected = tilted_grating_coupling(
        scalar_core, scalar, period=PERIOD, tilt=TILT, index_modulation=5e-4
    )
    members = family(fibre, order, rank)
    together = math.sqrt(sum(coupling(core, m, polarization) ** 2 for m in members))
    return expected, together


@pytest.mark.parametrize("order", [0, 1, 2])
@pytest.mark.parametrize("rank", [1, 3])
@pytest.mark.parametrize("polarization", ["p", "s"])
def test_where_the_cladding_guides_weakly_a_family_carries_what_its_lp_mode_does(
    order: int, rank: int, polarization: str
) -> None:
    """In a liquid of 1.43 the cladding step is small, and the LP picture holds."""
    expected, together = lp_and_family(1.43, order, rank, polarization)
    assert together == pytest.approx(expected, rel=0.02)


def test_a_weakly_coupled_family_converges_on_its_lp_mode_as_the_step_closes() -> None:
    """LP3 is reached a thousandth as strongly as LP1 at four degrees, through J_3.

    Its vector family is reached through J_1 and J_3 together, and the J_1 terms,
    on the LP1 scale, cancel only in the limit. So a small vector correction to
    a large term is left over -- and shrinks as the step at the cladding closes.
    """
    errors = []
    for surrounding in (1.43, 1.438, 1.4425):
        expected, together = lp_and_family(surrounding, 3, 2, "p")
        errors.append(abs(together - expected))
    assert errors[0] > errors[1] > errors[2]


# ---------------------------------------------------------------------------
# The block
# ---------------------------------------------------------------------------

CTX = SimulationContext(
    bit_rate=10e9, samples_per_symbol=16, sequence_length=16, seed=3, precision="double"
)


@pytest.fixture(scope="module")
def grating() -> TiltedFiberBraggGrating:
    return TiltedFiberBraggGrating(label="t", vector=True, azimuthal_orders=1.0, length=30.0)


@pytest.fixture(scope="module")
def combs(grating: TiltedFiberBraggGrating) -> dict[str, np.ndarray]:
    """Both polarizations' transmission a nanometre wide, five below the Bragg line.

    Near the cladding's cutoff the glass-air boundary splits the families most:
    TE and TM part by tens of picometres and HE and EH are weighted very
    differently, and a 30 mm grating's notches are narrow enough to show it.
    """
    bragg = grating.bragg_wavelength()
    out: dict[str, np.ndarray] = {}
    for polarization in ("tm", "te"):
        spectrum = grating.spectrum(
            bragg - 6.1e-9, bragg - 5.1e-9, points=801, polarization=polarization
        )
        out["wavelengths"] = spectrum.wavelengths
        out[polarization] = spectrum.power_db("out", "in")
    return out


def test_each_polarization_sees_its_own_comb(combs: dict[str, np.ndarray]) -> None:
    difference = np.abs(combs["tm"] - combs["te"])
    assert float(difference.max()) > 10.0, "a notch one polarization has and the other lacks"


def test_the_block_sends_x_through_p_and_y_through_s(
    grating: TiltedFiberBraggGrating, combs: dict[str, np.ndarray]
) -> None:
    where = int(np.argmax(np.abs(combs["tm"] - combs["te"])))
    frequency = 299792458.0 / float(combs["wavelengths"][where])

    samples = 64
    carrier = Band(
        Ex=np.full(samples, 1e-2, dtype=np.complex128),
        Ey=np.full(samples, 1e-2, dtype=np.complex128),
        f0=frequency,
        fs=10e9,
    )
    out = grating.run(CTX, {"in": OpticalSignal(bands=(carrier,))})["transmitted"]
    assert isinstance(out, OpticalSignal)
    band = out.bands[0]
    # The block interpolates its modes over the carrier's own few gigahertz rather
    # than the nanometre above, so the two agree to the interpolation, not the bit.
    x_db = 10 * np.log10(float(np.mean(np.abs(band.Ex) ** 2)) / 1e-4)
    y_db = 10 * np.log10(float(np.mean(np.abs(band.Ey) ** 2)) / 1e-4)
    assert x_db == pytest.approx(float(combs["tm"][where]), abs=0.3)
    assert y_db == pytest.approx(float(combs["te"][where]), abs=0.3)


def test_the_flag_is_off_by_default() -> None:
    assert not TiltedFiberBraggGrating().vector
