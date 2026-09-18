"""An edge coupler's facets as a cavity, and a grating coupler's passband from its stack.

Both were compact numbers before: the facets two independent losses, the grating a
peak loss and a bandwidth taken on trust from a PDK. Each is now physics, and each
is held to something that owes it nothing.

The etalon is summed bounce by bounce from Gaussian beams, so it is checked where
it must become the Airy function -- modes too wide to diffract across the gap --
and, where it must not, against a brute-force propagation of the cavity: an
angular-spectrum FFT, a tilted mirror as a phase ramp, and the bounces summed by
hand. They agree to 2e-4 square on and 7e-3 at six degrees of tilt.

The grating's directionality is a transfer-matrix result, checked against a
direct solve of every layer's boundary conditions with the grating as a jump in
the field's slope. Its overlap with the fibre is checked against the number the
grating-coupler literature is built on: an exponential beam meets a Gaussian at
best 80.1 percent of the way.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from maiman.components import EdgeCoupler, GratingCoupler
from maiman.photonics import (
    _grating_mode_overlap,
    edge_coupler,
    gaussian_coupling,
    gaussian_overlap,
    grating_coupler_centre,
    grating_coupler_stack,
    grating_directionality,
)
from maiman.units import C_LIGHT

WAVELENGTH = 1.55e-6
FIBRE = 5.2e-6
CHIP = 1.5e-6
R_FIBRE = (1.0 - 1.4682) / 2.4682
R_CHIP = (1.0 - 1.45) / 2.45


# ---------------------------------------------------------------------------
# The complex coupling the etalon is summed from
# ---------------------------------------------------------------------------


def test_the_amplitude_squares_to_the_overlap() -> None:
    wavelengths = np.linspace(1.5e-6, 1.6e-6, 7)
    for settings in (
        {"offset": 1e-6, "tilt": 0.02, "gap": 30e-6, "index": 1.0},
        {"offset": -2e-6, "tilt": -0.05, "gap": 0.0, "index": 1.45},
    ):
        amplitude = gaussian_coupling(FIBRE, CHIP, wavelength=wavelengths, **settings)
        power = gaussian_overlap(FIBRE, CHIP, wavelength=wavelengths, **settings)
        assert np.allclose(np.abs(amplitude) ** 2, power, rtol=0.0, atol=1e-15)


# A one-dimensional angular-spectrum propagator, shared by the brute-force checks.
X = np.linspace(-400e-6, 400e-6, 2**15)
DX = float(X[1] - X[0])
K = 2.0 * math.pi / WAVELENGTH
KX = 2.0 * math.pi * np.fft.fftfreq(X.size, DX)
KZ = np.where(KX**2 <= K**2, np.sqrt(np.abs(K**2 - KX**2)), -1j * np.sqrt(np.abs(KX**2 - K**2)))


def propagate(field: np.ndarray, distance: float) -> np.ndarray:
    """Exact in angle, with the carrier ``exp(-i k d)`` taken out."""
    return np.asarray(np.fft.ifft(np.fft.fft(field) * np.exp(-1j * (KZ - K) * distance)))


def gauss(radius: float, centre: float = 0.0) -> np.ndarray:
    return np.asarray(np.exp(-((X - centre) ** 2) / radius**2))


def project(field: np.ndarray, mode: np.ndarray) -> complex:
    return complex(np.sum(field * mode) * DX / math.sqrt(float(np.sum(mode**2)) * DX))


def test_the_gouy_phase_is_the_one_a_beam_really_has() -> None:
    """Against exact propagation; what is left, 1e-3, is the paraxial approximation itself."""
    source = gauss(3e-6, -1e-6)
    norm = math.sqrt(float(np.sum(source**2)) * DX)
    brute = project(propagate(source, 20e-6), gauss(2.5e-6)) / norm
    mine = complex(gaussian_coupling(3e-6, 2.5e-6, wavelength=WAVELENGTH, offset=-1e-6, gap=20e-6))
    assert abs(mine - brute) < 1.5e-3


# ---------------------------------------------------------------------------
# The etalon
# ---------------------------------------------------------------------------


def etalon(frequencies: np.ndarray, **changes: Any) -> np.ndarray:
    settings: dict[str, Any] = {
        "fibre_mode_radius": FIBRE,
        "chip_mode_radius_x": CHIP,
        "chip_mode_radius_y": CHIP,
        "gap": 50e-6,
    }
    settings.update(changes)
    return edge_coupler(frequencies, etalon=True, **settings).s


def test_without_diffraction_it_is_the_airy_function() -> None:
    """Modes a hundred microns wide cross five microns without noticing: a thin film."""
    frequencies = C_LIGHT / np.linspace(1.54e-6, 1.56e-6, 401)
    s = etalon(
        frequencies,
        fibre_mode_radius=50e-6,
        chip_mode_radius_x=50e-6,
        chip_mode_radius_y=50e-6,
        gap=5e-6,
    )
    transmitted, reflected = np.abs(s[:, 1, 0]) ** 2, np.abs(s[:, 0, 0]) ** 2
    delta = 2.0 * (2.0 * math.pi * frequencies / C_LIGHT) * 5e-6
    airy = (
        (1 - R_FIBRE**2) * (1 - R_CHIP**2) / np.abs(1 - R_FIBRE * R_CHIP * np.exp(-1j * delta)) ** 2
    )
    assert np.max(np.abs(transmitted - airy)) < 5e-5
    assert np.max(np.abs(transmitted + reflected - 1.0)) < 5e-7, "and nothing is lost"


def test_at_zero_gap_the_two_facets_are_one_interface() -> None:
    s = etalon(
        np.array([C_LIGHT / WAVELENGTH]),
        chip_mode_radius_x=FIBRE,
        chip_mode_radius_y=FIBRE,
        gap=0.0,
    )
    direct = 1.0 - ((1.4682 - 1.45) / (1.4682 + 1.45)) ** 2
    assert abs(s[0, 1, 0]) ** 2 == pytest.approx(direct, rel=0.0, abs=1e-12)


@pytest.mark.parametrize(("degrees", "tolerance"), [(0.0, 5e-4), (3.0, 5e-4), (6.0, 1e-2)])
def test_the_cavity_is_the_one_a_brute_force_propagation_builds(
    degrees: float, tolerance: float
) -> None:
    """Twelve bounces between a tilted fibre and a flat chip facet, done the long way.

    The fibre's facet is square to the fibre, so tilting it tilts one mirror
    against the other; paraxially that mirror is a phase ramp of ``2 k theta x``.
    The chip's facet is flat. Across the chip's plane the field is propagated
    exactly in angle, and each arrival projected onto the chip mode. The model
    unfolds the same cavity into a sum of Gaussian beams in closed form -- and
    the difference is the paraxial approximation both make, growing with tilt.
    """
    tilt = math.radians(degrees)
    gap, offset = 10e-6, 0.5e-6
    start = offset - gap * tilt
    field = gauss(FIBRE, start) * np.exp(-1j * K * math.sin(tilt) * X)
    norm = math.sqrt(float(np.sum(np.abs(field) ** 2)) * DX)
    chip = gauss(CHIP)
    total = 0j
    for n in range(12):
        arrive = propagate(field, gap)
        # Across the chip (y) nothing tilts: the same Gaussian coupling the model uses.
        across = complex(
            gaussian_coupling(FIBRE, FIBRE, wavelength=WAVELENGTH, gap=(2 * n + 1) * gap)
        )
        total += project(arrive, chip) / norm * across * np.exp(-1j * K * gap * (2 * n + 1))
        back = propagate(R_CHIP * arrive, gap)
        field = R_FIBRE * back * np.exp(-2j * K * tilt * (X - start))
    brute = math.sqrt((1 - R_FIBRE**2) * (1 - R_CHIP**2)) * total

    s = etalon(
        np.array([C_LIGHT / WAVELENGTH]),
        chip_mode_radius_y=FIBRE,
        offset_x=offset,
        tilt=tilt,
        gap=gap,
    )
    assert abs(complex(s[0, 1, 0]) - brute) < tolerance


def test_fifty_microns_of_air_ripples_by_half_a_decibel_every_three_terahertz() -> None:
    frequencies = np.linspace(C_LIGHT / 1.6e-6, C_LIGHT / 1.5e-6, 6001)
    power = np.abs(etalon(frequencies)[:, 1, 0]) ** 2
    ripple = 10.0 * math.log10(power.max() / power.min())
    assert ripple == pytest.approx(0.488, abs=0.005)
    peaks = np.flatnonzero((power[1:-1] > power[:-2]) & (power[1:-1] > power[2:])) + 1
    spacing = float(np.diff(frequencies[peaks]).mean())
    assert spacing == pytest.approx(C_LIGHT / (2.0 * 50e-6), rel=1e-3), "c / 2g, the etalon's own"


def test_an_angled_fibre_walks_the_cavity_off_and_the_ripple_goes_with_it() -> None:
    """Eight degrees takes 0.49 dB of ripple to 0.04: why fibre arrays are polished at an angle."""
    frequencies = np.linspace(C_LIGHT / 1.6e-6, C_LIGHT / 1.5e-6, 6001)
    power = np.abs(etalon(frequencies, tilt=math.radians(8.0))[:, 1, 0]) ** 2
    assert 10.0 * math.log10(power.max() / power.min()) == pytest.approx(0.0435, abs=0.003)


def test_an_index_matched_gap_has_nothing_to_resonate_with() -> None:
    """No reflection at the chip, one pass, and exactly the model without an etalon."""
    frequencies = C_LIGHT / np.linspace(1.5e-6, 1.6e-6, 11)
    settings: dict[str, Any] = {
        "fibre_mode_radius": FIBRE,
        "chip_mode_radius_x": CHIP,
        "chip_mode_radius_y": CHIP,
        "gap": 10e-6,
        "gap_index": 1.45,
        "mode_index": 1.45,
    }
    cavity = edge_coupler(frequencies, etalon=True, **settings).s
    plain = edge_coupler(frequencies, **settings).s
    assert np.allclose(np.abs(cavity[:, 1, 0]), np.abs(plain[:, 1, 0]), rtol=0.0, atol=1e-15)


def test_no_port_gives_back_more_than_it_was_given() -> None:
    frequencies = C_LIGHT / np.linspace(1.5e-6, 1.6e-6, 201)
    for tilt in (0.0, math.radians(4.0)):
        s = etalon(frequencies, tilt=tilt, gap=20e-6, offset_x=1e-6)
        for port in (0, 1):
            total = np.abs(s[:, 1 - port, port]) ** 2 + np.abs(s[:, port, port]) ** 2
            assert total.max() <= 1.0 + 1e-12


def test_the_block_carries_the_etalon_when_asked() -> None:
    frequencies = C_LIGHT / np.linspace(1.5e-6, 1.6e-6, 11)
    block = EdgeCoupler(gap=50.0, etalon=True, label="edge")
    expected = edge_coupler(
        frequencies,
        fibre_mode_radius=FIBRE,
        chip_mode_radius_x=CHIP,
        chip_mode_radius_y=CHIP,
        gap=block.si("gap"),
        etalon=True,
    )
    assert np.allclose(block.scattering_matrix(frequencies).s, expected.s, rtol=0.0, atol=1e-14)
    assert not EdgeCoupler(label="edge").etalon, "off by default, so no earlier result moves"


# ---------------------------------------------------------------------------
# The grating's directionality
# ---------------------------------------------------------------------------

STACK: dict[str, Any] = {
    "top_index": 1.0,
    "silicon_index": 3.476,
    "silicon_thickness": 220e-9,
    "box_index": 1.444,
    "box_thickness": 2e-6,
    "substrate_index": 3.476,
}


def directly(sine: float, **stack: float) -> float:
    """Every layer's boundary conditions solved at once, with the grating a jump in slope.

    TE fields ``A exp(i k_z z) + B exp(-i k_z z)``: nothing incoming from above
    or below, ``E`` and ``dE/dz`` continuous at every interface, and at the
    middle of the silicon ``E`` continuous while ``dE/dz`` jumps. Power leaving
    is ``k_z |E|^2`` in the top medium and in the substrate.
    """
    k = 2.0 * math.pi / WAVELENGTH
    n_top, n_si = stack["top_index"], stack["silicon_index"]
    n_box, n_sub = stack["box_index"], stack["substrate_index"]
    h, t_box = stack["silicon_thickness"] / 2.0, stack["box_thickness"]
    beta = k * n_top * sine

    def kz(n: float) -> complex:
        return complex(np.sqrt(complex((k * n) ** 2 - beta**2)))

    k1, k2, k3, k4 = kz(n_top), kz(n_si), kz(n_box), kz(n_sub)
    # Unknowns: a1 (top, up); a2 b2, Si above the source; a3 b3, below it; a4 b4, BOX;
    # b5, substrate, going down.
    rows, rhs = [], []

    def up(kk: complex, z: float) -> complex:
        return complex(np.exp(1j * kk * z))

    def dn(kk: complex, z: float) -> complex:
        return complex(np.exp(-1j * kk * z))

    def row(entries: dict[int, complex], value: complex = 0.0) -> None:
        line = np.zeros(8, dtype=np.complex128)
        for index, entry in entries.items():
            line[index] = entry
        rows.append(line)
        rhs.append(value)

    # z = h: top / Si upper
    row({0: up(k1, h), 1: -up(k2, h), 2: -dn(k2, h)})
    row({0: 1j * k1 * up(k1, h), 1: -1j * k2 * up(k2, h), 2: 1j * k2 * dn(k2, h)})
    # z = 0: the source
    row({1: 1.0, 2: 1.0, 3: -1.0, 4: -1.0})
    row({1: 1j * k2, 2: -1j * k2, 3: -1j * k2, 4: 1j * k2}, 1.0)
    # z = -h: Si lower / BOX
    z = -h
    row({3: up(k2, z), 4: dn(k2, z), 5: -up(k3, z), 6: -dn(k3, z)})
    row(
        {
            3: 1j * k2 * up(k2, z),
            4: -1j * k2 * dn(k2, z),
            5: -1j * k3 * up(k3, z),
            6: 1j * k3 * dn(k3, z),
        }
    )
    # z = -h - t_box: BOX / substrate
    z = -h - t_box
    row({5: up(k3, z), 6: dn(k3, z), 7: -dn(k4, z)})
    row({5: 1j * k3 * up(k3, z), 6: -1j * k3 * dn(k3, z), 7: 1j * k4 * dn(k4, z)})

    solution = np.linalg.solve(np.array(rows), np.array(rhs))
    upward = k1.real * abs(solution[0]) ** 2
    downward = k4.real * abs(solution[7]) ** 2
    return float(upward / (upward + downward))


@pytest.mark.parametrize("box", [1.0e-6, 1.7e-6, 2.0e-6, 2.3e-6])
@pytest.mark.parametrize("degrees", [0.0, 10.0, 25.0])
def test_the_directionality_is_every_boundary_condition_solved(box: float, degrees: float) -> None:
    sine = math.sin(math.radians(degrees))
    stack = {**STACK, "box_thickness": box}
    assert float(grating_directionality(WAVELENGTH, emission_sine=sine, **stack)) == pytest.approx(
        directly(sine, **stack), abs=1e-12
    )


def test_with_nothing_to_reflect_it_is_one_half() -> None:
    assert float(
        grating_directionality(
            WAVELENGTH,
            emission_sine=0.17,
            top_index=1.444,
            silicon_index=3.476,
            box_index=1.444,
            substrate_index=1.444,
        )
    ) == pytest.approx(0.5, abs=1e-15)


def test_the_oxide_is_a_cavity_that_repeats_every_half_wave() -> None:
    """``lambda / (2 n_box cos(theta_box))``; and 2.2 microns beats the standard 2 by 1.1 dB."""
    sine = math.sin(math.radians(10.0))
    inside = sine / 1.444
    period = WAVELENGTH / (2.0 * 1.444 * math.sqrt(1.0 - inside**2))
    first = grating_directionality(WAVELENGTH, emission_sine=sine, **STACK)
    later = grating_directionality(
        WAVELENGTH, emission_sine=sine, **{**STACK, "box_thickness": 2e-6 + period}
    )
    assert float(later) == pytest.approx(float(first), abs=1e-13)
    assert float(first) == pytest.approx(0.592, abs=0.001)


# ---------------------------------------------------------------------------
# The passband
# ---------------------------------------------------------------------------


def test_an_exponential_beam_meets_a_gaussian_one_at_best_eighty_percent() -> None:
    """The ceiling every uniform grating coupler lives under: 80.1 %, at ``alpha w = 0.70``."""
    best, where = 0.0, 0.0
    for strength in np.linspace(0.05e6, 0.6e6, 111):
        value = max(
            float(
                _grating_mode_overlap(
                    np.array([WAVELENGTH]),
                    np.zeros(1),
                    strength=float(strength),
                    length=200e-6,
                    fibre_radius=FIBRE,
                    position=float(place),
                    top_index=1.0,
                )[0]
            )
            for place in np.linspace(0.0, 20e-6, 81)
        )
        if value > best:
            best, where = value, float(strength)
    assert best == pytest.approx(0.801, abs=0.001)
    assert where * FIBRE == pytest.approx(0.70, abs=0.02)


GRATING: dict[str, Any] = {
    "period": 611e-9,
    "effective_index": 2.71,
    "group_index": 4.0,
    "reference_wavelength": 1.55e-6,
    "angle": math.radians(10.0),
}
BAND = np.linspace(1.45e-6, 1.65e-6, 4001)


def passband(**changes: Any) -> tuple[float, float, float]:
    s = grating_coupler_stack(C_LIGHT / BAND, **{**GRATING, **changes})
    power = np.abs(s.s[:, 1, 0]) ** 2
    inside = BAND[power >= power.max() * 10.0 ** (-0.1)]
    return (
        float(-10.0 * np.log10(power.max())),
        float(BAND[power.argmax()]),
        float(inside.max() - inside.min()),
    )


def test_the_default_stack_makes_the_passband_a_pdk_would_quote() -> None:
    """3.19 dB and 35.7 nm, from 220 nm of silicon on 2 microns of oxide and nothing else.

    It peaks 4 nm short of where phase matching centres it, because the share
    going up is still rising toward shorter wavelengths there.
    """
    loss, peak, width = passband()
    assert loss == pytest.approx(3.192, abs=0.005)
    assert width * 1e9 == pytest.approx(35.7, abs=0.2)
    centre = grating_coupler_centre(**GRATING)
    assert peak * 1e9 == pytest.approx(1545.85, abs=0.1)
    assert (centre - peak) * 1e9 == pytest.approx(3.95, abs=0.15)


def test_a_wider_fibre_mode_accepts_fewer_angles_and_so_fewer_wavelengths() -> None:
    assert passband(fibre_radius=10e-6)[2] * 1e9 == pytest.approx(24.6, abs=0.2)


def test_a_lower_group_index_turns_the_beam_more_slowly_and_widens_the_band() -> None:
    """The emission angle turns at ``(n_eff - n_g)/lambda_0 - 1/period`` per metre of wavelength."""
    assert passband(group_index=3.6)[2] * 1e9 == pytest.approx(39.5, abs=0.2)


def test_a_better_oxide_buys_back_a_decibel() -> None:
    assert passband(box_thickness=2.2e-6)[0] == pytest.approx(2.08, abs=0.01)


def test_tm_is_rejected_across_the_whole_band() -> None:
    frequencies = C_LIGHT / np.linspace(1.5e-6, 1.6e-6, 11)
    te = grating_coupler_stack(frequencies, **GRATING).s[:, 1, 0]
    tm = grating_coupler_stack(frequencies, extinction_db=25.0, **GRATING).s[:, 1, 0]
    assert np.allclose(np.abs(tm) ** 2, np.abs(te) ** 2 * 10**-2.5, rtol=1e-12, atol=0.0)


def test_where_nothing_can_radiate_nothing_is_coupled() -> None:
    """A period too short to phase match any real angle leaves the light in the chip."""
    s = grating_coupler_stack(np.array([C_LIGHT / WAVELENGTH]), **{**GRATING, "period": 300e-9})
    assert s.s[0, 1, 0] == 0.0


def test_what_cannot_be_a_grating_is_refused() -> None:
    frequencies = np.array([C_LIGHT / WAVELENGTH])
    with pytest.raises(ValueError, match="period"):
        grating_coupler_stack(frequencies, **{**GRATING, "period": 0.0})
    with pytest.raises(ValueError, match="strength and length"):
        grating_coupler_stack(frequencies, strength=0.0, **GRATING)
    with pytest.raises(ValueError, match="thickness"):
        grating_coupler_stack(frequencies, silicon_thickness=0.0, **GRATING)


def test_the_block_computes_its_passband_when_asked() -> None:
    block = GratingCoupler(from_stack=True, label="gc")
    loss, peak, width = block.passband()
    assert loss == pytest.approx(3.192, abs=0.005)
    assert width * 1e9 == pytest.approx(35.7, abs=0.1)
    assert block.directionality() == pytest.approx(0.593, abs=0.001)
    pdk = GratingCoupler(label="gc").passband()
    assert pdk[0] == pytest.approx(3.0, abs=1e-6), "and the PDK numbers when not"
    assert peak != pdk[1]
