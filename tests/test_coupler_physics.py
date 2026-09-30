"""An edge coupler's facets as a cavity, and a grating coupler's passband from its stack.

Both were compact numbers before: the facets two independent losses, the grating a
peak loss and a bandwidth taken on trust from a PDK. Each is now physics, and each
is held to something that owes it nothing.

The etalon is summed bounce by bounce from Gaussian beams, so it is checked where
it must become the Airy function -- modes too wide to diffract across the gap --
and, where it must not, against a brute-force propagation of the cavity: an
angular-spectrum FFT, a tilted mirror reflecting each plane wave by the exact law
of reflection, and the bounces summed by hand. They agree to 2e-4 square on and
to 7.2e-3 anywhere across +-30 degrees of tilt.

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
    finite_coupler_response,
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


#: A coarser grid for the exact reflection, which sums each bounce back onto the
#: grid directly: 4096 points over 300 um still resolves every propagating wave.
XR = np.linspace(-150e-6, 150e-6, 4096)
DXR = float(XR[1] - XR[0])
KXR = 2.0 * math.pi * np.fft.fftfreq(XR.size, DXR)
REACHING = np.abs(KXR) < K
KZR = np.sqrt(np.where(REACHING, K**2 - KXR**2, 0.0))


def propagate_exact(field: np.ndarray, distance: float) -> np.ndarray:
    """Exact in angle on the coarse grid, carrier out, evanescent waves dropped."""
    spectrum = np.where(REACHING, np.fft.fft(field) * np.exp(-1j * (KZR - K) * distance), 0.0)
    return np.asarray(np.fft.ifft(spectrum))


def reflect_off_tilted_facet(field: np.ndarray, tilt: float, through: float) -> np.ndarray:
    """A plane mirror through ``(through, 0)``, its normal ``tilt`` from the axis, exactly.

    Every plane wave the field is made of, arriving at the fibre, leaves as
    ``k - 2 (k . n) n`` -- the law of reflection, with no small-angle step in it --
    and the reflected field is summed back on the grid directly, because the
    reflected wavenumbers no longer sit on it. Evanescent components never reach
    the facet and are dropped.
    """
    spectrum = np.fft.fft(field) / field.size
    kx = KXR[REACHING]
    kz = -np.sqrt(K**2 - kx**2)  # travelling back towards the fibre
    amplitude = spectrum[REACHING] * np.exp(1j * kx * (through - XR[0]))
    # Tilted the way the model tilts the fibre: a positive tilt turns the
    # beam's tangential wavenumber negative, ``exp(-i k sin(tilt) x)``.
    normal = (-math.sin(tilt), math.cos(tilt))
    leaving = kx - 2.0 * (kx * normal[0] + kz * normal[1]) * normal[0]
    return np.asarray(np.exp(1j * np.outer(XR - through, leaving)) @ amplitude)


@pytest.mark.parametrize("degrees", [0.0, 3.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0, -8.0])
def test_the_cavity_is_the_one_an_exact_reflection_builds(degrees: float) -> None:
    """Twelve bounces between a tilted fibre and a flat chip facet, done the long way.

    The fibre's facet is square to the fibre, so tilting it tilts one mirror
    against the other. Here that mirror reflects every plane wave by the exact
    law of reflection, the field is propagated exactly in angle between the
    facets, and each arrival is projected onto the chip mode. The model unfolds
    the same cavity into a sum of paraxial Gaussian beams in closed form.

    Measured 1.7e-4 square on, 3.8e-3 at the 8 degrees an array is polished to,
    and no more than 7.2e-3 anywhere across the block's +-30 degrees. The last
    reference this was held to made the facet a paraxial phase ramp, and most of
    the 7e-3 it reported at six degrees was that ramp's own error.
    """
    tilt = math.radians(degrees)
    gap, offset = 10e-6, 0.5e-6
    start = offset - gap * math.tan(tilt)

    def gauss_r(radius: float, centre: float = 0.0) -> np.ndarray:
        return np.asarray(np.exp(-((XR - centre) ** 2) / radius**2))

    field = gauss_r(FIBRE, start) * np.exp(-1j * K * math.sin(tilt) * XR)
    norm = math.sqrt(float(np.sum(np.abs(field) ** 2)) * DXR)
    chip = gauss_r(CHIP)
    chip_norm = math.sqrt(float(np.sum(chip**2)) * DXR)
    total = 0j
    for n in range(12):
        arrive = propagate_exact(field, gap)
        # Across the chip (y) nothing tilts: the same Gaussian coupling the model uses.
        across = complex(
            gaussian_coupling(FIBRE, FIBRE, wavelength=WAVELENGTH, gap=(2 * n + 1) * gap)
        )
        projected = complex(np.sum(arrive * chip) * DXR / chip_norm)
        total += projected / norm * across * np.exp(-1j * K * gap * (2 * n + 1))
        back = propagate_exact(R_CHIP * arrive, gap)
        field = R_FIBRE * reflect_off_tilted_facet(back, tilt, start)
    exact = math.sqrt((1 - R_FIBRE**2) * (1 - R_CHIP**2)) * total

    s = etalon(
        np.array([C_LIGHT / WAVELENGTH]),
        chip_mode_radius_y=FIBRE,
        offset_x=offset,
        tilt=tilt,
        gap=gap,
    )
    assert abs(complex(s[0, 1, 0]) - exact) < 1e-2
    if abs(degrees) <= 3.0:
        assert abs(complex(s[0, 1, 0]) - exact) < 5e-4, "and paraxial really is exact near square"


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


def test_the_etalon_is_refused_past_the_tilt_it_is_validated_to() -> None:
    frequencies = np.array([C_LIGHT / WAVELENGTH])
    etalon(frequencies, tilt=math.radians(30.0))
    with pytest.raises(ValueError, match="validated to 30 degrees"):
        etalon(frequencies, tilt=math.radians(31.0))


# -- the teeth's own reflection in the fibre-chip gap (maiman-gz7) -------------------

GAP = {**GRATING, "fibre_height": 15e-6, "duty": 0.5}
GAP_BAND = C_LIGHT / np.linspace(1.53e-6, 1.56e-6, 121)


def gap_power(**changes: Any) -> np.ndarray:
    return np.abs(grating_coupler_stack(GAP_BAND, **{**GAP, **changes}).s[:, 1, 0]) ** 2


def test_the_teeth_reflection_needs_a_gap_to_act_in() -> None:
    """With the fibre touching the chip there is no cavity, so no etch changes anything."""
    flat = np.abs(grating_coupler_stack(GAP_BAND, **{**GRATING, "duty": 0.5}).s[:, 1, 0])
    etched = np.abs(
        grating_coupler_stack(GAP_BAND, **{**GRATING, "duty": 0.5}, etch_depth=70e-9).s[:, 1, 0]
    )
    assert np.array_equal(flat, etched)


def test_a_vanishing_etch_is_the_bare_stack_up_to_the_angle_it_is_seen_at() -> None:
    """Depth zero is the stack :func:`stack_reflection` folds -- the RCWA tests hold that exactly.

    The two differ here by 3e-3 in power, and not because of the solve: the bare
    stack is taken at each wavelength's emission angle and the teeth's at the
    fibre's, which is the angle the beam in the gap actually travels at.
    """
    assert np.max(np.abs(gap_power(etch_depth=1e-12) - gap_power())) < 5e-3


def test_the_etched_teeth_change_the_gap_ripple_and_keep_the_coupling_physical() -> None:
    """A 70 nm etch moves the ripple by 0.026 in power at most, where the bare stack's is 0.70 dB.

    The etched ripple is 1.02 dB peak to peak: the grating's own reflection is
    not the plain slab's, in phase as well as size. Coupling never passes unity.
    """
    bare, etched = gap_power(), gap_power(etch_depth=70e-9)
    assert np.max(np.abs(etched - bare)) == pytest.approx(0.026, abs=0.004)
    assert 10 * np.log10(bare.max() / bare.min()) == pytest.approx(0.70, abs=0.05)
    assert 10 * np.log10(etched.max() / etched.min()) == pytest.approx(1.02, abs=0.05)
    assert etched.max() <= 1.0


def test_the_teeth_are_refused_a_depth_the_silicon_cannot_have() -> None:
    with pytest.raises(ValueError, match="etch_depth"):
        grating_coupler_stack(GAP_BAND, **GAP, etch_depth=300e-9)
    with pytest.raises(ValueError, match="etch_depth"):
        grating_coupler_stack(GAP_BAND, **GAP, etch_depth=0.0)
    with pytest.raises(ValueError, match="teeth_harmonics"):
        grating_coupler_stack(GAP_BAND, **GAP, etch_depth=70e-9, teeth_harmonics=0)


# -- a finite coupler: the light that leaves down the waveguide (maiman-af0) --------

FINITE: dict[str, Any] = {
    "sine": math.sin(math.radians(10.0)),
    "pitch": 611e-9,
    "duty": 0.5,
    "etch_depth": 70e-9,
}


def test_a_finite_coupler_accounts_for_every_watt() -> None:
    """Reflected, transmitted and taken by the absorber sum to one.

    The balance an infinite grating cannot write, since what it couples into the
    waveguide comes back out of it as a resonance.
    """
    result = finite_coupler_response(1.55e-6, periods=12, beam_radius=3e-6, **FINITE)
    assert result.reflected + result.transmitted + result.coupled == pytest.approx(1.0, abs=1e-9)
    assert 0.2 < result.coupled < 0.5, "a 220 nm on 2 um oxide coupler takes a third of a beam"


def test_the_power_that_leaves_does_not_depend_on_how_it_is_absorbed() -> None:
    """Move the absorber's length and strength: the coupled power stays within half a percent.

    Past 30 micrometres of waveguide and an extinction of 0.05 the guided light is
    gone before it can return, so what the absorber took is what left: 0.3562 at
    30 um and 0.05, 0.3545 at 0.10, 0.3549 at 45 um.
    """
    taken = [
        finite_coupler_response(
            1.55e-6,
            periods=15,
            beam_radius=4e-6,
            absorber_length=length,
            absorber_extinction=extinction,
            **FINITE,
        ).coupled
        for length, extinction in ((30e-6, 0.05), (30e-6, 0.10), (45e-6, 0.05))
    ]
    assert max(taken) - min(taken) < 0.005
    assert taken[0] == pytest.approx(0.356, abs=0.01)


def test_light_coupled_in_is_radiated_out_again_by_the_grating_it_crosses() -> None:
    """A fixed 3 um beam: the more grating between it and the end, the less leaves.

    0.37, 0.29, 0.056 for 8, 16 and 32 periods. An infinite grating has no end, so all
    of what it couples in comes back as radiation -- the resonance the infinite
    solve shows -- and a finite one lets out only what reaches the end first.
    """
    taken = [
        finite_coupler_response(1.55e-6, periods=periods, beam_radius=3e-6, **FINITE).coupled
        for periods in (8, 16, 32)
    ]
    assert taken[0] > taken[1] > taken[2]
    assert taken == pytest.approx([0.367, 0.292, 0.056], abs=0.01)


def test_a_long_grating_under_a_wide_beam_reflects_what_the_infinite_one_does() -> None:
    """60 teeth under a 20 um beam: the power sent back is the infinite grating's, to 5 %.

    0.139 against 0.133 -- the infinite solve's whole reflectance, from
    :func:`maiman.rcwa.diffract_te` by a route that shares only its linear algebra.
    """
    from maiman.rcwa import GratingLayer, UniformLayer, diffract_te

    plane = diffract_te(
        1.55e-6,
        sine=FINITE["sine"],
        period=611e-9,
        top_index=1.0,
        layers=[
            GratingLayer(70e-9, 3.476, 1.0, 0.5),
            UniformLayer(150e-9, 3.476),
            UniformLayer(2e-6, 1.444),
        ],
        substrate_index=3.476,
        harmonics=20,
    )
    finite = finite_coupler_response(1.55e-6, periods=60, beam_radius=20e-6, **FINITE)
    assert finite.reflected == pytest.approx(float(plane.reflectance.sum()), rel=0.05)


def test_the_finite_teeth_reach_the_gap_and_keep_the_coupling_physical() -> None:
    """Three wavelengths through the coupler with ``finite_teeth``: a real, bounded passband."""
    wavelengths = np.array([1.545e-6, 1.550e-6, 1.555e-6])
    common = {**GAP, "length": 10e-6, "etch_depth": 70e-9}
    finite_s = grating_coupler_stack(C_LIGHT / wavelengths, finite_teeth=True, **common)
    infinite_s = grating_coupler_stack(C_LIGHT / wavelengths, **common)
    finite = np.abs(finite_s.s[:, 1, 0]) ** 2
    infinite = np.abs(infinite_s.s[:, 1, 0]) ** 2
    assert np.all(finite > 0.0) and np.all(finite <= 1.0)
    assert not np.allclose(finite, infinite, rtol=0, atol=1e-4), "the finite teeth were used"


def test_finite_teeth_need_an_etch_and_a_sensible_cell() -> None:
    with pytest.raises(ValueError, match="etch_depth"):
        grating_coupler_stack(GAP_BAND, finite_teeth=True, **GAP)
    with pytest.raises(ValueError, match="tooth"):
        finite_coupler_response(1.55e-6, periods=0, beam_radius=3e-6, **FINITE)
    with pytest.raises(ValueError, match="duty"):
        finite_coupler_response(1.55e-6, periods=4, beam_radius=3e-6, **{**FINITE, "duty": 1.0})
