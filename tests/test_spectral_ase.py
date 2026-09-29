"""An amplifier's ASE shaped by the erbium spectrum, not flat.

:class:`~maiman.components.EDFA` itself still emits its ASE flat across its
band -- it knows no cross sections. :func:`maiman.spectral_ase_noise_bin` is the
offline reshaping of that same total power into the standard steady-state
spectrum an erbium coil actually produces, and :func:`maiman.spectral_n_sp` and
:func:`maiman.spectral_ase_psd` are the pieces it is built from. The checks here
are the closed forms the module's own docstring claims: a flat spectrum changes
nothing, a curved one matches the textbook formula slice by slice, the slices
sum to the bin's total, and the spontaneous-emission factor sits at its quantum
floor and at the high-gain noise-figure limit where it must.

Reference: R. J. Giles and E. Desurvire, "Modeling erbium-doped fiber
amplifiers", J. Lightwave Technol. 9(2), 271, 1991; E. Desurvire, *Erbium-Doped
Fiber Amplifiers: Principles and Applications*, Wiley, 1994, ch. 2.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from maiman import (
    ErbiumSpectrum,
    gain_transient,
    saturating_weights,
    self_saturation_weight,
    self_saturation_weight_spectral,
    spectral_ase_noise_bin,
    spectral_ase_psd,
    spectral_gain_transient,
    spectral_n_sp,
    step_schedule,
)
from maiman.components import EDFA
from maiman.units import C_LIGHT, H_PLANCK, db_to_linear

CROSSOVER = 1531e-9


def flat_spectrum(level_db: float = 20.0) -> ErbiumSpectrum:
    """A spectrum whose ``A + G*`` does not depend on wavelength at all."""
    nm = np.linspace(1500.0, 1600.0, 51)
    return ErbiumSpectrum(
        wavelengths=nm * 1e-9,
        absorption_db=np.full_like(nm, level_db),
        full_inversion_gain_db=np.full_like(nm, level_db),
    )


def curved_spectrum() -> ErbiumSpectrum:
    """A 1530 nm absorption peak on a broad shoulder, emission by McCumber -- same
    shape :mod:`tests.test_gain_tilt` uses, so the two suites agree on what an
    "illustrative" erbium coil looks like."""
    nm = np.linspace(1500.0, 1600.0, 401)
    absorption = 30.0 * np.exp(-((nm - 1530.0) ** 2) / (2 * 8.0**2)) + 15.0 * np.exp(
        -((nm - 1548.0) ** 2) / (2 * 18.0**2)
    )
    return ErbiumSpectrum.from_mccumber(nm * 1e-9, absorption, crossover_wavelength=CROSSOVER)


def amplifier(gain: float = 20.0, bandwidth: float = 2.0) -> EDFA:
    return EDFA(gain=gain, noise_figure=5.0, bandwidth=bandwidth, label="edfa")


# ---------------------------------------------------------------------------
# A flat spectrum changes nothing
# ---------------------------------------------------------------------------


def test_a_flat_spectrum_reproduces_todays_numbers_exactly() -> None:
    amp = amplifier()
    gain_linear = db_to_linear(amp.gain)
    bin_ = spectral_ase_noise_bin(amp, flat_spectrum(), gain_linear)
    flat_psd = amp.ase_psd(gain_linear)

    assert bin_.psd_x == pytest.approx(flat_psd, rel=0.0, abs=0.0)
    assert bin_.psd_y == pytest.approx(flat_psd, rel=0.0, abs=0.0)
    assert bin_.shape is not None
    assert np.all(bin_.shape.weight_x == pytest.approx(1.0, abs=1e-12))
    assert np.all(bin_.shape.weight_y == pytest.approx(1.0, abs=1e-12))

    # Every reading a flat, unshaped bin would give is unchanged: density at any
    # frequency in the band, and the total power.
    centre = C_LIGHT / amp.si("center_wavelength")
    for offset in (-0.7e12, -0.1e12, 0.0, 0.3e12, 0.9e12):
        psd_x, psd_y = bin_.density_at(centre + offset)
        assert psd_x == pytest.approx(flat_psd, rel=1e-12)
        assert psd_y == pytest.approx(flat_psd, rel=1e-12)
    assert bin_.total_power() == pytest.approx(2.0 * flat_psd * amp.si("bandwidth"), rel=1e-12)


# ---------------------------------------------------------------------------
# A curved spectrum: per-slice power matches the textbook formula
# ---------------------------------------------------------------------------


def test_per_slice_ase_matches_the_spontaneous_emission_formula() -> None:
    spectrum = curved_spectrum()
    inversion = 0.65
    wavelengths = np.array([1530e-9, 1540e-9, 1550e-9, 1560e-9, 1570e-9])

    gain_db = spectrum.gain_db(wavelengths, inversion)
    gain_linear = 10.0 ** (gain_db / 10.0)
    absorption, emission = spectrum._curves(wavelengths)
    # n_sp = sigma_e N2 / (sigma_e N2 - sigma_a N1), Giles parameters standing in
    # for the cross sections (Giles & Desurvire 1991, sec. II).
    expected_n_sp = emission * inversion / (inversion * (absorption + emission) - absorption)
    frequency = C_LIGHT / wavelengths
    expected_psd = expected_n_sp * H_PLANCK * frequency * (gain_linear - 1.0)

    n_sp = spectral_n_sp(spectrum, wavelengths, inversion)
    psd = spectral_ase_psd(spectrum, wavelengths, inversion)

    assert n_sp == pytest.approx(expected_n_sp, rel=1e-12)
    assert psd == pytest.approx(expected_psd, rel=1e-12)
    # And the shortest wavelength here, near the absorption peak, needs more
    # excess noise than the reference the module's docstring quotes.
    assert n_sp[0] > n_sp[2] > 1.0


def test_the_shape_tracks_the_formula_not_the_declared_noise_figure() -> None:
    """The bin's mean stays whatever ``ase_psd`` (the noise figure) already gives,
    but where in the band it sits follows the formula's own relative profile."""
    amp = amplifier(bandwidth=3.0)
    spectrum = curved_spectrum()
    gain_linear = db_to_linear(amp.gain)
    bin_ = spectral_ase_noise_bin(amp, spectrum, gain_linear)

    assert bin_.shape is not None
    inversion = float(spectrum.inversion(amp.si("center_wavelength"), amp.gain))
    centre = C_LIGHT / amp.si("center_wavelength")
    # The exact grid the shape was built on -- comparing at these points avoids
    # any interpolation mismatch between an independently chosen grid and the
    # one the mean-normalisation was computed over.
    frequencies = bin_.shape.offsets + centre
    wavelengths = C_LIGHT / frequencies
    # The shape's own relative form: n_sp(lambda) (G(lambda) - 1), without the
    # h*nu(lambda) factor -- see transient._spectral_ase_relative's docstring for
    # why the shape leaves that (much smaller) variation out.
    gain_linear_slice = 10.0 ** (spectrum.gain_db(wavelengths, inversion) / 10.0)
    n_sp = spectral_n_sp(spectrum, wavelengths, inversion)
    relative_formula = n_sp * np.maximum(gain_linear_slice - 1.0, 0.0)
    relative = relative_formula / np.mean(relative_formula)

    assert bin_.shape.weight_x == pytest.approx(relative, rel=1e-9)
    assert bin_.shape.weight_y == pytest.approx(relative, rel=1e-9)

    # And at a handful of interior frequencies, reading the bin the way a
    # detector or an OSNR meter would (interpolated, through density_at) still
    # lands on the noise-figure total times this same relative profile.
    sample = frequencies[::40][:-1]  # skip the last point: density_at is [f_start, f_end)
    density = np.array([bin_.density_at(f)[0] for f in sample])
    on_sample = np.interp(sample, frequencies, relative)
    assert density / amp.ase_psd(gain_linear) == pytest.approx(on_sample, rel=1e-6)
    # The mean is exactly the declared-noise-figure total, not the formula's raw
    # absolute scale (which need not agree with a datasheet NF chosen freely).
    assert bin_.psd_x == pytest.approx(amp.ase_psd(gain_linear), rel=0.0, abs=0.0)


# ---------------------------------------------------------------------------
# Slices sum to the total
# ---------------------------------------------------------------------------


def test_total_ase_equals_the_sum_over_slices() -> None:
    amp = amplifier(bandwidth=4.0)
    spectrum = curved_spectrum()
    gain_linear = db_to_linear(amp.gain)
    bin_ = spectral_ase_noise_bin(amp, spectrum, gain_linear)

    centre = C_LIGHT / amp.si("center_wavelength")
    bandwidth = amp.si("bandwidth")
    edges = np.linspace(centre - bandwidth / 2.0, centre + bandwidth / 2.0, 33)
    summed_x = sum(bin_.power_between(lo, hi)[0] for lo, hi in itertools.pairwise(edges))
    summed_y = sum(bin_.power_between(lo, hi)[1] for lo, hi in itertools.pairwise(edges))

    # power_between's own numerical integration (a piecewise-midpoint average of
    # the interpolated shape, per slice) is not exact, so the tolerance here is
    # its accuracy, not the identity being tested -- which the finer check below
    # nails down directly.
    total_x, total_y = bin_.total_power() / 2.0, bin_.total_power() / 2.0
    assert summed_x == pytest.approx(total_x, rel=1e-3)
    assert summed_y == pytest.approx(total_y, rel=1e-3)
    assert summed_x + summed_y == pytest.approx(bin_.total_power(), rel=1e-3)

    # And directly, with no per-slice numerical integration in the way at all:
    # the shape is normalised to mean one by construction, so its own mean over
    # the band times the bandwidth is exactly the total the bin was given.
    assert bin_.shape is not None
    assert float(np.mean(bin_.shape.weight_x)) == pytest.approx(1.0, rel=1e-9)


# ---------------------------------------------------------------------------
# The quantum floor and the high-gain noise-figure limit
# ---------------------------------------------------------------------------


def test_n_sp_is_floored_at_the_quantum_limit() -> None:
    """Full inversion -- n = 1 -- puts every wavelength at its quantum floor,
    n_sp = 1, the 3 dB noise figure nothing can beat (Desurvire 1994, ch. 2)."""
    spectrum = curved_spectrum()
    wavelengths = np.array([1530e-9, 1540e-9, 1550e-9, 1560e-9])
    n_sp = spectral_n_sp(spectrum, wavelengths, 1.0)
    assert n_sp == pytest.approx(1.0, rel=1e-9)

    # And below any inversion where the coil still shows net gain at that
    # wavelength, the floor still holds -- it is never allowed to read quieter
    # than the quantum limit for any inversion at all.
    for inversion in (0.3, 0.5, 0.7, 0.9):
        assert np.all(spectral_n_sp(spectrum, wavelengths, inversion) >= 1.0 - 1e-12)


def test_the_noise_figure_implied_by_n_sp_reaches_2_n_sp_at_high_gain() -> None:
    """NF = 2 n_sp (1 - 1/G) -- the standard relation this project's own EDFA
    docstring states -- so at a wavelength held at ever higher gain (inversion
    climbing toward full inversion, n = 1), the implied noise figure closes in
    on ``2 n_sp``."""
    spectrum = curved_spectrum()
    wavelength = 1550e-9
    inversions = np.array([0.5, 0.8, 0.95, 0.999])
    gain_db = spectrum.gain_db(wavelength, inversions)
    gain_linear = 10.0 ** (gain_db / 10.0)
    n_sp = spectral_n_sp(spectrum, wavelength, inversions)
    implied_nf = 2.0 * n_sp * (1.0 - 1.0 / gain_linear)
    limit = 2.0 * n_sp

    gap = limit - implied_nf
    # The gap shrinks monotonically as gain climbs, and is small at the last,
    # highest-gain point.
    assert np.all(np.diff(gap) < 0.0)
    assert gap[-1] < 0.02 * limit[-1]


# ---------------------------------------------------------------------------
# What is refused
# ---------------------------------------------------------------------------


def test_a_shape_needs_gain_above_unity() -> None:
    from maiman import spectral_ase_shape

    with pytest.raises(ValueError, match="above unity"):
        spectral_ase_shape(amplifier(), curved_spectrum(), 1.0)


def test_a_shape_needs_positive_bandwidth() -> None:
    from maiman import spectral_ase_shape

    amp = amplifier(bandwidth=0.0)
    with pytest.raises(ValueError, match="positive bandwidth"):
        spectral_ase_shape(amp, curved_spectrum(), db_to_linear(amp.gain))


# ---------------------------------------------------------------------------
# The reservoir's own drain, made consistent with what is now emitted
# ---------------------------------------------------------------------------

#: 1500 to 1600 nm, matching tests/test_ase_saturation.py's own-ASE band grid.
GRID = np.linspace(1500e-9, 1600e-9, 201)


def own_amplifier(gain: float = 30.0) -> EDFA:
    return EDFA(gain=gain, saturate=True, saturation_power=17.0, self_saturation=True, label="e")


def test_a_flat_spectrum_the_two_weights_agree_exactly() -> None:
    """No wavelength dependence in n_sp or the gain either, so the emitted shape
    is one everywhere and this is :func:`self_saturation_weight` to rounding."""
    amp = own_amplifier()
    flat = ErbiumSpectrum(GRID, np.full(GRID.shape, 30.0), np.full(GRID.shape, 40.0))
    assert self_saturation_weight_spectral(amp, flat) == pytest.approx(
        self_saturation_weight(amp, flat), rel=1e-14
    )


def test_the_spectral_weight_matches_a_direct_trapezoid_of_the_formula() -> None:
    """Cross section times what each slice emits, ``n_sp (G - 1)``, the emitted
    share normalised to a mean of one -- the drain follows the emission, on a
    grid a hundred times finer than the function's."""
    amp = own_amplifier(gain=20.0)  # this spectrum reaches 24 dB at 1550 nm
    spectrum = curved_spectrum()
    inversion = float(spectrum.inversion(amp.si("center_wavelength"), amp.gain))
    centre = C_LIGHT / amp.si("center_wavelength")
    bandwidth = amp.si("bandwidth")
    frequencies = np.linspace(centre - bandwidth / 2.0, centre + bandwidth / 2.0, 20001)
    wavelengths = C_LIGHT / frequencies
    centre_wavelength = amp.si("center_wavelength")
    cross_section = saturating_weights(spectrum, wavelengths, centre_wavelength)
    n_sp = spectral_n_sp(spectrum, wavelengths, inversion)
    local_gain = 10.0 ** (spectrum.gain_db(wavelengths, inversion) / 10.0)
    emitted = n_sp * np.maximum(local_gain - 1.0, 0.0)
    share = emitted / float(np.trapezoid(emitted, frequencies) / bandwidth)
    combined = cross_section * share
    expected = float(np.trapezoid(combined, frequencies) / bandwidth)

    assert self_saturation_weight_spectral(amp, spectrum) == pytest.approx(expected, rel=1e-4)


def test_a_curved_spectrum_the_two_weights_differ() -> None:
    """n_sp itself moves with the same curvature that moves the gain, so weighing
    by what is actually emitted is not the same as weighing by cross section
    alone once the spectrum is not flat."""
    amp = own_amplifier(gain=20.0)
    spectrum = curved_spectrum()
    cross_section_only = self_saturation_weight(amp, spectrum)
    emitted_shape = self_saturation_weight_spectral(amp, spectrum)
    assert emitted_shape != pytest.approx(cross_section_only, rel=1e-6)


def test_the_flag_selects_the_emitted_shape_in_the_transient() -> None:
    """``spectral_ase=True`` drains the reservoir by ``self_saturation_weight_spectral``
    instead of ``self_saturation_weight`` -- the rest point in the dark moves to
    match, exactly as :mod:`tests.test_ase_saturation` pins the unweighted case."""
    amp = own_amplifier(gain=20.0)
    spectrum = curved_spectrum()
    times, power = step_schedule([(20e-3, 0.0)], points_per_segment=200)

    cross_section_run = spectral_gain_transient(
        amp, spectrum, times, power, [1550e-9], spectral_ase=False
    )
    emitted_run = spectral_gain_transient(amp, spectrum, times, power, [1550e-9], spectral_ase=True)

    small_signal = db_to_linear(amp.gain)
    saturation = amp.intrinsic_saturation_power(small_signal)

    def rest_gain(weight: float) -> float:
        beta = weight * amp.self_saturation_load() / saturation
        w = math.log1p(beta * small_signal)
        for _ in range(100):
            e = math.exp(w)
            f = w * e - beta * small_signal
            step = f / (e * (w + 1.0) - (w + 2.0) * f / (2.0 * w + 2.0))
            w -= step
            if abs(step) < 1e-15 * max(1.0, abs(w)):
                break
        return w / beta

    expected_cross_section = rest_gain(self_saturation_weight(amp, spectrum))
    expected_emitted = rest_gain(self_saturation_weight_spectral(amp, spectrum))

    assert cross_section_run.reference.gain[-1] == pytest.approx(expected_cross_section, rel=1e-9)
    assert emitted_run.reference.gain[-1] == pytest.approx(expected_emitted, rel=1e-9)
    assert emitted_run.reference.gain[-1] != pytest.approx(
        cross_section_run.reference.gain[-1], rel=1e-6
    )


def test_without_a_spectrum_the_flag_has_nothing_to_reach() -> None:
    """``spectral_ase`` only changes anything when ``self_saturation`` has a load
    to weigh at all -- without it, both paths are the unweighted reservoir."""
    amp = EDFA(gain=20.0, saturate=True, saturation_power=17.0, self_saturation=False, label="e")
    spectrum = curved_spectrum()
    times, power = step_schedule([(5e-3, 1e-6)], points_per_segment=64)
    plain = gain_transient(amp, times, power)
    flagged = spectral_gain_transient(amp, spectrum, times, power, [1550e-9], spectral_ase=True)
    assert flagged.reference.gain == pytest.approx(plain.gain, rel=1e-12)


# ---------------------------------------------------------------------------
# Against the paper's own relations (maiman-66t)
# ---------------------------------------------------------------------------
#
# Giles and Desurvire's figures and tables are for particular fibres, whose
# absorption and gain spectra are curves in Fig. 2 and are not tabulated, so no
# number of theirs can be reproduced without those curves -- and this library
# ships none, since a curve invented to look plausible is a number nobody can
# back. What the paper does state in closed form can be held to: Sec. VIII's
# noise factor of a highly pumped amplifier, eq. (31), its ASE power, eq. (32),
# and the sentence that with no stimulated emission at the pump the quantum
# limit n_sp = 1 is reached. Those are written in cross sections; the curves
# stand in for them because ``sigma_e / sigma_a = G* / A`` (the coil's length
# and doping cancel in the ratio).


@pytest.mark.parametrize("pump_ratio", [0.0, 0.1, 0.28, 0.6])
def test_a_highly_pumped_amplifier_has_the_papers_excess_noise_factor(pump_ratio: float) -> None:
    """Eq. (31): ``n_sp = 1 / (1 - (sigma_a / sigma_e)(sigma_ep / sigma_ap))``, at every wavelength.

    Fully pumped, the inversion is ``n = sigma_ap / (sigma_ap + sigma_ep) =
    1 / (1 + r)`` for ``r = sigma_ep / sigma_ap`` the pump's emission-to-absorption
    ratio, and the noise factor at each signal wavelength is the paper's -- here
    against :func:`spectral_n_sp`, which is written from the populations instead, so
    the two agree only if the paper's chain does. ``r = 0`` is a 980 nm pump.
    """
    spectrum = curved_spectrum()
    grid = np.linspace(1535e-9, 1565e-9, 61)
    absorption, emission = spectrum._curves(grid)
    inversion = 1.0 / (1.0 + pump_ratio)
    got = spectral_n_sp(spectrum, grid, inversion)
    with np.errstate(divide="ignore", invalid="ignore"):
        paper = 1.0 / (1.0 - (absorption / emission) * pump_ratio)
    usable = (spectrum.gain_db(grid, inversion) > 0.0) & np.isfinite(paper) & (paper > 0.0)
    assert usable.any()
    assert np.allclose(got[usable], paper[usable], rtol=1e-12)


def test_with_no_stimulated_emission_at_the_pump_the_quantum_limit_is_reached() -> None:
    """The paper's 980 nm sentence: complete inversion, ``n_sp = 1`` at every signal wavelength."""
    spectrum = curved_spectrum()
    grid = np.linspace(1520e-9, 1565e-9, 46)
    assert np.allclose(spectral_n_sp(spectrum, grid, 1.0), 1.0, rtol=0, atol=1e-12)


def test_the_ase_power_is_the_papers_eq_32_in_a_bandwidth() -> None:
    """Eq. (32): ``P_ASE = 2 n_sp (G - 1) h nu d_nu`` per bandwidth, both polarizations.

    :func:`spectral_ase_psd` is the one-sided density per polarization, so twice it
    times the bandwidth is the paper's power; checked at a wavelength where the
    coil has gain.
    """
    spectrum = curved_spectrum()
    wavelength, inversion = 1550e-9, 0.8
    bandwidth = 125e9  # the paper's 1 nm resolution
    gain = float(db_to_linear(float(np.ravel(spectrum.gain_db(wavelength, inversion))[0])))
    n_sp = float(np.ravel(spectral_n_sp(spectrum, wavelength, inversion))[0])
    photon = H_PLANCK * C_LIGHT / wavelength
    paper = 2.0 * n_sp * (gain - 1.0) * photon * bandwidth
    density = float(np.ravel(spectral_ase_psd(spectrum, wavelength, inversion))[0])
    assert 2.0 * density * bandwidth == pytest.approx(paper, rel=1e-12)
