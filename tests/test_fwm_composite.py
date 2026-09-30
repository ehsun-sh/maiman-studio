"""The composite split-step: every band, modulation and all, on one grid (maiman-edf).

``composite_fwm`` packs the bands onto a grid wide enough for the comb and its
products, runs the scalar split-step on the sum -- so mixing, cross-phase,
depletion, walk-off and each band's own dispersion are in the propagation -- and cuts
each band and product back out into its own window. The tone solver takes constants
exactly and the perturbative series takes modulation only through mean powers;
this is the one that takes both, at the cost of the comb's sampling rate.

What is checked, against things that share none of its packing: the tone solver
for constants, the one-band split-step across several spans, the linear
propagation of a modulated band alone, and -- for modulation -- the product a
dispersion-free fibre makes, which is ``-i gamma L E_a^2 E_b*`` in the time domain
at low power.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from maiman.components import Fiber
from maiman.kernels import (
    apply_pmd,
    differential_group_delay,
    pmd_jones_matrix,
    pmd_sections_from,
    propagate_dispersion,
)
from maiman.signals import Band, OpticalSignal
from test_fwm_tones import (
    ANCHOR,
    CTX,
    GAMMA,
    SPACING,
    jones_pumps,
    pmd_chain,
    split_step,
    two_pumps,
)


def link(
    spans: int, pump: float, dispersion: float, span: float, **settings: object
) -> tuple[dict[int, complex], OpticalSignal]:
    signal = two_pumps(pump)
    for index in range(spans):
        signal = Fiber(
            length=span,
            attenuation=0.0,
            dispersion=dispersion,
            nonlinearity=GAMMA,
            mixing_floor=250.0,
            label=f"span{index}",
            **settings,  # type: ignore[arg-type]
        ).run(CTX, {"in": signal})["out"]
    assert isinstance(signal, OpticalSignal)
    found = {}
    for m in (1, 2):
        band = [b for b in signal.bands if abs(b.f0 - (ANCHOR - m * SPACING)) < 1e3]
        found[m] = complex(np.mean(band[0].Ex)) if band else 0j
    return found, signal


@pytest.mark.parametrize("pump", [1e-3, 20e-3])
def test_constant_bands_land_on_the_tone_solver(pump: float) -> None:
    """The first product of two pumps: the composite grid and the tone solver agree to 5e-4.

    They share nothing -- one is a split-step on a wide grid, the other the coupled
    equations of the tones -- and at 20 mW the perturbative series is 25 % off.
    """
    composite, _ = link(1, pump, 2.0, 10.0, composite_fwm=True)
    tones, _ = link(1, pump, 2.0, 10.0, tone_solver=True)
    assert abs(composite[1]) == pytest.approx(abs(tones[1]), rel=1e-3)


@pytest.mark.parametrize(
    ("spans", "pump", "dispersion", "span"),
    [(1, 10e-3, 2.0, 10.0), (2, 10e-3, 2.0, 10.0), (4, 20e-3, 4.0, 5.0)],
)
def test_several_spans_land_on_the_one_band_split_step(
    spans: int, pump: float, dispersion: float, span: float
) -> None:
    """First and second product against the one-band split-step, over spans, to 2e-3.

    ``composite_order = 2`` gives the grid room for the second. Each span starts
    from bands stored without their carrier phase, so the composite puts back the
    curvature the path has accumulated before it propagates and takes it out after --
    without that the second span's mixing starts in the wrong phase, as the series'
    drawn phase did. Four 5 mW-per-tone spans at 20 mW with D = 4 were 2.3 times
    off in the series.
    """
    reference = split_step(pump, dispersion, span * 1e3, spans, 0.0, 0.0)
    composite, _ = link(spans, pump, dispersion, span, composite_fwm=True, composite_order=2.0)
    for m in (1, 2):
        assert abs(composite[m]) == pytest.approx(abs(reference[m]), rel=2e-3)


def modulated(power: float, f0: float, generator: np.random.Generator, bandwidth: float) -> Band:
    """A band-limited random envelope of a given mean power, ``bandwidth`` either side of zero."""
    n, rate = 256, 160e9
    noise = generator.normal(size=n) + 1j * generator.normal(size=n)
    spectrum = np.fft.fft(noise)
    spectrum[np.abs(np.fft.fftfreq(n, 1.0 / rate)) > bandwidth] = 0.0
    field = np.fft.ifft(spectrum)
    field *= math.sqrt(power / float(np.mean(np.abs(field) ** 2)))
    return Band(Ex=field.astype(np.complex128), Ey=np.zeros(n, dtype=np.complex128), f0=f0, fs=rate)


def test_a_dispersion_free_fibre_makes_the_time_domain_product_of_the_modulation() -> None:
    """``E_p(t) = -i gamma L E_b(t)^2 E_a(t)*``: the modulated product, sample for sample.

    No dispersion, so nothing walks and nothing mismatches, and at 0.5 mW per tone
    over 10 km ``gamma P L`` is 0.007. The library turns a field by ``exp(-i angle)``,
    hence the sign. The fitted coefficient is one in size to 5e-4, and what does not
    fit -- the product's own cross-phase, a few hundredths of a radian, and the
    third-order terms -- is under 1 % of it.
    """
    generator = np.random.default_rng(5)
    lower = modulated(0.5e-3, ANCHOR, generator, 8e9)
    upper = modulated(0.5e-3, ANCHOR + SPACING, generator, 8e9)
    length = 10.0
    out = Fiber(
        length=length,
        attenuation=0.0,
        dispersion=0.0,
        nonlinearity=GAMMA,
        mixing_floor=250.0,
        composite_fwm=True,
        label="span",
    ).run(CTX, {"in": OpticalSignal(bands=(lower, upper))})["out"]
    assert isinstance(out, OpticalSignal)
    product = next(b for b in out.bands if abs(b.f0 - (ANCHOR - SPACING)) < 1e3)
    expected = -1j * GAMMA * 1e-3 * length * 1e3 * lower.Ex**2 * np.conj(upper.Ex)
    coefficient = np.vdot(expected, product.Ex) / np.vdot(expected, expected)
    assert abs(coefficient) == pytest.approx(1.0, abs=5e-3)
    residue = np.linalg.norm(product.Ex - coefficient * expected) / np.linalg.norm(expected)
    assert residue < 0.02
    assert np.std(np.abs(product.Ex)) > 0.3 * np.mean(np.abs(product.Ex)), "and it is modulated"


def test_a_product_wider_than_its_slot_keeps_its_outer_skirt() -> None:
    """25 GHz of modulation makes a product 75 GHz either side, past its 50 GHz slot.

    The outermost product has no neighbour below it, so its slot reaches down to the
    band's window, and what lies more than 50 GHz below its carrier -- about 2 % of
    its power, which a slot of half the spacing dropped -- is the time-domain
    ``-i gamma L E_b^2 E_a*`` there, to 1 %. At 0.05 mW per tone, where that product
    is all there is: at ten times the power its own cross-phase moves the thin skirt
    by 5 %. The skirt on the other side is the channel's crosstalk.
    """
    generator = np.random.default_rng(5)
    lower = modulated(0.05e-3, ANCHOR, generator, 25e9)
    upper = modulated(0.05e-3, ANCHOR + SPACING, generator, 25e9)
    length = 10.0
    out = Fiber(
        length=length,
        attenuation=0.0,
        dispersion=0.0,
        nonlinearity=GAMMA,
        mixing_floor=250.0,
        composite_fwm=True,
        label="span",
    ).run(CTX, {"in": OpticalSignal(bands=(lower, upper))})["out"]
    assert isinstance(out, OpticalSignal)
    product = next(b for b in out.bands if abs(b.f0 - (ANCHOR - SPACING)) < 1e3)
    expected = -1j * GAMMA * 1e-3 * length * 1e3 * lower.Ex**2 * np.conj(upper.Ex)
    skirt = np.fft.fftfreq(product.Ex.size, 1.0 / product.fs) < -0.5 * SPACING
    wanted = np.fft.fft(expected)[skirt]
    got = np.fft.fft(product.Ex)[skirt]
    assert np.linalg.norm(wanted) ** 2 > 0.01 * np.linalg.norm(np.fft.fft(expected)) ** 2
    assert np.linalg.norm(got - wanted) < 0.01 * np.linalg.norm(wanted)


def test_without_a_nonlinearity_each_band_is_its_own_dispersed_self() -> None:
    """Linear limit: every band comes out as ``propagate_dispersion`` gives it alone, to 1e-9.

    In its own retarded frame with the carrier's constant phase taken out, which is
    what the library's bands are -- the composite's walk-off and curvature are gone
    again after the propagation.
    """
    generator = np.random.default_rng(6)
    bands = (
        modulated(1e-3, ANCHOR, generator, 8e9),
        modulated(1e-3, ANCHOR + SPACING, generator, 8e9),
    )
    signal = OpticalSignal(bands=bands)
    out = Fiber(
        length=40.0,
        attenuation=0.0,
        dispersion=17.0,
        nonlinearity=1e-12,
        mixing_floor=0.0,
        composite_fwm=True,
        label="span",
    ).run(CTX, {"in": signal})["out"]
    assert isinstance(out, OpticalSignal)
    beta2 = Fiber(dispersion=17.0).reference_beta2(signal)
    for source, result in zip(bands, out.bands, strict=False):
        wanted = propagate_dispersion(source.Ex, source.fs, beta2, 40e3)
        assert np.linalg.norm(result.Ex - wanted) < 1e-8 * np.linalg.norm(wanted)


def test_a_lossless_modulated_comb_keeps_its_power() -> None:
    generator = np.random.default_rng(7)
    bands = tuple(modulated(5e-3, ANCHOR + index * SPACING, generator, 8e9) for index in range(3))
    out = Fiber(
        length=20.0,
        attenuation=0.0,
        dispersion=4.0,
        nonlinearity=GAMMA,
        mixing_floor=250.0,
        composite_fwm=True,
        label="span",
    ).run(CTX, {"in": OpticalSignal(bands=bands)})["out"]
    assert isinstance(out, OpticalSignal)
    launched = sum(float(np.mean(np.abs(b.Ex) ** 2)) for b in bands)
    total = sum(float(np.mean(np.abs(b.Ex) ** 2 + np.abs(b.Ey) ** 2)) for b in out.bands)
    assert total == pytest.approx(launched, rel=5e-4)
    assert len(out.bands) > 3, "and products were made"


def test_carriers_off_the_windows_bins_are_refused_with_the_reason() -> None:
    signal = two_pumps(1e-3)
    second = signal.bands[1]
    shifted = Band(Ex=second.Ex, Ey=second.Ey, f0=second.f0 + 0.3e9, fs=second.fs)
    off = OpticalSignal(bands=(signal.bands[0], shifted))
    with pytest.raises(ValueError, match="frequency bins"):
        Fiber(length=1.0, nonlinearity=GAMMA, dispersion=2.0, composite_fwm=True, label="span").run(
            CTX, {"in": off}
        )


@pytest.mark.parametrize(
    "other",
    [
        {"tone_solver": True},
        {"pump_phase": True},
        {"cascaded_fwm": True},
        {"mixing_steps": 4.0},
        {"raman_gain_slope": 0.028},
    ],
)
def test_it_is_refused_beside_what_it_replaces(other: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="composite_fwm"):
        Fiber(composite_fwm=True, **other).validate()  # type: ignore[arg-type]
    Fiber(composite_fwm=True).validate()


# -- two polarizations and PMD on the grid -------------------------------------------

JONES = (math.cos(0.4), 1j * math.sin(0.4))


@pytest.mark.parametrize("coherent", [True, False])
@pytest.mark.parametrize("pump", [1e-3, 20e-3])
def test_two_polarizations_land_on_the_vector_tone_solver(coherent: bool, pump: float) -> None:
    """Elliptical pumps, the first product's Jones vector, composite against tone solver: 2e-3.

    The coupled split-step's phase-only form, or with ``coherent_polarization`` its
    circular-basis step, against the isotropic tensor written out over the tones.
    """

    def product(**settings: object) -> tuple[complex, complex]:
        out = Fiber(
            length=10.0,
            attenuation=0.0,
            dispersion=2.0,
            nonlinearity=GAMMA,
            mixing_floor=250.0,
            cross_polarization=True,
            coherent_polarization=coherent,
            label="span",
            **settings,  # type: ignore[arg-type]
        ).run(CTX, {"in": jones_pumps(pump, JONES)})["out"]
        assert isinstance(out, OpticalSignal)
        band = next(b for b in out.bands if abs(b.f0 - (ANCHOR - SPACING)) < 1e3)
        return complex(np.mean(band.Ex)), complex(np.mean(band.Ey))

    grid = product(composite_fwm=True)
    tones = product(tone_solver=True)
    for axis in (0, 1):
        assert abs(grid[axis]) == pytest.approx(abs(tones[axis]), rel=2e-3)


def pmd_span(nonlinearity: float = 1e-12, floor: float = 0.0, **settings: object) -> Fiber:
    return Fiber(
        label="pmd-span",
        length=20.0,
        attenuation=0.0,
        dispersion=0.0,
        nonlinearity=nonlinearity,
        mixing_floor=floor,
        composite_fwm=True,
        cross_polarization=True,
        pmd_coefficient=0.5,
        pmd_sections=12.0,
        **settings,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize("interleave", [False, True])
def test_one_band_meets_the_chain_as_apply_pmd_gives_it(interleave: bool) -> None:
    """A band at the grid's centre through the chain: ``apply_pmd``, to rounding.

    With nothing nonlinear the chain is the same whether it sits after the Kerr effect
    or between pieces of the span, and the same as the kernel gives the band alone.
    """
    band = jones_pumps(1e-6, JONES).bands[0]
    out = pmd_span(interleave_pmd=interleave).run(CTX, {"in": OpticalSignal(bands=(band,))})["out"]
    assert isinstance(out, OpticalSignal)
    ex, ey = apply_pmd(band.Ex, band.Ey, band.fs, pmd_chain("pmd-span", 0.5, 20.0, 12))
    assert np.linalg.norm(out.bands[0].Ex - ex) < 1e-9 * np.linalg.norm(ex)
    assert np.linalg.norm(out.bands[0].Ey - ey) < 1e-9 * np.linalg.norm(ey)


def test_two_carriers_meet_the_chain_at_their_own_frequencies() -> None:
    """Unlike the tone solver's, a chain on the grid rotates each carrier by what it does there.

    The chain is measured from the first band's carrier, so that band comes out as
    ``apply_pmd`` gives it alone, to rounding. The second, 100 GHz on, meets the
    chain 100 GHz on: a 2 ps chain turns it by half a radian more, and it differs
    from the rest-frame rotation by tenths.
    """
    signal = jones_pumps(1e-6, JONES)
    out = pmd_span().run(CTX, {"in": signal})["out"]
    assert isinstance(out, OpticalSignal)
    chain = pmd_chain("pmd-span", 0.5, 20.0, 12)
    gaps = []
    for source, result in zip(signal.bands, out.bands, strict=False):
        ex, _ = apply_pmd(source.Ex, source.Ey, source.fs, chain)
        gaps.append(np.linalg.norm(result.Ex - ex) / np.linalg.norm(ex))
    assert gaps[0] < 1e-9
    assert gaps[1] > 0.05


def test_a_band_meets_the_chain_whatever_else_is_on_the_grid() -> None:
    """Faint bands beside the first move the grid's centre and must not turn the first.

    Measured from the grid's centre, one to three bands at a millionth of the field
    turned the first band's state of polarization by 46 to 83 %.
    """
    first, second = jones_pumps(1e-6, JONES).bands
    ex, ey = apply_pmd(first.Ex, first.Ey, first.fs, pmd_chain("pmd-span", 0.5, 20.0, 12))
    for count in (1, 3):
        beside = tuple(
            Band(Ex=1e-6 * second.Ex, Ey=1e-6 * second.Ey, f0=first.f0 + k * SPACING, fs=first.fs)
            for k in range(1, count + 1)
        )
        out = pmd_span().run(CTX, {"in": OpticalSignal(bands=(first, *beside))})["out"]
        assert isinstance(out, OpticalSignal)
        assert np.linalg.norm(out.bands[0].Ex - ex) < 1e-9 * np.linalg.norm(ex)
        assert np.linalg.norm(out.bands[0].Ey - ey) < 1e-9 * np.linalg.norm(ey)


def test_a_chain_measured_from_elsewhere_is_the_chain_shifted() -> None:
    """The chain shifted by ``w0`` is, at ``w``, the chain itself at ``w + w0``; same DGD."""
    chain = pmd_chain("pmd-span", 0.5, 20.0, 12)
    w0 = 2.0 * math.pi * 137e9
    shifted = pmd_sections_from(chain, w0)
    for w in (0.0, 2.0 * math.pi * -40e9, 2.0 * math.pi * 5e9):
        assert np.allclose(
            pmd_jones_matrix(shifted, w), pmd_jones_matrix(chain, w + w0), atol=1e-12
        )
    assert differential_group_delay(shifted) == pytest.approx(
        differential_group_delay(chain), rel=1e-9
    )


def test_the_composite_reports_the_dgd_it_drew_and_keeps_the_power() -> None:
    generator = np.random.default_rng(9)
    bands = (
        modulated(5e-3, ANCHOR, generator, 8e9),
        modulated(5e-3, ANCHOR + SPACING, generator, 8e9),
    )
    span = pmd_span(GAMMA, 250.0, interleave_pmd=True)
    result = span.run(CTX, {"in": OpticalSignal(bands=bands)})
    diagnostics = result["diagnostics"]
    assert diagnostics.differential_group_delay == pytest.approx(
        differential_group_delay(pmd_chain("pmd-span", 0.5, 20.0, 12)), rel=1e-12
    )
    out = result["out"]
    assert isinstance(out, OpticalSignal)
    launched = sum(float(np.mean(np.abs(b.Ex) ** 2)) for b in bands)
    total = sum(float(np.mean(np.abs(b.Ex) ** 2 + np.abs(b.Ey) ** 2)) for b in out.bands)
    assert total == pytest.approx(launched, rel=1e-3)


def test_pmd_needs_the_two_polarizations_together_on_the_grid() -> None:
    with pytest.raises(ValueError, match="cross_polarization"):
        Fiber(composite_fwm=True, pmd_coefficient=0.1).validate()
    Fiber(composite_fwm=True, pmd_coefficient=0.1, cross_polarization=True).validate()
