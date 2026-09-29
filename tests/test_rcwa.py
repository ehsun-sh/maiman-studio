"""RCWA of a lamellar grating on a layered stack: checked against what shares no algebra with it.

Four exact limits -- a stack with no grating is the multilayer this library already
folds by Fresnel coefficients, a grating of zero depth or duty is that stack, power
is conserved when nothing absorbs, and a grating far finer than the wavelength is
the uniform layer of its averaged permittivity -- and the two ways it can be
wrong that they would not show: too few orders, and the wrong sign of a phase.
"""

from __future__ import annotations

import math

import pytest

from maiman.photonics import stack_reflection
from maiman.rcwa import GratingLayer, UniformLayer, diffract_te

WAVELENGTH = 1.55e-6
SINE = math.sin(math.radians(8.0))
SILICON, OXIDE = 3.476, 1.444


def teeth(depth: float = 70e-9, duty: float = 0.5) -> list[GratingLayer | UniformLayer]:
    return [
        GratingLayer(depth, SILICON, 1.0, duty),
        UniformLayer(220e-9 - depth, SILICON),
        UniformLayer(2e-6, OXIDE),
    ]


@pytest.mark.parametrize("sine", [0.0, SINE, 0.5])
@pytest.mark.parametrize("substrate", [3.476, 1.0, 0.5 + 16j])
def test_a_stack_with_no_grating_is_the_multilayer_the_module_already_folds(
    sine: float, substrate: complex
) -> None:
    """The same interfaces by Fresnel coefficients, to rounding -- sign and phase reference too."""
    layers = [UniformLayer(220e-9, SILICON), UniformLayer(2e-6, OXIDE)]
    got = diffract_te(
        WAVELENGTH,
        sine=sine,
        period=630e-9,
        top_index=1.0,
        layers=layers,
        substrate_index=substrate,
        harmonics=3,
    ).reflection
    expected = stack_reflection(
        WAVELENGTH,
        emission_sine=sine,
        top_index=1.0,
        silicon_index=SILICON,
        silicon_thickness=220e-9,
        box_index=OXIDE,
        box_thickness=2e-6,
        substrate_index=substrate,
    )
    assert got == pytest.approx(complex(expected), abs=1e-12)


@pytest.mark.parametrize(
    "layer",
    [
        GratingLayer(70e-9, SILICON, 1.0, 0.0),
        GratingLayer(70e-9, SILICON, 1.0, 1.0),
        GratingLayer(0.0, SILICON, 1.0, 0.5),
        GratingLayer(70e-9, SILICON, SILICON, 0.4),
    ],
)
def test_a_grating_with_nothing_to_diffract_is_its_uniform_layer(layer: GratingLayer) -> None:
    """Zero duty, full duty, zero depth, no index step: each is a plain slab."""
    index = layer.groove_index if layer.duty == 0.0 else layer.ridge_index
    with_layer = diffract_te(
        WAVELENGTH,
        sine=SINE,
        period=630e-9,
        top_index=1.0,
        layers=[layer, UniformLayer(150e-9, SILICON), UniformLayer(2e-6, OXIDE)],
        substrate_index=SILICON,
        harmonics=8,
    )
    if layer.thickness == 0.0:
        plain: list[GratingLayer | UniformLayer] = [UniformLayer(150e-9, SILICON)]
    else:
        plain = [UniformLayer(layer.thickness, index), UniformLayer(150e-9, SILICON)]
    plain.append(UniformLayer(2e-6, OXIDE))
    reference = diffract_te(
        WAVELENGTH,
        sine=SINE,
        period=630e-9,
        top_index=1.0,
        layers=plain,
        substrate_index=SILICON,
        harmonics=8,
    )
    assert with_layer.reflection == pytest.approx(reference.reflection, abs=1e-10)


@pytest.mark.parametrize("sine", [0.0, SINE, 0.3])
@pytest.mark.parametrize("duty", [0.3, 0.55])
def test_a_lossless_grating_keeps_all_the_power(sine: float, duty: float) -> None:
    """Reflected plus transmitted orders sum to one, to rounding: no order is lost or invented."""
    result = diffract_te(
        WAVELENGTH,
        sine=sine,
        period=630e-9,
        top_index=1.0,
        layers=teeth(duty=duty),
        substrate_index=SILICON,
        harmonics=20,
    )
    assert result.absorbed == pytest.approx(0.0, abs=1e-10)
    assert result.reflectance.sum() + result.transmittance.sum() == pytest.approx(1.0, abs=1e-10)


def test_an_absorbing_layer_keeps_what_the_stack_could_not_return() -> None:
    lossy = diffract_te(
        WAVELENGTH,
        sine=SINE,
        period=630e-9,
        top_index=1.0,
        layers=[
            GratingLayer(70e-9, SILICON + 0.2j, 1.0, 0.5),
            UniformLayer(150e-9, SILICON + 0.2j),
            UniformLayer(2e-6, OXIDE),
        ],
        substrate_index=SILICON,
        harmonics=20,
    )
    assert 0.0 < lossy.absorbed < 1.0
    assert lossy.reflectance.sum() + lossy.transmittance.sum() + lossy.absorbed == pytest.approx(
        1.0, abs=1e-10
    )


def test_under_an_absorbing_substrate_everything_that_crosses_is_taken() -> None:
    result = diffract_te(
        WAVELENGTH,
        sine=SINE,
        period=630e-9,
        top_index=1.0,
        layers=teeth(),
        substrate_index=0.5 + 16j,
        harmonics=20,
    )
    assert result.absorbed == pytest.approx(0.0, abs=1e-10)
    assert result.reflectance.sum() < 1.0
    assert result.transmittance.sum() == pytest.approx(1.0 - result.reflectance.sum(), abs=1e-10)


def test_the_answer_converges_as_orders_are_added() -> None:
    """Ten orders either side are good to 2e-4 of the amplitude, twenty to 3e-5 (70 nm etch)."""
    reference = diffract_te(
        WAVELENGTH,
        sine=SINE,
        period=630e-9,
        top_index=1.0,
        layers=teeth(),
        substrate_index=SILICON,
        harmonics=40,
    ).reflection
    errors = [
        abs(
            diffract_te(
                WAVELENGTH,
                sine=SINE,
                period=630e-9,
                top_index=1.0,
                layers=teeth(),
                substrate_index=SILICON,
                harmonics=m,
            ).reflection
            - reference
        )
        for m in (5, 10, 20)
    ]
    assert errors[0] > errors[1] > errors[2]
    assert errors[1] < 2e-4
    assert errors[2] < 3e-5


def test_a_grating_finer_than_the_wavelength_is_its_average_permittivity() -> None:
    """The zeroth-order effective medium, ``f eps_ridge + (1 - f) eps_groove`` for TE.

    Right to ``(period / lambda)^2``.

    Rytov's limit, an independent closed form: the gap to it falls by four each
    time the period halves (3.7, 3.9 measured), and is 1e-3 at 25 nm.
    """
    duty, depth = 0.45, 100e-9
    effective = math.sqrt(duty * SILICON**2 + (1.0 - duty) * 1.0**2)
    slab = diffract_te(
        WAVELENGTH,
        sine=SINE,
        period=1e-6,
        top_index=1.0,
        layers=[
            UniformLayer(depth, effective),
            UniformLayer(120e-9, SILICON),
            UniformLayer(2e-6, OXIDE),
        ],
        substrate_index=SILICON,
        harmonics=2,
    ).reflection
    gaps = []
    for period in (100e-9, 50e-9, 25e-9):
        fine = diffract_te(
            WAVELENGTH,
            sine=SINE,
            period=period,
            top_index=1.0,
            layers=[
                GratingLayer(depth, SILICON, 1.0, duty),
                UniformLayer(120e-9, SILICON),
                UniformLayer(2e-6, OXIDE),
            ],
            substrate_index=SILICON,
            harmonics=12,
        ).reflection
        gaps.append(abs(fine - slab))
    assert 3.4 < gaps[0] / gaps[1] < 4.3
    assert 3.4 < gaps[1] / gaps[2] < 4.3
    assert gaps[2] < 2e-3


def test_a_centred_ridge_reflects_the_same_from_either_side() -> None:
    """The grating is mirror-symmetric, so ``+theta`` and ``-theta`` give one zeroth order."""
    up = diffract_te(
        WAVELENGTH,
        sine=SINE,
        period=630e-9,
        top_index=1.0,
        layers=teeth(),
        substrate_index=SILICON,
        harmonics=15,
    ).reflection
    down = diffract_te(
        WAVELENGTH,
        sine=-SINE,
        period=630e-9,
        top_index=1.0,
        layers=teeth(),
        substrate_index=SILICON,
        harmonics=15,
    ).reflection
    assert up == pytest.approx(down, abs=1e-10)


def test_it_refuses_what_it_cannot_solve() -> None:
    kwargs = {
        "period": 630e-9,
        "top_index": 1.0,
        "layers": teeth(),
        "substrate_index": SILICON,
    }
    with pytest.raises(ValueError, match="sine"):
        diffract_te(WAVELENGTH, sine=1.0, **kwargs)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="lossless"):
        diffract_te(WAVELENGTH, sine=0.1, **{**kwargs, "top_index": 1.0 + 0.1j})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="duty"):
        diffract_te(
            WAVELENGTH,
            sine=0.1,
            **{**kwargs, "layers": [GratingLayer(70e-9, SILICON, 1.0, 1.5)]},  # type: ignore[arg-type]
        )
