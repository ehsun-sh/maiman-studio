"""Noise that carries the shape of what it went through.

A noise bin used to be flat by construction. That is the truth for an amplifier
and stops being true the moment ASE goes through anything wavelength-selective:
a ring passes its teeth and not its gaps, a grating reflects one band and passes
the rest. A flat bin could carry only the average of that, and everything read
at one frequency -- an OSNR meter at a carrier, a detector beating against the
ASE beside its signal, an analyser drawing a reflection -- read the average
instead.

These check the shape against things it has to agree with: the grating's own
reflectivity, the sensor's own wavelength shift, conservation of power, and every
flat bin reading exactly as it did before shapes existed.
"""

from __future__ import annotations

import numpy as np
import pytest

from maiman import SimulationContext
from maiman.components import (
    FiberBraggGrating,
    OpticalSpectrumAnalyzer,
    RingResonator,
    Waveguide,
)
from maiman.signals import NoiseBin, NoiseShape, OpticalSignal
from maiman.units import C_LIGHT

CTX = SimulationContext(bit_rate=10e9, samples_per_symbol=8, sequence_length=64)
PSD = 1e-15


def broadband(low_nm: float = 1548.0, high_nm: float = 1552.0) -> OpticalSignal:
    """Flat ASE across a few nanometres, the way an amplifier emits it."""
    return OpticalSignal(
        noise=(
            NoiseBin(
                f_start=C_LIGHT / (high_nm * 1e-9),
                f_end=C_LIGHT / (low_nm * 1e-9),
                psd_x=PSD,
                psd_y=PSD,
            ),
        )
    )


def amplifier_band() -> OpticalSignal:
    """Four terahertz of flat ASE about 193.1 THz."""
    return OpticalSignal(noise=(NoiseBin(f_start=191.1e12, f_end=195.1e12, psd_x=PSD, psd_y=PSD),))


def analyse(signal: OpticalSignal) -> tuple[np.ndarray, np.ndarray]:
    """Wavelength [nm] and density [W/Hz] as an analyser draws them."""
    trace = OpticalSpectrumAnalyzer(points=8192).run(CTX, {"in": signal})["out"]
    frequencies = np.asarray(trace.frequencies)
    step = (frequencies[-1] - frequencies[0]) / (len(frequencies) - 1)
    return C_LIGHT / frequencies * 1e9, np.asarray(trace.power_w) / step


def ring() -> RingResonator:
    return RingResonator(length=100.0, coupling=0.05, drop_coupling=0.05)


# --------------------------------------------------------------------------
# The instrument that could not be built
# --------------------------------------------------------------------------


def test_broadband_light_off_a_grating_shows_the_grating_on_an_analyser() -> None:
    """A white-light source and an OSA, which a flat bin made impossible.

    With flat bins the reflection of ASE off a grating was one averaged number and
    the analyser drew a flat floor. Now the peak is where the grating is and as
    tall as the grating's own reflectivity says -- input density times
    ``tanh**2(kappa L)`` -- and a nanometre away it is gone.
    """
    grating = FiberBraggGrating(bragg_wavelength=1550.0, length=10.0, index_modulation=1e-4)
    reflected = grating.run(CTX, {"in": broadband()})["reflected"]
    wavelengths, density = analyse(reflected)

    peak = int(np.argmax(density))
    assert wavelengths[peak] == pytest.approx(1550.0, abs=1e-3)
    # Normalised before comparing. pytest.approx keeps a 1e-12 absolute floor even
    # when only rel is given, and every density here is thousands of times smaller
    # than that -- compared directly, this assertion would accept any value at all.
    expected = 2.0 * PSD * grating.peak_reflectivity()
    assert density[peak] / expected == pytest.approx(1.0, rel=1e-3)

    far = np.abs(wavelengths - 1550.0) > 1.0
    assert density[far].max() < 0.01 * density[peak]


def test_a_strained_grating_moves_its_drawn_reflection_by_the_sensor_shift() -> None:
    """Read off an analyser rather than a sweep, and it agrees with the sweep."""
    peaks = []
    for strain in (0.0, 1000.0):
        grating = FiberBraggGrating(
            bragg_wavelength=1550.0, length=10.0, index_modulation=1e-4, strain=strain
        )
        wavelengths, density = analyse(grating.run(CTX, {"in": broadband()})["reflected"])
        peaks.append(float(wavelengths[int(np.argmax(density))]))
    assert peaks[1] - peaks[0] == pytest.approx(1.209, abs=2e-3)


def test_a_lossless_grating_splits_broadband_light_without_losing_any() -> None:
    """Reflected plus transmitted is the input, and a shape must not disturb that."""
    source = broadband()
    grating = FiberBraggGrating(bragg_wavelength=1550.0, length=10.0, index_modulation=1e-4)
    out = grating.run(CTX, {"in": source})
    total = out["reflected"].noise_power() + out["transmitted"].noise_power()
    assert total == pytest.approx(source.noise_power(), rel=1e-8)


# --------------------------------------------------------------------------
# A ring's teeth and gaps
# --------------------------------------------------------------------------


def test_a_ring_s_noise_is_read_at_its_teeth_not_at_its_average() -> None:
    """A couple of percent of the ASE survives in total, and none of it evenly.

    A 100 um ring passes about ninety percent of the density at a resonance and a
    few hundredths of a percent between two, and ``noise_psd_at`` now says so. The
    flat average -- the only thing a flat bin could hold -- is still exactly the
    bin's ``psd_x``.
    """
    device = ring()
    dropped = device.run(CTX, {"in": amplifier_band()})["drop"]

    period = device.free_spectral_range()
    frequencies = np.linspace(193.1e12 - period, 193.1e12 + period, 2001)
    density = np.array([dropped.noise_psd_at(f)[0] for f in frequencies]) / PSD

    assert density.max() == pytest.approx(0.91, abs=0.03)
    assert density.min() < 1e-3
    assert dropped.noise[0].psd_x / PSD == pytest.approx(0.0245, abs=0.002)


def test_a_shape_is_normalised_to_the_mean_the_bin_already_carries() -> None:
    """So a shape changes where the power is and never how much there is.

    And a device whose magnitude does not vary gets no shape at all: a waveguide
    differs from flat only in the last digit of a complex exponential, and storing
    thousands of ones for it would change nothing but the memory.
    """
    device = ring()
    dropped = device.run(CTX, {"in": amplifier_band()})["drop"].noise[0]
    assert dropped.shape is not None
    assert float(np.mean(dropped.shape.weight_x)) == pytest.approx(1.0, rel=1e-12)
    assert dropped.shape.period == pytest.approx(device.free_spectral_range(), rel=1e-12)

    guided = Waveguide(length=1000.0).run(CTX, {"in": amplifier_band()})["out"].noise[0]
    assert guided.shape is None


def test_clipping_a_shaped_bin_keeps_its_density_at_every_frequency() -> None:
    """A filter after a ring narrows the ASE; it must not re-flatten what is left.

    The mean over a narrower range is not the mean over the whole bin, so the mean
    density changes and the density at each frequency must not.
    """
    shaped = ring().run(CTX, {"in": amplifier_band()})["drop"].noise[0]
    clipped = shaped.clip(193.0e12, 193.3e12)
    assert clipped is not None
    assert clipped.shape is not None
    assert abs(clipped.psd_x / shaped.psd_x - 1.0) > 1e-3
    for frequency in np.linspace(193.01e12, 193.29e12, 7):
        # abs=0: these densities are far below approx's default 1e-12 floor.
        assert clipped.density_at(frequency)[0] == pytest.approx(
            shaped.density_at(frequency)[0], rel=1e-9, abs=0.0
        )


# --------------------------------------------------------------------------
# Flat bins, which every existing number depends on
# --------------------------------------------------------------------------


def test_a_flat_bin_reads_exactly_as_it_did_before_shapes_existed() -> None:
    """Bit for bit: every figure a flat bin has ever produced depends on it."""
    bin_ = NoiseBin(f_start=193.0e12, f_end=193.1e12, psd_x=3e-16, psd_y=1e-16)
    assert bin_.shape is None
    assert bin_.density_at(193.05e12) == (3e-16, 1e-16)
    assert bin_.density_at(194.0e12) == (0.0, 0.0)

    width = 193.04e12 - 193.02e12
    assert bin_.power_between(193.02e12, 193.04e12) == (3e-16 * width, 1e-16 * width)
    assert bin_.squared_density_integral() == (
        3e-16**2 * bin_.bandwidth,
        1e-16**2 * bin_.bandwidth,
    )

    clipped = bin_.clip(193.02e12, 193.2e12, factor=0.5)
    assert clipped is not None
    assert (clipped.f_start, clipped.f_end, clipped.psd_x, clipped.psd_y) == (
        193.02e12,
        193.1e12,
        3e-16 * 0.5,
        1e-16 * 0.5,
    )
    assert bin_.clip(194.0e12, 195.0e12) is None

    signal = OpticalSignal(noise=(bin_,))
    assert signal.noise_power_in(193.05e12, 12.5e9) == (3e-16 + 1e-16) * 12.5e9


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"offsets": [0.0], "weight_x": [1.0], "weight_y": [1.0]}, "at least two offsets"),
        (
            {"offsets": [0.0, 1.0], "weight_x": [1.0], "weight_y": [1.0, 1.0]},
            "one weight per offset",
        ),
        (
            {"offsets": [1.0, 0.0], "weight_x": [1.0, 1.0], "weight_y": [1.0, 1.0]},
            "strictly increasing",
        ),
        (
            {"offsets": [0.0, 1.0], "weight_x": [-1.0, 1.0], "weight_y": [1.0, 1.0]},
            "cannot be negative",
        ),
        (
            {"offsets": [0.0, 1.0], "weight_x": [1.0, 1.0], "weight_y": [1.0, 1.0], "period": 0.0},
            "period must be positive",
        ),
    ],
)
def test_a_noise_shape_refuses_what_it_cannot_describe(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        NoiseShape(centre=193.1e12, **kwargs)
