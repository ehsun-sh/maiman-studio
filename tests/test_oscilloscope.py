"""The optical oscilloscope: one pulse, unfolded, measured against closed forms.

A Gaussian has an exact answer for everything this block reports, so each
figure is held to it rather than to a number the block once printed.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from maiman import Graph, SimulationContext
from maiman.components import CWLaser, Fiber, GaussianPulse, Oscilloscope
from maiman.encoding import encode_results
from maiman.kernels import dispersion_to_beta2
from maiman.signals import ScopeTrace

#: 1024 samples over 400 ps, as the first-light pulse examples use.
CTX = SimulationContext(bit_rate=10e9, samples_per_symbol=256, sequence_length=4)
T0 = 10e-12
D_SMF = 17.0
BETA2 = dispersion_to_beta2(D_SMF * 1e-6, 1550e-9)


def trace(
    length_km: float = 0.0,
    *,
    chirp: float = 0.0,
    span: float = 0.0,
    centre: float = 0.0,
    points: float = 1024.0,
) -> ScopeTrace:
    graph = Graph(CTX)
    pulse = graph.add(GaussianPulse(peak_power=0.0, width=T0 * 1e12, chirp=chirp))
    meter = graph.add(Oscilloscope(span=span, centre=centre, points=points))
    if length_km:
        fibre = graph.add(Fiber(length=length_km, attenuation=0.0, dispersion=D_SMF))
        graph.connect(pulse, fibre["in"])
        graph.connect(fibre["out"], meter["in"])
    else:
        graph.connect(pulse, meter["in"])
    result = graph.run()[meter]
    assert isinstance(result, ScopeTrace)
    return result


def test_a_launched_gaussian_reads_its_own_definition() -> None:
    """Peak P0, FWHM 2 sqrt(ln 2) T0, rms T0/sqrt(2), energy sqrt(pi) P0 T0, centred."""
    launched = trace()
    assert launched.peak_power_w == pytest.approx(1e-3, rel=1e-6)
    assert launched.fwhm == pytest.approx(2.0 * math.sqrt(math.log(2.0)) * T0, rel=1e-3)
    assert launched.rms_width == pytest.approx(T0 / math.sqrt(2.0), rel=1e-6)
    assert launched.energy_j == pytest.approx(math.sqrt(math.pi) * 1e-3 * T0, rel=1e-6)
    assert launched.centroid == pytest.approx(0.0, abs=1e-15)


@pytest.mark.parametrize("length_km", [2.0, 5.0, 10.0])
def test_dispersion_broadens_it_by_the_closed_form(length_km: float) -> None:
    """T1/T0 = sqrt(1 + (z/L_D)^2), read off the scope at both the FWHM and the rms."""
    ld = T0**2 / abs(BETA2)
    factor = math.sqrt(1.0 + (length_km * 1e3 / ld) ** 2)
    launched, received = trace(), trace(length_km)
    assert received.fwhm / launched.fwhm == pytest.approx(factor, rel=2e-3)
    assert received.rms_width / launched.rms_width == pytest.approx(factor, rel=1e-3)
    assert received.peak_power_w == pytest.approx(1e-3 / factor, rel=2e-3)
    assert received.energy_j == pytest.approx(launched.energy_j, rel=1e-9)


def test_the_display_resolution_moves_nothing_it_measures() -> None:
    fine, coarse = trace(5.0, points=1024.0), trace(5.0, points=64.0)
    assert coarse.time.size == 64
    for name in ("peak_power_w", "fwhm", "rms_width", "centroid", "energy_j"):
        assert getattr(coarse, name) == getattr(fine, name), name
    # Averaging keeps the area: 64 bins of 16 samples, each the mean of its own.
    step = CTX.time_window / CTX.num_samples
    assert np.sum(coarse.power_w) * 16 * step == pytest.approx(fine.energy_j, rel=1e-9)


def test_the_chirp_trace_is_the_source_s_linear_ramp_with_its_sign() -> None:
    """A Gaussian of chirp C sweeps C T / (2 pi T0^2): an up-chirp rises with time."""
    for chirp in (2.0, -2.0):
        launched = trace(chirp=chirp)
        lit = np.isfinite(launched.chirp_hz) & (np.abs(launched.time) < 1.5 * T0)
        slope = np.polyfit(launched.time[lit], launched.chirp_hz[lit], 1)[0]
        assert slope == pytest.approx(chirp / (2.0 * math.pi * T0**2), rel=1e-2)


def test_the_tails_are_too_dark_to_have_a_frequency() -> None:
    launched = trace()
    dark = launched.power_w < 1e-3 * 1e-3
    assert dark.any() and np.isnan(launched.chirp_hz[dark]).all()


def test_span_and_centre_window_the_trace() -> None:
    windowed = trace(span=40.0, centre=10.0, points=16.0)
    assert windowed.time.min() >= -10e-12 and windowed.time.max() <= 30e-12
    # The peak is in the window, the left half of the pulse partly is not.
    assert windowed.peak_power_w == pytest.approx(1e-3, rel=1e-6)
    assert windowed.centroid > 0.0
    with pytest.raises(ValueError, match="holds no samples"):
        trace(span=1.0, centre=500.0)


def test_a_pulse_cut_by_the_window_has_no_fwhm_rather_than_a_wrong_one() -> None:
    assert math.isnan(trace(span=10.0).fwhm)


def test_a_cw_carrier_is_flat_and_has_no_fwhm() -> None:
    graph = Graph(CTX)
    laser = graph.add(CWLaser(power=0.0))
    meter = graph.add(Oscilloscope(points=32.0))
    graph.connect(laser, meter["in"])
    flat = graph.run()[meter]
    assert isinstance(flat, ScopeTrace)
    # The fields are single precision; 1e-6 is their resolution, not the model's.
    np.testing.assert_allclose(flat.power_w, 1e-3, rtol=1e-6)
    assert math.isnan(flat.fwhm)


def test_it_encodes_as_json_the_page_can_draw() -> None:
    graph = Graph(CTX)
    pulse = graph.add(GaussianPulse(peak_power=0.0, width=10.0, label="pulse"))
    graph.connect(pulse, graph.add(Oscilloscope(points=128.0, label="scope"))["in"])
    encoded = encode_results(graph.run())["scope"]["out"]
    json.dumps(encoded, allow_nan=False)
    assert encoded["kind"] == "scope"
    assert len(encoded["time_ps"]) == len(encoded["power_mw"]) == len(encoded["chirp_ghz"]) == 128
    assert encoded["peak_power_mw"] == pytest.approx(1.0, rel=1e-6)
    assert encoded["fwhm_ps"] == pytest.approx(16.65, abs=0.01)
    assert None in encoded["chirp_ghz"], "the dark tails travel as null, not as NaN"
