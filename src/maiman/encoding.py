"""Turning run results into JSON the interface can draw.

**Reduction happens here, in the engine, and never in the browser.** A run holds
waveforms of tens of thousands of samples per port; an eye diagram drawn from
them is a 96x96 histogram. Sending the samples and binning them in JavaScript
would move megabytes per run, put the binning where it cannot be tested, and
make the browser's answer a second implementation of the engine's. So a
signal-carrying port encodes to a *summary* — how many samples, at what rate,
how much power — and anything meant to be looked at arrives already reduced,
produced by a measurement component that the graph contains explicitly.

That rule is not a size optimisation, it is what keeps the interface honest:
every number the screen shows was computed by the engine and can be reproduced
by calling the same Python.

**Discriminated by ``kind``.** Every encoded value carries one, so the client
switches on a string rather than guessing from which fields happen to be
present.

**Non-finite numbers are nulled.** ``json.dumps`` emits bare ``NaN`` and
``Infinity``, which are not JSON and which ``JSON.parse`` rejects outright — one
infinite Q factor would fail the whole response rather than one field. Results
legitimately reach infinity (a Q of zero gives ``-inf`` decibels, an error-free
window gives an infinite estimated BER margin), so this is a normal path, not an
error path.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from .components.dsp import DispersionDiagnostics
from .kernels import PropagationDiagnostics
from .signals import (
    BinarySignal,
    ConstellationHistogram,
    ConstellationMeasurement,
    ElectricalSignal,
    EyeHistogram,
    EyeMeasurement,
    OpticalSignal,
    OpticalSpectrum,
    PowerReading,
    SymbolSignal,
)

if TYPE_CHECKING:
    from .graph import Results

#: Most numbers any single encoded value may carry. Display data is already
#: reduced by the component that produced it — the widest is an optical spectrum
#: at its maximum 16384 points, twice, for frequency and power — so this is a
#: guard against a component handing back something unreduced, not a budget
#: anyone should be spending. Exceeding it raises rather than truncating,
#: because a silently shortened trace is a plot that lies.
MAX_ENCODED_NUMBERS = 100_000


class EncodingError(TypeError):
    """A value could not be reduced to something the interface can draw."""


def number(value: float) -> float | None:
    """A JSON-safe number: non-finite becomes ``None``.

    See the module docstring — infinities are a normal result here, and bare
    ``Infinity`` in a response body is not JSON at all.
    """
    result = float(value)
    return result if math.isfinite(result) else None


def _numbers(values: np.ndarray) -> list[float | None]:
    array = np.asarray(values, dtype=np.float64)
    if array.size > MAX_ENCODED_NUMBERS:
        raise EncodingError(
            f"refusing to encode {array.size} numbers; the limit is "
            f"{MAX_ENCODED_NUMBERS}. Display data is reduced by the component "
            f"that produces it, so an array this large means a raw buffer "
            f"reached the encoder."
        )
    return [number(v) for v in array.tolist()]


def _grid(values: np.ndarray) -> list[list[float | None]]:
    array = np.asarray(values, dtype=np.float64)
    if array.size > MAX_ENCODED_NUMBERS:
        raise EncodingError(
            f"refusing to encode a {array.shape} grid of {array.size} numbers; "
            f"the limit is {MAX_ENCODED_NUMBERS}"
        )
    return [[number(v) for v in row] for row in array.tolist()]


def _points(values: np.ndarray) -> list[list[float | None]]:
    """Complex values as ``[real, imag]`` pairs, which is what a plot wants."""
    array = np.asarray(values, dtype=np.complex128)
    if array.size * 2 > MAX_ENCODED_NUMBERS:
        raise EncodingError(f"refusing to encode {array.size} complex points")
    return [[number(z.real), number(z.imag)] for z in array.tolist()]


# ---------------------------------------------------------------------------
# Signals: summarised, never shipped whole
# ---------------------------------------------------------------------------


def _electrical(signal: ElectricalSignal) -> dict[str, Any]:
    samples = np.asarray(signal.samples, dtype=np.float64)
    return {
        "kind": "electrical",
        "unit": signal.unit,
        "samples": int(samples.size),
        "sample_rate": number(signal.fs),
        "min": number(samples.min()) if samples.size else None,
        "max": number(samples.max()) if samples.size else None,
        "mean": number(samples.mean()) if samples.size else None,
        "rms": number(np.sqrt(np.mean(samples**2))) if samples.size else None,
    }


def _optical(signal: OpticalSignal) -> dict[str, Any]:
    return {
        "kind": "optical",
        "signal_power_w": number(signal.signal_power()),
        "noise_power_w": number(signal.noise_power()),
        "bands": [
            {
                "f0": number(band.f0),
                "wavelength_nm": number(band.wavelength * 1e9),
                "sample_rate": number(band.fs),
                "samples": band.num_samples,
                "power_w": number(band.average_power()),
            }
            for band in signal.bands
        ],
        "noise": [
            {
                "f_start": number(bin_.f_start),
                "f_end": number(bin_.f_end),
                "psd_x": number(bin_.psd_x),
                "psd_y": number(bin_.psd_y),
            }
            for bin_ in signal.noise
        ],
    }


def _binary(signal: BinarySignal) -> dict[str, Any]:
    bits = np.asarray(signal.bits)
    return {
        "kind": "binary",
        "bits": int(bits.size),
        "ones": int(bits.sum()),
        "symbol_rate": number(signal.symbol_rate),
    }


def _symbol(signal: SymbolSignal) -> dict[str, Any]:
    # The alphabet is at most 256 points and is what labels a constellation
    # plot; the symbols themselves are not sent — that is what a
    # ConstellationDiagram in the graph is for.
    return {
        "kind": "symbol",
        "symbols": signal.num_symbols,
        "symbol_rate": number(signal.symbol_rate),
        "bits_per_symbol": signal.bits_per_symbol,
        "order": signal.order,
        "constellation": _points(signal.constellation),
    }


# ---------------------------------------------------------------------------
# Measurements: already reduced, sent whole
# ---------------------------------------------------------------------------


def _power(reading: PowerReading) -> dict[str, Any]:
    return {
        "kind": "power",
        "signal_power_w": number(reading.signal_power_w),
        "noise_power_w": number(reading.noise_power_w),
        "power_w": number(reading.power_w),
        "power_dbm": number(reading.power_dbm),
        "bands": [
            {
                "f0": number(band.f0),
                "wavelength_nm": number(band.wavelength_nm),
                "power_w": number(band.power_w),
                "power_dbm": number(band.power_dbm),
            }
            for band in reading.bands
        ],
    }


def _spectrum(spectrum: OpticalSpectrum) -> dict[str, Any]:
    return {
        "kind": "spectrum",
        "frequencies": _numbers(spectrum.frequencies),
        "wavelengths_nm": _numbers(spectrum.wavelengths_nm),
        "power_w": _numbers(spectrum.power_w),
        "power_per_resolution_w": _numbers(spectrum.power_per_resolution()),
        "resolution_bandwidth": number(spectrum.resolution_bandwidth),
    }


def _eye_histogram(eye: EyeHistogram) -> dict[str, Any]:
    return {
        "kind": "eye",
        "counts": _grid(eye.counts),
        "time_edges": _numbers(eye.time_edges),
        "amplitude_edges": _numbers(eye.amplitude_edges),
        "unit": eye.unit,
    }


def _eye_measurement(eye: EyeMeasurement) -> dict[str, Any]:
    return {
        "kind": "eye_measurement",
        "q_factor": number(eye.q_factor),
        "q_db": number(eye.q_db),
        "ber_gaussian": number(eye.ber_gaussian),
        "ber_counted": number(eye.ber_counted),
        "mean_one": number(eye.mean_one),
        "mean_zero": number(eye.mean_zero),
        "std_one": number(eye.std_one),
        "std_zero": number(eye.std_zero),
        "threshold": number(eye.threshold),
        "sample_offset": eye.sample_offset,
        "bits_evaluated": eye.bits_evaluated,
        "errors": eye.errors,
    }


def _constellation_histogram(diagram: ConstellationHistogram) -> dict[str, Any]:
    return {
        "kind": "constellation",
        "counts": _grid(diagram.counts),
        "inphase_edges": _numbers(diagram.inphase_edges),
        "quadrature_edges": _numbers(diagram.quadrature_edges),
        "reference": _points(diagram.reference),
    }


def _constellation_measurement(measurement: ConstellationMeasurement) -> dict[str, Any]:
    return {
        "kind": "constellation_measurement",
        "evm": number(measurement.evm),
        "snr_db": number(measurement.snr_db),
        "mer_db": number(measurement.mer_db),
        "ser_counted": number(measurement.ser_counted),
        "ber_counted": number(measurement.ber_counted),
        "ber_estimated": number(measurement.ber_estimated),
        "symbols_evaluated": measurement.symbols_evaluated,
        "symbol_errors": measurement.symbol_errors,
        "bits_evaluated": measurement.bits_evaluated,
        "bit_errors": measurement.bit_errors,
        "frequency_offset": number(measurement.frequency_offset),
        "bits_per_symbol": measurement.bits_per_symbol,
    }


def _diagnostics(diagnostics: PropagationDiagnostics) -> dict[str, Any]:
    return {
        "kind": "propagation",
        "steps": diagnostics.steps,
        "distance": number(diagnostics.distance),
        "shortest_step": number(diagnostics.shortest_step),
        "longest_step": number(diagnostics.longest_step),
        "peak_nonlinear_phase": number(diagnostics.peak_nonlinear_phase),
        "differential_group_delay": number(diagnostics.differential_group_delay),
        "walkoff_span": number(diagnostics.walkoff_span),
        "peak_walkoff_slip": number(diagnostics.peak_walkoff_slip),
        "mixing_products": diagnostics.mixing_products,
    }


def _dispersion(diagnostics: DispersionDiagnostics) -> dict[str, Any]:
    return {
        "kind": "dispersion",
        "accumulated_dispersion": number(diagnostics.accumulated_dispersion),
        "removed_symbols": number(diagnostics.removed_symbols),
    }


_ENCODERS: dict[type, Any] = {
    DispersionDiagnostics: _dispersion,
    ElectricalSignal: _electrical,
    OpticalSignal: _optical,
    BinarySignal: _binary,
    SymbolSignal: _symbol,
    PowerReading: _power,
    OpticalSpectrum: _spectrum,
    EyeHistogram: _eye_histogram,
    EyeMeasurement: _eye_measurement,
    ConstellationHistogram: _constellation_histogram,
    ConstellationMeasurement: _constellation_measurement,
    PropagationDiagnostics: _diagnostics,
}


def register_encoder(result_type: type, encoder: Any) -> None:
    """Teach the interface to draw a result type this module does not know.

    Built-in encoders live here rather than beside the components that produce
    them, because turning a result into JSON is a presentation concern and a
    physics module has no business knowing about either. A component from
    another package cannot edit this file, though, so it registers instead —
    the same shape as the component registry itself, and for the same reason.

    The encoder is called with the value and must return a dictionary carrying a
    ``kind``. Without one, its results still arrive, tagged ``"opaque"``.
    """
    _ENCODERS[result_type] = encoder


def encode(value: object) -> dict[str, Any]:
    """Reduce one run result to a JSON-safe dictionary tagged with its ``kind``.

    Dispatch walks the method resolution order, so a subclass of a known result
    type is encoded as its base rather than falling through — a plugin that
    subclasses :class:`~maiman.signals.PowerReading` to add a field still draws.

    A type nothing knows how to reduce is not an error and is not guessed at.
    It comes back tagged ``"opaque"`` with its type name, which lets the
    interface say *this block produced something it cannot plot* instead of
    either failing the whole run or inventing a plausible-looking shape for it.
    """
    if isinstance(value, bool):  # bool is an int; check first or it encodes as one
        return {"kind": "scalar", "value": value}
    if isinstance(value, int | float | np.integer | np.floating):
        return {"kind": "scalar", "value": number(float(value))}

    for base in type(value).__mro__:
        encoder = _ENCODERS.get(base)
        if encoder is not None:
            return dict(encoder(value))

    return {
        "kind": "opaque",
        "type": type(value).__name__,
        "repr": repr(value)[:200],
    }


def encode_results(results: Results) -> dict[str, dict[str, Any]]:
    """Every retained port of a run, keyed by component label then port name."""
    encoded: dict[str, dict[str, Any]] = {}
    for (label, port), signal in results.items():
        encoded.setdefault(label, {})[port] = encode(signal)
    return encoded
