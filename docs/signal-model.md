---
title: "The signal model"
description: "Why an optical signal is a set of sampled bands plus noise bins, and what that buys."
---

# The signal model

The part of the engine most worth reviewing, and the most expensive to change. An optical signal
here is not one array of numbers, and the reasons it is not are the reasons the simulator can model
a WDM system at all. The full reasoning is in §3 of the [architecture document](ARCHITECTURE.md).

## The run owns the time window

Bit rate, oversampling, sequence length and the random seed live in a shared `SimulationContext`,
not in individual signals. Every sampled signal in a run shares the same number of samples, the
same sample rate and the same time origin:

```python
ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=16, sequence_length=64)
ctx.sample_rate    # 160000000000.0   Fs = bit_rate × samples_per_symbol
ctx.num_samples    # 1024             N = sequence_length × samples_per_symbol
ctx.time_window    # 6.4e-09          T = N / Fs
```

If each signal carried its own, two blocks could silently disagree about the window, and the
simulation would be meaningless while still producing plausible numbers. With one context, they
cannot, and results are reproducible from the seed.

## A signal is a set of bands

```python
@dataclass(frozen=True)
class Band:
    """One sampled band: complex envelope in two orthogonal polarizations (Jones vector)."""
    Ex: np.ndarray   # complex, shape (N,), read-only
    Ey: np.ndarray
    f0: float        # band centre frequency [Hz]
    fs: float        # band sample rate [Hz]

@dataclass(frozen=True)
class OpticalSignal:
    bands: tuple[Band, ...]
    noise: tuple[NoiseBin, ...]
    accumulated_gvd: float   # sum(beta2 * L) over the path so far [s^2]
```

A single scalar carrier frequency can represent exactly one carrier. A 40-channel DWDM system
spanning 4 THz cannot be sampled as one band without a sample rate around 8 THz, which no machine
can afford. So each carrier is its own band, sampled only across its own bandwidth and carrying its
own centre frequency, and channel spacing never enters the sample rate.

That is why dispersion is applied per band, at each band's own wavelength; why a filter tuned
between two channels attenuates both, rather than choosing an array index; and why two lasers
6 THz apart cost exactly what two lasers 125 GHz apart do. The engine enforces three rules on
bands: they are disjoint in frequency, a block that would make them overlap merges them onto a
common grid or says so, and a block that cannot handle several bands declares it rather than
processing the first and discarding the rest.

Fields are in units of √W, so instantaneous power is `|Ex|² + |Ey|²`, and the two components are
the Jones vector: polarization is carried from the laser to the detector.

## Noise is carried in bins

```python
@dataclass(frozen=True)
class NoiseBin:
    """Spectrally-resolved noise, carried separately from the sampled bands."""
    f_start: float
    f_end: float
    psd_x: float   # [W/Hz] per polarization
    psd_y: float
```

An amplifier's ASE spans terahertz; the signal occupies tens of gigahertz of it. Sampling both
together would demand the same impossible sample rate. So ASE is carried as a power spectral
density across a frequency range, and converted to samples only where a detector or a nonlinearity
needs it, and only over the bandwidth being sampled.

A bin can also carry a *shape*: noise that passed through a ring, an interferometer or a grating
acquires the device's response, and a density read at one frequency then comes out right. That is
what lets a filter act as an ASE gate, and what makes signal-ASE beat noise at the photodiode, and
so the Q-factor that follows from an OSNR, correct.

## The one piece of path state

`accumulated_gvd` is the only thing the signal remembers about where it has been. Four-wave mixing
products generated in different spans must add as fields, and how far they have drifted apart is
the phase mismatch integrated over the distance travelled, which depends on β₂ alone. Carrying it,
rather than each band's absolute phase, is not a convenience: over 80 km, β₀L is of order 10¹¹
radians, and reducing that modulo 2π in double precision would leave about five digits of the
answer.

## Signals are immutable

A 100 km WDM link with 2²⁰ samples per band is hundreds of megabytes, and value-copying between
blocks would make any simulator unusable, in any language. So a block receives read-only inputs
and returns new outputs; the containers are tuples and the arrays are marked non-writeable. Blocks
that only change metadata (an ideal attenuator, a frequency shift) share the underlying buffers,
and only blocks that genuinely transform samples allocate.

That is enforced by the type rather than documented, because the buffer sharing it enables would
be unsafe otherwise. It is also why fan-out is a copy and not a split.

## Precision

Sampled fields default to single precision, `complex64`, because double precision doubles memory
traffic and FFT cost for accuracy system-level simulation rarely needs. Accuracy studies can pass
`precision="double"` to the context; the validation suite runs in both.

## The other signal types

Electrical, binary, symbol, soft and metric signals are simpler: a waveform, bits, complex symbols,
log-likelihood ratios, or a measurement object such as a `PowerReading`. They share the same
context, and the same rule that nothing is mutated in place. See
[Typed ports](graphs.md#typed-ports).
