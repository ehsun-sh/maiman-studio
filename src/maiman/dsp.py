"""Receiver DSP: dispersion compensation, pulse shaping, adaptive equalisation.

Two impairments are undone here, and they are undone separately because they are
different kinds of problem. **Chromatic dispersion is deterministic**: a known,
static, all-pass phase that one fixed filter inverts exactly, and it can be
hundreds of taps long. **Polarization mixing is not**: it is a random rotation
that drifts on a millisecond timescale, so it needs a short adaptive filter that
tracks. Trying to do both with one adaptive filter needs it to be both long and
fast, which is the worst of each. Every deployed coherent receiver splits them in
exactly this order, and so does this module.

Model references: S. J. Savory, "Digital filters for coherent optical receivers",
Optics Express 16(2), 2008 (both the static CD filter and the butterfly);
D. N. Godard, "Self-recovering equalization and carrier tracking", IEEE Trans.
Comm. 28(11), 1980 (the constant-modulus algorithm).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .kernels import dispersion_slope_to_beta3, dispersion_to_beta2, propagate_dispersion
from .units import C_LIGHT

#: Cross-coupling placed in the initial filter to break the 45-degree symmetry.
#: See :func:`butterfly_equalize` for why a symmetric start cannot converge there.
SYMMETRY_TILT = 0.1


#: Metre used to turn an *accumulated* dispersion into the (D, length) pair the
#: forward propagator takes. The product is what matters physically, so any split
#: gives the same answer; one metre keeps the numbers readable in a debugger.
_UNIT_LENGTH = 1.0


def compensate_dispersion(
    baseband: np.ndarray,
    sample_rate: float,
    *,
    accumulated_dispersion: float,
    wavelength: float,
    accumulated_slope: float = 0.0,
) -> np.ndarray:
    """Undo accumulated chromatic dispersion on a complex baseband.

    ``accumulated_dispersion`` is D·L in s/m — the *product*, because that is the
    only thing a receiver can know. It cannot see how the span was built, and two
    links with the same total behave identically, so asking for D and L
    separately would invent a distinction that does not exist on this side of the
    fibre. ``wavelength`` is needed because β₂ is D scaled by λ².

    ``accumulated_slope`` is S·L for the same path, in ps/nm², and is what the
    filter needs to cancel the *cubic* phase the slope leaves behind. It is
    separate from the quadratic term because a receiver measures the two
    separately — the slope is a property of the fibre type and is known from the
    route, while D·L is what acquisition searches for. Leaving it at zero
    compensates β₂ alone, which is what a receiver that does not know its fibre
    can do; over a thousand kilometres of standard fibre that residue is tens of
    radians across a 32 GBd band and is not small.

    Chromatic dispersion is a pure phase, ``exp(i β₂ ω² z / 2)``, with unit
    magnitude at every frequency. Nothing is lost and nothing is buried under
    noise — the received field still holds all of it, just rearranged in time —
    which is why a link that is completely closed at the photodiode reopens to
    within a fraction of a dB of back-to-back. That is not error correction. It is
    the inverse of an invertible operation.

    **This calls the fibre's own propagator with a negated distance rather than
    writing the inverse transfer function out.** The two would be equivalent only
    as long as nobody edits one of them, and the sign of β₂ is exactly the kind of
    thing that drifts apart across two copies. Structurally there is one filter
    here, run in reverse; a sign error in it can only be a sign error in the
    forward model, which the propagation tests already pin.

    The window is periodic, so the filter wraps — as the forward propagation
    wrapped. That is self-consistent within a run, and it is why compensation
    returns *exactly* the launched waveform rather than approximately. Over a real
    span the wrap is a genuine edge effect and the outer symbols are dropped, as
    ``ignore_edges`` on the analysers already does.
    """
    beta2 = dispersion_to_beta2(accumulated_dispersion / _UNIT_LENGTH, wavelength)
    beta3 = (
        dispersion_slope_to_beta3(
            accumulated_dispersion / _UNIT_LENGTH, accumulated_slope / _UNIT_LENGTH, wavelength
        )
        if accumulated_slope != 0.0
        else 0.0
    )
    return propagate_dispersion(baseband, sample_rate, beta2, -_UNIT_LENGTH, beta3)


def dispersive_spread(
    accumulated_dispersion: float, bandwidth: float, wavelength: float, symbol_rate: float
) -> float:
    """How many symbol periods the channel smears one symbol across.

    ``Δτ = D·L · Δλ``, with the occupied optical bandwidth converted to a
    wavelength span by ``Δλ = λ²·Δf/c``. Reported in symbols because that is the
    number that decides whether an adaptive equaliser can plausibly cope: a
    seven-tap butterfly covers three symbols either side, so anything past that
    has to be removed statically first.
    """
    wavelength_span = wavelength**2 * bandwidth / C_LIGHT
    return abs(accumulated_dispersion * wavelength_span) * symbol_rate


#: Fraction of the beat spectrum's energy the acquisition scan keeps.
#:
#: The clock tone is a beat between frequencies one symbol rate apart, so only
#: bins where the spectrum overlaps its own shifted copy carry any of it; the
#: rest hold receiver noise. Summing all 65536 instead of the ~3800 that matter
#: costs a factor of twelve in time and buys nothing: it changes the scan by
#: 0.5 % of its peak at worst, and on the grid acquisition actually uses — half a
#: lobe, 61 ps/nm at 32 GBd — it moved the chosen candidate by zero steps at
#: every span tested. Probed on a ten-times finer grid it moves it by one.
BEAT_ENERGY_KEPT = 0.999

#: Candidates evaluated per matrix in :func:`scan_clock_tone`.
SCAN_BLOCK = 64

#: Points per refinement window. Nine leaves four either side of the minimum, so
#: the parabola through the best three is always fitted to points that bracket it.
REFINE_POINTS = 9

#: How many windows refinement may *move* before it starts narrowing.
#:
#: A window whose minimum sits on its own edge has not bracketed anything, and
#: narrowing it there would converge on the edge instead of on the answer. Moving
#: it costs one window and buys three grid steps of reach — about 2900 ps/nm at
#: 32 GBd over this many moves — which is what carries the estimate in from an
#: acquisition that is biased rather than merely imprecise. At roll-off 0.05 the
#: clock tone lands a consistent 950 ps/nm low, and walking is the whole reason
#: that case still resolves.
WALK_LIMIT = 16


@dataclass(frozen=True)
class DispersionEstimate:
    """What a blind search found, and how much to believe it."""

    accumulated_dispersion: float
    """The estimate [s/m]."""

    acquired: float
    """Where acquisition put it, before refinement [s/m]."""

    contrast: float
    """Clock-tone peak over the median of the scan.

    Near 1 the scan found no peak at all. A high value is *not* a promise of
    accuracy, and the two should not be read as the same measurement: at
    roll-off 0.05 the scan produced a contrast of 29 — indistinguishable from
    the 32 of a run that was right — while landing 950 ps/nm low. What rescued
    that case was refinement walking in, not the contrast warning about it.
    """

    step: float
    """Spacing of the acquisition grid [s/m], which is what refinement started from."""

    def __repr__(self) -> str:
        return (
            f"DispersionEstimate({self.accumulated_dispersion * 1e3:.1f} ps/nm, "
            f"acquired {self.acquired * 1e3:.0f}, contrast {self.contrast:.1f})"
        )


def _symbol_rate_bin(num_samples: int, sample_rate: float, symbol_rate: float) -> int:
    """The FFT bin the symbol rate lands on, refusing a window where it lands between.

    Every window this engine produces has an integer number of samples per
    symbol, so the symbol rate is always exactly on a bin. Interpolating a line
    that fell between two of them would be a different estimator with a different
    error, so the case is refused rather than quietly approximated.
    """
    exact = symbol_rate / sample_rate * num_samples
    if abs(exact - round(exact)) > 1e-9:
        raise ValueError(
            f"the symbol rate {symbol_rate:g} Hz does not land on an FFT bin of a "
            f"{num_samples}-sample window sampled at {sample_rate:g} Hz"
        )
    return round(exact)


def clock_tone(baseband: np.ndarray, sample_rate: float, *, symbol_rate: float) -> complex:
    """The symbol-rate line in the spectrum of the received *intensity*.

    A modulated signal is cyclostationary: its intensity repeats at the symbol
    rate, so ``|A|²`` carries a line there. Two different things about that line
    are useful, and this returns both in the one complex number.

    Its **magnitude** measures chromatic dispersion. The line is a beat between
    spectral components one symbol rate apart; dispersion gives every such pair a
    relative phase that grows with frequency, so their contributions rotate away
    from each other and partially cancel. Undo the dispersion and they add in
    phase again, which is the maximum. That is what :func:`estimate_dispersion`
    searches for.

    Its **phase** is the symbol timing — where in the symbol period the intensity
    peaks — which is a free clock recovery for anything that wants one. Nothing
    here does: refinement was written to use it and measured better without it,
    for which see :func:`intensity_cost`.

    **The line only exists if the signal has excess bandwidth.** Components one
    symbol rate apart must both fall inside the occupied band, and for a signal
    shaped to exactly the Nyquist bandwidth there are none: a raised cosine of
    zero roll-off is not cyclostationary at the symbol rate at all. No amount of
    processing recovers a tone that is not there, and that is a property of the
    signal rather than a limit of this implementation.
    """
    power = np.abs(np.asarray(baseband)) ** 2
    index = _symbol_rate_bin(len(power), sample_rate, symbol_rate)
    return complex(np.fft.fft(power)[index])


def clock_tone_lobe(wavelength: float, symbol_rate: float) -> float:
    """Narrowest the clock tone's peak can be, in accumulated dispersion [s/m].

    The beat between ``ω`` and ``ω + ω_s`` picks up a phase ``θ·ω·ω_s`` under a
    residual ``θ = A·λ²/2πc``. Summed over an overlap band of width ``Δf`` the
    contributions cancel once that phase has walked through ``2π`` across the
    band, which puts the first null at ``A = 2πc/(λ²·Δf·ω_s)``. The overlap band
    cannot be wider than the symbol rate itself, so the tightest the peak ever
    gets is

        ``A = c / (λ² · R_s²)``

    — 122 ps/nm at 32 GBd, 1.25 ns/nm at 10 GBd. The *narrowest* case is the one
    worth knowing, because the acquisition grid has to land inside the peak
    whatever the transmitter's roll-off turns out to be. Measured against an
    unshaped 32 GBd signal, whose beat band is the whole of it, the peak was
    100 ps/nm wide between its half-power points.
    """
    return C_LIGHT / (wavelength**2 * symbol_rate**2)


def scan_clock_tone(
    baseband: np.ndarray,
    sample_rate: float,
    *,
    symbol_rate: float,
    wavelength: float,
    candidates: np.ndarray,
    energy_kept: float = BEAT_ENERGY_KEPT,
) -> np.ndarray:
    """``|clock_tone|`` after compensating each candidate — without compensating any.

    The obvious implementation runs :func:`compensate_dispersion` and then a
    transform for every candidate, and a search wide enough to cover a thousand
    kilometres at 32 GBd has several hundred of them. This computes the same
    numbers from one forward transform.

    The line at the symbol rate is a correlation of the spectrum with itself,
    shifted by that rate::

        P[k_s] = (1/N) Σ_m  Y[m] · conj(Y[m - k_s])

    and compensation multiplies ``Y`` by a phase, so the compensated product is
    the *measured* product times the phase difference between the two bins::

        P_A[k_s] = (1/N) Σ_m  X[m]·conj(X[m-k_s]) · exp(i·θ(A)·(ω_m² - ω_{m-k_s}²)/2)

    The measured product and the frequency lever are computed once; each
    candidate is then one complex exponential and a dot product, with no
    transform at all. At ``energy_kept=1`` this is an identity rather than an
    approximation, and the tests hold it to 1e-12 relative against
    compensate-then-transform.

    The default keeps only the bins carrying :data:`BEAT_ENERGY_KEPT` of the beat
    energy, which is where the twelve-fold speed-up over 400 candidates actually
    comes from. That truncation is *not* exact, and it is least exact where it
    matters least: near the peak the bins it drops move the answer by 0.4 %,
    while at a candidate far enough out for the tone to have cancelled they are
    most of what is left. The peak is what this is read for.
    """
    baseband = np.asarray(baseband)
    n = len(baseband)
    shift = _symbol_rate_bin(n, sample_rate, symbol_rate)

    omega = 2.0 * np.pi * np.fft.fftfreq(n, 1.0 / sample_rate)
    spectrum = np.fft.fft(baseband)
    # Rolled by +shift so entry m holds bin m - shift, wrapping as the
    # correlation does. The wrap lands where a band-limited signal has nothing.
    beat = spectrum * np.conj(np.roll(spectrum, shift))
    lever = (omega**2 - np.roll(omega, shift) ** 2) / 2.0

    energy = np.abs(beat) ** 2
    total = float(energy.sum())
    if energy_kept < 1.0 and total > 0.0:
        order = np.argsort(energy)[::-1]
        held = np.cumsum(energy[order]) / total
        keep = order[: int(np.searchsorted(held, energy_kept)) + 1]
        beat, lever = beat[keep], lever[keep]

    theta = np.asarray(candidates, dtype=float) * wavelength**2 / (2.0 * np.pi * C_LIGHT)
    strength = np.empty(len(theta))
    # Blocked so the candidate-by-bin matrix stays a few megabytes rather than
    # growing with the product of a long window and a wide search.
    for start in range(0, len(theta), SCAN_BLOCK):
        block = theta[start : start + SCAN_BLOCK]
        strength[start : start + SCAN_BLOCK] = np.abs(np.exp(1j * np.outer(block, lever)) @ beat)
    return strength / n


def intensity_cost(
    baseband: np.ndarray,
    sample_rate: float,
    *,
    wavelength: float,
    accumulated: float,
) -> float:
    """How Gaussian the received intensity looks once a candidate is compensated.

    ``E[p²]/E[p]²`` of ``p = |A|²``, which is 2 for a complex Gaussian field and
    moves away from it for anything with structure left in it. Dispersion sums
    many independent symbols into every sample and drives the field towards that
    Gaussian; removing it puts the structure back. The minimum over
    ``accumulated`` is the estimate.

    This is the refinement stage, and it exists because the clock tone alone is
    not sharp enough to be useful. Around the true value the tone is flat — 99 %
    of its peak across a 140 ps/nm span at 32 GBd — while 20 ps/nm of residual
    already costs 8 dB of SNR. This cost has curvature exactly where the tone has
    none.

    **It reads the whole waveform, not one sample per symbol.** That was not the
    first version. Refinement used to interpolate to the symbol instant carried
    in :func:`clock_tone`'s phase, on the reasoning that what is being measured
    is inter-symbol interference and that interference is a property of the
    decision instants. Deleting the decimation changed no test result, which is
    how the reasoning was found to be decoration: measured across fifteen links
    it is not merely unnecessary but slightly worse — worst error 44 ps/nm
    against 63 — and the whole waveform won every case where noise or a short
    record made the estimate hard, by 18 ps/nm against 35 through an amplifier
    and 9 against 15 over a 512-symbol window.
    """
    baseband = np.asarray(baseband)
    compensated = (
        compensate_dispersion(
            baseband, sample_rate, accumulated_dispersion=accumulated, wavelength=wavelength
        )
        if accumulated != 0.0
        else baseband
    )
    power = np.abs(compensated) ** 2
    mean = float(power.mean())
    if mean <= 0.0:
        return 2.0
    return float((power**2).mean() / mean**2)


def estimate_dispersion(
    baseband: np.ndarray,
    sample_rate: float,
    *,
    symbol_rate: float,
    wavelength: float,
    search_range: float,
    rounds: int = 5,
) -> DispersionEstimate:
    """Find the accumulated dispersion in a received baseband, blind.

    A deployed receiver is not told how long its fibre is. It measures the
    dispersion during acquisition, from the signal itself, before anything
    downstream has converged — and that is a large part of what makes a coherent
    receiver deployable rather than merely correct in a laboratory.

    Two stages, because no one statistic does both jobs. **Acquisition** scans
    :func:`scan_clock_tone` across the whole of ``±search_range`` on a grid of
    half a :func:`clock_tone_lobe`: enormous capture range, an error of tens of
    ps/nm. **Refinement** then minimises :func:`intensity_cost` on successively
    halved windows around it, re-centring rather than shrinking whenever the
    minimum lands on an edge, which is what lets it walk in from an acquisition
    several hundred ps/nm out.

    **What it does, measured.** Over a 32 GBd 16-QAM link with 4096 symbols in
    the window, spans of 0, 20, 80, 400 and 1000 km — 0 to 17 000 ps/nm — were
    each estimated to within 9 ps/nm, and the EVM under the estimate was 1.8 to
    2.4 % against 1.7 % under the exact value. It held at 2, 4 and 16 samples per
    symbol, at QPSK and 64-QAM, for negative dispersion, at roll-offs from 0.02
    to 0.35 and with no pulse shaping at all, with a 500 kHz laser, over a window
    of 512 symbols, and through an amplifier at 5 and 9 dB noise figure.

    **What limits it.** Resolution scales as ``1/R_s²``, because the phase the
    beat accumulates does: the same link at 10 GBd landed 44 ps/nm out where
    32 GBd landed 5. Noise costs accuracy too — 18 ps/nm through a 5 dB
    amplifier. And a link that is *itself* sharply sensitive to residual
    dispersion is not made safe by a good estimate: unshaped and with no matched
    filter downstream, 11 ps/nm of residual takes the EVM from 5.1 to 18.1 %,
    and that margin belongs to the link rather than to anything here.

    **What it needs.** Excess bandwidth, for the reason given under
    :func:`clock_tone`: at zero roll-off the tone this searches for does not
    exist. Between there and roll-off 0.1 it exists but is biased — at 0.05 and
    0.02 acquisition came in a consistent 950 ps/nm low, on both halves of the
    record and at both 80 and 400 km — and it is refinement walking its window in
    that recovers those, not the contrast figure, which flagged nothing.
    """
    lobe = clock_tone_lobe(wavelength, symbol_rate)
    step = lobe / 2.0
    candidates = np.arange(-search_range, search_range + step / 2.0, step)
    strength = scan_clock_tone(
        baseband,
        sample_rate,
        symbol_rate=symbol_rate,
        wavelength=wavelength,
        candidates=candidates,
    )
    acquired = float(candidates[int(strength.argmax())])
    median = float(np.median(strength))
    contrast = float(strength.max() / median) if median > 0.0 else float("inf")

    centre, span = acquired, 3.0 * step
    walks = narrowings = 0
    while walks <= WALK_LIMIT and narrowings < max(rounds, 0):
        grid = np.linspace(centre - span, centre + span, REFINE_POINTS)
        cost = np.array(
            [
                intensity_cost(
                    baseband,
                    sample_rate,
                    wavelength=wavelength,
                    accumulated=float(value),
                )
                for value in grid
            ]
        )
        best = int(cost.argmin())
        spacing = float(grid[1] - grid[0])
        if best in (0, len(grid) - 1):
            # The minimum is outside the window, so nothing is bracketed yet:
            # move the window rather than narrow it, and do not spend a round.
            centre = float(grid[best])
            walks += 1
            continue

        low, middle, high = float(cost[best - 1]), float(cost[best]), float(cost[best + 1])
        curvature = low - 2.0 * middle + high
        centre = float(grid[best])
        if curvature > 0.0:
            centre += 0.5 * spacing * (low - high) / curvature
        narrowings += 1
        if span <= 1.5 * spacing:
            break
        span = 2.0 * spacing

    return DispersionEstimate(
        accumulated_dispersion=centre, acquired=acquired, contrast=contrast, step=step
    )


def root_raised_cosine(roll_off: float, span_symbols: int, samples_per_symbol: int) -> np.ndarray:
    """Root-raised-cosine impulse response, normalised to unit energy.

    A rectangular symbol has a sinc spectrum, which never ends: it is fine for a
    single channel in isolation and useless the moment neighbours are packed onto
    a grid beside it. A raised-cosine spectrum stops dead at ``(1 + roll_off)``
    times the symbol rate, which is what lets 32 GBd sit on a 50 GHz grid.

    The *root* is used because the shaping is split between the two ends. Cascade
    a transmitter's root with a receiver's matched root and the result is a full
    raised cosine, which is the shape that satisfies the Nyquist criterion: zero
    at every symbol instant but its own, so neighbouring symbols contribute
    nothing to a decision. Splitting it this way is also what makes the receiver
    filter *matched* to the transmitted pulse, and therefore optimal against
    additive noise. One end alone gives neither property.

    ``roll_off = 0`` degenerates to a sinc: the narrowest possible spectrum, and
    an impulse response that decays so slowly that any timing error is punished
    severely. Real systems use 0.05 to 0.3 for exactly that reason.
    """
    if not 0.0 <= roll_off <= 1.0:
        raise ValueError(f"roll_off must be in [0, 1], got {roll_off}")
    if span_symbols < 1:
        raise ValueError(f"span_symbols must be >= 1, got {span_symbols}")
    if samples_per_symbol < 1:
        raise ValueError(f"samples_per_symbol must be >= 1, got {samples_per_symbol}")

    length = span_symbols * samples_per_symbol
    if length % 2 == 0:
        length += 1  # odd, so the peak sits on a sample and the filter is symmetric
    t = (np.arange(length) - (length - 1) / 2.0) / samples_per_symbol

    beta = roll_off
    h = np.empty_like(t)
    for index, tau in enumerate(t):
        if abs(tau) < 1e-12:
            h[index] = 1.0 - beta + 4.0 * beta / np.pi
        elif beta > 0.0 and abs(abs(tau) - 1.0 / (4.0 * beta)) < 1e-12:
            # Removable singularity at t = +-T/(4*beta); the limit is closed-form.
            h[index] = (beta / np.sqrt(2.0)) * (
                (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * beta))
                + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * beta))
            )
        else:
            numerator = np.sin(np.pi * tau * (1.0 - beta)) + 4.0 * beta * tau * np.cos(
                np.pi * tau * (1.0 + beta)
            )
            denominator = np.pi * tau * (1.0 - (4.0 * beta * tau) ** 2)
            h[index] = numerator / denominator

    return h / np.sqrt(float(np.sum(h**2)))


def shape_symbols(
    symbols: np.ndarray, samples_per_symbol: int, filter_taps: np.ndarray
) -> np.ndarray:
    """Upsample by inserting zeros, then filter — circularly, to keep the window closed.

    Zero-insertion rather than sample-and-hold is what makes the filter the pulse
    shape rather than a correction applied on top of a rectangle.

    The convolution wraps because every other block in the engine treats the time
    window as periodic: a linear convolution would leave a transient at each end
    that the analyser would read as distortion, and the fix for that is the same
    ``ignore_edges`` everything else already uses.

    The result is scaled so the symbol instants keep the constellation's own
    level. Without it a unit-energy filter would silently rescale the alphabet,
    and every drive voltage downstream would be wrong by a factor depending on
    the oversampling. Note this fixes the *scale* only: a single root is not
    ISI-free, and the waveform sampled here still carries its neighbours'
    interference. Removing that is the receiver's matched root's job, and the two
    together are what satisfy the Nyquist criterion.
    """
    upsampled = np.zeros(symbols.shape[0] * samples_per_symbol, dtype=np.complex128)
    upsampled[::samples_per_symbol] = symbols.astype(np.complex128)
    peak = float(np.max(np.abs(filter_taps)))
    if peak <= 0.0:
        raise ValueError("the shaping filter is all zeros")
    return circular_filter(upsampled, filter_taps) / peak


def circular_filter(samples: np.ndarray, filter_taps: np.ndarray) -> np.ndarray:
    """Filter with a symmetric FIR, wrapping at the window edges and zero group delay."""
    taps = filter_taps.shape[0]
    padded = np.zeros(samples.shape[0], dtype=np.complex128)
    padded[:taps] = filter_taps
    spectrum = np.fft.fft(samples) * np.fft.fft(padded)
    # Undo the filter's own delay so the symbol instants stay where they were.
    delay = (taps - 1) // 2
    return np.roll(np.fft.ifft(spectrum), -delay)


def constellation_radii(constellation: np.ndarray) -> np.ndarray:
    """The distinct moduli a constellation uses, ascending.

    QPSK has one; 16-QAM has three. A one-radius constellation is what makes the
    plain constant-modulus algorithm exact, and the reason it only *approximately*
    works on QAM — which is what the radius-directed stage exists to fix.
    """
    moduli = np.abs(np.asarray(constellation).astype(np.complex128))
    return np.unique(np.round(moduli, 9))


def godard_radius(constellation: np.ndarray) -> float:
    """The CMA target ``R2 = E|c|^4 / E|c|^2``.

    For a constellation of one modulus this is that modulus squared. For QAM it
    is a compromise no symbol actually sits on, which is exactly why CMA opens
    the eye but does not close it, and why it is used as a *pre-convergence*
    stage rather than as the whole equaliser.
    """
    moduli = np.abs(np.asarray(constellation).astype(np.complex128))
    second = float(np.mean(moduli**2))
    if second <= 0.0:
        raise ValueError("the constellation carries no power")
    return float(np.mean(moduli**4) / second)


def radius_fit(symbols: np.ndarray, constellation: np.ndarray) -> float:
    """Mean squared distance from each sample's modulus to the nearest ring.

    The statistic the second stage adapts on, read as a score rather than a
    gradient. A tributary that has been separated sits on the rings; a mixture,
    or one an over-parameterised filter has smeared, does not.
    """
    radii = constellation_radii(np.asarray(constellation).astype(np.complex128))
    magnitude = np.abs(np.asarray(symbols))
    nearest = radii[np.argmin(np.abs(magnitude[:, None] - radii[None, :]), axis=1)]
    return float(np.mean((magnitude - nearest) ** 2))


def butterfly_separate(
    x: np.ndarray,
    y: np.ndarray,
    constellation: np.ndarray,
    **kwargs: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Equalise twice — settling the centre tap first, and not — and keep the better.

    Returns ``(out_x, out_y, weights, settled)``.

    The two ways of running the first stage each fail where the other works, and
    which one a channel needs is not knowable in advance: a rotation is
    memoryless and wants the centre tap alone, residual dispersion is not and
    wants every tap. So both are run and :func:`radius_fit` decides, which is not
    a new criterion — it is the same one the second stage already steers by, read
    as a score.

    Measured over ten cases, it picks right in all of them: 64-QAM through a
    rotated channel goes from 3568 symbol errors to zero at every angle, 16-QAM
    with 60 ps/nm of residual dispersion keeps the 44 it had, and 16-QAM at
    fifteen taps with nothing to correct improves from 544 to 53. Nothing it was
    tried on got worse.

    It costs one extra pass of an adaptation that is already the slowest thing in
    the receiver chain. That is the trade, and it is why the block exposes a way
    to turn it off.
    """
    candidates = []
    count = np.asarray(x).shape[0]
    for settle in (0, count):
        out_x, out_y, weights = butterfly_equalize(
            x,
            y,
            constellation,
            settle_symbols=settle,
            **kwargs,  # type: ignore[arg-type]
        )
        score = radius_fit(out_x, constellation) + radius_fit(out_y, constellation)
        candidates.append((score, settle > 0, out_x, out_y, weights))

    _, settled, out_x, out_y, weights = min(candidates, key=lambda entry: entry[0])
    return out_x, out_y, weights, settled


def butterfly_equalize(
    x: np.ndarray,
    y: np.ndarray,
    constellation: np.ndarray,
    *,
    taps: int = 7,
    step: float = 3e-3,
    cma_symbols: int | None = None,
    settle_symbols: int = 0,
    passes: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Blindly separate two mixed tributaries with an adaptive 2x2 FIR filter.

    Returns ``(out_x, out_y, weights)`` with ``weights`` the final ``(2, 2, taps)``
    filter, kept because a converged butterfly *is* the channel's inverse and
    reading it is how one checks what the channel did.

    **How it separates without being told anything.** Each output is a filtered
    combination of both inputs. The filters are adapted to drive every output
    symbol onto a modulus the constellation actually uses — no reference, no
    training sequence. A mixture of two independent tributaries has a modulus
    that wanders; a clean tributary does not. That difference is the entire
    signal the algorithm has, and it is enough.

    Two stages, because one is not enough for QAM. The first drives towards the
    single Godard radius, which opens the eye from nothing but converges to a
    compromise; the second switches to the *nearest* of the constellation's real
    radii, which is what actually closes it. On QPSK the two stages are identical
    because there is only one radius.

    **The first stage used to be what stopped this working at 64-QAM.** Driving
    every sample onto one radius is exactly right for QPSK, survivable for
    16-QAM, and destructive for a constellation whose points run from 0.218 to
    1.528 on a unit-power grid — the amplitude structure the format carries is
    the thing being flattened. It cost 3568 symbol errors out of 8192 on a
    channel that needed no equalisation at all.

    ``settle_symbols`` is what fixed it, and the reason it works says what the
    problem was. A polarization rotation is *memoryless*: a 2x2 complex matrix,
    four numbers. Fitting it with seven taps per path means twenty-eight, and the
    twenty-four that are not needed fill with gradient noise, which the
    single-radius cost has no per-sample truth to hold down. Adapting only the
    centre tap while that stage runs leaves 64-QAM at **zero errors at every
    rotation**, where a single tap had already been measured to leave a residual
    of 0.0013 against seven taps' 0.29.

    It is not free, which is why it is a parameter and not the new behaviour. An
    eye closed by *memory* rather than by rotation — residual dispersion after an
    imperfect compensator — needs every tap moving during the smooth first stage,
    because the radius-directed second one cannot bootstrap from a closed eye. At
    60 ps/nm of residual, settling the centre tap first costs 16-QAM 1955 errors
    where letting all the taps run costs 44. Choosing between them is
    :func:`butterfly_separate`'s job.

    Four other ways out were tried and measured, and none of them moved anything:
    gating the second stage's decision on whether the ring it picked was
    credible, normalising the update by the window energy so gradient noise stops
    scaling with the tap count, annealing the second stage's step, and leaking
    the weights towards zero. Freezing the second stage entirely still left 3249
    errors, which is what identified the first stage as the culprit in the first
    place.

    **The singularity.** Nothing in the cost function distinguishes the two
    outputs, so in principle both filters can converge onto the *same* tributary
    and leave the other unrecovered — the well-known failure of blind butterfly
    equalisation. What keeps them apart here is the initialisation: the two rows
    start orthogonal and the adaptation has no reason to bring them together.

    A textbook remedy is to re-seed the second row as the unitary complement of
    the first once the first has settled. That was tried and *measured*, and over
    ninety channel and format combinations it helped eight times, hurt nine, and
    changed nothing in the rest — so it is not here. The tilt below is the part
    that is actually load-bearing, and there is a test that fails without it.
    """
    if taps < 1 or taps % 2 == 0:
        raise ValueError(f"taps must be a positive odd number, got {taps}")
    if step <= 0.0:
        raise ValueError(f"step must be positive, got {step}")
    if passes < 1:
        raise ValueError(f"passes must be >= 1, got {passes}")
    if x.shape != y.shape:
        raise ValueError(f"tributaries differ in length: {x.shape} and {y.shape}")

    xs = x.astype(np.complex128)
    ys = y.astype(np.complex128)
    count = xs.shape[0]
    if count < taps:
        raise ValueError(f"need at least {taps} symbols to fill the filter, got {count}")

    points = np.asarray(constellation).astype(np.complex128)
    radii = constellation_radii(points)
    target = godard_radius(points)

    centre = taps // 2
    # h[out, in, tap]. A centre spike is the identity: pass each input straight
    # through to the matching output, then let the adaptation find the rotation.
    #
    # The small cross term is not decoration. A channel that mixes the two
    # tributaries exactly half and half — a 45 degree rotation — leaves the
    # identity initialisation equidistant from both valid solutions, which is a
    # saddle of the cost function rather than a minimum. The adaptation stalls
    # there, and running it longer makes it worse rather than better. Tilting the
    # start off the symmetry axis removes the saddle. It is deterministic, so the
    # result stays reproducible; 64-QAM at 45 degrees fails without it and lands
    # on the noise floor with it.
    weights = np.zeros((2, 2, taps), dtype=np.complex128)
    weights[0, 0, centre] = 1.0
    weights[1, 1, centre] = 1.0
    weights[0, 1, centre] = SYMMETRY_TILT
    weights[1, 0, centre] = -SYMMETRY_TILT

    if cma_symbols is None:
        cma_symbols = count // 2
    # During the settling phase only the centre tap moves. A polarization
    # rotation is memoryless — four complex numbers — and letting the other
    # twenty-four fit it as well is what fills them with gradient noise.
    settling = np.zeros(taps)
    settling[centre] = 1.0

    out_x = np.zeros(count, dtype=np.complex128)
    out_y = np.zeros(count, dtype=np.complex128)

    for pass_index in range(passes):
        for processed, k in enumerate(range(centre, count - centre)):
            window_x = xs[k - centre : k + centre + 1][::-1]
            window_y = ys[k - centre : k + centre + 1][::-1]

            ox = weights[0, 0] @ window_x + weights[0, 1] @ window_y
            oy = weights[1, 0] @ window_x + weights[1, 1] @ window_y
            out_x[k] = ox
            out_y[k] = oy

            # Radius-directed once the eye is open; Godard's single radius before
            # then, because a decision on a closed eye is worse than no decision.
            blind = pass_index == 0 and processed < cma_symbols
            if blind:
                error_x = target - abs(ox) ** 2
                error_y = target - abs(oy) ** 2
            else:
                error_x = radii[np.argmin(np.abs(radii - abs(ox)))] ** 2 - abs(ox) ** 2
                error_y = radii[np.argmin(np.abs(radii - abs(oy)))] ** 2 - abs(oy) ** 2

            gate = settling if (blind and processed < settle_symbols) else 1.0
            weights[0, 0] += step * error_x * ox * np.conj(window_x) * gate
            weights[0, 1] += step * error_x * ox * np.conj(window_y) * gate
            weights[1, 0] += step * error_y * oy * np.conj(window_x) * gate
            weights[1, 1] += step * error_y * oy * np.conj(window_y) * gate

    # The filter cannot produce an output for the first and last half-window.
    out_x[:centre] = out_x[centre]
    out_y[:centre] = out_y[centre]
    out_x[count - centre :] = out_x[count - centre - 1]
    out_y[count - centre :] = out_y[count - centre - 1]

    return out_x, out_y, weights
