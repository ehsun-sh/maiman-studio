"""Numerical kernels.

Everything computationally heavy goes through this module. It is deliberately
narrow — a handful of array-in/array-out functions with no knowledge of
components, graphs, or units — so that the back-end can change without touching
any physics code above it. Today that back-end is NumPy; CuPy (`cupy.fft` is a
drop-in for `numpy.fft`) and a native module are the intended next options.

**FFT library.** `numpy.fft` uses pocketfft (BSD). FFTW is *not* used and must
not be introduced: it is GPL-2.0-or-later, and linking it — directly or through
`pyFFTW` — would relicense the whole project.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .units import C_LIGHT


def angular_frequency_grid(num_samples: int, sample_rate: float) -> np.ndarray:
    """Angular frequency offsets from the band centre [rad/s], in FFT order.

    Returned in `numpy.fft` output order (positive frequencies first, then
    negative), so it multiplies an un-shifted spectrum directly.
    """
    return 2.0 * np.pi * np.fft.fftfreq(num_samples, d=1.0 / sample_rate)


def dispersion_to_beta2(dispersion: float, wavelength: float) -> float:
    """Convert the dispersion parameter D [s/m²] to the GVD parameter β₂ [s²/m].

    ``beta2 = -D * lambda**2 / (2*pi*c)``

    The sign matters: standard single-mode fiber has D > 0 at 1550 nm and
    therefore β₂ < 0 (anomalous dispersion).
    """
    return -dispersion * wavelength**2 / (2.0 * np.pi * C_LIGHT)


def dispersion_slope_to_beta3(dispersion: float, slope: float, wavelength: float) -> float:
    """Convert the dispersion slope S [s/m³] to the third-order parameter β₃ [s³/m].

    ``S = dD/dlambda``, and D is itself a function of β₂, so differentiating
    ``D = -2*pi*c*beta2/lambda**2`` gives

        ``S = (2*pi*c/lambda**2)**2 * beta3 + (4*pi*c/lambda**3) * beta2``

    which inverts, with ``beta2`` written out through :func:`dispersion_to_beta2`,
    to

        ``beta3 = (lambda**2 / 2*pi*c)**2 * (S + 2*D/lambda)``

    **The slope is not β₃ by another name, and D has to be passed in.** Even at
    zero slope a fibre has a nonzero β₃, because holding D constant across
    wavelength is itself a statement about how β₂ varies: the ``2*D/lambda``
    term is what a flat D costs. For standard fibre at 1550 nm, D = 17 ps/nm/km
    and S = 0.058 ps/nm²/km give β₃ = 0.13 ps³/km, which is the value the
    literature quotes; feeding it S = 0.09 — the slope at the *zero-dispersion*
    wavelength, which is the number datasheets lead with — gives 0.18 and is the
    easiest way to be forty percent wrong here.
    """
    return (wavelength**2 / (2.0 * np.pi * C_LIGHT)) ** 2 * (slope + 2.0 * dispersion / wavelength)


def propagate_dispersion(
    field: np.ndarray, sample_rate: float, beta2: float, distance: float, beta3: float = 0.0
) -> np.ndarray:
    """Propagate a complex envelope through group-velocity dispersion and its slope.

    Solves the linear part of the NLSE,
    ``dA/dz = -(i*beta2/2) d2A/dT2 + (beta3/6) d3A/dT3``, which in the frequency
    domain is an exact all-pass phase rotation::

        A(z, w) = A(0, w) * exp(i * (beta2 * w**2 / 2 - beta3 * w**3 / 6) * z)

    Because the transfer function has unit magnitude, this conserves energy
    exactly (up to floating-point) and is exactly invertible by propagating
    ``-distance`` — both of which are asserted in the test suite.

    **The two terms have opposite signs, and that is not a typo.** With this
    module's transform pair, ``x(t) = integral X(w) exp(+i w t) dw``, a time
    derivative is ``+i w``; the β₂ term carries two of them and the β₃ term
    three, so ``(i w)**2 = -w**2`` and ``(i w)**3 = -i w**3`` land the two on
    opposite sides. A β₂-only model is insensitive to the transform's sign
    convention because ω appears squared — that is exactly what stops being true
    here, which is why the β₃ term is derived from the same expansion of
    ``beta(omega)`` as the group-delay term in
    :func:`propagate_coupled_ssfm` rather than written down.

    β₃ is what makes dispersive broadening *asymmetric*: β₂ delays a frequency in
    proportion to its offset, so the two sides of a pulse spread alike, while β₃
    delays in proportion to the square and both sides move the same way. The
    tests measure that as a skewness, because it is the part of the answer a
    sign error in the cubic term would get wrong while every width still came out
    right.

    The sign of β₂ itself, however, is *not* free — and the unchirped broadening
    formula cannot detect an error in it, being even in β₂. Only the chirped case
    can, which is why ``test_chirped_pulse_compresses_before_broadening`` exists.

    The phase argument reaches thousands of radians over a realistic span, so the
    transform runs in double precision regardless of the storage precision and
    the caller casts the result back. Correctness first; if profiling later shows
    this matters, the precision policy belongs here, in one place.
    """
    if distance == 0.0 or (beta2 == 0.0 and beta3 == 0.0):
        return field.astype(np.complex128, copy=True)

    omega = angular_frequency_grid(field.shape[0], sample_rate)
    transfer = np.exp(1j * (0.5 * beta2 * omega**2 - beta3 * omega**3 / 6.0) * distance)
    return np.fft.ifft(np.fft.fft(field.astype(np.complex128)) * transfer)


def soliton_peak_power(beta2: float, gamma: float, width: float, order: int = 1) -> float:
    """Peak power of a soliton of the given order [W].

    A sech pulse of width ``T0`` is a soliton of order N when
    ``N**2 = gamma * P0 * T0**2 / |beta2|``. The fundamental (N = 1) is the case
    where the chirp Kerr imposes exactly cancels the chirp dispersion imposes,
    so the pulse propagates unchanged — which makes it the sharpest available
    check that both effects are implemented correctly and with the right signs
    relative to each other.

    Requires anomalous dispersion (``beta2 < 0``); in normal dispersion the two
    effects add instead of cancelling and no bright soliton exists.
    """
    if beta2 >= 0.0:
        raise ValueError(f"bright solitons need anomalous dispersion (beta2 < 0), got {beta2}")
    if gamma <= 0.0:
        raise ValueError(f"gamma must be positive, got {gamma}")
    return order**2 * abs(beta2) / (gamma * width**2)


def soliton_period(beta2: float, width: float) -> float:
    """Soliton period ``z0 = (pi/2) * T0**2 / |beta2|`` [m]."""
    return 0.5 * np.pi * width**2 / abs(beta2)


def attenuation_db_per_m_to_alpha(attenuation_db_per_m: float) -> float:
    """Convert a loss coefficient in dB/m to the power attenuation ``alpha`` [1/m].

    ``P(z) = P(0) * exp(-alpha * z)``, so ``alpha = ln(10)/10 * dB per metre``.
    """
    return attenuation_db_per_m * np.log(10.0) / 10.0


@dataclass(frozen=True)
class PropagationDiagnostics:
    """What the propagator actually did, so accuracy can be audited.

    A split-step result is only as good as its step size, and a fixed-step run
    produces answers that look plausible and are wrong. Reporting the step count
    and the largest nonlinear phase per step means the number can be checked
    rather than trusted.
    """

    steps: int
    distance: float
    shortest_step: float
    longest_step: float
    peak_nonlinear_phase: float
    """Largest nonlinear phase rotation applied in any single step [rad]."""

    differential_group_delay: float = 0.0
    """DGD realised on this run [s]. PMD is random, so this differs run to run;
    it is reported because the value a result depends on should be visible."""

    walkoff_span: float = 0.0
    """Largest relative group delay between bands over the span [s].

    Removed from the returned waveforms — each band comes back in its own
    retarded frame — but reported here, because it is what decides how much
    cross-phase modulation survives and a removed quantity that still governs
    the answer should not be invisible."""

    peak_walkoff_slip: float = 0.0
    """Largest relative slip between bands within any single step [samples].

    The nonlinear operator freezes the bands' relative positions for the length
    of a step, so this is the walk-off counterpart of
    ``peak_nonlinear_phase``: the quantity a shorter step buys accuracy in."""

    mixing_products: int = 0
    """Four-wave mixing products emitted at distinct frequencies."""

    raman_tilt: float = 0.0
    """Power the comb's extreme channels exchanged, in dB [longest minus shortest].

    Positive means the long-wavelength end gained, which is the only direction
    stimulated Raman scattering runs. Zero when the effect is switched off or
    when there is only one channel to tilt."""

    def __repr__(self) -> str:
        pmd = (
            f", DGD {self.differential_group_delay * 1e12:.2f} ps"
            if self.differential_group_delay
            else ""
        )
        walkoff = f", walk-off {self.walkoff_span * 1e12:.1f} ps" if self.walkoff_span else ""
        fwm = f", {self.mixing_products} FWM tones" if self.mixing_products else ""
        raman = f", Raman tilt {self.raman_tilt:+.2f} dB" if self.raman_tilt else ""
        return (
            f"PropagationDiagnostics({self.steps} steps over {self.distance / 1e3:.1f} km, "
            f"max phase {self.peak_nonlinear_phase:.4f} rad{pmd}{walkoff}{fwm}{raman})"
        )


def propagate_ssfm(
    field: np.ndarray,
    sample_rate: float,
    *,
    beta2: float,
    gamma: float,
    alpha: float,
    distance: float,
    max_nonlinear_phase: float = 0.005,
    max_step: float | None = None,
) -> tuple[np.ndarray, PropagationDiagnostics]:
    """Solve the nonlinear Schrödinger equation by symmetric split-step Fourier.

    ``dA/dz = -(alpha/2) A - (i beta2 / 2) d2A/dT2 + i gamma |A|**2 A``

    Each step applies half the linear operator in the frequency domain, the full
    nonlinear phase in the time domain, then the other half linear operator. The
    symmetric ordering makes the local error third order in the step size rather
    than second.

    **The step size is adaptive, and that is not optional.** The nonlinear term
    is a phase rotation proportional to instantaneous power, so a step long
    enough to rotate the peak by an appreciable angle stops commuting with
    dispersion in a way that quietly changes the answer. Steps here are bounded
    so the largest nonlinear rotation per step stays under
    ``max_nonlinear_phase``; the default of 5 mrad is conservative. Because the
    bound is recomputed from the current peak power, steps lengthen naturally as
    the pulse loses power to attenuation.

    This is the one-channel case of :func:`propagate_coupled_ssfm` and is written
    as a call to it rather than as a second implementation. With a single field
    the cross-phase term ``2*total - |A|**2`` collapses to ``|A|**2`` exactly, so
    the two agree to the last bit and cannot drift apart as either is changed.

    Returns the propagated field and :class:`PropagationDiagnostics`.
    """
    fields, diagnostics = propagate_coupled_ssfm(
        (field,),
        sample_rate,
        beta2=(beta2,),
        walkoff=(0.0,),
        gamma=gamma,
        alpha=alpha,
        distance=distance,
        max_nonlinear_phase=max_nonlinear_phase,
        max_step=max_step,
    )
    return fields[0], diagnostics


#: Weight the Kerr term gives power in the *orthogonal* polarization, relative to
#: power in its own.
#:
#: Two thirds, and it is the same two thirds in both places it appears: a
#: channel's own orthogonal component modulates it at 2/3 of the rate its
#: co-polarized component does, and a neighbour's orthogonal power at 2/3 of the
#: rate — which, since co-polarized cross-phase modulation carries the factor of
#: two, makes orthogonal cross-phase modulation exactly one third of co-polarized
#: cross-phase modulation. That ratio of three is the textbook number and is what
#: the tests measure.
#:
#: The factor comes from the tensor structure of chi(3) in an isotropic medium
#: and not from any averaging, so it is the fixed-axis value. A fibre whose
#: birefringence scrambles the polarization faster than the nonlinearity acts is
#: described instead by the Manakov equation, where the distinction between the
#: two components washes out into a single 8/9 on the total power. This model
#: applies PMD as a separate element rather than interleaving it, so it is the
#: fixed-axis form that is consistent with the rest of the block.
ORTHOGONAL_KERR_WEIGHT = 2.0 / 3.0


def propagate_coupled_ssfm(
    fields: Sequence[np.ndarray],
    sample_rate: float,
    *,
    beta2: Sequence[float],
    walkoff: Sequence[float],
    gamma: float,
    beta3: Sequence[float] | None = None,
    polarization: Sequence[int] | None = None,
    alpha: float,
    distance: float,
    max_nonlinear_phase: float = 0.005,
    max_walkoff_slip: float = 0.5,
    max_step: float | None = None,
) -> tuple[list[np.ndarray], PropagationDiagnostics]:
    """Co-propagate several channels through one fiber, coupled by the Kerr effect.

    ``dA_k/dz = -(alpha/2) A_k - d_k dA_k/dT - (i beta2_k / 2) d2A_k/dT2
    + (beta3_k / 6) d3A_k/dT3
    + i gamma (|A_k|**2 + 2 sum_{j != k} |A_j|**2 + (2/3) sum_{orthogonal} |A_j|**2) A_k``

    The whole content of the extension is that factor of two. A channel's own
    power rotates its own phase once; every *other* channel's power rotates it
    twice, and that asymmetry is not a fudge but falls out of expanding
    ``|A|**2 A`` for a sum of carriers — there are two ways to choose which of
    the two un-conjugated factors belongs to the neighbour and one way when it
    is the channel itself. Cross-phase modulation is therefore not a separate
    effect bolted on beside self-phase modulation; it is the same term, counted
    properly.

    Written as ``2 * total - |A_k|**2`` where ``total`` is the summed power of
    every field, so the cost is linear in the channel count rather than
    quadratic, and so one field reduces to plain SPM identically.

    **Walk-off is the reason the answer is not absurd.** Channels at different
    wavelengths travel at different group velocities, so a neighbour's power
    slides past rather than sitting on top of the channel it is modulating, and
    what survives is closer to the average of its pattern than to its peaks.
    Without that sliding the model would report an impairment several times too
    large and would get the dependence on dispersion backwards — it is the
    low-dispersion link, not the high-dispersion one, that suffers most from
    cross-phase modulation. ``walkoff[k]`` is the inverse group velocity of
    channel ``k`` relative to an arbitrary common frame [s/m]; the frame drops
    out, see below.

    **Sign convention.** :func:`propagate_dispersion` notes that a beta2-only
    model is insensitive to the transform's sign convention because omega
    appears squared, and that this stops being true the moment a group-delay
    term is added. This is that moment. Rather than guess, the walk-off operator
    is taken from the same Taylor expansion of ``beta(omega)`` that produced the
    dispersion operator: with ``exp(op * z)`` and ``op`` carrying
    ``0.5j * beta2 * omega**2`` for the quadratic term, the linear term is
    ``-1j * d_k * omega``, which delays a channel of larger inverse group
    velocity. Deriving it rather than asserting it is what makes it checkable.

    **Each field is returned in its own retarded frame.** The accumulated
    ``d_k * distance`` is divided out at the end, exactly, because it is a
    constant group delay: real hardware removes it in clock recovery, and
    keeping it would do nothing but slide every channel off its own sampler by
    tens of symbols. The walk-off still acts in full *during* propagation, where
    the physics is; only the bookkeeping delay is removed. Because it is
    removed, the choice of common frame cannot affect the result — and neither
    can the sign of the walk-off, since the impairment depends on the relative
    slip, which is even in it.

    **Two step-size bounds, for two ways of being wrong.** The nonlinear
    operator freezes both the power *and* the channels' relative positions for
    the length of a step, so besides the usual ``max_nonlinear_phase`` cap there
    is ``max_walkoff_slip``: the relative slip between the fastest and slowest
    channel within one step, in samples. Both are reported.

    The phase cap is on the largest rotation applied to *any* channel, which
    makes it the **weakest** channel that sets the step: it has the most
    neighbour power to be turned by and least of its own to subtract. So adding
    a dark or heavily attenuated band roughly halves the step size while
    changing the answer not at all. That is the conservative direction to err
    in, and the cost is bounded at a factor of two however faint the band is.

    Every field must share one time grid. Coupling is evaluated sample by sample
    and there is no meaningful way to add to it the power of a channel sampled
    on a different grid, so a mismatch raises rather than being papered over.

    **Polarization, when the caller asks for it.** ``polarization`` labels each
    field 0 or 1, and power on the other label enters the nonlinear term at
    :data:`ORTHOGONAL_KERR_WEIGHT` instead of the co-polarized weight::

        phase_k = gamma * (2 * P_same - |A_k|**2 + (2/3) * P_other) * step

    Left unset every field is on axis 0, the second sum is empty, and this is the
    scalar model term for term — which is what lets a caller propagate the two
    polarizations as two independent problems and get exactly what it got before.
    Pass both axes in one call with their labels and they couple: a channel is
    then modulated by its own orthogonal component at two thirds the rate, its
    neighbours' orthogonal power likewise, and — because the two axes of one
    channel no longer accumulate the same phase — the state of polarization
    rotates with the power, which is cross-polarization modulation.

    Returns the propagated fields, in input order, and
    :class:`PropagationDiagnostics`.
    """
    if distance < 0.0:
        raise ValueError(f"distance must be non-negative, got {distance}")
    if max_nonlinear_phase <= 0.0:
        raise ValueError(f"max_nonlinear_phase must be positive, got {max_nonlinear_phase}")
    if max_walkoff_slip <= 0.0:
        raise ValueError(f"max_walkoff_slip must be positive, got {max_walkoff_slip}")
    if len(beta2) != len(fields) or len(walkoff) != len(fields):
        raise ValueError(
            f"beta2 and walkoff must have one entry per field, got {len(beta2)} and "
            f"{len(walkoff)} for {len(fields)} fields"
        )
    slope = [0.0] * len(fields) if beta3 is None else list(beta3)
    if len(slope) != len(fields):
        raise ValueError(
            f"beta3 must have one entry per field, got {len(slope)} for {len(fields)} fields"
        )
    axis = [0] * len(fields) if polarization is None else list(polarization)
    if len(axis) != len(fields):
        raise ValueError(
            f"polarization must have one entry per field, got {len(axis)} for {len(fields)} fields"
        )
    if any(value not in (0, 1) for value in axis):
        raise ValueError(f"polarization entries must be 0 or 1, got {sorted(set(axis))}")

    a = [f.astype(np.complex128, copy=True) for f in fields]
    if not a:
        return a, PropagationDiagnostics(0, distance, 0.0, 0.0, 0.0)

    widths = {f.shape[0] for f in a}
    if len(widths) != 1:
        raise ValueError(
            "coupled propagation needs one common time grid; got lengths "
            f"{sorted(widths)}. Cross-phase modulation is evaluated sample by "
            "sample, so channels sampled differently cannot be coupled."
        )

    spread = max(walkoff) - min(walkoff)
    if distance == 0.0:
        return a, PropagationDiagnostics(0, 0.0, 0.0, 0.0, 0.0)

    omega = angular_frequency_grid(a[0].shape[0], sample_rate)
    # One expansion of beta(omega), one operator: the group delay, the dispersion
    # and its slope are the first three terms of the same series, which is what
    # fixes their relative signs. See propagate_dispersion.
    operators = [
        0.5j * b * omega**2 - 1j * w * omega - 1j * b3 * omega**3 / 6.0
        for b, w, b3 in zip(beta2, walkoff, slope, strict=True)
    ]
    ceiling = max_step if max_step is not None else distance

    travelled = 0.0
    steps = 0
    shortest = np.inf
    longest = 0.0
    peak_phase = 0.0
    peak_slip = 0.0

    while travelled < distance:
        remaining = distance - travelled
        step = min(ceiling, remaining)
        if gamma != 0.0:
            _axis_power = _power_per_axis(a, axis)
            # The largest rotation any channel will see. With one polarization
            # this is the smallest self power against the summed total, because
            # 2*total - |A_k|**2 grows as |A_k|**2 shrinks; written out per field
            # so that the orthogonal term is weighted here exactly as it is in
            # the rotation itself, and a co-polarized run picks the same step it
            # always did.
            peak_effective = 0.0
            for field, ax in zip(a, axis, strict=True):
                effective = 2.0 * _axis_power[ax] - np.abs(field) ** 2
                other = _axis_power.get(1 - ax)
                if other is not None:
                    effective = effective + ORTHOGONAL_KERR_WEIGHT * other
                peak_effective = max(peak_effective, float(np.max(effective)))
            if peak_effective > 0.0:
                step = min(step, max_nonlinear_phase / (abs(gamma) * peak_effective))
            # Only the nonlinear operator cares where the channels sit relative
            # to one another; the linear one is exact at any step length, so a
            # linear run needs no walk-off bound at all.
            if spread > 0.0:
                step = min(step, max_walkoff_slip / (spread * sample_rate))
        # A step can only be shortened to the point where it still advances;
        # without this an extreme peak power would stall the loop.
        step = max(min(step, remaining), remaining * 1e-9)

        half = [np.exp(-alpha * step / 4.0 + op * (step / 2.0)) for op in operators]
        a = [np.fft.ifft(np.fft.fft(f) * h) for f, h in zip(a, half, strict=True)]

        if gamma != 0.0:
            # Summed per polarization, because power in the orthogonal component
            # modulates at a different rate than power sharing an axis. With one
            # axis in use the second sum is zero and this is the scalar model
            # term for term.
            per_axis = _power_per_axis(a, axis)
            rotated = []
            for field, ax in zip(a, axis, strict=True):
                effective = 2.0 * per_axis[ax] - np.abs(field) ** 2
                other = per_axis.get(1 - ax)
                if other is not None:
                    effective = effective + ORTHOGONAL_KERR_WEIGHT * other
                phase = gamma * effective * step
                peak_phase = max(peak_phase, float(np.max(np.abs(phase))))
                rotated.append(field * np.exp(1j * phase))
            a = rotated

        a = [np.fft.ifft(np.fft.fft(f) * h) for f, h in zip(a, half, strict=True)]

        travelled += step
        steps += 1
        shortest = min(shortest, step)
        longest = max(longest, step)
        peak_slip = max(peak_slip, spread * step * sample_rate)

    if spread > 0.0:
        # Back into each channel's own retarded frame. This is the exact inverse
        # of the accumulated -1j*w*omega*distance, so it cannot remove anything
        # the propagation put there.
        a = [
            np.fft.ifft(np.fft.fft(f) * np.exp(1j * w * omega * distance))
            for f, w in zip(a, walkoff, strict=True)
        ]

    return a, PropagationDiagnostics(
        steps=steps,
        distance=distance,
        shortest_step=float(shortest),
        longest_step=longest,
        peak_nonlinear_phase=peak_phase,
        walkoff_span=spread * distance,
        peak_walkoff_slip=peak_slip,
    )


def _power_per_axis(fields: Sequence[np.ndarray], axis: Sequence[int]) -> dict[int, np.ndarray]:
    """Summed instantaneous power on each populated polarization axis [W].

    Keyed rather than a two-element list so that an unused axis is absent rather
    than an array of zeros. Numerically the two are the same — adding two thirds
    of nothing is nothing — so this buys no accuracy; what it buys is that a
    co-polarized run, which is every run that does not ask for the coupling,
    allocates and adds one array per step instead of two.
    """
    per_axis: dict[int, np.ndarray] = {}
    for which in (0, 1):
        members = [f for f, ax in zip(fields, axis, strict=True) if ax == which]
        if members:
            per_axis[which] = _total_power(members)
    return per_axis


def _total_power(fields: Sequence[np.ndarray]) -> np.ndarray:
    """Summed instantaneous power of co-propagating fields [W], sample by sample."""
    total = np.abs(fields[0]) ** 2
    for field in fields[1:]:
        total += np.abs(field) ** 2
    return total


#: Separation past which a triangular Raman gain profile stops being the truth.
#:
#: Silica's Raman gain rises roughly linearly with frequency separation, peaks
#: near 13.2 THz and falls away after it. Below this the linear approximation is
#: the standard one and is what :func:`raman_tilt` assumes; a comb wider than it
#: — the C and L bands together, say — has its far pairs past the peak, where the
#: model over-predicts the transfer. The fibre block reports how wide the comb it
#: was given actually is, so the case is visible rather than silent.
RAMAN_TRIANGLE_LIMIT = 13.2e12


def raman_tilt(
    frequencies: Sequence[float],
    powers: Sequence[float],
    *,
    gain_slope: float,
    effective_length: float,
) -> list[float]:
    """Power each channel keeps after stimulated Raman scattering, as a ratio.

    A photon can scatter off a silica vibration and come out at a lower
    frequency, and the process is stimulated: light already present at the lower
    frequency makes it more likely. So in a wavelength comb the short-wavelength
    channels pump the long-wavelength ones, and a flat launch does not arrive
    flat. Over one 80 km span of a filled C band it is most of a decibel, which
    is a large fraction of the margin a link is designed with.

    Closed form, from Zirngibl (*Electron. Lett.* 34(8), 1998), assuming the gain
    rises linearly with separation and every channel sees the same loss::

        P_n(L) = P_n(0) * P_total * exp(-C_R * P_total * L_eff * df_n)
                 / sum_m P_m(0) * exp(-C_R * P_total * L_eff * df_m)

    What is returned is the second factor alone — the redistribution, with the
    common ``exp(-alpha L)`` left out, because the caller has already applied the
    loss and applying it twice is the obvious way to get this wrong.

    **It conserves power.** Raman scattering moves power between channels; the
    quantum defect it loses to the lattice is a part in ten thousand at these
    separations and is not modelled. So ``sum(P_n * ratio_n) == sum(P_n)``, to
    floating point, and the tests hold it there — which is also what makes a
    sign error impossible to miss, since the two ends have to move in opposite
    directions by construction.

    ``df_n`` is measured from the mean of ``frequencies`` rather than from zero.
    The reference cancels between numerator and denominator, so this is not
    physics; it is what keeps the exponent near zero for a comb sitting at
    193 THz instead of asking ``exp`` for the ratio of two underflowed numbers.

    ``gain_slope`` is ``C_R`` in 1/(W·m·Hz) — 2.8e-17, or 0.028 1/(W·km·THz),
    for standard fibre at 1550 nm. ``effective_length`` is the span's, from
    :func:`effective_length`, because the transfer happens where the pump is
    still bright.
    """
    if len(frequencies) != len(powers):
        raise ValueError(
            f"one power per frequency, got {len(powers)} for {len(frequencies)} channels"
        )
    total = float(sum(powers))
    if gain_slope == 0.0 or total <= 0.0 or len(frequencies) < 2:
        return [1.0] * len(frequencies)

    offsets = np.asarray(frequencies, dtype=float)
    offsets = offsets - offsets.mean()
    weights = np.exp(-gain_slope * total * effective_length * offsets)
    normaliser = float(np.dot(np.asarray(powers, dtype=float), weights))
    if normaliser <= 0.0:
        return [1.0] * len(frequencies)
    return [float(total * w / normaliser) for w in weights]


def walkoff_from_dispersion(beta2: float, frequency_offset: float) -> float:
    """Inverse-group-velocity offset ``d`` [s/m] of a channel ``frequency_offset`` [Hz] away.

    ``d = beta2 * 2 * pi * frequency_offset``, the linear term of the same
    expansion of ``beta(omega)`` whose quadratic term is the dispersion the
    channel sees. Walk-off is therefore not an independent parameter: set the
    dispersion to zero and channels stop sliding past one another, which is
    exactly the condition under which cross-phase modulation is worst.

    In the units a link budget is written in this is ``d = D * delta_lambda``:
    at D = 17 ps/nm/km two channels 100 GHz apart (0.8 nm at 1550 nm) separate
    by about 13.6 ps for every kilometre they travel.
    """
    return beta2 * 2.0 * math.pi * frequency_offset


# --------------------------------------------------------------------------
# Four-wave mixing
# --------------------------------------------------------------------------


def effective_length(alpha: float, distance: float) -> float:
    """Nonlinear effective length ``L_eff = (1 - exp(-alpha*L)) / alpha`` [m].

    The length a lossless fiber would need to accumulate the same nonlinear
    effect as this lossy one. It saturates at ``1/alpha`` — about 21 km at
    0.2 dB/km — which is why the second half of an 80 km span contributes almost
    nothing nonlinear and why adding spans, not lengthening them, is what makes
    nonlinearity accumulate.

    Tends to ``distance`` as ``alpha`` tends to zero, and is evaluated that way
    rather than dividing by something near zero.
    """
    if alpha * distance < 1e-9:
        return distance
    return (1.0 - math.exp(-alpha * distance)) / alpha


def fwm_phase_mismatch(beta2: float, offset_i: float, offset_j: float, offset_k: float) -> float:
    """Linear phase mismatch ``delta_beta`` [rad/m] of the product at i + j - k.

    Four-wave mixing converts two photons from channels ``i`` and ``j`` into one
    at ``k`` and one at ``f_i + f_j - f_k``. Energy is conserved by construction;
    momentum is not, and the residual is what decides whether the product grows
    or oscillates away.

    Expanding ``beta(omega)`` to second order about any reference and forming
    ``beta_i + beta_j - beta_k - beta_F``, the constant and group-delay terms
    cancel exactly — they must, because the four frequencies satisfy
    ``omega_i + omega_j = omega_k + omega_F`` — and what is left is::

        delta_beta = -beta2 * (omega_i - omega_k) * (omega_j - omega_k)

    Two things follow, and they are the whole story of why dispersion suppresses
    four-wave mixing. It vanishes identically at zero dispersion, so a
    dispersion-shifted fiber operated at its zero is the worst possible place to
    put a WDM comb. And it grows as the *square* of the channel spacing, so
    widening the grid buys suppression quadratically.

    Offsets are in Hz from a common reference; only differences enter, so which
    reference is irrelevant. The sign of the result is unobservable — the
    efficiency below is even in it — but it is returned signed rather than
    absolute so that the expression above can be read off the code.
    """
    two_pi = 2.0 * math.pi
    return -beta2 * (two_pi * (offset_i - offset_k)) * (two_pi * (offset_j - offset_k))


def fwm_efficiency(phase_mismatch: float, alpha: float, distance: float) -> float:
    """Four-wave mixing efficiency, dimensionless and between 0 and 1.

    Under undepleted pumps the product field obeys
    ``dA_F/dz = i d gamma A_i A_j A_k* exp(i delta_beta z) - (alpha/2) A_F``,
    and with the pumps decaying as ``exp(-alpha z / 2)`` this integrates to a
    single mixing integral::

        A_F(L) proportional to integral_0^L exp((i delta_beta - alpha) z) dz

    The efficiency is that integral's squared magnitude normalised by
    ``L_eff**2``, so that it is 1 when the process is perfectly phase matched
    and the textbook form ``P_F = d**2 gamma**2 P_i P_j P_k L_eff**2 eta
    exp(-alpha L)`` holds with no further factors. Written out::

        eta = alpha**2 / (alpha**2 + delta_beta**2)
              * [1 + 4 exp(-alpha L) sin**2(delta_beta L / 2) / (1 - exp(-alpha L))**2]

    which is the expression usually quoted, and is what the complex integral
    above reduces to. The integral is evaluated instead of the expanded form
    because it stays well behaved in both limits: lossless, where the expression
    is 0/0 and the true answer is ``sinc**2(delta_beta L / 2)``, and phase
    matched, where it is 1.
    """
    if distance <= 0.0:
        return 0.0
    reference = effective_length(alpha, distance)
    if reference <= 0.0:
        return 0.0
    rate = complex(-alpha, phase_mismatch)
    if abs(rate) * distance < 1e-9:
        integral = complex(distance)
    else:
        integral = (np.exp(rate * distance) - 1.0) / rate
    return float(abs(integral) ** 2 / reference**2)


def fwm_product_power(
    power_i: float,
    power_j: float,
    power_k: float,
    *,
    gamma: float,
    alpha: float,
    distance: float,
    phase_mismatch: float,
    degenerate: bool,
) -> float:
    """Power generated at ``f_i + f_j - f_k`` over one span [W].

    ``P_F = d**2 gamma**2 P_i P_j P_k L_eff**2 eta exp(-alpha L)``

    ``d`` is the degeneracy factor, and it is not a fitted constant: expanding
    ``|A|**2 A = A A A*`` for a sum of carriers, the term oscillating at
    ``omega_i + omega_j - omega_k`` is assembled by choosing which of the two
    un-conjugated factors is ``i`` and which is ``j``. There are two ways when
    the pumps differ and one when they are the same channel, so ``d = 2`` for
    non-degenerate mixing and ``d = 1`` for degenerate. Non-degenerate products
    are therefore 6 dB stronger for the same pump powers — which is why the
    products that land *between* channels on a uniform grid, from three distinct
    pumps, dominate the ones that a single pump makes.

    The cubic dependence on power is the reason four-wave mixing is a launch
    power problem before it is anything else: 1 dB more per channel is 3 dB more
    product, and 2 dB less in the ratio that matters.

    Pump depletion is not modelled, so this overestimates once the product
    approaches the pumps. It never does at any power a link is operated at.
    """
    factor = 1.0 if degenerate else 2.0
    length = effective_length(alpha, distance)
    efficiency = fwm_efficiency(phase_mismatch, alpha, distance)
    return (
        factor**2
        * gamma**2
        * power_i
        * power_j
        * power_k
        * length**2
        * efficiency
        * math.exp(-alpha * distance)
    )


# --------------------------------------------------------------------------
# Polarization-mode dispersion
# --------------------------------------------------------------------------

#: Ratio of the mean square DGD to the squared mean, for a Maxwellian
#: distribution. It is a pure number, so measuring it is a shape test that does
#: not depend on getting the scale right.
MAXWELLIAN_MOMENT_RATIO = 3.0 * np.pi / 8.0


@dataclass(frozen=True)
class PMDSection:
    """One birefringent waveplate: a fixed delay between two random axes."""

    unitary: np.ndarray
    """2x2 unitary rotating into this section's principal states."""

    dgd: float
    """Differential group delay of this section [s]."""


def random_unitary_2x2(rng: np.random.Generator) -> np.ndarray:
    """A Haar-uniform SU(2) matrix, built from a random unit quaternion.

    Uniformity matters: birefringence axes in real fiber have no preferred
    orientation, and sampling them non-uniformly would bias the DGD statistics
    the whole model exists to reproduce.
    """
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    return np.array(
        [[q[0] + 1j * q[1], q[2] + 1j * q[3]], [-q[2] + 1j * q[3], q[0] - 1j * q[1]]],
        dtype=np.complex128,
    )


def random_pmd_sections(
    mean_dgd: float, sections: int, rng: np.random.Generator
) -> tuple[PMDSection, ...]:
    """Build a waveplate chain whose mean DGD is ``mean_dgd``.

    PMD is not a fixed impairment. Birefringence varies randomly along real
    fiber and drifts with temperature, so the differential group delay is a
    *random variable* with a Maxwellian distribution — which is why PMD is
    quoted as a coefficient in ps/sqrt(km) and why outage probability, rather
    than a worst case, is what gets designed against.

    Concatenating ``N`` randomly oriented waveplates of equal delay reproduces
    that distribution. Each section's delay follows from the second moment
    adding while the mean does not::

        <DGD> = section_dgd * sqrt(8N / (3*pi))

    so a target mean fixes the per-section delay. Around 50 sections is enough
    for the statistics to converge.
    """
    if sections < 1:
        raise ValueError(f"sections must be >= 1, got {sections}")
    if mean_dgd < 0.0:
        raise ValueError(f"mean_dgd must be non-negative, got {mean_dgd}")

    section_dgd = mean_dgd * np.sqrt(3.0 * np.pi / (8.0 * sections))
    return tuple(
        PMDSection(unitary=random_unitary_2x2(rng), dgd=float(section_dgd)) for _ in range(sections)
    )


def pmd_jones_matrix(sections: tuple[PMDSection, ...], omega: float) -> np.ndarray:
    """Total Jones transfer matrix of a waveplate chain at angular frequency ``omega``."""
    total = np.eye(2, dtype=np.complex128)
    for section in sections:
        phase = np.exp(0.5j * omega * section.dgd)
        delay = np.array([[phase, 0.0], [0.0, np.conj(phase)]], dtype=np.complex128)
        total = section.unitary @ delay @ total
    return total


def differential_group_delay(
    sections: tuple[PMDSection, ...], *, probe_spacing: float = 2.0 * np.pi * 1e9
) -> float:
    """Measure the chain's DGD by Jones matrix eigenanalysis [s].

    Takes the transfer matrix at two nearby frequencies, forms
    ``J(w2) @ inv(J(w1))``, and reads the delay off the argument of its
    eigenvalue ratio. This is the method a bench PMD analyser uses, which is a
    good reason to prefer it over reading the number back out of the parameters
    that generated it: it measures the chain rather than trusting it.

    ``probe_spacing`` must be small enough that ``probe_spacing * DGD`` stays
    below pi, or the phase wraps and the answer folds.
    """
    if not sections:
        return 0.0

    j1 = pmd_jones_matrix(sections, -probe_spacing / 2.0)
    j2 = pmd_jones_matrix(sections, probe_spacing / 2.0)
    eigenvalues = np.linalg.eigvals(j2 @ np.linalg.inv(j1))
    difference = np.angle(eigenvalues[0] / eigenvalues[1])
    return float(abs(difference) / probe_spacing)


def apply_pmd(
    ex: np.ndarray, ey: np.ndarray, sample_rate: float, sections: tuple[PMDSection, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate a Jones vector through a waveplate chain.

    Applied in the frequency domain, section by section, in the same order
    :func:`pmd_jones_matrix` multiplies them — so what a signal experiences and
    what the DGD measurement reports are the same chain.
    """
    if not sections:
        return ex.astype(np.complex128, copy=True), ey.astype(np.complex128, copy=True)

    omega = angular_frequency_grid(ex.shape[0], sample_rate)
    spectrum_x = np.fft.fft(ex.astype(np.complex128))
    spectrum_y = np.fft.fft(ey.astype(np.complex128))

    for section in sections:
        phase = np.exp(0.5j * omega * section.dgd)
        delayed_x = spectrum_x * phase
        delayed_y = spectrum_y * np.conj(phase)
        u = section.unitary
        spectrum_x = u[0, 0] * delayed_x + u[0, 1] * delayed_y
        spectrum_y = u[1, 0] * delayed_x + u[1, 1] * delayed_y

    return np.fft.ifft(spectrum_x), np.fft.ifft(spectrum_y)


def gaussian_lowpass_response(frequency: np.ndarray, bandwidth: float) -> np.ndarray:
    """Amplitude response of a Gaussian low-pass with 3 dB power bandwidth ``bandwidth``.

    ``|H(f)|**2 = exp(-ln2 * (f/B)**2)``, which is exactly 1/2 at ``f = B`` — that
    identity is the definition of the 3 dB point and is asserted in the tests.
    """
    if bandwidth <= 0.0:
        raise ValueError(f"bandwidth must be positive, got {bandwidth}")
    return np.exp(-0.5 * np.log(2.0) * (frequency / bandwidth) ** 2)


def super_gaussian_response(frequency: np.ndarray, bandwidth: float, order: int) -> np.ndarray:
    """Amplitude response of a super-Gaussian band-pass of 3 dB *full* width ``bandwidth``.

    ``|H(f)|**2 = exp(-ln2 * (2f/B)**(2n))``, which is exactly 1/2 at ``f = B/2``
    for every order — so the declared width means the same thing whatever the
    shape, and only the steepness of the skirts changes.

    Order 1 is an ordinary Gaussian, the shape of a thin-film filter. Raising the
    order flattens the top and steepens the edges towards the brick wall a
    wavelength-selective switch approximates; 3 to 5 is the usual range for a
    ROADM channel. The flat top matters for a signal passing through many of
    them, because a Gaussian's rounded peak narrows the passband a little at
    every hop while a flat one does not.
    """
    if bandwidth <= 0.0:
        raise ValueError(f"bandwidth must be positive, got {bandwidth}")
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}")
    return np.exp(-0.5 * np.log(2.0) * (2.0 * frequency / bandwidth) ** (2 * order))


def super_gaussian_noise_bandwidth(bandwidth: float, order: int) -> float:
    """Equivalent noise bandwidth of :func:`super_gaussian_response` [Hz].

    ``B_n = integral |H(f)|**2 df = B * Gamma(1 + 1/2n) / ln2**(1/2n)``.

    Having it in closed form is what lets a filtered ASE power be checked by
    arithmetic. At order 1 it is ``B * sqrt(pi/4ln2) ~ 1.0645 B``, which is
    :func:`gaussian_noise_bandwidth` and is asserted to equal it exactly.

    It does *not* approach ``B`` monotonically, which is worth knowing before
    reading a ratio near 1 as convergence: measured, the ratio is 1.0645, 0.9934,
    0.9862, 0.9869, 0.9915 at orders 1, 2, 3, 5 and 10. It undershoots around
    order 3 and comes back. A Gaussian's rounded shoulders pass more than its 3 dB
    width; a steep skirt passes slightly less before the flat top wins it back.
    """
    if bandwidth <= 0.0:
        raise ValueError(f"bandwidth must be positive, got {bandwidth}")
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}")
    exponent = 1.0 / (2.0 * order)
    return float(bandwidth * math.gamma(1.0 + exponent) / np.log(2.0) ** exponent)


def gaussian_noise_bandwidth(bandwidth: float) -> float:
    """Equivalent noise bandwidth of :func:`gaussian_lowpass_response` [Hz].

    ``B_n = integral of |H(f)|**2 df = B * sqrt(pi / (4 ln2)) ~ 1.0645 * B``.

    Having this in closed form is what makes the filter's effect on noise
    checkable by arithmetic rather than by eyeballing a variance.
    """
    return bandwidth * float(np.sqrt(np.pi / (4.0 * np.log(2.0))))


def lowpass_filter(samples: np.ndarray, sample_rate: float, bandwidth: float) -> np.ndarray:
    """Zero-phase Gaussian low-pass filter of a real waveform.

    The response is real and even, so the filter has no phase and no group delay:
    the output stays aligned with the input and nothing downstream has to
    compensate a delay. That is not causal, which a physical receiver is — a
    causal Bessel model, with the group delay that comes with it, is a later
    refinement and belongs in this same function.

    Filtering is circular, since the window is treated as periodic. The impulse
    response spans a couple of symbols, so a few symbols at each end of the
    window are contaminated by the wrap; analysis blocks drop them.
    """
    n = samples.shape[0]
    spectrum = np.fft.rfft(samples.astype(np.float64))
    response = gaussian_lowpass_response(np.fft.rfftfreq(n, d=1.0 / sample_rate), bandwidth)
    return np.fft.irfft(spectrum * response, n)
