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
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from .backend import array_module
from .units import C_LIGHT


def angular_frequency_grid(
    num_samples: int, sample_rate: float, *, like: object = None
) -> np.ndarray:
    """Angular frequency offsets from the band centre [rad/s], in FFT order.

    Returned in `numpy.fft` output order (positive frequencies first, then
    negative), so it multiplies an un-shifted spectrum directly.

    ``like`` names the array library to build it on — pass the field it is about
    to multiply and the grid is created wherever that field lives, which is the
    difference between one kernel and a device round trip per step. See
    :mod:`maiman.backend`.
    """
    xp = array_module(like)
    return 2.0 * xp.pi * xp.fft.fftfreq(num_samples, d=1.0 / sample_rate)


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

    An exact all-pass phase rotation in the frequency domain::

        A(z, w) = A(0, w) * exp(-i * (beta2 * w**2 / 2 + beta3 * w**3 / 6) * z)

    which is the expansion of ``beta(omega)`` past its group-delay term, carried
    with this library's ``exp(-i beta z)`` convention — the same one
    :func:`maiman.photonics.propagation_constant` and
    :func:`walkoff_from_dispersion` use.

    Because the transfer function has unit magnitude, this conserves energy
    exactly (up to floating-point) and is exactly invertible by propagating
    ``-distance`` — both of which are asserted in the test suite.

    **The β₂ term used to carry the opposite sign, and the story is worth
    keeping.** It was Agrawal's ``exp(+i beta2 w**2 z / 2)``, which is correct
    under the ``exp(-i w t)`` convention that book uses and wrong under the
    ``x(t) = integral X(w) exp(+i w t) dw`` that ``numpy.fft.ifft`` gives this
    one. A β₂-only model cannot feel the difference — ω appears squared, so every
    width came out right — and the β₃ term was never affected because it had been
    derived from the expansion of ``beta(omega)`` rather than transcribed.

    What could feel it was anything built on the other convention. Inside
    :func:`propagate_coupled_ssfm` the walk-off term and the β₃ term already
    followed ``exp(-i beta z)`` and the β₂ term beside them did not, so a WDM
    simulation slid its channels one way and dispersed each of them the other.
    Measured directly: over 20 km at D = +17 ps/nm/km, a component 200 GHz above
    the carrier arrived 1089.89 ps **late** where ``D * dlambda * L`` and
    :func:`walkoff_from_dispersion` both say it arrives 1089.89 ps early.

    Correcting it moved three other things, all of them sign conventions that had
    been matched to the old one: the Kerr rotation in
    :func:`propagate_coupled_ssfm` (which has to flip with β₂ or the soliton
    stops balancing), the chirp parameter of
    :class:`~maiman.components.GaussianPulse` (so that ``C > 0`` still means an
    up-chirp), and the trial phase the blind dispersion search builds in
    :func:`maiman.dsp.clock_tone_strength`.

    β₃ is what makes dispersive broadening *asymmetric*: β₂ delays a frequency in
    proportion to its offset, so the two sides of a pulse spread alike, while β₃
    delays in proportion to the square and both sides move the same way. The
    tests measure that as a skewness, because it is the part of the answer a
    sign error in the cubic term would get wrong while every width still came out
    right.

    The sign of β₂ is *not* free, and the unchirped broadening formula cannot
    detect an error in it, being even in β₂. Only the chirped case can, which is
    why ``test_chirped_pulse_compresses_before_broadening`` exists — though note
    that it did not catch the error above either, because the pulse it launches
    was defined in the same convention as the propagator and the two agreed with
    each other while disagreeing with the rest of the library. What caught it was
    putting a fibre and a grating in one graph.

    The phase argument reaches thousands of radians over a realistic span, so the
    transform runs in double precision regardless of the storage precision and
    the caller casts the result back. Correctness first; if profiling later shows
    this matters, the precision policy belongs here, in one place.
    """
    xp = array_module(field)
    if distance == 0.0 or (beta2 == 0.0 and beta3 == 0.0):
        return field.astype(xp.complex128, copy=True)

    omega = angular_frequency_grid(field.shape[0], sample_rate, like=field)
    transfer = xp.exp(-1j * (0.5 * beta2 * omega**2 + beta3 * omega**3 / 6.0) * distance)
    return xp.fft.ifft(xp.fft.fft(field.astype(xp.complex128)) * transfer)


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

    fwm_depletion: float = 0.0
    """Fraction of the power leaving the span that four-wave mixing moved into products.

    The pumps pay for it photon by photon, so this is also what they lost. Tens of
    parts per million at the powers a link runs at; a percent or more is the
    regime where an undepleted treatment would have created energy."""

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
        if self.fwm_depletion:
            fwm += f" ({self.fwm_depletion:.1e} depleted)"
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
    pairs: Sequence[tuple[int, int]] | None = None,
    coherent_polarization: bool = False,
    pmd: Sequence[PMDSection] | None = None,
    alpha: float,
    distance: float,
    max_nonlinear_phase: float = 0.005,
    max_walkoff_slip: float = 0.5,
    max_step: float | None = None,
    on_progress: Callable[[float], None] | None = None,
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

    **The coherent term, when asked for.** ``pairs`` names the two axes of each
    band as ``(x field, y field)``. With ``coherent_polarization`` those two are
    stepped together, including the ``(1/3) A_y**2 A_x*`` term that the
    phase-only form leaves out -- the one that moves power *between* the axes
    rather than only dephasing them. It is not a phase in the x/y basis, but in
    the circular basis ``A_{+-} = (A_x -+ i A_y) / sqrt(2)`` the self and
    cross-polarization terms become ``(2/3)(|A_{+-}|**2 + 2 |A_{-+}|**2)``, pure
    phases again, so the step stays exact.

    **Other bands turn a band's state, not only its phases.** Of the terms a
    neighbour ``p`` puts at a band's own frequency, the per-axis phases are only
    the diagonal: ``(s . p*) p`` -- and, with the coherent term, ``(p . s) p*`` --
    sit at the band's frequency exactly, do not oscillate, and mix its two axes.
    That is inter-channel cross-polarization modulation: a pump at 45 degrees
    precesses a probe's Stokes vector about its own, by ``(4/3) gamma P L`` with
    the coherent term and ``(2/3) gamma P L`` without. Each paired band is stepped
    through the exact exponential of its 2x2 Hermitian Kerr matrix; see
    :func:`_cross_polarization_coupling`. Neighbours whose axes are uncorrelated
    -- all on an axis -- leave the matrix diagonal, and the step is the per-axis
    phase it always was. What does average away are the terms that land
    elsewhere, ``p p s*``, which are four-wave mixing and handled as such.

    **PMD along the span, when asked for.** ``pmd`` sections are placed at the
    midpoints of equal lengths of the span and each is applied, to every pair,
    exactly where it sits -- the step is shortened to land on it. Between them the
    Kerr effect acts on a state of polarization that is still rotating, which is
    what applying the whole chain afterwards cannot represent. Applied in the order
    :func:`apply_pmd` applies them, so with ``gamma = 0`` the two agree exactly.

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
    couples = [] if pairs is None else [(int(x), int(y)) for x, y in pairs]
    for x_index, y_index in couples:
        if not (0 <= x_index < len(fields) and 0 <= y_index < len(fields)):
            raise ValueError(f"pair {(x_index, y_index)} indexes past the {len(fields)} fields")
        if axis[x_index] != 0 or axis[y_index] != 1:
            raise ValueError("each pair is (x field, y field): label the first 0 and the second 1")
    if (coherent_polarization or pmd) and not couples:
        raise ValueError(
            "the coherent polarization term and PMD along the span act on the two axes of a "
            "band together; name them with pairs"
        )
    partner: dict[int, int] = {}
    for x_index, y_index in couples:
        partner[x_index] = y_index
        partner[y_index] = x_index

    xp = array_module(*fields)
    a = [f.astype(xp.complex128, copy=True) for f in fields]
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

    omega = angular_frequency_grid(a[0].shape[0], sample_rate, like=a[0])
    # One expansion of beta(omega), one operator: the group delay, the dispersion
    # and its slope are the first three terms of the same series, which is what
    # fixes their relative signs. See propagate_dispersion.
    operators = [
        -1j * (0.5 * b * omega**2 + w * omega + b3 * omega**3 / 6.0)
        for b, w, b3 in zip(beta2, walkoff, slope, strict=True)
    ]
    ceiling = max_step if max_step is not None else distance
    sections = list(pmd or ())
    positions = [(index + 0.5) * distance / len(sections) for index in range(len(sections))]
    next_section = 0

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
                effective = 2.0 * _axis_power[ax] - xp.abs(field) ** 2
                other = _axis_power.get(1 - ax)
                if other is not None:
                    effective = effective + ORTHOGONAL_KERR_WEIGHT * other
                peak_effective = max(peak_effective, float(xp.max(effective)))
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
        if next_section < len(positions):
            # Land exactly on the next waveplate, so it acts where it sits.
            to_section = positions[next_section] - travelled
            if to_section > 0.0:
                step = min(step, to_section)

        half = [xp.exp(-alpha * step / 4.0 + op * (step / 2.0)) for op in operators]
        a = [xp.fft.ifft(xp.fft.fft(f) * h) for f, h in zip(a, half, strict=True)]

        if gamma != 0.0:
            # Summed per polarization, because power in the orthogonal component
            # modulates at a different rate than power sharing an axis. With one
            # axis in use the second sum is zero and this is the scalar model
            # term for term.
            per_axis = _power_per_axis(a, axis)
            rotated = list(a)
            phases = []
            for index, (field, ax) in enumerate(zip(a, axis, strict=True)):
                effective = 2.0 * per_axis[ax] - xp.abs(field) ** 2
                other = per_axis.get(1 - ax)
                if other is not None:
                    effective = effective + ORTHOGONAL_KERR_WEIGHT * other
                if coherent_polarization and index in partner:
                    # Only the other bands enter as a phase here. This band's own
                    # two axes are stepped together, exactly, just below.
                    effective = (
                        effective
                        - xp.abs(field) ** 2
                        - ORTHOGONAL_KERR_WEIGHT * xp.abs(a[partner[index]]) ** 2
                    )
                phase = gamma * effective * step
                peak_phase = max(peak_phase, float(xp.max(xp.abs(phase))))
                phases.append(phase)
            # What the phases above leave out: a neighbour whose two axes are
            # correlated turns this band's state of polarization, not only its
            # phase on each axis. See _cross_polarization_coupling.
            coupling = _cross_polarization_coupling(a, couples, coherent=coherent_polarization)
            stepped: set[int] = set()
            for (x_index, y_index), off in zip(couples, coupling, strict=True):
                if off is None:
                    continue
                rotated[x_index], rotated[y_index] = _hermitian_step(
                    a[x_index], a[y_index], phases[x_index], phases[y_index], gamma * step * off
                )
                peak_phase = max(peak_phase, float(xp.max(xp.abs(gamma * step * off))))
                stepped.update((x_index, y_index))
            for index, (field, phase) in enumerate(zip(a, phases, strict=True)):
                if index not in stepped:
                    rotated[index] = field * xp.exp(-1j * phase)
            if coherent_polarization:
                for x_index, y_index in couples:
                    rotated[x_index], rotated[y_index], turned = _coherent_kerr_step(
                        rotated[x_index], rotated[y_index], gamma * step
                    )
                    peak_phase = max(peak_phase, turned)
            a = rotated

        a = [xp.fft.ifft(xp.fft.fft(f) * h) for f, h in zip(a, half, strict=True)]

        travelled += step
        steps += 1
        shortest = min(shortest, step)
        longest = max(longest, step)
        peak_slip = max(peak_slip, spread * step * sample_rate)
        while next_section < len(positions) and positions[next_section] <= travelled * (
            1.0 + 1e-12
        ):
            a = _apply_pmd_section(a, couples, sections[next_section], omega)
            next_section += 1
        if on_progress is not None:
            # Distance, not step count. The step count is not known in advance —
            # the size is chosen each time from the peak power the fields
            # currently have — so a fraction of steps would need a denominator
            # that does not exist yet. Distance has one from the first line.
            on_progress(travelled / distance)

    if spread > 0.0:
        # Back into each channel's own retarded frame. This is the exact inverse
        # of the accumulated -1j*w*omega*distance, so it cannot remove anything
        # the propagation put there.
        a = [
            xp.fft.ifft(xp.fft.fft(f) * xp.exp(1j * w * omega * distance))
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


def _cross_polarization_coupling(
    fields: Sequence[np.ndarray], couples: Sequence[tuple[int, int]], *, coherent: bool
) -> list[np.ndarray | None]:
    """The off-diagonal entry the other bands put in each band's Kerr matrix [W].

    A band ``s`` beside a band ``p`` at another frequency is driven, at its own
    frequency, by the terms of ``|E|^2 E`` linear in ``s`` and quadratic in
    ``p``. With the isotropic tensor, ``(2/3)(E.E*)E + (1/3)(E.E)E*``, those are
    ``(2/3)[(p^H p) s + (p p^H) s + (p* p^T) s]``; with the phase-only form they
    are ``2|p_x|^2 + (2/3)|p_y|^2`` on the diagonal and ``(2/3) p_x p_y*`` off it.
    The diagonals are the per-axis phases every step already applies, so what is
    left is the off-diagonal: ``(2/3)(p_x p_y* + p_x* p_y)`` or ``(2/3) p_x p_y*``,
    summed over every band but ``s``.

    None of it oscillates. ``(s . p*) p`` sits at the frequency of ``s`` exactly,
    which is why a neighbour at 45 degrees turns a channel's state of
    polarization -- inter-channel cross-polarization modulation -- and why it
    cannot be dropped as a beat at the channel spacing. A band whose neighbours
    all sit on an axis gets ``None``, and the step it takes is the one it always
    took.
    """
    if len(couples) < 2:
        return [None] * len(couples)
    xp = array_module(*fields)
    terms = []
    for x_index, y_index in couples:
        px, py = fields[x_index], fields[y_index]
        cross = px * xp.conj(py)
        terms.append((2.0 / 3.0) * (cross + xp.conj(cross)) if coherent else (2.0 / 3.0) * cross)
    total = terms[0]
    for term in terms[1:]:
        total = total + term
    result: list[np.ndarray | None] = []
    for term in terms:
        off = total - term
        result.append(off if bool(xp.any(off != 0.0)) else None)
    return result


def _hermitian_step(
    ex: np.ndarray, ey: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """``exp(-i [[a, c], [c*, b]])`` applied to ``(ex, ey)``, sample by sample, exactly.

    With ``m = (a + b) / 2``, ``d = (a - b) / 2`` and ``w = sqrt(d^2 + |c|^2)``, the
    exponential of a 2x2 Hermitian matrix is
    ``exp(-i m) [cos w - i sin(w)/w (H - m)]``, and ``sin(w)/w`` goes to one as
    ``w`` does. The powers the matrix is built from do not change during the step
    -- it is unitary -- so this is the step's solution, not a Taylor term of it.
    """
    xp = array_module(ex, ey)
    mean = (a + b) / 2.0
    half = (a - b) / 2.0
    omega = xp.sqrt(half**2 + xp.abs(c) ** 2)
    cosine = xp.cos(omega)
    # sin(w)/w, with its limit where w is too small to divide by; the series'
    # next term is w^4/120, below double precision there.
    small = omega < 1e-4
    ratio = xp.where(small, 1.0 - omega**2 / 6.0, xp.sin(omega) / xp.where(small, 1.0, omega))
    common = xp.exp(-1j * mean)
    new_x = common * (cosine * ex - 1j * ratio * (half * ex + c * ey))
    new_y = common * (cosine * ey - 1j * ratio * (xp.conj(c) * ex - half * ey))
    return new_x, new_y


def _coherent_kerr_step(
    ex: np.ndarray, ey: np.ndarray, scale: float
) -> tuple[np.ndarray, np.ndarray, float]:
    """One band's own Kerr step with the coherent term, exact: ``(ex, ey, peak phase)``.

    In the circular basis the self and cross-polarization terms are phases,
    ``(2/3) scale (|A_{+-}|**2 + 2 |A_{-+}|**2)``, and the powers they depend on do
    not change during the step -- so rotating there and back is the solution,
    not an approximation of it. ``scale`` is ``gamma * step``.
    """
    xp = array_module(ex, ey)
    root = math.sqrt(2.0)
    plus = (ex - 1j * ey) / root
    minus = (ex + 1j * ey) / root
    power_plus = xp.abs(plus) ** 2
    power_minus = xp.abs(minus) ** 2
    turn_plus = (2.0 / 3.0) * scale * (power_plus + 2.0 * power_minus)
    turn_minus = (2.0 / 3.0) * scale * (power_minus + 2.0 * power_plus)
    plus = plus * xp.exp(-1j * turn_plus)
    minus = minus * xp.exp(-1j * turn_minus)
    peak = float(max(xp.max(xp.abs(turn_plus)), xp.max(xp.abs(turn_minus))))
    return (plus + minus) / root, 1j * (plus - minus) / root, peak


def _apply_pmd_section(
    fields: Sequence[np.ndarray],
    couples: Sequence[tuple[int, int]],
    section: PMDSection,
    omega: np.ndarray,
) -> list[np.ndarray]:
    """One waveplate on every ``(x, y)`` pair, in :func:`apply_pmd`'s own order."""
    xp = array_module(*fields)
    out = list(fields)
    phase = xp.exp(0.5j * omega * section.dgd)
    u = section.unitary
    for x_index, y_index in couples:
        delayed_x = xp.fft.fft(out[x_index]) * phase
        delayed_y = xp.fft.fft(out[y_index]) * xp.conj(phase)
        out[x_index] = xp.fft.ifft(u[0, 0] * delayed_x + u[0, 1] * delayed_y)
        out[y_index] = xp.fft.ifft(u[1, 0] * delayed_x + u[1, 1] * delayed_y)
    return out


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
    xp = array_module(fields[0])
    total = xp.abs(fields[0]) ** 2
    for field in fields[1:]:
        total += xp.abs(field) ** 2
    return total


#: Separation past which a triangular Raman gain profile stops being the truth.
#:
#: Silica's Raman gain rises roughly linearly with frequency separation, peaks
#: near 13.2 THz and falls away after it. Below this the linear approximation is
#: the standard one and is what :func:`raman_tilt` assumes. A comb wider than it
#: has its far pairs past the peak, where a straight line over-predicts the
#: transfer several times over, and the fibre block switches to
#: :func:`raman_transfer` with the measured shape instead.
#:
#: The C and L bands together are *not* that case: 1530 to 1610 nm is 9.7 THz. It
#: takes the S band as well — 1460 to 1625 nm, 20.8 THz — to cross the peak.
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


#: Silica's Raman gain against frequency separation, normalised to its peak.
#:
#: A coarse piecewise-linear trace of the measured spectrum (Stolen and Ippen,
#: *Appl. Phys. Lett.* 22, 1973; Agrawal, *Nonlinear Fiber Optics*, fig. 8.1), not
#: a fit. Below the peak it is exactly the straight line :func:`raman_tilt`
#: assumes, so the two models are the same device there. Past it, it follows the
#: steep fall with the 14.7 THz shoulder folded in, and the weak band near 24 THz.
#: Good to tens of percent past the peak, which is the region the straight line
#: gets wrong by several times.
RAMAN_SILICA_PROFILE: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),
    (RAMAN_TRIANGLE_LIMIT, 1.0),
    (15.0e12, 0.55),
    (18.0e12, 0.15),
    (24.0e12, 0.12),
    (30.0e12, 0.03),
    (40.0e12, 0.0),
)


def raman_gain(separation: np.ndarray, *, gain_slope: float, profile: str) -> np.ndarray:
    """Raman gain coefficient over effective area at ``separation`` [Hz], in 1/(W·m).

    ``"triangle"`` is ``gain_slope * separation`` at every separation, which is
    what the closed form is written in. ``"silica"`` is the same line up to the
    peak and :data:`RAMAN_SILICA_PROFILE` past it.
    """
    offsets = np.abs(np.asarray(separation, dtype=float))
    if profile == "triangle":
        return gain_slope * offsets
    if profile == "silica":
        xs = [point[0] for point in RAMAN_SILICA_PROFILE]
        ys = [point[1] for point in RAMAN_SILICA_PROFILE]
        peak = gain_slope * RAMAN_TRIANGLE_LIMIT
        return peak * np.interp(offsets, xs, ys, right=0.0)
    raise ValueError(f"unknown Raman profile {profile!r}; expected 'triangle' or 'silica'")


def raman_coupling(
    frequencies: Sequence[float],
    *,
    gain_slope: float,
    profile: str = "silica",
    photon_conserving: bool = True,
) -> np.ndarray:
    """``C[n, m]``: how channel ``n``'s power grows per watt of channel ``m`` [1/(W m)].

    Positive where ``m`` sits at a higher frequency than ``n`` and so pumps it,
    negative where it sits below and is pumped by ``n`` -- with ``photon_conserving``
    the loss carrying the factor ``f_n / f_m`` the quantum defect asks for -- and
    zero on the diagonal. ``dP_n/dz = P_n sum_m C[n, m] P_m`` is the redistribution
    :func:`raman_transfer` integrates and the tone solver adds to its amplitudes.
    """
    f = np.asarray(frequencies, dtype=float)
    separation = f[None, :] - f[:, None]  # [n, m] = f_m - f_n
    gain = raman_gain(separation, gain_slope=gain_slope, profile=profile)
    coupling = np.where(separation > 0.0, gain, -gain)
    if photon_conserving:
        coupling = np.where(separation < 0.0, coupling * (f[:, None] / f[None, :]), coupling)
    np.fill_diagonal(coupling, 0.0)
    return np.asarray(coupling)


def raman_transfer(
    frequencies: Sequence[float],
    powers: Sequence[float],
    *,
    gain_slope: float,
    effective_length: float,
    profile: str = "silica",
    photon_conserving: bool = True,
    steps: int | None = None,
) -> list[float]:
    """Power each channel keeps after stimulated Raman scattering, integrated.

    The coupled power equations, solved numerically where :func:`raman_tilt`
    has a closed form::

        dP_n/dz = -alpha P_n + P_n sum_{f_m > f_n} g(f_m - f_n) P_m
                              - P_n sum_{f_m < f_n} (f_n / f_m) g(f_n - f_m) P_m

    The loss is common to every channel, so writing ``P = exp(-alpha z) Q`` and
    measuring distance in effective length removes it exactly, and what is
    integrated — by fourth-order Runge-Kutta over ``effective_length`` — is the
    redistribution alone. The return value means what :func:`raman_tilt`'s does.

    Two things the closed form cannot do. **A real gain shape**: with
    ``profile="silica"`` the gain falls past 13.2 THz instead of rising forever,
    which is what a comb wider than :data:`RAMAN_TRIANGLE_LIMIT` needs.
    **Photons rather than watts**: with ``photon_conserving`` the pump loses
    ``f_n / f_m`` times what the signal gains, and the lattice keeps the
    difference. Then ``sum(P_n / f_n)`` is what is conserved, and the total power
    falls by the quantum defect.

    With ``profile="triangle"`` and ``photon_conserving=False`` this *is* the
    closed form, to the integrator's error, and the tests hold it there.
    """
    if len(frequencies) != len(powers):
        raise ValueError(
            f"one power per frequency, got {len(powers)} for {len(frequencies)} channels"
        )
    count = len(frequencies)
    launched = np.asarray(powers, dtype=float)
    total = float(launched.sum())
    if gain_slope == 0.0 or total <= 0.0 or count < 2 or effective_length <= 0.0:
        return [1.0] * count

    coupling = raman_coupling(
        frequencies, gain_slope=gain_slope, profile=profile, photon_conserving=photon_conserving
    )

    if steps is None:
        rate = float(np.abs(coupling).sum(axis=1).max()) * total * effective_length
        steps = int(min(20000, max(64, math.ceil(40.0 * rate))))
    h = effective_length / steps

    def slope(q: np.ndarray) -> np.ndarray:
        return q * (coupling @ q)

    q = launched.copy()
    for _ in range(steps):
        k1 = slope(q)
        k2 = slope(q + 0.5 * h * k1)
        k3 = slope(q + 0.5 * h * k2)
        k4 = slope(q + h * k3)
        q = q + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return [float(q[n] / launched[n]) if launched[n] > 0.0 else 1.0 for n in range(count)]


def band_to_grid(field: np.ndarray, factor: int) -> np.ndarray:
    """A band-limited window resampled ``factor`` times finer, amplitudes kept.

    Zero-padding the spectrum, which is exact for a window that is periodic and
    band-limited, as the simulated ones are taken to be. ``factor`` times the samples
    over the same span of time, so the sample rate rises by ``factor`` too.
    """
    n = int(field.shape[-1])
    if factor == 1:
        return np.asarray(field, dtype=np.complex128)
    spectrum = np.fft.fft(np.asarray(field, dtype=np.complex128))
    padded = np.zeros(n * factor, dtype=np.complex128)
    half = n // 2
    padded[:half] = spectrum[:half]
    padded[-half:] = spectrum[-half:]
    return np.asarray(np.fft.ifft(padded) * factor)


def grid_to_band(
    field: np.ndarray,
    samples: int,
    *,
    sample_rate: float,
    offset: float,
    half_width: float,
    below: float | None = None,
    above: float | None = None,
) -> np.ndarray:
    """The band ``offset`` [Hz] from the grid's centre, ``half_width`` [Hz] either side, cut out.

    The inverse of :func:`band_to_grid` for one band: shifted down by ``offset``,
    everything past ``half_width`` of it dropped (an ideal filter, so neighbours that
    are further apart than twice it do not leak), and the rest brought back to
    ``samples`` points over the same span of time, amplitudes kept. ``below`` and
    ``above``, where given, set the two edges apart, so that slots of unequal width
    can tile a spectrum with no gap between them. The lower edge is kept and the
    upper one dropped, so that two slots sharing an edge do not both take its bin.
    """
    total = int(field.shape[-1])
    factor = total // samples
    time = np.arange(total) / sample_rate
    spectrum = np.fft.fft(field * np.exp(-2j * np.pi * offset * time))
    frequency = np.fft.fftfreq(total, d=1.0 / sample_rate)
    if below is None and above is None:
        spectrum = np.where(np.abs(frequency) <= half_width, spectrum, 0.0)
    else:
        low = half_width if below is None else below
        high = half_width if above is None else above
        spectrum = np.where((frequency >= -low) & (frequency < high), spectrum, 0.0)
    half = samples // 2
    small = np.concatenate([spectrum[:half], spectrum[-half:]])
    return np.asarray(np.fft.ifft(small) / factor)


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


def apply_group_delay(field: np.ndarray, sample_rate: float, delay: float) -> np.ndarray:
    """Delay a sampled waveform by ``delay`` seconds [s], positive for later.

    The group-delay term of the same expansion whose quadratic term
    :func:`propagate_dispersion` applies, and carried with the same
    ``exp(-i beta z)`` convention: a constant delay is ``exp(-i omega tau)`` on
    the spectrum. Being all-pass it conserves energy exactly and is exactly
    invertible, and it is not restricted to whole samples -- a fraction of one is
    the interpolation this ramp implies, which is the right one for a waveform
    the sampling theorem already says is band-limited.

    **Circular, as every waveform here is.** What leaves the end of the window
    arrives at its start. That is the same convention the rest of the library
    runs on, and it is exact for the periodic signal a finite window stands for.

    This is deliberately *not* applied inside a band: there a common delay is
    unobservable. It is what one band has walked relative to another, which is
    what :class:`maiman.signals.WalkoffHistory` carries to the detector.
    """
    if delay == 0.0:
        return field
    xp = array_module(field)
    omega = angular_frequency_grid(int(field.shape[-1]), sample_rate, like=field)
    ramp = xp.exp(-1j * omega * delay)
    delayed = xp.fft.ifft(xp.fft.fft(field) * ramp)
    if not xp.issubdtype(field.dtype, xp.complexfloating):
        return xp.real(delayed)
    return delayed


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


def fwm_efficiency(
    phase_mismatch: float, alpha: float, distance: float, nonlinear_rate: float = 0.0
) -> float:
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

    With ``nonlinear_rate`` the pumps' own Kerr phase enters the mismatch as well
    (see :func:`fwm_mixing_integral`), and the efficiency can then exceed 1 where
    it undoes a linear mismatch -- nonlinear phase matching.
    """
    if distance <= 0.0:
        return 0.0
    reference = effective_length(alpha, distance)
    if reference <= 0.0:
        return 0.0
    integral = fwm_mixing_integral(phase_mismatch, alpha, distance, nonlinear_rate)
    return float(abs(integral) ** 2 / reference**2)


#: Largest ``|r / alpha|`` the mixing integral is summed as a series at. Beyond it
#: the terms grow before they fall and cancel each other; quadrature takes over.
MIXING_SERIES_LIMIT = 8.0


def kerr_rate(
    gamma: float,
    own: tuple[float, float],
    total: tuple[float, float],
    *,
    cross_phase: bool,
    orthogonal_weight: float = 0.0,
) -> tuple[float, float]:
    """How fast the split-step turns one carrier, per axis [rad/m], ``exp(-i rate z)``.

    :func:`propagate_coupled_ssfm`'s ``gamma (2 P_same - |A|^2 + w P_other)`` for a
    carrier of ``own`` power among ``total`` when the bands share one solve; its own
    ``gamma (P + w P_other_own)`` when each is propagated alone. ``w`` is
    ``orthogonal_weight``, two thirds with the polarizations coupled and zero
    without. ``own = (0, 0)`` is a weak carrier: a mixing product at a new
    frequency.
    """
    x, y = own
    if cross_phase:
        return (
            gamma * (2.0 * total[0] - x + orthogonal_weight * total[1]),
            gamma * (2.0 * total[1] - y + orthogonal_weight * total[0]),
        )
    return gamma * (x + orthogonal_weight * y), gamma * (y + orthogonal_weight * x)


#: How nearly one state of polarization a band has to be for its Kerr phase to
#: be taken in that state. A band carrying two independent tributaries has
#: none, and is left to the per-axis form.
POLARIZED = 0.999


def degree_of_polarization(state: np.ndarray) -> float:
    """``|S| / S0`` of a 2x2 coherency matrix: one for a single state, zero for none."""
    total = float(np.trace(state).real)
    if total <= 0.0:
        return 0.0
    spread = math.sqrt(float((state[0, 0] - state[1, 1]).real) ** 2 + 4.0 * abs(state[0, 1]) ** 2)
    return spread / total


#: Below this fraction of a state's power on x, its phase is taken on y: a state
#: with nothing on x has no phase there. A global phase cannot be chosen
#: continuously over every polarization, so the break is put where the light is
#: not -- rather than, say, on the circle of equal powers where circular and
#: diagonal light sit.
PHASE_REFERENCE_FLOOR = 1e-9


def phase_reference(vector: np.ndarray) -> int:
    """The axis a polarization's phase is read on: x, unless there is nothing on x."""
    v = np.asarray(vector)
    total = float(np.sum(np.abs(v) ** 2))
    return 0 if abs(v[0]) ** 2 > PHASE_REFERENCE_FLOOR * total else 1


def principal_state(state: np.ndarray) -> np.ndarray:
    """The unit Jones vector a coherency matrix is mostly made of, real on x.

    A coherency matrix carries no global phase, so one is chosen: the component
    on :func:`phase_reference`'s axis is real and positive. Every angle a Kerr
    history keeps is a phase in this gauge, which is what lets a state that
    moves be followed from one span to the next.
    """
    values, vectors = np.linalg.eigh(np.asarray(state, dtype=np.complex128))
    vector = vectors[:, int(np.argmax(values))]
    lead = vector[phase_reference(vector)]
    return vector * (abs(lead) / lead) if lead != 0 else vector


def geometric_phase(start: np.ndarray, end: np.ndarray) -> float:
    """What a state moving from ``start`` to ``end`` adds to its phase in that gauge [rad].

    A wave ``exp(-i phi) u`` with ``u`` held real on x, under ``dA/dz = -i H A``,
    turns at ``u^H H u + Im(u^H du/dz)``: the Kerr rate in its state, and a
    geometric term that is zero for a state that stays put and is
    ``sin^2(a) d(delta)`` for ``u = (cos a, sin a e^(i delta))`` -- Pancharatnam's
    connection (Berry, J. Mod. Opt. 34, 1401 (1987)). Over a short move it is
    ``arg(u_start^H u_end)``, the move taken along the shortest path; returned in
    the history's sense, ``exp(-i angle)``.
    """
    overlap = complex(np.vdot(np.asarray(start), np.asarray(end)))
    return float(np.angle(overlap)) if overlap != 0 else 0.0


def cross_kerr_matrix(state: np.ndarray, *, coherent: bool) -> np.ndarray:
    """What a band of coherency ``J`` does to a carrier at another frequency [W], as a matrix.

    ``(2/3)[tr(J) I + J + J*]`` with the coherent term -- ``(p^H p) + p p^H + p* p^T``
    for a polarized ``p`` -- and, phase-only, ``2 J_aa + (2/3) J_bb`` on the
    diagonal with ``(2/3) J_ab`` off it. The same matrices the coupled split-step
    steps a band through (:func:`_cross_polarization_coupling`), averaged.
    """
    j = np.asarray(state, dtype=np.complex128)
    if coherent:
        return (2.0 / 3.0) * (np.trace(j) * np.eye(2) + j + np.conj(j))
    return np.array(
        [
            [2.0 * j[0, 0] + (2.0 / 3.0) * j[1, 1], (2.0 / 3.0) * j[0, 1]],
            [(2.0 / 3.0) * j[1, 0], 2.0 * j[1, 1] + (2.0 / 3.0) * j[0, 0]],
        ]
    )


def self_kerr(state: np.ndarray, *, coherent: bool) -> float:
    """A polarized band's own Kerr rate over ``gamma`` [W], in its own state.

    Isotropic: ``(2/3 + |u^T u|^2 / 3) P`` -- one for linear light, two thirds
    for circular. Phase-only: ``(|u_x|^4 + |u_y|^4 + (4/3)|u_x|^2 |u_y|^2) P``,
    each axis turned by its own power and two thirds of the other's.
    """
    power = float(np.trace(state).real)
    if power <= 0.0:
        return 0.0
    u = principal_state(state)
    if coherent:
        return (2.0 / 3.0 + abs(complex(u @ u)) ** 2 / 3.0) * power
    x, y = abs(u[0]) ** 2, abs(u[1]) ** 2
    return (x * x + y * y + (4.0 / 3.0) * x * y) * power


def kerr_rate_in_state(
    gamma: float,
    states: Sequence[np.ndarray],
    probe: np.ndarray,
    *,
    own: int | None,
    coherent: bool,
    cross_phase: bool,
) -> float:
    """How fast the split-step turns a carrier in state ``probe`` [rad/m], ``exp(-i rate z)``.

    Every other band's :func:`cross_kerr_matrix` read in the probe's state, ``q^H M q``,
    when the bands share one solve; plus ``own``'s :func:`self_kerr` when the probe
    is that band. On an axis this is :func:`kerr_rate`'s ``2 P_same - |A|^2 + w
    P_other`` term for term. Off it, a 45-degree beam is turned by all of its
    power where the per-axis form turns each axis by five sixths.
    """
    q = np.asarray(probe, dtype=np.complex128)
    q = q / np.linalg.norm(q)
    rate = 0.0
    if cross_phase:
        for index, state in enumerate(states):
            if index != own:
                rate += float((np.conj(q) @ cross_kerr_matrix(state, coherent=coherent) @ q).real)
    if own is not None:
        rate += self_kerr(states[own], coherent=coherent)
    return gamma * rate


def fwm_nonlinear_rate(
    gamma: float,
    power_i: tuple[float, float],
    power_j: tuple[float, float],
    power_k: tuple[float, float],
    power_f: tuple[float, float] = (0.0, 0.0),
    *,
    cross_phase: bool,
    orthogonal_weight: float = 0.0,
) -> tuple[float, float]:
    """How fast the Kerr effect walks a product off its drive, per axis [rad/m].

    The product at ``f_i + f_j - f_k`` is driven by ``A_i A_j A_k*``, which the
    split-step turns at ``rate_i + rate_j - rate_k`` (:func:`kerr_rate`), and the
    carrier it lands on turns at its own ``rate_f``. The mixing integral needs the
    difference at the start of the span; the loss carries it down the span as the
    powers fall. Written out, on axis ``a``:

    * **cross-phase on**, every band in one solve: ``-gamma (P_i + P_j - P_k -
      P_f)``. The ``2T`` of cross-phase and the orthogonal axis's ``w T`` turn
      every carrier alike, and cancel. That is the textbook ``P1 + P2 - P3 - P4``.
    * **cross-phase off**, each band alone: ``+gamma (Q_a + w Q_b)`` with
      ``Q = P_i + P_j - P_k - P_f``, each carrier turned by its own power only.

    ``power_f`` is the power already at the product's frequency -- zero for a
    product at a new one, a channel's for one landing on it. ``power_*`` are
    ``(x, y)`` at the span's start [W], and the sign is the one the integral's
    ``delta_beta`` has. Checked against the split-step solution of a strong pump
    and a weak signal, whose product it brings from 12--29 % off to 2 % at a
    quarter of a radian of nonlinear phase.
    """
    # The total turns every carrier alike and cancels in ``i + j - k - f``, so it
    # is left at zero: what is left of each carrier's rate is its own part.
    rates = [
        kerr_rate(
            gamma, power, (0.0, 0.0), cross_phase=cross_phase, orthogonal_weight=orthogonal_weight
        )
        for power in (power_i, power_j, power_k, power_f)
    ]
    return (
        rates[0][0] + rates[1][0] - rates[2][0] - rates[3][0],
        rates[0][1] + rates[1][1] - rates[2][1] - rates[3][1],
    )


def fwm_mixing_integral(
    phase_mismatch: float, alpha: float, distance: float, nonlinear_rate: float = 0.0
) -> complex:
    """``integral_0^L exp((i delta_beta - alpha) z + i r L_eff(z)) dz`` [m], with its phase kept.

    :func:`fwm_efficiency` is this integral's squared magnitude and throws the
    argument away, which is all a single span needs: one span's product has
    whatever phase it has. A link of several spans needs the argument, because
    what decides whether the spans' contributions add or cancel is the phase of
    each relative to the others — see :func:`fwm_accumulated_phase`.

    Evaluated as the integral rather than as its expanded real form because that
    stays well behaved in both limits: lossless, where the expanded expression is
    0/0 and the answer is ``L sinc(delta_beta L / 2)``, and phase matched, where
    it is ``L_eff``.

    **The pumps' own phase.** ``r`` is :func:`fwm_nonlinear_rate`: self- and
    cross-phase modulation turn the drive and the product at different rates,
    and as the pumps decay that difference accumulates as ``r L_eff(z)`` rather
    than ``r z``. Lossless, it only shifts the mismatch to ``delta_beta + r``,
    closed form. With loss it is summed as a series in ``x = r / alpha`` --
    ``exp(i x (1 - e^{-alpha z}))`` expanded in powers of ``e^{-alpha z}``, each
    term a closed-form exponential integral -- and past
    :data:`MIXING_SERIES_LIMIT` by composite Gauss-Legendre quadrature, fine
    enough to resolve every turn of the integrand. The two agree to 1e-12 where
    both apply, which the tests hold them to.
    """
    rate = complex(-alpha, phase_mismatch)
    if nonlinear_rate == 0.0:
        if abs(rate) * distance < 1e-9:
            return complex(distance)
        return complex((np.exp(rate * distance) - 1.0) / rate)
    if alpha * distance < 1e-9:
        shifted = complex(-alpha, phase_mismatch + nonlinear_rate)
        if abs(shifted) * distance < 1e-9:
            return complex(distance)
        return complex((np.exp(shifted * distance) - 1.0) / shifted)
    if abs(nonlinear_rate / alpha) <= MIXING_SERIES_LIMIT:
        return _mixing_series(phase_mismatch, alpha, distance, nonlinear_rate)
    return _mixing_quadrature(phase_mismatch, alpha, distance, nonlinear_rate)


def _mixing_series(
    phase_mismatch: float, alpha: float, distance: float, nonlinear_rate: float
) -> complex:
    """The lossy mixing integral, summed term by term as a series in ``x = r / alpha``.

    ``e^{ix} sum_n (-ix)^n / n! int_0^L e^{(i delta_beta - (n + 1) alpha) z} dz``.
    """
    x = nonlinear_rate / alpha
    total = 0j
    coefficient = 1.0 + 0j
    for n in range(400):
        exponent = complex(-(n + 1) * alpha, phase_mismatch)
        term = coefficient * (np.exp(exponent * distance) - 1.0) / exponent
        total += term
        if n > abs(x) and abs(term) < 1e-17 * max(abs(total), 1e-300):
            break
        coefficient *= -1j * x / (n + 1)
    return complex(np.exp(1j * x) * total)


def _mixing_quadrature(
    phase_mismatch: float, alpha: float, distance: float, nonlinear_rate: float
) -> complex:
    """The same integral by composite Gauss-Legendre, half a radian of phase per panel."""
    turning = abs(phase_mismatch) + abs(nonlinear_rate) + alpha
    panels = max(16, math.ceil(distance * turning / 0.5))
    nodes, weights = np.polynomial.legendre.leggauss(8)
    edges = np.linspace(0.0, distance, panels + 1)
    half = 0.5 * np.diff(edges)
    z = (0.5 * (edges[:-1] + edges[1:]))[:, None] + half[:, None] * nodes[None, :]
    length = -np.expm1(-alpha * z) / alpha
    values = np.exp(complex(-alpha, phase_mismatch) * z + 1j * nonlinear_rate * length)
    return complex(np.sum(values * weights[None, :] * half[:, None]))


def fwm_accumulated_phase(
    accumulated_gvd: float, offset_i: float, offset_j: float, offset_k: float
) -> float:
    """How far a mixing product has rotated away from its pumps [rad].

    A product generated in the second span of a link is not in phase with the one
    generated in the first. Over the distance already travelled, the pump
    combination and the product have advanced at different rates, and the
    difference is exactly the phase mismatch integrated over that distance::

        delta_phi = integral delta_beta dz = -(sum beta2 L) * (w_i - w_k)(w_j - w_k)

    **This is why one scalar is enough to carry it.** The mismatch is a
    difference of four propagation constants at frequencies satisfying
    ``w_i + w_j = w_k + w_F``, so the constant and group-delay terms cancel
    identically and only the ``beta2`` term survives — see
    :func:`fwm_phase_mismatch`. Nothing about a band's absolute phase has to be
    tracked, which is fortunate, because ``beta_0 L`` is of order 1e11 radians
    over 80 km and reducing that modulo 2*pi in double precision would leave
    about five digits of the answer.

    Implemented as :func:`fwm_phase_mismatch` evaluated at the *accumulated* GVD
    rather than at ``beta2``: the two expressions are the same one, differing
    only in whether the length has been folded in, and writing it twice is how
    they would come to disagree.
    """
    return fwm_phase_mismatch(accumulated_gvd, offset_i, offset_j, offset_k)


def fwm_cascade_integral(
    delta_beta1: float, delta_beta2: float, alpha: float, distance: float
) -> complex:
    """A product driving a further product, within the span that made it [m^2].

    :func:`fwm_mixing_integral` is a single triad's build-up, ``A_i A_j A_k*``
    against a decaying, phase-slipping product. Undepleted-pump perturbation
    theory does not stop there: the product it makes, ``A_F(z)``, is itself a
    wave, and while the two original pumps are still present it drives a second
    product exactly as any other triad would -- the process the split-step
    resolves for free and the multi-band model, projecting each span to a set
    of final amplitudes, has always thrown away. Thompson & Roy call this
    *multiple four-wave mixing* and give the cascade to all orders for two
    equal pumps (Phys. Rev. A 43, 4987 (1991)); Cappellini & Trillo solve the
    three-wave case exactly (JOSA B 8, 824 (1991)). This is the second order of
    that cascade, generic in the two triads' mismatches and in the loss.

    Writing the first product's build-up as
    ``M1(z) = integral_0^z exp((i*delta_beta1 - alpha) z') dz'`` -- the same
    integrand :func:`fwm_mixing_integral` integrates to ``L``, here left
    running -- the second product obeys the same undepleted-pump equation
    driven by ``A_F(z)`` in place of a launched pump, and its own build-up is::

        integral_0^L M1(z) exp((i*delta_beta2 - alpha) z) dz

    which is what this function returns. Two nested exponentials integrate in
    closed form for any ``delta_beta1``, ``delta_beta2`` and ``alpha`` -- no
    series or quadrature needed, unlike :func:`fwm_mixing_integral` with a
    pump-phase rate folded in, because here the "rate" a product turns at
    inside the integral is a second *phase mismatch*, constant along ``z``,
    not a nonlinear rate riding the loss-shaped ``L_eff(z)``. ``delta_beta1``
    and ``delta_beta2`` may each carry a pump-phase correction added in by the
    caller when that correction is itself constant along ``z`` -- exact at
    ``alpha = 0``, where :func:`fwm_mixing_integral` folds a nonlinear rate in
    the same way for the same reason.

    Each stage's integral is one of two closed forms, chosen so both limits
    stay finite: ``phi(r, L) = (exp(rL) - 1) / r``, or ``L`` where that is
    ``0/0``, for the outer stage and any inner stage with a nonzero rate; and
    ``psi(r, L) = integral_0^L z exp(rz) dz``, or ``L**2/2`` at ``r = 0``, for
    the one case ``phi`` cannot reach -- a first mismatch of exactly zero,
    where ``M1(z) = z`` is linear rather than exponential.
    """
    r1 = complex(-alpha, delta_beta1)
    r2 = complex(-alpha, delta_beta2)

    def phi(rate: complex, length: float) -> complex:
        if abs(rate) * length < 1e-9:
            return complex(length)
        return complex((np.exp(rate * length) - 1.0) / rate)

    def psi(rate: complex, length: float) -> complex:
        if abs(rate) * length < 1e-9:
            return complex(length * length / 2.0)
        return complex(((rate * length - 1.0) * np.exp(rate * length) + 1.0) / (rate * rate))

    if abs(r1) * distance < 1e-9:
        return psi(r2, distance)
    return (phi(r1 + r2, distance) - phi(r2, distance)) / r1


def fwm_cascade_amplitude(
    power_i: float,
    power_j: float,
    power_k: float,
    power_p: float,
    power_q: float,
    *,
    gamma: float,
    alpha: float,
    distance: float,
    phase_mismatch_1: float,
    phase_mismatch_2: float,
    degenerate_1: bool,
    degenerate_2: bool,
    conjugate_first: bool = False,
) -> complex:
    """The second-order product's own complex field amplitude [sqrt(W)].

    ``i``, ``j``, ``k`` are the first triad -- the pumps that make the
    first-order product this one is cascaded from -- and ``p``, ``q`` are the
    second triad's other two legs, real launched carriers both. Where
    :func:`fwm_product_power` returns a *power* built from the undepleted-pump
    formula, this returns an *amplitude*, because the point of computing it is
    to add it coherently to whatever else lands on its frequency -- another
    second-order term from a different route, or, across spans, whatever the
    ordinary first-order treatment of :meth:`Fiber._mix` puts there next.

    Its squared magnitude is the power,
    ``d1**2 d2**2 gamma**4 P_i P_j P_k P_p P_q exp(-alpha L) |Omega|**2`` with
    ``Omega`` the cascade integral, the loss of the product's own propagation
    and of all five legs folded into it; the sign and the
    factor of ``gamma**2`` come from applying the single-triad field equation
    (:func:`fwm_efficiency`'s docstring) twice, the second time with the first
    product standing in for a launched pump. ``d1``, ``d2`` are 1 for a
    degenerate triad (two of its three legs the same carrier) and 2 otherwise,
    the same degeneracy :func:`fwm_product_power` takes as ``degenerate``.

    **The first product can drive the second either un-conjugated or
    conjugated**, and both have to be added: a triplet's nonlinear polarization
    is built from two un-conjugated factors and one conjugated one, and the
    first-order product can sit in either role. Un-conjugated (``F`` beside a
    launched pump, landing at ``f_F + f_p - f_q``) is the default; conjugated
    (two launched pumps beside ``F*``, landing at ``f_p + f_q - f_F``) is
    ``conjugate_first=True``, which flips the sign the two applications of
    ``i`` in the field equation leave (``i * i = -1`` un-conjugated,
    ``i * (-i) = +1`` conjugated -- the second stage drives off ``A_F(z)*``,
    and conjugating a wave conjugates its own growth rate too, which is why
    ``phase_mismatch_1`` enters negated in that case) and negates
    ``phase_mismatch_1`` inside the cascade integral to match. Missing this
    term entirely -- only the un-conjugated one -- left a two-pump second-order
    product 33% high in the ``P -> 0`` limit, where every higher-order
    correction this truncation also drops has vanished and the shortfall can
    only be a missing term (maiman-z8j).
    """
    if min(power_i, power_j, power_k, power_p, power_q) <= 0.0:
        return 0j
    d1 = 1.0 if degenerate_1 else 2.0
    d2 = 1.0 if degenerate_2 else 2.0
    first_mismatch = -phase_mismatch_1 if conjugate_first else phase_mismatch_1
    omega = fwm_cascade_integral(first_mismatch, phase_mismatch_2, alpha, distance)
    magnitude = d1 * d2 * gamma**2 * math.sqrt(power_i * power_j * power_k * power_p * power_q)
    sign = 1.0 if conjugate_first else -1.0
    return complex(sign * magnitude * math.exp(-alpha * distance / 2.0) * omega)


#: Butcher tableau of the Dormand-Prince 5(4) pair, the embedded Runge-Kutta the
#: tone solver steps with (Dormand and Prince, J. Comput. Appl. Math. 6, 19 (1980)).
_DP_A = (
    (),
    (1 / 5,),
    (3 / 40, 9 / 40),
    (44 / 45, -56 / 15, 32 / 9),
    (19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729),
    (9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656),
    (35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84),
)
_DP_C = (0.0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0)
_DP_B5 = (35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0.0)
_DP_B4 = (5179 / 57600, 0.0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40)

#: Two offsets closer than this [Hz] are the same tone.
TONE_TOLERANCE = 1e3


def fwm_tone_triples(offsets: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Every ordered ``(i, j, k, q)`` of tones with ``f_i + f_j - f_k = f_q``.

    Ordered, so a pair of different tones appears twice -- the factor of two a
    cross term has in ``|A|^2 A`` -- and a cross-phase or self-phase term
    (``k = i`` or ``k = j``) is in the list too, with no mismatch. Only the
    triples whose product is one of ``offsets`` are kept.
    """
    f = np.asarray(offsets, dtype=np.float64)
    n = f.size
    keys = np.round(f / TONE_TOLERANCE).astype(np.int64)
    index = {int(key): position for position, key in enumerate(keys)}
    ti, tj, tk, tq = [], [], [], []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                q = index.get(round((f[i] + f[j] - f[k]) / TONE_TOLERANCE))
                if q is not None:
                    ti.append(i)
                    tj.append(j)
                    tk.append(k)
                    tq.append(q)
    return (
        np.array(ti, dtype=np.intp),
        np.array(tj, dtype=np.intp),
        np.array(tk, dtype=np.intp),
        np.array(tq, dtype=np.intp),
    )


def fwm_tone_solve(
    offsets: np.ndarray,
    amplitudes: np.ndarray,
    *,
    beta2: float,
    gamma: float,
    alpha: float,
    distance: float,
    accumulated_gvd: float = 0.0,
    rtol: float = 1e-9,
    triples: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
    coherent: bool = True,
    raman: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    """Constant-amplitude tones through a span, every mixing order at once [sqrt(W)].

    The coupled-mode equations for tones ``a_n`` at frequency offsets ``f_n``, with
    the linear phase ``beta(omega_n) z`` taken out of each::

        da_q/dz = -alpha/2 a_q + i gamma sum a_i a_j a_k* exp(i delta_beta z)

    over the ordered triples with ``f_i + f_j - f_k = f_q`` (Agrawal, *Nonlinear
    Fiber Optics*, ch. 10). Self- and cross-phase modulation are the terms with
    ``k = i`` or ``k = j``, which have no mismatch; every other term is mixing.
    Nothing is undepleted, nothing truncated in order: a product drives whatever
    it can, the pumps give up what the products take, and the fields are exact
    to the integrator's tolerance. Total power is conserved to it at ``alpha = 0``.

    ``accumulated_gvd`` is what the fibre behind this span has added, so that the
    phase a mismatch has already turned through is not lost from one span to the
    next (:func:`fwm_accumulated_phase`). The amplitudes are in the convention
    of the equation above, ``exp(+i beta z)``; a caller whose fields turn the
    other way conjugates on the way in and out. Returns the amplitudes at the end
    of the span and the number of steps taken.

    **With two polarizations.** Give ``amplitudes`` as ``(tones, 2)`` -- each tone a
    Jones vector -- and the Kerr drive is silica's isotropic tensor, the cubic
    expansion of ``(2/3)(E . E*) E + (1/3)(E . E) E*``::

        da_q/dz = ... + i gamma sum [(2/3)(a_i . a_k*) a_j + (1/3)(a_i . a_j) a_k*] e^{i db z}

    over the same ordered triples, so one polarization reduces to the scalar
    equation identically, circular light turns at two thirds of the rate linear
    light does, and a tone's cross-phase on another is ``(2/3)[tr J + J + J*]``, the
    matrix :func:`cross_kerr_matrix` gives. With ``coherent`` false the coherent
    term ``(1/3)(a_i . a_j) a_k*`` is dropped where both pumps sit on the other
    axis, leaving the phase-only form the split-step's own default keeps,
    ``A_x A_x A_x* + (2/3)(A_y A_y*) A_x`` per axis. Power over both polarizations
    is conserved at ``alpha = 0``, either way.

    **Raman.** ``raman`` is the ``(tones, tones)`` matrix :func:`raman_coupling`
    gives, and adds ``(1/2) (C P)_n a_n`` to each amplitude -- the power equation
    ``dP_n/dz = P_n sum_m C[n, m] P_m`` written for a field, which has no phase of
    its own to add: scattering is incoherent, so it moves power and leaves the
    mixing and cross-phase terms as they were. ``P_m`` is a tone's power over both
    polarizations. With photons conserved the total power falls by the quantum
    defect; without, it is conserved.
    """
    f = np.asarray(offsets, dtype=np.float64)
    y0 = np.asarray(amplitudes, dtype=np.complex128)
    vector = y0.ndim == 2
    if f.ndim != 1 or y0.shape not in ((f.size,), (f.size, 2)):
        raise ValueError(
            "offsets are one-dimensional, and amplitudes one number a tone or a Jones vector each"
        )
    if distance <= 0.0 or f.size == 0:
        return y0.copy(), 0
    ti, tj, tk, tq = triples if triples is not None else fwm_tone_triples(f)
    delta = np.array(
        [fwm_phase_mismatch(beta2, f[i], f[j], f[k]) for i, j, k in zip(ti, tj, tk, strict=True)]
    )
    start = np.array(
        [
            fwm_accumulated_phase(accumulated_gvd, f[i], f[j], f[k])
            for i, j, k in zip(ti, tj, tk, strict=True)
        ]
    )
    n = f.size
    scale = float(np.max(np.abs(y0))) if y0.size else 0.0
    if scale == 0.0:
        return y0.copy(), 0
    atol = 1e-14 * scale

    def scattering(a: np.ndarray) -> np.ndarray:
        """``(1/2) (C P)_n a_n``: the Raman gain each tone's amplitude feels."""
        if raman is None:
            return np.zeros_like(a)
        power = np.abs(a) ** 2 if not vector else np.sum(np.abs(a) ** 2, axis=1)
        gain = 0.5 * (raman @ power)
        return gain * a if not vector else gain[:, None] * a

    def rhs(z: float, a: np.ndarray) -> np.ndarray:
        phase = np.exp(1j * (delta * z + start))
        if not vector:
            drive = 1j * gamma * a[ti] * a[tj] * np.conj(a[tk]) * phase
            out = np.bincount(tq, weights=drive.real, minlength=n) + 1j * np.bincount(
                tq, weights=drive.imag, minlength=n
            )
            return out - 0.5 * alpha * a + scattering(a)
        first, second, third = a[ti], a[tj], np.conj(a[tk])
        if coherent:
            term = (2.0 / 3.0) * np.sum(first * third, axis=1)[:, None] * second + (
                1.0 / 3.0
            ) * np.sum(first * second, axis=1)[:, None] * third
        else:
            term = first * second * third + (2.0 / 3.0) * (first[:, ::-1] * third[:, ::-1]) * second
        drive = 1j * gamma * term * phase[:, None]
        out = np.stack(
            [
                np.bincount(tq, weights=drive[:, axis].real, minlength=n)
                + 1j * np.bincount(tq, weights=drive[:, axis].imag, minlength=n)
                for axis in (0, 1)
            ],
            axis=1,
        )
        return out - 0.5 * alpha * a + scattering(a)

    z, a = 0.0, y0.copy()
    # One nonlinear or dispersive length, whichever is shorter, opens the step.
    rate = max(float(np.max(np.abs(delta))) if delta.size else 0.0, gamma * scale**2 * n, 1e-30)
    h = min(distance, 0.1 / rate)
    steps = 0
    k1 = rhs(z, a)
    while z < distance:
        h = min(h, distance - z)
        ks = [k1]
        for stage in range(1, 7):
            increment = sum(
                coefficient * k for coefficient, k in zip(_DP_A[stage], ks, strict=True)
            )
            ks.append(rhs(z + _DP_C[stage] * h, a + h * increment))
        high = a + h * sum(b * k for b, k in zip(_DP_B5, ks, strict=True))
        low = a + h * sum(b * k for b, k in zip(_DP_B4, ks, strict=True))
        error = float(
            np.max(np.abs(high - low) / (atol + rtol * np.maximum(np.abs(a), np.abs(high))))
        )
        if error <= 1.0:
            z, a = z + h, high
            k1 = ks[6]  # first same as last: the pair's own next stage
            steps += 1
        h *= min(5.0, max(0.2, 0.9 * error ** (-0.2))) if error > 0.0 else 5.0
        if h < 1e-12 * distance:
            raise ValueError("the tone solver's step collapsed: the equations are too stiff here")
    return a, steps


#: Silica's isotropic Kerr tensor, as the drive of one mixing product:
#: ``D^m = T^m_abc A_i^a A_j^b (A_k^c)*`` over the Jones components. From
#: ``(2/3)(E . E*) E + (1/3)(E . E) E*`` with the degeneracy factor divided out,
#: symmetrised in ``i`` and ``j`` -- after which the degenerate and the
#: non-degenerate products share it.
_FWM_TENSOR = np.zeros((2, 2, 2, 2))
for _m in range(2):
    for _a in range(2):
        for _b in range(2):
            for _c in range(2):
                _FWM_TENSOR[_m, _a, _b, _c] = (
                    (_a == _c) * (_m == _b) + (_b == _c) * (_m == _a) + (_a == _b) * (_m == _c)
                ) / 3.0

#: The same drive when the split-step keeps only the phase-only form,
#: ``|A_x|**2 A_x + (2/3) |A_y|**2 A_x``: what it drops is exactly the coherent
#: ``(1/3) A_y**2 A_x*`` term, whose part in the drive is the entries where both
#: pumps sit on the other axis and the conjugated one on this.
_FWM_TENSOR_PHASE_ONLY = _FWM_TENSOR.copy()
for _m in range(2):
    _FWM_TENSOR_PHASE_ONLY[_m, 1 - _m, 1 - _m, _m] = 0.0


def coherency(ex: np.ndarray, ey: np.ndarray) -> np.ndarray:
    """A band's 2x2 coherency matrix ``J[a, b] = <A^a (A^b)*>`` [W]."""
    fields = (np.asarray(ex, dtype=np.complex128), np.asarray(ey, dtype=np.complex128))
    return np.array([[np.mean(p * np.conj(q)) for q in fields] for p in fields])


def fwm_vector_drive(
    pump_i: np.ndarray, pump_j: np.ndarray, pump_k: np.ndarray, *, coherent: bool = True
) -> np.ndarray:
    """``C^mn = <D^m (D^n)*>``, the mixing drive's own coherency [W^3].

    The scalar product formula's ``P_i P_j P_k``, per axis and between the axes,
    for pumps in any state of polarization. Taking each axis as its own scalar
    problem builds the x product from the pumps' x components alone, which is
    right for light on an axis and a quarter of the answer for the same light at
    45 degrees -- where an isotropic fibre mixes exactly as it does on the axis.
    The tensor puts back the cross-polarized terms: ``(2/3)(A_i . A_k*) A_j`` and
    its partner, and ``(1/3)(A_i . A_j) A_k*``.

    The bands are taken as independent, so the average factors into their
    coherency matrices; a degenerate product uses its pump's twice, which is what
    the scalar formula's ``P_i**2`` already assumed. All three on one axis give
    ``P_i P_j P_k`` on it, exactly. Circularly polarized, ``A . A`` vanishes and
    ``(2/3)**2`` is left. Agrawal, *Nonlinear Fiber Optics*, 5th ed., sec. 6.1,
    for the tensor.

    ``coherent`` picks which Kerr term the drive belongs to, and it has to be the
    one the split-step is running: the isotropic tensor with the coherent
    polarization term, or the phase-only form without it, which mixes a 45-degree
    beam more weakly than an axial one because it is not isotropic.
    """
    tensor = _FWM_TENSOR if coherent else _FWM_TENSOR_PHASE_ONLY
    return np.einsum(
        "mabc,nxyz,ax,by,cz->mn",
        tensor,
        tensor,
        np.asarray(pump_i),
        np.asarray(pump_j),
        np.conj(np.asarray(pump_k)),
    )


def fwm_drive_phase(
    pump_i: np.ndarray,
    pump_j: np.ndarray,
    pump_k: np.ndarray,
    *,
    coherent: bool = True,
) -> float:
    """The phase the tensor gives a drive on its reference axis, for unit pumps [rad].

    :func:`fwm_vector_drive` is a coherency and has none. But ``T(u_i, u_j, u_k*)``
    for Jones vectors each real on x is a vector whose x component need not be
    real -- the cross terms ``(u_i . u_k*) u_j`` carry the pumps' relative phases
    between axes -- and a pump whose state moves moves it. Returned as the angle
    the product's field has, ``exp(+i phase)``, on :func:`phase_reference`'s axis.
    """
    tensor = _FWM_TENSOR if coherent else _FWM_TENSOR_PHASE_ONLY
    drive = np.einsum(
        "mabc,a,b,c->m",
        tensor,
        np.asarray(pump_i),
        np.asarray(pump_j),
        np.conj(np.asarray(pump_k)),
    )
    lead = drive[phase_reference(drive)]
    return float(np.angle(lead)) if lead != 0 else 0.0


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
    nonlinear_rate: float = 0.0,
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

    This is the undepleted-pump result: it assumes the pumps are unchanged by
    what they generate. The fibre block takes the power back out of them with
    :func:`fwm_power_transfer`, so energy is conserved, but the product itself is
    still this formula — right while it is small against the pumps, which at any
    power a link runs at it is by forty decibels.
    """
    factor = 1.0 if degenerate else 2.0
    length = effective_length(alpha, distance)
    efficiency = fwm_efficiency(phase_mismatch, alpha, distance, nonlinear_rate)
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


def fwm_power_transfer(
    frequency_i: float, frequency_j: float, frequency_k: float, product_power: float
) -> tuple[float, float, float]:
    """Who pays for a mixing product: ``(lost by i, lost by j, gained by k)`` [W].

    The process destroys one photon from each of ``i`` and ``j`` and creates one
    at ``k`` and one at the product, ``f_i + f_j - f_k``. So for every photon in
    the product, ``i`` and ``j`` each lose a photon's worth of their own
    frequency and ``k`` — the idler — gains one of its own. That is parametric
    amplification, and it is why ``k`` goes up rather than down.

    Energy is conserved exactly, because the frequencies are:
    ``lost_i + lost_j == gained_k + product_power``. When ``i`` and ``j`` are the
    same channel it loses both shares.
    """
    product_frequency = frequency_i + frequency_j - frequency_k
    if product_frequency <= 0.0:
        raise ValueError(f"the product frequency must be positive, got {product_frequency} Hz")
    photons = product_power / product_frequency
    return photons * frequency_i, photons * frequency_j, photons * frequency_k


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
        module = array_module(ex, ey)
        return (
            ex.astype(module.complex128, copy=True),
            ey.astype(module.complex128, copy=True),
        )

    xp = array_module(ex, ey)
    omega = angular_frequency_grid(ex.shape[0], sample_rate, like=ex)
    spectrum_x = xp.fft.fft(ex.astype(xp.complex128))
    spectrum_y = xp.fft.fft(ey.astype(xp.complex128))

    for section in sections:
        phase = xp.exp(0.5j * omega * section.dgd)
        delayed_x = spectrum_x * phase
        delayed_y = spectrum_y * xp.conj(phase)
        u = section.unitary
        spectrum_x = u[0, 0] * delayed_x + u[0, 1] * delayed_y
        spectrum_y = u[1, 0] * delayed_x + u[1, 1] * delayed_y

    return xp.fft.ifft(spectrum_x), xp.fft.ifft(spectrum_y)


def gaussian_lowpass_response(frequency: np.ndarray, bandwidth: float) -> np.ndarray:
    """Amplitude response of a Gaussian low-pass with 3 dB power bandwidth ``bandwidth``.

    ``|H(f)|**2 = exp(-ln2 * (f/B)**2)``, which is exactly 1/2 at ``f = B`` — that
    identity is the definition of the 3 dB point and is asserted in the tests.
    """
    if bandwidth <= 0.0:
        raise ValueError(f"bandwidth must be positive, got {bandwidth}")
    xp = array_module(frequency)
    return xp.exp(-0.5 * math.log(2.0) * (frequency / bandwidth) ** 2)


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
    xp = array_module(frequency)
    return xp.exp(-0.5 * math.log(2.0) * (2.0 * frequency / bandwidth) ** (2 * order))


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
    xp = array_module(samples)
    spectrum = xp.fft.rfft(samples.astype(xp.float64))
    frequency = xp.asarray(xp.fft.rfftfreq(n, d=1.0 / sample_rate))
    return xp.fft.irfft(spectrum * gaussian_lowpass_response(frequency, bandwidth), n)
