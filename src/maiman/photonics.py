"""Integrated-photonic device models, as scattering matrices.

Three devices, and one of them is built out of the other two. That is deliberate:
the ring resonator is not written down as a formula but *assembled* — a coupler
and a length of waveguide, wired into a loop and handed to
:meth:`maiman.circuit.Circuit.solve` — so that what the tests check is the
framework, not a transcription. The closed forms from the literature are in the
tests, on the other side of the comparison, where they belong.

Everything here is a function of a frequency grid and returns an
:class:`~maiman.circuit.SMatrix` on that grid. Nothing is sampled, nothing is
time-domain, and nothing carries state; turning a scattering matrix back into
something a link simulation can propagate is :mod:`maiman.components.photonic`'s
job.

**Sources.** The waveguide's index expansion is the standard one (Chrostowski &
Hochberg, *Silicon Photonics Design*, §3). The coupler and both ring
configurations follow Yariv, *Universal relations for coupling of optical power
between microresonators and dielectric waveguides*, Electron. Lett. 36(4), 2000,
and Bogaerts et al., *Silicon microring resonators*, Laser Photonics Rev. 6(1),
2012. Default index values are for a 500 x 220 nm silicon strip waveguide at
1550 nm.
"""

from __future__ import annotations

import numpy as np

from .circuit import Circuit, SMatrix
from .kernels import dispersion_to_beta2
from .units import C_LIGHT, frequency_to_wavelength

#: Effective index of a 500 x 220 nm silicon strip waveguide, TE, at 1550 nm.
SILICON_STRIP_NEFF = 2.44

#: Group index of the same waveguide. It is far larger than the effective index —
#: silicon waveguides are strongly dispersive by construction — and it is the one
#: that sets a ring's free spectral range, so confusing the two is the single
#: most common way to get an FSR wrong by a factor of two.
SILICON_STRIP_NGROUP = 4.20


def propagation_constant(
    frequencies: np.ndarray,
    *,
    n_eff: float,
    n_group: float,
    reference_frequency: float,
    dispersion: float = 0.0,
) -> np.ndarray:
    """``beta(omega)`` [rad/m] from the two indices and the dispersion parameter.

    The first three terms of the expansion about ``reference_frequency``:

        ``beta = 2*pi*f_ref*n_eff/c + (n_group/c) * W + beta2 * W**2 / 2``

    with ``W = 2*pi*(f - f_ref)``. ``n_eff`` sets the *phase* — which resonance a
    ring sits on — and ``n_group`` sets the *delay*, and therefore the spacing
    between resonances. They are different numbers and neither substitutes for
    the other.

    ``dispersion`` is D in the same units the fibre uses, so a delay line and a
    span of fibre are described by one parameter with one meaning.

    **A note on the sign of the quadratic term.** This is
    ``exp(-1j*beta(omega)*L)`` throughout: one expansion, one convention, with
    the delay term positive so that a longer waveguide arrives later and — for
    D > 0 — the longer wavelength arrives later still. That agrees with
    :func:`maiman.kernels.walkoff_from_dispersion`, and disagrees with
    :func:`maiman.kernels.propagate_dispersion`, whose quadratic term carries the
    opposite sign; the two cannot both be right and reconciling them is its own
    piece of work.

    It does not reach a resonator. At the very large D = -1000 ps/nm/km a silicon
    strip waveguide has, the quadratic term over a 100 um ring is 0.049 rad at the
    edge of the C band — three orders below the round trip's own phase, and far
    below a linewidth. It reaches a *delay line*: the same number over a 10 cm
    spiral is 49 rad, and there the sign is the difference between a pulse
    compressing and broadening. Both are measured in ``tests/test_photonics.py``
    rather than asserted, so that the day the kernel is reconciled the second
    number is already sitting there saying what changed.
    """
    omega = 2.0 * np.pi * (np.asarray(frequencies, dtype=np.float64) - reference_frequency)
    beta0 = 2.0 * np.pi * reference_frequency * n_eff / C_LIGHT
    beta1 = n_group / C_LIGHT
    beta2 = dispersion_to_beta2(dispersion, frequency_to_wavelength(reference_frequency))
    return beta0 + beta1 * omega + 0.5 * beta2 * omega**2


def free_spectral_range(length: float, n_group: float) -> float:
    """Resonance spacing ``c / (n_group * L)`` [Hz] of a loop of that length.

    The *group* index, not the effective index: a resonance moves when the round
    trip changes by one wavelength, and how fast that happens with frequency is a
    delay. At 4.20 a 100 um ring free-spectral-ranges every 714 GHz.
    """
    return C_LIGHT / (n_group * length)


def round_trip_amplitude(length: float, loss_db_per_m: float) -> float:
    """Field amplitude surviving one lap of a loop of that length."""
    return float(10.0 ** (-loss_db_per_m * length / 20.0))


def resonance_linewidth(
    length: float,
    n_group: float,
    *,
    coupling: float,
    drop_coupling: float = 0.0,
    loss_db_per_m: float = 0.0,
) -> float:
    """Full width at half depth of one resonance [Hz].

    ``FWHM = FSR * (1 - r) / (pi * sqrt(r))`` with ``r = t1 * t2 * a``, the
    amplitude a wave retains over one lap including both couplers. The standard
    Lorentzian approximation (Bogaerts et al., eq. 21), good wherever the
    resonance is narrow enough to be worth calling one.

    It is here because it is what a *sampling step* has to respect. A resonator's
    response is the narrowest feature in this library by orders of magnitude, and
    anything that integrates over it — averaging the ASE a ring passes, for
    instance — has to know how fine to look before it starts.
    """
    retained = (
        np.sqrt(1.0 - coupling)
        * np.sqrt(1.0 - drop_coupling)
        * round_trip_amplitude(length, loss_db_per_m)
    )
    if retained <= 0.0:
        return float("inf")
    spacing = free_spectral_range(length, n_group)
    return float(spacing * (1.0 - retained) / (np.pi * np.sqrt(retained)))


def straight_waveguide(
    frequencies: np.ndarray,
    *,
    length: float,
    n_eff: float = SILICON_STRIP_NEFF,
    n_group: float = SILICON_STRIP_NGROUP,
    reference_frequency: float,
    dispersion: float = 0.0,
    loss_db_per_m: float = 0.0,
    ports: tuple[str, str] = ("in", "out"),
) -> SMatrix:
    """A length of single-mode waveguide: delay, phase, and propagation loss.

    Matched at both ends and reciprocal, so the matrix is off-diagonal. That is a
    *model* choice, not a limitation of the solver — a facet reflection is a
    diagonal term and the reduction handles it — but a strip waveguide's
    sidewalls scatter light out of the circuit rather than back down it, and
    modelling that as loss is right.

    Loss is per metre here because everything inside the engine is SI, but the
    number a foundry quotes is per centimetre and is a hundred times smaller: a
    typical silicon strip is 2 dB/cm — 200 dB/m — and a good silicon nitride
    0.1 dB/cm.
    """
    frequencies = np.asarray(frequencies, dtype=np.float64)
    beta = propagation_constant(
        frequencies,
        n_eff=n_eff,
        n_group=n_group,
        reference_frequency=reference_frequency,
        dispersion=dispersion,
    )
    amplitude = 10.0 ** (-loss_db_per_m * length / 20.0)
    transfer = amplitude * np.exp(-1j * beta * length)

    s = np.zeros((frequencies.shape[0], 2, 2), dtype=np.complex128)
    s[:, 0, 1] = transfer
    s[:, 1, 0] = transfer
    return SMatrix(ports=ports, frequencies=frequencies, s=s)


def directional_coupler(
    frequencies: np.ndarray,
    *,
    coupling: float,
    insertion_loss_db: float = 0.0,
    ports: tuple[str, str, str, str] = ("in1", "in2", "out1", "out2"),
) -> SMatrix:
    """Two waveguides brought close enough to exchange power.

    ``coupling`` is the *power* fraction that crosses over. The amplitudes are
    ``t = sqrt(1 - coupling)`` straight through and ``1j * sqrt(coupling)``
    across, and **the factor of j is not decoration**: it is what makes the
    matrix unitary, and therefore what makes a lossless coupler conserve power
    at every phase rather than only on average. Drop it and a Mach-Zehnder built
    from two of these sends all its light out of one arm and none out of the
    other — an interferometer that gains energy at one wavelength and loses it at
    another.

    Frequency-independent, which is the standard idealisation and is good over a
    few tens of nanometres. A real coupler's ratio drifts across the C band; when
    that matters the fix is a fitted ``coupling(f)``, and the shape of this
    function is what a fit would slot into.

    Ports 1 and 2 are the two waveguides on the input side, ports 3 and 4 the
    same two on the output side: ``in1`` goes mostly to ``out1``, and the
    ``coupling`` fraction of it to ``out2``.
    """
    if not 0.0 <= coupling <= 1.0:
        raise ValueError(f"coupling is a power fraction in [0, 1], got {coupling}")
    frequencies = np.asarray(frequencies, dtype=np.float64)
    through = np.sqrt(1.0 - coupling)
    across = 1j * np.sqrt(coupling)
    amplitude = 10.0 ** (-insertion_loss_db / 20.0)

    block = amplitude * np.array([[through, across], [across, through]], dtype=np.complex128)
    s = np.zeros((frequencies.shape[0], 4, 4), dtype=np.complex128)
    s[:, 2:, :2] = block
    s[:, :2, 2:] = block
    return SMatrix(ports=ports, frequencies=frequencies, s=s)


def ring_resonator(
    frequencies: np.ndarray,
    *,
    length: float,
    coupling: float,
    drop_coupling: float = 0.0,
    n_eff: float = SILICON_STRIP_NEFF,
    n_group: float = SILICON_STRIP_NGROUP,
    reference_frequency: float,
    dispersion: float = 0.0,
    loss_db_per_m: float = 0.0,
) -> SMatrix:
    """A loop of waveguide beside one bus, or between two. Ports: in, add, through, drop.

    **Assembled, not written down.** Two couplers and two half-arcs are put in a
    :class:`~maiman.circuit.Circuit` and solved. The all-pass and add-drop
    transfer functions in the literature are what
    ``tests/test_photonics.py`` compares the result against; they are nowhere in
    this function, which is the point of having a solver at all.

    ``drop_coupling = 0`` leaves the second coupler in place with nothing
    crossing it, which is an all-pass ring: the loop no longer sees the second
    bus at all, and the drop port carries only whatever was put into the add port
    beside it. Physically exact, and it means the port set does not change when a
    parameter does — a block whose wiring becomes invalid because a number was
    edited is a bad block.

    **Critical coupling** is the case worth knowing: when the coupling exactly
    matches the round-trip loss, the through port goes to *zero* on resonance —
    the light coupled back out of the ring cancels the light that stayed on the
    bus. Under-couple or over-couple and the notch fills in from either side,
    with the same depth for two different couplings, which is why a measured
    notch depth alone does not identify a ring.
    """
    frequencies = np.asarray(frequencies, dtype=np.float64)
    # One arc, placed twice: the two halves of the loop are the same device, and
    # an SMatrix is a value rather than a thing with identity.
    arc = straight_waveguide(
        frequencies,
        length=length / 2.0,
        n_eff=n_eff,
        n_group=n_group,
        reference_frequency=reference_frequency,
        dispersion=dispersion,
        loss_db_per_m=loss_db_per_m,
    )
    circuit = Circuit()
    circuit.add("bus", directional_coupler(frequencies, coupling=coupling))
    circuit.add("drop_bus", directional_coupler(frequencies, coupling=drop_coupling))
    circuit.add("upper", arc)
    circuit.add("lower", arc)

    circuit.link("bus", "out2", "upper", "in")
    circuit.link("upper", "out", "drop_bus", "in2")
    circuit.link("drop_bus", "out2", "lower", "in")
    circuit.link("lower", "out", "bus", "in2")

    circuit.expose("in", "bus", "in1")
    circuit.expose("through", "bus", "out1")
    circuit.expose("add", "drop_bus", "in1")
    circuit.expose("drop", "drop_bus", "out1")
    return circuit.solve()
