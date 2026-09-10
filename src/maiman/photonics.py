"""Integrated-photonic device models, as scattering matrices.

Five devices, and two of them are built out of the others. That is deliberate:
the ring resonator and the Mach-Zehnder are not written down as formulas but
*assembled* — couplers and lengths of waveguide, wired up and handed to
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
2012. The MMI's phase relations are Bachmann, Besse & Melchior, *General
self-imaging properties in N x N multimode interference couplers including phase
relations*, Appl. Opt. 33(18), 1994. Default index values are for a 500 x 220 nm
silicon strip waveguide at 1550 nm.
"""

from __future__ import annotations

from typing import Any

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


def mmi_phase_relations(ports: int) -> np.ndarray:
    """The ``N x N`` self-imaging phase matrix ``phi[i, j]`` [rad], ports from 0.

    Bachmann, Besse & Melchior (1994), for a general-interference MMI at the
    length ``3 L_pi / N`` that images every input onto every output:

        ``phi_ij = phi_0 + (pi/4N) (j - i)(2N - j + i)``      i + j even
        ``phi_ij = phi_0 + (pi/4N) (i + j - 1)(2N - i - j + 1)``  i + j odd

    with ports numbered from **one** in the paper, which is why they are shifted
    here. ``phi_0`` is common to the whole matrix and is dropped: a global phase
    on a scattering matrix is unobservable, and carrying it would only invite
    someone to compare two of them and find a discrepancy that is not there.

    The relations are the whole content of the device. An MMI's *amplitudes* are
    the easy half — self-imaging splits the power evenly, so every path is
    ``1/sqrt(N)`` — and any two-port model gets that right by accident. What
    distinguishes an MMI from a box that divides power is the phase between the
    images, and it is what decides whether an interferometer built from two of
    them constructively interferes at the port you wanted.

    At ``N = 2`` this returns ``[[0, pi/2], [pi/2, 0]]``, which is the 3 dB
    directional coupler including its factor of j — the two devices are
    genuinely the same matrix, and a test holds them to it.

    **The returned matrix is not literally symmetric**, and the device still is.
    Above ``N = 3`` the two branches of the formula hand back phases for ``ij``
    and ``ji`` that differ by whole multiples of ``2 pi`` — at ``N = 4``,
    ``phi[0, 2]`` is ``0.75 pi`` and ``phi[2, 0]`` is ``-1.25 pi``. Only
    ``exp(1j * phi)`` is physical, and that *is* symmetric. Anyone comparing this
    matrix against another implementation's should compare the exponentials, or
    find a discrepancy that is not there.
    """
    if ports < 1:
        raise ValueError(f"an MMI needs at least one port a side, got {ports}")
    phi = np.zeros((ports, ports), dtype=np.float64)
    for i in range(1, ports + 1):
        for j in range(1, ports + 1):
            if (i + j) % 2 == 0:
                value = (j - i) * (2 * ports - j + i)
            else:
                value = (i + j - 1) * (2 * ports - i - j + 1)
            phi[i - 1, j - 1] = np.pi * value / (4.0 * ports)
    return phi


def mmi_coupler(
    frequencies: np.ndarray,
    *,
    ports: int,
    excess_loss_db: float = 0.0,
    imbalance_db: float = 0.0,
    ports_names: tuple[str, ...] | None = None,
) -> SMatrix:
    """An ``N x N`` multimode interference coupler.

    A section of waveguide wide enough to carry several modes. They travel at
    different speeds, and at the right length their relative phases bring them
    back into ``N`` copies of the input field — self-imaging. That is the whole
    device: no gap to control, no coupling length to hit, which is why an MMI is
    what a foundry gives you when a directional coupler's ratio would drift with
    a nanometre of lithography. It pays for that with excess loss, because the
    light that does not land in an image is radiated.

    ``imbalance_db`` tilts the split across the outputs, worst-to-best, and is
    the parameter a datasheet actually quotes; a perfect MMI is 0 and a real
    2 x 2 is a few tenths. It is applied as an amplitude taper and the matrix is
    then **not** unitary — which is correct, because an imbalanced MMI is lossy,
    and pretending otherwise would let a circuit built from it manufacture power.

    Ports are the ``N`` inputs then the ``N`` outputs, named ``in1..inN`` and
    ``out1..outN`` unless given.
    """
    if ports < 1:
        raise ValueError(f"an MMI needs at least one port a side, got {ports}")
    if excess_loss_db < 0.0:
        raise ValueError(f"excess loss is a loss, got {excess_loss_db} dB")
    frequencies = np.asarray(frequencies, dtype=np.float64)

    # Even split, then the taper. The nominal amplitude is 1/sqrt(N) so that a
    # perfect device passes all of its power; anything less is said explicitly by
    # a loss parameter rather than hidden in a normalisation.
    taper = np.ones(ports, dtype=np.float64)
    if imbalance_db and ports > 1:
        # Linear in dB across the outputs, then renormalised so the mean
        # *power* is one. Centring the tilt in decibels is not the same as
        # centring it in power — by Jensen it comes out slightly heavy — so
        # without this line an imbalanced MMI passes a little more light than
        # its excess loss says, and "0.45 dB excess" stops being a number you
        # can subtract. The rescale is a constant on every amplitude, so the
        # spread in dB is still exactly the imbalance asked for.
        tilt = np.linspace(-imbalance_db / 2.0, imbalance_db / 2.0, ports)
        taper = 10.0 ** (-tilt / 20.0)
        taper /= np.sqrt(np.mean(taper**2))
    amplitude = 10.0 ** (-excess_loss_db / 20.0) / np.sqrt(ports)
    # Rows are outputs, so the taper indexes rows. Tapering the columns instead
    # spreads the imbalance across the *inputs*, which leaves every output of a
    # singly-driven device at the same level — an imbalance parameter that does
    # nothing at all in the one configuration everybody uses.
    block = amplitude * taper[:, None] * np.exp(1j * mmi_phase_relations(ports))

    n = 2 * ports
    s = np.zeros((frequencies.shape[0], n, n), dtype=np.complex128)
    s[:, ports:, :ports] = block
    # Reciprocal: the same device run backwards, which is the transpose and not
    # the conjugate. Conjugating would reverse the sign of every self-imaging
    # phase and turn the reverse path into a different device.
    s[:, :ports, ports:] = block.T
    names = ports_names or tuple(
        [f"in{k + 1}" for k in range(ports)] + [f"out{k + 1}" for k in range(ports)]
    )
    return SMatrix(ports=names, frequencies=frequencies, s=s)


def mach_zehnder(
    frequencies: np.ndarray,
    *,
    arm_length: float,
    length_difference: float = 0.0,
    phase_shift: float = 0.0,
    splitter_coupling: float = 0.5,
    combiner_coupling: float = 0.5,
    use_mmi: bool = False,
    n_eff: float = SILICON_STRIP_NEFF,
    n_group: float = SILICON_STRIP_NGROUP,
    reference_frequency: float,
    dispersion: float = 0.0,
    loss_db_per_m: float = 0.0,
    insertion_loss_db: float = 0.0,
) -> SMatrix:
    """Two couplers with two paths between them. Ports: in1, in2, out1, out2.

    **Assembled, not written down**, exactly as the ring is: a splitter, two
    lengths of waveguide, a combiner, into a :class:`~maiman.circuit.Circuit`.
    The textbook ``cos``/``sin`` transfer function is in the tests, on the other
    side of the comparison.

    Two knobs move the output, and they are not the same knob.
    ``length_difference`` is *path* imbalance and its phase grows with frequency
    — that is what makes an unbalanced MZI a filter with a free spectral range
    of ``c / (n_g dL)``, and it is the interleaver every WDM transmitter has.
    ``phase_shift`` is a flat phase on the long arm, which is what a heater or a
    carrier-injection section does, and it slides the whole comb sideways
    without changing its period. A balanced MZI has no FSR at all and is a pure
    switch: it is ``phase_shift`` alone that decides which output the light
    leaves by.

    ``use_mmi`` swaps both directional couplers for 2 x 2 MMIs. At the nominal
    3 dB point the two are the same matrix, so a balanced device does not care;
    what changes is the behaviour when a coupler is *off* its ratio, which is
    the reason a foundry uses MMIs and the reason this is a switch rather than
    an assumption.
    """
    frequencies = np.asarray(frequencies, dtype=np.float64)
    settings: dict[str, Any] = {
        "n_eff": n_eff,
        "n_group": n_group,
        "reference_frequency": reference_frequency,
        "dispersion": dispersion,
        "loss_db_per_m": loss_db_per_m,
    }

    def coupler(coupling: float) -> SMatrix:
        if use_mmi:
            return mmi_coupler(frequencies, ports=2, excess_loss_db=insertion_loss_db)
        return directional_coupler(
            frequencies, coupling=coupling, insertion_loss_db=insertion_loss_db
        )

    # The imbalance goes entirely on one arm. Splitting it between the two would
    # give the same transfer function and a different physical device, and the
    # length that appears on a mask is this one.
    short = straight_waveguide(frequencies, length=arm_length, **settings)
    long_arm = straight_waveguide(frequencies, length=arm_length + length_difference, **settings)
    if phase_shift:
        # A pure phase, applied as its own two-port so that the arm's *length*
        # stays the length on the mask. Folding it into the arm would make a
        # heater setting indistinguishable from a fabrication error.
        shifter = np.zeros((frequencies.shape[0], 2, 2), dtype=np.complex128)
        shifter[:, 0, 1] = np.exp(-1j * phase_shift)
        shifter[:, 1, 0] = np.exp(-1j * phase_shift)
        phase = SMatrix(ports=("in", "out"), frequencies=frequencies, s=shifter)

    circuit = Circuit()
    circuit.add("split", coupler(splitter_coupling))
    circuit.add("combine", coupler(combiner_coupling))
    circuit.add("upper", short)
    circuit.add("lower", long_arm)

    circuit.link("split", "out1", "upper", "in")
    circuit.link("upper", "out", "combine", "in1")
    if phase_shift:
        circuit.add("heater", phase)
        circuit.link("split", "out2", "lower", "in")
        circuit.link("lower", "out", "heater", "in")
        circuit.link("heater", "out", "combine", "in2")
    else:
        circuit.link("split", "out2", "lower", "in")
        circuit.link("lower", "out", "combine", "in2")

    circuit.expose("in1", "split", "in1")
    circuit.expose("in2", "split", "in2")
    circuit.expose("out1", "combine", "out1")
    circuit.expose("out2", "combine", "out2")
    return circuit.solve()


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
