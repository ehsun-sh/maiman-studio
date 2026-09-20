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

import math
from collections.abc import Callable, Mapping, Sequence
from functools import lru_cache
from typing import Any

import numpy as np

from .circuit import Circuit, SMatrix
from .kernels import dispersion_to_beta2
from .modes import (
    Mode,
    StepIndexFibre,
    _legendre,
    bessel_j,
    cladding_modes,
    core_modes,
    lpg_coupling,
)
from .units import C_LIGHT, frequency_to_wavelength
from .vector_modes import VectorMode, vector_cladding_modes, vector_core_modes, vector_coupling

#: Effective index of a 500 x 220 nm silicon strip waveguide, TE, at 1550 nm.
SILICON_STRIP_NEFF = 2.44

#: Group index of the same waveguide. It is far larger than the effective index —
#: silicon waveguides are strongly dispersive by construction — and it is the one
#: that sets a ring's free spectral range, so confusing the two is the single
#: most common way to get an FSR wrong by a factor of two.
SILICON_STRIP_NGROUP = 4.20

#: The same waveguide's **TM** mode, which as far as every number here is
#: concerned is a different waveguide. A 220 nm-thick strip confines TM far more
#: weakly than TE — the mode spreads into the cladding and sees less silicon — and
#: both indices come down with it. Representative values for 500 x 220 nm at
#: 1550 nm from the same source as the TE pair; a real design takes them from a
#: mode solver or a PDK, because they move with every nanometre of width.
#:
#: The gap is not small, and that is the point: 2.44 against 1.78 is a 27 %
#: difference in phase index, so a ring resonates at two sets of wavelengths
#: nowhere near each other, and 4.20 against 3.80 puts the two combs on different
#: free spectral ranges as well.
SILICON_STRIP_NEFF_TM = 1.78
SILICON_STRIP_NGROUP_TM = 3.80

#: Effective index of the LP01 mode of a standard germanium-doped silica fibre
#: core at 1550 nm. It is here rather than beside the silicon numbers because it
#: belongs to a different device entirely: a fibre Bragg grating is written into
#: a drawn fibre, not patterned on a die, and taking silicon's 2.44 for it would
#: put the reflection 40 % away in wavelength. The Bragg condition is
#: ``lambda_B = 2 * n_eff * Lambda``, so this is the number that turns a period
#: into a wavelength and back.
SILICA_FIBER_NEFF = 1.4475

#: Separator between a port's name and the polarization it carries. No port name
#: in this library contains it, so a split is unambiguous and an un-suffixed name
#: stays a legal single-polarization port.
POLARIZATION_SEPARATOR = "@"

#: The two guided polarizations, in the order a dual-polarization matrix stacks
#: them. Named rather than numbered: "the second block" is not something anyone
#: can check against a foundry datasheet, and "tm" is.
POLARIZATIONS = ("te", "tm")


def polarized_port(port: str, polarization: str) -> str:
    """``"in"`` and ``"te"`` make ``"in@te"``, which is what the solver sees."""
    if POLARIZATION_SEPARATOR in port:
        raise ValueError(
            f"port name {port!r} already contains {POLARIZATION_SEPARATOR!r}, so it "
            f"cannot carry a polarization suffix unambiguously"
        )
    return f"{port}{POLARIZATION_SEPARATOR}{polarization}"


def dual_polarization(
    matrices: Mapping[str, SMatrix],
    *,
    cross: Mapping[tuple[str, str], np.ndarray] | None = None,
) -> SMatrix:
    """Stack one matrix per polarization into a single scattering matrix.

    **The solver needs nothing added to it for this, and that was worth checking
    rather than believing.** The roadmap described polarization resolution as
    "a second index on every device", which would have meant reworking every
    model and the reduction with them. It is not that.
    :class:`~maiman.circuit.SMatrix` identifies ports by *name*, so a device with
    an ``in`` and an ``out`` on each of two polarizations is a four-port device,
    and :meth:`~maiman.circuit.Circuit.solve` already solves those. All this does
    is build that four-port matrix out of two two-port ones.

    The result is **block diagonal**: light launched on one polarization stays on
    it. That is right for a straight waveguide and for a symmetric coupler, where
    the two modes are orthogonal solutions of the same structure and do not talk
    to each other. It is *not* right in general — a bend, a sidewall that is not
    vertical, and a mode converter placed there on purpose all couple TE to TM —
    so ``cross`` takes those terms, keyed by ``(from, to)`` and shaped like one
    device block. Nothing in this library produces one. The argument exists
    because the solver never needed the assumption and should not acquire it
    here, at the one place where leaving it out would be invisible.

    Every matrix must carry the same ports on the same frequency grid. Different
    ports would mean the polarizations were measured on different devices, which
    is not a device.
    """
    if not matrices:
        raise ValueError("a dual-polarization matrix needs at least one polarization")
    items = list(matrices.items())
    first = items[0][1]
    count = len(first.ports)
    for name, matrix in items[1:]:
        if matrix.ports != first.ports:
            raise ValueError(
                f"every polarization must describe the same device, but {name!r} has "
                f"ports {list(matrix.ports)} against {list(first.ports)}"
            )
        if not np.array_equal(matrix.frequencies, first.frequencies):
            raise ValueError(
                f"every polarization must be evaluated on the same frequency grid, "
                f"and {name!r} is not"
            )

    ports = tuple(polarized_port(port, name) for name, _ in items for port in first.ports)
    size = count * len(items)
    shape = (first.frequencies.shape[0], count, count)
    s = np.zeros((first.frequencies.shape[0], size, size), dtype=np.complex128)
    offsets = {name: index * count for index, (name, _) in enumerate(items)}
    for name, matrix in items:
        start = offsets[name]
        s[:, start : start + count, start : start + count] = matrix.s

    for (source, target), block in (cross or {}).items():
        for end, role in ((source, "source"), (target, "target")):
            if end not in offsets:
                raise KeyError(
                    f"cross term names {end!r} as its {role}, which is not a polarization "
                    f"here; have {sorted(offsets)}"
                )
        if source == target:
            raise ValueError(
                f"a cross term from {source!r} to itself is that polarization's own matrix"
            )
        values = np.asarray(block, dtype=np.complex128)
        if values.shape != shape:
            raise ValueError(
                f"cross term {source!r}->{target!r} must be shaped like one device "
                f"block, {shape}, got {values.shape}"
            )
        row, column = offsets[target], offsets[source]
        s[:, row : row + count, column : column + count] = values

    return SMatrix(ports=ports, frequencies=first.frequencies, s=s)


def link_polarizations(
    circuit: Circuit,
    a_instance: str,
    a_port: str,
    b_instance: str,
    b_port: str,
    *,
    polarizations: Sequence[str] = POLARIZATIONS,
) -> Circuit:
    """Wire a port to a port, on every polarization, as one call.

    A wire in a photonic circuit is a physical waveguide and it carries both
    modes. Writing the two :meth:`~maiman.circuit.Circuit.link` calls out by hand
    works, and is exactly where a circuit acquires a connection on TE that it
    does not have on TM — a fault that yields a perfectly plausible spectrum on
    one polarization and silence on the other.
    """
    for polarization in polarizations:
        circuit.link(
            a_instance,
            polarized_port(a_port, polarization),
            b_instance,
            polarized_port(b_port, polarization),
        )
    return circuit


def expose_polarizations(
    circuit: Circuit,
    name: str,
    instance: str,
    port: str,
    *,
    polarizations: Sequence[str] = POLARIZATIONS,
) -> Circuit:
    """Expose one instance's port on every polarization, under suffixed names."""
    for polarization in polarizations:
        circuit.expose(
            polarized_port(name, polarization), instance, polarized_port(port, polarization)
        )
    return circuit


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
    :func:`maiman.kernels.walkoff_from_dispersion`.

    It used to disagree with :func:`maiman.kernels.propagate_dispersion`, whose
    quadratic term carried the opposite sign, and this docstring said so for
    some time without the two ever being in one graph to argue about it. A
    chirped fibre Bragg grating put them there, the kernel turned out to be the
    one transcribed from a textbook using the other transform convention, and it
    has been corrected — see that function for what moved with it.

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


#: Apodization profiles a written grating can carry, as functions of the
#: normalised position ``u`` running from -0.5 at the input face to +0.5 at the
#: far one. Named rather than numbered, because "raised-cosine" is a word that
#: appears on a datasheet and ``2`` is not.
APODIZATIONS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    # A grating of constant strength. Its spectrum is the transform of a
    # rectangle, so it has a rectangle's sidelobes -- the first one about 9 dB
    # below the peak, which is far too much crosstalk for a channel filter and
    # is the whole reason the other two exist.
    "uniform": lambda u: np.ones_like(u),
    # Hann. The standard written apodization: zero strength at both faces, so
    # there is no abrupt start for the spectrum to ring on.
    "raised-cosine": lambda u: 0.5 * (1.0 + np.cos(2.0 * np.pi * u)),
    # A Gaussian truncated at +/- 2.5 sigma. Lower sidelobes still, at the cost
    # of a wider main lobe for the same length -- the trade every window makes.
    "gaussian": lambda u: np.exp(-0.5 * (u / 0.2) ** 2),
}

#: Sections a grating is cut into unless asked otherwise. Measured rather than
#: guessed: against the geometric dispersion of a linearly chirped grating, 50
#: sections are 2.4 % out on a 1 m grating chirped over 40 nm and 200 are 0.07 %,
#: and nothing between 200 and 4000 moves the answer at all. It is a *count*
#: rather than a length because what has to stay small is the shift in local
#: Bragg wavelength per section relative to that section's own bandwidth, and
#: both scale with the section count in the same direction.
DEFAULT_GRATING_SECTIONS = 200


#: Effective photoelastic constant of germanium-doped silica, ``p_e``. Stretching
#: a fibre lengthens the grating's period *and* lowers its index, and the two
#: fight: the index term cancels 22 % of the geometric one, so a grating stretched
#: by a part per million moves by 0.78 of a part per million rather than one.
#: Getting this constant wrong scales every strain reading by the same factor and
#: nothing about the spectrum looks unusual.
SILICA_PHOTOELASTIC = 0.22

#: Fractional shift of the Bragg wavelength per kelvin, ``alpha + xi``, for bare
#: germanium-doped silica.
#:
#: Two effects, and **they are not the same size**. Thermal expansion
#: ``alpha = 0.55e-6 /K`` lengthens the period; the thermo-optic coefficient
#: ``xi = (1/n) dn/dT = 6.67e-6 /K`` raises the index. The second is twelve times
#: the first, which is why a fibre grating is a thermometer made of glass rather
#: than a thermometer made of geometry -- and why **coating changes this number a
#: lot**: a metal or polymer jacket adds its own expansion to ``alpha`` and
#: nothing to ``xi``, so a packaged sensor is often two or three times as
#: sensitive as a bare one. Only the sum is observable, so only the sum is
#: offered, with both halves written down here.
SILICA_THERMAL_SENSITIVITY = 7.22e-6


def bragg_shift(
    bragg_wavelength: float,
    *,
    strain: float = 0.0,
    temperature_change: float = 0.0,
    photoelastic: float = SILICA_PHOTOELASTIC,
    thermal_sensitivity: float = SILICA_THERMAL_SENSITIVITY,
) -> float:
    """Where a grating reflects once it is stretched or warmed [m].

        ``dlambda / lambda = (1 - p_e) * epsilon + (alpha + xi) * dT``

    The whole of fibre Bragg sensing, and it is one line because the grating is
    the sensor: a period is a length, a length responds to strain and to
    temperature, and the reflection reports it. Nothing has to be added to the
    fibre and the reading is absolute -- a grating that loses power and comes back
    reads the same wavelength, which is why these are used on bridges and in
    boreholes where an intensity-based sensor drifts.

    At 1550 nm the two coefficients come to **1.21 pm per microstrain** and
    **11.2 pm per kelvin**, and dividing one by the other gives the number the
    whole field is organised around: **a kelvin of drift is indistinguishable
    from 9.2 microstrain.** One grating produces one wavelength and there are two
    unknowns behind it, so a strain measurement that does not also measure
    temperature is not a strain measurement. The usual answers are a second
    grating held strain-free beside the first, or two gratings whose coefficients
    differ enough to invert -- and the reason the second is harder than it looks
    is that both sensitivities are nearly proportional to ``lambda``, so two
    gratings of the same fibre at different wavelengths give a matrix that is
    very close to singular.

    ``strain`` is the fractional elongation, positive in tension. ``epsilon`` and
    ``dT`` are independent and add, which is the linearity the inversion above
    relies on and is why it is worth saying that this model is exactly linear:
    real fibre is too, to well past where it breaks.

    Hill & Meltz, *Fiber Bragg grating technology fundamentals and overview*,
    J. Lightwave Technol. 15(8), 1997, for the coefficients and their spread.
    """
    if bragg_wavelength <= 0.0:
        raise ValueError(f"bragg_wavelength must be positive, got {bragg_wavelength}")
    fractional = (1.0 - photoelastic) * strain + thermal_sensitivity * temperature_change
    shifted = bragg_wavelength * (1.0 + fractional)
    if shifted <= 0.0:
        raise ValueError(
            f"strain={strain} and temperature_change={temperature_change} move the Bragg "
            f"wavelength to {shifted} m, which is not a wavelength. A fibre breaks "
            f"somewhere past 10000 microstrain; this is past -1."
        )
    return shifted


def section_positions(sections: int) -> np.ndarray:
    """Normalised centres of ``sections`` equal pieces, -0.5 at the input face.

    The coordinate every profile below is a function of, named once so that a
    profile written by hand and one built here cannot disagree about which end
    is which — which on a chirped grating is the difference between a
    compensator and something that makes the span worse.
    """
    if sections < 1:
        raise ValueError(f"sections must be at least 1, got {sections}")
    return (np.arange(sections) + 0.5) / sections - 0.5


def sampled_profile(sections: int, *, periods: float, duty: float = 0.5) -> np.ndarray:
    """A coupling shape switched on and off along the length: the sampled grating.

    Write a grating and then erase it periodically — or, in practice, write it
    through a periodic amplitude mask — and the single reflection peak becomes a
    **comb** of them. It is the superstructure grating, and it is how one device
    addresses a whole WDM band: the tuning element of a sampled-grating DBR
    laser, a multi-channel dispersion compensator, an interrogator that reads
    many sensors at once.

    The peaks are spaced by ``lambda**2 / (2 n_eff * sampling_period)``, which is
    the same Fabry-Perot arithmetic as any other cavity of that length — the
    sampling period is the cavity. That expression is in
    ``tests/test_grating.py`` and not here, and the model reproduces it to four
    digits.

    ``periods`` is how many sampling periods fit in the grating and need not be a
    whole number; ``duty`` is the written fraction of each. A duty of 1 is not
    sampled at all and returns a uniform grating, which is the right degenerate
    answer rather than a special case.
    """
    if periods <= 0.0:
        raise ValueError(f"periods must be positive, got {periods}")
    if not 0.0 < duty <= 1.0:
        raise ValueError(f"duty must be in (0, 1], got {duty}")
    position = section_positions(sections) + 0.5
    return ((position * periods) % 1.0 < duty).astype(np.float64)


def phase_shift_profile(
    sections: int, *, shift: float = np.pi, position: float = 0.5
) -> np.ndarray:
    """A jump in the grating's phase partway along: the phase-shifted grating.

    Break the periodicity once and the stop band acquires a **transmission window
    in the middle of it** — a resonance between the two halves, which are two
    mirrors facing each other. At ``pi`` it sits exactly on the Bragg wavelength
    and is the narrowest feature a grating of that length can produce: this is
    the distributed-feedback laser's cavity, and the notch filter used to lock a
    laser to a line.

    ``position`` is where along the length the jump happens, 0 at the input face
    and 1 at the far one. Off centre the two mirrors are unequal and the
    resonance stops reaching full transmission, which is a real design
    sensitivity rather than a modelling artefact.

    The array is the phase *at* each section, so it is a step and not an impulse:
    every section past the jump carries it. Writing it the other way — a single
    section with a different phase — is a defect rather than a shift, and the two
    spectra are not the same.
    """
    if not 0.0 <= position <= 1.0:
        raise ValueError(f"position must be within the grating, got {position}")
    return np.where(section_positions(sections) + 0.5 >= position, shift, 0.0)


def fiber_bragg_grating(
    frequencies: np.ndarray,
    *,
    length: float,
    index_modulation: float,
    bragg_wavelength: float,
    n_eff: float = SILICA_FIBER_NEFF,
    chirp: float = 0.0,
    apodization: str = "uniform",
    coupling_profile: np.ndarray | None = None,
    bragg_profile: np.ndarray | None = None,
    phase_profile: np.ndarray | None = None,
    sections: int = DEFAULT_GRATING_SECTIONS,
    ports: tuple[str, str] = ("in", "out"),
) -> SMatrix:
    """A periodic index modulation in a fibre core: the library's first mirror.

    Every other device in this module is matched at its facets and its matrix is
    off-diagonal. This one is *made* of reflection — a few thousand weak partial
    reflections that add in phase at one wavelength and cancel at every other —
    so ``s[in, in]`` is the entry carrying the physics, and ``s[out, out]`` is
    not the same number once the grating is chirped.

    **Assembled, not written down**, in the sense the ring is. The grating is cut
    into ``sections`` short enough to be uniform, each gets the closed-form
    transfer matrix of a uniform grating at its own local period and its own
    local strength, and the product is taken. Erdogan's formula for a uniform
    grating is nowhere in this function; it is in ``tests/test_photonics.py`` on
    the other side of the comparison, and a uniform grating cut into two hundred
    sections has to reproduce it. That is what buys chirp and apodization for
    nothing — a linearly chirped grating is the same product with a different
    local period per section, and it has no closed form at all.

    **The parameters, and what each one is.**

    ``index_modulation`` is the amplitude of the index fringe, the ``delta-n`` a
    writing process is specified by: 1e-4 is a strong grating and 1e-5 a weak
    one. It sets the coupling ``kappa = pi * delta-n / lambda``, and ``kappa L``
    is the single number deciding how much comes back — ``tanh**2(kappa L)``,
    which is 0.58 at ``kappa L = 1`` and 0.9993 at 4.

    ``bragg_wavelength`` is where it reflects, ``2 * n_eff * Lambda`` for a
    period ``Lambda``. **The average index is taken as compensated**, which is
    what a modern writing process aims for and what makes this parameter mean
    what it says. Writing a grating raises the average index as well, and an
    uncompensated one reflects at ``lambda_B * (1 + delta-n / n_eff)`` — 0.107 nm
    high at 1550 nm and ``delta-n = 1e-4``, which is two channels on a 100 GHz
    grid. Apodizing an uncompensated grating is worse than a shift: the average
    index then varies along the length, which is a chirp nobody asked for.

    ``chirp`` is the *total* change in local Bragg wavelength from the input face
    to the far one, so a positive value puts the short wavelengths at the near
    end where they turn round early and the long ones arrive later. That is
    ``D > 0``, the sign standard fibre has — so a compensator is a *negative*
    chirp, or the same grating entered from its other end, which is why
    ``s[in, in]`` and ``s[out, out]`` are different entries. The slope is
    ``2 * n_eff * L / (c * chirp)``: 965 ps/nm for a 10 cm grating chirped over
    1 nm, 57 km of standard fibre undone by a part the length of a finger.

    ``apodization`` shapes ``kappa`` along the length; see :data:`APODIZATIONS`.

    ``sections`` has to be large enough that one section is uniform over its own
    length; :data:`DEFAULT_GRATING_SECTIONS` says what the default is measured
    against. A uniform grating needs only one section and is cut into two hundred
    anyway, because that is what makes the closed-form comparison a test of this
    function rather than of a transcription.

    **The chirped grating's dispersion is geometric only while the chirp
    dominates.** ``2 * n_eff * L / (c * chirp)`` assumes a wavelength turns round
    where the local period says it should, and that stops being true once the
    grating's own uniform bandwidth approaches the chirp: at eighteen times the
    bandwidth the slope is within 0.02 % of geometric, at nine times it is 1.5 %
    high, and at one time there is no chirp left to speak of — the device is a
    uniform grating wearing a chirp parameter. The model is right in all three
    cases; it is the pocket formula that stops applying.

    **What is deliberately not here.** Loss, because there is none worth a
    parameter: a grating is centimetres of fibre and fibre is 0.2 dB/km, so the
    strongest grating described above absorbs 2e-5 dB. Birefringence, because a
    fibre grating's is a per-draw number this library will not invent. And loss
    to cladding modes on the short-wavelength side, which is real and is a
    coupling to modes this model does not have.

    Erdogan, *Fiber Grating Spectra*, J. Lightwave Technol. 15(8), 1997, for the
    coupled-mode formulation and the per-section matrix.
    """
    if length <= 0.0:
        raise ValueError(f"length must be positive, got {length}")
    if bragg_wavelength <= 0.0:
        raise ValueError(f"bragg_wavelength must be positive, got {bragg_wavelength}")
    if sections < 1:
        raise ValueError(f"sections must be at least 1, got {sections}")
    if apodization not in APODIZATIONS:
        raise ValueError(f"unknown apodization {apodization!r}; have {sorted(APODIZATIONS)}")
    if coupling_profile is not None and apodization != "uniform":
        raise ValueError(
            f"coupling_profile and apodization={apodization!r} both shape the coupling; "
            f"pass one. APODIZATIONS[{apodization!r}] is the array the name stands for."
        )
    if bragg_profile is not None and chirp != 0.0:
        raise ValueError(
            "bragg_profile and chirp both set the local Bragg wavelength; pass one. "
            "A linear chirp is bragg_wavelength + chirp * section_positions(sections)."
        )

    # The profiles decide the section count when they are given, because a length
    # that disagreed with them would have to be resolved by truncating or padding
    # somebody's design, and neither is a thing to do quietly.
    given = {
        "coupling_profile": coupling_profile,
        "bragg_profile": bragg_profile,
        "phase_profile": phase_profile,
    }
    lengths = {name: len(np.asarray(a)) for name, a in given.items() if a is not None}
    if lengths:
        if len(set(lengths.values())) != 1:
            raise ValueError(f"the profiles have different lengths: {lengths}")
        sections = next(iter(lengths.values()))
        if sections < 1:
            raise ValueError("a profile must have at least one section")

    frequencies = np.asarray(frequencies, dtype=np.float64)
    if np.any(frequencies <= 0.0):
        raise ValueError("a grating is described against optical frequency, which is positive")
    wavelengths = C_LIGHT / frequencies

    # Section centres, from the input face at u = -0.5 to the far one at +0.5.
    u = section_positions(sections)
    dz = length / sections
    local_bragg = (
        bragg_wavelength + chirp * u
        if bragg_profile is None
        else np.asarray(bragg_profile, dtype=np.float64)
    )
    strength = index_modulation * (
        APODIZATIONS[apodization](u)
        if coupling_profile is None
        else np.asarray(coupling_profile, dtype=np.float64)
    )
    grating_phase = (
        np.zeros(sections) if phase_profile is None else np.asarray(phase_profile, dtype=np.float64)
    )
    if np.any(local_bragg <= 0.0):
        raise ValueError("every local Bragg wavelength must be positive")

    # [R(near); S(near)] = F @ [R(far); S(far)], R the forward amplitude and S
    # the backward one. The product runs from the input face outward, so a
    # chirped grating knows which of its two ends the light arrived at.
    f11 = np.ones_like(wavelengths, dtype=np.complex128)
    f12 = np.zeros_like(f11)
    f21 = np.zeros_like(f11)
    f22 = np.ones_like(f11)

    for index in range(sections):
        detuning = 2.0 * np.pi * n_eff * (1.0 / wavelengths - 1.0 / local_bragg[index])
        kappa = np.pi * strength[index] / wavelengths
        # Complex on purpose. Inside the stop band the coupling exceeds the
        # detuning and gamma is real, which is the exponential decay that makes a
        # mirror; outside it gamma is imaginary and cosh/sinh become cos/sin,
        # which is the ripple. One expression covers both because numpy takes
        # the branch rather than being told which case this is.
        gamma = np.sqrt((kappa**2 - detuning**2).astype(np.complex128))
        # sinh(gamma dz)/gamma is entire in gamma**2 and is dz at gamma = 0 —
        # an unmodulated section exactly on resonance. Dividing there would put
        # a nan in the middle of an otherwise ordinary spectrum.
        safe = np.where(gamma == 0.0, 1.0, gamma)
        ratio = np.where(gamma == 0.0, dz, np.sinh(gamma * dz) / safe)
        cosine = np.cosh(gamma * dz)

        # **Conjugated from the form Erdogan writes**, and this is not cosmetic.
        # The coupled-mode literature carries fields as ``exp(+i beta z)`` and
        # this library carries them as ``exp(-i beta z)`` -- the convention
        # ``straight_waveguide`` and ``propagation_constant`` already fix. Left
        # in Erdogan's sign the magnitudes come out right and every group delay
        # comes out *negative*: a 10 cm chirped grating returned -1018 ps, which
        # is the correct 1018 ps of round trip with light leaving before it
        # arrives. Flipping the explicit ``i`` is the whole of the change, and it
        # is safe to do termwise because gamma**2 is real -- gamma is real inside
        # the stop band and imaginary outside it, and cosh and sinh(x)/x are both
        # even, so those two factors are real either way.
        # The phase rides on the *coupling* and not on the detuning, which is
        # what makes it a shift in the grating rather than a change of period:
        # the two off-diagonal terms take it with opposite signs, so the section
        # stays unimodular and the product stays lossless. A constant phase over
        # the whole length therefore does nothing at all, and only a jump
        # partway along is visible -- which is the physically right answer,
        # since where a grating's fringes start is not observable.
        turn = np.exp(1j * grating_phase[index])
        s11 = cosine + 1j * detuning * ratio
        s12 = 1j * kappa * ratio * turn
        s21 = -1j * kappa * ratio / turn
        s22 = cosine - 1j * detuning * ratio

        f11, f12, f21, f22 = (
            f11 * s11 + f12 * s21,
            f11 * s12 + f12 * s22,
            f21 * s11 + f22 * s21,
            f21 * s12 + f22 * s22,
        )

    # Drive the near face and the far one has nothing coming back into it, so
    # transmission is 1/F11 and reflection F21/F11. Drive the far face and
    # reciprocity returns the same transmission — det F is 1 section by section
    # and therefore overall — while the reflection is -F12/F11, equal to the
    # first for a uniform grating and not for a chirped one, where the two ends
    # really are different devices.
    transmission = 1.0 / f11
    reflection_near = f21 / f11
    reflection_far = -f12 / f11

    s = np.empty((frequencies.shape[0], 2, 2), dtype=np.complex128)
    s[:, 0, 0] = reflection_near
    s[:, 0, 1] = transmission
    s[:, 1, 0] = transmission
    s[:, 1, 1] = reflection_far
    return SMatrix(ports=ports, frequencies=frequencies, s=s)


def circulator(
    frequencies: np.ndarray,
    *,
    insertion_loss_db: float = 0.0,
    isolation_db: float = 0.0,
    return_loss_db: float = 0.0,
    ports: tuple[str, str, str] = ("p1", "p2", "p3"),
) -> SMatrix:
    """Three ports and one direction — 1 to 2, 2 to 3, 3 to 1. The way to a mirror.

    The only non-reciprocal device in this library, and it is here because a
    reflective one is useless without it: a grating sends its channel back out of
    the fibre it arrived on, and a circulator is what turns that fibre back into
    a forward path.

    The matrix is a cyclic permutation scaled by one hop's loss, so it is
    **unitary at 0 dB and never symmetric** — ``s[p2, p1]`` is the transmission
    and ``s[p1, p2]`` is zero until an isolation is given.
    :meth:`maiman.circuit.Circuit.solve` has always handled that; until now
    nothing outside a test made it do so.

    ``isolation_db`` is the reverse leak on every hop -- 2 to 1, 3 to 2, 1 to 3 --
    and 0 means an ideal circulator with none. A real one leaks 40 to 60 dB.

    **It used to be refused as a loop, and that was wrong.** The reasoning was
    that in the configuration this part exists for -- signal in at port 1, a
    grating on port 2, the drop out of port 3 -- the leak sends light back round to
    the grating. It does not. Reflected light coming back into port 2 leaves at
    port 3, forward, or at port 1, back toward the source, and neither returns to
    the grating. Solved exactly with :meth:`maiman.circuit.Circuit.solve`, which
    sums every bounce there is, a 40 dB isolated circulator's drop port equals the
    feed-forward ``tau * r * tau + iota`` with a difference of exactly zero.

    What isolation does break is narrower. Port 3 now hears port 2 *and* the
    direct leak from port 1, while port 2 still hears port 1 alone; those
    dependencies overlap and are acyclic, which a partition of ports into groups
    could not express and :class:`~maiman.component.PortGroup` now can.

    ``return_loss_db`` is the reflection back out of the port light entered by, and
    0 means none. **That one is a real cavity.** In the drop configuration it sends
    reflected light back into the grating, which reflects it again: at 40 dB of it
    the exact solve departs from feed-forward by 7.9e-3 and matches
    ``tau * r * tau / (1 - rho * r) + iota`` to 4e-16, rippling the drop port by
    about 0.07 dB either way. In a graph that is a cycle, and it runs only with a
    :class:`~maiman.components.Feedback` closing it.
    """
    if isolation_db < 0.0:
        raise ValueError(f"isolation_db must be zero (ideal) or positive, got {isolation_db}")
    if return_loss_db < 0.0:
        raise ValueError(f"return_loss_db must be zero (none) or positive, got {return_loss_db}")
    frequencies = np.asarray(frequencies, dtype=np.float64)
    amplitude = 10.0 ** (-insertion_loss_db / 20.0)
    leak = 10.0 ** (-isolation_db / 20.0) if isolation_db > 0.0 else 0.0
    echo = 10.0 ** (-return_loss_db / 20.0) if return_loss_db > 0.0 else 0.0

    count = len(ports)
    s = np.zeros((frequencies.shape[0], count, count), dtype=np.complex128)
    for source in range(count):
        s[:, (source + 1) % count, source] = amplitude
        s[:, source, (source + 1) % count] = leak
        s[:, source, source] = echo
    return SMatrix(ports=ports, frequencies=frequencies, s=s)


# --------------------------------------------------------------------------
# Long-period gratings
# --------------------------------------------------------------------------

#: Chebyshev nodes the mode tables are solved at before interpolating across a
#: band. Effective indices are smooth in wavelength, so a handful reproduce a
#: direct solve far below anything a notch's position could show.
LPG_GRID_POINTS = 17


def _lpg_modes(
    fibre: StepIndexFibre, wavelength: float, count: int, vector: bool
) -> tuple[list[Any], list[Any], Callable[..., float]]:
    """The core mode and the cladding modes a long-period grating couples it to.

    Scalar: LP01 and the LP0m. Vector: HE11 and every cladding mode of order one,
    HE1m and EH1m interleaved, which is what the scalar LP0m stand in for plus
    what they leave out.
    """
    if vector:
        return (
            vector_core_modes(fibre, wavelength, order=1),
            vector_cladding_modes(fibre, wavelength, order=1, count=count),
            vector_coupling,
        )
    return (
        core_modes(fibre, wavelength),
        cladding_modes(fibre, wavelength, count=count),
        lpg_coupling,
    )


def _mode_tables(
    fibre: StepIndexFibre,
    wavelengths: np.ndarray,
    count: int,
    *,
    coupling: bool = True,
    vector: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Core index, cladding indices and per-unit-modulation couplings at each wavelength."""
    core = np.empty(len(wavelengths))
    cladding = np.empty((len(wavelengths), count))
    kappa = np.zeros((len(wavelengths), count))
    for row, wavelength in enumerate(wavelengths):
        guided, modes, couple = _lpg_modes(fibre, float(wavelength), count, vector)
        if not guided:
            raise ValueError(f"the fibre guides no core mode at {wavelength * 1e9:.1f} nm")
        if len(modes) < count:
            raise ValueError(
                f"asked for {count} cladding modes at {wavelength * 1e9:.1f} nm and the fibre "
                f"has {len(modes)}"
            )
        core[row] = guided[0].effective_index
        for column, mode in enumerate(modes):
            cladding[row, column] = mode.effective_index
            if coupling:
                kappa[row, column] = couple(guided[0], mode, 1.0)
    return core, cladding, kappa


def _chebyshev_nodes(lo: float, hi: float, points: int) -> tuple[np.ndarray, np.ndarray]:
    unit = np.cos(np.pi * (np.arange(points) + 0.5) / points)
    return unit, 0.5 * (lo + hi) + 0.5 * (hi - lo) * unit


def _interpolated_tables(
    fibre: StepIndexFibre, wavelengths: np.ndarray, count: int, points: int, vector: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique, inverse = np.unique(wavelengths, return_inverse=True)
    if unique.size <= points:
        core, cladding, kappa = _mode_tables(fibre, unique, count, vector=vector)
        return core[inverse], cladding[inverse], kappa[inverse]

    lo, hi = float(unique[0]), float(unique[-1])
    unit, nodes = _chebyshev_nodes(lo, hi, points)
    core, cladding, kappa = _mode_tables(fibre, nodes, count, vector=vector)
    x = (wavelengths - 0.5 * (lo + hi)) / (0.5 * (hi - lo))
    cheb = np.polynomial.chebyshev

    def through(values: np.ndarray) -> np.ndarray:
        fitted = cheb.chebval(x, cheb.chebfit(unit, values, points - 1))
        return np.asarray(fitted).T

    return through(core), through(cladding), through(kappa)


def long_period_grating(
    frequencies: np.ndarray,
    *,
    fibre: StepIndexFibre,
    period: float,
    length: float,
    index_modulation: float,
    cladding_modes: int = 8,
    grid_points: int = LPG_GRID_POINTS,
    vector: bool = False,
    ports: tuple[str, str] = ("in", "out"),
    separation: float | None = None,
    gap_loss_db: float = 0.0,
) -> SMatrix:
    """A long-period grating's transmission, from coupled modes solved exactly.

    The core mode ``a_0`` and the cladding modes ``a_m`` obey, in a frame
    rotating with the grating::

        da_0/dz = -i sum_m kappa_m a_m
        da_m/dz = -i kappa_m a_0 - i Delta_m a_m
        Delta_m = k (n_m - n_0) + 2 pi / period

    which has constant coefficients for a uniform grating, so its solution is a
    matrix exponential, taken here by diagonalising the real symmetric matrix at
    every frequency. ``Delta_m`` vanishes where ``(n_0 - n_m) * period`` is one
    wavelength, which is the phase matching. For one mode alone this is
    ``|t|^2 = cos^2(gamma L) + (Delta / 2 gamma)^2 sin^2(gamma L)`` with
    ``gamma^2 = kappa^2 + Delta^2 / 4``, and the tests hold it to that.

    Nothing is reflected: a long-period grating couples forwards. The cladding's
    share leaves the model, as it leaves a coated fibre. The phase is reported in
    the core mode's own retarded frame, so the bulk ``exp(-i beta L)`` of the
    fibre is not in it.

    Indices and couplings are solved at ``grid_points`` Chebyshev nodes across
    the band and interpolated, because a mode solve takes tens of milliseconds
    and a spectrum has thousands of points.

    ``vector`` swaps the LP modes for the true ones of :mod:`maiman.vector_modes`:
    HE11 in the core, and the first ``cladding_modes`` cladding modes of order
    one -- HE1m and EH1m alternating, so twice as many reach as deep as the
    scalar LP0m do. The EH1m are the modes the scalar model has no counterpart
    for at this order; they couple, weakly, because the glass-air step mixes
    them. The equations are otherwise the same.

    **A pair.** With ``separation`` the grating is written twice, ``separation``
    apart, over fibre whose cladding still guides -- a bare stretch rather than a
    coated one. The first grating hands part of the core's light to the cladding;
    in the gap each cladding mode runs ahead of or behind the core by ``k (n_m -
    n_0)`` per metre; and the second grating hands back what it can. That is a
    Mach-Zehnder interferometer inside one fibre: the notch the first grating
    cuts is filled with fringes ``lambda^2 / (dn_g d)`` apart, ``dn_g`` the two
    modes' group-index difference. The transfer of each grating is the full
    matrix exponential, cladding amplitudes and all, and the gap is diagonal,
    so ``T G T`` is exact. ``gap_loss_db`` is the power the cladding modes lose in
    the gap -- to a recoat, a bend, a dirty surface -- and it washes the fringes
    out. The second grating's fringes continue the first's, as from one phase
    mask with its middle left unexposed; a zero separation is one grating of
    twice the length, which the tests hold it to.
    """
    if period <= 0.0:
        raise ValueError(f"period must be positive, got {period} m")
    if length < 0.0:
        raise ValueError(f"length must not be negative, got {length} m")
    if index_modulation < 0.0:
        raise ValueError(f"index modulation must not be negative, got {index_modulation}")
    if cladding_modes < 1:
        raise ValueError(f"couple at least one cladding mode, got {cladding_modes}")
    if separation is not None and separation < 0.0:
        raise ValueError(f"separation must not be negative, got {separation} m")
    if gap_loss_db < 0.0:
        raise ValueError(f"the gap's loss must not be negative, got {gap_loss_db} dB")
    grid = np.asarray(frequencies, dtype=float)
    wavelengths = C_LIGHT / grid
    core, cladding, kappa = _interpolated_tables(
        fibre, wavelengths, cladding_modes, grid_points, vector
    )

    k = 2.0 * np.pi / wavelengths
    size = cladding_modes + 1
    matrix = np.zeros((grid.size, size, size))
    matrix[:, 0, 1:] = kappa * index_modulation
    matrix[:, 1:, 0] = kappa * index_modulation
    diagonal = np.arange(1, size)
    matrix[:, diagonal, diagonal] = k[:, None] * (cladding - core[:, None]) + 2.0 * np.pi / period
    values, vectors = np.linalg.eigh(matrix)
    if separation is None:
        through = np.sum(vectors[:, 0, :] ** 2 * np.exp(-1j * values * length), axis=1)
    else:
        # The whole transfer, not just its core entry: the cladding amplitudes
        # the first grating leaves are what the second one hands back.
        transfer = np.einsum("fij,fj,fkj->fik", vectors, np.exp(-1j * values * length), vectors)
        gap = np.ones((grid.size, size), dtype=np.complex128)
        surviving = 10.0 ** (-gap_loss_db / 20.0)
        gap[:, 1:] = surviving * np.exp(-1j * matrix[:, diagonal, diagonal] * separation)
        through = np.einsum("fj,fj,fj->f", transfer[:, 0, :], gap, transfer[:, :, 0])

    s = np.zeros((grid.size, 2, 2), dtype=np.complex128)
    s[:, 1, 0] = through
    s[:, 0, 1] = through
    return SMatrix(ports=ports, frequencies=grid, s=s)


def long_period_resonances(
    fibre: StepIndexFibre,
    *,
    period: float,
    count: int,
    band: tuple[float, float] = (1.2e-6, 1.7e-6),
    grid_points: int = 33,
    vector: bool = False,
) -> list[tuple[int, float]]:
    """``(cladding mode rank, wavelength [m])`` wherever ``(n_0 - n_m) * period`` is a wavelength.

    Found on a Chebyshev fit of the mismatch across ``band``, then polished with
    two Newton steps on direct mode solves, so the wavelengths are the solver's
    own and not the interpolation's. A mode can phase-match twice in one band --
    either side of the turning point where its group index equals the core's --
    and both are returned.

    With ``vector`` the rank counts order-one cladding modes of both families
    together, in the order :func:`long_period_grating` couples them.
    """
    lo, hi = band
    if not 0.0 < lo < hi:
        raise ValueError(f"band must be an increasing pair of positive wavelengths, got {band}")
    unit, nodes = _chebyshev_nodes(lo, hi, grid_points)
    core, cladding, _ = _mode_tables(fibre, nodes, count, coupling=False, vector=vector)
    mismatch = core[:, None] - cladding - nodes[:, None] / period
    cheb = np.polynomial.chebyshev
    coefficients = cheb.chebfit(unit, mismatch, grid_points - 1)
    slopes = cheb.chebder(coefficients) / (0.5 * (hi - lo))
    dense = np.linspace(-1.0, 1.0, 4001)
    sampled = np.asarray(cheb.chebval(dense, coefficients)).T

    def direct(wavelength: float, rank: int) -> float:
        guided, modes, _ = _lpg_modes(fibre, wavelength, rank, vector)
        return guided[0].effective_index - modes[rank - 1].effective_index - wavelength / period

    found: list[tuple[int, float]] = []
    for column in range(count):
        values = sampled[:, column]
        crossings = np.flatnonzero(np.sign(values[:-1]) * np.sign(values[1:]) < 0)
        for index in crossings:
            left, right = float(dense[index]), float(dense[index + 1])
            reference = np.sign(values[index])
            for _ in range(60):
                middle = 0.5 * (left + right)
                if np.sign(cheb.chebval(middle, coefficients[:, column])) == reference:
                    left = middle
                else:
                    right = middle
            wavelength = 0.5 * (lo + hi) + 0.5 * (hi - lo) * 0.5 * (left + right)
            for _ in range(2):
                x = (wavelength - 0.5 * (lo + hi)) / (0.5 * (hi - lo))
                slope = float(cheb.chebval(x, slopes[:, column]))
                wavelength -= direct(wavelength, column + 1) / slope
            found.append((column + 1, float(wavelength)))
    return sorted(found, key=lambda pair: pair[1])


# --------------------------------------------------------------------------
# Tilted fibre Bragg gratings
# --------------------------------------------------------------------------

#: Chebyshev nodes a tilted grating's mode table is solved at. Fewer than a
#: long-period grating's, because its band is a tenth as wide and each node
#: solves every azimuthal order.
TILTED_GRID_POINTS = 9

#: Modes solved together at each frequency, nearest to phase matching first. At
#: thirty-two the transmission is within 0.002 of a solve with every mode in --
#: measured, on a comb of thirty -- while the matrix stays small enough that the
#: mode solve, not the exponential, is what a spectrum costs.
TILTED_NEAREST = 32


def _expm(matrices: np.ndarray) -> np.ndarray:
    """``exp`` of a stack of small matrices, by scaling, a Taylor series, and squaring.

    Scaled per matrix until its one-norm is below a quarter, where sixteen terms
    leave an error below ``1e-25``, then squared back up.
    """
    norm = np.abs(matrices).sum(axis=-2).max(axis=-1)
    squarings = np.maximum(0, np.ceil(np.log2(np.maximum(norm, 1e-300) / 0.25))).astype(int)
    scaled = matrices / (2.0**squarings)[..., None, None]
    identity = np.broadcast_to(np.eye(matrices.shape[-1], dtype=matrices.dtype), matrices.shape)
    out = identity.copy()
    for term in range(16, 0, -1):
        out = identity + scaled @ out / term
    for step in range(int(squarings.max(initial=0))):
        out = np.where((step < squarings)[..., None, None], out @ out, out)
    return out


def tilted_grating_coupling(
    core: Mode, cladding: Mode, *, period: float, tilt: float, index_modulation: float
) -> float:
    """Contra-directional coupling from the core's LP01 to one LP mode, under tilted fringes [1/m].

    The fringes are ``dn cos(K (z cos(theta) + x sin(theta)))`` with ``K = 2 pi /
    period``, the period measured normal to them. Across the core the transverse
    part expands by Jacobi-Anger, ``exp(i K_t r cos(phi)) = sum_l i^l J_l(K_t r)
    exp(i l phi)``, so an ``LP_lm`` mode oriented along the tilt is reached through

        ``O = 2 pi int_core psi_0 psi_lm J_l(K_t r) r dr / sqrt(N_0 N_lm)``

    with ``N`` each mode's power over the whole cross-section, ``2 pi`` or ``pi``
    times its radial integral. Then ``kappa = k n1 dn |O| / (2 sqrt(n_0 n_lm))``,
    the long-period grating's coupling with the fringe's phase across the core
    put in. Untilted, ``J_l(0)`` is 1 for ``l = 0`` and 0 otherwise: the core
    reflects into itself and into nothing else, which is a Bragg grating. The
    mode oriented the other way, ``sin(l phi)``, has no overlap with a tilt in
    ``x`` and is not coupled.
    """
    if core.wavelength != cladding.wavelength or core.fibre != cladding.fibre:
        raise ValueError("a coupling is between two modes of one fibre at one wavelength")
    if core.order != 0:
        raise ValueError("the grating is driven by the core's LP01")
    k = 2.0 * math.pi / core.wavelength
    transverse = 2.0 * math.pi / period * math.sin(tilt)
    order = cladding.order
    a = core.fibre.core_radius
    x, w = _legendre(256)
    r = 0.5 * a * (x + 1.0)
    if transverse == 0.0:
        if order:
            return 0.0
        phase = np.ones_like(r)
    else:
        phase = bessel_j(order, abs(transverse) * r)
    shared = float(np.sum(0.5 * a * w * core.field(r) * cladding.field(r) * phase * r))
    angular = 2.0 * math.pi if order == 0 else math.pi
    overlap = (
        2.0
        * math.pi
        * shared
        / math.sqrt(2.0 * math.pi * core.power() * angular * cladding.power())
    )
    return (
        k
        * core.fibre.core_index
        * index_modulation
        * abs(overlap)
        / (2.0 * math.sqrt(core.effective_index * cladding.effective_index))
    )


def vector_tilted_coupling(
    core: VectorMode,
    partner: VectorMode,
    *,
    period: float,
    tilt: float,
    index_modulation: float,
    polarization: str,
) -> float:
    """Contra-directional coupling from the core's HE11 to a vector mode, tilted fringes [1/m].

    :func:`tilted_grating_coupling` with the true fields. The fringes are tilted
    in the ``x-z`` plane, and each mode of order ``nu`` comes in two orientations
    of one effective index. An ``x``-polarized HE11 -- in the plane of the tilt,
    p -- reaches the one with ``E_r`` on ``cos(nu phi)``, and a ``y``-polarized
    one -- s -- the other; the cross terms are odd in ``phi`` and vanish. Writing
    ``E . E'*`` out in ``nu - 1`` and ``nu + 1`` harmonics and expanding the
    fringe's phase by Jacobi-Anger, the overlap is

        ``pi int_core (J_{nu-1}(K_t r) S -+ J_{nu+1}(K_t r) D) r dr``

    with ``S = e_r e_r' + e_phi e_phi'`` and ``D = e_r e_r' - e_phi e_phi'``: minus
    for p, plus for s. That is the whole of the polarization dependence. For
    ``nu = 0`` it leaves a p-polarized core reaching only TM0m and an s-polarized
    one only TE0m; for the others it weights HE and EH differently, which is what
    splits a real tilted grating's comb. Normalised as :func:`vector_coupling`,
    with each mode's power over the whole cross-section -- ``2 pi`` or ``pi``
    times its radial integral -- and reducing to it untilted.

    ``polarization`` is ``"p"`` or ``"s"``.
    """
    if polarization not in ("p", "s"):
        raise ValueError(f"polarization must be 'p' or 's', got {polarization!r}")
    if core.wavelength != partner.wavelength:
        raise ValueError("a coupling is between two modes at one wavelength")
    if core.order != 1:
        raise ValueError("the grating is driven by the core's HE11")
    transverse = 2.0 * math.pi / period * math.sin(tilt)
    nu = partner.order
    sign = -1.0 if polarization == "p" else 1.0

    def overlap(r: np.ndarray) -> np.ndarray:
        e1, p1, _, _ = core.transverse(r)
        e2, p2, _, _ = partner.transverse(r)
        same, differ = e1 * e2 + p1 * p2, e1 * e2 - p1 * p2
        if transverse == 0.0:
            # Untilted: J_0(0) = 1 and every other order vanishes.
            lower = np.full_like(r, 1.0 if nu == 1 else 0.0)
            upper = np.zeros_like(r)
        else:
            lower = bessel_j(nu - 1, transverse * r) if nu >= 1 else -bessel_j(1, transverse * r)
            upper = bessel_j(nu + 1, transverse * r)
        return math.pi * (lower * same + sign * upper * differ)

    shared = core.integrate(overlap, upto=core.radii[0])
    angular_core = math.pi
    angular_partner = 2.0 * math.pi if nu == 0 else math.pi
    k0 = 2.0 * math.pi / core.wavelength
    norm = math.sqrt(angular_core * core.power() * angular_partner * partner.power())
    return k0 * core.indices[0] * index_modulation * abs(shared) / (2.0 * norm)


@lru_cache(maxsize=512)
def _vector_tilted_modes(
    fibre: StepIndexFibre, wavelength: float, counts: tuple[int, ...]
) -> tuple[VectorMode, tuple[tuple[VectorMode, ...], ...]]:
    """HE11 and the cladding modes of every order at one wavelength, solved once.

    Cached because the two polarizations see the same modes and differ only in
    the overlaps, and a vector cladding solve is seconds where an overlap is a
    millisecond.
    """
    guided = vector_core_modes(fibre, wavelength, order=1)
    if not guided:
        raise ValueError(f"the fibre guides no core mode at {wavelength * 1e9:.1f} nm")
    rows = tuple(
        tuple(vector_cladding_modes(fibre, wavelength, order=order, count=count)) if count else ()
        for order, count in enumerate(counts)
    )
    return guided[0], rows


def _tilted_tables(
    fibre: StepIndexFibre,
    wavelengths: np.ndarray,
    *,
    period: float,
    tilt: float,
    max_order: int,
    lowest_index: float,
    vector: bool = False,
    polarization: str = "p",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """Core index, every coupled mode's index and coupling per unit ``dn``, at each wavelength.

    Column 0 is the core mode reflected into itself. The rest are cladding modes
    of each order down to ``lowest_index``, as many as every wavelength has.

    With ``vector`` the modes are :mod:`maiman.vector_modes`' HE, EH, TE and TM,
    of orders up to ``max_order + 1`` -- the LP order ``l`` is HE of ``l + 1`` and
    EH of ``l - 1`` -- and twice as many of each, one per family; the couplings
    are those ``polarization`` sees.
    """
    k_top = 2.0 * math.pi / float(wavelengths.min())
    floor = max(lowest_index, fibre.surrounding_index + 1e-6)
    per_order: list[list[list[Any]]] = []
    cores: list[Any] = []
    orders = max_order + 2 if vector else max_order + 1
    for wavelength in wavelengths:
        # The cladding's own index at this wavelength, if its glass disperses.
        n2 = fibre.at(float(wavelength)).cladding_index
        counts = []
        for order in range(orders):
            if floor >= n2:
                counts.append(0)
                continue
            q = k_top * math.sqrt(n2**2 - floor**2)
            scalar = max(1, math.ceil(q * fibre.cladding_radius / math.pi - order / 2) + 3)
            counts.append(2 * scalar + 2 if vector else scalar)
        if vector:
            core, rows_v = _vector_tilted_modes(fibre, float(wavelength), tuple(counts))
            cores.append(core)
            per_order.append([list(row) for row in rows_v])
            continue
        guided = core_modes(fibre, float(wavelength))
        if not guided:
            raise ValueError(f"the fibre guides no core mode at {wavelength * 1e9:.1f} nm")
        cores.append(guided[0])
        rows: list[list[Any]] = [
            cladding_modes(fibre, float(wavelength), order=order, count=count) if count else []
            for order, count in enumerate(counts)
        ]
        per_order.append(rows)

    columns: list[tuple[int, int]] = [(0, 0)]
    for order in range(orders):
        common = min(len(rows[order]) for rows in per_order)
        # Keep a rank only if it is above the floor somewhere in the band.
        kept = [
            rank
            for rank in range(common)
            if max(rows[order][rank].effective_index for rows in per_order) >= floor
        ]
        columns.extend((order, rank + 1) for rank in kept)

    core_index = np.array([mode.effective_index for mode in cores])
    index = np.empty((len(wavelengths), len(columns)))
    coupling = np.empty((len(wavelengths), len(columns)))
    for row, (core, rows) in enumerate(zip(cores, per_order, strict=True)):
        for column, (order, rank) in enumerate(columns):
            partner = core if rank == 0 else rows[order][rank - 1]
            index[row, column] = partner.effective_index
            if vector:
                coupling[row, column] = vector_tilted_coupling(
                    core,
                    partner,
                    period=period,
                    tilt=tilt,
                    index_modulation=1.0,
                    polarization=polarization,
                )
            else:
                coupling[row, column] = tilted_grating_coupling(
                    core, partner, period=period, tilt=tilt, index_modulation=1.0
                )
    return core_index, index, coupling, columns


def _tilted_interpolated(
    fibre: StepIndexFibre,
    wavelengths: np.ndarray,
    points: int,
    **settings: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]]]:
    unique, inverse = np.unique(wavelengths, return_inverse=True)
    if unique.size <= points:
        core, index, coupling, columns = _tilted_tables(fibre, unique, **settings)
        return core[inverse], index[inverse], coupling[inverse], columns
    lo, hi = float(unique[0]), float(unique[-1])
    unit, nodes = _chebyshev_nodes(lo, hi, points)
    core, index, coupling, columns = _tilted_tables(fibre, nodes, **settings)
    x = (wavelengths - 0.5 * (lo + hi)) / (0.5 * (hi - lo))
    cheb = np.polynomial.chebyshev

    def through(values: np.ndarray) -> np.ndarray:
        return np.asarray(cheb.chebval(x, cheb.chebfit(unit, values, points - 1))).T

    return through(core), through(index), through(coupling), columns


def _tilted_setup(
    period: float, tilt: float, length: float, index_modulation: float, max_order: int
) -> None:
    if period <= 0.0:
        raise ValueError(f"period must be positive, got {period} m")
    if not 0.0 <= tilt < math.pi / 2:
        raise ValueError(f"tilt must be between 0 and 90 degrees, got {math.degrees(tilt)}")
    if length < 0.0:
        raise ValueError(f"length must not be negative, got {length} m")
    if index_modulation < 0.0:
        raise ValueError(f"index modulation must not be negative, got {index_modulation}")
    if max_order < 0:
        raise ValueError(f"max_order must not be negative, got {max_order}")


def tilted_grating_spectrum(
    frequencies: np.ndarray,
    *,
    fibre: StepIndexFibre,
    period: float,
    tilt: float,
    length: float,
    index_modulation: float,
    max_order: int = 6,
    nearest: int | None = TILTED_NEAREST,
    grid_points: int = TILTED_GRID_POINTS,
    vector: bool = False,
    polarization: str = "p",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transmission, the core's own reflection, and the power sent back into the cladding.

    The forward core mode ``a`` and the backward modes ``b_m`` -- the core mode
    itself and every LP cladding mode of order up to ``max_order`` -- obey, in a
    frame rotating with the fringes::

        da/dz   = -i sum_m kappa_m b_m
        db_m/dz = +i kappa_m a + i Delta_m b_m
        Delta_m = k (n_0 + n_m) - 2 pi cos(theta) / period

    With ``b_m(L) = 0`` and ``a(0) = 1``, the transfer from the far end is ``E =
    exp(-M L)`` and ``t = 1 / E_00``, ``r_m = E_m0 t``. For one mode that is
    ``t = 1 / cosh(kappa L)`` on resonance, which the tests hold it to, and
    ``|t|^2 + sum |r_m|^2 = 1`` holds exactly because the equations conserve it.

    Every mode is coupled to every other only through the core, so at each
    frequency the ``nearest`` modes to phase matching are solved together and
    the rest enter as the shift they leave on the core, ``sum kappa^2 / Delta``
    -- what eliminating a mode that follows the core rather than taking power
    from it leaves behind. ``nearest=None`` solves them all, at a cost that grows
    as the cube of the comb's length. At the default the transmission of a ten
    millimetre grating is within 0.002 of that solve, which the tests measure;
    the power it accounts for is exact either way, because a truncated system
    conserves its own.

    Returns ``(t, r_core, cladding_power)`` on the grid; ``t`` and ``r_core`` are
    amplitudes in the rotating frame, ``cladding_power`` the sum of ``|r_m|^2``
    over cladding modes, which a coated fibre strips.

    With ``vector`` the modes are the true HE, EH, TE and TM ones and the comb is
    the one ``polarization`` sees -- ``"p"``, in the plane of the tilt, or ``"s"``
    across it: :func:`vector_tilted_coupling`.
    """
    _tilted_setup(period, tilt, length, index_modulation, max_order)
    if nearest is not None and nearest < 1:
        raise ValueError(f"solve at least one mode together, got {nearest}")
    grid = np.asarray(frequencies, dtype=float)
    wavelengths = C_LIGHT / grid
    axial = 2.0 * math.pi * math.cos(tilt) / period
    # Only modes that can phase match somewhere in the band, with a margin of a
    # few resonance widths either side, are worth solving for.
    # The lowest cladding index phase matches at the shortest wavelength.
    k_top = 2.0 * math.pi / float(wavelengths.min())
    lowest = axial / k_top - fibre.at(float(wavelengths.min())).core_index - 2e-3
    core, index, coupling, _ = _tilted_interpolated(
        fibre,
        wavelengths,
        grid_points,
        period=period,
        tilt=tilt,
        max_order=max_order,
        lowest_index=lowest,
        vector=vector,
        polarization=polarization,
    )
    k = 2.0 * math.pi / wavelengths
    delta = k[:, None] * (core[:, None] + index) - axial
    kappa = coupling * index_modulation

    size = delta.shape[1] if nearest is None else min(nearest, delta.shape[1])
    chosen = np.argsort(np.abs(delta), axis=1)[:, :size]
    d = np.take_along_axis(delta, chosen, axis=1)
    c = np.take_along_axis(kappa, chosen, axis=1)
    # A mode far from phase matching follows the core rather than taking power
    # from it -- ``b = -kappa a / Delta`` -- and what it leaves behind is a shift
    # of the core's own propagation constant, ``sum kappa^2 / Delta``. Dropped
    # modes are put back that way, which is exact to their first order and costs
    # one sum; without it their absence moves every resonance slightly.
    shift = (kappa**2 / delta).sum(axis=1) - (c**2 / d).sum(axis=1)
    m = np.zeros((grid.size, size + 1, size + 1), dtype=np.complex128)
    m[:, 0, 0] = 1j * shift
    m[:, 0, 1:] = -1j * c
    m[:, 1:, 0] = 1j * c
    m[:, np.arange(1, size + 1), np.arange(1, size + 1)] = 1j * d
    transfer = _expm(-m * length)
    t = 1.0 / transfer[:, 0, 0]
    r = transfer[:, 1:, 0] * t[:, None]
    is_core = chosen == 0
    r_core = np.sum(np.where(is_core, r, 0.0), axis=1)
    cladding_power = np.sum(np.where(is_core, 0.0, np.abs(r) ** 2), axis=1)
    return t, r_core, cladding_power


def tilted_fiber_bragg_grating(
    frequencies: np.ndarray,
    *,
    fibre: StepIndexFibre,
    period: float,
    tilt: float,
    length: float,
    index_modulation: float,
    max_order: int = 6,
    nearest: int | None = TILTED_NEAREST,
    grid_points: int = TILTED_GRID_POINTS,
    ports: tuple[str, str] = ("in", "out"),
    vector: bool = False,
    polarization: str = "p",
) -> SMatrix:
    """A tilted fibre Bragg grating as a two-port: its comb of cladding notches, and its mirror.

    Tilting the fringes lets a short-period grating reflect the core mode into
    cladding modes travelling backwards, every one at its own wavelength
    ``(n_0 + n_m) period / cos(theta)`` -- a comb of narrow notches a few
    nanometres apart, shortward of the Bragg reflection the tilt weakens. The
    cladding modes feel what surrounds the fibre and the core mode does not,
    which is why the comb is a refractometer that carries its own temperature
    reference. See :func:`tilted_grating_spectrum` for the equations.

    ``s[out, in]`` is the transmission and ``s[in, in]`` the core's reflection;
    what went into the cladding is stripped. The grating is uniform, so both ends
    see the same. Scalar LP modes by default, in which the two polarizations are
    one; with ``vector`` the comb ``polarization`` sees, split as a real one is.
    """
    t, r_core, _ = tilted_grating_spectrum(
        frequencies,
        fibre=fibre,
        period=period,
        tilt=tilt,
        length=length,
        index_modulation=index_modulation,
        max_order=max_order,
        nearest=nearest,
        grid_points=grid_points,
        vector=vector,
        polarization=polarization,
    )
    grid = np.asarray(frequencies, dtype=float)
    s = np.zeros((grid.size, 2, 2), dtype=np.complex128)
    s[:, 1, 0] = s[:, 0, 1] = t
    s[:, 0, 0] = s[:, 1, 1] = r_core
    return SMatrix(ports=ports, frequencies=grid, s=s)


def tilted_grating_resonances(
    fibre: StepIndexFibre,
    *,
    period: float,
    tilt: float,
    band: tuple[float, float],
    max_order: int = 6,
    grid_points: int = TILTED_GRID_POINTS,
    vector: bool = False,
    polarization: str = "p",
) -> list[tuple[int, int, float, float]]:
    """``(order, rank, wavelength, coupling per unit dn)`` for every resonance in ``band``.

    ``rank`` 0 is the core's own Bragg reflection, at ``2 n_0 period / cos(theta)``.
    Found where ``k (n_0 + n_m) = 2 pi cos(theta) / period`` on the Chebyshev
    interpolant of the mode table, so the wavelengths carry its accuracy, which
    the tests measure against direct solves.
    """
    lo, hi = band
    if not 0.0 < lo < hi:
        raise ValueError(f"band must be an increasing pair of positive wavelengths, got {band}")
    _tilted_setup(period, tilt, 0.0, 0.0, max_order)
    axial = 2.0 * math.pi * math.cos(tilt) / period
    unit, nodes = _chebyshev_nodes(lo, hi, grid_points)
    core, index, coupling, columns = _tilted_tables(
        fibre,
        nodes,
        period=period,
        tilt=tilt,
        max_order=max_order,
        lowest_index=axial * lo / (2.0 * math.pi) - fibre.at(lo).core_index - 2e-3,
        vector=vector,
        polarization=polarization,
    )
    cheb = np.polynomial.chebyshev
    mismatch = core[:, None] + index - nodes[:, None] * axial / (2.0 * math.pi)
    fit = cheb.chebfit(unit, mismatch, grid_points - 1)
    fit_coupling = cheb.chebfit(unit, coupling, grid_points - 1)
    dense = np.linspace(-1.0, 1.0, 4001)
    sampled = np.asarray(cheb.chebval(dense, fit)).T
    found = []
    for column, (order, rank) in enumerate(columns):
        values = sampled[:, column]
        for i in np.flatnonzero(np.sign(values[:-1]) * np.sign(values[1:]) < 0):
            left, right = float(dense[i]), float(dense[i + 1])
            reference = np.sign(values[i])
            for _ in range(60):
                middle = 0.5 * (left + right)
                if np.sign(cheb.chebval(middle, fit[:, column])) == reference:
                    left = middle
                else:
                    right = middle
            x = 0.5 * (left + right)
            wavelength = 0.5 * (lo + hi) + 0.5 * (hi - lo) * x
            found.append((order, rank, wavelength, float(cheb.chebval(x, fit_coupling[:, column]))))
    return sorted(found, key=lambda item: item[2])


# --------------------------------------------------------------------------
# Coupling into the die
# --------------------------------------------------------------------------


def gaussian_overlap(
    source_radius: float,
    target_radius: float,
    *,
    wavelength: np.ndarray | float,
    offset: float = 0.0,
    tilt: float = 0.0,
    gap: float = 0.0,
    index: float = 1.0,
) -> np.ndarray:
    """Power coupled between two Gaussian beams along one transverse axis.

    The source leaves its facet with a waist of ``source_radius`` (1/e field
    radius), crosses ``gap`` of a medium of ``index``, and arrives displaced by
    ``offset`` and tilted by ``tilt`` [rad] onto a target whose waist of
    ``target_radius`` sits at its own facet. Both beams are written as
    ``exp(-alpha x^2 + beta x)`` with complex ``alpha`` -- the gap is a complex
    beam parameter ``q = gap + i z_R``, the tilt a linear phase, the offset a
    shift -- and the overlap of two such fields is a Gaussian integral in closed
    form, so every misalignment is handled at once and none is an expansion:

        eta = |int E_s E_t dx|^2 / (int |E_s|^2 dx  int |E_t|^2 dx)

    Offset alone gives the textbook ``exp(-d^2 / w^2)`` for equal waists, and
    unequal waists alone ``2 w_s w_t / (w_s^2 + w_t^2)``; the tests hold it to
    both and to brute-force integration. The offset is measured at the target's
    facet, so a tilted source pivots about the point it lands on.

    Two axes of an elliptical mode are two calls multiplied together, because a
    Gaussian separates.
    """
    lam = np.asarray(wavelength, dtype=float)
    if source_radius <= 0.0 or target_radius <= 0.0:
        raise ValueError("beam radii must be positive")
    if gap < 0.0:
        raise ValueError(f"the gap must not be negative, got {gap} m")
    k = 2.0 * np.pi * index / lam
    rayleigh = np.pi * source_radius**2 * index / lam
    alpha_s = 1j * k / (2.0 * (gap + 1j * rayleigh))
    alpha_t = 1.0 / target_radius**2
    beta = 2.0 * alpha_s * offset - 1j * k * np.sin(tilt)
    total = alpha_s + alpha_t
    field = np.exp(-alpha_s * offset**2) * np.sqrt(np.pi / total) * np.exp(beta**2 / (4.0 * total))
    source_norm = np.sqrt(np.pi / (2.0 * alpha_s.real))
    target_norm = np.sqrt(np.pi / (2.0 * alpha_t))
    return np.abs(field) ** 2 / (source_norm * target_norm)


def gaussian_coupling(
    source_radius: float,
    target_radius: float,
    *,
    wavelength: np.ndarray | float,
    offset: float = 0.0,
    tilt: float = 0.0,
    gap: float = 0.0,
    index: float = 1.0,
) -> np.ndarray:
    """The complex amplitude :func:`gaussian_overlap` is the squared magnitude of.

    The source beam carries its Gouy phase, ``sqrt(q_0 / q)`` along one axis,
    so two beams that crossed different gaps arrive with the phases they really
    have and can be added. The carrier ``exp(-i k gap)`` is left to the caller,
    who knows which path it took. ``|gaussian_coupling|**2`` is
    :func:`gaussian_overlap` exactly; the tests hold the two together.
    """
    lam = np.asarray(wavelength, dtype=float)
    if source_radius <= 0.0 or target_radius <= 0.0:
        raise ValueError("beam radii must be positive")
    if gap < 0.0:
        raise ValueError(f"the gap must not be negative, got {gap} m")
    k = 2.0 * np.pi * index / lam
    rayleigh = np.pi * source_radius**2 * index / lam
    q = gap + 1j * rayleigh
    alpha_s = 1j * k / (2.0 * q)
    alpha_t = 1.0 / target_radius**2
    beta = 2.0 * alpha_s * offset - 1j * k * np.sin(tilt)
    total = alpha_s + alpha_t
    field = np.exp(-alpha_s * offset**2) * np.sqrt(np.pi / total) * np.exp(beta**2 / (4.0 * total))
    # Normalised at the source's own waist, where its power is fixed; the Gouy
    # factor carries that power, and its phase, across the gap.
    gouy = np.sqrt(1j * rayleigh / q)
    source_norm = np.sqrt(np.sqrt(np.pi / 2.0) * source_radius)
    target_norm = np.sqrt(np.sqrt(np.pi / 2.0) * target_radius)
    return np.asarray(gouy * field / (source_norm * target_norm))


def fresnel_reflectance(first_index: float, second_index: float) -> float:
    """Power reflected at normal incidence between two media, ``((n1 - n2)/(n1 + n2))^2``."""
    if first_index <= 0.0 or second_index <= 0.0:
        raise ValueError("refractive indices must be positive")
    return ((first_index - second_index) / (first_index + second_index)) ** 2


def marcuse_mode_field_radius(fibre: StepIndexFibre, wavelength: float) -> float:
    """The Gaussian that best stands in for a step-index core's LP01 [m, 1/e field radius].

    ``w / a = 0.65 + 1.619 V^-1.5 + 2.879 V^-6`` (Marcuse, *Bell Syst. Tech. J.*
    56, 1977), good to about a percent for ``1.2 < V < 2.4``. For the default
    :class:`~maiman.modes.StepIndexFibre` at 1550 nm, V = 2.04, it gives 5.11
    microns: 0.7 % wider than the best Gaussian fit to the LP01 field the mode
    solver computes, and matching that field in power to 99.2 %. The tests hold
    it to both.
    """
    v = fibre.v_number(wavelength)
    return fibre.core_radius * (0.65 + 1.619 * v**-1.5 + 2.879 * v**-6)


def edge_coupler(
    frequencies: np.ndarray,
    *,
    fibre_mode_radius: float,
    chip_mode_radius_x: float,
    chip_mode_radius_y: float,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    tilt: float = 0.0,
    gap: float = 0.0,
    gap_index: float = 1.0,
    fibre_index: float = 1.4682,
    mode_index: float = 1.45,
    etalon: bool = False,
    ports: tuple[str, str] = ("in", "out"),
) -> SMatrix:
    """A fibre butted against a chip facet: mode overlap, and two facets in the way.

    Transmission is the Gaussian overlap along each axis -- a round fibre mode
    onto an elliptical chip mode, offset and tilted in the horizontal plane
    across a gap -- times what each facet lets through, ``1 - R``. Each port
    reflects its own facet's ``sqrt(R)``.

    By default the two facets are independent losses, which is the average over
    an etalon fringe and is right for an index-matched joint or a source broader
    than the fringes. ``etalon`` makes them a cavity instead; see
    :func:`_edge_etalon`.

    The mode radii are held fixed across the band, as a spot-size converter's
    are to first order; the gap's diffraction is not, and is computed at every
    wavelength.
    """
    grid = np.asarray(frequencies, dtype=float)
    wavelength = C_LIGHT / grid
    if etalon:
        return _edge_etalon(
            grid,
            fibre_mode_radius=fibre_mode_radius,
            chip_mode_radius_x=chip_mode_radius_x,
            chip_mode_radius_y=chip_mode_radius_y,
            offset_x=offset_x,
            offset_y=offset_y,
            tilt=tilt,
            gap=gap,
            gap_index=gap_index,
            fibre_index=fibre_index,
            mode_index=mode_index,
            ports=ports,
        )
    horizontal = gaussian_overlap(
        fibre_mode_radius,
        chip_mode_radius_x,
        wavelength=wavelength,
        offset=offset_x,
        tilt=tilt,
        gap=gap,
        index=gap_index,
    )
    vertical = gaussian_overlap(
        fibre_mode_radius,
        chip_mode_radius_y,
        wavelength=wavelength,
        offset=offset_y,
        gap=gap,
        index=gap_index,
    )
    fibre_facet = fresnel_reflectance(fibre_index, gap_index)
    chip_facet = fresnel_reflectance(mode_index, gap_index)
    through = np.sqrt(horizontal * vertical * (1.0 - fibre_facet) * (1.0 - chip_facet))

    s = np.zeros((grid.size, 2, 2), dtype=np.complex128)
    s[:, 1, 0] = through
    s[:, 0, 1] = through
    s[:, 0, 0] = np.sqrt(fibre_facet)
    s[:, 1, 1] = np.sqrt(chip_facet)
    return SMatrix(ports=ports, frequencies=grid, s=s)


#: Smallest bounce kept in an etalon sum, relative to the first. The sum is
#: geometric in the two facets' reflections, so for two glass-air facets it
#: takes about eight bounces to get here.
ETALON_TOLERANCE = 1e-12


def _edge_etalon(
    grid: np.ndarray,
    *,
    fibre_mode_radius: float,
    chip_mode_radius_x: float,
    chip_mode_radius_y: float,
    offset_x: float,
    offset_y: float,
    tilt: float,
    gap: float,
    gap_index: float,
    fibre_index: float,
    mode_index: float,
    ports: tuple[str, str],
) -> SMatrix:
    """The two facets as a Fabry-Perot cavity, bounce by bounce.

    Light crossing the gap is partly reflected at the chip's facet, again at the
    fibre's, and crosses once more -- so the chip mode receives a sum of beams,
    the ``n``-th having crossed the gap ``2n + 1`` times. Each is a Gaussian beam
    that has diffracted over its whole path, so each is projected onto the chip
    mode with its own overlap and its own Gouy phase, times ``(r_f r_c)^n`` and
    the carrier over the path:

        ``t = sqrt((1 - R_f)(1 - R_c)) sum_n (r_f r_c)^n A_n exp(-i k (2n+1) g)``

    With the two facets parallel and the gap much shorter than the Rayleigh
    range this is the Airy function of a plane-wave etalon, which the tests
    hold it to; at zero gap it is the fibre touching the chip, the two facets
    one interface. What it adds to the Airy function is what makes real joints
    ripple less than a thin film would: every bounce diffracts further and
    overlaps the chip mode less.

    **And a tilted fibre walks the cavity off.** The fibre's facet is square to
    the fibre, so tilting the fibre by ``theta`` tilts one mirror against the
    other. Unfolded, each round trip turns the beam by ``2 theta`` and carries it
    sideways, so the ``n``-th beam lands ``2 n (n + 1) g theta`` from the first
    at an angle of ``(2n + 1) theta`` and barely overlaps the chip mode at all --
    which is the whole reason fibre arrays are polished at an angle. Paraxial:
    path lengths are taken as multiples of the gap, not of ``g / cos(theta)``.

    Reflections are the same sums seen from each port: the fibre's own facet,
    plus every beam that comes back into the fibre mode having crossed the gap
    ``2(n + 1)`` times.
    """
    wavelength = C_LIGHT / grid
    k = 2.0 * np.pi * gap_index / wavelength
    r_fibre = (gap_index - fibre_index) / (gap_index + fibre_index)
    r_chip = (gap_index - mode_index) / (gap_index + mode_index)
    fibre_facet, chip_facet = r_fibre**2, r_chip**2
    loop = r_fibre * r_chip
    bounces = (
        1
        if loop == 0.0
        else int(min(400, math.ceil(math.log(ETALON_TOLERANCE) / math.log(abs(loop)))))
    )

    def beam(
        source: tuple[float, float],
        target: tuple[float, float],
        legs: list[float],
        shift: float,
        angle: float,
        vertical: float,
    ) -> np.ndarray:
        # Diffraction follows the whole path. The carrier follows the axis: a
        # leg crossing the gap at an angle phi advances k g cos(phi) along it,
        # which is what the tilt's second-order phase is.
        path = len(legs) * gap
        carrier = k * gap * sum(math.cos(leg) for leg in legs)
        horizontal = gaussian_coupling(
            source[0],
            target[0],
            wavelength=wavelength,
            offset=shift,
            tilt=angle,
            gap=path,
            index=gap_index,
        )
        across = gaussian_coupling(
            source[1], target[1], wavelength=wavelength, offset=vertical, gap=path, index=gap_index
        )
        return np.asarray(horizontal * across * np.exp(-1j * carrier))

    fibre = (fibre_mode_radius, fibre_mode_radius)
    chip = (chip_mode_radius_x, chip_mode_radius_y)
    through = np.zeros(grid.size, dtype=np.complex128)
    fibre_echo = np.zeros(grid.size, dtype=np.complex128)
    chip_echo = np.zeros(grid.size, dtype=np.complex128)
    for n in range(bounces):
        weight = loop**n
        # Forward legs turn by 2 theta a round trip; each return leg matches
        # the forward leg before it.
        forward = [(2 * m + 1) * tilt for m in range(n + 1)]
        through += weight * beam(
            fibre,
            chip,
            forward + forward[:-1],
            offset_x + 2 * n * (n + 1) * gap * tilt,
            (2 * n + 1) * tilt,
            offset_y,
        )
        walk = 2.0 * gap * tilt * (n + 1) ** 2
        fibre_echo += (
            weight * r_chip * beam(fibre, fibre, forward + forward, walk, 2 * (n + 1) * tilt, 0.0)
        )
        # The chip's beam leaves square to its facet, so its legs are even
        # multiples of theta: out at 0, back at 2 theta, out at 2 theta, ...
        outward = [2 * m * tilt for m in range(n + 1)]
        inward = [2 * (m + 1) * tilt for m in range(n + 1)]
        chip_echo += (
            weight * r_fibre * beam(chip, chip, outward + inward, walk, 2 * (n + 1) * tilt, 0.0)
        )

    s = np.zeros((grid.size, 2, 2), dtype=np.complex128)
    s[:, 1, 0] = s[:, 0, 1] = np.sqrt((1.0 - fibre_facet) * (1.0 - chip_facet)) * through
    s[:, 0, 0] = -r_fibre + (1.0 - fibre_facet) * fibre_echo
    s[:, 1, 1] = -r_chip + (1.0 - chip_facet) * chip_echo
    return SMatrix(ports=ports, frequencies=grid, s=s)


def grating_coupler_centre(
    *,
    period: float,
    effective_index: float,
    group_index: float,
    reference_wavelength: float,
    angle: float,
    medium_index: float = 1.0,
) -> float:
    """The wavelength a grating coupler couples best at [m].

    Phase matching along the grating: the fibre's tangential wavenumber plus one
    grating vector is the guided mode's, ``lambda = period (n_eff(lambda) - n sin theta)``,
    with ``theta`` the fibre's angle in the medium it sits in. ``n_eff`` moves with
    wavelength, and linearising it through the group index solves exactly::

        lambda_c = period (n_g - n sin theta) / (1 + period (n_g - n_eff) / lambda_0)

    which is ``period (n_eff - n sin theta)`` when the grating does not disperse.
    The denominator is why a real grating coupler tunes by several nanometres per
    degree of fibre angle rather than the ten-plus the dispersionless formula says.
    """
    if period <= 0.0:
        raise ValueError(f"period must be positive, got {period} m")
    tangential = medium_index * np.sin(angle)
    return float(
        period
        * (group_index - tangential)
        / (1.0 + period * (group_index - effective_index) / reference_wavelength)
    )


def grating_coupler(
    frequencies: np.ndarray,
    *,
    period: float,
    effective_index: float,
    group_index: float,
    reference_wavelength: float,
    angle: float,
    medium_index: float = 1.0,
    peak_loss_db: float = 3.0,
    bandwidth_1db: float = 35e-9,
    back_reflection_db: float = 20.0,
    ports: tuple[str, str] = ("in", "out"),
) -> SMatrix:
    """A grating coupler's passband, centred where its phase matching says.

    The centre is physics, from :func:`grating_coupler_centre`. The shape is a
    compact model, as a foundry's PDK gives one: a Gaussian in wavelength falling
    by one decibel ``bandwidth_1db / 2`` either side of the centre, under a peak
    loss. A grating's real passband is set by its directionality and mode match,
    which need the full vertical stack to compute; those are what the two
    numbers stand in for, and they are what a PDK measures. Each port reflects
    ``back_reflection_db`` below its input.
    """
    if bandwidth_1db <= 0.0:
        raise ValueError(f"the 1 dB bandwidth must be positive, got {bandwidth_1db} m")
    grid = np.asarray(frequencies, dtype=float)
    wavelength = C_LIGHT / grid
    centre = grating_coupler_centre(
        period=period,
        effective_index=effective_index,
        group_index=group_index,
        reference_wavelength=reference_wavelength,
        angle=angle,
        medium_index=medium_index,
    )
    # exp(-(dl)^2 / 2 sigma^2) is 10^-0.1 at dl = bandwidth / 2.
    sigma_squared = (bandwidth_1db / 2.0) ** 2 / (2.0 * 0.1 * np.log(10.0))
    power = 10.0 ** (-peak_loss_db / 10.0) * np.exp(
        -((wavelength - centre) ** 2) / (2.0 * sigma_squared)
    )

    s = np.zeros((grid.size, 2, 2), dtype=np.complex128)
    s[:, 1, 0] = np.sqrt(power)
    s[:, 0, 1] = np.sqrt(power)
    echo = 10.0 ** (-back_reflection_db / 20.0)
    s[:, 0, 0] = echo
    s[:, 1, 1] = echo
    return SMatrix(ports=ports, frequencies=grid, s=s)


# --------------------------------------------------------------------------
# A grating coupler from its vertical stack
# --------------------------------------------------------------------------


def _passive(index: complex, name: str) -> complex:
    """Refuse an index that would amplify under this module's sign convention.

    Fields here carry ``exp(+i k z)``, so a medium that absorbs has a *positive*
    imaginary index: aluminium is ``1.44 + 16j``. The other sign is a gain medium,
    and it makes a stack reflect more than it is given -- silently, and only in
    the third decimal of a directionality, which is why it is refused here rather
    than left to be noticed.
    """
    if complex(index).imag < 0.0:
        raise ValueError(
            f"{name} is {index}, which amplifies: an absorbing medium's index has a "
            "positive imaginary part here, as aluminium's 1.44 + 16j does"
        )
    return complex(index)


def grating_directionality(
    wavelength: np.ndarray | float,
    *,
    emission_sine: np.ndarray | float,
    top_index: float = 1.0,
    silicon_index: float = 3.476,
    silicon_thickness: float = 220e-9,
    box_index: float = 1.444,
    box_thickness: float = 2e-6,
    substrate_index: complex = 3.476,
) -> np.ndarray:
    """Share of a grating's radiated power that goes up, rather than into the substrate.

    The grating is a sheet source in the middle of the silicon layer, radiating
    a TE plane wave at the angle phase matching sets, ``n_top sin(theta) =
    emission_sine * n_top``, equally up and down. What happens next is the
    stack's: the upward wave is partly reflected at the silicon's top, the
    downward one at the buried oxide and again at the substrate beneath it, and
    every reflection comes back through the source. With ``Gamma_up`` and
    ``Gamma_down`` the reflection each half of the stack presents at the source
    plane, the self-consistent amplitudes leaving it are::

        U = (1 + Gamma_down) / (1 - Gamma_up Gamma_down)
        D = (1 + Gamma_up)   / (1 - Gamma_up Gamma_down)

    and the powers escaping are ``|U|^2 (1 - |Gamma_up|^2)`` up and ``|D|^2 (1 -
    |Gamma_down|^2)`` down. The buried oxide is a Fabry-Perot for the downward
    wave: at the right thickness its reflection off the substrate returns in
    phase and most of the light goes up, and at the wrong one it cancels. At
    1550 nm, ten degrees, the share going up swings between 0.39 and 0.77 as the
    oxide goes from one micron to three, repeating every ``lambda / (2 n_box
    cos(theta_box))`` = 0.54 microns -- and the standard 2 microns sits at 0.59,
    well short of the top, which is 1.1 dB a 2.2 micron oxide would buy back.

    ``emission_sine`` is ``sin(theta)`` in the top medium. A complex
    ``substrate_index`` is a bottom mirror: aluminium's ``1.44 + 16j`` under the
    oxide sends almost everything that goes down back up, and the share leaving
    upward rises from 0.59 to 0.99. The tests hold this against a direct solve
    of every layer's boundary conditions, which shares no algebra with it, and
    against 0.5 for a stack with nothing to reflect.
    """
    lam = np.asarray(wavelength, dtype=float)
    substrate_index = _passive(substrate_index, "the substrate's index")
    k = 2.0 * np.pi / lam
    transverse = top_index * np.asarray(emission_sine, dtype=float)

    def kz(n: complex) -> np.ndarray:
        return np.asarray(k * np.sqrt(np.asarray(n**2 - transverse**2, dtype=np.complex128)))

    k_top, k_si, k_box, k_sub = kz(top_index), kz(silicon_index), kz(box_index), kz(substrate_index)
    half = 0.5 * silicon_thickness

    def fresnel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.asarray((a - b) / (a + b))

    up = fresnel(k_si, k_top) * np.exp(2j * k_si * half)
    phase = np.exp(2j * k_box * box_thickness)
    first, second = fresnel(k_si, k_box), fresnel(k_box, k_sub)
    down = (first + second * phase) / (1.0 + first * second * phase) * np.exp(2j * k_si * half)
    loop = 1.0 - up * down
    upward = np.abs((1.0 + down) / loop) ** 2 * (1.0 - np.abs(up) ** 2)
    downward = np.abs((1.0 + up) / loop) ** 2 * (1.0 - np.abs(down) ** 2)
    # A mirror underneath keeps some of what it does not send back, so the share
    # is of what escapes the stack either way rather than of the whole.
    escaping = upward + downward
    return np.asarray(upward / np.where(escaping > 0.0, escaping, 1.0))


def _grating_profile(
    x: np.ndarray, *, strength: float, strength_end: float, length: float
) -> np.ndarray:
    """The field a grating emits along itself, normalised over its length.

    A grating radiating at ``alpha(x)`` per metre empties as ``exp(-int alpha)``
    and emits ``sqrt(2 alpha(x))`` times what is left, which is energy
    conservation and nothing else. Uniform, that is the usual ``exp(-alpha x)``;
    tapered -- ``strength`` at the start to ``strength_end`` at the far end, the
    apodization a real coupler is drawn with -- it is the profile that makes the
    emitted beam rounder, and a rounder beam meets the fibre's Gaussian better.
    """
    slope = (strength_end - strength) / length
    alpha = strength + slope * x
    integral = strength * x + 0.5 * slope * x**2
    emitted = np.sqrt(2.0 * alpha) * np.exp(-integral)
    # Normalised over the length: what leaves the grating at all is counted once,
    # by the caller, and this is its shape.
    escaping = 1.0 - math.exp(-(strength + strength_end) * length)
    return np.asarray(emitted / math.sqrt(escaping))


def _grating_mode_overlap(
    wavelength: np.ndarray,
    mismatch: np.ndarray,
    *,
    strength: float,
    length: float,
    fibre_radius: np.ndarray | float,
    position: float,
    top_index: float,
    strength_end: float | None = None,
    curvature: np.ndarray | float = 0.0,
    nodes: int = 400,
) -> np.ndarray:
    """``|int p(x) g(x) exp(i k n dsin x) dx|^2``: the grating's beam onto the fibre's.

    ``p`` is the grating's own beam from :func:`_grating_profile`, ``g`` the
    fibre's Gaussian footprint centred at ``position``, and the phase is the
    angle between them. ``curvature`` is ``1 / R`` of the fibre's wavefront where
    it lands, which is zero at the surface and not zero a height above it.
    """
    x, w = _legendre(nodes)
    x = 0.5 * length * (x + 1.0)
    w = 0.5 * length * w
    emitted = _grating_profile(
        x,
        strength=strength,
        strength_end=strength if strength_end is None else strength_end,
        length=length,
    )
    radius = np.atleast_1d(np.asarray(fibre_radius, dtype=float))[..., None]
    fibre = (2.0 / (math.pi * radius**2)) ** 0.25 * np.exp(-((x - position) ** 2) / radius**2)
    k = 2.0 * np.pi * top_index / np.asarray(wavelength, dtype=float)
    bend = np.asarray(curvature, dtype=float)
    phase = np.exp(
        1j
        * (
            (k * np.asarray(mismatch, dtype=float))[..., None] * x
            - 0.5 * (k * bend)[..., None] * (x - position) ** 2
        )
    )
    return np.asarray(np.abs((phase * (w * emitted * fibre)).sum(axis=-1)) ** 2)


def grating_harmonic(index_contrast: float, duty: float, order: int) -> float:
    """The ``order``-th Fourier amplitude of a grating's square tooth profile.

    Teeth of index ``n + dn`` over a fraction ``duty`` of each period and ``n``
    over the rest are ``dn [duty + sum_m (2 / pi m) sin(pi m duty) cos(m K z)]``.
    The coefficient of ``cos(m K z)`` is what couples: the first order radiates,
    the second reflects. A fifty-fifty grating has no second harmonic at all --
    ``sin(2 pi * 0.5)`` is zero -- so it reflects nothing this way, which is why
    a duty cycle is worth stating.
    """
    if order < 1:
        raise ValueError(f"the order must be positive, got {order}")
    if not 0.0 <= duty <= 1.0:
        raise ValueError(f"the duty cycle lies between 0 and 1, got {duty}")
    return 2.0 * index_contrast * math.sin(math.pi * order * duty) / (math.pi * order)


def grating_tooth_reflection(
    wavelength: np.ndarray | float,
    *,
    period: float,
    effective_index: float,
    group_index: float,
    reference_wavelength: float,
    index_contrast: float,
    duty: float = 0.5,
    length: float,
    order: int = 2,
) -> np.ndarray:
    """What a grating coupler's own teeth send back down the waveguide, as an amplitude.

    The teeth that radiate also couple the forward guided mode to the backward
    one, through the ``order``-th harmonic of their corrugation -- the second for
    a grating radiating on its first. That is a fibre Bragg grating written in
    the chip, and it obeys the same coupled-mode solution::

        r = kappa sinh(gamma L) / (delta sinh(gamma L) + i gamma cosh(gamma L))

    with ``kappa = pi c_m / lambda`` from :func:`grating_harmonic`, ``delta =
    beta - m pi / Lambda`` the detuning from its own Bragg wavelength ``2 n_eff
    Lambda / m``, and ``gamma = sqrt(kappa^2 - delta^2)``, imaginary where the
    detuning wins. A coupler radiating at 1550 nm sits far off that resonance --
    a 611 nm period at ``n_eff = 2.71`` reflects at 1656 nm on its second order --
    so what comes back is the tail of it, ``2 kappa / delta`` in amplitude and
    ringing with the grating's length. A fifty-fifty duty cycle reflects nothing
    at all.

    The mismatch is not free to choose: the same phase matching that sets the
    emission angle sets it, ``delta = 2 pi n_top sin(theta) / lambda`` for
    ``m = 2``, so a grating that radiates straight up reflects on resonance and a
    tilted one does not. That is the whole reason a fibre is angled.

    Checked against this module's own transfer-matrix grating, which shares no
    algebra with the closed form.
    """
    lam = np.asarray(wavelength, dtype=float)
    if period <= 0.0 or length <= 0.0:
        raise ValueError("the grating's period and length must both be positive")
    dispersed = (
        effective_index
        + (effective_index - group_index) * (lam - reference_wavelength) / reference_wavelength
    )
    harmonic = grating_harmonic(index_contrast, duty, order)
    kappa = math.pi * harmonic / lam
    delta = 2.0 * np.pi * dispersed / lam - order * np.pi / period
    gamma = np.sqrt((kappa**2 - delta**2).astype(np.complex128))
    # Where kappa and delta are both tiny the ratio is 0/0 and the answer is 0.
    small = np.abs(gamma) * length < 1e-12
    gamma = np.where(small, 1.0, gamma)
    reflection = (
        kappa
        * np.sinh(gamma * length)
        / (delta * np.sinh(gamma * length) + 1j * gamma * np.cosh(gamma * length))
    )
    return np.asarray(np.where(small, 0.0, reflection))


def stack_reflection(
    wavelength: np.ndarray | float,
    *,
    emission_sine: np.ndarray | float,
    top_index: float = 1.0,
    silicon_index: float = 3.476,
    silicon_thickness: float = 220e-9,
    box_index: float = 1.444,
    box_thickness: float = 2e-6,
    substrate_index: complex = 3.476,
) -> np.ndarray:
    """The chip's own reflection seen from above, as a TE amplitude.

    The same stack :func:`grating_directionality` radiates into, taken from the
    other side: what a beam coming down from the fibre finds. Three interfaces
    folded in from the bottom up, each ``(r + r' e^{2 i k t}) / (1 + r r' e^{2 i k
    t})``. It is what makes the gap between fibre and chip an etalon.
    """
    lam = np.asarray(wavelength, dtype=float)
    substrate_index = _passive(substrate_index, "the substrate's index")
    k = 2.0 * np.pi / lam
    transverse = top_index * np.asarray(emission_sine, dtype=float)

    def kz(n: complex) -> np.ndarray:
        return np.asarray(k * np.sqrt(np.asarray(n**2 - transverse**2, dtype=np.complex128)))

    k_top, k_si = kz(top_index), kz(silicon_index)
    k_box, k_sub = kz(box_index), kz(substrate_index)

    def fresnel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.asarray((a - b) / (a + b))

    def over(
        inner: np.ndarray, outer: np.ndarray, below: np.ndarray, thickness: float
    ) -> np.ndarray:
        """The reflection at ``inner``'s side of a layer of ``outer`` with ``below`` under it."""
        phase = np.exp(2j * outer * thickness)
        first = fresnel(inner, outer)
        return np.asarray((first + below * phase) / (1.0 + first * below * phase))

    at_box = fresnel(k_box, k_sub)
    at_silicon = over(k_si, k_box, at_box, box_thickness)
    return over(k_top, k_si, at_silicon, silicon_thickness)


def grating_coupler_stack(
    frequencies: np.ndarray,
    *,
    period: float,
    effective_index: float,
    group_index: float,
    reference_wavelength: float,
    angle: float,
    top_index: float = 1.0,
    silicon_index: float = 3.476,
    silicon_thickness: float = 220e-9,
    box_index: float = 1.444,
    box_thickness: float = 2e-6,
    substrate_index: float = 3.476,
    strength: float = 0.14e6,
    strength_end: float | None = None,
    length: float = 20e-6,
    fibre_radius: float = 5.2e-6,
    grating_radius_y: float = 5.2e-6,
    fibre_position: float | None = None,
    fibre_height: float = 0.0,
    fibre_index: float = 1.444,
    index_contrast: float | None = None,
    duty: float = 0.5,
    back_reflection_db: float = 20.0,
    extinction_db: float = 0.0,
    ports: tuple[str, str] = ("in", "out"),
) -> SMatrix:
    """A grating coupler's passband computed from its vertical stack, not declared.

    Three things multiply, each from physics this library already has or adds
    here:

    1. **What leaves the grating at all**, ``1 - exp(-2 strength length)``:
       ``strength`` is the field's decay rate along the grating, set by the etch.
    2. **What goes up rather than down**, :func:`grating_directionality`, from the
       silicon, the buried oxide and the substrate beneath it -- at the emission
       angle each wavelength actually leaves at.
    3. **What the fibre accepts.** The grating's beam is exponential, the fibre's
       Gaussian, and the overlap of the two is at best 80 percent. The emission
       angle moves with wavelength by phase matching, ``n_top sin(theta) =
       n_eff(lambda) - lambda / period``, with the same dispersion
       :func:`grating_coupler_centre` uses; the fibre's angle does not move, so
       away from the centre the two beams meet at an angle and the overlap
       falls. **That is the passband.** Its width is the fibre's angular
       acceptance divided by how fast the emission angle turns with wavelength,
       which is why a grating in thicker silicon, with a lower group index, is
       broader.

    Across the grating (``y``) the fibre meets the grating's lateral mode as two
    Gaussians, which does not move with wavelength. The fibre sits where it
    couples best at the centre wavelength unless ``fibre_position`` (from the
    grating's start) is given. ``extinction_db`` is taken off the whole
    passband, which is how a TM input is rejected.

    **What the teeth send back.** With ``index_contrast`` and ``duty`` the
    reflection into the waveguide is :func:`grating_tooth_reflection` rather than
    the flat ``back_reflection_db``: the same corrugation that radiates on its
    first order reflects on its second, far off resonance because the fibre is
    angled, and not at all at a duty cycle of exactly one half.

    **The height the fibre sits at.** ``fibre_height`` is the gap between the
    fibre's facet and the chip. Three things follow, and all three are measured
    rather than declared: the beam is wider where it lands, by
    ``sqrt(1 + (h / z_R)^2)``; it arrives curved, which the overlap integral
    carries as a phase; and the gap is an etalon between the fibre's facet and
    the chip's own reflection, :func:`stack_reflection`, whose fringes are
    ``lambda^2 / 2 n h`` apart. The tilt walks each round trip sideways by
    ``2 h tan(theta)``, so an angled fibre damps the ripple the way an angled
    facet damps an edge coupler's -- which is the same calculation, and the same
    functions.

    **Apodization.** ``strength_end`` tapers the radiation strength along the
    grating, from ``strength`` at the start. The emitted profile follows from
    energy conservation, :func:`_grating_profile`, and a taper makes it rounder
    and so closer to the fibre's Gaussian than the uniform grating's 80 percent.

    ``substrate_index`` may be complex, which is a bottom mirror.

    Not in it: the etalon's own effect on what the grating radiates, which is
    taken as a multiplier on the coupling rather than solved with the grating in
    the cavity, and capped at unity.
    """
    if period <= 0.0:
        raise ValueError(f"period must be positive, got {period} m")
    if strength <= 0.0 or length <= 0.0:
        raise ValueError("the grating's strength and length must both be positive")
    if silicon_thickness <= 0.0 or box_thickness < 0.0:
        raise ValueError("the silicon must have a thickness, and the oxide must not be negative")
    if fibre_height < 0.0:
        raise ValueError(f"the fibre's height must not be negative, got {fibre_height} m")
    if strength_end is not None and strength_end <= 0.0:
        raise ValueError(f"the grating's strength must stay positive, got {strength_end}")
    grid = np.asarray(frequencies, dtype=float)
    wavelength = C_LIGHT / grid
    dispersed = (
        effective_index
        + (effective_index - group_index)
        * (wavelength - reference_wavelength)
        / reference_wavelength
    )
    emission = (dispersed - wavelength / period) / top_index
    fibre_sine = math.sin(angle)
    # A beam that has travelled to the chip is wider there, and curved. Both come
    # from the same Rayleigh range, one per wavelength.
    travel = fibre_height / math.cos(angle)
    rayleigh = np.pi * fibre_radius**2 * top_index / wavelength
    landed = fibre_radius * np.sqrt(1.0 + (travel / rayleigh) ** 2)
    curvature = travel / (travel**2 + rayleigh**2) if fibre_height > 0.0 else np.zeros_like(landed)
    # The fibre's round mode lands on the chip stretched along the tilt.
    footprint = landed / math.cos(angle)

    if fibre_position is None:
        centre = grating_coupler_centre(
            period=period,
            effective_index=effective_index,
            group_index=group_index,
            reference_wavelength=reference_wavelength,
            angle=angle,
            medium_index=top_index,
        )
        candidates = np.linspace(0.0, length, 401)
        at_centre = float(
            fibre_radius
            * math.sqrt(1.0 + (travel / (math.pi * fibre_radius**2 * top_index / centre)) ** 2)
            / math.cos(angle)
        )
        matched = np.array(
            [
                _grating_mode_overlap(
                    np.array([centre]),
                    np.zeros(1),
                    strength=strength,
                    strength_end=strength_end,
                    length=length,
                    fibre_radius=at_centre,
                    position=float(place),
                    top_index=top_index,
                )[0]
                for place in candidates
            ]
        )
        fibre_position = float(candidates[int(matched.argmax())])

    radiating = np.abs(emission) < 1.0
    sine = np.where(radiating, emission, 0.0)
    upward = grating_directionality(
        wavelength,
        emission_sine=sine,
        top_index=top_index,
        silicon_index=silicon_index,
        silicon_thickness=silicon_thickness,
        box_index=box_index,
        box_thickness=box_thickness,
        substrate_index=substrate_index,
    )
    along = _grating_mode_overlap(
        wavelength,
        sine - fibre_sine,
        strength=strength,
        strength_end=strength_end,
        length=length,
        fibre_radius=footprint,
        position=fibre_position,
        top_index=top_index,
        curvature=curvature,
    )
    across = gaussian_overlap(
        fibre_radius, grating_radius_y, wavelength=wavelength, gap=travel, index=top_index
    )
    taper = strength if strength_end is None else strength_end
    leaves = 1.0 - math.exp(-(strength + taper) * length)
    power = np.where(radiating, leaves * upward * along * across, 0.0) * 10.0 ** (
        -extinction_db / 10.0
    )
    power = power * _fibre_gap_etalon(
        wavelength,
        emission_sine=sine,
        radius=fibre_radius,
        height=fibre_height,
        angle=angle,
        top_index=top_index,
        fibre_index=fibre_index,
        silicon_index=silicon_index,
        silicon_thickness=silicon_thickness,
        box_index=box_index,
        box_thickness=box_thickness,
        substrate_index=substrate_index,
    )

    s = np.zeros((grid.size, 2, 2), dtype=np.complex128)
    s[:, 1, 0] = s[:, 0, 1] = np.sqrt(np.minimum(power, 1.0))
    if index_contrast is None:
        s[:, 0, 0] = s[:, 1, 1] = 10.0 ** (-back_reflection_db / 20.0)
    else:
        teeth = grating_tooth_reflection(
            wavelength,
            period=period,
            effective_index=effective_index,
            group_index=group_index,
            reference_wavelength=reference_wavelength,
            index_contrast=index_contrast,
            duty=duty,
            length=length,
        )
        s[:, 0, 0] = s[:, 1, 1] = teeth
    return SMatrix(ports=ports, frequencies=grid, s=s)


def _fibre_gap_etalon(
    wavelength: np.ndarray,
    *,
    emission_sine: np.ndarray,
    radius: float,
    height: float,
    angle: float,
    top_index: float,
    fibre_index: float,
    **stack: object,
) -> np.ndarray:
    """``|1 / (1 - rho)|^2``: what the gap between fibre and chip does to the coupling.

    ``rho`` is one round trip -- off the chip's own reflection, back off the
    fibre's facet, and onto the fibre's mode again, displaced by ``2 h
    tan(theta)`` and diffracted over ``2 h / cos(theta)``. Zero height is no
    cavity and no ripple.
    """
    if height <= 0.0:
        return np.ones_like(wavelength)
    chip = stack_reflection(
        wavelength,
        emission_sine=emission_sine,
        top_index=top_index,
        **stack,  # type: ignore[arg-type]
    )
    facet = math.sqrt(fresnel_reflectance(fibre_index, top_index))
    path = 2.0 * height / math.cos(angle)
    returning = gaussian_coupling(
        radius,
        radius,
        wavelength=wavelength,
        offset=2.0 * height * math.tan(angle),
        gap=path,
        index=top_index,
    )
    carrier = np.exp(-2j * np.pi * top_index * path / wavelength)
    rho = facet * chip * returning * carrier
    return np.asarray(np.abs(1.0 / (1.0 - rho)) ** 2)
