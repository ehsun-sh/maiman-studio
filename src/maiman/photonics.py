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

from collections.abc import Callable, Mapping, Sequence
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

    **Return loss is the real cavity, and it is not modelled.** A reflection back
    out of the port light entered by bounces into the grating again: at 40 dB of it
    the exact solve departs from feed-forward by 7.9e-3 and matches
    ``tau * r * tau / (1 - rho * r) + iota`` to 4e-16, rippling the drop port by
    about 0.07 dB either way. That is a genuine loop and would need an iteration
    count.
    """
    if isolation_db < 0.0:
        raise ValueError(f"isolation_db must be zero (ideal) or positive, got {isolation_db}")
    frequencies = np.asarray(frequencies, dtype=np.float64)
    amplitude = 10.0 ** (-insertion_loss_db / 20.0)
    leak = 10.0 ** (-isolation_db / 20.0) if isolation_db > 0.0 else 0.0

    count = len(ports)
    s = np.zeros((frequencies.shape[0], count, count), dtype=np.complex128)
    for source in range(count):
        s[:, (source + 1) % count, source] = amplitude
        s[:, source, (source + 1) % count] = leak
    return SMatrix(ports=ports, frequencies=frequencies, s=s)
