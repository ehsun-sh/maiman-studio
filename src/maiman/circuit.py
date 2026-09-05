"""Scattering matrices, and the reduction that turns a circuit into one.

A fibre link is a chain: each block takes a waveform and returns a waveform, and
the scheduler runs them in order. A photonic integrated circuit is not. Light in
a ring goes round, comes back to the coupler it entered by, and interferes with
itself; a resonance is that feedback reaching steady state. There is no order to
run the blocks in, because the answer at every port depends on the answer at
every other port at the same time.

So a PIC needs a different kind of solve, and this is it. Every device is
described by a **scattering matrix** — for a unit amplitude going *into* port j,
``s[i, j]`` is what comes *out* of port i — and a circuit is described by which
ports are wired to which. This module reduces the second to the first: given the
instances and the connections, it returns the scattering matrix of the whole,
with the internal ports eliminated exactly.

**On not using an existing solver.** The roadmap said to integrate one rather
than write one, on the grounds that a bidirectional S-matrix solver is a large
project. That was worth checking before believing, and for the *solver* it does
not hold. The reduction is one linear identity: split the ports into external and
internal, write the connections as a permutation ``C`` that hands each internal
port's outgoing wave to its partner as an incoming wave, and

    a_int  = C S_ii a_int + C S_ie a_ext
    b_ext  = S_ee a_ext + S_ei a_int

eliminate ``a_int``, and the circuit's matrix is

    S = S_ee + S_ei (I - C S_ii)^-1 C S_ie

which is :meth:`Circuit.solve`, and it is thirty lines of numpy. What is
genuinely large in that ecosystem is the *PDK* side — process design kits,
layout, fitted component models — and that is still worth integrating rather
than inventing.

The cost of the other choice was measured rather than assumed. Installing SAX
resolves to **37 packages**, including jax and a 66 MB jaxlib, plus matplotlib,
pandas, scipy, sympy, xarray and pydantic — to perform one
``numpy.linalg.solve``. And one of them is disqualifying on its own: ``klujax``,
the sparse back-end, is **LGPL-2.0-only**. This project already refuses FFTW over
exactly that question (see the note in ``pyproject.toml``), and the architecture
document says dependency licences are checked before adoption, not after. So the
reduction is here, in numpy, and the answers it gives are checked against closed
forms from the literature in ``tests/test_circuit.py``.

**Reciprocity and reflection are not assumed.** The reduction never transposes
anything and never drops a term, so a non-reciprocal device (an isolator) and a
reflecting one (a facet, a Bragg grating) both solve correctly. The device
*models* in :mod:`maiman.photonics` happen to be reciprocal and matched; the
solver does not know that about them, and a test asserts it by solving a circuit
built from a deliberately non-reciprocal block.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SMatrix:
    """A scattering matrix on a frequency grid.

    ``s[f, i, j]`` is the complex amplitude leaving port ``ports[i]`` for unit
    amplitude entering port ``ports[j]``, at ``frequencies[f]``. Amplitude, not
    power: ``abs(s)**2`` is the power transmission, and the phase is what makes a
    ring resonate.

    The frequency axis is carried with the matrix rather than passed alongside
    it, because a circuit built from devices evaluated on different grids is not
    a circuit — it is two answers to different questions, and combining them
    silently is exactly the sort of thing that produces a plausible-looking
    spectrum with nothing behind it. :meth:`Circuit.solve` refuses it.
    """

    ports: tuple[str, ...]
    frequencies: np.ndarray
    s: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.ports)
        if len(set(self.ports)) != n:
            raise ValueError(f"port names must be unique, got {self.ports}")
        frequencies = np.asarray(self.frequencies, dtype=np.float64)
        if frequencies.ndim != 1:
            raise ValueError(f"frequencies must be 1-D, got {frequencies.ndim}-D")
        s = np.asarray(self.s, dtype=np.complex128)
        if s.shape != (frequencies.shape[0], n, n):
            raise ValueError(
                f"s must have shape (frequencies, ports, ports) = "
                f"{(frequencies.shape[0], n, n)}, got {s.shape}"
            )
        object.__setattr__(self, "frequencies", frequencies)
        object.__setattr__(self, "s", s)

    def index(self, port: str) -> int:
        """Row and column of ``port``."""
        try:
            return self.ports.index(port)
        except ValueError:
            raise KeyError(f"no port {port!r}; have {list(self.ports)}") from None

    def transmission(self, output: str, input_: str) -> np.ndarray:
        """Complex amplitude from ``input_`` to ``output``, one value per frequency."""
        return self.s[:, self.index(output), self.index(input_)]

    def power(self, output: str, input_: str) -> np.ndarray:
        """Power transmission from ``input_`` to ``output``, one value per frequency."""
        return np.abs(self.transmission(output, input_)) ** 2

    def is_unitary(self, *, tol: float = 1e-9) -> bool:
        """Whether the device is lossless: ``S^H S = I`` at every frequency.

        A check on a *model*, not on the solver. A coupler whose split ratio does
        not square up to one invents or destroys light, and it does so quietly —
        the spectrum still looks like a coupler's.
        """
        n = len(self.ports)
        product = np.conj(np.swapaxes(self.s, -1, -2)) @ self.s
        return bool(np.allclose(product, np.eye(n), atol=tol))


class Circuit:
    """Instances, the wires between them, and the ports left facing outward.

    Built by mutation and then solved, because that is what constructing a
    circuit is; the result — an :class:`SMatrix` — is immutable like every other
    value the engine passes around.

    A port that is neither linked nor exposed is **not an error**. It is a port
    facing a facet that nothing is driving, and ``a = 0`` is exactly the right
    boundary condition for it: light reaching it leaves the circuit and does not
    come back. The reduction carries it through as external and then drops it,
    which *is* that condition — arrived at by doing nothing rather than by
    special-casing.
    """

    def __init__(self) -> None:
        self._instances: dict[str, SMatrix] = {}
        self._links: list[tuple[tuple[str, str], tuple[str, str]]] = []
        self._exposed: dict[str, tuple[str, str]] = {}

    def add(self, name: str, matrix: SMatrix) -> Circuit:
        """Place a device in the circuit under ``name``."""
        if name in self._instances:
            raise ValueError(f"{name!r} is already in this circuit")
        self._instances[name] = matrix
        return self

    def link(self, a_instance: str, a_port: str, b_instance: str, b_port: str) -> Circuit:
        """Wire one port to another. A port may be wired at most once."""
        a, b = (a_instance, a_port), (b_instance, b_port)
        if a == b:
            raise ValueError(f"cannot link {a[0]}.{a[1]} to itself")
        for end in (a, b):
            self._check(end)
            if any(end in link for link in self._links):
                raise ValueError(f"{end[0]}.{end[1]} is already linked")
            if end in self._exposed.values():
                raise ValueError(f"{end[0]}.{end[1]} is exposed and cannot also be linked")
        self._links.append((a, b))
        return self

    def expose(self, name: str, instance: str, port: str) -> Circuit:
        """Make one instance's port a port of the circuit, under ``name``."""
        end = (instance, port)
        self._check(end)
        if name in self._exposed:
            raise ValueError(f"{name!r} is already an exposed port")
        if any(end in link for link in self._links):
            raise ValueError(f"{instance}.{port} is linked and cannot also be exposed")
        if end in self._exposed.values():
            raise ValueError(f"{instance}.{port} is already exposed")
        self._exposed[name] = end
        return self

    def _check(self, end: tuple[str, str]) -> None:
        instance, port = end
        if instance not in self._instances:
            raise KeyError(f"no instance {instance!r}; have {sorted(self._instances)}")
        if port not in self._instances[instance].ports:
            raise KeyError(
                f"{instance!r} has no port {port!r}; have {list(self._instances[instance].ports)}"
            )

    def solve(self) -> SMatrix:
        """The scattering matrix of the whole, over the exposed ports.

        Exact, not iterative. The feedback a circuit contains is a linear system
        and this solves it; there is no convergence criterion to pick and no
        number of round trips to truncate at. A ring's resonance appears as
        ``(I - C S_ii)^-1`` becoming large, which is the same statement as the
        round trip approaching unit gain — reached in one step rather than
        summed.
        """
        if not self._exposed:
            raise ValueError("a circuit with no exposed ports has nothing to solve for")
        grids = [m.frequencies for m in self._instances.values()]
        for grid in grids[1:]:
            if grid.shape != grids[0].shape or not np.array_equal(grid, grids[0]):
                raise ValueError(
                    "every device in a circuit must be evaluated on the same frequency "
                    "grid; combining two grids would silently answer a question nobody "
                    "asked"
                )
        frequencies = grids[0]
        rows = np.arange(frequencies.shape[0])

        keys = [(name, port) for name, m in self._instances.items() for port in m.ports]
        position = {key: i for i, key in enumerate(keys)}
        total = len(keys)

        # Block-diagonal: every device's own matrix, nothing coupling them yet.
        block = np.zeros((frequencies.shape[0], total, total), dtype=np.complex128)
        for name, matrix in self._instances.items():
            where = np.array([position[(name, port)] for port in matrix.ports], dtype=np.intp)
            block[np.ix_(rows, where, where)] = matrix.s

        partner: dict[int, int] = {}
        for a, b in self._links:
            partner[position[a]] = position[b]
            partner[position[b]] = position[a]
        internal = np.array(sorted(partner), dtype=np.intp)
        external = np.array([i for i in range(total) if i not in partner], dtype=np.intp)

        def sub(take_rows: np.ndarray, take_cols: np.ndarray) -> np.ndarray:
            return block[np.ix_(rows, take_rows, take_cols)]

        if internal.size:
            # C hands each internal port's outgoing wave to its partner as an
            # incoming one. It is a permutation, so it is its own inverse;
            # building it explicitly costs nothing at these sizes and puts the
            # identity in the code rather than only in the comment above.
            local = {int(index): k for k, index in enumerate(internal)}
            coupling = np.zeros((internal.size, internal.size), dtype=np.complex128)
            for index in local:
                coupling[local[index], local[partner[index]]] = 1.0

            interior = coupling @ sub(internal, internal)
            drive = coupling @ sub(internal, external)
            identity = np.eye(internal.size, dtype=np.complex128)
            circulating = np.linalg.solve(identity - interior, drive)
            reduced = sub(external, external) + sub(external, internal) @ circulating
        else:
            reduced = sub(external, external)

        # Dangling ports were carried through the reduction as external — which
        # is the a = 0 boundary condition they need — and are dropped here.
        outward = {int(index): k for k, index in enumerate(external)}
        names = tuple(self._exposed)
        take = np.array([outward[position[self._exposed[name]]] for name in names], dtype=np.intp)
        return SMatrix(
            ports=names,
            frequencies=frequencies,
            s=reduced[np.ix_(rows, take, take)],
        )
