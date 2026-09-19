"""Vector modes of a step-index fibre: HE, EH, TE and TM, from Maxwell's equations.

:mod:`maiman.modes` solves the scalar wave equation, which is exact only when
every index step is small. The core's step is. The cladding's is not -- glass to
air is a third of the index -- and there the LP modes are a label for families
of true modes that the scalar equation cannot tell apart. This module solves for
those modes themselves.

**The model.** Concentric layers of constant index, the innermost a core and the
outermost unbounded. In each layer the longitudinal fields are Bessel functions,

    Ez = F(r) cos(nu phi),    Z0 Hz = G(r) sin(nu phi),

``J`` and ``Y`` where the field oscillates and ``K`` outside, where it decays.
The transverse fields follow from ``Ez`` and ``Hz``, and at every interface the
four tangential components -- ``Ez``, ``Hz``, ``E_phi``, ``H_phi`` -- are
continuous. That is a four-by-four transfer from layer to layer; a mode is where
the two solutions regular on the axis can be matched onto the two that decay
outside, which is a determinant that vanishes. It has no poles, so its sign
changes are its roots.

**What it gives.** For ``nu = 0`` the problem splits into TE (``Ez = 0``) and TM
(``Hz = 0``). For ``nu >= 1`` the modes are hybrid and come in two families that
alternate in effective index, HE and EH. Under weak guidance ``HE_{l+1,m}`` and
``EH_{l-1,m}`` both tend to ``LP_lm``: HE1m to LP0m, EH1m to LP2m. That pairing
is what an untilted long-period grating sees. The core's HE11 couples to cladding
modes of ``nu = 1``, and those are the HE1m the scalar model has and the EH1m it
does not.

**How it is checked.** Two layers against the closed characteristic equation of
Snyder and Love, core and glass-air alike. Three layers with a vanishing core
step against two. The weak-guidance limit against :mod:`maiman.modes`. And the
fields against the property a wrong set cannot fake, orthogonality of the power
flow between distinct modes.

Units: lengths in metres; fields dimensionless, with ``H`` carried as ``Z0 H``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from .modes import (
    ORDER_FLOOR,
    StepIndexFibre,
    _legendre,
    _roots,
    bessel_j,
    bessel_k_scaled,
    bessel_y,
)

__all__ = [
    "VectorMode",
    "vector_cladding_modes",
    "vector_core_modes",
    "vector_coupling",
    "vector_modes",
]


def _j_prime(order: int, x: np.ndarray) -> np.ndarray:
    return bessel_j(order - 1, x) - (order / x) * bessel_j(order, x)


def _y_prime(order: int, x: np.ndarray) -> np.ndarray:
    return bessel_y(order - 1, x) - (order / x) * bessel_y(order, x)


def _k_prime_scaled(order: int, x: np.ndarray) -> np.ndarray:
    return -bessel_k_scaled(order - 1, x) - (order / x) * bessel_k_scaled(order, x)


# --------------------------------------------------------------------------
# Layer matrices
# --------------------------------------------------------------------------
#
# Everything is in the dimensionless radius rho = k0 r. In a layer of index n,
# with kappa^2 = n^2 - neff^2, the state carried across an interface is
#
#     (F, G, X3, X4),  X3 = (neff nu F / rho + G') / kappa^2
#                      X4 = (neff nu G / rho + n^2 F') / kappa^2
#
# where E_phi = i X3 sin(nu phi) and Z0 H_phi = -i X4 cos(nu phi): the four
# tangential components, each continuous, with the common factors of i removed.


def _oscillating(nu: int, n: float, neff: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """``(..., 4, 4)``: the state from ``(A, B, C, D)``, ``F = AJ + BY`` and ``G = CJ + DY``."""
    kappa2 = n**2 - neff**2
    kappa = np.sqrt(kappa2)
    x = kappa * rho
    j, y = bessel_j(nu, x), bessel_y(nu, x)
    jp, yp = _j_prime(nu, x), _y_prime(nu, x)
    zero = np.zeros_like(x)
    twist = neff * nu / (rho * kappa2)
    matrix = np.empty((*x.shape, 4, 4))
    matrix[..., 0, :] = np.stack([j, y, zero, zero], axis=-1)
    matrix[..., 1, :] = np.stack([zero, zero, j, y], axis=-1)
    matrix[..., 2, :] = np.stack([twist * j, twist * y, jp / kappa, yp / kappa], axis=-1)
    matrix[..., 3, :] = np.stack(
        [n**2 * jp / kappa, n**2 * yp / kappa, twist * j, twist * y], axis=-1
    )
    return matrix


def _decaying(nu: int, n: float, neff: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """``(..., 4, 2)``: state from ``F = A K``, ``G = C K``, with ``K`` scaled by ``exp(s rho)``."""
    s2 = neff**2 - n**2
    s = np.sqrt(s2)
    x = s * rho
    k, kp = bessel_k_scaled(nu, x), _k_prime_scaled(nu, x)
    zero = np.zeros_like(x)
    twist = -neff * nu / (rho * s2)
    matrix = np.empty((*x.shape, 4, 2))
    matrix[..., 0, :] = np.stack([k, zero], axis=-1)
    matrix[..., 1, :] = np.stack([zero, k], axis=-1)
    matrix[..., 2, :] = np.stack([twist * k, -kp / s], axis=-1)
    matrix[..., 3, :] = np.stack([-(n**2) * kp / s, twist * k], axis=-1)
    return matrix


@dataclass(frozen=True)
class _Layers:
    radii: tuple[float, ...]
    indices: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.indices) != len(self.radii) + 1:
            raise ValueError("need one more index than interface radii")
        if not all(a < b for a, b in pairwise((0.0, *self.radii))):
            raise ValueError(f"interface radii must be positive and increasing, got {self.radii}")
        if min(self.indices[:-1]) <= self.indices[-1]:
            raise ValueError("every inner layer must have a higher index than the outside")


def _system(
    layers: _Layers, nu: int, neff: np.ndarray, k0: float
) -> tuple[np.ndarray, list[np.ndarray]]:
    """The matching matrix at the last interface, and the coefficients carried through each layer.

    Columns 0 and 1 are the two solutions regular on the axis (``Ez`` and ``Hz``
    in the core), carried out to the last interface; columns 2 and 3 are the
    decaying ones outside, negated. A mode is a null vector.
    """
    ones = np.ones_like(neff)
    first = _oscillating(nu, layers.indices[0], neff, k0 * layers.radii[0] * ones)
    carried = [np.eye(4)[:, [0, 2]] * ones[..., None, None]]
    state = first[..., :, [0, 2]]
    for index, (inner, outer) in enumerate(pairwise(layers.radii), start=1):
        n = layers.indices[index]
        coefficients = np.linalg.solve(_oscillating(nu, n, neff, k0 * inner * ones), state)
        carried.append(coefficients)
        state = _oscillating(nu, n, neff, k0 * outer * ones) @ coefficients
    outside = _decaying(nu, layers.indices[-1], neff, k0 * layers.radii[-1] * ones)
    return np.concatenate([state, -outside], axis=-1), carried


def _rows_and_columns(nu: int, family: str | None) -> tuple[list[int], list[int]]:
    if nu == 0 and family == "TM":
        return [0, 3], [0, 2]
    if nu == 0 and family == "TE":
        return [1, 2], [1, 3]
    return [0, 1, 2, 3], [0, 1, 2, 3]


def _determinant(
    layers: _Layers, nu: int, neff: np.ndarray, k0: float, family: str | None
) -> np.ndarray:
    matrix, _ = _system(layers, nu, neff, k0)
    rows, columns = _rows_and_columns(nu, family)
    return np.asarray(np.linalg.det(matrix[..., rows, :][..., :, columns]))


# --------------------------------------------------------------------------
# Modes
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class VectorMode:
    """One vector mode at one wavelength.

    ``family`` is HE, EH, TE or TM; ``rank`` counts within the family from 1,
    highest effective index first. ``kind`` says which boundary guides it.
    """

    radii: tuple[float, ...]
    indices: tuple[float, ...]
    wavelength: float
    order: int
    family: str
    rank: int
    effective_index: float
    kind: str
    #: Coefficients ``(A, B, C, D)`` in each inner layer, and ``(A, C)`` outside.
    coefficients: tuple[tuple[float, ...], ...]

    @property
    def name(self) -> str:
        return f"{self.family}{self.order}{self.rank} ({self.kind})"

    @property
    def propagation_constant(self) -> float:
        return 2.0 * math.pi * self.effective_index / self.wavelength

    def longitudinal(self, radius: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
        """``F(r)`` and ``G(r)``: ``Ez = F cos(nu phi)`` and ``Z0 Hz = G sin(nu phi)``."""
        f, _, g, _ = self._radial(radius)
        return f, g

    def transverse(
        self, radius: np.ndarray | float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """``(e_r, e_phi, h_r, h_phi)`` along a radius, with the angular factors and ``i`` removed.

        ``E_r = -i e_r cos``, ``E_phi = i e_phi sin``, ``Z0 H_r = -i h_r sin`` and
        ``Z0 H_phi = -i h_phi cos`` (for ``nu = 0``, read every angular factor as
        1). Then ``E . E'* = e_r e_r' cos^2 + e_phi e_phi' sin^2`` and the axial
        power flow is ``(e_r h_phi cos^2 + e_phi h_r sin^2) / Z0``.
        """
        _, _, _, fields = self._radial(radius)
        return fields

    def power(self) -> float:
        """``int (e_r h_phi + e_phi h_r) rho d rho``, which fixes a mode's normalisation."""

        def flow(radius: np.ndarray) -> np.ndarray:
            e_r, e_phi, h_r, h_phi = self.transverse(radius)
            return e_r * h_phi + e_phi * h_r

        return self.integrate(flow)

    def integrate(
        self,
        integrand: Callable[[np.ndarray], np.ndarray],
        *,
        upto: float | None = None,
        nodes: int = 512,
    ) -> float:
        """``int f(r) rho d rho`` in ``rho = k0 r``, piecewise over the layers."""
        k0 = 2.0 * math.pi / self.wavelength
        s = k0 * math.sqrt(self.effective_index**2 - self.indices[-1] ** 2)
        edges = [0.0, *self.radii, self.radii[-1] + 40.0 / s]
        if upto is not None:
            edges = [e for e in edges if e < upto] + [upto]
        x, w = _legendre(nodes)
        total = 0.0
        for lo, hi in pairwise(edges):
            r = lo + 0.5 * (hi - lo) * (x + 1.0)
            total += float(np.sum(0.5 * (hi - lo) * w * integrand(r) * (k0 * r))) * k0
        return total

    def _radial(
        self, radius: np.ndarray | float
    ) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ]:
        r = np.atleast_1d(np.asarray(radius, dtype=float))
        k0 = 2.0 * math.pi / self.wavelength
        nu, neff = self.order, self.effective_index
        rho = np.maximum(k0 * r, 1e-9)
        f, fp, g, gp = (np.zeros_like(rho) for _ in range(4))
        kappa2 = np.empty_like(rho)
        n2 = np.empty_like(rho)
        bounds = [0.0, *self.radii, math.inf]
        for layer, (lo, hi) in enumerate(pairwise(bounds)):
            inside = (r >= lo) & (r < hi)
            if not np.any(inside):
                continue
            n = self.indices[layer]
            n2[inside] = n**2
            kappa2[inside] = n**2 - neff**2
            here = rho[inside]
            if layer < len(self.radii):
                a, b, c, d = self.coefficients[layer]
                kappa = math.sqrt(n**2 - neff**2)
                x = kappa * here
                j, y = bessel_j(nu, x), bessel_y(nu, x) if (b or d) else np.zeros_like(x)
                jp = _j_prime(nu, x)
                yp = _y_prime(nu, x) if (b or d) else np.zeros_like(x)
                f[inside], g[inside] = a * j + b * y, c * j + d * y
                fp[inside], gp[inside] = kappa * (a * jp + b * yp), kappa * (c * jp + d * yp)
            else:
                a, c = self.coefficients[layer]
                s = math.sqrt(neff**2 - n**2)
                x = s * here
                decay = np.exp(-s * (here - k0 * self.radii[-1]))
                k, kp = bessel_k_scaled(nu, x) * decay, _k_prime_scaled(nu, x) * decay
                f[inside], g[inside] = a * k, c * k
                fp[inside], gp[inside] = s * a * kp, s * c * kp
        e_r = (neff * fp + nu * g / rho) / kappa2
        e_phi = (neff * nu * f / rho + gp) / kappa2
        h_r = (neff * gp + n2 * nu * f / rho) / kappa2
        h_phi = (neff * nu * g / rho + n2 * fp) / kappa2
        return f, fp, g, (e_r, e_phi, h_r, h_phi)


def _classify(nu: int, core: np.ndarray, neff: float) -> str:
    """HE or EH, from the ratio of ``Z0 Hz`` to ``Ez`` in the core.

    With ``Ez`` on ``cos`` and ``Hz`` on ``sin``, that ratio sits near ``+neff``
    for HE modes and ``-neff`` for EH -- the two circular handednesses the
    families are made of -- and its sign stays put as the guidance strengthens.
    The tests hold the labels to mode counts: a core of V = 6 has HE11, EH11,
    HE12 at order one, which is LP01, LP21 and LP02.
    """
    del nu, neff
    a, c = core
    return "HE" if a * c > 0.0 else "EH"


def vector_modes(
    radii: Sequence[float],
    indices: Sequence[float],
    wavelength: float,
    *,
    order: int,
    window: tuple[float, float],
    kind: str = "cladding",
    samples: int | None = None,
) -> list[VectorMode]:
    """Every vector mode of azimuthal ``order`` with effective index inside ``window``.

    ``radii`` are the interfaces, innermost first; ``indices`` the layers, one
    more than the radii, the last unbounded. Every mode in ``window`` must see
    oscillating fields in every inner layer, so the window's top must sit below
    every inner index. Modes are found by sampling the determinant evenly in the
    transverse wavenumber of the layer at the window's top, in which modes are
    evenly spaced, and refining each sign change.
    """
    layers = _Layers(tuple(radii), tuple(indices))
    lo, hi = window
    if not layers.indices[-1] < lo < hi < min(layers.indices[:-1]) + 1e-15:
        raise ValueError(
            "the window must lie between the outside index and every inner layer's, got "
            f"{window} for layers {layers.indices}"
        )
    if order < 0:
        raise ValueError(f"azimuthal order must be non-negative, got {order}")
    k0 = 2.0 * math.pi / wavelength
    top = min(layers.indices[:-1])
    q_lo = k0 * math.sqrt(max(top**2 - hi**2, 0.0))
    q_hi = k0 * math.sqrt(top**2 - lo**2)
    if samples is None:
        # Two families of modes, each about pi / R apart in q.
        count = (q_hi - q_lo) * layers.radii[-1] / math.pi + 2 * order + 4
        samples = int(80 * count)
    grid = np.linspace(max(q_lo, 1e-3 / layers.radii[-1]), q_hi, samples)

    def index_of(q: np.ndarray) -> np.ndarray:
        return np.sqrt(top**2 - (q / k0) ** 2)

    families: tuple[str | None, ...] = ("TE", "TM") if order == 0 else (None,)
    found: list[tuple[float, str]] = []
    for family in families:

        def equation(q: np.ndarray, family: str | None = family) -> np.ndarray:
            return _determinant(layers, order, index_of(q), k0, family)

        roots = _roots(equation, grid, tolerance=1e-14 * q_hi)
        found.extend((float(index_of(np.array([q]))[0]), family or "") for q in roots)
    found.sort(key=lambda pair: -pair[0])

    modes: list[VectorMode] = []
    ranks: dict[str, int] = {}
    for neff, family in found:
        coefficients, core = _null_coefficients(layers, order, neff, k0, family or None)
        name = family or _classify(order, core, neff)
        ranks[name] = ranks.get(name, 0) + 1
        modes.append(
            VectorMode(
                radii=layers.radii,
                indices=layers.indices,
                wavelength=wavelength,
                order=order,
                family=name,
                rank=ranks[name],
                effective_index=neff,
                kind=kind,
                coefficients=coefficients,
            )
        )
    return modes


def _null_coefficients(
    layers: _Layers, nu: int, neff: float, k0: float, family: str | None
) -> tuple[tuple[tuple[float, ...], ...], np.ndarray]:
    matrix, carried = _system(layers, nu, np.array([neff]), k0)
    rows, columns = _rows_and_columns(nu, family)
    _, _, vh = np.linalg.svd(matrix[0][np.ix_(rows, columns)])
    null = np.zeros(4)
    null[columns] = vh[-1]
    core = null[:2]
    out = []
    for coefficients in carried:
        out.append(tuple(float(v) for v in coefficients[0] @ core))
    out.append((float(null[2]), float(null[3])))
    # Normalise to a positive, unit-sized core Ez (or Hz for TE).
    scale = core[0] if abs(core[0]) > abs(core[1]) else core[1]
    return tuple(tuple(v / scale for v in layer) for layer in out), core / scale


# --------------------------------------------------------------------------
# The fibre's modes, and the coupling a grating writes between them
# --------------------------------------------------------------------------


def vector_core_modes(
    fibre: StepIndexFibre, wavelength: float, *, order: int = 1
) -> list[VectorMode]:
    """Guided core modes of azimuthal ``order``: HE11 alone for a standard fibre at 1550 nm.

    Solved as core and unbounded cladding, as :func:`maiman.modes.core_modes` is.
    """
    fibre = fibre.at(wavelength)
    n1, n2 = fibre.core_index, fibre.cladding_index
    margin = 1e-12
    k0 = 2.0 * math.pi / wavelength
    v = fibre.v_number(wavelength)
    # HE11 alone can have a small u; above order one the floor the scalar solver
    # uses applies, for the same reason.
    top = n1 - margin
    if order >= 2:
        if v <= ORDER_FLOOR:
            return []
        top = math.sqrt(n1**2 - (ORDER_FLOOR / (k0 * fibre.core_radius)) ** 2)
    return vector_modes(
        [fibre.core_radius],
        [n1, n2],
        wavelength,
        order=order,
        window=(n2 + margin, top),
        kind="core",
        samples=max(400, int(400 * fibre.v_number(wavelength))),
    )


def vector_cladding_modes(
    fibre: StepIndexFibre, wavelength: float, *, order: int = 1, count: int = 10
) -> list[VectorMode]:
    """The first ``count`` cladding modes of azimuthal ``order``, every family, highest first."""
    if count < 1:
        raise ValueError(f"ask for at least one cladding mode, got {count}")
    fibre = fibre.at(wavelength)
    k0 = 2.0 * math.pi / wavelength
    b = fibre.cladding_radius
    n2, n3 = fibre.cladding_index, fibre.surrounding_index
    q_limit = k0 * math.sqrt(n2**2 - n3**2)
    q_top = min(q_limit * (1.0 - 1e-9), (count + 3 + order / 2) * math.pi / b)
    lowest = math.sqrt(n2**2 - (q_top / k0) ** 2)
    highest = math.sqrt(n2**2 - (max(1e-3, 0.5 * order) / b / k0) ** 2)
    modes = vector_modes(
        [fibre.core_radius, b],
        [fibre.core_index, n2, n3],
        wavelength,
        order=order,
        window=(lowest, highest),
        samples=int(80 * (count + 3 + order)),
    )
    return modes[:count]


def vector_coupling(first: VectorMode, second: VectorMode, index_modulation: float) -> float:
    """Coupling coefficient under a core index modulation ``dn cos(K z)`` [1/m].

    ``kappa = (omega eps0 / 4) int dEps E1 . E2* dA`` for fields carrying a watt,
    with ``dEps = n1 dn eps0`` the amplitude of the one exponential of the cosine
    that phase matches. In this module's units that is

        ``kappa = k0 n1 dn int_core (e_r e_r' + e_phi e_phi') / (2 sqrt(P P'))``,

    which is :func:`maiman.modes.lpg_coupling` under weak guidance, and ``pi dn
    / lambda`` scaled by confinement for a mode with itself. The longitudinal
    field's share is left out, as it is for every weakly modulated grating: it
    enters as ``Ez Ez'`` and the core's ``Ez`` is a hundredth of its transverse
    field. Modes of different order do not couple under a modulation that is the
    same all the way round.
    """
    if first.wavelength != second.wavelength:
        raise ValueError("a coupling is between two modes at one wavelength")
    if first.order != second.order:
        return 0.0
    core = min(first.radii[0], second.radii[0])
    n1 = first.indices[0]

    def overlap(r: np.ndarray) -> np.ndarray:
        e1, p1, _, _ = first.transverse(r)
        e2, p2, _, _ = second.transverse(r)
        return e1 * e2 + p1 * p2

    shared = first.integrate(overlap, upto=core)
    k0 = 2.0 * math.pi / first.wavelength
    return k0 * n1 * index_modulation * shared / (2.0 * math.sqrt(first.power() * second.power()))
