"""Guided and cladding modes of a step-index fibre, from the scalar wave equation.

A Bragg grating couples a core mode to itself travelling the other way, and all
it needs to know about the fibre is one effective index. A long-period grating
couples the core mode to modes of the *cladding* -- light guided by the
glass-air boundary sixty microns out -- and where it does so is set by the
difference between two effective indices, each of which moves with wavelength,
with the cladding's size, and with whatever surrounds the fibre. That is a mode
solver's job, and this is one.

**The model.** Linearly polarised (LP) modes of a circular step-index fibre: a
core of radius ``a`` and index ``n1``, a cladding out to ``b`` at ``n2``, and a
surrounding medium at ``n3``. The field is ``psi(r) cos(l phi)``, Bessel
functions in each layer, continuous with its derivative at both interfaces.

- **Core modes** have ``n2 < n_eff < n1`` and are found as two layers, core and
  infinite cladding. At 1550 nm the core mode has decayed by forty orders of
  magnitude before it reaches the cladding's edge, so the outer boundary is not
  an approximation anyone can measure.
- **Cladding modes** have ``n3 < n_eff < n2`` and need all three layers: the
  field oscillates through the core and the cladding and decays outside.

**What scalar means here.** The LP approximation treats the fields as if the
index steps were small. For the core that is the textbook case, good to parts in
a hundred thousand in effective index. The cladding-air step is not small, and
the vector modes the LP modes stand in for are split by it at around 1e-5 to
1e-4 in effective index -- which moves a long-period resonance by up to a few
nanometres. The tests check the solver against an independent finite-difference
solution of the *same* scalar equation, which pins the numerics, not the
approximation; the approximation is stated here instead.

**No SciPy.** The Bessel functions are evaluated from their integral
representations by quadrature, which converges exponentially for the smooth,
bounded integrands these are, and the values are checked against tabulated ones.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

# --------------------------------------------------------------------------
# Bessel functions of integer order, for real positive arguments
# --------------------------------------------------------------------------

#: Largest number of quadrature nodes times arguments evaluated in one block,
#: which caps memory for a wide argument array without changing any value.
_BLOCK = 2_000_000

_LEGENDRE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _legendre(count: int) -> tuple[np.ndarray, np.ndarray]:
    nodes = _LEGENDRE.get(count)
    if nodes is None:
        nodes = np.polynomial.legendre.leggauss(count)
        _LEGENDRE[count] = nodes
    return nodes


def _positive(x: np.ndarray | float) -> np.ndarray:
    values = np.atleast_1d(np.asarray(x, dtype=float))
    if values.size and not np.all(values > 0.0):
        raise ValueError("Bessel functions here take positive real arguments")
    return values


def _blocks(values: np.ndarray, count: int) -> list[slice]:
    size = max(1, _BLOCK // max(count, 1))
    return [slice(start, start + size) for start in range(0, values.size, size)]


def bessel_j(order: int, x: np.ndarray | float) -> np.ndarray:
    """``J_n(x)``: ``(1/2 pi)`` times the integral of ``cos(n t - x sin t)`` over a period.

    The integrand is periodic and entire, so the trapezoidal rule converges
    exponentially once it has more nodes than ``x + n``.
    """
    values = _positive(x)
    n = abs(order)
    top = float(values.max()) if values.size else 0.0
    count = int(min(16384, max(64, 2 * math.ceil((top + n) / 2) + 64)))
    angle = 2.0 * np.pi * np.arange(count) / count
    out = np.empty_like(values)
    for block in _blocks(values, count):
        chunk = values[block]
        out[block] = np.cos(n * angle[:, None] - chunk[None, :] * np.sin(angle)[:, None]).mean(
            axis=0
        )
    if order < 0 and n % 2:
        out = -out
    return out


def _tail_limit(
    values: np.ndarray, n: int, grows: Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:
    """Where an exponentially decaying tail integrand has fallen below ``exp(-50)``."""
    t = np.ones_like(values)
    for _ in range(8):
        t = grows((50.0 + n * t) / values)
    return np.maximum(t, 1.0)


def bessel_y(order: int, x: np.ndarray | float) -> np.ndarray:
    """``Y_n(x)``, from DLMF 10.9.7 for integer order.

    ``Y_n(x) = (1/pi) int_0^pi sin(x sin t - n t) dt
               - (1/pi) int_0^inf (e^{nt} + (-1)^n e^{-nt}) e^{-x sinh t} dt``
    """
    values = _positive(x)
    n = abs(order)
    top = float(values.max()) if values.size else 0.0
    count = int(min(16384, max(96, math.ceil(1.5 * (top + n)) + 96)))
    nodes, weights = _legendre(count)
    theta = 0.5 * np.pi * (nodes + 1.0)
    w_theta = 0.5 * np.pi * weights

    tail_nodes, tail_weights = _legendre(256)
    limit = _tail_limit(values, n, np.arcsinh)
    sign = -1.0 if n % 2 else 1.0

    out = np.empty_like(values)
    for block in _blocks(values, count):
        chunk = values[block]
        oscillating = (
            w_theta[:, None] * np.sin(chunk[None, :] * np.sin(theta)[:, None] - n * theta[:, None])
        ).sum(axis=0)
        span = limit[block]
        t = 0.5 * span[None, :] * (tail_nodes[:, None] + 1.0)
        decay = chunk[None, :] * np.sinh(t)
        tail = (
            0.5
            * span[None, :]
            * tail_weights[:, None]
            * (np.exp(n * t - decay) + sign * np.exp(-n * t - decay))
        ).sum(axis=0)
        out[block] = (oscillating - tail) / np.pi
    if order < 0 and n % 2:
        out = -out
    return out


def bessel_k_scaled(order: int, x: np.ndarray | float) -> np.ndarray:
    """``exp(x) K_n(x)``, from DLMF 10.32.9: ``K_n(x) = int_0^inf exp(-x cosh t) cosh(n t) dt``.

    Scaled because the arguments outside a fibre's cladding run to hundreds, where
    ``K_n`` itself underflows; only ratios of it enter a dispersion relation.
    ``K_{-n} = K_n``.
    """
    values = _positive(x)
    n = abs(order)
    nodes, weights = _legendre(256)
    limit = _tail_limit(values, n, lambda v: np.arccosh(1.0 + v))
    t = 0.5 * limit[None, :] * (nodes[:, None] + 1.0)
    decay = values[None, :] * (np.cosh(t) - 1.0)
    integrand = 0.5 * (np.exp(n * t - decay) + np.exp(-n * t - decay))
    return (0.5 * limit[None, :] * weights[:, None] * integrand).sum(axis=0)


def _j_prime(order: int, x: np.ndarray) -> np.ndarray:
    return bessel_j(order - 1, x) - (order / x) * bessel_j(order, x)


def _y_prime(order: int, x: np.ndarray) -> np.ndarray:
    return bessel_y(order - 1, x) - (order / x) * bessel_y(order, x)


def _k_log_derivative(order: int, x: np.ndarray) -> np.ndarray:
    """``K_n'(x) / K_n(x)``, from ``K_n' = -K_{n-1} - (n/x) K_n``; the scaling cancels."""
    return -bessel_k_scaled(order - 1, x) / bessel_k_scaled(order, x) - order / x


# --------------------------------------------------------------------------
# The fibre and its modes
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class StepIndexFibre:
    """Three concentric layers: core, cladding, and whatever surrounds the fibre.

    Lengths in metres. ``surrounding_index`` is air by default; set it to a
    liquid's and every cladding mode moves, which is what makes a long-period
    grating a refractometer. It has to stay below the cladding's, or the
    cladding guides nothing.
    """

    core_radius: float = 4.1e-6
    cladding_radius: float = 62.5e-6
    core_index: float = 1.4492
    cladding_index: float = 1.4440
    surrounding_index: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 < self.core_radius < self.cladding_radius:
            raise ValueError(
                f"need 0 < core radius < cladding radius, got {self.core_radius} and "
                f"{self.cladding_radius} m"
            )
        if not self.core_index > self.cladding_index > self.surrounding_index > 0.0:
            raise ValueError(
                "need core index > cladding index > surrounding index > 0, got "
                f"{self.core_index}, {self.cladding_index}, {self.surrounding_index}"
            )

    def v_number(self, wavelength: float) -> float:
        """Normalised frequency of the core, ``V = (2 pi a / lambda) sqrt(n1^2 - n2^2)``."""
        return (
            2.0
            * math.pi
            * self.core_radius
            / wavelength
            * math.sqrt(self.core_index**2 - self.cladding_index**2)
        )


@dataclass(frozen=True)
class Mode:
    """One LP mode at one wavelength: ``LP_{order, rank}`` of the core or the cladding."""

    fibre: StepIndexFibre
    wavelength: float
    order: int
    rank: int
    effective_index: float
    kind: str

    @property
    def name(self) -> str:
        return f"LP{self.order}{self.rank} ({self.kind})"

    @property
    def propagation_constant(self) -> float:
        return 2.0 * math.pi * self.effective_index / self.wavelength

    def field(self, radius: np.ndarray | float) -> np.ndarray:
        """The radial field ``psi(r)``, unnormalised: 1 on axis for a core mode."""
        r = np.atleast_1d(np.asarray(radius, dtype=float))
        if self.kind == "core":
            return _core_field(self, r)
        return _cladding_field(self, r)

    def power(self) -> float:
        """``int psi^2 r dr`` over the whole cross-section, the normalisation."""
        return _radial_integral(self, lambda r: self.field(r) ** 2, *_regions(self))


def _k0(wavelength: float) -> float:
    return 2.0 * math.pi / wavelength


def _roots(
    function: Callable[[np.ndarray], np.ndarray], grid: np.ndarray, tolerance: float
) -> np.ndarray:
    """Every sign change of ``function`` on ``grid``, refined by Illinois false position."""
    values = function(grid)
    finite = np.isfinite(values)
    change = np.flatnonzero(
        finite[:-1] & finite[1:] & (np.sign(values[:-1]) * np.sign(values[1:]) < 0)
    )
    if change.size == 0:
        return np.empty(0)
    lo, hi = grid[change].copy(), grid[change + 1].copy()
    f_lo, f_hi = values[change].copy(), values[change + 1].copy()
    side = np.zeros(change.size, dtype=int)
    for _ in range(100):
        guess = hi - f_hi * (hi - lo) / (f_hi - f_lo)
        f_guess = function(guess)
        left = np.sign(f_guess) == np.sign(f_lo)
        lo = np.where(left, guess, lo)
        f_lo = np.where(left, f_guess, f_lo)
        hi = np.where(left, hi, guess)
        f_hi = np.where(left, f_hi, f_guess)
        # Illinois: halve the stale end so one side cannot stall the iteration.
        f_hi = np.where(left & (side == 1), 0.5 * f_hi, f_hi)
        f_lo = np.where(~left & (side == -1), 0.5 * f_lo, f_lo)
        side = np.where(left, 1, -1)
        if np.all(np.abs(hi - lo) <= tolerance):
            break
    return 0.5 * (lo + hi)


def core_modes(fibre: StepIndexFibre, wavelength: float, *, order: int = 0) -> list[Mode]:
    """Every guided core mode of azimuthal ``order``, highest effective index first.

    The two-layer LP eigenvalue equation, written without division so it has no
    poles to mistake for roots: ``u J_{l-1}(u) K_l(w) + w K_{l-1}(w) J_l(u) = 0``
    with ``u^2 + w^2 = V^2``.
    """
    v = fibre.v_number(wavelength)
    if order < 0:
        raise ValueError(f"azimuthal order must be non-negative, got {order}")

    def equation(u: np.ndarray) -> np.ndarray:
        w = np.sqrt(np.maximum(v**2 - u**2, 1e-300))
        return u * bessel_j(order - 1, u) * bessel_k_scaled(order, w) + w * bessel_k_scaled(
            order - 1, w
        ) * bessel_j(order, u)

    grid = np.linspace(v * 1e-6, v * (1.0 - 1e-9), max(400, int(200 * v)))
    roots = _roots(equation, grid, tolerance=1e-14 * v)
    n1, n2 = fibre.core_index, fibre.cladding_index
    modes = []
    for rank, u in enumerate(sorted(roots), start=1):
        neff = math.sqrt(n1**2 - (u / (_k0(wavelength) * fibre.core_radius)) ** 2)
        modes.append(Mode(fibre, wavelength, order, rank, neff, "core"))
    del n2
    return modes


def cladding_modes(
    fibre: StepIndexFibre, wavelength: float, *, order: int = 0, count: int = 10
) -> list[Mode]:
    """The first ``count`` cladding modes of azimuthal ``order``, highest index first.

    Three layers, matched at the core and cladding boundaries. The unknown is
    swept as the cladding's transverse wavenumber ``q``, in which the modes are
    spaced by about ``pi / b`` -- evenly, which is what makes a fixed sampling
    density find every one.
    """
    if count < 1:
        raise ValueError(f"ask for at least one cladding mode, got {count}")
    k = _k0(wavelength)
    a, b = fibre.core_radius, fibre.cladding_radius
    n1, n2, n3 = fibre.core_index, fibre.cladding_index, fibre.surrounding_index
    nu = order

    def equation(q: np.ndarray) -> np.ndarray:
        neff2 = n2**2 - (q / k) ** 2
        p = k * np.sqrt(n1**2 - neff2)
        s = k * np.sqrt(np.maximum(neff2 - n3**2, 1e-300))
        at_core, slope_core = bessel_j(nu, p * a), p * _j_prime(nu, p * a)
        x = q * a
        amp_j = 0.5 * np.pi * x * (_y_prime(nu, x) * at_core - bessel_y(nu, x) * slope_core / q)
        amp_y = 0.5 * np.pi * x * (-_j_prime(nu, x) * at_core + bessel_j(nu, x) * slope_core / q)
        at_edge = amp_j * bessel_j(nu, q * b) + amp_y * bessel_y(nu, q * b)
        slope_edge = q * (amp_j * _j_prime(nu, q * b) + amp_y * _y_prime(nu, q * b))
        return s * _k_log_derivative(nu, s * b) * at_edge - slope_edge

    q_limit = k * math.sqrt(n2**2 - n3**2)
    q_top = min(q_limit * (1.0 - 1e-9), (count + 2 + nu / 2) * math.pi / b)
    samples = int(40 * (count + 3 + nu))
    grid = np.linspace(1e-3 / b, q_top, samples)
    roots = sorted(_roots(equation, grid, tolerance=1e-14 * q_limit))
    modes = []
    for rank, q in enumerate(roots[:count], start=1):
        neff = math.sqrt(n2**2 - (q / k) ** 2)
        modes.append(Mode(fibre, wavelength, nu, rank, neff, "cladding"))
    return modes


# --------------------------------------------------------------------------
# Fields, and the overlap a grating couples through
# --------------------------------------------------------------------------


def _core_field(mode: Mode, r: np.ndarray) -> np.ndarray:
    fibre, nu = mode.fibre, mode.order
    k, a = _k0(mode.wavelength), fibre.core_radius
    u = a * k * math.sqrt(fibre.core_index**2 - mode.effective_index**2)
    w = a * k * math.sqrt(mode.effective_index**2 - fibre.cladding_index**2)
    out = np.empty_like(r)
    inside = r < a
    rho = np.maximum(r / a, 1e-12)
    out[inside] = bessel_j(nu, u * rho[inside]) / bessel_j(nu, np.array([u]))[0]
    outside = ~inside
    if np.any(outside):
        out[outside] = (
            bessel_k_scaled(nu, w * rho[outside])
            / bessel_k_scaled(nu, np.array([w]))[0]
            * np.exp(-w * (rho[outside] - 1.0))
        )
    return out


def _cladding_field(mode: Mode, r: np.ndarray) -> np.ndarray:
    fibre, nu = mode.fibre, mode.order
    k = _k0(mode.wavelength)
    a, b = fibre.core_radius, fibre.cladding_radius
    neff = mode.effective_index
    p = k * math.sqrt(fibre.core_index**2 - neff**2)
    q = k * math.sqrt(fibre.cladding_index**2 - neff**2)
    s = k * math.sqrt(neff**2 - fibre.surrounding_index**2)
    one = np.ones(1)
    at_core = bessel_j(nu, p * a * one)[0]
    slope_core = p * _j_prime(nu, p * a * one)[0]
    x = q * a * one
    amp_j = (
        0.5 * math.pi * x[0] * (_y_prime(nu, x)[0] * at_core - bessel_y(nu, x)[0] * slope_core / q)
    )
    amp_y = (
        0.5 * math.pi * x[0] * (-_j_prime(nu, x)[0] * at_core + bessel_j(nu, x)[0] * slope_core / q)
    )
    at_edge = amp_j * bessel_j(nu, q * b * one)[0] + amp_y * bessel_y(nu, q * b * one)[0]

    out = np.empty_like(r)
    inner = r < a
    middle = (r >= a) & (r < b)
    outer = r >= b
    safe = np.maximum(r, 1e-12 * a)
    if np.any(inner):
        out[inner] = bessel_j(nu, p * safe[inner])
    if np.any(middle):
        out[middle] = amp_j * bessel_j(nu, q * safe[middle]) + amp_y * bessel_y(
            nu, q * safe[middle]
        )
    if np.any(outer):
        out[outer] = (
            at_edge
            * bessel_k_scaled(nu, s * safe[outer])
            / bessel_k_scaled(nu, s * b * one)[0]
            * np.exp(-s * (safe[outer] - b))
        )
    return out


def _regions(mode: Mode) -> tuple[float, ...]:
    """Radii bounding the pieces a radial integral is split at, out to a negligible tail."""
    fibre = mode.fibre
    k = _k0(mode.wavelength)
    if mode.kind == "core":
        w = k * math.sqrt(mode.effective_index**2 - fibre.cladding_index**2)
        return (0.0, fibre.core_radius, fibre.core_radius + 40.0 / w)
    s = k * math.sqrt(mode.effective_index**2 - fibre.surrounding_index**2)
    return (0.0, fibre.core_radius, fibre.cladding_radius, fibre.cladding_radius + 40.0 / s)


def _radial_integral(
    mode: Mode, integrand: Callable[[np.ndarray], np.ndarray], *edges: float, nodes: int = 512
) -> float:
    """``int f(r) r dr`` by Gauss-Legendre on each piece between ``edges``."""
    x, w = _legendre(nodes)
    total = 0.0
    for lo, hi in pairwise(edges):
        r = lo + 0.5 * (hi - lo) * (x + 1.0)
        total += float(np.sum(0.5 * (hi - lo) * w * integrand(r) * r))
    return total


def core_overlap(first: Mode, second: Mode) -> float:
    """``int_core psi_1 psi_2 r dr``, over the two modes' own normalisations.

    The quantity a grating written in the core couples two modes through: it is
    the fraction of each mode's field that sits where the index is modulated.
    Two modes of different azimuthal order have zero overlap under a modulation
    that is the same all the way round, which is why an untilted grating couples
    the core's LP01 to cladding LP0m modes and to nothing else.
    """
    if first.fibre != second.fibre or first.wavelength != second.wavelength:
        raise ValueError("an overlap is between two modes of one fibre at one wavelength")
    if first.order != second.order:
        return 0.0
    a = first.fibre.core_radius
    shared = _radial_integral(first, lambda r: first.field(r) * second.field(r), 0.0, a)
    return shared / math.sqrt(first.power() * second.power())


def lpg_coupling(core: Mode, cladding: Mode, index_modulation: float) -> float:
    """Coupling coefficient between two modes under a core index modulation [1/m].

    ``kappa = k^2 n1 dn O / (2 sqrt(beta_1 beta_2))``, which is ``pi dn / lambda``
    -- this library's Bragg coupling -- when the two modes are one and the same
    core mode. ``index_modulation`` means what it means for the Bragg grating.
    """
    k = _k0(core.wavelength)
    overlap = core_overlap(core, cladding)
    return (
        k**2
        * core.fibre.core_index
        * index_modulation
        * overlap
        / (2.0 * math.sqrt(core.propagation_constant * cladding.propagation_constant))
    )
