"""Rigorous coupled-wave analysis of a lamellar grating on a layered stack, TE polarization.

What a grating coupler sends back up at the fibre is the grating's own zeroth-order
reflection, and its phase is what decides the ripple of the gap between fibre and
chip. This solves it: a one-dimensional periodic structure of layers, each either
uniform or a lamellar grating of two indices, under a semi-infinite top medium and
above a semi-infinite substrate, illuminated by a plane wave in the plane
perpendicular to the grating's lines.

    M. G. Moharam, E. B. Grann, D. A. Pommet and T. K. Gaylord, "Formulation for
    stable and efficient implementation of the rigorous coupled-wave analysis of
    binary gratings", J. Opt. Soc. Am. A 12, 1068 (1995).
    L. Li, "Formulation and comparison of two recursive matrix algorithms for
    modeling layered diffraction gratings", J. Opt. Soc. Am. A 13, 1024 (1996).

**Conventions.** Fields carry ``exp(-i omega t)``, so ``exp(+i k z)`` travels down
into the stack and an absorbing medium has a positive imaginary index, as
:mod:`maiman.photonics` has it; ``z`` runs down from the top interface, where the
phase of every reflection is referenced. TE means the electric field lies along the
grating's lines, which is what a grating coupler's waveguide carries.

**How.** In each layer the field's harmonics obey ``u'' = k0^2 (Kx^2 - E) u``, with
``E`` the Toeplitz matrix of the layer's permittivity harmonics -- the Laurent rule,
which is the right factorization for TE. Its eigenmodes are ``exp(-+ k0 q z)`` with
``q^2`` the eigenvalues, and matching the field and its derivative at each
interface gives an interface scattering matrix. The reflection of the whole stack
is built up from the bottom by ``R -> S11 + S12 R (I - S22 R)^-1 S21`` with the
layers' decaying exponentials between, so no growing exponential ever appears --
that is what makes a 2 micron oxide under the teeth safe. Total power is conserved
to rounding for lossless media, which the tests hold it to.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

__all__ = ["Diffraction", "GratingLayer", "UniformLayer", "diffract_te"]


@dataclass(frozen=True)
class UniformLayer:
    """A slab of one index, ``thickness`` [m] thick."""

    thickness: float
    index: complex


@dataclass(frozen=True)
class GratingLayer:
    """A lamellar layer: ridges of ``ridge_index`` over a fraction ``duty`` of each period.

    The rest of the period is ``groove_index``. The ridge is centred on ``x = 0``,
    which does not matter for a single grating layer's power and only sets the
    phase of the higher orders.
    """

    thickness: float
    ridge_index: complex
    groove_index: complex
    duty: float = 0.5


@dataclass(frozen=True)
class Diffraction:
    """What a plane wave does to the stack.

    ``reflection`` is the zeroth order's amplitude, referenced to the top
    interface, and ``orders`` the integers it was computed for. ``reflectance`` and
    ``transmittance`` are each order's share of the incident power, zero for an
    evanescent one. ``transmittance`` is what crosses into the substrate, so under
    an absorbing substrate it is what that substrate takes; ``absorbed`` is what a
    lossy *layer* kept.
    """

    reflection: complex
    orders: np.ndarray
    reflectance: np.ndarray
    transmittance: np.ndarray

    @property
    def absorbed(self) -> float:
        """What the layers kept: one less every propagating order's share."""
        return float(1.0 - self.reflectance.sum() - self.transmittance.sum())


def _harmonics(layer: UniformLayer | GratingLayer, count: int) -> np.ndarray:
    """The permittivity's Fourier harmonics ``eps_h`` for ``h = -count .. count``."""
    h = np.arange(-count, count + 1)
    if isinstance(layer, UniformLayer):
        out = np.zeros(h.size, dtype=np.complex128)
        out[count] = layer.index**2
        return out
    if not 0.0 <= layer.duty <= 1.0:
        raise ValueError(f"the duty cycle lies between 0 and 1, got {layer.duty}")
    ridge, groove = layer.ridge_index**2, layer.groove_index**2
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(h == 0, layer.duty, np.sin(np.pi * h * layer.duty) / (np.pi * h))
    out = (ridge - groove) * term.astype(np.complex128)
    out[count] += groove
    return out


def _forward(q: np.ndarray) -> np.ndarray:
    """The root of each ``q^2`` that decays, or travels, downward (``exp(-k0 q z)``)."""
    q = np.sqrt(q.astype(np.complex128))
    flip = (q.real < -1e-14 * np.abs(q)) | ((np.abs(q.real) <= 1e-14 * np.abs(q)) & (q.imag > 0))
    return np.asarray(np.where(flip, -q, q))


def diffract_te(
    wavelength: float,
    *,
    sine: float,
    period: float,
    top_index: float,
    layers: Sequence[UniformLayer | GratingLayer],
    substrate_index: complex,
    harmonics: int = 15,
) -> Diffraction:
    """Diffraction of a TE plane wave by a layered stack with lamellar grating layers.

    ``sine`` is the sine of the incidence angle in the top medium, ``harmonics`` the
    number ``M`` of diffraction orders kept either side of the zeroth (``2 M + 1`` in
    all): the answer converges as it grows, and the tests show how fast. The top
    medium must be lossless.
    """
    if wavelength <= 0.0 or period <= 0.0:
        raise ValueError("the wavelength and the period must both be positive")
    if not -1.0 < sine < 1.0:
        raise ValueError(f"the sine of the incidence angle lies inside (-1, 1), got {sine}")
    if harmonics < 0:
        raise ValueError("the number of harmonics is not negative")
    if abs(complex(top_index).imag) > 0.0:
        raise ValueError("the top medium must be lossless: the incident wave travels in it")
    count = harmonics
    orders = np.arange(-count, count + 1)
    size = orders.size
    k0 = 2.0 * math.pi / wavelength
    kx = top_index * sine + orders * (wavelength / period)  # kx / k0
    identity = np.eye(size, dtype=np.complex128)
    kx2 = np.diag(kx**2).astype(np.complex128)

    def uniform_modes(index: complex) -> tuple[np.ndarray, np.ndarray]:
        return identity, _forward(kx**2 - index**2)

    def toeplitz(layer: UniformLayer | GratingLayer) -> np.ndarray:
        eps = _harmonics(layer, 2 * count)
        idx = orders[:, None] - orders[None, :]
        return eps[idx + 2 * count]

    def modes(layer: UniformLayer | GratingLayer) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(layer, UniformLayer):
            return uniform_modes(layer.index)
        values, vectors = np.linalg.eig(kx2 - toeplitz(layer))
        return vectors, _forward(values)

    def interface(
        above: tuple[np.ndarray, np.ndarray], below: tuple[np.ndarray, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        wa, qa = above
        wb, qb = below
        left = np.block([[wa, -wb], [wa * qa, wb * qb]])
        right = np.block([[-wa, wb], [wa * qa, wb * qb]])
        s = np.linalg.solve(left, right)
        return s[:size, :size], s[:size, size:], s[size:, :size], s[size:, size:]

    stack = [uniform_modes(top_index)]
    stack += [modes(layer) for layer in layers]
    stack.append(uniform_modes(substrate_index))
    thickness = [0.0] + [layer.thickness for layer in layers] + [0.0]
    joins = [interface(stack[i], stack[i + 1]) for i in range(len(stack) - 1)]

    # From the bottom: the substrate reflects nothing, so R at its top plane is zero.
    reflected = np.zeros((size, size), dtype=np.complex128)
    passes: list[np.ndarray] = []  # F_below = passes[i] F_above at interface i
    for i in range(len(joins) - 1, -1, -1):
        s11, s12, s21, s22 = joins[i]
        inverse = np.linalg.inv(identity - s22 @ reflected)
        passes.append(inverse @ s21)
        bottom = s11 + s12 @ reflected @ inverse @ s21
        if i > 0:
            decay = np.diag(np.exp(-k0 * stack[i][1] * thickness[i]))
            reflected = decay @ bottom @ decay
        else:
            reflected = bottom
    passes.reverse()
    zero = int(np.flatnonzero(orders == 0)[0])
    incident = np.zeros(size, dtype=np.complex128)
    incident[zero] = 1.0
    back = reflected @ incident

    # The transmitted amplitudes: pass down through every interface and layer.
    forward = incident
    for i in range(len(joins)):
        forward = passes[i] @ forward
        if i + 1 < len(joins):
            forward = np.exp(-k0 * stack[i + 1][1] * thickness[i + 1]) * forward
    wave_top = np.sqrt(top_index**2 - (kx * 1.0) ** 2 + 0j)
    wave_sub = np.sqrt(substrate_index**2 - kx**2 + 0j)
    kz0 = float(wave_top[zero].real)
    reflectance = np.where(wave_top.real > 1e-12, np.abs(back) ** 2 * wave_top.real / kz0, 0.0)
    transmittance = np.where(wave_sub.real > 1e-12, np.abs(forward) ** 2 * wave_sub.real / kz0, 0.0)
    return Diffraction(
        reflection=complex(back[zero]),
        orders=orders,
        reflectance=np.asarray(reflectance, dtype=float),
        transmittance=np.asarray(transmittance, dtype=float),
    )
