"""Rigorous coupled-wave analysis of a lamellar grating on a layered stack, TE and TM.

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
grating's lines, which is what a grating coupler's waveguide usually carries; TM the
magnetic field, solved with the inverse rule for the electric field that crosses the
grating's walls (see :func:`diffract`).

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

__all__ = [
    "Diffraction",
    "GratingLayer",
    "PatternedLayer",
    "UniformLayer",
    "diffract",
    "diffract_te",
    "diffract_tm",
]


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
class PatternedLayer:
    """A layer whose index steps through ``segments`` across one period.

    ``segments`` is a tuple of ``(width, index)`` from ``x = offset`` on, the widths
    adding to the period, the indices complex if a segment absorbs. It is what a
    lamellar :class:`GratingLayer` cannot say: a supercell -- a run of teeth and a
    stretch of plain waveguide in one period, which is a *finite* grating repeated
    far enough apart that the copies do not see each other.
    """

    thickness: float
    segments: tuple[tuple[float, complex], ...]
    offset: float = 0.0


@dataclass(frozen=True)
class Diffraction:
    """What a plane wave, or a beam of them, does to the stack.

    ``reflection`` is the amplitude the reflected field returns into the incident
    one -- ``<incident, reflected> / <incident, incident>``, which for a plane wave
    is the zeroth order's amplitude -- referenced to the top interface, and
    ``orders`` the integers it was computed for. ``reflectance`` and
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


def _harmonics(
    layer: UniformLayer | GratingLayer | PatternedLayer, count: int, period: float
) -> np.ndarray:
    """The permittivity's Fourier harmonics ``eps_h`` for ``h = -count .. count``."""
    h = np.arange(-count, count + 1)
    if isinstance(layer, UniformLayer):
        out = np.zeros(h.size, dtype=np.complex128)
        out[count] = layer.index**2
        return out
    if isinstance(layer, PatternedLayer):
        widths = np.array([width for width, _ in layer.segments], dtype=float)
        if np.any(widths < 0.0) or abs(widths.sum() - period) > 1e-9 * period:
            raise ValueError(
                f"a patterned layer's segments add to {widths.sum()} m, not the period {period} m"
            )
        edges = layer.offset + np.concatenate([[0.0], np.cumsum(widths)])
        out = np.zeros(h.size, dtype=np.complex128)
        nonzero = h != 0
        for (width, index), start, end in zip(layer.segments, edges[:-1], edges[1:], strict=True):
            eps = complex(index) ** 2
            out[count] += eps * width / period
            out[nonzero] += (
                eps
                * (
                    np.exp(-2j * np.pi * h[nonzero] * start / period)
                    - np.exp(-2j * np.pi * h[nonzero] * end / period)
                )
                / (2j * np.pi * h[nonzero])
            )
        return out
    if not 0.0 <= layer.duty <= 1.0:
        raise ValueError(f"the duty cycle lies between 0 and 1, got {layer.duty}")
    ridge, groove = layer.ridge_index**2, layer.groove_index**2
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(h == 0, layer.duty, np.sin(np.pi * h * layer.duty) / (np.pi * h))
    out = (ridge - groove) * term.astype(np.complex128)
    out[count] += groove
    return out


def _inverse_harmonics(
    layer: UniformLayer | GratingLayer | PatternedLayer, count: int, period: float
) -> np.ndarray:
    """The Fourier harmonics of ``1 / eps``, for the inverse rule TM needs.

    :func:`_harmonics` squares each index, so the same layer with every index
    replaced by its reciprocal gives ``1 / n^2`` -- for an absorbing one too.
    """
    if isinstance(layer, UniformLayer):
        return _harmonics(UniformLayer(layer.thickness, 1.0 / complex(layer.index)), count, period)
    if isinstance(layer, PatternedLayer):
        flipped = tuple((width, 1.0 / complex(index)) for width, index in layer.segments)
        return _harmonics(PatternedLayer(layer.thickness, flipped, layer.offset), count, period)
    return _harmonics(
        GratingLayer(
            layer.thickness,
            1.0 / complex(layer.ridge_index),
            1.0 / complex(layer.groove_index),
            layer.duty,
        ),
        count,
        period,
    )


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
    layers: Sequence[UniformLayer | GratingLayer | PatternedLayer],
    substrate_index: complex,
    harmonics: int = 15,
    incident: np.ndarray | None = None,
) -> Diffraction:
    """TE: :func:`diffract` with the electric field along the grating's lines."""
    return diffract(
        wavelength,
        sine=sine,
        period=period,
        top_index=top_index,
        layers=layers,
        substrate_index=substrate_index,
        harmonics=harmonics,
        incident=incident,
        polarization="te",
    )


def diffract_tm(
    wavelength: float,
    *,
    sine: float,
    period: float,
    top_index: float,
    layers: Sequence[UniformLayer | GratingLayer | PatternedLayer],
    substrate_index: complex,
    harmonics: int = 15,
    incident: np.ndarray | None = None,
) -> Diffraction:
    """TM: :func:`diffract` with the magnetic field along the grating's lines."""
    return diffract(
        wavelength,
        sine=sine,
        period=period,
        top_index=top_index,
        layers=layers,
        substrate_index=substrate_index,
        harmonics=harmonics,
        incident=incident,
        polarization="tm",
    )


def diffract(
    wavelength: float,
    *,
    sine: float,
    period: float,
    top_index: float,
    layers: Sequence[UniformLayer | GratingLayer | PatternedLayer],
    substrate_index: complex,
    harmonics: int = 15,
    incident: np.ndarray | None = None,
    polarization: str = "te",
) -> Diffraction:
    """Diffraction of a plane wave by a layered stack with lamellar grating layers.

    ``polarization`` is ``"te"``, the electric field along the grating's lines, or
    ``"tm"``, the magnetic field along them. In TM the amplitudes are those of
    ``H_y``, so ``reflection`` is the magnetic field's -- for one interface
    ``(eps_2 k_z1 - eps_1 k_z2) / (eps_2 k_z1 + eps_1 k_z2)`` -- and the electric
    field normal to the grating's walls, ``E_x``, has its product with the
    permittivity taken by the inverse rule (Lalanne and Morris, J. Opt. Soc. Am. A
    13, 779 (1996); Li, J. Opt. Soc. Am. A 13, 1870 (1996)): with ``A`` the Toeplitz
    matrix of ``1 / eps`` and ``E`` that of ``eps``, ``H'' = k0^2 A^-1 (Kx E^-1 Kx - I) H``,
    and what is matched at each interface is ``H_y`` and ``A H_y'``. The Laurent rule
    there converges like a staircase and is why TM had a name for being slow.

    ``sine`` is the sine of the incidence angle in the top medium, ``harmonics`` the
    number ``M`` of diffraction orders kept either side of the zeroth (``2 M + 1`` in
    all): the answer converges as it grows, and the tests show how fast. The top
    medium must be lossless.

    ``incident`` is the amplitude in each order, ``2 M + 1`` of them, for a beam
    rather than a plane wave: a beam on a period long enough that its copies do not
    overlap is a sum of the orders, and the stack is linear. Left ``None`` it is the
    zeroth order alone. Power is reckoned against the beam's own, so reflectance,
    transmittance and what a lossy layer kept still add to one.
    """
    if wavelength <= 0.0 or period <= 0.0:
        raise ValueError("the wavelength and the period must both be positive")
    if not -1.0 < sine < 1.0:
        raise ValueError(f"the sine of the incidence angle lies inside (-1, 1), got {sine}")
    if harmonics < 0:
        raise ValueError("the number of harmonics is not negative")
    if abs(complex(top_index).imag) > 0.0:
        raise ValueError("the top medium must be lossless: the incident wave travels in it")
    if polarization not in ("te", "tm"):
        raise ValueError(f"polarization is 'te' or 'tm', got {polarization!r}")
    tm = polarization == "tm"
    count = harmonics
    orders = np.arange(-count, count + 1)
    size = orders.size
    k0 = 2.0 * math.pi / wavelength
    kx = top_index * sine + orders * (wavelength / period)  # kx / k0
    identity = np.eye(size, dtype=np.complex128)
    Modes = tuple[np.ndarray, np.ndarray, np.ndarray]
    kx2 = np.diag(kx**2).astype(np.complex128)

    def uniform_modes(index: complex) -> Modes:
        # The field, its decay rates, and what is matched beside the field: ``W q``
        # for TE, ``W q / eps`` for TM.
        q = _forward(kx**2 - index**2)
        return identity, q, identity * (q / complex(index) ** 2 if tm else q)

    def toeplitz(eps: np.ndarray) -> np.ndarray:
        idx = orders[:, None] - orders[None, :]
        return np.asarray(eps[idx + 2 * count])

    def modes(layer: UniformLayer | GratingLayer | PatternedLayer) -> Modes:
        if isinstance(layer, UniformLayer):
            return uniform_modes(layer.index)
        eps = _harmonics(layer, 2 * count, period)
        if not tm:
            values, vectors = np.linalg.eig(kx2 - toeplitz(eps))
            q = _forward(values)
            return vectors, q, vectors * q
        inverse = toeplitz(_inverse_harmonics(layer, 2 * count, period))
        across = np.diag(kx).astype(np.complex128)
        operator = np.linalg.solve(
            inverse, across @ np.linalg.inv(toeplitz(eps)) @ across - identity
        )
        values, vectors = np.linalg.eig(operator)
        q = _forward(values)
        return vectors, q, inverse @ (vectors * q)

    def interface(
        above: Modes, below: Modes
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        wa, _, va = above
        wb, _, vb = below
        left = np.block([[wa, -wb], [va, vb]])
        right = np.block([[-wa, wb], [va, vb]])
        s = np.linalg.solve(left, right)
        return s[:size, :size], s[:size, size:], s[size:, :size], s[size:, size:]

    stack: list[Modes] = [uniform_modes(top_index)]
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
    if incident is None:
        beam = np.zeros(size, dtype=np.complex128)
        beam[zero] = 1.0
    else:
        beam = np.asarray(incident, dtype=np.complex128)
        if beam.shape != (size,):
            raise ValueError(f"incident is one amplitude per order, {size}, got {beam.shape}")
    back = reflected @ beam

    # The transmitted amplitudes: pass down through every interface and layer.
    forward = beam
    for i in range(len(joins)):
        forward = passes[i] @ forward
        if i + 1 < len(joins):
            forward = np.exp(-k0 * stack[i + 1][1] * thickness[i + 1]) * forward
    wave_top = np.sqrt(top_index**2 - (kx * 1.0) ** 2 + 0j)
    wave_sub = np.sqrt(substrate_index**2 - kx**2 + 0j)
    delivered = float(
        np.sum(np.abs(beam) ** 2 * np.where(wave_top.real > 1e-12, wave_top.real, 0.0))
    )
    if delivered <= 0.0:
        raise ValueError("the incident beam carries no propagating power")
    reflectance = np.where(
        wave_top.real > 1e-12, np.abs(back) ** 2 * wave_top.real / delivered, 0.0
    )
    # Each order's flux down the axis: ``Re(k_z)`` for TE's electric field, and for
    # TM's magnetic field ``Re(k_z / eps)`` against the incident medium's own ``eps``.
    flux_sub = (
        (wave_sub / complex(substrate_index) ** 2).real * top_index**2 if tm else wave_sub.real
    )
    transmittance = np.where(
        wave_sub.real > 1e-12, np.abs(forward) ** 2 * flux_sub / delivered, 0.0
    )
    return Diffraction(
        reflection=complex(np.vdot(beam, back) / np.vdot(beam, beam)),
        orders=orders,
        reflectance=np.asarray(reflectance, dtype=float),
        transmittance=np.asarray(transmittance, dtype=float),
    )
