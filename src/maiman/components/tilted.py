"""A tilted fibre Bragg grating: a mirror for the core, and a comb of doors to the cladding.

Write a Bragg grating with its fringes square to the axis and the core mode has
one partner to couple to, itself travelling backwards, and the grating is a
mirror at one wavelength. Tilt the fringes by a few degrees and the fringe front
now varies across the core, which lets the core mode reach every backward
*cladding* mode whose azimuthal shape matches the tilt -- each at its own
wavelength, a comb of narrow notches spread over tens of nanometres shortward of
the Bragg reflection, which the tilt weakens as it feeds the comb.

The point of the device is that one fibre now carries two rulers. The Bragg
reflection is guided by the core and is blind to what surrounds the fibre; the
cladding comb is guided by the glass-air boundary and moves when a liquid
touches it. Temperature moves both together. So the comb measured *against* the
Bragg line is a refractometer that carries its own temperature reference, which
is why these are used as biosensors and as refractive-index probes.

Everything is computed from :mod:`maiman.modes` and the coupled-mode equations
in :mod:`maiman.photonics` -- where each notch sits, how deep it is and how the
tilt divides the light between the comb and the mirror. Nothing is fitted.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from ..circuit import SMatrix
from ..component import BoolParam, Param, PortType
from ..context import SimulationContext
from ..modes import StepIndexFibre
from ..photonics import tilted_fiber_bragg_grating, tilted_grating_resonances
from ..signals import OpticalSignal, Signal
from ..units import C_LIGHT
from .photonic import ScatteringDevice, apply_response, port_response, solve_once


class TiltedFiberBraggGrating(ScatteringDevice):
    """Reflects the core mode into the cladding, mode by mode, and into itself.

    **What it models.** Contra-directional coupled modes between the core's LP01
    and every backward LP mode of the cladding up to ``azimuthal_orders``, plus
    the core mode itself. The fringes are tilted by ``tilt`` from square, so
    across the core they carry a transverse phase; expanded in azimuthal orders
    that phase is what opens each order's door, and the overlap it leaves is a
    Bessel function of ``2 pi sin(tilt) r / period``. The equations are solved
    exactly, every coupled mode at once. Cladding light is taken to be lost, as
    it is under a coating: it leaves through neither port.

    **What it does not.** The modes are scalar LP modes, so the splitting that
    makes a real tilted grating's comb depend on the input polarization is
    absent -- that needs the vector modes of :mod:`maiman.vector_modes`, which
    the long-period grating can use and this cannot yet. ``material_dispersion``
    lets the glass disperse, which moves the comb by picometres since its indices
    hold at 1550 nm, where the comb is. The average index the writing raises is
    not included, so a real grating's comb sits a few nanometres longward of this
    one.

    **Speed.** A mode solve per azimuthal order per wavelength node, and a
    tilted grating reaches dozens of cladding modes: expect seconds, and more of
    them for a wider band or a larger ``azimuthal_orders``.
    """

    display_name = "Tilted FBG"
    category = "Passive"

    period = Param(
        535.0, unit="nm", min=100.0, max=2000.0, doc="Period, measured normal to the fringes"
    )
    tilt = Param(4.0, unit="deg", min=0.0, max=45.0, doc="Fringe tilt from square to the axis")
    length = Param(10.0, unit="mm", min=0.1, max=500.0, doc="Written length")
    index_modulation = Param(
        5e-4, unit="", min=0.0, max=1e-2, doc="Index modulation of the core, as for a Bragg grating"
    )
    core_radius = Param(4.1, unit="um", min=0.5, max=50.0, doc="Core radius")
    cladding_radius = Param(62.5, unit="um", min=5.0, max=500.0, doc="Cladding radius")
    core_index = Param(1.4492, unit="", min=1.0, max=4.0, doc="Core refractive index")
    cladding_index = Param(1.4440, unit="", min=1.0, max=4.0, doc="Cladding refractive index")
    surrounding_index = Param(
        1.0,
        unit="",
        min=1.0,
        max=4.0,
        doc="What the fibre sits in: 1 is air, 1.333 water. Moves the comb and not the Bragg line",
    )
    material_dispersion = BoolParam(
        False,
        doc="Let the glass disperse: silica cladding, germania-doped core, "
        "the indices above holding at 1550 nm",
    )
    azimuthal_orders = Param(
        6.0, unit="", min=0.0, max=20.0, doc="Highest azimuthal order of cladding mode coupled"
    )

    inputs = {"in": PortType.OPTICAL}
    outputs = {"reflected": PortType.OPTICAL, "transmitted": PortType.OPTICAL}

    def fibre(self) -> StepIndexFibre:
        """The fibre the grating is written in, in SI units."""
        return StepIndexFibre(
            core_radius=self.si("core_radius"),
            cladding_radius=self.si("cladding_radius"),
            core_index=self.core_index,
            cladding_index=self.cladding_index,
            surrounding_index=self.surrounding_index,
            material_dispersion=self.material_dispersion,
        )

    def bragg_wavelength(self) -> float:
        """Where the core reflects into itself [m], ``2 n_0 period / cos(tilt)``.

        Solved, not assumed: ``n_0`` is the core mode's own effective index at
        the wavelength that comes back, found by two fixed-point steps.
        """
        from ..modes import core_modes

        axial = self.si("period") / math.cos(self.si("tilt"))
        wavelength = 2.0 * self.core_index * axial
        for _ in range(4):
            guided = core_modes(self.fibre(), wavelength)
            if not guided:
                raise ValueError(f"the fibre guides no core mode at {wavelength * 1e9:.1f} nm")
            wavelength = 2.0 * guided[0].effective_index * axial
        return wavelength

    def spectral_window(self) -> tuple[float, float]:
        """The comb below the Bragg line, and the line itself."""
        bragg = self.bragg_wavelength()
        return bragg - 6e-9, bragg + 1e-9

    def resonances(
        self, band: tuple[float, float] | None = None
    ) -> list[tuple[int, int, float, float]]:
        """``(order, rank, wavelength [m], coupling per unit dn)`` for every resonance in ``band``.

        Rank 0 is the Bragg reflection. The default band is the forty nanometres
        below the Bragg line and two above it, which is where a few degrees of
        tilt puts the comb.
        """
        bragg = self.bragg_wavelength()
        window = band if band is not None else (bragg - 40e-9, bragg + 2e-9)
        return tilted_grating_resonances(
            self.fibre(),
            period=self.si("period"),
            tilt=self.si("tilt"),
            band=window,
            max_order=int(self.azimuthal_orders),
        )

    def run(self, ctx: SimulationContext, inputs: dict[str, Signal]) -> dict[str, Signal]:
        signal: OpticalSignal = inputs["in"]
        matrix_for = self._matrix_factory()
        # A cladding notch is as narrow as the Bragg line is: its width is set by
        # the same kappa L and the same length. Averaging an ASE bin any coarser
        # than the ripple period would stride over notches and report where the
        # stride landed.
        resolution = C_LIGHT / (2.0 * self.core_index * self.si("length"))
        return {
            "reflected": apply_response(
                signal, port_response(matrix_for, "in", "in"), resolution=resolution
            ),
            "transmitted": apply_response(
                signal, port_response(matrix_for, "out", "in"), resolution=resolution
            ),
        }

    def _matrix_factory(self, polarization: str = "te") -> Callable[[np.ndarray], SMatrix]:
        # ``polarization`` is accepted and ignored: these are scalar LP modes,
        # in which the two polarizations are degenerate. A real tilted grating's
        # comb is not -- see the class docstring.
        fibre = self.fibre()
        period = self.si("period")
        tilt = self.si("tilt")
        length = self.si("length")
        modulation = self.index_modulation
        orders = int(self.azimuthal_orders)
        return solve_once(
            lambda f: tilted_fiber_bragg_grating(
                f,
                fibre=fibre,
                period=period,
                tilt=tilt,
                length=length,
                index_modulation=modulation,
                max_order=orders,
            )
        )
