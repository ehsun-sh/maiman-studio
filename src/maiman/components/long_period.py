"""A long-period fibre grating: the core's light handed to the cladding, and lost there.

A Bragg grating's period is half a wavelength, and it couples the core mode to
itself travelling backwards. Stretch the period to hundreds of microns and the
phase matching changes partner: the core mode now couples *forwards*, to modes of
the cladding, at the wavelengths where ``(n_core - n_cladding,m) * period`` is
one wavelength. The cladding light is stripped by the coating, so what comes out
is the input with a notch cut at each of those wavelengths -- broad notches,
tens of nanometres, where a Bragg grating's are fractions of one.

Where the notches sit depends on the cladding modes, and they feel the glass-air
boundary. That is the property this device is used for: dip a long-period
grating in a liquid and its notches move, which is a refractometer with no
electronics in the fibre. Setting ``surrounding_index`` is that experiment.

Everything below is computed by :mod:`maiman.modes` -- the effective indices,
their dispersion, and the overlaps that set each notch's depth -- and nothing is
a fitted constant.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ..circuit import SMatrix
from ..component import Param, PortType
from ..context import SimulationContext
from ..modes import StepIndexFibre
from ..photonics import long_period_grating, long_period_resonances
from ..signals import OpticalSignal, Signal
from .photonic import ScatteringDevice, apply_response, port_response, solve_once

#: Spectral resolution an ASE bin is averaged at [Hz]. A long-period notch is
#: tens of nanometres wide, thousands of gigahertz, so ten is far finer than any
#: feature in the response.
RESOLUTION = 10e9


class LongPeriodGrating(ScatteringDevice):
    """Couples the core mode to cladding modes, and transmits what is left.

    **What it models.** Coupled-mode theory between the core's LP01 and the first
    ``cladding_modes`` cladding modes of the same azimuthal order -- the only
    ones an untilted grating written evenly across the core can reach. The
    coupled equations for a uniform grating have constant coefficients, so they
    are solved exactly, all modes at once, by diagonalising a small matrix at
    every frequency. A notch's depth is ``cos^2(kappa L)`` at its centre, and its
    width is set by how fast the two indices part with wavelength.

    **What it does not.** The fields are scalar LP modes: the cladding-air step
    is not weak, and the vector modes these stand in for sit up to about 1e-4 away
    in effective index, a few nanometres in where a notch lands. Material
    dispersion is not included -- the indices are constants and only the
    waveguide's dispersion is computed -- and the average index the writing
    raises is not either, so a real grating's notches sit a few nanometres
    longward of these. The cladding light is taken to be lost, as it is under a
    coating; its recoupling at a second grating is not modelled.
    """

    display_name = "Long-Period Grating"
    category = "Passive"

    period = Param(500.0, unit="um", min=50.0, max=2000.0, doc="Grating period")
    length = Param(25.0, unit="mm", min=0.1, max=500.0, doc="Written length")
    index_modulation = Param(
        3e-4, unit="", min=0.0, max=1e-2, doc="Index modulation of the core, as for a Bragg grating"
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
        doc="What the fibre sits in: 1 is air, 1.333 water. Moves every notch",
    )
    cladding_modes = Param(
        8.0, unit="", min=1.0, max=40.0, doc="Cladding modes coupled, highest index first"
    )

    inputs = {"in": PortType.OPTICAL}
    outputs = {"transmitted": PortType.OPTICAL}

    def fibre(self) -> StepIndexFibre:
        """The fibre the grating is written in, in SI units."""
        return StepIndexFibre(
            core_radius=self.si("core_radius"),
            cladding_radius=self.si("cladding_radius"),
            core_index=self.core_index,
            cladding_index=self.cladding_index,
            surrounding_index=self.surrounding_index,
        )

    def resonances(self, band: tuple[float, float] = (1.2e-6, 1.7e-6)) -> list[tuple[int, float]]:
        """``(cladding mode rank, wavelength [m])`` for every notch inside ``band``."""
        return long_period_resonances(
            self.fibre(),
            period=self.si("period"),
            count=int(self.cladding_modes),
            band=band,
        )

    def run(self, ctx: SimulationContext, inputs: dict[str, Signal]) -> dict[str, Signal]:
        signal: OpticalSignal = inputs["in"]
        response = port_response(self._matrix_factory(), "out", "in")
        return {"transmitted": apply_response(signal, response, resolution=RESOLUTION)}

    def _matrix_factory(self, polarization: str = "te") -> Callable[[np.ndarray], SMatrix]:
        # ``polarization`` is accepted and ignored: LP modes are the scalar
        # approximation, in which the two polarizations are degenerate.
        fibre = self.fibre()
        period = self.si("period")
        length = self.si("length")
        modulation = self.index_modulation
        count = int(self.cladding_modes)
        return solve_once(
            lambda f: long_period_grating(
                f,
                fibre=fibre,
                period=period,
                length=length,
                index_modulation=modulation,
                cladding_modes=count,
            )
        )
