"""Getting light onto a chip: an edge coupler and a grating coupler.

Every photonic block in this library takes the signal's ``Ex`` to be the chip's
TE mode and ``Ey`` its TM, which silently says the die is perfectly aligned to
the fibre and that coupling onto it costs nothing. Neither is true, and both are
often the largest numbers in a photonic link budget -- a few decibels per facet,
where a ring filter costs tenths.

These two blocks are that step, with its alignment and its extinction.

- :class:`EdgeCoupler` butts a fibre against a polished facet. What it loses is
  mode mismatch and misalignment, computed as the overlap of two Gaussian beams,
  and the two facets' Fresnel reflections. It is broadband and treats both
  polarizations alike.
- :class:`GratingCoupler` diffracts light up out of the chip surface into a fibre
  held above it at an angle. Where it couples best is phase matching and is
  computed; its passband shape is a compact model, as a PDK gives one. It couples
  TE and rejects TM, which is the reason polarization-diverse receivers exist.

Both output the signal in the **chip's** frame. ``rotation`` is the angle of the
die's TE axis from the signal's x axis: the fields are projected onto the die's
axes first, then each polarization gets its own response. Noise passes the
projection unchanged, as it does through
:class:`~maiman.components.PolarizationRotator`, and is then scaled per axis.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace

import numpy as np

from ..circuit import SMatrix
from ..component import Param, PortType
from ..context import SimulationContext
from ..photonics import (
    edge_coupler,
    fresnel_reflectance,
    gaussian_overlap,
    grating_coupler,
    grating_coupler_centre,
)
from ..signals import OpticalSignal, Signal
from ..units import C_LIGHT
from .photonic import ScatteringDevice, apply_response, port_response, solve_once


def _onto_die(signal: OpticalSignal, angle: float) -> OpticalSignal:
    """Project the fields onto axes rotated by ``angle``: TE along it, TM across it."""
    if angle == 0.0:
        return signal
    c, s = math.cos(angle), math.sin(angle)
    bands = []
    for band in signal.bands:
        ex = band.Ex.astype(np.complex128)
        ey = band.Ey.astype(np.complex128)
        bands.append(
            replace(
                band,
                Ex=(c * ex + s * ey).astype(band.Ex.dtype),
                Ey=(-s * ex + c * ey).astype(band.Ey.dtype),
            )
        )
    return OpticalSignal(
        bands=tuple(bands), noise=signal.noise, accumulated_gvd=signal.accumulated_gvd
    )


class _ChipCoupler(ScatteringDevice):
    """The alignment and the per-polarization response both couplers share."""

    abstract = True
    category = "Photonic IC"

    rotation = Param(
        0.0, unit="deg", min=-90.0, max=90.0, doc="Die's TE axis against the signal's x axis"
    )

    inputs = {"in": PortType.OPTICAL}
    outputs = {"out": PortType.OPTICAL}

    def run(self, ctx: SimulationContext, inputs: dict[str, Signal]) -> dict[str, Signal]:
        signal: OpticalSignal = inputs["in"]
        aligned = _onto_die(signal, self.si("rotation"))
        te = port_response(self._matrix_factory("te"), "out", "in")
        tm = port_response(self._matrix_factory("tm"), "out", "in")
        return {"out": apply_response(aligned, te, response_y=tm)}


class EdgeCoupler(_ChipCoupler):
    """A fibre butted against a chip facet, through a spot-size converter.

    **What it loses.** The overlap of the fibre's mode with the chip's, each taken
    as a Gaussian -- a round 10.4 micron fibre mode onto an elliptical chip mode
    -- with a lateral offset, a tilt in the horizontal plane and a free-space gap
    between them, all in one closed form (see
    :func:`maiman.photonics.gaussian_overlap`). Then each facet's Fresnel
    reflection. A standard fibre onto a 3 micron mode loses 5.47 dB to mismatch
    alone and 5.78 dB with the facets. A lensed fibre or a larger spot-size
    converter buys the mismatch back; an index-matched gap removes the facets.

    **What it does not.** The etalon the two facets make across a gap -- a few
    percent of ripple for a bare air gap, nothing for an index-matched one -- and
    any difference between the chip mode's TE and TM sizes. Both polarizations
    see the same coupling.
    """

    display_name = "Edge Coupler"

    fibre_mfd = Param(
        10.4, unit="um", min=0.5, max=100.0, doc="Fibre mode field diameter; SMF-28 at 1550 nm"
    )
    chip_mfd_x = Param(
        3.0, unit="um", min=0.1, max=100.0, doc="Chip mode field diameter, in the chip's plane"
    )
    chip_mfd_y = Param(
        3.0, unit="um", min=0.1, max=100.0, doc="Chip mode field diameter, across the chip"
    )
    offset_x = Param(0.0, unit="um", min=-50.0, max=50.0, doc="Lateral misalignment, in plane")
    offset_y = Param(0.0, unit="um", min=-50.0, max=50.0, doc="Lateral misalignment, vertical")
    tilt = Param(0.0, unit="deg", min=-30.0, max=30.0, doc="Angle of the fibre in the chip's plane")
    gap = Param(0.0, unit="um", min=0.0, max=1000.0, doc="Fibre-to-facet distance")
    gap_index = Param(
        1.0, unit="", min=1.0, max=4.0, doc="What fills the gap; 1.45 is index-matched"
    )
    fibre_index = Param(1.4682, unit="", min=1.0, max=4.0, doc="Fibre core index at its facet")
    mode_index = Param(
        1.45, unit="", min=1.0, max=4.0, doc="Chip mode's index at the facet; low for a taper"
    )

    def coupled_fraction(self, wavelength: float = 1550e-9) -> float:
        """Power transmitted through the joint, both facets included."""
        matrix = self.scattering_matrix(np.array([C_LIGHT / wavelength]))
        return float(abs(matrix.s[0, 1, 0]) ** 2)

    def mode_overlap(self, wavelength: float = 1550e-9) -> float:
        """The mismatch and misalignment alone, without the facets."""
        fibre = self.si("fibre_mfd") / 2.0
        horizontal = gaussian_overlap(
            fibre,
            self.si("chip_mfd_x") / 2.0,
            wavelength=wavelength,
            offset=self.si("offset_x"),
            tilt=self.si("tilt"),
            gap=self.si("gap"),
            index=self.gap_index,
        )
        vertical = gaussian_overlap(
            fibre,
            self.si("chip_mfd_y") / 2.0,
            wavelength=wavelength,
            offset=self.si("offset_y"),
            gap=self.si("gap"),
            index=self.gap_index,
        )
        return float(horizontal * vertical)

    def facet_loss(self) -> float:
        """Power lost to the two facets' reflections, as a fraction."""
        return 1.0 - (1.0 - fresnel_reflectance(self.fibre_index, self.gap_index)) * (
            1.0 - fresnel_reflectance(self.mode_index, self.gap_index)
        )

    def _matrix_factory(self, polarization: str = "te") -> Callable[[np.ndarray], SMatrix]:
        # Both polarizations alike, as the docstring says.
        fibre = self.si("fibre_mfd") / 2.0
        chip_x, chip_y = self.si("chip_mfd_x") / 2.0, self.si("chip_mfd_y") / 2.0
        offset_x, offset_y = self.si("offset_x"), self.si("offset_y")
        tilt, gap = self.si("tilt"), self.si("gap")
        gap_index, fibre_index, mode_index = self.gap_index, self.fibre_index, self.mode_index
        return solve_once(
            lambda f: edge_coupler(
                f,
                fibre_mode_radius=fibre,
                chip_mode_radius_x=chip_x,
                chip_mode_radius_y=chip_y,
                offset_x=offset_x,
                offset_y=offset_y,
                tilt=tilt,
                gap=gap,
                gap_index=gap_index,
                fibre_index=fibre_index,
                mode_index=mode_index,
            )
        )


class GratingCoupler(_ChipCoupler):
    """A grating on the chip's surface, lifting TE light to a fibre held at an angle.

    **Where it couples** is phase matching along the grating, with the guided
    mode's dispersion in it: ``lambda_c = period (n_g - n sin theta) /
    (1 + period (n_g - n_eff) / lambda_0)``. The default grating -- a 611 nm period
    in 220 nm silicon, fibre at 10 degrees in air -- centres at 1549.8 nm and
    tunes by 7.0 nm per degree of fibre angle, where the same grating without
    dispersion would tune by 10.5. That is computed, not declared.

    **How well, and over how wide a band** is a compact model: a peak loss and a
    1 dB bandwidth, with a Gaussian passband between them, which is what a
    foundry's PDK measures and publishes. Computing those needs the grating's
    full vertical stack -- its directionality into the substrate and its mode
    match to the fibre -- and a number taken from a wafer is better than one
    invented here.

    **TM is rejected** by ``tm_extinction``: the grating is phase-matched for TE
    only, which is why a receiver that must accept any polarization uses a
    two-dimensional grating and two chips' worth of circuit.
    """

    display_name = "Grating Coupler"

    period = Param(611.0, unit="nm", min=100.0, max=2000.0, doc="Grating period")
    effective_index = Param(
        2.71, unit="", min=1.0, max=4.0, doc="Guided mode's index in the grating region"
    )
    group_index = Param(
        4.0, unit="", min=1.0, max=6.0, doc="Guided mode's group index; sets the angle tuning"
    )
    reference_wavelength = Param(
        1550.0, unit="nm", min=1200.0, max=1700.0, doc="Wavelength the two indices are quoted at"
    )
    fibre_angle = Param(10.0, unit="deg", min=-40.0, max=40.0, doc="Fibre's angle from vertical")
    medium_index = Param(
        1.0, unit="", min=1.0, max=2.0, doc="What the fibre sits in above the chip"
    )
    peak_loss = Param(3.0, unit="dB", min=0.0, max=30.0, doc="Loss at the centre wavelength")
    bandwidth_1db = Param(35.0, unit="nm", min=1.0, max=200.0, doc="Full width at 1 dB below peak")
    back_reflection = Param(20.0, unit="dB", min=0.0, max=80.0, doc="Reflection below the input")
    tm_extinction = Param(
        25.0, unit="dB", min=0.0, max=80.0, doc="How far below TE a TM input couples"
    )

    def centre_wavelength(self) -> float:
        """Where it couples best [m]."""
        return grating_coupler_centre(
            period=self.si("period"),
            effective_index=self.effective_index,
            group_index=self.group_index,
            reference_wavelength=self.si("reference_wavelength"),
            angle=self.si("fibre_angle"),
            medium_index=self.medium_index,
        )

    def angle_tuning(self) -> float:
        """How far the centre moves per radian of fibre angle [m/rad]."""
        angle = self.si("fibre_angle")
        return float(
            -self.si("period")
            * self.medium_index
            * math.cos(angle)
            / (
                1.0
                + self.si("period")
                * (self.group_index - self.effective_index)
                / self.si("reference_wavelength")
            )
        )

    def _matrix_factory(self, polarization: str = "te") -> Callable[[np.ndarray], SMatrix]:
        # Raw dB throughout: si() on a dB parameter hands back a linear ratio.
        peak = self.peak_loss + (self.tm_extinction if polarization == "tm" else 0.0)
        period, angle = self.si("period"), self.si("fibre_angle")
        effective, group = self.effective_index, self.group_index
        reference, medium = self.si("reference_wavelength"), self.medium_index
        bandwidth, echo = self.si("bandwidth_1db"), self.back_reflection
        return solve_once(
            lambda f: grating_coupler(
                f,
                period=period,
                effective_index=effective,
                group_index=group,
                reference_wavelength=reference,
                angle=angle,
                medium_index=medium,
                peak_loss_db=peak,
                bandwidth_1db=bandwidth,
                back_reflection_db=echo,
            )
        )
