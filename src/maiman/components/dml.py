"""A directly modulated laser: the drive current goes into the laser itself.

Every other transmitter here leaves the laser alone and modulates the light
afterwards. This one modulates the current, which is how a short-reach
transceiver is built -- one chip, no modulator, a fraction of the power -- and
the reason it is not used everywhere is in the output: the carrier density has
to move for the power to move, and the optical frequency moves with it.

The chirp is not a parameter. It comes out of
:mod:`maiman.laser`'s rate equations through the linewidth enhancement factor,
in both of its forms: the transient chirp that follows ``d(ln P)/dt`` and rings
with the relaxation oscillation, and the adiabatic chirp that holds the ones a
fixed distance in frequency from the zeros for as long as they last.
"""

from __future__ import annotations

import numpy as np

from ..component import Component, Param, PortType
from ..context import SimulationContext
from ..laser import LaserParameters, LaserWaveform, integrate_rate_equations
from ..signals import Band, ElectricalSignal, OpticalSignal, Signal
from ..units import C_LIGHT


class DirectlyModulatedLaser(Component):
    """Drive current in, chirped light out.

    The input is the driver's voltage; ``bias_current`` and ``transconductance``
    turn it into a current, ``I = bias + g V``, which is what a laser driver
    does. Bias below :meth:`threshold_current` and the laser is not lasing, which
    the model will show rather than refuse: the light is spontaneous emission,
    the extinction ratio collapses and the chirp runs to tens of gigahertz.

    **What comes out is a field, not an intensity.** The band carries
    ``sqrt(P(t)) exp(i phi(t))`` with the phase the rate equations produced, so a
    span of fibre downstream converts that chirp into pulse distortion on its own
    -- which is the whole reason a directly modulated laser has a reach limit.

    The parameters are the active region's, because that is what the model is
    written in; :meth:`threshold_current`, :meth:`slope_efficiency` and
    :meth:`relaxation_frequency` report the datasheet quantities they imply.
    """

    display_name = "Directly Modulated Laser"
    category = "Optical Sources"

    bias_current = Param(35.0, unit="mA", min=0.0, doc="Current with no drive applied")
    transconductance = Param(
        25.0, unit="mA/V", min=0.0, doc="Drive current per volt of input waveform"
    )
    wavelength = Param(1310.0, unit="nm", min=1200.0, max=1700.0, doc="Emission wavelength")
    linewidth_enhancement = Param(
        4.0, unit="", min=0.0, max=10.0, doc="alpha: how far the index moves with the gain"
    )
    confinement = Param(0.3, unit="", min=0.01, max=1.0, doc="Mode overlap with the active region")
    gain_slope = Param(2.1e-12, unit="", min=1e-14, max=1e-10, doc="g0 = v_g dg/dN [m^3/s]")
    transparency_density = Param(
        1.0e24, unit="", min=1e22, max=1e26, doc="N_t [1/m^3]: where absorption stops"
    )
    carrier_lifetime = Param(1.0, unit="ns", min=0.01, max=100.0, doc="tau_n")
    photon_lifetime = Param(2.5, unit="ps", min=0.1, max=100.0, doc="tau_p: the cavity's loss")
    gain_compression = Param(
        1.5e-23, unit="", min=0.0, max=1e-20, doc="epsilon [m^3]: damps the ringing"
    )
    spontaneous_coupling = Param(
        1e-4, unit="", min=0.0, max=1e-2, doc="beta: spontaneous emission into the mode"
    )
    active_volume = Param(9.0e-17, unit="", min=1e-19, max=1e-14, doc="V [m^3]")
    output_coupling = Param(
        0.2, unit="", min=0.001, max=1.0, doc="Photons out of the facet that reach the fibre"
    )
    substeps = Param(
        16.0,
        unit="",
        min=1.0,
        max=1024.0,
        doc="Integration steps per sample; the ringing needs them",
    )

    inputs = {"in": PortType.ELECTRICAL}
    outputs = {"out": PortType.OPTICAL}

    def parameters(self) -> LaserParameters:
        """The active region, in SI units."""
        return LaserParameters(
            confinement=self.confinement,
            gain_slope=self.gain_slope,
            transparency_density=self.transparency_density,
            carrier_lifetime=self.si("carrier_lifetime"),
            photon_lifetime=self.si("photon_lifetime"),
            gain_compression=self.gain_compression,
            spontaneous_coupling=self.spontaneous_coupling,
            active_volume=self.active_volume,
            output_coupling=self.output_coupling,
            wavelength=self.si("wavelength"),
            linewidth_enhancement=self.linewidth_enhancement,
        )

    def threshold_current(self) -> float:
        """The datasheet's ``I_th`` [A], from the active region's own numbers."""
        return self.parameters().threshold_current()

    def slope_efficiency(self) -> float:
        """The datasheet's ``dP/dI`` above threshold [W/A]."""
        return self.parameters().slope_efficiency()

    def relaxation_frequency(self) -> float:
        """Where the step response rings at this bias [Hz]."""
        return self.parameters().relaxation_frequency(self.si("bias_current"))

    def drive_current(self, waveform: ElectricalSignal) -> np.ndarray:
        """``bias + g V`` [A], clipped at zero: a driver cannot pull current out."""
        volts = np.asarray(waveform.samples, dtype=np.float64)
        current = self.si("bias_current") + self.si("transconductance") * volts
        return np.maximum(current, 0.0)

    def solve(self, waveform: ElectricalSignal) -> LaserWaveform:
        """Integrate the rate equations across this drive, without building a band."""
        return integrate_rate_equations(
            self.parameters(),
            self.drive_current(waveform),
            waveform.fs,
            substeps=int(self.substeps),
        )

    def run(self, ctx: SimulationContext, inputs: dict[str, Signal]) -> dict[str, Signal]:
        waveform: ElectricalSignal = inputs["in"]
        if waveform.samples.shape[0] != ctx.num_samples:
            raise ValueError(
                f"{self.label}: the drive has {waveform.samples.shape[0]} samples and the run "
                f"window holds {ctx.num_samples}"
            )
        solved = self.solve(waveform)
        field = solved.field
        band = Band(
            Ex=field.astype(ctx.complex_dtype),
            Ey=np.zeros(ctx.num_samples, dtype=ctx.complex_dtype),
            f0=C_LIGHT / self.si("wavelength"),
            fs=ctx.sample_rate,
        )
        return {"out": OpticalSignal(bands=(band,))}
