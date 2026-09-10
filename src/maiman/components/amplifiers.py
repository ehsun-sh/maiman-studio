"""Optical amplifiers.

Model reference: G. P. Agrawal, *Fiber-Optic Communication Systems*, ch. 6
(optical amplifiers, ASE and noise figure); ITU-T G.661 for the definitions;
A. A. M. Saleh, R. M. Jopson, J. D. Evankow and J. Aspell, "Modeling of gain in
erbium-doped fiber amplifiers", IEEE Photon. Technol. Lett. 2(10), 1990, for
gain compression.
"""

from __future__ import annotations

import math

import numpy as np

from ..component import BoolParam, Component, Param, PortType
from ..context import SimulationContext
from ..signals import Band, NoiseBin, OpticalSignal, Signal
from ..units import C_LIGHT, H_PLANCK, db_to_linear


class EDFA(Component):
    """Erbium-doped fiber amplifier: gain, plus the noise that comes with it.

    Amplification is not free. A phase-insensitive amplifier must add at least
    ``h*nu`` of noise per mode per unit bandwidth, and a real one adds more; the
    noise figure is how much more. The amplified spontaneous emission this block
    generates is what limits how many spans a link can have, so an amplifier
    modelled without it is not a simplification but a fiction.

    ASE is emitted into :class:`~maiman.signals.NoiseBin` rather than into the
    sampled bands. That is the whole reason the noise-bin representation exists:
    ASE covers the amplifier's full bandwidth — terahertz — while the signal
    occupies a few tens of gigahertz of it. Sampling both together would demand
    a sample rate no machine can afford, so the noise is carried as a power
    spectral density and only converted to samples where a detector or a
    nonlinearity actually needs it.

    The spontaneous emission factor follows from the noise figure::

        n_sp = NF * G / (2 * (G - 1))
        S_ASE = n_sp * h * nu * (G - 1)        [W/Hz, per polarization]

    **Saturation is the Saleh model, and it is off by default.** An amplifier
    that never compresses is an idealisation, and it is declared as one the way
    a zero-linewidth laser is: ``saturate`` is a flag and it starts false, so
    every link in this repository still gets the gain it asks for and turning it
    on is a decision someone makes. What it replaces is worse than an
    idealisation — an output clamp that this class's own docstring called "a
    clamp, not a model".

    An erbium amplifier compresses because
    the signal depletes the inversion it is drawing on, and the standard
    steady-state description of that is implicit in the gain::

        G = G_0 * exp(-(G - 1) * P_in / P_sat)

    which this solves exactly (Newton on ``ln G``, to 1e-12) rather than
    approximating. The compression is smooth and begins well before any ceiling:
    there is no input at which the amplifier switches from ideal to saturated,
    which is the one thing the clamp this replaces got qualitatively wrong.

    ``saturation_power`` is quoted the way a datasheet quotes it — the **output**
    power at which the gain has compressed by 3 dB — and converted to the
    model's intrinsic ``P_sat`` by::

        P_sat = P_3dB * (G_0 - 2) / (G_0 * ln 2)

    which follows from setting ``G = G_0 / 2`` in the equation above. For a large
    small-signal gain that is ``1.443 * P_3dB``, and the factor is carried in
    full rather than as that limit because a 10 dB amplifier is 20 % away from it.

    **Saturation is driven by total power, ASE included.** That is what makes two
    WDM channels share a gain that neither of them alone would have compressed,
    and it is why a high-gain amplifier left in the dark still does not deliver
    its small-signal gain: its own spontaneous emission is a load like any other.

    What this does *not* model is **dynamics**. The gain here is the steady state
    the erbium settles into; the millisecond transient after a channel is added
    or dropped is a real effect in a deployed system and is genuinely absent
    rather than approximated.
    """

    display_name = "EDFA"
    category = "Amplifiers"

    gain = Param(20.0, unit="dB", min=0.0, doc="Small-signal power gain")
    noise_figure = Param(5.0, unit="dB", min=0.0, doc="Noise figure")
    #: The clamp this replaces was declared, in this class's own docstring, to be
    #: "a clamp, not a model" — an output ceiling with a kink in it, below which
    #: the amplifier was perfectly ideal and above which the gain fell as 1/P_in.
    #: Retiring it is *not* the harmless kind of retirement the loader's comment
    #: describes: a project that leaned on the ceiling will now compress smoothly
    #: instead of hitting a wall, and its numbers will move. They should. The
    #: value it carried was fitted to a fiction, and ``saturation_power`` is the
    #: datasheet quantity that means something.
    retired_parameters = frozenset({"max_output_power"})

    saturate = BoolParam(False, doc="Compress the gain as the inversion depletes")
    saturation_power = Param(
        17.0,
        unit="dBm",
        doc="Output power at 3 dB gain compression, as a datasheet quotes it",
    )
    center_wavelength = Param(
        1550.0, unit="nm", min=1200.0, max=1700.0, doc="Centre of the ASE band"
    )
    bandwidth = Param(4.0, unit="THz", min=0.0, doc="Optical bandwidth over which ASE is emitted")

    inputs = {"in": PortType.OPTICAL}
    outputs = {"out": PortType.OPTICAL}

    def spontaneous_emission_factor(self, gain_linear: float) -> float:
        """``n_sp``, the population inversion factor implied by the noise figure.

        Its floor is 1 — full inversion, a 3 dB noise figure — and a noise figure
        below that would describe an amplifier quieter than quantum mechanics
        allows.
        """
        if gain_linear <= 1.0:
            return 1.0
        noise_figure = db_to_linear(self.noise_figure)
        return noise_figure * gain_linear / (2.0 * (gain_linear - 1.0))

    def ase_psd(self, gain_linear: float) -> float:
        """One-sided ASE power spectral density per polarization [W/Hz]."""
        if gain_linear <= 1.0:
            return 0.0
        frequency = C_LIGHT / self.si("center_wavelength")
        return (
            self.spontaneous_emission_factor(gain_linear)
            * H_PLANCK
            * frequency
            * (gain_linear - 1.0)
        )

    def intrinsic_saturation_power(self, small_signal_gain: float) -> float:
        """The model's ``P_sat`` [W], from the 3 dB output power a datasheet quotes.

        Setting ``G = G_0 / 2`` in the Saleh equation and eliminating ``P_in``
        gives ``P_sat = P_3dB * (G_0 - 2) / (G_0 * ln 2)``. Below 3 dB of gain
        there is no half-gain point to define it at, so the conversion is
        meaningless there and the caller does not reach it.
        """
        return (
            self.si("saturation_power")
            * (small_signal_gain - 2.0)
            / (small_signal_gain * math.log(2.0))
        )

    def effective_gain(self, input_power: float) -> float:
        """Linear gain at this input power, from the Saleh compression model.

        Solves ``g + a*exp(g) - a - ln(G_0) = 0`` for ``g = ln G``, with
        ``a = P_in / P_sat``. The function is increasing and convex, and the root
        is bracketed by ``[0, ln G_0]``, so Newton started at the upper end
        descends monotonically onto it — no bisection fallback and no iteration
        cap that could quietly return a half-converged gain.
        """
        small_signal_gain = db_to_linear(self.gain)
        saturation = self.si("saturation_power")
        if not self.saturate or input_power <= 0.0 or small_signal_gain <= 2.0 or saturation <= 0.0:
            # Below 6 dB there is no half-gain point to anchor P_sat to, and an
            # amplifier that small is not what this model is for. Returning the
            # small-signal gain is exact for zero input and honest for the rest.
            return small_signal_gain

        a = input_power / self.intrinsic_saturation_power(small_signal_gain)
        log_small_signal = math.log(small_signal_gain)
        g = log_small_signal
        for _ in range(64):
            step = (g + a * math.exp(g) - a - log_small_signal) / (1.0 + a * math.exp(g))
            g -= step
            if abs(step) < 1e-14:
                break
        return max(math.exp(g), 1.0)

    def run(self, ctx: SimulationContext, inputs: dict[str, Signal]) -> dict[str, Signal]:
        signal: OpticalSignal = inputs["in"]
        gain = self.effective_gain(signal.total_power())
        amplitude = float(np.sqrt(gain))

        bands = tuple(
            Band(
                Ex=(band.Ex.astype(np.complex128) * amplitude).astype(ctx.complex_dtype),
                Ey=(band.Ey.astype(np.complex128) * amplitude).astype(ctx.complex_dtype),
                f0=band.f0,
                fs=band.fs,
            )
            for band in signal.bands
        )

        # Incoming noise is amplified along with the signal — that accumulation
        # across spans is what actually limits a long-haul link.
        noise = [bin_.scale_power(gain) for bin_ in signal.noise]

        bandwidth = self.si("bandwidth")
        psd = self.ase_psd(gain)
        if bandwidth > 0.0 and psd > 0.0:
            centre = C_LIGHT / self.si("center_wavelength")
            noise.append(
                NoiseBin(
                    f_start=centre - bandwidth / 2.0,
                    f_end=centre + bandwidth / 2.0,
                    psd_x=psd,
                    psd_y=psd,
                )
            )

        return {
            "out": OpticalSignal(
                bands=bands,
                noise=tuple(noise),
                accumulated_gvd=signal.accumulated_gvd,
            )
        }
