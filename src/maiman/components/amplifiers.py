"""Optical amplifiers.

Model reference: G. P. Agrawal, *Fiber-Optic Communication Systems*, ch. 6
(optical amplifiers, ASE and noise figure); ITU-T G.661 for the definitions;
A. A. M. Saleh, R. M. Jopson, J. D. Evankow and J. Aspell, "Modeling of gain in
erbium-doped fiber amplifiers", IEEE Photon. Technol. Lett. 2(10), 1990, for
gain compression.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from ..component import BoolParam, Component, Param, PortType
from ..context import SimulationContext
from ..signals import Band, NoiseBin, OpticalSignal, Signal
from ..units import C_LIGHT, H_PLANCK, db_to_linear

if TYPE_CHECKING:
    from ..transient import ErbiumSpectrum


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

    **This block itself emits it flat.** It knows no cross sections, so every
    hertz of ``bandwidth`` gets the same density. Given an
    :class:`~maiman.transient.ErbiumSpectrum`, :func:`~maiman.transient.spectral_ase_noise_bin`
    reshapes that same total power the way a real coil would — more of it where
    the erbium's own gain curve is higher, none of it moved into or out of the
    band — as an offline step over this component's output, not something the
    component does on every run.

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

    **Saturation is driven by total power, incoming ASE included.** That is what
    makes two WDM channels share a gain that neither of them alone would have
    compressed.

    **And, with** ``self_saturation``, **by the amplifier's own ASE.** Saleh's
    equation is energy conservation for the reservoir: ``P_sat ln(G_0 / G)`` is
    everything the inversion hands out, output minus input over every field in
    the fibre. The spontaneous emission the amplifier generates is one of those
    fields, and it has no input. It leaves both ends -- a uniformly inverted
    fibre sends as much backwards as forwards -- and each end carries what this
    block emits, ``2 S_ASE B = NF G h nu B`` over both polarizations. So::

        ln(G_0 / G) = [(G - 1) P_in + 2 NF G h nu B] / P_sat

    A lightly loaded amplifier is where that matters: with no input at all the
    gain is ``W(beta G_0) / beta``, Lambert's W, with ``beta = 2 NF h nu B /
    P_sat`` -- a ceiling on how much gain an erbium coil can hold before its own
    noise empties it. This block knows no cross sections, so here it drains the
    reservoir at the centre wavelength's rate; given an
    :class:`~maiman.transient.ErbiumSpectrum`, the transient analysis weighs each
    part of the band by its own. Off by default, because it moves every saturated
    amplifier's gain.

    **The gain here is the steady state the erbium settles into**, and within any
    window this engine runs that is not an approximation but the right answer.
    The transient after a channel is added or dropped relaxes with
    ``tau / (1 + P_out / P_sat)``, which for this amplifier is between 3.3 and
    9.9 ms; a window of 4096 symbols at 32 GBd is 128 ns. The fastest transient
    is twenty-five thousand windows long, so the gain across one of them is a
    constant to a part in ten thousand.

    What happens *between* steady states is :mod:`maiman.transient`, which
    integrates the dynamic form of the very equation solved above — same ``G_0``,
    same ``P_sat``, one lifetime added — and a test runs it to rest against
    :meth:`effective_gain` to make sure the two never drift apart. It is an
    analysis on its own time axis rather than a parameter here, because a
    parameter that cannot change the result of a run is a control this library
    will not show.
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
        applies_when="saturate",
    )
    self_saturation = BoolParam(
        False,
        doc="Let the amplifier's own ASE, forwards and backwards, deplete its inversion",
        applies_when="saturate",
    )
    center_wavelength = Param(
        1550.0, unit="nm", min=1200.0, max=1700.0, doc="Centre of the ASE band"
    )
    bandwidth = Param(4.0, unit="THz", min=0.0, doc="Optical bandwidth over which ASE is emitted")

    inputs = {"in": PortType.OPTICAL}
    outputs = {"out": PortType.OPTICAL}

    def __init__(
        self,
        erbium_spectrum: object = None,
        *,
        label: str | None = None,
        **params: float | bool,
    ) -> None:
        """``erbium_spectrum``: the coil's measured Giles curves, to shape the ASE by.

        Structural rather than a parameter, because it is two curves and not a
        number, and the library ships none -- an erbium spectrum belongs to the
        glass it was measured in. Given one, the ASE the block emits keeps the total
        the noise figure sets and takes the shape ``n_sp(lambda) (G(lambda) - 1)`` of
        :func:`~maiman.transient.spectral_ase_noise_bin`, where without one it is
        flat across ``bandwidth``. A dictionary of the spectrum's three fields, as a
        project file stores it, is read back into one.
        """
        from ..transient import ErbiumSpectrum

        super().__init__(label=label, **params)
        if isinstance(erbium_spectrum, dict):
            erbium_spectrum = ErbiumSpectrum(
                wavelengths=np.asarray(erbium_spectrum["wavelengths"], dtype=np.float64),
                absorption_db=np.asarray(erbium_spectrum["absorption_db"], dtype=np.float64),
                full_inversion_gain_db=np.asarray(
                    erbium_spectrum["full_inversion_gain_db"], dtype=np.float64
                ),
            )
        if erbium_spectrum is not None and not isinstance(erbium_spectrum, ErbiumSpectrum):
            raise TypeError(
                f"erbium_spectrum must be an ErbiumSpectrum or its fields, got {erbium_spectrum!r}"
            )
        self.erbium_spectrum: ErbiumSpectrum | None = erbium_spectrum

    def structural_config(self) -> dict[str, Any]:
        spectrum = self.erbium_spectrum
        if spectrum is None:
            return {}
        return {
            "erbium_spectrum": {
                "wavelengths": [float(v) for v in spectrum.wavelengths],
                "absorption_db": [float(v) for v in spectrum.absorption_db],
                "full_inversion_gain_db": [float(v) for v in spectrum.full_inversion_gain_db],
            }
        }

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

    def self_saturation_load(self) -> float:
        """The amplifier's own ASE per unit gain, both ends, both polarizations [W].

        ``2 NF h nu B``: the ASE this block emits from one end is ``2 S_ASE B``,
        which for the noise figure's ``n_sp`` is ``NF G h nu B``, and the other
        end emits the same. Zero unless ``self_saturation`` is set. Multiply by
        the gain for the power the reservoir spends on it.
        """
        if not (self.saturate and self.self_saturation):
            return 0.0
        frequency = C_LIGHT / self.si("center_wavelength")
        return 2.0 * db_to_linear(self.noise_figure) * H_PLANCK * frequency * self.si("bandwidth")

    def effective_gain(self, input_power: float, *, self_load_weight: float = 1.0) -> float:
        """Linear gain at this input power, from the Saleh compression model.

        Solves ``g + a*exp(g) - a + b*exp(g) - ln(G_0) = 0`` for ``g = ln G``,
        with ``a = P_in / P_sat`` and ``b`` the amplifier's own ASE per unit gain
        over ``P_sat`` (zero without ``self_saturation``). The function is
        increasing and convex, and the root is bracketed by ``[0, ln G_0]``, so
        Newton started at the upper end descends monotonically onto it — no
        bisection fallback and no iteration cap that could quietly return a
        half-converged gain.

        ``self_load_weight`` scales ``b``: how hard the ASE drains the reservoir
        against the same power at the centre wavelength. One here, where nothing
        knows the erbium's cross sections; :func:`maiman.transient.self_saturation_weight`
        computes it from a spectrum that does.
        """
        small_signal_gain = db_to_linear(self.gain)
        saturation = self.si("saturation_power")
        load = self.self_saturation_load() * self_load_weight
        if (
            not self.saturate
            or (input_power <= 0.0 and load <= 0.0)
            or small_signal_gain <= 2.0
            or saturation <= 0.0
        ):
            # Below 6 dB there is no half-gain point to anchor P_sat to, and an
            # amplifier that small is not what this model is for. Returning the
            # small-signal gain is exact for zero input and honest for the rest.
            return small_signal_gain

        intrinsic = self.intrinsic_saturation_power(small_signal_gain)
        a = max(input_power, 0.0) / intrinsic
        b = load / intrinsic
        log_small_signal = math.log(small_signal_gain)
        if b >= log_small_signal:
            # Its own noise would empty it before it reached unity gain.
            return 1.0
        g = log_small_signal
        for _ in range(64):
            grown = (a + b) * math.exp(g)
            step = (g + grown - a - log_small_signal) / (1.0 + grown)
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
        if bandwidth > 0.0 and psd > 0.0 and self.erbium_spectrum is not None:
            from ..transient import spectral_ase_noise_bin

            noise.append(spectral_ase_noise_bin(self, self.erbium_spectrum, gain))
        elif bandwidth > 0.0 and psd > 0.0:
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
                walkoff=signal.walkoff,
                nonlinear_history=signal.nonlinear_history,
            )
        }
