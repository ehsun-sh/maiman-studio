"""A mirror in the line, and the part that lets a one-way graph reach it.

The first reflecting device in this library. Everything else is matched at both
ends and light goes in one port and out another, which is what a dataflow graph
draws. A Bragg grating sends its channel back out of the fibre it arrived on, and
in hardware the only way to that channel is a circulator.

Five things are worth reading off the output.

**The grating is assembled, not written down.** Two hundred per-section transfer
matrices multiplied together, against Erdogan's closed form for a uniform
grating, which appears nowhere in the model. They agree to twelve digits — and
the point of building it that way is the next two items, which have no closed
form at all.

**Apodization is not only about sidelobes.** The textbook reason to shape a
grating's strength is crosstalk: a uniform grating's first sidelobe is 7.6 dB
down, which is hopeless for dropping a channel, and a raised-cosine profile takes
it to 31 dB. The reason that matters for a *compensator* is different and larger.
An unapodized chirped grating has 101 ps rms of group-delay ripple — a whole bit
at 10 Gb/s — and apodizing it leaves 6.5 ps. Same profile, a different failure
avoided.

**A chirped grating is a dispersion compensator that works without coherent
detection.** ``DispersionCompensator`` is DSP: it needs the field, so it needs a
coherent receiver. This is glass. A 10 cm grating undoes 80 km of standard fibre
and would do it in front of a photodiode that destroys the phase.

**The circulator's loss is paid twice**, because the light passes through it
twice, and that is the price of reaching a reflective device at all. The drop
port below is exactly two hops and one reflectivity, and nothing else.

**And the sign is not what the grating's own physics says it is.** That is a
disagreement between two kernels in this engine rather than a property of
gratings, and it is printed rather than hidden — see the last section.
"""

from __future__ import annotations

import numpy as np

from maiman import Graph, SimulationContext
from maiman.circuit import SMatrix
from maiman.components import (
    Circulator,
    Combiner,
    CWLaser,
    Fiber,
    FiberBraggGrating,
    GaussianPulse,
    PowerMeter,
)
from maiman.photonics import APODIZATIONS, SILICA_FIBER_NEFF, fiber_bragg_grating
from maiman.signals import OpticalSignal, PowerReading
from maiman.units import C_LIGHT

BRAGG_NM = 1550.0
BRAGG = BRAGG_NM * 1e-9

#: 10 cm chirped over 0.71 nm is -1360 ps/nm by the geometric formula, which is
#: 80 km of standard fibre at 17 ps/nm/km.
COMPENSATOR: dict[str, float] = {
    "length": 100.0,  # mm
    "index_modulation": 2e-4,
    "bragg_wavelength": BRAGG_NM,
    "chirp": 0.71,  # nm — the sign is discussed at the end
}


def compensator() -> FiberBraggGrating:
    """The chirped grating used in sections 3 and 5, built the same way twice."""
    return FiberBraggGrating(
        "raised-cosine",
        length=COMPENSATOR["length"],
        index_modulation=COMPENSATOR["index_modulation"],
        bragg_wavelength=COMPENSATOR["bragg_wavelength"],
        chirp=COMPENSATOR["chirp"],
    )


SPAN_KM = 80.0


def erdogan_uniform(
    wavelengths: np.ndarray, *, length: float, index_modulation: float
) -> np.ndarray:
    """Reflectivity from the literature, sharing no arithmetic with the model."""
    detuning = 2.0 * np.pi * SILICA_FIBER_NEFF * (1.0 / wavelengths - 1.0 / BRAGG)
    kappa = np.pi * index_modulation / wavelengths
    gamma = np.sqrt((kappa**2 - detuning**2).astype(np.complex128))
    top = -kappa * np.sinh(gamma * length)
    bottom = detuning * np.sinh(gamma * length) + 1j * gamma * np.cosh(gamma * length)
    return np.abs(top / bottom) ** 2


def group_delay(matrix: SMatrix, port: str = "in") -> np.ndarray:
    """Group delay of a reflection [s]."""
    phase = np.unwrap(np.angle(matrix.transmission(port, port)))
    return -np.gradient(phase, 2.0 * np.pi * matrix.frequencies)


def rms_width_ps(signal: OpticalSignal) -> float:
    """RMS width of a pulse [ps], centred first so a bulk delay does not count."""
    (band,) = signal.bands
    power = np.abs(band.Ex) ** 2 + np.abs(band.Ey) ** 2
    power = np.roll(power, band.num_samples // 2 - int(np.argmax(power)))
    times = np.arange(band.num_samples) / band.fs
    centre = (power * times).sum() / power.sum()
    return float(np.sqrt(((times - centre) ** 2 * power).sum() / power.sum())) * 1e12


def against_the_closed_form() -> None:
    """Two hundred matrices, against one formula."""
    print("1. A uniform grating, assembled, against Erdogan's closed form")
    print(f"     {'delta-n':>9}  {'kappa L':>8}  {'tanh^2(kL)':>11}  {'model':>10}  {'max err':>9}")

    wavelengths = np.linspace(BRAGG - 1e-9, BRAGG + 1e-9, 2001)
    for index_modulation in (1e-5, 5e-5, 1e-4, 5e-4):
        length = 0.01
        kappa_l = np.pi * index_modulation / BRAGG * length
        model = fiber_bragg_grating(
            C_LIGHT / wavelengths,
            length=length,
            index_modulation=index_modulation,
            bragg_wavelength=BRAGG,
        ).power("in", "in")
        reference = erdogan_uniform(wavelengths, length=length, index_modulation=index_modulation)
        peak = model[len(model) // 2]
        print(
            f"     {index_modulation:9.0e}  {kappa_l:8.3f}  {np.tanh(kappa_l) ** 2:11.6f}  "
            f"{peak:10.6f}  {np.abs(model - reference).max():9.2e}"
        )
    print("     the formula is in this file, not in the model it is checked against.\n")


def apodization_trade() -> None:
    """Both of the things shaping the coupling buys, and what it costs."""
    print("2. Apodization: sidelobes, and the ripple that matters more")
    print(f"     {'profile':>14}  {'peak R':>8}  {'sidelobe':>9}  {'GD ripple':>10}")

    wavelengths = np.linspace(BRAGG - 2e-9, BRAGG + 2e-9, 40001)
    chirped = np.linspace(BRAGG - 0.12e-9, BRAGG + 0.12e-9, 2001)

    for profile in APODIZATIONS:
        reflectivity = fiber_bragg_grating(
            C_LIGHT / wavelengths,
            length=0.01,
            index_modulation=1e-4,
            bragg_wavelength=BRAGG,
            apodization=profile,
        ).power("in", "in")
        peak = int(np.argmax(reflectivity))
        lobes = [
            i
            for i in range(peak + 1, len(reflectivity) - 1)
            if reflectivity[i] > reflectivity[i - 1] and reflectivity[i] >= reflectivity[i + 1]
        ]
        sidelobe = 10.0 * np.log10(reflectivity[lobes[0]] / reflectivity[peak])

        # The same profile on the compensator, where the ripple is the number
        # that decides whether it is usable.
        matrix = fiber_bragg_grating(
            C_LIGHT / chirped,
            length=0.10,
            index_modulation=2e-4,
            bragg_wavelength=BRAGG,
            chirp=0.71e-9,
            apodization=profile,
        )
        delay = group_delay(matrix)
        fit = np.polyfit(chirped, delay, 1)
        ripple = np.std(delay - np.polyval(fit, chirped)) * 1e12

        print(f"     {profile:>14}  {reflectivity[peak]:8.4f}  {sidelobe:8.1f}dB  {ripple:9.1f}ps")
    print("     uniform is unusable as a compensator for the third column, not the second.\n")


def the_compensator() -> None:
    """What the chirped grating is, before it is put in a link."""
    print("3. The compensator, on its own")
    grating = compensator()
    geometric = grating.dispersion() * 1e3

    band = np.linspace(BRAGG - 0.12e-9, BRAGG + 0.12e-9, 2001)
    matrix = grating.scattering_matrix(C_LIGHT / band)
    reflectivity = matrix.power("in", "in")
    slope = np.polyfit(band, group_delay(matrix), 1)[0] * 1e3

    print(f"     length            {grating.si('length') * 1e3:.0f} mm")
    print(f"     chirp             {grating.si('chirp') * 1e9:.2f} nm")
    print(f"     reflectivity      {reflectivity.min():.4f} to {reflectivity.max():.4f} in band")
    print(f"     D, geometric      {geometric:.0f} ps/nm")
    print(f"     D, from the model {slope:.0f} ps/nm")
    print(f"     round trip 2nL/c  {2 * SILICA_FIBER_NEFF * 0.1 / C_LIGHT * 1e12:.0f} ps")
    print(f"     undoes            {abs(geometric) / 17.0:.0f} km of standard fibre\n")


def the_drop() -> None:
    """A channel dropped through a circulator, with the budget it costs.

    Every figure in the table is arithmetic: two hops of insertion loss and one
    reflectivity on the drop, one hop and one transmission on the express.
    """
    print("4. A channel dropped, and what the circulator charges for it")
    ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=8, sequence_length=64)
    graph = Graph(ctx)

    dropped = graph.add(CWLaser(power=0.0, wavelength=1550.0, label="ch1550"))
    express = graph.add(CWLaser(power=0.0, wavelength=1552.0, label="ch1552"))
    mux = graph.add(Combiner(2, label="mux"))
    circ = graph.add(Circulator(insertion_loss=0.7, label="circ"))
    grating = graph.add(
        FiberBraggGrating(bragg_wavelength=1550.0, length=10.0, index_modulation=1e-4)
    )
    drop_meter = graph.add(PowerMeter(label="drop"))
    thru_meter = graph.add(PowerMeter(label="thru"))

    graph.connect(dropped, mux["in0"])
    graph.connect(express, mux["in1"])
    graph.connect(mux, circ["in1"])
    graph.connect(circ["out2"], grating["in"])
    graph.connect(grating["reflected"], circ["in2"])
    graph.connect(circ["out3"], drop_meter["in"])
    graph.connect(grating["transmitted"], thru_meter["in"])

    results = graph.run()
    reflectivity = grating.peak_reflectivity()

    def at(reading: PowerReading, wavelength_nm: float) -> float:
        (band,) = [b for b in reading.bands if round(b.wavelength_nm, 2) == wavelength_nm]
        return band.power_dbm

    print(f"     grating: R = {reflectivity:.4f} over {grating.bandwidth() * 1e9:.3f} nm")
    print(f"     {'':>10}  {'1550 nm':>10}  {'1552 nm':>10}   accounting")
    print(
        f"     {'drop':>10}  {at(results[drop_meter], 1550.0):9.3f}dB  "
        f"{at(results[drop_meter], 1552.0):9.3f}dB   two hops "
        f"({-1.4:.1f}) + R ({10 * np.log10(reflectivity):.2f})"
    )
    print(
        f"     {'express':>10}  {at(results[thru_meter], 1550.0):9.3f}dB  "
        f"{at(results[thru_meter], 1552.0):9.3f}dB   one hop (-0.7) + T "
        f"({10 * np.log10(1 - reflectivity):.2f})"
    )
    print(
        "     the neighbour's rejection on the drop port is the grating's own\n"
        "     sidelobe floor two nanometres out, not an isolation figure declared\n"
        "     anywhere. Apodize it and the number moves.\n"
    )


def a_span_undone() -> None:
    """The grating in a line, against the fibre it is there to cancel."""
    print(f"5. {SPAN_KM:.0f} km of standard fibre, undone by 10 cm of glass")

    def width(length_km: float, *, compensate: bool) -> float:
        ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=64, sequence_length=256, seed=1)
        graph = Graph(ctx)
        source = graph.add(GaussianPulse(width=30.0, peak_power=1.0, wavelength=BRAGG_NM))
        fiber = graph.add(Fiber(length=length_km, attenuation=0.0, dispersion=17.0))
        graph.connect(source, fiber["in"])

        if not compensate:
            meter = graph.add(PowerMeter(label="meter"))
            graph.connect(fiber, meter["in"])
            return rms_width_ps(graph.run(keep=[fiber]).port(fiber, "out"))

        circ = graph.add(Circulator(insertion_loss=0.0, label="circ"))
        grating = graph.add(compensator())
        meter = graph.add(PowerMeter(label="meter"))
        graph.connect(fiber, circ["in1"])
        graph.connect(circ["out2"], grating["in"])
        graph.connect(grating["reflected"], circ["in2"])
        graph.connect(circ["out3"], meter["in"])
        return rms_width_ps(graph.run(keep=[circ]).port(circ, "out3"))

    launched = width(0.0, compensate=False)
    print(f"     {'span':>6}  {'bare':>9}  {'+ grating':>10}")
    for length_km in (0.0, 40.0, 80.0, 120.0):
        print(
            f"     {length_km:5.0f}km  {width(length_km, compensate=False):8.2f}ps  "
            f"{width(length_km, compensate=True):9.2f}ps"
        )
    print(f"     launched {launched:.2f} ps rms; the grating is matched to {SPAN_KM:.0f} km, so")
    print("     40 km is over-compensated and 120 km is not compensated enough.")
    print("     None of this needs a coherent receiver: it happens in the glass.\n")


def the_sign() -> None:
    """The one result here that is about this engine rather than about gratings."""
    print("6. A sign worth stating out loud")
    print(
        "     The grating's own physics says a positive chirp puts short wavelengths\n"
        "     at the near end, so they turn round first and the long ones arrive\n"
        "     later -- dtau/dlambda > 0, which is D > 0, the sign standard fibre has.\n"
        "     By that reading a compensator is a NEGATIVE chirp.\n"
        "\n"
        "     Section 5 used a positive one, because that is what actually cancels\n"
        "     this engine's Fiber. The two disagree because kernels.propagate_dispersion\n"
        "     carries the opposite quadratic sign from photonics.propagation_constant,\n"
        "     which the latter's docstring has said since before this device existed.\n"
        "     The grating is simply the first component to put both in one graph.\n"
        "\n"
        "     tests/test_grating.py pins the behaviour as it stands, so that the day\n"
        "     the kernel is reconciled, the failure names the compensator as one of\n"
        "     the things that moved with it."
    )


def main() -> None:
    against_the_closed_form()
    apodization_trade()
    the_compensator()
    the_drop()
    a_span_undone()
    the_sign()


if __name__ == "__main__":
    main()
