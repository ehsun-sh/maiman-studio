"""A grating is a sensor, and one wavelength is not two numbers.

A Bragg grating's period is a length. Lengths respond to being stretched and to
being warmed, and the reflection reports it — so the same device that drops a
channel is a strain gauge and a thermometer, with nothing added to the fibre and
nothing electrical anywhere near the measurement. That is why these end up in
boreholes, inside composite spars, and along bridges: the sensing element is
glass, it is immune to electromagnetic interference, and because the reading is a
*wavelength* it does not drift when the light gets dimmer.

Five things are worth reading off the output.

**The two sensitivities are one coefficient each.** 1.21 pm per microstrain and
11.2 pm per kelvin at 1550 nm, both of them a material number times the Bragg
wavelength. Nothing here is fitted.

**One grating cannot tell you two things.** Dividing one sensitivity by the other
gives 9.26 microstrain per kelvin: a degree of drift is indistinguishable from
nine microstrain of load, and the wavelength that comes back is the same
wavelength. Every real installation is arranged around this fact.

**And a second grating at another wavelength does not fix it**, which is the
trap. Both sensitivities scale with lambda, so their ratio does not, and the
two-by-two you would invert has two nearly identical rows. The measured condition
number below says how hopeless it is.

**What does fix it is one grating that carries no load.** A reference grating
beside the working one, loose in its tube, measures the temperature alone — and
subtracting it recovers the strain to well under a microstrain.

**Interrogation is a wavelength sweep**, the way a real tunable-laser
interrogator works: step the probe across the band, record the reflected power,
and take the centroid of what comes back. The array is read down one fibre
through one circulator, because each grating passes what it does not reflect.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np

from maiman import Graph, SimulationContext, sweep
from maiman.components import (
    Circulator,
    CWLaser,
    FiberBraggGrating,
    PowerMeter,
)
from maiman.photonics import (
    SILICA_PHOTOELASTIC,
    SILICA_THERMAL_SENSITIVITY,
    bragg_shift,
)

BRAGG_NM = 1550.0
BRAGG = BRAGG_NM * 1e-9


def sensor(
    *,
    bragg_nm: float = BRAGG_NM,
    strain: float = 0.0,
    warming: float = 0.0,
    label: str | None = None,
) -> FiberBraggGrating:
    """One sensing grating: short and strong.

    A sensor wants a narrow peak whose centre it can find, not a wide one it can
    reflect all of — 10 mm at ``delta-n = 1.5e-4`` is 0.2 nm wide and reflects
    essentially everything at the middle of it.
    """
    return FiberBraggGrating(
        length=10.0,
        index_modulation=1.5e-4,
        bragg_wavelength=bragg_nm,
        strain=strain,
        temperature_change=warming,
        label=label,
    )


def sensitivities() -> None:
    """The two coefficients, and the one number that follows from both."""
    print("1. What a grating feels")
    grating = sensor()

    print(f"     strain       {grating.strain_sensitivity() * 1e-6 * 1e12:6.3f} pm per microstrain")
    print(f"                  = (1 - p_e) * lambda_B, p_e = {SILICA_PHOTOELASTIC}")
    print(f"     temperature  {grating.temperature_sensitivity() * 1e12:6.3f} pm per kelvin")
    print(
        f"                  = (alpha + xi) * lambda_B, "
        f"alpha + xi = {SILICA_THERMAL_SENSITIVITY:.2e} /K"
    )
    print(f"     cross        {grating.cross_sensitivity() * 1e6:6.2f} microstrain per kelvin")
    print()
    print(f"     {'load':>18}  {'reflects at':>13}")
    for strain, warming, label in (
        (0.0, 0.0, "unloaded"),
        (1000.0, 0.0, "1000 ustrain"),
        (0.0, 50.0, "+50 K"),
        (500.0, 20.0, "500 ue and +20 K"),
    ):
        moved = bragg_shift(BRAGG, strain=strain * 1e-6, temperature_change=warming)
        print(f"     {label:>18}  {moved * 1e9:12.4f} nm")
    print()


def one_wavelength_two_unknowns() -> None:
    """The ambiguity, shown as two different loads that read identically."""
    print("2. Why one grating is not a strain gauge")
    warmed = bragg_shift(BRAGG, temperature_change=10.0)
    pulled = bragg_shift(BRAGG, strain=92.6e-6)
    print(f"     +10 K, no load        -> {warmed * 1e9:.6f} nm")
    print(f"     92.6 ustrain, no heat -> {pulled * 1e9:.6f} nm")
    print(f"     difference             {abs(warmed - pulled) * 1e15:.3f} fm")
    print("     the same reading. An interrogator cannot tell these apart, and")
    print("     nothing about the spectrum hints that it should try.\n")


def the_two_wavelength_trap() -> None:
    """Why the obvious fix is not one: the matrix is nearly singular."""
    print("3. Two gratings at two wavelengths do not separate them")
    print(f"     {'lambda_B':>10}  {'pm/ustrain':>11}  {'pm/K':>8}  {'ratio':>8}")
    rows = []
    for nm in (1530.0, 1550.0, 1570.0):
        grating = sensor(bragg_nm=nm)
        strain = grating.strain_sensitivity() * 1e-6
        thermal = grating.temperature_sensitivity()
        rows.append([strain, thermal])
        print(
            f"     {nm:9.1f}nm  {strain * 1e12:11.4f}  {thermal * 1e12:8.4f}  "
            f"{grating.cross_sensitivity() * 1e6:8.3f}"
        )

    matrix = np.array([rows[0], rows[2]])
    print(f"     condition number of the 1530/1570 pair: {np.linalg.cond(matrix):.3e}")
    print("     the ratio column is the problem: it is the same number every time,")
    print("     because both sensitivities scale with lambda and the ratio does not.")
    print("     Two rows that differ only by a scale factor invert into noise.\n")


def interrogate(
    gratings: list[FiberBraggGrating], *, span_nm: float, points: int
) -> dict[str, float]:
    """Sweep a tunable laser across the array and return each peak by centroid.

    How a real interrogator works, and the reason it is a sweep here rather than
    a broadband source and an optical spectrum analyser: this engine carries
    broadband light as a noise bin with a *flat* power density, so a
    wavelength-resolved reflection off one would come back as a single scalar
    with no peak in it at all. A tunable laser has no such problem, and every
    point below is a real run of a real graph.

    **The array is one fibre.** The probe goes out through a circulator, and the
    gratings are in series because each passes what it does not reflect — which
    is the entire architecture: ten sensors, one fibre, one instrument.

    **What is metered is each grating's own reflected port**, and there is a
    reason rather than a convenience. On hardware all the reflections come back
    down that one fibre and the circulator hands the instrument their sum. Wiring
    that here means summing several bands that are all at the *probe's*
    wavelength, since a single-wavelength probe is what every grating is
    reflecting a copy of — and :class:`~maiman.components.Combiner` refuses
    exactly that, because co-located carriers interfere and have to be added as
    fields on a common grid rather than multiplexed. It is right to refuse. The
    quantity being measured is per grating anyway, and off resonance the others
    contribute nothing to it.
    """
    ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=8, sequence_length=32)
    graph = Graph(ctx)
    probe = graph.add(CWLaser(power=0.0, wavelength=BRAGG_NM, label="probe"))
    # One hop: out of port 1 and into the array at port 2. The return hop is what
    # a real instrument uses and what the drop configuration in
    # ``fbg_circulator.py`` wires; here each grating is metered where it sits.
    circ = graph.add(Circulator(driven=(1,), insertion_loss=0.0, label="circ"))
    for grating in gratings:
        graph.add(grating)
    meters = {
        grating.label: graph.add(PowerMeter(label=f"meter_{grating.label}")) for grating in gratings
    }

    graph.connect(probe, circ["in1"])
    graph.connect(circ["out2"], gratings[0]["in"])
    for upstream, downstream in pairwise(gratings):
        graph.connect(upstream["transmitted"], downstream["in"])
    for grating in gratings:
        graph.connect(grating["reflected"], meters[grating.label]["in"])
    # The last grating passes what nothing reflected; that is the end of the array.
    end = graph.add(PowerMeter(label="beyond"))
    graph.connect(gratings[-1]["transmitted"], end["in"])

    peaks: dict[str, float] = {}
    for grating in gratings:
        centre = grating.sensed_bragg_wavelength() * 1e9
        probes = np.linspace(centre - span_nm / 2, centre + span_nm / 2, points)
        result = sweep(graph, {(probe, "wavelength"): list(probes)})
        power = result.metric(meters[grating.label], lambda reading: reading.power_dbm)
        linear = 10.0 ** (power[:, 0] / 10.0)
        # Centroid of the top of the peak, not the largest sample: the peak sits
        # between grid points and an argmax can never beat the step. This is what
        # an interrogator reports, and it is why 121 points over 0.6 nm -- a 5 pm
        # grid -- recovers a shift to a fraction of a picometre.
        top = linear > 0.5 * linear.max()
        peaks[grating.label] = float((probes[top] * linear[top]).sum() / linear[top].sum())
    return peaks


def recovering_a_load() -> None:
    """A working grating and a strain-free reference, read down one fibre."""
    print("4. Interrogating an array, and recovering what was applied")
    applied_strain, applied_warming = 400.0, 15.0

    working = sensor(
        bragg_nm=1545.0, strain=applied_strain, warming=applied_warming, label="working"
    )
    # Loose in its tube: it feels the temperature and none of the load.
    reference = sensor(bragg_nm=1555.0, strain=0.0, warming=applied_warming, label="reference")

    peaks = interrogate([working, reference], span_nm=0.6, points=121)
    print(f"     applied: {applied_strain:.0f} ustrain and +{applied_warming:.0f} K")
    print(f"     {'grating':>10}  {'nominal':>11}  {'measured':>12}  {'shift':>10}")
    for grating in (working, reference):
        nominal = grating.si("bragg_wavelength") * 1e9
        measured = peaks[grating.label]
        print(
            f"     {grating.label:>10}  {nominal:10.4f}nm  {measured:11.5f}nm  "
            f"{(measured - nominal) * 1e3:9.2f}pm"
        )

    # The reference sees temperature alone, so its shift *is* the temperature.
    reference_shift = (peaks["reference"] - 1555.0) * 1e-9
    recovered_warming = reference_shift / reference.temperature_sensitivity()

    # Subtract that much drift from the working grating and what is left is load.
    working_shift = (peaks["working"] - 1545.0) * 1e-9
    thermal_part = recovered_warming * working.temperature_sensitivity()
    recovered_strain = (working_shift - thermal_part) / working.strain_sensitivity() * 1e6

    print()
    print(
        f"     recovered temperature  {recovered_warming:8.3f} K   "
        f"(error {recovered_warming - applied_warming:+.4f})"
    )
    print(
        f"     recovered strain       {recovered_strain:8.3f} ue  "
        f"(error {recovered_strain - applied_strain:+.4f})"
    )
    print("     the reference is the whole method: it is the second equation.\n")


def what_drift_costs_without_one() -> None:
    """The same array read as though the temperature were known to be zero."""
    print("5. The same measurement with no reference grating")
    print(f"     {'applied':>22}  {'strain read':>12}  {'error':>10}")
    for warming in (0.0, 1.0, 5.0, 15.0):
        working = sensor(bragg_nm=1545.0, strain=400.0, warming=warming, label="working")
        peaks = interrogate([working], span_nm=0.6, points=121)
        shift = (peaks["working"] - 1545.0) * 1e-9
        read = shift / working.strain_sensitivity() * 1e6
        print(f"     400 ue and +{warming:4.1f} K   {read:11.2f}ue  {read - 400.0:+9.2f}ue")
    print("     9.26 ustrain per kelvin, arriving exactly on schedule. An")
    print("     uncompensated fibre strain gauge is a thermometer with extra steps.\n")


def main() -> None:
    sensitivities()
    one_wavelength_two_unknowns()
    the_two_wavelength_trap()
    recovering_a_load()
    what_drift_costs_without_one()


if __name__ == "__main__":
    main()
