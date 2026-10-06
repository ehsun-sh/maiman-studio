"""A process design kit, and what changes when a design stops being nominal.

The last of the three things Phase 4's roadmap row named. A PDK here is not a
component library and adds no devices: it is the half of a real design this
engine cannot derive. :mod:`maiman.photonics` knows what a directional coupler
*is*. It has no way to know that this process makes a nominal 3 dB coupler that
measures 0.48 at 1550 nm and drifts nearly a fifth of its ratio across the C
band, or that the nitride guide beside the silicon one is twenty times quieter.
Those numbers come off a wafer.

The file is JSON and reading it executes nothing — the same rule the `.maiman`
project format follows, for the same reason.

Three things are worth reading off the output.

**A fit is not a constant.** Every value in a kit may be a polynomial in
``(lambda - lambda_ref)``, which is the form a foundry quotes one in, and it is
evaluated when the component is built. That is the whole of the wavelength
dependence and it is resolved at build time: the device models here are
frequency-flat by construction, so what a kit buys is the right constant for the
band you are working in, chosen by the file rather than guessed.

**A fit has a window, and running outside it is refused.** This is the part
worth the code. A first-order C-band fit of that coupler, extrapolated to
1310 nm, returns *minus* 0.46 — a negative power fraction, out of arithmetic
that never complained once. `valid_wavelengths` turns that into a sentence.

**Nominal is not what you build.** The last table runs the same 1 x 4 splitter
twice: once as the textbook says, and once as the kit says. The textbook one
divides by four exactly. The kit's has excess loss and 0.35 dB of imbalance, so
its four arms are not equal — which is what a real splitter tree does, and what
decides whether the design margin you calculated survives.
"""

from __future__ import annotations

import math
from pathlib import Path

from maiman import Component, Graph, SimulationContext
from maiman.components import MMI, CWLaser, PowerMeter
from maiman.pdk import PDKError, load_pdk

KIT = Path(__file__).parent / "silicon_220nm.pdk.json"


def main() -> None:
    pdk = load_pdk(KIT)
    low, high = pdk.valid_wavelengths or (0.0, 0.0)
    print(f"{pdk}")
    print(f"  fitted over {low:.0f}-{high:.0f} nm")
    print(f"  {pdk.description[:96]}...\n")

    print("Cross-sections, at the reference wavelength")
    print(f"  {'':10}  {'n_eff':>8}  {'n_group':>8}  {'loss':>12}")
    print("  " + "-" * 46)
    for name, section in pdk.cross_sections.items():
        values = section.at(pdk.reference_wavelength, pdk.reference_wavelength)
        print(
            f"  {name:10}  {values['n_eff']:8.4f}  {values['n_group']:8.4f}  "
            f"{values['propagation_loss']:8.3f} dB/cm"
        )
    print("  one definition each, reaching every device drawn in it.\n")

    print("The nominal 3 dB coupler is not 3 dB, and is not one number")
    print(f"  {'wavelength':>12}  {'coupling':>9}  {'as dB split':>22}")
    print("  " + "-" * 50)
    for wavelength in (1500.0, 1525.0, 1550.0, 1575.0, 1600.0):
        coupling = pdk.parameters("dc_3db", wavelength=wavelength)["coupling"]
        through = 10.0 * math.log10(1.0 - coupling)
        cross = 10.0 * math.log10(coupling)
        print(f"  {wavelength:9.0f} nm  {coupling:9.4f}  {through:8.2f} / {cross:.2f} dB")
    print("  which is the entire reason the MMI beside it exists in the kit.\n")

    print("Outside the window it was fitted in, the kit refuses")
    try:
        pdk.parameters("dc_3db", wavelength=1310.0)
    except PDKError as error:
        for line in str(error).split(" — "):
            print(f"  {line}")
    print()

    print("Nominal against the kit: a 1 x 4 splitter, twice")
    ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=8, sequence_length=64, seed=1)

    def split(splitter: Component, what: str) -> None:
        graph = Graph(ctx)
        laser = graph.add(CWLaser(power=0.0, wavelength=1550.0, label="tx"))
        graph.add(splitter)
        meters = [graph.add(PowerMeter(label=f"pm{k}")) for k in range(4)]
        graph.connect(laser, splitter["in1"])
        for index, meter in enumerate(meters):
            graph.connect(splitter[f"out{index + 1}"], meter["in"])
        results = graph.run()
        levels = [results[m].power_dbm for m in meters]
        total = 10.0 * math.log10(sum(10.0 ** (level / 10.0) for level in levels))
        arms = "  ".join(f"{level:7.3f}" for level in levels)
        spread = max(levels) - min(levels)
        print(f"  {what:14}  {arms}   total {total:+.3f} dB   spread {spread:.3f} dB")

    print(f"  {'':14}  {'out1':>7}  {'out2':>7}  {'out3':>7}  {'out4':>7}")
    print("  " + "-" * 78)
    split(MMI(4, excess_loss=0.0, imbalance=0.0, label="ideal"), "textbook")
    split(pdk.make("mmi_1x4", label="kit"), "silicon-220nm")
    print("  the textbook one divides by four exactly. The other is the device.")


if __name__ == "__main__":
    main()
