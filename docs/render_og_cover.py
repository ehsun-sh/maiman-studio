"""Render the repository's social preview card, from real engine output.

    python docs/render_og_cover.py

The image GitHub shows when the repository is linked anywhere. 1200x630 is the
size every platform crops from, and it is read at about a third of that in a
timeline — so there is room for one idea, a name, and nothing else.

**The idea is the constellation, and it is a real one.** The points on the right
come from running the coherent link the studio ships — 32 GBd through 80 km of
fiber, recovered — right here, at 320 bins instead of the interface's 128. Not an
illustration of a constellation: the constellation. A project whose first claim
is that every number on screen came from the engine cannot draw a decorative one
on its cover.

The run rather than the baked export because of resolution, not principle. The
export bins over the range that keeps a badly distorted constellation inside its
frame, which leaves a clean one occupying the middle half; blown up to a
400-pixel hero it is a handful of bins per cluster and interpolates into fog.
Measuring the same symbols more finely, over the range they actually occupy, is
what a bench analyser's zoom does.

Rendered through the same headless Chrome the interface screenshots use, at
twice the scale so the type holds up on a retina timeline.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from base64 import b64encode
from pathlib import Path

import numpy as np
from capture_screenshots import find_browser

DOCS = Path(__file__).parent
ROOT = DOCS.parent
MARK = ROOT / "assets" / "logo-mark-tight.png"
OUTPUT = ROOT / "assets" / "og-cover.png"

#: Bins across the constellation, and the window they cover. Far finer than the
#: interface's own diagram because this is drawn ten times larger than the dock
#: ever draws it, and tighter for the same reason.
#:
#: ``extent`` is in units of the largest constellation point's *magnitude*, not
#: of the axes — for 16-QAM the corner point is at 1.342, so 0.9 here is a frame
#: half-width of about 1.21 against outer points at 0.949. Reading it as an axis
#: limit put the reference crosses half a cluster away from the clusters, which
#: is why the edges below are taken from the histogram rather than computed.
BINS, EXTENT = 320.0, 0.9

#: Symbols to run, against the interface's 4096. Four times the data at four
#: times the cost of a run that takes under a second, and it is what puts a
#: bright core in each cluster instead of a scatter of single counts — the
#: density ramp needs something to be dense. The EVM printed under the plot is
#: measured on these, so the caption stays the run's own number.
SYMBOLS = 16384

#: The size every platform crops its preview from.
WIDTH, HEIGHT = 1200, 630

#: Rendered at 2x and left that way: GitHub downsamples, and the type survives it
#: far better than it survives being drawn at one pixel per point.
SCALE = 2

#: Long enough for the canvas to be drawn. It happens in a rAF callback, so a
#: shorter budget captures an empty card — the same trap the interface
#: screenshots fall into.
TIME_BUDGET_MS = 3000


def measure() -> dict[str, object]:
    """Run the shipped coherent link and return its constellation, finely binned.

    The graph is ``examples/export_ui_data.build()`` — the same one the interface
    draws and the front page quotes — with the diagram's resolution turned up.
    Nothing else about it is touched, so the EVM printed under the plot is the
    EVM of the link on the cover.
    """
    sys.path.insert(0, str(ROOT / "examples"))
    from export_ui_data import build

    from maiman.components import ConstellationDiagram

    graph = build(sequence_length=SYMBOLS)
    diagram = next(c for c in graph.components if isinstance(c, ConstellationDiagram))
    analyzer = next(c for c in graph.components if c.label == "vsa")
    results = graph.run(overrides={(diagram, "bins"): BINS, (diagram, "extent"): EXTENT})
    histogram = results[diagram]
    measurement = results[analyzer]
    return {
        "counts": np.asarray(histogram.counts).astype(int).tolist(),
        # The window the engine actually binned over, not the parameter that
        # asked for it. One is a number this file can be wrong about; the other
        # is what the histogram is.
        "edges": [
            float(np.asarray(histogram.inphase_edges)[0]),
            float(np.asarray(histogram.inphase_edges)[-1]),
        ],
        "reference": [[float(p.real), float(p.imag)] for p in np.asarray(histogram.reference)],
        "evm": measurement.evm * 100.0,
    }


def page() -> str:
    """The card, as one self-contained document."""
    constellation = measure()
    mark = b64encode(MARK.read_bytes()).decode("ascii")

    return f"""<title>Maiman Studio</title>
<style>
  /* The graphite ground the interface ships, so the card and the tool are
     recognisably the same object. */
  :root {{
    --ground: #0b0e13;
    --panel: #171c24;
    --ink: #e8edf5;
    --ink-2: #aab6c8;
    --ink-3: #7a869b;
    --rule: #2a323d;
    --optical: #22d3c5;
    --electrical: #f0a030;
    --binary: #7c86a0;
    --symbol: #9b7cf6;
    --metric: #e85d9b;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    width: {WIDTH}px; height: {HEIGHT}px; overflow: hidden;
    background: var(--ground);
    font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto,
                 "Helvetica Neue", Arial, sans-serif;
    color: var(--ink);
    display: grid; grid-template-columns: 1fr 520px;
  }}
  /* A single soft light from the upper left, so the flat ground has somewhere
     to fall away to rather than reading as a rectangle of paint. */
  body::before {{
    content: ""; position: absolute; inset: 0;
    background: radial-gradient(1100px 700px at 8% -10%, #1b2430 0%, transparent 62%);
  }}

  /* Vertically centred rather than pinned to the top. The first draft hung the
     name at the top and the facts at the bottom and left a third of the card
     empty between them, which at thumbnail size reads as a mistake. */
  .left {{
    position: relative; padding: 0 40px 0 78px;
    display: flex; flex-direction: column; justify-content: center; gap: 30px;
  }}

  .lockup {{ display: flex; align-items: center; gap: 18px; }}
  .lockup img {{ height: 66px; width: auto; display: block; margin-top: -3px; }}
  .name {{ font-size: 58px; font-weight: 620; letter-spacing: -0.018em; }}
  .name span {{ color: var(--ink-3); font-weight: 400; }}

  .tagline {{
    max-width: 560px;
    font-size: 25px; line-height: 1.4; color: var(--ink-2); font-weight: 380;
  }}
  .tagline b {{ color: var(--ink); font-weight: 560; }}

  /* Facts, not adjectives. Each of these is checked by the test suite, which is
     the only reason any of them is allowed on a cover. */
  /* One line each. Three of them, because at the size this is actually read a
     fourth is a smudge. Each is checked by the test suite, which is the only
     reason any of them is allowed on a cover. */
  .facts {{ display: flex; flex-direction: column; gap: 11px; font-size: 19px; }}
  .facts div {{ display: flex; align-items: baseline; gap: 12px; color: var(--ink-2); }}
  .facts b {{
    min-width: 108px; color: var(--ink); font-weight: 560; letter-spacing: -0.01em;
  }}
  .facts i {{ font-style: normal; color: var(--optical); font-size: 15px; }}

  .right {{ position: relative; display: grid; place-items: center; }}
  .plot {{ position: relative; }}
  .plot canvas {{ display: block; width: 408px; height: 408px; border-radius: 3px; }}
  /* The cyan bleeding past the frame, which is what a dense constellation
     actually looks like on a bright analyser and what stops the panel reading
     as a screenshot pasted on. */
  .plot::after {{
    content: ""; position: absolute; inset: -70px; border-radius: 50%;
    background: radial-gradient(closest-side, rgba(34,211,197,0.16), transparent 70%);
    pointer-events: none; z-index: -1;
  }}
  .caption {{
    position: absolute; left: 0; right: 0; bottom: -34px; text-align: center;
    font: 15px ui-monospace, "SF Mono", Menlo, Consolas, monospace;
    color: var(--ink-3); letter-spacing: 0.01em;
  }}

  /* The port identity, as the one rule on the card. Every wire colour in the
     tool is a wavelength; this is that palette with nothing else to say. */
  .spectrum {{
    position: absolute; left: 0; right: 0; bottom: 0; height: 5px;
    background: linear-gradient(90deg,
      var(--optical) 0%, var(--optical) 20%,
      var(--electrical) 20%, var(--electrical) 40%,
      var(--binary) 40%, var(--binary) 60%,
      var(--symbol) 60%, var(--symbol) 80%,
      var(--metric) 80%, var(--metric) 100%);
  }}
</style>

<div class="left">
  <div class="lockup">
    <img src="data:image/png;base64,{mark}" alt="">
    <div class="name">Maiman<span>&nbsp;Studio</span></div>
  </div>

  <p class="tagline">
    An open-source simulator for <b>optical communication links</b>
    and <b>photonic systems</b>.
  </p>

  <div class="facts">
    <div><b>Validated</b>every physics block against a closed form</div>
    <div><b>Coherent</b>BPSK to 256-QAM, dual polarization</div>
    <div><b>Photonic</b>S-matrix circuits, MMI, rings, PDK import</div>
  </div>
</div>

<div class="right">
  <div class="plot">
    <canvas id="con" width="900" height="900"></canvas>
    <div class="caption">
      16-QAM &middot; 32 GBd &middot; 80 km &middot; EVM {constellation["evm"]:.2f} %
    </div>
  </div>
</div>

<div class="spectrum"></div>

<script>
const DATA = {json.dumps(constellation)};

/* The window the histogram was binned over, straight from the engine. */
const [LOW, HIGH] = DATA.edges;

/* The same density ramp the interface draws its constellation with — ground to
   cyan to white — so distance from the page means how many symbols landed
   there, here as much as in the tool. */
function draw() {{
  const canvas = document.getElementById("con");
  const ctx = canvas.getContext("2d");
  const size = canvas.width;
  const counts = DATA.counts;
  const rows = counts.length, cols = counts[0].length;

  ctx.fillStyle = "#0d1219";
  ctx.fillRect(0, 0, size, size);

  const stops = [[13, 18, 25], [34, 211, 197], [236, 253, 250]];
  const KNEE = 0.55;
  const ramp = new Array(256);
  for (let i = 0; i < 256; i++) {{
    const t = i / 255;
    const [a, b, u] = t < KNEE ? [stops[0], stops[1], t / KNEE]
                               : [stops[1], stops[2], (t - KNEE) / (1 - KNEE)];
    ramp[i] = [
      Math.round(a[0] + (b[0] - a[0]) * u),
      Math.round(a[1] + (b[1] - a[1]) * u),
      Math.round(a[2] + (b[2] - a[2]) * u),
    ];
  }}

  let max = 0;
  for (const row of counts) for (const v of row) if (v > max) max = v;

  /* Painted one histogram bin per *pixel* and then scaled up interpolated,
     rather than one bin per square. A 96-bin histogram blown up to 900 px is
     nine-pixel blocks, and blocks read as a low-resolution image rather than as
     a dense measurement — which is the opposite of the impression a
     constellation is meant to give. The engine's bins are unchanged; only the
     way they are drawn is. */
  const bins = document.createElement("canvas");
  bins.width = cols; bins.height = rows;
  const cell = bins.getContext("2d");
  const image = cell.createImageData(cols, rows);
  for (let r = 0; r < rows; r++) {{
    for (let k = 0; k < cols; k++) {{
      const level = Math.round(Math.pow(counts[r][k] / max, 0.5) * 255);
      const [red, green, blue] = ramp[level];
      // Rows run bottom-up in the histogram and top-down in an image.
      const at = ((rows - 1 - r) * cols + k) * 4;
      image.data[at] = red;
      image.data[at + 1] = green;
      image.data[at + 2] = blue;
      image.data[at + 3] = 255;
    }}
  }}
  cell.putImageData(image, 0, 0);

  /* Every bin drawn, because the measurement was already taken over the range
     the symbols occupy. Smoothed on the way up: at 320 bins the interpolation
     is under three pixels a bin and softens the edges of a cluster rather than
     inventing its shape. */
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(bins, 0, 0, size, size);

  // Axes through the origin: a constellation is read relative to zero.
  const X = (v) => ((v - LOW) / (HIGH - LOW)) * size;
  const Y = (v) => size - ((v - LOW) / (HIGH - LOW)) * size;
  ctx.strokeStyle = "rgba(122,134,155,0.35)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(0, Y(0)); ctx.lineTo(size, Y(0));
  ctx.moveTo(X(0), 0); ctx.lineTo(X(0), size);
  ctx.stroke();

  ctx.strokeStyle = "rgba(122,134,155,0.5)";
  ctx.lineWidth = 2;
  ctx.strokeRect(1, 1, size - 2, size - 2);

  // Ideal points as crosses, not dots, so a cluster's own centre stays visible
  // underneath — the same reason the interface marks them this way.
  ctx.strokeStyle = "rgba(155,124,246,0.85)";
  ctx.lineWidth = 2.5;
  const arm = 13;
  for (const [re, im] of DATA.reference) {{
    const x = X(re), y = Y(im);
    ctx.beginPath();
    ctx.moveTo(x - arm, y); ctx.lineTo(x + arm, y);
    ctx.moveTo(x, y - arm); ctx.lineTo(x, y + arm);
    ctx.stroke();
  }}
}}
draw();
</script>
"""


def main() -> None:
    if not MARK.is_file():
        raise SystemExit(f"{MARK} not found")

    browser = find_browser()
    with tempfile.TemporaryDirectory() as work:
        source = Path(work) / "og-cover.html"
        source.write_text(page(), encoding="utf-8")
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                browser,
                "--headless",
                "--disable-gpu",
                "--hide-scrollbars",
                f"--force-device-scale-factor={SCALE}",
                f"--window-size={WIDTH},{HEIGHT}",
                f"--virtual-time-budget={TIME_BUDGET_MS}",
                f"--screenshot={OUTPUT}",
                source.resolve().as_uri(),
            ],
            check=True,
            capture_output=True,
        )
    if not OUTPUT.is_file():
        raise SystemExit(f"{browser} reported success but wrote no file to {OUTPUT}")
    print(f"{OUTPUT.relative_to(ROOT)}  {OUTPUT.stat().st_size / 1024:.0f} kB")


if __name__ == "__main__":
    main()
