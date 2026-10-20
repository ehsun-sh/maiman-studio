"""Render the repository's social preview card, from real engine output.

    python docs/render_og_cover.py

The image GitHub, the site and every chat app show when Maiman Studio is linked.
1200x630 is the size every platform crops from, and it is read at about a third
of that in a timeline — so there is room for a name, one line, and the two
instruments people recognise an optical link by.

**Both instruments are real.** The eye on the right is the direct-detection link
the studio ships — 10 Gb/s on-off keying through 20 km of fiber — and the
constellation is the coherent one, 16-QAM at 32 GBd through 80 km, recovered.
Both are run right here, finely binned, and the Q and EVM under them are the
runs' own numbers. A project whose first claim is that every number on screen
came from the engine cannot draw decorative ones on its cover.

The runs rather than the baked exports because of resolution, not principle. The
exports bin coarsely, for a dock a few hundred pixels across; blown up to a hero
they are a handful of bins per feature and interpolate into fog. Measuring the
same signals more finely is what a bench instrument's zoom does.

The counts on the card are counted, not typed: the components from the engine's
own manifests, the guided examples from the projects the Examples menu opens,
the version from the package. A release that adds a block updates the card the
next time this runs.

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

#: The projects the Examples menu opens, one ``.maiman`` file each.
EXAMPLES = ROOT / "examples" / "maiman"

#: Bins across the constellation, and the window they cover. Far finer than the
#: interface's own diagram because this is drawn several times larger than the
#: dock ever draws it, and tighter for the same reason.
#:
#: ``extent`` is in units of the largest constellation point's *magnitude*, not
#: of the axes — for 16-QAM the corner point is at 1.342, so 0.9 here is a frame
#: half-width of about 1.21 against outer points at 0.949. Reading it as an axis
#: limit put the reference crosses half a cluster away from the clusters, which
#: is why the edges below are taken from the histogram rather than computed.
BINS, EXTENT = 240.0, 0.9

#: Symbols to run, against the interface's 4096. Four times the data at four
#: times the cost of a run that takes under a second, and it is what puts a
#: bright core in each cluster instead of a scatter of single counts — the
#: density ramp needs something to be dense. The EVM printed under the plot is
#: measured on these, so the caption stays the run's own number.
SYMBOLS = 16384

#: The eye's resolution, against the shipped project's 64 x 72. Fine enough that
#: each trace is a line rather than a staircase at the size the card draws it.
EYE_TIME_BINS, EYE_AMPLITUDE_BINS = 256.0, 200.0

#: The size every platform crops its preview from.
WIDTH, HEIGHT = 1200, 630

#: Rendered at 2x and left that way: GitHub downsamples, and the type survives it
#: far better than it survives being drawn at one pixel per point.
SCALE = 2

#: Long enough for the canvases to be drawn. It happens in a rAF callback, so a
#: shorter budget captures an empty card — the same trap the interface
#: screenshots fall into.
TIME_BUDGET_MS = 3000


def measure() -> dict[str, object]:
    """Run the shipped coherent link and return its constellation, finely binned.

    The graph is ``examples/python/export_ui_data.build()`` — the same one the interface
    draws and the front page quotes — with the diagram's resolution turned up.
    Nothing else about it is touched, so the EVM printed under the plot is the
    EVM of the link on the cover.
    """
    sys.path.insert(0, str(ROOT / "examples" / "python"))
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


def measure_eye() -> dict[str, object]:
    """Run the shipped direct-detection link and return its eye, finely binned.

    The graph is ``export_ui_data.ook_link()``, the one behind the eye the
    studio opens, with only the eye's resolution turned up — so the Q printed
    under it is that link's Q.
    """
    sys.path.insert(0, str(ROOT / "examples" / "python"))
    from export_ui_data import of_type, ook_link

    from maiman.components import BERAnalyzer, EyeDiagram

    graph = ook_link()
    eye = of_type(graph, EyeDiagram)
    analyzer = of_type(graph, BERAnalyzer)
    results = graph.run(
        overrides={(eye, "time_bins"): EYE_TIME_BINS, (eye, "amplitude_bins"): EYE_AMPLITUDE_BINS}
    )
    return {
        "counts": np.asarray(results[eye].counts).astype(int).tolist(),
        "q": float(results[analyzer].q_factor),
        "rate": graph.ctx.bit_rate / 1e9,
    }


def facts() -> dict[str, object]:
    """What the card counts: the version, the blocks, the guided examples."""
    import maiman
    from maiman import manifests

    return {
        "version": maiman.__version__,
        "components": len(manifests()),
        "examples": len(list(EXAMPLES.glob("*.maiman"))),
    }


def page() -> str:
    """The card, as one self-contained document."""
    constellation = measure()
    eye = measure_eye()
    counted = facts()
    mark = b64encode(MARK.read_bytes()).decode("ascii")

    return f"""<title>Maiman Studio</title>
<style>
  /* The graphite ground the interface ships, so the card and the tool are
     recognisably the same object. */
  :root {{
    --ground: #0b0e13;
    --panel: #0d1219;
    --ink: #eef2f8;
    --ink-2: #aab6c8;
    --ink-3: #7a869b;
    --rule: #2a323d;
    --optical: #22d3c5;
    --electrical: #f0a030;
    --binary: #7c86a0;
    --symbol: #9b7cf6;
    --metric: #e85d9b;
    --sans: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto,
            "Helvetica Neue", Arial, sans-serif;
    --mono: "JetBrains Mono", ui-monospace, "SF Mono", Menlo, Consolas, monospace;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    position: relative;
    width: {WIDTH}px; height: {HEIGHT}px; overflow: hidden;
    background: var(--ground);
    font-family: var(--sans);
    color: var(--ink);
    display: grid; grid-template-columns: 1fr 548px;
  }}

  /* Light from two corners: cool behind the instruments, where the signal is,
     and a faint warm one low on the left, the laser's end of the spectrum. */
  body::before {{
    content: ""; position: absolute; inset: 0;
    background:
      radial-gradient(760px 560px at 86% 30%, rgba(34,211,197,0.13) 0%, transparent 62%),
      radial-gradient(620px 420px at 0% 110%, rgba(232,93,155,0.12) 0%, transparent 65%),
      radial-gradient(900px 600px at 10% -20%, #1b2430 0%, transparent 60%);
  }}
  /* The editor's dot grid, fading out toward the type, so the right half reads
     as a canvas the instruments sit on. */
  body::after {{
    content: ""; position: absolute; inset: 0;
    background-image: radial-gradient(rgba(170,182,200,0.16) 1px, transparent 1.4px);
    background-size: 22px 22px;
    -webkit-mask-image: linear-gradient(90deg, transparent 30%, #000 75%);
    pointer-events: none;
  }}

  .left {{
    position: relative; z-index: 1; padding: 0 0 0 72px;
    display: flex; flex-direction: column; justify-content: center;
  }}

  .lockup {{ display: flex; align-items: center; gap: 14px; }}
  .lockup img {{ height: 50px; width: auto; display: block; margin-top: -2px; }}
  .name {{ font-size: 40px; font-weight: 600; letter-spacing: -0.02em; }}
  .name span {{ color: var(--ink-3); font-weight: 400; }}

  /* The promise in two short lines, the second in the mark's own spectrum. */
  h1 {{
    margin-top: 34px;
    font-size: 60px; line-height: 1.04; font-weight: 800; letter-spacing: -0.035em;
  }}
  h1 em {{
    font-style: normal; display: inline-block; padding-bottom: 4px;
    background: linear-gradient(90deg, #468ec7 0%, #22d3c5 30%, #b2bb63 52%,
                                #f8981c 72%, #ef4d67 100%);
    -webkit-background-clip: text; background-clip: text; color: transparent;
  }}

  .tagline {{
    margin-top: 20px; max-width: 520px;
    font-size: 20px; line-height: 1.42; color: var(--ink-2); font-weight: 400;
  }}
  .tagline b {{ color: var(--ink); font-weight: 600; }}

  /* Counted, not typed: see facts(). */
  .chips {{ margin-top: 28px; display: flex; gap: 10px; flex-wrap: wrap; }}
  .chips div {{
    padding: 8px 15px; border-radius: 999px;
    background: rgba(255,255,255,0.045); border: 1px solid rgba(255,255,255,0.08);
    font-size: 17px; font-weight: 500; color: var(--ink);
  }}
  .chips b {{ color: var(--optical); font-weight: 700; }}

  .foot {{
    position: absolute; left: 72px; bottom: 34px; z-index: 1;
    font: 15px var(--mono); color: var(--ink-3); letter-spacing: 0.01em;
  }}
  .foot b {{ color: var(--ink-2); font-weight: 500; }}

  /* Two instruments as the studio's dock draws them: dark panels with a tab
     strip naming what is on them, the constellation over the eye's shoulder. */
  .right {{ position: relative; z-index: 1; }}
  .panel {{
    position: absolute; background: var(--panel); border-radius: 12px;
    border: 1px solid rgba(122,134,155,0.32);
    box-shadow: 0 24px 60px rgba(0,0,0,0.55), 0 0 0 1px rgba(0,0,0,0.4);
    overflow: hidden; display: flex; flex-direction: column;
  }}
  .panel canvas {{ display: block; width: 100%; flex: 1; min-height: 0; }}
  .head {{
    display: flex; align-items: baseline; gap: 8px; padding: 9px 14px 8px;
    border-bottom: 1px solid rgba(122,134,155,0.22); background: rgba(255,255,255,0.025);
    font: 600 14px var(--sans); color: var(--ink); white-space: nowrap;
  }}
  .head i {{ font: 400 12px var(--mono); font-style: normal; color: var(--ink-3); }}
  .eye {{ left: 0; top: 232px; width: 476px; height: 300px; }}
  .con {{ left: 290px; top: 40px; width: 230px; height: 264px; }}
  /* A teal halo behind the constellation, which is what a dense one looks like
     on a bright analyser and what stops it reading as a pasted screenshot. */
  .halo {{
    position: absolute; left: 235px; top: 0; width: 340px; height: 340px;
    border-radius: 50%;
    background: radial-gradient(closest-side, rgba(34,211,197,0.20), transparent 72%);
  }}

  /* The port identity, as the one rule on the card. Every wire colour in the
     tool is a wavelength; this is that palette with nothing else to say. */
  .spectrum {{
    position: absolute; left: 0; right: 0; bottom: 0; height: 5px; z-index: 2;
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

  <h1>Build optical links.<br><em>See every signal.</em></h1>

  <p class="tagline">
    An open-source simulator for <b>optical communication links</b> and
    <b>photonic systems</b>. Every number on screen comes from the engine.
  </p>

  <div class="chips">
    <div><b>{counted["components"]}</b> components</div>
    <div><b>{counted["examples"]}</b> guided examples</div>
    <div><b>Desktop</b> &amp; Python</div>
  </div>
</div>

<div class="foot">
  <b>v{counted["version"]}</b> &middot; pip install maiman &middot; maimanstudio.com
</div>

<div class="right">
  <div class="panel eye">
    <div class="head">Eye diagram<i>{eye["rate"]:.0f} Gb/s &middot; Q {eye["q"]:.1f}</i></div>
    <canvas id="eye" width="952" height="530"></canvas>
  </div>
  <div class="halo"></div>
  <div class="panel con">
    <div class="head">16-QAM<i>EVM {constellation["evm"]:.1f} %</i></div>
    <canvas id="con" width="460" height="460"></canvas>
  </div>
</div>

<div class="spectrum"></div>

<script>
const CON = {json.dumps(constellation)};
const EYE = {json.dumps(eye)};

/* The same density ramp the interface draws its instruments with — ground to
   cyan to white — so brightness means how many samples landed there, here as
   much as in the tool. */
const RAMP = (() => {{
  const stops = [[13, 18, 25], [34, 211, 197], [236, 253, 250]];
  const KNEE = 0.55;
  const ramp = new Array(256);
  for (let i = 0; i < 256; i++) {{
    const t = i / 255;
    const [a, b, u] = t < KNEE ? [stops[0], stops[1], t / KNEE]
                               : [stops[1], stops[2], (t - KNEE) / (1 - KNEE)];
    ramp[i] = [0, 1, 2].map((c) => Math.round(a[c] + (b[c] - a[c]) * u));
  }}
  return ramp;
}})();

/* Painted one histogram bin per *pixel* and then scaled up interpolated,
   rather than one bin per square: blocks read as a low-resolution image rather
   than as a dense measurement. The engine's bins are unchanged; only the way
   they are drawn is. */
function paint(ctx, counts, w, h, gamma) {{
  const rows = counts.length, cols = counts[0].length;
  let max = 0;
  for (const row of counts) for (const v of row) if (v > max) max = v;
  const bins = document.createElement("canvas");
  bins.width = cols; bins.height = rows;
  const cell = bins.getContext("2d");
  const image = cell.createImageData(cols, rows);
  for (let r = 0; r < rows; r++) {{
    for (let k = 0; k < cols; k++) {{
      const [red, green, blue] = RAMP[Math.round(Math.pow(counts[r][k] / max, gamma) * 255)];
      // Rows run bottom-up in a histogram and top-down in an image.
      const at = ((rows - 1 - r) * cols + k) * 4;
      image.data[at] = red;
      image.data[at + 1] = green;
      image.data[at + 2] = blue;
      image.data[at + 3] = 255;
    }}
  }}
  cell.putImageData(image, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(bins, 0, 0, w, h);
}}

function drawEye() {{
  const canvas = document.getElementById("eye");
  const ctx = canvas.getContext("2d");
  const w = canvas.width, h = canvas.height;
  ctx.fillStyle = "#0d1219";
  ctx.fillRect(0, 0, w, h);
  // A little headroom above and below, so the rails do not touch the frame.
  const pad = 0.1 * h;
  ctx.save();
  ctx.translate(0, pad);
  paint(ctx, EYE.counts, w, h - 2 * pad, 0.42);
  // The phosphor bloom a scope's traces have: the same picture, blurred and
  // added back over itself. It brightens what is there and adds nothing new.
  const trace = document.createElement("canvas");
  trace.width = w; trace.height = h - 2 * pad;
  trace.getContext("2d").drawImage(canvas, 0, pad, w, h - 2 * pad, 0, 0, w, h - 2 * pad);
  ctx.globalCompositeOperation = "lighter";
  ctx.globalAlpha = 0.55;
  ctx.filter = "blur(7px)";
  ctx.drawImage(trace, 0, 0);
  ctx.restore();
  // The graticule a scope draws: centre lines, and one division per half symbol.
  ctx.strokeStyle = "rgba(122,134,155,0.22)";
  ctx.lineWidth = 2;
  ctx.setLineDash([6, 8]);
  ctx.beginPath();
  for (const x of [0.25, 0.5, 0.75]) {{ ctx.moveTo(x * w, 0); ctx.lineTo(x * w, h); }}
  ctx.moveTo(0, h / 2); ctx.lineTo(w, h / 2);
  ctx.stroke();
}}

function drawConstellation() {{
  const canvas = document.getElementById("con");
  const ctx = canvas.getContext("2d");
  const size = canvas.width;
  ctx.fillStyle = "#0d1219";
  ctx.fillRect(0, 0, size, size);
  paint(ctx, CON.counts, size, size, 0.5);

  /* The window the histogram was binned over, straight from the engine. */
  const [LOW, HIGH] = CON.edges;
  const X = (v) => ((v - LOW) / (HIGH - LOW)) * size;
  const Y = (v) => size - ((v - LOW) / (HIGH - LOW)) * size;

  // Axes through the origin: a constellation is read relative to zero.
  ctx.strokeStyle = "rgba(122,134,155,0.35)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(0, Y(0)); ctx.lineTo(size, Y(0));
  ctx.moveTo(X(0), 0); ctx.lineTo(X(0), size);
  ctx.stroke();

  // Ideal points as crosses, not dots, so a cluster's own centre stays visible
  // underneath — the same reason the interface marks them this way.
  ctx.strokeStyle = "rgba(155,124,246,0.85)";
  ctx.lineWidth = 2;
  const arm = 8;
  for (const [re, im] of CON.reference) {{
    const x = X(re), y = Y(im);
    ctx.beginPath();
    ctx.moveTo(x - arm, y); ctx.lineTo(x + arm, y);
    ctx.moveTo(x, y - arm); ctx.lineTo(x, y + arm);
    ctx.stroke();
  }}
}}

drawEye();
drawConstellation();
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
