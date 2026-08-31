"""Regenerate the README's interface screenshots from the mockup.

Run after any change to ``docs/ui-mockup.html``, so the front page shows what the
current build renders rather than an older one. The README is the first thing a
visitor sees; a screenshot that has drifted from the build is worse than none,
because it is wrong rather than merely missing.

    python docs/capture_screenshots.py

Both grounds are captured, because the interface ships both and the README says
so. Graphite is taken from a throwaway copy whose theme bootstrap is forced —
the page otherwise defaults to paper and headless Chrome has no session storage
to read a choice out of.

Chrome is used directly rather than through Playwright or Selenium: it is already
on every machine that would run this, and a screenshot needs none of what a
driver adds. ``--virtual-time-budget`` matters — the plots are drawn on a canvas
from a ``requestAnimationFrame`` callback, so a capture taken too early gets an
empty dock.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

DOCS = Path(__file__).parent
MOCKUP = DOCS / "ui-mockup.html"
OUTPUT = DOCS / "images"

#: The viewport the design is specified at. DESIGN.md's layout numbers and every
#: contrast audit are quoted at this width; capturing at another one would show a
#: layout nothing else in the project describes.
WIDTH, HEIGHT = 1440, 900

#: Long enough for the canvas plots to be drawn. They are not on the critical
#: rendering path, so a shorter budget yields a blank results dock.
TIME_BUDGET_MS = 4000

THEME_DEFAULT = 'sessionStorage.getItem("maiman-theme") || "light"'

CANDIDATES = (
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "google-chrome",
    "chromium",
)


def find_browser() -> str:
    for candidate in CANDIDATES:
        if Path(candidate).is_file():
            return candidate
        found = shutil.which(candidate)
        if found:
            return found
    raise SystemExit(
        "no Chrome or Edge found; install one, or add its path to CANDIDATES in this file"
    )


def capture(browser: str, page: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            browser,
            "--headless",
            "--disable-gpu",
            "--hide-scrollbars",
            "--force-device-scale-factor=1",
            f"--window-size={WIDTH},{HEIGHT}",
            f"--virtual-time-budget={TIME_BUDGET_MS}",
            f"--screenshot={destination}",
            page.resolve().as_uri(),
        ],
        check=True,
        capture_output=True,
    )
    if not destination.is_file():
        raise SystemExit(f"{browser} reported success but wrote no file to {destination}")


def main() -> None:
    if not MOCKUP.is_file():
        raise SystemExit(f"{MOCKUP} not found")
    browser = find_browser()
    source = MOCKUP.read_text(encoding="utf-8")
    if THEME_DEFAULT not in source:
        raise SystemExit(
            "the mockup's theme bootstrap has changed; update THEME_DEFAULT to match, "
            "or the graphite capture will silently be a second paper one"
        )

    capture(browser, MOCKUP, OUTPUT / "studio-paper.png")

    with tempfile.TemporaryDirectory() as work:
        forced = Path(work) / "graphite.html"
        forced.write_text(source.replace(THEME_DEFAULT, '"dark"'), encoding="utf-8")
        capture(browser, forced, OUTPUT / "studio-graphite.png")

    for name in ("studio-paper.png", "studio-graphite.png"):
        path = OUTPUT / name
        print(f"{path.relative_to(DOCS.parent)}  {path.stat().st_size / 1024:.0f} kB")


if __name__ == "__main__":
    sys.exit(main())
