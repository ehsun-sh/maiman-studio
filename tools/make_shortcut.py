"""Create a "Maiman Studio" shortcut, with the Maiman icon, in the repository root.

Run it once with the Python that has Maiman's dependencies:

    python tools/make_shortcut.py

Double-clicking the shortcut then runs tools/maiman_app.py with that same
Python, which opens the studio in the browser. A shortcut has to hold absolute
paths, so it is made on your machine rather than kept in git; run this again
if you move the checkout or change Python.

  Windows  Maiman Studio.lnk
  macOS    Maiman Studio.app
  Linux    Maiman Studio.desktop
"""

from __future__ import annotations

import os
import plistlib
import shlex
import shutil
import stat
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LAUNCHER = ROOT / "tools" / "maiman_app.py"
ASSETS = ROOT / "assets"
NAME = "Maiman Studio"


def _windows() -> Path:
    exe = Path(sys.executable)
    # pythonw has no console window; the launcher shows its own small window.
    pythonw = exe.with_name("pythonw.exe")
    target = pythonw if pythonw.exists() else exe
    link = ROOT / f"{NAME}.lnk"

    def ps(value: object) -> str:
        return "'" + str(value).replace("'", "''") + "'"

    script = "; ".join(
        [
            "$s = (New-Object -ComObject WScript.Shell).CreateShortcut(" + ps(link) + ")",
            "$s.TargetPath = " + ps(target),
            "$s.Arguments = " + ps(f'"{LAUNCHER}"'),
            "$s.WorkingDirectory = " + ps(ROOT),
            "$s.IconLocation = " + ps(f"{ASSETS / 'icon.ico'},0"),
            "$s.Description = 'Open Maiman Studio'",
            "$s.Save()",
        ]
    )
    subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
        check=True,
    )
    return link


def _macos() -> Path:
    app = ROOT / f"{NAME}.app"
    if app.exists():
        shutil.rmtree(app)
    macos = app / "Contents" / "MacOS"
    resources = app / "Contents" / "Resources"
    macos.mkdir(parents=True)
    resources.mkdir()
    shutil.copy(ASSETS / "icon.icns", resources / "icon.icns")
    with open(app / "Contents" / "Info.plist", "wb") as f:
        plistlib.dump(
            {
                "CFBundleName": NAME,
                "CFBundleDisplayName": NAME,
                "CFBundleIdentifier": "io.github.ehsun-sh.maiman-studio",
                "CFBundleExecutable": "maiman-studio",
                "CFBundleIconFile": "icon",
                "CFBundlePackageType": "APPL",
                "CFBundleShortVersionString": "1.0",
            },
            f,
        )
    run = macos / "maiman-studio"
    run.write_text(f"#!/bin/sh\nexec {shlex.quote(sys.executable)} {shlex.quote(str(LAUNCHER))}\n")
    run.chmod(run.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    # Finder caches icons; touching the bundle makes it look again.
    os.utime(app)
    return app


def _linux() -> Path:
    entry = ROOT / f"{NAME}.desktop"

    def q(value: object) -> str:
        return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'

    entry.write_text(
        "[Desktop Entry]\n"
        "Type=Application\n"
        f"Name={NAME}\n"
        "Comment=Open Maiman Studio\n"
        f"Exec={q(sys.executable)} {q(LAUNCHER)}\n"
        f"Path={ROOT}\n"
        f"Icon={ASSETS / 'icon-512.png'}\n"
        "Terminal=false\n"
        "Categories=Science;Education;\n"
    )
    entry.chmod(entry.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    # GNOME will not launch a .desktop file it has not been told to trust.
    if shutil.which("gio"):
        subprocess.run(
            ["gio", "set", str(entry), "metadata::trusted", "true"],
            check=False,
            capture_output=True,
        )
    return entry


def main() -> int:
    if sys.platform == "win32":
        made = _windows()
    elif sys.platform == "darwin":
        made = _macos()
    else:
        made = _linux()
    print(f"Created {made}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
