"""What has to be true of the distribution, not just of the source tree.

The engine was importable from a wheel from the first day. The interface was
not: the studio page sat in ``docs/``, addressed from the server as three
directories above itself, which resolves in a checkout and nowhere else. So
``pip install maiman`` produced a server that answered every API route and
returned 404 for the page it exists to serve, and nothing failed — the source
tree it was tested from always had the file.

That is the shape of every packaging bug: the thing under test is the artefact,
and every test was reading the workspace. These read the artefact.
"""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

import maiman
from maiman import server

ROOT = Path(__file__).resolve().parent.parent

#: Where the page must be found inside an installed distribution. Written out
#: rather than derived, so that moving the file has to change this line too.
PACKAGED = "maiman/studio/index.html"


def test_the_studio_page_lives_inside_the_package() -> None:
    """Not next to it, not above it: under the directory that gets installed.

    This is the cheap half of the check and the one that runs everywhere. A page
    outside the package can still be served from a checkout, so nothing else in
    the suite would notice it moving back out.
    """
    package = Path(maiman.__file__).resolve().parent
    studio = Path(str(server.STUDIO)).resolve()
    assert studio.is_file(), f"{studio} is missing"
    assert studio.is_relative_to(package), (
        f"the studio page is at {studio}, outside the package at {package}; "
        f"it will not be installed with it"
    )


@pytest.fixture(scope="session")
def wheel(tmp_path_factory: pytest.TempPathFactory) -> zipfile.ZipFile:
    """A wheel, built for real, once for the session.

    Assembled without build isolation, which is why hatchling is a dev
    dependency: a test that needs the network to prove a packaging claim is a
    test that gets skipped on the day it matters.
    """
    pytest.importorskip("build", reason="the wheel cannot be built without it")
    pytest.importorskip("hatchling", reason="isolation is off, so the backend must be present")

    out = tmp_path_factory.mktemp("wheel")
    completed = subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--no-isolation", "--outdir", str(out)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout

    built = list(out.glob("*.whl"))
    assert len(built) == 1, f"expected one wheel, got {[w.name for w in built]}"
    return zipfile.ZipFile(built[0])


def test_the_wheel_contains_the_studio_page(wheel: zipfile.ZipFile) -> None:
    """Built for real, then opened and read."""
    names = set(wheel.namelist())
    assert PACKAGED in names, (
        f"the wheel has no {PACKAGED}. The server will 404 the interface on every "
        f"installed copy. Wheel contains: {sorted(n for n in names if 'studio' in n)}"
    )
    packaged = wheel.read(PACKAGED)

    assert packaged == Path(str(server.STUDIO)).read_bytes(), (
        "the page in the wheel is not the page in the source tree"
    )
    assert b'id="maiman-data"' in packaged, "the page shipped without its baked-in data"


def test_the_wheel_declares_the_maiman_command(wheel: zipfile.ZipFile) -> None:
    """``maiman serve`` has to exist after an install, or the docs are wrong.

    Read out of the built wheel rather than out of pyproject.toml, because what
    is declared and what is recorded in the metadata are two different files and
    only one of them is installed.
    """
    entry_points = next(n for n in wheel.namelist() if n.endswith("entry_points.txt"))
    text = wheel.read(entry_points).decode("utf-8")

    assert "[console_scripts]" in text, text
    assert "maiman = maiman.cli:main" in text, text
