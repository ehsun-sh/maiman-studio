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

import re
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


def test_the_test_count_the_product_brief_claims_is_a_floor_and_is_met(
    request: pytest.FixtureRequest,
) -> None:
    """PRODUCT.md says "more than N tests". It must be true, and it must stay true.

    The number drifted twice in one afternoon — written as an exact count, it is
    wrong the moment anybody adds a test, and nothing notices. So it is a floor
    now: only a claim that gets *smaller* can falsify it, which happens when
    tests are deleted and is exactly when somebody should be told.

    Skipped on a partial run, because ``testscollected`` counts what this
    session collected and running one file would fail this for the wrong reason.
    """
    brief = (ROOT / "PRODUCT.md").read_text(encoding="utf-8")
    match = re.search(r"more than ([\d,]+) tests", brief)
    assert match, "PRODUCT.md no longer states a test count; the guard has nothing to hold"
    claimed = int(match.group(1).replace(",", ""))

    collected = request.session.testscollected
    if collected < claimed // 2:
        pytest.skip(f"partial run ({collected} collected); this needs the whole suite")

    assert collected > claimed, (
        f"PRODUCT.md claims more than {claimed} tests and the suite collects "
        f"{collected}. Lower the claim, or find out which tests went missing."
    )


# --------------------------------------------------------------------------
# The files that describe the project to people who are not reading the code
# --------------------------------------------------------------------------


def test_the_citation_file_names_the_version_the_package_does() -> None:
    """A citation that points at the wrong version is worse than none at all.

    Whoever cites this is recording what they ran. The version is the only part
    of that a reader can check, so it has to be the version the package
    actually declares — and it is exactly the sort of field that is updated in
    one file and forgotten in the other.
    """
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    declared = re.search(r"^version:\s*(\S+)\s*$", citation, re.MULTILINE)
    assert declared, "CITATION.cff has no version line"
    assert declared.group(1) == maiman.__version__, (
        f"CITATION.cff says {declared.group(1)} and the package says {maiman.__version__}"
    )


def test_the_citation_file_has_what_a_citation_needs() -> None:
    """The fields CFF 1.2.0 requires, checked without taking a YAML dependency.

    Deliberately not a schema validation: pulling in a parser to check five
    lines would cost every contributor an install to run the suite. What breaks
    in practice is a field going missing, and that is what this sees.
    """
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    for field in ("cff-version:", "title:", "message:", "type:", "authors:", "license:"):
        assert field in citation, f"CITATION.cff is missing {field!r}"
    assert "cff-version: 1.2.0" in citation


def test_contributing_only_names_tests_that_exist() -> None:
    """Its table of guards is a promise about this suite, so it has to be true.

    A contributing guide that sends someone to a test that was renamed two
    months ago wastes exactly the person who was trying to help. Every
    backticked ``test_...`` name in the file is checked against the suite.
    """
    guide = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    named = set(re.findall(r"`(test_[a-z0-9_]+)`", guide))
    assert named, "no test names found in CONTRIBUTING.md — has its shape changed?"

    sources = "\n".join(
        path.read_text(encoding="utf-8") for path in (ROOT / "tests").glob("test_*.py")
    )
    missing = sorted(name for name in named if f"def {name}(" not in sources)
    assert not missing, f"CONTRIBUTING.md names tests that do not exist: {missing}"


def test_contributing_only_links_to_files_that_exist() -> None:
    """Same argument, for the paths it points at."""
    guide = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    targets = re.findall(r"\]\(([^)#][^)]*)\)", guide)
    missing = sorted(target for target in targets if not (ROOT / target.split("#")[0]).exists())
    assert not missing, f"CONTRIBUTING.md links to paths that do not exist: {missing}"


def test_every_document_points_at_the_same_repository() -> None:
    """One rename, and half of these send people somewhere that does not exist.

    The issue-template config links to a security advisory page by absolute URL,
    and the citation file names the repository. Those are the two places a fork
    or a rename leaves stale, and neither fails in a way anyone notices until a
    contributor follows one.
    """
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    declared = re.search(r"repository-code:\s*\"?(\S+?)\"?\s*$", citation, re.MULTILINE)
    assert declared, "CITATION.cff has no repository-code"
    repository = declared.group(1).rstrip("/")

    config = (ROOT / ".github" / "ISSUE_TEMPLATE" / "config.yml").read_text(encoding="utf-8")
    for url in re.findall(r"url:\s*(\S+)", config):
        assert url.startswith(repository), (
            f"{url} does not point at {repository}, which CITATION.cff names"
        )


def test_the_issue_templates_are_forms_github_will_accept() -> None:
    """Structural, and without a YAML dependency for the same reason as above.

    GitHub does not report a malformed issue form anywhere a maintainer sees it;
    it silently falls back to a blank issue, and the first sign is that reports
    stop arriving with the fields they were meant to have.
    """
    directory = ROOT / ".github" / "ISSUE_TEMPLATE"
    forms = sorted(path for path in directory.glob("*.yml") if path.name != "config.yml")
    assert forms, "no issue forms found"

    for form in forms:
        text = form.read_text(encoding="utf-8")
        for field in ("name:", "description:", "body:"):
            assert field in text, f"{form.name} is missing {field!r}"
        # Every entry in `body` needs a type, and GitHub knows only these.
        types = set(re.findall(r"^\s*-\s*type:\s*(\S+)", text, re.MULTILINE))
        unknown = types - {"markdown", "textarea", "input", "dropdown", "checkboxes"}
        assert not unknown, f"{form.name} uses field types GitHub does not have: {unknown}"


def test_the_security_policy_says_which_versions_are_supported() -> None:
    """And has to keep saying something true as soon as there *is* a release.

    Right now the honest answer is "only main", because nothing has been
    released. The day a version ships, this file needs revisiting -- so the
    guard is tied to the package being a development version, and starts
    failing the moment that stops being true.
    """
    policy = (ROOT / "SECURITY.md").read_text(encoding="utf-8")
    assert "## Supported versions" in policy
    assert "ehsun.ca@gmail.com" in policy, "no reporting channel in the security policy"

    if "dev" in maiman.__version__:
        assert "no release" in policy.lower(), (
            "the policy should say there has been no release while the version is a dev one"
        )
    else:
        raise AssertionError(
            f"the package now declares {maiman.__version__}, which is not a development "
            f"version. SECURITY.md still says only `main` is supported -- decide what "
            f"that means for the released version and update both."
        )
