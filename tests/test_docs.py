"""The documentation in docs/ is what the website publishes, so it is held like code.

Three things can go wrong without anybody noticing: a generated reference page
that no longer matches the class it describes, a page the site's sidebar names
that does not exist, and a relative link that points at a file or a heading
that moved. Each has a test here.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"


def _generator():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(
        "gen_reference_docs", ROOT / "tools" / "gen_reference_docs.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_generated_reference_is_what_the_code_says() -> None:
    """A component added, removed or changed shows up in docs/ in the same commit."""
    expected = _generator().generated()
    stale = [
        str(path.relative_to(ROOT))
        for path, text in expected.items()
        if not path.exists() or path.read_text(encoding="utf-8") != text
    ]
    orphans = [
        str(p.relative_to(ROOT)) for p in (DOCS / "components").glob("*.md") if p not in expected
    ]
    assert not stale and not orphans, (
        f"run `python tools/gen_reference_docs.py`: stale {stale[:5]}, orphaned {orphans[:5]}"
    )


def test_every_page_in_the_sidebar_exists() -> None:
    files = re.findall(r"file: ([^,}\s]+)", (DOCS / "nav.yml").read_text(encoding="utf-8"))
    assert files, "docs/nav.yml lists no pages"
    missing = [f for f in files if not (DOCS / f).is_file()]
    assert not missing, f"docs/nav.yml names pages that do not exist: {missing}"


def _anchors(path: Path) -> set[str]:
    text = re.sub(r"```.*?```", "", path.read_text(encoding="utf-8"), flags=re.S)
    slugs = set()
    for heading in re.findall(r"^#{1,6} (.+)$", text, flags=re.M):
        slug = re.sub(r"[^\w\- ]", "", heading.strip().lower()).replace(" ", "-")
        slugs.add(slug)
    return slugs


_PAGES = sorted([*DOCS.rglob("*.md"), ROOT / "README.md"])


@pytest.mark.parametrize("page", _PAGES, ids=lambda p: str(p.relative_to(ROOT)))
def test_relative_links_resolve(page: Path) -> None:
    text = re.sub(r"```.*?```", "", page.read_text(encoding="utf-8"), flags=re.S)
    broken = []
    for target in re.findall(r"\]\(([^)\s]+)\)", text):
        if re.match(r"[a-z]+:", target):
            continue
        file_part, _, anchor = target.partition("#")
        dest = (page.parent / file_part).resolve() if file_part else page
        if not dest.exists() or (anchor and dest.suffix == ".md" and anchor not in _anchors(dest)):
            broken.append(target)
    assert not broken, f"{page.relative_to(ROOT)} has broken links: {broken}"
