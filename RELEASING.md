# Releasing

`maiman` is not on PyPI yet. This is what stands between here and there.

Everything mechanical is in [`.github/workflows/release.yml`](.github/workflows/release.yml). The
steps below are the ones a workflow cannot do: they involve web interfaces and an account, and
they are deliberately the only ones.

## The version

The package declares **`0.1.0`** — a real release, not a `.devN` placeholder, so
`pip install maiman` will find it. The `Development Status :: 2 - Pre-Alpha` classifier says what
it is, and `0.1.0` does not claim otherwise.

The number lives in **three** places and they are checked against each other on every commit:
`pyproject.toml`, `src/maiman/__init__.py`, and `CITATION.cff`. Change one and the suite tells you
about the other two.

**A version number on PyPI can never be reused.** Deleting a release does not free it. That is the
one irreversible step in this document, and the reason the workflow refuses to publish when the git
tag and `pyproject.toml` disagree.

Do a TestPyPI run first. It costs one workflow dispatch and it is the only way to see the page the
way a stranger will.

## One-time setup, on the web

### 1. PyPI trusted publisher

No API token is involved anywhere in this repository, and none should be. PyPI mints trust from
GitHub's OIDC identity instead — nothing long-lived exists to leak or rotate.

Because the project does not exist on PyPI yet, register it as a **pending publisher**:
<https://pypi.org/manage/account/publishing/>

| Field | Value |
| :--- | :--- |
| PyPI project name | `maiman` |
| Owner | `ehsun-sh` |
| Repository name | `maiman-studio` |
| Workflow name | `release.yml` |
| Environment name | `pypi` |

Repeat it at <https://test.pypi.org/manage/account/publishing/> with the environment name
`testpypi` if you want the dry run, which you do.

### 2. GitHub environments

Settings → Environments. Create **`pypi`** and **`testpypi`**. They need no secrets — the whole
point is that there are none — but the names must match the workflow and the publisher
registration above, and an environment is where a required reviewer goes if you ever want a human
in the loop before an upload.

## Releasing

```bash
# 1. Decide and set the version in pyproject.toml, then:
git commit -am "Release 0.1.0" && git push

# 2. Tag it. The workflow refuses if this disagrees with pyproject.toml.
git tag v0.1.0 && git push origin v0.1.0
```

Then publish a GitHub release against that tag. The workflow builds, runs `twine check --strict`,
installs the wheel into a clean environment and runs a link through it, and only then uploads.

For a dry run instead: Actions → Release → *Run workflow* → target `testpypi`.

## What is verified before anything is uploaded

The same things `tests/test_packaging.py` asserts on every commit, plus a clean-room check the test
suite cannot do:

* the metadata renders on PyPI (`twine check --strict`),
* the wheel installs into an empty environment,
* a link actually runs from the installed package, outside any source tree,
* `src/maiman/studio/index.html` is inside the wheel — a wheel without it installs a server that
  answers every API route and 404s the page it exists to serve,
* more than forty components register.


## After the first successful upload

Three statements in this repository are true only until the moment the upload lands, and nothing
can check them from inside:

* `README.md` says "Not on PyPI yet" above the install instructions. Replace it with
  `pip install maiman`.
* `RELEASING.md` — this file — opens with the same claim.
* `CITATION.cff` has no `date-released`, on the grounds that there has been no release. Add the
  date the release was actually published, not the date the version was bumped.

None of these is guarded, because a test cannot ask PyPI whether an upload happened without a
network call in CI. They are listed here instead, which is the honest substitute.
