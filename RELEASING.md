# Releasing

`maiman` is on PyPI. This is how a release is cut.

Everything mechanical is in [`.github/workflows/release.yml`](.github/workflows/release.yml). The
steps below are the ones a workflow cannot do: they involve web interfaces and an account, and
they are deliberately the only ones.

## The version

The package declares **`0.2.1`**, which is published. The next release picks the next number;
a `.devN` version is not worth publishing, because pip excludes pre-releases and
`pip install maiman` would not find it.

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

Already done for this repository, and recorded here for anyone forking it. The first
registration is a **pending publisher**, because the project does not exist on the index yet:
<https://pypi.org/manage/account/publishing/>. Afterwards it lives under the project's own
settings.

| Field | Value |
| :--- | :--- |
| PyPI project name | `maiman` |
| Owner | `ehsun-sh` |
| Repository name | `maiman-studio` |
| Workflow name | `release.yml` |
| Environment name | `pypi` |

### 2. TestPyPI trusted publisher — a *separate* registration

**TestPyPI is a different service with a different account and its own publisher list.** Registering
on PyPI does not register you on TestPyPI, and the dry run fails with `invalid-publisher` if you
skip this. That is exactly what happened the first time.

<https://test.pypi.org/manage/account/publishing/>, same table as above, except:

| Field | Value |
| :--- | :--- |
| Environment name | `testpypi` |

### 3. GitHub environments

Settings → Environments. Create **`pypi`** and **`testpypi`**. They need no secrets — the whole
point is that there are none — but the names must match the workflow and the publisher
registration above, and an environment is where a required reviewer goes if you ever want a human
in the loop before an upload.

## Releasing

```bash
git tag v0.2.1
git push origin v0.2.1
```

Two lines, not one joined by `&&` — **Windows PowerShell 5.1 has no `&&`**, and a chained command
there is a parser error that runs neither half. The first attempt at this produced no tag and no
error anyone noticed, which is exactly how a parser error looks when you expected a git message.

The workflow refuses to publish if the tag disagrees with `pyproject.toml`.

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


## After a release

Three things go stale the moment an upload lands, and no test can check them without a network
call in CI:

* `README.md` and this file describe the install. They were updated at 0.2.1.
* `CITATION.cff` needs `date-released` set to the date the release was *published*, not the date
  the version was bumped. Its `version` is already checked against the package on every commit.
* The version in `pyproject.toml`, `src/maiman/__init__.py` and `CITATION.cff` must move together
  for the next release — that one *is* checked.

## When the upload fails with `invalid-publisher`

```
* `invalid-publisher`: valid token, but no corresponding publisher
```

The OIDC token was fine; PyPI has no publisher matching it. The action prints the claims it sent,
and the fix is to compare them field by field against the registration:

| Claim in the log | Must equal |
| :--- | :--- |
| `repository` | Owner + repository name in the form |
| `workflow_ref` | ends with the **Workflow name** you registered, e.g. `release.yml` |
| `environment` | the **Environment name** you registered |

If all three already match, check *which service* you are looking at. A publisher on
<https://pypi.org> does nothing for an upload to <https://test.pypi.org>, and the error is identical
either way.

Nothing is uploaded when this fails, so the version number is not spent and there is nothing to
clean up. Fix the registration and run it again.

## Reading the run

A dispatch with `target: testpypi` runs **Build and check**, then **Publish to TestPyPI**, and
*skips* **Publish to PyPI** — which GitHub renders greyed out with a dash, next to a job that
failed with a red cross. The two look similar at a glance and mean opposite things: skipped is
correct, failed is not. `gh run view <id>` prints them unambiguously.
