# Vega package review: remaining recommendations

Reviewed on 2026-09-15 at commit `01f1a9784767e43e68c89ee4464c2bdcb15420c0`
(`master`, synchronized with `origin/master`). This recheck covers package metadata,
installation, release artifacts, GitHub Actions, developer tooling, tests, and Sphinx
documentation. It compares the current repository with the preceding package review.

## Items completed since the preceding review

The following recommendations are complete and do not need further work:

- **Installable source archives:** `.git_archival.txt` and the `export-subst` rule in
  `.gitattributes` let `setuptools-scm` recover versions from GitHub tag archives.
  The current CI explicitly constructs and validates a Git archive without `.git`.
- **Validated distributions:** CI builds the sdist and wheel, applies strict Twine
  checks, rebuilds a wheel from the extracted sdist, installs the exact wheel in an
  isolated environment, and checks representative scientific resources and scripts.
- **Release procedure:** the manual release workflow validates an existing stable tag,
  builds once, verifies checksums, transfers the same artifacts between jobs, and
  creates a GitHub Release without moving or replacing a tag.
- **Published release artifacts:** Vega 1.7.8 has a wheel, sdist, and `SHA256SUMS` on the
  GitHub Releases page. The corresponding build and release jobs passed.
- **Canonical installation guidance:** `README.rst` and `docs/install.rst` recommend the
  tested GitHub Release wheel, checksum verification, the sdist as the source-build
  alternative, and an editable clone for development. They correctly distinguish the
  generated GitHub source snapshot from the tested release artifacts.
- **Build tools:** the development extra now declares `build` and `twine`, and the
  `Makefile` distribution target uses `python -m build` plus strict Twine validation.
- **License metadata:** `pyproject.toml` now uses the SPDX expression
  `GPL-3.0-or-later` and declares the license file using current core metadata.
- **Ruff and pre-commit:** Ruff linting and formatting are active in CI; the pre-commit
  configuration is valid and uses the same Ruff release as CI. The repository was
  formatted and the contributor documentation describes the hook workflow.
- **Most GitHub Action pinning:** checkout, Python setup, artifact upload, and artifact
  download actions are pinned to complete commit SHAs. Dependabot monitors GitHub
  Actions monthly.
- **Contributor release instructions:** `CONTRIBUTING.rst` now describes the current
  `pyproject.toml`, pre-commit, test, and manually dispatched release workflow rather
  than the removed `setup.py` release procedure.

The latest [master CI run](https://github.com/andreicuceu/vega/actions/runs/34999478412)
passed Ruff, Python 3.9--3.13 tests, distribution validation, and artifact transfer. The
[Vega 1.7.8 release run](https://github.com/andreicuceu/vega/actions/runs/34911959607)
also passed. A local Python 3.13 run collected 14 tests; all passed, with 36% total
coverage.

## Remaining recommendations

### 1. Resolve or explicitly document the Python distribution-name collision

**Priority: high.** The `[project]` name is `vega`, but the `vega` project on PyPI is an
unrelated Jupyter visualization package. Consequently, `python -m pip install vega`
does not install this Lyα-analysis package.

- If PyPI publication is intended, choose a distinct distribution name such as
  `desi-vega` or `vega-lya` after checking availability. The import package can remain
  `vega`.
- If GitHub Releases will remain the only supported channel, add a prominent warning
  that the PyPI project is unrelated and that `pip install vega` must not be used.
- Keep the selected distribution name consistent in `pyproject.toml`, the release
  validator, artifact names, release workflow, documentation, and
  `importlib.metadata` lookups.

**Completion criterion:** a new user cannot plausibly install the unrelated PyPI
project by following or extrapolating from Vega's installation documentation.

### 2. Declare every directly imported dependency and make extras coherent

**Priority: high.** A clean Vega installation currently succeeds partly because
`picca` happens to bring in transitive dependencies. That is not a stable package
contract.

- Add `GitPython` as a direct core dependency because `vega.build_config` imports
  `git`, and `BuildConfig` is imported by `vega.__init__`.
- Add a template-generation extra containing `camb` and `fitsio`, which are imported
  directly by the installed `make_template.py` script. Alternatively, make them core
  dependencies if template generation is considered core behavior.
- Replace the generic `samplers = ["pocomc"]` extra with explicit, behavior-complete
  extras. The PocoMC MPI path also imports `mpi4py` and `schwimmbad`; PolyChord imports
  `pypolychord` and is installed by a separate procedure.
- Document the supported combinations, for example `.[mpi]`, `.[pocomc]`,
  `.[templates]`, and `.[docs]`. Do not describe `.[dev]` as including optional runtime
  capabilities unless it actually does so.
- Add isolated import or `--help` probes for each extra so a missing direct dependency
  cannot be hidden by `picca` or another transitive dependency.

**Completion criterion:** each supported feature installs and starts in a clean
environment using only its documented extra, independent of transitive dependency
contents.

### 3. Repair the Sphinx documentation and enforce a warning-free build

**Priority: high.** A strict build with Sphinx 9.1, MyST-NB 1.4, and
Sphinx Book Theme 1.4 exits nonzero with 18 diagnostics. The current problems are:

- all eight notebook targets in `docs/index.rst` point to nonexistent
  `docs/examples/*` documents, while the notebooks reside in the repository-level
  `examples/` directory;
- `docs/modules.rst` refers to `vega.plots.rt_wedge.RtWedge`, but the module is
  `vega.plots.rt_wedges`;
- autodoc imports `vega.samplers.polychord` although `pypolychord` is optional and not
  declared by the docs extra;
- malformed reStructuredText remains in `BuildConfig.build`,
  `CorrelationItem.get_undist_xi_marg_templates`, `get_legendre_bins`, and two lists in
  `HISTORY.rst`;
- `html_static_path` names `docs/_static`, which does not exist.

Make the notebooks available inside the Sphinx source tree or replace those toctree
entries with external links. Mock genuinely optional imports in autodoc, or isolate
optional API pages so the core docs environment does not require MPI and sampler
libraries. Then add a CI job that installs `.[docs]` and runs:

```console
python -m sphinx -W --keep-going -b html docs docs/_build/html
```

**Completion criterion:** the strict command passes in a fresh environment and every
tutorial entry resolves to a rendered page or an intentional external link.

### 4. Test an installed wheel with pytest, not only with import probes

**Status: done.** The `wheel_tests` CI job installs the exact wheel built by the `package`
job into a clean Python 3.14 environment and runs the test suite from a staged copy of
`tests/` outside the checkout (`.github/scripts/run_installed_tests.py`). It asserts that
`vega` is imported from site-packages. Tests that inspect the source tree are skipped there
through `tests/conftest.py`; the editable Python 3.11--3.14 matrix is unchanged.

### 5. Update the supported Python-version policy

**Status: done.** Python 3.11--3.14 is supported: `requires-python = ">=3.11"`, classifiers,
Ruff's `py311` target, the CI matrix, and the README/CONTRIBUTING examples agree
(`tests/test_package_metadata.py::test_supported_python_versions_are_consistent` enforces
this). Python 3.14 is the default for the lint, documentation, packaging, feature and
release jobs and for Read the Docs. Ruff rule B905 is ignored because the existing `zip()`
calls rely on truncation.

### 6. Separate core tests from MPI tests

**Status: partly done.** The ordinary Python matrix installs `.[dev]` without OpenMPI
or `mpi4py`; the `features` job still installs OpenMPI for the `mpi` and `pocomc` probes.
The dedicated bounded `mpiexec` smoke job is still open.

### 7. Replace legacy script installation with console entry points

**Priority: medium.** `pyproject.toml` still uses setuptools `script-files`, and
`vega/cli.py` is an unused cookiecutter placeholder that prints a replacement message.
The README examples use `python run_vega.py`, which looks in the current directory and
does not reliably invoke an installed script.

- Implement small `main()` functions inside the package and expose stable commands
  through `[project.scripts]`, for example `vega-run`, `vega-run-mpi`, and
  `vega-make-template`.
- Keep compatibility aliases for the existing `.py` script names for one deprecation
  interval if analysis scripts depend on them.
- Remove the placeholder CLI and document both installed commands and source-checkout
  invocation unambiguously.
- Extend distribution validation to run each non-destructive command with `--help`.

**Completion criterion:** the documented commands work from an arbitrary directory in
a clean wheel environment.

### 8. Make local developer commands agree with CI

**Status: done.** `make lint` runs Ruff format and lint checks, `make docs` runs the
strict Sphinx build used by CI (and no longer deletes `docs/modules.rst`), `make coverage`
uses pytest, and `tox` and the `servedocs` target were removed. Tests guard against the
obsolete tools returning.

### 9. Finish CI hardening

**Status: done.** Codecov is pinned to the `v7.1.1` commit and `python_package.yml`
declares top-level `permissions: contents: read`. A test requires every workflow action
to be a full commit SHA.

### 10. Correct coverage configuration and establish a useful regression floor

**Priority: medium.** `.coveragerc` sets `source` to the repository URL rather than the
Python package. Pytest currently bypasses this mistake with `--cov=vega`, but direct
coverage commands do not share a reliable configuration. The suite reports 36% total
coverage and has no failure threshold; several orchestration, output, plotting, and
sampler modules are nearly untested.

- Set coverage source to `vega`, normalize omit patterns, and use one coverage command
  locally and in CI.
- Establish an initial floor at or just below the measured baseline, then raise it as
  targeted regression tests are added.
- Prioritize scientifically consequential branches: covariance handling, parameter
  transformations, output round trips, configuration construction, and optional
  sampler dispatch. Use explicit numerical tolerances for model tests.

**Completion criterion:** coverage configuration is internally consistent, a decrease
fails CI, and new scientific behavior is accompanied by focused regression tests.

### 11. Define and test the dependency-compatibility policy

**Status: done.** Each lower bound is documented in `pyproject.toml`, and
`CONTRIBUTING.rst` has a dependency policy: latest-dependencies CI through the unpinned
matrix, no older-environment support, and `freeze-*` artifacts for reconstructing a CI
environment with `pip install -c freeze.txt`. No minimum-dependency job or committed
constraints file was added, so Dependabot has nothing to monitor for Python dependencies.

### 12. Complete documentation and repository metadata cleanup

**Priority: lower.** These do not block installation, but they leave avoidable ambiguity:

- add `CITATION.cff` with the preferred software citation, authors, repository URL,
  license, and DOI or related publications where appropriate;
- correct README typographical errors and replace the nonexistent
  `run_pypolychord.py` example with an actual, bounded validation command;
- either populate `.github/ISSUE_TEMPLATE.md` with useful environment, configuration,
  traceback, and minimal-reproduction fields, replace it with GitHub issue forms, or
  remove the empty file;
- remove obsolete commented Sphinx boilerplate from `docs/conf.py`, update the
  copyright year, and change the release-workflow input example from a stale concrete
  version to `vX.Y.Z`;
- optionally add Sphinx link checking after the strict HTML build is clean.

**Completion criterion:** citation, issue-reporting, and documentation metadata point
to real files and current commands, with no empty or placeholder interfaces.

## No structural change currently required

The flat `vega/` package layout is conventional and working; migration to `src/` would
cause substantial churn without addressing an observed defect. Likewise, packaged
scientific resources are present in the wheel and are now protected by release
validation, so a separate manifest rewrite is not urgent unless the resource set
changes.
