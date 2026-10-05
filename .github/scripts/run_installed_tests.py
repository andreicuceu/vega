#!/usr/bin/env python3
"""Run the Vega test suite against an installed wheel, outside the source checkout."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import venv
from pathlib import Path

# Run inside the staged tests, before pytest, to confirm that ``vega`` is the installed copy.
LOCATION_CHECK = """
import importlib.metadata
import sys
from pathlib import Path

import vega

source_root = Path(sys.argv[1]).resolve()
distribution = importlib.metadata.distribution("vega")
package_file = Path(vega.__file__).resolve()
site_root = Path(distribution.locate_file("")).resolve()
assert site_root in package_file.parents, (site_root, package_file)
assert source_root not in package_file.parents, package_file
assert distribution.version == vega.__version__
print(f"testing {package_file} ({distribution.version})")
"""


def run(command: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    """Run a command, streaming its output, and raise if it fails."""
    subprocess.run(command, cwd=cwd, env=env, check=True)


def stage_tests(source: Path, stage_dir: Path) -> Path:
    """Copy the tests and the benchmark configurations they read into a staging directory.

    ``vega.utils.find_file`` resolves relative paths from the working directory, so the
    benchmark configurations are nested as ``examples/picca_benchmarks`` inside the staged
    tests, which become the working directory.

    Parameters
    ----------
    source : pathlib.Path
        Vega source checkout containing ``tests/`` and ``examples/``.
    stage_dir : pathlib.Path
        Empty directory outside the checkout that receives the copies.

    Returns
    -------
    pathlib.Path
        The staged ``tests`` directory, to be used as the working directory.
    """
    staged_tests = stage_dir / "tests"
    shutil.copytree(
        source / "tests", staged_tests, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    shutil.copytree(
        source / "examples" / "picca_benchmarks",
        staged_tests / "examples" / "picca_benchmarks",
        ignore=shutil.ignore_patterns("*.ipynb", ".ipynb_checkpoints"),
    )
    return staged_tests


def main() -> None:
    """Install the wheel in a clean environment and run pytest from the staged tests."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("pytest_args", nargs="*", help="extra arguments passed to pytest")
    args = parser.parse_args()

    source = args.source.resolve()
    work_dir = args.work_dir.resolve()
    wheel = args.wheel.resolve()
    if not (source / "pyproject.toml").is_file():
        parser.error(f"Not a Vega source tree: {source}")
    if work_dir == source or source in work_dir.parents:
        parser.error("--work-dir must be outside the source checkout")
    if work_dir.exists() and any(work_dir.iterdir()):
        parser.error(f"--work-dir must be empty: {work_dir}")
    if not wheel.is_file() or wheel.suffix != ".whl":
        parser.error(f"Expected a wheel file: {wheel}")
    work_dir.mkdir(parents=True, exist_ok=True)

    environment_dir = work_dir / "venv"
    venv.EnvBuilder(with_pip=True).create(environment_dir)
    python = str(environment_dir / "bin" / "python")
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYTHONNOUSERSITE"] = "1"

    run(
        [python, "-m", "pip", "install", "--disable-pip-version-check", str(wheel)],
        cwd=work_dir,
        env=environment,
    )
    run(
        [python, "-m", "pip", "install", "--disable-pip-version-check", "pytest", "pytest-cov"],
        cwd=work_dir,
        env=environment,
    )
    run([python, "-m", "pip", "check"], cwd=work_dir, env=environment)

    staged_tests = stage_tests(source, work_dir / "stage")
    run([python, "-c", LOCATION_CHECK, str(source)], cwd=staged_tests, env=environment)

    # The working directory is deliberately on ``sys.path`` (no ``-I``) because relative
    # configuration and data paths are resolved from it; it contains no ``vega`` package.
    run(
        [python, "-m", "pytest", "--cov=vega", "--cov-report=term", "-p", "no:cacheprovider"]
        + args.pytest_args,
        cwd=staged_tests,
        env=environment,
    )


if __name__ == "__main__":
    main()
