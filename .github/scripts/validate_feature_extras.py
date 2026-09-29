#!/usr/bin/env python3
"""Probe one Vega wheel in separate environments for its optional features."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import venv
from pathlib import Path
from typing import Optional

FEATURES = ("core", "templates", "mpi", "pocomc")
SCRIPTS = {
    "core": ("run_vega.py", "make_configs.py"),
    "templates": ("make_template.py",),
    "mpi": ("run_vega_mpi.py", "run_vega_mc_mpi.py", "run_vega_mc_fits_mpi.py"),
    "pocomc": ("run_vega_mpi.py",),
}

PROBE = """
import importlib
import importlib.metadata
import sys
from pathlib import Path

feature = sys.argv[1]
source_root = Path(sys.argv[2]).resolve()
distribution = importlib.metadata.distribution("vega")
import vega

package_file = Path(vega.__file__).resolve()
site_root = Path(distribution.locate_file("")).resolve()
assert site_root in package_file.parents, (site_root, package_file)
assert source_root not in package_file.parents, package_file
assert package_file.is_file(), package_file
assert distribution.version == vega.__version__
assert hasattr(vega, "BuildConfig")

if feature == "templates":
    import camb
    import fitsio
elif feature == "mpi":
    from mpi4py import MPI
elif feature == "pocomc":
    import pocomc
    from mpi4py import MPI
    from schwimmbad import MPIPool
    importlib.import_module("vega.samplers.pocomc")

print(f"{feature}: {package_file} ({distribution.version})")
"""


def run(command: list[str], *, cwd: Path, env: dict[str, str], show_output: bool = False) -> None:
    """Run a bounded probe and retain full subprocess output on failure."""
    result = subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(command)}\n"
            f"{result.stdout}\n{result.stderr}"
        )
    if show_output and result.stdout.strip():
        print(result.stdout.strip())


def wheel_for_probe(source: Path, work_dir: Path, supplied_wheel: Optional[Path]) -> Path:
    """Build one wheel, or use the exact wheel provided by the caller."""
    if supplied_wheel is not None:
        wheel = supplied_wheel.resolve()
        if not wheel.is_file() or wheel.suffix != ".whl":
            raise ValueError(f"Expected a wheel file: {wheel}")
        return wheel

    dist_dir = work_dir / "dist"
    dist_dir.mkdir()
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYTHONNOUSERSITE"] = "1"
    run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(dist_dir), str(source)],
        cwd=work_dir,
        env=environment,
    )
    wheels = list(dist_dir.glob("*.whl"))
    if len(wheels) != 1:
        raise ValueError(f"Expected exactly one built wheel in {dist_dir}: {wheels}")
    return wheels[0]


def probe_feature(feature: str, wheel: Path, source: Path, work_dir: Path) -> None:
    """Install a feature in a fresh virtual environment and test its imports and scripts."""
    feature_dir = work_dir / feature
    feature_dir.mkdir()
    environment_dir = feature_dir / "venv"
    venv.EnvBuilder(with_pip=True).create(environment_dir)
    python = environment_dir / "bin" / "python"
    probe_dir = feature_dir / "outside-checkout"
    probe_dir.mkdir()
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYTHONNOUSERSITE"] = "1"

    requirement = str(wheel) if feature == "core" else f"{wheel}[{feature}]"
    run(
        [str(python), "-m", "pip", "install", "--disable-pip-version-check", requirement],
        cwd=probe_dir,
        env=environment,
    )
    run([str(python), "-m", "pip", "check"], cwd=probe_dir, env=environment)
    run(
        [str(python), "-I", "-c", PROBE, feature, str(source)],
        cwd=probe_dir,
        env=environment,
        show_output=True,
    )
    for script in SCRIPTS[feature]:
        installed_script = python.parent / script
        if not installed_script.is_file():
            raise FileNotFoundError(f"Missing installed script: {installed_script}")
        run([str(python), str(installed_script), "--help"], cwd=probe_dir, env=environment)
    print(f"validated {feature} from {wheel.name}", flush=True)


def main() -> None:
    """Select a wheel and probe requested features outside the source checkout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--feature", choices=FEATURES, action="append", dest="features")
    args = parser.parse_args()
    source = args.source.resolve()
    work_dir = args.work_dir.resolve()
    if not (source / "pyproject.toml").is_file():
        parser.error(f"Not a Vega source tree: {source}")
    if work_dir == source or source in work_dir.parents:
        parser.error("--work-dir must be outside the source checkout")
    if work_dir.exists() and any(work_dir.iterdir()):
        parser.error(f"--work-dir must be empty: {work_dir}")
    work_dir.mkdir(parents=True, exist_ok=True)

    wheel = wheel_for_probe(source, work_dir, args.wheel)
    for feature in dict.fromkeys(args.features or FEATURES):
        probe_feature(feature, wheel, source, work_dir)


if __name__ == "__main__":
    main()
