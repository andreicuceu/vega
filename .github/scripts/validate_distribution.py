#!/usr/bin/env python3
"""Build and validate Vega source and wheel distributions."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import tarfile
import textwrap
import venv
import zipfile
from email import policy
from email.message import Message
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from typing import Dict, Optional

DISTRIBUTION_NAME = "vega"
EXPECTED_RESOURCES = (
    "vega/models/Planck18/Planck18.ini",
    "vega/templates/parameters.ini",
    "vega/parameters/default_values.txt",
)
EXPECTED_SCRIPTS = (
    "make_configs.py",
    "make_template.py",
    "run_vega.py",
    "run_vega_mc_fits_mpi.py",
    "run_vega_mc_mpi.py",
    "run_vega_mpi.py",
)


class ValidationError(RuntimeError):
    """Indicate that a distribution does not satisfy Vega's release contract."""


def require(condition: bool, message: str) -> None:
    """Raise a readable validation error when a release invariant fails."""
    if not condition:
        raise ValidationError(message)


def run(
    command: list[str], *, cwd: Optional[Path] = None, env: Optional[Dict[str, str]] = None
) -> None:
    """Run one validation command and display its exact invocation."""
    print(f"+ {shlex.join(command)}", flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def prepare_empty_directory(path: Path) -> None:
    """Create an empty directory, refusing stale validation products."""
    if path.exists():
        require(path.is_dir(), f"Validation path is not a directory: {path}")
        require(not any(path.iterdir()), f"Validation directory is not empty: {path}")
    else:
        path.mkdir(parents=True)


def parse_metadata(contents: bytes, source: str) -> Message:
    """Parse core metadata and require the fields used for release identity."""
    metadata = BytesParser(policy=policy.default).parsebytes(contents)
    require(metadata.get("Name") is not None, f"Missing Name metadata in {source}")
    require(metadata.get("Version") is not None, f"Missing Version metadata in {source}")
    return metadata


def read_sdist_metadata(sdist: Path) -> Message:
    """Read the top-level PKG-INFO from an sdist."""
    with tarfile.open(sdist, mode="r:gz") as archive:
        candidates = [
            member
            for member in archive.getmembers()
            if len(PurePosixPath(member.name).parts) == 2
            and PurePosixPath(member.name).name == "PKG-INFO"
        ]
        require(len(candidates) == 1, f"Expected one top-level PKG-INFO in {sdist}")
        metadata_file = archive.extractfile(candidates[0])
        require(metadata_file is not None, f"Could not read PKG-INFO from {sdist}")
        return parse_metadata(metadata_file.read(), str(sdist))


def read_wheel_metadata(wheel: Path) -> Message:
    """Read the core METADATA file from a wheel."""
    with zipfile.ZipFile(wheel) as archive:
        candidates = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        require(len(candidates) == 1, f"Expected one dist-info/METADATA in {wheel}")
        return parse_metadata(archive.read(candidates[0]), str(wheel))


def validate_wheel_payload(wheel: Path) -> None:
    """Require representative scientific resources in the built wheel."""
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
    missing = sorted(set(EXPECTED_RESOURCES) - names)
    require(not missing, f"Missing required wheel resources: {', '.join(missing)}")


def require_one(directory: Path, pattern: str, description: str) -> Path:
    """Select exactly one matching artifact and reject stale or ambiguous products."""
    matches = sorted(directory.glob(pattern))
    require(len(matches) == 1, f"Expected one {description} in {directory}; found {matches}")
    return matches[0]


def validate_artifacts(dist_dir: Path, expected_version: Optional[str]) -> tuple[Path, Path, str]:
    """Validate artifact count, filenames, metadata, and resource payload."""
    sdist = require_one(dist_dir, "*.tar.gz", "source distribution")
    wheel = require_one(dist_dir, "*.whl", "wheel")
    sdist_metadata = read_sdist_metadata(sdist)
    wheel_metadata = read_wheel_metadata(wheel)

    names = {sdist_metadata["Name"], wheel_metadata["Name"]}
    require(names == {DISTRIBUTION_NAME}, f"Unexpected distribution names: {sorted(names)}")

    versions = {sdist_metadata["Version"], wheel_metadata["Version"]}
    require(len(versions) == 1, f"Source and wheel versions differ: {sorted(versions)}")
    version = versions.pop()
    if expected_version is not None:
        require(
            version == expected_version, f"Expected version {expected_version}, found {version}"
        )

    require(sdist.name == f"vega-{version}.tar.gz", f"Unexpected sdist filename: {sdist.name}")
    require(
        wheel.name == f"vega-{version}-py3-none-any.whl",
        f"Unexpected wheel filename: {wheel.name}",
    )
    validate_wheel_payload(wheel)
    return sdist, wheel, version


def extract_sdist(sdist: Path, destination: Path, version: str) -> Path:
    """Extract an sdist safely and return its single source root."""
    destination.mkdir()
    with tarfile.open(sdist, mode="r:gz") as archive:
        roots = {
            PurePosixPath(member.name).parts[0]
            for member in archive.getmembers()
            if PurePosixPath(member.name).parts
        }
        require(roots == {f"vega-{version}"}, f"Unexpected sdist roots: {sorted(roots)}")
        archive.extractall(destination, filter="data")

    source_root = destination / f"vega-{version}"
    require((source_root / "pyproject.toml").is_file(), "Extracted sdist lacks pyproject.toml")
    require(not (source_root / ".git").exists(), "Extracted sdist unexpectedly contains .git")
    return source_root


def rebuild_sdist(sdist: Path, work_dir: Path, version: str) -> None:
    """Build and validate a wheel from an independently extracted sdist."""
    source_root = extract_sdist(sdist, work_dir / "sdist-source", version)
    rebuilt_dir = work_dir / "sdist-wheel"
    rebuilt_dir.mkdir()
    run([sys.executable, "-m", "build", "--wheel", "--outdir", str(rebuilt_dir), str(source_root)])
    rebuilt_wheel = require_one(rebuilt_dir, "*.whl", "wheel rebuilt from the sdist")
    metadata = read_wheel_metadata(rebuilt_wheel)
    require(metadata["Name"] == DISTRIBUTION_NAME, "Rebuilt wheel has the wrong name")
    require(metadata["Version"] == version, "Rebuilt wheel has the wrong version")
    require(
        rebuilt_wheel.name == f"vega-{version}-py3-none-any.whl",
        f"Unexpected rebuilt wheel filename: {rebuilt_wheel.name}",
    )
    validate_wheel_payload(rebuilt_wheel)
    run([sys.executable, "-m", "twine", "check", "--strict", str(rebuilt_wheel)])


def clean_install_probe(wheel: Path, work_dir: Path, source_root: Path, version: str) -> None:
    """Install the exact wheel in a fresh environment and probe it outside the checkout."""
    environment_dir = work_dir / "wheel-environment"
    venv.EnvBuilder(with_pip=True).create(environment_dir)
    environment_python = environment_dir / "bin" / "python"
    run(
        [
            str(environment_python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            str(wheel),
        ]
    )

    probe_dir = work_dir / "installed-probe"
    probe_dir.mkdir()
    probe = textwrap.dedent(
        f"""
        import importlib.metadata
        import sys
        from pathlib import Path

        import vega

        expected_version = {version!r}
        source_root = Path({str(source_root)!r}).resolve()
        distribution = importlib.metadata.distribution("vega")
        package_file = Path(vega.__file__).resolve()
        site_packages = Path(distribution.locate_file("")).resolve()

        assert distribution.version == expected_version
        assert vega.__version__ == expected_version
        assert site_packages in package_file.parents
        assert source_root not in package_file.parents

        required_symbols = (
            "BuildConfig",
            "FitResults",
            "RtWedge",
            "VegaInterface",
            "VegaPlots",
            "Wedge",
            "run_vega",
        )
        assert all(hasattr(vega, name) for name in required_symbols)

        required_resources = {EXPECTED_RESOURCES!r}
        missing_resources = [
            relative_path
            for relative_path in required_resources
            if not (site_packages / relative_path).is_file()
        ]
        assert not missing_resources, missing_resources

        scripts_dir = Path(sys.executable).parent
        required_scripts = {EXPECTED_SCRIPTS!r}
        missing_scripts = [name for name in required_scripts if not (scripts_dir / name).is_file()]
        assert not missing_scripts, missing_scripts
        print(f"validated {{package_file}} at version {{distribution.version}}")
        """
    )
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYTHONNOUSERSITE"] = "1"
    run([str(environment_python), "-I", "-c", probe], cwd=probe_dir, env=environment)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--dist-dir", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--expected-version")
    parser.add_argument("--version-file", type=Path)
    parser.add_argument("--skip-install", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Build Vega and exercise its distribution contracts."""
    args = parse_args()
    source_root = args.source.resolve()
    dist_dir = args.dist_dir.resolve()
    work_dir = args.work_dir.resolve()
    require((source_root / "pyproject.toml").is_file(), f"Not a source tree: {source_root}")
    prepare_empty_directory(dist_dir)
    prepare_empty_directory(work_dir)

    run([sys.executable, "-m", "build", "--outdir", str(dist_dir), str(source_root)])
    sdist, wheel, version = validate_artifacts(dist_dir, args.expected_version)
    run([sys.executable, "-m", "twine", "check", "--strict", str(sdist), str(wheel)])
    rebuild_sdist(sdist, work_dir, version)
    if not args.skip_install:
        clean_install_probe(wheel, work_dir, source_root, version)

    if args.version_file is not None:
        args.version_file.parent.mkdir(parents=True, exist_ok=True)
        args.version_file.write_text(f"{version}\n", encoding="utf-8")
    print(f"Validated Vega distributions for version {version}")


if __name__ == "__main__":
    main()
