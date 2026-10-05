"""Guard direct imports and optional features in installed wheel metadata."""

import ast
import re
import tomllib
from importlib.metadata import distribution
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1]
STDLIB = {
    "argparse",
    "configparser",
    "copy",
    "dataclasses",
    "datetime",
    "functools",
    "importlib",
    "multiprocessing",
    "os",
    "pathlib",
    "sys",
    "time",
    "typing",
}
CORE_IMPORTS = {
    "numpy": "numpy",
    "scipy": "scipy",
    "astropy": "astropy",
    "numba": "numba",
    "h5py": "h5py",
    "iminuit": "iminuit",
    "cachetools": "cachetools",
    "matplotlib": "matplotlib",
    "mcfit": "mcfit",
    "getdist": "getdist",
    "picca": "picca",
    "git": "gitpython",
}
OPTIONAL_IMPORTS = {
    "camb": ("templates", {"bin/make_template.py"}),
    "fitsio": ("templates", {"bin/make_template.py"}),
    "mpi4py": (
        "mpi",
        {
            "bin/run_vega_mpi.py",
            "bin/run_vega_mc_mpi.py",
            "bin/run_vega_mc_fits_mpi.py",
            "vega/samplers/sampler_interface.py",
        },
    ),
    "pocomc": ("pocomc", {"bin/run_vega_mpi.py", "vega/samplers/pocomc.py"}),
    "schwimmbad": ("pocomc", {"bin/run_vega_mpi.py"}),
    "pypolychord": (None, {"vega/samplers/polychord.py"}),
    "desimeter": (
        None,
        {"vega/models/instrumental_systematics/write_desi_instrumental_syst_table.py"},
    ),
    "desimodel": (
        None,
        {"vega/models/instrumental_systematics/write_desi_instrumental_syst_table.py"},
    ),
}


def requirement_names_by_extra():
    """Collect directly declared requirements without resolving transitive packages."""
    names = {"core": set()}
    for requirement in distribution("vega").metadata.get_all("Requires-Dist", []):
        name = re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", requirement).group()
        name = re.sub(r"[-_.]+", "-", name).lower()
        extra = re.search(r"\bextra\s*==\s*['\"]([^'\"]+)['\"]", requirement)
        names.setdefault(extra.group(1) if extra else "core", set()).add(name)
    return names


def test_direct_imports_have_direct_requirements():
    requirements = requirement_names_by_extra()
    assert {
        "numpy",
        "scipy",
        "astropy",
        "numba",
        "h5py",
        "iminuit",
        "cachetools",
        "matplotlib",
        "mcfit",
        "getdist",
        "picca",
        "gitpython",
    } <= requirements["core"]
    assert requirements["mpi"] == {"mpi4py"}
    assert requirements["templates"] == {"camb", "fitsio"}
    assert requirements["pocomc"] == {"pocomc", "mpi4py", "schwimmbad"}
    assert requirements["samplers"] == requirements["pocomc"]


def test_package_and_installed_script_import_inventory():
    """Reject undeclared imports even when another package installs them transitively."""
    requirements = requirement_names_by_extra()
    sources = sorted((SOURCE / "vega").rglob("*.py")) + sorted((SOURCE / "bin").glob("*.py"))
    for source in sources:
        relative = source.relative_to(SOURCE).as_posix()
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = [alias.name.split(".", 1)[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                roots = [node.module.split(".", 1)[0]]
            else:
                continue
            for root in roots:
                if root in STDLIB or root == "vega":
                    continue
                if root in OPTIONAL_IMPORTS:
                    extra, allowed_paths = OPTIONAL_IMPORTS[root]
                    assert relative in allowed_paths, (relative, root)
                    if extra is not None:
                        assert root in requirements[extra], (relative, root, extra)
                        if root == "mpi4py":
                            assert root in requirements["pocomc"]
                    continue
                assert root in CORE_IMPORTS, (relative, root)
                assert CORE_IMPORTS[root] in requirements["core"], (relative, root)


def test_supported_python_versions_are_consistent():
    """Require metadata, Ruff, CI, and documentation to name the same Python releases."""
    pyproject = tomllib.loads((SOURCE / "pyproject.toml").read_text(encoding="utf-8"))
    floor = re.fullmatch(r">=3\.(\d+)", pyproject["project"]["requires-python"])
    assert floor, pyproject["project"]["requires-python"]
    minor_floor = int(floor.group(1))

    classifier_prefix = "Programming Language :: Python :: 3."
    classifier_minors = sorted(
        int(name.removeprefix(classifier_prefix))
        for name in pyproject["project"]["classifiers"]
        if name.startswith(classifier_prefix)
    )
    supported_minors = list(range(minor_floor, classifier_minors[-1] + 1))
    assert classifier_minors == supported_minors

    # Ruff must not target a release older than the oldest supported one.
    assert pyproject["tool"]["ruff"]["target-version"] == f"py3{minor_floor}"

    # The editable test matrix covers exactly the supported releases.
    workflow = (SOURCE / ".github" / "workflows" / "python_package.yml").read_text(encoding="utf-8")
    matrix = re.search(r"python-version:\s*\[([^\]]*)\]", workflow)
    assert matrix
    matrix_minors = [int(item.split(".")[1]) for item in re.findall(r"3\.\d+", matrix.group(1))]
    assert matrix_minors == supported_minors

    # Every single-version job, the release workflow, the documentation build, and the
    # installation examples use the newest supported release.
    default = f"3.{supported_minors[-1]}"
    pinned_versions = set()
    for path in (".github/workflows/python_package.yml", ".github/workflows/release.yml"):
        text = (SOURCE / path).read_text(encoding="utf-8")
        pinned_versions.update(re.findall(r'python-version:\s*"(3\.\d+)"', text))
    pinned_versions.update(
        re.findall(r'python:\s*"(3\.\d+)"', (SOURCE / ".readthedocs.yaml").read_text())
    )
    for path in ("README.rst", "CONTRIBUTING.rst"):
        text = (SOURCE / path).read_text(encoding="utf-8")
        pinned_versions.update(re.findall(r"python=(3\.\d+)", text))
    assert pinned_versions == {default}


def test_makefile_targets_match_documented_tooling():
    """Keep local developer commands on the tools that CI and the extras declare."""
    makefile = (SOURCE / "Makefile").read_text(encoding="utf-8")
    targets = set(re.findall(r"^([a-zA-Z_-]+):", makefile, flags=re.MULTILINE))
    for obsolete in ("flake8", "tox", "watchmedo", "sphinx-apidoc"):
        assert obsolete not in makefile, obsolete
    assert not {"test-all", "servedocs"} & targets

    documentation = "\n".join(
        (SOURCE / name).read_text(encoding="utf-8")
        for name in ("README.rst", "CONTRIBUTING.rst", "docs/install.rst")
    )
    # Only inline literals such as ``make lint``; PolyChord's own build steps also use make.
    for target in re.findall(r"``make\s+([a-zA-Z_-]+)``", documentation):
        assert target in targets, target


def test_workflow_actions_are_pinned_and_token_is_read_only():
    """Require immutable third-party actions and an explicit read-only default token."""
    workflows = sorted((SOURCE / ".github" / "workflows").glob("*.yml"))
    assert workflows
    for workflow in workflows:
        text = workflow.read_text(encoding="utf-8")
        for action in re.findall(r"^\s*(?:-\s+)?uses:\s*(\S+)", text, flags=re.MULTILINE):
            assert re.fullmatch(r"[\w.-]+/[\w./-]+@[0-9a-f]{40}", action), (workflow.name, action)
        top_level = re.search(
            r"^permissions:\s*\n\s+contents:\s*read\s*$", text, flags=re.MULTILINE
        )
        assert top_level, workflow.name
