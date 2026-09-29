"""Guard direct imports and optional features in installed wheel metadata."""

import ast
import re
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
