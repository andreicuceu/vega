"""Check the linked notebook pages and local assets in a Vega HTML build."""

import argparse
import subprocess
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

NOTEBOOKS = (
    "Vega_tutorial",
    "SimpleModelTutorial",
    "FitResultsTutorial",
    "VegaPlots",
    "Plots_tutorial",
    "VegaPlots2Datasets",
    "config_creation",
    "Sensitivity_tutorial",
)


class LocalReferences(HTMLParser):
    """Collect local HTML links and asset paths."""

    def __init__(self):
        super().__init__()
        self.references = []

    def handle_starttag(self, tag, attributes):
        attributes = dict(attributes)
        reference = attributes.get("src" if tag in ("img", "script") else "href")
        if reference and tag in ("a", "img", "script", "link"):
            url = urlsplit(reference)
            if not url.scheme and not url.netloc and url.path:
                self.references.append(unquote(url.path))


def validate(html_root):
    repository = Path(__file__).resolve().parents[2]
    source_paths = [Path("examples") / f"{name}.ipynb" for name in NOTEBOOKS]
    for name, source_path in zip(NOTEBOOKS, source_paths):
        link = repository / "docs" / "examples" / f"{name}.ipynb"
        expected = Path("../../") / source_path
        assert link.is_symlink() and link.readlink() == expected, link
        assert link.resolve() == (repository / source_path).resolve(), link

    # Sphinx only reads the canonical notebooks. A clean checkout must remain
    # clean for these files after the full and incremental builds.
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", *(str(path) for path in source_paths)],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    assert not status.stdout, f"Canonical example notebooks changed:\n{status.stdout}"

    pages = [html_root / "index.html"]
    index = pages[0].read_text()
    for name in NOTEBOOKS:
        page = html_root / "examples" / f"{name}.html"
        assert page.is_file() and page.stat().st_size > 1000, page
        assert f"examples/{name}.html" in index, name
        rendered = page.read_text()
        assert 'class="cell docutils container"' in rendered, page
        assert 'class="cell_input docutils container"' in rendered, page
        pages.append(page)

    references_checked = 0
    for page in pages:
        parser = LocalReferences()
        parser.feed(page.read_text())
        for reference in parser.references:
            target = (page.parent / reference).resolve()
            assert target.exists(), f"Broken local reference in {page}: {reference}"
            references_checked += 1

    print(f"Validated {len(NOTEBOOKS)} notebook pages and {references_checked} local references")


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("html_root", type=Path, help="Sphinx HTML output directory")
    arguments = argument_parser.parse_args()
    validate(arguments.html_root.resolve())
