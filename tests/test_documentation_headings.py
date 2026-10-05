"""Regression checks for headings in linked notebook documentation."""

import importlib.util
import json
from pathlib import Path

HELPER = Path(__file__).resolve().parents[1] / "docs" / "notebook_headings.py"
SPEC = importlib.util.spec_from_file_location("notebook_headings", HELPER)
notebook_headings = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(notebook_headings)


def notebook_source(markdown):
    return [json.dumps({"cells": [{"cell_type": "markdown", "source": [markdown]}]})]


def rendered_markdown(source):
    return "".join(json.loads(source[0])["cells"][0]["source"])


def test_skipped_levels_preserve_siblings_children_and_newlines():
    source = notebook_source(
        "# Correlation model\r\n### First result\r\n### Second result\r\n"
        "#### Multipoles\r\n## Separate analysis\r\n### Data comparison\r\n"
    )
    notebook_headings.normalize_notebook_headings(None, "examples/tutorial", source)
    assert rendered_markdown(source) == (
        "# Correlation model\r\n## First result\r\n## Second result\r\n"
        "### Multipoles\r\n## Separate analysis\r\n### Data comparison\r\n"
    )


def test_fenced_code_headings_are_not_rewritten():
    source = notebook_source(
        "# Tutorial\n```md\n```python\n### In code\n```\n### Rendered section\n"
    )
    notebook_headings.normalize_notebook_headings(None, "examples/tutorial", source)
    assert rendered_markdown(source) == (
        "# Tutorial\n```md\n```python\n### In code\n```\n## Rendered section\n"
    )


def test_source_is_unchanged_for_other_documents_and_linked_notebook(tmp_path):
    notebook = {
        "cells": [
            {"cell_type": "markdown", "source": ["# Tutorial\n### Section\n"]},
            {"cell_type": "code", "source": ["print(1)"], "outputs": [{"text": ["1\n"]}]},
        ]
    }
    path = tmp_path / "tutorial.ipynb"
    path.write_text(json.dumps(notebook))
    original_bytes = path.read_bytes()
    source = [path.read_text()]
    notebook_headings.normalize_notebook_headings(None, "intro", source)
    assert source[0].encode() == original_bytes

    notebook_headings.normalize_notebook_headings(None, "examples/tutorial", source)
    assert path.read_bytes() == original_bytes
    assert json.loads(source[0])["cells"][1] == notebook["cells"][1]
    assert rendered_markdown(source) == "# Tutorial\n## Section\n"
