"""Normalize heading depth in linked example notebooks during Sphinx rendering."""

import json
import re

HEADING = re.compile(r"^( {0,3})(#{1,6})([ \t]+[^\r\n]*)(\r?\n)?$")
FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})([^\r\n]*)")


def normalize_notebook_headings(app, docname, source):
    """Close skipped heading levels without modifying the repository notebooks.

    A Markdown H3 directly beneath an H1 is rendered as H2, including later
    H3 siblings. The original hierarchy determines which headings are siblings
    and which are children. Headings inside fenced code blocks are left intact.
    """
    if not docname.startswith("examples/"):
        return

    notebook = json.loads(source[0])
    heading_stack = []
    changed = False
    for cell in notebook["cells"]:
        if cell["cell_type"] != "markdown":
            continue
        fence_character = None
        fence_length = 0
        lines = "".join(cell["source"]).splitlines(keepends=True)
        for index, line in enumerate(lines):
            fence = FENCE.match(line)
            if fence:
                marker = fence.group(1)
                if fence_character is None:
                    fence_character, fence_length = marker[0], len(marker)
                elif (
                    marker[0] == fence_character
                    and len(marker) >= fence_length
                    and not fence.group(2).strip()
                ):
                    fence_character = None
                continue
            if fence_character is not None:
                continue
            heading = HEADING.match(line)
            if heading is None:
                continue
            original_level = len(heading.group(2))
            while heading_stack and heading_stack[-1][0] >= original_level:
                heading_stack.pop()
            level = heading_stack[-1][1] + 1 if heading_stack else original_level
            if level != original_level:
                lines[index] = (
                    f"{heading.group(1)}{'#' * level}{heading.group(3)}{heading.group(4) or ''}"
                )
                changed = True
            heading_stack.append((original_level, level))
        cell["source"] = lines

    if changed:
        source[0] = json.dumps(notebook)
