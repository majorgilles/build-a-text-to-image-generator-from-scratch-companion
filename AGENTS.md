# Agent coding guidelines

## Python typing

- Add type annotations to all function and method parameters and return values.
- Prefer precise built-in generics and stable library types.
- Avoid `Any` unless a third-party boundary has no useful type information.
- Write Python docstrings in Google style, including `Args:`, `Returns:`, and `Raises:` when applicable.

## Learning notebooks

- Keep examples beginner-readable and split distinct concepts into focused cells.
- Add nearby Markdown for each new concept and concise intent-focused comments.
- Let each notebook read as one continuous chapter with one final consolidated summary.
- Treat `glossary.md` as the source of truth for terminology and identifiers.
- Keep Markdown, comments, type annotations, tensor shapes, and executable code synchronized.
- Preserve book behavior during documentation- or typing-only work.
- Validate notebook JSON, unique cell IDs, and Python code-cell syntax after editing.

## nbdev exports

- Treat notebooks as the source of truth for reusable code.
- Export reusable definitions to the most appropriate package module.
- Run `uv run nbdev-export` after changing exported definitions.
- Do not edit generated package modules manually.
