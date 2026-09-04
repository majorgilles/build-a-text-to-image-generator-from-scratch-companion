# Agent coding guidelines

## Python typing

- Add type annotations to all new or modified Python functions, methods, parameters, return values, and meaningful module-level variables.
- Prefer precise built-in generics and stable library types; avoid `Any` unless a third-party boundary provides no useful type information.
- Keep annotations synchronized with runtime behavior and tensor shapes.
- Write Google-style docstrings for public modules, classes, functions, and methods. Include `Args:`, `Returns:`, and `Raises:` sections when applicable.

## Comments and documentation

- Add concise comments where they explain intent, assumptions, non-obvious math, or implementation choices.
- Do not add comments that merely restate the code.
- Update nearby documentation whenever behavior, terminology, inputs, outputs, or shapes change.
- Treat `glossary.md` as the source of truth for terminology and identifiers.

## Learning notebooks

- Keep notebooks beginner-readable and organize distinct concepts into focused cells.
- Add a Markdown cell before each new concept or substantial code section to explain its purpose, approach, and expected result.
- Add concise, intent-focused comments to non-obvious code.
- Document tensor shapes near every important tensor creation or transformation, including inputs, outputs, intermediate dimensions, and dimension-order conventions.
- Express shapes consistently, for example: `(batch_size, sequence_length, embedding_dim)` or `(batch_size, channels, height, width)`.
- Explain shape-changing operations such as `reshape`, `view`, `permute`, `transpose`, broadcasting, attention projections, and concatenation.
- Add type annotations to reusable Python definitions in notebook code cells.
- Keep Markdown, comments, docstrings, type annotations, tensor-shape descriptions, and executable code synchronized.
- Preserve the notebook's existing behavior during documentation-only or typing-only changes.
- Make each notebook read as one continuous chapter and end it with one consolidated summary.
- After editing a notebook, validate its JSON, ensure cell IDs are unique, and check Python code-cell syntax.

## Package modules

- `build_a_text_to_image_generator_from_scratch_companion/utils/` holds hand-written ports of the
  upstream `txt2img` utilities; edit them directly and keep the chapter notebooks importing from them.
- Run `uv run ruff format` and `uv run ruff check` after editing a module.
