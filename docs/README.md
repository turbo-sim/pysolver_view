# PySolverView documentation

The Sphinx site starts at [source/index.md](source/index.md). All authored
pages use Markdown with [MyST](https://myst-parser.readthedocs.io/en/latest/).
The build script generates the API reference from Python docstrings.

## Build and preview

Run from the repository root:

```bash
poetry install --with dev
poetry run python docs/build_docs.py
```

The default command opens a live preview at `http://127.0.0.1:8000` and rebuilds
when documentation changes. Stop it with Ctrl+C. For a single HTML build:

```bash
poetry run python docs/build_docs.py --no-autobuild
```

HTML is written to `docs/_build/html/`, the same directory published by the
GitHub Pages workflow. Builds fail on Sphinx warnings. The script resolves
paths relative to itself and uses tools from the active virtual environment.
It also works from `docs/` as `poetry run python build_docs.py`.

Other useful commands:

```bash
poetry run python docs/build_docs.py --no-open-browser --port 8080
poetry run python docs/build_docs.py --force-apidoc --no-autobuild
poetry run python docs/build_docs.py --clean-only
```

## Source structure

- `source/index.md`: navigation and landing page.
- `source/introduction.md`: overview and installation.
- `source/tutorial_*.md`: optimization and nonlinear system tutorials.
- `source/source_layout.md`: package organization and compatibility.
- `source/figures/`: committed tutorial illustrations.
- `source/references/`: bibliography page and local BibTeX source.
- `source/api/`: generated RST stubs; recreated on each build and ignored by Git.
- `_build/`: generated HTML and Sphinx caches; ignored by Git.
- `development/` and `publication.md`: unpublished notes outside the source tree.
- `build_docs.py` and `conf.py`: shared build workflow and project configuration.

Add authored pages as `.md` files under `source/` and include them in a
`{toctree}`. Use `$...$` for inline math and `$$...$$` for display math; use
`aligned` inside a display block for multiline equations. Use MyST fenced
directives for figures, notes, and Sphinx cross-references. Do not edit the
generated API stubs; update Python docstrings instead.

`build_docs.py` is byte-for-byte identical to `turbodash/docs/build_docs.py`.
Project paths and optional bibliography export settings live in
`docs_build_config` in `conf.py`. The configuration shares Turbodash's theme,
extensions, warning policy, and bibliography rebuild hook, with PySolverView
metadata, source layout, and canonical-module autodoc settings. The release
version is read from `pysolver_view.__version__`.

## Bibliography

Normal builds use the committed `source/references/bibliography.bib` and do not
contact Zotero. Add entries there and cite them using MyST citation roles.
The bibliography hook refreshes the references page during incremental builds.

To enable optional Zotero export, set `zotero_group_id` in `conf.py` to the
project's library ID and set `ZOTERO_API_KEY` in the environment if the library
requires authentication. Then run:

```bash
poetry run python docs/build_docs.py --build-bib --no-autobuild
# Export only, without building the site:
poetry run python docs/build_docs.py --build-bib-only
```
