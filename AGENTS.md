# AGENTS.md

## Overview

CutePandas provides Qt item models, proxy models and widgets for displaying pandas
DataFrames (including MultiIndex rows/columns) in Qt applications. It is built on top of
[prettyqt](https://github.com/phil65/prettyqt) (`from prettyqt import core, widgets, ...`),
which wraps PySide6 or PyQt6. Not copier-managed: all files are hand-maintained.

## Architecture

Flat package layout (no `src/`): the package lives in `cutepandas/`.

- `cutepandas/pandasmodels/`: Qt models over pandas objects (exported in `__init__.py`).
  - `pandasdataframemodel.py`: `PandasDataFrameModel` (cells; joins MultiIndex levels with a
    separator), `VerticalHeaderModel` / `HorizontalHeaderModel` (one cell per index level).
  - `pandascolumnlistmodel.py`: `PandasColumnListModel`, `PandasIndexListModel` (list
    columns / index levels of a DataFrame, with dtype info and icons).
  - `pandascategorylistmodel.py`: `PandasCategoryListModel` (categories of a category column).
  - `dataframeholder.py`: `DataFrameHolder` (shared, observable frame) and the
    `DataFrameHolderMixin` the pandas models use for `self.df`.
  - `pandasindexfilterproxymodel.py`: `BasePandasIndexFilterProxyModel` and subclasses
    (`PandasStringColumnFilterProxyModel`, `PandasEvalFilterProxyModel`,
    `PandasMultiStringColumnFilterProxyModel`). These are `AbstractProxyModel`s that filter
    via a NumPy boolean mask (`compute_filter`) plus row mapping instead of
    `QSortFilterProxyModel`; any structural change of the source recomputes the mask.
  - `polarsdataframemodel.py`: `PolarsDataFrameModel`; needs the `polars` extra, not
    imported by the package `__init__`.
- `cutepandas/pandaswidgets/`: widgets (exported in `__init__.py`).
  - `dataframeviewer.py`: `DataFrameViewer`, a composite of data table + column header +
    index header (`HeaderView`, a prettyqt `OrientedTableView`) kept in sync on
    scroll/resize/selection; supports MultiIndex spans.
  - `dataframeviewer2.py`: alternative `DataFrameViewer` implementation (not exported).
  - `dataframewidget.py`: `DataFrameWidget` (main window: viewer plus dock widgets for
    index list, column list, categories and `df.attrs`).
  - `dataframelistwidget.py`: `DataFrameListWidget` (table of several DataFrames).
  - `dataframemanagerwidget.py`: `DataFrameManagerWidget` (list + `DataFrameWidget`).
- `cutepandas/qtframe/qtframe.py`: experimental `QtFrame` DataFrame subclass that emits a
  Qt signal when an operation changes the shape. Not exported.
- `cutepandas/utils/`: `icons.py` (dtype to icon), `helpers.py` (`infer_type`, misc).
- `cutepandas/constants/iconnames.py`: icon name constants.

Models keep the DataFrame on `self.df`; views find it through
`view.get_model(skip_proxies=True).df`.

`self.df` lives in a `DataFrameHolder` (`pandasmodels/dataframeholder.py`). Pass one
holder to several models / `DataFrameViewer` to share a frame: assigning `model.df` (or
`holder.df`) emits `about_to_change` / `changed`, and every other model of that holder
resets itself. `DataFrameWidget` uses one holder for all its panels. So a model must
**replace** the frame (`self.df = df.rename(...)`), building the new frame completely
before assigning it; changing it in place (`df.index = ...`, `df[col] = ...`) notifies
nobody. Cell edits via `iat` are the one in-place exception (all models share the object).

## Where to look

| Task | Location |
| --- | --- |
| Cell display / editing / roles | `cutepandas/pandasmodels/pandasdataframemodel.py` |
| Fast filtering | `cutepandas/pandasmodels/pandasindexfilterproxymodel.py` |
| MultiIndex header rendering, synced scrolling | `cutepandas/pandaswidgets/dataframeviewer.py` |
| Explorer window with docks | `cutepandas/pandaswidgets/dataframewidget.py` |
| Dtype icons | `cutepandas/utils/icons.py`, `cutepandas/constants/iconnames.py` |
| Tests and fixtures | `tests/`, `tests/conftest.py` |
| CI | `.github/workflows/build.yml`, `.github/workflows/documentation.yml` |
| Docs config | `mkdocs.yml` |

## Commands

```bash
uv sync --all-extras            # installs pyside6 + pyqt6 + polars + dev/lint/docs groups
uv run pytest                   # tests (Qt runs offscreen, set in tests/conftest.py)
uv run ruff check .             # lint (ruff pinned in the `lint` group)
uv run ruff format .            # format
uv run ty check                 # type check (`cutepandas/` only); warnings fail too
uv run deptry .                 # dependency hygiene
uv run mkdocs-mknodes build     # docs (needs Python >= 3.13)
uv build                        # sdist + wheel
```

Pre-commit hooks (`.pre-commit-config.yaml`, run via prek/pre-commit) call
`uv run ruff format`, `uv run ruff check` and `uv run ty check`, and commitizen checks
commit messages (conventional commits).

## Conventions

- `from __future__ import annotations` everywhere; ruff line length 90; isort with
  `force-sort-within-sections` and 2 lines after imports.
- Use prettyqt wrappers (`core.ModelIndex`, `constants.DISPLAY_ROLE`, `widgets.TableView`,
  `set_model`, ...) rather than raw Qt bindings.
- Qt overrides keep Qt's camelCase names (`rowCount`, `data`, `keyPressEvent`).
- Signals are declared as `core.Signal(...)`; ty cannot type them across both bindings, so
  they carry `# ty: ignore[missing-argument]`.
- Tests use pytest-qt (`qtbot`, `qtmodeltester`); the `df` fixture is a small 2-column frame.

## Pitfalls

- A Qt binding is required at runtime: install with an extra (`cutepandas[pyside6]` or
  `cutepandas[pyqt6]`). Extras are ignored for DEP002 in `[tool.deptry]`.
- `uv.lock` is gitignored.
- The `docs` group's `mkdocs-mknodes` only installs on Python >= 3.13 (marker); CI tests
  run with `--no-group docs`.
- Docs are the mkdocs-mknodes default site: `README.md` is the index page; API pages are
  generated. mkdocs-mknodes >= 2 builds with Zensical (not MkDocs); the generated API
  pages need `mkdocstrings[python]` (in the `docs` group), and the build writes a
  gitignored `mkdocs.generated.yml`. `docs/changelog.md` is written by commitizen (`cz bump`).
- `Makefile` still references poetry and is outdated; prefer the commands above.
- Modules with `if __name__ == "__main__":` demos create a QApplication; never put demo
  code at module level (docs build and tests import every module).
