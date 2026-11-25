# AGENTS.md

## Overview

CutePandas provides Qt item models, delegates and widgets for dataframes in Qt
applications. pandas is the first-class backend (MultiIndex rows/columns, pyarrow-backed
dtypes); the table / schema / category models also take pyarrow Tables and polars
DataFrames. It is built on [prettyqt](https://github.com/phil65/prettyqt)
(`from prettyqt import core, widgets, ...`), which wraps PySide6 or PyQt6. Not
copier-managed: all files are hand-maintained. pyarrow is a core dependency; polars is
the optional `polars` extra (never import it at module level outside
`frames/polarssource.py`).

## Architecture

Flat package layout (no `src/`): the package lives in `cutepandas/`.

- `cutepandas/frames/` (Qt-free): the backend-agnostic core.
  - `source.py`: `FrameSource` protocol (`row_count`, `columns` -> `ColumnInfo`,
    `fetch(start, stop, cols)` -> `ColumnChunk` (Python values + null mask),
    `fetch_arrow`, `null_count`, `nbytes`, `value_range`, `value_counts`,
    `categories`, `describe` -> `ColumnStats`), optional `SortableSource` /
    `PermutableSource` (`argsort`, `fetch_rows`: display-only sort) /
    `EditableSource`, the `Kind` enum (dtype family).
  - `pandassource.py`, `arrowsource.py`, `polarssource.py`: adapters + dtype -> `Kind`
    mapping. `frames.as_source(obj)` picks the adapter.
  - `formatters.py`: `FormatOptions`, per-`Kind` formatters, `format_chunk`.
  - `chunkcache.py`: `ChunkCache`, thread-safe LRU of (row block, column) -> chunk with
    generations (stale worker results are dropped).
- `cutepandas/pandasmodels/`: Qt models (exported in `__init__.py`).
  - `dataframemodel.py`: `DataFrameModel` (virtual, any backend, windowed chunk cache,
    `color_values`, `NormalizedRole`, `prefetch()` for workers),
    `VerticalHeaderModel` / `HorizontalHeaderModel` (one cell per pandas index level;
    row numbers / single row for other backends).
  - `schemamodel.py`: `SchemaModel` (columns, dtype, icon, nulls, size; read-only).
    `pandascolumnlistmodel.py`: `PandasColumnListModel(SchemaModel)` adds pandas editing;
    `PandasIndexListModel` (index levels).
  - `categorymodel.py`: `CategoryModel` (read-only, any backend);
    `pandascategorylistmodel.py`: `PandasCategoryListModel(CategoryModel)` edits.
  - `dataframeholder.py`: `DataFrameHolder` (shared, observable frame of any backend;
    `df` = native frame, `source` = adapter) and the `DataFrameHolderMixin`.
  - `pandasindexfilterproxymodel.py`: `BasePandasIndexFilterProxyModel` and subclasses
    (`PandasStringColumnFilterProxyModel`, `PandasEvalFilterProxyModel`,
    `PandasMultiStringColumnFilterProxyModel`). These are `AbstractProxyModel`s that filter
    via a NumPy boolean mask (`compute_filter`) plus row mapping instead of
    `QSortFilterProxyModel`; any structural change of the source recomputes the mask.
    pandas only.
- `cutepandas/delegates/`: `DTypeDelegate` (dtype combo box), `ValueColorDelegate`
  (data bar / heat map from `NormalizedRole`).
- `cutepandas/pandaswidgets/`: widgets (exported in `__init__.py`).
  - `dataframeviewer.py`: `DataFrameViewer`, a composite of data table + column header +
    index header (`HeaderView`, a prettyqt `OrientedTableView`) kept in sync on
    scroll/resize/selection; supports MultiIndex spans (pandas, `axis_spans()`), any
    backend.
  - `gridviewer.py`: `GridDataFrameViewer`, a 2 x 2 grid (`CornerLevelModel` with the
    level names, column header, index, data).
  - `simpledataframeviewer.py`: `SimpleDataFrameViewer`, one `QTableView` with native
    headers (MultiIndex levels joined).
  - `sortcontroller.py`: `SortController` (header-click sorting of one
    `DataFrameModel`: copy / permutation mode, worker, stale drop, sort state) and
    `SortableViewerMixin` (the viewers' sort API). All three viewers use it.
  - `_viewer_style.py`: `ViewerColors` (from the palette), `DataCellDelegate`,
    `HeaderCellDelegate`, `CornerCellDelegate`, sizing constants / helpers. Styling
    goes through palettes and delegates, not style sheets (QStyleSheetStyle slows
    painting); only under an app / parent style sheet do the tables get a small reset
    sheet. Delegates paint per cell: no enum flag operations or extra `data()` calls
    there.
  - `columnprofilewidget.py`: `ColumnProfileWidget` (stats + top values), `TopValuesModel`.
  - `dataframewidget.py`: `DataFrameWidget` (main window: viewer plus dock widgets for
    index list, column list, categories, column profile and `df.attrs`).
  - `dataframelistwidget.py`: `DataFrameListWidget` (table of several DataFrames).
  - `dataframemanagerwidget.py`: `DataFrameManagerWidget` (list + `DataFrameWidget`).
- `cutepandas/qtframe/qtframe.py`: experimental `QtFrame` DataFrame subclass that emits a
  Qt signal when an operation changes the shape. Not exported.
- `cutepandas/utils/`: `icons.py` (dtype / `Kind` / index to icon), `helpers.py`,
  `i18n.py` (`N_` marker, `TranslateMixin` / `TranslatableHeaderMixin`: labels go
  through an optional `translate` callable, `retranslate()` re-shows them).
- `cutepandas/constants/iconnames.py`: icon name constants.

Models keep the frame on `self.df` (native) and `self.source` (adapter); views find it
through `view.get_model(skip_proxies=True).df`.

`self.df` lives in a `DataFrameHolder` (`pandasmodels/dataframeholder.py`). Pass one
holder to several models / `DataFrameViewer` to share a frame: assigning `model.df` (or
`holder.df`) emits `about_to_change` / `changed`, and every other model of that holder
resets itself (and drops its caches; `DataFrameModel` also drops its cache for its own
changes). So a model must **replace** the frame (`self.df = df.rename(...)`), building
the new frame completely before assigning it; changing it in place (`df.index = ...`,
`df[col] = ...`) notifies nobody. Cell edits via `EditableSource.set_value` are the one
in-place exception (other `DataFrameModel`s on the same holder keep stale cache blocks).

Sorting: `DataFrameModel(sort_mode="copy")` replaces the frame by a sorted copy;
`sort_mode="permutation"` stores a row order (`DataFrameHolder.row_order`, cleared when
the frame is replaced) that `DataFrameModel` / `VerticalHeaderModel` read through.
`DataFrameViewer(sortable=False)` (likewise the other viewers, via `SortController`)
only emits `sort_requested`; in permutation mode the controller computes the order in a
worker thread (latest request wins, stale results are
dropped via the model's data generation).

New fixed English labels: mark them with `N_` and show them via `self._tr(...)`.

Threads: Qt methods, holder and widgets on the GUI thread; `DataFrameModel.prefetch`,
`compute_sort_permutation`, `ChunkCache` and `FrameSource` reads on any thread. No
module-level mutable state.

## Where to look

| Task | Location |
| --- | --- |
| Cell display / editing / roles | `cutepandas/pandasmodels/dataframemodel.py` |
| Backend adapters, dtype kinds | `cutepandas/frames/` |
| Value formatting | `cutepandas/frames/formatters.py` |
| Fast filtering | `cutepandas/pandasmodels/pandasindexfilterproxymodel.py` |
| MultiIndex header rendering, synced scrolling | `cutepandas/pandaswidgets/dataframeviewer.py`, `gridviewer.py` |
| Viewer colours, cell / header painting | `cutepandas/pandaswidgets/_viewer_style.py` |
| Header-click sorting | `cutepandas/pandaswidgets/sortcontroller.py` |
| Explorer window with docks | `cutepandas/pandaswidgets/dataframewidget.py` |
| Dtype icons / delegates | `cutepandas/utils/icons.py`, `cutepandas/delegates/` |
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
uv run mknodes build            # docs (needs Python >= 3.13)
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
  `tests/test_frames.py` and `tests/test_dataframemodel.py` run every test for pandas,
  pandas with pyarrow dtypes, pyarrow and polars (`BACKENDS`).

## Pitfalls

- A Qt binding is required at runtime: install with an extra (`cutepandas[pyside6]` or
  `cutepandas[pyqt6]`). Extras are ignored for DEP002 in `[tool.deptry]`.
- `uv.lock` is gitignored.
- The `docs` group's `mknodes` only installs on Python >= 3.13 (marker); CI tests
  run with `--no-group docs`.
- Docs are the mknodes default site: `README.md` is the index page; API pages are
  generated. mknodes builds with Zensical (not MkDocs); the generated API
  pages need `mkdocstrings[python]` (in the `docs` group), and the build writes a
  gitignored `mkdocs.generated.yml`. `docs/changelog.md` is written by commitizen (`cz bump`).
- `Makefile` still references poetry and is outdated; prefer the commands above.
- Modules with `if __name__ == "__main__":` demos create a QApplication; never put demo
  code at module level (docs build and tests import every module).
