# CutePandas: dataframe models and widgets for Qt

[![PyPI License](https://img.shields.io/pypi/l/cutepandas.svg)](https://pypi.org/project/cutepandas/)
[![Package status](https://img.shields.io/pypi/status/cutepandas.svg)](https://pypi.org/project/cutepandas/)
[![Python version](https://img.shields.io/pypi/pyversions/cutepandas.svg)](https://pypi.org/project/cutepandas/)
[![Build](https://github.com/phil65/cutepandas/actions/workflows/build.yml/badge.svg)](https://github.com/phil65/cutepandas/actions/workflows/build.yml)

**CutePandas** provides Qt item models, delegates and ready-made widgets for showing
dataframes in Qt applications. pandas is the first-class backend (incl. MultiIndex rows /
columns and pyarrow-backed dtypes); the table, schema and category models also show
pyarrow Tables and polars DataFrames. It is built on
[prettyqt](https://github.com/phil65/prettyqt) and works with PySide6 or PyQt6.

Documentation: https://phil65.github.io/cutepandas/

## Installation

A Qt binding is required, pick one via an extra:

```sh
pip install "cutepandas[pyside6]"
# or
pip install "cutepandas[pyqt6]"
# polars support (optional)
pip install "cutepandas[pyside6,polars]"
```

## Quickstart

```python
import pandas as pd
from prettyqt import widgets

from cutepandas import delegates, pandasmodels, pandaswidgets

app = widgets.app()
df = pd.DataFrame({"name": ["apple", "banana", "cherry"], "price": [1.2, 0.5, 3.0]})

# Composite viewer: data table plus synced row/column headers (MultiIndex aware)
viewer = pandaswidgets.DataFrameViewer(df)
viewer.show()

# Or use the models directly with any view, here with a fast string filter
model = pandasmodels.DataFrameModel(df, color_values=True)
proxy = pandasmodels.PandasStringColumnFilterProxyModel()
proxy.setSourceModel(model)
proxy.set_filter_column(0)
proxy.set_search_term("b")  # rows whose "name" starts with "b"
table = widgets.TableView()
table.set_model(proxy)
table.set_delegate(delegates.ValueColorDelegate(mode="bar"), column=1)
table.show()

app.exec()
```

## Choosing a viewer

Three viewer widgets share the same `DataFrameModel`, cell delegate, sort behaviour
(`sortable`, `sort_mode`, `sort_requested`, `sorting_changed`, see [Sorting](#sorting))
and `DataFrameHolder` sharing; they differ in how they lay out the headers.

| | `DataFrameViewer` | `GridDataFrameViewer` | `SimpleDataFrameViewer` |
| --- | --- | --- | --- |
| Structure | data table + column / index header tables kept in sync | 2 x 2 grid: level-name corner, column header, index, data | one `QTableView` with native headers |
| MultiIndex (pandas) | one header row / index column per level, equal neighbours merged into spans | same spans | levels joined (`"a \| x"`) in one header row / column |
| Level names | strips beside / above the headers, only when the levels are named | corner table, always flush with both headers | none |
| Header sizes | follow the labels | fixed per frame (rows of equal height) | native |
| Resizing | drag header edges (columns, rows), double-click fits | drag header edges (columns, rows), double-click fits | native handles, double-click fits |
| Sort indicator | painted arrow on the leaf header | painted arrow on the leaf header | native indicator |

pyarrow and polars frames have one header row and row numbers in every viewer. More than
`MAX_SPAN_COUNT` (20,000) spans on an axis switch the spans off (Qt's span list gets slow);
a display-only sort switches the index spans off while it is active.

All three paint through palette-derived colours (light and dark themes, restyled on a
palette or style change) without per-widget style sheets. Measured with 10M rows x 6
columns in a 1600 x 900 window (offscreen, median of 40 random jumps, each fetching and
painting the viewport; polars / pandas with a 2-level MultiIndex on both axes):

| | `DataFrameViewer` | `GridDataFrameViewer` | `SimpleDataFrameViewer` |
| --- | --- | --- | --- |
| jump: fetch + paint | 8.8 / 8.1 ms | 8.9 / 9.6 ms | 8.1 / 13.0 ms |
| jump, permutation-sorted | 9.0 / 9.6 ms | 8.5 / 10.0 ms | 10.1 / 8.6 ms |
| repaint (cached) | 4.8 / 4.9 ms | 4.9 / 4.3 ms | 5.1 / 4.9 ms |

Which one:

- `DataFrameViewer`: the default (also inside `DataFrameWidget`); MultiIndex frames
  where unnamed levels should take no space.
- `GridDataFrameViewer`: MultiIndex rows and columns with level names, which the corner
  shows aligned with both headers.
- `SimpleDataFrameViewer`: flat frames (or pyarrow / polars), or where a single
  `QTableView` with native, style-following headers fits better.

## Backends: `FrameSource`

`DataFrameModel` never touches a frame directly. It reads through a `FrameSource`
(`cutepandas.frames`, Qt-free), a small protocol with adapters for pandas
(`PandasSource`), pyarrow (`ArrowSource`) and polars (`PolarsSource`, imported only when
a polars frame is passed):

| Member | Purpose |
| --- | --- |
| `backend`, `native` | backend name, the wrapped frame |
| `row_count`, `columns` | shape; `ColumnInfo(name, dtype, kind, native_dtype)` per column |
| `fetch(start, stop, columns)` | a window as Python values (`ColumnChunk`: values + null mask) |
| `fetch_arrow(start, stop, columns)` | the same window as a `pyarrow.Table` |
| `null_count`, `nbytes`, `value_range`, `value_counts`, `categories`, `describe` | column metadata and statistics |
| `sort(column, descending)` (optional, `SortableSource`) | a sorted copy (stable, nulls last) |
| `argsort(column, descending)`, `fetch_rows(rows, columns)` (optional, `PermutableSource`) | sort order as row positions; a window in any row order |
| `set_value(row, column, value)` (optional, `EditableSource`) | in-place cell edit |

`Kind` groups the backend dtypes (integer, float, string, categorical, datetime, ...) and
drives formatting, alignment and icons. Anything that implements the protocol (e.g. a
lazy source of an application) can be passed to the models instead of a frame.

```python
import polars as pl

from cutepandas import frames, pandasmodels

source = frames.as_source(pl.DataFrame({"a": [1, 2, None]}))
source.fetch(0, 2, [0])[0].values  # [1, 2]
model = pandasmodels.DataFrameModel(source)
```

### Windowed model

`DataFrameModel` fetches rows in blocks (`chunk_rows`, default 256) per column, formats
each block once with per-dtype formatters (`FormatOptions`: float precision, thousands
separator, null text, text truncation) and keeps at most `max_chunks` blocks (default
256) in an LRU `ChunkCache`. Only the visible window is ever converted to Python
objects, so memory is the frame plus a small, bounded cache. Background colours
(`color_values=True`) and normalised values (`NormalizedRole`, used by
`ValueColorDelegate`) are computed per block with NumPy, not per `data()` call.

Measured on a 10M-row x 10-column frame (4 float, 2 int, datetime, categorical,
unique strings, bool; about 700 MB), four roles of every cell of a 40 x 10 viewport at
random positions, cold cache: median 4.5 ms, p95 8 ms, max 11 ms for pandas, pyarrow
and polars alike. The full cache (256 blocks) adds about 9 MB. With `color_values`, the
first paint scans each numeric column once for its range (0.04-0.4 s on 10M rows).

### Sorting

A header click (the leaf level of a MultiIndex) in any of the viewers emits
`sort_requested(column, order)` (a `Qt.SortOrder`; the same column again reverses).
What happens next:

- `sortable=False`: nothing else. The application sorts under its own control (e.g.
  within a memory budget, or as a step of a pipeline), shows the result with
  `set_df()` and restores the indicator with `set_sort_state(column, order)`.
- `sortable=True` (default) and `sort_mode="copy"` (default): the source builds a sorted
  copy (`SortableSource.sort`) that replaces the frame, as large as the frame itself.
- `sortable=True` and `sort_mode="permutation"`: a display-only sort. The source
  computes the order as row positions (`PermutableSource.argsort`; 8 bytes per row),
  stored as `DataFrameHolder.row_order`; the frame is neither copied nor changed and
  every visible block is read through the order (`fetch_rows`). The order is computed
  in a worker thread (`sorting_changed(True / False)`); only the latest request is
  applied, only if the frame was not replaced or edited meanwhile, and
  `cancel_sort()` drops a running one. Replacing the frame or `clear_sort()` restores
  the frame's order; row labels follow the order (MultiIndex spans are off while
  sorted).

The viewers share this logic through a `SortController` (one per `DataFrameModel`,
`cutepandas.pandaswidgets.sortcontroller`), which a custom view can use as well. The
building blocks are public: `DataFrameModel.compute_sort_permutation(column,
order)` (any thread) returns a `SortPermutation`, `apply_sort_permutation()` (GUI
thread) applies it or returns False if it is stale, `source_row(row)` maps a model row
to a frame row. Sort orders are stable with nulls last; pandas uses NumPy's stable
argsort for NumPy dtypes and Arrow's sort kernel for Arrow-backed columns (incl. the
pandas 3 `str` dtype), pyarrow Arrow's sort kernel, polars its own (multi-threaded)
sort.

Measured on 10M rows (one column sorted, descending): computing the order takes
1.6 s (int) / 2.1 s (float) / 6.1 s (str) for pandas, 1.8 / 2.2 / 4.9 s for pyarrow and
0.35 / 0.33 / 1.3 s for polars; the order takes 80 MB. A cold 60 x 3 viewport read
through it: median 0.8-1.2 ms, max 3.3 ms on every backend.

### Translations

cutepandas ships English labels only. `SchemaModel`, `PandasColumnListModel`,
`PandasIndexListModel`, `CategoryModel`, `TopValuesModel`, `ColumnProfileWidget` and
`DataFrameWidget` take `translate=` (a `str -> str` callable such as a gettext
function) and look each label up when they show it; `set_translate()` swaps it and
`retranslate()` re-shows the labels after a language switch. The English labels are
marked with `cutepandas.utils.i18n.N_`, so `xgettext --keyword=N_` (or babel
`-k N_`) extracts them for a catalog.

### Threads

- GUI thread only: every Qt method of the models (`data`, `setData`, `sort`, assigning
  `df`, the setters), `DataFrameHolder`, all widgets, `ColumnProfileWidget.set_stats`.
- Any thread: `DataFrameModel.compute_sort_permutation`,
  `DataFrameModel.prefetch(first_row, last_row, columns)` (fills the cache,
  e.g. ahead of scrolling), the `ChunkCache` (all mutations under a lock; `clear()` /
  `invalidate()` start a new generation so a block built from old data is dropped) and
  the read methods of a `FrameSource` (`fetch`, `describe`, ...; e.g. compute column
  statistics in a worker and hand them to `ColumnProfileWidget.set_stats`).
- `set_value` edits the frame in place: don't run it while a worker reads the frame.

There is no module-level mutable state.

## Features

- **Models**: `DataFrameModel` (any backend; MultiIndex levels joined by a configurable
  separator), `VerticalHeaderModel` / `HorizontalHeaderModel` (one cell per index level),
  `SchemaModel` (columns with dtype, icon, null count, size) and `CategoryModel`
  (categories with code and count; pandas `category`, polars `Enum` / `Categorical`,
  Arrow dictionaries). The pandas models `PandasColumnListModel`,
  `PandasCategoryListModel` and `PandasIndexListModel` also edit the frame.
- **Filter proxies** (pandas) that filter with NumPy boolean indexes instead of
  per-cell Python calls, which is much faster than `QSortFilterProxyModel` on large
  frames: `PandasStringColumnFilterProxyModel`,
  `PandasMultiStringColumnFilterProxyModel`, `PandasEvalFilterProxyModel` (filter with
  a `DataFrame.eval` expression).
- **Delegates**: `DTypeDelegate` (pick a dtype from a combo box with dtype icons),
  `ValueColorDelegate` (data bar or heat map from `NormalizedRole`).
- **Widgets**: `DataFrameViewer`, `GridDataFrameViewer` and `SimpleDataFrameViewer`
  (see [Choosing a viewer](#choosing-a-viewer)), `ColumnProfileWidget` (dtype, counts, min / max / mean / std, top values
  with share bars), `DataFrameWidget` (viewer with docks for index, columns,
  categories, column profile and `df.attrs`), `DataFrameListWidget` and
  `DataFrameManagerWidget` for several DataFrames.
- **Icons**: `cutepandas.utils.icons.icon_for_dtype` / `icon_for_kind` for pandas,
  pyarrow and polars dtypes, tinted to the palette; `KindIconCache` caches them for
  views that paint them often.

## Development

```sh
uv sync --all-extras
uv run pytest
uv run ruff check . && uv run ty check
```

## License

[MIT](https://github.com/phil65/cutepandas/blob/main/LICENSE)
