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

### Threads

- GUI thread only: every Qt method of the models (`data`, `setData`, `sort`, assigning
  `df`, the setters), `DataFrameHolder`, all widgets, `ColumnProfileWidget.set_stats`.
- Any thread: `DataFrameModel.prefetch(first_row, last_row, columns)` (fills the cache,
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
- **Widgets**: `DataFrameViewer` (data and header tables kept in sync, spans for
  MultiIndex), `ColumnProfileWidget` (dtype, counts, min / max / mean / std, top values
  with share bars), `DataFrameWidget` (viewer with docks for index, columns,
  categories, column profile and `df.attrs`), `DataFrameListWidget` and
  `DataFrameManagerWidget` for several DataFrames.
- **Icons**: `cutepandas.utils.icons.icon_for_dtype` / `icon_for_kind` for pandas,
  pyarrow and polars dtypes.

## Development

```sh
uv sync --all-extras
uv run pytest
uv run ruff check . && uv run ty check
```

## License

[MIT](https://github.com/phil65/cutepandas/blob/main/LICENSE)
