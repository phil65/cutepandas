# CutePandas: Pandas models and widgets for Qt

[![PyPI License](https://img.shields.io/pypi/l/cutepandas.svg)](https://pypi.org/project/cutepandas/)
[![Package status](https://img.shields.io/pypi/status/cutepandas.svg)](https://pypi.org/project/cutepandas/)
[![Python version](https://img.shields.io/pypi/pyversions/cutepandas.svg)](https://pypi.org/project/cutepandas/)
[![Build](https://github.com/phil65/cutepandas/actions/workflows/build.yml/badge.svg)](https://github.com/phil65/cutepandas/actions/workflows/build.yml)

**CutePandas** provides Qt item models, fast filter proxies and ready-made widgets for
displaying pandas DataFrames in Qt applications, including MultiIndex rows and columns.
It is built on [prettyqt](https://github.com/phil65/prettyqt) and works with PySide6 or PyQt6.

Documentation: https://phil65.github.io/cutepandas/

## Installation

A Qt binding is required, pick one via an extra:

```sh
pip install "cutepandas[pyside6]"
# or
pip install "cutepandas[pyqt6]"
```

## Quickstart

```python
import pandas as pd
from prettyqt import widgets

from cutepandas import pandasmodels, pandaswidgets

app = widgets.app()
df = pd.DataFrame({"name": ["apple", "banana", "cherry"], "price": [1.2, 0.5, 3.0]})

# Composite viewer: data table plus synced row/column headers (MultiIndex aware)
viewer = pandaswidgets.DataFrameViewer(df)
viewer.show()

# Or use the models directly with any view, here with a fast string filter
model = pandasmodels.PandasDataFrameModel(df)
proxy = pandasmodels.PandasStringColumnFilterProxyModel()
proxy.setSourceModel(model)
proxy.set_filter_column(0)
proxy.set_search_term("b")  # rows whose "name" starts with "b"
table = widgets.TableView()
table.set_model(proxy)
table.show()

app.exec()
```

## Features

- **Models**: `PandasDataFrameModel` (cells, MultiIndex levels joined by a configurable
  separator), `VerticalHeaderModel` / `HorizontalHeaderModel` (one cell per index level),
  `PandasColumnListModel`, `PandasIndexListModel`, `PandasCategoryListModel`.
- **Filter proxies** that filter with NumPy boolean indexes instead of per-cell Python
  calls, which is much faster than `QSortFilterProxyModel` on large frames:
  `PandasStringColumnFilterProxyModel`, `PandasMultiStringColumnFilterProxyModel`,
  `PandasEvalFilterProxyModel` (filter with a `DataFrame.eval` expression).
- **Widgets**: `DataFrameViewer` (data and header tables kept in sync, spans for
  MultiIndex), `DataFrameWidget` (viewer with docks for index, columns, categories and
  `df.attrs`), `DataFrameListWidget` and `DataFrameManagerWidget` for several DataFrames.
- Optional `PolarsDataFrameModel` in `cutepandas.pandasmodels.polarsdataframemodel`
  (install the `polars` extra).

## Development

```sh
uv sync --all-extras
uv run pytest
uv run ruff check . && uv run ty check
```

## License

[MIT](https://github.com/phil65/cutepandas/blob/main/LICENSE)
