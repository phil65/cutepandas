"""Pandas models.

CutePandas contains a range of tools to work with pandas data structures.


## Models

`DataFrameModel` is a virtual table model for pandas, pyarrow and polars frames. It
reads through a `FrameSource` (see `cutepandas.frames`) and converts only the visible
window: rows are fetched in blocks per column, formatted once by per-dtype formatters
and kept in a bounded, thread-safe LRU cache. pandas MultiIndex columns are shown by
joining the levels with a separator (a Qt Property).

`VerticalHeaderModel` / `HorizontalHeaderModel` show one cell per index level; the
`DataFrameViewer` widget combines them with the data table (synced on scrolling /
resizing).

`SchemaModel` lists the columns of any frame (dtype, icon, null count, size),
`CategoryModel` the categories of a categorical / enum column. Their pandas
subclasses `PandasColumnListModel` and `PandasCategoryListModel` also edit the frame;
`PandasIndexListModel` shows (and edits) the index levels.

Models can share one frame through a DataFrameHolder: a change made through one model
(rename, dtype change, sorting, ...) shows up in all others.


## Proxies

Since working with pandas often means working with tables containing several hundred
thousands of rows, the default QSortFilterProxyModel does not work that well.

CutePandas includes several proxy models which try to improve this.

PandasStringColumnFilterProxyModel can be used to filter a column based on a search
string. Instead of looping over the cells, a filter index is built using NumPy
operations. This makes filtering super fast, even with several thousands of rows.
A quick benchmark showed an almost 100x (!!) performance increase compared to
QSortFilterProxyModel.

PandasEvalFilterProxyModel works in a similar way, but filtering is done by a
`DataFrame.eval` expression (example: 'a > 10' shows all rows where the value of
column "a" is greater than 10).

PandasMultiStringColumnFilterProxyModel shows the rows where several columns each
start with a given prefix.

The proxies compute their masks with pandas: they need a pandas frame.
"""

from .categorymodel import CategoryModel
from .dataframeholder import DataFrameHolder
from .dataframemodel import (
    CellChunk,
    DataFrameModel,
    HorizontalHeaderModel,
    VerticalHeaderModel,
)
from .pandascategorylistmodel import PandasCategoryListModel
from .pandascolumnlistmodel import PandasColumnListModel, PandasIndexListModel
from .pandasindexfilterproxymodel import (
    PandasEvalFilterProxyModel,
    PandasMultiStringColumnFilterProxyModel,
    PandasStringColumnFilterProxyModel,
)
from .schemamodel import SchemaModel


__all__ = [
    "CategoryModel",
    "CellChunk",
    "DataFrameHolder",
    "DataFrameModel",
    "HorizontalHeaderModel",
    "PandasCategoryListModel",
    "PandasColumnListModel",
    "PandasEvalFilterProxyModel",
    "PandasIndexListModel",
    "PandasMultiStringColumnFilterProxyModel",
    "PandasStringColumnFilterProxyModel",
    "SchemaModel",
    "VerticalHeaderModel",
]
