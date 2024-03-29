"""Pandas models.

CutePandas contains a range of tools to work with pandas data structures.


## Models

CutePandas includes multiple models to display Pandas dataframes.
One of the key "issues" is that in contrast to QAbstractTableModels, a pandas DataFrame can have MultiIndexes.
CutePandas offers several approaches to display these MultiIndexes.

PandasDataFrameModel is a very simple model which is displaying MultiIndexes by joining
the multiple index levels using a separator. (the separator value is exposed as a Qt Property)

In addition, two composed widgets are included which consist of three different tables (one for data, one for the index, one for the headers) which are synced on scrolling / resizing.

There is also a model to get a detailed view on an index and another one to display the categories of
a pandas category column.


## Proxies
Since working with pandas often means working with tables containing several hundred thousands of rows,
the default QSortFilterProxyModel does not work that well.

CutePandas includes several proxy models which try to improve this.

PandasStringColumnFilterProxyModel can be used to filter a column based on a search string.
Instead of looping over the cells, a filter index is built using NumPy operations.
This makes filtering super fast, even with several thousands of rows.
A quick benchmark showed an almost 100x (!!) performance increase compared to QSortFilterProxyModel.

PandasEvalFilterProxyModel works in a similar way, but filtering is done by a Python statement.
(example: '"a" > 10' would show all rows where the value of column "a" is greater than 10.)

To display heatmaps, CutePandas also includes a proxy model to color the cells according to their values.
That proxy model includes several modes, also including modes which dont need to pre-compute min-max values,
"""

from .pandascategorylistmodel import PandasCategoryListModel
from .pandascolumnlistmodel import PandasColumnListModel, PandasIndexListModel
from .pandasdataframemodel import (
    HorizontalHeaderModel,
    PandasDataFrameModel,
    VerticalHeaderModel,
)
from .pandasindexfilterproxymodel import (
    PandasEvalFilterProxyModel,
    PandasMultiStringColumnFilterProxyModel,
    PandasStringColumnFilterProxyModel,
)

__all__ = [
    "PandasCategoryListModel",
    "PandasColumnListModel",
    "PandasIndexListModel",
    "PandasDataFrameModel",
    "VerticalHeaderModel",
    "HorizontalHeaderModel",
    "PandasEvalFilterProxyModel",
    "PandasStringColumnFilterProxyModel",
    "PandasMultiStringColumnFilterProxyModel",
]
