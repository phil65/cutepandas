from __future__ import annotations

import numpy as np
import pandas as pd
from prettyqt import core, itemmodels, widgets

from cutepandas import pandasmodels, pandaswidgets


class DataFrameWidget(widgets.MainWindow):
    current_index_changed = core.Signal(core.ModelIndex)  # ty: ignore[missing-argument]
    current_row_changed = core.Signal(int)  # ty: ignore[missing-argument]

    def __init__(
        self,
        df: pd.DataFrame | None = None,
        object_name: str = "pandas_window",
        **kwargs,
    ):
        super().__init__(object_name=object_name, **kwargs)

        self.index_table = widgets.TableView(selection_behavior="rows")
        self.column_table = widgets.TableView(selection_behavior="rows")
        self.data_table = pandaswidgets.DataFrameViewer(df)
        self.setCentralWidget(self.data_table)
        self.category_table = widgets.TableView(selection_behavior="rows")
        self.attrs_table = widgets.TreeView(
            selection_behavior="rows", root_is_decorated=False
        )
        self.add_dockwidget(self.index_table, window_title="Index list")
        self.add_dockwidget(self.column_table, window_title="Column list")
        self.add_dockwidget(self.category_table, window_title="Category list")
        self.add_dockwidget(self.attrs_table, window_title="MetaData")
        self._set_list_models(df)

    def set_df(self, df: pd.DataFrame | None):
        self.data_table.set_df(df)
        self._set_list_models(df)

    def _set_list_models(self, df: pd.DataFrame | None):
        attrs = df.attrs if df is not None else {}
        self.attrs_table.set_model(itemmodels.JsonModel(attrs, show_root=False))
        self.index_table.set_model(pandasmodels.PandasIndexListModel(df))
        self.column_table.set_model(pandasmodels.PandasColumnListModel(df))
        # A new model comes with a new selection model: connect it again.
        self.column_table.selectionModel().currentChanged.connect(self._on_current_change)
        self.category_table.set_model(None)

    def _on_current_change(self, new: core.ModelIndex, old: core.ModelIndex):
        model = self.column_table.get_model(skip_proxies=True)
        if not new.isValid() or model.df is None:
            self.category_table.set_model(None)
            return
        column_name = model.df.columns[new.row()]
        category_model = pandasmodels.PandasCategoryListModel(model.df, column_name)
        self.category_table.set_model(category_model)


if __name__ == "__main__":
    app = widgets.app()
    app.set_style("Fusion")
    tuples = [
        ("bar", "one", "q"),
        ("bar", "two", "q"),
        ("baz", "one", "q"),
        ("baz", "two", "q"),
        ("foo", "one", "q"),
        ("foo", "two", "q"),
        ("qux", "one", "q"),
        ("qux", "two", "q"),
    ]
    index = pd.MultiIndex.from_tuples(tuples, names=["first", "second", "third"])
    df = pd.DataFrame(np.random.randn(8, 8), index=index, columns=index)
    df.attrs = {"test": "test"}
    widget = DataFrameWidget(df)
    widget.show()
    app.exec()
