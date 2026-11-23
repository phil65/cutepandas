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

        # Every panel works on this holder, so an edit in one shows up in all of them.
        self.holder = pandasmodels.DataFrameHolder(df, parent=self)
        self.index_table = widgets.TableView(selection_behavior="rows")
        self.index_table.set_model(pandasmodels.PandasIndexListModel(self.holder))
        self.column_table = widgets.TableView(selection_behavior="rows")
        self.column_table.set_model(pandasmodels.PandasColumnListModel(self.holder))
        self.column_table.selectionModel().currentChanged.connect(self._on_current_change)
        self.data_table = pandaswidgets.DataFrameViewer(self.holder)
        self.setCentralWidget(self.data_table)
        self.category_table = widgets.TableView(selection_behavior="rows")
        self.attrs_table = widgets.TreeView(
            selection_behavior="rows", root_is_decorated=False
        )
        self.add_dockwidget(self.index_table, window_title="Index list")
        self.add_dockwidget(self.column_table, window_title="Column list")
        self.add_dockwidget(self.category_table, window_title="Category list")
        self.add_dockwidget(self.attrs_table, window_title="MetaData")
        self.holder.changed.connect(self._on_df_changed)
        self._on_df_changed()

    @property
    def df(self) -> pd.DataFrame | None:
        return self.holder.df

    def set_df(self, df: pd.DataFrame | None):
        self.holder.df = df

    def _on_df_changed(self, origin=None):
        attrs = self.df.attrs if self.df is not None else {}
        self.attrs_table.set_model(itemmodels.JsonModel(attrs, show_root=False))

    def _on_current_change(self, new: core.ModelIndex, old: core.ModelIndex):
        df = self.df
        model = None
        if new.isValid() and df is not None and new.row() < len(df.columns):
            column_name = df.columns[new.row()]
            model = pandasmodels.PandasCategoryListModel(
                self.holder, column_name, parent=self
            )
        # Delete the previous model: it is connected to the holder and would
        # otherwise stay alive (and keep resetting) for the widget's lifetime.
        if (old := self.category_table.model()) is not None:
            old.deleteLater()
        self.category_table.set_model(model)


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
