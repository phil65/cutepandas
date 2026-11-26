from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from prettyqt import core, itemmodels, widgets

from cutepandas import delegates, pandasmodels, pandaswidgets
from cutepandas.utils.i18n import N_, TranslateMixin


if TYPE_CHECKING:
    from cutepandas.utils.i18n import Translate


logger = logging.getLogger(__name__)


class DataFrameWidget(TranslateMixin, widgets.MainWindow):
    """Viewer plus docks: index levels, columns, categories, column profile, attrs.

    Shows pandas, pyarrow and polars frames. For pandas frames the column and
    category lists edit the frame (rename, dtype, reorder, ...); for other backends
    they are read-only and the index list stays empty.

    Dock titles and the labels of the panels go through `translate` (`str -> str`);
    `set_translate()` / `retranslate()` switch the language at runtime.
    """

    DOCK_TITLES = (
        N_("Index list"),
        N_("Column list"),
        N_("Category list"),
        N_("Column profile"),
        N_("MetaData"),
    )

    current_index_changed = core.Signal(core.ModelIndex)  # ty: ignore[missing-argument]
    current_row_changed = core.Signal(int)  # ty: ignore[missing-argument]

    def __init__(
        self,
        df: Any = None,
        object_name: str = "pandas_window",
        *,
        translate: Translate | None = None,
        **kwargs,
    ):
        super().__init__(object_name=object_name, **kwargs)
        self._translate = translate

        # Every panel works on this holder, so an edit in one shows up in all of them.
        self.holder = pandasmodels.DataFrameHolder(df, parent=self)
        self._is_pandas: bool | None = None
        self.index_table = widgets.TableView(selection_behavior="rows")
        self.column_table = widgets.TableView(selection_behavior="rows")
        self.column_table.set_delegate(delegates.DTypeDelegate(), column=1)
        self.data_table = pandaswidgets.DataFrameViewer(self.holder)
        self.setCentralWidget(self.data_table)
        self.category_table = widgets.TableView(selection_behavior="rows")
        self.profile = pandaswidgets.ColumnProfileWidget(translate=translate)
        self.attrs_table = widgets.TreeView(
            selection_behavior="rows", root_is_decorated=False
        )
        panels = (
            self.index_table,
            self.column_table,
            self.category_table,
            self.profile,
            self.attrs_table,
        )
        self.docks = [self.add_dockwidget(panel) for panel in panels]
        self.holder.changed.connect(self._on_df_changed)
        self._on_df_changed()
        self.retranslate()

    def retranslate(self):
        """Re-show dock titles and panel labels with the current `translate`."""
        for dock, title in zip(self.docks, self.DOCK_TITLES):
            dock.setWindowTitle(self._tr(title))
        self.profile.set_translate(self._translate)
        for view in (self.index_table, self.column_table, self.category_table):
            if isinstance(model := view.model(), TranslateMixin):
                model.set_translate(self._translate)

    @property
    def df(self) -> Any:
        return self.holder.df

    def set_df(self, df: Any):
        self.holder.df = df

    def _setup_models(self, is_pandas: bool):
        """(Re)create the list models when the backend changes."""
        self._is_pandas = is_pandas
        for view in (self.index_table, self.column_table):
            if (old := view.model()) is not None:
                old.deleteLater()
        if is_pandas:
            index_model = pandasmodels.PandasIndexListModel(
                self.holder, translate=self._translate
            )
            column_model = pandasmodels.PandasColumnListModel(
                self.holder, translate=self._translate
            )
        else:
            index_model = None
            column_model = pandasmodels.SchemaModel(
                self.holder, translate=self._translate
            )
        self.index_table.set_model(index_model)
        self.column_table.set_model(column_model)
        self.column_table.selectionModel().currentChanged.connect(self._on_current_change)

    def _on_df_changed(self, origin=None):
        df = self.df
        is_pandas = isinstance(df, pd.DataFrame)
        if is_pandas != self._is_pandas:
            self._setup_models(is_pandas)
        attrs = df.attrs if is_pandas else {}
        self.attrs_table.set_model(itemmodels.JsonModel(attrs, show_root=False))

    def _on_current_change(self, new: core.ModelIndex, old: core.ModelIndex):
        source = self.holder.source
        model = None
        if new.isValid() and source is not None and new.row() < len(source.columns):
            column_name = source.columns[new.row()].name
            cls = (
                pandasmodels.PandasCategoryListModel
                if self._is_pandas
                else pandasmodels.CategoryModel
            )
            model = cls(self.holder, column_name, translate=self._translate, parent=self)
            try:
                self.profile.set_column(self.holder, new.row())
            except Exception:
                logger.exception("Could not profile column %r", column_name)
                self.profile.clear()
        else:
            self.profile.clear()
        # Delete the previous model: it is connected to the holder and would
        # otherwise stay alive (and keep resetting) for the widget's lifetime.
        if (old_model := self.category_table.model()) is not None:
            old_model.deleteLater()
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
