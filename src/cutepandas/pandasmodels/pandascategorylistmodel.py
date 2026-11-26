import logging
from typing import TYPE_CHECKING, Any

import pandas as pd
from prettyqt import constants, core
from prettyqt.utils import helpers

from cutepandas.pandasmodels.categorymodel import CategoryModel


if TYPE_CHECKING:
    from cutepandas.pandasmodels.dataframeholder import DataFrameHolder


logger = logging.getLogger(__name__)


class PandasCategoryListModel(CategoryModel):
    """`CategoryModel` for a pandas `category` column that also edits it.

    Rename, reorder (drag and drop, sorting), add categories, toggle `ordered`,
    remove unused ones. `USER_ROLE` returns the rows of a category.
    """

    MIME_TYPE = "mime_categories"

    def __init__(
        self,
        df: pd.DataFrame | DataFrameHolder | None = None,
        col=None,
        parent=None,
        **kwargs,
    ):
        super().__init__(df, col, parent=parent, **kwargs)

    @property
    def series(self):
        assert self.df is not None
        return self.df[self.col]

    @series.setter
    def series(self, series):
        assert self.df is not None
        # A new frame (instead of df[col] = ...) so shared models get notified.
        # With copy-on-write the shallow copy doesn't touch the caller's frame.
        df = self.df.copy(deep=False)
        df[self.col] = series
        self.df = df

    def setData(
        self,
        index: core.ModelIndex,
        value: Any,
        role: constants.ItemDataRole = constants.EDIT_ROLE,
    ) -> bool:
        if not index.isValid():
            return False
        if role == constants.EDIT_ROLE and index.column() == 0:
            if value in self.series.cat.categories:
                logger.error("Error: Category names must be unique.")
                return False
            rename_dct = {index.data(constants.NAME_ROLE): value}
            with self.change_layout():
                self.series = self.series.cat.rename_categories(rename_dct)
            return True
        return False

    def flags(self, index: core.ModelIndex) -> constants.ItemFlag:
        if not index.isValid():
            return constants.DROP_ENABLED
        return self.DEFAULT_FLAGS | constants.IS_EDITABLE

    def data(
        self,
        index: core.ModelIndex,
        role: constants.ItemDataRole = constants.DISPLAY_ROLE,
    ):
        if role == constants.USER_ROLE and index.isValid():
            category = self.categories[index.row()]
            assert self.df is not None
            return self.df[self.series == category]
        return super().data(index, role)

    def mimeData(self, indexes):
        mime_data = core.MimeData()
        data = [i.row() for i in indexes if i.column() == 0]
        mime_data.set_json_data(self.MIME_TYPE, data)
        return mime_data

    def mimeTypes(self):
        return [self.MIME_TYPE]

    def dropMimeData(self, mime_data, action, row, column, parent_index):
        if parent_index.isValid():
            return False

        if mime_data.hasFormat(self.MIME_TYPE):
            indexes = mime_data.get_json_data(self.MIME_TYPE)
            cats = self.series.cat.categories.tolist()
            cats = helpers.move_in_list(cats, indexes, row)
            self.reorder_categories(cats)
            return False
        return None

    def sort(self, ncol, order):
        if ncol != 0:
            return
        is_descending = order == constants.DESCENDING
        new_order = sorted(self.series.cat.categories, reverse=is_descending)
        self.reorder_categories(new_order)

    def reorder_categories(self, order):
        with self.change_layout():
            self.series = self.series.cat.reorder_categories(new_categories=order)

    def add_category(self, new_categories):
        if not pd.api.types.is_list_like(new_categories):
            new_categories = [new_categories]
        with self.append_rows(len(new_categories)):
            self.series = self.series.cat.add_categories(new_categories=new_categories)

    def toggle_ordered(self):
        with self.change_layout():
            if self.series.cat.ordered:
                self.series = self.series.cat.set_unordered()
            else:
                self.series = self.series.cat.set_ordered()

    def remove_unused(self):
        with self.change_layout():
            self.series = self.series.cat.remove_unused()


if __name__ == "__main__":
    from prettyqt import widgets

    app = widgets.app()
    # Prepare data
    df = pd.DataFrame({"A": ["a", "b", "c", "a"]})

    df["B"] = df["A"].astype("category")
    model = PandasCategoryListModel(df, "B")
    tableview = widgets.TreeView()
    tableview.setup_dragdrop_move()
    # tableview.set_delegate("editor")
    tableview.set_model(model)

    tableview.show()
    app.exec()
