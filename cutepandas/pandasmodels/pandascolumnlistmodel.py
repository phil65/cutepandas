from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from prettyqt import constants, core, itemmodels
from prettyqt.utils import helpers

from cutepandas.pandasmodels.dataframeholder import DataFrameHolder, DataFrameHolderMixin
from cutepandas.pandasmodels.schemamodel import SchemaModel
from cutepandas.utils.i18n import N_, TranslatableHeaderMixin


if TYPE_CHECKING:
    from cutepandas.utils.i18n import Translate


logger = logging.getLogger(__name__)


class PandasColumnListModel(SchemaModel):
    """`SchemaModel` for pandas frames that also edits them.

    Rename (column 0), change the dtype (column 1, see `DTypeDelegate`), remove,
    reorder by drag and drop or by sorting, append columns and move index levels into
    the columns (drop from `PandasIndexListModel`).
    """

    MIME_TYPE = "mime_columns"
    content_type = "features"

    def mimeData(self, indexes):
        mime_data = core.MimeData()
        data = [i.row() for i in indexes if i.column() == 0]
        mime_data.set_json_data(self.MIME_TYPE, data)
        return mime_data

    def dropMimeData(self, mime_data, action, row, column, parent_index):
        df = self.df
        if parent_index.isValid() or df is None:
            # Since we only drop in between items, parent_index must be invalid
            return False
        if mime_data.hasFormat(self.MIME_TYPE):
            data = mime_data.get_json_data(self.MIME_TYPE)
            positions = helpers.move_in_list(list(range(len(df.columns))), data, row)
            with self.change_layout():
                self.df = df.iloc[:, positions]
            self.update_all()
            return False

        if mime_data.hasFormat("mime_indexes"):
            data = mime_data.get_json_data("mime_indexes")
            with self.reset_model():
                self.df = df.reset_index(level=data)
            self.update_all()
        return False

    def mimeTypes(self):
        return [self.MIME_TYPE]

    def setData(
        self,
        index: core.ModelIndex,
        value: Any,
        role: constants.ItemDataRole = constants.EDIT_ROLE,
    ) -> bool:
        if not index.isValid() or self.df is None:
            return False
        match role, index.column():
            case constants.EDIT_ROLE, 0:
                return self._rename_column(index.row(), value)
            case constants.EDIT_ROLE, 1:
                return self._set_dtype(index.row(), value)
            case constants.USER_ROLE, _:
                return self._replace_column(index.row(), value)
            case self.Roles.DTypeRole, _:
                return self._set_dtype(index.row(), value)
        return False

    def _replace_column(self, pos: int, values: Any) -> bool:
        df = self.df
        assert df is not None
        # Build the new frame first: assigning self.df notifies the other models.
        new = df.copy(deep=False)
        new.isetitem(pos, values)
        self.df = new
        self.update_all()
        return True

    def _set_dtype(self, pos: int, dtype: Any) -> bool:
        df = self.df
        assert df is not None
        try:
            column = df.iloc[:, pos].astype(dtype)
        except TypeError, ValueError:
            logger.exception("Could not convert column %r to %r", pos, dtype)
            return False
        return self._replace_column(pos, column)

    def flags(self, index: core.ModelIndex) -> constants.ItemFlag:
        if not index.isValid():
            return constants.DROP_ENABLED
        if index.column() in {0, 1}:
            return self.DEFAULT_FLAGS | constants.IS_EDITABLE
        return self.DEFAULT_FLAGS

    def removeRows(self, row, count, parent):
        if self.df is None or count <= 0 or row < 0 or row + count > self.rowCount():
            return False
        keep = [i for i in range(len(self.df.columns)) if not row <= i < row + count]
        with self.remove_rows(row, row + count - 1, parent):
            self.df = self.df.iloc[:, keep]
        return True

    def _rename_column(self, pos: int, name: Any) -> bool:
        df = self.df
        assert df is not None
        if isinstance(df.columns, pd.MultiIndex) or name in df.columns:
            return False  # a label can't replace a tuple; names stay unique
        labels = list(df.columns)
        labels[pos] = name
        self.df = df.set_axis(labels, axis="columns")
        self.update_all()
        return True

    def data(
        self,
        index: core.ModelIndex,
        role: constants.ItemDataRole = constants.DISPLAY_ROLE,
    ):
        if index.isValid() and role == constants.USER_ROLE and self.df is not None:
            return self.data_by_index(index)
        return super().data(index, role)

    def sort(self, ncol: int, order: constants.SortOrder = constants.ASCENDING):
        if self.df is None:
            return
        if not 0 <= ncol < len(self.HEADER):
            return
        keys = [
            self.data(self.index(row, ncol), constants.SORT_ROLE)
            for row in range(self.rowCount())
        ]
        positions = sorted(
            range(len(keys)),
            key=keys.__getitem__,
            reverse=order == constants.DESCENDING,
        )
        with self.change_layout():
            # positional, so duplicate column labels survive the reorder
            self.df = self.df.iloc[:, positions]

    def data_by_index(self, index: core.ModelIndex):
        df = self.df
        assert df is not None
        return df.iloc[:, index.row()]

    def add_feature(self, series: pd.Series):
        """Append a Series as a new column (named after the Series)."""
        df = self.df
        assert df is not None
        new = df.copy(deep=False)
        with self.append_rows(1):
            new[series.name] = series
            self.df = new


def label_for_index(index: pd.Index) -> str:
    """English label of the index type (a msgid, see `cutepandas.utils.i18n`)."""
    match index:
        case pd.RangeIndex():
            return N_("Range")
        case pd.DatetimeIndex():
            return N_("Datetime")
        case pd.CategoricalIndex():
            return N_("Categorical")
        case pd.IntervalIndex():
            return N_("Interval")
        case pd.TimedeltaIndex():
            return N_("Timedelta")
        case pd.MultiIndex():
            return N_("Multi-index")
        case pd.PeriodIndex():
            return N_("Period")
        case _:
            return N_("Regular")


class PandasIndexListModel(
    TranslatableHeaderMixin,
    DataFrameHolderMixin,
    itemmodels.ModelMixin,
    core.AbstractTableModel,
):
    """Index levels of a pandas frame: name, index type, monotonic (editable).

    Labels go through `translate` (see `cutepandas.utils.i18n`).
    """

    HEADER = [N_("Name"), N_("Index type"), N_("Monotonic")]
    MIME_TYPE = "mime_indexes"

    def __init__(
        self,
        ds: pd.DataFrame | DataFrameHolder | None = None,
        *,
        translate: Translate | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._translate = translate
        self._init_holder(ds)

    def setData(
        self,
        index: core.ModelIndex,
        value: Any,
        role: constants.ItemDataRole = constants.EDIT_ROLE,
    ) -> bool:
        if not index.isValid():
            return False
        assert self.df is not None
        match role:
            case constants.EDIT_ROLE if index.column() == 0:
                level = index.row() if isinstance(self.df.index, pd.MultiIndex) else None
                self._set_index(self.df.index.set_names(names=value, level=level))
                self.dataChanged.emit(index, index)
                return True
            case constants.USER_ROLE if not isinstance(self.df.index, pd.MultiIndex):
                with self.change_layout():
                    self._set_index(value)
                return True
            case self.DTYPE_ROLE:
                try:
                    new_index = self.df.index.astype(value)
                except TypeError, ValueError:
                    logger.exception("Could not convert index to %r", value)
                    return False
                with self.change_layout():
                    self._set_index(new_index)
                return True
        return False

    def removeRows(self, row, count, parent=None):
        """Remove index levels (the rows of this model) from the DataFrame."""
        if self.df is None or count <= 0 or row < 0 or row + count > self.rowCount():
            return False
        with self.remove_rows(row, row + count - 1, parent):
            # drop=True discards the level values instead of moving them into columns.
            # Removing every level leaves a fresh RangeIndex.
            self.df = self.df.reset_index(level=list(range(row, row + count)), drop=True)
        return True

    def flags(self, index: core.ModelIndex) -> constants.ItemFlag:
        if not index.isValid():
            return constants.DROP_ENABLED
        assert self.df is not None
        is_range = self.df.index.name is None and isinstance(self.df.index, pd.RangeIndex)
        match index.column(), is_range:
            case 0, True:
                # do not allow moving the "default" RangeIndex.
                return (
                    constants.IS_ENABLED | constants.IS_SELECTABLE | constants.IS_EDITABLE
                )
            case _, True:
                return constants.IS_ENABLED | constants.IS_SELECTABLE
            case 0 | 1, False:
                return self.DEFAULT_FLAGS | constants.IS_EDITABLE
            case _, False:
                return self.DEFAULT_FLAGS

    def data(
        self,
        index: core.ModelIndex,
        role: constants.ItemDataRole = constants.DISPLAY_ROLE,
    ):
        if not index.isValid():
            return None
        assert self.df is not None
        match role, index.column():
            # case constants.DECORATION_ROLE, 0:
            #     index = self.get_index(index.row())
            #     return icons.icon_for_index(index)

            case constants.TOOLTIP_ROLE, _:
                index = self.get_index(index.row())
                return (
                    f"<b>{index.name}</b><br>{self._tr(N_('Contains NaNs'))}: "
                    f"{index.hasnans}<br>"
                    # f"{_('Monotonic')}: {index.is_monotonic}"
                )

            case constants.DISPLAY_ROLE | constants.EDIT_ROLE, 0:
                index_name = self.df.index.names[index.row()]
                return index_name or self._tr(N_("[Unnamed]"))

            case constants.DISPLAY_ROLE | constants.EDIT_ROLE, 1:
                return self._tr(label_for_index(self.get_index(index.row())))

            case constants.DISPLAY_ROLE | constants.EDIT_ROLE, 2:
                index = self.get_index(index.row())
                if index.is_monotonic_increasing:
                    return self._tr(N_("Increasing"))
                elif index.is_monotonic_decreasing:
                    return self._tr(N_("Decreasing"))

            case constants.USER_ROLE, _:
                name = self.df.index.names[index.row()]
                return self.df.index.get_level_values(name)

            case constants.NAME_ROLE, _:
                return self.df.index.names[index.row()]

            case self.DTYPE_ROLE, _:
                return self.get_index(index.row()).dtype

    def rowCount(self, parent: core.ModelIndex | None = None) -> int:
        if self.df is None or (parent is not None and parent.isValid()):
            return 0
        return self.df.index.nlevels

    def columnCount(self, parent: core.ModelIndex | None = None) -> int:
        if parent is not None and parent.isValid():
            return 0
        return len(self.HEADER)

    def mimeData(self, indexes):
        mime_data = core.MimeData()
        data = [i.row() for i in indexes if i.column() == 0]
        mime_data.set_json_data(self.MIME_TYPE, data)
        return mime_data

    def dropMimeData(self, mime_data, action, row, column, parent_index):
        if parent_index.isValid():
            return False
        assert self.df is not None
        if mime_data.hasFormat(self.MIME_TYPE):
            data = mime_data.get_json_data(self.MIME_TYPE)
            names = list(self.df.index.names)
            names = helpers.move_in_list(names, data, row)
            with self.change_layout():
                self.df = self.df.reorder_levels(order=names)
        elif mime_data.hasFormat("mime_columns"):
            data = mime_data.get_json_data("mime_columns")
            cols = self.df.columns.tolist()
            to_append = [cols[i] for i in data]
            # TODO: make sure drop offset is taken into account
            with self.reset_model():
                self.df = self.df.set_index(keys=to_append, append=True)
        return False

    def mimeTypes(self):
        return [self.MIME_TYPE]

    def get_index(self, row: int = 0):
        assert self.df is not None
        idx = self.df.index
        # using get_level_values() is too slow for large datasets with multiindex
        return idx.levels[row] if isinstance(idx, pd.MultiIndex) else idx

    def _set_index(self, new_index):
        # A new frame (instead of assigning df.index) so shared models get notified.
        assert self.df is not None
        self.df = self.df.set_axis(new_index, axis="index")

    def to_datetime(self, level: int | None = None, fmt: str | None = None):
        """Convert the index (or one level of a MultiIndex) to datetimes."""
        assert self.df is not None
        index = self.df.index
        if isinstance(index, pd.MultiIndex):
            level = level or 0
            converted = pd.to_datetime(index.levels[level], format=fmt)
            new_index = index.set_levels(converted, level=level)
        else:
            new_index = pd.to_datetime(index, format=fmt)
        with self.change_layout():
            self._set_index(new_index)

    def filter_range(self, start: int, end: int):
        assert self.df is not None
        with self.change_layout():
            self.df = self.df.iloc[start:end]
        return True


if __name__ == "__main__":
    from prettyqt import widgets

    app = widgets.app()
    # Prepare data
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
    df = pd.DataFrame(np.random.randn(6, 6), index=index[:6], columns=index[:6])
    model = PandasColumnListModel(df)
    tableview = widgets.TreeView()
    tableview.setup_dragdrop_move()
    tableview.set_delegate("editor")
    tableview.set_model(model)

    tableview.show()
    app.exec()
