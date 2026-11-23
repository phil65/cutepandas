from __future__ import annotations

import enum
import logging
from typing import Any

import numpy as np
import pandas as pd
from prettyqt import constants, core


logger = logging.getLogger(__name__)


def _is_missing(value: Any) -> bool:
    """Return True for scalar NA values (None, NaN, NaT, pd.NA); False for containers."""
    return bool(pd.api.types.is_scalar(value) and pd.isna(value))


def _to_python(value: Any) -> Any:
    """Convert NumPy scalars to builtins so Qt can wrap them in a QVariant."""
    return value.item() if isinstance(value, np.generic) else value


class PandasDataFrameModel(core.AbstractTableModel):
    """Table model for pandas DataFrames."""

    SUPPORTS = pd.DataFrame

    class Roles(enum.IntEnum):
        """Role names."""

        DTypeRole = constants.USER_ROLE + 4555
        ColumnNameRole = constants.USER_ROLE + 4556

    def __init__(self, df: pd.DataFrame | pd.Series | None, *args, **kwargs):
        self._multiindex_separator = " | "
        # A Series is shown as a one-column frame, so all positional access is 2D.
        self.df = df.to_frame() if isinstance(df, pd.Series) else df
        super().__init__(*args, **kwargs)

    @classmethod
    def supports(cls, instance) -> bool:
        return isinstance(instance, pd.DataFrame)

    def headerData(
        self,
        section: int,
        orientation: constants.Orientation,
        role: constants.ItemDataRole = constants.DISPLAY_ROLE,
    ):
        match role, orientation:
            case constants.ALIGNMENT_ROLE, constants.HORIZONTAL:
                return constants.ALIGN_CENTER | constants.ALIGN_BOTTOM
            case constants.DISPLAY_ROLE, constants.HORIZONTAL:
                header = self.df.columns[section]
                if isinstance(header, tuple | list):
                    return self._multiindex_separator.join(str(i) for i in header)
                return str(header)
                # label = helpers.format_name(self.df.columns[section])
                # return f"{label}\n{self.df.iloc[:, section].dtype}"
            # case constants.DECORATION_ROLE, constants.HORIZONTAL if self.show_icons:
            #     dtype = self.df.iloc[:, section].dtype
            #     return icons.icon_for_dtype(dtype)
            case _, _:
                return None

    def set_multiindex_separator(self, separator: str):
        self._multiindex_separator = separator

    def get_multiindex_separator(self) -> str:
        return self._multiindex_separator

    def columnCount(self, parent: core.ModelIndex | None = None) -> int:
        if self.df is None or (parent is not None and parent.isValid()):
            return 0
        return len(self.df.columns)

    def rowCount(self, parent: core.ModelIndex | None = None) -> int:
        if self.df is None or (parent is not None and parent.isValid()):
            return 0
        return len(self.df.index)

    def data(
        self,
        index: core.ModelIndex,
        role: constants.ItemDataRole = constants.DISPLAY_ROLE,
    ):
        if not index.isValid() or self.df is None:
            return None
        match role:
            case constants.DISPLAY_ROLE:
                cell = self.df.iat[index.row(), index.column()]
                return "" if _is_missing(cell) else str(cell)
            case constants.EDIT_ROLE:
                cell = self.df.iat[index.row(), index.column()]
                return None if _is_missing(cell) else _to_python(cell)
            case constants.TOOLTIP_ROLE:
                cell = self.df.iat[index.row(), index.column()]
                return "NaN" if _is_missing(cell) else str(cell)
            case self.Roles.DTypeRole:
                return self.df.iloc[:, index.column()].dtype
            case constants.USER_ROLE:
                return self.df.iat[index.row(), index.column()]
            case self.Roles.ColumnNameRole:
                return self.df.columns[index.column()]
        return None

    def flags(self, index: core.ModelIndex) -> constants.ItemFlag:
        if not index.isValid():
            return constants.NO_FLAGS
        return constants.IS_EDITABLE | constants.IS_ENABLED | constants.IS_SELECTABLE

    #     return cur_flags if self.is_read_only else cur_flags | constants.IS_EDITABLE

    def setData(
        self,
        index: core.ModelIndex,
        value: Any,
        role: constants.ItemDataRole = constants.EDIT_ROLE,
    ) -> bool:
        match role:
            case constants.USER_ROLE | constants.EDIT_ROLE:
                row = index.row()
                col = index.column()
                try:
                    self.df.iat[row, col] = value
                except Exception as e:
                    logger.exception(e)
                    return False
                else:
                    self.dataChanged.emit(index, index)
                    return True
        return False

    def sort(self, ncol: int, order: constants.SortOrder = constants.ASCENDING):
        if self.df is None or not 0 <= ncol < len(self.df.columns):
            return
        is_ascending = order == constants.ASCENDING
        with self.change_layout():
            # sort by position: label-based sorting is ambiguous for duplicate columns
            column = self.df.iloc[:, ncol].reset_index(drop=True)
            column = column.sort_values(ascending=is_ascending, kind="stable")
            self.df = self.df.iloc[column.index.to_numpy()]

    multiindex_separator = core.Property(
        str,
        get_multiindex_separator,
        set_multiindex_separator,
        doc="Separator to use for MultiIndexes",
    )


class VerticalHeaderModel(core.AbstractTableModel):
    def __init__(self, df: pd.DataFrame | None, **kwargs):
        super().__init__(**kwargs)
        self.df = df

    def columnCount(self, parent: core.ModelIndex | None = None) -> int:
        if self.df is None or (parent is not None and parent.isValid()):
            return 0
        return self.df.index.nlevels

    def rowCount(self, parent: core.ModelIndex | None = None) -> int:
        if self.df is None or (parent is not None and parent.isValid()):
            return 0
        return self.df.index.shape[0]

    def data(
        self,
        index: core.ModelIndex,
        role: constants.ItemDataRole = constants.DISPLAY_ROLE,
    ):
        match role:
            case constants.DISPLAY_ROLE | constants.TOOLTIP_ROLE:
                row = index.row()
                col = index.column()
                if isinstance(self.df.index, pd.MultiIndex):
                    return str(self.df.index.values[row][col])
                else:
                    return str(self.df.index.values[row])
        return None

    def headerData(
        self,
        section: int,
        orientation: constants.Orientation,
        role: constants.ItemDataRole = constants.DISPLAY_ROLE,
    ) -> str | None:
        match role, orientation:
            case constants.DISPLAY_ROLE | constants.TOOLTIP_ROLE, constants.HORIZONTAL:
                if isinstance(self.df.index, pd.MultiIndex):
                    return str(self.df.index.names[section])
                else:
                    return str(self.df.index.name)
        return None


class HorizontalHeaderModel(core.AbstractTableModel):
    def __init__(self, df: pd.DataFrame | None, **kwargs):
        super().__init__(**kwargs)
        self.df = df

    def columnCount(self, parent: core.ModelIndex | None = None) -> int:
        if self.df is None or (parent is not None and parent.isValid()):
            return 0
        return self.df.columns.shape[0]

    def rowCount(self, parent: core.ModelIndex | None = None) -> int:
        if self.df is None or (parent is not None and parent.isValid()):
            return 0
        return self.df.columns.nlevels

    def data(
        self,
        index: core.ModelIndex,
        role: constants.ItemDataRole = constants.DISPLAY_ROLE,
    ):
        match role:
            case constants.DISPLAY_ROLE | constants.TOOLTIP_ROLE:
                row = index.row()
                col = index.column()
                if isinstance(self.df.columns, pd.MultiIndex):
                    return str(self.df.columns.values[col][row])
                else:
                    return str(self.df.columns.values[col])
            case constants.ALIGNMENT_ROLE:
                return constants.ALIGN_CENTER

    def headerData(
        self,
        section: int,
        orientation: constants.Orientation,
        role: constants.ItemDataRole = constants.DISPLAY_ROLE,
    ):
        match role, orientation:
            case constants.DISPLAY_ROLE | constants.TOOLTIP_ROLE, constants.VERTICAL:
                if isinstance(self.df.columns, pd.MultiIndex):
                    return str(self.df.columns.names[section])
                else:
                    return str(self.df.columns.name)


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
    ] * 10
    index = pd.MultiIndex.from_tuples(tuples, names=["first", "second", "third"])
    df = pd.DataFrame(np.random.randn(60, 60), index=index, columns=index)
    table = widgets.TableView()
    table.show()
    model = PandasDataFrameModel(df, parent=table)
    model = model.proxifier[6:7, ::2].style("red", role=constants.FOREGROUND_ROLE)
    model = model.proxifier[1:4, ::2].style(
        "Courier", role=constants.FONT_ROLE, point_size=30
    )
    # model = model.proxifier[1:4, ::2].filter()
    table.set_model(model)
    print(table.get_models())
    # table.set_model(table.model().proxifier.get_proxy("color_values", parent=table))
    # model = model.proxifier.get_proxy("read_only")
    table.set_delegate("editor")

    app.exec()
