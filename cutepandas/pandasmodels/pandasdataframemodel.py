from __future__ import annotations

import enum
import logging
from typing import Any

import numpy as np
import pandas as pd
from prettyqt import constants, core

from cutepandas.pandasmodels.dataframeholder import DataFrameHolder, DataFrameHolderMixin


logger = logging.getLogger(__name__)


def _is_missing(value: Any) -> bool:
    """Return True for scalar NA values (None, NaN, NaT, pd.NA); False for containers."""
    return bool(pd.api.types.is_scalar(value) and pd.isna(value))


def _to_python(value: Any) -> Any:
    """Convert NumPy scalars to builtins so Qt can wrap them in a QVariant."""
    return value.item() if isinstance(value, np.generic) else value


class PandasDataFrameModel(DataFrameHolderMixin, core.AbstractTableModel):
    """Table model for pandas DataFrames."""

    SUPPORTS = pd.DataFrame

    class Roles(enum.IntEnum):
        """Role names."""

        DTypeRole = constants.USER_ROLE + 4555
        ColumnNameRole = constants.USER_ROLE + 4556

    def __init__(
        self,
        df: pd.DataFrame | pd.Series | DataFrameHolder | None,
        *args,
        **kwargs,
    ):
        """Pass a DataFrameHolder to share the frame with other models.

        A Series is shown as a one-column frame.
        """
        self._multiindex_separator = " | "
        super().__init__(*args, **kwargs)
        self._init_holder(df)

    @classmethod
    def supports(cls, instance) -> bool:
        return isinstance(instance, pd.DataFrame)

    def headerData(
        self,
        section: int,
        orientation: constants.Orientation,
        role: constants.ItemDataRole = constants.DISPLAY_ROLE,
    ):
        df = self.df
        match role, orientation:
            case constants.ALIGNMENT_ROLE, constants.HORIZONTAL:
                return constants.ALIGN_CENTER | constants.ALIGN_BOTTOM
            case constants.DISPLAY_ROLE, constants.HORIZONTAL if df is not None:
                header = df.columns[section]
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
        df = self.df  # hot path: one lookup through the holder per call
        if df is None or not index.isValid():
            return None
        match role:
            case constants.DISPLAY_ROLE:
                cell = df.iat[index.row(), index.column()]
                return "" if _is_missing(cell) else str(cell)
            case constants.EDIT_ROLE:
                cell = df.iat[index.row(), index.column()]
                return None if _is_missing(cell) else _to_python(cell)
            case constants.TOOLTIP_ROLE:
                cell = df.iat[index.row(), index.column()]
                return "NaN" if _is_missing(cell) else str(cell)
            case self.Roles.DTypeRole:
                return df.dtypes.iloc[index.column()]
            case constants.USER_ROLE:
                return df.iat[index.row(), index.column()]
            case self.Roles.ColumnNameRole:
                return df.columns[index.column()]
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
        df = self.df
        if df is None or not index.isValid():
            return False
        match role:
            case constants.USER_ROLE | constants.EDIT_ROLE:
                # In place: every model sharing the holder sees the same object.
                try:
                    df.iat[index.row(), index.column()] = value
                except Exception:
                    logger.exception("Could not set %r at %s", value, index)
                    return False
                else:
                    self.dataChanged.emit(index, index)
                    return True
        return False

    def sort(self, ncol: int, order: constants.SortOrder = constants.ASCENDING):
        df = self.df
        if df is None or not 0 <= ncol < len(df.columns):
            return
        # By position (labels are ambiguous for duplicate columns) and stable, so
        # sorting by one column after another keeps the earlier order for ties.
        column = df.iloc[:, ncol]
        if order == constants.ASCENDING:
            positions = column.argsort(kind="stable").to_numpy()  # fastest; NaN last
        else:
            # argsort can't sort descending; this keeps ties stable and NaN last
            ranked = column.reset_index(drop=True)
            ranked = ranked.sort_values(ascending=False, kind="stable")
            positions = ranked.index.to_numpy()
        with self.change_layout():
            self.df = df.take(positions)

    multiindex_separator = core.Property(
        str,
        get_multiindex_separator,
        set_multiindex_separator,
        doc="Separator to use for MultiIndexes",
    )


class VerticalHeaderModel(DataFrameHolderMixin, core.AbstractTableModel):
    def __init__(self, df: pd.DataFrame | DataFrameHolder | None, **kwargs):
        super().__init__(**kwargs)
        self._init_holder(df)

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
        df = self.df
        if df is None or not index.isValid():
            return None
        match role:
            case constants.DISPLAY_ROLE | constants.TOOLTIP_ROLE:
                value = df.index[index.row()]
                if isinstance(df.index, pd.MultiIndex):
                    value = value[index.column()]
                return str(value)
        return None

    def headerData(
        self,
        section: int,
        orientation: constants.Orientation,
        role: constants.ItemDataRole = constants.DISPLAY_ROLE,
    ) -> str | None:
        df = self.df
        match role, orientation:
            case constants.DISPLAY_ROLE | constants.TOOLTIP_ROLE, constants.HORIZONTAL:
                if df is None or not 0 <= section < df.index.nlevels:
                    return None
                return str(df.index.names[section])
        return None


class HorizontalHeaderModel(DataFrameHolderMixin, core.AbstractTableModel):
    def __init__(self, df: pd.DataFrame | DataFrameHolder | None, **kwargs):
        super().__init__(**kwargs)
        self._init_holder(df)

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
        df = self.df
        if df is None or not index.isValid():
            return None
        match role:
            case constants.DISPLAY_ROLE | constants.TOOLTIP_ROLE:
                value = df.columns[index.column()]
                if isinstance(df.columns, pd.MultiIndex):
                    value = value[index.row()]
                return str(value)
            case constants.ALIGNMENT_ROLE:
                return constants.ALIGN_CENTER
        return None

    def headerData(
        self,
        section: int,
        orientation: constants.Orientation,
        role: constants.ItemDataRole = constants.DISPLAY_ROLE,
    ):
        df = self.df
        match role, orientation:
            case constants.DISPLAY_ROLE | constants.TOOLTIP_ROLE, constants.VERTICAL:
                if df is None or not 0 <= section < df.columns.nlevels:
                    return None
                return str(df.columns.names[section])
        return None


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
