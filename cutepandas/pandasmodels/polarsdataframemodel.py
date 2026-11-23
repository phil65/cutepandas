from __future__ import annotations

import enum
import logging
from typing import Any

import polars as pl
from prettyqt import constants, core


logger = logging.getLogger(__name__)


class PolarsDataFrameModel(core.AbstractTableModel):
    """Model to display polars DataFrames."""

    SUPPORTS = pl.DataFrame

    class Roles(enum.IntEnum):
        """Role names."""

        DTypeRole = constants.USER_ROLE + 4555
        ColumnNameRole = constants.USER_ROLE + 4556

    def __init__(self, df: pl.DataFrame, **kwargs):
        super().__init__(**kwargs)
        self.df = df

    @classmethod
    def supports(cls, instance) -> bool:
        return isinstance(instance, pl.DataFrame)

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
                return str(header)
            case _, _:
                return None

    def columnCount(self, parent: core.ModelIndex | None = None) -> int:
        if self.df is None or (parent is not None and parent.isValid()):
            return 0
        return len(self.df.columns)

    def rowCount(self, parent: core.ModelIndex | None = None) -> int:
        if self.df is None or (parent is not None and parent.isValid()):
            return 0
        return len(self.df)

    def data(
        self,
        index: core.ModelIndex,
        role: constants.ItemDataRole = constants.DISPLAY_ROLE,
    ):
        if not index.isValid() or self.df is None:
            return None
        match role:
            case constants.DISPLAY_ROLE:
                cell = self.df[index.row(), index.column()]
                return "" if cell is None else str(cell)
            case constants.EDIT_ROLE | constants.USER_ROLE:
                return self.df[index.row(), index.column()]
            case self.Roles.DTypeRole:
                return self.df.dtypes[index.column()]
            case self.Roles.ColumnNameRole:
                return self.df.columns[index.column()]
        return None

    def flags(self, index: core.ModelIndex) -> constants.ItemFlag:
        if not index.isValid():
            return constants.NO_FLAGS
        return constants.IS_EDITABLE | constants.IS_ENABLED | constants.IS_SELECTABLE

    def setData(
        self,
        index: core.ModelIndex,
        value: Any,
        role: constants.ItemDataRole = constants.EDIT_ROLE,
    ) -> bool:
        match role:
            case constants.USER_ROLE | constants.EDIT_ROLE:
                name = self.df.columns[index.column()]
                try:
                    self.df[index.row(), name] = value
                except Exception:
                    logger.exception("Could not set %r at %s", value, index.row())
                    return False
                else:
                    self.dataChanged.emit(index, index)
                    return True
        return False

    def sort(self, ncol: int, order: constants.SortOrder = constants.ASCENDING):
        if self.df is None or not 0 <= ncol < len(self.df.columns):
            return
        with self.change_layout():
            self.df = self.df.sort(
                self.df.columns[ncol],
                descending=order == constants.DESCENDING,
                nulls_last=True,
                maintain_order=True,
            )


if __name__ == "__main__":
    from datetime import datetime

    from prettyqt import widgets

    app = widgets.app()
    df = pl.DataFrame({
        "integer": [1, 2, 3],
        "date": [
            datetime(2022, 1, 1),
            datetime(2022, 1, 2),
            datetime(2022, 1, 3),
        ],
        "float": [4.0, 5.0, 6.0],
    })
    table = widgets.TableView()
    table.show()
    model = PolarsDataFrameModel(df, parent=table)
    table.set_model(model)
    table.set_delegate("editor")

    app.exec()
