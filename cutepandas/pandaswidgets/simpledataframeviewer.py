"""Single-table DataFrame viewer built on one QTableView and DataFrameModel."""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any

from prettyqt import constants, core, gui, widgets

from cutepandas import pandasmodels
from cutepandas.pandasmodels import DataFrameHolder
from cutepandas.pandaswidgets import _viewer_style
from cutepandas.utils import helpers


if TYPE_CHECKING:
    from cutepandas.pandasmodels.dataframemodel import SortMode


logger = logging.getLogger(__name__)

ROW_PADDING = 10
MAX_COLUMN_WIDTH = 400
AUTOSIZE_COLUMNS = 50
AUTOSIZE_ROWS = 75


class SimpleDataFrameViewer(widgets.TableView):
    """Lightweight single-table viewer backed by `DataFrameModel`.

    Uses Qt's native horizontal and vertical `QHeaderView`s (joining MultiIndex
    levels with a separator) instead of separate header tables. Supports shared
    `DataFrameHolder`s, dtype icons, clipboard copy/paste, and both `"copy"` and
    `"permutation"` sort modes with native sort indicators.
    """

    sort_requested = core.Signal(int, object)  # ty: ignore[missing-argument]
    sorting_changed = core.Signal(bool)  # ty: ignore[missing-argument]
    _permutation_ready = core.Signal(int, object)  # ty: ignore[missing-argument]

    def __init__(
        self,
        df: Any = None,
        *,
        sortable: bool = True,
        sort_mode: SortMode = "copy",
        dtype_icons: bool = True,
        color_values: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._sortable = sortable
        self._dtype_icons = dtype_icons
        self._sort_state: tuple[int, bool] | None = None
        self._sort_request = 0
        self._sorting = False
        self._autosized_cols: set[int] = set()
        self._permutation_ready.connect(self._on_permutation_ready)

        if isinstance(df, DataFrameHolder):
            self.holder = df
        else:
            self.holder = DataFrameHolder(df, parent=self)

        self.setAlternatingRowColors(True)
        self.set_scroll_mode("pixel")
        self.setWordWrap(False)
        self.set_size_policy("expanding", "expanding")
        self.setItemDelegate(_viewer_style.DataCellDelegate(self))

        self.h_header.setHighlightSections(True)
        self.h_header.setSectionsClickable(True)
        self.h_header.setSortIndicatorShown(False)
        self.h_header.setStretchLastSection(False)
        self.h_header.setDefaultAlignment(constants.ALIGN_CENTER)
        self.h_header.sectionClicked.connect(self._on_header_clicked)
        self.h_header.sectionHandleDoubleClicked.connect(self.auto_size_column)

        self.v_header.setHighlightSections(True)
        self.v_header.setDefaultAlignment(
            constants.ALIGN_RIGHT | constants.ALIGN_V_CENTER
        )

        model = pandasmodels.DataFrameModel(
            self.holder,
            dtype_icons=dtype_icons,
            color_values=color_values,
            sort_mode=sort_mode,
            parent=self,
        )
        self.set_model(model)
        self.h_scrollbar.valueChanged.connect(self._autosize_visible_columns)

        self._apply_style()

        self.holder.changed.connect(self._on_df_changed)
        self.holder.row_order_changed.connect(self._on_row_order_changed)
        self._on_df_changed()

    def _apply_style(self):
        if getattr(self, "_applying_style", False):
            return
        self._applying_style = True
        try:
            colors = _viewer_style.ViewerColors.from_palette(
                widgets.QApplication.palette(self)
            )
            _viewer_style.apply_native_table_style(self, colors)
            delegate = self.itemDelegate()
            if isinstance(delegate, _viewer_style.DataCellDelegate):
                delegate.set_colors(colors)
        finally:
            self._applying_style = False

    def changeEvent(self, event: core.QEvent):
        super().changeEvent(event)
        if event.type() == core.QEvent.Type.PaletteChange:
            self._apply_style()

    @property
    def df(self) -> Any:
        return self.holder.df

    @df.setter
    def df(self, df: Any):
        self.set_df(df)

    def set_df(self, df: Any):
        """Show another frame (in every model / widget sharing the holder)."""
        self.holder.df = df

    def _on_df_changed(self, origin: Any = None):
        self._sort_state = None
        self._autosized_cols.clear()
        self.h_header.setSortIndicatorShown(False)
        self._resize_columns_and_rows()

    def _on_row_order_changed(self, origin: Any = None):
        if self.holder.row_order is None:
            self._sort_state = None
            self.h_header.setSortIndicatorShown(False)

    def _resize_columns_and_rows(self):
        model = self.model()
        if model is None:
            return
        row_height = self.fontMetrics().height() + ROW_PADDING
        self.v_header.setDefaultSectionSize(row_height)
        for col in range(min(model.columnCount(), AUTOSIZE_COLUMNS)):
            self.auto_size_column(col)

    def _autosize_visible_columns(self, _value: int = 0):
        """Lazily auto-size wide-frame columns as they scroll into view."""
        model = self.model()
        if model is None:
            return
        col_count = model.columnCount()
        if len(self._autosized_cols) >= col_count:
            return
        vp_width = self.viewport().width()
        first = self.columnAt(0)
        if first < 0:
            return
        last = self.columnAt(max(0, vp_width - 1))
        end = col_count if last < 0 else min(col_count, last + 1)
        for col in range(first, end):
            if col not in self._autosized_cols:
                self.auto_size_column(col)

    def auto_size_column(self, column_index: int):
        """Fit `column_index` to its header (including icon/sort arrow) and first rows."""
        self._autosized_cols.add(column_index)
        model = self.model()
        if model is None or not 0 <= column_index < model.columnCount():
            return
        header_text = str(
            model.headerData(column_index, constants.HORIZONTAL, constants.DISPLAY_ROLE)
            or ""
        )
        header_font = gui.Font(self.h_header.font())
        header_font.set_weight("demi_bold")
        header_metrics = gui.FontMetrics(header_font)
        header_extra = _viewer_style.HEADER_HPAD * 2 + _viewer_style.SORT_ARROW_SPACE
        if self._dtype_icons:
            header_extra += 22
        width = header_metrics.horizontalAdvance(header_text) + header_extra

        cell_metrics = self.fontMetrics()
        cell_pad = _viewer_style.CELL_HPAD * 2 + 12
        for row in range(min(model.rowCount(), AUTOSIZE_ROWS)):
            text = str(model.data(model.index(row, column_index)) or "")
            width = max(width, cell_metrics.horizontalAdvance(text) + cell_pad)

        self.setColumnWidth(column_index, min(width, MAX_COLUMN_WIDTH))

    # --- sorting ---

    def _on_header_clicked(self, column: int):
        if self.df is None:
            return
        ascending = self._sort_state != (column, True)
        order = constants.ASCENDING if ascending else constants.DESCENDING
        self.sort_requested.emit(column, order)
        if self._sortable:
            self.sort_by(column, order)

    def sort_by(self, column: int, order: constants.SortOrder = constants.ASCENDING):
        """Sort by a column position (in the model's `sort_mode`)."""
        model = self.get_model(skip_proxies=True)
        if model.get_sort_mode() != "permutation":
            model.sort(column, order)
            self.set_sort_state(column, order)
            return
        self._sort_request += 1
        request = self._sort_request
        self._set_sorting(True)

        def work():
            try:
                result = model.compute_sort_permutation(column, order)
            except Exception as e:
                result = e
            self._permutation_ready.emit(request, result)

        threading.Thread(target=work, name="cutepandas-sort", daemon=True).start()

    def _on_permutation_ready(self, request: int, result: Any):
        if request != self._sort_request:
            return
        self._set_sorting(False)
        if isinstance(result, BaseException):
            logger.error("Sorting failed", exc_info=result)
            return
        model = self.get_model(skip_proxies=True)
        if result is not None and model.apply_sort_permutation(result):
            order = constants.DESCENDING if result.descending else constants.ASCENDING
            self.set_sort_state(result.column, order)

    def cancel_sort(self):
        self._sort_request += 1
        self._set_sorting(False)

    def is_sorting(self) -> bool:
        return self._sorting

    def _set_sorting(self, value: bool):
        if value != self._sorting:
            self._sorting = value
            self.sorting_changed.emit(value)

    def clear_sort(self):
        self.cancel_sort()
        self.get_model(skip_proxies=True).clear_sort()
        self.set_sort_state(None)

    def set_sort_state(
        self, column: int | None, order: constants.SortOrder | None = None
    ):
        if column is None:
            self._sort_state = None
            self.h_header.setSortIndicatorShown(False)
        else:
            ascending = order != constants.DESCENDING
            self._sort_state = (column, ascending)
            qt_order = constants.ASCENDING if ascending else constants.DESCENDING
            self.h_header.setSortIndicator(column, qt_order)
            self.h_header.setSortIndicatorShown(True)

    def get_sort_state(self) -> tuple[int, constants.SortOrder] | None:
        if self._sort_state is None:
            return None
        column, ascending = self._sort_state
        return column, constants.ASCENDING if ascending else constants.DESCENDING

    def set_sortable(self, value: bool):
        self._sortable = value

    def is_sortable(self) -> bool:
        return self._sortable

    def set_sort_mode(self, mode: SortMode):
        self.get_model(skip_proxies=True).set_sort_mode(mode)

    def get_sort_mode(self) -> SortMode:
        return self.get_model(skip_proxies=True).get_sort_mode()

    # --- clipboard ---

    def copy(self):
        """Copy the bounding box of the selected cells as tab-separated text."""
        indexes = self.selectionModel().selection().indexes()
        model = self.get_model(skip_proxies=True)
        if not indexes or model.source is None:
            return
        rows = [ix.row() for ix in indexes]
        cols = range(
            min(ix.column() for ix in indexes), max(ix.column() for ix in indexes) + 1
        )
        chunks = model.fetch_window(min(rows), max(rows) + 1, cols)
        lines = [
            "\t".join(
                "" if chunk.values[i] is None else str(chunk.values[i])
                for chunk in chunks
            )
            for i in range(max(rows) + 1 - min(rows))
        ]
        gui.GuiApplication.clipboard().setText("\n".join(lines))

    def paste(self):
        """Paste tab-separated clipboard text, starting at the current cell."""
        text = gui.GuiApplication.clipboard().text()
        model = self.model()
        if not text or model is None:
            return
        current = self.currentIndex()
        top, left = (current.row(), current.column()) if current.isValid() else (0, 0)
        for i, line in enumerate(text.splitlines()):
            for j, cell in enumerate(line.split("\t")):
                index = model.index(top + i, left + j)
                if index.isValid():
                    model.setData(index, helpers.infer_type(cell), constants.EDIT_ROLE)

    def keyPressEvent(self, event: gui.QKeyEvent):
        if event.matches(gui.KeySequence.StandardKey.Copy):
            self.copy()
        elif event.matches(gui.KeySequence.StandardKey.Paste):
            self.paste()
        else:
            super().keyPressEvent(event)


if __name__ == "__main__":
    import numpy as np
    import pandas as pd

    app = widgets.app()
    df = pd.DataFrame(
        np.random.default_rng(0).normal(size=(20, 4)).round(2),
        index=pd.MultiIndex.from_product([["North", "South"], range(10)]),
        columns=pd.MultiIndex.from_product([["Revenue", "Margin"], ["2025", "2026"]]),
    )
    view = SimpleDataFrameViewer(df)
    view.show()
    app.exec()
