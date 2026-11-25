"""Four-table 2x2 grid DataFrame viewer (corner levels, column headers, index, data)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from prettyqt import constants, core, gui, widgets

from cutepandas import pandasmodels
from cutepandas.pandasmodels import DataFrameHolder
from cutepandas.pandasmodels.dataframeholder import DataFrameHolderMixin
from cutepandas.pandaswidgets import _viewer_style
from cutepandas.pandaswidgets.dataframeviewer import (
    MAX_ROW_HEIGHT,
    ROW_PADDING,
    DataTableView,
    _pd_axis,
    axis_spans,
)
from cutepandas.pandaswidgets.sortcontroller import SortableViewerMixin, SortController


if TYPE_CHECKING:
    from cutepandas.pandasmodels.dataframemodel import SortMode


logger = logging.getLogger(__name__)

SelectionFlag = core.ItemSelectionModel.SelectionFlag
AUTOSIZE_COLUMNS = 50
AUTOSIZE_ROWS = 75
MAX_COLUMN_WIDTH = 400
EDGE_TOLERANCE = 4


class CornerLevelModel(DataFrameHolderMixin, core.AbstractTableModel):
    """Top-left corner model showing pandas index & column level names."""

    def __init__(self, df: Any = None, **kwargs):
        super().__init__(**kwargs)
        self._init_holder(df)

    def _col_names(self) -> list[str]:
        axis = _pd_axis(self.df, horizontal=True)
        if axis is None:
            return [""]
        return ["" if n is None else str(n) for n in axis.names]

    def _idx_names(self) -> list[str]:
        axis = _pd_axis(self.df, horizontal=False)
        if axis is None:
            return [""]
        return ["" if n is None else str(n) for n in axis.names]

    def rowCount(self, parent: core.ModelIndex | None = None) -> int:
        if self.df is None or (parent is not None and parent.isValid()):
            return 0
        return len(self._col_names())

    def columnCount(self, parent: core.ModelIndex | None = None) -> int:
        if self.df is None or (parent is not None and parent.isValid()):
            return 0
        return len(self._idx_names())

    def data(
        self,
        index: core.ModelIndex,
        role: constants.ItemDataRole = constants.DISPLAY_ROLE,
    ):
        if self.df is None or not index.isValid():
            return None
        col_names = self._col_names()
        idx_names = self._idx_names()
        last_row = len(col_names) - 1
        last_col = len(idx_names) - 1
        r, c = index.row(), index.column()
        match role:
            case constants.DISPLAY_ROLE | constants.TOOLTIP_ROLE:
                if r == last_row and c == last_col:
                    left = idx_names[c]
                    right = col_names[r]
                    if left and right:
                        return f"{left} / {right}"
                    return left or right
                if r == last_row:
                    return idx_names[c]
                if c == last_col:
                    return col_names[r]
                return ""
            case constants.ALIGNMENT_ROLE:
                if r == last_row and c < last_col:
                    return constants.ALIGN_LEFT | constants.ALIGN_V_CENTER
                if c == last_col and r < last_row:
                    return constants.ALIGN_RIGHT | constants.ALIGN_V_CENTER
                return constants.ALIGN_CENTER
        return None


def _apply_axis_spans(view: widgets.QTableView, df: Any, *, horizontal: bool):
    """Merge adjacent equal (Multi)Index labels into spanning cells."""
    view.clearSpans()
    for level, start, length in axis_spans(df, horizontal):
        if horizontal:
            view.setSpan(level, start, 1, length)
        else:
            view.setSpan(start, level, length, 1)


class GridDataFrameViewer(SortableViewerMixin, widgets.Widget):
    """Four-table 2x2 grid viewer with flush corner, column header, index, and data.

    Layout:
        table_level  | table_header | (corner)
        table_index  | table_data   | vscroll
                     | hscroll      |

    Because `table_level` occupies the top-left quadrant with the exact height of
    `table_header` and exact width of `table_index`, headers sit completely flush
    against `table_data` with no spacer gaps — even when both row and column
    MultiIndexes have level names.
    """

    sort_requested = core.Signal(int, object)  # ty: ignore[missing-argument]
    sorting_changed = core.Signal(bool)  # ty: ignore[missing-argument]

    def __init__(
        self,
        df: Any = None,
        *,
        sortable: bool = True,
        sort_mode: SortMode = "copy",
        dtype_icons: bool = True,
        parent: widgets.QWidget | None = None,
        **kwargs,
    ) -> None:
        super().__init__(parent=parent, **kwargs)
        if not self.objectName():
            self.setObjectName("cutepandas_grid_dataframe_viewer")
        self._dtype_icons = dtype_icons
        self._syncing_selection = False
        self._autosized_cols: set[int] = set()
        self._resizing_col: int | None = None
        self._resize_start_x = 0
        self._resize_initial_w = 0
        self._resizing_row: int | None = None
        self._resize_start_y = 0
        self._resize_initial_h = 0

        if isinstance(df, DataFrameHolder):
            self.holder = df
        else:
            self.holder = DataFrameHolder(df, parent=self)

        self.set_size_policy("expanding", "expanding")

        # 1. Data table (bottom-right)
        self.table_data = DataTableView(self.holder, parent=self)
        data_model = self.table_data.get_model(skip_proxies=True)
        data_model.set_sort_mode(sort_mode)
        self.sorter = SortController(data_model, sortable=sortable, parent=self)
        self._connect_sorter()

        # 2. Corner level table (top-left)
        self.table_level = widgets.TableView(parent=self)
        self.table_level.h_header.hide()
        self.table_level.v_header.hide()
        self.table_level.set_edit_triggers("none")
        self.table_level.set_scrollbar_policy("always_off")
        self.table_level.setSelectionMode(
            widgets.QAbstractItemView.SelectionMode.NoSelection
        )
        self.table_level.setFocusPolicy(constants.FocusPolicy.NoFocus)
        self.table_level.setWordWrap(False)
        self.table_level.setItemDelegate(
            _viewer_style.CornerCellDelegate(self.table_level)
        )
        self.table_level.set_model(CornerLevelModel(self.holder, parent=self.table_level))

        # 3. Column header table (top-right)
        self.table_header = widgets.TableView(parent=self)
        self.table_header.h_header.hide()
        self.table_header.v_header.hide()
        self.table_header.set_edit_triggers("none")
        self.table_header.set_scrollbar_policy("always_off")
        self.table_header.set_scroll_mode("pixel")
        self.table_header.setWordWrap(False)
        self.table_header.setMouseTracking(True)
        self.table_header.viewport().setMouseTracking(True)
        self.table_header.viewport().installEventFilter(self)

        def _kind_for_col(col: int):
            cols = self.table_data.get_model(skip_proxies=True).column_info
            return cols[col].kind if 0 <= col < len(cols) else None

        self.table_header.setItemDelegate(
            _viewer_style.HeaderCellDelegate(
                constants.HORIZONTAL,
                self.table_header,
                sort_state_getter=self.get_sort_state,
                kind_getter=_kind_for_col,
                show_dtype_icons=dtype_icons,
            )
        )
        self.table_header.set_model(
            pandasmodels.HorizontalHeaderModel(self.holder, parent=self.table_header)
        )
        self.table_header.clicked.connect(self._on_header_clicked)
        self.sorter.sort_state_changed.connect(
            lambda _state: self.table_header.viewport().update()
        )

        # 4. Row index table (bottom-left)
        self.table_index = widgets.TableView(parent=self)
        self.table_index.h_header.hide()
        self.table_index.v_header.hide()
        self.table_index.set_edit_triggers("none")
        self.table_index.set_scrollbar_policy("always_off")
        self.table_index.set_scroll_mode("pixel")
        self.table_index.setWordWrap(False)
        self.table_index.setMouseTracking(True)
        self.table_index.viewport().setMouseTracking(True)
        self.table_index.viewport().installEventFilter(self)
        self.table_index.setItemDelegate(
            _viewer_style.HeaderCellDelegate(constants.VERTICAL, self.table_index)
        )
        self.table_index.set_model(
            pandasmodels.VerticalHeaderModel(self.holder, parent=self.table_index)
        )

        # Link scrollbars
        self.table_data.h_scrollbar.valueChanged.connect(
            self.table_header.h_scrollbar.setValue
        )
        self.table_data.h_scrollbar.valueChanged.connect(self._autosize_visible_columns)
        self.table_header.h_scrollbar.valueChanged.connect(
            self.table_data.h_scrollbar.setValue
        )
        self.table_data.v_scrollbar.valueChanged.connect(
            self.table_index.v_scrollbar.setValue
        )
        self.table_index.v_scrollbar.valueChanged.connect(
            self.table_data.v_scrollbar.setValue
        )

        # Link selections
        self.table_data.selectionModel().selectionChanged.connect(
            self._on_data_selection_changed
        )
        self.table_header.selectionModel().selectionChanged.connect(
            self._on_header_selection_changed
        )
        self.table_index.selectionModel().selectionChanged.connect(
            self._on_index_selection_changed
        )

        # 2x2 grid layout + scrollbars
        layout = self.set_layout("grid", spacing=0, margin=0)
        layout[0, 0] = self.table_level
        layout[0, 1] = self.table_header
        layout[1, 0] = self.table_index
        layout[1, 1] = self.table_data
        layout[1, 2] = self.table_data.v_scrollbar
        layout[2, 1] = self.table_data.h_scrollbar
        layout.setColumnStretch(1, 1)
        layout.setRowStretch(1, 1)

        self.setFocusProxy(self.table_data)
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
                widgets.QApplication.palette(self.table_data)
            )
            _viewer_style.apply_composite_frame_style(self, colors)
            _viewer_style.apply_data_table_style(self.table_data, colors)
            _viewer_style.apply_corner_table_style(self.table_level, colors)
            _viewer_style.apply_header_table_style(
                self.table_header, colors, constants.HORIZONTAL
            )
            _viewer_style.apply_header_table_style(
                self.table_index, colors, constants.VERTICAL
            )
            data_delegate = self.table_data.itemDelegate()
            if isinstance(data_delegate, _viewer_style.DataCellDelegate):
                data_delegate.set_colors(colors)
            for view in (self.table_header, self.table_index):
                delegate = view.itemDelegate()
                if isinstance(delegate, _viewer_style.HeaderCellDelegate):
                    delegate.set_colors(colors)
            corner_delegate = self.table_level.itemDelegate()
            if isinstance(corner_delegate, _viewer_style.CornerCellDelegate):
                corner_delegate.set_colors(colors)
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

    def _on_row_order_changed(self, origin: Any = None):
        # Spans group adjacent equal labels: only meaningful in the frame's order.
        if self.holder.row_order is None:
            _apply_axis_spans(self.table_index, self.df, horizontal=False)
        else:
            self.table_index.clearSpans()

    def _on_df_changed(self, origin: Any = None):
        self._autosized_cols.clear()
        _apply_axis_spans(self.table_header, self.df, horizontal=True)
        _apply_axis_spans(self.table_index, self.df, horizontal=False)
        self._resize_all()
        self.table_header.viewport().update()

    def jump_to_column(self, col_num: int):
        """Scroll horizontally so `col_num` is visible."""
        if (model := self.table_data.model()) is not None:
            idx = model.index(0, col_num)
            self.table_data.scrollTo(idx)
        if (model := self.table_header.model()) is not None:
            idx = model.index(0, col_num)
            self.table_header.scrollTo(idx)
        self._autosize_visible_columns()

    def _resize_all(self):
        data_model = self.table_data.model()
        header_model = self.table_header.model()
        index_model = self.table_index.model()
        level_model = self.table_level.model()
        if data_model is None or header_model is None or index_model is None:
            return

        # 1. Row height for data + index
        row_h = min(self.table_data.fontMetrics().height() + ROW_PADDING, MAX_ROW_HEIGHT)
        self.table_data.v_header.setDefaultSectionSize(row_h)
        self.table_index.v_header.setDefaultSectionSize(row_h)

        # 2. Header row height (for table_header & table_level)
        hdr_row_h = self.table_header.fontMetrics().height() + ROW_PADDING + 2
        self.table_header.v_header.setDefaultSectionSize(hdr_row_h)
        self.table_level.v_header.setDefaultSectionSize(hdr_row_h)

        # 3. Fit data / header columns
        for col in range(min(data_model.columnCount(), AUTOSIZE_COLUMNS)):
            self.auto_size_column(col)

        # 4. Fit index / corner columns (bounded sample + last row for trailing width)
        metrics = self.table_index.fontMetrics()
        pad = _viewer_style.HEADER_HPAD * 2 + 12
        row_count = index_model.rowCount()
        sample = list(range(min(row_count, AUTOSIZE_ROWS)))
        if row_count > AUTOSIZE_ROWS:
            sample.append(row_count - 1)
        idx_total_w = 0
        for col in range(index_model.columnCount()):
            w = 36
            if level_model is not None:
                for r in range(level_model.rowCount()):
                    txt = str(level_model.data(level_model.index(r, col)) or "")
                    w = max(w, metrics.horizontalAdvance(txt) + pad)
            for r in sample:
                txt = str(index_model.data(index_model.index(r, col)) or "")
                w = max(w, metrics.horizontalAdvance(txt) + pad)
            w = min(w, MAX_COLUMN_WIDTH)
            self.table_index.setColumnWidth(col, w)
            # Qt may round up to its minimum section size: use the applied width.
            w = self.table_index.columnWidth(col)
            self.table_level.setColumnWidth(col, w)
            idx_total_w += w

        hdr_total_h = header_model.rowCount() * hdr_row_h + 1  # +1 for divider border
        idx_total_w = idx_total_w + 1 if index_model.columnCount() > 0 else 0

        self.table_header.setFixedHeight(hdr_total_h)
        self.table_level.setFixedHeight(hdr_total_h)
        self.table_index.setFixedWidth(idx_total_w)
        self.table_level.setFixedWidth(idx_total_w)

    def _autosize_visible_columns(self, _value: int = 0):
        """Lazily auto-size wide-frame columns as they scroll into view."""
        model = self.table_data.model()
        if model is None:
            return
        col_count = model.columnCount()
        if len(self._autosized_cols) >= col_count:
            return
        vp_width = self.table_data.viewport().width()
        first = self.table_data.columnAt(0)
        if first < 0:
            return
        last = self.table_data.columnAt(max(0, vp_width - 1))
        end = col_count if last < 0 else min(col_count, last + 1)
        for col in range(first, end):
            if col not in self._autosized_cols:
                self.auto_size_column(col)

    def auto_size_column(self, column_index: int):
        """Fit `column_index` in both `table_header` and `table_data`."""
        self._autosized_cols.add(column_index)
        data_model = self.table_data.model()
        header_model = self.table_header.model()
        if (
            data_model is None
            or header_model is None
            or not 0 <= column_index < data_model.columnCount()
        ):
            return
        header_font = gui.Font(self.table_header.font())
        header_font.set_weight("demi_bold")
        header_metrics = gui.FontMetrics(header_font)
        header_extra = _viewer_style.HEADER_HPAD * 2 + _viewer_style.SORT_ARROW_SPACE
        if self._dtype_icons:
            header_extra += 20

        width = 60
        for r in range(header_model.rowCount()):
            text = str(header_model.data(header_model.index(r, column_index)) or "")
            width = max(width, header_metrics.horizontalAdvance(text) + header_extra)

        cell_metrics = self.table_data.fontMetrics()
        cell_pad = _viewer_style.CELL_HPAD * 2 + 12
        for r in range(min(data_model.rowCount(), AUTOSIZE_ROWS)):
            text = str(data_model.data(data_model.index(r, column_index)) or "")
            width = max(width, cell_metrics.horizontalAdvance(text) + cell_pad)

        width = min(width, MAX_COLUMN_WIDTH)
        self.table_header.setColumnWidth(column_index, width)
        self.table_data.setColumnWidth(column_index, width)

    # --- interactive header column resizing ---

    def _header_edge_at(self, x: int) -> int | None:
        header_x = x + self.table_header.h_scrollbar.value()
        col = self.table_header.columnAt(x)
        if col < 0:
            return None
        right_edge = self.table_header.columnViewportPosition(
            col
        ) + self.table_header.columnWidth(col)
        if abs(x - right_edge) <= EDGE_TOLERANCE:
            return col
        left_edge = self.table_header.columnViewportPosition(col)
        if col > 0 and abs(x - left_edge) <= EDGE_TOLERANCE and header_x >= 0:
            return col - 1
        return None

    def _index_edge_at(self, y: int) -> int | None:
        row = self.table_index.rowAt(y)
        if row < 0:
            return None
        bottom_edge = self.table_index.rowViewportPosition(
            row
        ) + self.table_index.rowHeight(row)
        if abs(y - bottom_edge) <= EDGE_TOLERANCE:
            return row
        top_edge = self.table_index.rowViewportPosition(row)
        if row > 0 and abs(y - top_edge) <= EDGE_TOLERANCE:
            return row - 1
        return None

    def eventFilter(self, watched: core.QObject, event: core.QEvent) -> bool:
        if watched is self.table_header.viewport() and isinstance(event, gui.QMouseEvent):
            x = int(event.position().x())
            match event.type():
                case core.QEvent.Type.MouseButtonPress:
                    col = self._header_edge_at(x)
                    if col is not None:
                        self._resizing_col = col
                        self._resize_start_x = x
                        self._resize_initial_w = self.table_header.columnWidth(col)
                        return True
                case core.QEvent.Type.MouseButtonRelease:
                    if self._resizing_col is not None:
                        self._resizing_col = None
                        return True
                case core.QEvent.Type.MouseButtonDblClick:
                    col = self._header_edge_at(x)
                    if col is not None:
                        self.auto_size_column(col)
                        return True
                case core.QEvent.Type.MouseMove:
                    if self._resizing_col is not None:
                        new_w = max(24, self._resize_initial_w + x - self._resize_start_x)
                        self.table_header.setColumnWidth(self._resizing_col, new_w)
                        self.table_data.setColumnWidth(self._resizing_col, new_w)
                        return True
                    cursor = (
                        constants.CursorShape.SplitHCursor
                        if self._header_edge_at(x) is not None
                        else constants.CursorShape.ArrowCursor
                    )
                    self.table_header.viewport().setCursor(gui.Cursor(cursor))
        elif watched is self.table_index.viewport() and isinstance(
            event, gui.QMouseEvent
        ):
            y = int(event.position().y())
            match event.type():
                case core.QEvent.Type.MouseButtonPress:
                    row = self._index_edge_at(y)
                    if row is not None:
                        self._resizing_row = row
                        self._resize_start_y = y
                        self._resize_initial_h = self.table_index.rowHeight(row)
                        return True
                case core.QEvent.Type.MouseButtonRelease:
                    if self._resizing_row is not None:
                        self._resizing_row = None
                        return True
                case core.QEvent.Type.MouseMove:
                    if self._resizing_row is not None:
                        new_h = max(18, self._resize_initial_h + y - self._resize_start_y)
                        self.table_index.setRowHeight(self._resizing_row, new_h)
                        self.table_data.setRowHeight(self._resizing_row, new_h)
                        return True
                    cursor = (
                        constants.CursorShape.SplitVCursor
                        if self._index_edge_at(y) is not None
                        else constants.CursorShape.ArrowCursor
                    )
                    self.table_index.viewport().setCursor(gui.Cursor(cursor))
        return super().eventFilter(watched, event)

    # --- selection syncing ---

    def _on_data_selection_changed(self, selected, deselected):
        if self._syncing_selection:
            return
        self._syncing_selection = True
        try:
            self.table_header.selectionModel().select(
                selected, SelectionFlag.Columns | SelectionFlag.ClearAndSelect
            )
            self.table_index.selectionModel().select(
                selected, SelectionFlag.Rows | SelectionFlag.ClearAndSelect
            )
        finally:
            self._syncing_selection = False

    def _on_header_selection_changed(self, selected, deselected):
        if self._syncing_selection or not self.table_header.hasFocus():
            return
        self._syncing_selection = True
        try:
            self.table_data.selectionModel().select(
                selected, SelectionFlag.Columns | SelectionFlag.ClearAndSelect
            )
            self.table_index.selectionModel().clear()
        finally:
            self._syncing_selection = False

    def _on_index_selection_changed(self, selected, deselected):
        if self._syncing_selection or not self.table_index.hasFocus():
            return
        self._syncing_selection = True
        try:
            self.table_data.selectionModel().select(
                selected, SelectionFlag.Rows | SelectionFlag.ClearAndSelect
            )
            self.table_header.selectionModel().clear()
        finally:
            self._syncing_selection = False

    # --- sorting ---

    def _on_header_clicked(self, ix: core.QModelIndex):
        header_model = self.table_header.model()
        if not ix.isValid() or header_model is None:
            return
        if ix.row() == header_model.rowCount() - 1:
            self.sorter.request_sort(ix.column())


if __name__ == "__main__":
    app = widgets.app()
    df = pd.DataFrame(
        np.random.default_rng(42).uniform(10.0, 500.0, size=(12, 4)).round(2),
        index=pd.MultiIndex.from_product(
            [["North", "South", "West"], ["Q1", "Q2", "Q3", "Q4"]],
            names=["Region", "Quarter"],
        ),
        columns=pd.MultiIndex.from_product(
            [["Revenue", "Margin"], ["2025", "2026"]],
            names=["Metric", "Year"],
        ),
    )
    table = GridDataFrameViewer(df)
    table.show()
    app.exec()
