from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from prettyqt import constants, core, custom_widgets, gui, widgets

from cutepandas import pandasmodels
from cutepandas.pandasmodels import DataFrameHolder
from cutepandas.pandaswidgets import _viewer_style
from cutepandas.pandaswidgets.sortcontroller import SortableViewerMixin, SortController
from cutepandas.utils import helpers


if TYPE_CHECKING:
    from cutepandas.pandasmodels.dataframemodel import SortMode


logger = logging.getLogger(__name__)

SelectionFlag = core.ItemSelectionModel.SelectionFlag
QWIDGETSIZE_MAX = (1 << 24) - 1
ROW_PADDING = 10
MAX_ROW_HEIGHT = 100
MAX_SPAN_COUNT = 20_000
SAMPLE_ROWS = 75


def _pd_axis(df: Any, horizontal: bool) -> pd.Index | None:
    """Columns / index of a pandas frame; None for other backends (one level)."""
    if not isinstance(df, pd.DataFrame):
        return None
    return df.columns if horizontal else df.index


def _nlevels(df: Any, horizontal: bool) -> int:
    axis = _pd_axis(df, horizontal)
    return 1 if axis is None else axis.nlevels


def _has_level_names(df: Any, horizontal: bool) -> bool:
    axis = _pd_axis(df, horizontal)
    return axis is not None and any(name is not None for name in axis.names)


def axis_spans(df: Any, horizontal: bool) -> list[tuple[int, int, int]]:
    """(level, start, length) of each run of equal adjacent labels of a pandas axis.

    A run on an inner level also breaks wherever an outer level changes, so
    ("a", 1), ("b", 1) doesn't merge the two 1s across the a/b boundary. NaN labels
    count as equal. Empty for other backends, unique flat axes and when there would
    be more than `MAX_SPAN_COUNT` spans (QTableView's span list degrades on tens of
    thousands of spans).
    """
    axis = _pd_axis(df, horizontal)
    if axis is None or len(axis) < 2:
        return []
    if isinstance(axis, pd.MultiIndex):
        levels = [np.asarray(codes) for codes in axis.codes]
    elif axis.is_unique:  # cached by pandas, O(1) for a RangeIndex
        return []
    else:
        levels = [pd.factorize(axis, use_na_sentinel=False)[0]]
    boundary = np.zeros(len(axis) - 1, dtype=bool)  # True: i and i+1 differ
    spans: list[tuple[int, int, int]] = []
    for level, codes in enumerate(levels):
        boundary |= codes[:-1] != codes[1:]
        starts = np.concatenate(([0], np.flatnonzero(boundary) + 1))
        lengths = np.diff(np.append(starts, len(axis)))
        mask = lengths > 1
        if len(spans) + int(mask.sum()) > MAX_SPAN_COUNT:
            return []
        spans.extend(
            (level, int(start), int(length))
            for start, length in zip(starts[mask], lengths[mask], strict=True)
        )
    return spans


class DataTableView(widgets.TableView):
    def __init__(self, df: Any = None, **kwargs):
        super().__init__(**kwargs)

        self.h_header.hide()
        self.v_header.hide()
        self.setAlternatingRowColors(True)
        self.set_scroll_mode("pixel")
        self.set_margin(0)
        self.set_scrollbar_policy("always_off")
        self.setWordWrap(False)  # single-line cells (row height assumes it)
        self.setItemDelegate(_viewer_style.DataCellDelegate(self))
        self.set_size_policy("expanding", "expanding")
        self.set_model(pandasmodels.DataFrameModel(df, parent=self))

    def set_df(self, df: Any):
        self.get_model(skip_proxies=True).holder.df = df

    def copy(self):
        """Copy the bounding box of the selected cells as tab-separated text."""
        indexes = self.selectionModel().selection().indexes()
        if not indexes or self.get_model(skip_proxies=True).source is None:
            return
        rows = [ix.row() for ix in indexes]
        cols = range(
            min(ix.column() for ix in indexes), max(ix.column() for ix in indexes) + 1
        )
        model = self.get_model(skip_proxies=True)
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
                if index.isValid():  # clip to the frame
                    model.setData(index, helpers.infer_type(cell), constants.EDIT_ROLE)

    def keyPressEvent(self, event: gui.QKeyEvent):
        # Handled here: the base class would otherwise copy only the current cell.
        if event.matches(gui.KeySequence.StandardKey.Copy):
            self.copy()
        elif event.matches(gui.KeySequence.StandardKey.Paste):
            self.paste()
        else:
            super().keyPressEvent(event)

    def sizeHint(self):
        if self.get_model(skip_proxies=True).df is None:
            return core.Size(0, 0)
        width = 2 * self.frameWidth()  # Account for border & padding
        # width += self.v_scrollbar.width()
        max_cols = min(100, self.model().columnCount())
        for i in range(max_cols):
            width += self.columnWidth(i)

        # Height
        height = 2 * self.frameWidth()  # Account for border & padding
        # height += self.h_scrollbar.height()
        max_rows = min(100, self.model().rowCount())
        for i in range(max_rows):
            height += self.rowHeight(i)

        return core.Size(width, height)


class DataFrameViewer(SortableViewerMixin, widgets.Widget):
    """Data table with synced column / index headers (MultiIndex spans for pandas).

    Clicking a column header emits `sort_requested(column, order)` (`order` is a
    `Qt.SortOrder`; clicking the same column again reverses it). With
    `sortable=True` (default) the viewer then sorts itself, see `sort_mode`; with
    `sortable=False` it only emits the signal, so an application can sort under its
    own control (e.g. a memory budget) and call `set_df()` / `set_sort_state()`.

    `sort_mode="copy"` (default) replaces the frame by a sorted copy (on the GUI
    thread); `sort_mode="permutation"` sorts for display only (a row order of 8 bytes
    per row in the holder, no frame copy). The order is computed in a worker thread:
    `sorting_changed(True)` while it runs, `sorting_changed(False)` when it was
    applied or dropped. Only the latest request is applied, and only if the frame
    was not replaced or edited meanwhile; `cancel_sort()` drops a running one.
    See `DataFrameModel`.
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
        **kwargs,
    ) -> None:
        """Pass a DataFrameHolder to share the frame with other models / widgets."""
        super().__init__(**kwargs)
        if not self.objectName():
            self.setObjectName("cutepandas_dataframe_viewer")
        # Indicates whether the widget has been shown yet. Set to True in showEvent.
        self._loaded = False
        self._dtype_icons = dtype_icons
        self._autosized_cols: set[int] = set()
        if isinstance(df, DataFrameHolder):
            self.holder = df
        else:
            self.holder = DataFrameHolder(df, parent=self)
        # Set up DataFrame TableView and Model
        self.table_data = DataTableView(self.holder, parent=self)
        data_model = self.table_data.get_model(skip_proxies=True)
        data_model.set_sort_mode(sort_mode)
        self.sorter = SortController(data_model, sortable=sortable, parent=self)
        self._connect_sorter()
        self.set_size_policy("expanding", "expanding")

        self.table_data.selectionModel().selectionChanged.connect(
            self._on_selection_changed
        )
        # Create headers
        self.table_columns = HeaderView(
            orientation=constants.HORIZONTAL,
            parent=self,
            show_dtype_icons=dtype_icons,
        )
        self.table_columns.clicked.connect(self.on_clicked)
        self.sorter.sort_state_changed.connect(
            lambda _state: self.table_columns.viewport().update()
        )

        self.table_index = HeaderView(orientation=constants.VERTICAL, parent=self)
        self.table_columns.selectionModel().selectionChanged.connect(
            self.on_column_selection_changed
        )
        self.table_index.selectionModel().selectionChanged.connect(
            self.on_index_selection_changed
        )

        # Set up layout
        self.layout_grid = self.set_layout("grid", margin=0, spacing=0)
        # Link scrollbars
        self.table_data.h_scrollbar.valueChanged.connect(
            self.table_columns.h_scrollbar.setValue
        )
        self.table_data.h_scrollbar.valueChanged.connect(self._autosize_visible_columns)
        self.table_data.v_scrollbar.valueChanged.connect(
            self.table_index.v_scrollbar.setValue
        )
        self.table_columns.h_scrollbar.valueChanged.connect(
            self.table_data.h_scrollbar.setValue
        )
        self.table_index.v_scrollbar.valueChanged.connect(
            self.table_data.v_scrollbar.setValue
        )

        # Disable scrolling on the headers. Even when hidden, dragging desyncs them
        self.table_index.h_scrollbar.valueChanged.connect(lambda: None)

        # Add items to layout

        # SpacerItem |            Table_columns
        #       table index             | TrackingSpacer
        #       Table Index             | Table Data      | V Scrollbar
        #            |  Tracking Spacer  |H Scrollbar
        self._top_spacer = TrackingSpacer(ref_y=self.table_index.h_header)
        self._bottom_spacer = TrackingSpacer(ref_x=self.table_columns.v_header)
        self.layout_grid[0, 0] = widgets.SpacerItem(0, 0, "expanding", "expanding")
        self.layout_grid[0, 1:2] = self.table_columns
        self.layout_grid[1:2, 0:1] = self.table_index
        self.layout_grid[1, 2] = self._top_spacer
        self.layout_grid[2, 2] = self.table_data
        self.layout_grid[2, 3] = self.table_data.v_scrollbar
        self.layout_grid[3, 1] = self._bottom_spacer
        self.layout_grid[3, 2] = self.table_data.h_scrollbar

        # The data table takes all extra space, so the viewer fills its parent
        # (a small frame leaves empty table area instead of a shrunken widget).
        self.layout_grid.setColumnStretch(2, 1)
        self.layout_grid.setRowStretch(2, 1)
        # Headers keep their content size: pin them to the data table's edges.
        self.layout_grid.setAlignment(self.table_columns, constants.ALIGN_LEFT)
        self.layout_grid.setAlignment(self.table_index, constants.ALIGN_TOP)

        self._apply_style()

        # Connected after the models were created, so they are reset before
        # the header layout gets rebuilt.
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
            _viewer_style.apply_header_table_style(
                self.table_columns, colors, constants.HORIZONTAL
            )
            _viewer_style.apply_header_table_style(
                self.table_index, colors, constants.VERTICAL
            )
            data_delegate = self.table_data.itemDelegate()
            if isinstance(data_delegate, _viewer_style.DataCellDelegate):
                data_delegate.set_colors(colors)
            for view in (self.table_columns, self.table_index):
                delegate = view.itemDelegate()
                if isinstance(delegate, _viewer_style.HeaderCellDelegate):
                    delegate.set_colors(colors)
            div = _viewer_style._hex(colors.divider)
            bg = _viewer_style._hex(colors.header_outer_bg)
            self._top_spacer.setStyleSheet(
                f"QFrame {{ background-color: {bg}; border: none;"
                f" border-bottom: 1px solid {div}; }}"
            )
            self._bottom_spacer.setStyleSheet(
                f"QFrame {{ background-color: {bg}; border: none; }}"
            )
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

    def _on_row_order_changed(self, origin=None):
        # Spans group adjacent equal labels: only meaningful in the frame's order.
        self.table_index.refresh(set_spans=self.holder.row_order is None)

    def _on_df_changed(self, origin=None):
        """Rebuild header spans and sizes for the holder's current frame."""
        df = self.df
        self._autosized_cols.clear()
        self.table_columns.refresh()
        self.table_index.refresh()
        # Only show the level-name headers if there are level names
        v_header = self.table_columns.v_header
        h_header = self.table_index.h_header
        if _has_level_names(df, horizontal=True):
            v_header.setMinimumWidth(0)
            v_header.setMaximumWidth(QWIDGETSIZE_MAX)
        else:
            v_header.setFixedWidth(0)
        if _has_level_names(df, horizontal=False):
            h_header.setMinimumHeight(0)
            h_header.setMaximumHeight(QWIDGETSIZE_MAX)
        else:
            h_header.setFixedHeight(0)
        self._resize_columns()
        self.table_data.updateGeometry()
        self.table_columns.updateGeometry()
        self.table_index.updateGeometry()  # its width follows the index labels

    def jump_to_column(self, col_num: int):
        """Make sure column at given index is visible.

        Scrolls to column at given index.

        Arguments:
            col_num: column to scroll to
        """
        if (model := self.table_data.model()) is not None:
            idx = model.index(0, col_num)
            self.table_data.scrollTo(idx)
        if (model := self.table_index.model()) is not None:
            idx = model.index(0, col_num)
            self.table_index.scrollTo(idx)
        if (model := self.table_columns.model()) is not None:
            idx = model.index(0, col_num)
            self.table_columns.scrollTo(idx)

    # def on_column_selection_changed(self, selected, deselected):
    #     if self.table_columns.hasFocus():
    #         selected = self.table_columns.get_higher_levels()
    #         self.table_data.selectionModel().select(
    #             selected,
    #             SelectionFlag.Columns
    #             | SelectionFlag.ClearAndSelect,
    #         )
    #     self.table_columns._on_selection_changed(selected, deselected)

    # def on_index_selection_changed(self, selected, deselected):
    #     if not self.table_index.hasFocus():
    #         return
    #     selected = self.table_index.get_higher_levels()
    #     self.table_data.selectionModel().select(
    #         selected,
    #         SelectionFlag.Rows
    #         | SelectionFlag.ClearAndSelect,
    #     )
    #     self.table_index._on_selection_changed(selected, deselected)

    def on_clicked(self, ix: core.QModelIndex):
        """Request a sort by the clicked leaf column; clicking again reverses."""
        col_model = self.table_columns.model()
        if not ix.isValid() or col_model is None:
            return
        if ix.row() == col_model.rowCount() - 1:
            self.sorter.request_sort(ix.column())

    def showEvent(self, event: gui.QShowEvent):
        """Initialize column and row sizes on the first time the widget is shown."""
        if not self._loaded and self.df is not None:
            # Set column widths
            self._resize_columns()

            self._loaded = True
        event.accept()

    def _resize_columns(self):
        """Fit the first columns to header + first rows; one single-line row height."""
        n = 50
        columns = min(self.table_data.model().columnCount(), n)
        for column_index in range(columns):
            self.auto_size_column(column_index)
        # Cells do not wrap, so a row is one line of text.
        height = min(self.table_data.fontMetrics().height() + ROW_PADDING, MAX_ROW_HEIGHT)
        self.table_index.v_header.setDefaultSectionSize(height)
        self.table_data.v_header.setDefaultSectionSize(height)

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

    def on_column_selection_changed(self, selected, deselected):
        if not self.table_columns.hasFocus():
            return
        higher_levels = self.table_columns.get_higher_levels()
        # Set selection mode so selecting one row or column at a time
        # adds to selection each time
        selected.merge(higher_levels, SelectionFlag.Deselect)
        # Select the cells in the data view
        self.table_data.selectionModel().select(
            selected, SelectionFlag.Columns | SelectionFlag.ClearAndSelect
        )

    def on_index_selection_changed(self, selected, deselected):
        if not self.table_index.hasFocus():
            return
        higher_levels = self.table_index.get_higher_levels()
        selected.merge(higher_levels, SelectionFlag.Deselect)
        self.table_data.selectionModel().select(
            selected, SelectionFlag.Rows | SelectionFlag.ClearAndSelect
        )

    def _on_selection_changed(self, selected, deselected):
        """Runs when cells are selected in the main table.

        This logic highlights the correct
        cells in the vertical and horizontal headers when a data cell is selected
        """
        # The two blocks below check what columns or rows are selected in the data table
        # and highlights thecorresponding ones in the two headers. The if statements
        # check for focus on headers, because if the user clicks a header that will
        # auto-select all cells in that row or column which will trigger this function
        # and cause and infinite loop

        if not self.table_columns.hasFocus():
            self.table_columns.selectionModel().select(
                selected, SelectionFlag.Columns | SelectionFlag.ClearAndSelect
            )

        if not self.table_index.hasFocus():
            self.table_index.selectionModel().select(
                selected, SelectionFlag.Rows | SelectionFlag.ClearAndSelect
            )

    def auto_size_column(self, column_index: int):
        """Set the size of column at column_index to fit its contents."""
        self._autosized_cols.add(column_index)
        col_model = self.table_columns.model()
        data_model = self.table_data.model()
        if col_model is None or data_model is None:
            return

        header_font = gui.Font(self.table_columns.font())
        header_font.set_weight("demi_bold")
        header_metrics = gui.FontMetrics(header_font)
        header_extra = _viewer_style.HEADER_HPAD * 2 + _viewer_style.SORT_ARROW_SPACE
        if self._dtype_icons:
            header_extra += 20

        width = 60
        for r in range(col_model.rowCount()):
            txt = str(col_model.data(col_model.index(r, column_index)) or "")
            width = max(width, header_metrics.horizontalAdvance(txt) + header_extra)

        # Only check the first SAMPLE_ROWS rows for O(1) performance.
        rows = min(data_model.rowCount(), SAMPLE_ROWS)
        cell_metrics = self.table_data.fontMetrics()
        cell_pad = _viewer_style.CELL_HPAD * 2 + 12
        for i in range(rows):
            mi = data_model.index(i, column_index)
            text = str(data_model.data(mi) or "")
            width = max(width, cell_metrics.horizontalAdvance(text) + cell_pad)

        max_allowable_width = 400
        width = min(width, max_allowable_width)

        self.table_columns.setColumnWidth(column_index, width)
        self.table_data.setColumnWidth(column_index, width)

        self.table_data.updateGeometry()
        self.table_columns.updateGeometry()

    def auto_size_row(self, row_index: int):
        """Set the size of row at row_index to fix its contents."""
        padding = 20

        self.table_index.resizeRowToContents(row_index)
        height = self.table_index.rowHeight(row_index)

        # Iterate over the row's columns and check the width of each to
        # determine the max height for the row
        # Only check the first N columns for performance.
        N = 100
        model = self.table_data.model()
        for i in range(min(N, model.columnCount())):
            mi = model.index(row_index, i)
            cell_width = self.table_columns.columnWidth(i)
            text = model.data(mi)
            # Gets row height at a constrained width (the column width).
            # This constrained width, with the flag of Qt.TextWordWrap
            # gets the height the cell would have to be to fit the text.
            constrained_rect = core.Rect(0, 0, cell_width, 0)
            text = str(text)  # TODO: remove this?
            h = (
                self.table_data
                .fontMetrics()
                .boundingRect(constrained_rect, constants.TextFlag.TextWordWrap, text)
                .height()
            )

            height = max(height, h)

        height += padding

        self.table_index.setRowHeight(row_index, height)
        self.table_data.setRowHeight(row_index, height)

        self.table_data.updateGeometry()
        self.table_index.updateGeometry()

    def data_changed(self):
        # Call dataChanged on all models for all data
        for model in [
            self.table_data.model(),
            self.table_columns.model(),
            self.table_index.model(),
        ]:
            model.update_all()


class HeaderView(custom_widgets.OrientedTableView):
    """Displays the DataFrame index or columns depending on orientation."""

    def __init__(
        self,
        parent: DataFrameViewer,
        orientation: constants.Orientation,
        *,
        show_dtype_icons: bool = False,
    ):
        super().__init__(orientation, parent=parent)

        self.table = parent.table_data
        # These are used during column resizing
        self._header_is_resizing = None
        self.resize_start_pos: int | None = None
        self.initial_header_size = None

        # Handled by self.eventFilter()
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        self.viewport().installEventFilter(self)

        # Settings
        self.set_size_policy("maximum", "maximum")
        self.setWordWrap(False)
        self.set_horizontal_scrollbar_policy("always_off")
        self.set_scroll_mode("pixel")
        self.set_margin(0)

        def _kind_for_col(col: int):
            cols = self.table.get_model(skip_proxies=True).column_info
            return cols[col].kind if 0 <= col < len(cols) else None

        self.setItemDelegate(
            _viewer_style.HeaderCellDelegate(
                orientation,
                self,
                sort_state_getter=parent.get_sort_state if self.is_horizontal() else None,
                kind_getter=_kind_for_col if self.is_horizontal() else None,
                show_dtype_icons=show_dtype_icons and self.is_horizontal(),
            )
        )

        # Orientation specific settings
        if self.is_horizontal():
            self.set_horizontal_scrollbar_policy("always_off")
            self.h_header.hide()
            self.v_header.setSectionsClickable(False)
            # Selection lags without this
            self.v_header.setHighlightSections(False)
        else:
            self.set_vertical_scrollbar_policy("always_off")
            self.v_header.hide()
            self.h_header.setSectionsClickable(False)
            self.h_header.setHighlightSections(False)
        if self.is_horizontal():
            model = pandasmodels.HorizontalHeaderModel(parent.holder, parent=self)
        else:
            model = pandasmodels.VerticalHeaderModel(parent.holder, parent=self)
        self.set_model(model)

    def refresh(self, set_spans: bool = True):
        """Rebuild spans and sizes after the frame changed."""
        self.clearSpans()  # spans of a previous frame
        if set_spans and self.get_model(skip_proxies=True).df is not None:
            self.set_spans()
        self.init_size()
        # Set initial size
        self.resize(self.sizeHint())

    def get_higher_levels(self, levels: int | None = None) -> core.ItemSelection:
        model = self.model()
        if self.is_horizontal():
            # Get the header's selected columns
            # Removes the higher levels so that only the lowest level of the header
            # affects the data table selection
            last_row_ix = _nlevels(model.df, horizontal=True) - 1
            last_col_ix = model.columnCount() - 1
            return core.ItemSelection(
                model.index(0, 0), model.index(last_row_ix - 1, last_col_ix)
            )
        else:
            last_row_ix = model.rowCount() - 1
            last_col_ix = _nlevels(model.df, horizontal=False) - 1
            return core.ItemSelection(
                model.index(0, 0), model.index(last_row_ix, last_col_ix - 1)
            )

    # Take the current set of selected cells and make it so that any spanning cell above
    # a selected cell is selected too This should happen after every selection change
    def selectionChanged(self, selected, deselected):
        super().selectionChanged(selected, deselected)  # repaints the selection
        df = self.get_model(skip_proxies=True).df
        if df is None:
            return
        if self.is_horizontal():
            if _nlevels(df, horizontal=True) == 1:
                return
            for ix in self.selectedIndexes():
                for row in range(ix.row()):
                    ix2 = self.model().index(row, ix.column())
                    self.setSelection(self.visualRect(ix2), SelectionFlag.Select)
        else:
            if _nlevels(df, horizontal=False) == 1:
                return
            for ix in self.selectedIndexes():
                for col in range(ix.column()):
                    ix2 = self.model().index(ix.row(), col)
                    self.setSelection(self.visualRect(ix2), SelectionFlag.Select)

    # Fits columns to contents in O(1) time (bounded sample + last row for index width)
    def init_size(self):
        model = self.model()
        if model is None:
            return
        metrics = self.fontMetrics()
        pad = _viewer_style.HEADER_HPAD * 2 + 12
        colcount = min(model.columnCount(), 50)
        if self.is_horizontal():
            min_size = 80
            for col in range(colcount):
                width = min_size
                for row in range(model.rowCount()):
                    txt = str(model.data(model.index(row, col)) or "")
                    width = max(width, metrics.horizontalAdvance(txt) + pad)
                self.setColumnWidth(col, width)
                self.table.setColumnWidth(col, width)
        else:
            row_count = model.rowCount()
            sample = list(range(min(row_count, SAMPLE_ROWS)))
            if row_count > SAMPLE_ROWS:
                sample.append(row_count - 1)  # fit widest trailing row number
            for col in range(model.columnCount()):
                hdr = str(
                    model.headerData(col, constants.HORIZONTAL, constants.DISPLAY_ROLE)
                    or ""
                )
                width = max(36, metrics.horizontalAdvance(hdr) + pad)
                for row in sample:
                    txt = str(model.data(model.index(row, col)) or "")
                    width = max(width, metrics.horizontalAdvance(txt) + pad)
                self.setColumnWidth(col, min(width, 400))

    def set_spans(self):
        """Merge adjacent equal header labels into one spanning cell per level."""
        df = self.get_model(skip_proxies=True).df
        for level, start, length in axis_spans(df, self.is_horizontal()):
            self.set_section_span(level, start, length)

    def eventFilter(self, source: core.QObject, event: core.QEvent):
        if not isinstance(event, gui.QMouseEvent):
            return False
        mouse_pos = event.position().x() if self.is_horizontal() else event.position().y()
        mouse_pos = int(mouse_pos)
        # If mouse is on an edge, start the drag resize process
        match event.type():
            case core.QEvent.Type.MouseButtonPress:
                if self.over_header_edge(mouse_pos) is not None:
                    self._header_is_resizing = self.over_header_edge(mouse_pos)
                    self.resize_start_pos = mouse_pos
                    self.initial_header_size = self.sectionWidth(self._header_is_resizing)
                    return True
                else:
                    self._header_is_resizing = None
            # End the drag process
            case core.QEvent.Type.MouseButtonRelease:
                self._header_is_resizing = None

            case core.QEvent.Type.MouseButtonDblClick if self.is_horizontal():
                if (header_index := self.over_header_edge(mouse_pos)) is not None:
                    self.parent().auto_size_column(header_index)
                    return True

            case core.QEvent.Type.MouseButtonDblClick:
                if (header_index := self.over_header_edge(mouse_pos)) is not None:
                    self.parent().auto_size_row(header_index)
                    return True
            case core.QEvent.Type.MouseMove:
                # If this is None, there is no drag resize happening
                if self._header_is_resizing is not None:
                    size = self.initial_header_size + mouse_pos - self.resize_start_pos
                    if size > 10:
                        table_data = self.parent().table_data
                        self.setSectionWidth(self._header_is_resizing, size)
                        if self.is_horizontal():
                            table_data.setColumnWidth(self._header_is_resizing, size)
                        else:
                            table_data.setRowHeight(self._header_is_resizing, size)
                        self.updateGeometry()
                        table_data.updateGeometry()
                    return True

                # Set the cursor shape
                if self.over_header_edge(mouse_pos) is not None:
                    self.viewport().setCursor(self.get_split_cursor())
                else:
                    cursor = gui.Cursor(constants.CursorShape.ArrowCursor)
                    self.viewport().setCursor(cursor)
        return False

    # Return the size of the header needed to match the corresponding DataTableView
    def sizeHint(self):
        if self.model() is None:
            return core.Size(0, 0)
        # Horizontal HeaderView
        if self.is_horizontal():
            # Width of DataTableView
            width = self.table.sizeHint().width() + self.v_header.width()
            # Height (+1 for the bottom divider border)
            height = 2 * self.frameWidth() + 1
            max_rows = min(100, self.model().rowCount())
            for i in range(max_rows):
                height += self.rowHeight(i)

        # Vertical HeaderView
        else:
            # Height of DataTableView
            height = self.table.sizeHint().height() + self.h_header.height()
            # Width (+1 for the right divider border)
            width = 2 * self.frameWidth() + 1
            max_cols = min(100, self.model().columnCount())
            for i in range(max_cols):
                width += self.columnWidth(i)
        return core.Size(width, height)

    # This is needed because otherwise when the horizontal header is a single row
    # it will add whitespace to be bigger
    def minimumSizeHint(self):
        if self.is_horizontal():
            return core.Size(0, self.sizeHint().height())
        else:
            return core.Size(self.sizeHint().width(), 0)


# This is a fixed size widget with a size that tracks some other widget
class TrackingSpacer(widgets.Frame):
    def __init__(self, ref_x=None, ref_y=None):
        super().__init__()
        self.ref_x = ref_x
        self.ref_y = ref_y

    def minimumSizeHint(self):
        width = self.ref_x.width() if self.ref_x else 0
        height = self.ref_y.height() if self.ref_y else 0
        return core.Size(width, height)


if __name__ == "__main__":
    from prettyqt import widgets

    app = widgets.app()
    df = pd.DataFrame(
        np.random.default_rng().random((20, 4)),
        index=pd.MultiIndex.from_product([["a", "b"], range(10)]),
        columns=pd.MultiIndex.from_product([["x", "y"], ["1", "2"]]),
    )
    with app.debug_mode():
        view2 = DataFrameViewer(df)
        view2.show()
        app.exec()
