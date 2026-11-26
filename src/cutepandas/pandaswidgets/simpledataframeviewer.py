"""Single-table DataFrame viewer built on one QTableView and DataFrameModel."""

import logging
from typing import TYPE_CHECKING, Any

from prettyqt import constants, core, gui, widgets

from cutepandas import pandasmodels
from cutepandas.pandasmodels import DataFrameHolder
from cutepandas.pandaswidgets import _viewer_style
from cutepandas.pandaswidgets.sortcontroller import SortableViewerMixin, SortController
from cutepandas.utils import helpers


if TYPE_CHECKING:
    from cutepandas.pandasmodels.dataframemodel import SortMode


logger = logging.getLogger(__name__)

ROW_PADDING = 10


class SimpleDataFrameViewer(SortableViewerMixin, widgets.TableView):
    """Lightweight single-table viewer backed by `DataFrameModel`.

    Uses Qt's native horizontal and vertical `QHeaderView`s (joining MultiIndex
    levels with a separator) instead of separate header tables. Supports shared
    `DataFrameHolder`s, dtype icons, clipboard copy/paste, and both `"copy"` and
    `"permutation"` sort modes with native sort indicators.
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
        color_values: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._dtype_icons = dtype_icons
        self._autosized_cols: set[int] = set()

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
        self.sorter = SortController(model, sortable=sortable, parent=self)
        self._connect_sorter()
        self.sorter.sort_state_changed.connect(self._on_sort_state_changed)
        self.h_scrollbar.valueChanged.connect(self._autosize_visible_columns)

        self._apply_style()

        self.holder.changed.connect(self._on_df_changed)
        self._on_df_changed()

    def _apply_style(self):
        """Derive the colours from the palette (light or dark theme).

        Only the viewport and the header widgets get palettes / style sheets, so the
        view's own palette keeps following the application's.
        """
        colors = _viewer_style.ViewerColors.from_palette(self.palette())
        self.setShowGrid(False)  # the delegate paints hairlines
        viewport = self.viewport()
        palette: Any = gui.QPalette(viewport.palette())
        palette.setColor(gui.Palette.ColorRole.Base, colors.base)
        palette.setColor(gui.Palette.ColorRole.AlternateBase, colors.alt_row)
        viewport.setPalette(palette)
        header_style = _viewer_style.native_header_style(colors)
        self.h_header.setStyleSheet(header_style)
        self.v_header.setStyleSheet(header_style)
        _viewer_style.restyle_view(self, colors)

    def changeEvent(self, event: core.QEvent):
        super().changeEvent(event)
        if event.type() in {
            core.QEvent.Type.PaletteChange,
            core.QEvent.Type.StyleChange,  # e.g. an application style sheet (theme)
        }:
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
        self._autosized_cols.clear()
        self._resize_columns_and_rows()

    def _resize_columns_and_rows(self):
        model = self.model()
        if model is None:
            return
        row_height = self.fontMetrics().height() + ROW_PADDING
        self.v_header.setDefaultSectionSize(row_height)
        for col in range(min(model.columnCount(), _viewer_style.AUTOSIZE_COLUMNS)):
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
        header_text = model.headerData(
            column_index, constants.HORIZONTAL, constants.DISPLAY_ROLE
        )
        rows = range(min(model.rowCount(), _viewer_style.SAMPLE_ROWS))
        width = _viewer_style.fitted_column_width(
            [str(header_text or "")],
            self.h_header.font(),
            (str(model.data(model.index(r, column_index)) or "") for r in rows),
            self.fontMetrics(),
            icon_space=_viewer_style.NATIVE_HEADER_ICON_SPACE if self._dtype_icons else 0,
        )
        self.setColumnWidth(column_index, width)

    # --- sorting ---

    def _on_header_clicked(self, column: int):
        self.sorter.request_sort(column)

    def _on_sort_state_changed(self, state: tuple[int, constants.SortOrder] | None):
        if state is None:
            self.h_header.setSortIndicatorShown(False)
        else:
            self.h_header.setSortIndicator(*state)
            self.h_header.setSortIndicatorShown(True)

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
