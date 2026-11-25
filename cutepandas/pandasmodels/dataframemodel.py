"""`DataFrameModel`: a virtual table model over any `FrameSource`."""

from __future__ import annotations

from dataclasses import dataclass, field
import enum
import logging
import threading
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
from prettyqt import constants, core, gui
import pyarrow as pa

from cutepandas.frames import (
    ChunkCache,
    ColumnChunk,
    ColumnInfo,
    EditableSource,
    FormatOptions,
    FrameSource,
    Kind,
    PermutableSource,
    SortableSource,
    format_chunk,
    format_name,
    formatter_for,
    is_polars_frame,
)
from cutepandas.pandasmodels.dataframeholder import DataFrameHolderMixin
from cutepandas.utils import icons


if TYPE_CHECKING:
    from collections.abc import Sequence

    from cutepandas.frames.formatters import Formatter


logger = logging.getLogger(__name__)

_RIGHT = constants.ALIGN_RIGHT | constants.ALIGN_V_CENTER
_LEFT = constants.ALIGN_LEFT | constants.ALIGN_V_CENTER
_PALETTE_STEPS = 64

type SortMode = Literal["copy", "permutation"]


@dataclass(slots=True)
class CellChunk:
    """One converted block of a column: everything `data()` returns, per row."""

    start: int
    values: list[Any]
    nulls: list[bool]
    display: list[str]
    scaled: bool = False
    """True once `normalized` / `colors` were computed (both stay None if the
    column is not numeric / boolean)."""
    normalized: list[float | None] | None = None
    colors: list[Any] | None = None


@dataclass(slots=True)
class _FrameState:
    """Everything derived from one frame.

    Replaced as a whole when the frame changes, so a worker that builds chunks always
    sees a consistent snapshot.
    """

    source: FrameSource | None
    columns: Sequence[ColumnInfo]
    formatters: list[Formatter]
    alignments: list[Any]
    row_count: int
    row_order: np.ndarray | None = None
    """Display order (row positions) or None, see `DataFrameHolder.row_order`."""
    ranges: dict[int, tuple[float, float] | None] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass(frozen=True, slots=True)
class SortPermutation:
    """A display-only sort order, see `DataFrameModel.compute_sort_permutation`."""

    column: int
    descending: bool
    positions: np.ndarray
    """Row positions of the frame in display order (8 bytes per row)."""
    generation: int
    """Data generation of the model it was computed for (stale check)."""


def _make_state(
    source: FrameSource | None,
    options: FormatOptions,
    row_order: np.ndarray | None = None,
) -> _FrameState:
    if source is None:
        return _FrameState(None, [], [], [], 0)
    columns = list(source.columns)
    return _FrameState(
        source=source,
        columns=columns,
        formatters=[formatter_for(c.kind, options) for c in columns],
        alignments=[_RIGHT if c.kind.is_numeric else _LEFT for c in columns],
        row_count=source.row_count,
        row_order=row_order,
    )


class DataFrameModel(DataFrameHolderMixin, core.AbstractTableModel):
    """Virtual table model for pandas, pyarrow and polars frames.

    The model reads through a `FrameSource` and converts only what the view asks for:
    rows are fetched in blocks of `chunk_rows` per column, formatted once (per-dtype
    formatters, see `FormatOptions`) and kept in a bounded LRU `ChunkCache`. A 10M-row
    frame therefore costs the frame itself plus at most `max_chunks` converted blocks.

    With `color_values=True`, numeric cells get a background whose intensity follows
    the value within the column's (min, max); booleans are green / red. The colours
    are computed per block (vectorised), not per `data()` call.

    Sorting (`sort`, e.g. from a header click) has two modes:

    - `"copy"` (default): the source builds a sorted copy of the frame
      (`SortableSource.sort`), which replaces the holder's frame. The copy is as large
      as the frame and the old frame stays alive as long as someone references it.
    - `"permutation"`: display-only. The source returns the sort order as row
      positions (`PermutableSource.argsort`, 8 bytes per row), stored as the holder's
      `row_order`; the frame is not copied or changed, and every visible block is
      read through the order (`PermutableSource.fetch_rows`). Row `i` of the model is
      row `source_row(i)` of the frame. Replacing the frame or `clear_sort()` drops
      the order.

    Threads: all Qt methods (`data`, `setData`, `sort`, assigning `df`, setters) run
    on the GUI thread. `prefetch()` may run on any thread; it only fills the cache.
    """

    SUPPORTS = pd.DataFrame

    class Roles(enum.IntEnum):
        """Role names."""

        DTypeRole = constants.USER_ROLE + 4555
        """Backend dtype object of the column."""
        ColumnNameRole = constants.USER_ROLE + 4556
        """Column label."""
        KindRole = constants.USER_ROLE + 4557
        """`Kind` of the column."""
        IsNullRole = constants.USER_ROLE + 4558
        """True if the cell is missing."""
        NormalizedRole = constants.USER_ROLE + 4559
        """Numeric cell value scaled to 0..1 within the column (None if not numeric)."""

    def __init__(
        self,
        df: Any = None,
        *args,
        chunk_rows: int = 256,
        max_chunks: int = 256,
        format_options: FormatOptions | None = None,
        color_values: bool = False,
        dtype_icons: bool = False,
        read_only: bool = False,
        sort_mode: SortMode = "copy",
        **kwargs,
    ):
        """Show a frame.

        Args:
            df: pandas DataFrame / Series, pyarrow Table, polars DataFrame, a
                `FrameSource` or a `DataFrameHolder` (to share the frame with
                other models).
            chunk_rows: Rows per cached block.
            max_chunks: Maximum number of cached blocks (all columns together).
            format_options: How values are formatted for display.
            color_values: Colour numeric / boolean cells by value.
            dtype_icons: Show a dtype icon in the column headers.
            read_only: Disable editing even if the backend supports it.
            sort_mode: `"copy"` (sorted copy of the frame) or `"permutation"`
                (display-only row order, no copy); see the class docstring.
            args: Passed on to the Qt base class.
            kwargs: Passed on to the Qt base class.
        """
        self._multiindex_separator = " | "
        self._options = format_options or FormatOptions()
        self._color_values = color_values
        self._dtype_icons = dtype_icons
        self._read_only = read_only
        self._sort_mode: SortMode = sort_mode
        self._data_generation = 0
        self._palette: list[Any] | None = None
        self._cache: ChunkCache[CellChunk] = ChunkCache(chunk_rows, max_chunks)
        self._chunk_rows = chunk_rows
        self._state = _make_state(None, self._options)
        super().__init__(*args, **kwargs)
        self._init_holder(df)
        self.holder.row_order_about_to_change.connect(self._on_row_order_about_to_change)
        self.holder.row_order_changed.connect(self._on_row_order_changed)
        self._refresh_state()

    @classmethod
    def supports(cls, instance) -> bool:
        return isinstance(instance, pd.DataFrame | pa.Table) or is_polars_frame(instance)

    # --- frame state ---

    def _refresh_state(self):
        self._cache.clear()
        self._state = _make_state(self.source, self._options, self.holder.row_order)

    def _on_holder_changed(self, origin: Any):
        # Also for changes made by this model (sort): the cache belongs to the old frame.
        self._data_generation += 1
        self._refresh_state()
        super()._on_holder_changed(origin)

    def _on_row_order_about_to_change(self, origin: Any):
        self.layoutAboutToBeChanged.emit()

    def _on_row_order_changed(self, origin: Any):
        self._refresh_state()  # blocks were read in the old order
        self.layoutChanged.emit()

    @property
    def cache(self) -> ChunkCache[CellChunk]:
        return self._cache

    @property
    def column_info(self) -> Sequence[ColumnInfo]:
        return self._state.columns

    # --- chunk building (thread-safe) ---

    def _column_range(self, state: _FrameState, col: int) -> tuple[float, float] | None:
        with state.lock:
            if col in state.ranges:
                return state.ranges[col]
        assert state.source is not None
        value_range = state.source.value_range(col)  # full scan, outside the lock
        with state.lock:
            state.ranges[col] = value_range
        return value_range

    def _get_palette(self) -> list[Any]:
        if self._palette is None:
            color = gui.QColor("#e4572e")
            self._palette = [
                gui.QColor(
                    color.red(),
                    color.green(),
                    color.blue(),
                    int(i * 200 / (_PALETTE_STEPS - 1)),
                )
                for i in range(_PALETTE_STEPS)
            ]
        return self._palette

    def _add_colors(self, state: _FrameState, col: int, chunk: CellChunk):
        chunk.scaled = True
        kind = state.columns[col].kind
        if kind == Kind.BOOLEAN:
            true, false = gui.QColor(0, 160, 0, 50), gui.QColor(220, 0, 0, 50)
            chunk.normalized = [None if v is None else float(v) for v in chunk.values]
            chunk.colors = [
                None if v is None else (true if v else false) for v in chunk.values
            ]
            return
        if kind not in {Kind.INTEGER, Kind.FLOAT, Kind.DECIMAL}:
            return
        value_range = self._column_range(state, col)
        if value_range is None:
            return
        lo, hi = value_range
        try:
            values = np.array(chunk.values, dtype=float)
        except TypeError, ValueError:  # complex numbers
            return
        span = hi - lo
        scaled = (values - lo) / span if span else np.zeros_like(values)
        valid = np.isfinite(scaled)
        scaled = np.where(valid, np.clip(scaled, 0.0, 1.0), 0.0)
        steps = (scaled * (_PALETTE_STEPS - 1)).astype(np.intp)
        palette = self._get_palette()
        chunk.normalized = [
            s if ok else None for s, ok in zip(scaled.tolist(), valid.tolist())
        ]
        chunk.colors = [
            palette[s] if ok else None for s, ok in zip(steps.tolist(), valid.tolist())
        ]

    def _build_chunk(self, block: int, col: int) -> CellChunk:
        state = self._state
        assert state.source is not None
        start = block * self._chunk_rows
        stop = min(start + self._chunk_rows, state.row_count)
        if state.row_order is None:
            [fetched] = state.source.fetch(start, stop, [col])
        else:
            assert isinstance(state.source, PermutableSource)
            [fetched] = state.source.fetch_rows(state.row_order[start:stop], [col])
        display = format_chunk(
            fetched.values, fetched.nulls, state.formatters[col], self._options.null_text
        )
        chunk = CellChunk(start, fetched.values, fetched.nulls.tolist(), display)
        if self._color_values:
            self._add_colors(state, col, chunk)
        return chunk

    def _chunk(self, row: int, col: int) -> CellChunk:
        block = row // self._chunk_rows
        chunk = self._cache.get(block, col)
        if chunk is None:
            chunk = self._cache.get_or_build(block, col, self._build_chunk)
        return chunk

    def _scaled_chunk(self, row: int, col: int) -> CellChunk:
        chunk = self._chunk(row, col)
        if not chunk.scaled:  # on demand, once per block (GUI thread)
            self._add_colors(self._state, col, chunk)
        return chunk

    def source_row(self, row: int) -> int:
        """Row position in the frame of the model's row `row` (differs once sorted)."""
        order = self._state.row_order
        return row if order is None else int(order[row])

    def fetch_window(
        self, start: int, stop: int, columns: Sequence[int]
    ) -> list[ColumnChunk]:
        """Model rows `start:stop` of the given columns, in display order."""
        state = self._state
        if state.source is None:
            return []
        if state.row_order is None:
            return state.source.fetch(start, stop, columns)
        assert isinstance(state.source, PermutableSource)
        return state.source.fetch_rows(state.row_order[start:stop], columns)

    def prefetch(
        self, first_row: int, last_row: int, columns: Sequence[int] | None = None
    ):
        """Convert the blocks covering rows `first_row..last_row` (any thread)."""
        state = self._state
        if state.source is None or state.row_count == 0:
            return
        last_row = min(last_row, state.row_count - 1)
        cols = range(len(state.columns)) if columns is None else columns
        for block in range(
            first_row // self._chunk_rows, last_row // self._chunk_rows + 1
        ):
            for col in cols:
                self._cache.get_or_build(block, col, self._build_chunk)

    # --- Qt model API ---

    def rowCount(self, parent: core.ModelIndex | None = None) -> int:
        if parent is not None and parent.isValid():
            return 0
        return self._state.row_count

    def columnCount(self, parent: core.ModelIndex | None = None) -> int:
        if parent is not None and parent.isValid():
            return 0
        return len(self._state.columns)

    def data(
        self,
        index: core.ModelIndex,
        role: constants.ItemDataRole = constants.DISPLAY_ROLE,
    ):
        if not index.isValid() or self._state.source is None:
            return None
        row, col = index.row(), index.column()
        match role:
            case constants.DISPLAY_ROLE:
                chunk = self._chunk(row, col)
                return chunk.display[row - chunk.start]
            case constants.ALIGNMENT_ROLE:
                return self._state.alignments[col]
            case constants.BACKGROUND_ROLE if self._color_values:
                chunk = self._scaled_chunk(row, col)
                return None if chunk.colors is None else chunk.colors[row - chunk.start]
            case constants.EDIT_ROLE | constants.USER_ROLE:
                chunk = self._chunk(row, col)
                return chunk.values[row - chunk.start]
            case constants.TOOLTIP_ROLE:
                chunk = self._chunk(row, col)
                offset = row - chunk.start
                return "null" if chunk.nulls[offset] else str(chunk.values[offset])
            case self.Roles.IsNullRole:
                chunk = self._chunk(row, col)
                return chunk.nulls[row - chunk.start]
            case self.Roles.NormalizedRole:
                chunk = self._scaled_chunk(row, col)
                if chunk.normalized is None:
                    return None
                return chunk.normalized[row - chunk.start]
            case self.Roles.DTypeRole:
                return self._state.columns[col].native_dtype
            case self.Roles.KindRole:
                return self._state.columns[col].kind
            case self.Roles.ColumnNameRole:
                return self._state.columns[col].name
        return None

    def headerData(
        self,
        section: int,
        orientation: constants.Orientation,
        role: constants.ItemDataRole = constants.DISPLAY_ROLE,
    ):
        columns = self._state.columns
        if orientation == constants.VERTICAL:
            if not 0 <= section < self._state.row_count:
                return None
            if role in {constants.DISPLAY_ROLE, constants.TOOLTIP_ROLE}:
                row = self.source_row(section)
                if isinstance(self.df, pd.DataFrame):
                    return format_name(self.df.index[row], self._multiindex_separator)
                return str(row)
            if role == constants.ALIGNMENT_ROLE:
                return constants.ALIGN_RIGHT | constants.ALIGN_V_CENTER
            return None
        if not 0 <= section < len(columns):
            return None
        info = columns[section]
        match role:
            case constants.DISPLAY_ROLE:
                return format_name(info.name, self._multiindex_separator)
            case constants.TOOLTIP_ROLE:
                label = format_name(info.name, self._multiindex_separator)
                return f"{label}\n{info.dtype}"
            case constants.ALIGNMENT_ROLE:
                return constants.ALIGN_CENTER
            case constants.DECORATION_ROLE if self._dtype_icons:
                return icons.icon_for_kind(info.kind)
            case self.Roles.DTypeRole:
                return info.native_dtype
            case self.Roles.KindRole:
                return info.kind
            case self.Roles.ColumnNameRole:
                return info.name
        return None

    def flags(self, index: core.ModelIndex) -> constants.ItemFlag:
        if not index.isValid():
            return constants.NO_FLAGS
        flags = constants.IS_ENABLED | constants.IS_SELECTABLE
        if self.is_editable():
            flags |= constants.IS_EDITABLE
        return flags

    def is_editable(self) -> bool:
        return not self._read_only and isinstance(self._state.source, EditableSource)

    def setData(
        self,
        index: core.ModelIndex,
        value: Any,
        role: constants.ItemDataRole = constants.EDIT_ROLE,
    ) -> bool:
        source = self._state.source
        if not index.isValid() or not self.is_editable():
            return False
        if role not in {constants.EDIT_ROLE, constants.USER_ROLE}:
            return False
        assert isinstance(source, EditableSource)
        row, col = index.row(), index.column()
        try:
            # In place: every model sharing the holder sees the same object.
            source.set_value(self.source_row(row), col, value)
        except Exception:
            logger.exception("Could not set %r at %s", value, (row, col))
            return False
        self._data_generation += 1  # a sort order computed before is stale now
        state = self._state
        with state.lock:
            state.ranges.pop(col, None)
        self._cache.invalidate(column=col)
        if self._color_values:  # the column range may have changed
            last = self.index(state.row_count - 1, col)
            self.dataChanged.emit(self.index(0, col), last)
        else:
            self.dataChanged.emit(index, index)
        return True

    def sort(self, ncol: int, order: constants.SortOrder = constants.ASCENDING):
        """Sort by column `ncol` (see `sort_mode`); unsupported sources are ignored."""
        source = self._state.source
        if not 0 <= ncol < self.columnCount():
            return
        descending = order == constants.DESCENDING
        if self._sort_mode == "permutation":
            if (permutation := self.compute_sort_permutation(ncol, order)) is not None:
                self.apply_sort_permutation(permutation)
            return
        if not isinstance(source, SortableSource):
            return
        new = source.sort(ncol, descending=descending)
        with self.change_layout():
            self.df = new

    def compute_sort_permutation(
        self, column: int, order: constants.SortOrder = constants.ASCENDING
    ) -> SortPermutation | None:
        """Compute a display-only sort order (any thread, e.g. a worker).

        Reads the frame through `PermutableSource.argsort` (stable, nulls last);
        the result is one integer per row. Returns None if the source can't sort.
        Apply it with `apply_sort_permutation` on the GUI thread.
        """
        generation = self._data_generation  # before reading: edits make it stale
        source = self._state.source
        if not isinstance(source, PermutableSource):
            return None
        descending = order == constants.DESCENDING
        positions = source.argsort(column, descending=descending)
        return SortPermutation(column, descending, positions, generation)

    def apply_sort_permutation(self, permutation: SortPermutation) -> bool:
        """Show the rows in a computed order (GUI thread).

        Returns False (and changes nothing) if the frame was replaced or edited
        since the order was computed.
        """
        if permutation.generation != self._data_generation:
            return False
        self.holder.set_row_order(permutation.positions, origin=self)
        return True

    def can_sort(self) -> bool:
        """True if the source supports sorting in the current `sort_mode`."""
        protocol = (
            PermutableSource if self._sort_mode == "permutation" else SortableSource
        )
        return isinstance(self._state.source, protocol)

    def clear_sort(self):
        """Back to the frame's own row order (drops a display-only sort)."""
        if self.holder.row_order is not None:
            self.holder.set_row_order(None, origin=self)

    @property
    def row_order(self) -> np.ndarray | None:
        """Display order of the rows (row positions), None if not permuted."""
        return self._state.row_order

    # --- options ---

    def _emit_all_changed(self):
        rows, cols = self.rowCount(), self.columnCount()
        if rows and cols:
            self.dataChanged.emit(self.index(0, 0), self.index(rows - 1, cols - 1))

    def set_format_options(self, options: FormatOptions):
        self._options = options
        self._refresh_state()
        self._emit_all_changed()

    def get_format_options(self) -> FormatOptions:
        return self._options

    def set_color_values(self, value: bool):
        self._color_values = value
        self._cache.clear()
        self._emit_all_changed()

    def get_color_values(self) -> bool:
        return self._color_values

    def set_dtype_icons(self, value: bool):
        self._dtype_icons = value
        if self.columnCount():
            self.headerDataChanged.emit(constants.HORIZONTAL, 0, self.columnCount() - 1)

    def get_dtype_icons(self) -> bool:
        return self._dtype_icons

    def set_sort_mode(self, mode: SortMode):
        if mode not in {"copy", "permutation"}:
            msg = f"Invalid sort mode {mode!r}"
            raise ValueError(msg)
        self._sort_mode = mode

    def get_sort_mode(self) -> SortMode:
        return self._sort_mode

    def set_read_only(self, value: bool):
        self._read_only = value

    def is_read_only(self) -> bool:
        return self._read_only

    def set_multiindex_separator(self, separator: str):
        self._multiindex_separator = separator

    def get_multiindex_separator(self) -> str:
        return self._multiindex_separator

    multiindex_separator = core.Property(
        str,
        get_multiindex_separator,
        set_multiindex_separator,
        doc="Separator to use for MultiIndexes",
    )
    color_values = core.Property(
        bool,
        get_color_values,
        set_color_values,
        doc="Colour numeric / boolean cells by value",
    )


class VerticalHeaderModel(DataFrameHolderMixin, core.AbstractTableModel):
    """Row labels: one column per pandas index level; row numbers for other backends."""

    def __init__(self, df: Any = None, **kwargs):
        super().__init__(**kwargs)
        self._init_holder(df)
        # A display-only sort (holder.row_order) moves the row labels too.
        self.holder.row_order_about_to_change.connect(self._on_row_order_about_to_change)
        self.holder.row_order_changed.connect(self._on_row_order_changed)

    def _on_row_order_about_to_change(self, origin: Any):
        self.layoutAboutToBeChanged.emit()

    def _on_row_order_changed(self, origin: Any):
        self.layoutChanged.emit()

    def _row(self, row: int) -> int:
        order = self.holder.row_order
        return row if order is None else int(order[row])

    def _index(self) -> pd.Index | None:
        df = self.df
        return df.index if isinstance(df, pd.DataFrame) else None

    def columnCount(self, parent: core.ModelIndex | None = None) -> int:
        if self.df is None or (parent is not None and parent.isValid()):
            return 0
        index = self._index()
        return 1 if index is None else index.nlevels

    def rowCount(self, parent: core.ModelIndex | None = None) -> int:
        if parent is not None and parent.isValid():
            return 0
        source = self.source
        return 0 if source is None else source.row_count

    def data(
        self,
        index: core.ModelIndex,
        role: constants.ItemDataRole = constants.DISPLAY_ROLE,
    ):
        if self.df is None or not index.isValid():
            return None
        match role:
            case constants.DISPLAY_ROLE | constants.TOOLTIP_ROLE:
                pd_index = self._index()
                row = self._row(index.row())
                if pd_index is None:
                    return str(row)
                value = pd_index[row]
                if isinstance(pd_index, pd.MultiIndex):
                    value = value[index.column()]
                return str(value)
        return None

    def headerData(
        self,
        section: int,
        orientation: constants.Orientation,
        role: constants.ItemDataRole = constants.DISPLAY_ROLE,
    ) -> str | None:
        match role, orientation:
            case constants.DISPLAY_ROLE | constants.TOOLTIP_ROLE, constants.HORIZONTAL:
                pd_index = self._index()
                if pd_index is None:
                    return "" if section == 0 and self.df is not None else None
                if not 0 <= section < pd_index.nlevels:
                    return None
                name = pd_index.names[section]
                return "" if name is None else str(name)
        return None


class HorizontalHeaderModel(DataFrameHolderMixin, core.AbstractTableModel):
    """Column labels: one row per pandas column level; one row for other backends."""

    def __init__(self, df: Any = None, **kwargs):
        super().__init__(**kwargs)
        self._init_holder(df)

    def _pd_columns(self) -> pd.Index | None:
        df = self.df
        return df.columns if isinstance(df, pd.DataFrame) else None

    def columnCount(self, parent: core.ModelIndex | None = None) -> int:
        if parent is not None and parent.isValid():
            return 0
        source = self.source
        return 0 if source is None else len(source.columns)

    def rowCount(self, parent: core.ModelIndex | None = None) -> int:
        if self.df is None or (parent is not None and parent.isValid()):
            return 0
        pd_columns = self._pd_columns()
        return 1 if pd_columns is None else pd_columns.nlevels

    def data(
        self,
        index: core.ModelIndex,
        role: constants.ItemDataRole = constants.DISPLAY_ROLE,
    ):
        source = self.source
        if source is None or not index.isValid():
            return None
        match role:
            case constants.DISPLAY_ROLE | constants.TOOLTIP_ROLE:
                value = source.columns[index.column()].name
                if isinstance(self._pd_columns(), pd.MultiIndex):
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
        match role, orientation:
            case constants.DISPLAY_ROLE | constants.TOOLTIP_ROLE, constants.VERTICAL:
                pd_columns = self._pd_columns()
                if pd_columns is None:
                    return "" if section == 0 and self.df is not None else None
                if not 0 <= section < pd_columns.nlevels:
                    return None
                name = pd_columns.names[section]
                return "" if name is None else str(name)
        return None


if __name__ == "__main__":
    from prettyqt import widgets

    app = widgets.app()
    frame = pd.DataFrame(np.random.randn(100_000, 20)).add_prefix("col_")
    table = widgets.TableView()
    table.set_model(DataFrameModel(frame, color_values=True, parent=table))
    table.show()
    app.exec()
