from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal, get_args

import numpy as np
import pandas as pd
from prettyqt import constants, core


if TYPE_CHECKING:
    from collections.abc import Sequence


logger = logging.getLogger(__name__)


FilterModeStr = Literal["startswith", "endswith", "contains", "match"]
FILTER_MODES: tuple[str, ...] = get_args(FilterModeStr)

# Source signals after which the row mapping has to be rebuilt. Each pair is
# (about-to signal, done signal); the proxy answers them with a model reset.
_STRUCTURE_SIGNALS = [
    ("modelAboutToBeReset", "modelReset"),
    ("layoutAboutToBeChanged", "layoutChanged"),
    ("rowsAboutToBeInserted", "rowsInserted"),
    ("rowsAboutToBeRemoved", "rowsRemoved"),
    ("rowsAboutToBeMoved", "rowsMoved"),
    ("columnsAboutToBeInserted", "columnsInserted"),
    ("columnsAboutToBeRemoved", "columnsRemoved"),
    ("columnsAboutToBeMoved", "columnsMoved"),
]


def _as_string_series(series: pd.Series) -> pd.Series:
    """Return a Series usable with the `.str` accessor, keeping missing values."""
    if series.dtype == object or pd.api.types.is_string_dtype(series.dtype):
        return series
    return series.astype("string")


def _to_mask(result: Any, length: int, na_value: bool = False) -> np.ndarray:
    """Turn a boolean-ish Series into a plain NumPy bool mask (NA -> na_value)."""
    if not isinstance(result, pd.Series) or len(result) != length:
        return np.zeros(length, dtype=bool)
    if not (pd.api.types.is_bool_dtype(result.dtype) or result.dtype == object):
        return np.zeros(length, dtype=bool)
    # Object / nullable boolean results may hold NA; fillna keeps the mask boolean.
    return result.fillna(na_value).to_numpy(dtype=bool)


class BasePandasIndexFilterProxyModel(core.AbstractProxyModel):
    """Base proxy to filter a pandas DataFrame based on a calculated boolean index.

    Subclasses implement `compute_filter`, returning a boolean mask (one entry per
    source row) or None to show all rows. Mappings in both directions are then
    calculated with NumPy.

    Faster than the row-based SortFilterProxyModel approach, and it doesn't need a
    copy of the DataFrame: only the mapping arrays are built.

    The mask is recomputed whenever the source model changes its structure (reset,
    layout change such as sorting, inserted / removed / moved rows or columns).

    Note: the DataFrame is taken from the innermost source model, so these proxies
    must sit directly on top of a DataFrame model (they can't be chained).
    """

    def __init__(self, **kwargs):
        self._filter_index = np.ones(0, dtype=bool)
        self._source_to_proxy = np.zeros(0, dtype=np.intp)
        self._proxy_to_source = np.zeros(0, dtype=np.intp)
        self._row_count = 0
        self._connections: list[tuple[Any, Any]] = []
        super().__init__(**kwargs)

    # --- filter computation ---

    def compute_filter(self, df: pd.DataFrame) -> np.ndarray | None:
        """Return a boolean mask with one entry per row, or None for "show all"."""
        return None

    def get_df(self) -> pd.DataFrame | None:
        source = self.get_source_model(skip_proxies=True)
        return getattr(source, "df", None)

    def invalidate_filter(self):
        """Recompute the filter mask and reset the proxy."""
        with self.reset_model():
            self._apply_filter()

    def _apply_filter(self):
        source = self.sourceModel()
        rowcount = source.rowCount() if source is not None else 0
        mask = None
        df = self.get_df()
        if df is not None and rowcount:
            try:
                mask = self.compute_filter(df)
            except Exception:
                logger.exception("Filtering failed, showing no rows")
                mask = np.zeros(rowcount, dtype=bool)
        if mask is None:
            mask = np.ones(rowcount, dtype=bool)
        mask = np.asarray(mask, dtype=bool)
        if len(mask) != rowcount:
            logger.warning("Filter mask has %s entries for %s rows", len(mask), rowcount)
            mask = np.zeros(rowcount, dtype=bool)
        self._filter_index = mask
        self._source_to_proxy = np.cumsum(mask) - 1
        self._proxy_to_source = np.flatnonzero(mask)
        self._row_count = len(self._proxy_to_source)

    # --- source model handling ---

    def setSourceModel(self, model):
        for signal, slot in self._connections:
            signal.disconnect(slot)
        self._connections = []
        with self.reset_model():
            super().setSourceModel(model)
            self._apply_filter()
        if model is None:
            return
        for about_to, done in _STRUCTURE_SIGNALS:
            self._connect(getattr(model, about_to), self._on_source_about_to_change)
            self._connect(getattr(model, done), self._on_source_changed)
        self._connect(model.dataChanged, self._on_source_data_changed)
        self._connect(model.headerDataChanged, self._on_source_header_changed)

    def _connect(self, signal, slot):
        signal.connect(slot)
        self._connections.append((signal, slot))

    def _on_source_about_to_change(self, *_args):
        self.beginResetModel()

    def _on_source_changed(self, *_args):
        self._apply_filter()
        self.endResetModel()

    def _on_source_data_changed(
        self,
        top_left: core.ModelIndex,
        bottom_right: core.ModelIndex,
        roles: Sequence[int] = (),
    ):
        first, last = top_left.row(), bottom_right.row()
        visible = np.flatnonzero(self._filter_index[first : last + 1])
        if not len(visible):
            return
        top = int(self._source_to_proxy[first + visible[0]])
        bottom = int(self._source_to_proxy[first + visible[-1]])
        self.dataChanged.emit(
            self.index(top, top_left.column()),
            self.index(bottom, bottom_right.column()),
            list(roles),
        )

    def _on_source_header_changed(
        self, orientation: constants.Orientation, first: int, last: int
    ):
        if orientation == constants.HORIZONTAL:
            self.headerDataChanged.emit(orientation, first, last)
        elif self._row_count:
            self.headerDataChanged.emit(orientation, 0, self._row_count - 1)

    # --- QAbstractItemModel interface ---

    def rowCount(self, parent: core.ModelIndex | None = None) -> int:
        if parent is not None and parent.isValid():
            return 0
        return self._row_count

    def columnCount(self, parent: core.ModelIndex | None = None) -> int:
        source = self.sourceModel()
        if source is None or (parent is not None and parent.isValid()):
            return 0
        return source.columnCount()

    def hasChildren(self, parent: core.ModelIndex | None = None) -> bool:
        if parent is not None and parent.isValid():
            return False
        return self.rowCount() > 0 and self.columnCount() > 0

    def index(
        self, row: int, column: int, parent: core.ModelIndex | None = None
    ) -> core.ModelIndex:
        if (parent is not None and parent.isValid()) or not (
            0 <= row < self._row_count and 0 <= column < self.columnCount()
        ):
            return core.ModelIndex()
        return self.createIndex(row, column)

    def parent(self, *args):
        if args:  # parent(index): a flat table, so there is never a parent index
            return core.ModelIndex()
        return super().parent()  # QObject.parent()

    def headerData(
        self,
        section: int,
        orientation: constants.Orientation,
        role: constants.ItemDataRole = constants.DISPLAY_ROLE,
    ):
        """Map header data to proxy by using proxy-to-source-map."""
        source = self.sourceModel()
        if source is None:
            return None
        if orientation == constants.HORIZONTAL:
            return source.headerData(section, orientation, role)
        if not 0 <= section < self._row_count:
            return None
        return source.headerData(int(self._proxy_to_source[section]), orientation, role)

    def mapToSource(self, proxy_idx: core.ModelIndex) -> core.ModelIndex:
        """Map index to source by using proxy-to-source map."""
        source = self.sourceModel()
        if (
            source is None
            or not proxy_idx.isValid()
            or not 0 <= proxy_idx.row() < self._row_count
        ):
            return core.ModelIndex()
        row_pos = int(self._proxy_to_source[proxy_idx.row()])
        return source.index(row_pos, proxy_idx.column())

    def mapFromSource(self, source_index: core.ModelIndex) -> core.ModelIndex:
        row = source_index.row()
        if (
            self.sourceModel() is None
            or not source_index.isValid()
            or not 0 <= row < len(self._filter_index)
            or not self._filter_index[row]  # filtered out: no proxy index
        ):
            return core.ModelIndex()
        return self.index(int(self._source_to_proxy[row]), source_index.column())


class PandasStringColumnFilterProxyModel(BasePandasIndexFilterProxyModel):
    """Filters a DataFrame by df.iloc[:, column].str.<filter_mode>(search_term)."""

    ID = "pandas_str_filter"

    def __init__(self, **kwargs):
        self._filter_column = 0
        self._filter_mode: FilterModeStr = "startswith"
        self._case_sensitive = True
        self._flags = 0
        self._search_term = ""
        self._na_value = False
        super().__init__(**kwargs)

    def compute_filter(self, df: pd.DataFrame) -> np.ndarray | None:
        if not self._search_term:
            return None
        if not 0 <= self._filter_column < len(df.columns):
            return np.zeros(len(df), dtype=bool)
        column = _as_string_series(df.iloc[:, self._filter_column])
        term = self._search_term
        match self._filter_mode:
            case "startswith" | "endswith" if not self._case_sensitive:
                method = getattr(column.str.lower().str, self._filter_mode)
                result = method(term.lower(), na=self._na_value)
            case "startswith" | "endswith":
                result = getattr(column.str, self._filter_mode)(term, na=self._na_value)
            case "contains" | "match":
                method = getattr(column.str, self._filter_mode)
                result = method(
                    term, case=self._case_sensitive, flags=self._flags, na=self._na_value
                )
        return _to_mask(result, len(df), self._na_value)

    def _refilter(self):
        if self.sourceModel() is not None:
            self.invalidate_filter()

    def set_search_term(self, search_term: str):
        self._search_term = search_term
        self._refilter()

    def get_search_term(self) -> str:
        return self._search_term

    def set_filter_column(self, column: int):
        self._filter_column = column
        self._refilter()

    def get_filter_column(self) -> int:
        return self._filter_column

    def set_filter_mode(self, mode: FilterModeStr):
        if mode not in FILTER_MODES:
            msg = f"Invalid filter mode {mode!r}. Valid: {FILTER_MODES}"
            raise ValueError(msg)
        self._filter_mode = mode
        self._refilter()

    def get_filter_mode(self) -> FilterModeStr:
        return self._filter_mode

    def set_case_sensitive(self, mode: bool):
        self._case_sensitive = mode
        self._refilter()

    def is_case_sensitive(self) -> bool:
        return self._case_sensitive

    def set_flags(self, flags: int):
        self._flags = flags
        self._refilter()

    def get_flags(self) -> int:
        return self._flags

    def set_na_value(self, value: bool):
        self._na_value = value
        self._refilter()

    def get_na_value(self) -> bool:
        return self._na_value

    search_term = core.Property(
        str,
        get_search_term,
        set_search_term,
        doc="Term to use for filtering",
    )
    filter_column = core.Property(
        int,
        get_filter_column,
        set_filter_column,
        doc="Column to use for filtering",
    )
    filter_mode = core.Property(
        str,
        get_filter_mode,
        set_filter_mode,
        doc="Filter mode",
    )
    case_sensitive = core.Property(
        bool,
        is_case_sensitive,
        set_case_sensitive,
        doc="Case sensitivity of the filter",
    )
    re_flags = core.Property(
        int,
        get_flags,
        set_flags,
        doc="Regex Flags for match and contains mode",
    )
    na_value = core.Property(
        bool,
        get_na_value,
        set_na_value,
        doc="Whether rows with missing values are shown",
    )


class PandasEvalFilterProxyModel(BasePandasIndexFilterProxyModel):
    """Filters a DataFrame by a boolean `DataFrame.eval` expression, like 'a > 10'.

    Invalid expressions, or ones that don't evaluate to a boolean Series, show no rows.
    """

    ID = "pandas_eval_filter"

    def __init__(self, **kwargs):
        self._expression = ""
        super().__init__(**kwargs)

    def compute_filter(self, df: pd.DataFrame) -> np.ndarray | None:
        if not self._expression:
            return None
        try:
            result = df.eval(self._expression)
        except Exception:  # noqa: BLE001 (user input: any syntax / name / type error)
            logger.debug("Invalid filter expression %r", self._expression)
            return np.zeros(len(df), dtype=bool)
        if not pd.api.types.is_bool_dtype(getattr(result, "dtype", None)):
            return np.zeros(len(df), dtype=bool)
        return _to_mask(result, len(df))

    def set_expression(self, expression: str):
        self._expression = expression
        if self.sourceModel() is not None:
            self.invalidate_filter()

    def get_expression(self) -> str:
        return self._expression

    expression = core.Property(
        str,
        get_expression,
        set_expression,
        doc="Expression to use for filtering",
    )


class PandasMultiStringColumnFilterProxyModel(BasePandasIndexFilterProxyModel):
    """Shows rows where every given column starts with its given prefix.

    Filters are a mapping of column label to prefix; empty prefixes are ignored.
    """

    def __init__(self, **kwargs):
        self._filters: dict[str, str] = {}
        super().__init__(**kwargs)

    def compute_filter(self, df: pd.DataFrame) -> np.ndarray | None:
        active = {k: v for k, v in self._filters.items() if v}
        if not active:
            return None
        mask = np.ones(len(df), dtype=bool)
        for column, prefix in active.items():
            if column not in df.columns:
                return np.zeros(len(df), dtype=bool)
            series = _as_string_series(df[column])
            mask &= _to_mask(series.str.startswith(prefix, na=False), len(df))
        return mask

    def set_filters(self, filters: dict[str, str]):
        self._filters = dict(filters)
        if self.sourceModel() is not None:
            self.invalidate_filter()

    def get_filters(self) -> dict[str, str]:
        return self._filters

    filters = core.Property(
        dict,
        get_filters,
        set_filters,
        doc="Dictionary containing a Column->Prefix mapping",
    )


if __name__ == "__main__":
    from prettyqt import debugging, widgets

    from cutepandas import pandasmodels

    app = widgets.app()
    a = pd.Series(["a", "bc", "c", "d", "aa"] * 100000, dtype=pd.StringDtype())
    b = pd.Series(["a", "b", "c", "fjkdsj", "fdf"] * 100000, dtype=pd.StringDtype())
    df = pd.DataFrame(dict(a=a, b=b))
    model = pandasmodels.DataFrameModel(df)
    proxy = PandasStringColumnFilterProxyModel(parent=model)
    proxy.setSourceModel(model)
    w = debugging.ProxyComparerWidget(proxy)
    w.show()
    with app.debug_mode():
        app.exec()
