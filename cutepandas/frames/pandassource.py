"""`FrameSource` adapter for pandas DataFrames (the first-class backend)."""

from __future__ import annotations

from functools import cached_property
import math
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from pandas.api import types as ptypes
import pyarrow as pa

from cutepandas.frames.arrowsource import kind_for_arrow_type
from cutepandas.frames.source import ColumnChunk, ColumnInfo, ColumnStats, Kind


if TYPE_CHECKING:
    from collections.abc import Sequence


def kind_for_pandas_dtype(dtype: Any) -> Kind:
    """Map a pandas / NumPy dtype (incl. `pd.ArrowDtype`, or its name) to a `Kind`."""
    if isinstance(dtype, str):
        dtype = ptypes.pandas_dtype(dtype)
    if isinstance(dtype, pd.ArrowDtype):
        return kind_for_arrow_type(dtype.pyarrow_dtype)
    if isinstance(dtype, pd.CategoricalDtype):
        return Kind.CATEGORICAL
    if ptypes.is_bool_dtype(dtype):
        return Kind.BOOLEAN
    if ptypes.is_integer_dtype(dtype):
        return Kind.INTEGER
    if ptypes.is_float_dtype(dtype) or ptypes.is_complex_dtype(dtype):
        return Kind.FLOAT
    if ptypes.is_datetime64_any_dtype(dtype) or isinstance(dtype, pd.PeriodDtype):
        return Kind.DATETIME
    if ptypes.is_timedelta64_dtype(dtype):
        return Kind.DURATION
    if isinstance(dtype, pd.StringDtype):
        return Kind.STRING
    return Kind.OBJECT


def _python(value: Any) -> Any:
    """NumPy scalars to builtins (pandas results of min / max / mean)."""
    if isinstance(value, np.generic):
        value = value.item()
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


class PandasSource:
    """Source over a pandas DataFrame (a Series is shown as a one-column frame).

    Supports every pandas dtype, incl. pyarrow-backed ones (`string[pyarrow]`,
    `pd.ArrowDtype`). Implements the optional `SortableSource` and `EditableSource`
    hooks: `set_value` writes in place (`DataFrame.iat`).
    """

    backend = "pandas"

    def __init__(self, df: pd.DataFrame | pd.Series):
        self._df = df.to_frame() if isinstance(df, pd.Series) else df

    @property
    def native(self) -> pd.DataFrame:
        return self._df

    @property
    def row_count(self) -> int:
        return len(self._df)

    @cached_property
    def columns(self) -> Sequence[ColumnInfo]:
        return [
            ColumnInfo(name, str(dtype), kind_for_pandas_dtype(dtype), dtype)
            for name, dtype in zip(self._df.columns, self._df.dtypes)
        ]

    def _column(self, column: int) -> pd.Series:
        return self._df.iloc[:, column]

    def fetch(self, start: int, stop: int, columns: Sequence[int]) -> list[ColumnChunk]:
        window = self._df.iloc[start:stop]  # a view (copy-on-write)
        chunks = []
        for j in columns:
            series = window.iloc[:, j]
            nulls = series.isna().to_numpy(dtype=bool)
            values = series.tolist()
            if nulls.any():
                values = [None if n else v for v, n in zip(values, nulls)]
            chunks.append(ColumnChunk(values, nulls))
        return chunks

    def fetch_arrow(self, start: int, stop: int, columns: Sequence[int]) -> pa.Table:
        window = self._df.iloc[start:stop, list(columns)]
        names = [str(c) for c in window.columns]
        arrays = [pa.array(window.iloc[:, j]) for j in range(window.shape[1])]
        return pa.table(arrays, names=names)

    def null_count(self, column: int) -> int:
        return int(self._column(column).isna().sum())

    def nbytes(self, column: int) -> int:
        return int(self._column(column).nbytes)

    def value_range(self, column: int) -> tuple[float, float] | None:
        info = self.columns[column]
        if not (info.kind.is_numeric or info.kind == Kind.BOOLEAN):
            return None
        series = self._column(column)
        if ptypes.is_complex_dtype(series.dtype):
            series = series.abs()
        try:
            lo, hi = _python(series.min()), _python(series.max())
        except TypeError:
            return None
        if lo is None or hi is None:
            return None
        return float(lo), float(hi)

    def value_counts(
        self, column: int, limit: int | None = None
    ) -> list[tuple[Any, int]]:
        counts = self._column(column).value_counts(dropna=True, sort=True)
        if limit is not None:
            counts = counts.head(limit)
        return [(_python(k), int(v)) for k, v in counts.items()]

    def categories(self, column: int) -> list[Any] | None:
        dtype = self._column(column).dtype
        if isinstance(dtype, pd.CategoricalDtype):
            return [_python(c) for c in dtype.categories]
        if isinstance(dtype, pd.ArrowDtype) and pa.types.is_dictionary(
            dtype.pyarrow_dtype
        ):
            return _arrow_dictionary(self._column(column))
        return None

    def describe(self, column: int, top: int = 10) -> ColumnStats:
        info = self.columns[column]
        series = self._column(column)
        lo = hi = mean = std = None
        orderable = info.kind.is_numeric or info.kind.is_temporal
        if orderable or info.kind == Kind.STRING:
            try:
                lo, hi = _python(series.min()), _python(series.max())
            except TypeError:
                lo = hi = None
        if info.kind.is_numeric and not ptypes.is_complex_dtype(series.dtype):
            mean, std = _python(series.mean()), _python(series.std())
        try:
            unique = int(series.nunique(dropna=True))
        except TypeError:  # unhashable objects
            unique = None
        return ColumnStats(
            count=len(series),
            null_count=int(series.isna().sum()),
            unique=unique,
            min=lo,
            max=hi,
            mean=None if mean is None else float(mean),
            std=None if std is None else float(std),
            top=self.value_counts(column, limit=top) if unique is not None else [],
        )

    def sort(self, column: int, descending: bool = False) -> pd.DataFrame:
        # By position (labels are ambiguous for duplicate columns) and stable, so
        # sorting by one column after another keeps the earlier order for ties.
        series = self._column(column)
        if not descending:
            positions = series.argsort(kind="stable").to_numpy()  # fastest; NaN last
        else:
            # argsort can't sort descending; this keeps ties stable and NaN last
            ranked = series.reset_index(drop=True)
            ranked = ranked.sort_values(ascending=False, kind="stable")
            positions = ranked.index.to_numpy()
        return self._df.take(positions)

    def set_value(self, row: int, column: int, value: Any) -> None:
        self._df.iat[row, column] = value


def _arrow_dictionary(series: pd.Series) -> list[Any]:
    array = pa.array(series)
    if isinstance(array, pa.ChunkedArray):
        if array.num_chunks == 0:
            return []
        array = array.unify_dictionaries().chunk(0)
    return array.dictionary.to_pylist()
