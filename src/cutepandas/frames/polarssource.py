"""`FrameSource` adapter for polars DataFrames (needs the `polars` extra)."""

from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from cutepandas.frames.source import ColumnChunk, ColumnInfo, ColumnStats, Kind


if TYPE_CHECKING:
    from collections.abc import Sequence

    import pyarrow as pa


def kind_for_polars_dtype(dtype: pl.DataType) -> Kind:
    """Map a polars dtype to a `Kind`."""
    match dtype:
        case pl.Categorical() | pl.Enum():
            return Kind.CATEGORICAL
        case pl.Boolean():
            return Kind.BOOLEAN
        case pl.Decimal():
            return Kind.DECIMAL
        case pl.String():
            return Kind.STRING
        case pl.Datetime():
            return Kind.DATETIME
        case pl.Date():
            return Kind.DATE
        case pl.Time():
            return Kind.TIME
        case pl.Duration():
            return Kind.DURATION
        case pl.Binary():
            return Kind.BINARY
        case pl.Null():
            return Kind.NULL
        case pl.List() | pl.Array() | pl.Struct():
            return Kind.NESTED
    if dtype.is_integer():
        return Kind.INTEGER
    if dtype.is_float():
        return Kind.FLOAT
    return Kind.OBJECT


def _number(value: Any) -> float | None:
    return None if value is None else float(value)


def _chunk(series: pl.Series) -> ColumnChunk:
    return ColumnChunk(series.to_list(), series.is_null().to_numpy())


class PolarsSource:
    """Source over a polars DataFrame.

    Implements the optional `SortableSource`, `PermutableSource` and
    `EditableSource` hooks; `set_value`
    changes the frame in place (`df[row, column] = value`).
    """

    backend = "polars"

    def __init__(self, df: pl.DataFrame):
        self._df = df

    @property
    def native(self) -> pl.DataFrame:
        return self._df

    @property
    def row_count(self) -> int:
        return self._df.height

    @cached_property
    def columns(self) -> Sequence[ColumnInfo]:
        return [
            ColumnInfo(name, str(dtype), kind_for_polars_dtype(dtype), dtype)
            for name, dtype in self._df.schema.items()
        ]

    def _column(self, column: int) -> pl.Series:
        return self._df.to_series(column)

    def fetch(self, start: int, stop: int, columns: Sequence[int]) -> list[ColumnChunk]:
        length = max(0, stop - start)
        return [_chunk(self._column(j).slice(start, length)) for j in columns]

    def fetch_rows(self, rows: np.ndarray, columns: Sequence[int]) -> list[ColumnChunk]:
        return [_chunk(self._column(j).gather(rows)) for j in columns]

    def fetch_arrow(self, start: int, stop: int, columns: Sequence[int]) -> pa.Table:
        names = [self._df.columns[j] for j in columns]
        return self._df.slice(start, max(0, stop - start)).select(names).to_arrow()

    def null_count(self, column: int) -> int:
        return self._column(column).null_count()

    def nbytes(self, column: int) -> int:
        return int(self._column(column).estimated_size())

    def value_range(self, column: int) -> tuple[float, float] | None:
        info = self.columns[column]
        if not (info.kind.is_numeric or info.kind == Kind.BOOLEAN):
            return None
        series = self._column(column)
        if info.kind == Kind.BOOLEAN:
            series = series.cast(pl.Int8)
        lo, hi = series.min(), series.max()
        if lo is None or hi is None:
            return None
        return float(lo), float(hi)  # ty: ignore[invalid-argument-type]

    def value_counts(
        self, column: int, limit: int | None = None
    ) -> list[tuple[Any, int]]:
        series = self._column(column).drop_nulls()
        if series.is_empty():
            return []
        counts = series.value_counts(sort=True, name="__count__")
        if limit is not None:
            counts = counts.head(limit)
        return list(zip(counts.to_series(0).to_list(), counts.to_series(1).to_list()))

    def categories(self, column: int) -> list[Any] | None:
        series = self._column(column)
        dtype = series.dtype
        if isinstance(dtype, pl.Enum):
            return dtype.categories.to_list()
        if isinstance(dtype, pl.Categorical):
            # polars 2 categoricals have no fixed category list: the values present
            return series.drop_nulls().unique(maintain_order=True).to_list()
        return None

    def describe(self, column: int, top: int = 10) -> ColumnStats:
        info = self.columns[column]
        series = self._column(column)
        lo = hi = mean = std = None
        if info.kind.is_numeric or info.kind.is_temporal or info.kind == Kind.STRING:
            lo, hi = series.min(), series.max()
        if info.kind.is_numeric:
            mean = _number(series.mean())
            std = _number(series.std())
        try:
            unique = series.drop_nulls().n_unique()
        except pl.exceptions.InvalidOperationError:
            unique = None
        return ColumnStats(
            count=series.len(),
            null_count=series.null_count(),
            unique=unique,
            min=lo,
            max=hi,
            mean=mean,
            std=std,
            top=self.value_counts(column, limit=top) if unique is not None else [],
        )

    def argsort(self, column: int, descending: bool = False) -> np.ndarray:
        positions = self._df.select(
            pl.int_range(pl.len(), dtype=pl.Int64).sort_by(
                pl.nth(column),
                descending=descending,
                nulls_last=True,
                maintain_order=True,
            )
        ).to_series()
        return np.asarray(positions.to_numpy(), dtype=np.intp)

    def sort(self, column: int, descending: bool = False) -> pl.DataFrame:
        return self._df.sort(
            self._df.columns[column],
            descending=descending,
            nulls_last=True,
            maintain_order=True,
        )

    def set_value(self, row: int, column: int, value: Any) -> None:
        self._df[row, self._df.columns[column]] = value
