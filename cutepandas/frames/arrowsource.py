"""`FrameSource` adapter for pyarrow tables."""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as _pc

from cutepandas.frames.source import ColumnChunk, ColumnInfo, ColumnStats, Kind


if TYPE_CHECKING:
    from collections.abc import Sequence

# The compute functions are generated at import time; type checkers can't see them.
pc: Any = _pc


def kind_for_arrow_type(typ: pa.DataType) -> Kind:
    """Map an Arrow type to a `Kind`."""
    t = pa.types
    if t.is_dictionary(typ):
        return Kind.CATEGORICAL
    if t.is_boolean(typ):
        return Kind.BOOLEAN
    if t.is_integer(typ):
        return Kind.INTEGER
    if t.is_floating(typ):
        return Kind.FLOAT
    if t.is_decimal(typ):
        return Kind.DECIMAL
    if t.is_string(typ) or t.is_large_string(typ) or t.is_string_view(typ):
        return Kind.STRING
    if t.is_timestamp(typ):
        return Kind.DATETIME
    if t.is_date(typ):
        return Kind.DATE
    if t.is_time(typ):
        return Kind.TIME
    if t.is_duration(typ):
        return Kind.DURATION
    if t.is_binary(typ) or t.is_large_binary(typ) or t.is_fixed_size_binary(typ):
        return Kind.BINARY
    if t.is_null(typ):
        return Kind.NULL
    if t.is_nested(typ):
        return Kind.NESTED
    return Kind.OBJECT


def _scalar(value: pa.Scalar) -> Any:
    return value.as_py() if value.is_valid else None


class ArrowSource:
    """Read-only source over a `pyarrow.Table` (or `RecordBatch`)."""

    backend = "pyarrow"

    def __init__(self, table: pa.Table | pa.RecordBatch):
        if isinstance(table, pa.RecordBatch):
            table = pa.Table.from_batches([table])
        self._table = table

    @property
    def native(self) -> pa.Table:
        return self._table

    @property
    def row_count(self) -> int:
        return self._table.num_rows

    @cached_property
    def columns(self) -> Sequence[ColumnInfo]:
        return [
            ColumnInfo(f.name, str(f.type), kind_for_arrow_type(f.type), f.type)
            for f in self._table.schema
        ]

    def fetch(self, start: int, stop: int, columns: Sequence[int]) -> list[ColumnChunk]:
        length = max(0, stop - start)
        chunks = []
        for j in columns:
            col = self._table.column(j).slice(start, length)
            nulls = np.asarray(col.is_null().to_numpy(zero_copy_only=False), dtype=bool)
            chunks.append(ColumnChunk(col.to_pylist(), nulls))
        return chunks

    def fetch_arrow(self, start: int, stop: int, columns: Sequence[int]) -> pa.Table:
        return self._table.select(list(columns)).slice(start, max(0, stop - start))

    def null_count(self, column: int) -> int:
        return self._table.column(column).null_count

    def nbytes(self, column: int) -> int:
        return self._table.column(column).nbytes

    def value_range(self, column: int) -> tuple[float, float] | None:
        info = self.columns[column]
        if not (info.kind.is_numeric or info.kind == Kind.BOOLEAN):
            return None
        col = self._table.column(column)
        if info.kind == Kind.BOOLEAN:
            col = col.cast(pa.int8())
        result = pc.min_max(col)
        lo, hi = _scalar(result["min"]), _scalar(result["max"])
        if lo is None or hi is None:
            return None
        return float(lo), float(hi)

    def value_counts(
        self, column: int, limit: int | None = None
    ) -> list[tuple[Any, int]]:
        col = self._table.column(column)
        if col.null_count:
            col = col.drop_null()
        if not len(col):
            return []
        counts = pc.value_counts(col)
        values = counts.field("values").to_pylist()
        numbers = counts.field("counts").to_pylist()
        pairs = sorted(zip(values, numbers), key=lambda p: -p[1])
        return pairs if limit is None else pairs[:limit]

    def categories(self, column: int) -> list[Any] | None:
        col = self._table.column(column)
        if not pa.types.is_dictionary(col.type):
            return None
        if col.num_chunks == 0:
            return []
        return col.unify_dictionaries().chunk(0).dictionary.to_pylist()

    def describe(self, column: int, top: int = 10) -> ColumnStats:
        info = self.columns[column]
        col = self._table.column(column)
        lo = hi = mean = std = None
        if info.kind.is_numeric or info.kind.is_temporal or info.kind == Kind.STRING:
            result = pc.min_max(col)
            lo, hi = _scalar(result["min"]), _scalar(result["max"])
        if info.kind.is_numeric:
            mean = _scalar(pc.mean(col))
            std = _scalar(pc.stddev(col, ddof=1))
        if pa.types.is_dictionary(col.type):
            col = col.cast(col.type.value_type)
        try:
            unique = pc.count_distinct(col, mode="only_valid").as_py()
        except pa.ArrowNotImplementedError:  # e.g. nested types
            unique = None
        return ColumnStats(
            count=len(col),
            null_count=col.null_count,
            unique=unique,
            min=lo,
            max=hi,
            mean=None if mean is None else float(mean),
            std=None if std is None else float(std),
            top=self.value_counts(column, limit=top) if unique is not None else [],
        )

    def sort(self, column: int, descending: bool = False) -> pa.Table:
        order = "descending" if descending else "ascending"
        indices = pc.array_sort_indices(
            self._table.column(column), order=order, null_placement="at_end"
        )
        return self._table.take(indices)
