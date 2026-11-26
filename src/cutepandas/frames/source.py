"""The `FrameSource` protocol: what the Qt models need from a dataframe.

Everything in `cutepandas.frames` is Qt-free. A source is a thin adapter over one
backend frame (pandas, pyarrow, polars). Models only talk to the source, so they never
convert more than the requested window into Python objects.

Thread safety: sources are read-only views. All read methods (`fetch`, `fetch_arrow`,
`null_count`, `describe`, ...) may run on any thread, e.g. a worker that prefetches
chunks. The optional write hook `set_value` changes the frame in place and must run on
the GUI thread (the model invalidates its caches afterwards).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import enum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable


if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    import pyarrow as pa


class Kind(enum.StrEnum):
    """Backend-independent family of a column dtype (drives formatting and icons)."""

    INTEGER = "integer"
    FLOAT = "float"
    DECIMAL = "decimal"
    BOOLEAN = "boolean"
    STRING = "string"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    DATE = "date"
    TIME = "time"
    DURATION = "duration"
    BINARY = "binary"
    NESTED = "nested"
    NULL = "null"
    OBJECT = "object"

    @property
    def is_numeric(self) -> bool:
        return self in {Kind.INTEGER, Kind.FLOAT, Kind.DECIMAL}

    @property
    def is_temporal(self) -> bool:
        return self in {Kind.DATETIME, Kind.DATE, Kind.TIME, Kind.DURATION}


def format_name(name: Any, separator: str = " | ") -> str:
    """Column label as text; tuples (pandas MultiIndex columns) are joined."""
    if isinstance(name, tuple | list):
        return separator.join(str(i) for i in name)
    return str(name)


@dataclass(frozen=True, slots=True)
class ColumnInfo:
    """Name and dtype of one column."""

    name: Any
    """Column label (a tuple for pandas MultiIndex columns)."""
    dtype: str
    """The backend dtype as text (`int64`, `string[pyarrow]`, `Datetime(...)`)."""
    kind: Kind
    native_dtype: Any = field(default=None, compare=False, repr=False)
    """The backend dtype object."""

    @property
    def label(self) -> str:
        return format_name(self.name)


@dataclass(frozen=True, slots=True)
class ColumnChunk:
    """A window of one column as Python values."""

    values: list[Any]
    """One Python value per row; missing values are None."""
    nulls: np.ndarray
    """Boolean mask, True where the value is missing."""


@dataclass(frozen=True, slots=True)
class ColumnStats:
    """Basic statistics of one column (see `FrameSource.describe`)."""

    count: int
    null_count: int
    unique: int | None = None
    min: Any = None
    max: Any = None
    mean: float | None = None
    std: float | None = None
    top: list[tuple[Any, int]] = field(default_factory=list)
    """Most frequent non-null values with their counts, most frequent first."""


@runtime_checkable
class FrameSource(Protocol):
    """Read access to a rectangular frame, independent of the backend."""

    backend: str
    """Short backend name: `pandas`, `pyarrow` or `polars`."""

    @property
    def native(self) -> Any:
        """The wrapped backend frame."""
        ...

    @property
    def row_count(self) -> int: ...

    @property
    def columns(self) -> Sequence[ColumnInfo]: ...

    def fetch(self, start: int, stop: int, columns: Sequence[int]) -> list[ColumnChunk]:
        """Rows `start:stop` of the given column positions as Python values."""
        ...

    def fetch_arrow(self, start: int, stop: int, columns: Sequence[int]) -> pa.Table:
        """Rows `start:stop` of the given column positions as an Arrow table."""
        ...

    def null_count(self, column: int) -> int: ...

    def nbytes(self, column: int) -> int:
        """Memory used by the column's buffers (backend estimate)."""
        ...

    def value_range(self, column: int) -> tuple[float, float] | None:
        """(min, max) of a numeric or boolean column, None for other kinds / no data."""
        ...

    def value_counts(
        self, column: int, limit: int | None = None
    ) -> list[tuple[Any, int]]:
        """Non-null values with their counts, most frequent first."""
        ...

    def categories(self, column: int) -> list[Any] | None:
        """Categories of a categorical / enum / dictionary column, else None."""
        ...

    def describe(self, column: int, top: int = 10) -> ColumnStats:
        """Basic statistics of one column.

        Scans the whole column: call it off the GUI thread for large frames.
        """
        ...


@runtime_checkable
class SortableSource(Protocol):
    """Optional hook: a source that can return a sorted copy of its frame."""

    def sort(self, column: int, descending: bool = False) -> Any:
        """Return a new native frame, sorted stably by one column, nulls last."""
        ...


@runtime_checkable
class PermutableSource(Protocol):
    """Optional hook: sort order as row positions, and reads in any row order.

    Used for a display-only sort: the model keeps the permutation (one integer per
    row) and reads each visible window through it, so the frame is never copied.
    """

    def argsort(self, column: int, descending: bool = False) -> np.ndarray:
        """Row positions that sort the frame stably by one column, nulls last."""
        ...

    def fetch_rows(self, rows: np.ndarray, columns: Sequence[int]) -> list[ColumnChunk]:
        """The given row positions (in that order) of the given columns."""
        ...


@runtime_checkable
class EditableSource(Protocol):
    """Optional hook: a source whose frame can be changed cell by cell (GUI thread)."""

    def set_value(self, row: int, column: int, value: Any) -> None:
        """Change one cell in place; raises on invalid values."""
        ...
