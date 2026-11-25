"""Backend-agnostic, Qt-free access to dataframes.

`FrameSource` is the protocol the Qt models of cutepandas are built on. `as_source`
wraps a pandas DataFrame / Series, a pyarrow Table / RecordBatch or a polars DataFrame
(polars is only imported when a polars frame is passed). Objects that already follow
the protocol (e.g. a lazy source of an application) are returned unchanged.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pyarrow as pa

from cutepandas.frames.arrowsource import ArrowSource, kind_for_arrow_type
from cutepandas.frames.chunkcache import ChunkCache
from cutepandas.frames.formatters import FormatOptions, format_chunk, formatter_for
from cutepandas.frames.pandassource import PandasSource, kind_for_pandas_dtype
from cutepandas.frames.source import (
    ColumnChunk,
    ColumnInfo,
    ColumnStats,
    EditableSource,
    FrameSource,
    Kind,
    SortableSource,
    format_name,
)


def is_polars_frame(obj: Any) -> bool:
    """True for a polars DataFrame, without importing polars."""
    cls = type(obj)
    return cls.__module__.startswith("polars") and cls.__name__ == "DataFrame"


def as_source(obj: Any) -> FrameSource:
    """Wrap a supported frame in its `FrameSource` adapter."""
    if isinstance(obj, pd.DataFrame | pd.Series):
        return PandasSource(obj)
    if isinstance(obj, pa.Table | pa.RecordBatch):
        return ArrowSource(obj)
    if is_polars_frame(obj):
        from cutepandas.frames.polarssource import PolarsSource  # noqa: PLC0415

        return PolarsSource(obj)
    if isinstance(obj, FrameSource):
        return obj
    msg = f"Unsupported frame type: {type(obj).__qualname__}"
    raise TypeError(msg)


__all__ = [
    "ArrowSource",
    "ChunkCache",
    "ColumnChunk",
    "ColumnInfo",
    "ColumnStats",
    "EditableSource",
    "FormatOptions",
    "FrameSource",
    "Kind",
    "PandasSource",
    "SortableSource",
    "as_source",
    "format_chunk",
    "format_name",
    "formatter_for",
    "is_polars_frame",
    "kind_for_arrow_type",
    "kind_for_pandas_dtype",
]
