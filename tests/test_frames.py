"""Tests for `cutepandas.frames` (Qt-free sources, formatters, chunk cache)."""

from __future__ import annotations

import datetime as dt
import threading

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from cutepandas import frames
from cutepandas.frames import ChunkCache, FormatOptions, Kind


pl = pytest.importorskip("polars")

BACKENDS = ["pandas", "pandas-arrow", "pyarrow", "polars"]


def base_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "i": pd.array([3, 1, None, 2], dtype="Int64"),
        "f": [1.5, np.nan, 2.0, -1.0],
        "s": ["x", None, "z", "x"],
        "c": pd.Categorical(["a", "b", "a", None]),
        "d": pd.to_datetime(["2020-01-01", None, "2021-01-01", "2020-06-01"]),
        "b": [True, False, True, True],
    })


def make(backend: str):
    df = base_frame()
    match backend:
        case "pandas":
            return df
        case "pandas-arrow":
            return df.convert_dtypes(dtype_backend="pyarrow")
        case "pyarrow":
            return pa.Table.from_pandas(df, preserve_index=False)
        case "polars":
            return pl.from_pandas(df)
    raise ValueError(backend)


@pytest.fixture(params=BACKENDS)
def source(request):
    return frames.as_source(make(request.param))


def test_as_source_types():
    assert isinstance(frames.as_source(base_frame()), frames.PandasSource)
    assert isinstance(frames.as_source(base_frame()["f"]), frames.PandasSource)
    table = pa.Table.from_pandas(base_frame())
    assert isinstance(frames.as_source(table), frames.ArrowSource)
    assert isinstance(frames.as_source(table.to_batches()[0]), frames.ArrowSource)
    assert frames.as_source(pl.from_pandas(base_frame())).backend == "polars"
    source = frames.as_source(base_frame())
    assert frames.as_source(source) is source
    with pytest.raises(TypeError):
        frames.as_source([1, 2])


def test_protocols(source):
    assert isinstance(source, frames.FrameSource)
    assert isinstance(source, frames.SortableSource)
    assert isinstance(source, frames.EditableSource) == (source.backend != "pyarrow")


def test_schema(source):
    assert source.row_count == 4
    assert [c.name for c in source.columns] == ["i", "f", "s", "c", "d", "b"]
    kinds = [c.kind for c in source.columns]
    assert kinds == [
        Kind.INTEGER,
        Kind.FLOAT,
        Kind.STRING,
        Kind.CATEGORICAL,
        Kind.DATETIME,
        Kind.BOOLEAN,
    ]


def test_fetch_window(source):
    i, f, s, c, d, b = source.fetch(1, 3, range(6))
    assert i.values == [1, None]
    assert i.nulls.tolist() == [False, True]
    assert f.values[0] is None
    assert f.values[1] == 2.0
    assert s.values == [None, "z"]
    assert c.values == ["b", "a"]
    assert d.values[0] is None
    assert pd.Timestamp(d.values[1]) == pd.Timestamp("2021-01-01")
    assert b.values == [False, True]


def test_fetch_clips_and_empty(source):
    [chunk] = source.fetch(3, 10, [0])
    assert chunk.values == [2]
    [chunk] = source.fetch(4, 4, [0])
    assert chunk.values == []


def test_fetch_arrow(source):
    table = source.fetch_arrow(0, 2, [1, 2])
    assert table.num_rows == 2
    assert table.column(1).to_pylist() == ["x", None]


def test_counts_and_ranges(source):
    assert source.null_count(0) == 1
    assert source.null_count(5) == 0
    assert source.nbytes(1) > 0
    assert source.value_range(1) == (-1.0, 2.0)
    assert source.value_range(0) == (1.0, 3.0)
    assert source.value_range(5) == (0.0, 1.0)
    assert source.value_range(2) is None
    assert source.value_counts(2) == [("x", 2), ("z", 1)]
    assert source.value_counts(2, limit=1) == [("x", 2)]


def test_categories(source):
    assert sorted(source.categories(3)) == ["a", "b"]
    assert source.categories(2) is None


def test_describe(source):
    stats = source.describe(1, top=2)
    assert stats.count == 4
    assert stats.null_count == 1
    assert stats.unique == 3
    assert stats.min == -1.0
    assert stats.max == 2.0
    assert stats.mean == pytest.approx(2.5 / 3)
    assert stats.std is not None
    assert len(stats.top) == 2
    text = source.describe(2)
    assert (text.min, text.max, text.mean) == ("x", "z", None)


@pytest.mark.parametrize("descending", [False, True])
def test_sort_nulls_last(source, descending):
    assert isinstance(source, frames.SortableSource)
    result = frames.as_source(source.sort(1, descending=descending))
    [chunk] = result.fetch(0, 4, [1])
    expected = [2.0, 1.5, -1.0, None] if descending else [-1.0, 1.5, 2.0, None]
    assert chunk.values == expected


def test_set_value(source):
    if not isinstance(source, frames.EditableSource):
        pytest.skip("read-only backend")
    source.set_value(0, 1, 9.5)
    assert source.fetch(0, 1, [1])[0].values == [9.5]


def test_polars_enum_and_nested():
    df = pl.DataFrame({
        "e": pl.Series(["u", "v", "u"], dtype=pl.Enum(["v", "u", "w"])),
        "l": [[1], [2, 3], []],
        "t": [dt.time(1), dt.time(2), None],
    })
    source = frames.as_source(df)
    assert [c.kind for c in source.columns] == [Kind.CATEGORICAL, Kind.NESTED, Kind.TIME]
    assert source.categories(0) == ["v", "u", "w"]


def test_kind_for_names():
    assert frames.kind_for_pandas_dtype("string[pyarrow]") == Kind.STRING
    assert frames.kind_for_pandas_dtype("int64[pyarrow]") == Kind.INTEGER
    assert frames.kind_for_pandas_dtype("category") == Kind.CATEGORICAL
    assert frames.kind_for_arrow_type(pa.list_(pa.int8())) == Kind.NESTED


# --- formatters ---


def test_formatters():
    options = FormatOptions(float_precision=3, max_text_length=5)
    fmt = frames.formatter_for(Kind.FLOAT, options)
    assert [fmt(v) for v in (3.0, 1.23456, 1e20, float("nan"))] == [
        "3.0",
        "1.23",
        "1e+20",
        "NaN",
    ]
    assert frames.formatter_for(Kind.STRING, options)("abcdefgh") == "abcd…"
    grouped = FormatOptions(thousands_separator=True)
    assert frames.formatter_for(Kind.INTEGER, grouped)(1234567) == "1,234,567"
    nulls = np.array([False, True])
    out = frames.format_chunk([1, None], nulls, str, "<NA>")
    assert out == ["1", "<NA>"]


# --- chunk cache ---


def test_chunk_cache_lru():
    cache: ChunkCache[str] = ChunkCache(chunk_rows=10, max_chunks=2)
    assert cache.block_of(25) == 2
    assert cache.block_bounds(2, 23) == (20, 23)
    cache.put(0, 0, "a")
    cache.put(1, 0, "b")
    assert cache.get(0, 0) == "a"  # touch: (1, 0) is now the oldest
    cache.put(2, 0, "c")
    assert cache.get(1, 0) is None
    assert len(cache) == 2


def test_chunk_cache_invalidate_and_generation():
    cache: ChunkCache[str] = ChunkCache()
    cache.put(0, 0, "a")
    cache.put(0, 1, "b")
    generation = cache.generation
    cache.invalidate(column=1)
    assert cache.get(0, 0) == "a"
    assert cache.get(0, 1) is None
    # a block built before the invalidation is not stored
    assert not cache.put(0, 1, "stale", generation)
    cache.clear()
    assert len(cache) == 0


def test_chunk_cache_threads():
    cache: ChunkCache[tuple[int, int]] = ChunkCache(chunk_rows=1, max_chunks=50)
    calls = []

    def build(block, column):
        calls.append((block, column))
        return block, column

    def work():
        for block in range(200):
            assert cache.get_or_build(block, 0, build) == (block, 0)

    threads = [threading.Thread(target=work) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(cache) == 50


def test_chunk_cache_rejects_bad_sizes():
    with pytest.raises(ValueError, match="positive"):
        ChunkCache(chunk_rows=0)
