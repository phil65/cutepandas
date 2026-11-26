"""Display-only sort (row permutation), `sortable` / `sort_requested` (all backends)."""

import tracemalloc

import numpy as np
import pandas as pd
from prettyqt import constants, gui
import pyarrow as pa
import pytest

from cutepandas import frames, pandasmodels, pandaswidgets
from cutepandas.frames.pandassource import numpy_argsort


pl = pytest.importorskip("polars")

BACKENDS = ["pandas", "pandas-arrow", "pyarrow", "polars"]


def base_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "i": pd.array([3, 1, None, 1, 3], dtype="Int64"),
        "f": [1.5, np.nan, 2.0, 1.5, -1.0],
        "s": ["x", None, "z", "x", "a"],
        "d": pd.to_datetime([
            "2020-01-01",
            None,
            "2021-01-01",
            "2020-01-01",
            "2019-01-01",
        ]),
        "n": [2, 1, 2, 1, 2],
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
def frame(request):
    return make(request.param)


def expected_order(column: str, descending: bool) -> list[int]:
    """Stable, nulls last, computed independently of the sources."""
    values = base_frame()[column].tolist()
    valid = [i for i, v in enumerate(values) if not pd.isna(v)]
    nulls = [i for i, v in enumerate(values) if pd.isna(v)]
    return sorted(valid, key=lambda i: values[i], reverse=descending) + nulls


def cell(model, row: int, col: int):
    return model.data(model.index(row, col))


# --- sources ---


@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("column", range(5))
def test_argsort_stable_nulls_last(frame, column, descending):
    source = frames.as_source(frame)
    assert isinstance(source, frames.PermutableSource)
    positions = source.argsort(column, descending=descending)
    assert positions.dtype == np.intp
    name = base_frame().columns[column]
    # sorted() with reverse=True keeps ties in their original order too
    assert positions.tolist() == expected_order(name, descending)
    # the copy sort uses the same order
    sorted_source = frames.as_source(source.sort(column, descending=descending))
    assert (
        sorted_source.fetch(0, 5, [column])[0].values
        == source.fetch_rows(positions, [column])[0].values
    )


def test_fetch_rows(frame):
    source = frames.as_source(frame)
    [chunk] = source.fetch_rows(np.array([4, 1, 0]), [1])
    assert chunk.values == [-1.0, None, 1.5]
    assert chunk.nulls.tolist() == [False, True, False]


def test_fetch_rows_multi_chunk_arrow():
    column = pa.chunked_array([["a", None], ["c"], ["d", "e"]])
    table = pa.table({"s": column})
    [chunk] = frames.as_source(table).fetch_rows(np.array([4, 0, 2, 1, 3]), [0])
    assert chunk.values == ["e", "a", "c", None, "d"]
    assert chunk.nulls.tolist() == [False, False, False, True, False]
    series = pd.Series(pd.arrays.ArrowExtensionArray(column))
    [chunk] = frames.as_source(series).fetch_rows(np.array([3, 1]), [0])
    assert chunk.values == ["d", None]


@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("dtype", ["int64", "float64", "datetime64[ns]", "bool"])
def test_numpy_argsort_matches_pandas(dtype, descending):
    rng = np.random.default_rng(0)
    values = rng.integers(0, 20, 500)
    match dtype:
        case "float64":
            values = values.astype(float)
            values[rng.integers(0, 500, 50)] = np.nan
        case "datetime64[ns]":
            values = values.astype("datetime64[ns]")
            values[rng.integers(0, 500, 50)] = np.datetime64("NaT", "ns")
        case "bool":
            values = values > 10
    series = pd.Series(values)
    expected = series.sort_values(
        ascending=not descending, kind="stable", na_position="last"
    ).index
    assert numpy_argsort(values, descending).tolist() == expected.tolist()


# --- model ---


@pytest.mark.parametrize("descending", [False, True])
def test_permutation_sort_shows_sorted_rows_without_copy(frame, descending):
    model = pandasmodels.DataFrameModel(frame, sort_mode="permutation")
    original = model.df
    order = constants.DESCENDING if descending else constants.ASCENDING
    model.sort(1, order)
    assert model.df is original  # no copy, frame untouched
    expected = expected_order("f", descending)
    assert model.row_order is not None
    assert model.row_order.tolist() == expected
    assert [model.source_row(r) for r in range(5)] == expected
    values = [model.data(model.index(r, 1), constants.USER_ROLE) for r in range(5)]
    f = base_frame()["f"].tolist()
    assert values[:4] == [f[i] for i in expected[:4]]
    assert values[4] is None  # NaN shown as null, last
    # vertical header shows the original row position
    assert model.headerData(0, constants.VERTICAL) == str(expected[0])
    chunks = model.fetch_window(0, 2, [1])
    assert chunks[0].values == values[:2]
    model.clear_sort()
    assert model.row_order is None
    assert cell(model, 0, 1) == cell(pandasmodels.DataFrameModel(frame), 0, 1)


def test_permutation_shared_holder_and_reset(frame):
    holder = pandasmodels.DataFrameHolder(frame)
    model = pandasmodels.DataFrameModel(holder, sort_mode="permutation")
    other = pandasmodels.DataFrameModel(holder)
    header = pandasmodels.VerticalHeaderModel(holder)
    model.sort(4, constants.ASCENDING)  # n: 1, 1, 2, 2, 2
    assert holder.row_order.tolist() == [1, 3, 0, 2, 4]
    assert cell(other, 0, 4) == "1"  # same holder: same display order
    assert header.data(header.index(0, 0)) == "1"
    holder.df = make("pandas")  # a new frame drops the order
    assert holder.row_order is None
    assert model.row_order is None
    assert cell(model, 0, 4) == "2"


def test_permutation_edit_maps_rows():
    model = pandasmodels.DataFrameModel(base_frame(), sort_mode="permutation")
    model.sort(1, constants.ASCENDING)  # f: -1.0 (row 4) first
    assert model.setData(model.index(0, 1), 9.0)
    assert model.df["f"].iloc[4] == 9.0
    assert cell(model, 0, 1) == "9.0"


def test_stale_permutation_is_dropped():
    model = pandasmodels.DataFrameModel(base_frame(), sort_mode="permutation")
    permutation = model.compute_sort_permutation(1, constants.ASCENDING)
    assert permutation is not None
    model.setData(model.index(0, 1), 100.0)  # edited after computing
    assert not model.apply_sort_permutation(permutation)
    assert model.row_order is None
    permutation = model.compute_sort_permutation(1, constants.ASCENDING)
    model.df = base_frame()  # replaced after computing
    assert not model.apply_sort_permutation(permutation)
    permutation = model.compute_sort_permutation(1, constants.ASCENDING)
    assert model.apply_sort_permutation(permutation)
    assert model.row_order is not None


def test_permutation_layout_signals(qtbot):
    model = pandasmodels.DataFrameModel(base_frame(), sort_mode="permutation")
    with qtbot.waitSignals([model.layoutAboutToBeChanged, model.layoutChanged]):
        model.sort(0, constants.ASCENDING)


def test_holder_rejects_bad_row_order():
    holder = pandasmodels.DataFrameHolder(base_frame())
    with pytest.raises(ValueError, match="positions"):
        holder.set_row_order([0, 1])
    with pytest.raises(ValueError, match="outside"):
        holder.set_row_order([0, 1, 2, 3, 5])


def test_permutation_memory():
    """The display-only sort allocates the permutation, not a copy of the frame."""
    rows = 200_000
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "a": rng.random(rows),
        "b": rng.integers(0, 100, rows),
        "c": rng.random(rows),
        "d": rng.random(rows),
    })
    frame_bytes = int(df.memory_usage(index=False).sum())
    model = pandasmodels.DataFrameModel(df, sort_mode="permutation")
    tracemalloc.start()
    try:
        model.sort(0, constants.DESCENDING)
        retained, _peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert model.df is df
    assert model.row_order is not None
    assert model.row_order.nbytes == 8 * rows
    # what stays allocated: the permutation (8 bytes / row), far below the frame
    assert retained < 8 * rows + frame_bytes // 4
    copy_model = pandasmodels.DataFrameModel(df)
    tracemalloc.start()
    try:
        copy_model.sort(0, constants.DESCENDING)
        retained_copy, _ = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert copy_model.df is not df
    assert retained_copy >= frame_bytes  # the copy mode holds a full sorted copy


def test_permutation_mode_unsupported_source():
    class Plain:
        """A FrameSource without the PermutableSource hook."""

        def __init__(self):
            self._inner = frames.as_source(base_frame())

        def __getattr__(self, name):
            if name in {"argsort", "fetch_rows", "sort", "set_value"}:
                raise AttributeError(name)
            return getattr(self._inner, name)

    model = pandasmodels.DataFrameModel(sort_mode="permutation")
    model.holder._source = Plain()
    assert not model.can_sort()
    model.sort(0, constants.ASCENDING)  # ignored
    assert model.row_order is None


def test_invalid_sort_mode():
    with pytest.raises(ValueError, match="sort mode"):
        pandasmodels.DataFrameModel().set_sort_mode("nope")  # ty: ignore[invalid-argument-type]


# --- viewer ---


def click_header(widget, column: int):
    widget.table_columns.clicked.emit(widget.table_columns.model().index(0, column))


def test_viewer_not_sortable_only_emits(qtbot):
    df = pd.DataFrame({"a": [3, 1, 2]})
    widget = pandaswidgets.DataFrameViewer(df, sortable=False)
    qtbot.addWidget(widget)
    with qtbot.waitSignal(widget.sort_requested) as blocker:
        click_header(widget, 0)
    assert blocker.args == [0, constants.ASCENDING]
    assert widget.df is df
    assert widget.get_sort_state() is None
    # the application sorts and restores the state: next click reverses
    widget.set_df(df.sort_values("a"))
    widget.set_sort_state(0, constants.ASCENDING)
    with qtbot.waitSignal(widget.sort_requested) as blocker:
        click_header(widget, 0)
    assert blocker.args == [0, constants.DESCENDING]


def test_viewer_copy_mode_emits_and_sorts(qtbot):
    widget = pandaswidgets.DataFrameViewer(pd.DataFrame({"a": [3, 1, 2]}))
    qtbot.addWidget(widget)
    with qtbot.waitSignal(widget.sort_requested):
        click_header(widget, 0)
    assert widget.df["a"].tolist() == [1, 2, 3]
    assert widget.get_sort_state() == (0, constants.ASCENDING)


@pytest.mark.parametrize("backend", BACKENDS)
def test_viewer_permutation_mode_async(qtbot, backend):
    frame = make(backend)
    widget = pandaswidgets.DataFrameViewer(frame, sort_mode="permutation")
    qtbot.addWidget(widget)
    assert widget.get_sort_mode() == "permutation"
    with qtbot.waitSignals([widget.sort_requested, widget.sorting_changed]):
        click_header(widget, 1)
    qtbot.waitUntil(lambda: not widget.is_sorting())
    assert widget.df is frame
    assert widget.holder.row_order.tolist() == expected_order("f", False)
    assert widget.get_sort_state() == (1, constants.ASCENDING)
    index_model = widget.table_index.model()
    assert index_model.data(index_model.index(0, 0)) == "4"
    click_header(widget, 1)  # reverses
    qtbot.waitUntil(lambda: not widget.is_sorting())
    assert widget.holder.row_order.tolist() == expected_order("f", True)
    widget.clear_sort()
    assert widget.holder.row_order is None
    assert widget.get_sort_state() is None


def test_viewer_permutation_latest_wins_and_cancel(qtbot):
    widget = pandaswidgets.DataFrameViewer(base_frame(), sort_mode="permutation")
    qtbot.addWidget(widget)
    widget.sort_by(1, constants.ASCENDING)
    widget.sort_by(4, constants.DESCENDING)  # supersedes the first
    qtbot.waitUntil(lambda: not widget.is_sorting())
    qtbot.wait(50)  # the first result may arrive later: must be dropped
    assert widget.holder.row_order.tolist() == expected_order("n", True)
    widget.clear_sort()
    widget.sort_by(1, constants.ASCENDING)
    widget.cancel_sort()
    assert not widget.is_sorting()
    qtbot.wait(100)
    assert widget.holder.row_order is None


def test_viewer_permutation_copy_to_clipboard(qtbot):
    widget = pandaswidgets.DataFrameViewer(
        pd.DataFrame({"a": [3, 1, 2], "b": ["c", "a", "b"]}), sort_mode="permutation"
    )
    qtbot.addWidget(widget)
    widget.sort_by(0, constants.ASCENDING)
    qtbot.waitUntil(lambda: widget.holder.row_order is not None)
    widget.table_data.selectAll()
    widget.table_data.copy()
    text = gui.GuiApplication.clipboard().text()
    assert text.splitlines() == ["1\ta", "2\tb", "3\tc"]
