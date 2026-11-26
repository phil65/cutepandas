"""The three DataFrame viewers: sorting, MultiIndex headers, backends."""

import numpy as np
import pandas as pd
from prettyqt import constants, gui
import pyarrow as pa
import pytest

from cutepandas import pandasmodels
from cutepandas.frames import Kind
from cutepandas.pandaswidgets import (
    DataFrameViewer,
    GridDataFrameViewer,
    SimpleDataFrameViewer,
    dataframeviewer,
)
from cutepandas.pandaswidgets._viewer_style import ViewerColors
from cutepandas.pandaswidgets.sortcontroller import SortController
from cutepandas.utils import icons


pl = pytest.importorskip("polars")

VIEWERS = [DataFrameViewer, GridDataFrameViewer, SimpleDataFrameViewer]
VIEWER_IDS = ["composite", "grid", "simple"]
BACKENDS = ["pandas", "pandas-arrow", "pyarrow", "polars"]


def make(backend: str, data: dict):
    df = pd.DataFrame(data)
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


# --- per-viewer accessors ---


def data_view(widget):
    return widget if isinstance(widget, SimpleDataFrameViewer) else widget.table_data


def column_header(widget):
    """The column header table (None for the single-table viewer)."""
    match widget:
        case GridDataFrameViewer():
            return widget.table_header
        case DataFrameViewer():
            return widget.table_columns
    return None


def click_header(widget, column: int, row: int | None = None):
    """Click the header of `column` (`row`: header level, default the leaf)."""
    if isinstance(widget, SimpleDataFrameViewer):
        widget.h_header.sectionClicked.emit(column)
        return
    header = column_header(widget)
    model = header.model()
    row = model.rowCount() - 1 if row is None else row
    header.clicked.emit(model.index(row, column))


def cells(widget, column: int) -> list[str]:
    model = data_view(widget).model()
    return [model.data(model.index(r, column)) for r in range(model.rowCount())]


def row_label(widget, row: int, level: int = 0) -> str:
    if isinstance(widget, SimpleDataFrameViewer):
        return widget.model().headerData(row, constants.VERTICAL)
    model = widget.table_index.model()
    return model.data(model.index(row, level))


@pytest.fixture(params=VIEWERS, ids=VIEWER_IDS)
def viewer_cls(request):
    return request.param


# --- SortController ---


def test_sort_controller_copy_mode(qtbot):
    model = pandasmodels.DataFrameModel(pd.DataFrame({"a": [3, 1, 2]}))
    sorter = SortController(model)
    states = []
    sorter.sort_state_changed.connect(states.append)
    with qtbot.waitSignal(sorter.sort_requested) as blocker:
        sorter.request_sort(0)
    assert blocker.args == [0, constants.ASCENDING]
    assert model.df["a"].tolist() == [1, 2, 3]
    assert sorter.get_sort_state() == (0, constants.ASCENDING)
    sorter.request_sort(0)  # reverses
    assert model.df["a"].tolist() == [3, 2, 1]
    assert sorter.get_sort_state() == (0, constants.DESCENDING)
    model.holder.df = pd.DataFrame({"a": [1]})  # a new frame resets the state
    assert sorter.get_sort_state() is None
    assert states[-1] is None


def test_sort_controller_not_sortable(qtbot):
    df = pd.DataFrame({"a": [3, 1, 2]})
    model = pandasmodels.DataFrameModel(df)
    sorter = SortController(model, sortable=False)
    assert not sorter.is_sortable()
    with qtbot.waitSignal(sorter.sort_requested):
        sorter.request_sort(0)
    assert model.df is df
    sorter.set_sortable(True)
    sorter.request_sort(0)
    assert model.df["a"].tolist() == [1, 2, 3]


def test_sort_controller_empty_model(qtbot):
    sorter = SortController(pandasmodels.DataFrameModel())
    with qtbot.assertNotEmitted(sorter.sort_requested):
        sorter.request_sort(0)


def test_sort_controller_permutation_stale_result_dropped(qtbot):
    model = pandasmodels.DataFrameModel(
        pd.DataFrame({"a": [3, 1, 2]}), sort_mode="permutation"
    )
    sorter = SortController(model)
    with qtbot.waitSignal(sorter.sorting_changed) as blocker:
        sorter.sort_by(0)
    assert blocker.args == [True]
    model.holder.df = pd.DataFrame({"a": [5, 4]})  # replaced while sorting
    qtbot.waitUntil(lambda: not sorter.is_sorting())
    assert model.holder.row_order is None
    assert sorter.get_sort_state() is None


def test_sort_controller_error_is_logged(qtbot, caplog, monkeypatch):
    model = pandasmodels.DataFrameModel(
        pd.DataFrame({"a": [3, 1, 2]}), sort_mode="permutation"
    )

    def fail(*args):
        raise RuntimeError("boom")

    monkeypatch.setattr(model, "compute_sort_permutation", fail)
    sorter = SortController(model)
    sorter.sort_by(0)
    qtbot.waitUntil(lambda: not sorter.is_sorting())
    assert "Sorting failed" in caplog.text
    assert sorter.get_sort_state() is None


# --- sorting through each viewer ---


def test_viewer_not_sortable_only_emits(qtbot, viewer_cls):
    df = pd.DataFrame({"a": [3, 1, 2]})
    widget = viewer_cls(df, sortable=False)
    qtbot.addWidget(widget)
    assert not widget.is_sortable()
    with qtbot.waitSignal(widget.sort_requested) as blocker:
        click_header(widget, 0)
    assert blocker.args == [0, constants.ASCENDING]
    assert widget.df is df
    assert widget.get_sort_state() is None
    widget.set_df(df.sort_values("a"))
    widget.set_sort_state(0, constants.ASCENDING)
    with qtbot.waitSignal(widget.sort_requested) as blocker:
        click_header(widget, 0)
    assert blocker.args == [0, constants.DESCENDING]


def test_viewer_copy_mode(qtbot, viewer_cls):
    widget = viewer_cls(pd.DataFrame({"a": [3, 1, 2]}))
    qtbot.addWidget(widget)
    assert widget.get_sort_mode() == "copy"
    click_header(widget, 0)
    assert widget.df["a"].tolist() == [1, 2, 3]
    assert widget.get_sort_state() == (0, constants.ASCENDING)
    click_header(widget, 0)
    assert widget.df["a"].tolist() == [3, 2, 1]
    assert widget.get_sort_state() == (0, constants.DESCENDING)


@pytest.mark.parametrize("backend", BACKENDS)
def test_viewer_permutation_mode(qtbot, viewer_cls, backend):
    frame = make(backend, {"a": [3, 1, 2], "b": ["x", "y", "z"]})
    widget = viewer_cls(frame, sort_mode="permutation")
    qtbot.addWidget(widget)
    with qtbot.waitSignals([widget.sort_requested, widget.sorting_changed]):
        click_header(widget, 0)
    qtbot.waitUntil(lambda: not widget.is_sorting())
    assert widget.df is frame
    assert widget.holder.row_order.tolist() == [1, 2, 0]
    assert cells(widget, 0) == ["1", "2", "3"]
    assert row_label(widget, 0) == "1"  # the labels move with the rows
    assert widget.get_sort_state() == (0, constants.ASCENDING)
    click_header(widget, 0)
    qtbot.waitUntil(lambda: not widget.is_sorting())
    assert widget.holder.row_order.tolist() == [0, 2, 1]
    widget.clear_sort()
    assert widget.holder.row_order is None
    assert widget.get_sort_state() is None


def test_viewer_permutation_latest_wins_and_cancel(qtbot, viewer_cls):
    widget = viewer_cls(
        pd.DataFrame({"a": [3, 1, 2], "b": [1, 3, 2]}), sort_mode="permutation"
    )
    qtbot.addWidget(widget)
    widget.sort_by(0, constants.ASCENDING)
    widget.sort_by(1, constants.DESCENDING)  # supersedes the first
    qtbot.waitUntil(lambda: not widget.is_sorting())
    qtbot.wait(50)  # the first result may arrive later: must be dropped
    assert widget.holder.row_order.tolist() == [1, 2, 0]
    assert widget.get_sort_state() == (1, constants.DESCENDING)
    widget.clear_sort()
    widget.sort_by(0, constants.ASCENDING)
    widget.cancel_sort()
    assert not widget.is_sorting()
    qtbot.wait(100)
    assert widget.holder.row_order is None


def test_viewer_set_sort_mode(qtbot, viewer_cls):
    widget = viewer_cls(pd.DataFrame({"a": [3, 1, 2]}))
    qtbot.addWidget(widget)
    widget.set_sort_mode("permutation")
    assert widget.get_sort_mode() == "permutation"
    widget.set_sortable(False)
    assert not widget.is_sortable()


def test_viewer_outer_header_level_does_not_sort(qtbot, viewer_cls):
    if viewer_cls is SimpleDataFrameViewer:
        pytest.skip("one header row (levels joined)")
    cols = pd.MultiIndex.from_tuples([("m", "x"), ("m", "y")])
    widget = viewer_cls(pd.DataFrame([[2, 1], [1, 2]], columns=cols))
    qtbot.addWidget(widget)
    with qtbot.assertNotEmitted(widget.sort_requested):
        click_header(widget, 0, row=0)
    assert widget.get_sort_state() is None


def test_simple_viewer_sort_indicator(qtbot):
    widget = SimpleDataFrameViewer(pd.DataFrame({"a": [3, 1, 2], "b": [1, 2, 3]}))
    qtbot.addWidget(widget)
    click_header(widget, 1)
    click_header(widget, 1)
    assert widget.h_header.isSortIndicatorShown()
    assert widget.h_header.sortIndicatorSection() == 1
    assert widget.h_header.sortIndicatorOrder() == constants.DESCENDING
    widget.set_df(pd.DataFrame({"a": [1]}))
    assert not widget.h_header.isSortIndicatorShown()


# --- MultiIndex ---

INDEX_CASES = {
    "flat": pd.Index(["r0", "r1", "r1", "r3"]),
    "2-named": pd.MultiIndex.from_tuples(
        [("a", 1), ("a", 1), ("b", 1), ("b", 2)], names=["grp", "sub"]
    ),
    "3-unnamed-nan": pd.MultiIndex.from_tuples([
        ("a", "x", 1),
        ("a", "x", 2),
        ("a", np.nan, 2),
        (np.nan, np.nan, 3),
    ]),
}
COLUMN_CASES = {
    "flat": pd.Index(["c0", "c1", "c2"]),
    "2-named": pd.MultiIndex.from_tuples(
        [("m", "x"), ("m", "y"), ("n", "x")], names=["metric", "var"]
    ),
    "3-mixed-names": pd.MultiIndex.from_tuples(
        [("m", "x", 1), ("m", "x", 2), ("m", np.nan, 2)], names=["l0", None, "l2"]
    ),
}


def multi_frame(index_case: str, column_case: str) -> pd.DataFrame:
    return pd.DataFrame(
        [[4, 1, 0], [3, 2, 0], [2, 3, 0], [1, 4, 0]],
        index=INDEX_CASES[index_case],
        columns=COLUMN_CASES[column_case],
    )


def spans(view, *, horizontal: bool) -> set[tuple[int, int, int]]:
    """(level, start, length) of the spans of a header table."""
    model = view.model()
    found = set()
    levels = model.rowCount() if horizontal else model.columnCount()
    count = model.columnCount() if horizontal else model.rowCount()
    for level in range(levels):
        pos = 0
        while pos < count:
            n = view.columnSpan(level, pos) if horizontal else view.rowSpan(pos, level)
            if n > 1:
                found.add((level, pos, n))
            pos += max(n, 1)
    return found


@pytest.mark.parametrize("index_case", list(INDEX_CASES))
@pytest.mark.parametrize("column_case", list(COLUMN_CASES))
def test_multiindex_headers(qtbot, viewer_cls, index_case, column_case):
    df = multi_frame(index_case, column_case)
    widget = viewer_cls(df)
    qtbot.addWidget(widget)
    widget.resize(600, 400)
    widget.show()
    qtbot.waitExposed(widget)
    assert data_view(widget).model().rowCount() == 4
    assert data_view(widget).model().columnCount() == 3
    if isinstance(widget, SimpleDataFrameViewer):
        model = widget.model()
        label = model.headerData(0, constants.HORIZONTAL)
        assert label == " | ".join(str(i) for i in np.atleast_1d(df.columns[0]))
        assert model.headerData(0, constants.VERTICAL) == " | ".join(
            str(i) for i in np.atleast_1d(df.index[0])
        )
        assert widget.v_header.width() > 0
        assert widget.h_header.height() > 0
        return
    header = column_header(widget)
    index = widget.table_index
    assert header.model().rowCount() == df.columns.nlevels
    assert index.model().columnCount() == df.index.nlevels
    for level in range(df.index.nlevels):
        values = df.index.get_level_values(level)
        assert [row_label(widget, r, level) for r in range(4)] == [str(v) for v in values]
    expected_rows = set(dataframeviewer.axis_spans(df, horizontal=False))
    expected_cols = set(dataframeviewer.axis_spans(df, horizontal=True))
    assert spans(index, horizontal=False) == expected_rows
    assert spans(header, horizontal=True) == expected_cols
    # header sizing: every level row / column is visible
    header_rows = sum(header.rowHeight(r) for r in range(df.columns.nlevels))
    assert header.height() >= header_rows > 0
    index_cols = sum(index.columnWidth(c) for c in range(df.index.nlevels))
    assert index.width() >= index_cols > 0
    if isinstance(widget, GridDataFrameViewer):
        corner = widget.table_level
        assert corner.model().rowCount() == df.columns.nlevels
        assert corner.model().columnCount() == df.index.nlevels
        assert corner.height() == header.height()
        assert corner.width() == index.width()
        for c in range(df.index.nlevels):
            assert corner.columnWidth(c) == index.columnWidth(c)


def test_axis_spans_values():
    df = multi_frame("3-unnamed-nan", "3-mixed-names")
    # level 0: a a a | nan; level 1: x x | nan | nan (breaks at the level-0 change)
    assert dataframeviewer.axis_spans(df, horizontal=False) == [(0, 0, 3), (1, 0, 2)]
    assert dataframeviewer.axis_spans(df, horizontal=True) == [
        (0, 0, 3),
        (1, 0, 2),
    ]
    flat_unique = pd.DataFrame({"a": [1, 2]})
    assert dataframeviewer.axis_spans(flat_unique, horizontal=False) == []
    assert dataframeviewer.axis_spans(pa.table({"a": [1, 1]}), horizontal=False) == []


def test_corner_level_names(qtbot):
    widget = GridDataFrameViewer(multi_frame("2-named", "2-named"))
    qtbot.addWidget(widget)
    corner = widget.table_level.model()
    texts = [
        [corner.data(corner.index(r, c)) for c in range(corner.columnCount())]
        for r in range(corner.rowCount())
    ]
    assert texts == [["", "metric"], ["grp", "sub / var"]]
    widget.set_df(multi_frame("3-unnamed-nan", "flat"))
    assert corner.rowCount() == 1
    assert corner.columnCount() == 3
    assert all(corner.data(corner.index(0, c)) == "" for c in range(3))


def test_composite_level_name_headers(qtbot):
    widget = DataFrameViewer(multi_frame("2-named", "2-named"))
    qtbot.addWidget(widget)
    widget.show()
    assert widget.table_columns.v_header.maximumWidth() > 0
    assert widget.table_index.h_header.maximumHeight() > 0
    widget.set_df(multi_frame("3-unnamed-nan", "flat"))
    assert widget.table_columns.v_header.maximumWidth() == 0
    assert widget.table_index.h_header.maximumHeight() == 0


@pytest.mark.parametrize("sort_mode", ["copy", "permutation"])
def test_multiindex_sort_by_leaf(qtbot, viewer_cls, sort_mode):
    df = multi_frame("2-named", "2-named")  # column (m, y) = 1 2 3 4
    widget = viewer_cls(df, sort_mode=sort_mode)
    qtbot.addWidget(widget)
    widget.sort_by(1, constants.DESCENDING)
    qtbot.waitUntil(lambda: not widget.is_sorting())
    assert cells(widget, 1) == ["4", "3", "2", "1"]
    assert widget.get_sort_state() == (1, constants.DESCENDING)
    if isinstance(widget, SimpleDataFrameViewer):
        assert row_label(widget, 0) == "b | 2"
        return
    assert [row_label(widget, r, 1) for r in range(4)] == ["2", "1", "1", "1"]
    index_spans = spans(widget.table_index, horizontal=False)
    if sort_mode == "permutation":
        assert index_spans == set()  # spans only in the frame's own order
        widget.clear_sort()
        assert spans(widget.table_index, horizontal=False) == {
            (0, 0, 2),
            (0, 2, 2),
            (1, 0, 2),
        }
    else:  # the sorted copy is a new frame with its own spans
        assert index_spans == set(dataframeviewer.axis_spans(widget.df, horizontal=False))


def test_max_span_count_fallback(qtbot, viewer_cls, monkeypatch):
    monkeypatch.setattr(dataframeviewer, "MAX_SPAN_COUNT", 2)
    index = pd.MultiIndex.from_product([["a", "b", "c"], ["x", "y"], [1, 2]])
    widget = viewer_cls(pd.DataFrame({"v": range(12)}, index=index))
    qtbot.addWidget(widget)
    assert dataframeviewer.axis_spans(widget.df, horizontal=False) == []
    if not isinstance(widget, SimpleDataFrameViewer):
        assert spans(widget.table_index, horizontal=False) == set()
        assert row_label(widget, 11, 2) == "2"


@pytest.mark.parametrize("backend", ["pyarrow", "polars"])
def test_non_pandas_backends_degrade(qtbot, viewer_cls, backend):
    widget = viewer_cls(make(backend, {"a": [1, 1, 2], "b": ["x", "x", "y"]}))
    qtbot.addWidget(widget)
    widget.show()
    assert data_view(widget).model().rowCount() == 3
    if isinstance(widget, SimpleDataFrameViewer):
        assert widget.model().headerData(2, constants.VERTICAL) == "2"
        return
    assert column_header(widget).model().rowCount() == 1
    assert widget.table_index.model().columnCount() == 1
    assert [row_label(widget, r) for r in range(3)] == ["0", "1", "2"]
    assert spans(widget.table_index, horizontal=False) == set()
    if isinstance(widget, GridDataFrameViewer):
        corner = widget.table_level.model()
        assert (corner.rowCount(), corner.columnCount()) == (1, 1)


def test_viewer_copy_to_clipboard_permutation(qtbot, viewer_cls):
    widget = viewer_cls(
        pd.DataFrame({"a": [3, 1, 2], "b": ["c", "a", "b"]}), sort_mode="permutation"
    )
    qtbot.addWidget(widget)
    widget.sort_by(0, constants.ASCENDING)
    qtbot.waitUntil(lambda: widget.holder.row_order is not None)
    view = data_view(widget)
    view.selectAll()
    view.copy()
    text = gui.GuiApplication.clipboard().text()
    assert text.splitlines() == ["1\ta", "2\tb", "3\tc"]


# --- palette / theme ---


def dark_palette() -> gui.QPalette:
    palette = gui.QPalette()
    role = gui.QPalette.ColorRole
    for r, color in [
        (role.Window, "#202020"),
        (role.WindowText, "#f0f0f0"),
        (role.Base, "#181818"),
        (role.Text, "#eeeeee"),
        (role.Highlight, "#3070d0"),
    ]:
        palette.setColor(r, gui.QColor(color))
    return palette


def icon_color(icon) -> str:
    """Colour of the most opaque pixel of a rendered icon."""
    image = icon.pixmap(32, 32).toImage()
    best = max(
        (image.pixelColor(x, y) for x in range(32) for y in range(32)),
        key=lambda c: c.alpha(),
    )
    return best.name()


def test_kind_icon_cache_follows_app_palette(qapp):
    original = gui.GuiApplication.palette()
    cache = icons.KindIconCache()
    try:
        first = cache.icon(Kind.INTEGER)
        assert cache.icon(Kind.INTEGER) is first  # cached
        assert icon_color(first) == icons.default_icon_color()
        gui.GuiApplication.setPalette(dark_palette())
        dark = cache.icon(Kind.INTEGER)
        assert dark is not first  # the palette changed: rebuilt
        assert icons.default_icon_color() == icons.default_icon_color(dark_palette())
        assert icon_color(dark) == icons.default_icon_color()
        explicit = cache.icon(Kind.FLOAT, color="#ff0000")
        assert cache.icon(Kind.FLOAT, color="#ff0000") is explicit
        assert icon_color(explicit) == "#ff0000"
    finally:
        gui.GuiApplication.setPalette(original)


def test_dataframe_model_icons_cached(qapp):
    model = pandasmodels.DataFrameModel(pd.DataFrame({"a": [1]}), dtype_icons=True)
    role = constants.DECORATION_ROLE
    icon = model.headerData(0, constants.HORIZONTAL, role)
    assert model.headerData(0, constants.HORIZONTAL, role) is icon


def delegates(widget) -> list:
    if isinstance(widget, SimpleDataFrameViewer):
        return [widget.itemDelegate()]
    views = [widget.table_data, column_header(widget), widget.table_index]
    if isinstance(widget, GridDataFrameViewer):
        views.append(widget.table_level)
    return [view.itemDelegate() for view in views]


def test_viewer_restyles_on_palette_change(qtbot, viewer_cls):
    widget = viewer_cls(pd.DataFrame({"a": [1, 2]}))
    qtbot.addWidget(widget)
    light = ViewerColors.from_palette(widget.palette())
    assert all(d._colors == light for d in delegates(widget))
    widget.setPalette(dark_palette())
    dark = ViewerColors.from_palette(widget.palette())
    assert dark.base.name() == "#181818"
    assert dark != light
    assert all(d._colors == dark for d in delegates(widget))
    data = data_view(widget)
    viewport_base = data.viewport().palette().color(gui.QPalette.ColorRole.Base)
    if isinstance(widget, SimpleDataFrameViewer):
        assert viewport_base.name() == dark.base.name()
    else:
        assert data.palette().color(gui.QPalette.ColorRole.Base) == dark.base
    # no style sheet on the views without an application / parent style sheet
    assert data.styleSheet() == ""


def test_viewer_header_icon_tint_follows_palette(qtbot):
    widget = DataFrameViewer(pd.DataFrame({"a": [1, 2]}))
    qtbot.addWidget(widget)
    delegate = column_header(widget).itemDelegate()
    widget.setPalette(dark_palette())
    muted = delegate._colors.muted_text.name()
    icon = delegate._kind_icons.icon(Kind.INTEGER, color=muted)
    assert icon_color(icon) == muted


def test_viewer_under_app_style_sheet(qtbot, qapp, viewer_cls):
    """A theme's QTableView border / padding is reset on the composite tables."""
    qapp.setStyleSheet("QTableView { border: 3px solid red; padding-top: 5px; }")
    try:
        widget = viewer_cls(pd.DataFrame({"a": [1, 2]}))
        qtbot.addWidget(widget)
        widget.show()
        if isinstance(widget, SimpleDataFrameViewer):
            assert widget.itemDelegate()._colors is not None
            return
        for view in (widget.table_data, column_header(widget), widget.table_index):
            assert "border: none" in view.styleSheet()
            assert view.frameWidth() == 0
    finally:
        qapp.setStyleSheet("")
    qapp.processEvents()  # StyleChange: restyled without the app style sheet
    if not isinstance(widget, SimpleDataFrameViewer):
        assert widget.table_data.styleSheet() == ""
