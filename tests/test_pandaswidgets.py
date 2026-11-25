"""Tests for `cutepandas` package."""

from __future__ import annotations

import pandas as pd
from prettyqt import core, gui, widgets

from cutepandas import pandaswidgets
from cutepandas.pandasmodels import DataFrameHolder
from cutepandas.qtframe.qtframe import QtFrame


def test_dataframeviewer(qtbot, df):
    widget = pandaswidgets.DataFrameViewer(df)
    qtbot.addWidget(widget)
    widget.show()


def test_dataframewidget(qtbot, df):
    df.attrs = {"source": "test"}
    widget = pandaswidgets.DataFrameWidget(df)
    qtbot.addWidget(widget)
    widget.set_df(df)


def test_dataframemanagerwidget(qtbot, df):
    widget = pandaswidgets.DataFrameManagerWidget()
    qtbot.addWidget(widget)
    widget.add_df(df)
    widget.add_df(pd.DataFrame({"x": [1.0, 2.0]}))


def test_dataframelistwidget(qtbot, df):
    widget = pandaswidgets.DataFrameListWidget()
    qtbot.addWidget(widget)
    widget.add_df(df)


def test_dataframeviewer_empty_then_set(qtbot, df):
    widget = pandaswidgets.DataFrameViewer()
    qtbot.addWidget(widget)
    widget.set_df(df)
    assert widget.table_data.model().rowCount() == 3


def test_dataframeviewer_header_click_sorts(qtbot):
    widget = pandaswidgets.DataFrameViewer(pd.DataFrame({"a": [3, 1, 2]}))
    qtbot.addWidget(widget)
    header_index = widget.table_columns.model().index(0, 0)
    widget.table_columns.clicked.emit(header_index)
    model = widget.table_data.model()
    assert [model.data(model.index(r, 0)) for r in range(3)] == ["1", "2", "3"]
    index_model = widget.table_index.model()
    assert index_model.data(index_model.index(0, 0)) == "1"


def test_dataframeviewer_copy_paste(qtbot):
    widget = pandaswidgets.DataFrameViewer(pd.DataFrame({"a": [1, 2], "b": [3, 4]}))
    qtbot.addWidget(widget)
    table = widget.table_data
    table.copy()  # nothing selected: must not raise
    table.selectAll()
    table.copy()
    clipboard = gui.GuiApplication.clipboard()
    assert clipboard.text().splitlines() == ["1\t3", "2\t4"]
    clipboard.setText("7\t8")
    table.setCurrentIndex(table.model().index(1, 0))
    table.paste()
    assert widget.df is table.get_model(skip_proxies=True).df
    assert table.get_model(skip_proxies=True).df.iloc[1].tolist() == [7, 8]


def test_dataframemanagerwidget_select(qtbot, df):
    widget = pandaswidgets.DataFrameManagerWidget()
    qtbot.addWidget(widget)
    other = pd.DataFrame({"x": [1.0, 2.0]})
    widget.add_df(df)
    widget.add_df(other)
    view = widget.dataframe_list
    flag = core.ItemSelectionModel.SelectionFlag.ClearAndSelect
    view.selectionModel().setCurrentIndex(view.model().index(1, 0), flag)
    assert widget.dataframe_editor.data_table.df is other


def test_dataframewidget_category_list(qtbot):
    df = pd.DataFrame({"n": [1, 2], "c": pd.Categorical(["a", "b"])})
    widget = pandaswidgets.DataFrameWidget(df)
    qtbot.addWidget(widget)
    view = widget.column_table
    flag = core.ItemSelectionModel.SelectionFlag.ClearAndSelect
    view.selectionModel().setCurrentIndex(view.model().index(1, 0), flag)
    assert widget.category_table.model().rowCount() == 2
    # a new frame gets a fresh column list that is still connected
    widget.set_df(df.copy())
    view.selectionModel().setCurrentIndex(view.model().index(1, 0), flag)
    assert widget.category_table.model().rowCount() == 2


def test_qtframe_reductions():
    frame = QtFrame({"a": [1, 2]})
    assert frame.sum()["a"] == 3
    assert frame.T.shape == (1, 2)


def test_dataframeviewer_spans(qtbot):
    index = pd.MultiIndex.from_tuples([("a", 1), ("a", 1), ("b", 1), ("b", 2)])
    widget = pandaswidgets.DataFrameViewer(pd.DataFrame({"v": range(4)}, index=index))
    qtbot.addWidget(widget)
    header = widget.table_index
    assert header.rowSpan(0, 0) == 2  # "a"
    assert header.rowSpan(2, 0) == 2  # "b"
    assert header.rowSpan(0, 1) == 2  # the two 1s inside "a"
    assert header.rowSpan(2, 1) == 1  # the 1 in "b" doesn't merge across a/b


def test_dataframeviewer_spans_unique_index(qtbot):
    widget = pandaswidgets.DataFrameViewer(pd.DataFrame({"v": range(1000)}))
    qtbot.addWidget(widget)
    assert widget.table_index.rowSpan(0, 0) == 1


def test_dataframeviewer_fills_parent(qtbot):
    """A small frame must not shrink the viewer to its content size."""
    parent = widgets.Widget()
    qtbot.addWidget(parent)
    layout = parent.set_layout("vertical", margin=0)
    viewer = pandaswidgets.DataFrameViewer(pd.DataFrame({"a": [1, 2]}))
    layout.add(viewer)
    parent.resize(900, 700)
    parent.show()
    qtbot.waitExposed(parent)
    assert viewer.width() >= 890
    # the data table (not an empty grid cell) takes the extra space
    assert viewer.table_data.width() > 700
    assert viewer.table_data.height() > 550


def test_dataframeviewer_fits_columns_and_row_height(qtbot):
    df = pd.DataFrame({
        "a": [1, 2],
        "ts": pd.to_datetime(["2023-01-01 08:01:00", "2023-01-02 09:00:00"]),
        "text": ["a rather long text value in this cell", "x"],
    })
    viewer = pandaswidgets.DataFrameViewer(df)
    qtbot.addWidget(viewer)
    viewer.resize(1200, 400)
    viewer.show()
    table = viewer.table_data
    metrics = table.fontMetrics()
    texts = {1: "2023-01-01 08:01:00", 2: "a rather long text value in this cell"}
    for col, text in texts.items():  # every column fitted, not only the first
        assert table.columnWidth(col) >= metrics.horizontalAdvance(text)
        assert viewer.table_columns.columnWidth(col) == table.columnWidth(col)
    # one line per row, not the height of text wrapped into a narrow column
    assert table.rowHeight(0) <= metrics.height() * 2
    assert viewer.table_index.rowHeight(0) == table.rowHeight(0)


def test_dataframeviewer_index_header_after_set_df(qtbot):
    """A frame set after the viewer is shown (inside a layout) gets a visible index."""
    parent = widgets.QWidget()
    qtbot.addWidget(parent)
    layout = widgets.QVBoxLayout(parent)
    holder = DataFrameHolder()
    viewer = pandaswidgets.DataFrameViewer(holder)
    layout.addWidget(viewer)
    parent.resize(800, 400)
    parent.show()
    widgets.QApplication.processEvents()  # laid out while still empty
    holder.df = pd.DataFrame({"a": range(10)})
    qtbot.wait(50)
    index = viewer.table_index
    assert index.width() >= index.columnWidth(0) > 0
