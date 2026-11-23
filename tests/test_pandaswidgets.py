"""Tests for `cutepandas` package."""

from __future__ import annotations

import pandas as pd
from prettyqt import core, gui

from cutepandas import pandaswidgets
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
