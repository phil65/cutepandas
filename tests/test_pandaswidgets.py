"""Tests for `cutepandas` package."""

from __future__ import annotations

import pandas as pd

from cutepandas import pandaswidgets


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
