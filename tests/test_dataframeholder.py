"""Tests for sharing one DataFrame between models and widgets."""

import pandas as pd
from prettyqt import constants, core
import pytest

from cutepandas import pandasmodels, pandaswidgets


@pytest.fixture
def frame():
    return pd.DataFrame({
        "n": [3, 1, 2],
        "c": pd.Categorical(["x", "y", "x"]),
    })


def select_row(view, row: int):
    flag = core.ItemSelectionModel.SelectionFlag.ClearAndSelect
    view.selectionModel().setCurrentIndex(view.model().index(row, 0), flag)


def header_texts(viewer) -> list[str]:
    model = viewer.table_columns.model()
    return [model.data(model.index(0, c)) for c in range(model.columnCount())]


def test_holder_signals(qtbot, frame):
    holder = pandasmodels.DataFrameHolder(frame)
    assert holder.df is frame
    with (
        qtbot.waitSignal(holder.about_to_change) as about,
        qtbot.waitSignal(holder.changed) as changed,
    ):
        holder.set_df(frame.head(1), origin="me")
    assert about.args == ["me"]
    assert changed.args == ["me"]
    assert len(holder.df) == 1


def test_holder_wraps_series():
    holder = pandasmodels.DataFrameHolder(pd.Series([1, 2], name="s"))
    assert list(holder.df.columns) == ["s"]


def test_models_share_holder(qtbot, frame):
    holder = pandasmodels.DataFrameHolder(frame)
    data_model = pandasmodels.DataFrameModel(holder)
    column_model = pandasmodels.PandasColumnListModel(holder)
    with qtbot.waitSignal(data_model.modelReset):
        assert column_model.setData(column_model.index(0, 0), "renamed")
    assert data_model.headerData(0, constants.HORIZONTAL) == "renamed"
    assert data_model.df is column_model.df is holder.df


def test_model_owns_private_holder(frame):
    model = pandasmodels.DataFrameModel(frame)
    other = pandasmodels.DataFrameModel(frame)
    model.sort(0, constants.ASCENDING)
    assert other.data(other.index(0, 0)) == "3"  # not shared: unaffected


def test_proxy_follows_holder(frame):
    holder = pandasmodels.DataFrameHolder(frame)
    source = pandasmodels.DataFrameModel(holder)
    proxy = pandasmodels.PandasEvalFilterProxyModel()
    proxy.setSourceModel(source)
    proxy.set_expression("n > 1")
    assert proxy.rowCount() == 2
    holder.df = pd.DataFrame({"n": [5, 6, 7, 0]})
    assert proxy.rowCount() == 3


def test_widget_column_rename_updates_viewer(qtbot, frame):
    widget = pandaswidgets.DataFrameWidget(frame)
    qtbot.addWidget(widget)
    column_model = widget.column_table.model()
    assert column_model.setData(column_model.index(0, 0), "renamed")
    assert header_texts(widget.data_table) == ["renamed", "c"]
    assert widget.data_table.df is column_model.df


def test_widget_remove_column_updates_viewer(qtbot, frame):
    widget = pandaswidgets.DataFrameWidget(frame)
    qtbot.addWidget(widget)
    widget.column_table.model().removeRows(0, 1, core.ModelIndex())
    assert widget.data_table.table_data.model().columnCount() == 1
    assert header_texts(widget.data_table) == ["c"]


def test_widget_dtype_change_updates_viewer(qtbot, frame):
    widget = pandaswidgets.DataFrameWidget(frame)
    qtbot.addWidget(widget)
    column_model = widget.column_table.model()
    role = column_model.Roles.DTypeRole
    assert column_model.setData(column_model.index(0, 1), "float64", role)
    data_model = widget.data_table.table_data.model()
    assert data_model.data(data_model.index(0, 0)) == "3.0"


def test_widget_index_rename_updates_viewer(qtbot, frame):
    widget = pandaswidgets.DataFrameWidget(frame)
    qtbot.addWidget(widget)
    index_model = widget.index_table.model()
    assert index_model.setData(index_model.index(0, 0), "idx")
    header_model = widget.data_table.table_index.model()
    assert header_model.headerData(0, constants.HORIZONTAL) == "idx"
    assert widget.data_table.df.index.name == "idx"
    assert frame.index.name is None  # the caller's frame is left alone


def test_widget_category_rename_updates_viewer(qtbot, frame):
    widget = pandaswidgets.DataFrameWidget(frame)
    qtbot.addWidget(widget)
    select_row(widget.column_table, 1)
    category_model = widget.category_table.model()
    assert category_model.setData(category_model.index(0, 0), "z")
    data_model = widget.data_table.table_data.model()
    assert data_model.data(data_model.index(0, 1)) == "z"
    assert list(frame["c"]) == ["x", "y", "x"]  # the caller's frame is left alone


def test_widget_viewer_sort_updates_lists(qtbot, frame):
    widget = pandaswidgets.DataFrameWidget(frame)
    qtbot.addWidget(widget)
    viewer = widget.data_table
    viewer.table_columns.clicked.emit(viewer.table_columns.model().index(0, 0))
    assert list(viewer.df["n"]) == [1, 2, 3]
    assert widget.column_table.model().df is viewer.df
    assert widget.index_table.model().df is viewer.df


def test_widget_set_df_keeps_sync(qtbot, frame):
    widget = pandaswidgets.DataFrameWidget()
    qtbot.addWidget(widget)
    widget.set_df(frame)
    column_model = widget.column_table.model()
    column_model.setData(column_model.index(1, 0), "cat")
    assert header_texts(widget.data_table) == ["n", "cat"]
    select_row(widget.column_table, 1)
    assert widget.category_table.model().rowCount() == 2


def test_manager_list_follows_edits(qtbot, frame):
    widget = pandaswidgets.DataFrameManagerWidget()
    qtbot.addWidget(widget)
    widget.add_df(frame)
    select_row(widget.dataframe_list, 0)
    editor = widget.dataframe_editor
    editor.column_table.model().removeRows(0, 1, core.ModelIndex())
    list_model = widget.dataframe_list.model()
    assert list_model.items[0] is editor.data_table.df
    assert list_model.data(list_model.index(0, 1)) == "1"  # "Columns" column


def test_widget_replaces_category_models(qtbot, frame):
    widget = pandaswidgets.DataFrameWidget(frame)
    qtbot.addWidget(widget)
    for row in [1, 0, 1, 0, 1]:
        select_row(widget.column_table, row)
    qtbot.wait(10)  # let deleteLater run
    models = widget.findChildren(pandasmodels.PandasCategoryListModel)
    assert models == [widget.category_table.model()]


def test_column_drag_reorder_updates_viewer(qtbot, frame):
    widget = pandaswidgets.DataFrameWidget(frame)
    qtbot.addWidget(widget)
    column_model = widget.column_table.model()
    mime = column_model.mimeData([column_model.index(1, 0)])
    column_model.dropMimeData(
        mime, constants.DropAction.MoveAction, 0, 0, core.ModelIndex()
    )
    assert header_texts(widget.data_table) == ["c", "n"]


def test_widget_multiindex_frame(qtbot):
    index = pd.MultiIndex.from_product([["a", "b"], [1, 2]], names=["k", "i"])
    columns = pd.MultiIndex.from_product([["x", "y"], ["1", "2"]])
    df = pd.DataFrame(range(16), dtype=float).values.reshape(4, 4)
    df = pd.DataFrame(df, index=index, columns=columns)
    widget = pandaswidgets.DataFrameWidget(df)
    qtbot.addWidget(widget)
    widget.show()
    assert widget.index_table.model().rowCount() == 2
    widget.index_table.model().removeRows(0, 1, core.ModelIndex())
    header_model = widget.data_table.table_index.model()
    assert header_model.columnCount() == 1
    assert header_model.headerData(0, constants.HORIZONTAL) == "i"
    widget.column_table.model().removeRows(0, 2, core.ModelIndex())
    assert widget.data_table.table_columns.model().columnCount() == 2
