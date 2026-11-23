"""Tests for `cutepandas.pandasmodels`."""

from __future__ import annotations

import numpy as np
import pandas as pd
from prettyqt import constants, core
import pytest

from cutepandas import pandasmodels


@pytest.fixture
def multi_df():
    index = pd.MultiIndex.from_tuples(
        [("a", "1"), ("a", "2"), ("b", "1")], names=["l0", "l1"]
    )
    return pd.DataFrame({"x": [1.0, 2.0, np.nan], "y": ["p", "q", None]}, index=index)


@pytest.fixture
def cat_df():
    return pd.DataFrame({"c": pd.Categorical(["b", "a", "b", "c"])})


@pytest.fixture
def filter_df():
    return pd.DataFrame({"a": ["ab", "b", "abc", None], "n": [1, 5, 10, 20]})


def make_proxy(cls, df):
    source = pandasmodels.PandasDataFrameModel(df)
    proxy = cls()
    proxy.setSourceModel(source)
    return source, proxy


# --- PandasDataFrameModel ---


def test_dataframemodel_modeltester(qtmodeltester, multi_df):
    model = pandasmodels.PandasDataFrameModel(multi_df)
    qtmodeltester.check(model, force_py=True)


def test_dataframemodel_series():
    model = pandasmodels.PandasDataFrameModel(pd.Series([1, 2], name="s"))
    assert model.columnCount() == 1
    assert model.data(model.index(1, 0)) == "2"
    assert model.data(model.index(1, 0), constants.EDIT_ROLE) == 2
    assert model.headerData(0, constants.HORIZONTAL) == "s"


def test_dataframemodel_nan_display(multi_df):
    model = pandasmodels.PandasDataFrameModel(multi_df)
    assert model.data(model.index(2, 0)) == ""
    assert model.data(model.index(2, 0), constants.TOOLTIP_ROLE) == "NaN"


def test_dataframemodel_sort(df):
    model = pandasmodels.PandasDataFrameModel(df)
    model.sort(0, constants.DESCENDING)
    assert model.data(model.index(0, 0)) == "c"


# --- PandasColumnListModel ---


def test_columnlistmodel_modeltester(qtmodeltester, multi_df):
    model = pandasmodels.PandasColumnListModel(multi_df)
    qtmodeltester.check(model, force_py=True)


def test_columnlistmodel_set_dtype():
    model = pandasmodels.PandasColumnListModel(pd.DataFrame({"a": [1, 2], "b": [3, 4]}))
    role = model.Roles.DTypeRole
    assert model.setData(model.index(1, 1), "float64", role)
    assert model.df["b"].dtype == np.float64
    assert model.df["a"].dtype == np.int64


def test_columnlistmodel_set_invalid_dtype():
    model = pandasmodels.PandasColumnListModel(pd.DataFrame({"a": ["x"]}))
    assert not model.setData(model.index(0, 1), "int64", model.Roles.DTypeRole)


def test_columnlistmodel_sort_order():
    df = pd.DataFrame({"b": [1], "a": [1], "c": [1]})
    model = pandasmodels.PandasColumnListModel(df)
    model.sort(0, constants.ASCENDING)
    assert list(model.df.columns) == ["a", "b", "c"]
    model.sort(0, constants.DESCENDING)
    assert list(model.df.columns) == ["c", "b", "a"]


def test_columnlistmodel_sort_by_size_is_numeric():
    df = pd.DataFrame({"big": np.zeros(13, dtype=np.int64), "small": np.zeros(13, "i1")})
    model = pandasmodels.PandasColumnListModel(df)
    # 104 bytes vs 13 bytes: a string comparison would put "104" first.
    model.sort(2, constants.ASCENDING)
    assert list(model.df.columns) == ["small", "big"]


def test_columnlistmodel_remove_rows(qtbot):
    model = pandasmodels.PandasColumnListModel(pd.DataFrame({"a": [1], "b": [2]}))
    with qtbot.waitSignal(model.rowsAboutToBeRemoved) as blocker:
        assert model.removeRows(0, 1, core.ModelIndex())
    assert blocker.args[1:] == [0, 0]
    assert list(model.df.columns) == ["b"]


# --- PandasIndexListModel ---


def test_indexlistmodel_modeltester(qtmodeltester, multi_df):
    model = pandasmodels.PandasIndexListModel(multi_df)
    qtmodeltester.check(model, force_py=True)


def test_indexlistmodel_data(df, multi_df):
    model = pandasmodels.PandasIndexListModel(df)
    assert model.data(model.index(0, 0)) == "[Unnamed]"
    assert model.data(model.index(0, 1)) == "Range"
    assert model.flags(model.index(0, 0)) & constants.IS_EDITABLE
    model = pandasmodels.PandasIndexListModel(multi_df)
    assert model.rowCount() == 2
    assert model.data(model.index(1, 0)) == "l1"
    assert model.data(model.index(1, 1)) == "Regular"


def test_indexlistmodel_rename(multi_df):
    model = pandasmodels.PandasIndexListModel(multi_df)
    assert model.setData(model.index(0, 0), "renamed")
    assert list(model.df.index.names) == ["renamed", "l1"]


def test_indexlistmodel_remove_rows(multi_df):
    model = pandasmodels.PandasIndexListModel(multi_df)
    assert model.removeRows(0, 1, core.ModelIndex())
    assert list(model.df.index.names) == ["l1"]
    assert list(model.df.columns) == ["x", "y"]


# --- PandasCategoryListModel ---


def test_categorylistmodel_modeltester(qtmodeltester, cat_df):
    model = pandasmodels.PandasCategoryListModel(cat_df, "c")
    qtmodeltester.check(model, force_py=True)


def test_categorylistmodel_data(cat_df):
    model = pandasmodels.PandasCategoryListModel(cat_df, "c")
    assert model.rowCount() == 3
    assert [model.data(model.index(r, 0)) for r in range(3)] == ["a", "b", "c"]
    # The code of a category is its position, not the code of the n-th item.
    assert [model.data(model.index(r, 1)) for r in range(3)] == ["0", "1", "2"]
    assert "Items: 2" in model.data(model.index(1, 0), constants.TOOLTIP_ROLE)


def test_categorylistmodel_user_role_does_not_mutate(cat_df):
    model = pandasmodels.PandasCategoryListModel(cat_df, "c")
    subset = model.data(model.index(1, 0), constants.USER_ROLE)
    assert len(subset) == 2
    assert len(model.df) == 4
    assert model.rowCount() == 3


def test_categorylistmodel_sort_order(cat_df):
    model = pandasmodels.PandasCategoryListModel(cat_df, "c")
    model.sort(0, constants.DESCENDING)
    assert list(cat_df["c"].cat.categories) == ["c", "b", "a"]
    model.sort(0, constants.ASCENDING)
    assert list(cat_df["c"].cat.categories) == ["a", "b", "c"]


def test_categorylistmodel_add_categories(qtbot, cat_df):
    model = pandasmodels.PandasCategoryListModel(cat_df, "c")
    with qtbot.waitSignal(model.rowsInserted) as blocker:
        model.add_category(["d", "e"])
    assert blocker.args[1:] == [3, 4]
    assert model.rowCount() == 5


def test_categorylistmodel_empty():
    model = pandasmodels.PandasCategoryListModel()
    assert model.rowCount() == 0


# --- Filter proxies ---


def test_string_column_filter_proxy(df):
    _, proxy = make_proxy(pandasmodels.PandasStringColumnFilterProxyModel, df)
    proxy.set_filter_column(0)
    proxy.set_search_term("b")
    assert proxy.rowCount() == 1


def test_string_filter_modes(qtmodeltester, filter_df):
    _, proxy = make_proxy(pandasmodels.PandasStringColumnFilterProxyModel, filter_df)
    proxy.set_search_term("b")
    assert proxy.rowCount() == 1
    # changing settings re-applies the current search term
    proxy.set_filter_mode("contains")
    assert proxy.rowCount() == 3
    proxy.set_filter_mode("endswith")
    assert proxy.rowCount() == 2
    proxy.set_search_term("")
    assert proxy.rowCount() == 4
    qtmodeltester.check(proxy, force_py=True)


def test_string_filter_non_string_column(filter_df):
    _, proxy = make_proxy(pandasmodels.PandasStringColumnFilterProxyModel, filter_df)
    proxy.set_filter_column(1)
    proxy.set_search_term("1")
    assert proxy.rowCount() == 2


def test_filter_mapping(filter_df):
    source, proxy = make_proxy(pandasmodels.PandasStringColumnFilterProxyModel, filter_df)
    proxy.set_search_term("a")
    assert proxy.rowCount() == 2
    assert proxy.mapToSource(proxy.index(1, 0)).row() == 2
    assert proxy.mapFromSource(source.index(2, 0)).row() == 1
    # filtered-out rows have no proxy index
    assert not proxy.mapFromSource(source.index(1, 0)).isValid()
    assert proxy.data(proxy.index(1, 1)) == "10"


def test_filter_follows_source_changes(filter_df):
    source, proxy = make_proxy(pandasmodels.PandasStringColumnFilterProxyModel, filter_df)
    proxy.set_search_term("a")
    assert proxy.rowCount() == 2
    with source.reset_model():
        source.df = pd.concat(
            [source.df, pd.DataFrame({"a": ["az"], "n": [0]})], ignore_index=True
        )
    assert proxy.rowCount() == 3
    source.sort(1, constants.ASCENDING)
    assert proxy.data(proxy.index(0, 0)) == "az"


def test_filter_forwards_data_changes(qtbot, filter_df):
    source, proxy = make_proxy(pandasmodels.PandasStringColumnFilterProxyModel, filter_df)
    proxy.set_search_term("a")
    with qtbot.waitSignal(proxy.dataChanged) as blocker:
        source.setData(source.index(2, 1), 99)
    assert blocker.args[0].row() == 1
    assert proxy.data(proxy.index(1, 1)) == "99"


def test_eval_filter(qtmodeltester, filter_df):
    _, proxy = make_proxy(pandasmodels.PandasEvalFilterProxyModel, filter_df)
    proxy.set_expression("n > 2")
    assert proxy.rowCount() == 3
    proxy.set_expression("")
    assert proxy.rowCount() == 4
    proxy.set_expression("not valid python (")
    assert proxy.rowCount() == 0
    proxy.set_expression("n + 1")  # not a boolean result
    assert proxy.rowCount() == 0
    qtmodeltester.check(proxy, force_py=True)


def test_multi_string_filter(filter_df):
    _, proxy = make_proxy(pandasmodels.PandasMultiStringColumnFilterProxyModel, filter_df)
    proxy.set_filters({"a": "a"})
    assert proxy.rowCount() == 2
    proxy.set_filters({"a": "a", "n": "1"})
    assert proxy.rowCount() == 2
    proxy.set_filters({"a": "ab", "n": "10"})
    assert proxy.rowCount() == 1
    proxy.set_filters({"a": "it's"})
    assert proxy.rowCount() == 0
    proxy.set_filters({})
    assert proxy.rowCount() == 4


def test_multi_string_filter_is_real_startswith():
    df = pd.DataFrame({"a": ["a", "a~", "aé", "b"]})
    _, proxy = make_proxy(pandasmodels.PandasMultiStringColumnFilterProxyModel, df)
    proxy.set_filters({"a": "a"})
    assert proxy.rowCount() == 3


def test_polars_model(qtmodeltester):
    pl = pytest.importorskip("polars")
    from cutepandas.pandasmodels.polarsdataframemodel import (  # noqa: PLC0415
        PolarsDataFrameModel,  # needs the optional polars dependency
    )

    model = PolarsDataFrameModel(pl.DataFrame({"a": [3, 1, 2], "b": ["x", "y", "z"]}))
    qtmodeltester.check(model, force_py=True)
    assert model.setData(model.index(0, 0), 5)
    assert model.df[0, "a"] == 5
    model.sort(0, constants.ASCENDING)
    assert model.df["a"].to_list() == [1, 2, 5]
