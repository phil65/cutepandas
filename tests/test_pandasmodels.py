"""Tests for `cutepandas` package."""

from cutepandas import pandasmodels


# def test_categorylistmodel(qtmodeltester, df):
#     model = pandasmodels.PandasCategoryListModel(df)
#     assert model
# qtmodeltester.check(model, force_py=True)


def test_pandascolumnlistmodel(qtmodeltester, df):
    model = pandasmodels.PandasColumnListModel(df)
    assert model
    # qtmodeltester.check(model, force_py=True)


def test_string_column_filter_proxy(df):
    model = pandasmodels.PandasDataFrameModel(df)
    proxy = pandasmodels.PandasStringColumnFilterProxyModel()
    proxy.setSourceModel(model)
    proxy.set_filter_column(0)
    proxy.set_search_term("b")
    assert proxy.rowCount() == 1
