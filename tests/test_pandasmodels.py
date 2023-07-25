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
