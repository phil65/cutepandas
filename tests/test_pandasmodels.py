"""Tests for `cutepandas` package."""

from cutepandas import pandasmodels


def test_pandasmodel(qtmodeltester, df):
    model = pandasmodels.PandasModel(df)
    qtmodeltester.check(model, force_py=True)

def test_pandasindexmodel(qtmodeltester, df):
    model = pandasmodels.PandasIndexModel(df)
    qtmodeltester.check(model, force_py=True)
