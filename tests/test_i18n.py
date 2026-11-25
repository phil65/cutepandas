"""Translatable labels (`translate` callables, `retranslate()`)."""

from __future__ import annotations

import pandas as pd
from prettyqt import constants

from cutepandas import pandasmodels, pandaswidgets
from cutepandas.utils.i18n import N_


GERMAN = {
    "Name": "Name",
    "Type": "Typ",
    "Nulls": "Leer",
    "Size": "Größe",
    "Code": "Code",
    "Count": "Anzahl",
    "Value": "Wert",
    "Share": "Anteil",
    "Rows": "Zeilen",
    "Index type": "Indextyp",
    "Range": "Bereich",
    "Column profile": "Spaltenprofil",
    "Items": "Einträge",
}


def german(text: str) -> str:
    return GERMAN.get(text, text)


def headers(model) -> list[str]:
    return [model.headerData(i, constants.HORIZONTAL) for i in range(model.columnCount())]


def test_n_is_identity():
    assert N_("Rows") == "Rows"


def test_schema_model_translated(qtbot):
    df = pd.DataFrame({"a": [1, None]})
    model = pandasmodels.SchemaModel(df)
    assert headers(model) == ["Name", "Type", "Nulls", "Size"]
    with qtbot.waitSignals([model.headerDataChanged, model.dataChanged]):
        model.set_translate(german)
    assert headers(model) == ["Name", "Typ", "Leer", "Größe"]
    tooltip = model.data(model.index(0, 0), constants.TOOLTIP_ROLE)
    assert "Typ: float64" in tooltip
    assert "Leer: 1" in tooltip
    model.set_translate(None)
    assert headers(model)[1] == "Type"
    column_model = pandasmodels.PandasColumnListModel(df, translate=german)
    assert headers(column_model)[1] == "Typ"


def test_category_and_index_models_translated():
    df = pd.DataFrame({"c": pd.Categorical(["a", "b", "a"])})
    model = pandasmodels.CategoryModel(df, "c", translate=german)
    assert headers(model) == ["Name", "Code", "Anzahl"]
    assert "Einträge: 2" in model.data(model.index(0, 0), constants.TOOLTIP_ROLE)
    pandas_model = pandasmodels.PandasCategoryListModel(df, "c", translate=german)
    assert headers(pandas_model)[2] == "Anzahl"
    index_model = pandasmodels.PandasIndexListModel(df, translate=german)
    assert headers(index_model)[1] == "Indextyp"
    assert index_model.data(index_model.index(0, 1)) == "Bereich"


def test_column_profile_widget_translated(qtbot):
    df = pd.DataFrame({"a": [1, 1, 2]})
    widget = pandaswidgets.ColumnProfileWidget(df, "a", translate=german)
    qtbot.addWidget(widget)
    assert widget.field_labels["Rows"].text() == "Zeilen:"
    top = widget.top_table.model()
    assert headers(top) == ["Wert", "Anzahl", "Anteil"]
    widget.set_translate(None)
    assert widget.field_labels["Rows"].text() == "Rows:"
    assert headers(widget.top_table.model())[0] == "Value"
    widget.clear()
    widget.set_translate(german)
    assert headers(widget.top_table.model())[0] == "Wert"


def test_dataframe_widget_translated(qtbot):
    widget = pandaswidgets.DataFrameWidget(pd.DataFrame({"a": [1]}), translate=german)
    qtbot.addWidget(widget)
    titles = [dock.windowTitle() for dock in widget.docks]
    assert "Spaltenprofil" in titles
    assert headers(widget.column_table.model())[1] == "Typ"
    widget.set_translate(None)
    assert "Column profile" in [dock.windowTitle() for dock in widget.docks]
    assert headers(widget.column_table.model())[1] == "Type"
    assert widget.profile.field_labels["Rows"].text() == "Rows:"
