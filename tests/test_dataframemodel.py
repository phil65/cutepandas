"""Tests for table, schema and category models, delegates and widgets (all backends)."""

import threading

import numpy as np
import pandas as pd
from prettyqt import constants, widgets
import pyarrow as pa
import pytest

from cutepandas import delegates, frames, pandasmodels, pandaswidgets
from cutepandas.frames import FormatOptions, Kind


pl = pytest.importorskip("polars")

Roles = pandasmodels.DataFrameModel.Roles
BACKENDS = ["pandas", "pandas-arrow", "pyarrow", "polars"]


def base_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "n": [10, 20, 30, 40],
        "f": [0.5, np.nan, 2.0, 1.0],
        "s": ["x", None, "z", "x"],
        "c": pd.Categorical(["a", "b", "a", "a"]),
        "b": [True, False, True, False],
    })


def make(backend: str):
    df = base_frame()
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


@pytest.fixture(params=BACKENDS)
def frame(request):
    return make(request.param)


def cell(model, row, col, role=constants.DISPLAY_ROLE):
    return model.data(model.index(row, col), role)


# --- DataFrameModel ---


def test_modeltester(qtmodeltester, frame):
    model = pandasmodels.DataFrameModel(frame, chunk_rows=2)
    qtmodeltester.check(model, force_py=True)


def test_shape_and_header(frame):
    model = pandasmodels.DataFrameModel(frame)
    assert (model.rowCount(), model.columnCount()) == (4, 5)
    assert model.headerData(0, constants.HORIZONTAL) == "n"
    assert model.headerData(2, constants.VERTICAL) == "2"
    assert "\n" in model.headerData(0, constants.HORIZONTAL, constants.TOOLTIP_ROLE)
    assert model.headerData(1, constants.HORIZONTAL, Roles.KindRole) == Kind.FLOAT


def test_display_roles(frame):
    model = pandasmodels.DataFrameModel(frame, chunk_rows=3)
    assert [cell(model, r, 0) for r in range(4)] == ["10", "20", "30", "40"]
    assert cell(model, 0, 1) == "0.5"
    assert cell(model, 1, 1) == ""  # missing
    assert cell(model, 1, 1, constants.TOOLTIP_ROLE) == "null"
    assert cell(model, 1, 1, Roles.IsNullRole) is True
    assert cell(model, 1, 1, constants.EDIT_ROLE) is None
    assert cell(model, 2, 2) == "z"
    assert cell(model, 3, 4, constants.EDIT_ROLE) is False
    assert cell(model, 0, 3) == "a"
    assert cell(model, 0, 0, Roles.ColumnNameRole) == "n"
    assert cell(model, 0, 0, Roles.KindRole) == Kind.INTEGER
    align = cell(model, 0, 0, constants.ALIGNMENT_ROLE)
    assert align & constants.ALIGN_RIGHT
    assert not cell(model, 0, 2, constants.ALIGNMENT_ROLE) & constants.ALIGN_RIGHT


def test_format_options(frame):
    model = pandasmodels.DataFrameModel(frame)
    assert cell(model, 1, 1) == ""
    model.set_format_options(FormatOptions(null_text="NULL"))
    assert cell(model, 1, 1) == "NULL"
    assert model.get_format_options().null_text == "NULL"


def test_normalized_and_colors(frame):
    model = pandasmodels.DataFrameModel(frame)
    assert cell(model, 0, 0, constants.BACKGROUND_ROLE) is None  # colouring off
    assert cell(model, 0, 0, Roles.NormalizedRole) == 0.0
    assert cell(model, 3, 0, Roles.NormalizedRole) == 1.0
    assert cell(model, 1, 1, Roles.NormalizedRole) is None  # null
    assert cell(model, 0, 2, Roles.NormalizedRole) is None  # text
    model.set_color_values(True)
    low = cell(model, 0, 0, constants.BACKGROUND_ROLE)
    high = cell(model, 3, 0, constants.BACKGROUND_ROLE)
    assert low.alpha() < high.alpha()
    true, false = (
        cell(model, 0, 4, constants.BACKGROUND_ROLE),
        cell(model, 1, 4, constants.BACKGROUND_ROLE),
    )
    assert true.green() > true.red()
    assert false.red() > false.green()
    assert cell(model, 0, 2, constants.BACKGROUND_ROLE) is None


def test_sort(frame):
    model = pandasmodels.DataFrameModel(frame)
    model.sort(1, constants.DESCENDING)
    assert [cell(model, r, 1) for r in range(4)] == ["2.0", "1.0", "0.5", ""]
    model.sort(0, constants.ASCENDING)
    assert cell(model, 0, 0) == "10"


def test_edit(frame):
    model = pandasmodels.DataFrameModel(frame)
    editable = not isinstance(frame, pa.Table)
    assert bool(model.flags(model.index(0, 0)) & constants.IS_EDITABLE) == editable
    assert cell(model, 0, 0) == "10"
    assert model.setData(model.index(0, 0), 99) == editable
    assert cell(model, 0, 0) == ("99" if editable else "10")
    model.set_read_only(True)
    assert not model.flags(model.index(0, 0)) & constants.IS_EDITABLE
    assert not model.setData(model.index(0, 0), 5)


def test_edit_invalid_value_fails():
    model = pandasmodels.DataFrameModel(pd.DataFrame({"n": [1, 2]}))
    assert not model.setData(model.index(0, 0), "not a number")
    assert cell(model, 0, 0) == "1"


def test_cache_is_bounded():
    df = pd.DataFrame({"a": np.arange(10_000), "b": np.arange(10_000) * 0.5})
    model = pandasmodels.DataFrameModel(df, chunk_rows=100, max_chunks=8)
    for row in range(0, 10_000, 50):
        cell(model, row, 0)
        cell(model, row, 1)
    assert len(model.cache) == 8
    assert cell(model, 9_999, 0) == "9999"


def test_prefetch_from_worker_thread():
    df = pd.DataFrame({"a": np.arange(1_000)})
    model = pandasmodels.DataFrameModel(df, chunk_rows=100)
    worker = threading.Thread(target=model.prefetch, args=(0, 499))
    worker.start()
    worker.join()
    assert len(model.cache) == 5
    assert cell(model, 450, 0) == "450"


def test_frame_change_clears_cache(qtbot):
    model = pandasmodels.DataFrameModel(pd.DataFrame({"a": [1, 2]}))
    assert cell(model, 0, 0) == "1"
    with qtbot.waitSignal(model.modelReset):
        model.holder.df = pl.DataFrame({"x": ["p", "q", "r"]})
    assert model.rowCount() == 3
    assert cell(model, 0, 0) == "p"
    model.holder.df = None
    assert model.rowCount() == 0
    assert cell(model, 0, 0) is None


def test_source_object_and_supports():
    source = frames.as_source(pa.table({"a": [1, 2]}))
    model = pandasmodels.DataFrameModel(source)
    assert model.source is source
    assert cell(model, 1, 0) == "2"
    assert pandasmodels.DataFrameModel.supports(pl.DataFrame())
    assert not pandasmodels.DataFrameModel.supports([1])


def test_dtype_icons_in_header(qapp):
    model = pandasmodels.DataFrameModel(base_frame(), dtype_icons=True)
    assert model.headerData(0, constants.HORIZONTAL, constants.DECORATION_ROLE)
    model.set_dtype_icons(False)
    assert model.headerData(0, constants.HORIZONTAL, constants.DECORATION_ROLE) is None


def test_viewer_with_other_backends(qtbot, frame):
    viewer = pandaswidgets.DataFrameViewer(frame)
    qtbot.addWidget(viewer)
    viewer.show()
    assert viewer.table_index.model().rowCount() == 4
    assert viewer.table_columns.model().columnCount() == 5
    viewer.on_clicked(viewer.table_columns.model().index(0, 1))
    data = viewer.table_data.model()
    assert cell(data, 0, 1) == "0.5"
    viewer.table_data.selectAll()
    viewer.table_data.copy()


# --- SchemaModel / CategoryModel ---


def test_schema_model(qtmodeltester, frame):
    model = pandasmodels.SchemaModel(frame)
    qtmodeltester.check(model, force_py=True)
    assert model.rowCount() == 5
    assert [cell(model, r, 0) for r in range(5)] == ["n", "f", "s", "c", "b"]
    assert cell(model, 1, 2) == "1"
    assert cell(model, 1, 2, constants.SORT_ROLE) == 1
    assert cell(model, 0, 3, constants.SORT_ROLE) > 0
    assert cell(model, 3, 0, model.Roles.KindRole) == Kind.CATEGORICAL
    assert cell(model, 0, 0, constants.DECORATION_ROLE) is not None
    assert "Nulls: 0" in cell(model, 0, 0, constants.TOOLTIP_ROLE)
    info = cell(model, 2, 0, model.Roles.ColumnInfoRole)
    assert info.kind == Kind.STRING


def test_schema_model_follows_holder(frame):
    holder = pandasmodels.DataFrameHolder(frame)
    model = pandasmodels.SchemaModel(holder)
    assert cell(model, 0, 2) == "0"
    holder.df = pd.DataFrame({"z": [None, None]})
    assert model.rowCount() == 1
    assert cell(model, 0, 2) == "2"


def test_category_model(qtmodeltester, frame):
    model = pandasmodels.CategoryModel(frame, "c")
    qtmodeltester.check(model, force_py=True)
    assert model.rowCount() == 2
    names = [cell(model, r, 0) for r in range(2)]
    assert sorted(names) == ["a", "b"]
    counts = {cell(model, r, 0): cell(model, r, 2) for r in range(2)}
    assert counts == {"a": "3", "b": "1"}
    assert "Items: 3" in cell(model, names.index("a"), 0, constants.TOOLTIP_ROLE)
    assert pandasmodels.CategoryModel(frame, "s").rowCount() == 0
    assert pandasmodels.CategoryModel(frame, "missing").rowCount() == 0
    assert pandasmodels.CategoryModel().rowCount() == 0


# --- delegates and widgets ---


def test_value_color_delegate_paints(qtbot):
    view = widgets.TableView()
    qtbot.addWidget(view)
    model = pandasmodels.DataFrameModel(base_frame(), parent=view)
    view.set_model(model)
    for mode in ("bar", "heat"):
        delegate = delegates.ValueColorDelegate(mode=mode, parent=view)
        assert delegate.get_mode() == mode
        view.setItemDelegate(delegate)
        view.resize(400, 200)
        assert not view.grab().isNull()


def test_dtype_delegate_sets_dtype(qtbot):
    view = widgets.TableView()
    qtbot.addWidget(view)
    model = pandasmodels.PandasColumnListModel(pd.DataFrame({"a": [1, 2]}))
    view.set_model(model)
    delegate = delegates.DTypeDelegate(parent=view)
    view.set_delegate(delegate, column=1)
    index = model.index(0, 1)
    editor = delegate.createEditor(view.viewport(), None, index)
    delegate.setEditorData(editor, index)
    assert editor.currentText() == "int64"
    editor.setCurrentText("float64")
    delegate.setModelData(editor, model, index)
    assert model.df["a"].dtype == np.float64


def test_column_profile_widget(qtbot, frame):
    widget = pandaswidgets.ColumnProfileWidget(frame, "f")
    qtbot.addWidget(widget)
    assert widget.value_labels["Rows"].text() == "4"
    assert widget.value_labels["Nulls"].text() == "1 (25.0%)"
    assert widget.value_labels["Max"].text() == "2"
    top = widget.top_table.model()
    assert top.rowCount() == 3
    assert top.data(top.index(0, 2), Roles.NormalizedRole) == 1.0
    widget.set_column(pandasmodels.DataFrameHolder(frame), 2)
    assert widget.value_labels["Unique"].text() == "2"
    top = widget.top_table.model()
    assert top.data(top.index(0, 0)) == "x"
    assert top.data(top.index(0, 2)) == "50.0%"
    with pytest.raises(KeyError):
        widget.set_column(frame, "missing")
    widget.clear()
    assert widget.value_labels["Rows"].text() == ""


def test_dataframewidget_other_backend(qtbot, frame):
    widget = pandaswidgets.DataFrameWidget(frame)
    qtbot.addWidget(widget)
    columns = widget.column_table.model()
    assert columns.rowCount() == 5
    widget.column_table.setCurrentIndex(columns.index(3, 0))
    assert widget.category_table.model().rowCount() == 2
    assert widget.profile.value_labels["Rows"].text() == "4"
