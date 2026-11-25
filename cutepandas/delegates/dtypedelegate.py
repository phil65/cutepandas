"""`DTypeDelegate`: pick a dtype from a combo box.

Meant for the Type column of a `PandasColumnListModel`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from prettyqt import constants, widgets

from cutepandas.utils import icons


if TYPE_CHECKING:
    from collections.abc import Sequence

    from prettyqt import core


# Default choices: common pandas dtypes, incl. pyarrow-backed ones.
PANDAS_DTYPES: tuple[str, ...] = (
    "int64",
    "Int64",
    "float64",
    "Float64",
    "bool",
    "boolean",
    "str",
    "category",
    "datetime64[ns]",
    "timedelta64[ns]",
    "object",
    "int64[pyarrow]",
    "double[pyarrow]",
    "bool[pyarrow]",
    "string[pyarrow]",
    "timestamp[us][pyarrow]",
)


class DTypeDelegate(widgets.StyledItemDelegate):
    """Edits a dtype with an editable combo box (each entry with its dtype icon).

    The chosen text is written back with `setData(index, text, role)`; the model
    converts the column (`PandasColumnListModel` does `astype`). A failed conversion
    leaves the column unchanged.
    """

    ID = "dtype"

    def __init__(
        self,
        dtypes: Sequence[str] = PANDAS_DTYPES,
        role: constants.ItemDataRole = constants.EDIT_ROLE,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dtypes = list(dtypes)
        self.role = role

    def createEditor(self, parent, option, index: core.ModelIndex):
        combo = widgets.ComboBox(parent)
        combo.set_editable(True)
        for dtype in self.dtypes:
            try:
                icon = icons.icon_for_dtype(dtype)
            except TypeError, ValueError:
                icon = None
            combo.add(dtype, icon=icon)
        combo.activated.connect(lambda _: self.commitData.emit(combo))
        return combo

    def setEditorData(self, editor, index: core.ModelIndex):
        editor.setCurrentText(str(index.data(constants.DISPLAY_ROLE)))

    def setModelData(self, editor, model, index: core.ModelIndex):
        text = editor.currentText().strip()
        if text and text != index.data(constants.DISPLAY_ROLE):
            model.setData(index, text, self.role)
