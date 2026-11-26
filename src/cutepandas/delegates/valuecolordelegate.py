"""`ValueColorDelegate`: paint a cell by its value (data bar or heat map)."""

from typing import TYPE_CHECKING, Literal

from prettyqt import gui, widgets

from cutepandas.pandasmodels.dataframemodel import DataFrameModel


if TYPE_CHECKING:
    from prettyqt import core as qcore


ModeStr = Literal["bar", "heat"]


class ValueColorDelegate(widgets.StyledItemDelegate):
    """Paints a data bar or a heat-map fill behind the text of a cell.

    The value (0..1) is read from `role`; `mode="bar"` fills that share of the cell
    width, `mode="heat"` the whole cell with an opacity that follows the value.

    With `DataFrameModel` the default role (`NormalizedRole`) is computed once per
    cached block (vectorised), so painting costs one lookup per cell. Any other model
    can provide the role, too; cells without a value are painted normally.
    """

    ID = "value_color"

    def __init__(
        self,
        mode: ModeStr = "bar",
        color: str | gui.QColor = "#4c8bf5",
        role: int = DataFrameModel.Roles.NormalizedRole,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._mode: ModeStr = mode
        self._color = gui.QColor(color)
        self._role = role

    def paint(
        self,
        painter: gui.QPainter,
        option: widgets.QStyleOptionViewItem,
        index: qcore.ModelIndex,
    ):
        value = index.data(self._role)
        if isinstance(value, int | float) and 0.0 <= value <= 1.0:
            rect = option.rect
            width = rect.width()
            color = gui.QColor(self._color)
            if self._mode == "bar":
                color.setAlpha(110)
                width = round(width * value)
            else:
                color.setAlpha(round(40 + 180 * value))
            painter.fillRect(rect.x(), rect.y(), width, rect.height(), color)  # ty: ignore[no-matching-overload]
        super().paint(painter, option, index)

    def set_mode(self, mode: ModeStr):
        self._mode = mode

    def get_mode(self) -> ModeStr:
        return self._mode

    def set_color(self, color: str | gui.QColor):
        self._color = gui.QColor(color)

    def get_color(self) -> gui.QColor:
        return self._color
