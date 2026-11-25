"""Shared palette-aware visual styling and delegates for DataFrame viewers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from prettyqt import constants, core, gui, widgets

from cutepandas.pandasmodels.dataframemodel import DataFrameModel
from cutepandas.utils import icons


if TYPE_CHECKING:
    from collections.abc import Callable


CELL_HPAD = 6
HEADER_HPAD = 8
SORT_ARROW_SPACE = 16


def blend(c1: gui.QColor, c2: gui.QColor, ratio: float) -> gui.QColor:
    """Blend `ratio` of `c1` with `1 - ratio` of `c2`."""
    inv = 1.0 - ratio
    return gui.QColor(
        round(c1.red() * ratio + c2.red() * inv),
        round(c1.green() * ratio + c2.green() * inv),
        round(c1.blue() * ratio + c2.blue() * inv),
    )


def _hex(color: Any) -> str:
    return color.name()


@dataclass(frozen=True)
class ViewerColors:
    """Cohesive color tokens derived from a QPalette (works in light & dark themes)."""

    base: gui.QColor
    alt_row: gui.QColor
    text: gui.QColor
    muted_text: gui.QColor
    header_bg: gui.QColor
    header_outer_bg: gui.QColor
    header_fg: gui.QColor
    header_selected_bg: gui.QColor
    selection_bg: gui.QColor
    selection_fg: gui.QColor
    grid: gui.QColor
    divider: gui.QColor
    accent: gui.QColor

    @classmethod
    def from_palette(cls, palette: gui.QPalette) -> ViewerColors:
        role = gui.Palette.ColorRole
        base = gui.QColor(palette.color(role.Base))
        text = gui.QColor(palette.color(role.Text))
        window_text = gui.QColor(palette.color(role.WindowText))
        highlight = gui.QColor(palette.color(role.Highlight))
        header_bg = blend(window_text, base, 0.055)
        return cls(
            base=base,
            alt_row=blend(window_text, base, 0.03),
            text=text,
            muted_text=blend(text, base, 0.42),
            header_bg=header_bg,
            header_outer_bg=blend(window_text, base, 0.09),
            header_fg=window_text,
            header_selected_bg=blend(highlight, header_bg, 0.18),
            selection_bg=blend(highlight, base, 0.20),
            selection_fg=text,
            grid=blend(window_text, base, 0.13),
            divider=blend(window_text, base, 0.28),
            accent=highlight,
        )


class DataCellDelegate(widgets.StyledItemDelegate):
    """Clean cell renderer: horizontal padding, muted null marker, no dotted focus."""

    def __init__(
        self,
        parent: core.QObject | None = None,
        *,
        null_placeholder: str = "—",
    ):
        super().__init__(parent=parent)
        self._null_placeholder = null_placeholder
        self._colors: ViewerColors | None = None

    def set_colors(self, colors: ViewerColors):
        self._colors = colors

    def paint(
        self,
        painter: Any,
        option: Any,
        index: core.ModelIndex,
    ):
        colors = self._colors or ViewerColors.from_palette(option.palette)
        rect = option.rect
        state = option.state
        is_selected = bool(state & widgets.QStyle.StateFlag.State_Selected)
        has_focus = bool(state & widgets.QStyle.StateFlag.State_HasFocus)

        painter.save()
        # Background
        if is_selected:
            bg = colors.selection_bg
        elif (role_bg := index.data(constants.BACKGROUND_ROLE)) is not None:
            bg = role_bg
        elif option.features & widgets.QStyleOptionViewItem.ViewItemFeature.Alternate:
            bg = colors.alt_row
        else:
            bg = colors.base
        painter.fillRect(rect.x(), rect.y(), rect.width(), rect.height(), bg)

        # Text (only query IsNullRole when display text is empty)
        raw_text = index.data(constants.DISPLAY_ROLE)
        text = "" if raw_text is None else str(raw_text)
        text_rect = rect.adjusted(CELL_HPAD, 0, -CELL_HPAD, 0)
        if text_rect.width() > 0:
            is_null_placeholder = False
            if (
                not text
                and self._null_placeholder
                and index.data(DataFrameModel.Roles.IsNullRole)
            ):
                text = self._null_placeholder
                is_null_placeholder = True

            if text:
                if is_null_placeholder:
                    font: Any = gui.QFont(option.font)
                    font.setItalic(True)
                    painter.setFont(font)
                    fg = colors.muted_text
                else:
                    painter.setFont(option.font)
                    if is_selected:
                        fg = colors.selection_fg
                    elif (role_fg := index.data(constants.FOREGROUND_ROLE)) is not None:
                        fg = role_fg
                    else:
                        fg = colors.text
                painter.setPen(fg)
                align = index.data(constants.ALIGNMENT_ROLE) or (
                    constants.ALIGN_LEFT | constants.ALIGN_V_CENTER
                )
                elided = painter.fontMetrics().elidedText(
                    text, constants.TextElideMode.ElideRight, text_rect.width()
                )
                painter.drawText(text_rect, int(align), elided)

        # Crisp accent inset border on the focused cell
        if has_focus:
            painter.setPen(gui.Pen(colors.accent, 1))
            painter.setBrush(constants.BrushStyle.NoBrush)
            painter.drawRect(rect.adjusted(0, 0, -1, -1))
        painter.restore()


class HeaderCellDelegate(widgets.StyledItemDelegate):
    """Polished renderer for multi-level column and row header tables."""

    def __init__(
        self,
        orientation: constants.Orientation,
        parent: core.QObject | None = None,
        *,
        sort_state_getter: Callable[[], tuple[int, constants.SortOrder] | None]
        | None = None,
        kind_getter: Callable[[int], Any] | None = None,
        show_dtype_icons: bool = False,
    ):
        super().__init__(parent=parent)
        self._orientation = orientation
        self._sort_state_getter = sort_state_getter
        self._kind_getter = kind_getter
        self._show_dtype_icons = show_dtype_icons
        self._colors: ViewerColors | None = None

    def set_colors(self, colors: ViewerColors):
        self._colors = colors

    def paint(
        self,
        painter: Any,
        option: Any,
        index: core.ModelIndex,
    ):
        colors = self._colors or ViewerColors.from_palette(option.palette)
        rect = option.rect
        is_selected = bool(option.state & widgets.QStyle.StateFlag.State_Selected)
        is_horizontal = self._orientation == constants.HORIZONTAL

        model = index.model()
        is_leaf = True
        if model is not None:
            if is_horizontal:
                is_leaf = index.row() == model.rowCount() - 1
            else:
                is_leaf = index.column() == model.columnCount() - 1

        painter.save()
        # Surface fill: slightly richer tone for outer MultiIndex levels
        if is_selected:
            bg = colors.header_selected_bg
        elif not is_leaf:
            bg = colors.header_outer_bg
        else:
            bg = colors.header_bg
        painter.fillRect(rect.x(), rect.y(), rect.width(), rect.height(), bg)

        # Accent edge on selected leaf header touching the data table
        if is_selected and is_leaf:
            if is_horizontal:
                painter.fillRect(
                    rect.left(), rect.bottom() - 1, rect.width(), 2, colors.accent
                )
            else:
                painter.fillRect(
                    rect.right() - 1, rect.top(), 2, rect.height(), colors.accent
                )

        # Sort indicator on leaf column header
        sort_order: constants.SortOrder | None = None
        if is_horizontal and is_leaf and self._sort_state_getter is not None:
            sort_state = self._sort_state_getter()
            if sort_state is not None and sort_state[0] == index.column():
                sort_order = sort_state[1]

        left_pad = HEADER_HPAD
        right_pad = HEADER_HPAD + (SORT_ARROW_SPACE if sort_order is not None else 0)

        # Optional dtype icon on leaf horizontal headers
        if (
            is_horizontal
            and is_leaf
            and self._show_dtype_icons
            and self._kind_getter is not None
        ):
            kind = self._kind_getter(index.column())
            if kind is not None:
                icon = icons.icon_for_kind(kind, color=_hex(colors.muted_text))
                icon_size = 13
                iy = rect.top() + (rect.height() - icon_size) // 2
                icon.paint(painter, rect.left() + left_pad, iy, icon_size, icon_size)
                left_pad += icon_size + 5

        raw_text = index.data(constants.DISPLAY_ROLE)
        text = "" if raw_text is None else str(raw_text)
        text_rect = rect.adjusted(left_pad, 0, -right_pad, 0)
        if text and text_rect.width() > 0:
            font: Any = gui.Font(option.font)
            if is_horizontal:
                font.set_weight("demi_bold" if is_leaf else "medium")
            elif not is_leaf:
                font.set_weight("medium")
            painter.setFont(font)
            painter.setPen(colors.header_fg)
            align = (
                constants.ALIGN_CENTER
                if is_horizontal
                else (constants.ALIGN_LEFT | constants.ALIGN_V_CENTER)
            )
            elided = painter.fontMetrics().elidedText(
                text, constants.TextElideMode.ElideRight, text_rect.width()
            )
            painter.drawText(text_rect, int(align), elided)

        if sort_order is not None:
            _draw_sort_arrow(
                painter,
                rect.right() - 10,
                rect.center().y(),
                ascending=sort_order != constants.DESCENDING,
                color=colors.accent,
            )

        painter.restore()


class CornerCellDelegate(widgets.StyledItemDelegate):
    """Renderer for the top-left level-name corner table in the 4-table viewer."""

    def __init__(self, parent: core.QObject | None = None):
        super().__init__(parent=parent)
        self._colors: ViewerColors | None = None

    def set_colors(self, colors: ViewerColors):
        self._colors = colors

    def paint(
        self,
        painter: Any,
        option: Any,
        index: core.ModelIndex,
    ):
        colors = self._colors or ViewerColors.from_palette(option.palette)
        rect = option.rect

        painter.save()
        painter.fillRect(
            rect.x(), rect.y(), rect.width(), rect.height(), colors.header_outer_bg
        )
        raw_text = index.data(constants.DISPLAY_ROLE)
        text = "" if raw_text is None else str(raw_text)
        text_rect = rect.adjusted(HEADER_HPAD, 0, -HEADER_HPAD, 0)
        if text and text_rect.width() > 0:
            font: Any = gui.Font(option.font)
            font.setItalic(True)
            font.set_weight("medium")
            painter.setFont(font)
            painter.setPen(colors.muted_text)
            align = index.data(constants.ALIGNMENT_ROLE) or constants.ALIGN_CENTER
            elided = painter.fontMetrics().elidedText(
                text, constants.TextElideMode.ElideRight, text_rect.width()
            )
            painter.drawText(text_rect, int(align), elided)
        painter.restore()


def _draw_sort_arrow(
    painter: Any,
    cx: int,
    cy: int,
    *,
    ascending: bool,
    color: gui.QColor,
):
    painter.save()
    painter.setRenderHint(gui.QPainter.RenderHint.Antialiasing, True)
    painter.setPen(constants.PenStyle.NoPen)
    painter.setBrush(color)
    if ascending:
        pts = [
            core.QPointF(cx, cy - 3.0),
            core.QPointF(cx - 4.0, cy + 2.5),
            core.QPointF(cx + 4.0, cy + 2.5),
        ]
    else:
        pts = [
            core.QPointF(cx - 4.0, cy - 2.5),
            core.QPointF(cx + 4.0, cy - 2.5),
            core.QPointF(cx, cy + 3.0),
        ]
    painter.drawPolygon(gui.PolygonF(pts))
    painter.restore()


def apply_composite_frame_style(widget: Any, colors: ViewerColors):
    """Unified outer border & clean surface for multi-table composite viewers."""
    widget.setAttribute(constants.WidgetAttribute.WA_StyledBackground, True)
    widget.setStyleSheet(
        f"QWidget#{widget.objectName()} {{"
        f" background-color: {_hex(colors.header_outer_bg)};"
        f" border: 1px solid {_hex(colors.divider)};"
        "}"
    )


def apply_data_table_style(view: Any, colors: ViewerColors):
    """Frameless, low-contrast hairline grid style for the inner data table."""
    view.setFrameShape(widgets.QFrame.Shape.NoFrame)
    view.setStyleSheet(
        "QTableView {"
        f" background-color: {_hex(colors.base)};"
        f" alternate-background-color: {_hex(colors.alt_row)};"
        f" gridline-color: {_hex(colors.grid)};"
        " border: none;"
        " outline: 0;"
        "}"
    )


def apply_header_table_style(
    view: Any,
    colors: ViewerColors,
    orientation: constants.Orientation,
):
    """Frameless header table style with a crisp divider edge touching the data table."""
    view.setFrameShape(widgets.QFrame.Shape.NoFrame)
    edge = (
        f"border-bottom: 1px solid {_hex(colors.divider)};"
        if orientation == constants.HORIZONTAL
        else f"border-right: 1px solid {_hex(colors.divider)};"
    )
    view.setStyleSheet(
        "QTableView {"
        f" background-color: {_hex(colors.header_bg)};"
        f" gridline-color: {_hex(colors.grid)};"
        " border: none;"
        f" {edge}"
        " outline: 0;"
        "}"
        "QHeaderView::section {"
        f" background-color: {_hex(colors.header_outer_bg)};"
        f" color: {_hex(colors.muted_text)};"
        " font-style: italic;"
        " padding: 2px 6px;"
        " border: none;"
        f" border-right: 1px solid {_hex(colors.grid)};"
        f" border-bottom: 1px solid {_hex(colors.grid)};"
        "}"
    )


def apply_corner_table_style(view: Any, colors: ViewerColors):
    """Frameless corner table style with crisp bottom & right divider edges."""
    view.setFrameShape(widgets.QFrame.Shape.NoFrame)
    view.setStyleSheet(
        "QTableView {"
        f" background-color: {_hex(colors.header_outer_bg)};"
        f" gridline-color: {_hex(colors.grid)};"
        " border: none;"
        f" border-right: 1px solid {_hex(colors.divider)};"
        f" border-bottom: 1px solid {_hex(colors.divider)};"
        " outline: 0;"
        "}"
    )


def apply_native_table_style(view: widgets.QTableView, colors: ViewerColors):
    """Clean, modern stylesheet for the single-table viewer."""
    view.setStyleSheet(
        "QTableView {"
        f" background-color: {_hex(colors.base)};"
        f" alternate-background-color: {_hex(colors.alt_row)};"
        f" gridline-color: {_hex(colors.grid)};"
        f" border: 1px solid {_hex(colors.divider)};"
        " outline: 0;"
        "}"
        "QHeaderView::section:horizontal {"
        f" background-color: {_hex(colors.header_bg)};"
        f" color: {_hex(colors.header_fg)};"
        " font-weight: 600;"
        " padding: 4px 8px;"
        " border: none;"
        f" border-right: 1px solid {_hex(colors.grid)};"
        f" border-bottom: 1px solid {_hex(colors.divider)};"
        "}"
        "QHeaderView::section:horizontal:hover {"
        f" background-color: {_hex(colors.header_outer_bg)};"
        "}"
        "QHeaderView::section:horizontal:checked {"
        f" background-color: {_hex(colors.header_selected_bg)};"
        "}"
        "QHeaderView::section:vertical {"
        f" background-color: {_hex(colors.header_bg)};"
        f" color: {_hex(colors.header_fg)};"
        " padding: 2px 8px;"
        " border: none;"
        f" border-bottom: 1px solid {_hex(colors.grid)};"
        f" border-right: 1px solid {_hex(colors.divider)};"
        "}"
        "QHeaderView::section:vertical:checked {"
        f" background-color: {_hex(colors.header_selected_bg)};"
        "}"
        "QTableCornerButton::section {"
        f" background-color: {_hex(colors.header_outer_bg)};"
        " border: none;"
        f" border-right: 1px solid {_hex(colors.divider)};"
        f" border-bottom: 1px solid {_hex(colors.divider)};"
        "}"
    )
