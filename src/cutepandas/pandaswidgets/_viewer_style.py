"""Shared palette-aware visual styling and delegates for DataFrame viewers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from prettyqt import constants, core, gui, widgets

from cutepandas.pandasmodels.dataframemodel import DataFrameModel
from cutepandas.utils import icons


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


# Horizontal text padding inside a data cell (each side).
CELL_HPAD = 6
# Horizontal text padding inside a header cell (each side).
HEADER_HPAD = 8
# Room for the sort arrow at the right edge of the sorted leaf header.
SORT_ARROW_SPACE = 16
# Size of the dtype icon painted by `HeaderCellDelegate`.
HEADER_ICON_SIZE = 13
# Gap between the dtype icon and the header text.
HEADER_ICON_GAP = 5
# Width a dtype icon adds to a fitted header column (icon, gap, 2 px slack).
HEADER_ICON_SPACE = HEADER_ICON_SIZE + HEADER_ICON_GAP + 2
# Width a dtype icon adds to a native `QHeaderView` section (16 px icon + margins).
NATIVE_HEADER_ICON_SPACE = 22
# Extra width for fitted columns, so measured text is not elided by rounding.
FIT_SLACK = 12
# Narrowest fitted data column.
MIN_COLUMN_WIDTH = 60
# Narrowest fitted index level column.
MIN_INDEX_COLUMN_WIDTH = 36
# Widest fitted column (data and index); wider text is elided.
MAX_COLUMN_WIDTH = 400
# Columns fitted when a frame is set (the others when they scroll into view).
AUTOSIZE_COLUMNS = 50
# Rows measured to fit a column (bounded: O(1) in the frame length).
SAMPLE_ROWS = 75
# Fonts kept per (base font, weight, italic) before the cache starts over.
_MAX_CACHED_FONTS = 32


def sample_rows(row_count: int) -> list[int]:
    """Rows measured to fit a column: the first `SAMPLE_ROWS` and the last one."""
    rows = list(range(min(row_count, SAMPLE_ROWS)))
    if row_count > SAMPLE_ROWS:
        rows.append(row_count - 1)  # widest trailing row number
    return rows


def fitted_column_width(
    header_texts: Iterable[str],
    header_font: gui.QFont,
    cell_texts: Iterable[str],
    cell_metrics: gui.QFontMetrics,
    *,
    icon_space: int = 0,
) -> int:
    """Width of a data column fitting its (bold) header labels and sampled cells."""
    font = gui.Font(header_font)
    font.set_weight("demi_bold")
    header_metrics = gui.FontMetrics(font)
    header_extra = HEADER_HPAD * 2 + SORT_ARROW_SPACE + icon_space
    width = MIN_COLUMN_WIDTH
    for text in header_texts:
        width = max(width, header_metrics.horizontalAdvance(text) + header_extra)
    cell_extra = CELL_HPAD * 2 + FIT_SLACK
    for text in cell_texts:
        width = max(width, cell_metrics.horizontalAdvance(text) + cell_extra)
    return min(width, MAX_COLUMN_WIDTH)


def fitted_index_width(texts: Iterable[str], metrics: gui.QFontMetrics) -> int:
    """Width of an index level column fitting `texts` (level name, sampled labels)."""
    extra = HEADER_HPAD * 2 + FIT_SLACK
    width = MIN_INDEX_COLUMN_WIDTH
    for text in texts:
        width = max(width, metrics.horizontalAdvance(text) + extra)
    return min(width, MAX_COLUMN_WIDTH)


def blend(c1: gui.QColor, c2: gui.QColor, ratio: float) -> gui.QColor:
    """Blend `ratio` of `c1` with `1 - ratio` of `c2`."""
    inv = 1.0 - ratio
    return gui.QColor(
        round(c1.red() * ratio + c2.red() * inv),
        round(c1.green() * ratio + c2.green() * inv),
        round(c1.blue() * ratio + c2.blue() * inv),
    )


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


# Plain ints: Python enum flag operations cost more than the rest of a cell's paint.
_SELECTED = widgets.QStyle.StateFlag.State_Selected.value
_HAS_FOCUS = widgets.QStyle.StateFlag.State_HasFocus.value
_ALTERNATE = widgets.QStyleOptionViewItem.ViewItemFeature.Alternate.value
_ALIGN_CELL = (constants.ALIGN_LEFT | constants.ALIGN_V_CENTER).value
_ELIDE_RIGHT = constants.TextElideMode.ElideRight


def _align_value(align: Any, default: int) -> int:
    if align is None:
        return default
    return align if isinstance(align, int) else align.value


class _PaletteColorsMixin:
    """Colours set by the viewer (`set_colors`), else derived from the palette."""

    _colors: ViewerColors | None = None
    _fallback: tuple[int, ViewerColors] | None = None

    def set_colors(self, colors: ViewerColors):
        self._colors = colors

    def colors_for(self, palette: gui.QPalette) -> ViewerColors:
        if self._colors is not None:
            return self._colors
        key = palette.cacheKey()
        if self._fallback is None or self._fallback[0] != key:
            self._fallback = (key, ViewerColors.from_palette(palette))
        return self._fallback[1]


class DataCellDelegate(_PaletteColorsMixin, widgets.StyledItemDelegate):
    """Clean cell renderer: horizontal padding, muted null marker, no dotted focus."""

    def __init__(
        self,
        parent: core.QObject | None = None,
        *,
        null_placeholder: str = "—",
    ):
        super().__init__(parent=parent)
        self._null_placeholder = null_placeholder

    def paint(
        self,
        painter: Any,
        option: Any,
        index: core.ModelIndex,
    ):
        colors = self.colors_for(option.palette)
        rect = option.rect
        state = option.state.value
        is_selected = state & _SELECTED
        has_focus = state & _HAS_FOCUS
        # A plain DataFrameModel has no colour roles unless it colours values: skip
        # those queries (each data() call is a round trip into Python).
        model = index.model()
        style_roles = not isinstance(model, DataFrameModel) or model.get_color_values()

        painter.save()
        # Background
        if is_selected:
            bg = colors.selection_bg
        elif (
            style_roles and (role_bg := index.data(constants.BACKGROUND_ROLE)) is not None
        ):
            bg = role_bg
        elif option.features.value & _ALTERNATE:
            bg = colors.alt_row
        else:
            bg = colors.base
        x, y, w, h = rect.getRect()
        painter.fillRect(x, y, w, h, bg)
        # grid hairlines (right, bottom); the view's own grid is off
        painter.fillRect(x + w - 1, y, 1, h, colors.grid)
        painter.fillRect(x, y + h - 1, w, 1, colors.grid)

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
                    elif (
                        style_roles
                        and (role_fg := index.data(constants.FOREGROUND_ROLE)) is not None
                    ):
                        fg = role_fg
                    else:
                        fg = colors.text
                painter.setPen(fg)
                align = _align_value(index.data(constants.ALIGNMENT_ROLE), _ALIGN_CELL)
                elided = painter.fontMetrics().elidedText(
                    text, _ELIDE_RIGHT, text_rect.width()
                )
                painter.drawText(text_rect, align, elided)

        # Crisp accent inset border on the focused cell
        if has_focus:
            painter.setPen(gui.Pen(colors.accent, 1))
            painter.setBrush(constants.BrushStyle.NoBrush)
            painter.drawRect(rect.adjusted(0, 0, -1, -1))
        painter.restore()


def _draw_grid_edges(painter: Any, rect: Any, right: gui.QColor, bottom: gui.QColor):
    """Hairlines on the right and bottom edge of a cell (the views hide Qt's grid)."""
    painter.fillRect(rect.right(), rect.top(), 1, rect.height(), right)
    painter.fillRect(rect.left(), rect.bottom(), rect.width(), 1, bottom)


class _FontCache:
    """Weighted / italic variants of the view font, built once per base font."""

    def __init__(self):
        self._fonts: dict[tuple[str, str, bool], gui.QFont] = {}

    def get(self, base: gui.QFont, weight: str | None, italic: bool = False) -> Any:
        key = (base.key(), weight or "", italic)
        if (font := self._fonts.get(key)) is None:
            # the base font changed often: start over
            if len(self._fonts) > _MAX_CACHED_FONTS:
                self._fonts.clear()
            font = gui.Font(base)
            if weight is not None:
                font.set_weight(weight)  # ty: ignore[invalid-argument-type]
            font.setItalic(italic)
            self._fonts[key] = font
        return font


_ALIGN_HEADER_H = constants.ALIGN_CENTER.value
_ALIGN_HEADER_V = (constants.ALIGN_LEFT | constants.ALIGN_V_CENTER).value


class HeaderCellDelegate(_PaletteColorsMixin, widgets.StyledItemDelegate):
    """Renderer for the multi-level column and row header tables.

    Outer MultiIndex levels get a slightly stronger surface, the leaf level a divider
    edge towards the data table, the sorted leaf column a sort arrow and (optionally)
    leaf column headers a palette-tinted dtype icon.
    """

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
        self._horizontal = orientation == constants.HORIZONTAL
        self._sort_state_getter = sort_state_getter
        self._kind_getter = kind_getter
        self._show_dtype_icons = show_dtype_icons
        self._kind_icons = icons.KindIconCache()
        self._fonts = _FontCache()

    def set_colors(self, colors: ViewerColors):
        super().set_colors(colors)
        self._kind_icons.clear()

    def paint(
        self,
        painter: Any,
        option: Any,
        index: core.ModelIndex,
    ):
        colors = self.colors_for(option.palette)
        rect = option.rect
        is_selected = option.state.value & _SELECTED
        is_horizontal = self._horizontal

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
        painter.fillRect(rect, bg)
        # Grid hairlines; the leaf level's edge towards the data table is a divider.
        if is_horizontal:
            bottom = colors.divider if is_leaf else colors.grid
            _draw_grid_edges(painter, rect, colors.grid, bottom)
        else:
            right = colors.divider if is_leaf else colors.grid
            _draw_grid_edges(painter, rect, right, colors.grid)

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
                icon = self._kind_icons.icon(kind, color=colors.muted_text.name())
                size = HEADER_ICON_SIZE
                iy = rect.top() + (rect.height() - size) // 2
                icon.paint(painter, rect.left() + left_pad, iy, size, size)
                left_pad += size + HEADER_ICON_GAP

        raw_text = index.data(constants.DISPLAY_ROLE)
        text = "" if raw_text is None else str(raw_text)
        text_rect = rect.adjusted(left_pad, 0, -right_pad, 0)
        if text and text_rect.width() > 0:
            if is_horizontal:
                weight = "demi_bold" if is_leaf else "medium"
            else:
                weight = None if is_leaf else "medium"
            painter.setFont(self._fonts.get(option.font, weight))
            painter.setPen(colors.header_fg)
            align = _ALIGN_HEADER_H if is_horizontal else _ALIGN_HEADER_V
            elided = painter.fontMetrics().elidedText(
                text, _ELIDE_RIGHT, text_rect.width()
            )
            painter.drawText(text_rect, align, elided)

        if sort_order is not None:
            _draw_sort_arrow(
                painter,
                rect.right() - 10,
                rect.center().y(),
                ascending=sort_order != constants.DESCENDING,
                color=colors.accent,
            )

        painter.restore()


class CornerCellDelegate(_PaletteColorsMixin, widgets.StyledItemDelegate):
    """Renderer for the top-left level-name corner table of the grid viewer."""

    def __init__(self, parent: core.QObject | None = None):
        super().__init__(parent=parent)
        self._fonts = _FontCache()

    def paint(
        self,
        painter: Any,
        option: Any,
        index: core.ModelIndex,
    ):
        colors = self.colors_for(option.palette)
        rect = option.rect

        painter.save()
        painter.fillRect(rect, colors.header_outer_bg)
        model = index.model()
        last_row = model is None or index.row() == model.rowCount() - 1
        last_col = model is None or index.column() == model.columnCount() - 1
        _draw_grid_edges(
            painter,
            rect,
            colors.divider if last_col else colors.grid,
            colors.divider if last_row else colors.grid,
        )
        raw_text = index.data(constants.DISPLAY_ROLE)
        text = "" if raw_text is None else str(raw_text)
        text_rect = rect.adjusted(HEADER_HPAD, 0, -HEADER_HPAD, 0)
        if text and text_rect.width() > 0:
            painter.setFont(self._fonts.get(option.font, "medium", italic=True))
            painter.setPen(colors.muted_text)
            align = _align_value(index.data(constants.ALIGNMENT_ROLE), _ALIGN_HEADER_H)
            elided = painter.fontMetrics().elidedText(
                text, _ELIDE_RIGHT, text_rect.width()
            )
            painter.drawText(text_rect, align, elided)
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


def inherits_style_sheet(widget: Any) -> bool:
    """True if the application or an ancestor of `widget` sets a style sheet."""
    app: Any = widgets.QApplication.instance()
    if app is not None and app.styleSheet():
        return True
    parent = widget.parentWidget()
    while parent is not None:
        if parent.styleSheet():
            return True
        parent = parent.parentWidget()
    return False


def style_table(view: Any, base: gui.QColor, alt: gui.QColor | None = None):
    """Frameless table without Qt's grid (the delegates paint hairlines).

    Colours go through the palette, not a style sheet: a style sheet would route all
    painting of the view through QStyleSheetStyle. Only when an application / parent
    style sheet (a theme) already does that, the view gets a small sheet that resets
    the theme's table border and padding (the composite viewers align their tables
    edge to edge) and keeps the palette colours.
    """
    view.setFrameShape(widgets.QFrame.Shape.NoFrame)
    view.setShowGrid(False)
    palette: Any = gui.QPalette(view.palette())
    role = gui.Palette.ColorRole
    palette.setColor(role.Base, base)
    palette.setColor(role.Window, base)
    palette.setColor(role.AlternateBase, alt or base)
    view.setPalette(palette)
    sheet = ""
    if inherits_style_sheet(view):
        sheet = (
            "QTableView {"
            " border: none; padding: 0px; margin: 0px; outline: 0;"
            f" background-color: {base.name()};"
            f" alternate-background-color: {(alt or base).name()};"
            "}"
        )
    if view.styleSheet() != sheet:
        view.setStyleSheet(sheet)


def level_name_header_style(colors: ViewerColors) -> str:
    """Style sheet for the native level-name `QHeaderView`s of the composite viewer.

    Set on the header widget itself (not on the table), so only the header goes
    through QStyleSheetStyle.
    """
    return (
        "QHeaderView::section {"
        f" background-color: {colors.header_outer_bg.name()};"
        f" color: {colors.muted_text.name()};"
        " font-style: italic;"
        " padding: 2px 6px;"
        " border: none;"
        f" border-right: 1px solid {colors.grid.name()};"
        f" border-bottom: 1px solid {colors.grid.name()};"
        "}"
    )


def native_header_style(colors: ViewerColors) -> str:
    """Style sheet for the native `QHeaderView`s of the single-table viewer."""
    return (
        "QHeaderView::section:horizontal {"
        f" background-color: {colors.header_bg.name()};"
        f" color: {colors.header_fg.name()};"
        " font-weight: 600;"
        " padding: 4px 8px;"
        " border: none;"
        f" border-right: 1px solid {colors.grid.name()};"
        f" border-bottom: 1px solid {colors.divider.name()};"
        "}"
        "QHeaderView::section:horizontal:hover {"
        f" background-color: {colors.header_outer_bg.name()};"
        "}"
        "QHeaderView::section:horizontal:checked {"
        f" background-color: {colors.header_selected_bg.name()};"
        "}"
        "QHeaderView::section:vertical {"
        f" background-color: {colors.header_bg.name()};"
        f" color: {colors.header_fg.name()};"
        " padding: 2px 8px;"
        " border: none;"
        f" border-bottom: 1px solid {colors.grid.name()};"
        f" border-right: 1px solid {colors.divider.name()};"
        "}"
        "QHeaderView::section:vertical:checked {"
        f" background-color: {colors.header_selected_bg.name()};"
        "}"
    )


def restyle_view(view: Any, colors: ViewerColors):
    """Hand `colors` to the view's delegate (if it is one of ours) and repaint."""
    delegate = view.itemDelegate()
    if isinstance(delegate, _PaletteColorsMixin):
        delegate.set_colors(colors)
    view.viewport().update()


def paint_frame(widget: Any, colors: ViewerColors):
    """Paint the composite viewers' surface and 1 px outer border (in paintEvent)."""
    painter: Any = gui.QPainter(widget)
    rect = widget.rect()
    painter.fillRect(rect, colors.header_outer_bg)
    painter.setPen(colors.divider)
    painter.drawRect(rect.adjusted(0, 0, -1, -1))
    painter.end()
