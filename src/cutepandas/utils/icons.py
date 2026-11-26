"""Icons for dtypes and indexes (any backend).

See https://pictogrammers.com/library/mdi/ for the names.
"""

from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import pandas as pd
from prettyqt import iconprovider
import pyarrow as pa

from cutepandas.constants import iconnames
from cutepandas.frames import Kind, kind_for_arrow_type, kind_for_pandas_dtype


if TYPE_CHECKING:
    from prettyqt import gui


ICON_FOR_KIND: MappingProxyType[Kind, str] = MappingProxyType({
    Kind.INTEGER: iconnames.NUMERIC,
    Kind.FLOAT: iconnames.NUMERIC,
    Kind.DECIMAL: iconnames.NUMERIC,
    Kind.BOOLEAN: iconnames.BOOL,
    Kind.STRING: iconnames.STRING,
    Kind.CATEGORICAL: iconnames.BOOKMARK,
    Kind.DATETIME: iconnames.TIME,
    Kind.DATE: iconnames.DATE,
    Kind.TIME: iconnames.TIME,
    Kind.DURATION: iconnames.TIMELINE,
    Kind.BINARY: iconnames.BINARY,
    Kind.NESTED: iconnames.NESTED,
    Kind.NULL: iconnames.NAN,
    Kind.OBJECT: iconnames.OBJECT,
})


def kind_of_dtype(dtype: Any) -> Kind:
    """`Kind` of a pandas / NumPy, pyarrow or polars dtype."""
    if isinstance(dtype, pa.DataType):
        return kind_for_arrow_type(dtype)
    if type(dtype).__module__.startswith("polars"):
        from cutepandas.frames.polarssource import kind_for_polars_dtype

        return kind_for_polars_dtype(dtype)
    return kind_for_pandas_dtype(dtype)


# Share of the palette's text colour in the icon tint (rest: window background).
ICON_FOREGROUND_RATIO = 0.68


def default_icon_color(palette: gui.QPalette | None = None) -> str | None:
    """Muted foreground hex colour for icons, from `palette` or the app palette.

    None without a QGuiApplication (the icon provider's default colour then).
    """
    from prettyqt import gui

    if palette is None:
        if gui.GuiApplication.instance() is None:
            return None
        palette = gui.GuiApplication.palette()
    fg = palette.color(gui.Palette.ColorRole.WindowText)
    bg = palette.color(gui.Palette.ColorRole.Window)
    ratio = ICON_FOREGROUND_RATIO
    r = round(fg.red() * ratio + bg.red() * (1 - ratio))
    g = round(fg.green() * ratio + bg.green() * (1 - ratio))
    b = round(fg.blue() * ratio + bg.blue() * (1 - ratio))
    return f"#{r:02x}{g:02x}{b:02x}"


def icon_for_kind(kind: Kind, color: str | None = None) -> gui.QIcon:
    """Icon for a `Kind`, tinted `color` (default: `default_icon_color()`).

    Not cached here: views asking per paint use a `KindIconCache`.
    """
    resolved = default_icon_color() if color is None else color
    return iconprovider.get_icon(ICON_FOR_KIND[kind], color=resolved, as_qicon=True)


class KindIconCache:
    """Per-owner cache of `Kind` icons for models / delegates that paint them often.

    Without an explicit colour the icons follow the application palette: the cache
    is dropped when the palette changes (`QPalette.cacheKey()`). Explicit colours are
    cached per (kind, colour); `clear()` drops everything (e.g. on a restyle).
    """

    def __init__(self):
        self._icons: dict[tuple[Kind, str | None], gui.QIcon] = {}
        self._palette_key: int | None = None

    def icon(self, kind: Kind, color: str | None = None) -> gui.QIcon:
        if color is None:
            from prettyqt import gui

            key = (
                gui.GuiApplication.palette().cacheKey()
                if gui.GuiApplication.instance() is not None
                else None
            )
            if key != self._palette_key:
                self._icons.clear()
                self._palette_key = key
        elif len(self._icons) > len(ICON_FOR_KIND) * 4:  # many colours: start over
            self._icons.clear()
        if (icon := self._icons.get((kind, color))) is None:
            icon = self._icons[kind, color] = icon_for_kind(kind, color=color)
        return icon

    def clear(self):
        self._icons.clear()
        self._palette_key = None


def icon_for_dtype(dtype: Any, color: str | None = None) -> gui.QIcon:
    return icon_for_kind(kind_of_dtype(dtype), color=color)


def icon_for_ds(ds: pd.DataFrame) -> gui.QIcon:
    if isinstance(ds.index, pd.MultiIndex):
        icon = iconnames.MULTIINDEX
    else:
        icon = iconnames.SINGLEINDEX
    return iconprovider.get_icon(icon, as_qicon=True)


def icon_for_index(index: pd.Index) -> gui.QIcon:
    match index:
        case pd.RangeIndex():
            icon = iconnames.SORT
        case pd.DatetimeIndex():
            icon = iconnames.TIME
        case pd.CategoricalIndex():
            icon = iconnames.CATEGORY
        case pd.TimedeltaIndex():
            icon = iconnames.TIMELINE
        case pd.PeriodIndex():
            icon = iconnames.TIMELAPSE
        case _:
            icon = ICON_FOR_KIND[kind_for_pandas_dtype(index.dtype)]
    return iconprovider.get_icon(icon, as_qicon=True)
