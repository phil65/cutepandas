"""Icons for dtypes and indexes (any backend).

See https://pictogrammers.com/library/mdi/ for the names.
"""

from __future__ import annotations

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
        from cutepandas.frames.polarssource import kind_for_polars_dtype  # noqa: PLC0415

        return kind_for_polars_dtype(dtype)
    return kind_for_pandas_dtype(dtype)


def _default_icon_color() -> str | None:
    """Return a muted foreground hex color matching the active Qt palette."""
    from prettyqt import gui  # noqa: PLC0415

    app = gui.GuiApplication.instance()
    if app is None:
        return None
    pal = gui.GuiApplication.palette()
    fg = pal.color(gui.Palette.ColorRole.WindowText)
    bg = pal.color(gui.Palette.ColorRole.Window)
    # 68% foreground blended toward window background for crisp but unobtrusive icons
    r = round(fg.red() * 0.68 + bg.red() * 0.32)
    g = round(fg.green() * 0.68 + bg.green() * 0.32)
    b = round(fg.blue() * 0.68 + bg.blue() * 0.32)
    return f"#{r:02x}{g:02x}{b:02x}"


def icon_for_kind(kind: Kind, color: str | None = None) -> gui.QIcon:
    resolved = _default_icon_color() if color is None else color
    return iconprovider.get_icon(ICON_FOR_KIND[kind], color=resolved, as_qicon=True)


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
