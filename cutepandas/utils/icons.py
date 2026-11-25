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


def icon_for_kind(kind: Kind) -> gui.QIcon:
    return iconprovider.get_icon(ICON_FOR_KIND[kind], as_qicon=True)


def icon_for_dtype(dtype: Any) -> gui.QIcon:
    return icon_for_kind(kind_of_dtype(dtype))


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
