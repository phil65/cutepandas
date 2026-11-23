"""A shared, observable container for one DataFrame."""

from __future__ import annotations

from typing import Any

import pandas as pd
from prettyqt import core


class DataFrameHolder(core.Object):
    """Holds the current DataFrame for several models / widgets.

    pandas operations like `rename` or `astype` return a new frame instead of
    changing the old one. Models that share a holder always see the current frame:
    replacing it emits `about_to_change` and `changed`, and every model except the one
    that made the change (the `origin`) resets itself.

    A Series is stored as a one-column DataFrame.
    """

    about_to_change = core.Signal(object)  # ty: ignore[missing-argument]
    changed = core.Signal(object)  # ty: ignore[missing-argument]

    def __init__(self, df: pd.DataFrame | pd.Series | None = None, **kwargs):
        super().__init__(**kwargs)
        self._df = _as_frame(df)

    @property
    def df(self) -> pd.DataFrame | None:
        return self._df

    @df.setter
    def df(self, df: pd.DataFrame | pd.Series | None):
        self.set_df(df)

    def set_df(self, df: pd.DataFrame | pd.Series | None, origin: Any = None):
        """Replace the frame. `origin` is passed on with the signals."""
        self.about_to_change.emit(origin)
        self._df = _as_frame(df)
        self.changed.emit(origin)


def _as_frame(df: pd.DataFrame | pd.Series | None) -> pd.DataFrame | None:
    return df.to_frame() if isinstance(df, pd.Series) else df


class DataFrameHolderMixin:
    """Model mixin: keeps `self.df` in a (possibly shared) DataFrameHolder.

    Call `_init_holder` from `__init__` after the Qt base class is initialized.
    Assigning `self.df` replaces the holder's frame; the model itself is then expected
    to emit the matching signals (e.g. inside `change_layout()`), while all other
    models of the holder are reset.
    """

    _holder: DataFrameHolder

    def _init_holder(self, df: pd.DataFrame | pd.Series | DataFrameHolder | None):
        if isinstance(df, DataFrameHolder):
            holder = df
        else:
            holder = DataFrameHolder(df, parent=self)
        self._holder = holder
        holder.about_to_change.connect(self._on_holder_about_to_change)
        holder.changed.connect(self._on_holder_changed)

    @property
    def holder(self) -> DataFrameHolder:
        return self._holder

    @property
    def df(self) -> pd.DataFrame | None:
        return self._holder.df

    @df.setter
    def df(self, df: pd.DataFrame | pd.Series | None):
        self._holder.set_df(df, origin=self)

    def _on_holder_about_to_change(self, origin: Any):
        if origin is not self:
            self.beginResetModel()  # ty: ignore[missing-attribute]

    def _on_holder_changed(self, origin: Any):
        if origin is not self:
            self.endResetModel()  # ty: ignore[missing-attribute]
