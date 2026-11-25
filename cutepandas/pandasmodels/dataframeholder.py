"""A shared, observable container for one frame (pandas, polars, pyarrow)."""

from __future__ import annotations

from typing import Any

import pandas as pd
from prettyqt import core

from cutepandas.frames import FrameSource, as_source


class DataFrameHolder(core.Object):
    """Holds the current frame for several models / widgets (GUI thread only).

    Frame operations like `rename` or `astype` return a new frame instead of changing
    the old one. Models that share a holder always see the current frame: replacing it
    emits `about_to_change` and `changed`, and every model except the one that made the
    change (the `origin`) resets itself.

    Accepts a pandas DataFrame / Series (a Series is stored as a one-column frame), a
    pyarrow Table, a polars DataFrame or any `FrameSource`. `df` is the native frame,
    `source` the backend-agnostic adapter the table models read through.
    """

    about_to_change = core.Signal(object)  # ty: ignore[missing-argument]
    changed = core.Signal(object)  # ty: ignore[missing-argument]

    def __init__(self, df: Any = None, **kwargs):
        super().__init__(**kwargs)
        self._df: Any = None
        self._source: FrameSource | None = None
        self._store(df)

    def _store(self, df: Any):
        if isinstance(df, pd.Series):
            df = df.to_frame()
        if df is None:
            self._df, self._source = None, None
        elif isinstance(df, FrameSource) and not isinstance(df, pd.DataFrame):
            self._df, self._source = df.native, df
        else:
            self._df, self._source = df, None  # adapter created on first use

    @property
    def df(self) -> Any:
        """The native frame (pandas DataFrame, pyarrow Table, polars DataFrame)."""
        return self._df

    @df.setter
    def df(self, df: Any):
        self.set_df(df)

    @property
    def source(self) -> FrameSource | None:
        if self._source is None and self._df is not None:
            self._source = as_source(self._df)
        return self._source

    def set_df(self, df: Any, origin: Any = None):
        """Replace the frame. `origin` is passed on with the signals."""
        self.about_to_change.emit(origin)
        self._store(df)
        self.changed.emit(origin)


class DataFrameHolderMixin:
    """Model mixin: keeps `self.df` in a (possibly shared) DataFrameHolder.

    Call `_init_holder` from `__init__` after the Qt base class is initialized.
    Assigning `self.df` replaces the holder's frame; the model itself is then expected
    to emit the matching signals (e.g. inside `change_layout()`), while all other
    models of the holder are reset.
    """

    _holder: DataFrameHolder

    def _init_holder(self, df: Any):
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
    def df(self) -> Any:
        return self._holder.df

    @df.setter
    def df(self, df: Any):
        self._holder.set_df(df, origin=self)

    @property
    def source(self) -> FrameSource | None:
        return self._holder.source

    def _on_holder_about_to_change(self, origin: Any):
        if origin is not self:
            self.beginResetModel()  # ty: ignore[missing-attribute]

    def _on_holder_changed(self, origin: Any):
        if origin is not self:
            self.endResetModel()  # ty: ignore[missing-attribute]
