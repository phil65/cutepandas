"""`SchemaModel`: the columns of a frame with dtype, icon, null count and size."""

from __future__ import annotations

import enum
from typing import TYPE_CHECKING, Any, ClassVar

from prettyqt import constants, core, itemmodels

from cutepandas.pandasmodels.dataframeholder import DataFrameHolderMixin
from cutepandas.utils import icons
from cutepandas.utils.i18n import N_, TranslatableHeaderMixin


if TYPE_CHECKING:
    from cutepandas.frames import ColumnInfo
    from cutepandas.utils.i18n import Translate


class SchemaModel(
    TranslatableHeaderMixin,
    DataFrameHolderMixin,
    itemmodels.ModelMixin,
    core.AbstractTableModel,
):
    """One row per column of a frame (any backend), read-only.

    Null counts and sizes are computed on first display of a row and cached until the
    frame changes (a null count scans the column; polars and pyarrow keep it in
    metadata). Sort it with a `SortFilterProxyModel` on `SORT_ROLE`.

    Labels (`HEADER`, tooltips) go through `translate` (see `cutepandas.utils.i18n`).
    """

    HEADER: ClassVar[list[str]] = [N_("Name"), N_("Type"), N_("Nulls"), N_("Size")]

    class Roles(enum.IntEnum):
        """Role names."""

        DTypeRole = constants.USER_ROLE + 4555
        ColumnNameRole = constants.USER_ROLE + 4556
        KindRole = constants.USER_ROLE + 4557
        ColumnInfoRole = constants.USER_ROLE + 4560

    def __init__(self, df: Any = None, *, translate: Translate | None = None, **kwargs):
        super().__init__(**kwargs)
        self._translate = translate
        self._null_counts: dict[int, int] = {}
        self._sizes: dict[int, int] = {}
        self._kind_icons = icons.KindIconCache()
        self._init_holder(df)

    def _on_holder_changed(self, origin: Any):
        self._null_counts.clear()
        self._sizes.clear()
        super()._on_holder_changed(origin)

    def column(self, row: int) -> ColumnInfo:
        source = self.source
        assert source is not None
        return source.columns[row]

    def null_count(self, row: int) -> int:
        if (count := self._null_counts.get(row)) is None:
            assert self.source is not None
            count = self._null_counts[row] = self.source.null_count(row)
        return count

    def nbytes(self, row: int) -> int:
        if (size := self._sizes.get(row)) is None:
            assert self.source is not None
            size = self._sizes[row] = self.source.nbytes(row)
        return size

    def rowCount(self, parent: core.ModelIndex | None = None) -> int:
        source = self.source
        if source is None or (parent is not None and parent.isValid()):
            return 0
        return len(source.columns)

    def columnCount(self, parent: core.ModelIndex | None = None) -> int:
        if parent is not None and parent.isValid():
            return 0
        return len(self.HEADER)

    def data(
        self,
        index: core.ModelIndex,
        role: constants.ItemDataRole = constants.DISPLAY_ROLE,
    ):
        if not index.isValid() or self.source is None:
            return None
        row = index.row()
        info = self.column(row)
        match role, index.column():
            case constants.DISPLAY_ROLE | constants.EDIT_ROLE, 0:
                return info.label
            case constants.DISPLAY_ROLE | constants.EDIT_ROLE, 1:
                return info.dtype
            case constants.DISPLAY_ROLE, 2:
                return str(self.null_count(row))
            case constants.DISPLAY_ROLE, 3:
                return core.Locale().get_formatted_data_size(self.nbytes(row))
            case constants.SORT_ROLE, 0:
                return info.label
            case constants.SORT_ROLE, 1:
                return info.dtype
            case constants.SORT_ROLE, 2:
                return self.null_count(row)
            case constants.SORT_ROLE, 3:
                return self.nbytes(row)
            case constants.ALIGNMENT_ROLE, 2 | 3:
                return constants.ALIGN_RIGHT | constants.ALIGN_V_CENTER
            case constants.DECORATION_ROLE, 0:
                return self._kind_icons.icon(info.kind)
            case constants.TOOLTIP_ROLE, _:
                return (
                    f"<b>{info.label}</b><br>{self._tr(N_('Type'))}: {info.dtype}"
                    f" ({info.kind.value})<br>{self._tr(N_('Nulls'))}: "
                    f"{self.null_count(row)}"
                )
            case self.Roles.DTypeRole, _:
                return info.native_dtype
            case self.Roles.KindRole, _:
                return info.kind
            case self.Roles.ColumnNameRole | constants.NAME_ROLE, _:
                return info.name
            case self.Roles.ColumnInfoRole, _:
                return info
        return None
