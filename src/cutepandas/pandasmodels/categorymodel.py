"""`CategoryModel`: the categories of a categorical / enum column (any backend)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from prettyqt import constants, core, iconprovider, itemmodels

from cutepandas.constants import iconnames
from cutepandas.frames import format_name
from cutepandas.pandasmodels.dataframeholder import DataFrameHolderMixin
from cutepandas.utils.i18n import N_, TranslatableHeaderMixin


if TYPE_CHECKING:
    from cutepandas.utils.i18n import Translate


class CategoryModel(
    TranslatableHeaderMixin,
    DataFrameHolderMixin,
    itemmodels.ModelMixin,
    core.AbstractTableModel,
):
    """One row per category of a column, with its code and number of rows, read-only.

    Works for pandas `category` and dictionary-encoded `pd.ArrowDtype` columns, polars
    `Enum` (fixed categories) and `Categorical` (the values present), and pyarrow
    dictionary columns. Other columns show no rows. Categories and counts are computed
    once per frame (one `value_counts` scan).
    """

    HEADER: ClassVar[list[str]] = [N_("Name"), N_("Code"), N_("Count")]

    def __init__(
        self,
        df: Any = None,
        col: Any = None,
        *,
        translate: Translate | None = None,
        **kwargs,
    ):
        """Show the categories of column `col` (a label).

        `translate` maps the English labels (header, tooltip) to the shown text.
        """
        super().__init__(**kwargs)
        self._translate = translate
        self.col = col
        self._categories: list[Any] | None = None
        self._counts: dict[Any, int] = {}
        self._init_holder(df)

    def _on_holder_changed(self, origin: Any):
        self._categories = None
        super()._on_holder_changed(origin)

    def column_position(self) -> int | None:
        source = self.source
        if source is None or self.col is None:
            return None
        for pos, info in enumerate(source.columns):
            if info.name == self.col:
                return pos
        return None

    @property
    def categories(self) -> list[Any]:
        if self._categories is None:
            pos = self.column_position()
            source = self.source
            if pos is None or source is None:
                self._categories, self._counts = [], {}
            else:
                self._categories = source.categories(pos) or []
                if self._categories:
                    self._counts = dict(source.value_counts(pos))
        return self._categories

    def count(self, category: Any) -> int:
        """Number of rows with this category."""
        _ = self.categories
        return self._counts.get(category, 0)

    def rowCount(self, parent: core.ModelIndex | None = None) -> int:
        if parent is not None and parent.isValid():
            return 0
        return len(self.categories)

    def columnCount(self, parent: core.ModelIndex | None = None) -> int:
        if parent is not None and parent.isValid():
            return 0
        return len(self.HEADER)

    def data(
        self,
        index: core.ModelIndex,
        role: constants.ItemDataRole = constants.DISPLAY_ROLE,
    ):
        if not index.isValid() or index.row() >= len(self.categories):
            return None
        category = self.categories[index.row()]
        match role, index.column():
            case constants.DISPLAY_ROLE | constants.EDIT_ROLE, 0:
                return format_name(category)
            case constants.DISPLAY_ROLE | constants.EDIT_ROLE, 1:
                # A category's code is its position in the categories.
                return str(index.row())
            case constants.DISPLAY_ROLE, 2:
                return str(self.count(category))
            case constants.SORT_ROLE, 0:
                return format_name(category)
            case constants.SORT_ROLE, 1:
                return index.row()
            case constants.SORT_ROLE, 2:
                return self.count(category)
            case constants.ALIGNMENT_ROLE, 1 | 2:
                return constants.ALIGN_RIGHT | constants.ALIGN_V_CENTER
            case constants.DECORATION_ROLE, 0:
                return iconprovider.get_icon(iconnames.CATEGORY, as_qicon=True)
            case constants.NAME_ROLE | constants.USER_ROLE, _:
                return category
            case constants.TOOLTIP_ROLE, _:
                items = self._tr(N_("Items"))
                return f"<b>{category}</b><br>{items}: {self.count(category)}"
        return None
