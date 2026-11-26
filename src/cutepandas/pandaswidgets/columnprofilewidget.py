"""`ColumnProfileWidget`: basic statistics and top values of one column."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from prettyqt import constants, core, widgets

from cutepandas.delegates import ValueColorDelegate
from cutepandas.frames import ColumnInfo, ColumnStats, FrameSource, as_source, format_name
from cutepandas.pandasmodels.dataframeholder import DataFrameHolder
from cutepandas.pandasmodels.dataframemodel import DataFrameModel
from cutepandas.utils import icons
from cutepandas.utils.i18n import N_, TranslatableHeaderMixin, TranslateMixin


if TYPE_CHECKING:
    from collections.abc import Sequence

    from cutepandas.utils.i18n import Translate


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return format(value, ".6g")
    return str(value)


class TopValuesModel(TranslatableHeaderMixin, core.AbstractTableModel):
    """Most frequent values of a column: value, count and share of all rows."""

    HEADER: ClassVar[list[str]] = [N_("Value"), N_("Count"), N_("Share")]

    def __init__(
        self,
        top: Sequence[tuple[Any, int]] = (),
        total: int = 0,
        *,
        translate: Translate | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._translate = translate
        self._top = list(top)
        self._total = total
        self._max = max((count for _, count in self._top), default=0)

    def rowCount(self, parent: core.ModelIndex | None = None) -> int:
        return 0 if parent is not None and parent.isValid() else len(self._top)

    def columnCount(self, parent: core.ModelIndex | None = None) -> int:
        return 0 if parent is not None and parent.isValid() else len(self.HEADER)

    def data(
        self,
        index: core.ModelIndex,
        role: constants.ItemDataRole = constants.DISPLAY_ROLE,
    ):
        if not index.isValid():
            return None
        value, count = self._top[index.row()]
        match role, index.column():
            case constants.DISPLAY_ROLE, 0:
                return format_name(value)
            case constants.DISPLAY_ROLE, 1:
                return str(count)
            case constants.DISPLAY_ROLE, 2:
                return f"{count / self._total:.1%}" if self._total else ""
            case constants.ALIGNMENT_ROLE, 1 | 2:
                return constants.ALIGN_RIGHT | constants.ALIGN_V_CENTER
            case DataFrameModel.Roles.NormalizedRole, 2:
                return count / self._max if self._max else None
            case constants.USER_ROLE, _:
                return value
        return None


class ColumnProfileWidget(TranslateMixin, widgets.Widget):
    """Basic statistics and the most frequent values of one column of any frame.

    Shows dtype, row / null / unique counts, min / max / mean / std and the top values
    with a share bar.

    `set_column()` computes the statistics right away (`FrameSource.describe`, a full
    column scan). For large frames compute them in a worker thread and pass the
    result to `set_stats()`, which must run on the GUI thread.

    The field labels (`FIELDS`) and the top values header go through `translate`
    (`str -> str`, e.g. gettext); `set_translate()` / `retranslate()` switch the
    language at runtime (see `cutepandas.utils.i18n`).
    """

    FIELDS: ClassVar = (
        N_("Rows"),
        N_("Nulls"),
        N_("Unique"),
        N_("Min"),
        N_("Max"),
        N_("Mean"),
        N_("Std"),
    )

    def __init__(
        self,
        df: Any = None,
        column: Any = None,
        *,
        top: int = 10,
        translate: Translate | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._translate = translate
        self.top = top
        layout = self.set_layout("vertical", margin=4)
        self.title = widgets.Label()
        self.title.setTextFormat(constants.TextFormat.RichText)
        layout.add(self.title)
        form = widgets.FormLayout()
        self.value_labels: dict[str, widgets.Label] = {}
        self.field_labels: dict[str, widgets.Label] = {}
        for name in self.FIELDS:
            label = widgets.Label()
            label.set_text_interaction("by_mouse")
            self.value_labels[name] = label
            self.field_labels[name] = widgets.Label()
            form.addRow(self.field_labels[name], label)
        layout.add(form)
        self.top_table = widgets.TableView(selection_behavior="rows")
        self.top_table.v_header.hide()
        self.top_table.set_delegate(ValueColorDelegate(mode="bar"), column=2)
        layout.add(self.top_table)
        self.retranslate()
        self.clear()
        if df is not None and column is not None:
            self.set_column(df, column)

    def retranslate(self):
        """Re-show the labels with the current `translate` callable."""
        for name, label in self.field_labels.items():
            label.set_text(f"{self._tr(name)}:")
        if isinstance(model := self.top_table.model(), TopValuesModel):
            model.set_translate(self._translate)

    def clear(self):
        self.title.set_text("")
        for label in self.value_labels.values():
            label.set_text("")
        self.top_table.set_model(
            TopValuesModel(translate=self._translate, parent=self.top_table)
        )

    def set_column(self, df: Any, column: Any):
        """Profile a column (position or label) of a frame, source or holder."""
        source = df.source if isinstance(df, DataFrameHolder) else as_source(df)
        if source is None:
            self.clear()
            return
        pos = self._position(source, column)
        self.set_stats(source.columns[pos], source.describe(pos, top=self.top))

    @staticmethod
    def _position(source: FrameSource, column: Any) -> int:
        names = [info.name for info in source.columns]
        if column in names:
            return names.index(column)
        if isinstance(column, int) and 0 <= column < len(names):
            return column
        msg = f"No column {column!r}"
        raise KeyError(msg)

    def set_stats(self, info: ColumnInfo, stats: ColumnStats):
        """Show precomputed statistics (GUI thread)."""
        self.title.set_text(f"<b>{info.label}</b> &nbsp; {info.dtype}")
        self.setWindowIcon(icons.icon_for_kind(info.kind))
        share = f" ({stats.null_count / stats.count:.1%})" if stats.count else ""
        values = {
            "Rows": str(stats.count),
            "Nulls": f"{stats.null_count}{share}",
            "Unique": _fmt(stats.unique),
            "Min": _fmt(stats.min),
            "Max": _fmt(stats.max),
            "Mean": _fmt(stats.mean),
            "Std": _fmt(stats.std),
        }
        for name, text in values.items():
            self.value_labels[name].set_text(text)
        old = self.top_table.model()
        self.top_table.set_model(
            TopValuesModel(stats.top, stats.count, translate=self._translate, parent=self)
        )
        if old is not None:
            old.deleteLater()
