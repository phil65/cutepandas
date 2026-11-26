"""Translatable UI labels.

cutepandas ships no translations. Widgets and models with fixed English labels
(header titles, field names) take a `translate` callable (`str -> str`, e.g. a
gettext function) and look each label up when they show it; `set_translate()`
swaps the callable and `retranslate()` re-shows the labels (language switch at
runtime). This matches prettyqt's `PydanticForm(translate=...)`.

The English labels are marked with `N_` (a no-op), so a message catalog can be
extracted with `xgettext --keyword=N_` (or babel `-k N_`) over the package.
"""

from collections.abc import Callable
from typing import Any, ClassVar

from prettyqt import constants


# Maps an English label (the msgid) to the text to show.
type Translate = Callable[[str], str]


def N_(text: str) -> str:
    """Mark a label for extraction; returns it unchanged."""
    return text


class TranslateMixin:
    """Keeps an optional `translate` callable; `_tr(text)` applies it.

    Classes using it implement `retranslate()` to re-show their labels.
    """

    _translate: Translate | None = None

    def _tr(self, text: str) -> str:
        return text if self._translate is None else self._translate(text)

    def set_translate(self, translate: Translate | None):
        """Use another translate callable (None: English) and re-show the labels."""
        self._translate = translate
        self.retranslate()

    def get_translate(self) -> Translate | None:
        return self._translate

    def retranslate(self):
        """Re-show the labels (after the translate callable changed its language)."""


class TranslatableHeaderMixin(TranslateMixin):
    """Item model mixin: horizontal header labels from `HEADER`, translated.

    Put it before the Qt model base class. `retranslate()` emits
    `headerDataChanged` and `dataChanged` (cells and tooltips may contain labels).
    """

    HEADER: ClassVar[list[str]] = []

    def headerData(
        self,
        offset: int,
        orientation: constants.Orientation,
        role: constants.ItemDataRole = constants.DISPLAY_ROLE,
    ):
        if (
            orientation == constants.HORIZONTAL
            and role == constants.DISPLAY_ROLE
            and 0 <= offset < len(self.HEADER)
        ):
            return self._tr(self.HEADER[offset])
        return None

    def retranslate(self):
        model: Any = self
        columns = len(self.HEADER)
        if not columns:
            return
        model.headerDataChanged.emit(constants.HORIZONTAL, 0, columns - 1)
        if rows := model.rowCount():
            model.dataChanged.emit(model.index(0, 0), model.index(rows - 1, columns - 1))
