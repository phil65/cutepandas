"""Header-click sorting shared by the DataFrame viewers."""

import logging
import threading
from typing import TYPE_CHECKING, Any

from prettyqt import constants, core


if TYPE_CHECKING:
    from cutepandas.pandasmodels import DataFrameModel
    from cutepandas.pandasmodels.dataframemodel import SortMode


logger = logging.getLogger(__name__)


class SortController(core.Object):
    """Sort state and sorting of one `DataFrameModel`, driven by header clicks.

    `request_sort(column)` (a header click) emits `sort_requested(column, order)`
    (clicking the sorted column again reverses the order) and, if `sortable`, sorts:

    - `sort_mode="copy"`: the model replaces the frame by a sorted copy (GUI thread).
    - `sort_mode="permutation"`: the row order is computed in a worker thread
      (`sorting_changed(True)` while it runs, `sorting_changed(False)` when it was
      applied or dropped). Only the latest request is applied, and only if the frame
      was not replaced or edited meanwhile; `cancel_sort()` drops a running one.

    Replacing the frame, or clearing the row order, resets the sort state.
    `sort_state_changed` is emitted with `(column, order)` or None whenever the
    state changes, so a viewer can repaint its sort indicator.
    """

    sort_requested = core.Signal(int, object)  # ty: ignore[missing-argument]
    sorting_changed = core.Signal(bool)  # ty: ignore[missing-argument]
    sort_state_changed = core.Signal(object)  # ty: ignore[missing-argument]
    # (request id, SortPermutation | BaseException | None) from the worker thread
    _permutation_ready = core.Signal(int, object)  # ty: ignore[missing-argument]

    def __init__(
        self,
        model: DataFrameModel,
        *,
        sortable: bool = True,
        parent: core.QObject | None = None,
    ):
        super().__init__(parent=parent)
        self.model = model
        self._sortable = sortable
        self._state: tuple[int, bool] | None = None  # (column, ascending)
        self._request = 0
        self._sorting = False
        self._permutation_ready.connect(self._on_permutation_ready)
        model.holder.changed.connect(self._on_df_changed)
        model.holder.row_order_changed.connect(self._on_row_order_changed)

    def _on_df_changed(self, origin: Any = None):
        self.set_sort_state(None)

    def _on_row_order_changed(self, origin: Any = None):
        if self.model.holder.row_order is None:
            self.set_sort_state(None)

    def request_sort(self, column: int):
        """Header click on `column`: emit `sort_requested`, sort if `sortable`."""
        if self.model.df is None:
            return
        ascending = self._state != (column, True)
        order = constants.ASCENDING if ascending else constants.DESCENDING
        self.sort_requested.emit(column, order)
        if self._sortable:
            self.sort_by(column, order)

    def sort_by(self, column: int, order: constants.SortOrder = constants.ASCENDING):
        """Sort by a column position (in the model's `sort_mode`).

        In permutation mode this returns right away; the order is computed in a
        worker thread and applied later (see `sorting_changed`).
        """
        model = self.model
        if model.get_sort_mode() != "permutation":
            model.sort(column, order)
            self.set_sort_state(column, order)  # after: a frame change resets it
            return
        self._request += 1
        request = self._request
        self._set_sorting(True)

        def work():
            try:
                result = model.compute_sort_permutation(column, order)
            except Exception as e:  # noqa: BLE001 (reported on the GUI thread)
                result = e
            self._permutation_ready.emit(request, result)  # queued to the GUI thread

        threading.Thread(target=work, name="cutepandas-sort", daemon=True).start()

    def _on_permutation_ready(self, request: int, result: Any):
        if request != self._request:  # superseded or cancelled
            return
        self._set_sorting(False)
        if isinstance(result, BaseException):
            logger.error("Sorting failed", exc_info=result)
            return
        if result is not None and self.model.apply_sort_permutation(result):
            order = constants.DESCENDING if result.descending else constants.ASCENDING
            self.set_sort_state(result.column, order)

    def cancel_sort(self):
        """Drop a display-only sort that is still being computed."""
        self._request += 1
        self._set_sorting(False)

    def is_sorting(self) -> bool:
        """True while a display-only sort order is computed."""
        return self._sorting

    def _set_sorting(self, value: bool):
        if value != self._sorting:
            self._sorting = value
            self.sorting_changed.emit(value)

    def clear_sort(self):
        """Drop a display-only sort (permutation mode) and the sort state."""
        self.cancel_sort()
        self.model.clear_sort()
        self.set_sort_state(None)

    def set_sort_state(
        self, column: int | None, order: constants.SortOrder | None = None
    ):
        """Remember the current sort (the next request for `column` reverses it).

        Replacing the frame resets it; an application that sorts on
        `sort_requested` and then sets the sorted frame restores it here.
        """
        state = None if column is None else (column, order != constants.DESCENDING)
        if state == self._state:
            return
        self._state = state
        self.sort_state_changed.emit(self.get_sort_state())

    def get_sort_state(self) -> tuple[int, constants.SortOrder] | None:
        """(column, order) of the current sort, None if unsorted."""
        if self._state is None:
            return None
        column, ascending = self._state
        return column, constants.ASCENDING if ascending else constants.DESCENDING

    def set_sortable(self, value: bool):
        self._sortable = value

    def is_sortable(self) -> bool:
        return self._sortable

    def set_sort_mode(self, mode: SortMode):
        self.model.set_sort_mode(mode)

    def get_sort_mode(self) -> SortMode:
        return self.model.get_sort_mode()


class SortableViewerMixin:
    """Viewer sort API delegating to `self.sorter` (a `SortController`).

    The viewer declares the `sort_requested` / `sorting_changed` signals and forwards
    the controller's (`_connect_sorter()`).
    """

    sorter: SortController
    sort_requested: Any  # signals of the viewer class
    sorting_changed: Any

    def _connect_sorter(self):
        self.sorter.sort_requested.connect(self.sort_requested)
        self.sorter.sorting_changed.connect(self.sorting_changed)

    def sort_by(self, column: int, order: constants.SortOrder = constants.ASCENDING):
        """Sort by a column position (see `SortController.sort_by`)."""
        self.sorter.sort_by(column, order)

    def cancel_sort(self):
        """Drop a display-only sort that is still being computed."""
        self.sorter.cancel_sort()

    def is_sorting(self) -> bool:
        """True while a display-only sort order is computed."""
        return self.sorter.is_sorting()

    def clear_sort(self):
        """Drop a display-only sort (permutation mode) and the sort state."""
        self.sorter.clear_sort()

    def set_sort_state(
        self, column: int | None, order: constants.SortOrder | None = None
    ):
        """Remember the current sort (the next click on `column` reverses it).

        Replacing the frame resets it; an application that sorts on
        `sort_requested` and then sets the sorted frame restores it here.
        """
        self.sorter.set_sort_state(column, order)

    def get_sort_state(self) -> tuple[int, constants.SortOrder] | None:
        """(column, order) of the current sort, None if unsorted."""
        return self.sorter.get_sort_state()

    def set_sortable(self, value: bool):
        self.sorter.set_sortable(value)

    def is_sortable(self) -> bool:
        return self.sorter.is_sortable()

    def set_sort_mode(self, mode: SortMode):
        self.sorter.set_sort_mode(mode)

    def get_sort_mode(self) -> SortMode:
        return self.sorter.get_sort_mode()
