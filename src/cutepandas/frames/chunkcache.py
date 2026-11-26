"""A thread-safe, bounded LRU cache of per-column row chunks."""

from collections import OrderedDict
import threading
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Callable


class ChunkCache[T]:
    """Caches converted windows of a frame, keyed by (row block, column).

    A row block covers `chunk_rows` consecutive rows of one column, so a wide frame only
    converts the visible columns. At most `max_chunks` blocks are kept; the least
    recently used one is dropped first, so memory stays bounded however far the user
    scrolls.

    Every mutation happens under a lock, so a worker thread may fill the cache
    (`get_or_build`) while the GUI thread reads it. `clear()` and `invalidate()` start a
    new generation: a block built from the old data by a slow worker is dropped instead
    of stored.
    """

    def __init__(self, chunk_rows: int = 256, max_chunks: int = 256):
        if chunk_rows < 1 or max_chunks < 1:
            msg = "chunk_rows and max_chunks must be positive"
            raise ValueError(msg)
        self.chunk_rows = chunk_rows
        self.max_chunks = max_chunks
        self._lock = threading.Lock()
        self._chunks: OrderedDict[tuple[int, int], T] = OrderedDict()
        self._generation = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._chunks)

    @property
    def generation(self) -> int:
        return self._generation

    def block_of(self, row: int) -> int:
        return row // self.chunk_rows

    def block_bounds(self, block: int, row_count: int) -> tuple[int, int]:
        """Row range (start, stop) of a block, clipped to `row_count`."""
        start = block * self.chunk_rows
        return start, min(start + self.chunk_rows, row_count)

    def get(self, block: int, column: int) -> T | None:
        key = (block, column)
        with self._lock:
            chunk = self._chunks.get(key)
            if chunk is not None:
                self._chunks.move_to_end(key)
            return chunk

    def put(
        self, block: int, column: int, chunk: T, generation: int | None = None
    ) -> bool:
        """Store a block; returns False if it was built for an older generation."""
        with self._lock:
            if generation is not None and generation != self._generation:
                return False
            self._chunks[block, column] = chunk
            self._chunks.move_to_end((block, column))
            while len(self._chunks) > self.max_chunks:
                self._chunks.popitem(last=False)
            return True

    def get_or_build(self, block: int, column: int, build: Callable[[int, int], T]) -> T:
        """Return the cached block or build it (outside the lock) and store it."""
        if (chunk := self.get(block, column)) is not None:
            return chunk
        generation = self._generation
        chunk = build(block, column)
        self.put(block, column, chunk, generation)
        return chunk

    def invalidate(self, *, block: int | None = None, column: int | None = None):
        """Drop the blocks of one row block and/or column (all if both are None)."""
        with self._lock:
            # A new generation also drops blocks that are being built right now.
            self._generation += 1
            if block is None and column is None:
                self._chunks.clear()
                return
            for key in list(self._chunks):
                if (block is None or key[0] == block) and (
                    column is None or key[1] == column
                ):
                    del self._chunks[key]

    def clear(self):
        """Drop everything and start a new generation (call when the frame changes)."""
        self.invalidate()
