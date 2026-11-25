## v0.7.0 (2026-11-25)

### Feat

- `GridDataFrameViewer`: a 2 x 2 grid viewer (level-name corner, column header, index,
  data) with MultiIndex spans, drag-resizable header columns / rows
- `SimpleDataFrameViewer`: a single `QTableView` with native headers (MultiIndex levels
  joined), dtype icons and the native sort indicator
- `SortController` / `SortableViewerMixin`: header-click sorting shared by all three
  viewers (copy / permutation mode, latest-wins worker, `sortable=False`)
- palette-aware viewer styling (`ViewerColors`, cell / header delegates): follows light
  and dark palettes and restyles on a palette or style change; dtype icons in the
  column headers, tinted to the palette (`KindIconCache`); a muted placeholder for nulls
- `DataFrameViewer` fits columns lazily as they scroll into view (bounded row sample)
- `DataFrameModel`'s vertical header shows the index labels (pandas)

### Fix

- sorting only from the leaf row of a MultiIndex column header
- the level-name headers stay aligned under an application style sheet (e.g. the
  prettyqt dark theme)

### Perf

- no per-widget style sheets (QStyleSheetStyle); fewer `data()` calls and no enum flag
  operations per painted cell; dtype icons cached instead of rebuilt per paint

## v0.6.2 (2026-11-25)

### Fix

- `DataFrameViewer` fits every column to its header and first rows (only the first
  column was fitted before) and uses one single-line row height (rows were measured
  with word wrap in 100 px columns: 40-55 px tall); data cells no longer wrap
- the index header gets its width when a frame is set after the viewer was shown (it
  stayed 2 px wide inside a layout)

## v0.6.1 (2026-11-25)

### Fix

- `DataFrameViewer` fills its parent: the data table takes the extra space (it kept its
  content size before, leaving an empty grid cell); the column / index headers stay
  pinned to the table's top-left edge

## v0.6.0 (2026-11-25)

### Feat

- `DataFrameViewer(sortable=..., sort_mode=...)` and the `sort_requested(column, order)`
  signal: with `sortable=False` a header click only requests the sort, so an
  application can sort under its own control (`set_sort_state()` restores the
  indicator)
- display-only sort (`sort_mode="permutation"`, opt-in): a row order of 8 bytes per row
  in `DataFrameHolder.row_order` instead of a sorted copy of the frame; computed in a
  worker thread by the viewer (`sorting_changed`, `cancel_sort()`, latest request wins,
  stale orders dropped); `DataFrameModel.compute_sort_permutation()` /
  `apply_sort_permutation()` / `source_row()` / `clear_sort()`
- `PermutableSource` hook (`argsort`, `fetch_rows`) for the pandas, pyarrow and polars
  sources; pandas sorts NumPy columns with NumPy and Arrow-backed columns with Arrow's
  sort kernel
- translatable labels: `translate=` / `set_translate()` / `retranslate()` on
  `SchemaModel`, `PandasColumnListModel`, `PandasIndexListModel`, `CategoryModel`,
  `TopValuesModel`, `ColumnProfileWidget` and `DataFrameWidget`; English labels marked
  with `cutepandas.utils.i18n.N_` for catalog extraction

## v0.5.0 (2026-11-25)

### Feat

- backend-agnostic core `cutepandas.frames` (Qt-free): `FrameSource` protocol with
  adapters for pandas (incl. pyarrow-backed dtypes), pyarrow and polars, `Kind`,
  per-dtype formatters (`FormatOptions`), thread-safe `ChunkCache`
- `DataFrameModel`: one virtual table model for every backend; fetches the visible
  window in cached blocks, colours / normalises values per block, `prefetch()` from a
  worker thread
- `SchemaModel`, `CategoryModel` (any backend), `ColumnProfileWidget`,
  `DTypeDelegate`, `ValueColorDelegate`, dtype icons for all backends
- `DataFrameViewer` / `DataFrameWidget` show pyarrow and polars frames too

### BREAKING CHANGE

- `PandasDataFrameModel` and `PolarsDataFrameModel` are replaced by `DataFrameModel`;
  the unexported `dataframeviewer2` module is removed
- `PandasColumnListModel` has a new "Nulls" column (size moved to column 3)
- pyarrow is now a required dependency

## v0.3.2 (2023-07-25)

### Fix

- fix build pipeline

## v0.3.1 (2023-07-25)

### Fix

- disable tests for now

## v0.3.0 (2023-07-25)

### Feat

- merge from prettyqt

## v0.2.2 (2022-11-11)

## v0.2.1 (2022-11-11)

### Fix

- fix tests

## v0.2.0 (2022-11-11)

### Feat

- initial commit
