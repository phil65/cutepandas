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
