"""cutepandas: main package.

Qt item models, delegates and widgets for dataframes (pandas, polars, pyarrow).
"""

from importlib.metadata import version

__version__ = version("cutepandas")
__title__ = "cutepandas"

__author__ = "Philipp Temminghoff"
__author_email__ = "philipptemminghoff@googlemail.com"
__copyright__ = "Copyright (c) 2019 Philipp Temminghoff"
__license__ = "MIT"
__url__ = "https://github.com/phil65/cutepandas"

__all__ = [
    "__version__",
    "delegates",
    "frames",
    "pandasmodels",
    "pandaswidgets",
]
