"""Generate the code reference pages and navigation."""


from __future__ import annotations

import logging
import sys

from cutepandas import pandasmodels, pandaswidgets
from prettyqt import prettyqtmarkdown
import mknodes

logger = logging.getLogger(__name__)

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

root_nav = mknodes.MkNav()

model_docs = root_nav.add_doc(
    pandasmodels, section_name="models", class_page=prettyqtmarkdown.PrettyQtClassPage
)
widgets_docs = root_nav.add_doc(
    pandaswidgets, section_name="widgets", class_page=prettyqtmarkdown.PrettyQtClassPage
)
model_docs.collect_classes()
widgets_docs.collect_classes()

root_nav.write()
