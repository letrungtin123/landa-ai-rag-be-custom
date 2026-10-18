"""Old import path of ``app.content.ordered_learning_content`` (moved verbatim in refactor wave 2, step 2.1).

Importing this path returns the moved module object itself (``sys.modules`` alias), so every name,
private helper and ``unittest.mock.patch`` target of the old path keeps working for the tests and the
backend's cross-language scripts; type checkers see the public names.
Application code imports the new path.
"""

import sys
from typing import TYPE_CHECKING

from app.content import ordered_learning_content as _moved

if TYPE_CHECKING:
    from app.content.ordered_learning_content import *

sys.modules[__name__] = _moved
