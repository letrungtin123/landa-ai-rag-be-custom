"""Old import path of ``app.content.learner_content_purity`` (moved verbatim in refactor wave 2, step 2.1).

Importing this path returns the moved module object itself (``sys.modules`` alias), so every name,
private helper and ``unittest.mock.patch`` target of the old path keeps working for the tests and the
backend's cross-language scripts; type checkers see the public names.
Application code imports the new path.
"""

import sys
from typing import TYPE_CHECKING

from app.content import learner_content_purity as _moved

if TYPE_CHECKING:
    from app.content.learner_content_purity import *

sys.modules[__name__] = _moved
