"""Old import path of ``app.services.staged_unit.validation`` (moved verbatim in refactor wave 2, step 2.1).

Importing this path returns the moved module object itself (``sys.modules`` alias), so every name,
private helper and ``unittest.mock.patch`` target of the old path keeps working for the tests and the
backend's cross-language scripts.
Application code imports the new path.
"""

import sys

from app.services.staged_unit import validation as _moved

sys.modules[__name__] = _moved
