"""Old import path of ``app.contracts.orchestration_v2.blueprint`` (moved verbatim in refactor wave 2, step 2.1).

Importing this path returns the moved module object itself (``sys.modules`` alias). It is kept only for the Node
backend: its IDM fixture (``lesson-author-idm.fixture.ts``) runs ``from app.lesson_author_orchestration_v2 import
canonical_hash`` against this checkout. Remove it once the backend imports ``app.hashing`` instead. Application and
test code import the new path (``tests/contract/test_architecture_layers.py`` checks it).
"""

import sys

from app.contracts.orchestration_v2 import blueprint as _moved

sys.modules[__name__] = _moved
