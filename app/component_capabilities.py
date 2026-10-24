"""Old import path of ``app.contracts.component_capabilities``, moved there in refactor wave 3b.

Strict code imports the new path; the legacy lesson-author pipeline and its tests keep importing these
names from here.
"""

from app.contracts.component_capabilities import (
    ComponentCapabilities,
    component_capabilities,
    validate_instance_plan,
)

__all__ = [
    "ComponentCapabilities",
    "component_capabilities",
    "validate_instance_plan",
]
