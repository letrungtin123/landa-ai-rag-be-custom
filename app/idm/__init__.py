"""IDM (Instructional Design Methodology) pipeline for Lesson Author orchestration V2.

The package never imports ``app.main``; the service layer injects an
:class:`app.idm.runtime.IdmRuntime` and calls the stage entry points.
"""
