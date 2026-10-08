"""Route modules, one per route family.

Each module holds thin endpoint functions and a ``register(app)`` that declares them on the
application with ``app.add_api_route``. Routes are registered on the app itself rather than
through ``include_router``: FastAPI 0.142 keeps an included router as one lazy
``_IncludedRouter`` entry, while the application, its OpenAPI document and the route tests
(auth dependency, route snapshots) work on a flat list of ``APIRoute`` objects.
"""
