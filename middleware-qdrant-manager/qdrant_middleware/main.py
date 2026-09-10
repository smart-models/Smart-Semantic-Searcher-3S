"""Application factory and package entrypoint for qdrant-middleware."""

import json
import logging
import pathlib
import threading

from fastapi import Depends, FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response

from qdrant_middleware.api.dependencies import (
    require_bearer_auth,
    require_operational_readiness,
)
from qdrant_middleware.api.routers import collections, monitoring, points, query
from qdrant_middleware.core.config import Settings, get_settings
from qdrant_middleware.core.lifespan import lifespan
from qdrant_middleware.core.runtime import create_runtime_state
from qdrant_middleware.middleware.body_limit import BodySizeLimitMiddleware
from qdrant_middleware.middleware.metrics import TimeoutMetricsMiddleware
from qdrant_middleware.middleware.rate_limit import rate_limit_middleware
from qdrant_middleware.middleware.shutdown import shutdown_middleware
from qdrant_middleware.services.vector_reindex import VectorReindexManager

# Defense-in-depth headers for the operator console. The SPA loads every script
# from its own origin and ships no inline script or event handler, so
# default-src 'self' covers scripts, fonts and XHR; img-src adds the one data:
# URI favicon; connect-src 'self' lets the console reach the API it is served
# with; frame-ancestors 'none' plus X-Frame-Options keep it out of frames.
#
# style-src is the one directive that has to allow 'unsafe-inline', and the
# reason is narrow: the result cards size their score bars with a computed
# `style="width:NN%"` attribute, and style-src governs style *attributes* as
# well as <style> blocks (style-src-attr falls back to it). Without the
# allowance those bars silently render at zero width -- a broken console, not a
# blocked attack. The exposure it buys back is style injection, which needs an
# injection point to begin with: every value the console interpolates goes
# through esc(), and script-src stays strict, which is where the real risk
# lives. Tightening this again means moving those three widths to the CSSOM
# (`element.style.width = ...`), which CSP does not police, and rebuilding
# frontend/dist.
_CONSOLE_CSP = (
    "default-src 'self'; "
    "img-src 'self' data:; "
    "connect-src 'self'; "
    "style-src 'self' 'unsafe-inline'; "
    "frame-ancestors 'none'; "
    "base-uri 'none'"
)
_CONSOLE_SECURITY_HEADERS = {
    "Content-Security-Policy": _CONSOLE_CSP,
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
}


def configure_package_logging() -> None:
    """Attach one stream handler to this package's logger, at INFO.

    Uvicorn configures only its own ``uvicorn*`` loggers, and nothing else in
    the process calls ``basicConfig``, so without this every ``logger.info``
    in ``qdrant_middleware`` falls through to logging's last-resort handler --
    which drops anything below WARNING. The auth label of each request and the
    effective ACL logged at startup would exist in the code and nowhere in the
    output. Scoped to this package rather than the root logger so enabling it
    does not also turn on INFO for httpx and qdrant-client.
    """
    package_logger = logging.getLogger("qdrant_middleware")
    if package_logger.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s:     %(message)s"))
    package_logger.addHandler(handler)
    package_logger.setLevel(logging.INFO)
    # The root logger has no handlers, so propagation would reach only the
    # last-resort handler; keeping it off avoids a duplicate line if a future
    # deployment does call basicConfig.
    package_logger.propagate = False


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create an independent FastAPI app with settings-bound runtime state."""
    configure_package_logging()
    resolved_settings = settings or get_settings()
    app = FastAPI(
        title="Smart Semantic Searcher",
        version="1.0.0",
        docs_url="/docs" if resolved_settings.expose_middleware_swagger else None,
        redoc_url="/redoc" if resolved_settings.expose_middleware_swagger else None,
        openapi_url=(
            "/openapi.json" if resolved_settings.expose_middleware_swagger else None
        ),
        lifespan=lifespan,
    )

    @app.exception_handler(RequestValidationError)
    async def _validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content=jsonable_encoder({"detail": exc.errors()}),
        )

    app.state.settings = resolved_settings
    app.state.runtime = create_runtime_state(resolved_settings)
    app.state.vector_reindex_manager = VectorReindexManager(app.state.runtime.mutations)
    app.state.embedder = None
    app.state.embedder_lock = threading.Lock()
    protected = [Depends(require_bearer_auth), Depends(require_operational_readiness)]
    app.include_router(collections.router, dependencies=protected)
    app.include_router(points.router, dependencies=protected)
    app.include_router(query.router, dependencies=protected)
    app.include_router(monitoring.router)

    if resolved_settings.expose_frontend:

        @app.middleware("http")
        async def _console_security_headers(request: Request, call_next) -> Response:
            # The console pages and assets get defense-in-depth headers; the JSON
            # API does not render as a document, so it does not need them.
            response = await call_next(request)
            path = request.url.path
            if path == "/" or path.startswith("/assets/"):
                for header, value in _CONSOLE_SECURITY_HEADERS.items():
                    response.headers.setdefault(header, value)
            return response

        @app.get("/assets/config.js", include_in_schema=False)
        async def _frontend_config() -> Response:
            # The token itself must never reach the browser unauthenticated:
            # the console only learns WHETHER auth is required, and the user
            # registers the key manually from the API KEY popover.
            config = {
                "authRequired": bool(resolved_settings.api_tokens),
                "queryDefaultLimit": resolved_settings.query_default_limit,
                "queryMaxLimit": resolved_settings.query_max_limit,
                "queryDefaultCandidateLimit": (
                    resolved_settings.query_default_candidate_limit
                ),
                "queryMaxCandidateLimit": resolved_settings.query_max_candidate_limit,
                "queryDefaultFusion": resolved_settings.query_default_fusion,
                "queryDefaultRerank": resolved_settings.query_default_rerank,
                "queryDefaultRerankTopK": (
                    resolved_settings.query_default_rerank_top_k
                ),
                "queryMaxRerankTopK": resolved_settings.query_max_rerank_top_k,
                "queryRerankMaxChars": resolved_settings.query_rerank_max_chars,
                "denseVectorName": resolved_settings.dense_vector_name,
                "sparseVectorName": resolved_settings.sparse_vector_name,
                "queryFilterMaxDepth": resolved_settings.query_filter_max_depth,
                "queryFilterMaxConditions": (
                    resolved_settings.query_filter_max_conditions
                ),
            }
            body = f"window.QHM_CONFIG={json.dumps(config)};"
            return Response(
                content=body,
                media_type="application/javascript",
                headers={"Cache-Control": "no-store"},
            )

        _frontend_dir = pathlib.Path(__file__).parent.parent / "frontend" / "dist"
        app.frontend("/", directory=str(_frontend_dir), check_dir=False)

    if not resolved_settings.expose_middleware_swagger:
        # FastAPI's frontend SPA fallback otherwise turns disabled docs paths
        # into UI responses. Keep those exact documentation surfaces hidden.

        @app.middleware("http")
        async def _hide_disabled_swagger(request: Request, call_next) -> Response:
            path = request.url.path
            if path in {"/docs", "/redoc", "/openapi.json"} or path.startswith(
                ("/docs/", "/redoc/")
            ):
                return Response(status_code=404)
            return await call_next(request)

    app.add_middleware(
        BodySizeLimitMiddleware,
        max_bytes=resolved_settings.request_body_max_bytes,
    )
    app.middleware("http")(rate_limit_middleware)
    app.middleware("http")(shutdown_middleware)
    app.add_middleware(TimeoutMetricsMiddleware)
    return app


def __getattr__(name: str):
    """Build ``app`` lazily on first access, then cache it (PEP 562).

    ``create_app()`` reads the process environment and requires either a
    non-empty token list or the explicit unauthenticated flag. Eagerly
    assigning ``app = create_app()`` here would run that validation at import
    time, which breaks every test that imports this module for ``create_app``
    even though those tests never touch ``app`` at all. Deferring
    construction to attribute access keeps ``uvicorn qdrant_middleware.main:app``
    working identically in the container, where the real environment is
    always set, without forcing every importer to pay for it.

    The built app is cached into the module's globals on success, so a
    second access finds ``app`` there directly and never re-enters
    ``__getattr__`` at all — module attribute lookup only falls back to
    ``__getattr__`` when the plain lookup misses. Without this, every access
    would call ``create_app()`` again, building a new app each time. A
    genuine configuration error still raises on every access, uncached: the
    exception happens before the assignment below runs, so there is nothing
    to cache and nothing is swallowed.
    """
    if name == "app":
        built = create_app()
        globals()["app"] = built
        return built
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
