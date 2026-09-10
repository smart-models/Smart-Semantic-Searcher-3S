"""Every middleware route must be reachable through a tool or a resource.

This is the test that keeps the coverage claim honest: adding an endpoint to the
middleware without exposing it here fails the suite.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from smart_qdrant_mcp.tool_map import RESOURCE_ONLY_ROUTES, TOOL_ROUTES

# Documentation and schema surfaces served by FastAPI itself, not middleware
# operations: they are not API endpoints and have nothing to expose over MCP.
# This set holds only genuinely framework-owned paths — see
# HIDDEN_MIDDLEWARE_ROUTES below for middleware-authored routes that merely
# opt out of the schema.
FRAMEWORK_PATHS = frozenset(
    {"/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"}
)

# Middleware-authored routes registered with include_in_schema=False. These
# are real endpoints the middleware owns, not framework surfaces — they are
# simply invisible to app.openapi(), and therefore to both coverage
# assertions below. Each entry here is a deliberate decision, not a silent
# omission: test_every_hidden_route_is_acknowledged fails the suite if the
# source acquires a hidden route not named here.
HIDDEN_MIDDLEWARE_ROUTES = frozenset(
    {
        # The static console's bootstrap config: tells a browser whether an
        # API token is required so it can prompt for one, and never leaks the
        # token itself (see qdrant_middleware/main.py). Deliberately not an
        # MCP surface — an agent has qdrant_check_health for "is this
        # deployment up," and no use for a browser-bootstrap payload.
        "/assets/config.js",
    }
)

# The only verbs that constitute a distinct operation. HEAD/OPTIONS are
# transport-level affordances FastAPI adds automatically alongside GET, not
# operations a client (or an MCP tool) would ever call directly.
_OPERATION_METHODS = frozenset({"GET", "POST", "PUT", "PATCH", "DELETE"})


def middleware_operations() -> set[tuple[str, str]]:
    """Return (METHOD, path) for every real middleware operation.

    In the installed FastAPI version (0.138), ``app.routes`` does not expose a
    flat list of ``APIRoute`` objects for routers added via
    ``include_router``: they are held behind a lazy ``_IncludedRouter``
    wrapper that ``app.routes`` does not resolve, so walking ``app.routes``
    and filtering ``isinstance(route, APIRoute)`` finds only the one route
    mounted directly on the app (``/assets/config.js``) and misses all 42
    router-mounted endpoint functions. ``app.openapi()`` forces FastAPI to
    resolve the full effective route table into its schema, which is the
    reliable way to enumerate everything actually served. It also already
    excludes the framework's own docs/schema surfaces and HEAD/OPTIONS, but
    both exclusions are made explicit below anyway so this test documents,
    rather than silently relies on, that behavior.

    ``app.openapi()`` has one blind spot of its own: a route registered with
    ``include_in_schema=False`` never enters ``schema["paths"]`` at all, so it
    is invisible here regardless of ``FRAMEWORK_PATHS``. That is not a
    framework surface being filtered out — it is a real middleware endpoint
    the schema simply cannot see. ``test_every_hidden_route_is_acknowledged``
    below closes that gap by scanning source instead of the schema.

    Stacked decorators register one Route per HTTP method, so PUT and PATCH
    on /collections/{n}/schema-toggles arrive as two routes over one endpoint
    function. Both are kept here — the tool map lists both as well — but the
    deduplicated *operation* count is what the arithmetic assertion below
    uses.
    """
    from qdrant_middleware.main import create_app

    app = create_app()
    schema = app.openapi()
    operations: set[tuple[str, str]] = set()
    for path, methods_by_verb in schema["paths"].items():
        if path in FRAMEWORK_PATHS:
            continue
        for verb in methods_by_verb:
            method = verb.upper()
            if method not in _OPERATION_METHODS:
                continue
            operations.add((method, path))
    return operations


def declared_routes() -> set[tuple[str, str]]:
    """Return every (METHOD, path) the MCP server claims to cover."""
    covered = {pair for pairs in TOOL_ROUTES.values() for pair in pairs}
    return covered | set(RESOURCE_ONLY_ROUTES)


def test_no_middleware_operation_is_unexposed():
    """Every real middleware operation must have a tool or a resource."""
    missing = middleware_operations() - declared_routes()
    assert not missing, (
        "these middleware operations have no tool and no resource: "
        f"{sorted(missing)}. Add a tool, or add the route to "
        "RESOURCE_ONLY_ROUTES with a comment saying why."
    )


def test_no_declared_route_is_stale():
    """Nothing in the map may reference a route the middleware no longer has."""
    stale = declared_routes() - middleware_operations()
    assert not stale, (
        f"these declared routes no longer exist in the middleware: {sorted(stale)}"
    )


# Tool input fields that intentionally diverge from the middleware body schema's
# field names, or that the tool wraps in a different shape. Each entry maps a
# (middleware_schema_name, required_field_name) to the tool input field that
# covers it, or to None when the tool's input model shape intentionally doesn't
# mirror the body schema for that field. When the middleware adds a NEW required
# field to a body schema, it won't be in this dict, and
# test_every_middleware_required_body_field_is_known_to_a_tool will fail — which
# is exactly the class of drift this test exists to catch (the audit called it
# G3: a field-level change that the route-level coverage test above cannot see).
SCHEMA_FIELD_OVERRIDES: dict[tuple[str, str], str | None] = {
    # The tool calls it "confirm"; the middleware body field is
    # "confirm_collection". The tool code maps confirm -> confirm_collection
    # when building the request body (see tools/reindex.py).
    ("ReindexStartRequest", "confirm_collection"): "confirm",
    # UpsertPointRequest (POST /points, single-point upsert) requires point_id
    # and text as top-level body fields. The tool's UpsertPointsInput instead
    # takes a "points" list (the batch shape from POST /points/batch) and
    # dispatches to POST /points or POST /points/batch based on list length.
    # The tool covers these fields through its points list items, not as
    # top-level input fields — a deliberate abstraction that keeps the tool's
    # surface uniform regardless of upsert cardinality.
    ("UpsertPointRequest", "point_id"): None,
    ("UpsertPointRequest", "text"): None,
}


async def test_every_middleware_required_body_field_is_known_to_a_tool(mcp_client):
    """Every required field in a middleware request body schema must be known.

    This is the schema-level complement to
    test_no_middleware_operation_is_unexposed: that test catches a whole
    endpoint disappearing from the tool surface; this one catches a new
    required field appearing in an existing endpoint's body schema without
    the covering tool's input model knowing about it. Without it, the
    middleware could add a required field to a request body and every tool
    would silently keep sending the old shape, getting 422s at runtime with
    no test-time signal.

    For each (METHOD, path) that has a JSON request body, the middleware
    OpenAPI schema's ``required`` list is compared against the properties of
    the tool's inner input model (the Pydantic model nested under ``params``
    in the tool's input schema). Every required middleware field must appear
    as a property in that model, unless it is listed in
    SCHEMA_FIELD_OVERRIDES — the explicit allowlist for intentional
    field-name mappings and shape divergences.
    """
    from qdrant_middleware.main import create_app

    app = create_app()
    openapi = app.openapi()
    components = openapi["components"]["schemas"]

    # Reverse map: (METHOD, path) -> tool_name, for body-bearing routes only.
    route_to_tool: dict[tuple[str, str], str] = {}
    for tool_name, routes in TOOL_ROUTES.items():
        for method, path in routes:
            route_to_tool[(method, path)] = tool_name

    tools = {tool.name: tool for tool in (await mcp_client.list_tools()).tools}

    missing: list[str] = []
    for path, methods in openapi["paths"].items():
        if path in FRAMEWORK_PATHS:
            continue
        for verb, op in methods.items():
            method = verb.upper()
            if method not in _OPERATION_METHODS:
                continue
            body = op.get("requestBody", {})
            if not body:
                continue
            body_schema = (
                body.get("content", {}).get("application/json", {}).get("schema", {})
            )
            ref = body_schema.get("$ref", "")
            if not ref:
                continue
            schema_name = ref.split("/")[-1]
            required = components.get(schema_name, {}).get("required", [])
            if not required:
                continue

            tool_name = route_to_tool.get((method, path))
            if tool_name is None:
                # Resource-only or otherwise not tool-covered;
                # test_no_middleware_operation_is_unexposed handles that.
                continue

            tool = tools.get(tool_name)
            if tool is None:
                missing.append(f"{method} {path}: tool {tool_name} not registered")
                continue

            params_ref = (
                tool.input_schema.get("properties", {})
                .get("params", {})
                .get("$ref", "")
            )
            if not params_ref:
                continue
            model_name = params_ref.split("/")[-1]
            model_def = tool.input_schema.get("$defs", {}).get(model_name, {})
            tool_fields = set(model_def.get("properties", {}).keys())

            for field in required:
                override = SCHEMA_FIELD_OVERRIDES.get((schema_name, field), field)
                if override is None:
                    continue
                if override not in tool_fields:
                    missing.append(
                        f"{method} {path}: middleware body schema "
                        f"{schema_name} requires field {field!r} but tool "
                        f"{tool_name}'s input model {model_name} does not "
                        f"declare it"
                    )

    assert not missing, (
        "these required fields in middleware request body schemas are not "
        "declared in the covering tool's input model (if the field is "
        "intentionally renamed or wrapped differently, add it to "
        "SCHEMA_FIELD_OVERRIDES with a comment): " + "; ".join(missing)
    )


def test_operation_count_matches_the_verified_surface():
    """Pin the numbers verified against the live middleware.

    43 (method, path) pairs over 35 distinct paths. Eight paths answer two
    methods each; one of those eight — /collections/{n}/schema-toggles — is a
    single stacked-decorator handler, so the 43 pairs correspond to 42
    distinct endpoint functions.
    """
    pairs = middleware_operations()
    assert len(pairs) == 43
    assert len({path for _method, path in pairs}) == 35


@pytest.mark.parametrize("name", sorted(TOOL_ROUTES))
def test_tool_names_are_prefixed_and_snake_case(name):
    """Every declared tool name follows the qdrant_ / snake_case convention."""
    assert name.startswith("qdrant_")
    assert name == name.lower()
    assert " " not in name and "-" not in name


class UnresolvableHiddenRouteError(AssertionError):
    """A schema-hiding decorator's path could not be resolved to a literal.

    Raised instead of silently skipping the decorator: a route registered
    with ``include_in_schema=False`` whose path this scanner cannot read is
    exactly as invisible to coverage as one it never noticed at all. The
    original scanner only recorded a hidden path when the decorator's first
    positional argument was an ``ast.Constant``; anything else (a keyword
    ``path=`` argument, or a non-literal expression such as a module-level
    name) fell through every check silently, so the route escaped
    ``test_every_hidden_route_is_acknowledged`` -- and therefore both
    coverage assertions above it -- with no failure at all. "Cannot resolve
    this" must fail loudly, not count as "nothing hidden here."
    """


def _has_false_keyword(call: ast.Call, name: str) -> bool:
    """Return True if ``call`` passes ``name=False`` as a literal keyword."""
    return any(
        keyword.arg == name
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value is False
        for keyword in call.keywords
    )


def _literal_path_argument(call: ast.Call) -> str | None:
    """Return a route decorator's literal path string, or None if unresolvable.

    Checks the first positional argument first (``@app.get("/x", ...)``),
    then the ``path=`` keyword (``@app.get(path="/x", ...)``) -- the original
    scanner only ever checked the former, so a hidden route declared with
    ``path=`` as a keyword was invisible to it. Returns None both when no
    path-shaped argument exists and when one exists but is not a string
    literal (e.g. a module-level constant); the caller must treat both cases
    as unresolvable, not as "no hidden path here."
    """
    if call.args:
        first = call.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            return first.value
        return None
    for keyword in call.keywords:
        if keyword.arg == "path":
            value = keyword.value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                return value.value
            return None
    return None


def _call_target_name(call: ast.Call) -> str | None:
    """Return a call's callee name.

    Handles both bound calls (``app.include_router``, attribute access) and
    bare ones (``APIRouter(...)``, a plain name).
    """
    func = call.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def _hidden_routes_from_source() -> set[str]:
    """Return every route/router hidden from the schema via include_in_schema=False.

    This walks the AST of every module in the qdrant_middleware package
    rather than any FastAPI route table, on purpose: app.openapi() cannot see
    these routes at all (that is what makes them hidden), and app.routes
    cannot be trusted either — this FastAPI version resolves include_router
    output lazily behind a private _IncludedRouter wrapper (see
    middleware_operations()'s docstring). Python's own parser is the one
    thing here that isn't tied to a FastAPI version's internal route
    representation, so it is the stable way to find every
    include_in_schema=False occurrence regardless of how FastAPI stores it
    at runtime.

    Three shapes are covered, all as ``include_in_schema=False``: a route
    decorator (``@app.get(...)`` / ``@router.get(...)``, keyed by its literal
    path — see ``_literal_path_argument`` for why an unresolvable path is a
    hard failure rather than a skip), an ``include_router(...)`` call (which
    can hide an entire router's worth of endpoints in one place), and an
    ``APIRouter(...)`` constructor call (which hides every route ever added
    to that router, at its point of creation). The latter two have no single
    "path" to key on, so they are recorded as a marker naming the call site.
    """
    import qdrant_middleware

    package_dir = Path(qdrant_middleware.__file__).parent
    hidden: set[str] = set()
    for module_path in package_dir.rglob("*.py"):
        tree = ast.parse(
            module_path.read_text(encoding="utf-8"), filename=str(module_path)
        )
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for decorator in node.decorator_list:
                    if not isinstance(decorator, ast.Call):
                        continue
                    if not _has_false_keyword(decorator, "include_in_schema"):
                        continue
                    path = _literal_path_argument(decorator)
                    if path is None:
                        raise UnresolvableHiddenRouteError(
                            f"{module_path}:{decorator.lineno}: a route is "
                            "registered with include_in_schema=False but its "
                            "path is not a literal string constant, so this "
                            "scanner cannot verify it is acknowledged in "
                            "HIDDEN_MIDDLEWARE_ROUTES. Express the path as a "
                            "literal, or extend this scanner to resolve it "
                            "explicitly."
                        )
                    hidden.add(path)
                continue
            if isinstance(node, ast.Call) and _has_false_keyword(
                node, "include_in_schema"
            ):
                target = _call_target_name(node)
                if target in ("include_router", "APIRouter"):
                    location = module_path.relative_to(package_dir.parent)
                    hidden.add(f"<{target}:include_in_schema=False> {location}")
    return hidden


def test_every_hidden_route_is_acknowledged():
    """A route hidden from the schema must be a decision, not an omission.

    include_in_schema=False routes are invisible to both
    test_no_middleware_operation_is_unexposed and
    test_no_declared_route_is_stale, since both read app.openapi(). Without
    this test, a future hidden route would silently escape coverage
    entirely — exactly the gap this whole file exists to close. This forces
    every such route to be named in HIDDEN_MIDDLEWARE_ROUTES with a reason,
    the same way a covered route must be named in TOOL_ROUTES or
    RESOURCE_ONLY_ROUTES.
    """
    found = _hidden_routes_from_source()
    unacknowledged = found - HIDDEN_MIDDLEWARE_ROUTES
    assert not unacknowledged, (
        "these routes are registered with include_in_schema=False, which "
        "makes them invisible to app.openapi() and therefore to the "
        f"coverage tests above, but are not in HIDDEN_MIDDLEWARE_ROUTES: "
        f"{sorted(unacknowledged)}. Expose them through a tool, add them to "
        "RESOURCE_ONLY_ROUTES, or add them to HIDDEN_MIDDLEWARE_ROUTES with "
        "a comment explaining why they are deliberately hidden."
    )
    stale = HIDDEN_MIDDLEWARE_ROUTES - found
    assert not stale, (
        "HIDDEN_MIDDLEWARE_ROUTES acknowledges routes that are no longer "
        f"registered with include_in_schema=False in the middleware source: "
        f"{sorted(stale)}"
    )
