"""Cross-service bearer-token preflight for the supported launchers.

No single container sees every token list, so the invariants that span
services are checked here, where the launchers have already resolved ``.env``.
This script never parses ``.env`` itself: the launcher resolves the values and
sends them as ``NAME=VALUE`` lines on stdin, so the repository keeps one answer
on quoting, comments and blank handling. Stdin rather than arguments because
these values are bearer tokens and process arguments are world-readable through
``ps``.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Mapping

LABEL_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
RESERVED_LABEL_PREFIX = "@"
PUBLISH_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1"})
REMOVED_ENV_NAMES = {
    "MIDDLEWARE_API_TOKEN": "MIDDLEWARE_API_TOKENS or MIDDLEWARE_CLIENT_TOKEN",
    "SMART_EMBEDDER_API_TOKEN": (
        "SMART_EMBEDDER_API_TOKENS or SMART_EMBEDDER_CLIENT_TOKEN"
    ),
    "ALLOW_UNAUTHENTICATED_LOCAL": "SMART_EMBEDDER_ALLOW_UNAUTHENTICATED_LOCAL",
}
INBOUND_LISTS = {
    "middleware": "MIDDLEWARE_API_TOKENS",
    "smart-embedder": "SMART_EMBEDDER_API_TOKENS",
    "mcp": "MCP_API_TOKENS",
}
# Which client token each service auto-accepts from its caller.
SERVICE_CLIENT_TOKENS = {
    "middleware": "MIDDLEWARE_CLIENT_TOKEN",
    "smart-embedder": "SMART_EMBEDDER_CLIENT_TOKEN",
}
# The opt-out each service's own startup check accepts in place of a
# non-empty inbound token list. Mirrors qdrant_middleware/core/config.py,
# smart-embedder.py and smart_qdrant_mcp/config.py: an empty list without this
# flag set is a boot-time crash there, not just a warning here.
SERVICE_UNAUTH_FLAGS = {
    "middleware": "MIDDLEWARE_ALLOW_UNAUTHENTICATED_LOCAL",
    "smart-embedder": "SMART_EMBEDDER_ALLOW_UNAUTHENTICATED_LOCAL",
    "mcp": "MCP_ALLOW_UNAUTHENTICATED_LOCAL",
}


def _is_true(value: str) -> bool:
    """Return whether an environment string means true."""
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse(raw: str, variable: str, errors: list[str]) -> set[str]:
    """Return the tokens in ``raw``, appending a message per malformed entry."""
    tokens: set[str] = set()
    for entry in (part.strip() for part in raw.split(",")):
        if not entry:
            continue
        token, separator, label = (piece.strip() for piece in entry.partition(":"))
        if not separator or not token or not label:
            errors.append(f"{variable} has an entry that is not token:label")
            continue
        if label.startswith(RESERVED_LABEL_PREFIX):
            errors.append(
                f"{variable} label {label!r} must not start with "
                f"{RESERVED_LABEL_PREFIX!r}; that prefix is reserved"
            )
            continue
        if not LABEL_PATTERN.match(label):
            errors.append(f"{variable} label {label!r} must match [A-Za-z0-9._-]+")
            continue
        if token in tokens:
            errors.append(f"{variable} contains a duplicate token")
            continue
        tokens.add(token)
    return tokens


def _is_publish_loopback(host: str) -> bool:
    """Return whether a published address stays on the loopback interface."""
    normalized = host.strip().lower()
    return normalized in PUBLISH_LOOPBACK_HOSTS or normalized.startswith("127.")


def check_token_config(values: Mapping[str, str]) -> list[str]:
    """Return every configuration error found across the three services.

    Args:
        values: Resolved environment values, as the launchers computed them.

    Returns:
        One message per problem, empty when the configuration is usable. Token
        values never appear in a message.
    """
    errors: list[str] = []
    for name, replacement in REMOVED_ENV_NAMES.items():
        if values.get(name, "").strip():
            errors.append(f"{name} was removed -- use {replacement}")

    mcp_enabled = _is_true(values.get("MCP_SERVER_ENABLED", "true"))
    embedder_only = _is_true(values.get("EMBEDDER_ONLY", "false"))

    acls: dict[str, set[str]] = {}
    for service, variable in INBOUND_LISTS.items():
        acls[service] = _parse(values.get(variable, ""), variable, errors)
    for service, variable in SERVICE_CLIENT_TOKENS.items():
        client_token = values.get(variable, "").strip()
        if client_token:
            acls[service].add(client_token)

    services = sorted(acls)
    for index, first in enumerate(services):
        for second in services[index + 1 :]:
            shared = acls[first] & acls[second]
            if shared:
                errors.append(
                    f"token overlap: {len(shared)} credential(s) open both "
                    f"{first} and {second}; every service needs its own"
                )

    if mcp_enabled and not embedder_only:
        if acls["middleware"] and not values.get("MIDDLEWARE_CLIENT_TOKEN", "").strip():
            errors.append(
                "MIDDLEWARE_CLIENT_TOKEN is empty while MIDDLEWARE_API_TOKENS "
                "requires authentication; the MCP server would get HTTP 401"
            )
    if not embedder_only:
        if (
            acls["smart-embedder"]
            and not values.get("SMART_EMBEDDER_CLIENT_TOKEN", "").strip()
        ):
            errors.append(
                "SMART_EMBEDDER_CLIENT_TOKEN is empty while "
                "SMART_EMBEDDER_API_TOKENS requires authentication; the "
                "middleware would get HTTP 401"
            )

    # Each service refuses to start at boot when its effective inbound ACL is
    # empty and its *_ALLOW_UNAUTHENTICATED_LOCAL flag is not true. The ACL is
    # the parsed *_API_TOKENS list plus the *_CLIENT_TOKEN its caller presents
    # (already folded into ``acls`` above), so a client token alone is enough to
    # boot. The MCP server skips this rule under MCP_AUTH_MODE=oauth -- there the
    # token list is ignored -- so it is the one mode where the check does not
    # apply. The comparison is case-sensitive, matching the service's
    # Literal["static", "oauth"] field; any other value (including a typo) would
    # crash the service at boot, so flagging an empty list there is correct. The
    # launchers feed MCP_AUTH_MODE alongside the other names so this branch sees
    # the real value; an unset/empty value means the static default, which is the
    # mode where the rule does apply.
    mcp_auth_mode = values.get("MCP_AUTH_MODE", "static").strip() or "static"
    active_services = {"smart-embedder"}
    if not embedder_only:
        active_services.add("middleware")
        if mcp_enabled and mcp_auth_mode != "oauth":
            active_services.add("mcp")

    for service in sorted(active_services):
        if acls[service]:
            continue
        flag = SERVICE_UNAUTH_FLAGS[service]
        if not _is_true(values.get(flag, "false")):
            errors.append(
                f"{INBOUND_LISTS[service]} is empty and {flag} is not true; "
                f"{service} would refuse to start"
            )

    if _is_true(values.get("MCP_ALLOW_UNAUTHENTICATED_LOCAL", "false")):
        bind_host = values.get("MCP_BIND_HOST", "127.0.0.1")
        if not _is_publish_loopback(bind_host):
            errors.append(
                "MCP_ALLOW_UNAUTHENTICATED_LOCAL=true requires a loopback "
                f"MCP_BIND_HOST; refusing to publish unauthenticated on {bind_host}"
            )
    return errors


def main() -> int:
    """Validate the ``NAME=VALUE`` lines on stdin and report every problem.

    A value may not contain a newline; the launchers only ever send values read
    from ``.env``, where a newline is not expressible either.
    """
    values: dict[str, str] = {}
    for line in sys.stdin.read().splitlines():
        if not line.strip():
            continue
        name, separator, value = line.partition("=")
        if not separator or not name.strip():
            print("[ERROR] stdin expects NAME=VALUE lines", file=sys.stderr)
            return 2
        values[name.strip()] = value
    errors = check_token_config(values)
    for error in errors:
        print(f"[ERROR] {error}", file=sys.stderr)
    return 2 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
