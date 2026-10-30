"""Inbound bearer-token verification for static and OAuth deployments."""

from __future__ import annotations

import hmac
import logging
import time
from typing import Any

import httpx
from mcp.server.auth.provider import AccessToken, TokenVerifier
from mcp.server.auth.settings import AuthSettings

from smart_qdrant_mcp.config import Settings, url_host

logger = logging.getLogger(__name__)


def _constant_time_equals(presented: str, known: str) -> bool:
    """Compare two strings in constant time, whatever bytes they carry.

    ``hmac.compare_digest`` refuses ``str`` arguments that are not ASCII-only,
    raising ``TypeError`` rather than returning False. Both values compared in
    this module arrive from outside the process — a bearer token from an
    unauthenticated HTTP header (ASGI decodes headers latin-1, so any byte
    >= 0x80 lands here as a non-ASCII character) and an ``aud`` claim from the
    identity provider — so that exception is reachable by anyone who can reach
    the port, and it turns a plain authentication failure into a server error.

    Encoding both sides to UTF-8 first keeps the comparison constant-time (the
    bytes overload has no such restriction) while making every possible input a
    verdict instead of a crash.
    """
    return hmac.compare_digest(presented.encode("utf-8"), known.encode("utf-8"))


class StaticTokenVerifier(TokenVerifier):
    """Verify a bearer token against a fixed token-to-label mapping.

    The label becomes the ``client_id`` of the resulting access token, which is
    what gets logged: it gives per-person attribution and lets a single token be
    revoked without rotating the others.
    """

    def __init__(self, tokens: dict[str, str]) -> None:
        """Store an immutable copy of the token-to-label mapping."""
        self._tokens = dict(tokens)

    async def verify_token(self, token: str) -> AccessToken | None:
        """Return an access token for a known bearer value, otherwise None.

        Compares against every known token with ``hmac.compare_digest``
        rather than a plain ``dict`` lookup, so a wrong guess cannot be
        distinguished from a right one by how long the comparison took (a
        ``dict`` hash lookup is not constant-time, and neither is
        ``str.__eq__``, both of which can leak a token's length or a
        matching prefix through timing). Every known token is compared
        unconditionally -- the loop does not stop at the first match --
        so the total time depends only on the number of configured tokens,
        never on which one (if any) matched.

        Only the label is ever logged. Never the ``AccessToken`` object --
        its repr includes ``token``, which is plaintext -- and never the
        presented value on a failed lookup: logging *that* would defeat the
        entire point of authenticating by label instead of by raw secret.
        """
        matched_label: str | None = None
        for known_token, label in self._tokens.items():
            if _constant_time_equals(token, known_token):
                matched_label = label
        if matched_label is None:
            logger.info("token verification failed")
            return None
        logger.info("token verified: %s", matched_label)
        return AccessToken(token=token, client_id=matched_label, scopes=[])


class OAuthIntrospectionTokenVerifier(TokenVerifier):
    """Validate opaque OAuth access tokens through RFC 7662 introspection."""

    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """Build the introspection client from the OAuth settings.

        Args:
            settings: Source of the issuer, audience, introspection endpoint and
                the client credentials sent as HTTP Basic auth.
            transport: Optional httpx transport override, for tests that answer
                introspection without a network round trip.
        """
        self._issuer = str(settings.oauth_issuer_url)
        self._audience = settings.oauth_expected_audience
        self._introspection_url = str(settings.oauth_introspection_url)
        self._client = httpx.AsyncClient(
            timeout=settings.http_timeout,
            auth=httpx.BasicAuth(
                settings.oauth_client_id,
                settings.oauth_client_secret.get_secret_value(),
            ),
            transport=transport,
        )

    async def aclose(self) -> None:
        """Close the connection pool used for token introspection."""
        await self._client.aclose()

    async def verify_token(self, token: str) -> AccessToken | None:
        """Return a validated access token, failing closed on every anomaly."""
        try:
            response = await self._client.post(
                self._introspection_url,
                data={"token": token, "token_type_hint": "access_token"},
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError):
            logger.warning("OAuth token introspection failed")
            return None
        if not isinstance(payload, dict) or payload.get("active") is not True:
            logger.info("OAuth token is inactive")
            return None
        if payload.get("iss") not in (None, self._issuer):
            logger.warning("OAuth token issuer mismatch")
            return None
        if not _audience_contains(payload.get("aud"), self._audience):
            logger.warning("OAuth token audience mismatch")
            return None
        raw_expiry = payload.get("exp")
        expires_at = _optional_int(raw_expiry)
        if raw_expiry is not None and expires_at is None:
            logger.warning("OAuth introspection response has an invalid expiry")
            return None
        if expires_at is not None and expires_at <= int(time.time()):
            logger.info("OAuth token is expired")
            return None
        scopes = _scopes(payload.get("scope"))
        client_id = str(payload.get("client_id") or payload.get("sub") or "")
        if not client_id:
            logger.warning("OAuth introspection response has no principal")
            return None
        subject = str(payload["sub"]) if payload.get("sub") is not None else None
        logger.info("OAuth token verified: %s", client_id)
        return AccessToken(
            token=token,
            client_id=client_id,
            scopes=scopes,
            expires_at=expires_at,
            resource=self._audience,
            subject=subject,
            claims=_safe_claims(payload),
        )


def _audience_contains(raw: object, expected: str) -> bool:
    """Return whether an RFC-style string or array audience contains expected."""
    if isinstance(raw, str):
        return _constant_time_equals(raw, expected)
    if isinstance(raw, list):
        return any(
            isinstance(value, str) and _constant_time_equals(value, expected)
            for value in raw
        )
    return False


def _optional_int(value: object) -> int | None:
    """Parse an optional integer claim without accepting booleans."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _scopes(value: object) -> list[str]:
    """Normalize the RFC 7662 space-delimited scope member."""
    if isinstance(value, str):
        return [scope for scope in value.split() if scope]
    if isinstance(value, list):
        return [scope for scope in value if isinstance(scope, str) and scope]
    return []


def _safe_claims(payload: dict[str, Any]) -> dict[str, Any]:
    """Retain useful introspection claims without duplicating token material."""
    return {key: value for key, value in payload.items() if key != "token"}


def is_unauthenticated(settings: Settings) -> bool:
    """Return whether this configuration deliberately serves without auth."""
    return (
        settings.auth_mode == "static"
        and settings.allow_unauthenticated_local
        and not settings.api_tokens
    )


def build_token_verifier(
    settings: Settings,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> TokenVerifier | None:
    """Build the verifier selected by ``MCP_AUTH_MODE``, or none when open.

    ``StaticTokenVerifier({})`` would reject every request, including one
    carrying no token at all, so an empty ACL must install no verifier rather
    than an empty one -- otherwise the "authentication disabled" configuration
    would answer 401 to everything.
    """
    if is_unauthenticated(settings):
        return None
    if settings.auth_mode == "oauth":
        return OAuthIntrospectionTokenVerifier(settings, transport=transport)
    return StaticTokenVerifier(settings.api_tokens)


def build_auth_settings(settings: Settings) -> AuthSettings | None:
    """Configure authentication and publish metadata only for OAuth mode.

    Static tokens are provisioned out of band, so advertising OAuth discovery
    for that mode would send clients to an issuer that cannot authorize them.
    The SDK nevertheless requires an ``AuthSettings`` instance whenever a
    token verifier is installed, so static mode uses an inert local issuer and
    deliberately leaves ``resource_server_url`` unset. A deliberately open
    configuration installs no verifier, so it needs no ``AuthSettings`` either.
    """
    if is_unauthenticated(settings):
        return None
    if settings.auth_mode == "static":
        return AuthSettings(
            issuer_url=f"http://{url_host(settings.bind_host)}:{settings.port}",
            resource_server_url=None,
            required_scopes=[],
        )
    # The introspection verifier already enforces the expected audience, which
    # need not equal the resource URL the SDK's own check would compare against.
    return AuthSettings(
        issuer_url=settings.oauth_issuer_url,
        resource_server_url=settings.oauth_resource_server_url,
        required_scopes=settings.oauth_required_scopes,
        validate_token_resource=False,
    )
