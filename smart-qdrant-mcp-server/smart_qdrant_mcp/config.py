"""Configuration for the smart_qdrant_mcp server. Single source of every MCP_* var."""

from __future__ import annotations

import ipaddress
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import (
    AnyHttpUrl,
    Field,
    SecretStr,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

LOOPBACK = "127.0.0.1"
# Publish addresses are literal IPs, so the hostname "localhost" is not a
# member here -- unlike the middleware's LOOPBACK_HOSTS, which describes a
# bind host. Mirrors scripts/verify_compose_security.py.
PUBLISH_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1"})
PRIVATE_IPV4_NETWORKS = tuple(
    ipaddress.IPv4Network(cidr)
    for cidr in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")
)
# smart-qdrant-mcp-server/smart_qdrant_mcp/config.py -> repository root
REPO_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class ConfigError(RuntimeError):
    """Raised when the environment cannot produce a usable configuration."""


REMOVED_ENV_NAMES = {
    "MIDDLEWARE_API_TOKEN": (
        "MIDDLEWARE_API_TOKENS (inbound list) or MIDDLEWARE_CLIENT_TOKEN "
        "(the credential this server presents to the middleware)"
    ),
    "SMART_EMBEDDER_API_TOKEN": (
        "SMART_EMBEDDER_API_TOKENS (inbound list) or SMART_EMBEDDER_CLIENT_TOKEN "
        "(the credential the middleware presents to the embedder)"
    ),
    "ALLOW_UNAUTHENTICATED_LOCAL": "SMART_EMBEDDER_ALLOW_UNAUTHENTICATED_LOCAL",
}


def _names_in_env_file(env_file: Path) -> set[str]:
    """Return the removed variable names assigned in ``env_file``."""
    if not env_file.is_file():
        return set()
    found: set[str] = set()
    for line in env_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export ") :].lstrip()
        name, separator, value = stripped.partition("=")
        # An empty assignment is ignored, so `MIDDLEWARE_API_TOKEN=` left
        # behind in a migrated .env is not an error. check_token_config in
        # scripts/verify_token_config.py tests the value the same way; the
        # two must agree or the preflight and the boot disagree on one file.
        if separator and value.strip() and name.strip() in REMOVED_ENV_NAMES:
            found.add(name.strip())
    return found


def reject_removed_env_names(env_file: Path | None) -> None:
    """Refuse to start when a variable removed by the multi-token change is set.

    Args:
        env_file: The ``.env`` file this process would read, or ``None`` to scan
            the process environment only.

    Raises:
        ConfigError: If any removed name is present, naming its replacement.
            Silently ignoring one would produce a server whose actual auth state
            contradicts what the operator configured.
    """
    found = {name for name in REMOVED_ENV_NAMES if os.environ.get(name, "").strip()}
    if env_file is not None:
        found |= _names_in_env_file(env_file)
    if found:
        details = "; ".join(
            f"{name} was removed -- use {REMOVED_ENV_NAMES[name]}"
            for name in sorted(found)
        )
        raise ConfigError(details)


LABEL_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
RESERVED_LABEL_PREFIX = "@"


def url_host(host: str) -> str:
    """Format an IP literal safely for a URL authority or Host header."""
    raw = host.strip()
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1].strip()
    try:
        address = ipaddress.ip_address(raw)
    except ValueError:
        return host
    return f"[{address}]" if address.version == 6 else str(address)


def is_publish_loopback(host: str) -> bool:
    """Return whether a published address stays on the loopback interface."""
    raw = host.strip().strip("[]")
    try:
        return ipaddress.ip_address(raw).is_loopback
    except ValueError:
        return False


def is_wildcard_bind(host: str) -> bool:
    """Return whether a published address binds every interface.

    Compares semantically with ``ipaddress`` rather than against a literal
    ``{"0.0.0.0", "::"}`` set, so the alternate spellings of the unspecified
    address that slip past a textual match are refused: bracketed (``[::]``),
    compressed (``::0``), whitespace-padded, and the IPv4-mapped
    ``::ffff:0.0.0.0`` (which still names the unspecified IPv4 address). Values
    that are not syntactically valid IP literals (hostnames) are not wildcards
    and are handled by the other checks.
    """
    text = host.strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1].strip()
    try:
        address = ipaddress.ip_address(text)
    except ValueError:
        return False
    if address.is_unspecified:
        return True
    # An IPv4-mapped IPv6 address names an IPv4 host; mapped 0.0.0.0 is still
    # the unspecified address under a v6 spelling.
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        return address.ipv4_mapped.is_unspecified
    return False


def is_allowed_publish_address(host: str) -> bool:
    """Return whether MCP may be published on this literal host address."""
    text = host.strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1].strip()
    try:
        address = ipaddress.ip_address(text)
    except ValueError:
        return False
    if address.is_loopback:
        return True
    if not isinstance(address, ipaddress.IPv4Address):
        return False
    return any(address in network for network in PRIVATE_IPV4_NETWORKS)


def is_oauth_loopback_url(url: AnyHttpUrl) -> bool:
    """Return whether an OAuth URL's host is an explicit loopback target."""
    host = str(url.host or "").strip().lower()
    if host == "localhost":
        return True
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def parse_api_tokens(raw: str, variable: str = "MCP_API_TOKENS") -> dict[str, str]:
    """Parse ``token:label,token:label`` into a mapping of token to label.

    Args:
        raw: The raw environment value.
        variable: Name of the variable being parsed, used in error messages so
            each service reports its own name.

    Returns:
        Every token mapped to its audit label, in declaration order.

    Raises:
        ConfigError: On a malformed entry, an out-of-charset label, a label
            using the reserved prefix, or a duplicate token. The token value is
            never included in the message.
    """
    tokens: dict[str, str] = {}
    for entry in (part.strip() for part in raw.split(",")):
        if not entry:
            continue
        token, separator, label = (piece.strip() for piece in entry.partition(":"))
        if not separator or not token or not label:
            raise ConfigError(f"{variable} has an entry that is not token:label")
        if label.startswith(RESERVED_LABEL_PREFIX):
            raise ConfigError(
                f"{variable} label {label!r} must not start with "
                f"{RESERVED_LABEL_PREFIX!r}; that prefix is reserved for the "
                "credentials services present to each other"
            )
        if not LABEL_PATTERN.match(label):
            raise ConfigError(
                f"{variable} label {label!r} must match [A-Za-z0-9._-]+. A label "
                "containing ':' usually means the token itself contains ':', "
                "which this format cannot express"
            )
        if token in tokens:
            raise ConfigError(f"{variable} contains a duplicate token")
        tokens[token] = label
    return tokens


class Settings(BaseSettings):
    """Runtime configuration, read once at process start."""

    # The repository-root .env is read the same way the middleware reads it
    # (core/config.py:34-40). In the container the values arrive from the
    # compose environment and no .env file is present, which is fine: explicit
    # environment variables win over the file either way.
    model_config = SettingsConfigDict(
        env_file=REPO_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
        case_sensitive=False,
    )

    api_tokens_raw: str = Field(default="", alias="MCP_API_TOKENS")
    auth_mode: Literal["static", "oauth"] = Field(
        default="static", alias="MCP_AUTH_MODE"
    )
    oauth_issuer_url: AnyHttpUrl | None = Field(
        default=None, alias="MCP_OAUTH_ISSUER_URL"
    )
    oauth_resource_server_url: AnyHttpUrl | None = Field(
        default=None, alias="MCP_OAUTH_RESOURCE_SERVER_URL"
    )
    oauth_introspection_url: AnyHttpUrl | None = Field(
        default=None, alias="MCP_OAUTH_INTROSPECTION_URL"
    )
    oauth_client_id: str = Field(default="", alias="MCP_OAUTH_CLIENT_ID")
    oauth_client_secret: SecretStr = Field(
        default=SecretStr(""), alias="MCP_OAUTH_CLIENT_SECRET"
    )
    oauth_audience: str = Field(default="", alias="MCP_OAUTH_AUDIENCE")
    oauth_required_scopes_raw: str = Field(
        default="", alias="MCP_OAUTH_REQUIRED_SCOPES"
    )
    # Loopback by default so local development works without compose; the
    # container overrides this with the private-network hostname.
    middleware_url: str = Field(
        default="http://127.0.0.1:6336", alias="MCP_MIDDLEWARE_URL"
    )
    middleware_client_token: str = Field(default="", alias="MIDDLEWARE_CLIENT_TOKEN")
    # The MCP server never calls the embedder. This is read only so startup
    # validation can refuse a token that would open two services at once.
    smart_embedder_client_token: str = Field(
        default="", alias="SMART_EMBEDDER_CLIENT_TOKEN"
    )
    allow_unauthenticated_local: bool = Field(
        default=False, alias="MCP_ALLOW_UNAUTHENTICATED_LOCAL"
    )
    # The published/policed address: what compose publishes on, what
    # scripts/verify_compose_security.py checks, and what seeds the Host
    # allowlist and the issuer/resource URL. It never reaches a socket
    # itself -- see listen_host below for the setting that does.
    bind_host: str = Field(default=LOOPBACK, alias="MCP_BIND_HOST")
    # The address uvicorn actually binds inside this process's own network
    # namespace. Defaults to loopback so running outside Docker (where there
    # is no container namespace or published-port restriction to lean on) is
    # safe by default. Unlike bind_host, a wildcard here is not refused: the
    # compose service sets this to 0.0.0.0 deliberately, because inside its
    # own private namespace binding all interfaces is exactly what the
    # published-port restriction assumes it does.
    listen_host: str = Field(default=LOOPBACK, alias="MCP_LISTEN_HOST")
    port: int = Field(default=6337, ge=1, le=65535, alias="MCP_PORT")
    allowed_hosts_raw: str = Field(default="", alias="MCP_ALLOWED_HOSTS")
    allowed_origins_raw: str = Field(default="", alias="MCP_ALLOWED_ORIGINS")
    http_timeout: float = Field(default=120.0, gt=0, alias="MCP_HTTP_TIMEOUT")
    retry_attempts: int = Field(default=3, ge=1, le=10, alias="MCP_RETRY_ATTEMPTS")
    retry_backoff: float = Field(default=0.5, gt=0, alias="MCP_RETRY_BACKOFF_SECONDS")

    @field_validator(
        "oauth_issuer_url",
        "oauth_resource_server_url",
        "oauth_introspection_url",
        mode="before",
    )
    @classmethod
    def blank_url_is_unset(cls, value: object) -> object:
        """Treat blank optional OAuth URL variables as absent."""
        return None if isinstance(value, str) and not value.strip() else value

    @field_validator("middleware_url")
    @classmethod
    def strip_trailing_slash(cls, value: str) -> str:
        """Normalise the base URL so path joining is unambiguous."""
        return value.rstrip("/")

    @field_validator("listen_host")
    @classmethod
    def reject_blank_listen_host(cls, value: str) -> str:
        """A blank value would pass an empty string to uvicorn.run(host=...).

        Wildcards are deliberately still accepted here -- see the field's
        comment -- this only guards against the one value that would make
        the socket bind unpredictable rather than merely permissive.
        """
        if not value.strip():
            raise ConfigError("MCP_LISTEN_HOST must not be blank")
        return value

    @field_validator("bind_host")
    @classmethod
    def normalize_bind_host(cls, value: str) -> str:
        """Strip the published address and reject an empty value."""
        normalized = value.strip()
        if not normalized:
            raise ConfigError("MCP_BIND_HOST must not be blank")
        return normalized

    @property
    def api_tokens(self) -> dict[str, str]:
        """Inbound bearer tokens mapped to their audit labels."""
        return parse_api_tokens(self.api_tokens_raw)

    @property
    def oauth_required_scopes(self) -> list[str]:
        """Scopes every OAuth access token must carry."""
        return [
            scope.strip()
            for scope in self.oauth_required_scopes_raw.split(",")
            if scope.strip()
        ]

    @property
    def oauth_expected_audience(self) -> str:
        """Audience required in an active introspection response."""
        if self.oauth_audience:
            return self.oauth_audience
        return str(self.oauth_resource_server_url or "")

    @property
    def allowed_hosts(self) -> list[str]:
        """Host allowlist for transport security, always including the bind host.

        A real client sends the port in the Host header, and the SDK matches
        that header exactly or against a ``name:*`` wildcard — a bare name
        never matches ``127.0.0.1:6337``. Each name is therefore published in
        both forms. The wildcard is the right granularity here: this check
        defends against DNS rebinding, which turns on the name, not the port,
        and the port that is reachable at all is already fixed by the compose
        publish and by MCP_LISTEN_HOST.
        """
        extra = [h.strip() for h in self.allowed_hosts_raw.split(",") if h.strip()]
        names = {LOOPBACK, "localhost", url_host(self.bind_host), *extra}
        allowed = {name for name in names if name}
        allowed |= {f"{name}:*" for name in allowed if not name.endswith(":*")}
        return sorted(allowed)

    @property
    def allowed_origins(self) -> list[str]:
        """Origin allowlist; empty means only same-origin requests are accepted."""
        return [o.strip() for o in self.allowed_origins_raw.split(",") if o.strip()]

    @model_validator(mode="after")
    def validate_startup(self) -> Settings:
        """Fail fast on any configuration that would be unsafe to serve."""
        tokens = parse_api_tokens(self.api_tokens_raw)
        if is_wildcard_bind(self.bind_host):
            raise ConfigError("MCP_BIND_HOST must not bind all interfaces")
        if not is_allowed_publish_address(self.bind_host):
            raise ConfigError(
                "MCP_BIND_HOST must be a loopback or private RFC 1918 IPv4 "
                "literal; public addresses and hostnames are not supported"
            )
        if self.auth_mode == "static" and not tokens:
            if not self.allow_unauthenticated_local:
                raise ConfigError(
                    "MCP_API_TOKENS must define at least one token:label entry "
                    "when MCP_AUTH_MODE=static, unless "
                    "MCP_ALLOW_UNAUTHENTICATED_LOCAL=true"
                )
            if not is_publish_loopback(self.bind_host):
                raise ConfigError(
                    "MCP_ALLOW_UNAUTHENTICATED_LOCAL=true requires a loopback "
                    "MCP_BIND_HOST; refusing to serve unauthenticated on "
                    f"{self.bind_host}"
                )
        if self.auth_mode == "oauth":
            missing = [
                name
                for name, value in (
                    ("MCP_OAUTH_ISSUER_URL", self.oauth_issuer_url),
                    (
                        "MCP_OAUTH_RESOURCE_SERVER_URL",
                        self.oauth_resource_server_url,
                    ),
                    ("MCP_OAUTH_INTROSPECTION_URL", self.oauth_introspection_url),
                    ("MCP_OAUTH_CLIENT_ID", self.oauth_client_id),
                    (
                        "MCP_OAUTH_CLIENT_SECRET",
                        self.oauth_client_secret.get_secret_value(),
                    ),
                )
                if not value
            ]
            if missing:
                raise ConfigError("MCP_AUTH_MODE=oauth requires " + ", ".join(missing))
            if not self.oauth_expected_audience:
                raise ConfigError(
                    "MCP_AUTH_MODE=oauth requires MCP_OAUTH_AUDIENCE or "
                    "MCP_OAUTH_RESOURCE_SERVER_URL"
                )
            oauth_urls = (
                ("MCP_OAUTH_ISSUER_URL", self.oauth_issuer_url),
                ("MCP_OAUTH_RESOURCE_SERVER_URL", self.oauth_resource_server_url),
                ("MCP_OAUTH_INTROSPECTION_URL", self.oauth_introspection_url),
            )
            for variable, url in oauth_urls:
                if (
                    url is not None
                    and url.scheme != "https"
                    and not is_oauth_loopback_url(url)
                ):
                    raise ConfigError(
                        f"{variable} must use https except for an explicit "
                        "loopback development endpoint"
                    )
        # MIDDLEWARE_CLIENT_TOKEN is not unconditionally required: the
        # middleware may itself be serving without authentication. The preflight
        # enforces the pair when the middleware has a non-empty ACL; here we
        # only refuse a collision with this service's own inbound tokens.
        reserved = set()
        if self.middleware_client_token:
            reserved.add(self.middleware_client_token)
        if self.smart_embedder_client_token:
            reserved.add(self.smart_embedder_client_token)
        if self.auth_mode == "static" and reserved & set(tokens):
            raise ConfigError(
                "MCP_API_TOKENS must be distinct from MIDDLEWARE_CLIENT_TOKEN "
                "and SMART_EMBEDDER_CLIENT_TOKEN"
            )
        return self


def load_settings(**overrides: object) -> Settings:
    """Build settings, surfacing configuration problems as ConfigError.

    Two error paths land here. Pydantic only wraps ValueError/TypeError/
    AssertionError raised inside a validator into a ValidationError (with the
    original exception recoverable at ``error["ctx"]["error"]``); a custom
    exception of any other type — ConfigError is a RuntimeError subclass — is
    not wrapped and propagates out of ``Settings(**overrides)`` unchanged, so
    it passes straight through the except clause below and out of this
    function. Genuine pydantic-native failures (e.g. ``MCP_PORT`` outside its
    ge/le range, or a value that fails type coercion) do raise a real
    ValidationError; those are caught here and normalised into ConfigError via
    the ``str(exc)`` fallback. The ctx-unwrapping branch is kept because it is
    the correct way to recover a wrapped ValueError/TypeError/AssertionError
    if a future validator raises one of those instead of ConfigError directly.
    """
    try:
        return Settings(**overrides)
    except ValidationError as exc:
        for error in exc.errors():
            cause = error.get("ctx", {}).get("error")
            if isinstance(cause, ConfigError):
                raise ConfigError(str(cause)) from exc
        raise ConfigError(str(exc)) from exc


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings, validated once.

    The removed-name scan runs before any Settings instance exists: a stale
    variable must abort the boot rather than be quietly ignored by the
    ``extra="ignore"`` policy.
    """
    reject_removed_env_names(REPO_ENV_FILE)
    return load_settings()
