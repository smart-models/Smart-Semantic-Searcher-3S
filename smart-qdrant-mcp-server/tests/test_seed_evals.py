"""The eval seed script must speak the real server's tool contract."""

import importlib.util
from pathlib import Path

from mcp.client import Client

SEED = Path(__file__).parent.parent / "evals" / "seed_evals.py"


def load_seed_module():
    """Import evals/seed_evals.py, which is a script rather than a package."""
    spec = importlib.util.spec_from_file_location("seed_evals", SEED)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def test_seed_runs_clean_against_the_real_server(mcp_server, routes, monkeypatch):
    """Every seed call passes the server's validation and reaches the middleware.

    A live seed once failed every call with "params: Field required" because
    the script sent flat arguments, yet exited 0 after printing the errors.
    The script now raises on any refused call, so this run fails if a tool
    rejects its arguments.
    """
    seed_evals = load_seed_module()
    routes.add("POST", "/collections", body={"ok": True, "created": True})
    routes.add(
        "POST",
        "/collections/mcp_eval/payload-indexes",
        body={"ok": True, "created": [k for k, _ in seed_evals.INDEXED_KEYS]},
    )
    routes.add(
        "POST",
        "/points/ingest-chunks",
        body={"ok": True, "ingested": 72, "skipped": 0, "collection_name": "mcp_eval"},
    )
    routes.add("POST", "/points/count", body={"count": 72})
    monkeypatch.setattr(seed_evals, "streamable_http_client", lambda *a, **k: None)
    monkeypatch.setattr(seed_evals, "Client", lambda transport: Client(mcp_server))

    await seed_evals.seed("http://unused", "unused", recreate=True)

    assert [request.url.path for request in routes.calls] == [
        "/collections",
        "/collections/mcp_eval/payload-indexes",
        "/points/ingest-chunks",
        "/points/count",
    ]
