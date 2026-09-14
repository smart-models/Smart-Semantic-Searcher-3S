"""Calibrate Smart Embedder in a pre-start process, then exec Uvicorn.

``prepare()`` isolates profile-specific calibration and ``main()`` always
continues into the serving process, including after a visible fallback.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Optional

from advanced_autotune import AdvancedAutotuneConfig
from basic_autotune import AutotuneConfig, ProfileLock

APP_VERSION = "1.2.0"
BGE_MODEL = "BAAI/bge-m3"
BASIC_PROFILE = "basic"
ADVANCED_PROFILE = "advanced"


def _last_json_line(output: str) -> Optional[dict]:
    for line in reversed(output.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def basic_calibration_command(config: AutotuneConfig, *, script_dir: Path) -> list[str]:
    """Build the isolated BASIC calibration command.

    Args:
        config: Resolved BASIC calibration settings.
        script_dir: Directory containing the BASIC runner.

    Returns:
        Command arguments for the current Python interpreter.
    """
    return [
        sys.executable,
        str(script_dir / "basic_autotune_runner.py"),
        "--mode",
        config.mode,
        "--objective",
        config.objective,
        "--profile",
        config.profile_path,
        "--app-version",
        APP_VERSION,
        "--model-revision",
        BGE_MODEL,
    ]


def advanced_calibration_command(
    config: AdvancedAutotuneConfig, *, script_dir: Path
) -> list[str]:
    """Build the isolated ADVANCED calibration command.

    Args:
        config: Resolved ADVANCED calibration settings.
        script_dir: Directory containing the ADVANCED runner.

    Returns:
        Command arguments for the current Python interpreter.
    """
    return [
        sys.executable,
        str(script_dir / "advanced_autotune_runner.py"),
        "--mode",
        config.mode,
        "--objective",
        config.objective,
        "--profile",
        config.profile_path,
        "--max-seconds",
        str(config.max_seconds),
        "--app-version",
        APP_VERSION,
    ]


def _advanced_device_kind() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except (ImportError, RuntimeError):
        return "cpu"


def prepare(
    env: Optional[Mapping[str, str]] = None,
    *,
    script_dir: Optional[Path] = None,
) -> dict:
    """Run the applicable pre-start tuner with bounded isolation.

    Calibration failures are converted into explicit fallback results so they
    never prevent the serving process from starting with conservative defaults.

    Args:
        env: Environment passed unchanged to the calibration subprocess.
        script_dir: Directory containing the profile-specific runners.

    Returns:
        The runner's JSON result or a disabled/fallback result produced at the
        isolation boundary.

    Raises:
        ValueError: If ``EMBEDDER_PROFILE`` is not ``basic`` or ``advanced``.
    """
    values = os.environ if env is None else env
    active_profile = values.get("EMBEDDER_PROFILE", BASIC_PROFILE).strip().lower()
    if active_profile not in {BASIC_PROFILE, ADVANCED_PROFILE}:
        raise ValueError(
            f"unsupported EMBEDDER_PROFILE {active_profile!r}; "
            f"expected {BASIC_PROFILE!r} or {ADVANCED_PROFILE!r}"
        )
    if active_profile == ADVANCED_PROFILE:
        config = AdvancedAutotuneConfig.from_env(
            values, device_kind=_advanced_device_kind()
        )
    else:
        config = AutotuneConfig.from_env(values)
    if config.mode == "off":
        return {"action": "disabled", "reason": "autotune is off"}
    directory = script_dir or Path(__file__).resolve().parent
    command = (
        advanced_calibration_command(config, script_dir=directory)
        if active_profile == ADVANCED_PROFILE
        else basic_calibration_command(config, script_dir=directory)
    )
    budget_limit = (
        config.budget_ceiling_seconds
        if active_profile == ADVANCED_PROFILE
        else config.max_seconds
    )
    timeout = max(30.0, budget_limit + 60.0)
    logging.info(
        "%s hardware autotune pre-start begin: mode=%s objective=%s budget=%.0fs "
        "ceiling=%.0fs",
        active_profile.upper(),
        config.mode,
        config.objective,
        config.max_seconds,
        budget_limit,
    )
    try:
        profile_path = Path(config.profile_path)
        lock_path = profile_path.with_suffix(profile_path.suffix + ".calibration.lock")
        with ProfileLock(
            lock_path,
            timeout_seconds=timeout,
            stale_seconds=timeout + 300.0,
        ):
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=dict(values),
            )
    except subprocess.TimeoutExpired:
        return {
            "action": "fallback",
            "reason": f"pre-start calibration exceeded {timeout:.0f}s",
        }
    except TimeoutError:
        return {
            "action": "fallback",
            "reason": f"pre-start calibration lock exceeded {timeout:.0f}s",
        }
    payload = _last_json_line(completed.stdout)
    if completed.returncode != 0 or payload is None:
        if (
            payload is not None
            and payload.get("action") == "fallback"
            and isinstance(payload.get("reason"), str)
        ):
            return {**payload, "returncode": completed.returncode}
        stderr = completed.stderr.strip().splitlines()
        reason = (
            stderr[-1]
            if stderr
            else (
                "calibration returned no JSON result"
                if payload is None
                else f"calibration exited with status {completed.returncode}"
            )
        )
        return {
            "action": "fallback",
            "reason": reason,
            "returncode": completed.returncode,
        }
    for warning in payload.get("warnings") or ():
        logging.warning("%s hardware autotune: %s", active_profile.upper(), warning)
    return payload


def main() -> int:
    """Calibrate when needed, then replace this process with Uvicorn.

    Returns:
        Only returns a status if process replacement fails to occur.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    result = prepare()
    if result.get("action") in {"fallback"}:
        logging.warning("Hardware autotune pre-start fallback: %s", result)
    else:
        logging.info("Hardware autotune pre-start result: %s", result)

    command = [
        "uvicorn",
        "smart-embedder:app",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--workers",
        "1",
    ]
    os.execvp(command[0], command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
