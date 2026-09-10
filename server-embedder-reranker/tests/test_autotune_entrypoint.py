"""Tests for the pre-start autotune entrypoint.

``autotune_entrypoint`` dispatches to the basic or advanced tuner from the
environment; these checks pin that dispatch and its exit codes.
"""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

import autotune_entrypoint  # noqa: E402


class AutotuneEntrypointTests(unittest.TestCase):
    """Profile dispatch, timeout fallback, and the final exec into uvicorn."""

    def test_nonzero_fallback_retains_the_structured_failure_reason(self):
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=2,
            stdout='{"action":"fallback","reason":"baseline failed","stage":"probe"}',
            stderr="",
        )
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(subprocess, "run", return_value=completed):
                result = autotune_entrypoint.prepare(
                    {
                        "SMART_EMBEDDER_AUTOTUNE": "quick",
                        "SMART_EMBEDDER_AUTOTUNE_PROFILE": str(
                            Path(temporary) / "hardware.json"
                        ),
                    }
                )
        self.assertEqual(result["action"], "fallback")
        self.assertEqual(result["reason"], "baseline failed")
        self.assertEqual(result["stage"], "probe")
        self.assertEqual(result["returncode"], 2)

    def test_off_profile_skips_subprocess_and_unsupported_fails(self):
        with patch.object(subprocess, "run") as runner:
            disabled = autotune_entrypoint.prepare({"SMART_EMBEDDER_AUTOTUNE": "off"})
            with self.assertRaisesRegex(ValueError, "unsupported EMBEDDER_PROFILE"):
                autotune_entrypoint.prepare(
                    {
                        "SMART_EMBEDDER_AUTOTUNE": "quick",
                        "EMBEDDER_PROFILE": "experimental",
                    }
                )

        self.assertEqual(disabled["action"], "disabled")
        runner.assert_not_called()

    def test_nonzero_exit_cannot_promote_a_success_payload(self):
        completed = subprocess.CompletedProcess(
            args=[], returncode=2, stdout='{"action":"calibrated"}', stderr=""
        )
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(subprocess, "run", return_value=completed):
                result = autotune_entrypoint.prepare(
                    {
                        "SMART_EMBEDDER_AUTOTUNE": "quick",
                        "SMART_EMBEDDER_AUTOTUNE_PROFILE": str(
                            Path(temporary) / "hardware.json"
                        ),
                    }
                )
        self.assertEqual(result["action"], "fallback")
        self.assertIn("status 2", result["reason"])

    def test_advanced_profile_uses_independent_runner_and_profile(self):
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout='{"action":"calibrated","selected":{"advanced":{}}}\n',
            stderr="",
        )
        with tempfile.TemporaryDirectory() as temporary:
            hf_home = str(Path(temporary) / "models")
            env = {
                "SMART_EMBEDDER_AUTOTUNE": "quick",
                "EMBEDDER_PROFILE": "advanced",
                "HF_HOME": hf_home,
            }
            with (
                patch.object(
                    autotune_entrypoint, "_advanced_device_kind", return_value="cuda"
                ),
                patch.object(subprocess, "run", return_value=completed) as worker,
            ):
                result = autotune_entrypoint.prepare(env, script_dir=Path("/app"))

        command = worker.call_args.args[0]
        self.assertEqual(result["action"], "calibrated")
        self.assertIn("advanced_autotune_runner.py", command[1])
        self.assertTrue(
            any("advanced-hardware-profile.json" in value for value in command)
        )
        self.assertIn("240.0", command)

    def test_successful_prestart_returns_runner_payload(self):
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout='loading\n{"action":"calibrated","selected":{"threads":4}}\n',
            stderr="",
        )
        with tempfile.TemporaryDirectory() as temporary:
            profile_path = str(Path(temporary) / "hardware.json")
            env = {
                "SMART_EMBEDDER_AUTOTUNE": "quick",
                "SMART_EMBEDDER_AUTOTUNE_PROFILE": profile_path,
                "EMBEDDER_PROFILE": "basic",
            }
            with patch.object(subprocess, "run", return_value=completed) as runner:
                result = autotune_entrypoint.prepare(env, script_dir=Path("/app"))

        self.assertEqual(result["action"], "calibrated")
        command = runner.call_args.args[0]
        self.assertIn("basic_autotune_runner.py", command[1])
        self.assertIn(profile_path, command)
        self.assertEqual(runner.call_args.kwargs["env"], env)

    def test_timeout_degrades_to_visible_fallback(self):
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(
                subprocess,
                "run",
                side_effect=subprocess.TimeoutExpired(["python"], 180),
            ),
        ):
            result = autotune_entrypoint.prepare(
                {
                    "SMART_EMBEDDER_AUTOTUNE": "quick",
                    "SMART_EMBEDDER_AUTOTUNE_MAX_SECONDS": "120",
                    "SMART_EMBEDDER_AUTOTUNE_PROFILE": str(
                        Path(temporary) / "hardware.json"
                    ),
                }
            )

        self.assertEqual(result["action"], "fallback")
        self.assertIn("180s", result["reason"])

    def test_main_execs_uvicorn_even_after_fallback(self):
        with (
            patch.object(
                autotune_entrypoint,
                "prepare",
                return_value={"action": "fallback", "reason": "test"},
            ),
            patch.object(os, "execvp", side_effect=RuntimeError("exec")) as execvp,
            self.assertRaisesRegex(RuntimeError, "exec"),
        ):
            autotune_entrypoint.main()

        self.assertEqual(execvp.call_args.args[0], "uvicorn")


if __name__ == "__main__":
    unittest.main()
