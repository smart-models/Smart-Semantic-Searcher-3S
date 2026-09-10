"""Tests for the model preparation step run before the server starts.

``model_prepare`` must fetch exactly the revisions pinned in
``model_revisions`` and fail loudly rather than fall back to a branch.
"""

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

import model_prepare  # noqa: E402
from model_revisions import MODEL_REVISIONS  # noqa: E402


class ModelPrepareTests(unittest.TestCase):
    """Model sets stay pinned, and prefetch honours the exact revision."""

    def test_profile_model_sets_are_complete_and_pinned(self):
        self.assertEqual(
            model_prepare.models_for_profile("basic"),
            ("BAAI/bge-m3", "BAAI/bge-reranker-v2-m3"),
        )
        self.assertEqual(len(model_prepare.models_for_profile("advanced")), 4)
        for models in model_prepare.PROFILE_MODELS.values():
            self.assertTrue(all(model in MODEL_REVISIONS for model in models))

    def test_unsupported_profile_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unsupported EMBEDDER_PROFILE"):
            model_prepare.models_for_profile("experimental")

    def test_prefetch_reuses_local_snapshot_without_network_fallback(self):
        calls = []
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def downloader(**kwargs):
                calls.append(kwargs)
                snapshot = root / kwargs["revision"]
                snapshot.mkdir()
                return str(snapshot)

            result = model_prepare.prefetch_models(
                "basic", downloader=downloader, cache_dir=root / "hub"
            )

        self.assertEqual(len(calls), 2)
        self.assertTrue(all(call["local_files_only"] for call in calls))
        self.assertEqual(set(result), set(model_prepare.PROFILE_MODELS["basic"]))

    def test_prefetch_downloads_exact_revision_after_local_miss(self):
        calls = []
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def downloader(**kwargs):
                calls.append(kwargs)
                if kwargs.get("local_files_only"):
                    raise OSError("not cached")
                snapshot = root / kwargs["revision"]
                snapshot.mkdir()
                return str(snapshot)

            model_prepare.prefetch_models(
                "basic", downloader=downloader, cache_dir=root / "hub"
            )

        self.assertEqual(len(calls), 4)
        for local_call, download_call in zip(calls[::2], calls[1::2]):
            self.assertTrue(local_call["local_files_only"])
            self.assertNotIn("local_files_only", download_call)
            self.assertEqual(download_call["revision"], local_call["revision"])

    def test_prefetch_imports_the_default_downloader_only_when_needed(self):
        calls = []
        fake_hub = types.ModuleType("huggingface_hub")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def downloader(**kwargs):
                calls.append(kwargs)
                snapshot = root / kwargs["revision"]
                snapshot.mkdir(exist_ok=True)
                return str(snapshot)

            fake_hub.snapshot_download = downloader
            with patch.dict(sys.modules, {"huggingface_hub": fake_hub}):
                model_prepare.prefetch_models("basic", cache_dir=root / "hub")

        self.assertEqual(len(calls), 2)
        self.assertTrue(all(call["local_files_only"] for call in calls))

    def test_revision_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            wrong = Path(temporary) / "moving-main"
            wrong.mkdir()
            with self.assertRaisesRegex(RuntimeError, "revision mismatch"):
                model_prepare.prefetch_models(
                    "basic",
                    downloader=lambda **_kwargs: str(wrong),
                    cache_dir=Path(temporary) / "hub",
                )

    def test_main_prefetches_before_calibration(self):
        order = []
        with (
            patch.dict("os.environ", {"EMBEDDER_PROFILE": "advanced"}),
            patch.object(
                model_prepare,
                "prefetch_models",
                side_effect=lambda profile: order.append(("prefetch", profile)),
            ),
            patch.object(
                model_prepare,
                "prepare",
                side_effect=lambda: (
                    order.append(("prepare", None)) or {"action": "reused"}
                ),
            ),
        ):
            self.assertEqual(model_prepare.main(), 0)

        self.assertEqual(order, [("prefetch", "advanced"), ("prepare", None)])


if __name__ == "__main__":
    unittest.main()
