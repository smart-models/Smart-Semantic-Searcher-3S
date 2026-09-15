"""Tests for the model preparation step run before the server starts.

``model_prepare`` must fetch exactly the revisions pinned in
``model_revisions`` and fail loudly rather than fall back to a branch.
"""

import fnmatch
import runpy
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

REPOSITORY_FILES = ("config.json", "model.safetensors", "tokenizer.json")


class FakeHubCache:
    """Stand-in for ``snapshot_download`` that keeps the hub cache's weak spot.

    A local-only lookup returns any snapshot folder that exists, exactly as
    ``huggingface_hub`` does, even when an interrupted download left that folder
    holding only the files that finished before the connection dropped.
    """

    def __init__(self, root, fail_download_of=(), partial_download_of=()):
        self.root = Path(root)
        self.fail_download_of = set(fail_download_of)
        self.partial_download_of = set(partial_download_of)
        self.reachable = True
        self.calls = []

    def snapshot(self, revision):
        return self.root / "snapshots" / revision

    def list_repo_files(self, repo_id, *, revision=None):
        """Return the repository listing, as ``huggingface_hub.list_repo_files``."""
        if not self.reachable:
            raise OSError("hub unreachable")
        return list(REPOSITORY_FILES)

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        snapshot = self.snapshot(kwargs["revision"])
        if kwargs.get("local_files_only"):
            if not snapshot.is_dir():
                raise OSError("not cached")
            return str(snapshot)
        snapshot.mkdir(parents=True, exist_ok=True)
        if kwargs["repo_id"] in self.fail_download_of:
            raise OSError("connection dropped mid-download")
        if not self.reachable or kwargs["repo_id"] in self.partial_download_of:
            # With the Hub unreachable and no cached tree listing,
            # snapshot_download returns the folder as-is, however partial.
            (snapshot / "config.json").touch()
            return str(snapshot)
        ignored = kwargs.get("ignore_patterns") or ()
        for name in REPOSITORY_FILES:
            if not any(fnmatch.fnmatch(name, pattern) for pattern in ignored):
                (snapshot / name).touch()
        return str(snapshot)

    def online_repositories(self):
        return [
            call["repo_id"] for call in self.calls if not call.get("local_files_only")
        ]


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

    def test_prefetch_reuses_a_completed_snapshot_without_network(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary) / "hub"
            hub = FakeHubCache(Path(temporary) / "fake")
            model_prepare.prefetch_models(
                "basic", downloader=hub, lister=hub.list_repo_files, cache_dir=cache
            )
            hub.calls.clear()
            hub.reachable = False
            result = model_prepare.prefetch_models(
                "basic", downloader=hub, lister=hub.list_repo_files, cache_dir=cache
            )

        self.assertEqual(len(hub.calls), 2)
        self.assertTrue(all(call["local_files_only"] for call in hub.calls))
        self.assertEqual(set(result), set(model_prepare.PROFILE_MODELS["basic"]))

    def test_prefetch_downloads_the_exact_pinned_revision(self):
        with tempfile.TemporaryDirectory() as temporary:
            hub = FakeHubCache(Path(temporary) / "fake")
            model_prepare.prefetch_models(
                "basic",
                downloader=hub,
                lister=hub.list_repo_files,
                cache_dir=Path(temporary) / "hub",
            )

        online = [call for call in hub.calls if not call.get("local_files_only")]
        self.assertEqual(
            [call["repo_id"] for call in online],
            list(model_prepare.PROFILE_MODELS["basic"]),
        )
        for call in online:
            self.assertEqual(call["revision"], MODEL_REVISIONS[call["repo_id"]])

    def test_snapshot_folder_without_a_finished_download_is_fetched_again(self):
        """A snapshot folder alone does not prove its download completed."""
        with tempfile.TemporaryDirectory() as temporary:
            hub = FakeHubCache(Path(temporary) / "fake")
            for model_id in model_prepare.PROFILE_MODELS["basic"]:
                hub.snapshot(MODEL_REVISIONS[model_id]).mkdir(parents=True)
            model_prepare.prefetch_models(
                "basic",
                downloader=hub,
                lister=hub.list_repo_files,
                cache_dir=Path(temporary) / "hub",
            )

        self.assertEqual(
            hub.online_repositories(), list(model_prepare.PROFILE_MODELS["basic"])
        )

    def test_download_interrupted_mid_way_resumes_on_the_next_prepare(self):
        interrupted = model_prepare.PROFILE_MODELS["basic"][1]
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary) / "hub"
            hub = FakeHubCache(Path(temporary) / "fake", fail_download_of={interrupted})
            with self.assertRaises(OSError):
                model_prepare.prefetch_models(
                    "basic", downloader=hub, lister=hub.list_repo_files, cache_dir=cache
                )
            hub.fail_download_of.clear()
            hub.calls.clear()
            model_prepare.prefetch_models(
                "basic", downloader=hub, lister=hub.list_repo_files, cache_dir=cache
            )

        self.assertEqual(hub.online_repositories(), [interrupted])

    def test_advanced_prefetch_skips_encoder_weights_milco_never_loads(self):
        with tempfile.TemporaryDirectory() as temporary:
            hub = FakeHubCache(Path(temporary) / "fake")
            model_prepare.prefetch_models(
                "advanced",
                downloader=hub,
                lister=hub.list_repo_files,
                cache_dir=Path(temporary) / "hub",
            )

        patterns = {
            call["repo_id"]: call.get("ignore_patterns")
            for call in hub.calls
            if not call.get("local_files_only")
        }
        self.assertEqual(
            patterns.pop("BAAI/bge-m3-unsupervised"), ["*.safetensors", "*.bin"]
        )
        self.assertEqual(set(patterns.values()), {None})

    def test_changed_download_patterns_fetch_the_snapshot_again(self):
        """A snapshot completed without some files is incomplete for a wider set."""
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary) / "hub"
            hub = FakeHubCache(Path(temporary) / "fake")
            model_prepare.prefetch_models(
                "advanced", downloader=hub, lister=hub.list_repo_files, cache_dir=cache
            )
            hub.calls.clear()
            with patch.dict(model_prepare.DOWNLOAD_IGNORE_PATTERNS, clear=True):
                model_prepare.prefetch_models(
                    "advanced",
                    downloader=hub,
                    lister=hub.list_repo_files,
                    cache_dir=cache,
                )

        self.assertEqual(hub.online_repositories(), ["BAAI/bge-m3-unsupervised"])

    def test_prefetch_imports_the_default_downloader_only_when_needed(self):
        fake_hub = types.ModuleType("huggingface_hub")

        with tempfile.TemporaryDirectory() as temporary:
            hub = FakeHubCache(Path(temporary) / "fake")
            fake_hub.snapshot_download = hub
            fake_hub.list_repo_files = hub.list_repo_files
            with patch.dict(sys.modules, {"huggingface_hub": fake_hub}):
                model_prepare.prefetch_models(
                    "basic", cache_dir=Path(temporary) / "hub"
                )

        self.assertEqual(len(hub.online_repositories()), 2)

    def test_a_partial_snapshot_returned_by_the_hub_is_not_certified(self):
        """snapshot_download hands back a partial folder when the Hub drops out."""
        partial = model_prepare.PROFILE_MODELS["basic"][0]
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary) / "hub"
            hub = FakeHubCache(Path(temporary) / "fake", partial_download_of={partial})
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                model_prepare.prefetch_models(
                    "basic", downloader=hub, lister=hub.list_repo_files, cache_dir=cache
                )
            marker = model_prepare.completion_marker(
                cache, partial, MODEL_REVISIONS[partial]
            )

            self.assertFalse(marker.exists())

    def test_an_unreachable_hub_never_certifies_an_unverified_snapshot(self):
        model_id = model_prepare.PROFILE_MODELS["basic"][0]
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary) / "hub"
            hub = FakeHubCache(Path(temporary) / "fake")
            hub.snapshot(MODEL_REVISIONS[model_id]).mkdir(parents=True)
            hub.reachable = False
            with self.assertRaisesRegex(RuntimeError, "cannot verify"):
                model_prepare.prefetch_models(
                    "basic", downloader=hub, lister=hub.list_repo_files, cache_dir=cache
                )
            marker = model_prepare.completion_marker(
                cache, model_id, MODEL_REVISIONS[model_id]
            )

            self.assertFalse(marker.exists())

    def test_a_marker_older_releases_wrote_is_validated_again(self):
        """Older markers were written even for partial snapshots; they prove nothing."""
        model_id = model_prepare.PROFILE_MODELS["basic"][0]
        revision = MODEL_REVISIONS[model_id]
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary) / "hub"
            hub = FakeHubCache(Path(temporary) / "fake")
            hub.snapshot(revision).mkdir(parents=True)
            (hub.snapshot(revision) / "config.json").touch()
            legacy = model_prepare.completion_marker(cache, model_id, revision)
            legacy.parent.mkdir(parents=True)
            legacy.write_text("", encoding="utf-8")  # the pattern-only record
            hub.reachable = False
            with self.assertRaisesRegex(RuntimeError, "cannot verify"):
                model_prepare.prefetch_models(
                    "basic", downloader=hub, lister=hub.list_repo_files, cache_dir=cache
                )
            hub.reachable = True
            hub.calls.clear()
            model_prepare.prefetch_models(
                "basic", downloader=hub, lister=hub.list_repo_files, cache_dir=cache
            )
            complete = all(
                (hub.snapshot(revision) / name).exists() for name in REPOSITORY_FILES
            )

        self.assertIn(model_id, hub.online_repositories())
        self.assertTrue(complete)

    def test_a_file_pruned_after_verification_is_fetched_again(self):
        model_id = model_prepare.PROFILE_MODELS["basic"][0]
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary) / "hub"
            hub = FakeHubCache(Path(temporary) / "fake")
            model_prepare.prefetch_models(
                "basic", downloader=hub, lister=hub.list_repo_files, cache_dir=cache
            )
            (hub.snapshot(MODEL_REVISIONS[model_id]) / "model.safetensors").unlink()
            hub.calls.clear()
            model_prepare.prefetch_models(
                "basic", downloader=hub, lister=hub.list_repo_files, cache_dir=cache
            )

        self.assertEqual(hub.online_repositories(), [model_id])

    def test_revision_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            wrong = Path(temporary) / "moving-main"
            wrong.mkdir()
            with self.assertRaisesRegex(RuntimeError, "revision mismatch"):
                model_prepare.prefetch_models(
                    "basic",
                    downloader=lambda **_kwargs: str(wrong),
                    lister=lambda *_args, **_kwargs: [],
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

    def test_running_the_file_as_a_script_prepares_models(self):
        """model_bootstrap execs this file, so running it must run main()."""
        calibrations = []
        fake_hub = types.ModuleType("huggingface_hub")
        fake_autotune = types.ModuleType("autotune_entrypoint")
        fake_autotune.prepare = lambda: (
            calibrations.append("prepare") or {"action": "reused"}
        )
        with tempfile.TemporaryDirectory() as temporary:
            hub = FakeHubCache(Path(temporary) / "fake")
            fake_hub.snapshot_download = hub
            fake_hub.list_repo_files = hub.list_repo_files
            modules = {
                "huggingface_hub": fake_hub,
                "autotune_entrypoint": fake_autotune,
            }
            environment = {
                "EMBEDDER_PROFILE": "basic",
                "HF_HUB_CACHE": str(Path(temporary) / "hub"),
            }
            with (
                patch.dict(sys.modules, modules),
                patch.dict("os.environ", environment),
                patch("logging.basicConfig"),
                self.assertRaises(SystemExit) as exited,
            ):
                runpy.run_path(
                    str(SERVER_DIR / "model_prepare.py"), run_name="__main__"
                )

        self.assertEqual(exited.exception.code, 0)
        self.assertEqual(
            hub.online_repositories(), list(model_prepare.PROFILE_MODELS["basic"])
        )
        self.assertEqual(calibrations, ["prepare"])


if __name__ == "__main__":
    unittest.main()
