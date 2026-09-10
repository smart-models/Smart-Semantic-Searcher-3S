"""Tests for the privileged, minimal named-volume bootstrap."""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import call, patch

import model_bootstrap


class ModelBootstrapTests(unittest.TestCase):
    """The init container prepares mounts, then drops privileges before exec."""

    def test_numeric_identity_rejects_root_and_invalid_values(self):
        for value in ("0", "-1", "not-a-number"):
            with (
                self.subTest(value=value),
                patch.dict(os.environ, {"TEST_IDENTITY": value}),
            ):
                with self.assertRaises(ValueError):
                    model_bootstrap._numeric_identity("TEST_IDENTITY", 1000)

    @patch("model_bootstrap.os.chown", create=True)
    def test_prepare_mount_ownership_touches_only_mount_roots(self, chown):
        with patch.object(Path, "is_dir", return_value=True):
            model_bootstrap.prepare_mount_ownership(
                (Path("/models"), Path("/state")), 1000, 1000
            )

        self.assertEqual(
            chown.call_args_list,
            [call(Path("/models"), 1000, 1000), call(Path("/state"), 1000, 1000)],
        )

    def test_drop_privileges_clears_groups_before_gid_and_uid(self):
        operations = []
        with (
            patch("model_bootstrap.os.geteuid", side_effect=[0, 1000], create=True),
            patch("model_bootstrap.os.getegid", return_value=1000, create=True),
            patch(
                "model_bootstrap.os.setgroups",
                side_effect=lambda groups: operations.append(("groups", groups)),
                create=True,
            ),
            patch(
                "model_bootstrap.os.setgid",
                side_effect=lambda gid: operations.append(("gid", gid)),
                create=True,
            ),
            patch(
                "model_bootstrap.os.setuid",
                side_effect=lambda uid: operations.append(("uid", uid)),
                create=True,
            ),
        ):
            model_bootstrap.drop_privileges(1000, 1000)

        self.assertEqual(operations, [("groups", []), ("gid", 1000), ("uid", 1000)])

    @patch("model_bootstrap.os.execv")
    @patch("model_bootstrap.drop_privileges")
    @patch("model_bootstrap.prepare_mount_ownership")
    def test_main_drops_privileges_before_exec(
        self, prepare_ownership, drop_privileges, execv
    ):
        model_bootstrap.main()

        prepare_ownership.assert_called_once_with(
            model_bootstrap.WRITABLE_MOUNTS, 1000, 1000
        )
        drop_privileges.assert_called_once_with(1000, 1000)
        executable, arguments = execv.call_args.args
        self.assertEqual(executable, sys.executable)
        self.assertEqual(arguments[0], sys.executable)
        self.assertTrue(arguments[1].endswith("model_prepare.py"))


if __name__ == "__main__":
    unittest.main()
