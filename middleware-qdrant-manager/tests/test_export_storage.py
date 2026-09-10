"""Tests for export naming, HTTP headers and filesystem containment."""

from urllib.parse import unquote

import pytest

from qdrant_middleware.services.export_storage import (
    attachment_disposition,
    export_filename,
    export_target,
)


def test_unicode_attachment_is_ascii_on_the_wire():
    """The extended filename parameter preserves Unicode without header bytes."""
    filename = "audit-à漢_export_2026-09-05-12-00-00_0123456789ab.json"
    header = attachment_disposition(filename)

    header.encode("ascii")
    assert unquote(header.split("filename*=UTF-8''", 1)[1]) == filename


@pytest.mark.parametrize(
    "filename", ["../escape.json", r"..\escape.json", "/escape.json", "x/y.json"]
)
def test_export_target_refuses_non_basenames(tmp_path, filename):
    """Both native and Windows path spellings remain inside the export root."""
    with pytest.raises(ValueError):
        export_target(tmp_path, filename)
    assert list(tmp_path.iterdir()) == []


def test_export_filename_removes_path_and_control_characters():
    """External collection validation is not relied on for output paths."""
    filename = export_filename(" ..\\bad/\r\n ", "2026-09-05-12-00-00", "0123456789ab")

    assert filename.endswith("_export_2026-09-05-12-00-00_0123456789ab.json")
    assert not set(filename) & {"/", "\\", "\r", "\n"}


def test_export_filename_avoids_windows_device_names():
    """A valid basename remains valid when a host mounts exports on Windows."""
    filename = export_filename("CON", "2026-09-05-12-00-00", "0123456789ab")

    assert filename.startswith("CON__export_")
