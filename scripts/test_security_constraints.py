"""Tests for advisory-driven transitive lock overrides."""

from scripts import security_constraints as sc


def _write(path, text):
    """Write a test manifest and return its path."""
    path.write_text(text, encoding="utf-8")
    return path


def test_parse_exact_pins_normalizes_names_and_ignores_comments(tmp_path):
    constraints = _write(
        tmp_path / "security.txt",
        "# advisory\nMy_Package==2.0  # CVE\n",
    )

    assert sc.parse_exact_pins(constraints) == {"my-package": ("My_Package", "2.0")}


def test_check_reports_a_stale_constraint_and_a_version_mismatch(tmp_path):
    constraints = _write(tmp_path / "security.txt", "h2==4.4.1\ndatasets==5.0.1\n")
    lock = _write(tmp_path / "middleware.txt", "h2==4.4.0\nhttpx==0.28.1\n")

    errors, changed = sc.security_constraint_errors(constraints, [lock])

    assert any("h2==4.4.0" in error for error in errors)
    assert any("datasets" in error and "stale" in error for error in errors)
    assert changed[lock].startswith("h2==4.4.1\n")


def test_write_mode_updates_only_packages_already_in_each_lock(tmp_path, capsys):
    constraints = _write(tmp_path / "security.txt", "datasets==5.0.1\nh2==4.4.1\n")
    locks = tmp_path / "locks"
    locks.mkdir()
    middleware = _write(locks / "middleware.txt", "h2==4.4.0\nhttpx==0.28.1\n")
    embedder = _write(locks / "embedder.txt", "datasets==5.0.0\ntorch==2.13.0\n")

    result = sc.main(
        [
            "--constraints",
            str(constraints),
            "--locks-dir",
            str(locks),
            "--write",
        ]
    )

    assert result == 0
    assert "h2==4.4.1" in middleware.read_text(encoding="utf-8")
    assert "datasets" not in middleware.read_text(encoding="utf-8")
    assert "datasets==5.0.1" in embedder.read_text(encoding="utf-8")
    assert "h2" not in embedder.read_text(encoding="utf-8")
    assert "Security constraints verified" in capsys.readouterr().out


def test_repository_locks_satisfy_security_constraints():
    """Committed locks must carry every active transitive security override."""
    locks = sorted(sc.DEFAULT_LOCKS_DIR.glob("*.txt"))

    errors, changed = sc.security_constraint_errors(sc.DEFAULT_CONSTRAINTS, locks)

    assert errors == []
    assert changed == {}


def test_docker_builds_apply_one_target_lock_to_both_install_phases():
    """Bootstrap and runtime resolution must share the target's sole lock."""
    dockerfile = (sc.REPOSITORY_ROOT / "Dockerfile").read_text(encoding="utf-8")
    lock_names = (
        "middleware-py311-linux.txt",
        "mcp-py311-linux.txt",
        "embedder-cpu-py311-linux.txt",
        "embedder-gpu-py310-linux.txt",
    )

    for lock_name in lock_names:
        constraint = f"-c requirements/locks/{lock_name}"
        assert dockerfile.count(constraint) == 2, lock_name
    assert "requirements/security-constraints.txt" not in dockerfile
