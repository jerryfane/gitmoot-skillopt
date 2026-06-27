"""Tests for the budgeted trajectory digest and its judge-prompt wiring (#348 Phase 1)."""

from __future__ import annotations

import json

import pytest

from skillopt.envs.gitmoot.evaluator import _judge_score, _process_summary_sections
from skillopt.envs.gitmoot.trajectory_digest import (
    DEFAULT_MAX_BYTES,
    build_trajectory_digest,
)

PLANTED_TOKEN = "ghp_AAAABBBBCCCCDDDDEEEEFFFFGGGG1234"

REALISTIC_RAW = f"""[2026-06-27T10:00:00] OpenAI Codex
user
Implement the feature and run the tests.

codex
I'll inspect the repo and add the function.

exec
bash -lc 'ls -la'
total 8
drwxr-xr-x 3 user user 4096 .
succeeded in 0.12s

exec
apply_patch <<'EOF'
*** Begin Patch
*** Update File: skillopt/feature.py
+def feature():
+    return 42
*** Add File: tests/test_feature.py
+def test_feature():
+    assert feature() == 42
*** End Patch
EOF
succeeded in 0.34s

exec
bash -lc 'curl -H "Authorization: Bearer {PLANTED_TOKEN}" https://api.example.com'
succeeded in 0.20s

exec
bash -lc 'pytest -q tests/test_feature.py'
F
error: assertion failed
failed in 1.20s

codex
Fixing the off-by-one bug.

exec
bash -lc 'pytest -q tests/test_feature.py'
.
1 passed
succeeded in 0.98s

exec
bash -lc 'cat /etc/shadow'
Refusing to expose denied data directory to exec backend
failed in 0.05s

codex
Done. The feature is implemented and the tests pass.
"""


def _write_raw(tmp_path, raw: str, name: str = "target_exec_raw.txt") -> str:
    pred_dir = tmp_path / "pred"
    pred_dir.mkdir(exist_ok=True)
    (pred_dir / name).write_text(raw, encoding="utf-8")
    return str(pred_dir)


def _judge_item() -> dict[str, object]:
    return {"id": "digest-item", "prompt": "Do the task.", "metadata": {}}


def _capture_judge_user(monkeypatch, item, config):
    captured: dict[str, object] = {}

    def fake_chat_optimizer(**kwargs):
        captured["user"] = kwargs.get("user")
        return (json.dumps({"hard": 1, "soft": 0.9, "fail_reason": "", "reasoning": "ok"}), {})

    monkeypatch.setattr("skillopt.envs.gitmoot.evaluator.chat_optimizer", fake_chat_optimizer)
    _judge_score(item, "candidate response", config)
    return str(captured.get("user") or "")


# --- byte-identical OFF-by-default headline test --------------------------------


def test_judge_prompt_byte_identical_when_disabled(monkeypatch, tmp_path):
    pred_dir = _write_raw(tmp_path, REALISTIC_RAW)
    # Default config (no digest knobs) -> never any section.
    base_user = _capture_judge_user(monkeypatch, _judge_item(), {})
    # Even with a valid exec raw on disk, an unset flag must change nothing.
    with_pred_user = _capture_judge_user(monkeypatch, _judge_item(), {"_prediction_dir": pred_dir})

    assert "## Process Summary" not in base_user
    assert "## Process Summary" not in with_pred_user
    # Byte-identical: the _prediction_dir key is the ONLY config delta, and it is
    # only ever surfaced inside the (already-dumped) Evaluation Config JSON.
    expected = with_pred_user.replace(
        json.dumps({"_prediction_dir": pred_dir}, indent=2, sort_keys=True),
        json.dumps({}, indent=2, sort_keys=True),
    )
    assert expected == base_user


def test_disabled_with_explicit_false_flag(monkeypatch, tmp_path):
    pred_dir = _write_raw(tmp_path, REALISTIC_RAW)
    user = _capture_judge_user(
        monkeypatch,
        _judge_item(),
        {"_prediction_dir": pred_dir, "trace_digest_enabled": False, "trace_digest": False},
    )
    assert "## Process Summary" not in user


# --- ON: section present with expected fields -----------------------------------


def test_judge_prompt_includes_section_when_enabled(monkeypatch, tmp_path):
    pred_dir = _write_raw(tmp_path, REALISTIC_RAW)
    user = _capture_judge_user(
        monkeypatch,
        _judge_item(),
        {"_prediction_dir": pred_dir, "trace_digest_enabled": True},
    )
    assert "## Process Summary" in user
    # Section sits before the candidate response.
    assert user.index("## Process Summary") < user.index("## Candidate Response")
    for marker in (
        "### Tool calls",
        "### Commands run",
        "### Tests",
        "### Files touched",
        "### Errors / retries / recovery",
        "### Constraint violations",
        "### Elapsed",
        "### Final artifact summary",
    ):
        assert marker in user


def test_digest_fields_from_parse_codex_raw():
    digest = build_trajectory_digest_from(REALISTIC_RAW)
    assert "pytest -q tests/test_feature.py" in digest
    assert "update: skillopt/feature.py" in digest
    assert "add: tests/test_feature.py" in digest
    # The failed-then-passed test run is recognized as a recovery.
    assert "retry succeeded after failure" in digest
    assert "failed commands: 2" in digest
    # Constraint-violation hint surfaced.
    assert "Refusing to expose denied data directory" in digest


# --- redaction ------------------------------------------------------------------


def test_redaction_removes_planted_secret():
    digest = build_trajectory_digest_from(REALISTIC_RAW)
    assert PLANTED_TOKEN not in digest
    assert "[REDACTED]" in digest


def test_redaction_covers_kv_credentials():
    raw = REALISTIC_RAW + "\nexec\nbash -lc 'export API_KEY=supersecretvalue12345'\nsucceeded in 0.01s\n"
    digest = build_trajectory_digest_from(raw)
    assert "supersecretvalue12345" not in digest


# --- cap enforcement ------------------------------------------------------------


def test_cap_enforced(tmp_path):
    huge = REALISTIC_RAW + "\n".join(
        f"\nexec\nbash -lc 'echo line-{i} with some padding text to grow the trace'\nsucceeded in 0.01s"
        for i in range(5000)
    )
    pred_dir = _write_raw(tmp_path, huge)
    digest = build_trajectory_digest(pred_dir, max_bytes=2000)
    assert digest
    assert len(digest.encode("utf-8")) <= 2000


def test_default_cap_enforced(tmp_path):
    huge = "user\nx\n" + "\n".join(
        f"exec\nbash -lc 'echo {i} aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'\nsucceeded in 0.01s"
        for i in range(20000)
    )
    pred_dir = _write_raw(tmp_path, huge)
    digest = build_trajectory_digest(pred_dir)
    assert len(digest.encode("utf-8")) <= DEFAULT_MAX_BYTES


# --- fail-closed ----------------------------------------------------------------


def test_missing_prediction_dir_returns_empty():
    assert build_trajectory_digest(None) == ""
    assert build_trajectory_digest("") == ""
    assert build_trajectory_digest("/nonexistent/path/abc123") == ""


def test_garbage_raw_returns_empty(tmp_path):
    pred_dir = _write_raw(tmp_path, "this is not a codex trace at all\njust noise\n")
    assert build_trajectory_digest(pred_dir) == ""


def test_empty_raw_returns_empty(tmp_path):
    pred_dir = _write_raw(tmp_path, "   \n  \n")
    assert build_trajectory_digest(pred_dir) == ""


def test_process_summary_sections_fail_closed_when_no_pred_dir():
    # Flag ON but no prediction dir -> no section, no exception.
    assert _process_summary_sections({"trace_digest_enabled": True}) == []


def test_process_summary_sections_fail_closed_on_garbage(tmp_path):
    pred_dir = _write_raw(tmp_path, "no markers here")
    assert _process_summary_sections({"trace_digest_enabled": True, "_prediction_dir": pred_dir}) == []


# --- exec-backends-only gating --------------------------------------------------


def test_exec_backends_only_no_raw_file(tmp_path):
    # A non-exec backend writes conversation.json, never the raw exec files.
    pred_dir = tmp_path / "pred"
    pred_dir.mkdir()
    (pred_dir / "conversation.json").write_text("[]", encoding="utf-8")
    assert build_trajectory_digest(str(pred_dir)) == ""


def test_codex_raw_fallback_used(tmp_path):
    # When only codex_raw.txt exists (no alias yet) it is still read.
    pred_dir = _write_raw(tmp_path, REALISTIC_RAW, name="codex_raw.txt")
    digest = build_trajectory_digest(pred_dir)
    assert "### Commands run" in digest


# --- settings resolution --------------------------------------------------------


@pytest.mark.parametrize(
    "config",
    [
        {"trace_digest_enabled": True},
        {"trace_digest": True},
        {"trace_digest": {"enabled": True}},
    ],
)
def test_enabled_variants(monkeypatch, tmp_path, config):
    pred_dir = _write_raw(tmp_path, REALISTIC_RAW)
    full = {**config, "_prediction_dir": pred_dir}
    user = _capture_judge_user(monkeypatch, _judge_item(), full)
    assert "## Process Summary" in user


def test_custom_max_bytes_via_config(tmp_path):
    huge = "user\nx\n" + "\n".join(
        f"exec\nbash -lc 'echo {i} padding padding padding'\nsucceeded in 0.01s" for i in range(5000)
    )
    pred_dir = _write_raw(tmp_path, huge)
    sections = _process_summary_sections(
        {"trace_digest": {"enabled": True, "max_bytes": 1500}, "_prediction_dir": pred_dir}
    )
    assert sections
    assert len(sections[1].encode("utf-8")) <= 1500


# helper that writes a temp dir on the fly ---------------------------------------


def build_trajectory_digest_from(raw: str) -> str:
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "target_exec_raw.txt").write_text(raw, encoding="utf-8")
        return build_trajectory_digest(tmp)
