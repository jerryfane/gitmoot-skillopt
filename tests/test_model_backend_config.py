from __future__ import annotations

import skillopt.model as model
import skillopt.model.codex_harness as codex_harness


def test_codex_optimizer_backend_routes_chat_optimizer(monkeypatch):
    calls = {}

    def fake_chat_optimizer(**kwargs):
        calls.update(kwargs)
        return "ok", {"prompt_tokens": 1, "completion_tokens": 1}

    monkeypatch.setattr(model._codex, "chat_optimizer", fake_chat_optimizer)
    model.set_optimizer_backend("codex")
    try:
        response, usage = model.chat_optimizer(system="system", user="user", stage="test")
    finally:
        model.set_optimizer_backend("openai_chat")

    assert response == "ok"
    assert usage["prompt_tokens"] == 1
    assert calls["system"] == "system"
    assert calls["user"] == "user"
    assert calls["stage"] == "test"


def test_codex_optimizer_deployment_is_configurable():
    original = model._codex.OPTIMIZER_DEPLOYMENT
    try:
        model.set_optimizer_deployment("gpt-5.5")
        assert model._codex.OPTIMIZER_DEPLOYMENT == "gpt-5.5"
    finally:
        model.set_optimizer_deployment(original)


def test_codex_target_backend_alias_routes_to_exec_backend():
    original = model.get_target_backend()
    try:
        model.set_target_backend("codex")
        assert model.get_target_backend() == "codex_exec"
        assert model.is_target_exec_backend() is True
    finally:
        model.set_target_backend(original)


def test_codex_target_exec_receives_file_edit_mode(monkeypatch, tmp_path):
    calls = {}

    def fake_sdk_exec(**kwargs):
        raise ModuleNotFoundError("openai_codex_sdk")

    def fake_cli_exec(**kwargs):
        calls.update(kwargs)
        return "done", "raw"

    monkeypatch.setattr(codex_harness, "_run_codex_sdk_exec", fake_sdk_exec)
    monkeypatch.setattr(codex_harness, "_run_codex_cli_exec", fake_cli_exec)

    response, raw = codex_harness.run_codex_exec(
        work_dir=str(tmp_path),
        prompt="Build a Vue/Vite preview.",
        model="gpt-test",
        timeout=1,
        allow_file_edits=True,
    )

    assert response == "done"
    assert raw.startswith("===== CODEX")
    assert calls["allow_file_edits"] is True


def test_optimize_cli_model_defaults_are_empty_for_backend_resolution():
    # A hardcoded gpt-5.5 default overrode the per-backend model defaults, so
    # the claude backend was asked for an OpenAI model (404 at preflight).
    from gitmoot_skillopt.cli import build_parser

    args = build_parser().parse_args(
        [
            "optimize",
            "--training-package", "pkg.json",
            "--artifact-root", "blobs",
            "--out-root", "out",
            "--candidate-output", "out/candidate.json",
        ]
    )
    assert args.optimizer_model == ""
    assert args.target_model == ""


def test_default_model_for_backend_claude():
    from skillopt.model.common import default_model_for_backend

    assert default_model_for_backend("claude").startswith("claude-")
    assert not default_model_for_backend("codex").startswith("claude-")


def test_claude_cli_error_includes_stdout_tail(monkeypatch):
    import subprocess as subprocess_module
    from types import SimpleNamespace

    import pytest

    from skillopt.model import claude_backend as cb

    def fake_run(cmd, **kwargs):
        return SimpleNamespace(
            returncode=1,
            stdout='{"type":"result","is_error":true,"result":"model not available"}',
            stderr="",
        )

    monkeypatch.setattr(cb.subprocess, "run", fake_run)
    # A recognizable stdout error is classified into actionable guidance even
    # though stderr is empty (JSON-mode failures land on stdout).
    with pytest.raises(RuntimeError) as excinfo:
        cb._run_claude_print(
            system="s",
            prompt="p",
            model="claude-sonnet-4-6",
            tools=None,
            tool_choice=None,
            return_message=False,
            timeout=5,
        )
    assert "rejected it" in str(excinfo.value)

    # An unrecognized stdout error surfaces its tail instead of a bare exit code.
    def fake_run_unclassified(cmd, **kwargs):
        return SimpleNamespace(
            returncode=1,
            stdout='{"type":"result","is_error":true,"result":"something exploded"}',
            stderr="",
        )

    monkeypatch.setattr(cb.subprocess, "run", fake_run_unclassified)
    with pytest.raises(RuntimeError) as excinfo:
        cb._run_claude_print(
            system="s",
            prompt="p",
            model="claude-sonnet-4-6",
            tools=None,
            tool_choice=None,
            return_message=False,
            timeout=5,
        )
    assert "something exploded" in str(excinfo.value)
    assert "exited with code 1" in str(excinfo.value)
