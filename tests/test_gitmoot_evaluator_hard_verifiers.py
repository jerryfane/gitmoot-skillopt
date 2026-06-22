"""Tests for the default-on deterministic hard-verifier floor (#346).

The floor runs before the LLM judge for every non-fixture task kind and
short-circuits with a ``hard==0`` failure packet on the first failed required
check. These tests assert the floor catches bad candidates without ever calling
the judge, lets clean candidates through to the judge, and leaves the
fixture/contains/landing-page paths untouched.
"""

from __future__ import annotations

import json

import pytest

from skillopt.envs.gitmoot.evaluator import evaluate_response

# --- helpers ---------------------------------------------------------------


def _no_judge(monkeypatch):
    """Monkeypatch chat_optimizer to fail loudly if the judge is ever called."""

    def fail_chat_optimizer(**kwargs):
        raise AssertionError("LLM judge must not run after a hard-verifier failure")

    monkeypatch.setattr("skillopt.envs.gitmoot.evaluator.chat_optimizer", fail_chat_optimizer)


def _stub_judge(monkeypatch):
    """Monkeypatch chat_optimizer to a fixed valid judge response and record calls."""
    calls: dict[str, object] = {"count": 0}

    def stub_chat_optimizer(**kwargs):
        calls["count"] = int(calls["count"]) + 1
        calls["system"] = kwargs.get("system")
        return (json.dumps({"hard": 1, "soft": 0.8, "fail_reason": "", "reasoning": "ok"}), {})

    monkeypatch.setattr("skillopt.envs.gitmoot.evaluator.chat_optimizer", stub_chat_optimizer)
    return calls


def _agent_template_item(**metadata):
    base = {"task_kind": "agent_template"}
    base.update(metadata)
    return {"id": "tmpl-item", "prompt": "Improve the agent template.", "metadata": base}


def _package_item(**metadata):
    base = {"task_kind": "package"}
    base.update(metadata)
    return {"id": "pkg-item", "prompt": "Build the package.", "metadata": base}


_GOOD_TEMPLATE = """---
id: planner
name: Planner
kind: agent-template
---

# Planner

Plan the work carefully.

## Update Format
Return a complete replacement skill document.

```json
{"action": "plan", "steps": 3}
```
"""


def _failed_check_ids(score):
    return [check["check"] for check in score["failure"]["failed_checks"]]


# --- agent_template: bad candidates short-circuit (judge NOT called) --------


def test_agent_template_missing_frontmatter_hard_fails(monkeypatch):
    _no_judge(monkeypatch)
    response = "# Planner\n\n## Update Format\nReturn a replacement skill.\n"
    score = evaluate_response(_agent_template_item(), response, {})
    assert score["hard"] == 0
    assert score["soft"] == 0.0
    assert any("frontmatter" in check_id for check_id in _failed_check_ids(score))


def test_agent_template_invalid_example_json_hard_fails(monkeypatch):
    _no_judge(monkeypatch)
    response = _GOOD_TEMPLATE.replace(
        '{"action": "plan", "steps": 3}', '{"action": "plan", "steps": 3,}'
    )
    score = evaluate_response(_agent_template_item(), response, {})
    assert score["hard"] == 0
    assert any("fenced_json" in check_id for check_id in _failed_check_ids(score))


def test_agent_template_absolute_path_hard_fails(monkeypatch):
    _no_judge(monkeypatch)
    response = _GOOD_TEMPLATE + "\nUse the file at /home/user/secret/config.yaml when running.\n"
    score = evaluate_response(_agent_template_item(), response, {})
    assert score["hard"] == 0
    assert any("absolute_path" in check_id for check_id in _failed_check_ids(score))


def test_agent_template_missing_required_section_hard_fails(monkeypatch):
    _no_judge(monkeypatch)
    response = "---\nid: planner\nname: Planner\nkind: agent-template\n---\n\n# Planner\n\nPlan work.\n"
    score = evaluate_response(_agent_template_item(), response, {})
    assert score["hard"] == 0
    assert any("sections" in check_id for check_id in _failed_check_ids(score))


def test_agent_template_secret_hard_fails(monkeypatch):
    _no_judge(monkeypatch)
    response = _GOOD_TEMPLATE + "\nexport OPENAI_API_KEY=sk-abcdefghijklmnopqrstuvwxyz0123\n"
    score = evaluate_response(_agent_template_item(), response, {})
    assert score["hard"] == 0
    assert any("secret" in check_id for check_id in _failed_check_ids(score))


def test_agent_template_remote_mutation_language_hard_fails(monkeypatch):
    _no_judge(monkeypatch)
    response = _GOOD_TEMPLATE + "\nWhen done, git push to origin and auto-merge the PR.\n"
    score = evaluate_response(_agent_template_item(), response, {})
    assert score["hard"] == 0
    assert any("unsafe_language" in check_id for check_id in _failed_check_ids(score))


def test_agent_template_failure_packet_shape(monkeypatch):
    _no_judge(monkeypatch)
    response = "# no frontmatter\n\n## Update Format\nstuff\n"
    score = evaluate_response(_agent_template_item(), response, {})
    failure = score["failure"]
    assert failure["primary_reason"] == "hard_verifier_failure"
    assert failure["human_reason"]
    assert failure["optimizer_hint"]
    assert failure["failed_dimensions"] == ["hard_verifier"]
    assert failure["stage_status"] == [{"stage": "hard_verifier", "status": "failed"}]
    assert all({"check", "severity", "reason", "evidence"} <= set(c) for c in failure["failed_checks"])
    assert score["metadata"]["failure"] == failure


# --- agent_template: clean candidate proceeds to the judge ------------------


def test_agent_template_clean_candidate_reaches_judge(monkeypatch):
    calls = _stub_judge(monkeypatch)
    score = evaluate_response(_agent_template_item(), _GOOD_TEMPLATE, {})
    assert calls["count"] == 1
    assert score["hard"] == 1
    assert score["soft"] == 0.8


# --- package: invalid vs valid ---------------------------------------------


def test_package_non_json_hard_fails(monkeypatch):
    _no_judge(monkeypatch)
    score = evaluate_response(_package_item(), "this is not json at all", {})
    assert score["hard"] == 0
    assert any("package.json" in check_id for check_id in _failed_check_ids(score))


def test_package_wrong_contract_version_hard_fails(monkeypatch):
    _no_judge(monkeypatch)
    response = json.dumps({"contract_version": 2, "templates": []})
    score = evaluate_response(_package_item(), response, {})
    assert score["hard"] == 0
    assert any("contract_version" in check_id for check_id in _failed_check_ids(score))


def test_package_valid_candidate_reaches_judge(monkeypatch):
    calls = _stub_judge(monkeypatch)
    response = json.dumps({"contract_version": 1, "templates": [{"id": "planner"}]})
    score = evaluate_response(_package_item(), response, {})
    assert calls["count"] == 1
    assert score["hard"] == 1


# --- declared checks bridge (config["checks"]) -----------------------------


def test_declared_check_runs_for_unknown_task_kind(monkeypatch):
    _no_judge(monkeypatch)
    item = {"id": "x", "prompt": "p", "metadata": {"task_kind": "custom_kind"}}
    config = {"checks": [{"id": "secrets-floor", "type": "no_secrets", "required": True}]}
    response = "some output with AKIA0123456789ABCDEF embedded"
    score = evaluate_response(item, response, config)
    assert score["hard"] == 0
    assert any("secret" in check_id for check_id in _failed_check_ids(score))


def test_declared_check_skipped_when_not_required(monkeypatch):
    calls = _stub_judge(monkeypatch)
    item = {"id": "x", "prompt": "p", "metadata": {"task_kind": "custom_kind"}}
    config = {"checks": [{"id": "secrets-floor", "type": "no_secrets", "required": False}]}
    response = "some output with AKIA0123456789ABCDEF embedded"
    score = evaluate_response(item, response, config)
    # Not required -> floor skips it, candidate reaches the judge.
    assert calls["count"] == 1
    assert score["hard"] == 1


# --- no-op for unknown task kinds with no declared checks -------------------


def test_unknown_task_kind_no_declared_checks_is_noop(monkeypatch):
    calls = _stub_judge(monkeypatch)
    item = {"id": "x", "prompt": "p", "metadata": {"task_kind": "x_post"}}
    # Candidate has an absolute path + secret, but no floor applies -> judge runs.
    response = "post about /home/user/file with sk-abcdefghijklmnopqrstuvwxyz0123"
    score = evaluate_response(item, response, {})
    assert calls["count"] == 1
    assert score["hard"] == 1


# --- fixture / contains / landing-page paths unchanged ---------------------


def test_fixture_path_unchanged(monkeypatch):
    _no_judge(monkeypatch)
    item = {"id": "f", "metadata": {"task_kind": "agent_template", "expected_hard": True}}
    score = evaluate_response(item, "ignored", {"mode": "fixture"})
    assert score["hard"] == 1
    assert score["metadata"]["evaluator"] == "fixture"


def test_contains_path_unchanged(monkeypatch):
    _no_judge(monkeypatch)
    item = {"id": "c", "metadata": {"task_kind": "agent_template"}}
    config = {"mode": "contains", "required_text": "hello"}
    # An agent_template that would fail the floor still goes through contains only.
    score = evaluate_response(item, "hello world /home/user/x", config)
    assert score["hard"] == 1
    assert score["metadata"]["evaluator"] == "contains"


def test_landing_page_path_unchanged(monkeypatch):
    # A non-bundle landing-page response still hits the landing-page floor
    # (wrong_artifact_type), not the new hard-verifier floor.
    def fail_chat_optimizer(**kwargs):
        raise AssertionError("landing page judge should not run for artifact contract failures")

    monkeypatch.setattr("skillopt.envs.gitmoot.evaluator.chat_optimizer", fail_chat_optimizer)
    response = "---\nid: landing-page-builder\nkind: agent-template\n---\n\n# X\n\n## Update Format\nReturn a skill.\n"
    score = evaluate_response(
        {"prompt": "Build a Vue landing page."},
        response,
        {"mode": "landing_page_v1", "artifact_contract": "vue_vite_bundle"},
    )
    assert score["hard"] == 0
    assert score["failure"]["primary_reason"] == "wrong_artifact_type"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
