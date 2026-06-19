"""Tests for per-task_kind judge system-prompt resolution (#345 Phase 2)."""

from __future__ import annotations

import json

from skillopt.envs.gitmoot.evaluator import (
    _human_feedback_judge_system_prompt,
    _judge_score,
    _resolve_judge_system_prompt,
)

GENERIC_DEFAULT = (
    "You are evaluating a Gitmoot SkillOpt candidate response. "
    "Return only JSON with keys hard, soft, fail_reason, and reasoning. "
    "hard must be 0 or 1. soft must be a number from 0 to 1."
)


def _default_fn() -> str:
    return "DEFAULT-PROMPT"


def _judge_item(**metadata) -> dict[str, object]:
    return {
        "id": "judge-item",
        "prompt": "Do the task.",
        "metadata": metadata,
    }


# --- _resolve_judge_system_prompt ----------------------------------------


def test_resolve_selects_variant_per_task_kind():
    config = {
        "judge_prompt_templates": {
            "landing_page": "LANDING-PAGE-PROMPT",
            "x_post": "X-POST-PROMPT",
        }
    }
    assert (
        _resolve_judge_system_prompt(_judge_item(task_kind="landing_page"), config, _default_fn)
        == "LANDING-PAGE-PROMPT"
    )
    assert (
        _resolve_judge_system_prompt(_judge_item(task_kind="x_post"), config, _default_fn)
        == "X-POST-PROMPT"
    )


def test_resolve_falls_back_when_no_variant_for_task_kind():
    config = {"judge_prompt_templates": {"landing_page": "LANDING-PAGE-PROMPT"}}
    assert (
        _resolve_judge_system_prompt(_judge_item(task_kind="unmapped_kind"), config, _default_fn)
        == "DEFAULT-PROMPT"
    )


def test_resolve_falls_back_when_no_templates_configured():
    assert _resolve_judge_system_prompt(_judge_item(task_kind="landing_page"), {}, _default_fn) == "DEFAULT-PROMPT"


def test_resolve_falls_back_when_no_task_kind():
    config = {"judge_prompt_templates": {"landing_page": "LANDING-PAGE-PROMPT"}}
    assert _resolve_judge_system_prompt(_judge_item(), config, _default_fn) == "DEFAULT-PROMPT"


def test_resolve_reads_nested_judge_config_location():
    config = {"judge": {"config": {"judge_prompt_templates": {"landing_page": "NESTED-PROMPT"}}}}
    assert (
        _resolve_judge_system_prompt(_judge_item(task_kind="landing_page"), config, _default_fn)
        == "NESTED-PROMPT"
    )


def test_resolve_reads_evaluation_location():
    config = {"evaluation": {"judge_prompt_templates": {"landing_page": "EVAL-PROMPT"}}}
    assert (
        _resolve_judge_system_prompt(_judge_item(task_kind="landing_page"), config, _default_fn)
        == "EVAL-PROMPT"
    )


def test_resolve_top_level_wins_over_nested():
    config = {
        "judge_prompt_templates": {"landing_page": "TOP-LEVEL"},
        "judge": {"config": {"judge_prompt_templates": {"landing_page": "NESTED"}}},
    }
    assert (
        _resolve_judge_system_prompt(_judge_item(task_kind="landing_page"), config, _default_fn)
        == "TOP-LEVEL"
    )


def test_resolve_task_kind_from_top_level_item_and_config():
    config = {"judge_prompt_templates": {"landing_page": "VARIANT"}}
    item_top = {"id": "x", "prompt": "p", "task_kind": "landing_page"}
    assert _resolve_judge_system_prompt(item_top, config, _default_fn) == "VARIANT"

    config_with_kind = {
        "task_kind": "landing_page",
        "judge_prompt_templates": {"landing_page": "VARIANT"},
    }
    assert _resolve_judge_system_prompt({"id": "x", "prompt": "p"}, config_with_kind, _default_fn) == "VARIANT"


def test_resolve_defensive_on_bad_config_types():
    # Wrong-typed templates / variants must all fall back, never raise.
    for config in (
        None,
        {"judge_prompt_templates": "not-a-dict"},
        {"judge_prompt_templates": {"landing_page": 123}},
        {"judge_prompt_templates": {"landing_page": None}},
        {"judge_prompt_templates": {"landing_page": "   "}},
        {"judge": "not-a-dict"},
        {"judge": {"config": "not-a-dict"}},
        {"evaluation": ["not", "a", "dict"]},
    ):
        assert (
            _resolve_judge_system_prompt(_judge_item(task_kind="landing_page"), config, _default_fn)
            == "DEFAULT-PROMPT"
        )


# --- _judge_score wiring --------------------------------------------------


def _capture_judge_system(monkeypatch, item, config):
    captured: dict[str, object] = {}

    def fake_chat_optimizer(**kwargs):
        captured["system"] = kwargs.get("system")
        return (json.dumps({"hard": 1, "soft": 0.9, "fail_reason": "", "reasoning": "ok"}), {})

    monkeypatch.setattr("skillopt.envs.gitmoot.evaluator.chat_optimizer", fake_chat_optimizer)
    result = _judge_score(item, "candidate response", config)
    return captured.get("system"), result


def test_judge_score_uses_configured_variant(monkeypatch):
    config = {"judge_prompt_templates": {"x_post": "X-POST-JUDGE-PROMPT"}}
    system, result = _capture_judge_system(monkeypatch, _judge_item(task_kind="x_post"), config)
    assert system == "X-POST-JUDGE-PROMPT"
    assert result["hard"] == 1


def test_judge_score_default_unchanged_without_templates(monkeypatch):
    # No templates configured -> byte-identical generic default system prompt.
    system, result = _capture_judge_system(monkeypatch, _judge_item(task_kind="x_post"), {})
    assert system == GENERIC_DEFAULT
    assert result["hard"] == 1
    assert result["soft"] == 0.9


def test_judge_score_feedback_default_unchanged_without_templates(monkeypatch):
    # Human-feedback path default must remain the feedback judge system prompt.
    item = {
        "id": "fb",
        "prompt": "p",
        "metadata": {"task_kind": "x_post"},
        "feedback_context": {"feedback_target": "live_candidate"},
        "ranked_feedback_events": [
            {"choice": "best", "ranking": ["A"], "promote": "no", "continue_mode": "refine"}
        ],
    }
    system, _ = _capture_judge_system(monkeypatch, item, {})
    assert system == _human_feedback_judge_system_prompt()
