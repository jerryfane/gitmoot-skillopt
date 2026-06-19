"""#345 Phase 2: the per-task_kind judge prompt must bridge end-to-end —
from the contract (evaluator_profile.judge.config, stamped by the gitmoot Go
side) through the dataloader's evaluator_config into the evaluator's
per-task_kind resolution. Regression guard for the integration gap where the
dataloader read judge.model but dropped judge.config.
"""
from types import SimpleNamespace

from gitmoot_skillopt.contracts import EvaluatorProfile
from skillopt.envs.gitmoot.dataloader import _evaluator_profile_config
from skillopt.envs.gitmoot.evaluator import _resolve_judge_system_prompt


def _profile_with_judge_templates():
    return EvaluatorProfile.from_dict(
        {
            "task_kind": "vue_landing_page",
            "judge": {
                "type": "llm_judge",
                "config": {
                    "judge_prompt_templates": {"vue_landing_page": "VARIANT PROMPT"},
                    "judge_prompt_version": "jp-2",
                },
            },
        }
    )


def test_evaluator_profile_config_surfaces_judge_prompt_fields():
    package = SimpleNamespace(evaluator_profile=_profile_with_judge_templates())
    cfg = _evaluator_profile_config(package)
    assert cfg["judge_prompt_templates"]["vue_landing_page"] == "VARIANT PROMPT"
    assert cfg["judge_prompt_version"] == "jp-2"


def test_dataloader_config_drives_per_task_kind_resolution():
    cfg = _evaluator_profile_config(SimpleNamespace(evaluator_profile=_profile_with_judge_templates()))
    # Matching task_kind picks the configured variant; non-matching falls back.
    assert (
        _resolve_judge_system_prompt({"metadata": {"task_kind": "vue_landing_page"}}, cfg, lambda: "DEFAULT")
        == "VARIANT PROMPT"
    )
    assert (
        _resolve_judge_system_prompt({"metadata": {"task_kind": "other"}}, cfg, lambda: "DEFAULT") == "DEFAULT"
    )


def test_no_judge_config_is_backward_compatible():
    profile = EvaluatorProfile.from_dict({"task_kind": "generic", "judge": {"type": "llm_judge"}})
    cfg = _evaluator_profile_config(SimpleNamespace(evaluator_profile=profile))
    assert "judge_prompt_templates" not in cfg
    assert _resolve_judge_system_prompt({"metadata": {"task_kind": "generic"}}, cfg, lambda: "DEFAULT") == "DEFAULT"
