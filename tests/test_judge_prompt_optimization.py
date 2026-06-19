"""Tests for #345 Phase 2: human-agreement gate + judge-prompt optimization.

Covers:
  (a) the ``human_agreement`` gate accepts only when agreement improves;
  (b) a synthetic-label mechanism E2E — given two candidate judge prompts, the
      one that agrees more with humans is selected;
  (c) per-task_kind variant selection is honored in the judge-tuning path;
  plus the held-out human-labeled loader and config-flag scaffolding.
"""
from __future__ import annotations

import json

import pytest

from skillopt.config import (
    flatten_config,
    judge_optimization_config,
    judge_optimization_enabled,
)
from skillopt.datasets.base import (
    HumanLabeledItem,
    load_human_labeled_set,
)
from skillopt.engine.trainer import (
    JudgePromptCandidate,
    run_judge_prompt_optimization,
)
from skillopt.envs.gitmoot.dataloader import GitmootDataLoader
from skillopt.evaluation.gate import (
    evaluate_human_agreement_gate,
    normalize_human_verdict,
    score_human_agreement,
    select_gate_score,
)
from tests.test_gitmoot_dataloader import write_training_package

# ── score_human_agreement ─────────────────────────────────────────────────


def test_score_human_agreement_fraction_matching_verdicts():
    labeled = [
        {"id": "a", "human_verdict": "promote"},
        {"id": "b", "human_verdict": "reject"},
        {"id": "c", "human_verdict": "promote"},
        {"id": "d", "human_verdict": "reject"},
    ]
    # judge agrees on a, b; disagrees on c, d  → 0.5
    verdicts = {"a": "promote", "b": "reject", "c": "reject", "d": "promote"}
    assert score_human_agreement(labeled, verdicts) == 0.5
    assert score_human_agreement(labeled, {"a": "promote", "b": "reject", "c": "promote", "d": "reject"}) == 1.0
    assert score_human_agreement([], {}) == 0.0


def test_score_human_agreement_missing_judge_verdict_counts_as_disagreement():
    labeled = [{"id": "a", "human_verdict": "promote"}, {"id": "b", "human_verdict": "reject"}]
    assert score_human_agreement(labeled, {"a": "promote"}) == 0.5


def test_score_human_agreement_positional_list_alignment():
    labeled = [{"human_verdict": "promote"}, {"human_verdict": "reject"}]
    assert score_human_agreement(labeled, ["promote", "reject"]) == 1.0
    assert score_human_agreement(labeled, ["reject", "reject"]) == 0.5


def test_normalize_human_verdict_synonyms_and_errors():
    assert normalize_human_verdict("promote") is True
    assert normalize_human_verdict("accept") is True
    assert normalize_human_verdict("reject") is False
    assert normalize_human_verdict(True) is True
    with pytest.raises(ValueError):
        normalize_human_verdict("maybe")


def test_select_gate_score_human_agreement_passthrough():
    assert select_gate_score(0.83, 0.0, "human_agreement") == 0.83


# ── (a) human_agreement gate accepts only when agreement improves ──────────


def test_human_agreement_gate_accepts_only_when_agreement_improves():
    labeled = [
        {"id": "a", "human_verdict": "promote"},
        {"id": "b", "human_verdict": "reject"},
    ]
    baseline_verdicts = {"a": "reject", "b": "reject"}  # agreement 0.5
    better_verdicts = {"a": "promote", "b": "reject"}  # agreement 1.0
    worse_verdicts = {"a": "reject", "b": "promote"}  # agreement 0.0

    baseline_agreement = score_human_agreement(labeled, baseline_verdicts)

    improved = evaluate_human_agreement_gate(
        candidate_prompt="better judge prompt",
        cand_agreement=score_human_agreement(labeled, better_verdicts),
        current_prompt="baseline prompt",
        current_agreement=baseline_agreement,
        best_prompt="baseline prompt",
        best_agreement=baseline_agreement,
        best_step=0,
        global_step=1,
    )
    regressed = evaluate_human_agreement_gate(
        candidate_prompt="worse judge prompt",
        cand_agreement=score_human_agreement(labeled, worse_verdicts),
        current_prompt="baseline prompt",
        current_agreement=baseline_agreement,
        best_prompt="baseline prompt",
        best_agreement=baseline_agreement,
        best_step=0,
        global_step=2,
    )

    assert improved.action == "accept_new_best"
    assert improved.best_skill == "better judge prompt"
    assert improved.best_score == 1.0
    assert regressed.action == "reject"
    assert regressed.best_skill == "baseline prompt"


# ── (b) synthetic-label mechanism E2E: better judge prompt is selected ─────


def test_judge_prompt_optimization_selects_better_agreeing_candidate():
    labeled = [
        HumanLabeledItem(id="a", human_verdict="promote", artifact="A"),
        HumanLabeledItem(id="b", human_verdict="reject", artifact="B"),
        HumanLabeledItem(id="c", human_verdict="promote", artifact="C"),
        HumanLabeledItem(id="d", human_verdict="reject", artifact="D"),
    ]
    # Baseline judge agrees on 2/4 (0.5).
    baseline_verdicts = {"a": "promote", "b": "promote", "c": "reject", "d": "reject"}
    # Worse candidate: 1/4. Better candidate: 4/4.
    worse = JudgePromptCandidate(
        prompt="judge: promote everything",
        verdicts={"a": "promote", "b": "promote", "c": "promote", "d": "promote"},
        origin="judge_candidate_worse",
    )
    better = JudgePromptCandidate(
        prompt="judge: match the humans",
        verdicts={"a": "promote", "b": "reject", "c": "promote", "d": "reject"},
        origin="judge_candidate_better",
    )

    result = run_judge_prompt_optimization(
        baseline_prompt="baseline judge prompt",
        candidates=[worse, better],
        labeled_items=labeled,
        baseline_verdicts=baseline_verdicts,
        base_version="v1",
    )

    assert result.baseline_agreement == 0.5
    assert result.best_origin == "judge_candidate_better"
    assert result.best_prompt == "judge: match the humans"
    assert result.best_agreement == 1.0
    # judge_prompt_version is recorded on the produced judge-prompt artifact.
    assert result.judge_prompt_version == "v1+judge1"
    actions = [rec["action"] for rec in result.history]
    assert "reject" in actions  # worse candidate rejected
    assert "accept_new_best" in actions  # better candidate promoted


def test_judge_prompt_optimization_keeps_baseline_when_no_candidate_improves():
    labeled = [
        HumanLabeledItem(id="a", human_verdict="promote"),
        HumanLabeledItem(id="b", human_verdict="reject"),
    ]
    baseline_verdicts = {"a": "promote", "b": "reject"}  # already perfect 1.0
    worse = JudgePromptCandidate(
        prompt="worse",
        verdicts={"a": "reject", "b": "reject"},
        origin="worse",
    )

    result = run_judge_prompt_optimization(
        baseline_prompt="frozen baseline judge",
        candidates=[worse],
        labeled_items=labeled,
        baseline_verdicts=baseline_verdicts,
        base_version="v3",
    )

    assert result.best_origin == "baseline_judge"
    assert result.best_prompt == "frozen baseline judge"
    assert result.best_agreement == 1.0
    assert result.judge_prompt_version == "v3"


# ── (c) per-task_kind variant selection in the judge-tuning path ───────────


def test_judge_prompt_optimization_honors_task_kind_variant_selection():
    labeled = [
        HumanLabeledItem(id="lp-1", human_verdict="promote", task_kind="vue_landing_page"),
        HumanLabeledItem(id="lp-2", human_verdict="reject", task_kind="vue_landing_page"),
        HumanLabeledItem(id="other-1", human_verdict="promote", task_kind="text_reply"),
    ]
    # Candidate tagged for the wrong variant must be skipped; the matching one wins.
    wrong_variant = JudgePromptCandidate(
        prompt="judge for text replies",
        verdicts={"lp-1": "promote", "lp-2": "reject"},
        origin="text_reply_candidate",
        task_kind="text_reply",
    )
    right_variant = JudgePromptCandidate(
        prompt="judge for landing pages",
        verdicts={"lp-1": "promote", "lp-2": "reject"},
        origin="landing_page_candidate",
        task_kind="vue_landing_page",
    )

    result = run_judge_prompt_optimization(
        baseline_prompt="baseline landing-page judge",
        candidates=[wrong_variant, right_variant],
        labeled_items=labeled,
        # baseline disagrees on lp-2 → 0.5 over the 2 landing-page items
        baseline_verdicts={"lp-1": "promote", "lp-2": "promote"},
        base_version="lp",
        task_kind="vue_landing_page",
    )

    # Only the two landing-page items are scored (the text_reply label is excluded).
    assert result.baseline_agreement == 0.5
    assert result.best_origin == "landing_page_candidate"
    assert result.best_agreement == 1.0
    skipped = [rec for rec in result.history if rec["action"] == "skipped_task_kind"]
    assert len(skipped) == 1
    assert skipped[0]["origin"] == "text_reply_candidate"


# ── held-out human-labeled loader ──────────────────────────────────────────


def test_load_human_labeled_set_inline_list_canonicalizes_verdicts():
    items = load_human_labeled_set(
        [
            {"id": "x", "human_verdict": "accept", "artifact": "A"},
            {"id": "y", "human_verdict": "no", "trajectory": "T"},
            {"human_verdict": True, "output": "O"},
        ]
    )
    assert [it.human_verdict for it in items] == ["promote", "reject", "promote"]
    assert items[0].artifact == "A"
    assert items[1].artifact == "T"  # falls back to trajectory
    assert items[2].artifact == "O"  # falls back to output
    assert items[2].id == "2"  # positional id fallback


def test_load_human_labeled_set_from_json_path(tmp_path):
    path = tmp_path / "labels.json"
    path.write_text(
        json.dumps(
            [
                {"id": "a", "human_verdict": "promote"},
                {"id": "b", "human_verdict": "reject"},
            ]
        ),
        encoding="utf-8",
    )
    items = load_human_labeled_set(str(path))
    assert len(items) == 2
    assert items[0].id == "a"


def test_load_human_labeled_set_jsonl_path(tmp_path):
    path = tmp_path / "labels.jsonl"
    path.write_text(
        '{"id": "a", "human_verdict": "promote"}\n{"id": "b", "human_verdict": "reject"}\n',
        encoding="utf-8",
    )
    items = load_human_labeled_set(str(path))
    assert [it.id for it in items] == ["a", "b"]


def test_load_human_labeled_set_task_kind_filter():
    raw = [
        {"id": "a", "human_verdict": "promote", "task_kind": "vue_landing_page"},
        {"id": "b", "human_verdict": "reject", "task_kind": "text_reply"},
        {"id": "c", "human_verdict": "promote"},  # untagged → always kept
    ]
    items = load_human_labeled_set(raw, task_kind="vue_landing_page")
    assert [it.id for it in items] == ["a", "c"]


def test_load_human_labeled_set_rejects_bad_verdict_and_missing_field():
    with pytest.raises(ValueError):
        load_human_labeled_set([{"id": "a", "human_verdict": "maybe"}])
    with pytest.raises(ValueError):
        load_human_labeled_set([{"id": "a"}])


def test_load_human_labeled_set_empty_source_returns_empty():
    assert load_human_labeled_set(None) == []
    assert load_human_labeled_set([]) == []
    assert load_human_labeled_set("") == []


# ── config flag scaffolding (freeze-and-alternate) ─────────────────────────


def test_judge_optimization_enabled_flag_and_mode():
    assert judge_optimization_enabled({}) is False
    assert judge_optimization_enabled({"judge": {"prompt_optimization": True}}) is True
    assert judge_optimization_enabled({"judge": {"mode": "judge"}}) is True
    assert judge_optimization_enabled({"judge_prompt_optimization": True}) is True


def test_judge_optimization_config_selects_one_mode():
    # Judge mode → human_agreement gate (freeze the skill).
    judge_cfg = judge_optimization_config({"judge": {"prompt_optimization": True}})
    assert judge_cfg["mode"] == "judge"
    assert judge_cfg["gate_metric"] == "human_agreement"

    # Default → skill mode, keeps its configured gate metric (freeze the judge).
    skill_cfg = judge_optimization_config({"evaluation": {"gate_metric": "soft"}})
    assert skill_cfg["mode"] == "skill"
    assert skill_cfg["gate_metric"] == "soft"


def test_judge_section_flattens_to_trainer_keys():
    flat = flatten_config(
        {
            "env": {"name": "gitmoot"},
            "judge": {
                "prompt_optimization": True,
                "prompt_init": "JUDGE PROMPT",
                "prompt_version": "v7",
                "human_labeled_path": "/tmp/labels.json",
                "edit_budget": 6,
            },
        }
    )
    assert flat["judge_prompt_optimization"] is True
    assert flat["judge_prompt_init"] == "JUDGE PROMPT"
    assert flat["judge_prompt_version"] == "v7"
    assert flat["judge_human_labeled_path"] == "/tmp/labels.json"
    assert flat["judge_edit_budget"] == 6


# ── dataloader held-out human-labeled wiring ───────────────────────────────


def test_dataloader_loads_human_labeled_set_from_config(tmp_path):
    package_path, artifact_root = write_training_package(tmp_path)
    loader = GitmootDataLoader()
    loader.setup(
        {
            "training_package": package_path,
            "artifact_root": artifact_root,
            "judge": {
                "human_labeled": [
                    {"id": "a", "human_verdict": "promote", "artifact": "A"},
                    {"id": "b", "human_verdict": "reject", "artifact": "B"},
                ]
            },
        }
    )
    labeled = loader.human_labeled_items()
    assert [it.id for it in labeled] == ["a", "b"]
    assert labeled[0].human_verdict == "promote"


def test_dataloader_human_labeled_task_kind_filter(tmp_path):
    package_path, artifact_root = write_training_package(tmp_path)
    loader = GitmootDataLoader()
    loader.setup(
        {
            "training_package": package_path,
            "artifact_root": artifact_root,
            "judge_human_labeled": [
                {"id": "lp", "human_verdict": "promote", "task_kind": "vue_landing_page"},
                {"id": "tr", "human_verdict": "reject", "task_kind": "text_reply"},
            ],
        }
    )
    assert [it.id for it in loader.human_labeled_items("vue_landing_page")] == ["lp"]
    assert [it.id for it in loader.human_labeled_items("text_reply")] == ["tr"]


def test_dataloader_without_human_labeled_set_is_empty(tmp_path):
    package_path, artifact_root = write_training_package(tmp_path)
    loader = GitmootDataLoader()
    loader.setup({"training_package": package_path, "artifact_root": artifact_root})
    assert loader.human_labeled_items() == []


# ── reflect: judge-prompt analyst disagreement formatting ──────────────────


def test_fmt_human_labeled_disagreements_only_shows_disagreements():
    from skillopt.gradient.reflect import fmt_human_labeled_disagreements

    labeled = [
        HumanLabeledItem(id="a", human_verdict="promote", artifact="good"),
        HumanLabeledItem(id="b", human_verdict="reject", artifact="bad"),
    ]
    # Judge agrees on a, disagrees on b.
    text = fmt_human_labeled_disagreements(labeled, {"a": "promote", "b": "promote"})
    assert "id=b" in text
    assert "id=a" not in text
    assert "Human verdict: reject" in text


def test_fmt_human_labeled_disagreements_empty_when_all_agree():
    from skillopt.gradient.reflect import fmt_human_labeled_disagreements

    labeled = [HumanLabeledItem(id="a", human_verdict="promote")]
    assert fmt_human_labeled_disagreements(labeled, {"a": "promote"}) == ""


# ── trainer: human_agreement is a valid gate metric ────────────────────────
#
# NOTE: the upstream ``gitmoot_skillopt.optimize.build_trainer_config`` (a
# separate package, NOT owned by this task) still rejects gate metrics outside
# {hard, soft, mixed} at config-build time. We therefore build a valid config
# and override ``cfg["gate_metric"]`` afterwards, which exercises the trainer's
# own gate-metric validation (the part this task owns).


class _GateMetricFakeAdapter:
    def setup(self, cfg):
        pass

    def get_dataloader(self):
        return None

    def requires_ray(self):
        return False

    def build_eval_env(self, **kwargs):
        return ["val-1"]

    def rollout(self, env_manager, skill_content, out_dir, **kwargs):
        return [
            {
                "id": "val-1",
                "hard": 1,
                "soft": 0.9,
                "score_status": "scored",
                "target_status": "passed",
                "evaluator_status": "passed",
            }
        ]

    def get_task_types(self):
        return ["gitmoot-skillopt"]


def _gate_metric_trainer_config(tmp_path):
    from gitmoot_skillopt.contracts import TrainingPackage
    from gitmoot_skillopt.optimize import build_trainer_config

    package_path, artifact_root = write_training_package(tmp_path)
    package = TrainingPackage.load(package_path)
    out_root = tmp_path / "out"
    initial_skill_path = out_root / "initial_skill.md"
    out_root.mkdir(parents=True)
    initial_skill_path.write_text(package.template.content, encoding="utf-8")
    cfg = build_trainer_config(
        package_path=package_path,
        artifact_root=artifact_root,
        out_root=out_root,
        initial_skill_path=initial_skill_path,
        dry_run=False,
        num_epochs=1,
        batch_size=1,
        seed=1,
        optimizer_model="gpt-test",
        target_model="gpt-test",
        optimizer_backend="openai_chat",
        target_backend="openai_chat",
        evaluator_config={"mode": "fixture"},
        gate_metric="hard",
        reasoning_effort="",
        skill_update_mode="patch",
    )
    cfg["num_epochs"] = 0
    cfg["train_size"] = 1
    cfg["eval_test"] = False
    return cfg


def test_trainer_accepts_human_agreement_gate_metric(tmp_path):
    from skillopt.engine.trainer import ReflACTTrainer

    cfg = _gate_metric_trainer_config(tmp_path)
    cfg["gate_metric"] = "human_agreement"

    # Should not raise the "must be 'hard'|'soft'|'mixed'|'human_agreement'"
    # validation error from the trainer.
    summary = ReflACTTrainer(cfg, _GateMetricFakeAdapter()).train()
    assert summary["gate_status"] in {"passed", "no_promotion"}


def test_trainer_rejects_unknown_gate_metric(tmp_path):
    from skillopt.engine.trainer import ReflACTTrainer

    cfg = _gate_metric_trainer_config(tmp_path)
    cfg["gate_metric"] = "bogus"

    with pytest.raises(ValueError, match="gate_metric"):
        ReflACTTrainer(cfg, _GateMetricFakeAdapter()).train()
