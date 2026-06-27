"""Command line interface for Gitmoot-SkillOpt."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from . import package_version


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gitmoot-skillopt",
        description=(
            "Optimize Gitmoot agent templates from Gitmoot SkillOpt training "
            "packages and emit pending candidate packages for Gitmoot review."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {package_version()}",
    )

    subcommands = parser.add_subparsers(dest="command", metavar="<command>")

    optimize = subcommands.add_parser(
        "optimize",
        help="optimize a Gitmoot training package into a candidate package",
        description=(
            "Optimize a Gitmoot training package with the Gitmoot SkillOpt "
            "adapter and write a Gitmoot-compatible candidate package plus "
            "verified output artifacts."
        ),
    )
    optimize.add_argument(
        "--training-package",
        required=True,
        help="path to a Gitmoot skillopt training package JSON file",
    )
    optimize.add_argument(
        "--artifact-root",
        required=True,
        help="path to Gitmoot's content-addressed artifact blob root",
    )
    optimize.add_argument(
        "--out-root",
        required=True,
        help="directory for optimizer run output",
    )
    optimize.add_argument(
        "--candidate-output",
        required=True,
        help="path where the candidate package JSON should be written",
    )
    optimize.add_argument(
        "--artifact-dir",
        default="",
        help="directory where candidate artifact files should be written; defaults to OUT_ROOT/artifacts",
    )
    optimize.add_argument(
        "--dry-run",
        action="store_true",
        help="produce a candidate package with zero training epochs; useful for fixture smoke tests",
    )
    optimize.add_argument("--num-epochs", type=int, default=1, help="number of optimization epochs")
    optimize.add_argument("--batch-size", type=int, default=4, help="training batch size")
    optimize.add_argument(
        "--optimizer-views",
        type=_positive_int,
        default=1,
        help="independent optimizer perspectives to run over imported human review feedback",
    )
    optimize.add_argument(
        "--retry-optimizer-views",
        type=_retry_optimizer_views,
        default="auto",
        help=(
            "optimizer perspectives for gate-reject retries: auto, inherit, "
            "or a positive integer"
        ),
    )
    optimize.add_argument("--seed", type=int, default=42, help="random seed")
    optimize.add_argument("--optimizer-model", default="", help="optimizer model name (empty = backend default)")
    optimize.add_argument("--target-model", default="", help="target model name (empty = backend default)")
    optimize.add_argument("--optimizer-backend", default="openai_chat", help="optimizer backend")
    optimize.add_argument(
        "--target-backend",
        default="openai_chat",
        help="target backend; use 'codex' for the Codex execution target",
    )
    optimize.add_argument("--evaluator-id", default="", help="evaluator id, such as landing_page_v1")
    optimize.add_argument(
        "--evaluator-model",
        default="",
        help="evaluator model name; defaults to the optimizer model",
    )
    optimize.add_argument(
        "--evaluator-backend",
        default="",
        help="evaluator backend; defaults to the optimizer backend",
    )
    optimize.add_argument(
        "--gate-metric",
        default="hard",
        choices=["hard", "soft", "mixed"],
        help="selection gate metric used by SkillOpt when accepting candidate updates",
    )
    optimize.add_argument(
        "--reasoning-effort",
        default="",
        help="optional reasoning effort for models that support it; omit for standard OpenAI chat models",
    )
    optimize.add_argument(
        "--skill-update-mode",
        default="patch",
        choices=["patch", "rewrite_from_suggestions", "full_rewrite_minibatch"],
        help="SkillOpt update mode",
    )
    optimize.add_argument(
        "--noop-retry-budget",
        type=int,
        default=1,
        help="number of no-op optimizer retries before skipping a step",
    )
    optimize.add_argument(
        "--gate-reject-retry-budget",
        type=int,
        default=3,
        help="maximum adaptive retries after a near-miss selection gate rejection",
    )
    optimize.add_argument(
        "--wrong-artifact-retry-budget",
        type=int,
        default=1,
        help="number of retries for wrong-artifact evaluator rejections",
    )
    optimize.add_argument(
        "--gate-reject-retry-close-gap",
        type=float,
        default=0.03,
        help=(
            "baseline-minus-candidate gate score gap recorded as large-regression "
            "retry context when actionable gate-reject retry is enabled"
        ),
    )
    optimize.add_argument(
        "--feedback-direct-mode",
        default="auto",
        choices=["auto", "on", "off"],
        help="use ranked human feedback as the first optimizer signal before target rollout",
    )
    optimize.add_argument(
        "--target-artifact-retry-budget",
        type=int,
        default=1,
        help="number of target retries for structured artifact-contract failures before optimizer reflection",
    )
    optimize.add_argument(
        "--hard-failure-retry-budget",
        type=int,
        default=1,
        help="number of reflection retries when structured evaluator failures produce no usable patch",
    )
    optimize.add_argument(
        "--evaluator-schema-retry-budget",
        type=int,
        default=1,
        help="number of evaluator retries for malformed judge schema before failing the evaluator",
    )
    optimize.add_argument(
        "--eval-test",
        action="store_true",
        help="run final test evaluation after selection; disabled by default for Gitmoot human-review flows",
    )
    optimize.add_argument(
        "--judge-prompt-optimization",
        action="store_true",
        help=(
            "tune the JUDGE PROMPT against held-out human agreement instead of the "
            "skill (#345 Phase 2 freeze-and-alternate); requires --judge-human-labeled-path"
        ),
    )
    optimize.add_argument(
        "--judge-human-labeled-path",
        default="",
        help="path to a JSON/JSONL held-out human-labeled set (required with --judge-prompt-optimization)",
    )
    optimize.add_argument(
        "--judge-prompt-init",
        default="",
        help="initial judge prompt text; defaults to the built-in verdict judge prompt",
    )
    optimize.add_argument(
        "--judge-prompt-version",
        default="v0",
        help="base judge prompt version tag stamped on accepted variants",
    )
    optimize.add_argument(
        "--judge-edit-budget",
        type=int,
        default=4,
        help="maximum edits per judge-prompt reflect pass",
    )
    optimize.set_defaults(func=_run_optimize)

    pairwise = subcommands.add_parser(
        "pairwise",
        help="opt-in live-pairwise eval: rerun promoted + candidate live and emit a blinded review packet",
        description=(
            "Opt-in live-pairwise evaluation (#77a). Reruns BOTH the promoted "
            "template and the candidate template live over the validation split, "
            "then writes a blinded paired review packet plus per-item artifacts "
            "(both outputs, token/cost, runtime metadata, and per-item failures). "
            "Does not run the optimizer rewrite or the score-gate, never promotes, "
            "and never writes ~/.gitmoot. The default saved-baseline 'optimize' "
            "path is unchanged."
        ),
    )
    pairwise.add_argument(
        "--training-package",
        required=True,
        help="path to a Gitmoot skillopt training package JSON file",
    )
    pairwise.add_argument(
        "--artifact-root",
        required=True,
        help="path to Gitmoot's content-addressed artifact blob root",
    )
    pairwise.add_argument(
        "--candidate",
        required=True,
        help=(
            "candidate template to compare: a template markdown file, a candidate "
            "package JSON, or an optimizer run directory containing best_skill.md"
        ),
    )
    pairwise.add_argument(
        "--out-root",
        required=True,
        help="directory for live-pairwise run output",
    )
    pairwise.add_argument(
        "--candidate-output",
        default="",
        help="ignored placeholder kept for flag parity with optimize; pairwise never emits a candidate package",
    )
    pairwise.add_argument(
        "--artifact-dir",
        default="",
        help="directory where review packet artifacts should be written; defaults to OUT_ROOT/artifacts",
    )
    pairwise.add_argument(
        "--mode",
        default="live-pairwise",
        choices=["live-pairwise"],
        help="pairwise evaluation mode (only live-pairwise is supported)",
    )
    pairwise.add_argument("--seed", type=int, default=42, help="random seed for split and A/B blinding")
    pairwise.add_argument(
        "--limit",
        type=int,
        default=0,
        help="optional cap on validation items per side (0 = all)",
    )
    pairwise.add_argument(
        "--max-completion-tokens",
        type=int,
        default=4096,
        help="max completion tokens for chat-target live rollouts",
    )
    pairwise.set_defaults(func=_run_pairwise)

    return parser


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _retry_optimizer_views(value: str) -> str:
    raw = str("auto" if value is None else value).strip().lower()
    if raw in {"auto", "inherit"}:
        return raw
    try:
        parsed = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be auto, inherit, or a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be auto, inherit, or a positive integer")
    return str(parsed)


def _run_judge_optimize(args: argparse.Namespace) -> int:
    from .judge_optimize import run_judge_optimize

    if not str(args.judge_human_labeled_path).strip():
        print("error: --judge-prompt-optimization requires --judge-human-labeled-path")
        return 2
    result = run_judge_optimize(
        training_package=args.training_package,
        out_root=args.out_root,
        candidate_output=args.candidate_output,
        human_labeled_path=args.judge_human_labeled_path,
        judge_prompt_init=args.judge_prompt_init,
        judge_prompt_version=args.judge_prompt_version,
        edit_budget=args.judge_edit_budget,
        optimizer_backend=args.optimizer_backend,
        optimizer_model=args.optimizer_model,
        evaluator_id=args.evaluator_id,
        evaluator_backend=args.evaluator_backend,
        evaluator_model=args.evaluator_model,
    )
    print(f"wrote judge candidate package: {args.candidate_output}")
    for name, variant in result.get("variants", {}).items():
        print(
            f"  variant {name}: n_items={variant['n_items']} "
            f"baseline_agreement={variant['baseline_agreement']:.3f} "
            f"best_agreement={variant['best_agreement']:.3f} "
            f"accepted={variant['accepted']} version={variant['judge_prompt_version']}"
        )
    return 0


def _run_optimize(args: argparse.Namespace) -> int:
    if getattr(args, "judge_prompt_optimization", False):
        return _run_judge_optimize(args)

    from .optimize import run_optimize

    candidate = run_optimize(
        training_package=args.training_package,
        artifact_root=args.artifact_root,
        out_root=args.out_root,
        candidate_output=args.candidate_output,
        artifact_dir=args.artifact_dir,
        dry_run=args.dry_run,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        optimizer_views=args.optimizer_views,
        retry_optimizer_views=args.retry_optimizer_views,
        seed=args.seed,
        optimizer_model=args.optimizer_model,
        target_model=args.target_model,
        optimizer_backend=args.optimizer_backend,
        target_backend=args.target_backend,
        evaluator_id=args.evaluator_id,
        evaluator_model=args.evaluator_model,
        evaluator_backend=args.evaluator_backend,
        gate_metric=args.gate_metric,
        reasoning_effort=args.reasoning_effort,
        skill_update_mode=args.skill_update_mode,
        noop_retry_budget=args.noop_retry_budget,
        gate_reject_retry_budget=args.gate_reject_retry_budget,
        wrong_artifact_retry_budget=args.wrong_artifact_retry_budget,
        gate_reject_retry_close_gap=args.gate_reject_retry_close_gap,
        feedback_direct_mode=args.feedback_direct_mode,
        target_artifact_retry_budget=args.target_artifact_retry_budget,
        hard_failure_retry_budget=args.hard_failure_retry_budget,
        evaluator_schema_retry_budget=args.evaluator_schema_retry_budget,
        eval_test=args.eval_test,
    )
    summary = getattr(candidate, "summary", None)
    metadata = summary.metadata if summary is not None and isinstance(summary.metadata, dict) else {}
    no_candidate_reason = str(metadata.get("no_candidate_reason") or "").strip()
    if no_candidate_reason:
        print(f"wrote no-candidate package: {args.candidate_output}")
        print(f"no_candidate_reason: {no_candidate_reason}")
        details = metadata.get("no_candidate_details")
        if isinstance(details, dict):
            if attempted_patch := str(details.get("attempted_patch") or "").strip():
                print(f"attempted_patch: {attempted_patch}")
            if baseline_gate := _optional_detail(details, "baseline_gate"):
                print(f"baseline_gate: {baseline_gate}")
            if candidate_gate := _optional_detail(details, "candidate_gate"):
                print(f"candidate_gate: {candidate_gate}")
            if "duplicate_retry_detected" in details:
                print(f"duplicate_retry_detected: {bool(details.get('duplicate_retry_detected'))}")
            for key in ("review_feedback_items", "optimizer_context_items"):
                value = details.get(key)
                if isinstance(value, list) and value:
                    print(f"{key}: {','.join(str(item) for item in value)}")
            for key in ("selection_failed_item", "retry_decision", "score_gap_handling", "hard_score_handling"):
                if value := str(details.get(key) or "").strip():
                    print(f"{key}: {value}")
            if evaluator_reason := str(details.get("evaluator_reason") or "").strip():
                print(f"evaluator_reason: {evaluator_reason}")
            rejection = details.get("rejection")
            if isinstance(rejection, dict):
                baseline = rejection.get("baseline") if isinstance(rejection.get("baseline"), dict) else {}
                candidate_scores = rejection.get("candidate") if isinstance(rejection.get("candidate"), dict) else {}
                print(
                    "rejection: "
                    f"baseline_gate={baseline.get('gate_score')} "
                    f"candidate_gate={candidate_scores.get('gate_score')}"
                )
            if retry_attempts := str(details.get("retry_attempts") or "").strip():
                print(f"retry_attempts: {retry_attempts}")
            next_actions = details.get("next_actions")
            if isinstance(next_actions, list):
                for action in next_actions:
                    text = str(action or "").strip()
                    if text:
                        print(f"next_action_option: {text}")
        print(f"next_action: {metadata.get('next_action')}")
    else:
        print(f"wrote candidate package: {args.candidate_output}")
    return 0


def _run_pairwise(args: argparse.Namespace) -> int:
    from .pairwise import run_pairwise_eval

    summary = run_pairwise_eval(
        training_package=args.training_package,
        artifact_root=args.artifact_root,
        candidate=args.candidate,
        out_root=args.out_root,
        artifact_dir=args.artifact_dir,
        seed=args.seed,
        max_completion_tokens=args.max_completion_tokens,
        limit=args.limit,
        mode=args.mode,
    )
    print(f"wrote live-pairwise review packet: {summary['packet_markdown_path']}")
    print(f"packet json: {summary['packet_json_path']}")
    print(f"secret map (admin/debug, separate): {summary['secret_map_path']}")
    print(f"val items: {summary['item_count']} failed_sides: {summary['failed_sides']}")
    return 0


def _optional_detail(details: dict[str, object], key: str) -> str:
    value = details.get(key)
    if value is None or value == "":
        return ""
    return str(value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = getattr(args, "func", None)
    if command is None:
        parser.print_help()
        return 0
    return int(command(args) or 0)
