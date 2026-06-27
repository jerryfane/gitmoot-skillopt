"""Gitmoot rollout execution."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import re
from dataclasses import dataclass
from typing import Any

from skillopt.envs.gitmoot.evaluator import _check_vue_vite_bundle, evaluate_response
from skillopt.envs.gitmoot.package import safe_item_path_segment
from skillopt.envs.gitmoot.result_contract import (
    EVALUATOR_FAILED,
    EVALUATOR_NOT_RUN,
    STRUCTURED_EVALUATOR_FIELDS,
    TARGET_FAILED,
    TARGET_PASSED,
    make_unscored_evaluation,
    normalize_scored_evaluation,
)
from skillopt.model import chat_target, get_target_backend, is_target_chat_backend, is_target_exec_backend
from skillopt.model.codex_harness import prepare_workspace, render_skill_md, run_target_exec
from skillopt.model.common import default_model_for_backend

SKILLOPT_TARGET_START = "<!-- SKILLOPT_TARGET_START -->"
SKILLOPT_TARGET_END = "<!-- SKILLOPT_TARGET_END -->"
SKILLOPT_OPTIMIZER_START = "<!-- SKILLOPT_OPTIMIZER_START -->"
SKILLOPT_OPTIMIZER_END = "<!-- SKILLOPT_OPTIMIZER_END -->"
VUE_VITE_REQUIRED_FILES = ("package.json", "index.html", "src/main.js", "src/App.vue")
VUE_VITE_ARTIFACT_MARKERS = {"vue_vite_bundle", "vue-vite-bundle"}


@dataclass(frozen=True)
class TargetSkillContext:
    content: str
    metadata: dict[str, Any]


def run_batch(
    *,
    items: list[dict[str, Any]],
    skill_content: str,
    out_root: str,
    max_completion_tokens: int = 4096,
    evaluator_config: dict[str, Any] | None = None,
    target_artifact_retry_budget: int = 1,
    skip_evaluation: bool = False,
) -> list[dict[str, Any]]:
    return [
        process_one(
            item=item,
            skill_content=skill_content,
            out_root=out_root,
            max_completion_tokens=max_completion_tokens,
            evaluator_config=evaluator_config,
            target_artifact_retry_budget=target_artifact_retry_budget,
            skip_evaluation=skip_evaluation,
        )
        for item in items
    ]


def process_one(
    *,
    item: dict[str, Any],
    skill_content: str,
    out_root: str,
    max_completion_tokens: int = 4096,
    evaluator_config: dict[str, Any] | None = None,
    target_artifact_retry_budget: int = 1,
    skip_evaluation: bool = False,
) -> dict[str, Any]:
    item_id = str(item["id"])
    prediction_id = safe_item_path_segment(item_id)
    pred_dir = os.path.join(out_root, "predictions", prediction_id)
    os.makedirs(pred_dir, exist_ok=True)
    target_trace_path = _target_trace_path(pred_dir, item)
    evaluator_trace_path = os.path.join(pred_dir, "result.json")
    target_skill = extract_target_skill_content(skill_content)
    system_prompt = _build_system_prompt(target_skill.content)
    user_prompt = str(item.get("prompt") or "")
    response = ""
    fail_reason = ""
    agent_ok = False
    agent_failed = False
    repair_attempts: list[dict[str, Any]] = []
    # Per-item token/cost capture: the agent path (exec or chat) populates this
    # sink so usage/cost lands in result.json for downstream per-item artifacts
    # (e.g. the #77a live-pairwise review packet). Empty when usage is unknown.
    exec_usage: dict[str, Any] = {}

    try:
        config = evaluator_config if evaluator_config is not None else item.get("evaluator_config")
        response = _call_run_agent(
            item,
            target_skill.content,
            system_prompt,
            user_prompt,
            max_completion_tokens,
            pred_dir,
            config=config,
            usage_sink=exec_usage,
        )
        agent_ok = True
    except Exception as exc:  # noqa: BLE001 - rollout result records the failure.
        agent_failed = True
        fail_reason = str(exc) or "agent execution failed"

    if agent_failed:
        score = make_unscored_evaluation(
            fail_reason=fail_reason,
            target_status=TARGET_FAILED,
            evaluator_status=EVALUATOR_NOT_RUN,
            blocker="target_rollout_failed",
            target_trace_path=target_trace_path,
            evaluator_trace_path=evaluator_trace_path,
            metadata={"evaluator": "not_run", "agent_error": True},
        )
    elif skip_evaluation:
        # Response-only rollout (e.g. live-pairwise): the agent response is the
        # only product, so never invoke the evaluator/judge. This avoids a
        # per-item LLM-judge call (and its network dependency / spend) for a mode
        # that discards all hard/soft scores.
        score = make_unscored_evaluation(
            fail_reason="",
            target_status=TARGET_PASSED,
            evaluator_status=EVALUATOR_NOT_RUN,
            blocker="",
            target_trace_path=target_trace_path,
            evaluator_trace_path=evaluator_trace_path,
            metadata={"evaluator": "not_run", "evaluation_skipped": True},
        )
    else:
        try:
            score = _evaluate_target_response(
                item=item,
                response=response,
                config=config,
                pred_dir=pred_dir,
                target_trace_path=target_trace_path,
                evaluator_trace_path=evaluator_trace_path,
            )
            for attempt in range(max(0, int(target_artifact_retry_budget or 0))):
                if not _is_artifact_contract_failure(score):
                    break
                repair_prompt = _build_artifact_repair_prompt(
                    original_prompt=user_prompt,
                    score=score,
                    attempt=attempt + 1,
                    budget=max(0, int(target_artifact_retry_budget or 0)),
                )
                repair_record = _artifact_repair_record(score=score, attempt=attempt + 1)
                try:
                    repaired_response = _call_run_agent(
                        item,
                        target_skill.content,
                        system_prompt,
                        repair_prompt,
                        max_completion_tokens,
                        pred_dir,
                        config=config,
                        usage_sink=exec_usage,
                    )
                except Exception as repair_exc:  # noqa: BLE001 - keep the last scored artifact failure.
                    repair_record["status"] = "target_failed"
                    repair_record["error"] = str(repair_exc) or "artifact repair target failed"
                    repair_attempts.append(repair_record)
                    break
                repaired_score = _evaluate_target_response(
                    item=item,
                    response=repaired_response,
                    config=config,
                    pred_dir=pred_dir,
                    target_trace_path=target_trace_path,
                    evaluator_trace_path=evaluator_trace_path,
                )
                repair_record["status"] = "accepted" if not _is_artifact_contract_failure(repaired_score) else "failed"
                repair_record["after_primary_reason"] = str(repaired_score.get("primary_reason") or "")
                repair_record["after_fail_reason"] = str(repaired_score.get("fail_reason") or "")
                repair_attempts.append(repair_record)
                response = repaired_response
                user_prompt = repair_prompt
                score = repaired_score
        except Exception as exc:  # noqa: BLE001 - rollout result records the failure.
            fail_reason = str(exc) or "evaluator execution failed"
            score = make_unscored_evaluation(
                fail_reason=fail_reason,
                target_status=TARGET_PASSED,
                evaluator_status=EVALUATOR_FAILED,
                blocker="evaluator_failed",
                target_trace_path=target_trace_path,
                evaluator_trace_path=evaluator_trace_path,
                metadata={"evaluator": "unknown"},
            )
    result = {
        "id": item_id,
        "hard": score.get("hard"),
        "soft": score.get("soft"),
        "response": response,
        "fail_reason": str(score.get("fail_reason") or ""),
        "agent_ok": agent_ok,
        "n_turns": 1 if agent_ok else 0,
        "task_type": item.get("task_type", "gitmoot-skillopt"),
        "task_description": item.get("task_description", item_id),
        "metadata": {
            **(item.get("metadata") if isinstance(item.get("metadata"), dict) else {}),
            "target_skill": target_skill.metadata,
            **_target_note_metadata(response),
            **(score.get("metadata") if isinstance(score.get("metadata"), dict) else {}),
            **({"target_artifact_repair_attempts": repair_attempts} if repair_attempts else {}),
        },
        "target_status": score.get("target_status"),
        "evaluator_status": score.get("evaluator_status"),
        "score_status": score.get("score_status"),
        "blocker": score.get("blocker", ""),
        "evaluator_id": score.get("evaluator_id", ""),
        "evaluator_version": score.get("evaluator_version", ""),
        "target_trace_path": score.get("target_trace_path", ""),
        "evaluator_trace_path": score.get("evaluator_trace_path", ""),
        "target_system_prompt": system_prompt,
        "target_user_prompt": user_prompt,
        "token_usage": dict(exec_usage),
    }
    for key in STRUCTURED_EVALUATOR_FIELDS:
        if key in score:
            result[key] = score[key]
    _write_prediction(pred_dir, result)
    return result


def _evaluate_target_response(
    *,
    item: dict[str, Any],
    response: str,
    config: Any,
    pred_dir: str,
    target_trace_path: str,
    evaluator_trace_path: str,
) -> dict[str, Any]:
    raw_score = evaluate_response(item, response, _evaluator_config_for_result(config, pred_dir, response))
    return normalize_scored_evaluation(
        raw_score,
        target_trace_path=target_trace_path,
        evaluator_trace_path=evaluator_trace_path,
    )


def _is_artifact_contract_failure(score: dict[str, Any]) -> bool:
    primary_reason = str(score.get("primary_reason") or "").strip().lower()
    if primary_reason in {"wrong_artifact_type", "artifact_contract_failure"}:
        return True
    failed_dimensions = score.get("failed_dimensions")
    if isinstance(failed_dimensions, list) and "artifact_contract" in {
        str(item).strip().lower() for item in failed_dimensions
    }:
        return True
    failure = score.get("failure") if isinstance(score.get("failure"), dict) else {}
    failure_reason = str(failure.get("primary_reason") or "").strip().lower()
    if failure_reason in {"wrong_artifact_type", "artifact_contract_failure"}:
        return True
    failure_dimensions = failure.get("failed_dimensions")
    return isinstance(failure_dimensions, list) and "artifact_contract" in {
        str(item).strip().lower() for item in failure_dimensions
    }


def _artifact_repair_record(*, score: dict[str, Any], attempt: int) -> dict[str, Any]:
    failed_checks = score.get("failed_checks")
    if not isinstance(failed_checks, list):
        failure = score.get("failure") if isinstance(score.get("failure"), dict) else {}
        failed_checks = failure.get("failed_checks") if isinstance(failure.get("failed_checks"), list) else []
    return {
        "attempt": attempt,
        "before_primary_reason": str(score.get("primary_reason") or ""),
        "before_fail_reason": str(score.get("fail_reason") or ""),
        "failed_checks": failed_checks,
    }


def _build_artifact_repair_prompt(*, original_prompt: str, score: dict[str, Any], attempt: int, budget: int) -> str:
    failed_checks = score.get("failed_checks")
    if not isinstance(failed_checks, list):
        failure = score.get("failure") if isinstance(score.get("failure"), dict) else {}
        failed_checks = failure.get("failed_checks") if isinstance(failure.get("failed_checks"), list) else []
    optimizer_hint = str(score.get("optimizer_hint") or "")
    if not optimizer_hint:
        failure = score.get("failure") if isinstance(score.get("failure"), dict) else {}
        optimizer_hint = str(failure.get("optimizer_hint") or "")
    failure_summary = {
        "primary_reason": str(score.get("primary_reason") or ""),
        "fail_reason": str(score.get("fail_reason") or ""),
        "optimizer_hint": optimizer_hint,
        "failed_checks": failed_checks,
    }
    return "\n\n".join(
        [
            original_prompt,
            f"## Artifact Contract Repair Attempt {attempt}/{budget}",
            "The previous target response failed the required artifact contract.",
            "Return only the corrected deliverable. Do not return a skill, prompt, YAML, markdown explanation, or prose.",
            "Use this structured failure packet:",
            json.dumps(failure_summary, indent=2, ensure_ascii=False),
        ]
    )


def _evaluator_config_for_result(config: Any, pred_dir: str, response: str) -> dict[str, Any]:
    configured = dict(config) if isinstance(config, dict) else {}
    configured.pop("artifact_dir", None)
    configured["render_artifact_dir"] = os.path.join(
        pred_dir,
        "render-smoke",
        hashlib.sha256(response.encode("utf-8", errors="replace")).hexdigest()[:12],
    )
    configured["_prediction_dir"] = pred_dir
    return configured


def _run_agent(
    item: dict[str, Any],
    skill_content: str,
    system_prompt: str,
    user_prompt: str,
    max_completion_tokens: int,
    pred_dir: str,
    *,
    config: Any = None,
    usage_sink: dict[str, Any] | None = None,
) -> str:
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    if _has_mock_response(item):
        return str(metadata["mock_response"])
    if is_target_exec_backend():
        response = _run_exec_agent(
            item, skill_content, system_prompt, user_prompt, pred_dir, config=config, usage_sink=usage_sink
        )
        if not response.strip():
            raise RuntimeError("exec target returned empty response")
        return response
    if not is_target_chat_backend():
        raise RuntimeError("Gitmoot rollout requires a chat target backend, exec target backend, or metadata.mock_response")
    response, _usage = chat_target(
        system=system_prompt,
        user=user_prompt,
        max_completion_tokens=max_completion_tokens,
        retries=2,
        stage="gitmoot_rollout",
    )
    if usage_sink is not None and isinstance(_usage, dict) and _usage:
        usage_sink["usage"] = _usage
    return response


def _call_run_agent(
    item: dict[str, Any],
    skill_content: str,
    system_prompt: str,
    user_prompt: str,
    max_completion_tokens: int,
    pred_dir: str,
    *,
    config: Any = None,
    usage_sink: dict[str, Any] | None = None,
) -> str:
    parameters = inspect.signature(_run_agent).parameters
    kwargs: dict[str, Any] = {}
    if "config" in parameters:
        kwargs["config"] = config
    if "usage_sink" in parameters:
        kwargs["usage_sink"] = usage_sink
    return _run_agent(
        item,
        skill_content,
        system_prompt,
        user_prompt,
        max_completion_tokens,
        pred_dir,
        **kwargs,
    )


def _run_exec_agent(
    item: dict[str, Any],
    skill_content: str,
    system_prompt: str,
    user_prompt: str,
    pred_dir: str,
    *,
    config: Any = None,
    usage_sink: dict[str, Any] | None = None,
) -> str:
    work_dir = os.path.join(pred_dir, "target_exec")
    skill_md = render_skill_md(
        skill_content,
        description="Dynamic Gitmoot SkillOpt agent-template guidance.",
        preamble="Use this Gitmoot agent-template guidance to complete the task in `task.md`.",
    )
    prepare_workspace(
        work_dir=work_dir,
        skill_md=skill_md,
        task_text=user_prompt,
    )
    model = os.environ.get("TARGET_DEPLOYMENT") or default_model_for_backend(get_target_backend())
    collect_workspace_artifact = _requires_vue_vite_workspace_artifact(item, config)
    try:
        response, raw = run_target_exec(
            work_dir=work_dir,
            prompt=_build_exec_prompt(system_prompt, collect_workspace_artifact=collect_workspace_artifact),
            model=model,
            timeout=900,
            allow_file_edits=collect_workspace_artifact,
        )
    except Exception as exc:
        _write_target_exec_trace_alias(pred_dir, error=str(exc) or "exec target failed")
        raise
    if usage_sink is not None:
        usage_sink.update(_extract_exec_usage(raw))
    _write_target_exec_trace_alias(pred_dir, raw=raw)
    with open(os.path.join(pred_dir, "target_exec_response.txt"), "w", encoding="utf-8") as handle:
        handle.write(response)
    if collect_workspace_artifact:
        artifact_response, _is_complete = _collect_vue_vite_workspace_artifact(work_dir, target_note=response)
        if artifact_response is not None and not _is_valid_vue_vite_response(response):
            with open(os.path.join(pred_dir, "target_exec_artifact.json"), "w", encoding="utf-8") as handle:
                handle.write(artifact_response)
            return artifact_response
    return response


def _requires_vue_vite_workspace_artifact(item: dict[str, Any], config: Any) -> bool:
    sources: list[dict[str, Any]] = []
    if isinstance(config, dict):
        sources.append(config)
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    sources.append(metadata)
    item_config = item.get("evaluator_config")
    if isinstance(item_config, dict):
        sources.append(item_config)
    for artifact in (item.get("artifacts") or {}).values() if isinstance(item.get("artifacts"), dict) else []:
        if isinstance(artifact, dict):
            sources.append(artifact)

    prompt = str(item.get("prompt") or "").strip().lower()
    if "vue/vite" in prompt or "vue-vite" in prompt:
        return True

    for source in sources:
        artifact_contract = str(source.get("artifact_contract") or source.get("output_contract") or "").strip().lower()
        output_type = str(source.get("output_type") or "").strip().lower()
        profile_id = str(source.get("profile_id") or "").strip().lower()
        task_kind = str(source.get("task_kind") or "").strip().lower()
        driver = str(source.get("driver") or "").strip().lower()
        mode = str(source.get("mode") or "").strip().lower()
        checks = source.get("checks")
        if bool(source.get("require_vue_vite_bundle")) or bool(source.get("require_vue_render_smoke")):
            return True
        if _has_vue_render_smoke_check(checks):
            return True
        if artifact_contract in VUE_VITE_ARTIFACT_MARKERS:
            return True
        if output_type in VUE_VITE_ARTIFACT_MARKERS:
            return True
        if profile_id in {"landing_page_v1", "vue_landing_page_v1"}:
            return True
        if task_kind in {"landing_page", "vue_landing_page"}:
            return True
        if driver == "vue-vite":
            return True
        if mode in {"landing_page_v1", "landing-page-v1", "landing_page"}:
            return True
    return False


def _has_vue_render_smoke_check(checks: Any) -> bool:
    if not isinstance(checks, list):
        return False
    for check in checks:
        if not isinstance(check, dict):
            continue
        check_id = str(check.get("id") or "").strip().lower().replace("-", "_")
        check_type = str(check.get("type") or "").strip().lower().replace("-", "_")
        if check_id == "render_smoke" or check_type == "render_smoke":
            return True
    return False


def _collect_vue_vite_workspace_artifact(work_dir: str, *, target_note: str) -> tuple[str | None, bool]:
    files: list[dict[str, str]] = []
    for rel_path, path in _iter_vue_vite_workspace_files(work_dir):
        try:
            with open(path, encoding="utf-8") as handle:
                files.append({"path": rel_path, "content": handle.read()})
        except UnicodeDecodeError:
            continue
    if not files:
        return None, False
    is_complete = set(VUE_VITE_REQUIRED_FILES).issubset({entry["path"] for entry in files})
    return json.dumps(
        {
            "renderer": "vue-vite",
            "build_command": "npm run build",
            "dist_dir": "dist",
            "files": files,
            "target_note": target_note,
            "artifact_source": "codex_exec_workspace",
        },
        ensure_ascii=False,
    ), is_complete


def _iter_vue_vite_workspace_files(work_dir: str) -> list[tuple[str, str]]:
    files: list[tuple[str, str]] = []
    workspace_root = os.path.realpath(work_dir)
    for root, dirs, filenames in os.walk(work_dir):
        dirs[:] = [
            dirname
            for dirname in dirs
            if dirname not in {".agents", ".git", "dist", "node_modules"}
            and not dirname.startswith(".")
            and _is_workspace_child(os.path.join(root, dirname), workspace_root)
        ]
        for filename in filenames:
            path = os.path.join(root, filename)
            rel_path = os.path.relpath(path, work_dir).replace(os.sep, "/")
            if _is_collectable_vue_vite_workspace_file(rel_path) and _is_workspace_child(path, workspace_root):
                files.append((rel_path, path))
    return sorted(files, key=lambda item: (item[0] not in VUE_VITE_REQUIRED_FILES, item[0]))


def _is_workspace_child(path: str, workspace_root: str) -> bool:
    if os.path.islink(path):
        return False
    real_path = os.path.realpath(path)
    return os.path.commonpath([workspace_root, real_path]) == workspace_root


def _is_collectable_vue_vite_workspace_file(rel_path: str) -> bool:
    if rel_path in VUE_VITE_REQUIRED_FILES:
        return True
    if rel_path.startswith("src/") or rel_path.startswith("public/"):
        return not rel_path.endswith((".map",))
    return rel_path in {"vite.config.js", "vite.config.mjs", "vite.config.ts"}


def _is_valid_vue_vite_response(response: str) -> bool:
    return _check_vue_vite_bundle(response) is None


def _target_note_metadata(response: str) -> dict[str, str]:
    try:
        parsed = json.loads(response)
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    target_note = parsed.get("target_note")
    if not isinstance(target_note, str) or not target_note.strip():
        return {}
    artifact_source = parsed.get("artifact_source")
    metadata = {"target_note": target_note}
    if isinstance(artifact_source, str) and artifact_source.strip():
        metadata["target_artifact_source"] = artifact_source.strip()
    return metadata


def _build_exec_prompt(system_prompt: str, *, collect_workspace_artifact: bool = False) -> str:
    instructions = [
        "Use the `skillopt-target` skill available in this workspace.",
        "Read `.agents/skills/skillopt-target/SKILL.md` directly; do not call a Skill tool.",
        "Read `task.md` and complete the Gitmoot SkillOpt target task.",
    ]
    if collect_workspace_artifact:
        instructions.extend(
            [
                "Create the requested Vue/Vite preview as files in this workspace before answering.",
                "At minimum write package.json, index.html, src/main.js, and src/App.vue.",
                "Gitmoot will collect those workspace files as the artifact, so do not return a skill file, prompt template, YAML/frontmatter, or Markdown code blocks as the deliverable.",
                "After writing the files, return a short plain-text completion note.",
            ]
        )
    else:
        instructions.append("Return only the requested response text.")
    instructions.extend(["## System Instructions", system_prompt])
    return "\n\n".join(instructions)


def _write_target_exec_trace_alias(pred_dir: str, raw: str = "", error: str = "") -> None:
    alias_path = os.path.join(pred_dir, "target_exec_raw.txt")
    content = raw
    if not content:
        for name in ("codex_raw.txt", "claude_raw.txt"):
            source_path = os.path.join(pred_dir, name)
            if os.path.exists(source_path):
                with open(source_path, encoding="utf-8") as handle:
                    content = handle.read()
                break
    if not content:
        content = f"exec target failed before raw trace was captured: {error}"
    with open(alias_path, "w", encoding="utf-8") as handle:
        handle.write(content)


def _usage_from_json_blob(text: str) -> dict[str, Any]:
    """Pull token/cost fields out of a single JSON-looking trace body."""
    usage: dict[str, Any] = {}
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return usage
    if not isinstance(data, dict):
        return usage
    cost = data.get("total_cost_usd")
    if isinstance(cost, int | float) and not isinstance(cost, bool):
        usage["cost_usd"] = float(cost)
    claude_usage = data.get("usage")
    if isinstance(claude_usage, dict) and claude_usage:
        usage["usage"] = claude_usage
    for key in ("num_turns", "duration_ms"):
        value = data.get(key)
        if isinstance(value, int | float) and not isinstance(value, bool):
            usage[key] = value
    return usage


def _exec_attempt_segments(raw: str) -> list[str]:
    """Split a raw trace into per-attempt bodies, stripping ``===== ATTEMPT ====`` banners.

    ``run_target_exec`` joins each attempt as ``===== <BACKEND> <CLI|SDK> ATTEMPT N =====\\n<body>``
    (newline-joined), so the usage JSON never sits at the start of the blob. Strip
    those banners and return each attempt body so callers can parse them.
    """
    banner = re.compile(r"^=====.*ATTEMPT\s+\d+\s+=====\s*$", re.MULTILINE)
    if not banner.search(raw):
        return [raw]
    segments = [segment.strip() for segment in banner.split(raw)]
    return [segment for segment in segments if segment]


def _extract_exec_usage(raw: str) -> dict[str, Any]:
    """Parse token/cost usage from an exec target's raw trace.

    Handles both exec backends without raising: the Claude Code / Codex SDK paths
    emit a JSON object carrying ``total_cost_usd``/``usage``/``num_turns``, while
    the Codex CLI path emits a plain ``tokens used`` line. ``run_target_exec``
    wraps every attempt body behind a ``===== ... ATTEMPT N =====`` banner, so we
    strip those banners and parse each attempt, keeping the last attempt that
    carries usage (the returned/successful one). Returns an empty dict when no
    usage signal is present so callers can treat usage as best-effort.
    """
    usage: dict[str, Any] = {}
    if not isinstance(raw, str) or not raw.strip():
        return usage
    for segment in _exec_attempt_segments(raw):
        if segment.startswith("{"):
            segment_usage = _usage_from_json_blob(segment)
            if segment_usage:
                usage = segment_usage
    if usage:
        return usage
    lines = [line.rstrip() for line in raw.splitlines()]
    for index, line in enumerate(lines):
        if line.strip() == "tokens used" and index + 1 < len(lines):
            token_text = lines[index + 1].strip().replace(",", "")
            try:
                usage["total_tokens"] = int(token_text)
            except ValueError:
                pass
            break
    return usage


def _has_mock_response(item: dict[str, Any]) -> bool:
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    return metadata.get("mock_response") is not None


def _target_trace_path(pred_dir: str, item: dict[str, Any]) -> str:
    if is_target_exec_backend() and not _has_mock_response(item):
        return os.path.join(pred_dir, "target_exec_raw.txt")
    return os.path.join(pred_dir, "conversation.json")


def _build_system_prompt(skill_content: str) -> str:
    return "\n\n".join(
        [
            "You are solving one Gitmoot task.",
            f"## Skill\n{skill_content.strip()}",
            "## Output Contract\nReturn exactly the required deliverable.",
        ]
    )


def extract_target_skill_content(skill_content: str) -> TargetSkillContext:
    text = skill_content or ""
    target_bounds = _marker_bounds(text, SKILLOPT_TARGET_START, SKILLOPT_TARGET_END)
    optimizer_bounds = _marker_bounds(text, SKILLOPT_OPTIMIZER_START, SKILLOPT_OPTIMIZER_END)
    target_malformed = _marker_pair_is_malformed(text, SKILLOPT_TARGET_START, SKILLOPT_TARGET_END)
    optimizer_malformed = _marker_pair_is_malformed(text, SKILLOPT_OPTIMIZER_START, SKILLOPT_OPTIMIZER_END)
    metadata: dict[str, Any] = {
        "sectioned": False,
        "target_section_present": target_bounds is not None,
        "optimizer_section_present": optimizer_bounds is not None,
        "isolation": "legacy_full_skill",
    }

    if target_malformed:
        metadata["warning"] = "malformed_target_section"
        return TargetSkillContext(content=text, metadata=metadata)

    if target_bounds is None:
        if optimizer_bounds is not None or optimizer_malformed:
            metadata["warning"] = "optimizer_section_without_target_section"
        else:
            metadata["warning"] = "skillopt_sections_absent"
        return TargetSkillContext(content=text, metadata=metadata)

    start, end = target_bounds
    metadata["sectioned"] = True
    metadata["isolation"] = "target_section"
    if optimizer_malformed:
        metadata["warning"] = "malformed_optimizer_section"
    return TargetSkillContext(content=text[start:end].strip(), metadata=metadata)


def _marker_bounds(text: str, start_marker: str, end_marker: str) -> tuple[int, int] | None:
    start = text.find(start_marker)
    if start == -1:
        return None
    content_start = start + len(start_marker)
    end = text.find(end_marker, content_start)
    if end == -1:
        return None
    return content_start, end


def _marker_pair_is_malformed(text: str, start_marker: str, end_marker: str) -> bool:
    return (start_marker in text or end_marker in text) and _marker_bounds(text, start_marker, end_marker) is None


def _write_prediction(pred_dir: str, result: dict[str, Any]) -> None:
    with open(os.path.join(pred_dir, "result.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    conversation = [
        {"role": "system", "content": result["target_system_prompt"]},
        {"role": "user", "content": result["target_user_prompt"]},
        {"role": "assistant", "content": result["response"]},
    ]
    with open(os.path.join(pred_dir, "conversation.json"), "w", encoding="utf-8") as handle:
        json.dump(conversation, handle, indent=2)
        handle.write("\n")
    with open(os.path.join(pred_dir, "target_system_prompt.txt"), "w", encoding="utf-8") as handle:
        handle.write(result["target_system_prompt"])
    with open(os.path.join(pred_dir, "target_user_prompt.txt"), "w", encoding="utf-8") as handle:
        handle.write(result["target_user_prompt"])
