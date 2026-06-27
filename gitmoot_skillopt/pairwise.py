"""Gitmoot SkillOpt live-pairwise evaluation mode (#77a).

An opt-in, higher-fidelity alternative to the default saved-baseline candidate
review. Instead of comparing a candidate against saved baseline outputs, this
mode reruns BOTH the currently-promoted template and the candidate template
LIVE over the validation split, then emits a BLINDED paired review packet plus
per-item artifacts for human preference review.

This module is a pure PRODUCER of a verifiable package:

* it NEVER runs the optimizer rewrite (reflect) and NEVER runs the score-gate;
* it NEVER promotes a candidate and NEVER writes ``~/.gitmoot``/SQLite/blobs;
* it only writes review artifacts under the caller-provided output directories.

The default ``run_optimize`` saved-baseline path is untouched; nothing here runs
unless the ``pairwise`` subcommand is explicitly selected.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from gitmoot_skillopt.artifacts import OutputArtifactWriter, content_hash
from gitmoot_skillopt.contracts import CONTRACT_VERSION, TrainingPackage
from gitmoot_skillopt.optimize import _read_best_skill
from skillopt.envs.gitmoot.dataloader import GitmootDataLoader
from skillopt.envs.gitmoot.package import safe_item_path_segment
from skillopt.envs.gitmoot.rollout import run_batch

PAIRWISE_REVIEW_KIND = "gitmoot-skillopt-pairwise-review-packet"
PAIRWISE_MODE = "live-pairwise"

# Anonymized side labels shown to the human reviewer. The mapping back to
# promoted (champion) / candidate (challenger) lives ONLY in the secret map.
SIDE_A = "A"
SIDE_B = "B"


def run_pairwise_eval(
    *,
    training_package: str,
    artifact_root: str,
    candidate: str,
    out_root: str,
    artifact_dir: str = "",
    seed: int = 42,
    max_completion_tokens: int = 4096,
    limit: int = 0,
    mode: str = PAIRWISE_MODE,
) -> dict[str, Any]:
    """Run the live-pairwise evaluation and write the blinded review packet.

    Returns a summary dict describing the run, the written artifacts, and the
    paths to the blinded packet and the (separate) secret map. Promotion stays
    manual: this function returns a producer summary and never promotes.
    """
    if str(mode or "").strip() != PAIRWISE_MODE:
        raise ValueError(f"unsupported pairwise mode: {mode!r} (only {PAIRWISE_MODE!r})")

    package_path = _require_file(training_package, "training package")
    artifact_root_path = _require_dir(artifact_root, "artifact root")
    out_root_path = Path(out_root).expanduser()
    out_root_path.mkdir(parents=True, exist_ok=True)
    artifact_dir_path = (
        Path(artifact_dir).expanduser() if str(artifact_dir).strip() else out_root_path / "artifacts"
    )

    package = TrainingPackage.load(package_path)
    promoted_content = package.template.content
    candidate_content = _resolve_candidate_content(candidate, fallback=promoted_content)

    loader = GitmootDataLoader(
        training_package=str(package_path),
        artifact_root=str(artifact_root_path),
        seed=seed,
        limit=limit,
    )
    loader.setup({})
    evaluator_config = dict(loader.evaluator_config)
    items = loader.get_split_items("val")
    if not items:
        raise ValueError("live-pairwise requires a non-empty val split")

    promoted_dir = str(out_root_path / "rollout" / "promoted")
    candidate_dir = str(out_root_path / "rollout" / "candidate")

    promoted_results = _safe_run_batch(
        items=items,
        skill_content=promoted_content,
        out_root=promoted_dir,
        evaluator_config=evaluator_config,
        max_completion_tokens=max_completion_tokens,
    )
    candidate_results = _safe_run_batch(
        items=items,
        skill_content=candidate_content,
        out_root=candidate_dir,
        evaluator_config=evaluator_config,
        max_completion_tokens=max_completion_tokens,
    )

    packet, secret_map = build_pairwise_packet(
        run_id=package.eval_run.id or package.template.id,
        template_id=package.template.id,
        base_version_id=package.template.version_id,
        items=items,
        promoted_results=_results_by_id(promoted_results),
        candidate_results=_results_by_id(candidate_results),
        seed=seed,
        promoted_content=promoted_content,
        candidate_content=candidate_content,
    )

    written = write_pairwise_artifacts(
        packet=packet,
        secret_map=secret_map,
        out_root=out_root_path,
        artifact_dir=artifact_dir_path,
        run_id=package.eval_run.id or package.template.id,
    )

    return {
        "mode": PAIRWISE_MODE,
        "template_id": package.template.id,
        "base_version_id": package.template.version_id,
        "val_items": [item["id"] for item in items],
        "item_count": len(items),
        "failed_sides": packet["summary"]["failed_sides"],
        "promoted_rollout_dir": promoted_dir,
        "candidate_rollout_dir": candidate_dir,
        "artifacts": written["artifacts"],
        "packet_markdown_path": written["packet_markdown_path"],
        "packet_json_path": written["packet_json_path"],
        "secret_map_path": written["secret_map_path"],
        "secret_map_artifact_id": written["secret_map_artifact_id"],
    }


def build_pairwise_packet(
    *,
    run_id: str,
    template_id: str,
    base_version_id: str,
    items: list[dict[str, Any]],
    promoted_results: dict[str, dict[str, Any]],
    candidate_results: dict[str, dict[str, Any]],
    seed: int,
    promoted_content: str = "",
    candidate_content: str = "",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Assemble the blinded packet and the (separate) secret unblinding map.

    The packet is human-visible and must NOT reveal which anonymized side (A/B)
    is the promoted champion: per item a deterministic RNG decides A/B placement,
    and no template id, "promoted"/"candidate"/"champion" label, score, or
    ordering signal leaks the mapping. The secret map is returned separately and
    is the only place that records the A/B -> promoted/candidate correspondence.
    """
    rng = random.Random(f"gitmoot-skillopt-pairwise:{seed}:{run_id}")
    packet_items: list[dict[str, Any]] = []
    secret_items: list[dict[str, Any]] = []
    failed_sides = 0

    for item in items:
        item_id = str(item["id"])
        promoted_result = promoted_results.get(item_id) or _missing_side_result(item_id)
        candidate_result = candidate_results.get(item_id) or _missing_side_result(item_id)
        champion_is_a = rng.random() < 0.5

        if champion_is_a:
            side_a_result, side_b_result = promoted_result, candidate_result
        else:
            side_a_result, side_b_result = candidate_result, promoted_result

        side_a = _blinded_side(SIDE_A, side_a_result)
        side_b = _blinded_side(SIDE_B, side_b_result)
        failed_sides += int(side_a["failed"]) + int(side_b["failed"])

        packet_items.append(
            {
                "item_id": item_id,
                "title": str(item.get("title") or ""),
                "prompt": str(item.get("prompt") or ""),
                "outputs": [side_a, side_b],
            }
        )
        secret_items.append(
            {
                "item_id": item_id,
                "champion_label": SIDE_A if champion_is_a else SIDE_B,
                "challenger_label": SIDE_B if champion_is_a else SIDE_A,
                "mapping": {
                    (SIDE_A if champion_is_a else SIDE_B): "promoted",
                    (SIDE_B if champion_is_a else SIDE_A): "candidate",
                },
                # Role-revealing trace paths live ONLY here, never in the packet.
                "promoted_trace_path": str(promoted_result.get("target_trace_path") or ""),
                "candidate_trace_path": str(candidate_result.get("target_trace_path") or ""),
            }
        )

    packet = {
        "kind": PAIRWISE_REVIEW_KIND,
        "contract_version": CONTRACT_VERSION,
        "mode": PAIRWISE_MODE,
        "template_id": template_id,
        "base_version_id": base_version_id,
        "run_id": run_id,
        "instructions": (
            "Two anonymized outputs (A and B) were produced live for each item. "
            "Record which output you prefer per item. The A/B placement is "
            "randomized per item and does not indicate any ranking."
        ),
        "items": packet_items,
        "summary": {
            "item_count": len(packet_items),
            "failed_sides": failed_sides,
        },
    }
    secret_map = {
        "kind": "gitmoot-skillopt-pairwise-secret-map",
        "contract_version": CONTRACT_VERSION,
        "run_id": run_id,
        "template_id": template_id,
        "warning": (
            "ADMIN/DEBUG ONLY. This unblinds the A/B mapping. Do not include it "
            "in the human review packet."
        ),
        "champion_role": "promoted",
        "challenger_role": "candidate",
        "promoted_content_hash": content_hash(promoted_content.encode()) if promoted_content else "",
        "candidate_content_hash": content_hash(candidate_content.encode()) if candidate_content else "",
        "items": secret_items,
    }
    return packet, secret_map


def write_pairwise_artifacts(
    *,
    packet: dict[str, Any],
    secret_map: dict[str, Any],
    out_root: Path,
    artifact_dir: Path,
    run_id: str,
) -> dict[str, Any]:
    """Write the blinded packet, per-item artifacts, and (separate) secret map.

    Mirrors ``optimize.write_candidate_package``'s use of ``OutputArtifactWriter``
    + content hashing. The blinded packet's manifest does NOT reference the
    secret map, keeping the unblinding map out of the human-visible deliverable.
    """
    writer = OutputArtifactWriter(out_root, artifact_dir)
    artifacts: list[Any] = []

    markdown = render_pairwise_markdown(packet)
    packet_md_entry = writer.write_bytes(
        "pairwise-review.md",
        markdown.encode(),
        artifact_id=f"{run_id}/pairwise-review",
        media_type="text/markdown",
        driver="gitmoot-skillopt",
    )
    artifacts.append(packet_md_entry)

    packet_json_entry = writer.write_bytes(
        "pairwise-review.json",
        json.dumps(packet, indent=2, sort_keys=True).encode() + b"\n",
        artifact_id=f"{run_id}/pairwise-review-json",
        media_type="application/json",
        driver="gitmoot-skillopt",
    )
    artifacts.append(packet_json_entry)

    for packet_item in packet["items"]:
        segment = safe_item_path_segment(str(packet_item["item_id"]))
        for side in packet_item["outputs"]:
            label = str(side["label"]).lower()
            artifacts.append(
                writer.write_bytes(
                    f"items/{segment}/output-{label}.txt",
                    str(side["response"]).encode(),
                    artifact_id=f"{run_id}/pairwise-item/{segment}/output-{label}",
                    media_type="text/plain",
                    driver="gitmoot-skillopt",
                )
            )
        artifacts.append(
            writer.write_bytes(
                f"items/{segment}/pairwise-item.json",
                json.dumps(packet_item, indent=2, sort_keys=True).encode() + b"\n",
                artifact_id=f"{run_id}/pairwise-item/{segment}/item",
                media_type="application/json",
                driver="gitmoot-skillopt",
            )
        )

    # Secret map is written as its OWN artifact and is intentionally NOT added to
    # the blinded packet manifest entries above (admin/debug unblind only).
    secret_entry = writer.write_bytes(
        "pairwise-secret-map.json",
        json.dumps(secret_map, indent=2, sort_keys=True).encode() + b"\n",
        artifact_id=f"{run_id}/pairwise-secret-map",
        media_type="application/json",
        driver="gitmoot-skillopt",
    )

    return {
        "artifacts": [entry.to_dict() for entry in artifacts],
        "packet_markdown_path": str(artifact_dir / packet_md_entry.path),
        "packet_json_path": str(artifact_dir / packet_json_entry.path),
        "secret_map_path": str(artifact_dir / secret_entry.path),
        "secret_map_artifact_id": secret_entry.id,
    }


def render_pairwise_markdown(packet: dict[str, Any]) -> str:
    """Render the blinded paired review packet as markdown.

    Shows both outputs anonymously (A/B) per item. Emits no template id,
    promoted/candidate/champion wording, or score so the champion stays hidden.
    """
    lines = [
        "# Gitmoot SkillOpt Live-Pairwise Review",
        "",
        packet.get("instructions", ""),
        "",
        f"Items: {packet['summary']['item_count']}",
        "",
    ]
    for index, packet_item in enumerate(packet["items"], start=1):
        title = packet_item.get("title") or packet_item["item_id"]
        lines.append(f"## Item {index}: {title}")
        lines.append("")
        for side in packet_item["outputs"]:
            lines.append(f"### Output {side['label']}")
            if side["failed"]:
                lines.append(f"_Live run failed: {side['fail_reason'] or 'unknown error'}_")
            lines.append("")
            lines.append("```")
            lines.append(str(side["response"]).rstrip("\n"))
            lines.append("```")
            lines.append("")
        lines.append("Preferred output (A or B): ____")
        lines.append("")
    return "\n".join(lines)


def _blinded_side(label: str, result: dict[str, Any]) -> dict[str, Any]:
    target_status = str(result.get("target_status") or "")
    fail_reason = str(result.get("fail_reason") or "")
    failed = bool(result.get("agent_ok") is False or target_status == "failed" or result.get("agent_error"))
    token_usage = result.get("token_usage") if isinstance(result.get("token_usage"), dict) else {}
    # NOTE: never copy role/filesystem-derived fields (e.g. ``target_trace_path``,
    # which encodes the ``promoted``/``candidate`` rollout dir) into the blinded
    # side. Doing so would unblind the champion. Those live only in the secret map.
    return {
        "label": label,
        "response": str(result.get("response") or ""),
        "failed": failed,
        "fail_reason": fail_reason,
        "target_status": target_status,
        "token_usage": dict(token_usage),
        "runtime": {
            "n_turns": result.get("n_turns"),
            "blocker": str(result.get("blocker") or ""),
        },
    }


def _missing_side_result(item_id: str) -> dict[str, Any]:
    # fail_reason is surfaced in the blinded packet, so it must stay role-neutral.
    return {
        "id": item_id,
        "response": "",
        "agent_ok": False,
        "agent_error": True,
        "target_status": "failed",
        "fail_reason": "live rollout produced no result for this item",
        "blocker": "pairwise_side_missing",
        "token_usage": {},
    }


def _results_by_id(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    mapped: dict[str, dict[str, Any]] = {}
    for result in results:
        if isinstance(result, dict) and result.get("id") is not None:
            mapped[str(result["id"])] = result
    return mapped


def _safe_run_batch(
    *,
    items: list[dict[str, Any]],
    skill_content: str,
    out_root: str,
    evaluator_config: dict[str, Any],
    max_completion_tokens: int,
) -> list[dict[str, Any]]:
    """Run one side's live rollout, fail-closed per item.

    Runs response-only (``skip_evaluation=True``): the blinded packet uses only
    the agent response/usage/agent_ok, never the hard/soft scores, so we never
    invoke the per-item evaluator/judge (avoiding needless LLM/jury spend and a
    network-dependent failure surface). ``process_one`` still captures per-item
    agent failures as structured unscored results. If the whole batch raises
    before returning (a catastrophic side failure), degrade to per-item failure
    results so one bad live run never aborts the pairwise packet.
    """
    try:
        return run_batch(
            items=items,
            skill_content=skill_content,
            out_root=out_root,
            evaluator_config=evaluator_config,
            max_completion_tokens=max_completion_tokens,
            skip_evaluation=True,
        )
    except Exception as exc:  # noqa: BLE001 - fail-closed: record, do not abort.
        reason = str(exc) or "live rollout batch failed"
        return [
            {
                "id": str(item["id"]),
                "response": "",
                "agent_ok": False,
                "agent_error": True,
                "target_status": "failed",
                "fail_reason": reason,
                "blocker": "pairwise_batch_failed",
                "token_usage": {},
            }
            for item in items
        ]


def _resolve_candidate_content(candidate: str, *, fallback: str) -> str:
    text = str(candidate or "").strip()
    if not text:
        raise ValueError("live-pairwise requires a --candidate template, package, or run directory")
    path = Path(text).expanduser()
    if path.is_dir():
        return _read_best_skill(path, fallback)
    if not path.is_file():
        raise FileNotFoundError(f"candidate not found: {path}")
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            nested = data.get("candidate")
            if isinstance(nested, dict) and isinstance(nested.get("content"), str) and nested["content"].strip():
                return nested["content"]
    return raw


def _require_file(path_text: str, label: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _require_dir(path_text: str, label: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_dir():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path
