from __future__ import annotations

import json

from gitmoot_skillopt.cli import build_parser, main
from gitmoot_skillopt.contracts import CANDIDATE_PACKAGE_KIND, CandidatePackage
from gitmoot_skillopt.pairwise import (
    PAIRWISE_REVIEW_KIND,
    build_pairwise_packet,
    render_pairwise_markdown,
    run_pairwise_eval,
    write_pairwise_artifacts,
)
from tests.test_gitmoot_dataloader import write_training_package

_ROLE_LEAK_WORDS = ("promoted", "candidate", "champion", "challenger", "baseline", "initial")


def _val_items(ids: list[str]) -> list[dict[str, object]]:
    return [{"id": item_id, "title": f"Item {item_id}", "prompt": f"do {item_id}"} for item_id in ids]


def _side_result(item_id: str, response: str, **overrides: object) -> dict[str, object]:
    result = {
        "id": item_id,
        "response": response,
        "agent_ok": True,
        "target_status": "passed",
        "fail_reason": "",
        "n_turns": 1,
        "blocker": "",
        "token_usage": {},
    }
    result.update(overrides)
    return result


def _results(ids: list[str], prefix: str, **overrides: object) -> dict[str, dict[str, object]]:
    return {item_id: _side_result(item_id, f"{prefix}-{item_id}", **overrides) for item_id in ids}


def _write_multi_val_package(tmp_path):
    package_path, artifact_root = write_training_package(tmp_path)
    package = json.loads(package_path.read_text(encoding="utf-8"))
    package["items"].append(
        {
            "id": "val-2",
            "title": "Val item 2",
            "baseline_artifact_id": "baseline",
            "candidate_artifact_id": "candidate",
            "metadata": {"split": "val", "mock_response": "better val2", "expected_hard": False, "expected_soft": 0.25},
        }
    )
    package_path.write_text(json.dumps(package), encoding="utf-8")
    return package_path, artifact_root


def test_blinded_packet_hides_champion_mapping():
    ids = [f"item-{n}" for n in range(8)]
    items = _val_items(ids)
    # Feed role-revealing rollout-scoped trace paths on every side, exactly as a
    # real run does (out_root/rollout/promoted|candidate/...): the blinded packet
    # must NOT echo these back or it trivially unblinds the champion.
    promoted = _results(
        ids,
        "alpha",
        target_trace_path="/abs/out/rollout/promoted/predictions/x/target_exec_raw.txt",
    )
    candidate = _results(
        ids,
        "beta",
        target_trace_path="/abs/out/rollout/candidate/predictions/x/target_exec_raw.txt",
    )
    # Drop one side for one item so the missing-side fallback path is exercised:
    # its fail_reason must stay role-neutral (no "promoted"/"candidate").
    promoted.pop("item-3")

    packet, secret_map = build_pairwise_packet(
        run_id="run-1",
        template_id="planner",
        base_version_id="planner@v1",
        items=items,
        promoted_results=promoted,
        candidate_results=candidate,
        seed=7,
        promoted_content="promoted body",
        candidate_content="candidate body",
    )

    markdown = render_pairwise_markdown(packet)
    packet_json = json.dumps(packet)

    # No role wording leaks into the human-visible packet (md or json).
    for word in _ROLE_LEAK_WORDS:
        assert word not in markdown.lower()
        assert word not in packet_json.lower()

    # The rollout-scoped trace path (which encodes the role dir) never appears in
    # the packet; it is recorded only in the secret map.
    assert "/rollout/" not in packet_json
    assert "target_trace_path" not in packet_json
    secret_json = json.dumps(secret_map)
    assert "/rollout/promoted/" in secret_json
    assert "/rollout/candidate/" in secret_json

    # Per-output entries are labeled only A/B; nothing names the role.
    for packet_item in packet["items"]:
        labels = {side["label"] for side in packet_item["outputs"]}
        assert labels == {"A", "B"}
        for side in packet_item["outputs"]:
            assert "role" not in side
            assert "origin" not in side

    # A/B placement is randomized: the champion is not always the same label,
    # and side A carries a mix of promoted (alpha) and candidate (beta) outputs,
    # so position alone cannot reveal the champion.
    champion_labels = {entry["champion_label"] for entry in secret_map["items"]}
    assert champion_labels == {"A", "B"}
    a_outputs = {
        item["outputs"][0]["response"].split("-")[0]
        for item in packet["items"]
        if item["outputs"][0]["response"]
    }
    assert a_outputs == {"alpha", "beta"}


def test_secret_map_is_separate_and_round_trips(tmp_path):
    ids = ["item-0", "item-1", "item-2"]
    items = _val_items(ids)
    promoted = _results(ids, "alpha")
    candidate = _results(ids, "beta")
    packet, secret_map = build_pairwise_packet(
        run_id="run-1",
        template_id="planner",
        base_version_id="planner@v1",
        items=items,
        promoted_results=promoted,
        candidate_results=candidate,
        seed=3,
        promoted_content="promoted body",
        candidate_content="candidate body",
    )
    written = write_pairwise_artifacts(
        packet=packet,
        secret_map=secret_map,
        out_root=tmp_path / "out",
        artifact_dir=tmp_path / "out" / "artifacts",
        run_id="run-1",
    )

    # Secret map written as its own file, NOT referenced by the packet manifest.
    packet_artifact_ids = {artifact["id"] for artifact in written["artifacts"]}
    assert written["secret_map_artifact_id"] not in packet_artifact_ids
    assert (tmp_path / "out" / "artifacts" / "pairwise-secret-map.json").is_file()

    # Unblinding with the secret map recovers the promoted/candidate mapping.
    packet_by_id = {item["item_id"]: item for item in packet["items"]}
    for entry in secret_map["items"]:
        item_id = entry["item_id"]
        outputs = {side["label"]: side["response"] for side in packet_by_id[item_id]["outputs"]}
        for label, role in entry["mapping"].items():
            expected = promoted[item_id]["response"] if role == "promoted" else candidate[item_id]["response"]
            assert outputs[label] == expected


def test_per_item_token_cost_and_failure_captured(tmp_path):
    ids = ["item-0", "item-1"]
    items = _val_items(ids)
    promoted = _results(ids, "alpha", token_usage={"cost_usd": 0.02, "usage": {"input_tokens": 12}})
    candidate = _results(ids, "beta", token_usage={"total_tokens": 99})
    # One candidate-side live run failed; it must be captured, not abort the run.
    candidate["item-1"] = _side_result(
        "item-1",
        "",
        agent_ok=False,
        target_status="failed",
        fail_reason="exec target failed",
        token_usage={},
    )

    packet, secret_map = build_pairwise_packet(
        run_id="run-1",
        template_id="planner",
        base_version_id="planner@v1",
        items=items,
        promoted_results=promoted,
        candidate_results=candidate,
        seed=1,
        promoted_content="promoted body",
        candidate_content="candidate body",
    )
    write_pairwise_artifacts(
        packet=packet,
        secret_map=secret_map,
        out_root=tmp_path / "out",
        artifact_dir=tmp_path / "out" / "artifacts",
        run_id="run-1",
    )

    # Token/cost present per anonymized side in the per-item artifact.
    item0 = json.loads(
        (tmp_path / "out" / "artifacts" / "items" / "item-0" / "pairwise-item.json").read_text(encoding="utf-8")
    )
    token_blobs = [side["token_usage"] for side in item0["outputs"]]
    assert {"cost_usd": 0.02, "usage": {"input_tokens": 12}} in token_blobs
    assert {"total_tokens": 99} in token_blobs

    # The failing side is recorded per item without aborting the whole packet.
    assert packet["summary"]["item_count"] == 2
    assert packet["summary"]["failed_sides"] == 1
    item1 = json.loads(
        (tmp_path / "out" / "artifacts" / "items" / "item-1" / "pairwise-item.json").read_text(encoding="utf-8")
    )
    failed = [side for side in item1["outputs"] if side["failed"]]
    assert len(failed) == 1
    assert failed[0]["fail_reason"] == "exec target failed"


def test_safe_run_batch_swallows_catastrophic_failure(tmp_path, monkeypatch):
    from gitmoot_skillopt import pairwise

    def boom(**kwargs):
        raise RuntimeError("backend unavailable")

    monkeypatch.setattr(pairwise, "run_batch", boom)
    results = pairwise._safe_run_batch(
        items=_val_items(["a", "b"]),
        skill_content="skill",
        out_root=str(tmp_path),
        evaluator_config={"mode": "fixture"},
        max_completion_tokens=4096,
    )

    assert [r["id"] for r in results] == ["a", "b"]
    assert all(r["target_status"] == "failed" for r in results)
    assert all(r["fail_reason"] == "backend unavailable" for r in results)


def test_run_pairwise_eval_dual_rolls_out_both_templates_over_val_split(tmp_path, monkeypatch):
    from gitmoot_skillopt import pairwise

    package_path, artifact_root = _write_multi_val_package(tmp_path)
    candidate_path = tmp_path / "candidate.md"
    candidate_path.write_text("# Candidate\n\nDifferent guidance.\n", encoding="utf-8")

    calls: list[tuple[str, list[str]]] = []

    def fake_run_batch(*, items, skill_content, out_root, evaluator_config=None, max_completion_tokens=4096, **kwargs):
        del out_root, evaluator_config, max_completion_tokens, kwargs
        ids = [str(item["id"]) for item in items]
        calls.append((skill_content, ids))
        tag = "P" if "Candidate" not in skill_content else "C"
        return [_side_result(item_id, f"{tag}-{item_id}") for item_id in ids]

    monkeypatch.setattr(pairwise, "run_batch", fake_run_batch)

    summary = run_pairwise_eval(
        training_package=str(package_path),
        artifact_root=str(artifact_root),
        candidate=str(candidate_path),
        out_root=str(tmp_path / "out"),
        seed=5,
    )

    # Both templates rolled out live over the same val split (val-1, val-2).
    assert len(calls) == 2
    rolled_skills = {call[0] for call in calls}
    assert any("Candidate" in skill for skill in rolled_skills)
    assert any("Candidate" not in skill for skill in rolled_skills)
    for _skill, ids in calls:
        assert set(ids) == {"val-1", "val-2"}

    assert sorted(summary["val_items"]) == ["val-1", "val-2"]
    assert summary["mode"] == "live-pairwise"
    packet = json.loads(open(summary["packet_json_path"], encoding="utf-8").read())
    assert packet["kind"] == PAIRWISE_REVIEW_KIND
    assert packet["contract_version"] == 1
    assert {item["item_id"] for item in packet["items"]} == {"val-1", "val-2"}
    assert open(summary["packet_markdown_path"], encoding="utf-8").read().count("### Output") == 4


def test_cli_pairwise_invocation_writes_packet(tmp_path, monkeypatch, capsys):
    from gitmoot_skillopt import pairwise

    package_path, artifact_root = write_training_package(tmp_path)
    candidate_path = tmp_path / "candidate.md"
    candidate_path.write_text("# Candidate\n\nDifferent guidance.\n", encoding="utf-8")

    def fake_run_batch(*, items, skill_content, out_root, evaluator_config=None, max_completion_tokens=4096, **kwargs):
        del out_root, evaluator_config, max_completion_tokens, kwargs
        return [_side_result(str(item["id"]), f"out-{item['id']}") for item in items]

    monkeypatch.setattr(pairwise, "run_batch", fake_run_batch)

    out_root = tmp_path / "out"
    result = main(
        [
            "pairwise",
            "--training-package",
            str(package_path),
            "--artifact-root",
            str(artifact_root),
            "--candidate",
            str(candidate_path),
            "--out-root",
            str(out_root),
            "--mode",
            "live-pairwise",
        ]
    )

    assert result == 0
    output = capsys.readouterr().out
    assert "wrote live-pairwise review packet" in output
    assert (out_root / "artifacts" / "pairwise-review.md").is_file()
    assert (out_root / "artifacts" / "pairwise-review.json").is_file()
    assert (out_root / "artifacts" / "pairwise-secret-map.json").is_file()


def test_process_one_threads_exec_token_and_cost(tmp_path, monkeypatch):
    from skillopt.envs.gitmoot import rollout

    # run_target_exec wraps each attempt body behind a "===== ... ATTEMPT N ====="
    # banner, so usage parsing must survive the banner (not require a bare JSON).
    claude_json = json.dumps(
        {"total_cost_usd": 0.0123, "usage": {"input_tokens": 10, "output_tokens": 5}, "num_turns": 2}
    )
    claude_raw = f"===== CLAUDE SDK ATTEMPT 1 =====\n{claude_json}"

    def fake_exec(**kwargs):
        return "exec response", claude_raw

    monkeypatch.setenv("TARGET_DEPLOYMENT", "gpt-test")
    monkeypatch.setattr(rollout, "is_target_exec_backend", lambda: True)
    monkeypatch.setattr(rollout, "is_target_chat_backend", lambda: False)
    monkeypatch.setattr(rollout, "get_target_backend", lambda: "codex_exec")
    monkeypatch.setattr(rollout, "run_target_exec", fake_exec)
    item = {"id": "exec-item", "prompt": "Prompt", "metadata": {"expected_hard": True}, "evaluator_config": {"mode": "fixture"}}

    result = rollout.process_one(item=item, skill_content="skill", out_root=str(tmp_path))

    assert result["token_usage"]["cost_usd"] == 0.0123
    assert result["token_usage"]["usage"] == {"input_tokens": 10, "output_tokens": 5}
    assert result["token_usage"]["num_turns"] == 2
    persisted = json.loads((tmp_path / "predictions" / "exec-item" / "result.json").read_text(encoding="utf-8"))
    assert persisted["token_usage"]["cost_usd"] == 0.0123


def test_process_one_threads_codex_tokens(tmp_path, monkeypatch):
    from skillopt.envs.gitmoot import rollout

    def fake_exec(**kwargs):
        return "exec response", "some trace\ntokens used\n1,234\nmore"

    monkeypatch.setenv("TARGET_DEPLOYMENT", "gpt-test")
    monkeypatch.setattr(rollout, "is_target_exec_backend", lambda: True)
    monkeypatch.setattr(rollout, "is_target_chat_backend", lambda: False)
    monkeypatch.setattr(rollout, "get_target_backend", lambda: "codex_exec")
    monkeypatch.setattr(rollout, "run_target_exec", fake_exec)
    item = {"id": "codex-item", "prompt": "Prompt", "metadata": {"expected_hard": True}, "evaluator_config": {"mode": "fixture"}}

    result = rollout.process_one(item=item, skill_content="skill", out_root=str(tmp_path))

    assert result["token_usage"] == {"total_tokens": 1234}


def test_default_optimize_path_is_untouched(tmp_path, capsys):
    # The pairwise mode is a separate subcommand; the default saved-baseline
    # optimize path still produces a candidate package unchanged.
    parser = build_parser()
    subactions = [action for action in parser._actions if action.dest == "command"]
    assert subactions and {"optimize", "pairwise"} <= set(subactions[0].choices)

    package_path, artifact_root = write_training_package(tmp_path)
    out_root = tmp_path / "out"
    candidate_output = out_root / "candidate.json"
    result = main(
        [
            "optimize",
            "--training-package",
            str(package_path),
            "--artifact-root",
            str(artifact_root),
            "--out-root",
            str(out_root),
            "--candidate-output",
            str(candidate_output),
            "--dry-run",
        ]
    )

    assert result == 0
    loaded = CandidatePackage.from_dict(json.loads(candidate_output.read_text(encoding="utf-8")))
    assert loaded.kind == CANDIDATE_PACKAGE_KIND
    assert "wrote" in capsys.readouterr().out
