"""Budgeted, secrets-redacted trajectory digest for the Gitmoot judge (#348 Phase 1).

This module turns the on-disk raw exec output of an exec-backend rollout
(``target_exec_raw.txt`` / ``codex_raw.txt`` under a prediction directory) into a
compact ``## Process Summary`` body the LLM judge can read. It REUSES
:func:`skillopt.model.codex_harness.parse_codex_raw` to structure the raw trace
and never emits raw chain-of-thought.

Hard rules (all fail-closed / best-effort):

* Only exec backends write the raw files, so a missing file (any other backend)
  yields an empty digest and therefore NO section.
* Any IO/parse error, a missing/empty/garbage trace, or a missing prediction
  directory yields an empty digest. This module NEVER raises.
* The output is REDACTED (tokens/keys/credentials removed) and byte-CAPPED so the
  digest can never exceed its budget.

The caller (the evaluator) is responsible for gating this behind an
``evaluator_config`` flag so it is OFF BY DEFAULT and the judge prompt stays
byte-identical when disabled.
"""

from __future__ import annotations

import os
import re

from skillopt.model.codex_harness import parse_codex_raw

# ~3k tokens of budget for the digest body (a conservative slice of the ~4k cap).
DEFAULT_MAX_BYTES = 12000

# Hard ceiling on how much raw trace we materialize before any cap runs, so a
# pathological multi-hundred-MB raw file cannot be fully read into memory. Far
# larger than any sane digest budget, so it never truncates a real trace.
_MAX_RAW_CHARS = 8_000_000

_REDACTED = "[REDACTED]"

# Exec-backend raw trace files, in preference order. ``target_exec_raw.txt`` is
# the canonical alias written for every exec backend; ``codex_raw.txt`` is the
# codex-specific fallback. Non-exec backends write neither (they emit
# ``conversation.json``), which is exactly how exec-only gating happens.
_RAW_FILE_NAMES = ("target_exec_raw.txt", "codex_raw.txt")

# Standalone secret token shapes -> wholesale [REDACTED].
_SECRET_PATTERNS = (
    re.compile(r"gh[pousr]_[A-Za-z0-9]{16,}"),
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}"),
    # Scheme-aware auth credentials: redact the token after the scheme word so
    # two-token headers like ``Authorization: Basic <base64-cred>`` do not leak
    # the credential (only the scheme word would otherwise be caught by _KV_SECRET).
    re.compile(r"(?i)(?:bearer|basic|token|digest)\s+[A-Za-z0-9._\-=/+]{8,}"),
    re.compile(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"
    ),
)

# ``key=value`` / ``key: value`` credential assignments -> keep the key, drop the value.
# The key match covers identifiers that merely CONTAIN a sensitive word so compound
# env-var names (AWS_SECRET_ACCESS_KEY, GITHUB_TOKEN, DATABASE_PASSWORD,
# MY_REFRESH_TOKEN) are redacted, not just clean-boundary names like API_KEY.
_KV_SECRET = re.compile(
    r"(?i)\b([A-Za-z0-9_-]*"
    r"(?:authorization|token|secret|password|passwd|pwd|api[_-]?key|apikey|"
    r"credential|auth)"
    r"[A-Za-z0-9_-]*)(\s*[:=]\s*)[\"']?[^\s\"'`]+"
)

_TEST_CMD = re.compile(
    r"\b(pytest|unittest|tox|nox|go\s+test|npm\s+(?:run\s+)?test|yarn\s+test|"
    r"pnpm\s+test|jest|vitest|mocha|ruff|flake8|mypy|pyright|eslint|"
    r"cargo\s+test|make\s+test|ctest|rspec|phpunit)\b"
)

_VIOLATION_HINT = re.compile(
    r"(?i)(refus(?:e|ing)\b|permission denied|operation not permitted|not allowed|"
    r"sandbox|blocked by|access denied|forbidden|EACCES|read-only file system|"
    r"network (?:is )?unreachable|denied data director)"
)

_PATCH_FILE = re.compile(r"\*\*\*\s+(Add|Update|Delete) File:\s+(.+?)\s*$", re.MULTILINE)
_DURATION = re.compile(r"\b(?:succeeded|failed)\s+in\s+([\d.]+)\s*(ms|s|m)\b", re.IGNORECASE)

# Per-section item caps keep the pre-byte-cap rendering bounded.
_MAX_COMMANDS = 25
_MAX_FILES = 30
_MAX_TESTS = 20
_MAX_ERRORS = 8
_MAX_VIOLATIONS = 8
_MAX_SEQUENCE = 40
_CMD_CHARS = 200
_FINAL_CHARS = 600


def build_trajectory_digest(prediction_dir: str | None, *, max_bytes: int = DEFAULT_MAX_BYTES) -> str:
    """Return a budgeted, redacted process digest for *prediction_dir*.

    Returns an empty string (caller omits the section) for every failure mode:
    no/garbage prediction dir, a non-exec backend (no raw file), an empty or
    unparseable trace, or any unexpected error. Never raises.
    """
    try:
        raw = _read_exec_raw(prediction_dir)
        if not raw or not raw.strip():
            return ""
        parsed = parse_codex_raw(raw)
        steps = parsed.get("steps") if isinstance(parsed, dict) else None
        if not steps:
            return ""
        body = _render(steps)
        if not body.strip():
            return ""
        body = _redact(body)
        return _enforce_byte_cap(body, max_bytes)
    except Exception:
        # Fail-closed: a digest must never break scoring.
        return ""


def _read_exec_raw(prediction_dir: str | None) -> str:
    pred = str(prediction_dir or "").strip()
    if not pred or not os.path.isdir(pred):
        return ""
    for name in _RAW_FILE_NAMES:
        path = os.path.join(pred, name)
        if os.path.isfile(path):
            with open(path, encoding="utf-8", errors="replace") as handle:
                return handle.read(_MAX_RAW_CHARS)
    return ""


def _render(steps: list[dict]) -> str:
    type_counts: dict[str, int] = {}
    sequence: list[str] = []
    commands: list[tuple[str, str, float | None]] = []
    tests: list[tuple[str, str]] = []
    files: list[str] = []
    seen_files: set[str] = set()
    error_lines: list[str] = []
    seen_errors: set[str] = set()
    error_line_count = 0
    violations: list[str] = []
    seen_violations: set[str] = set()
    failed_commands = 0
    total_seconds = 0.0
    last_codex_message = ""

    for step in steps:
        stype = str(step.get("type") or "?")
        type_counts[stype] = type_counts.get(stype, 0) + 1
        sequence.append(stype)
        content = str(step.get("content") or "")

        for match in _PATCH_FILE.finditer(content):
            path = match.group(2).strip()
            if path and path not in seen_files:
                seen_files.add(path)
                files.append(f"{match.group(1).lower()}: {path}")

        for hint in _VIOLATION_HINT.finditer(content):
            if len(violations) >= _MAX_VIOLATIONS:
                break
            line = _line_of(content, hint.start())
            # Bound collection (not just the render slice) and use a set for O(1)
            # dedup so a verbose trace cannot drive O(n^2) membership scans.
            if line and line not in seen_violations:
                seen_violations.add(line)
                violations.append(line)

        if stype == "exec":
            cmd, status, seconds = _summarize_exec(content)
            if seconds is not None:
                total_seconds += seconds
            if status == "failed":
                failed_commands += 1
            commands.append((cmd, status, seconds))
            if cmd and _TEST_CMD.search(cmd):
                tests.append((cmd, status))
            for line in content.splitlines():
                low = line.lower()
                if "error" in low or "traceback" in low or "exception" in low:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    error_line_count += 1
                    # Bound the distinct-line collection and use a set for O(1)
                    # dedup; a huge failing log cannot drive O(n^2) scans or grow
                    # the stored list without limit (the count metric stays exact).
                    if len(error_lines) < _MAX_ERRORS and stripped not in seen_errors:
                        seen_errors.add(stripped)
                        error_lines.append(stripped)
        elif stype == "codex":
            text = " ".join(part.strip() for part in content.splitlines() if part.strip())
            if text:
                last_codex_message = text

    out: list[str] = []

    seq_render = " -> ".join(sequence[:_MAX_SEQUENCE])
    if len(sequence) > _MAX_SEQUENCE:
        seq_render += " -> ..."
    counts_render = ", ".join(f"{k}: {v}" for k, v in sorted(type_counts.items()))
    out.append("### Tool calls")
    out.append(f"- total steps: {len(steps)} ({counts_render})")
    if seq_render:
        out.append(f"- sequence: {seq_render}")

    out.append("")
    out.append(f"### Commands run ({len(commands)})")
    if commands:
        for cmd, status, seconds in commands[:_MAX_COMMANDS]:
            out.append(f"- `{_clip(cmd, _CMD_CHARS)}` — {_status_label(status, seconds)}")
        if len(commands) > _MAX_COMMANDS:
            out.append(f"- ...(+{len(commands) - _MAX_COMMANDS} more)")
    else:
        out.append("- none detected")

    out.append("")
    out.append(f"### Tests ({len(tests)})")
    if tests:
        for cmd, status in tests[:_MAX_TESTS]:
            out.append(f"- `{_clip(cmd, _CMD_CHARS)}` — {status}")
        if len(tests) > _MAX_TESTS:
            out.append(f"- ...(+{len(tests) - _MAX_TESTS} more)")
    else:
        out.append("- none detected")

    out.append("")
    out.append(f"### Files touched ({len(files)})")
    if files:
        for path in files[:_MAX_FILES]:
            out.append(f"- {_clip(path, _CMD_CHARS)}")
        if len(files) > _MAX_FILES:
            out.append(f"- ...(+{len(files) - _MAX_FILES} more)")
    else:
        out.append("- none detected")

    out.append("")
    out.append("### Errors / retries / recovery")
    out.append(f"- failed commands: {failed_commands}")
    out.append(f"- error lines observed: {error_line_count}")
    out.append(f"- recovery: {_recovery_note(commands, failed_commands)}")
    for line in error_lines[:_MAX_ERRORS]:
        out.append(f"- {_clip(line, _CMD_CHARS)}")

    out.append("")
    out.append("### Constraint violations")
    if violations:
        for line in violations[:_MAX_VIOLATIONS]:
            out.append(f"- {_clip(line, _CMD_CHARS)}")
    else:
        out.append("- none detected")

    out.append("")
    out.append("### Elapsed")
    out.append(f"- total measured exec time: ~{total_seconds:.1f}s across {len(commands)} command(s)")

    out.append("")
    out.append("### Final artifact summary")
    out.append(f"- files changed: {len(files)}")
    if last_codex_message:
        out.append(f"- final agent note: {_clip(last_codex_message, _FINAL_CHARS)}")

    return "\n".join(out)


def _summarize_exec(content: str) -> tuple[str, str, float | None]:
    body = [ln.rstrip() for ln in content.splitlines()]
    non_empty = [ln for ln in body if ln.strip()]
    cmd = non_empty[0].strip() if non_empty else ""
    if cmd.startswith("$ "):
        cmd = cmd[2:].strip()
    status = "ran"
    seconds: float | None = None
    for line in non_empty:
        low = line.lower()
        match = _DURATION.search(line)
        if match:
            seconds = _seconds(match.group(1), match.group(2))
        if "succeeded in" in low:
            status = "succeeded"
        elif "failed in" in low or low.startswith("error") or " error" in low:
            status = "failed"
        elif "timed out" in low:
            status = "timed out"
    return cmd, status, seconds


def _seconds(value: str, unit: str) -> float | None:
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return None
    unit = unit.lower()
    if unit == "ms":
        return amount / 1000.0
    if unit == "m":
        return amount * 60.0
    return amount


def _status_label(status: str, seconds: float | None) -> str:
    if seconds is None:
        return status
    return f"{status} ({seconds:.2f}s)"


def _recovery_note(commands: list[tuple[str, str, float | None]], failed_commands: int) -> str:
    if failed_commands == 0:
        return "no failures"
    # A failed command whose identical command later succeeds = a retry/recovery.
    last_status: dict[str, str] = {}
    recovered = False
    for cmd, status, _ in commands:
        prev = last_status.get(cmd)
        if prev == "failed" and status == "succeeded":
            recovered = True
        last_status[cmd] = status
    return "retry succeeded after failure" if recovered else "no successful retry of a failed command"


def _line_of(content: str, index: int) -> str:
    start = content.rfind("\n", 0, index) + 1
    end = content.find("\n", index)
    if end == -1:
        end = len(content)
    return content[start:end].strip()


def _clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _redact(text: str) -> str:
    redacted = text
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub(_REDACTED, redacted)
    redacted = _KV_SECRET.sub(lambda m: f"{m.group(1)}{m.group(2)}{_REDACTED}", redacted)
    return redacted


def _enforce_byte_cap(text: str, max_bytes: int) -> str:
    if max_bytes <= 0:
        return ""
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    marker = "\n…[digest truncated]"
    marker_bytes = len(marker.encode("utf-8"))
    budget = max_bytes - marker_bytes
    if budget <= 0:
        # Marker alone overflows the budget; return a hard byte slice.
        return encoded[:max_bytes].decode("utf-8", errors="ignore")
    head = encoded[:budget].decode("utf-8", errors="ignore")
    return head + marker
