# Contact and Balance Diagnostic Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline for this single bounded task.

**Goal:** Replace the optimization framing with actual ball output and balance, and provide faithful contact episode diagnostics.

**Architecture:** A new offline module replays existing 5 ms contact/ball/root records; old control, defaults and frozen audits stay unchanged.

**Tech Stack:** Python/NumPy, existing recorded IsaacLab evidence, unittest.

**Spec:** `docs/superpowers/specs/2026-11-04-contact-and-balance-design.md`

## Constraints and review focus

- Branch version/v0.1.6, preserve prior history and Tang assets.
- Do not count zero-force manifolds, merge separate force episodes, or infer physical separation from force alone.
- Preserve sign of directional impulse; label missing friction/COM/ground-contact evidence.
- Missing/NaN/gapped samples fail measurement; incomplete windows are explicit.
- No new controller-success claim or GPU search.

## Task 1

- [x] Write contact and balance diagnostics tests; observe missing module RED.
- [x] Implement `tools/kick_contact_metrics.py` and pass tests.
- [x] Replay matched existing demos, document objective correction and evidence gaps.
- [x] Run regression checks and one read-only final code review; publish version branch after fixes.

Independent read-only contact and balance investigations found no contact/initial-output gain despite increased instep travel, and worse outside recovery. Implementer owns all edits; no repeated experimental search.
