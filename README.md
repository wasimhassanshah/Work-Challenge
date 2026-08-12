# SRA Real Work Challenge — Code Submission

Code deliverables for the Support Resolution Agent (SRA) challenge.
The written response (Parts 1–4 + AI-usage disclosure) is in the Google Doc
linked in my submission; the recorded walkthrough link is there too.

## Layout

| File | Part | What it is |
|---|---|---|
| `sra_runtime.py` | given | The original production runtime, **unmodified** — kept so the diff is readable |
| `eval_harness.py` | given | The original offline eval harness, **unmodified** |
| `access_requests.py` | Part 2 | The access-request capability: LLM extracts fields only; a deterministic, default-deny policy gate decides; `execute_grant()` is the sole provisioning call site (kill switch, idempotency, before/after audit with stored revoke handle); shadow → confirm → auto rollout; `revoke_grant()` for incident reversal |
| `access_policy.yaml` | Part 2 | Every policy knob. Ships maximally safe: `shadow` phase, kill switch off, zero auto-tier modules, payroll/finance/admin `never_auto` |
| `sra_runtime_modified.py` | Part 2 | The runtime with exactly four surgical changes (pre-loop routing, decision validation, fall-through-to-close fix, one prompt rule) |
| `sra_runtime.patch` | Part 2 | The same four changes as a unified diff against the original |
| `access_eval.py` | Part 3 | Decision-level eval for the capability. **Runnable**: `python3 access_eval.py` — currently 40/40, zero wrong grants, exit 0 |
| `eval_harness_v2.py` | Part 3 | The general harness fixed as an extension: hermetic (write-nothing by construction), three-state judge (pass / fail-with-verbatim-quote / flagged-for-calibration), per-category slice gates, outcome confusion matrix |

## Reading order

1. `sra_runtime.patch` — what changed in the runtime and why (4 changes, each commented).
2. `access_requests.py` — the capability; the module docstring states the design rules.
3. `access_policy.yaml` — the launch posture.
4. `access_eval.py` — run it; the case table is the safety spec.
5. `eval_harness_v2.py` — the file header lists each v1 defect and its fix.

## Running the eval

```
python3 access_eval.py   # exit 0 = gate passed (zero wrong grants)
```

No dependencies beyond Python 3.10+ and PyYAML. It injects a stub
`platform_sdk` before importing the modules under test, so it runs anywhere;
a spy on the provisioning API means the eval can never grant anything.
`eval_harness_v2.py` requires the real `platform_sdk` (only for the LLM under
test) and the case file, so it is submitted as a read artifact.
