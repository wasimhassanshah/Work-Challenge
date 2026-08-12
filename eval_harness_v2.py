"""
Offline evaluation harness v2 — an extension of eval_harness.py, not a
rewrite: load_cases, the case JSON format, and the judge concept are kept.

Fixes, in the order the defects were diagnosed (Part 1):

1. Grades the reply the agent ACTUALLY POSTED. The v1 harness judged
   result.get("body", "") but run() returns only {ticket_id, outcome} — the
   judge graded the empty string on all 240 cases. Here a RecordingTicketing
   fake captures the posted body, so run()'s contract is unchanged.
2. Hermetic. v1 posts, closes, and queues 240 REAL tickets per run — and the
   moment a write path exists in the runtime, any access-request case it
   exercises would fire a real, no-undo grant, with no seam to stop it.
   Here fixture-backed fakes replace platform_sdk BEFORE the runtime import;
   only the LLM under test is real. The access path is stubbed inert with a
   shadow fixture config and a temp ledger (it has its own dedicated gate,
   access_eval.py), and a provisioning tripwire is asserted clean after
   every case: an eval run writes nothing, anywhere.
3. Versioned fixtures. v1's launch-era references would (once the plumbing
   is fixed) PASS a Trace-A-style v13 answer. Cases here carry
   product_version and are served a KB fixture for that version, so post-v14
   cases can be represented. Re-authoring the error/troubleshooting and
   configuration references post-v14 is a content task this file enables
   but cannot do.
4. Judge v2, three-state. Different model from the agent (v1 self-graded
   with frontier-model-v2 on both sides), JSON rubric verdict instead of
   startswith("PASS"), and a mandatory verbatim quote for any FAIL. A FAIL
   whose quote is missing or does not appear verbatim in the reply — or
   judge output that does not parse — is FLAGGED, not scored: flagged cases
   are excluded from the pass-rate denominator, reported separately, and a
   high flag rate fails the run (the judge needs calibration — against the
   weekly n=50 human QA labels, the only trusted signal the org has).
   Expected escalations/clarifies are credited as OUTCOMES, never judged as
   text.
5. Reporting. Never a single number: per-category slices with thresholds,
   an expected-vs-actual outcome confusion matrix, flagged-case counts,
   per-case llm-call counts and near-duplicate-search flags (token-overlap,
   so Trace B's four near-identical queries would be flagged), and a
   nonzero exit code so this can gate deploys in CI.

Case schema (superset of v1's; v1 cases still load with defaults):
  {"ticket": {...TicketContext fields...},
   "reference_answer": str,
   "category": "informational" | "configuration" | "error_troubleshooting",
   "expected_outcome": "resolved" | "clarify" | "escalated",
   "product_version": "v14",
   "fixtures": {"kb_docs": [...], "customer": {...}}}

Note on the seam: this is module-level substitution (sys.modules) rather
than a runtime dependency-injection seam — deliberately, so the runtime diff
stays at zero. It requires a fresh process (nothing may import the real
platform_sdk first).
"""

import importlib
import json
import sys
import tempfile
import types

# Eval knobs — inline so this stays a single-file extension of v1; move to a
# yaml beside access_policy.yaml if they grow.
JUDGE_MODEL = "frontier-model-v3"   # any model family != the agent's
                                    # frontier-model-v2 (name presumed)
# Gates are deliberately NOT the launch accuracy values (93/89/86): an
# offline LLM judge is not commensurable with human QA sampling. They are
# regression floors for THIS suite; tune from the first calibrated runs.
SLICE_THRESHOLDS = {"informational": 0.90, "configuration": 0.85,
                    "error_troubleshooting": 0.80, "_default": 0.80}
MAX_JUDGE_FLAG_RATE = 0.10          # more than this and the JUDGE fails the run
MAX_STEPS_FLAG = 5                  # flag cases that use more (Trace B: 8)


# ------------------------------------------------------------------- fakes

class ForbiddenProvisioning:
    """The eval must NEVER provision. The runtime's own exception handling
    could swallow a raise, so detection is a touched-flag asserted after
    every case, not just the exception."""
    def __init__(self):
        self.touched = False

    def __getattr__(self, name):
        if name.startswith("_") or name == "touched":
            raise AttributeError(name)
        def _refuse(*a, **k):
            self.touched = True
            raise AssertionError(f"eval touched provisioning.{name} — forbidden")
        return _refuse


class RecordingTicketing:
    """Captures what would have been posted; writes nothing anywhere."""
    def __init__(self):
        self.replies, self.notes, self.queues = [], [], []

    def post_reply(self, ticket_id, body, close):
        self.replies.append({"ticket_id": ticket_id, "body": body, "close": close})

    def add_note(self, ticket_id, note):
        self.notes.append((ticket_id, note))

    def assign_to_queue(self, ticket_id, queue):
        self.queues.append((ticket_id, queue))

    def recent_tickets(self, account_id, limit=20):
        return []

    def history(self, *a, **k):
        return []

    def last_reply_body(self, ticket_id):
        for r in reversed(self.replies):
            if r["ticket_id"] == ticket_id:
                return r["body"]
        return ""

    def reset(self):
        self.replies.clear(); self.notes.clear(); self.queues.clear()


class FixtureKB:
    """Serves the case's versioned KB fixture. Token-overlap retrieval — a
    stated fidelity limit vs production semantic search, not a claim of
    equivalence. Near-duplicate searches are flagged by token overlap."""
    def __init__(self):
        self.docs = []
        self.search_args = []

    def load(self, docs):
        self.docs = docs or []
        self.search_args = []

    def fetch_product_brain(self):
        return self.docs

    def search(self, query, **kwargs):
        self.search_args.append(str(query))
        q = set(str(query).lower().split())
        scored = sorted(
            ((len(q & set(d.get("text", "").lower().split())), d) for d in self.docs),
            key=lambda x: -x[0])
        return [d for score, d in scored if score > 0][:5]

    def known_issues(self, *a, **k):
        return [d for d in self.docs if d.get("known_issue")]

    def near_duplicate_searches(self, threshold=0.6):
        seen, dupes = [], 0
        for q in self.search_args:
            toks = set(q.lower().split())
            if any(toks and s and len(toks & s) / len(toks | s) >= threshold
                   for s in seen):
                dupes += 1
            seen.append(toks)
        return dupes


class FixtureCRM:
    def __init__(self):
        self.record = {}

    def fetch_customer_record(self, account_id):
        return self.record

    def get_field(self, *a, **k):
        return self.record.get(a[-1]) if a else None


class CountingLLM:
    """Passes through to the real LLM; counts calls per case (cost proxy)."""
    def __init__(self, real_llm):
        self._real = real_llm
        self.calls = 0

    def complete(self, *a, **k):
        self.calls += 1
        return self._real.complete(*a, **k)


def build_hermetic_runtime():
    real_sdk = importlib.import_module("platform_sdk")  # real LLM only
    fake = types.ModuleType("platform_sdk")
    fake.llm = CountingLLM(real_sdk.llm)
    fake.kb = FixtureKB()
    fake.crm = FixtureCRM()
    fake.ticketing = RecordingTicketing()
    fake.provisioning = ForbiddenProvisioning()
    sys.modules["platform_sdk"] = fake
    runtime = importlib.import_module("sra_runtime_modified")

    # Stub the access path inert: it has its own dedicated gate
    # (access_eval.py). A fixture config with a TEMP ledger means no eval run
    # can ever read live rollout config or write to the production audit
    # ledger, and a canned non-access extractor means no per-case extraction
    # spend and no access-path routing inside the general suite.
    access = importlib.import_module("access_requests")
    eval_cfg = {
        "rollout_phase": "shadow", "granting_enabled": False,
        "confirm_queue": "eval-confirm", "escalate_queue": "tier2",
        "ledger_path": tempfile.mkstemp(suffix="_eval_ledger.jsonl")[1],
        "extraction": {"model": "unused", "max_ticket_chars": 6000},
        "policy": {"admin_record_max_age_days": 90,
                   "rate_limit_grants_per_window": 3,
                   "rate_limit_window_hours": 24,
                   "require_target_domain_match": True},
        "module_tiers": {"payroll": "never_auto"},
        "replies": {"granted": "unused"},
    }
    access.load_policy_config = lambda path=None, _c=eval_cfg: _c
    access.llm = types.SimpleNamespace(complete=lambda **kw: types.SimpleNamespace(
        text='{"is_access_request": false}', tool_calls=[]))
    return runtime, fake


# ------------------------------------------------------------------- judge

JUDGE_PROMPT_V2 = """You are grading a support agent's reply against a reference answer.

Rubric — the reply FAILS if any of these hold:
- It contradicts the reference answer on a fact, step, or error-code meaning.
- It invents product behaviour, settings, or steps absent from the reference.
- It answers a different question than the ticket asks.
The reply PASSES if it is factually consistent with the reference, even if
shorter, differently worded, or if it reasonably defers detail.

Ticket: {ticket}
Reference answer (current product version {version}): {reference}
Agent reply: {response}

Respond with ONLY this JSON:
{{"verdict": "PASS" or "FAIL",
  "quote": "<for FAIL: the exact offending text quoted verbatim from the reply, else null>",
  "reason": "<one sentence>"}}
A FAIL without a verbatim quote from the reply is invalid — re-check before failing.
"""


def _extract_json(text):
    s, e = text.find("{"), text.rfind("}")
    if s == -1 or e <= s:
        raise json.JSONDecodeError("no json object found", text, 0)
    return json.loads(text[s:e + 1])


def judge_v2(llm, case, response_body):
    """Returns (status, reason) with status in {'pass', 'fail', 'flagged'}.
    Flagged = the JUDGE misbehaved (missing/hallucinated quote, unparseable
    output). Flagged cases are excluded from pass-rate math and counted
    against MAX_JUDGE_FLAG_RATE — the judge gets calibrated, not obeyed, and
    the agent is neither rewarded nor punished for a judge failure."""
    verdict = llm.complete(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": JUDGE_PROMPT_V2.format(
            ticket=case["body"],
            reference=case["reference_answer"],
            version=case.get("product_version", "unversioned"),
            response=response_body,
        )}],
        temperature=0.0,
    )
    try:
        parsed = _extract_json(verdict.text)
    except (json.JSONDecodeError, AttributeError):
        return "flagged", "judge_output_unparseable"
    if parsed.get("verdict") == "PASS":
        return "pass", parsed.get("reason", "")
    quote = parsed.get("quote")
    if not quote or quote not in response_body:
        return "flagged", "judge_fail_without_verbatim_quote"
    return "fail", parsed.get("reason", "")


# -------------------------------------------------------------------- main

def load_cases(path="eval_cases.json"):
    """Kept from v1. Old-format cases get defaults so the suite still runs
    while references are re-authored post-v14."""
    with open(path) as f:
        cases = json.load(f)
    for c in cases:
        c.setdefault("category", "uncategorized")
        c.setdefault("expected_outcome", "resolved")
        c.setdefault("product_version", "v13-era")
        c.setdefault("fixtures", {})
    return cases


def main():
    runtime, sdk = build_hermetic_runtime()
    cases = load_cases()
    rows, slices, confusion = [], {}, {}
    safety_breach = False

    for case in cases:
        sdk.ticketing.reset()
        sdk.kb.load(case["fixtures"].get("kb_docs"))
        sdk.crm.record = case["fixtures"].get("customer", {})
        sdk.llm.calls = 0
        sdk.provisioning.touched = False

        ctx = runtime.TicketContext(**case["ticket"])
        result = runtime.run(ctx)
        outcome = result["outcome"].split(":")[0]  # escalated:<reason> -> escalated
        expected = case["expected_outcome"]
        confusion[(expected, outcome)] = confusion.get((expected, outcome), 0) + 1

        if sdk.provisioning.touched:
            status, why = "fail", "EVAL SAFETY: provisioning was touched"
            safety_breach = True
        elif expected in ("escalated", "clarify"):
            # Correct escalation/clarify is an OUTCOME, credited directly —
            # v1 judged these paths as empty text and failed them.
            status = "pass" if outcome == expected else "fail"
            why = f"outcome={result['outcome']}"
        elif outcome != "resolved":
            status, why = "fail", f"expected resolved, got {result['outcome']}"
        else:
            body = sdk.ticketing.last_reply_body(ctx.ticket_id)
            status, why = judge_v2(sdk.llm._real, case, body)

        flags = []
        if sdk.llm.calls > MAX_STEPS_FLAG:
            flags.append(f"steps={sdk.llm.calls}>{MAX_STEPS_FLAG}")
        dupes = sdk.kb.near_duplicate_searches()
        if dupes:
            flags.append(f"near_duplicate_searches={dupes}")

        cat = case["category"]
        slices.setdefault(cat, []).append(status)
        rows.append((case["ticket"].get("ticket_id", "?"), cat, expected,
                     outcome, status, ";".join(flags), why))

    print(f"{'ticket':10} {'category':22} {'expected':10} {'got':10} "
          f"{'result':8} {'flags':28} note")
    for t, cat, exp, got, status, flags, why in rows:
        print(f"{t:10} {cat:22} {exp:10} {got:10} {status.upper():8} "
              f"{flags:28} {why if status != 'pass' else ''}")

    print("\nExpected vs actual outcome (all cases):")
    for (exp, got), n in sorted(confusion.items()):
        print(f"  {exp:10} -> {got:10} x{n}")

    print("\nPer-category slices (flagged excluded from denominator):")
    exit_code = 0
    total_flagged = sum(s.count("flagged") for s in slices.values())
    for cat, statuses in sorted(slices.items()):
        p, f = statuses.count("pass"), statuses.count("fail")
        judged = p + f
        rate = (p / judged) if judged else 0.0
        thresh = SLICE_THRESHOLDS.get(cat, SLICE_THRESHOLDS["_default"])
        ok = rate >= thresh
        if not ok:
            exit_code = 1
        print(f"  {cat:24} {p}/{judged} = {rate:.0%}  (gate {thresh:.0%})  "
              f"flagged={statuses.count('flagged')}  {'OK' if ok else 'BELOW THRESHOLD'}")

    flag_rate = total_flagged / max(1, len(rows))
    if flag_rate > MAX_JUDGE_FLAG_RATE:
        print(f"\nJUDGE FLAG RATE {flag_rate:.0%} > {MAX_JUDGE_FLAG_RATE:.0%} — "
              f"the judge needs calibration (vs the weekly human QA labels); "
              f"run is not trustworthy.")
        exit_code = 1
    if safety_breach:
        print("\nEVAL SAFETY BREACH: provisioning was touched during an eval run.")
        exit_code = 1
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
