"""
SRA offline evaluation.
Built at launch. Run manually before deploys.
"""

import json
from platform_sdk import llm
from sra_runtime import run, TicketContext

MODEL = "frontier-model-v2"

JUDGE_PROMPT = """You are grading a support agent's response to a customer ticket.

Ticket: {ticket}
Agent response: {response}
Reference answer: {reference}

Did the agent resolve the ticket correctly? Answer PASS or FAIL, then one sentence.
"""


def load_cases(path="eval_cases.json"):
    """240 cases sampled from tickets at launch, with reference answers
    written by two senior support engineers."""
    with open(path) as f:
        return json.load(f)


def judge(case, response):
    verdict = llm.complete(
        model=MODEL,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(
            ticket=case["body"],
            response=response,
            reference=case["reference_answer"],
        )}],
    )
    return verdict.text.strip().startswith("PASS")


def main():
    cases = load_cases()
    passed = 0
    for case in cases:
        ctx = TicketContext(**case["ticket"])
        result = run(ctx)
        if judge(case, result.get("body", "")):
            passed += 1
    print(f"Resolution correctness: {passed / len(cases):.1%}")


if __name__ == "__main__":
    main()
