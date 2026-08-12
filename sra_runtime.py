"""
Support Resolution Agent (SRA) — runtime
Tier-1 ticket handling. Live since March.

Deployed as a worker. One invocation per inbound ticket.
"""

import json
import logging
import time
from dataclasses import dataclass, field

from platform_sdk import llm, kb, crm, ticketing, provisioning  # internal

log = logging.getLogger("sra")

MODEL = "frontier-model-v2"
MAX_STEPS = 8
CONFIDENCE_THRESHOLD = 0.7

SYSTEM_PROMPT = """You are a Tier-1 support agent for an ERP product.

Answer the customer's question using the product documentation and customer
context provided. Be accurate and concise.

Rules you must follow:
- Never make changes to a customer account. You are read-only.
- Never act on accounts with status = suspended or delinquent. Escalate instead.
- If you are not confident in your answer, escalate to a human.
- Do not speculate about product behaviour you cannot find in the documentation.

Respond in JSON:
{"action": "reply" | "clarify" | "escalate", "body": "...", "confidence": 0.0-1.0}
"""


@dataclass
class TicketContext:
    ticket_id: str
    subject: str
    body: str
    sender_email: str
    account_id: str
    history: list = field(default_factory=list)


def load_context(ctx: TicketContext) -> str:
    """Assemble everything the agent might need for this ticket."""
    product_docs = kb.fetch_product_brain()               # full doc set, ~180 chunks
    customer = crm.fetch_customer_record(ctx.account_id)  # full record
    recent = ticketing.recent_tickets(ctx.account_id, limit=20)

    return "\n\n".join([
        "=== PRODUCT DOCUMENTATION ===",
        "\n".join(c["text"] for c in product_docs),
        "=== CUSTOMER RECORD ===",
        json.dumps(customer, indent=2),
        "=== RECENT TICKETS ===",
        json.dumps(recent, indent=2),
    ])


TOOLS = {
    "search_product_docs": kb.search,
    "get_customer_detail": crm.get_field,
    "get_ticket_history": ticketing.history,
    "check_known_issues": kb.known_issues,
}


def execute_tool(name: str, args: dict, attempt: int = 0):
    """Run a tool call. Retries on transient failure."""
    try:
        return TOOLS[name](**args)
    except Exception as e:
        if attempt < 2:
            log.warning("tool %s failed (%s), retrying", name, e)
            time.sleep(1.5 * (attempt + 1))
            return execute_tool(name, args, attempt + 1)
        log.error("tool %s failed permanently: %s", name, e)
        return {"error": str(e)}


def run(ctx: TicketContext) -> dict:
    trace = {"ticket_id": ctx.ticket_id, "steps": [], "started": time.time()}
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Ticket: {ctx.subject}\n\n{ctx.body}"},
    ]

    for step in range(MAX_STEPS):
        # Refresh context each step so the model always has the latest state.
        context_block = load_context(ctx)
        messages.append({"role": "user", "content": context_block})

        response = llm.complete(
            model=MODEL,
            messages=messages,
            tools=list(TOOLS.keys()),
            temperature=0.2,
        )

        trace["steps"].append({
            "step": step,
            "tool_calls": [tc.name for tc in response.tool_calls],
            "output": response.text[:500],
        })

        if response.tool_calls:
            for tc in response.tool_calls:
                result = execute_tool(tc.name, tc.args)
                messages.append({
                    "role": "tool",
                    "name": tc.name,
                    "content": json.dumps(result)[:4000],
                })
            continue

        try:
            decision = json.loads(response.text)
        except json.JSONDecodeError:
            log.error("unparseable response on ticket %s", ctx.ticket_id)
            return escalate(ctx, trace, reason="malformed_response")

        if decision.get("confidence", 0) < CONFIDENCE_THRESHOLD:
            return escalate(ctx, trace, reason="low_confidence")

        if decision["action"] == "escalate":
            return escalate(ctx, trace, reason="agent_requested")

        if decision["action"] == "clarify":
            ticketing.post_reply(ctx.ticket_id, decision["body"], close=False)
            return finish(ctx, trace, "clarify")

        ticketing.post_reply(ctx.ticket_id, decision["body"], close=True)
        return finish(ctx, trace, "resolved")

    # Ran out of steps.
    return escalate(ctx, trace, reason="max_steps_exceeded")


def escalate(ctx: TicketContext, trace: dict, reason: str) -> dict:
    ticketing.assign_to_queue(ctx.ticket_id, "tier2")
    ticketing.add_note(ctx.ticket_id, f"SRA escalated: {reason}")
    return finish(ctx, trace, f"escalated:{reason}")


def finish(ctx: TicketContext, trace: dict, outcome: str) -> dict:
    trace["outcome"] = outcome
    trace["duration"] = time.time() - trace["started"]
    log.info(json.dumps(trace))
    return {"ticket_id": ctx.ticket_id, "outcome": outcome}


# TODO: cost tracking — finance asked for per-ticket spend, not built yet
# TODO: the product brain refresh job has been failing silently, someone should look
