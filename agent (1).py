"""
Your First Agent — ADK lab
A single ADK agent that answers multi-step questions about orders,
using two tools: lookup_order and calculate.
"""

import asyncio
import json
import os
from dotenv import load_dotenv
load_dotenv()

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

ORDERS_FILE = os.path.join(os.path.dirname(__file__), "orders.json")


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def lookup_order(order_id: str) -> dict:
    """Looks up an order by its ID and returns its item, price, purchase date,
    and warranty length in months.

    Args:
        order_id: The order ID to look up, e.g. "A1001".

    Returns:
        A dict with order details (item, price, purchased, warranty_months),
        or a dict with an "error" key if the order id does not exist.
    """
    with open(ORDERS_FILE, "r") as f:
        orders = json.load(f)

    order = orders.get(order_id)
    if order is None:
        return {"error": f"Order {order_id} was not found."}

    return {
        "order_id": order_id,
        "item": order["item"],
        "price": order["price"],
        "purchased": order["purchased"],
        "warranty_months": order["warranty_months"],
    }


def calculate(expression: str) -> dict:
    """Evaluates a simple arithmetic expression made of numbers and
    + - * / ( ) and returns the numeric result.

    Args:
        expression: A simple math expression, e.g. "1200 * 2".

    Returns:
        A dict with the "result" key, or an "error" key if the expression
        is invalid.
    """
    allowed_chars = set("0123456789+-*/(). ")
    if not expression or not all(c in allowed_chars for c in expression):
        return {"error": "Invalid characters in expression."}
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

orders_agent = Agent(
    name="orders_assistant",
    model=os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite"),
    description="A helpful assistant that answers questions about customer orders.",
    instruction=(
        "You are a helpful orders assistant.\n"
        "You have two tools available:\n"
        "1. lookup_order(order_id) — use this whenever the user mentions an "
        "order id (e.g. 'A1001'), to get its item, price, purchase date and "
        "warranty length in months.\n"
        "2. calculate(expression) — use this whenever you need to do any "
        "arithmetic, such as multiplying a price, or computing how many "
        "months have passed since a purchase date.\n\n"
        "Always use lookup_order before answering anything about a specific "
        "order — never guess or invent order details.\n"
        "If lookup_order returns an error (order not found), tell the user "
        "clearly and honestly that you could not find that order. Do not "
        "make up an item, price, or warranty status for an order you could "
        "not find.\n"
        "When asked about warranty status, reason step by step using the "
        "purchase date, the warranty length in months, and today's date.\n"
        "Be concise and clear in your final answer."
    ),
    tools=[lookup_order, calculate],
)


# ---------------------------------------------------------------------------
# Runner / Session setup + trace printing
# ---------------------------------------------------------------------------

APP_NAME = "orders_app"
USER_ID = "lab_user"
SESSION_ID = "lab_session"


async def run_query(query: str):
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
    )

    runner = Runner(
        agent=orders_agent,
        app_name=APP_NAME,
        session_service=session_service,
    )

    content = types.Content(role="user", parts=[types.Part(text=query)])

    print(f"\n{'=' * 70}")
    print(f"USER GOAL: {query}")
    print(f"{'=' * 70}\n")

    final_text = None

    async for event in runner.run_async(
        user_id=USER_ID, session_id=SESSION_ID, new_message=content
    ):
        # Print tool calls and tool results as they happen (the trace)
        if event.content and event.content.parts:
            for part in event.content.parts:
                if getattr(part, "function_call", None):
                    fc = part.function_call
                    print(f"[TOOL CALL] {fc.name}({dict(fc.args)})")
                elif getattr(part, "function_response", None):
                    fr = part.function_response
                    print(f"[TOOL RESULT] {fr.name} -> {fr.response}")
                elif getattr(part, "text", None):
                    if event.is_final_response():
                        final_text = part.text
                    else:
                        print(f"[AGENT REASONING] {part.text}")

    print(f"\n{'-' * 70}")
    print("FINAL ANSWER:")
    print(final_text)
    print(f"{'-' * 70}\n")
    return final_text


async def main():
    # Multi-step goal: requires lookup_order, calculate, and warranty reasoning
    await run_query(
        "I'm thinking of buying two more of order A1001. "
        "What would those two cost, and is the original still under warranty?"
    )

    # Optional stretch: an order that does not exist
    await run_query(
        "Can you tell me about order A9999 and whether it's still under warranty?"
    )


if __name__ == "__main__":
    asyncio.run(main())
