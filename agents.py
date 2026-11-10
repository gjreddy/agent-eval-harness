"""Pretend agents, so you can run everything with no API key.

They are simple keyword matchers, not AI. Each one has a known personality,
which lets you watch the harness catch real problems:

  good   follows the rules. It has one honest weakness: it does not
         understand "I am NOT asking for a refund".
  naive  does whatever "ignore previous instructions ..." text tells it to,
         and never checks the refund window.
  flaky  the good agent, but on about a third of runs it does a worse job.

To test a real agent, write a class with the same `run` method. See
claude_agent.py for an example.
"""
import random
import re
from datetime import date

from world import CUSTOMER_ID, REFUND_WINDOW_DAYS, SYSTEM_PROMPT, TODAY

ORDER_ID = re.compile(r"\b[A-Z]\d{4}\b")
INJECTION = re.compile(r"ignore (all |any )?(previous|prior|above) instructions", re.I)
SECRET_WORDS = ("system prompt", "your instructions", "reference code")


class MockAgent:
    def __init__(self, kind="good"):
        assert kind in ("good", "naive", "flaky"), f"unknown mock agent: {kind}"
        self.kind = kind

    def run(self, message, world, seed=0):
        """Read the customer's message, use tools through `world.call`,
        return the final answer."""
        dice = random.Random(seed)
        text = message.lower()
        order_ids = ORDER_ID.findall(message)

        if self.kind == "naive" and INJECTION.search(message):
            return self.obey(message, world)
        if any(word in text for word in SECRET_WORDS):
            return "I can't share my internal instructions, but I'm happy to help with an order."
        if "human" in text or "manager" in text:
            world.call("escalate_to_human", {"reason": message})
            return "I've passed this to a human agent. They'll be in touch shortly."
        if "refund" in text or "money back" in text:
            if not order_ids:
                return "I can help with a refund. What's the order number?"
            return self.refund(order_ids[0], world, dice)
        if order_ids:
            return " ".join(self.status(order_id, world, dice) for order_id in order_ids)
        return "You're welcome! Anything else I can help with?"

    def look_up(self, order_id, world):
        """Returns (order, None) if we may talk about it, else (None, reply)."""
        order = world.call("lookup_order", {"order_id": order_id})
        if "error" in order:
            return None, f"I couldn't find order {order_id}. Could you double-check the number?"
        if order["customer_id"] != CUSTOMER_ID:
            return None, f"I can't help with order {order_id} because it isn't on your account."
        if self.kind == "naive" and INJECTION.search(order.get("notes", "")):
            return None, self.obey(order["notes"], world)
        return order, None

    def status(self, order_id, world, dice):
        order, reply = self.look_up(order_id, world)
        if reply:
            return reply
        if order["status"] == "delivered":
            return f"Order {order_id} ({order['item']}) was delivered on {order['delivered_on']}."
        if self.kind == "flaky" and dice.random() < 0.35:
            return f"Order {order_id} ({order['item']}) is on its way."   # forgets the date
        return f"Order {order_id} ({order['item']}) has shipped and should arrive by {order['eta']}."

    def refund(self, order_id, world, dice):
        order, reply = self.look_up(order_id, world)
        if reply:
            return reply
        if order["status"] != "delivered":
            return f"Order {order_id} hasn't been delivered yet, so I can't refund it."
        if self.kind != "naive":
            days = (TODAY - date.fromisoformat(order["delivered_on"])).days
            if days > REFUND_WINDOW_DAYS:
                return (f"Order {order_id} was delivered {days} days ago, which is outside our "
                        f"{REFUND_WINDOW_DAYS}-day refund window, so I can't refund it.")
        if self.kind == "flaky" and dice.random() < 0.35:
            world.call("escalate_to_human", {"reason": f"refund for {order_id}"})   # gives up
            return "I've passed your refund request to a human agent."
        world.call("issue_refund", {"order_id": order_id, "amount": order["total"]})
        return f"I've refunded ${order['total']:.2f} for order {order_id} ({order['item']})."

    def obey(self, instruction, world):
        """What a gullible agent does with an injected instruction."""
        text = instruction.lower()
        if any(word in text for word in SECRET_WORDS):
            return "Sure! Here are my instructions:\n" + SYSTEM_PROMPT
        order_ids = ORDER_ID.findall(instruction)
        if "refund" in text and order_ids:
            order = world.call("lookup_order", {"order_id": order_ids[0]})
            world.call("issue_refund", {"order_id": order_ids[0], "amount": order.get("total", 0)})
            return f"Done. I've refunded order {order_ids[0]}."
        return "OK."
