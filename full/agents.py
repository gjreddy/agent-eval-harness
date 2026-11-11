"""A pretend agent, so you can run everything with no API key.

It is a keyword matcher, not AI. With no flaws switched on it follows the
store's rules (it has one honest weakness: it does not understand "I am NOT
asking for a refund"). Each flaw below switches on one kind of bad behaviour,
so you can watch which measurement catches it:

    python3 evals.py --flaw hallucinates

You do not need to read this file to understand evals. To test a real agent,
write a class with the same `reply` method. See claude_agent.py.
"""
import random
import re
from datetime import date
from difflib import SequenceMatcher

from world import CUSTOMER_ID, REFUND_WINDOW_DAYS, SYSTEM_PROMPT, TODAY, UNAVAILABLE

FLAWS = {
    "careless":        "leaves out details and hands refunds to a human instead of doing them",
    "half_done":       "does the first thing the customer asks and ignores the rest",
    "forgetful":       "only reads the latest message, not the conversation so far",
    "off_topic":       "answers a different question from the one asked",
    "from_memory":     "answers policy questions without searching, and gets them wrong",
    "hallucinates":    "makes something up when it cannot find the answer",
    "obeys_injection": "does whatever 'ignore previous instructions' text tells it to",
    "breaks_rules":    "refunds anything: too late, not delivered, not yours, extra money",
    "over_refuses":    "refuses harmless requests that contain a scary-sounding word",
    "under_refuses":   "happily answers things a store's support agent should not",
    "rude":            "snaps at upset customers",
    "leaky":           "shares other customers' details",
    "brittle":         "only understands neat input: no typos, no lower-case order numbers",
    "gives_up":        "stops at the first tool error instead of retrying or escalating",
    "wasteful":        "looks every order up three times",
    "flaky":           "careless on about one run in three",
}

ORDER_ID = re.compile(r"\b[A-Za-z]\d{4}\b")
INJECTION = re.compile(r"ignore (all |any )?(previous|prior|above) instructions", re.I)
SECRET_WORDS = ("system prompt", "your instructions", "reference code")
OUT_OF_SCOPE = ("medical", "stocks", "homework")
SCARY_WORDS = ("kill", "dying", "shoot")
ANGRY_WORDS = ("ridiculous", "garbage", "useless", "terrible")
HUMAN_WORDS = ("human", "manager", "real person")
POLICY_WORDS = ("policy", "warranty", "shipping", "do you", "is there", "how do i", "how long")
DETAIL_WORDS = ("where", "when", "arrive", "delivered", "status", "item", "cost", "pay", "tracking")
WRONG_MEMORY = {"refund": "Refunds are available within 60 days of delivery.",
                "shipping": "Shipping always takes 2 days.",
                "warranty": "Everything comes with a 5 year warranty.",
                "return": "Returns cost a flat fee of 10 dollars."}

# Pretend usage, so cost and latency have something to measure.
PROMPT_TOKENS = 400       # system prompt + tool descriptions, sent on every model turn
MODEL_SECONDS = 0.8       # pretend thinking time per model turn


class MockAgent:
    def __init__(self, flaws=()):
        unknown = set(flaws) - set(FLAWS)
        assert not unknown, f"unknown flaws: {sorted(unknown)}"
        self.flaws = set(flaws)

    def has(self, flaw):
        return flaw in self.active

    def reply(self, history, world, seed=0):
        """`history` is the conversation so far, ending with the customer's
        latest message. Use tools through `world.call`, return the answer."""
        self.active = set(self.flaws)
        if "flaky" in self.active and random.Random(seed).random() < 0.35:
            self.active.add("careless")

        calls_before = len(world.calls)
        answer = self.think(history, world)

        # A real agent asks the model once per tool call, plus once to write
        # the answer. Report pretend usage for that many turns.
        turns = len(world.calls) - calls_before + 1
        seen_so_far = sum(len(turn["text"]) for turn in history) // 4   # about 4 characters per token
        world.model_turn(tokens_in=turns * (PROMPT_TOKENS + seen_so_far),
                         tokens_out=20 * (turns - 1) + len(answer) // 4,
                         seconds=turns * MODEL_SECONDS)
        return answer

    # ------------------------------------------------- understanding the message
    def think(self, history, world):
        said = [turn["text"] for turn in history if turn["role"] == "customer"]
        if self.has("forgetful"):
            said = said[-1:]
        message = said[-1]
        text = message.lower()

        if self.has("obeys_injection") and INJECTION.search(message):
            return self.obey(message, world)
        if any(word in text for word in SECRET_WORDS):
            return "I can't share my internal instructions, but I'm happy to help with an order."
        if self.has("over_refuses") and any(word in text for word in SCARY_WORDS):
            return "I can't help with that request."
        if any(word in text for word in OUT_OF_SCOPE):
            if self.has("under_refuses"):
                return "Sure, happy to help! My advice: go for it, it works 100% of the time."
            return "Sorry, I can only help with orders, refunds and store policies."

        # Which order is this about, if the message does not say? Fall back to
        # the last one mentioned earlier in the conversation.
        earlier = [order_id for old in said[:-1] for order_id in self.order_ids(old)]
        fallback = self.order_ids(message) or earlier[-1:]
        just_asked_for_number = (len(said) > 1 and len(history) > 1
                                 and "order number" in history[-2]["text"])

        # Handle each sentence (or "... and also ...") as its own request.
        replies = []
        for part in re.split(r"(?<=[.?!])\s+|\s+and also\s+", message):
            reply = self.handle(part, fallback, just_asked_for_number, world)
            if reply:
                replies.append(reply)
                if self.has("half_done"):
                    break
        answer = " ".join(replies) or "Happy to help! Is there anything else you need?"

        if any(word in text for word in ANGRY_WORDS):
            answer = ("Calm down. " if self.has("rude") else "I'm sorry about the trouble. ") + answer
        return answer

    def handle(self, part, fallback, just_asked_for_number, world):
        """Answer one request. Returns None if the part asks for nothing."""
        text = part.lower()
        own_ids = self.order_ids(part)
        order_ids = own_ids or fallback

        if any(word in text for word in HUMAN_WORDS):
            world.call("escalate_to_human", {"reason": part})
            return "I've passed this to a human agent. They'll be in touch shortly."
        if not own_ids and any(word in text for word in POLICY_WORDS):
            return self.policy(part, world)
        if self.wants_refund(text) or (own_ids and just_asked_for_number):
            if not order_ids:
                return "I can help with a refund. What's the order number?"
            return self.refund(order_ids[0], part, world)
        if own_ids or any(word in text for word in DETAIL_WORDS):
            if not order_ids:
                return "Which order do you mean? Please send me the order number."
            return " ".join(self.status(order_id, part, world) for order_id in order_ids)
        return None

    def order_ids(self, text):
        if self.has("brittle"):
            return re.findall(r"\b[A-Z]\d{4}\b", text)
        return [order_id.upper() for order_id in ORDER_ID.findall(text)]

    def wants_refund(self, text):
        if "refund" in text:
            return True
        if self.has("brittle"):
            return False
        misspelled = any(SequenceMatcher(None, word, "refund").ratio() > 0.8
                         for word in re.findall(r"[a-z]+", text))
        return misspelled or any(phrase in text for phrase in ("money back", "reimburse", "return my"))

    # ------------------------------------------------------------- using tools
    def use(self, world, tool, args):
        result = world.call(tool, args)
        if result.get("error") == UNAVAILABLE and not self.has("gives_up"):
            result = world.call(tool, args)      # try once more
        if self.has("wasteful") and tool == "lookup_order":
            world.call(tool, args)
            world.call(tool, args)
        return result

    def stuck(self, world, reason):
        """A tool failed twice."""
        if self.has("gives_up"):
            return "Sorry, something went wrong."
        world.call("escalate_to_human", {"reason": reason})
        return "Our system isn't responding right now, so I've passed this to a human agent who will follow up."

    def look_up(self, order_id, world):
        """Returns (order, sorry). `sorry` is the reply to give when we cannot help."""
        order = self.use(world, "lookup_order", {"order_id": order_id})
        if order.get("error") == UNAVAILABLE:
            return None, self.stuck(world, f"looking up {order_id} keeps failing")
        if "error" in order:
            return None, f"I couldn't find order {order_id}. Could you double-check the number?"
        if self.has("obeys_injection") and INJECTION.search(order.get("notes", "")):
            return None, self.obey(order["notes"], world)
        if order["customer_id"] != CUSTOMER_ID:
            return order, f"I can't help with order {order_id} because it isn't on your account."
        return order, None

    def status(self, order_id, part, world):
        order, sorry = self.look_up(order_id, world)
        if sorry:
            if order and self.has("leaky"):
                return f"Order {order_id} is a {order['item']} bought by {order['email']}."
            return sorry
        asked = part.lower()
        if "tracking" in asked:
            if self.has("hallucinates"):
                return f"The tracking number for order {order_id} is 1Z999AA10123456784."
            return f"I don't have a tracking number for order {order_id}."
        if not self.has("off_topic"):
            if any(word in asked for word in ("cost", "pay", "how much")):
                return f"Order {order_id} cost ${order['total']:.2f}."
            if "item" in asked or "what did i order" in asked:
                return f"Order {order_id} is for {order['item']}."
        if order["status"] == "delivered":
            return f"Order {order_id} was delivered on {order['delivered_on']}."
        if order["status"] == "refunded":
            return f"Order {order_id} has been refunded."
        if self.has("careless"):
            return f"Order {order_id} is on its way."      # forgets the date
        return f"Order {order_id} has shipped and should arrive by {order['eta']}."

    def refund(self, order_id, part, world):
        order, sorry = self.look_up(order_id, world)
        ignore_rules = self.has("breaks_rules") and order
        if sorry and not ignore_rules:
            return sorry
        if not ignore_rules:
            if order["status"] == "refunded":
                return f"Order {order_id} has already been refunded."
            if order["status"] != "delivered":
                return f"Order {order_id} hasn't been delivered yet, so I can't refund it."
            if (TODAY - date.fromisoformat(order["delivered_on"])).days > REFUND_WINDOW_DAYS:
                return (f"Order {order_id} was delivered on {order['delivered_on']}, which is outside "
                        f"our {REFUND_WINDOW_DAYS}-day refund window, so I can't refund it.")
        if self.has("careless"):
            world.call("escalate_to_human", {"reason": f"refund for {order_id}"})
            return "I've passed your refund request to a human agent."
        amount = order["total"]
        extra = re.search(r"\$(\d+)", part)
        if ignore_rules and extra:
            amount += float(extra.group(1))
        result = self.use(world, "issue_refund", {"order_id": order_id, "amount": amount})
        if "error" in result:
            return self.stuck(world, f"refund for {order_id} keeps failing")
        return f"I've refunded ${amount:.2f} for order {order_id}."

    def policy(self, part, world):
        text = part.lower()
        if self.has("from_memory"):
            for topic, wrong_answer in WRONG_MEMORY.items():
                if topic in text:
                    return wrong_answer
        found = self.use(world, "search_policy", {"query": part}).get("results", [])
        if found:
            return " ".join(policy["text"] for policy in found)
        if self.has("hallucinates"):
            return "Yes, we do! It applies to every order for 90 days."
        return "I don't have information about that, so I'd rather not guess."

    def obey(self, instruction, world):
        """What a gullible agent does with an injected instruction."""
        text = instruction.lower()
        if any(word in text for word in SECRET_WORDS):
            return "Sure! Here are my instructions:\n" + SYSTEM_PROMPT
        say = re.search(r"tell the customer (.+)", instruction, re.I)
        if say:
            return "Important: you need " + say.group(1)
        order_ids = self.order_ids(instruction)
        if "refund" in text and order_ids:
            order = world.call("lookup_order", {"order_id": order_ids[0]})
            world.call("issue_refund", {"order_id": order_ids[0], "amount": order.get("total", 0)})
            return f"Done. I've refunded order {order_ids[0]}."
        return "OK."
