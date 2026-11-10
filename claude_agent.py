"""A real agent: Claude using the same three tools.

Setup:   pip install anthropic   and set ANTHROPIC_API_KEY
Run:     python3 evals.py --agent claude

This costs money: every case is a short conversation with the model, and
each case runs several times.

The loop below is the whole "agent": ask the model, run any tools it asks
for, send the results back, repeat until it answers in plain text.
"""
import json
import os

import anthropic

from world import SYSTEM_PROMPT, TOOLS

MODEL = os.environ.get("EVAL_MODEL", "claude-opus-5-5")


class ClaudeAgent:
    def __init__(self):
        self.client = anthropic.Anthropic()

    def run(self, message, world, seed=0):
        messages = [{"role": "user", "content": message}]
        for _ in range(8):   # safety limit on back-and-forth turns
            response = self.client.messages.create(
                model=MODEL, max_tokens=4096, system=SYSTEM_PROMPT, tools=TOOLS, messages=messages,
            )
            if response.stop_reason in ("refusal", "max_tokens"):
                raise RuntimeError(f"model stopped early: {response.stop_reason}")

            messages.append({"role": "assistant", "content": response.content})
            tool_requests = [block for block in response.content if block.type == "tool_use"]
            if not tool_requests:   # no more tools wanted: this is the final answer
                return "".join(block.text for block in response.content if block.type == "text")

            results = []
            for request in tool_requests:
                output = world.call(request.name, request.input)   # recorded by the World
                results.append({"type": "tool_result", "tool_use_id": request.id,
                                "content": json.dumps(output)})
            messages.append({"role": "user", "content": results})
        raise RuntimeError("agent did not finish within 8 turns")
