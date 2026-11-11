"""A real agent: Claude using the same four tools.

Setup:   pip install anthropic   and set ANTHROPIC_API_KEY
Run:     python3 evals.py --agent claude

This costs money: every case is a short conversation with the model, and
each case runs several times. It has not been run against the live API yet,
so expect to adjust a case or two.

The loop below is the whole "agent harness": ask the model, run any tools it
asks for, send the results back, repeat until it answers in plain text.
"""
import json
import os
import time

import anthropic

from world import SYSTEM_PROMPT, TOOLS

MODEL = os.environ.get("EVAL_MODEL", "claude-opus-5-5")


class ClaudeAgent:
    def __init__(self):
        self.client = anthropic.Anthropic()

    def reply(self, history, world, seed=0):
        # Earlier turns are passed as plain text. (The tool calls made in
        # earlier turns are not replayed, to keep this short.)
        messages = [{"role": "user" if turn["role"] == "customer" else "assistant", "content": turn["text"]}
                    for turn in history]
        for _ in range(8):   # safety limit on back-and-forth turns
            started = time.time()
            response = self.client.messages.create(
                model=MODEL, max_tokens=4096, system=SYSTEM_PROMPT, tools=TOOLS, messages=messages,
            )
            # Report real usage, so cost and latency are measured, not guessed.
            world.model_turn(response.usage.input_tokens, response.usage.output_tokens,
                             time.time() - started)
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
