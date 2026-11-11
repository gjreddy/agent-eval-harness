"""Optional: use a model as the judge.

Some qualities cannot be checked with simple code. "Is this answer polite?"
"Does it answer the question that was asked?" "Is every claim backed by what
the agent read?" The usual approach is to ask a second model to grade the
answer. That is called a model judge (or "LLM as judge").

Setup:   pip install anthropic   and set ANTHROPIC_API_KEY
Run:     python3 evals.py --judge

This costs money: one extra model request per run. It has not been run
against the live API yet. A judge is itself a model and can be wrong, so
check its verdicts by hand on a sample before you trust its scores.
"""
import json
import os

MODEL = os.environ.get("JUDGE_MODEL", "claude-opus-5-5")

QUESTIONS = {
    "grounded": "Every factual claim in the agent's answers is supported by the sources.",
    "relevant": "The agent's answers address what the customer actually asked.",
    "polite": "The agent's answers are polite and professional.",
}

INSTRUCTIONS = """You are grading a customer support agent. Below are the sources the agent had
(its instructions and the tool results it read) and the conversation.

For each statement, decide if it is true or false:
{statements}

Reply with only a JSON object like {{"grounded": true, "relevant": true, "polite": true, "why": "one short sentence"}}.

<sources>
{sources}
</sources>

<conversation>
{conversation}
</conversation>"""


class ModelJudge:
    def __init__(self):
        import anthropic     # imported here so the rest works without it installed
        self.client = anthropic.Anthropic()

    def __call__(self, history, sources):
        """Returns a list of problems, like the checks in checks.py."""
        prompt = INSTRUCTIONS.format(
            statements="\n".join(f"- {name}: {text}" for name, text in QUESTIONS.items()),
            sources="\n".join(sources),
            conversation="\n".join(f"{turn['role']}: {turn['text']}" for turn in history))
        response = self.client.messages.create(
            model=MODEL, max_tokens=1024, messages=[{"role": "user", "content": prompt}])
        text = "".join(block.text for block in response.content if block.type == "text")
        return problems_from(text)


def problems_from(reply):
    """Turn the judge's reply into a list of problems."""
    try:
        verdict = json.loads(reply[reply.index("{"):reply.rindex("}") + 1])
    except ValueError:
        return [f"judge gave an unreadable reply: {reply[:80]!r}"]
    return [f"judge: not {name} ({verdict.get('why', 'no reason given')})"
            for name in QUESTIONS if verdict.get(name) is not True]
