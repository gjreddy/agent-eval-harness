"""The eval harness. Read it top to bottom; it is five small steps.

  1. load the test cases           load_cases
  2. run the agent on one case     run_once     (the checks are in checks.py)
  3. repeat and add up             run_all, numbers
  4. print the scorecard           print_report
  5. compare with last time        compare_with_baseline

Usage:
  python3 evals.py                              the rule-following pretend agent
  python3 evals.py --flaw hallucinates          the same agent with one flaw switched on
  python3 evals.py --flaw flaky --runs 5
  python3 evals.py --save-baseline baseline.json
  python3 evals.py --flaw wasteful --compare baseline.json
"""
import argparse
import json
import math
import sys
import zlib
from pathlib import Path

import checks
from world import SYSTEM_PROMPT, World

HERE = Path(__file__).parent

# The scorecard: each question is answered by one or more measurements.
# Each measurement is a file of test cases in cases/.
SCORECARD = [
    ("Did it do the right thing?", ["correctness", "tool_use", "task_completion", "context", "relevance"]),
    ("Did it tell the truth?",     ["groundedness", "hallucination"]),
    ("Did it stay safe?",          ["injection", "policy", "refusal", "tone_safety"]),
    ("Did it hold up?",            ["robustness", "error_recovery"]),
    ("What did it cost?",          ["efficiency"]),
]

# Example prices in dollars per million tokens. They are not any real model's
# prices: put your own model's prices here.
PRICE_IN, PRICE_OUT = 3.00, 15.00
ALLOWED_INCREASE = 0.10    # cost, latency or tool calls may grow 10% before it counts as a regression


# ---------------------------------------------------------------- 1. load
def load_cases(folder=HERE / "cases"):
    """Each file in cases/ is one measurement; each line is one test case."""
    cases = []
    for path in sorted(Path(folder).glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            case = json.loads(line)
            case["category"] = path.stem
            # A misspelled check name would be silently ignored and the case
            # would always pass, so refuse to load it.
            unknown = set(case["expect"]) - checks.ALLOWED_CHECKS
            if unknown or not case["expect"]:
                raise ValueError(f"case {case['id']}: bad or missing checks {sorted(unknown)}")
            cases.append(case)
    ids = [case["id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("two cases have the same id")
    order = [category for _, categories in SCORECARD for category in categories]
    unknown = {case["category"] for case in cases} - set(order)
    if unknown:
        raise ValueError(f"cases/ has files that are not on the scorecard: {sorted(unknown)}")
    return sorted(cases, key=lambda case: order.index(case["category"]))   # scorecard order


# ----------------------------------------------------------------- 2. run
def run_once(agent, case, run_number, judge=None):
    """One conversation in a fresh world, then every check."""
    world = World(notes=case.get("notes"), failures=case.get("failures"))
    seed = zlib.crc32(f"{case['id']}:{run_number}".encode())   # makes the flaky agent repeatable
    history, answers, made_up = [], [], []
    try:
        for message in case["messages"]:                 # most cases have one message
            history.append({"role": "customer", "text": message})
            answers.append(agent.reply(history, world, seed=seed))
            history.append({"role": "agent", "text": answers[-1]})

        sources = [SYSTEM_PROMPT] + case["messages"] + world.read   # all it was told or read
        made_up = checks.made_up_facts(answers, sources)
        problems = checks.expected(case, answers, world) + checks.safety(answers) + made_up
        if judge:                                        # optional: a model grades it too
            problems += judge(history, sources)
    except Exception as crash:                           # a crashing agent fails the run
        problems = [f"agent crashed: {crash}"]

    return {"passed": not problems, "problems": problems, "answers": answers,
            "tools": [call["tool"] for call in world.calls],
            "made_up": bool(made_up), "seconds": world.seconds,
            "cost": (world.tokens_in * PRICE_IN + world.tokens_out * PRICE_OUT) / 1_000_000}


# ------------------------------------------------------- 3. repeat, add up
def run_all(agent, cases, runs=3, judge=None):
    results = {}
    for case in cases:
        attempts = [run_once(agent, case, n, judge) for n in range(runs)]
        results[case["id"]] = {"category": case["category"], "runs": runs, "attempts": attempts,
                               "passes": sum(a["passed"] for a in attempts)}
    return results


def numbers(results):
    """The measurements that are numbers rather than pass/fail."""
    attempts = [a for r in results.values() for a in r["attempts"]]
    seconds = sorted(a["seconds"] for a in attempts)
    return {
        "hallucination_rate": sum(a["made_up"] for a in attempts) / len(attempts),
        "tool_calls": sum(len(a["tools"]) for a in attempts) / len(attempts),
        "seconds": sum(seconds) / len(seconds),
        "seconds_p95": seconds[math.ceil(0.95 * len(seconds)) - 1],   # 95% of runs were this fast or faster
        "cost": sum(a["cost"] for a in attempts) / len(attempts),
        "total_cost": sum(a["cost"] for a in attempts),
    }


# ---------------------------------------------------------------- 4. report
def print_report(agent_name, results):
    runs = next(iter(results.values()))["runs"]
    print(f"Agent: {agent_name}   {len(results)} cases x {runs} runs each\n")

    for case_id, r in results.items():
        label = "PASS " if r["passes"] == runs else "FLAKY" if r["passes"] else "FAIL "
        print(f"  {label}  {r['category']:<16} {case_id:<30} {r['passes']}/{runs}")
        if r["passes"] < runs:
            bad = next(a for a in r["attempts"] if not a["passed"])
            for problem in bad["problems"]:
                print(f"           - {problem}")
            print(f"           tools called: {', '.join(bad['tools']) or 'none'}")
            print(f"           answer: {(bad['answers'] or [''])[-1][:100]!r}")

    # Some measurements are numbers, not pass/fail. Show each under its question.
    n = numbers(results)
    flaky = [case_id for case_id, r in results.items() if 0 < r["passes"] < runs]
    always = sum(r["passes"] == runs for r in results.values())
    extra = {
        "Did it tell the truth?": [
            f"hallucination rate {n['hallucination_rate']:>4.0%}   of runs stated a fact found in nothing the agent read"],
        "Did it hold up?": [
            f"consistency        {always}/{len(results)} cases passed on every run; flaky: {', '.join(flaky) or 'none'}"],
        "What did it cost?": [
            f"tool calls         {n['tool_calls']:.2f} per run",
            f"latency            {n['seconds']:.2f}s average, {n['seconds_p95']:.2f}s at the 95th percentile",
            f"cost               ${n['cost']:.5f} per run, ${n['total_cost']:.4f} for everything"],
    }

    print("\nSCORECARD (share of runs that passed)")
    for question, categories in SCORECARD:
        print(f"\n  {question}")
        for category in categories:
            group = [r for r in results.values() if r["category"] == category]
            if group:
                passed, total = sum(r["passes"] for r in group), sum(r["runs"] for r in group)
                print(f"    {category:<18} {passed / total:>4.0%}   ({passed}/{total})")
        for line in extra.get(question, []):
            print(f"    {line}")


# ------------------------------------------------- 5. compare with baseline
def snapshot(results):
    """What we save as a baseline: each case's pass rate, plus the numbers."""
    n = numbers(results)
    return {"cases": {case_id: r["passes"] / r["runs"] for case_id, r in results.items()},
            "numbers": {key: round(n[key], 6) for key in ("tool_calls", "seconds", "cost")}}


def compare_with_baseline(results, baseline):
    """Print what changed since the baseline. Returns True if nothing got worse."""
    now = snapshot(results)
    then_cases, now_cases = baseline["cases"], now["cases"]
    worse = [c for c in now_cases if c in then_cases and now_cases[c] < then_cases[c]]
    better = [c for c in now_cases if c in then_cases and now_cases[c] > then_cases[c]]
    print("\nCompared with baseline")
    for case_id in worse:
        print(f"  WORSE   {case_id:<30} {then_cases[case_id]:.0%} -> {now_cases[case_id]:.0%}")
    for case_id in better:
        print(f"  BETTER  {case_id:<30} {then_cases[case_id]:.0%} -> {now_cases[case_id]:.0%}")
    for case_id in sorted(set(now_cases) - set(then_cases)):
        print(f"  NEW     {case_id}")
    for case_id in sorted(set(then_cases) - set(now_cases)):
        print(f"  REMOVED {case_id}   (was a failing test deleted?)")

    # The numbers can regress too: same answers, but slower or more expensive.
    for key, label in (("tool_calls", "tool calls per run"), ("seconds", "seconds per run"), ("cost", "cost per run")):
        before, after = baseline["numbers"][key], now["numbers"][key]
        if after > before * (1 + ALLOWED_INCREASE):
            worse.append(key)
            print(f"  WORSE   {label:<30} {before:g} -> {after:g}")

    print(f"  Result: {'REGRESSION, ' + str(len(worse)) + ' thing(s) got worse' if worse else 'OK, nothing got worse'}")
    return not worse


def main(argv=None):
    from agents import FLAWS, MockAgent
    parser = argparse.ArgumentParser(description="Evaluate a tool-using support agent.")
    parser.add_argument("--agent", default="mock", choices=["mock", "claude"])
    parser.add_argument("--flaw", action="append", default=[], choices=sorted(FLAWS),
                        help="switch on a flaw in the pretend agent (can be repeated)")
    parser.add_argument("--runs", type=int, default=3, help="times to run each case (default 3)")
    parser.add_argument("--judge", action="store_true", help="also grade answers with a model (needs an API key)")
    parser.add_argument("--save-baseline", metavar="FILE", help="save these results as the new baseline")
    parser.add_argument("--compare", metavar="FILE", help="compare with a baseline; exit 1 if anything got worse")
    args = parser.parse_args(argv)

    if args.agent == "claude":
        from claude_agent import ClaudeAgent     # imported here so the pretend agent needs no install
        agent, name = ClaudeAgent(), "claude"
    else:
        agent, name = MockAgent(args.flaw), f"mock, flaws: {', '.join(args.flaw) or 'none'}"
    judge = None
    if args.judge:
        from judge import ModelJudge
        judge = ModelJudge()

    results = run_all(agent, load_cases(), runs=args.runs, judge=judge)
    print_report(name, results)

    if args.save_baseline:
        with open(args.save_baseline, "w", encoding="utf-8") as f:
            json.dump(snapshot(results), f, indent=2)
            f.write("\n")
        print(f"\nBaseline saved to {args.save_baseline}")
    if args.compare:
        with open(args.compare, encoding="utf-8") as f:
            if not compare_with_baseline(results, json.load(f)):
                return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
