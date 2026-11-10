"""The eval harness. Read it top to bottom; it is five small steps.

  1. load the test cases           load_cases
  2. run the agent on one case     run_once
  3. check what it did             check
  4. repeat and add up             run_all, print_report
  5. compare with last time        compare_with_baseline

Usage:
  python3 evals.py --agent good
  python3 evals.py --agent naive
  python3 evals.py --agent flaky --runs 5
  python3 evals.py --agent good --save-baseline baseline.json
  python3 evals.py --agent good --compare baseline.json
"""
import argparse
import json
import sys
import zlib

from world import World

ALLOWED_CHECKS = {"must_call", "must_not_call", "answer_has", "answer_lacks", "refunds"}


# ---------------------------------------------------------------- 1. load
def load_cases(path="cases.jsonl"):
    cases = []
    for line in open(path, encoding="utf-8"):
        if line.strip():
            case = json.loads(line)
            # A misspelled check name would be silently ignored and the case
            # would always pass, so refuse to load it.
            unknown = set(case["expect"]) - ALLOWED_CHECKS
            if unknown or not case["expect"]:
                raise ValueError(f"case {case['id']}: bad or missing checks {sorted(unknown)}")
            cases.append(case)
    return cases


# --------------------------------------------------------------- 3. check
def same(expected, actual):
    """Compare loosely: numbers within a cent, text ignoring case."""
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return abs(expected - actual) < 0.01
    return str(expected).strip().lower() == str(actual).strip().lower()


def args_match(expected_args, actual_args):
    """Every expected argument must be there and equal. Extra ones are fine."""
    return all(key in actual_args and same(value, actual_args[key])
               for key, value in expected_args.items())


def check(case, answer, world):
    """Return a list of problems. An empty list means the run passed."""
    expect = case["expect"]
    problems = []
    answer_text = answer.lower()

    # Did it make the tool calls it had to? Each expected call needs its own
    # real call, so looking up the same order twice does not count as two.
    unused = list(world.calls)
    for wanted in expect.get("must_call", []):
        match = next((call for call in unused if call["tool"] == wanted["tool"]
                      and args_match(wanted.get("args", {}), call["args"])), None)
        if match:
            unused.remove(match)
        else:
            problems.append(f"did not call {wanted['tool']} {wanted.get('args', '')}".strip())

    # Did it avoid the tool calls it must never make here?
    called = [call["tool"] for call in world.calls]
    for tool in expect.get("must_not_call", []):
        if tool in called:
            problems.append(f"called {tool}, which is not allowed in this case")

    # Does the answer contain what it should, and nothing it should not?
    for phrase in expect.get("answer_has", []):
        if phrase.lower() not in answer_text:
            problems.append(f"answer is missing '{phrase}'")
    for phrase in expect.get("answer_lacks", []):
        if phrase.lower() in answer_text:
            problems.append(f"answer contains '{phrase}'")

    # Is the world in the right state afterwards? This looks at the refunds
    # that actually exist, not at whether the agent claims it refunded.
    if "refunds" in expect:
        wanted, actual = expect["refunds"], world.refunds
        if len(wanted) != len(actual) or not all(
                any(args_match(w, a) for a in actual) for w in wanted):
            problems.append(f"expected refunds {wanted}, found {actual}")

    return problems


# ----------------------------------------------------------------- 2. run
def run_once(agent, case, run_number):
    world = World(notes=case.get("notes"))               # fresh world every time
    seed = zlib.crc32(f"{case['id']}:{run_number}".encode())   # makes the flaky mock repeatable
    try:
        answer = agent.run(case["message"], world, seed=seed)
        problems = check(case, answer, world)
    except Exception as crash:                           # a crashing agent fails the run
        answer, problems = "", [f"agent crashed: {crash}"]
    return {"passed": not problems, "problems": problems, "answer": answer,
            "tools": [call["tool"] for call in world.calls]}


# ------------------------------------------------------- 4. repeat, report
def run_all(agent, cases, runs=3):
    results = {}
    for case in cases:
        attempts = [run_once(agent, case, n) for n in range(runs)]
        results[case["id"]] = {
            "category": case["category"],
            "passes": sum(a["passed"] for a in attempts),
            "runs": runs,
            "attempts": attempts,
        }
    return results


def print_report(agent_name, results):
    runs = next(iter(results.values()))["runs"]
    print(f"Agent: {agent_name}   {len(results)} cases x {runs} runs each\n")

    for case_id, r in results.items():
        label = "PASS " if r["passes"] == runs else "FLAKY" if r["passes"] else "FAIL "
        print(f"  {label}  {case_id:<28} {r['passes']}/{runs}")
        if r["passes"] < runs:
            bad = next(a for a in r["attempts"] if not a["passed"])
            for problem in bad["problems"]:
                print(f"           - {problem}")
            print(f"           tools called: {', '.join(bad['tools']) or 'none'}")
            print(f"           answer: {bad['answer'][:90]!r}")

    print("\nScore by category (share of runs that passed)")
    for category in dict.fromkeys(r["category"] for r in results.values()):
        group = [r for r in results.values() if r["category"] == category]
        passed, total = sum(r["passes"] for r in group), sum(r["runs"] for r in group)
        print(f"  {category:<12} {passed / total:>4.0%}   ({passed}/{total})")

    flaky = [cid for cid, r in results.items() if 0 < r["passes"] < runs]
    always = sum(r["passes"] == runs for r in results.values())
    print("\nConsistency")
    print(f"  passed on every run:  {always}/{len(results)} cases")
    print(f"  flaky (mixed results): {', '.join(flaky) or 'none'}")


# ------------------------------------------------- 5. compare with baseline
def pass_rates(results):
    return {case_id: r["passes"] / r["runs"] for case_id, r in results.items()}


def compare_with_baseline(results, baseline):
    """Print what changed since the baseline. Returns True if nothing got worse."""
    now = pass_rates(results)
    worse = [c for c in now if c in baseline and now[c] < baseline[c]]
    better = [c for c in now if c in baseline and now[c] > baseline[c]]
    print("\nCompared with baseline")
    for case_id in worse:
        print(f"  WORSE   {case_id:<28} {baseline[case_id]:.0%} -> {now[case_id]:.0%}")
    for case_id in better:
        print(f"  BETTER  {case_id:<28} {baseline[case_id]:.0%} -> {now[case_id]:.0%}")
    for case_id in sorted(set(now) - set(baseline)):
        print(f"  NEW     {case_id}")
    for case_id in sorted(set(baseline) - set(now)):
        print(f"  REMOVED {case_id}   (was a failing test deleted?)")
    print(f"  Result: {'REGRESSION, ' + str(len(worse)) + ' case(s) got worse' if worse else 'OK, nothing got worse'}")
    return not worse


def make_agent(name):
    if name == "claude":
        from claude_agent import ClaudeAgent   # imported here so the mocks need no install
        return ClaudeAgent()
    from agents import MockAgent
    return MockAgent(name)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate a tool-using support agent.")
    parser.add_argument("--agent", default="good", choices=["good", "naive", "flaky", "claude"])
    parser.add_argument("--runs", type=int, default=3, help="times to run each case (default 3)")
    parser.add_argument("--save-baseline", metavar="FILE", help="save these results as the new baseline")
    parser.add_argument("--compare", metavar="FILE", help="compare with a baseline; exit 1 if anything got worse")
    args = parser.parse_args(argv)

    results = run_all(make_agent(args.agent), load_cases(), runs=args.runs)
    print_report(args.agent, results)

    if args.save_baseline:
        with open(args.save_baseline, "w", encoding="utf-8") as f:
            json.dump(pass_rates(results), f, indent=2)
            f.write("\n")
        print(f"\nBaseline saved to {args.save_baseline}")
    if args.compare:
        with open(args.compare, encoding="utf-8") as f:
            if not compare_with_baseline(results, json.load(f)):
                return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
