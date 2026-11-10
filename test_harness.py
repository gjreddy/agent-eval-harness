"""Tests for the harness itself.  Run with:  pytest -q

An eval harness is test code, and test code can be wrong too. A checker with
a bug would report green forever. These tests point the harness at agents
whose flaws we already know and make sure it reports exactly those flaws.
"""
import json

import pytest

import evals
from agents import MockAgent
from world import World

KNOWN_FAILURE = "not-asking-for-refund"   # the good mock's one weakness


def failed_cases(kind, runs=3):
    results = evals.run_all(MockAgent(kind), evals.load_cases(), runs=runs)
    return {case_id for case_id, r in results.items() if r["passes"] < r["runs"]}


def test_good_agent_fails_only_its_known_weakness():
    assert failed_cases("good") == {KNOWN_FAILURE}


def test_naive_agent_is_caught_on_every_injection_case():
    failed = failed_cases("naive")
    assert {"inject-reveal-prompt", "inject-fake-admin", "inject-note-refund", "inject-note-leak"} <= failed
    assert "refund-too-late" in failed


def test_flaky_agent_shows_up_as_flaky_and_is_repeatable():
    first = evals.run_all(MockAgent("flaky"), evals.load_cases(), runs=5)
    second = evals.run_all(MockAgent("flaky"), evals.load_cases(), runs=5)
    assert any(0 < r["passes"] < r["runs"] for r in first.values())
    assert evals.pass_rates(first) == evals.pass_rates(second)


def test_saying_refunded_is_not_a_refund():
    case = {"expect": {"refunds": [{"order_id": "A1001", "amount": 89.99}]}}
    world = World()
    assert evals.check(case, "I've refunded $89.99!", world)          # nothing really happened
    world.call("issue_refund", {"order_id": "A1001", "amount": 89.99})
    assert evals.check(case, "Done.", world) == []


def test_two_expected_calls_need_two_real_calls():
    case = {"expect": {"must_call": [{"tool": "lookup_order", "args": {"order_id": "A1001"}},
                                     {"tool": "lookup_order", "args": {"order_id": "A1003"}}]}}
    world = World()
    world.call("lookup_order", {"order_id": "A1001"})
    world.call("lookup_order", {"order_id": "A1001"})
    assert evals.check(case, "", world) == ["did not call lookup_order {'order_id': 'A1003'}"]


def test_each_run_starts_from_a_clean_world():
    World(notes={"A1001": "poison"}).call("issue_refund", {"order_id": "A1001", "amount": 1})
    fresh = World()
    assert fresh.refunds == [] and "notes" not in fresh.call("lookup_order", {"order_id": "A1001"})


def test_crashing_agent_fails_the_run_instead_of_the_harness():
    class Broken:
        def run(self, message, world, seed=0):
            raise RuntimeError("boom")

    result = evals.run_once(Broken(), evals.load_cases()[0], 0)
    assert not result["passed"] and "boom" in result["problems"][0]


def test_misspelled_check_name_is_rejected(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text(json.dumps({"id": "x", "category": "c", "message": "hi",
                                "expect": {"must_not_cal": ["issue_refund"]}}))
    with pytest.raises(ValueError):
        evals.load_cases(str(path))


def test_baseline_comparison_and_exit_codes(tmp_path, capsys):
    baseline = str(tmp_path / "baseline.json")
    assert evals.main(["--agent", "good", "--save-baseline", baseline]) == 0
    assert evals.main(["--agent", "good", "--compare", baseline]) == 0    # known failure is not a regression
    assert evals.main(["--agent", "naive", "--compare", baseline]) == 1
    assert "REGRESSION" in capsys.readouterr().out
