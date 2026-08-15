"""Tests for Demo 0 claim-intake routing fixtures."""

import json
from pathlib import Path


SCENARIOS_PATH = Path(__file__).parent.parent / "docs" / "scenarios.json"


def _load_scenarios():
    with SCENARIOS_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def _scenario(pack, scenario_id):
    for sc in pack["scenarios"]:
        if sc["id"] == scenario_id:
            return sc
    raise AssertionError(f"missing scenario {scenario_id}")


def test_claim_intake_066_routes_to_human_handoff():
    pack = _load_scenarios()
    sc = _scenario(pack, "claim_intake_066")
    assert sc["expected_output"]["routing_decision"] == "human_handoff"


def test_claim_intake_058_routes_to_ai_handle():
    pack = _load_scenarios()
    sc = _scenario(pack, "claim_intake_058")
    assert sc["expected_output"]["routing_decision"] == "ai_handle"
