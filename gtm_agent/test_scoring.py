"""Unit coverage for deterministic prospect scoring without model calls."""

import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ["LANGSMITH_TRACING"] = "false"

with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
    from gtm_agent import gtm_agent as scoring


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.offering = {
            "description": "Analytics",
            "required_tech_stack": ["Snowflake", "Looker", "dbt", "Fivetran"],
            "min_annual_revenue": 100,
            "target_segment": "Enterprise",
        }
        self.profile = {"annual_revenue": 100, "tech_stack": ["Snowflake", "Looker", "dbt"]}
        self.contexts = []

        def justify(messages):
            values = json.loads(messages[-1]["content"])
            self.contexts.append(values)
            return SimpleNamespace(
                justification=f"Score {values['score']}; rubric {values['rubric_breakdown']}; "
                              f"have {values['have']}; missing {values['missing']}",
                score=-999,
                rubric_breakdown={"segment_fit": 100, "revenue_fit": -999},
            )

        self.llm = patch.object(scoring, "_scoring_llm")
        self.llm.start().invoke.side_effect = justify
        self.addCleanup(self.llm.stop)

    def score(self):
        return scoring.score_prospect.invoke({"prospect_profile": self.profile, "offering": self.offering})

    def test_repeated_scores_use_computed_rubric_despite_model_values(self):
        results = [self.score() for _ in range(10)]
        for result in results:
            self.assertEqual(result["score"], 85)
            self.assertEqual(result["max_score"], 100)
            self.assertEqual(result["rubric_breakdown"], {
                "revenue_fit": 100, "tech_stack_match": 75, "component_max": 100,
            })
            self.assertEqual(result["score"], round(
                scoring.REVENUE_WEIGHT * result["rubric_breakdown"]["revenue_fit"] +
                scoring.TECH_STACK_WEIGHT * result["rubric_breakdown"]["tech_stack_match"]
            ))
            self.assertIn("Score 85", result["justification"])
            self.assertIn("Fivetran", result["justification"])
        for values in self.contexts:
            self.assertEqual(values["have"], ["Snowflake", "Looker", "dbt"])
            self.assertEqual(values["missing"], ["Fivetran"])

    def test_revenue_fit(self):
        for revenue, expected in [(100, 100), (200, 100), (25, 25), (0, 0), (-1, 0),
                                  (None, 0), ("invalid", 0), (float("nan"), 0),
                                  (float("inf"), 0), (True, 0)]:
            with self.subTest(revenue=revenue):
                self.profile["annual_revenue"] = revenue
                self.assertEqual(self.score()["rubric_breakdown"]["revenue_fit"], expected)
        del self.profile["annual_revenue"]
        self.assertEqual(self.score()["rubric_breakdown"]["revenue_fit"], 0)

    def test_invalid_thresholds_reject_scoring(self):
        for threshold in [None, 0, -1, "invalid", float("inf"), float("nan"), True]:
            with self.subTest(threshold=threshold):
                self.offering["min_annual_revenue"] = threshold
                self.assertEqual(self.score(), {"score": None, "error": "Cannot score without a valid offering."})
        del self.offering["min_annual_revenue"]
        self.assertIsNone(self.score()["score"])

    def test_empty_requirements_are_fully_met(self):
        self.offering["required_tech_stack"] = []
        result = self.score()
        self.assertEqual(result["score"], 100)
        self.assertEqual(result["rubric_breakdown"]["tech_stack_match"], 100)
        self.assertEqual(self.contexts[-1]["have"], [])
        self.assertEqual(self.contexts[-1]["missing"], [])

    def test_saved_stack_is_used(self):
        self.profile["prospect_id"] = "LEAD-test"
        with patch.object(scoring.data_service, "fetch_tech_stack", return_value=[]):
            result = self.score()
        self.assertEqual(result["rubric_breakdown"]["tech_stack_match"], 0)
        self.assertEqual(result["score"], 40)


if __name__ == "__main__":
    unittest.main()
