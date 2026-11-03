import copy
import json
import os
import unittest
from unittest.mock import Mock, patch

os.environ["LANGSMITH_TRACING"] = "false"
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from gtm_agent import data_service
from gtm_agent import gtm_agent as agent


class ProspectPrivacyTests(unittest.TestCase):
    def setUp(self):
        self.prospect_id = next(iter(data_service.PROSPECTS))
        self.record = copy.deepcopy(data_service.PROSPECTS[self.prospect_id])
        self.record["billing_qualification"] = {"tax_id": "private-tax-id"}
        self.record["extra_field"] = "excluded"
        records = patch.dict(data_service.PROSPECTS, {self.prospect_id: self.record})
        records.start()
        self.addCleanup(records.stop)
        profiles = patch.dict(data_service._PROFILES, {}, clear=True)
        profiles.start()
        self.addCleanup(profiles.stop)

    def test_contact_contains_only_allowed_fields(self):
        result = agent.get_prospect.invoke({"prospect_id": self.prospect_id})
        self.assertTrue(result["found"])
        self.assertEqual(result["prospect"], {
            "prospect_id": self.prospect_id,
            **{key: self.record[key] for key in ("name", "email", "disqualified")},
        })

    def test_fresh_profile_contains_only_scoring_fields(self):
        result = agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})
        expected = {
            "prospect_id": self.prospect_id,
            **{key: self.record[key] for key in (
                "name", "annual_revenue", "tech_stack", "account_details", "engagement_history"
            )},
        }
        self.assertEqual(result, {"prospect_profile": expected, "found": True})
        self.assertEqual(data_service.get_profile_from_db(self.prospect_id)["prospect_profile"], expected)

    def test_cached_profile_is_sanitized_before_return(self):
        cached = {"prospect_id": self.prospect_id, "name": self.record["name"]}
        data_service._PROFILES[self.prospect_id] = {
            **cached, "billing_qualification": self.record["billing_qualification"]
        }
        self.assertEqual(data_service.get_profile_from_db(self.prospect_id), {
            "prospect_profile": cached
        })
        self.assertEqual(agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id}), {
            "prospect_profile": cached, "found": True
        })
        self.assertEqual(data_service._PROFILES[self.prospect_id], cached)

    def test_scoring_excludes_billing_for_direct_callers(self):
        for profile in ({"name": "Prospect"}, {"prospect_id": self.prospect_id}):
            with self.subTest(profile=profile):
                supplied = {**profile, "billing_qualification": self.record["billing_qualification"]}
                self.assert_scoring_is_private(supplied)
                self.assertIn("billing_qualification", supplied)

    def assert_scoring_is_private(self, profile):
        captured = []

        def score(messages):
            captured.extend(messages)
            return Mock(model_dump=lambda: {"score": 75})

        with patch.object(agent, "_scoring_llm", Mock(invoke=score)):
            result = agent.score_prospect.invoke({
                "prospect_profile": profile,
                "offering": next(iter(data_service.OFFERINGS.values())),
            })
        self.assertEqual(result, {"score": 75})
        serialized = json.dumps(captured)
        self.assertNotIn("billing_qualification", serialized)
        self.assertNotIn("private-tax-id", serialized)

    def test_lookup_build_score_flow_for_all_prospects(self):
        for prospect_id in data_service.PROSPECTS:
            with self.subTest(prospect_id=prospect_id):
                contact = agent.get_prospect.invoke({"prospect_id": prospect_id})
                profile = agent.build_prospect_profile.invoke({"prospect_id": prospect_id})
                cached = agent.build_prospect_profile.invoke({"prospect_id": prospect_id})
                self.assertNotIn("billing_qualification", json.dumps([contact, profile, cached]))
                self.assert_scoring_is_private(profile["prospect_profile"])


if __name__ == "__main__":
    unittest.main()
