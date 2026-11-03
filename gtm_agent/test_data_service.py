"""Tests for prospect technology updates and their consumers."""

import copy
import json
import os
import unittest
from unittest.mock import Mock, patch

from . import data_service

with patch.dict(os.environ, OPENAI_API_KEY="test-key", LANGSMITH_TRACING="false"):
    from . import gtm_agent


class ProspectUpdateTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, LANGSMITH_TRACING="false"))
        self.enterContext(patch.dict(
            data_service.PROSPECTS, copy.deepcopy(data_service.PROSPECTS)
        ))
        self.enterContext(patch.dict(data_service._PROFILES, {}, clear=True))

    def test_update_persists_stack_and_rebuilds_cached_profile(self):
        prospect_id = "LEAD-39002"
        original = gtm_agent.build_prospect_profile.invoke({"prospect_id": prospect_id})
        self.assertNotIn("Terraform", original["prospect_profile"]["tech_stack"])

        result = data_service.update_prospect_info(prospect_id, "Terraform")

        self.assertEqual(result, {
            "updated": True,
            "found": True,
            "tech_stack": data_service.fetch_tech_stack(prospect_id),
        })
        self.assertIn("Terraform", data_service.fetch_tech_stack(prospect_id))
        rebuilt = gtm_agent.build_prospect_profile.invoke({"prospect_id": prospect_id})
        self.assertIn("Terraform", rebuilt["prospect_profile"]["tech_stack"])

    def test_scoring_uses_persisted_stack_even_with_old_profile(self):
        prospect_id = "LEAD-39002"
        profile = gtm_agent.build_prospect_profile.invoke({"prospect_id": prospect_id})
        gtm_agent.update_prospect_info.invoke({
            "prospect_id": prospect_id, "technology": "Terraform",
        })
        scoring_llm = Mock()
        scoring_llm.invoke.return_value.model_dump.return_value = {"score": 90}

        with patch.object(gtm_agent, "_scoring_llm", scoring_llm):
            result = gtm_agent.score_prospect.invoke({
                "prospect_profile": profile["prospect_profile"],
                "offering": data_service.get_offering("OFFER-10004"),
            })

        messages = scoring_llm.invoke.call_args.args[0]
        scored_profile = json.loads(messages[1]["content"].split("Prospect profile:\n")[1])
        self.assertIn("Terraform", scored_profile["tech_stack"])
        self.assertEqual(result, {"score": 90})

    def test_repeated_update_does_not_duplicate_technology(self):
        data_service.update_prospect_info("LEAD-39002", "Terraform")
        data_service.update_prospect_info("LEAD-39002", "Terraform")

        self.assertEqual(data_service.fetch_tech_stack("LEAD-39002").count("Terraform"), 1)

    def test_unknown_prospect_is_not_updated(self):
        self.assertEqual(
            data_service.update_prospect_info("missing-prospect", "Terraform"),
            {"updated": False, "found": False},
        )
