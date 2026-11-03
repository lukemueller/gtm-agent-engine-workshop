import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "LANGSMITH_TRACING": "false"}):
    from gtm_agent import gtm_agent


class SendProspectEmailTests(unittest.TestCase):
    def setUp(self):
        self.prospect = {"prospect_id": "LEAD-TEST", "name": "Test Prospect", "email": "prospect@example.com"}
        self.rep = {"name": "Test Rep", "email": "rep@example.com"}
        self.runtime = SimpleNamespace(config={"metadata": {"user_id": "test-rep"}})

    def send(self, prospect):
        return gtm_agent.send_prospect_email.func(
            prospect, "Demo invite", "Please join our demo.", self.runtime, self.rep
        )

    def assert_sent(self, result):
        self.assertEqual(result, {
            "status": "sent",
            "message_id": result["message_id"],
            "to": self.prospect["email"],
            "to_name": self.prospect["name"],
            "from": self.rep["email"],
            "from_name": self.rep["name"],
            "subject": "Demo invite",
            "body": "Please join our demo.",
        })
        self.assertTrue(result["message_id"].startswith("msg-"))

    def test_disqualified_source_blocks_even_when_contact_says_qualified(self):
        prospect = {**self.prospect, "disqualified": False}
        with patch.dict(gtm_agent.data_service.PROSPECTS, {"LEAD-TEST": {"disqualified": True}}):
            result = self.send(prospect)
        self.assertEqual(result, {
            "status": "blocked",
            "error": "Prospect is disqualified; email not sent.",
            "prospect_id": "LEAD-TEST",
        })
        self.assertNotIn("message_id", result)

    def test_qualified_source_sends(self):
        with patch.dict(gtm_agent.data_service.PROSPECTS, {"LEAD-TEST": {"disqualified": False}}):
            self.assert_sent(self.send(self.prospect))

    def test_missing_source_record_sends(self):
        with patch.dict(gtm_agent.data_service.PROSPECTS, {}, clear=True):
            self.assert_sent(self.send(self.prospect))

    def test_missing_prospect_id_sends(self):
        prospect = {k: v for k, v in self.prospect.items() if k != "prospect_id"}
        self.assert_sent(self.send(prospect))

    def test_missing_email_preserves_failure(self):
        for email in (None, ""):
            with self.subTest(email=email):
                prospect = {**self.prospect, "email": email}
                with patch.dict(gtm_agent.data_service.PROSPECTS, {"LEAD-TEST": {"disqualified": True}}):
                    self.assertEqual(self.send(prospect), {
                        "status": "failed", "error": "Prospect record has no email address."
                    })


if __name__ == "__main__":
    unittest.main()
