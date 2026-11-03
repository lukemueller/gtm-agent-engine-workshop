"""Offline checks for rep context and ID-based scoring."""

import importlib
import json
import os
import unittest
from unittest.mock import Mock, patch

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

with patch.dict(os.environ, {"OPENAI_API_KEY": "test", "LANGSMITH_TRACING": "false"}):
    module = importlib.import_module("gtm_agent.gtm_agent")


class ToolCallingModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, **kwargs):
        context = "\n".join(message.text for message in messages if message.type == "system")
        rep = module.data_service.get_rep("rep_amills")
        if rep["name"] not in context or rep["email"] not in context:
            raise AssertionError("Signed-in rep missing from model context")
        return super()._generate(messages, **kwargs)


class AgentTests(unittest.TestCase):
    def setUp(self):
        module.data_service._PROFILES.clear()
        self.addCleanup(module.data_service._PROFILES.clear)

    def test_email_uses_rep_context_without_rep_tool(self):
        prospect = module.data_service.get_prospect_record("LEAD-71001")
        model = ToolCallingModel(responses=[
            AIMessage(content="", tool_calls=[{
                "name": "send_prospect_email", "id": "email",
                "args": {"prospect": {"name": prospect["name"], "email": prospect["email"]},
                         "subject": "Demo", "body": "Best, Ava Mills"},
            }]),
            AIMessage(content="Email sent."),
        ])
        graph = module.create_deep_agent(
            model=model, tools=[module.send_prospect_email], system_prompt=module.SYSTEM_PROMPT,
        )
        with patch.object(module, "gtm_agent", graph), patch.object(
            module.get_current_rep, "func", side_effect=AssertionError("Unnecessary rep lookup"),
        ):
            result = module.run_agent("Email LEAD-71001", user_id="rep_amills")
        sent = next(json.loads(message.content) for message in result["messages"]
                    if message.type == "tool")
        self.assertEqual(sent["status"], "sent")
        self.assertEqual(sent["from"], module.data_service.get_rep("rep_amills")["email"])
        self.assertEqual(result["reply"], "Email sent.")

    def test_missing_rep_does_not_crash(self):
        graph = Mock()
        graph.invoke.return_value = {"messages": [AIMessage(content="Ready.")]}
        with patch.object(module, "gtm_agent", graph):
            result = module.run_agent("Hello", user_id="missing-rep")
        self.assertEqual(result["reply"], "Ready.")
        context = graph.invoke.call_args.args[0]["messages"][0]
        self.assertEqual(context, {"role": "system", "content": "Signed-in rep not found."})

    def test_score_fetches_records_from_ids(self):
        score = {"score": 75, "justification": "Fit", "rubric_breakdown": {}}
        scoring = Mock()
        scoring.invoke.return_value.model_dump.return_value = score
        model = ToolCallingModel(responses=[
            AIMessage(content="", tool_calls=[{
                "name": "score_prospect", "id": "score",
                "args": {"prospect_id": "LEAD-71001", "offering_id": "OFFER-10005"},
            }]),
            AIMessage(content="Score: 75"),
        ])
        graph = module.create_deep_agent(
            model=model, tools=[module.score_prospect], system_prompt=module.SYSTEM_PROMPT,
        )
        with patch.object(module, "_scoring_llm", scoring), patch.object(module, "gtm_agent", graph):
            result = module.run_agent("Score LEAD-71001 against OFFER-10005", user_id="rep_amills")
        tool_results = [json.loads(message.content) for message in result["messages"]
                        if message.type == "tool"]
        self.assertEqual(tool_results, [score])
        content = scoring.invoke.call_args.args[0][1]["content"]
        self.assertIn(json.dumps(module.data_service.get_offering("OFFER-10005"), indent=2), content)
        profile = module.build_prospect_profile.invoke({"prospect_id": "LEAD-71001"})
        self.assertTrue(profile["found"])
        self.assertIn(json.dumps(profile["prospect_profile"], indent=2), content)
        schema = module.score_prospect.get_input_schema().model_json_schema()
        self.assertEqual(set(schema["properties"]), {"prospect_id", "offering_id"})
        self.assertEqual(set(schema["required"]), {"prospect_id", "offering_id"})

    def test_missing_scoring_records_do_not_call_model(self):
        for prospect_id, offering_id, missing_id in [
            ("missing-prospect", "OFFER-10005", "missing-prospect"),
            ("LEAD-71001", "missing-offering", "missing-offering"),
        ]:
            with self.subTest(missing_id=missing_id), patch.object(module, "_scoring_llm") as scoring:
                result = module.score_prospect.invoke({
                    "prospect_id": prospect_id, "offering_id": offering_id,
                })
                self.assertIsNone(result["score"])
                self.assertIn(missing_id, result["error"])
                scoring.invoke.assert_not_called()


if __name__ == "__main__":
    unittest.main()
