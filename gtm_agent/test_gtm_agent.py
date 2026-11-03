import importlib
import json
import os
import unittest
import uuid
from unittest.mock import patch

import httpx
from langchain_openai import ChatOpenAI


class AgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.requests = []
        cls.responses = []

        def respond(request):
            cls.requests.append(json.loads(request.content))
            message = cls.responses.pop(0)
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "gpt-4o-mini",
                    "choices": [
                        {
                            "index": 0,
                            "message": message,
                            "finish_reason": "tool_calls"
                            if message.get("tool_calls")
                            else "stop",
                        }
                    ],
                },
            )

        client = httpx.Client(transport=httpx.MockTransport(respond))
        cls.addClassCleanup(client.close)

        def model(**kwargs):
            return ChatOpenAI(**kwargs, http_client=client)

        with (
            patch.dict(
                os.environ, {"OPENAI_API_KEY": "test", "LANGSMITH_TRACING": "false"}
            ),
            patch("langchain_openai.ChatOpenAI", side_effect=model),
        ):
            cls.agent = importlib.import_module("gtm_agent.gtm_agent")

    def setUp(self):
        self.requests.clear()
        self.responses.clear()

    def queue_tool(self, name, arguments):
        self.responses.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": name,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(arguments),
                        },
                    }
                ],
            }
        )

    def test_first_call_contains_only_application_prompt_and_tools(self):
        self.responses.append({"role": "assistant", "content": "Ready."})
        with patch.dict(os.environ, {"LANGSMITH_TRACING": "false"}):
            self.agent.run_agent("Look up OFFER-10001", user_id="REP-001")
        request = self.requests[0]
        self.assertEqual(
            request["messages"],
            [
                {"role": "system", "content": self.agent.SYSTEM_PROMPT},
                {"role": "user", "content": "Look up OFFER-10001"},
            ],
        )
        self.assertEqual(
            {tool["function"]["name"] for tool in request["tools"]},
            {
                "lookup_offering",
                "build_prospect_profile",
                "get_prospect",
                "send_prospect_email",
                "score_prospect",
                "update_prospect_info",
                "get_current_rep",
            },
        )
        payload = json.dumps({key: request[key] for key in ("messages", "tools")})
        self.assertLess(len(payload), 6000)
        print(f"First-call messages and schemas: {len(payload)} JSON characters")

    def test_run_agent_propagates_signed_in_rep_to_tools(self):
        prospect_id = next(iter(self.agent.data_service.PROSPECTS))
        prospect = self.agent.data_service.PROSPECTS[prospect_id]
        for rep in self.agent.REP_IDS[:2]:
            with self.subTest(user_id=rep["rep_id"]):
                self.queue_tool("get_current_rep", {})
                self.queue_tool("get_prospect", {"prospect_id": prospect_id})
                self.queue_tool(
                    "send_prospect_email",
                    {
                        "prospect": {
                            "name": prospect["name"],
                            "email": prospect["email"],
                        },
                        "subject": "Hello",
                        "body": "Following up.",
                    },
                )
                self.responses.append({"role": "assistant", "content": "Email sent."})
                with patch.dict(os.environ, {"LANGSMITH_TRACING": "false"}):
                    result = self.agent.run_agent(
                        f"Email {prospect_id}",
                        user_id=rep["rep_id"],
                        environment="test",
                        thread_id="test-thread",
                    )
                outputs = {
                    message.name: json.loads(message.content)
                    for message in result["messages"]
                    if message.type == "tool"
                }
                self.assertEqual(outputs["get_current_rep"]["rep"], rep)
                self.assertTrue(outputs["get_prospect"]["found"])
                self.assertEqual(outputs["send_prospect_email"]["status"], "sent")
                self.assertEqual(outputs["send_prospect_email"]["from"], rep["email"])
                self.assertEqual(result["reply"], "Email sent.")
                self.assertEqual(str(uuid.UUID(result["run_id"])), result["run_id"])
                self.assertFalse(self.responses)


if __name__ == "__main__":
    unittest.main()
