"""Offline tests for provider-specific ranking request translation."""
# ruff: noqa: D101, D102

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_client import RankingClient  # noqa: E402


class FakeBedrockClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class FakeOpenAIClient:
    def __init__(self, message):
        self.message = message
        self.calls = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=self.message)])


class FakeResponses:
    def __init__(self, output_text):
        self.output_text = output_text
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_text=self.output_text)


class FakeAzureOpenAIClient:
    def __init__(self, output_text):
        self.responses = FakeResponses(output_text)


class RankingClientTests(unittest.TestCase):
    def test_bedrock_uses_native_converse_shape_and_collects_text(self):
        """Bedrock must receive Converse blocks, not OpenAI-style messages."""
        transport = FakeBedrockClient(
            {
                "output": {
                    "message": {
                        "content": [
                            {"reasoningContent": {"reasoningText": {"text": "hidden"}}},
                            {"text": "A"},
                            {"text": "B"},
                        ]
                    }
                }
            }
        )
        client = RankingClient(
            "openai.gpt-oss-20b-1:0",
            provider="amazon-bedrock",
            region="ap-southeast-2",
            client=transport,
        )

        self.assertEqual(client.generate("rank this", max_tokens=3), "A\nB")
        self.assertEqual(
            transport.calls,
            [
                {
                    "modelId": "openai.gpt-oss-20b-1:0",
                    "messages": [{"role": "user", "content": [{"text": "rank this"}]}],
                    "inferenceConfig": {"maxTokens": 1024, "temperature": 0},
                }
            ],
        )

    def test_gpt_oss_filter_uses_low_reasoning_effort(self):
        """Reserve filter output tokens for the cleaned passage, not deliberation."""
        transport = FakeBedrockClient(
            {
                "stopReason": "end_turn",
                "output": {"message": {"content": [{"text": "clean"}]}},
            }
        )
        client = RankingClient(
            "openai.gpt-oss-20b-1:0",
            provider="amazon-bedrock",
            client=transport,
        )

        self.assertEqual(
            client.generate(
                "clean this",
                max_tokens=1024,
                require_complete=True,
                reasoning_effort="low",
            ),
            "clean",
        )
        self.assertEqual(
            transport.calls[0]["additionalModelRequestFields"],
            {"reasoning_effort": "low"},
        )

    def test_bedrock_usage_is_returned_when_requested(self):
        """Expose Converse usage for persistent evaluation accounting."""
        transport = FakeBedrockClient(
            {
                "output": {"message": {"content": [{"text": "A"}]}},
                "usage": {
                    "inputTokens": 11,
                    "outputTokens": 3,
                    "totalTokens": 14,
                },
            }
        )
        client = RankingClient("model", provider="amazon-bedrock", client=transport)

        self.assertEqual(
            client.generate("rank this", max_tokens=3, return_usage=True),
            ("A", {"input_tokens": 11, "output_tokens": 3, "total_tokens": 14}),
        )

    def test_openai_keeps_vllm_thinking_override(self):
        """The existing vLLM path still needs its chat-template extension."""
        message = SimpleNamespace(content=None, model_extra={"reasoning_content": "B"})
        transport = FakeOpenAIClient(message)
        client = RankingClient("local-model", client=transport)

        self.assertEqual(client.generate("rank this", max_tokens=3), "B")
        self.assertEqual(
            transport.calls[0]["extra_body"],
            {"chat_template_kwargs": {"enable_thinking": False}},
        )

    def test_azure_openai_uses_responses_api_for_gpt_5_6(self):
        """Azure v1 requests must not inherit legacy vLLM request fields."""
        transport = FakeAzureOpenAIClient(" C ")
        client = RankingClient(
            "gpt-5.6",
            provider="azure-openai",
            base_url="https://example.openai.azure.com/openai/v1",
            client=transport,
        )

        self.assertEqual(client.generate("rank this", max_tokens=3), "C")
        self.assertEqual(
            transport.responses.calls,
            [
                {
                    "model": "gpt-5.6",
                    "input": "rank this",
                    "max_output_tokens": 128,
                    "reasoning": {"effort": "none"},
                    "text": {"verbosity": "low"},
                }
            ],
        )

    def test_region_precedence_matches_existing_bedrock_setup(self):
        """An explicit region wins over Bedrock and standard AWS environment vars."""
        with patch.dict(
            os.environ,
            {"BEDROCK_REGION": "us-west-2", "AWS_REGION": "us-east-1"},
            clear=False,
        ):
            explicit = RankingClient(
                "model",
                provider="amazon-bedrock",
                region="ap-southeast-2",
                client=object(),
            )
            environment = RankingClient(
                "model", provider="amazon-bedrock", client=object()
            )

        self.assertEqual(explicit.region, "ap-southeast-2")
        self.assertEqual(environment.region, "us-west-2")

    def test_unknown_provider_is_rejected(self):
        """A typo must fail before an experiment launches thousands of tasks."""
        with self.assertRaisesRegex(ValueError, "Unsupported provider"):
            RankingClient("model", provider="bedrok", client=object())


if __name__ == "__main__":
    unittest.main()
