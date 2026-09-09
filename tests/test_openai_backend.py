import asyncio

import httpx
import pytest
from openai import BadRequestError

from gateway.backends.base import BackendUpstreamError
from gateway.backends.openai import OpenAIBackend
from gateway.config import BackendDefinition

def test_openai_backend_forwards_timeout_and_headers(monkeypatch):
    captured = {}

    class DummyClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("gateway.backends.openai.OpenAI", DummyClient)

    backend = OpenAIBackend(
        BackendDefinition(
            type="openai",
            name="openai-primary",
            base_url="https://api.openai.com/v1",
            api_key="sk-test",
            request_timeout_s=123.0,
            extra_headers={"x-test-header": "value"},
        )
    )

    assert backend._client is not None
    assert {key: value for key, value in captured.items() if key != "http_client"} == {
        "api_key": "sk-test",
        "base_url": "https://api.openai.com/v1",
        "timeout": 123.0,
        "default_headers": {"x-test-header": "value"},
        "max_retries": 0,
    }
    http_client = captured["http_client"]
    assert http_client.timeout.connect == 123.0
    assert http_client.timeout.read == 123.0
    assert http_client.timeout.write == 123.0
    assert http_client.timeout.pool == 123.0
    assert http_client.headers["x-test-header"] == "value"
    http_client.close()


def test_openai_backend_optionally_normalizes_aqueduct_chat_payload(monkeypatch):
    forwarded = []

    class DummyCompletions:
        def create(self, **kwargs):
            forwarded.append(kwargs)
            if kwargs["stream"]:
                return iter(())
            return type("Response", (), {"model_dump": lambda self: {"usage": {}}})()

    class DummyClient:
        def __init__(self, **kwargs):
            self.chat = type("Chat", (), {"completions": DummyCompletions()})()

    monkeypatch.setattr("gateway.backends.openai.OpenAI", DummyClient)
    backend = OpenAIBackend(
        BackendDefinition(
            type="openai",
            name="aqueduct",
            base_url="https://aqueduct.example",
            api_key="test-key",
            strip_reasoning_effort=True,
            reorder_system_messages_to_front=True,
            coalesce_system_messages=True,
        )
    )
    tool_definition = {"type": "function", "function": {"name": "lookup", "parameters": {}}}
    tool_result = {"role": "tool", "tool_call_id": "call-1", "content": "result"}
    payload = {
        "model": "qwen",
        "reasoning_effort": "medium",
        "tools": [tool_definition],
        "messages": [
            {"role": "system", "content": "First instruction"},
            {"role": "user", "content": "Question"},
            {"role": "system", "content": "Second instruction"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "call-1"}]},
            tool_result,
        ],
    }

    asyncio.run(backend.chat("qwen", payload))
    stream = asyncio.run(backend.chat("qwen", payload, stream=True))
    asyncio.run(stream.aclose())

    for forwarded_payload in forwarded:
        assert "reasoning_effort" not in forwarded_payload
        assert forwarded_payload["tools"] == [tool_definition]
        assert forwarded_payload["messages"] == [
            {"role": "system", "content": "First instruction\n\nSecond instruction"},
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "call-1"}]},
            tool_result,
        ]


def test_openai_backend_preserves_payload_when_cleanup_flags_are_disabled(monkeypatch):
    forwarded = {}

    class DummyCompletions:
        def create(self, **kwargs):
            forwarded.update(kwargs)
            return type("Response", (), {"model_dump": lambda self: {"usage": {}}})()

    class DummyClient:
        def __init__(self, **kwargs):
            self.chat = type("Chat", (), {"completions": DummyCompletions()})()

    monkeypatch.setattr("gateway.backends.openai.OpenAI", DummyClient)
    backend = OpenAIBackend(
        BackendDefinition(type="openai", name="openai", base_url="https://example", api_key="test-key")
    )
    messages = [{"role": "user", "content": "Question"}, {"role": "system", "content": "Instruction"}]
    asyncio.run(backend.chat("model", {"model": "model", "reasoning_effort": "medium", "messages": messages}))

    assert forwarded["reasoning_effort"] == "medium"
    assert forwarded["messages"] == messages


@pytest.mark.parametrize("stream", [False, True])
def test_openai_backend_converts_upstream_http_400_to_bounded_gateway_error(monkeypatch, stream):
    class DummyCompletions:
        def create(self, **kwargs):
            response = httpx.Response(
                400,
                request=httpx.Request("POST", "https://aqueduct.example/v1/chat/completions"),
            )
            raise BadRequestError("invalid request", response=response, body={"error": "invalid reasoning_effort"})

    class DummyClient:
        def __init__(self, **kwargs):
            self.chat = type("Chat", (), {"completions": DummyCompletions()})()

    monkeypatch.setattr("gateway.backends.openai.OpenAI", DummyClient)
    backend = OpenAIBackend(
        BackendDefinition(type="openai", name="aqueduct", base_url="https://aqueduct.example", api_key="test-key")
    )

    with pytest.raises(BackendUpstreamError) as exc_info:
        asyncio.run(backend.chat("qwen", {"model": "qwen", "messages": []}, stream=stream))

    assert exc_info.value.status_code == 400
    assert exc_info.value.body_preview == '{"error":"invalid reasoning_effort"}'
