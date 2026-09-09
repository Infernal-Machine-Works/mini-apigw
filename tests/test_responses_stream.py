import asyncio
import json

import pytest
from pydantic import TypeAdapter
from openai._types import omit
from openai.types.responses import ResponseStreamEvent
from openai.types.responses import response_create_params
from openai.lib.streaming.responses import ResponseStreamState
from openai.types.responses.response import Response

from gateway.backends.base import BackendUsage
from gateway.config import BackendDefinition, BackendSupports, ResponsesShimBackendConfig
from gateway.responses import (
    ResponsesJobRegistry,
    ResponsesShim,
    ResponsesStateStore,
    ResponsesToolRegistry,
    ShimContext,
)


class _FakeClient:
    async def chat(self, model: str, payload, stream: bool = False):
        if not stream:
            raise AssertionError("stream flag expected in test")

        async def _generator():
            chunk = {
                "choices": [
                    {
                        "delta": {
                            "content": [
                                {"type": "output_text", "text": "Hello"},
                                {"type": "output_text", "text": " world"},
                            ]
                        }
                    }
                ]
            }
            yield b"data: " + json.dumps(chunk).encode("utf-8") + b"\n\n"
            yield b"data: [DONE]\n\n"

        return _generator()


class _FakeNonStreamClient:
    async def chat(self, model: str, payload, stream: bool = False):
        if stream:
            raise AssertionError("non-stream flag expected in test")
        return type(
            "Result",
            (),
            {
                "body": {
                    "id": "chatcmpl-test",
                    "choices": [{"message": {"role": "assistant", "content": "Hello world"}}],
                },
                "usage": BackendUsage(input_tokens=3, output_tokens=2),
            },
        )()


class _FakeResponsesClient:
    def __init__(self):
        self.calls = []

    async def responses(self, model: str, payload, stream: bool = False):
        if stream:
            raise AssertionError("responses shim should force non-stream backend call")
        from gateway.backends.base import BackendResult

        self.calls.append((model, payload, stream))

        return BackendResult(
            body={
                "id": "resp_test",
                "object": "response",
                "created_at": 0,
                "model": model,
                "status": "completed",
                "output": [
                    {
                        "id": "msg_test",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "hello", "annotations": []}],
                    }
                ],
            },
            usage=BackendUsage(input_tokens=2, output_tokens=3, total_tokens=5),
        )


@pytest.mark.asyncio
async def test_responses_shim_stream_produces_responses_events():
    backend = BackendDefinition(
        type="shim",
        name="shim-backend",
        base_url="http://shim",
        supports=BackendSupports(chat=["shim:model"]),
        responses_shim=ResponsesShimBackendConfig(enabled=True, operation="chat"),
    )
    ctx = ShimContext(backend=backend, backend_model="shim:model", shim_config=backend.responses_shim)
    shim = ResponsesShim(ResponsesStateStore(), ResponsesToolRegistry(), ResponsesJobRegistry())
    fake_client = _FakeClient()
    stream = await shim.handle(
        ctx,
        fake_client,
        {"model": "shim-model", "input": [{"role": "user", "content": [{"type": "input_text", "text": "hi"}]}]},
        stream=True,
    )
    events = []
    async for chunk in stream:
        data = chunk.decode("utf-8").strip()
        if not data.startswith("data: "):
            continue
        payload = data[6:]
        if payload == "[DONE]":
            continue
        events.append(json.loads(payload))
    assert events[0]["type"] == "response.created"
    assert events[0]["response"]["output"] == []
    assert events[1]["type"] == "response.output_item.added"
    assert events[1]["item"]["id"].startswith("msg-")
    assert events[1]["item"]["status"] == "in_progress"
    assert events[2]["type"] == "response.content_part.added"
    assert events[2]["part"]["annotations"] == []
    assert events[3]["type"] == "response.output_text.delta"
    assert events[3]["delta"] == "Hello world"
    assert events[-1]["type"] == "response.completed"
    assert events[-1]["response"]["output"][0]["content"][0]["text"] == "Hello world"


@pytest.mark.asyncio
async def test_responses_shim_can_wrap_non_stream_responses_backend_result():
    backend = BackendDefinition(
        type="openai",
        name="shim-backend",
        base_url="http://shim",
        supports=BackendSupports(responses=[]),
        responses_shim=ResponsesShimBackendConfig(enabled=True, operation="responses"),
    )
    ctx = ShimContext(backend=backend, backend_model="shim:model", shim_config=backend.responses_shim)
    shim = ResponsesShim(ResponsesStateStore(), ResponsesToolRegistry(), ResponsesJobRegistry())
    result = await shim.handle(
        ctx,
        _FakeResponsesClient(),
        {"model": "shim-model", "input": "hi"},
        stream=True,
    )
    assert result.body["output"][0]["content"][0]["text"] == "hello"
    assert result.usage.total_tokens == 5


@pytest.mark.asyncio
async def test_responses_shim_native_fallback_strips_unsupported_fields_and_maps_client_metadata():
    backend = BackendDefinition(
        type="openai",
        name="shim-backend",
        base_url="http://shim",
        supports=BackendSupports(responses=[]),
        responses_shim=ResponsesShimBackendConfig(enabled=True, operation="responses"),
    )
    ctx = ShimContext(backend=backend, backend_model="shim:model", shim_config=backend.responses_shim)
    shim = ResponsesShim(ResponsesStateStore(), ResponsesToolRegistry(), ResponsesJobRegistry())
    client = _FakeResponsesClient()
    await shim.handle(
        ctx,
        client,
        {
            "model": "shim-model",
            "input": "hi",
            "client_metadata": {"source": "codex", "session": "abc"},
            "reasoning": {"effort": "medium"},
            "text": {"verbosity": "low"},
            "include": ["reasoning.encrypted_content"],
            "truncation": "auto",
            "metadata": {"origin": "existing"},
            "extra_field": "drop-me",
        },
        stream=True,
    )
    _, payload, stream = client.calls[0]
    assert stream is False
    assert payload["input"] == "hi"
    assert payload["metadata"] == {"origin": "existing"}
    assert payload["reasoning"] == {"effort": "medium"}
    assert payload["text"] == {"verbosity": "low"}
    assert payload["include"] == ["reasoning.encrypted_content"]
    assert payload["truncation"] == "auto"
    assert "client_metadata" not in payload
    assert "extra_field" not in payload


@pytest.mark.asyncio
async def test_responses_shim_native_fallback_uses_client_metadata_when_metadata_missing():
    backend = BackendDefinition(
        type="openai",
        name="shim-backend",
        base_url="http://shim",
        supports=BackendSupports(responses=[]),
        responses_shim=ResponsesShimBackendConfig(enabled=True, operation="responses"),
    )
    ctx = ShimContext(backend=backend, backend_model="shim:model", shim_config=backend.responses_shim)
    shim = ResponsesShim(ResponsesStateStore(), ResponsesToolRegistry(), ResponsesJobRegistry())
    client = _FakeResponsesClient()
    await shim.handle(
        ctx,
        client,
        {
            "model": "shim-model",
            "input": "hi",
            "client_metadata": {"source": "codex"},
        },
        stream=False,
    )
    _, payload, _ = client.calls[0]
    assert payload["metadata"] == {"source": "codex"}
    assert "client_metadata" not in payload


def test_responses_shim_native_fallback_normalizes_output_text_message_parts_for_sdk_schema():
    shim = ResponsesShim(ResponsesStateStore(), ResponsesToolRegistry(), ResponsesJobRegistry())
    payload = shim._translate_responses_payload(
        {
            "input": [
                {
                    "type": "message",
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "Follow instructions"}],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Previous answer", "annotations": []}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": ["Next question"],
                },
            ]
        }
    )
    assert payload["input"][1]["content"] == [{"type": "input_text", "text": "Previous answer"}]
    assert payload["input"][2]["content"] == [{"type": "input_text", "text": "Next question"}]
    adapter = TypeAdapter(response_create_params.ResponseCreateParamsNonStreaming)
    validated = adapter.validate_python({"model": "glm-5.2", "input": payload["input"], "stream": False})
    assert validated["input"][1]["content"][0]["type"] == "input_text"


def test_responses_shim_non_stream_payload_matches_openai_response_schema():
    backend = BackendDefinition(
        type="shim",
        name="shim-backend",
        base_url="http://shim",
        supports=BackendSupports(chat=["shim:model"]),
        responses_shim=ResponsesShimBackendConfig(enabled=True, operation="chat"),
    )
    shim = ResponsesShim(ResponsesStateStore(), ResponsesToolRegistry(), ResponsesJobRegistry())
    body = shim._normalize_chat_response(
        "shim-model",
        {
            "id": "chatcmpl-test",
            "choices": [{"message": {"role": "assistant", "content": "Hello world"}}],
            "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
        },
    )
    parsed = Response.model_validate(body)
    assert parsed.status == "completed"
    assert parsed.output[0].content[0].text == "Hello world"
    assert parsed.usage.input_tokens == 3


def test_responses_shim_maps_chat_tool_calls_to_response_output_items():
    shim = ResponsesShim(ResponsesStateStore(), ResponsesToolRegistry(), ResponsesJobRegistry())
    body = shim._normalize_chat_response(
        "shim-model",
        {
            "id": "chatcmpl-test",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "add_todo",
                                    "arguments": '{"item":"Milk"}',
                                },
                            }
                        ],
                    }
                }
            ],
        },
    )
    parsed = Response.model_validate(body)
    assert parsed.output[0].type == "function_call"
    assert parsed.output[0].call_id == "call_123"
    assert parsed.output[0].name == "add_todo"
    assert parsed.output[0].arguments == '{"item":"Milk"}'


def test_responses_shim_coerces_iso_created_at_to_numeric_timestamp():
    shim = ResponsesShim(ResponsesStateStore(), ResponsesToolRegistry(), ResponsesJobRegistry())
    body = shim._normalize_chat_response(
        "shim-model",
        {
            "id": "chatcmpl-test",
            "created_at": "2026-07-02T17:34:14.089848Z",
            "choices": [{"message": {"role": "assistant", "content": "Hello world"}}],
            "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
        },
    )
    assert isinstance(body["created_at"], int)


def test_single_response_stream_merges_backend_usage():
    shim = ResponsesShim(ResponsesStateStore(), ResponsesToolRegistry(), ResponsesJobRegistry())
    body = shim._normalize_chat_response(
        "shim-model",
        {
            "id": "chatcmpl-test",
            "choices": [{"message": {"role": "assistant", "content": "Hello world"}}],
        },
    )

    async def collect():
        stream = shim._single_response_stream(body, BackendUsage(input_tokens=2, output_tokens=3, total_tokens=5))
        events = []
        async for chunk in stream:
            data = chunk.decode("utf-8").strip()
            if not data.startswith("data: "):
                continue
            payload = data[6:]
            if payload == "[DONE]":
                continue
            events.append(json.loads(payload))
        return events

    events = asyncio.run(collect())
    assert events[-1]["response"]["usage"]["input_tokens"] == 2
    assert events[-1]["response"]["usage"]["output_tokens"] == 3
    assert events[-1]["response"]["usage"]["total_tokens"] == 5


def test_responses_shim_omits_usage_when_backend_does_not_report_it():
    shim = ResponsesShim(ResponsesStateStore(), ResponsesToolRegistry(), ResponsesJobRegistry())
    body = shim._normalize_chat_response(
        "shim-model",
        {
            "id": "chatcmpl-test",
            "choices": [{"message": {"role": "assistant", "content": "Hello world"}}],
        },
    )
    assert "usage" not in body


def test_responses_shim_preserves_partial_usage_fields():
    shim = ResponsesShim(ResponsesStateStore(), ResponsesToolRegistry(), ResponsesJobRegistry())
    body = shim._normalize_chat_response(
        "shim-model",
        {
            "id": "chatcmpl-test",
            "choices": [{"message": {"role": "assistant", "content": "Hello world"}}],
            "usage": {"input_tokens": 3},
        },
    )
    assert body["usage"] == {
        "input_tokens": 3,
        "input_tokens_details": {"cached_tokens": 0},
    }


def test_responses_shim_translates_responses_tool_roundtrip_to_chat_messages():
    shim = ResponsesShim(ResponsesStateStore(), ResponsesToolRegistry(), ResponsesJobRegistry())
    messages = shim._build_messages_from_input(
        {
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Add milk"}],
                },
                {
                    "type": "function_call",
                    "call_id": "call_123",
                    "name": "add_todo",
                    "arguments": '{"item":"Milk"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_123",
                    "output": "done",
                },
            ]
        }
    )
    assert messages == [
        {"role": "user", "content": "Add milk"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {"name": "add_todo", "arguments": '{"item":"Milk"}'},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_123",
            "content": "done",
        },
    ]


def test_responses_shim_translates_responses_tools_to_chat_tool_shape():
    shim = ResponsesShim(ResponsesStateStore(), ResponsesToolRegistry(), ResponsesJobRegistry())
    payload = shim._translate_chat_payload(
        {
            "input": "hi",
            "tools": [
                {
                    "type": "function",
                    "name": "add_todo",
                    "description": "Add todo",
                    "parameters": {
                        "type": "object",
                        "properties": {"item": {"type": "string"}},
                        "required": ["item"],
                    },
                }
            ],
        }
    )
    assert payload["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "add_todo",
                "description": "Add todo",
                "parameters": {
                    "type": "object",
                    "properties": {"item": {"type": "string"}},
                    "required": ["item"],
                },
            },
        }
    ]


def test_responses_shim_normalizes_function_tools_without_explicit_type():
    shim = ResponsesShim(ResponsesStateStore(), ResponsesToolRegistry(), ResponsesJobRegistry())
    payload = shim._translate_chat_payload(
        {
            "input": "hi",
            "tools": [
                {
                    "name": "shell",
                    "description": "Run a shell command",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                    "strict": True,
                }
            ],
        }
    )
    assert payload["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "shell",
                "description": "Run a shell command",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
                "strict": True,
            },
        }
    ]


def test_responses_shim_flattens_namespace_tools_and_drops_non_function_tools():
    shim = ResponsesShim(ResponsesStateStore(), ResponsesToolRegistry(), ResponsesJobRegistry())
    payload = shim._translate_chat_payload(
        {
            "input": "hi",
            "tools": [
                {
                    "type": "namespace",
                    "name": "example_namespace",
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "nested_tool",
                                "description": "Nested tool",
                                "parameters": {"type": "object", "properties": {}},
                            },
                        }
                    ],
                },
                {"type": "web_search", "search_content_types": ["text"]},
            ],
        }
    )
    assert payload["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "nested_tool",
                "description": "Nested tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]


def test_responses_shim_chat_fallback_strips_responses_only_fields_for_openai_chat():
    shim = ResponsesShim(ResponsesStateStore(), ResponsesToolRegistry(), ResponsesJobRegistry())
    payload = shim._translate_chat_payload(
        {
            "input": "hi",
            "client_metadata": {"source": "codex"},
            "reasoning": {"effort": "medium"},
            "text": {"verbosity": "low"},
            "include": ["reasoning.encrypted_content"],
            "truncation": "auto",
            "max_output_tokens": 321,
        },
        backend_type="openai",
    )
    assert payload["messages"] == [{"role": "user", "content": "hi"}]
    assert payload["max_tokens"] == 321
    assert "reasoning" not in payload
    assert "text" not in payload
    assert "include" not in payload
    assert "truncation" not in payload
    assert "client_metadata" not in payload


def test_responses_shim_chat_fallback_maps_text_json_schema_and_num_predict_for_ollama():
    shim = ResponsesShim(ResponsesStateStore(), ResponsesToolRegistry(), ResponsesJobRegistry())
    payload = shim._translate_chat_payload(
        {
            "input": "hi",
            "max_output_tokens": 123,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "structured_output",
                    "schema": {
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                    },
                }
            },
        },
        backend_type="ollama",
    )
    assert payload["messages"] == [{"role": "user", "content": "hi"}]
    assert payload["num_predict"] == 123
    assert payload["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "structured_output",
            "schema": {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
            },
        },
    }


def test_responses_shim_chat_fallback_normalizes_developer_messages_to_system():
    shim = ResponsesShim(ResponsesStateStore(), ResponsesToolRegistry(), ResponsesJobRegistry())
    payload = shim._translate_chat_payload(
        {
            "input": [
                {
                    "type": "message",
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "Follow the rules"}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Say hello"}],
                },
            ]
        },
        backend_type="ollama",
    )
    assert payload["messages"] == [
        {"role": "system", "content": "Follow the rules"},
        {"role": "user", "content": "Say hello"},
    ]


def test_responses_shim_chat_fallback_moves_system_messages_to_front():
    shim = ResponsesShim(ResponsesStateStore(), ResponsesToolRegistry(), ResponsesJobRegistry())
    payload = shim._translate_chat_payload(
        {
            "messages": [
                {"role": "user", "content": "First question"},
                {"role": "developer", "content": "Use terse answers"},
                {"role": "assistant", "content": "Previous answer"},
                {"role": "system", "content": "Follow policy"},
            ]
        },
        reorder_system_messages_to_front=True,
    )
    assert payload["messages"] == [
        {"role": "system", "content": "Use terse answers"},
        {"role": "system", "content": "Follow policy"},
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "Previous answer"},
    ]


def test_responses_shim_chat_fallback_preserves_message_order_by_default():
    shim = ResponsesShim(ResponsesStateStore(), ResponsesToolRegistry(), ResponsesJobRegistry())
    payload = shim._translate_chat_payload(
        {
            "messages": [
                {"role": "user", "content": "First question"},
                {"role": "developer", "content": "Use terse answers"},
                {"role": "assistant", "content": "Previous answer"},
            ]
        }
    )
    assert payload["messages"] == [
        {"role": "user", "content": "First question"},
        {"role": "system", "content": "Use terse answers"},
        {"role": "assistant", "content": "Previous answer"},
    ]


def test_responses_shim_chat_fallback_coalesces_leading_system_messages():
    shim = ResponsesShim(ResponsesStateStore(), ResponsesToolRegistry(), ResponsesJobRegistry())
    payload = shim._translate_chat_payload(
        {
            "messages": [
                {"role": "system", "content": "Primary instructions"},
                {"role": "developer", "content": "Secondary instructions"},
                {"role": "user", "content": "Quick test"},
            ]
        },
        coalesce_system_messages=True,
    )
    assert payload["messages"] == [
        {"role": "system", "content": "Primary instructions\n\nSecondary instructions"},
        {"role": "user", "content": "Quick test"},
    ]


def test_responses_shim_chat_fallback_reorders_then_coalesces_system_messages():
    shim = ResponsesShim(ResponsesStateStore(), ResponsesToolRegistry(), ResponsesJobRegistry())
    payload = shim._translate_chat_payload(
        {
            "messages": [
                {"role": "system", "content": "You are a coding agent"},
                {"role": "user", "content": "<environment_context>...</environment_context>"},
                {"role": "developer", "content": "<permissions instructions>...</permissions instructions>"},
                {"role": "user", "content": "Quick test - is this model working?"},
            ]
        },
        reorder_system_messages_to_front=True,
        coalesce_system_messages=True,
    )
    assert payload["messages"] == [
        {
            "role": "system",
            "content": "You are a coding agent\n\n<permissions instructions>...</permissions instructions>",
        },
        {"role": "user", "content": "<environment_context>...</environment_context>"},
        {"role": "user", "content": "Quick test - is this model working?"},
    ]


@pytest.mark.asyncio
async def test_responses_shim_non_stream_merges_backend_usage_and_derives_total_tokens():
    backend = BackendDefinition(
        type="shim",
        name="shim-backend",
        base_url="http://shim",
        supports=BackendSupports(chat=["shim:model"]),
        responses_shim=ResponsesShimBackendConfig(enabled=True, operation="chat"),
    )
    ctx = ShimContext(backend=backend, backend_model="shim:model", shim_config=backend.responses_shim)
    shim = ResponsesShim(ResponsesStateStore(), ResponsesToolRegistry(), ResponsesJobRegistry())
    result = await shim.handle(
        ctx,
        _FakeNonStreamClient(),
        {"model": "shim-model", "input": "hi"},
        stream=False,
    )
    assert result.usage.input_tokens == 3
    assert result.usage.output_tokens == 2
    assert result.body["usage"] == {
        "input_tokens": 3,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": 2,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 5,
    }
    parsed = Response.model_validate(result.body)
    assert parsed.usage.total_tokens == 5


def test_single_response_stream_derives_total_tokens_and_details_from_backend_usage():
    shim = ResponsesShim(ResponsesStateStore(), ResponsesToolRegistry(), ResponsesJobRegistry())
    body = shim._normalize_chat_response(
        "shim-model",
        {
            "id": "chatcmpl-test",
            "choices": [{"message": {"role": "assistant", "content": "Hello world"}}],
        },
    )

    async def collect():
        stream = shim._single_response_stream(body, BackendUsage(input_tokens=2, output_tokens=3))
        events = []
        async for chunk in stream:
            data = chunk.decode("utf-8").strip()
            if not data.startswith("data: "):
                continue
            payload = data[6:]
            if payload == "[DONE]":
                continue
            events.append(json.loads(payload))
        return events

    events = asyncio.run(collect())
    usage = events[-1]["response"]["usage"]
    assert usage["input_tokens"] == 2
    assert usage["input_tokens_details"] == {"cached_tokens": 0}
    assert usage["output_tokens"] == 3
    assert usage["output_tokens_details"] == {"reasoning_tokens": 0}
    assert usage["total_tokens"] == 5


def test_single_response_stream_preserves_function_call_output_items():
    shim = ResponsesShim(ResponsesStateStore(), ResponsesToolRegistry(), ResponsesJobRegistry())
    body = shim._normalize_chat_response(
        "shim-model",
        {
            "id": "chatcmpl-test",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {"name": "add_todo", "arguments": '{"item":"Milk"}'},
                            }
                        ],
                    }
                }
            ],
        },
    )

    async def collect():
        stream = shim._single_response_stream(body, BackendUsage(input_tokens=2, output_tokens=3))
        events = []
        async for chunk in stream:
            data = chunk.decode("utf-8").strip()
            if not data.startswith("data: "):
                continue
            payload = data[6:]
            if payload == "[DONE]":
                continue
            events.append(json.loads(payload))
        return events

    events = asyncio.run(collect())
    output = events[-1]["response"]["output"]
    assert output[0]["type"] == "function_call"
    assert output[0]["call_id"] == "call_123"
    assert output[0]["name"] == "add_todo"
    assert output[0]["arguments"] == '{"item":"Milk"}'


def test_single_response_stream_emits_canonical_responses_event_sequence():
    shim = ResponsesShim(ResponsesStateStore(), ResponsesToolRegistry(), ResponsesJobRegistry())
    body = shim._normalize_chat_response(
        "shim-model",
        {
            "id": "chatcmpl-test",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Hello world",
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {"name": "add_todo", "arguments": '{"item":"Milk"}'},
                            }
                        ],
                    }
                }
            ],
        },
    )

    async def collect():
        stream = shim._single_response_stream(body, BackendUsage(input_tokens=2, output_tokens=3))
        events = []
        async for chunk in stream:
            data = chunk.decode("utf-8").strip()
            if not data.startswith("data: "):
                continue
            payload = data[6:]
            if payload == "[DONE]":
                continue
            events.append(json.loads(payload))
        return events

    events = asyncio.run(collect())
    assert [event["type"] for event in events] == [
        "response.created",
        "response.output_item.added",
        "response.content_part.added",
        "response.output_text.delta",
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
        "response.output_item.added",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "response.output_item.done",
        "response.completed",
    ]
    assert [event["sequence_number"] for event in events] == list(range(len(events)))


def test_single_response_stream_events_are_accepted_by_openai_stream_state():
    shim = ResponsesShim(ResponsesStateStore(), ResponsesToolRegistry(), ResponsesJobRegistry())
    body = shim._normalize_chat_response(
        "shim-model",
        {
            "id": "chatcmpl-test",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Hello world",
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {"name": "add_todo", "arguments": '{"item":"Milk"}'},
                            }
                        ],
                    }
                }
            ],
        },
    )

    async def collect():
        stream = shim._single_response_stream(body, BackendUsage(input_tokens=2, output_tokens=3))
        events = []
        async for chunk in stream:
            data = chunk.decode("utf-8").strip()
            if not data.startswith("data: "):
                continue
            payload = data[6:]
            if payload == "[DONE]":
                continue
            events.append(json.loads(payload))
        return events

    raw_events = asyncio.run(collect())
    adapter = TypeAdapter(ResponseStreamEvent)
    state = ResponseStreamState(input_tools=[], text_format=omit)
    for payload in raw_events:
        state.handle_event(adapter.validate_python(payload))

    final = state._completed_response
    assert final is not None
    assert final.output[0].type == "message"
    assert final.output[0].content[0].text == "Hello world"
    assert final.output[1].type == "function_call"
    assert final.output[1].call_id == "call_123"
    assert final.output[1].arguments == '{"item":"Milk"}'


@pytest.mark.asyncio
async def test_responses_shim_stream_maps_chat_usage_aliases_to_response_usage():
    class _FakeClientWithChatUsage:
        async def chat(self, model: str, payload, stream: bool = False):
            if not stream:
                raise AssertionError("stream flag expected in test")

            async def _generator():
                chunk = {
                    "choices": [
                        {
                            "delta": {
                                "content": [
                                    {"type": "output_text", "text": "Hello world"},
                                ]
                            }
                        }
                    ]
                }
                usage_chunk = {
                    "choices": [],
                    "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
                }
                yield b"data: " + json.dumps(chunk).encode("utf-8") + b"\n\n"
                yield b"data: " + json.dumps(usage_chunk).encode("utf-8") + b"\n\n"
                yield b"data: [DONE]\n\n"

            return _generator()

    backend = BackendDefinition(
        type="shim",
        name="shim-backend",
        base_url="http://shim",
        supports=BackendSupports(chat=["shim:model"]),
        responses_shim=ResponsesShimBackendConfig(enabled=True, operation="chat"),
    )
    ctx = ShimContext(backend=backend, backend_model="shim:model", shim_config=backend.responses_shim)
    shim = ResponsesShim(ResponsesStateStore(), ResponsesToolRegistry(), ResponsesJobRegistry())
    stream = await shim.handle(
        ctx,
        _FakeClientWithChatUsage(),
        {"model": "shim-model", "input": "hi"},
        stream=True,
    )
    raw_events = []
    async for chunk in stream:
        data = chunk.decode("utf-8").strip()
        if not data.startswith("data: "):
            continue
        payload = data[6:]
        if payload == "[DONE]":
            continue
        raw_events.append(json.loads(payload))

    adapter = TypeAdapter(ResponseStreamEvent)
    state = ResponseStreamState(input_tools=[], text_format=omit)
    for payload in raw_events:
        state.handle_event(adapter.validate_python(payload))

    final = state._completed_response
    assert final is not None
    assert final.usage is not None
    assert final.usage.input_tokens == 11
    assert final.usage.output_tokens == 7
    assert final.usage.total_tokens == 18


@pytest.mark.asyncio
async def test_responses_shim_stream_ignores_reasoning_chunks_and_collects_late_content():
    class _FakeClientWithReasoningFirstStream:
        async def chat(self, model: str, payload, stream: bool = False):
            if not stream:
                raise AssertionError("stream flag expected in test")

            async def _generator():
                chunks = [
                    {
                        "choices": [
                            {
                                "delta": {
                                    "content": "",
                                    "role": "assistant",
                                }
                            }
                        ]
                    },
                    {
                        "choices": [
                            {
                                "delta": {
                                    "reasoning_content": "Thinking",
                                    "reasoning": "Thinking",
                                }
                            }
                        ]
                    },
                    {
                        "choices": [
                            {
                                "delta": {
                                    "content": "Hello",
                                }
                            }
                        ]
                    },
                    {
                        "choices": [
                            {
                                "delta": {
                                    "content": " world",
                                }
                            }
                        ]
                    },
                    {
                        "choices": [],
                        "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
                    },
                ]
                for chunk in chunks:
                    yield b"data: " + json.dumps(chunk).encode("utf-8") + b"\n\n"
                yield b"data: [DONE]\n\n"

            return _generator()

    backend = BackendDefinition(
        type="shim",
        name="shim-backend",
        base_url="http://shim",
        supports=BackendSupports(chat=["shim:model"]),
        responses_shim=ResponsesShimBackendConfig(enabled=True, operation="chat"),
    )
    ctx = ShimContext(backend=backend, backend_model="shim:model", shim_config=backend.responses_shim)
    shim = ResponsesShim(ResponsesStateStore(), ResponsesToolRegistry(), ResponsesJobRegistry())
    stream = await shim.handle(
        ctx,
        _FakeClientWithReasoningFirstStream(),
        {"model": "shim-model", "input": "hi"},
        stream=True,
    )
    raw_events = []
    async for chunk in stream:
        data = chunk.decode("utf-8").strip()
        if not data.startswith("data: "):
            continue
        payload = data[6:]
        if payload == "[DONE]":
            continue
        raw_events.append(json.loads(payload))

    adapter = TypeAdapter(ResponseStreamEvent)
    state = ResponseStreamState(input_tools=[], text_format=omit)
    for payload in raw_events:
        state.handle_event(adapter.validate_python(payload))

    final = state._completed_response
    assert final is not None
    assert final.output[0].content[0].text == "Hello world"
    assert final.usage is not None
    assert final.usage.input_tokens == 4
    assert final.usage.output_tokens == 2
    assert final.usage.total_tokens == 6


@pytest.mark.asyncio
async def test_responses_shim_stream_preserves_tool_calls_from_chat_chunks():
    class _FakeClientWithToolCallStream:
        async def chat(self, model: str, payload, stream: bool = False):
            if not stream:
                raise AssertionError("stream flag expected in test")

            async def _generator():
                chunks = [
                    {
                        "choices": [
                            {
                                "delta": {
                                    "content": "\n\n",
                                    "role": "assistant",
                                }
                            }
                        ]
                    },
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": "call_123",
                                            "type": "function",
                                            "function": {"name": "exec_command", "arguments": "{\"cmd\":\"echo"},
                                        }
                                    ]
                                }
                            }
                        ]
                    },
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "function": {"arguments": " hi\"}"},
                                        }
                                    ]
                                }
                            }
                        ]
                    },
                    {
                        "choices": [],
                        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
                    },
                ]
                for chunk in chunks:
                    yield b"data: " + json.dumps(chunk).encode("utf-8") + b"\n\n"
                yield b"data: [DONE]\n\n"

            return _generator()

    backend = BackendDefinition(
        type="shim",
        name="shim-backend",
        base_url="http://shim",
        supports=BackendSupports(chat=["shim:model"]),
        responses_shim=ResponsesShimBackendConfig(enabled=True, operation="chat"),
    )
    ctx = ShimContext(backend=backend, backend_model="shim:model", shim_config=backend.responses_shim)
    shim = ResponsesShim(ResponsesStateStore(), ResponsesToolRegistry(), ResponsesJobRegistry())
    stream = await shim.handle(
        ctx,
        _FakeClientWithToolCallStream(),
        {"model": "shim-model", "input": "hi"},
        stream=True,
    )
    raw_events = []
    async for chunk in stream:
        data = chunk.decode("utf-8").strip()
        if not data.startswith("data: "):
            continue
        payload = data[6:]
        if payload == "[DONE]":
            continue
        raw_events.append(json.loads(payload))

    adapter = TypeAdapter(ResponseStreamEvent)
    state = ResponseStreamState(input_tools=[], text_format=omit)
    for payload in raw_events:
        state.handle_event(adapter.validate_python(payload))

    final = state._completed_response
    assert final is not None
    assert final.output[0].type == "message"
    assert final.output[0].content[0].text == "\n\n"
    assert final.output[1].type == "function_call"
    assert final.output[1].name == "exec_command"
    assert final.output[1].arguments == '{"cmd":"echo hi"}'
    assert final.usage is not None
    assert final.usage.input_tokens == 5
    assert final.usage.output_tokens == 3
    assert final.usage.total_tokens == 8
