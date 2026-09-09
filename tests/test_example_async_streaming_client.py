"""Behavioral tests for the async streaming example helpers."""
from __future__ import annotations

import asyncio
from typing import Any

from tests.example_async_chat_client import _append_tool_call_fragment, _stream_chat_completion


class _Chunk:
    def __init__(self, payload):
        self._payload = payload

    def model_dump(self):
        return self._payload


class _FinalResponse:
    def __init__(self, payload):
        self._payload = payload

    def model_dump(self):
        return self._payload


class _DummyStream:
    def __init__(self, chunks, final_response):
        self._chunks = list(chunks)
        self._final_response = final_response

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)

    async def get_final_response(self):
        return self._final_response


class _DummyStreamNoFinal:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)


class _DummyChatCompletions:
    def __init__(self, responses):
        self._responses = list(responses)

    async def create(self, **kwargs):
        if not self._responses:
            raise RuntimeError("No more responses")
        return self._responses.pop(0)


class _DummyChat:
    def __init__(self, responses):
        self.completions = _DummyChatCompletions(responses)


class _DummyAsyncClient:
    def __init__(self, responses):
        self.chat = _DummyChat(responses)


def test_stream_chat_completion_returns_final_payload_and_streams_text():
    chunks = [
        _Chunk(
            {
                "choices": [
                    {
                        "delta": {
                            "content": [{"type": "text", "text": "Hello"}],
                        }
                    }
                ]
            }
        )
    ]
    final_response = _FinalResponse(
        {
            "model": "dummy",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello"}}],
        }
    )
    stream = _DummyStream(chunks, final_response)
    client = _DummyAsyncClient([stream])

    seen_chunks: list[str] = []

    def _on_chunk(text: str) -> None:
        seen_chunks.append(text)

    payload = asyncio.run(
        _stream_chat_completion(
            client,
            model="dummy",
            messages=[{"role": "user", "content": "hi"}],
            response_format=None,
            on_text_chunk=_on_chunk,
        )
    )

    assert payload["choices"][0]["message"]["content"] == "Hello"
    assert seen_chunks == ["Hello"]


def test_stream_chat_completion_fallback_without_final_response():
    chunks = [
        _Chunk(
            {
                "choices": [
                    {
                        "delta": {
                            "content": [{"type": "text", "text": "Partial"}],
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "function": {"name": "add_todo", "arguments": '{"item": "A"}'},
                                    "type": None,
                                }
                            ],
                        },
                        "message": {"tool_calls": [{"index": 0, "function": {"name": "add_todo", "arguments": '{"item": "A"}'}}]},
                        "finish_reason": "tool_calls",
                    }
                ]
            }
        )
    ]
    stream = _DummyStreamNoFinal(chunks)
    client = _DummyAsyncClient([stream])

    payload = asyncio.run(
        _stream_chat_completion(
            client,
            model="dummy",
            messages=[{"role": "user", "content": "hi"}],
            response_format=None,
        )
    )

    message = payload["choices"][0]["message"]
    assert message["content"] == "Partial"
    assert message["tool_calls"][0]["type"] == "function"


def test_append_tool_call_fragment_defaults_type_and_arguments():
    fragments: dict[int, dict[str, Any]] = {}
    fragment = {"function": {"arguments": '{"item": "Task"}'}}
    _append_tool_call_fragment(fragments, fragment)

    assert fragments[0]["type"] == "function"
    assert fragments[0]["function"]["arguments"] == '{"item": "Task"}'
