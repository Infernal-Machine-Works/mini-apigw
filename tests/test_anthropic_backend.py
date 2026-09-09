"""Unit tests covering Anthropic backend helpers."""
from __future__ import annotations

from gateway.backends.anthropic import AnthropicBackend
from gateway.config import BackendDefinition


def _make_backend() -> AnthropicBackend:
    definition = BackendDefinition(
        type="anthropic",
        name="anthropic-test",
        base_url="https://example.com",
        api_key="key",
    )
    return AnthropicBackend(definition)


def test_finalize_stream_tool_inputs_overrides_placeholder_input():
    backend = _make_backend()
    message = {
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_123",
                "name": "add_todo",
                "input": {},
                "_input_json_buffer": '{"item": "Write report"}',
            }
        ]
    }

    backend._finalize_stream_tool_inputs(message)

    block = message["content"][0]
    assert block["input"] == {"item": "Write report"}


def test_finalize_stream_tool_inputs_handles_invalid_json():
    backend = _make_backend()
    message = {
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_999",
                "name": "add_todo",
                "input": {"existing": True},
                "_input_json_buffer": '{"item": "test"',  # malformed JSON
            }
        ]
    }

    backend._finalize_stream_tool_inputs(message)

    block = message["content"][0]
    assert block["input"] == {}
