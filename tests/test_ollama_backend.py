import asyncio
import json
from copy import deepcopy

import httpx
import pytest

from gateway.backends.base import BackendUpstreamError
from gateway.backends.ollama import OllamaBackend
from gateway.config import BackendDefinition
from tests.test_api_responses import DummyRuntime, create_test_app


def make_backend():
    return OllamaBackend(BackendDefinition(
        type="ollama", name="local", base_url="http://ollama.test:8182", request_timeout_s=123,
    ))


@pytest.fixture
def backend():
    instance = make_backend()
    yield instance
    instance._client._client.close()


def mock_http(monkeypatch, handler):
    original_client = httpx.AsyncClient

    def client(**kwargs):
        assert kwargs["timeout"] == 123
        return original_client(**kwargs, transport=httpx.MockTransport(handler))

    monkeypatch.setattr("gateway.backends.ollama.httpx.AsyncClient", client)


def test_tool_round_trip_preserves_wire_history_and_parallel_results(backend, monkeypatch):
    requests = []
    arguments = {"city": "Wien", "nested": {"flag": False, "count": 0, "items": [1, None]}, "text": "<function>ä</parameter>"}

    def handler(request):
        assert request.url == "http://ollama.test:8182/api/chat"
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 1:
            return httpx.Response(200, json={
                "model": "qwen", "message": {
                    "role": "assistant", "content": "Checking both.", "thinking": "Compare cities.",
                    "tool_calls": [
                        {"function": {"name": "weather", "arguments": arguments}},
                        {"function": {"name": "clock", "arguments": {}}},
                    ],
                }, "done": True, "prompt_eval_count": 11, "eval_count": 7,
            })
        return httpx.Response(200, json={"model": "qwen", "message": {"role": "assistant", "content": "Done."}, "done": True})

    mock_http(monkeypatch, handler)
    payload = {
        "messages": [{"role": "user", "content": "Check both"}],
        "tools": [{"type": "function", "function": {"name": name, "parameters": {"type": "object"}}} for name in ("weather", "clock")],
        "options": {"num_ctx": 8192}, "temperature": 0, "think": True, "keep_alive": "10m",
    }
    first = asyncio.run(backend.chat("qwen", payload))
    choice = first.body["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    calls = choice["message"]["tool_calls"]
    assert len({call["id"] for call in calls}) == 2
    assert json.loads(calls[0]["function"]["arguments"]) == arguments
    assert first.usage.input_tokens == 11
    continuation = deepcopy(payload)
    continuation["messages"] += [
        choice["message"],
        {"role": "tool", "tool_call_id": calls[1]["id"], "content": "12:00"},
        {"role": "tool", "tool_call_id": calls[0]["id"], "content": "Sunny"},
    ]
    original = deepcopy(continuation)
    second = asyncio.run(backend.chat("qwen", continuation, stream=True))
    assert continuation == original
    assert second.body["choices"][0]["message"]["content"] == "Done."
    wire = requests[1]
    assert wire["stream"] is False
    assert wire["tools"] == payload["tools"]
    assert wire["options"] == {"num_ctx": 8192, "temperature": 0}
    assert wire["think"] is True
    assert wire["keep_alive"] == "10m"
    assert wire["messages"][1] == {
        "role": "assistant", "content": "Checking both.", "thinking": "Compare cities.",
        "tool_calls": [
            {"function": {"name": "weather", "arguments": arguments}},
            {"function": {"name": "clock", "arguments": {}}},
        ],
    }
    assert wire["messages"][2:] == [
        {"role": "tool", "tool_name": "clock", "content": "12:00"},
        {"role": "tool", "tool_name": "weather", "content": "Sunny"},
    ]

    async def chunks():
        return [chunk async for chunk in first.as_sse("chatcmpl-test")]

    stream = asyncio.run(chunks())
    assert json.loads(stream[0].decode().removeprefix("data: "))["choices"][0]["finish_reason"] == "tool_calls"
    assert stream[-1] == b"data: [DONE]\n\n"


def test_legacy_and_native_history_and_multimodal_content(backend):
    payload = {"functions": [{"name": "lookup", "parameters": {"type": "object"}}], "messages": [
        {"role": "user", "content": [{"type": "text", "text": "Look"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,YQ=="}}]},
        {"role": "assistant", "content": None, "function_call": {"name": "lookup", "arguments": "{}"}},
        {"role": "function", "name": "lookup", "content": "found"},
        {"role": "assistant", "content": "Again", "tool_calls": [{"function": {"name": "lookup", "arguments": {}}}]},
        {"role": "tool", "tool_name": "lookup", "content": "found again"},
    ]}
    original = deepcopy(payload)
    native = backend._prepare_chat_payload("model", payload, False)
    assert payload == original
    assert native["tools"] == [{"type": "function", "function": payload["functions"][0]}]
    assert "functions" not in native
    assert native["messages"][0] == {"role": "user", "content": "Look", "images": ["YQ=="]}
    assert native["messages"][1]["tool_calls"][0]["function"]["arguments"] == {}
    assert native["messages"][2] == {"role": "tool", "tool_name": "lookup", "content": "found"}
    assert native["messages"][4]["tool_name"] == "lookup"


@pytest.mark.parametrize("arguments", ['{"broken":', '[]', 'null', '1', '"text"', [], None, True])
def test_invalid_arguments_fail_before_transport(backend, arguments):
    payload = {"messages": [{"role": "assistant", "tool_calls": [
        {"function": {"name": "lookup", "arguments": arguments}},
    ]}]}
    with pytest.raises(BackendUpstreamError) as exc:
        asyncio.run(backend.chat("model", payload))
    assert exc.value.status_code == 400
    assert exc.value.body_preview == "Tool-call arguments must be a JSON object"


@pytest.mark.parametrize("message", [
    {"role": "tool", "tool_call_id": "missing", "content": "result"},
    {"role": "tool", "tool_call_id": "a", "name": "wrong", "content": "result"},
    {"role": "tool", "tool_call_id": [], "content": "result"},
    {"role": "assistant", "tool_calls": "broken"},
    {"role": "assistant", "tool_calls": [{"type": "custom", "custom": {}}]},
    {"role": "assistant", "tool_calls": [{"function": {"arguments": {}}}]},
])
def test_invalid_tool_history_is_not_silently_dropped(backend, message):
    with pytest.raises(BackendUpstreamError) as exc:
        backend._prepare_chat_payload("model", {"messages": [
            {"role": "assistant", "tool_calls": [{"id": "a", "function": {"name": "lookup", "arguments": {}}}]}, message,
        ]}, False)
    assert exc.value.status_code == 400


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("status,detail", [
    (500, "XML syntax error on line 294: element <function> closed by </parameter>"),
    (404, "model missing"),
])
def test_upstream_errors_are_captured_and_returned_as_http_errors(backend, monkeypatch, caplog, stream, status, detail):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(status, json={"error": detail})

    mock_http(monkeypatch, handler)

    class Runtime(DummyRuntime):
        async def execute_operation(self, *, model, payload, stream, **kwargs):
            return await backend.chat(model, payload, stream)

    with create_test_app(Runtime()) as client:
        response = client.post("/v1/chat/completions", headers={"Authorization": "Bearer test-key"}, json={
            "model": "qwen", "stream": stream, "messages": [{"role": "user", "content": "private prompt"}],
        })
    assert response.status_code == status
    assert response.json() == {"detail": detail}
    assert len(requests) == 1
    assert "backend=local model=qwen" in caplog.text
    assert detail in caplog.text
    assert "private prompt" not in caplog.text


def test_non_json_error_is_bounded(backend, monkeypatch, caplog):
    mock_http(monkeypatch, lambda request: httpx.Response(502, text="bad gateway\n" + "x" * 1000))
    with pytest.raises(BackendUpstreamError) as exc:
        asyncio.run(backend.chat("qwen", {"messages": []}))
    assert exc.value.status_code == 502
    assert len(exc.value.body_preview) == 512
    assert "\n" not in exc.value.body_preview


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("completion", [{"done": False}, {}, {"done": "true"}])
@pytest.mark.parametrize("message", [
    {"role": "assistant", "content": "", "thinking": "/" * 31},
    {"role": "assistant", "content": "private partial answer", "tool_calls": [
        {"function": {"name": "lookup", "arguments": {"private": "partial"}}},
    ]},
])
def test_incomplete_native_response_is_never_reported_as_success(
    backend, monkeypatch, caplog, stream, completion, message,
):
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"model": "qwen", "message": message, **completion})

    mock_http(monkeypatch, handler)

    class Runtime(DummyRuntime):
        async def execute_operation(self, *, model, payload, stream, **kwargs):
            return await backend.chat(model, payload, stream)

    with create_test_app(Runtime()) as client:
        response = client.post("/v1/chat/completions", headers={"Authorization": "Bearer test-key"}, json={
            "model": "qwen", "stream": stream, "tools": TOOLS,
            "messages": [{"role": "user", "content": "private prompt"}],
        })
    assert response.status_code == 502
    assert "without a completion marker" in response.json()["detail"]
    assert "text/event-stream" not in response.headers["content-type"]
    assert len(requests) == 1  # No XML fallback or blind generation retry.
    assert requests[0]["stream"] is False
    assert "Ollama incomplete response" in caplog.text
    assert "private" not in caplog.text + response.text


def test_raw_recovery_also_rejects_incomplete_generation(backend, monkeypatch):
    backend.definition.ollama_tool_call_parser = "qwen_xml"
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return httpx.Response(200, json={"done": False, "_debug_info": {"rendered_template": PROMPT}})
        return httpx.Response(200, json={"done": False, "response": RAW_RESPONSE})

    mock_http(monkeypatch, handler)
    with pytest.raises(BackendUpstreamError, match="without a completion marker"):
        asyncio.run(backend.chat("qwen", {"tools": TOOLS, "messages": []}))
    assert len(requests) == 2


XML_ERROR = "XML syntax error on line 188: element <function> closed by </parameter>"
TOOLS = [{"type": "function", "function": {"name": "lookup", "parameters": {
    "type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"],
}}}]
PROMPT = "Qwen tools: <function=NAME><parameter=NAME>VALUE\n<|im_start|>assistant\n<think>\n"
RAW_RESPONSE = "Inspect. </think>Checking.<tool_call><function=lookup><parameter=text><function>ä & x</parameter></function></parameter></tool_call>"


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("mode,status", [("auto", 200), ("auto", 500), ("qwen_xml", 200)])
@pytest.mark.parametrize("raw_response,expected_text", [
    (RAW_RESPONSE, "<function>ä & x"),
    ('</think><tool_call><function=lookup><parameter=text>print("<tool_call><function=lookup><parameter=text>example</parameter></function></tool_call>")</parameter></function></tool_call>',
     'print("<tool_call><function=lookup><parameter=text>example</parameter></function></tool_call>")'),
    ('</think><tool_call><function=lookup><parameter=text>missing end tags', 'missing end tags'),
])
def test_xml_repair_through_openai_frontend(backend, monkeypatch, stream, mode, status, raw_response, expected_text):
    backend.definition.ollama_tool_call_parser = mode
    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append((request.url.path, body))
        if body.get("_debug_render_only"):
            return httpx.Response(200, json={"_debug_info": {"rendered_template": PROMPT}})
        if request.url.path == "/api/chat":
            return httpx.Response(status, json={"error": XML_ERROR})
        assert request.url.path == "/api/generate"
        assert body["prompt"] == PROMPT
        assert body["raw"] is True
        return httpx.Response(200, json={
            "model": "qwen", "response": raw_response, "done": True,
            "done_reason": "stop", "prompt_eval_count": 15, "eval_count": 9,
        })

    mock_http(monkeypatch, handler)

    class Runtime(DummyRuntime):
        async def execute_operation(self, *, model, payload, stream, **kwargs):
            result = await backend.chat(model, payload, stream)
            _, record, _ = await super().execute_operation(model=model, payload=payload, stream=stream, **kwargs)
            return result, record, "ollama"

    with create_test_app(Runtime()) as client:
        response = client.post("/v1/chat/completions", headers={"Authorization": "Bearer test-key"}, json={
            "model": "qwen", "stream": stream, "tools": TOOLS,
            "messages": [{"role": "user", "content": "Look up"}],
        })
    assert response.status_code == 200
    if stream:
        chunks = [json.loads(line[6:]) for line in response.text.splitlines()
                  if line.startswith("data: ") and line != "data: [DONE]"]
        choice = next(chunk["choices"][0] for chunk in chunks if chunk.get("choices"))
        message = choice["delta"]
        assert "data: [DONE]" in response.text
    else:
        choice = response.json()["choices"][0]
        message = choice["message"]
    assert choice["finish_reason"] == "tool_calls"
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {"text": expected_text}
    assert len(requests) == (3 if mode == "auto" else 2)
    assert requests[-2][1]["tools"] == TOOLS
    assert requests[-2][1]["messages"] == [{"role": "user", "content": "Look up"}]


def test_raw_xml_preserves_history_options_and_input(backend, monkeypatch):
    backend.definition.ollama_tool_call_parser = "qwen_xml"
    requests = []
    payload = {
        "tools": TOOLS, "think": True, "keep_alive": "10m",
        "options": {"seed": 42, "temperature": 0, "num_predict": 1024},
        "messages": [
            {"role": "system", "content": "Original instructions"},
            {"role": "user", "content": "private prompt"},
            {"role": "assistant", "content": "", "thinking": "Reasoning", "tool_calls": [
                {"id": "a", "function": {"name": "lookup", "arguments": '{"text":"<a>&"}'}},
            ]},
            {"role": "tool", "tool_call_id": "a", "content": "private result"},
        ],
    }
    original = deepcopy(payload)

    def handler(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return httpx.Response(200, json={"_debug_info": {"rendered_template": PROMPT}})
        return httpx.Response(200, json={"response": RAW_RESPONSE, "done": True, "prompt_eval_count": 12, "eval_count": 3})

    mock_http(monkeypatch, handler)
    result = asyncio.run(backend.chat("qwen", payload))
    assert payload == original
    assert result.usage.input_tokens == 12
    assert result.usage.output_tokens == 3
    assert result.body["choices"][0]["message"]["thinking"] == "Inspect. "
    assert "response" not in result.body
    expected = backend._prepare_chat_payload("qwen", payload, False)
    assert requests[0] == {**expected, "_debug_render_only": True}
    assert requests[1] == {
        "model": "qwen", "options": payload["options"], "think": True,
        "keep_alive": "10m", "raw": True, "stream": False, "prompt": PROMPT,
    }


@pytest.mark.parametrize("mode,status,detail", [
    ("native", 500, XML_ERROR), ("auto", 400, XML_ERROR),
    ("auto", 500, "CUDA error"), ("auto", 503, "temporarily unavailable"),
])
def test_recovery_is_specific(backend, monkeypatch, mode, status, detail):
    backend.definition.ollama_tool_call_parser = mode
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(status, json={"error": detail})

    mock_http(monkeypatch, handler)
    with pytest.raises(BackendUpstreamError) as exc:
        asyncio.run(backend.chat("qwen", {"tools": TOOLS, "messages": []}))
    assert len(requests) == 1
    assert exc.value.status_code == status


@pytest.mark.parametrize("rendered", [
    {"message": {"content": "debug unsupported"}},
    {"_debug_info": {"rendered_template": "Other template"}},
    {"_debug_info": {"rendered_template": PROMPT, "image_count": 1}},
])
def test_unsupported_raw_render_fails_before_generation(backend, monkeypatch, rendered):
    backend.definition.ollama_tool_call_parser = "qwen_xml"
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=rendered)

    mock_http(monkeypatch, handler)
    with pytest.raises(BackendUpstreamError) as exc:
        asyncio.run(backend.chat("qwen", {"tools": TOOLS, "messages": []}))
    assert len(requests) == 1
    assert exc.value.status_code == 502


def test_raw_failure_does_not_loop_or_log_arguments(backend, monkeypatch, caplog):
    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        if body.get("_debug_render_only"):
            return httpx.Response(200, json={"_debug_info": {"rendered_template": PROMPT}})
        if request.url.path == "/api/chat":
            return httpx.Response(500, json={"error": XML_ERROR})
        return httpx.Response(200, json={"response": "<tool_call>private broken argument", "done": True})

    mock_http(monkeypatch, handler)
    with pytest.raises(BackendUpstreamError) as exc:
        asyncio.run(backend.chat("qwen", {"tools": TOOLS, "messages": []}))
    assert len(requests) == 3
    assert exc.value.status_code == 502
    assert "private" not in caplog.text + exc.value.body_preview
