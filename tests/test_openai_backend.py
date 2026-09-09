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
