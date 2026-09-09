from gateway.config import BackendConfig, BackendDefinition, BackendSupports, ResponsesShimBackendConfig
from gateway.routing import ModelRouter


def test_alias_resolution():
    supports = BackendSupports(chat=["openai:gpt-4o-mini"])
    backend = BackendDefinition(type="openai", name="openai", base_url="https://api", supports=supports)
    config = BackendConfig(aliases={"gpt-4o-mini": "openai:gpt-4o-mini"}, sequence_groups={}, backends=[backend])
    router = ModelRouter(config)
    candidates = router.candidates("gpt-4o-mini", "chat")
    assert candidates[0].backend.name == "openai"

def test_responses_operation_resolves_backend():
    supports = BackendSupports(responses=["openai:gpt-4o"], chat=["openai:gpt-4o"])
    backend = BackendDefinition(type="openai", name="openai", base_url="https://api", supports=supports)
    config = BackendConfig(aliases={}, sequence_groups={}, backends=[backend])
    router = ModelRouter(config)
    candidates = router.candidates("openai:gpt-4o", "responses")
    assert candidates[0].backend.name == "openai"


def test_responses_operation_falls_back_to_shim():
    supports = BackendSupports(chat=["ollama:gpt-oss"], responses=[])
    shim_cfg = ResponsesShimBackendConfig(enabled=True, operation="chat")
    backend = BackendDefinition(
        type="ollama",
        name="ollama",
        base_url="http://localhost",
        supports=supports,
        responses_shim=shim_cfg,
    )
    config = BackendConfig(aliases={"gpt-oss": "ollama:gpt-oss"}, sequence_groups={}, backends=[backend])
    router = ModelRouter(config)
    candidates = router.candidates("gpt-oss", "responses")
    assert candidates[0].backend.name == "ollama"
    assert candidates[0].shim_operation == "chat"


def test_responses_operation_can_fall_back_to_responses_shim_mode():
    supports = BackendSupports(chat=["glm-5.2-744b-preview"], responses=[])
    shim_cfg = ResponsesShimBackendConfig(enabled=True, operation="responses")
    backend = BackendDefinition(
        type="openai",
        name="openai-fallback",
        base_url="http://localhost",
        supports=supports,
        responses_shim=shim_cfg,
    )
    config = BackendConfig(aliases={"glm-5.2": "glm-5.2-744b-preview"}, sequence_groups={}, backends=[backend])
    router = ModelRouter(config)
    candidates = router.candidates("glm-5.2", "responses")
    assert candidates[0].backend.name == "openai-fallback"
    assert candidates[0].shim_operation == "responses"


def test_responses_shim_mode_does_not_capture_unrelated_models_from_other_backends():
    tuwien = BackendDefinition(
        type="openai",
        name="tuwien",
        base_url="https://aqueduct",
        supports=BackendSupports(chat=["glm-5.2-744b-preview"], responses=[]),
        responses_shim=ResponsesShimBackendConfig(enabled=True, operation="responses"),
    )
    ollama = BackendDefinition(
        type="ollama",
        name="ollama-local",
        base_url="http://localhost",
        supports=BackendSupports(chat=["llama3.2:latest"], responses=[]),
        responses_shim=ResponsesShimBackendConfig(enabled=True, operation="chat"),
    )
    config = BackendConfig(
        aliases={"glm-5.2": "glm-5.2-744b-preview", "llama3.2": "llama3.2:latest"},
        sequence_groups={},
        backends=[tuwien, ollama],
    )
    router = ModelRouter(config)
    candidates = router.candidates("llama3.2", "responses")
    assert candidates[0].backend.name == "ollama-local"
    assert candidates[0].shim_operation == "chat"
