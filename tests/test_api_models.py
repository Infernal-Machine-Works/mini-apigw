import json
from types import SimpleNamespace

import pytest

from fastapi import HTTPException

from gateway.api import v1
from gateway.auth import AuthError
from gateway.policies import PolicyDecision


class _FakeRouter:
    def __init__(self, models, aliases=None):
        self._models = list(models)
        self._aliases = dict(aliases or {})

    def build_models_payload(self):
        return {"data": list(self._models)}

    def aliases(self):
        return dict(self._aliases)

    def expand_alias(self, alias):
        return self._aliases.get(alias, alias)


class _FakeRuntime:
    def __init__(self, *, models, allowed_models, api_key="good", aliases=None):
        self._router = _FakeRouter(models, aliases=aliases)
        self._allowed = set(allowed_models)
        self._api_key = api_key

    def router(self):
        return self._router

    async def authenticate(self, api_key):
        if api_key != self._api_key:
            raise AuthError("Unknown API key")
        return SimpleNamespace(app=SimpleNamespace(app_id="app"))

    async def check_policy(self, app, model):
        return PolicyDecision(allowed=model in self._allowed)


@pytest.mark.asyncio
async def test_models_endpoint_filters_by_policy():
    models = [
        {"id": "allowed", "object": "model", "owned_by": "backend"},
        {"id": "denied", "object": "model", "owned_by": "backend"},
    ]
    runtime = _FakeRuntime(models=models, allowed_models={"allowed"})

    response = await v1.models(authorization="Bearer good", runtime=runtime)

    payload = json.loads(response.body)
    assert payload == {"data": [models[0]]}


@pytest.mark.asyncio
async def test_models_endpoint_includes_allowed_alias():
    models = [{"id": "canonical", "object": "model", "owned_by": "backend"}]
    runtime = _FakeRuntime(
        models=models,
        allowed_models={"alias"},
        aliases={"alias": "canonical"},
    )

    response = await v1.models(authorization="Bearer good", runtime=runtime)

    payload = json.loads(response.body)
    assert payload == {"data": [{"id": "alias", "object": "model", "owned_by": "backend"}]}


@pytest.mark.asyncio
async def test_models_endpoint_requires_authorization():
    runtime = _FakeRuntime(models=[], allowed_models=set())

    with pytest.raises(HTTPException) as exc_info:
        await v1.models(authorization=None, runtime=runtime)

    assert exc_info.value.status_code == 401
