from __future__ import annotations

from typing import Any, AsyncIterator, Dict
import asyncio
import json
import httpx

from openai import OpenAI

from .base import BackendClient, BackendResult, BackendUsage

class OpenAIBackend(BackendClient):
    def __init__(self, definition):
        super().__init__(definition)
        if not definition.api_key:
            raise ValueError("OpenAI backend requires api_key")
        client_kwargs = {
            "api_key": definition.api_key,
            "base_url": definition.base_url,
        }
        if definition.request_timeout_s is not None:
            client_kwargs["timeout"] = definition.request_timeout_s
            long_timeout_client = httpx.Client(
                timeout=httpx.Timeout(
                    timeout=definition.request_timeout_s,
                    connect=definition.request_timeout_s,
                    read=definition.request_timeout_s,
                    write=definition.request_timeout_s,
                    pool=definition.request_timeout_s,
                ),
                headers=dict(definition.extra_headers or {}) or None,
            )
            client_kwargs["max_retries"] = 0
            client_kwargs["http_client"] = long_timeout_client
        if definition.extra_headers:
            client_kwargs["default_headers"] = dict(definition.extra_headers)
        self._client = OpenAI(**client_kwargs)

    async def chat(self, model: str, payload: Dict[str, Any], stream: bool = False):
        request_payload = dict(payload)
        request_payload.pop("model", None)
        request_payload.pop("stream", None)

        if stream:
            return await self._stream_chat(model, request_payload)

        def _call():
            return self._client.chat.completions.create(model=model, stream=False, **request_payload)

        response = await self._asyncify(_call)
        body = response.model_dump()
        usage = BackendUsage(
            input_tokens=body.get("usage", {}).get("prompt_tokens"),
            output_tokens=body.get("usage", {}).get("completion_tokens"),
            total_tokens=body.get("usage", {}).get("total_tokens"),
        )
        return BackendResult(body=body, usage=usage)

    async def _stream_chat(self, model: str, payload: Dict[str, Any]) -> AsyncIterator[bytes]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Any] = asyncio.Queue()
        sentinel = object()

        def _enqueue(value: Any) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, value)

        def _producer() -> None:
            try:
                stream = self._client.chat.completions.create(model=model, stream=True, **payload)
                for chunk in stream:
                    _enqueue(chunk.model_dump())
            except Exception as exc:  # pragma: no cover - depends on network runtime
                _enqueue(exc)
            finally:
                _enqueue(sentinel)

        loop.run_in_executor(None, _producer)

        async def _generator() -> AsyncIterator[bytes]:
            while True:
                item = await queue.get()
                if item is sentinel:
                    break
                if isinstance(item, Exception):
                    raise item
                data = json.dumps(item).encode("utf-8")
                yield b"data: " + data + b"\n\n"
            yield b"data: [DONE]\n\n"

        return _generator()


    async def responses(self, model: str, payload: Dict[str, Any], stream: bool = False):
        request_payload = dict(payload)
        request_payload.pop("model", None)
        request_payload.pop("stream", None)

        if stream:
            return await self._stream_responses(model, request_payload)

        def _call():
            return self._client.responses.create(model=model, **request_payload)

        response = await self._asyncify(_call)
        body = response.model_dump()
        usage_data = body.get("usage") or {}
        usage = BackendUsage(
            input_tokens=usage_data.get("input_tokens"),
            output_tokens=usage_data.get("output_tokens"),
            total_tokens=usage_data.get("total_tokens"),
        )
        return BackendResult(body=body, usage=usage)

    async def _stream_responses(self, model: str, payload: Dict[str, Any]) -> AsyncIterator[bytes]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Any] = asyncio.Queue()
        sentinel = object()

        def _enqueue(value: Any) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, value)

        def _producer() -> None:
            try:
                with self._client.responses.with_streaming_response.create(
                    model=model, stream=True, **payload
                ) as response:
                    buffer: list[str] = []
                    for line in response.iter_lines():
                        if line is None:
                            continue
                        buffer.append(line)
                        if line == "":
                            chunk = "\n".join(buffer) + "\n"
                            buffer.clear()
                            _enqueue(chunk.encode("utf-8"))
                    if buffer:
                        chunk = "\n".join(buffer) + "\n"
                        _enqueue(chunk.encode("utf-8"))
            except Exception as exc:  # pragma: no cover - depends on network runtime
                _enqueue(exc)
            finally:
                _enqueue(sentinel)

        loop.run_in_executor(None, _producer)

        async def _generator() -> AsyncIterator[bytes]:
            while True:
                item = await queue.get()
                if item is sentinel:
                    break
                if isinstance(item, Exception):
                    raise item
                if isinstance(item, (bytes, bytearray)):
                    yield bytes(item)
                else:
                    yield str(item).encode("utf-8")

        return _generator()

    async def completions(self, model: str, payload: Dict[str, Any]) -> BackendResult:
        request_payload = dict(payload)
        request_payload.pop("model", None)
        response = await self._asyncify(lambda: self._client.completions.create(model=model, **request_payload))
        body = response.model_dump()
        usage = BackendUsage(
            input_tokens=body.get("usage", {}).get("prompt_tokens"),
            output_tokens=body.get("usage", {}).get("completion_tokens"),
            total_tokens=body.get("usage", {}).get("total_tokens"),
        )
        return BackendResult(body=body, usage=usage)

    async def embeddings(self, model: str, payload: Dict[str, Any]) -> BackendResult:
        request_payload = dict(payload)
        request_payload.pop("model", None)
        response = await self._asyncify(lambda: self._client.embeddings.create(model=model, **request_payload))
        return BackendResult(body=response.model_dump())

    async def images(self, model: str, payload: Dict[str, Any]) -> BackendResult:
        request_payload = dict(payload)
        request_payload.pop("model", None)
        response = await self._asyncify(lambda: self._client.images.generate(model=model, **request_payload))
        return BackendResult(body=response.model_dump())

    async def models(self) -> Dict[str, Any]:
        response = await self._asyncify(lambda: self._client.models.list())
        return response.model_dump()


__all__ = ["OpenAIBackend"]
