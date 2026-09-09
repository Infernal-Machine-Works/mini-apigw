from __future__ import annotations

from typing import Any, AsyncIterator, Dict
import asyncio
import json
import httpx

from openai import APIStatusError, OpenAI

from .base import BackendClient, BackendResult, BackendUpstreamError, BackendUsage

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
        request_payload = self._normalize_chat_payload(payload)

        if stream:
            return await self._stream_chat(model, request_payload)

        def _call():
            try:
                return self._client.chat.completions.create(model=model, stream=False, **request_payload)
            except APIStatusError as exc:
                raise self._upstream_error(exc) from exc

        response = await self._asyncify(_call)
        body = response.model_dump()
        usage = BackendUsage(
            input_tokens=body.get("usage", {}).get("prompt_tokens"),
            output_tokens=body.get("usage", {}).get("completion_tokens"),
            total_tokens=body.get("usage", {}).get("total_tokens"),
        )
        return BackendResult(body=body, usage=usage)

    async def _stream_chat(self, model: str, payload: Dict[str, Any]) -> AsyncIterator[bytes]:
        def _open_stream():
            try:
                return self._client.chat.completions.create(model=model, stream=True, **payload)
            except APIStatusError as exc:
                raise self._upstream_error(exc) from exc

        stream = await self._asyncify(_open_stream)
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Any] = asyncio.Queue()
        sentinel = object()

        def _enqueue(value: Any) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, value)

        def _producer() -> None:
            try:
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

    def _normalize_chat_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        request_payload = dict(payload)
        request_payload.pop("model", None)
        request_payload.pop("stream", None)
        if self.definition.strip_reasoning_effort:
            request_payload.pop("reasoning_effort", None)

        messages = request_payload.get("messages")
        if not isinstance(messages, list):
            return request_payload
        if not (
            self.definition.reorder_system_messages_to_front
            or self.definition.coalesce_system_messages
        ):
            return request_payload

        normalized_messages = list(messages)
        if self.definition.reorder_system_messages_to_front:
            system_messages = [message for message in normalized_messages if self._is_system_message(message)]
            non_system_messages = [message for message in normalized_messages if not self._is_system_message(message)]
            normalized_messages = system_messages + non_system_messages
        if self.definition.coalesce_system_messages:
            normalized_messages = self._coalesce_leading_system_messages(normalized_messages)
        request_payload["messages"] = normalized_messages
        return request_payload

    @staticmethod
    def _is_system_message(message: Any) -> bool:
        return isinstance(message, dict) and message.get("role") == "system"

    @classmethod
    def _coalesce_leading_system_messages(cls, messages: list[Any]) -> list[Any]:
        leading_system_messages: list[Dict[str, Any]] = []
        for message in messages:
            if not cls._is_system_message(message):
                break
            leading_system_messages.append(message)
        if len(leading_system_messages) < 2:
            return messages
        contents = [cls._system_message_text(message.get("content")) for message in leading_system_messages]
        return [{"role": "system", "content": "\n\n".join(contents)}] + messages[len(leading_system_messages) :]

    @staticmethod
    def _system_message_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts: list[str] = []
            for part in content:
                if isinstance(part, str):
                    text_parts.append(part)
                elif isinstance(part, dict) and isinstance(part.get("text"), str):
                    text_parts.append(part["text"])
            return "".join(text_parts)
        return ""

    @staticmethod
    def _upstream_error(exc: APIStatusError) -> BackendUpstreamError:
        body = getattr(exc, "body", None)
        if isinstance(body, (dict, list)):
            preview = json.dumps(body, ensure_ascii=True, separators=(",", ":"))
        elif isinstance(body, str):
            preview = body
        else:
            preview = "Upstream request failed"
        preview = " ".join(preview.split())[:512]
        return BackendUpstreamError(exc.status_code, preview or "Upstream request failed")


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
