"""Tolerant decoding of Qwen's XML-like tool protocol, not arbitrary XML.

Parameter bodies are opaque text. In particular, XML in code arguments must
not be interpreted as nested function/parameter elements or entity references.
Only protocol delimiters are repaired; missing argument data is never invented.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

from .base import BackendUpstreamError


_NAME = r'([^\s<>="\']+)'
_FUNCTION = re.compile(r"<function=" + _NAME + r">")
_PARAMETER = re.compile(r"<parameter=" + _NAME + r">")
_FUNCTION_END = re.compile(r"</function>\s*(?:</parameter>\s*)*$")
_PARAMETER_END = re.compile(r"</parameter>\s*$")
_PARAMETER_TOKEN = re.compile(r"<parameter=" + _NAME + r">|</parameter>")
_AFTER_PARAMETER = re.compile(r"\s*(?:<parameter=|</function>|</tool_call>|<tool_call>|$)")


def _protect_parameters(text: str, names: set[str]) -> tuple[str, dict[str, str]]:
    """Hide explicitly delimited values before interpreting wrapper tags.

    Balance named parameter tags so an XML tool-call example inside a code
    argument stays in that argument, including its own parameter tags. An
    unmatched opening is left for the missing-delimiter repair in _call.
    """
    stack = []
    spans = []
    for token in _PARAMETER_TOKEN.finditer(text):
        if token[1] is not None:
            stack.append((token.end(), token[1], token.start()))
        elif stack and (_AFTER_PARAMETER.match(text, token.end())
                        or (len(stack) > 1 and text[token.end():].lstrip().startswith("</parameter>"))):
            start, name, opening = stack.pop()
            # A matched outer value supersedes any matched inner values.
            while spans and spans[-1][0] >= start:
                spans.pop()
            spans.append((start, token.start(), name, opening))
    # An unmatched apparent parameter opening can itself be literal data:
    # e.g. code containing '<parameter=example>' before the real closing tag.
    # Keep declared, distinct sibling parameters available for missing-close
    # recovery; quoted, undeclared, or repeated openings belong to the value.
    for start, name, opening in reversed(stack):
        nested = [span for span in spans if span[0] >= start]
        if len(nested) != 1:
            continue
        inner = nested[0]
        quoted = inner[3] > 0 and text[inner[3] - 1] in "\"'`"
        if (inner[2] in names and inner[2] != name and not quoted):
            continue
        if re.search(r"</(?:function|tool_call)>", text[start:inner[3]]):
            continue
        spans[spans.index(inner)] = (start, inner[1], name, opening)
    # Use a delimiter absent from the original response, so literal content
    # can never impersonate a protected value.
    delimiter = "\x00"
    while delimiter in text:
        delimiter += "\x00"
    values = {}
    parts = []
    position = 0
    for index, (start, end, _, _) in enumerate(spans):
        placeholder = f"{delimiter}{index}{delimiter}"
        values[placeholder] = text[start:end]
        parts.extend((text[position:start], placeholder))
        position = end
    parts.append(text[position:])
    return "".join(parts), values


def _restore_values(text: str, values: dict[str, str]) -> str:
    # The delimiter was chosen to be absent from every original value.
    if not values:
        return text
    key = next(iter(values))
    delimiter = key[:len(key) - len(key.lstrip("\x00"))]
    return re.sub(delimiter + r"[0-9]+" + delimiter,
                  lambda match: values.get(match[0], match[0]), text)


def _fail(reason: str):
    # Diagnostics deliberately exclude model-generated text and arguments.
    raise BackendUpstreamError(502, "Cannot repair Ollama tool-call XML: " + reason)


def _types(schema: Any, root: dict, seen: tuple = ()) -> set[str]:
    if not isinstance(schema, dict):
        return set()
    ref = schema.get("$ref")
    if ref is not None:
        if not isinstance(ref, str) or not ref.startswith("#/") or ref in seen:
            _fail("unsupported or cyclic parameter schema reference")
        target = root
        for part in ref[2:].split("/"):
            if not isinstance(target, dict):
                _fail("unresolved parameter schema reference")
            target = target.get(part.replace("~1", "/").replace("~0", "~"))
        if target is None:
            _fail("unresolved parameter schema reference")
        return _types(target, root, (*seen, ref))
    result = schema.get("type", [])
    result = {result} if isinstance(result, str) else set(result)
    for key in ("anyOf", "oneOf", "allOf"):
        for variant in schema.get(key, []):
            result.update(_types(variant, root, seen))
    return result


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            _fail("duplicate JSON argument key")
        result[key] = value
    return result


def _reject_constant(value):
    _fail("non-finite JSON argument")


def _finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        _fail("non-finite JSON argument")
    return number


def _value(raw: str, schema: Any, root: dict):
    # Qwen's template adds one framing newline at each end, not arbitrary
    # whitespace. Preserve indentation, blank lines, ampersands and entities.
    raw = raw.removeprefix("\n").removesuffix("\n")
    types = _types(schema, root)
    if not types or types == {"string"}:
        return raw
    try:
        value = json.loads(raw, object_pairs_hook=_unique_object,
                           parse_constant=_reject_constant, parse_float=_finite_float)
    except (ValueError, RecursionError):
        if "string" in types:
            return raw
        _fail("argument does not match its declared type")
    valid = (
        (value is None and "null" in types)
        or (type(value) is bool and "boolean" in types)
        or (type(value) is int and bool(types & {"integer", "number"}))
        or (type(value) is float and "number" in types and math.isfinite(value))
        or (isinstance(value, list) and "array" in types)
        or (isinstance(value, dict) and "object" in types)
    )
    if valid:
        return value
    if "string" in types:
        return raw
    _fail("argument does not match its declared type")


def _call(block: str, tools: dict, values: dict[str, str], normal_end: bool = False) -> dict:
    block = block.strip()
    function = _FUNCTION.match(block)
    if function is None or function[1] not in tools:
        _fail("missing or unknown function name")
    name = function[1]
    tail = block[function.end():]
    end = _FUNCTION_END.search(tail)
    if end:
        tail = tail[:end.start()]
    elif not normal_end and not _PARAMETER_END.search(tail) and tail.strip():
        _fail("missing function and parameter terminators")
    # A missing </function> is recoverable inside a complete <tool_call>.
    # Named opening tags inside values are ambiguous; do not guess nesting.
    if _FUNCTION.search(tail):
        _fail("ambiguous nested function")
    parameters = list(_PARAMETER.finditer(tail))
    prefix = tail[:parameters[0].start()] if parameters else tail
    if re.sub(r"</parameter>\s*", "", prefix).strip():
        _fail("unexpected text outside parameters")
    schema = tools[name].get("parameters") or {}
    properties = schema.get("properties") or {}
    arguments = {}
    for index, parameter in enumerate(parameters):
        key = parameter[1]
        if key in arguments:
            _fail("duplicate parameter")
        if key not in properties and schema.get("additionalProperties") is False:
            _fail("unknown parameter")
        limit = parameters[index + 1].start() if index + 1 < len(parameters) else len(tail)
        raw = tail[parameter.end():limit]
        closing = _PARAMETER_END.search(raw)
        if closing:
            raw = raw[:closing.start()]
        raw = _restore_values(raw, values)
        # Missing </parameter> is bounded by the next named parameter or the
        # function terminator. Literal <function> and other XML remain data.
        arguments[key] = _value(raw, properties.get(key, schema.get("additionalProperties", {})), schema)
    if any(key not in arguments for key in schema.get("required", [])):
        _fail("required parameter is missing")
    return {"function": {"name": name, "arguments": arguments}}


def parse_qwen_response(body: dict, tools: list, prompt: str) -> dict:
    text = body.get("response")
    if not isinstance(text, str) or body.get("done") is not True:
        _fail("incomplete raw response")
    thinking = body.get("thinking") or ""
    # A raw generation may include the thinking delimiters, or the opening
    # delimiter may already be in the rendered assistant prefill.
    implicit_think = prompt.rstrip().endswith("<think>") and not thinking
    if text.lstrip().startswith("<think>") or implicit_think:
        if not implicit_think:
            text = text.lstrip().removeprefix("<think>")
        elif text.lstrip().startswith("<think>"):
            text = text.lstrip().removeprefix("<think>")
        if "</think>" not in text:
            _fail("unfinished thinking block")
        thought, text = text.split("</think>", 1)
        thinking += thought
    text = re.sub(r"(?:<\|im_end\|>|<\|endoftext\|>)\s*$", "", text)
    names = {name for tool in tools
             for name in (tool.get("function", {}).get("parameters") or {}).get("properties", {})}
    text, values = _protect_parameters(text, names)
    # Duplicate wrappers add no argument data. Values have already been
    # protected, so these substitutions cannot rewrite code arguments.
    text = re.sub(r"(?:<tool_call>\s*){2,}", "<tool_call>", text)
    text = re.sub(r"(?:</tool_call>\s*){2,}", "</tool_call>", text)
    available = {}
    for tool in tools:
        function = tool.get("function", {})
        name = function.get("name")
        if not name or name in available:
            _fail("invalid or duplicate tool definition")
        available[name] = function
    calls = []
    content = []
    position = 0
    while True:
        start = text.find("<tool_call>", position)
        if start < 0:
            remainder = text[position:]
            if ("<tool_call" in remainder or "</tool_call>" in remainder
                    or re.match(r"\s*<(?:function|parameter)=", remainder)):
                _fail("unframed tool call")
            content.append(remainder)
            break
        if body.get("done_reason") == "length":
            _fail("generation hit its token limit")
        content.append(text[position:start])
        begin = start + len("<tool_call>")
        end = text.find("</tool_call>", begin)
        next_start = text.find("<tool_call>", begin)
        missing_close = False
        if next_start >= 0 and (end < 0 or next_start < end):
            previous = text[begin:next_start]
            if not (_FUNCTION_END.search(previous) or _PARAMETER_END.search(previous)):
                _fail("ambiguous nested tool call")
            # A completed function or parameter gives us a sibling boundary;
            # close the preceding function/wrapper implicitly as needed.
            end = next_start
            missing_close = True
        if end < 0:
            # Explicit inner closes, or a normal stop (not a token limit), let
            # us close the remaining structure without inventing value bytes.
            if not (_FUNCTION_END.search(text[begin:]) or _PARAMETER_END.search(text[begin:])
                    or body.get("done_reason") == "stop"):
                _fail("unterminated tool call")
            end = len(text)
        block = text[begin:end]
        if "<tool_call>" in block:
            _fail("ambiguous nested tool call")
        calls.append(_call(block, available, values, normal_end=(
            end == len(text) and body.get("done_reason") == "stop")))
        position = end if missing_close else min(end + len("</tool_call>"), len(text))
    message = {"role": "assistant", "content": _restore_values("".join(content), values).strip()}
    if thinking:
        message["thinking"] = thinking
    if calls:
        message["tool_calls"] = calls
    return message
