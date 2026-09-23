import pytest

from gateway.backends.base import BackendUpstreamError
from gateway.backends.ollama_xml import parse_qwen_response


TOOLS = [{"type": "function", "function": {"name": "run", "parameters": {
    "type": "object", "properties": {
        "code": {"type": "string"}, "count": {"type": "integer"},
        "flag": {"type": "boolean"}, "data": {"type": "object"},
        "items": {"type": "array"}, "maybe": {"type": ["null", "string"]},
    }, "required": ["code"], "additionalProperties": False,
}}}]


def parse(text, **kwargs):
    return parse_qwen_response({"response": text, "done": True, **kwargs}, TOOLS, "assistant\n")


def args(message):
    return message["tool_calls"][0]["function"]["arguments"]


@pytest.mark.parametrize("code", [
    '<function>some code</parameter>', '<function>unclosed',
    'if a < b && b > c: print("ä &amp; &#65;")',
    '<root><child>XML</child></root>', '</parameter> literal in code',
    '  indented\n\n  text  ', '', 'null', '123', 'false',
    'print("<tool_call>")', 'print("</tool_call>")',
    '<tool_call>literal XML</tool_call>',
    '<function=run>literal function markup</function>',
    '<tool_call><function=run><parameter=code>example</parameter></function></tool_call>',
    '<parameter=literal>example</parameter>',
    '\x00 literal null character',
    '<parameter=example>', '<parameter=code>',
    'print("<parameter=count>")',
])
def test_parameter_bodies_are_opaque_text(code):
    result = parse(f"<tool_call><function=run><parameter=code>\n{code}\n</parameter></function></tool_call>")
    assert args(result) == {"code": code}


@pytest.mark.parametrize("xml", [
    # Missing parameter close before function close.
    "<tool_call><function=run><parameter=code>x</function></tool_call>",
    # Stray parameter close outside the function (the reported error shape).
    "<tool_call><function=run><parameter=code>x</parameter></function></parameter></tool_call>",
    # Missing function close, but parameter and enclosing call are complete.
    "<tool_call><function=run><parameter=code>x</parameter></tool_call>",
    # Missing enclosing close, but function is complete and generation stopped.
    "<tool_call><function=run><parameter=code>x</parameter></function>",
    # Stray close before the first parameter.
    "<tool_call><function=run></parameter><parameter=code>x</parameter></function></tool_call>",
])
def test_recovers_delimiters_without_changing_values(xml):
    assert args(parse(xml)) == {"code": "x"}


def test_missing_parameter_close_before_next_parameter_and_typed_values():
    result = parse('''<tool_call><function=run>
<parameter=code>
  x & <function>
<parameter=count>0</parameter>
<parameter=flag>false</parameter>
<parameter=data>{"nested":{"items":[0,false,null]},"literal":"<xml>&"}</parameter>
<parameter=items>[1,"ä",null]</parameter>
<parameter=maybe>null</parameter>
</function></tool_call>''')
    assert args(result) == {
        "code": "  x & <function>", "count": 0, "flag": False,
        "data": {"nested": {"items": [0, False, None]}, "literal": "<xml>&"},
        "items": [1, "ä", None], "maybe": None,
    }


def test_parallel_calls_content_and_thinking():
    call = "<tool_call><function=run><parameter=code>x</parameter></function></tool_call>"
    result = parse(f"<think>Do not execute: {call}</think>Before {call} between {call} after<|im_end|>")
    assert len(result["tool_calls"]) == 2
    assert result["thinking"] == f"Do not execute: {call}"
    assert result["content"] == "Before  between  after"


def test_implicit_thinking_prefill_and_plain_response():
    result = parse_qwen_response({"done": True, "response": "Reasoning</think>Answer"}, TOOLS, "assistant\n<think>\n")
    assert result == {"role": "assistant", "thinking": "Reasoning", "content": "Answer"}
    assert parse("Plain answer") == {"role": "assistant", "content": "Plain answer"}


@pytest.mark.parametrize("xml", [
    "<tool_call><function=run><parameter=code>truncated",
    "<tool_call",
    "<function=",
    "<tool_call><function=unknown></function></tool_call>",
    "<tool_call><function=run></function></tool_call>",
    "<tool_call><function=run><parameter=code>x</parameter><parameter=code>y</parameter></function></tool_call>",
    "<tool_call><function=run><parameter=unknown>x</parameter></function></tool_call>",
    "<tool_call><function=run><parameter=code>x<function=run></function></tool_call>",
    "<tool_call><function=run><parameter=code>x<tool_call></tool_call>",
    "<function=run><parameter=code>x</parameter></function>",
    "<think><tool_call><function=run><parameter=code>x</parameter></function></tool_call>",
])
def test_ambiguous_or_incomplete_calls_fail(xml):
    with pytest.raises(BackendUpstreamError) as exc:
        parse(xml)
    assert exc.value.status_code == 502


@pytest.mark.parametrize("name,value", [
    ("count", "nonsense"), ("count", "true"), ("count", "1.5"),
    ("flag", "nonsense"), ("data", '{"truncated":'),
    ("data", '{"a":1,"a":2}'), ("data", '{"a":NaN}'),
    ("data", '{"a":1e999}'),
    ("items", "{}"),
])
def test_no_fabrication_of_typed_arguments(name, value):
    with pytest.raises(BackendUpstreamError):
        parse(f"<tool_call><function=run><parameter=code>x</parameter><parameter={name}>{value}</parameter></function></tool_call>")


def test_truncated_parallel_generation_is_not_partially_executed():
    call = "<tool_call><function=run><parameter=code>x</parameter></function></tool_call>"
    with pytest.raises(BackendUpstreamError):
        parse(call + "<tool_call><function=run>", done_reason="length")


def test_schema_refs_and_unions():
    tools = [{"function": {"name": "f", "parameters": {
        "$defs": {"number": {"type": "integer"}}, "properties": {
            "n": {"$ref": "#/$defs/number"},
            "optional": {"anyOf": [{"type": "null"}, {"type": "object"}]},
        },
    }}}]
    result = parse_qwen_response({"done": True, "response":
        '<tool_call><function=f><parameter=n>42</parameter><parameter=optional>{"a":1}</parameter></function></tool_call>'}, tools, "")
    assert args(result) == {"n": 42, "optional": {"a": 1}}


@pytest.mark.parametrize("separator", ["", "\n\n"])
def test_missing_outer_close_between_parallel_calls(separator):
    first = "<tool_call><function=run><parameter=code>first</parameter></function>"
    second = "<tool_call><function=run><parameter=code>second</parameter></function></tool_call>"
    result = parse(first + separator + second)
    assert [call["function"]["arguments"] for call in result["tool_calls"]] == [
        {"code": "first"}, {"code": "second"},
    ]


def test_repeated_outer_wrappers_do_not_duplicate_calls():
    result = parse("<tool_call>\n<tool_call><function=run><parameter=code>x</parameter></function></tool_call>\n</tool_call>")
    assert len(result["tool_calls"]) == 1
    assert args(result) == {"code": "x"}


def test_nested_example_stays_in_one_argument_followed_by_parallel_call():
    example = "<tool_call><function=run><parameter=code>inner</parameter></function></tool_call>"
    code = "Before\n" + example + "\nAfter"
    outer = f"<tool_call><function=run><parameter=code>\n{code}\n</parameter><parameter=count>2</parameter></function></tool_call>"
    sibling = "<tool_call><function=run><parameter=code>second</parameter></function></tool_call>"
    result = parse(outer + sibling)
    assert len(result["tool_calls"]) == 2
    assert args(result) == {"code": code, "count": 2}
    assert result["tool_calls"][1]["function"]["arguments"] == {"code": "second"}


def test_tool_markup_inside_json_argument():
    import json
    data = {"code": "<tool_call><function=run><parameter=code>inner</parameter></function></tool_call>"}
    result = parse(f'<tool_call><function=run><parameter=code>x</parameter><parameter=data>{json.dumps(data)}</parameter></function></tool_call>')
    assert args(result) == {"code": "x", "data": data}


@pytest.mark.parametrize("ending", ["</parameter>", "</parameter></function>", ""])
def test_normal_stop_can_close_omitted_final_tags(ending):
    result = parse("<tool_call><function=run><parameter=code>complete value" + ending, done_reason="stop")
    assert args(result) == {"code": "complete value"}


def test_missing_function_and_wrapper_close_before_sibling():
    first = "<tool_call><function=run><parameter=code>first</parameter>"
    second = "<tool_call><function=run><parameter=code>second</parameter></function></tool_call>"
    result = parse(first + second)
    assert [call["function"]["arguments"]["code"] for call in result["tool_calls"]] == ["first", "second"]


def test_delimiters_inside_plain_text_are_not_replaced_by_placeholders():
    text = "Example <parameter=literal>x</parameter></function> in plain text"
    assert parse(text)["content"] == text


@pytest.mark.parametrize("text", [
    "<tool_call><function=run><parameter=code>incomplete",
    "<tool_call><function=run><parameter=code>complete</parameter>",
])
def test_token_limit_never_repairs_away_missing_data(text):
    with pytest.raises(BackendUpstreamError):
        parse(text, done_reason="length")
