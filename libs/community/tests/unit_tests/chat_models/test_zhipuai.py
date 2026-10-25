"""Test ZhipuAI Chat API wrapper."""

from functools import reduce
from typing import Any, Dict, Type

import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessage,
    ChatMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    SystemMessage,
    SystemMessageChunk,
    ToolMessage,
)
from langchain_core.runnables import Runnable, RunnableBinding
from pydantic import BaseModel, Field
from pytest import MonkeyPatch

from langchain_community.chat_models.zhipuai import (
    API_BASE_ENV_VARS,
    API_KEY_ENV_VARS,
    ZHIPUAI_API_BASE,
    ChatZhipuAI,
    _convert_delta_to_message_chunk,
    _convert_dict_to_message,
    _convert_message_to_dict,
    _get_jwt_token,
    _truncate_params,
)


class _Weather(BaseModel):
    """Get the current weather in a given location."""

    location: str = Field(..., description="The city, state, e.g. LA, CA")


ALL_ENV_VARS = (*API_KEY_ENV_VARS, *API_BASE_ENV_VARS)


@pytest.fixture
def clean_env(monkeypatch: MonkeyPatch) -> MonkeyPatch:
    """Clear every supported env var so tests control their own inputs."""
    for var in ALL_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


# ---------------------------------------------------------------- model params


@pytest.mark.requires("httpx", "httpx_sse", "jwt")
def test_zhipuai_model_param() -> None:
    llm = ChatZhipuAI(api_key="test", model="foo")
    assert llm.model_name == "foo"
    llm = ChatZhipuAI(api_key="test", model_name="foo")  # type: ignore[call-arg]
    assert llm.model_name == "foo"


def test_llm_metadata(clean_env: MonkeyPatch) -> None:
    llm = ChatZhipuAI(api_key="test")
    assert llm._llm_type == "zhipuai-chat"
    assert llm.lc_secrets == {"zhipuai_api_key": "ZHIPUAI_API_KEY"}
    assert llm.lc_attributes == {"zhipuai_api_base": ZHIPUAI_API_BASE}
    assert ChatZhipuAI.get_lc_namespace() == ["langchain", "chat_models", "zhipuai"]


# ----------------------------------------------------- API key env var fallback


def test_api_key_from_explicit_param_beats_env(clean_env: MonkeyPatch) -> None:
    clean_env.setenv("ZHIPUAI_API_KEY", "from-env")
    llm = ChatZhipuAI(api_key="from-alias")
    assert llm.zhipuai_api_key == "from-alias"
    llm = ChatZhipuAI(zhipuai_api_key="from-field")  # type: ignore[call-arg]
    assert llm.zhipuai_api_key == "from-field"


@pytest.mark.parametrize(
    ("env_var", "expected"),
    [
        ("ZHIPUAI_API_KEY", "zhipuai-key"),
        ("BIGMODEL_API_KEY", "bigmodel-key"),
        ("ZAI_API_KEY", "zai-key"),
    ],
)
def test_api_key_from_single_env_var(
    clean_env: MonkeyPatch, env_var: str, expected: str
) -> None:
    clean_env.setenv(env_var, expected)
    llm = ChatZhipuAI()
    assert llm.zhipuai_api_key == expected


@pytest.mark.parametrize(
    ("env_vars", "expected"),
    [
        (
            {
                "ZHIPUAI_API_KEY": "zhipuai-key",
                "BIGMODEL_API_KEY": "bigmodel-key",
                "ZAI_API_KEY": "zai-key",
            },
            "zhipuai-key",
        ),
        (
            {
                "BIGMODEL_API_KEY": "bigmodel-key",
                "ZAI_API_KEY": "zai-key",
            },
            "bigmodel-key",
        ),
        ({"ZAI_API_KEY": "zai-key"}, "zai-key"),
    ],
)
def test_api_key_env_var_precedence(
    clean_env: MonkeyPatch, env_vars: Dict[str, str], expected: str
) -> None:
    for var, value in env_vars.items():
        clean_env.setenv(var, value)
    llm = ChatZhipuAI()
    assert llm.zhipuai_api_key == expected


def test_api_key_empty_env_var_falls_through(clean_env: MonkeyPatch) -> None:
    clean_env.setenv("ZHIPUAI_API_KEY", "")
    clean_env.setenv("BIGMODEL_API_KEY", "bigmodel-key")
    llm = ChatZhipuAI()
    assert llm.zhipuai_api_key == "bigmodel-key"


def test_api_key_missing_raises(clean_env: MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="Did not find zhipuai_api_key"):
        ChatZhipuAI()


# --------------------------------------------------- api_base env var fallback


def test_api_base_from_explicit_param_beats_env(clean_env: MonkeyPatch) -> None:
    clean_env.setenv("ZHIPUAI_API_BASE", "https://env.test/api")
    llm = ChatZhipuAI(api_key="test", api_base="https://alias.test/api")
    assert llm.zhipuai_api_base == "https://alias.test/api"
    llm = ChatZhipuAI(api_key="test", zhipuai_api_base="https://field.test/api")  # type: ignore[call-arg]
    assert llm.zhipuai_api_base == "https://field.test/api"


def test_api_base_defaults_to_zhipuai_url(clean_env: MonkeyPatch) -> None:
    llm = ChatZhipuAI(api_key="test")
    assert llm.zhipuai_api_base == ZHIPUAI_API_BASE


@pytest.mark.parametrize(
    ("env_var", "expected"),
    [
        ("ZHIPUAI_API_BASE", "https://zhipuai.test/api"),
        ("BIGMODEL_API_BASE", "https://bigmodel.test/api"),
        ("ZAI_API_BASE", "https://zai.test/api"),
    ],
)
def test_api_base_from_single_env_var(
    clean_env: MonkeyPatch, env_var: str, expected: str
) -> None:
    clean_env.setenv(env_var, expected)
    llm = ChatZhipuAI(api_key="test")
    assert llm.zhipuai_api_base == expected


@pytest.mark.parametrize(
    ("env_vars", "expected"),
    [
        (
            {
                "ZHIPUAI_API_BASE": "https://zhipuai.test/api",
                "BIGMODEL_API_BASE": "https://bigmodel.test/api",
                "ZAI_API_BASE": "https://zai.test/api",
            },
            "https://zhipuai.test/api",
        ),
        (
            {
                "BIGMODEL_API_BASE": "https://bigmodel.test/api",
                "ZAI_API_BASE": "https://zai.test/api",
            },
            "https://bigmodel.test/api",
        ),
        ({"ZAI_API_BASE": "https://zai.test/api"}, "https://zai.test/api"),
    ],
)
def test_api_base_env_var_precedence(
    clean_env: MonkeyPatch, env_vars: Dict[str, str], expected: str
) -> None:
    for var, value in env_vars.items():
        clean_env.setenv(var, value)
    llm = ChatZhipuAI(api_key="test")
    assert llm.zhipuai_api_base == expected


# --------------------------------------------------------- message -> dict


def test__convert_message_to_dict_with_tool() -> None:
    message = ToolMessage(name="foo", content="bar", tool_call_id="abc123")
    result = _convert_message_to_dict(message)
    expected_output = {
        "name": "foo",
        "content": "bar",
        "tool_call_id": "abc123",
        "role": "tool",
    }
    assert result == expected_output


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        (SystemMessage(content="sys"), {"role": "system", "content": "sys"}),
        (HumanMessage(content="hi"), {"role": "user", "content": "hi"}),
        (AIMessage(content="hello"), {"role": "assistant", "content": "hello"}),
        (ChatMessage(role="custom", content="x"), {"role": "custom", "content": "x"}),
    ],
)
def test__convert_message_to_dict(
    message: BaseMessage, expected: Dict[str, Any]
) -> None:
    assert _convert_message_to_dict(message) == expected


def test__convert_message_to_dict_tool_without_name() -> None:
    result = _convert_message_to_dict(ToolMessage(content="bar", tool_call_id="abc"))
    assert result == {
        "role": "tool",
        "content": "bar",
        "tool_call_id": "abc",
        "name": None,
    }


def test__convert_message_to_dict_unknown_type_raises() -> None:
    with pytest.raises(TypeError, match="Got unknown type 'str'"):
        _convert_message_to_dict("not a message")  # type: ignore[arg-type]


# --------------------------------------------------------- dict -> message


@pytest.mark.parametrize(
    ("dct", "expected_type", "expected_content"),
    [
        ({"role": "system", "content": "sys"}, SystemMessage, "sys"),
        ({"role": "user", "content": "hi"}, HumanMessage, "hi"),
        ({"role": "assistant", "content": "hello"}, AIMessage, "hello"),
        ({"role": "user"}, HumanMessage, ""),
        ({"role": "custom", "content": "x"}, ChatMessage, "x"),
    ],
)
def test__convert_dict_to_message_roles(
    dct: Dict[str, Any],
    expected_type: Type[BaseMessage],
    expected_content: str,
) -> None:
    message = _convert_dict_to_message(dct)
    assert isinstance(message, expected_type)
    assert message.content == expected_content
    assert message.additional_kwargs == {}


def test__convert_dict_to_message_tool_with_name() -> None:
    message = _convert_dict_to_message(
        {"role": "tool", "content": "bar", "name": "foo", "tool_call_id": "abc"}
    )
    assert isinstance(message, ToolMessage)
    assert message.content == "bar"
    assert message.tool_call_id == "abc"
    assert message.additional_kwargs == {"name": "foo"}


def test__convert_dict_to_message_tool_without_name() -> None:
    message = _convert_dict_to_message(
        {"role": "tool", "content": "bar", "tool_call_id": "abc"}
    )
    assert isinstance(message, ToolMessage)
    assert message.additional_kwargs == {}


def test__convert_dict_to_message_assistant_tool_calls() -> None:
    tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"location": "LA"}'},
        }
    ]
    message = _convert_dict_to_message(
        {"role": "assistant", "content": "", "tool_calls": tool_calls}
    )
    assert isinstance(message, AIMessage)
    assert message.content == ""
    assert message.additional_kwargs == {"tool_calls": tool_calls}


def test__convert_dict_to_message_reasoning_content() -> None:
    message = _convert_dict_to_message(
        {"role": "assistant", "content": "42", "reasoning_content": "thinking..."}
    )
    assert isinstance(message, AIMessage)
    assert message.content == "42"
    assert message.additional_kwargs == {"reasoning_content": "thinking..."}


def test__convert_dict_to_message_reasoning_content_with_tool_calls() -> None:
    tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "get_weather", "arguments": "{}"},
        }
    ]
    message = _convert_dict_to_message(
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "should call a tool",
            "tool_calls": tool_calls,
        }
    )
    assert isinstance(message, AIMessage)
    assert message.additional_kwargs == {
        "tool_calls": tool_calls,
        "reasoning_content": "should call a tool",
    }


def test__convert_dict_to_message_omits_reasoning_content_when_absent() -> None:
    message = _convert_dict_to_message({"role": "assistant", "content": "hi"})
    assert isinstance(message, AIMessage)
    assert message.additional_kwargs == {}


# --------------------------------------------------- dict -> message chunk


@pytest.mark.parametrize(
    ("dct", "default_class", "expected_type"),
    [
        ({"role": "system", "content": "s"}, AIMessageChunk, SystemMessageChunk),
        ({"role": "user", "content": "u"}, AIMessageChunk, HumanMessageChunk),
        ({"role": "assistant", "content": "a"}, AIMessageChunk, AIMessageChunk),
        ({}, AIMessageChunk, AIMessageChunk),
        ({"content": "a"}, AIMessageChunk, AIMessageChunk),
        # an explicit default_class takes precedence over the delta role
        (
            {"role": "assistant", "content": "a"},
            SystemMessageChunk,
            SystemMessageChunk,
        ),
        ({"role": "tool", "content": "t"}, AIMessageChunk, AIMessageChunk),
        ({"role": "tool", "content": "t"}, ChatMessageChunk, ChatMessageChunk),
    ],
)
def test__convert_delta_to_message_chunk_roles(
    dct: Dict[str, Any],
    default_class: Type[BaseMessageChunk],
    expected_type: Type[BaseMessageChunk],
) -> None:
    chunk = _convert_delta_to_message_chunk(dct, default_class)
    assert isinstance(chunk, expected_type)
    assert chunk.content == dct.get("content", "")


def test__convert_delta_to_message_chunk_tool_calls() -> None:
    tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"location": '},
        }
    ]
    chunk = _convert_delta_to_message_chunk(
        {"role": "assistant", "content": "", "tool_calls": tool_calls}, AIMessageChunk
    )
    assert isinstance(chunk, AIMessageChunk)
    assert chunk.additional_kwargs == {"tool_calls": tool_calls}


def test__convert_delta_to_message_chunk_reasoning_content() -> None:
    chunk = _convert_delta_to_message_chunk(
        {"role": "assistant", "content": "", "reasoning_content": "Let me think"},
        AIMessageChunk,
    )
    assert isinstance(chunk, AIMessageChunk)
    assert chunk.additional_kwargs == {"reasoning_content": "Let me think"}


def test__convert_delta_to_message_chunk_reasoning_content_empty_string() -> None:
    # Streams may emit an empty reasoning_content on non-reasoning chunks.
    chunk = _convert_delta_to_message_chunk(
        {"role": "assistant", "content": "x", "reasoning_content": ""}, AIMessageChunk
    )
    assert isinstance(chunk, AIMessageChunk)
    assert chunk.additional_kwargs == {"reasoning_content": ""}


def test__convert_delta_to_message_chunk_omits_reasoning_content_when_absent() -> None:
    chunk = _convert_delta_to_message_chunk(
        {"role": "assistant", "content": "x"}, AIMessageChunk
    )
    assert isinstance(chunk, AIMessageChunk)
    assert chunk.additional_kwargs == {}


def test_reasoning_content_accumulates_when_chunks_merged() -> None:
    deltas = [
        {"role": "assistant", "content": "", "reasoning_content": "Let me "},
        {"content": "2", "reasoning_content": "think about "},
        {"content": "!"},
    ]
    merged = reduce(
        lambda left, right: (
            left + _convert_delta_to_message_chunk(right, AIMessageChunk)
        ),
        deltas[1:],
        _convert_delta_to_message_chunk(deltas[0], AIMessageChunk),
    )
    assert isinstance(merged, AIMessageChunk)
    assert merged.content == "2!"
    assert merged.additional_kwargs == {"reasoning_content": "Let me think about "}


# ----------------------------------------------------------- param truncation


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"temperature": 0}, {"temperature": 0.01}),
        ({"temperature": 0.001}, {"temperature": 0.01}),
        ({"temperature": 0.01}, {"temperature": 0.01}),
        ({"temperature": 0.5}, {"temperature": 0.5}),
        ({"temperature": 0.99}, {"temperature": 0.99}),
        ({"temperature": 1}, {"temperature": 0.99}),
        ({"temperature": 3.7}, {"temperature": 0.99}),
        ({"temperature": None}, {"temperature": None}),
        ({"top_p": 0}, {"top_p": 0.01}),
        ({"top_p": 0.001}, {"top_p": 0.01}),
        ({"top_p": 0.7}, {"top_p": 0.7}),
        ({"top_p": 1}, {"top_p": 0.99}),
        ({"top_p": None}, {"top_p": None}),
        (
            {"temperature": 0, "top_p": 1, "model": "glm-4"},
            {"temperature": 0.01, "top_p": 0.99, "model": "glm-4"},
        ),
        ({}, {}),
    ],
)
def test__truncate_params(payload: Dict[str, Any], expected: Dict[str, Any]) -> None:
    _truncate_params(payload)
    assert payload == expected


def test__truncate_params_ignores_absent_params() -> None:
    payload = {"model": "glm-4", "messages": []}
    _truncate_params(payload)
    assert payload == {"model": "glm-4", "messages": []}


# --------------------------------------------------------- request building


def test__default_params(clean_env: MonkeyPatch) -> None:
    llm = ChatZhipuAI(api_key="test", model="glm-4", max_tokens=128)
    assert llm._default_params == {
        "model": "glm-4",
        "stream": False,
        "temperature": 0.95,
        "max_tokens": 128,
    }


def test__create_message_dicts(clean_env: MonkeyPatch) -> None:
    llm = ChatZhipuAI(api_key="test", model="glm-4", max_tokens=128)
    message_dicts, params = llm._create_message_dicts(
        [HumanMessage(content="hi")], stop=["END"]
    )
    assert message_dicts == [{"role": "user", "content": "hi"}]
    assert params["model"] == "glm-4"
    assert params["max_tokens"] == 128
    assert params["stop"] == ["END"]


def test__create_chat_result(clean_env: MonkeyPatch) -> None:
    llm = ChatZhipuAI(api_key="test", model="glm-4.5")
    response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "42",
                    "reasoning_content": "deep thought",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }
    result = llm._create_chat_result(response)
    generation = result.generations[0]
    assert isinstance(generation.message, AIMessage)
    assert generation.message.content == "42"
    assert generation.message.additional_kwargs == {"reasoning_content": "deep thought"}
    assert generation.generation_info == {"finish_reason": "stop"}
    assert result.llm_output == {
        "token_usage": {
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "total_tokens": 5,
        },
        "model_name": "glm-4.5",
    }


# ----------------------------------------------------------------- tool calls


def test_bind_tools_pydantic_schema(clean_env: MonkeyPatch) -> None:
    llm = ChatZhipuAI(api_key="test")
    runnable = llm.bind_tools([_Weather])
    assert isinstance(runnable, RunnableBinding)
    tools = runnable.kwargs["tools"]
    assert tools[0]["function"]["name"] == _Weather.__name__
    assert "tool_choice" not in runnable.kwargs


def test_bind_tools_openai_dict(clean_env: MonkeyPatch) -> None:
    tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        },
    }
    runnable = ChatZhipuAI(api_key="test").bind_tools([tool])
    assert isinstance(runnable, RunnableBinding)
    assert runnable.kwargs["tools"] == [tool]


def test_bind_tools_tool_choice_auto(clean_env: MonkeyPatch) -> None:
    runnable = ChatZhipuAI(api_key="test").bind_tools([_Weather], tool_choice="auto")
    assert isinstance(runnable, RunnableBinding)
    assert runnable.kwargs["tool_choice"] == "auto"


@pytest.mark.parametrize("tool_choice", ["any", "none", "required", "Weather", True])
def test_bind_tools_unsupported_tool_choice_raises(
    clean_env: MonkeyPatch, tool_choice: Any
) -> None:
    with pytest.raises(ValueError, match="only supports `auto` tool choice"):
        ChatZhipuAI(api_key="test").bind_tools([_Weather], tool_choice=tool_choice)


def test_bind_tools_glm_4v_raises(clean_env: MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="does not support tool calling"):
        ChatZhipuAI(api_key="test", model="glm-4v").bind_tools([_Weather])


# ------------------------------------------------------- structured output


def test_with_structured_output_returns_runnable(clean_env: MonkeyPatch) -> None:
    structured = ChatZhipuAI(api_key="test").with_structured_output(_Weather)
    assert isinstance(structured, Runnable)


def test_with_structured_output_requires_schema(clean_env: MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="schema must be specified"):
        ChatZhipuAI(api_key="test").with_structured_output(None)


def test_with_structured_output_rejects_unknown_kwargs(
    clean_env: MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="unsupported arguments"):
        ChatZhipuAI(api_key="test").with_structured_output(_Weather, foo="bar")


def test_with_structured_output_rejects_json_mode(clean_env: MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="Unrecognized method"):
        ChatZhipuAI(api_key="test").with_structured_output(_Weather, method="json_mode")


# ----------------------------------------------------------------- jwt token


@pytest.mark.requires("jwt")
def test_get_jwt_token_rejects_malformed_api_key() -> None:
    with pytest.raises(ValueError, match="Invalid API key"):
        _get_jwt_token("missing-the-dot")


@pytest.mark.requires("jwt")
def test_get_jwt_token_returns_signed_token() -> None:
    token = _get_jwt_token("id.secret")
    assert isinstance(token, str)
    assert token.count(".") == 2
