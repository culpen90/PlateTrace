import copy
import json

import httpx
import pytest

from platetrace import providers

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "vehicle_specs",
            "description": "Look up public vehicle specifications",
            "parameters": {"type": "object", "properties": {"vin": {"type": "string"}}},
        },
    }
]


@pytest.fixture(autouse=True)
def clear_provider_environment(monkeypatch):
    for name in ("OPENROUTER_API_KEY", "OLLAMA_API_KEY", "OLLAMA_BASE_URL"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def mock_http(monkeypatch):
    real_client = httpx.AsyncClient

    def install(handler):
        def client(**kwargs):
            return real_client(transport=httpx.MockTransport(handler), **kwargs)

        monkeypatch.setattr(providers.httpx, "AsyncClient", client)

    return install


async def test_openrouter_tools_and_reasoning_round_trip(mock_http, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret-env-key")
    reasoning = [{"type": "reasoning.encrypted", "data": "opaque-signature"}]
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        assert str(request.url) == "https://openrouter.ai/api/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer secret-env-key"
        assert request.extensions["timeout"]["connect"] == 10
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "reasoning_details": reasoning,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "vehicle_specs",
                                        "arguments": '{"vin":"TESTVIN"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
        )

    mock_http(handler)
    message = await providers.complete(
        "openrouter", "test/model", [{"role": "user", "content": "Research"}], TOOLS
    )
    assert message["content"] == ""
    assert message["reasoning_details"] == reasoning
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {"vin": "TESTVIN"}
    assert requests[0]["tools"] == TOOLS
    assert requests[0]["provider"] == {"require_parameters": True}
    assert requests[0]["stream"] is False
    await providers.complete("openrouter", "test/model", [message], TOOLS)
    assert requests[1]["messages"][0]["reasoning_details"] == reasoning


async def test_ollama_native_tool_round_trip_preserves_names(mock_http, monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434/")
    monkeypatch.setenv("OLLAMA_API_KEY", "ollama-key")
    requests = []

    def handler(request):
        assert str(request.url) == "http://localhost:11434/api/chat"
        assert request.headers["authorization"] == "Bearer ollama-key"
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "",
                    "thinking": "private reasoning",
                    "tool_calls": [{"function": {"name": "vehicle_specs", "arguments": {"vin": "TESTVIN"}}}],
                }
            },
        )

    mock_http(handler)
    message = await providers.complete(
        "ollama", "test:latest", [{"role": "user", "content": "Research"}], TOOLS
    )
    call_id = message["tool_calls"][0]["id"]
    history = [message, {"role": "tool", "tool_call_id": call_id, "content": "public specs"}]
    original = copy.deepcopy(history)
    await providers.complete("ollama", "test:latest", history, TOOLS)
    assert history == original
    assert requests[0]["stream"] is False
    assert requests[0]["tools"] == TOOLS
    assert requests[1]["messages"][0]["thinking"] == "private reasoning"
    assert requests[1]["messages"][0]["tool_calls"] == [
        {"function": {"name": "vehicle_specs", "arguments": {"vin": "TESTVIN"}}}
    ]
    assert requests[1]["messages"][1] == {
        "role": "tool",
        "content": "public specs",
        "tool_name": "vehicle_specs",
    }


@pytest.mark.parametrize("provider", ["openrouter", "ollama"])
@pytest.mark.parametrize("arguments", ['{"broken":', "[]", "null", '{"bad": NaN}', 7])
async def test_malformed_tool_arguments_fail_without_leaking_data(mock_http, provider, arguments):
    message = {
        "role": "assistant",
        "tool_calls": [
            {
                "function": {
                    "name": "vehicle_specs",
                    "arguments": arguments,
                }
            }
        ],
    }
    payload = {"choices": [{"message": message}]} if provider == "openrouter" else {"message": message}
    mock_http(lambda _: httpx.Response(200, json=payload))
    with pytest.raises(providers.ProviderError, match="invalid tool arguments") as error:
        await providers.complete(provider, "model", [], TOOLS, "secret-key")
    assert "secret-key" not in str(error.value)
    assert "broken" not in str(error.value)


@pytest.mark.parametrize("provider", ["openrouter", "ollama"])
async def test_text_only_response_is_normalized(mock_http, provider):
    message = {"role": "assistant", "content": "Source-cited answer"}
    payload = {"choices": [{"message": message}]} if provider == "openrouter" else {"message": message}
    mock_http(lambda _: httpx.Response(200, json=payload))
    assert await providers.complete(provider, "model", [], [], "key") == {
        "role": "assistant",
        "content": "Source-cited answer",
        "tool_calls": [],
    }


async def test_openrouter_requires_key_before_network(mock_http):
    mock_http(lambda _: pytest.fail("No request should be made without a key"))
    with pytest.raises(providers.ProviderError, match="OPENROUTER_API_KEY"):
        await providers.complete("openrouter", "model", [], TOOLS)


@pytest.mark.parametrize(
    "status,detail,expected",
    [
        (401, "secret key and request echoed", "authentication"),
        (402, "secret key and request echoed", "credits"),
        (429, "secret key and request echoed", "rate limit"),
        (400, "model does not support tools: secret", "tool-capable"),
        (404, "No endpoints support tool use: secret", "tool-capable"),
        (503, "secret key and request echoed", "temporarily unavailable"),
        (400, "secret key and request echoed", "rejected the request"),
    ],
)
async def test_provider_errors_are_actionable_and_redacted(mock_http, status, detail, expected):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status, json={"error": {"message": detail}})

    mock_http(handler)
    with pytest.raises(providers.ProviderError, match=expected) as error:
        await providers.complete("openrouter", "model", [], TOOLS, "secret")
    assert "secret" not in str(error.value)
    assert len(calls) == 1


async def test_ollama_unreachable_is_actionable(mock_http):
    def handler(request):
        raise httpx.ConnectError("could expose secret host here", request=request)

    mock_http(handler)
    with pytest.raises(providers.ProviderError, match="ollama serve") as error:
        await providers.list_models("ollama")
    assert "secret host" not in str(error.value)


async def test_timeout_is_redacted(mock_http):
    def handler(request):
        raise httpx.ReadTimeout("sensitive request", request=request)

    mock_http(handler)
    with pytest.raises(providers.ProviderError, match="timed out") as error:
        await providers.complete("ollama", "model", [], TOOLS)
    assert "sensitive" not in str(error.value)


async def test_openrouter_public_model_list_filters_tools(mock_http):
    def handler(request):
        assert request.method == "GET"
        assert request.url.params["supported_parameters"] == "tools"
        assert "authorization" not in request.headers
        return httpx.Response(
            200,
            json={
                "data": [
                    {"id": "a", "name": "Model A", "supported_parameters": ["tools", "temperature"]},
                    {"id": "b", "name": "No tools", "supported_parameters": ["temperature"]},
                    {"id": "c"},
                    {"id": "a", "name": "Duplicate", "supported_parameters": ["tools"]},
                    {"id": "d", "supported_parameters": ["tools"]},
                ]
            },
        )

    mock_http(handler)
    assert await providers.list_models("openrouter") == [
        {"id": "a", "name": "Model A"},
        {"id": "d", "name": "d"},
    ]


async def test_ollama_model_list(mock_http):
    def handler(request):
        assert str(request.url) == "http://127.0.0.1:11434/api/tags"
        assert "authorization" not in request.headers
        return httpx.Response(
            200,
            json={
                "models": [
                    {"name": "qwen3:8b", "model": "qwen3:8b"},
                    {"name": "llama3.2:latest"},
                    {"name": "qwen3:8b"},
                    {},
                    "invalid",
                ]
            },
        )

    mock_http(handler)
    assert await providers.list_models("ollama") == [
        {"id": "qwen3:8b", "name": "qwen3:8b"},
        {"id": "llama3.2:latest", "name": "llama3.2:latest"},
    ]


@pytest.mark.parametrize("payload", [{}, {"choices": []}, [], {"choices": [{"message": {}}]}])
async def test_invalid_completion_response_is_safe(mock_http, payload):
    mock_http(lambda _: httpx.Response(200, json=payload))
    with pytest.raises(providers.ProviderError):
        await providers.complete("openrouter", "model", [], TOOLS, "key")


async def test_embedded_error_with_http_200(mock_http):
    mock_http(lambda _: httpx.Response(200, json={"error": {"code": 429, "message": "sensitive"}}))
    with pytest.raises(providers.ProviderError, match="rate limit"):
        await providers.complete("openrouter", "model", [], TOOLS, "key")


async def test_non_json_response(mock_http):
    mock_http(lambda _: httpx.Response(200, text="private proxy HTML"))
    with pytest.raises(providers.ProviderError, match="invalid response"):
        await providers.list_models("ollama")


@pytest.mark.parametrize(
    "base", ["file:///etc/passwd", "http://user:secret@localhost:11434", "http://[bad", ""]
)
async def test_invalid_ollama_configuration_is_not_echoed(monkeypatch, base):
    monkeypatch.setenv("OLLAMA_BASE_URL", base)
    with pytest.raises(providers.ProviderError, match="OLLAMA_BASE_URL") as error:
        await providers.list_models("ollama")
    assert "secret" not in str(error.value)


async def test_unknown_provider():
    with pytest.raises(providers.ProviderError, match="Choose OpenRouter or Ollama"):
        await providers.list_models("unrecognized")


async def test_orphan_ollama_tool_result_fails_before_network(mock_http):
    mock_http(lambda _: pytest.fail("No request should be made with an orphaned tool result"))
    with pytest.raises(providers.ProviderError, match="no matching call"):
        await providers.complete(
            "ollama", "model", [{"role": "tool", "tool_call_id": "missing", "content": "x"}], TOOLS
        )
