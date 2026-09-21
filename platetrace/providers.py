"""Small, asynchronous adapters for OpenRouter and Ollama tool conversations.

Public errors deliberately exclude upstream response bodies, URLs, and credentials.
Reasoning metadata is passed through for conversation continuity, not UI rendering.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

import httpx

OPENROUTER_URL = "https://openrouter.ai/api/v1"
REQUEST_TIMEOUT = httpx.Timeout(120.0, connect=10.0, pool=10.0)
TOTAL_TIMEOUT = 150.0


class ProviderError(Exception):
    """An actionable provider failure that is safe to show to the user."""


def _provider_name(provider: str) -> str:
    if provider not in {"openrouter", "ollama"}:
        raise ProviderError("Choose OpenRouter or Ollama as the model provider.")
    return "OpenRouter" if provider == "openrouter" else "Ollama"


def _configuration(provider: str, api_key: str, *, require_key: bool) -> tuple[str, dict[str, str]]:
    _provider_name(provider)
    headers = {"Accept": "application/json"}
    if provider == "openrouter":
        key = api_key.strip() or os.getenv("OPENROUTER_API_KEY", "").strip()
        if require_key and not key:
            raise ProviderError("Add an OpenRouter API key in settings or set OPENROUTER_API_KEY.")
        base = OPENROUTER_URL
        headers["X-OpenRouter-Title"] = "PlateTrace"
    else:
        key = api_key.strip() or os.getenv("OLLAMA_API_KEY", "").strip()
        base = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").strip().rstrip("/")
        try:
            parsed = urlsplit(base)
            valid = (
                parsed.scheme in {"http", "https"}
                and parsed.hostname
                and not parsed.username
                and not parsed.password
                and not parsed.query
                and not parsed.fragment
            )
            _ = parsed.port
        except ValueError:
            valid = False
        if not valid:
            raise ProviderError("Set OLLAMA_BASE_URL to an HTTP(S) server URL without embedded credentials.")
    if key:
        if any(ord(char) < 32 or ord(char) > 126 for char in key):
            raise ProviderError("The API key contains invalid characters. Copy the key again in settings.")
        headers["Authorization"] = f"Bearer {key}"
    return base, headers


def _error_message(provider: str, status: int, data: Any = None) -> str:
    name = _provider_name(provider)
    error = data.get("error", "") if isinstance(data, dict) else ""
    detail = error.get("message", "") if isinstance(error, dict) else error
    detail = detail.lower() if isinstance(detail, str) else ""
    if status in {401, 403}:
        return f"{name} rejected authentication or access. Check your API key and model permissions."
    if status == 402:
        return f"{name} has insufficient credits. Add credits or choose a model within your account limits."
    if status == 429:
        return f"{name} rate limit reached. Wait a moment before starting another research run."
    if "tool" in detail and any(word in detail for word in ("support", "endpoint", "capab")):
        return f"This {name} model does not support the required tools. Select a tool-capable model."
    if status == 404:
        if provider == "ollama":
            return (
                "Ollama could not find the model or API endpoint. Pull the model and check OLLAMA_BASE_URL."
            )
        return "OpenRouter could not route this model. Refresh models and select a tool-capable model."
    if status in {408, 504}:
        return f"{name} timed out. Try a smaller model or start the research again."
    if status >= 500:
        return f"{name} is temporarily unavailable. Check the service and try again later."
    return f"{name} rejected the request. Check the selected model and its tool support (HTTP {status})."


async def _request_json(
    provider: str,
    method: str,
    url: str,
    headers: dict[str, str],
    *,
    payload: dict | None = None,
    params: dict | None = None,
) -> dict:
    name = _provider_name(provider)
    try:
        async with asyncio.timeout(TOTAL_TIMEOUT):
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT, follow_redirects=False) as client:
                response = await client.request(method, url, headers=headers, json=payload, params=params)
    except (httpx.TimeoutException, TimeoutError):
        raise ProviderError(f"{name} timed out. Try a smaller model or start the research again.") from None
    except httpx.HTTPError:
        if provider == "ollama":
            message = "Cannot reach Ollama. Start Ollama (ollama serve) and check OLLAMA_BASE_URL."
        else:
            message = "Cannot reach OpenRouter. Check your internet connection and try again."
        raise ProviderError(message) from None
    try:
        data = response.json()
    except (ValueError, UnicodeError):
        data = None
    if not response.is_success:
        raise ProviderError(_error_message(provider, response.status_code, data))
    if not isinstance(data, dict):
        raise ProviderError(f"{name} returned an invalid response. Try again or choose another model.")
    if data.get("error"):
        error = data["error"]
        code = error.get("code", 400) if isinstance(error, dict) else 400
        try:
            status = int(code)
        except (TypeError, ValueError):
            status = 400
        raise ProviderError(_error_message(provider, status, data))
    return data


def _arguments(value: Any) -> dict:
    """Require a JSON object: never repair or silently execute malformed calls."""
    try:
        if isinstance(value, str):
            value = json.loads(value)
        if not isinstance(value, dict):
            raise TypeError
        # Also reject non-JSON values if a caller supplies a Python dictionary.
        json.dumps(value, allow_nan=False)
    except (ValueError, TypeError, RecursionError):
        raise ProviderError(
            "The model returned invalid tool arguments. Retry or choose another tool-capable model."
        ) from None
    return value


def _normalize_message(message: Any, provider: str) -> dict:
    if not isinstance(message, dict) or message.get("role", "assistant") != "assistant":
        raise ProviderError(f"{_provider_name(provider)} returned an invalid assistant message.")
    content = message.get("content") or ""
    if not isinstance(content, str):
        raise ProviderError(f"{_provider_name(provider)} returned unsupported message content.")
    calls = message.get("tool_calls") or []
    if not isinstance(calls, list):
        raise ProviderError("The model returned invalid tool calls. Select another tool-capable model.")
    normalized: dict[str, Any] = {"role": "assistant", "content": content, "tool_calls": []}
    seen_ids: set[str] = set()
    for call in calls:
        if not isinstance(call, dict) or call.get("type", "function") != "function":
            raise ProviderError("The model returned an unsupported tool call.")
        function = call.get("function")
        if (
            not isinstance(function, dict)
            or not isinstance(function.get("name"), str)
            or not function["name"]
        ):
            raise ProviderError("The model returned a tool call without a function name.")
        arguments = _arguments(function.get("arguments", {}))
        call_id = call.get("id")
        if not isinstance(call_id, str) or not call_id or call_id in seen_ids:
            call_id = f"call_{uuid4().hex}"
        seen_ids.add(call_id)
        normalized["tool_calls"].append(
            {
                "id": call_id,
                "type": "function",
                "function": {"name": function["name"], "arguments": json.dumps(arguments, allow_nan=False)},
            }
        )
    # Some reasoning models require these opaque fields on subsequent tool turns.
    fields = ("reasoning_details", "reasoning") if provider == "openrouter" else ("thinking",)
    for field in fields:
        if field in message:
            normalized[field] = message[field]
    if not content and not normalized["tool_calls"]:
        raise ProviderError("The model returned no answer or tool calls. Try another tool-capable model.")
    return normalized


def _ollama_messages(messages: list[dict]) -> list[dict]:
    """Translate the shared OpenAI message history to Ollama's native format."""
    result = []
    names: dict[str, str] = {}
    for message in messages:
        converted = {"role": message["role"], "content": message.get("content") or ""}
        if "thinking" in message:
            converted["thinking"] = message["thinking"]
        if "images" in message:
            converted["images"] = message["images"]
        if message.get("tool_calls"):
            converted["tool_calls"] = []
            for call in message["tool_calls"]:
                function = call["function"]
                if call.get("id"):
                    names[call["id"]] = function["name"]
                converted["tool_calls"].append(
                    {
                        "function": {
                            "name": function["name"],
                            "arguments": _arguments(function.get("arguments", {})),
                        }
                    }
                )
        if message["role"] == "tool":
            name = message.get("tool_name") or message.get("name") or names.get(message.get("tool_call_id"))
            if not name:
                raise ProviderError("A tool result has no matching call. Start a new research run.")
            converted["tool_name"] = name
        result.append(converted)
    return result


async def complete(
    provider: str, model: str, messages: list[dict], tools: list[dict], api_key: str = ""
) -> dict:
    """Return one normalized assistant message, retaining tool reasoning metadata."""
    base, headers = _configuration(provider, api_key, require_key=True)
    if not isinstance(model, str) or not model.strip():
        raise ProviderError("Select a model before starting research.")
    if provider == "openrouter":
        payload = {"model": model.strip(), "messages": messages, "stream": False}
        if tools:
            payload.update({"tools": tools, "tool_choice": "auto", "provider": {"require_parameters": True}})
        data = await _request_json(provider, "POST", f"{base}/chat/completions", headers, payload=payload)
        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError):
            raise ProviderError(
                "OpenRouter returned no assistant message. Try again or choose another model."
            ) from None
    else:
        payload = {"model": model.strip(), "messages": _ollama_messages(messages), "stream": False}
        if tools:
            payload["tools"] = tools
        data = await _request_json(provider, "POST", f"{base}/api/chat", headers, payload=payload)
        message = data.get("message")
    return _normalize_message(message, provider)


async def list_models(provider: str, api_key: str = "") -> list[dict[str, str]]:
    """List tool-capable OpenRouter models or installed Ollama models.

    Ollama's tags endpoint does not report tool capability. Unsupported models are
    diagnosed when a tool request is made, without extra per-model network calls.
    """
    base, headers = _configuration(provider, api_key, require_key=False)
    if provider == "openrouter":
        data = await _request_json(
            provider, "GET", f"{base}/models", headers, params={"supported_parameters": "tools"}
        )
        models = data.get("data")
    else:
        data = await _request_json(provider, "GET", f"{base}/api/tags", headers)
        models = data.get("models")
    if not isinstance(models, list):
        raise ProviderError(f"{_provider_name(provider)} returned an invalid model list.")
    result = []
    seen = set()
    for model in models:
        if not isinstance(model, dict):
            continue
        if provider == "openrouter":
            parameters = model.get("supported_parameters")
            if not isinstance(parameters, list) or "tools" not in parameters:
                continue
            model_id = model.get("id")
        else:
            model_id = model.get("model") or model.get("name")
        if not isinstance(model_id, str) or not model_id or model_id in seen:
            continue
        seen.add(model_id)
        name = model.get("name")
        result.append({"id": model_id, "name": name if isinstance(name, str) and name else model_id})
    return result
