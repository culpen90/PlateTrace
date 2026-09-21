import base64
import io
import json
import struct
import threading
import zlib

import httpx
import pytest
from PIL import Image, PngImagePlugin
from pydantic import ValidationError

from platetrace import photos, providers
from platetrace.photos import PhotoError, PhotoRequest, read_plate


def image_bytes(format="PNG", size=(32, 16), **save_options):
    output = io.BytesIO()
    Image.new("RGB", size, "white").save(output, format=format, **save_options)
    return output.getvalue()


def data_url(raw=None, mime="image/png"):
    return f"data:{mime};base64," + base64.b64encode(raw if raw is not None else image_bytes()).decode()


def photo_request(**overrides):
    fields = {"provider": "ollama", "model_id": "vision:test", "image_data_url": data_url()}
    return PhotoRequest(**(fields | overrides))


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


@pytest.mark.parametrize("provider", ["openrouter", "ollama"])
async def test_both_provider_adapters_send_sanitized_photo_without_tools(mock_http, provider):
    exif = Image.Exif()
    exif[274] = 6  # Rotate 90 degrees clockwise before dropping EXIF.
    exif[270] = "private-photo-comment"
    raw = image_bytes("JPEG", size=(48, 16), exif=exif, icc_profile=b"private-color-profile")
    calls = []

    def handler(request):
        calls.append(request)
        assert request.headers["authorization"] == "Bearer photo-secret"
        body = json.loads(request.content)
        assert body["model"] == "vision:test"
        assert body["stream"] is False
        assert "tools" not in body
        assert "tool_choice" not in body
        assert len(body["messages"]) == 2
        system, user = body["messages"]
        assert system["role"] == "system"
        assert "untrusted data" in system["content"]
        assert "never guess" in system["content"]
        assert "multiple license plates" in system["content"]
        if provider == "openrouter":
            assert str(request.url) == "https://openrouter.ai/api/v1/chat/completions"
            assert "images" not in user
            assert user["content"][0]["type"] == "text"
            image = user["content"][1]
            assert image["type"] == "image_url"
            assert image["image_url"]["url"].startswith("data:image/jpeg;base64,")
            encoded = image["image_url"]["url"].split(",", 1)[1]
        else:
            assert str(request.url) == "http://127.0.0.1:11434/api/chat"
            assert isinstance(user["content"], str)
            assert len(user["images"]) == 1
            encoded = user["images"][0]
            assert not encoded.startswith("data:")
        sanitized = base64.b64decode(encoded, validate=True)
        assert b"private-photo-comment" not in sanitized
        assert b"private-color-profile" not in sanitized
        with Image.open(io.BytesIO(sanitized)) as image:
            assert image.format == "JPEG"
            assert image.size == (16, 48)
            assert not image.getexif()
            assert "icc_profile" not in image.info
        message = {"role": "assistant", "content": '{"plate":" abc-123 ","jurisdiction":" Maine ","warnings":[]}'}
        response = {"choices": [{"message": message}]} if provider == "openrouter" else {"message": message}
        return httpx.Response(200, json=response)

    mock_http(handler)
    result = await read_plate(photo_request(
        provider=provider, model_id=" vision:test ", api_key="photo-secret",
        image_data_url=data_url(raw, "image/jpeg"),
    ))
    assert result == {"plate": "ABC-123", "jurisdiction": "Maine", "warnings": []}
    assert len(calls) == 1


@pytest.mark.parametrize("format,mime", [("PNG", "image/png"), ("JPEG", "image/jpeg"), ("WEBP", "image/webp")])
def test_supported_photos_are_resized_to_jpeg(format, mime):
    encoded = photos._prepare_image(data_url(image_bytes(format, size=(3000, 1200)), mime))
    with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
        assert image.format == "JPEG"
        assert image.size == (2048, 819)


def test_png_text_and_transparency_are_removed():
    info = PngImagePlugin.PngInfo()
    info.add_text("comment", "private-png-note")
    raw = io.BytesIO()
    Image.new("RGBA", (16, 16), (0, 0, 0, 0)).save(raw, format="PNG", pnginfo=info)
    encoded = photos._prepare_image(data_url(raw.getvalue()))
    clean = base64.b64decode(encoded)
    assert b"private-png-note" not in clean
    with Image.open(io.BytesIO(clean)) as image:
        assert image.mode == "RGB"
        assert image.getpixel((0, 0)) == (255, 255, 255)


@pytest.mark.parametrize("url", [
    "https://example.com/plate.jpg",
    "data:image/svg+xml;base64,PHN2Zz4=",
    "data:image/gif;base64,R0lGODlh",
    "data:image/png,not-base64",
    "data:image/png;base64,@@@@",
    "data:image/png;base64,",
    "data:image/png;base64,éééé",
    "data:image/png;base64,YQ==\n",
    "data:image/png;base64,YR==",  # Non-canonical padding bits.
    data_url(b"private-non-image-input"),
    data_url(image_bytes("JPEG")),  # Incorrect declared format.
    data_url(image_bytes()[:45]),
])
async def test_invalid_photo_never_reaches_provider(monkeypatch, url):
    async def unexpected(*args, **kwargs):
        pytest.fail("An invalid photo must not be sent to a provider")

    monkeypatch.setattr(photos, "complete", unexpected)
    with pytest.raises(PhotoError) as error:
        await read_plate(photo_request(image_data_url=url))
    assert url not in str(error.value)
    assert "private-non-image-input" not in str(error.value)


def test_oversize_encoded_and_decoded_images_are_rejected(monkeypatch):
    monkeypatch.setattr(photos, "MAX_PHOTO_BYTES", 8)
    for raw in (b"x" * 9, b"x" * 10):
        with pytest.raises(PhotoError, match="too large"):
            photos._prepare_image(data_url(raw))


def test_pixel_limit_is_checked_before_loading():
    raw = bytearray(image_bytes())
    raw[16:24] = struct.pack(">II", 5000, 5000)
    raw[29:33] = struct.pack(">I", zlib.crc32(raw[12:29]) & 0xFFFFFFFF)
    with pytest.raises(PhotoError, match="24 megapixels"):
        photos._prepare_image(data_url(bytes(raw)))


@pytest.mark.parametrize("format,mime", [("PNG", "image/png"), ("WEBP", "image/webp")])
def test_animated_images_are_rejected(format, mime):
    output = io.BytesIO()
    Image.new("RGB", (16, 16), "white").save(
        output, format=format, save_all=True, append_images=[Image.new("RGB", (16, 16), "black")], duration=100,
    )
    with pytest.raises(PhotoError, match="Animated"):
        photos._prepare_image(data_url(output.getvalue(), mime))


async def test_decode_runs_off_event_loop(monkeypatch):
    event_thread = threading.get_ident()
    prepare = photos._prepare_image
    threads = []

    def wrapped(url):
        threads.append(threading.get_ident())
        return prepare(url)

    async def complete(*args, **kwargs):
        return {"content": '{"plate":"","jurisdiction":"","warnings":["No plate is visible."]}'}

    monkeypatch.setattr(photos, "_prepare_image", wrapped)
    monkeypatch.setattr(photos, "complete", complete)
    await read_plate(photo_request())
    assert len(threads) == 1
    assert threads[0] != event_thread


@pytest.mark.parametrize("overrides", [
    {"provider": "demo"}, {"model_id": "   "}, {"model_id": "x" * 201},
    {"api_key": "x" * 501}, {"unexpected": "value"},
])
def test_request_validation(overrides):
    with pytest.raises(ValidationError):
        photo_request(**overrides)


def test_photo_and_key_are_not_in_request_repr_or_validation_error():
    request = photo_request(api_key="private-api-key")
    assert request.image_data_url not in repr(request)
    assert request.api_key not in repr(request)
    with pytest.raises(ValidationError) as error:
        photo_request(api_key="private-api-key" * 100)
    assert "private-api-key" not in str(error.value)


@pytest.mark.parametrize("result", [
    {"plate": "", "jurisdiction": "", "warnings": ["No plate is visible."]},
    {"plate": "", "jurisdiction": "", "warnings": ["Characters are uncertain; try a clearer photo."]},
    {"plate": "", "jurisdiction": "", "warnings": ["Multiple plates are visible. Crop to one plate."]},
])
def test_empty_and_uncertain_results_remain_empty(result):
    assert photos._read_result({"content": json.dumps(result)}) == result


def test_missing_values_receive_actionable_warnings():
    empty = photos._read_result({"content": '{"plate":"","jurisdiction":"","warnings":[]}'})
    assert empty["plate"] == empty["jurisdiction"] == ""
    assert "No plate" in empty["warnings"][0]
    unknown = photos._read_result({"content": '{"plate":"ABC123","jurisdiction":"","warnings":[]}'})
    assert unknown["plate"] == "ABC123"
    assert "jurisdiction" in unknown["warnings"][0]


def test_json_code_fences_are_accepted():
    result = {"plate": "ABC123", "jurisdiction": "Maine", "warnings": []}
    assert photos._read_result({"content": "```json\n" + json.dumps(result) + "\n```"}) == result


@pytest.mark.parametrize("content", [
    "private-bad-response", "x" * (photos.MAX_RESPONSE_CHARS + 1), "[]", "null", "{bad}",
    '{"plate":"ABC123","jurisdiction":"Maine"}',
    '{"plate":"ABC123","plate":"OTHER","jurisdiction":"Maine","warnings":[]}',
    json.dumps({"plate": "ABC123", "jurisdiction": "Maine", "warnings": [], "owner": "private-name"}),
    json.dumps({"plate": ["ABC123", "DEF456"], "jurisdiction": "Maine", "warnings": []}),
    json.dumps({"plate": "ABC?123", "jurisdiction": "Maine", "warnings": []}),
    json.dumps({"plate": "ABC/123", "jurisdiction": "Maine", "warnings": []}),
    json.dumps({"plate": "- -", "jurisdiction": "Maine", "warnings": []}),
    json.dumps({"plate": "X" * 21, "jurisdiction": "Maine", "warnings": []}),
    json.dumps({"plate": "ABC123", "jurisdiction": "X" * 81, "warnings": []}),
    json.dumps({"plate": "ABC123", "jurisdiction": "A", "warnings": []}),
    json.dumps({"plate": "ABC123", "jurisdiction": "Maine\x00hidden", "warnings": []}),
    json.dumps({"plate": "ABC123", "jurisdiction": "Maine", "warnings": "private-warning"}),
    json.dumps({"plate": "ABC123", "jurisdiction": "Maine", "warnings": [None]}),
    json.dumps({"plate": "ABC123", "jurisdiction": "Maine", "warnings": ["x" * 301]}),
    json.dumps({"plate": "ABC123", "jurisdiction": "Maine", "warnings": ["warning"] * 9}),
])
def test_invalid_model_results_have_safe_error(content):
    with pytest.raises(providers.ProviderError) as error:
        photos._read_result({"content": content})
    assert "private" not in str(error.value)
    assert "vision-capable" in str(error.value)


def test_unexpected_tool_calls_are_rejected():
    with pytest.raises(providers.ProviderError):
        photos._read_result({
            "content": '{"plate":"ABC123","jurisdiction":"Maine","warnings":[]}',
            "tool_calls": [{"function": {"name": "search", "arguments": {}}}],
        })


@pytest.mark.parametrize("provider", ["openrouter", "ollama"])
async def test_vision_rejection_is_actionable_and_does_not_leak_upstream_data(mock_http, provider):
    mock_http(lambda _: httpx.Response(400, json={"error": "model does not support images private-upstream-key"}))
    with pytest.raises(providers.ProviderError) as error:
        await read_plate(photo_request(provider=provider, api_key="private-api-key"))
    assert "vision-capable" in str(error.value)
    assert "private" not in str(error.value)


async def test_provider_authentication_error_remains_safe(mock_http):
    mock_http(lambda _: httpx.Response(401, json={"error": "private-api-key rejected"}))
    with pytest.raises(providers.ProviderError, match="authentication") as error:
        await read_plate(photo_request(provider="openrouter", api_key="private-api-key"))
    assert "private-api-key" not in str(error.value)
