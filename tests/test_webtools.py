import asyncio
import socket
from unittest.mock import AsyncMock

import httpx
import pytest

from platetrace import webtools


@pytest.fixture
def mock_http(monkeypatch):
    real_client = httpx.AsyncClient

    def install(handler):
        def client(**kwargs):
            assert kwargs["trust_env"] is False
            assert kwargs["follow_redirects"] is False
            return real_client(transport=httpx.MockTransport(handler), **kwargs)

        monkeypatch.setattr(webtools.httpx, "AsyncClient", client)

    return install


def address_info(ip):
    return (socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 443))


@pytest.mark.parametrize(
    "addresses", [["127.0.0.1"], ["10.0.0.1"], ["169.254.169.254"], ["::1"], ["93.184.216.34", "192.168.1.1"]]
)
async def test_dns_rejects_private_addresses_including_mixed_answers(monkeypatch, addresses):
    lookup = AsyncMock(return_value=[address_info(ip) for ip in addresses])
    monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", lookup)
    with pytest.raises(webtools.WebError, match="private"):
        await webtools.public_address("https://vehicles.example/source")
    lookup.assert_awaited_once()


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "http://host.example:8000/",
        "https://u:p@host.example",
        "http://[invalid",
        "https://host.example/" + "x" * 2000,
    ],
)
async def test_invalid_urls_are_rejected_before_dns(monkeypatch, url):
    lookup = AsyncMock()
    monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", lookup)
    with pytest.raises(webtools.WebError):
        await webtools.public_address(url)
    lookup.assert_not_awaited()


async def test_request_is_pinned_to_resolved_public_ip_with_original_host_and_tls_name(
    monkeypatch, mock_http
):
    lookup = AsyncMock(return_value=[address_info("93.184.216.34")])
    monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", lookup)

    def handler(request):
        assert request.url.host == "93.184.216.34"
        assert request.headers["host"] == "vehicles.example"
        assert request.extensions["sni_hostname"] == "vehicles.example"
        return httpx.Response(200, headers={"content-type": "text/plain"}, text="Evidence")

    mock_http(handler)
    assert await webtools.public_get("https://vehicles.example/source") == (
        "https://vehicles.example/source",
        "Evidence",
        "text/plain",
    )
    lookup.assert_awaited_once()


async def test_redirect_to_private_host_is_revalidated_before_network(monkeypatch, mock_http):
    lookup = AsyncMock(side_effect=[[address_info("93.184.216.34")], [address_info("127.0.0.1")]])
    monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", lookup)
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(302, headers={"location": "http://localhost/private"})

    mock_http(handler)
    with pytest.raises(webtools.WebError, match="private"):
        await webtools.public_get("https://vehicles.example/source")
    assert len(requests) == 1
    assert lookup.await_count == 2


@pytest.mark.parametrize(
    "destination", ["https://destination.example/source", "http://search.example/source"]
)
async def test_cross_origin_redirect_drops_search_credentials(monkeypatch, mock_http, destination):
    monkeypatch.setattr(
        webtools,
        "public_address",
        AsyncMock(
            side_effect=[
                ("93.184.216.34", "search.example"),
                ("93.184.216.35", "destination.example"),
            ]
        ),
    )
    requests = []

    def handler(request):
        requests.append(request)
        if len(requests) == 1:
            assert request.headers["x-subscription-token"] == "private-search-key"
            return httpx.Response(302, headers={"location": destination})
        assert "x-subscription-token" not in request.headers
        return httpx.Response(200, headers={"content-type": "text/plain"}, text="Evidence")

    mock_http(handler)
    await webtools.public_get("https://search.example/query", {"X-Subscription-Token": "private-search-key"})
    assert len(requests) == 2


@pytest.mark.parametrize(
    "response,expected",
    [
        (httpx.Response(200, headers={"content-type": "text/plain"}, content=b"x" * 1_000_001), "1 MB"),
        (
            httpx.Response(200, headers={"content-type": "application/octet-stream"}, content=b"binary"),
            "supports HTML",
        ),
        (
            httpx.Response(403, headers={"content-type": "text/html"}, text="Private source body"),
            "No access bypass",
        ),
        (httpx.Response(302), "without a destination"),
    ],
)
async def test_response_size_mime_and_access_limits(monkeypatch, mock_http, response, expected):
    monkeypatch.setattr(
        webtools, "public_address", AsyncMock(return_value=("93.184.216.34", "vehicles.example"))
    )
    mock_http(lambda _: response)
    with pytest.raises(webtools.WebError, match=expected):
        await webtools.public_get("https://vehicles.example/source")


async def test_redirect_loop_is_bounded(monkeypatch, mock_http):
    address = AsyncMock(return_value=("93.184.216.34", "vehicles.example"))
    monkeypatch.setattr(webtools, "public_address", address)
    mock_http(lambda _: httpx.Response(302, headers={"location": "/again"}))
    with pytest.raises(webtools.WebError, match="Too many redirects"):
        await webtools.public_get("https://vehicles.example/source")
    assert address.await_count == 6


async def test_html_reader_removes_executable_content_and_bounds_text_links(monkeypatch):
    body = "<title>Public specifications</title><script>SECRET_SCRIPT</script><nav>NAVIGATION</nav>"
    body += "<p>" + "x" * 23000 + "</p>"
    body += "".join(f'<a href="/source/{index}">Source {index}</a>' for index in range(50))
    monkeypatch.setattr(
        webtools, "public_get", AsyncMock(return_value=("https://vehicles.example", body, "text/html"))
    )
    result = await webtools.fetch_page("https://vehicles.example")
    assert result["title"] == "Public specifications"
    assert len(result["text"]) == 22000
    assert result["truncated"] is True
    assert "SECRET_SCRIPT" not in result["text"] and "NAVIGATION" not in result["text"]
    assert len(result["links"]) == 35
    assert result["links"][0]["url"] == "https://vehicles.example/source/0"
    assert "Untrusted source text" in result["notice"]


async def test_search_snippets_stay_discovery_leads_and_captcha_is_not_bypassed(monkeypatch):
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
    body = '<div class="result"><a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fvehicles.example%2Fspecs">Vehicle</a><div class="result__snippet">Possible match</div></div>'
    get = AsyncMock(return_value=("https://html.duckduckgo.com/html/", body, "text/html"))
    monkeypatch.setattr(webtools, "public_get", get)
    results = await webtools.search_web("ABC123 vehicle")
    assert results["results"][0]["url"] == "https://vehicles.example/specs"
    assert "fetch original sources" in results["notice"]
    get.return_value = ("https://html.duckduckgo.com/html/", "CAPTCHA challenge", "text/html")
    with pytest.raises(webtools.WebError, match="No CAPTCHA bypass"):
        await webtools.search_web("ABC123 vehicle")
