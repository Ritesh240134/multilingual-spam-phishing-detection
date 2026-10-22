"""Static URL extraction and risk indicators with no URL fetching."""

from __future__ import annotations

import ipaddress
import re
from urllib.parse import urlsplit

from .schemas import URLFeatures

_URL_PATTERN = re.compile(r"(?i)\b(?:https?://|www\.)[^\s<>\"']+")
_TRAILING_PUNCTUATION = ".,;:!?)]}>\"'"
_SHORTENER_DOMAINS = {
    "bit.ly",
    "buff.ly",
    "goo.gl",
    "is.gd",
    "ow.ly",
    "rb.gy",
    "rebrand.ly",
    "t.co",
    "tinyurl.com",
    "tiny.one",
}
_SUSPICIOUS_TLDS = {"click", "country", "gq", "link", "live", "monster", "rest", "top", "work", "xyz", "zip"}


def extract_urls(text: str, html_urls: tuple[str, ...] = ()) -> tuple[str, ...]:
    """Extract URL-like values from supplied text and HTML attributes only."""

    candidates = [* _URL_PATTERN.findall(text), *html_urls]
    normalized: list[str] = []
    for candidate in candidates:
        cleaned = candidate.strip().rstrip(_TRAILING_PUNCTUATION)
        if cleaned:
            normalized.append(cleaned)
    return tuple(normalized)


def _hostname(url: str) -> str | None:
    """Return a hostname without performing DNS resolution."""

    candidate = url if re.match(r"(?i)^https?://", url) else f"http://{url}"
    try:
        return urlsplit(candidate).hostname
    except ValueError:
        return None


def _is_ip_address(hostname: str | None) -> bool:
    if not hostname:
        return False
    try:
        ipaddress.ip_address(hostname)
        return True
    except ValueError:
        return False


def _is_suspicious(url: str, hostname: str | None, is_ip: bool) -> bool:
    """Evaluate static lexical indicators only; no reputation or network calls."""

    lower_url = url.lower()
    host = (hostname or "").lower().rstrip(".")
    tld = host.rsplit(".", 1)[-1] if "." in host else ""
    has_non_standard_port = False
    try:
        parsed = urlsplit(url if re.match(r"(?i)^https?://", url) else f"http://{url}")
        has_non_standard_port = parsed.port not in (None, 80, 443)
    except ValueError:
        return True
    return any(
        (
            is_ip,
            "@" in url,
            "xn--" in host,
            "%" in lower_url,
            len(url) > 75,
            has_non_standard_port,
            tld in _SUSPICIOUS_TLDS,
        )
    )


def build_url_features(urls: tuple[str, ...]) -> URLFeatures:
    """Build deterministic static indicators from extracted URLs."""

    unique_urls = tuple(dict.fromkeys(urls))
    http_count = sum(url.lower().startswith("http://") for url in urls)
    https_count = sum(url.lower().startswith("https://") for url in urls)
    ip_count = 0
    shortener_count = 0
    suspicious_urls: list[str] = []
    for url in urls:
        hostname = _hostname(url)
        is_ip = _is_ip_address(hostname)
        ip_count += int(is_ip)
        if hostname and hostname.lower().rstrip(".") in _SHORTENER_DOMAINS:
            shortener_count += 1
        if _is_suspicious(url, hostname, is_ip):
            suspicious_urls.append(url)
    lengths = [len(url) for url in urls]
    return URLFeatures(
        url_count=len(urls),
        unique_url_count=len(unique_urls),
        http_url_count=http_count,
        https_url_count=https_count,
        ip_address_url_count=ip_count,
        shortener_url_count=shortener_count,
        suspicious_url_count=len(suspicious_urls),
        longest_url_length=max(lengths, default=0),
        average_url_length=round(sum(lengths) / len(lengths), 2) if lengths else 0.0,
        suspicious_urls=tuple(suspicious_urls),
    )
