"""
Web Scraping & Content Extraction API
--------------------------------------
Accepts a URL, fetches the page, strips clutter (scripts, nav, ads, footers),
and returns structured article data: title, author, publish date, body text.

Security notes:
- Blocks requests to private/internal/loopback IPs (SSRF protection) since
  this API accepts arbitrary user-supplied URLs.
- Enforces request timeout and max response size to avoid hanging or being
  used to DoS your server / download huge files.
- Only allows http/https schemes.
"""

import ipaddress
import re
import socket
from datetime import datetime
from typing import Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from dateutil import parser as date_parser
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REQUEST_TIMEOUT = 8            # seconds — fail fast, don't hang the server
MAX_CONTENT_BYTES = 5_000_000  # 5 MB cap — refuse to download huge pages
USER_AGENT = (
    "Mozilla/5.0 (compatible; ContentExtractorBot/1.0; "
    "+https://example.com/bot)"
)

# Tags that are pure clutter and should never contribute to article text
NOISE_TAGS = [
    "script", "style", "noscript", "iframe", "svg", "form",
    "nav", "footer", "header", "aside", "button", "input",
]

# Class/id substrings commonly used for ads, navigation, and boilerplate.
# Matched case-insensitively against an element's id/class attributes.
NOISE_PATTERNS = re.compile(
    r"(nav|menu|sidebar|footer|header|advert|banner|cookie|popup|"
    r"subscribe|newsletter|social|share|related|comment|widget|breadcrumb)",
    re.IGNORECASE,
)

app = FastAPI(
    title="Content Extraction API",
    description="Extracts clean article content (title, author, date, body) from a URL.",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ExtractRequest(BaseModel):
    url: str = Field(..., description="Full URL of the article to extract, e.g. https://example.com/article")

    @field_validator("url")
    @classmethod
    def validate_scheme(cls, v: str) -> str:
        parsed = urlparse(v)
        if parsed.scheme not in ("http", "https"):
            raise ValueError("URL must start with http:// or https://")
        if not parsed.netloc:
            raise ValueError("URL must include a valid domain")
        return v


class ExtractResponse(BaseModel):
    url: str
    title: Optional[str] = None
    author: Optional[str] = None
    publish_date: Optional[str] = None
    body_text: Optional[str] = None
    word_count: int = 0


# ---------------------------------------------------------------------------
# SSRF protection
# ---------------------------------------------------------------------------

def _is_private_host(hostname: str) -> bool:
    """
    Resolve the hostname and check whether it points at a private, loopback,
    link-local, or reserved IP range. This stops the API being used as a
    proxy to hit internal services (e.g. http://169.254.169.254 metadata
    endpoints, http://localhost, http://10.x.x.x internal servers).
    """
    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        raise HTTPException(status_code=400, detail="Could not resolve host")

    for info in infos:
        ip_str = info[4][0]
        ip = ipaddress.ip_address(ip_str)
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            return True
    return False


def _safe_get(url: str) -> requests.Response:
    """Fetch a URL with SSRF, timeout, and size protections applied."""
    hostname = urlparse(url).hostname
    if not hostname:
        raise HTTPException(status_code=400, detail="Invalid URL")

    if _is_private_host(hostname):
        raise HTTPException(status_code=400, detail="Requests to internal/private addresses are not allowed")

    try:
        resp = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT,
            stream=True,  # stream so we can enforce a size cap before reading it all
            allow_redirects=True,
        )
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Request to target URL timed out")
    except requests.exceptions.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Failed to fetch URL: {exc}")

    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Target site returned status {resp.status_code}")

    content_type = resp.headers.get("Content-Type", "")
    if "text/html" not in content_type:
        raise HTTPException(status_code=415, detail="URL did not return an HTML page")

    # Enforce max size while streaming, instead of trusting Content-Length
    size = 0
    chunks = []
    for chunk in resp.iter_content(chunk_size=8192):
        size += len(chunk)
        if size > MAX_CONTENT_BYTES:
            raise HTTPException(status_code=413, detail="Page too large to process")
        chunks.append(chunk)

    resp._content = b"".join(chunks)
    return resp


# ---------------------------------------------------------------------------
# Extraction logic
# ---------------------------------------------------------------------------

def _clean_soup(soup: BeautifulSoup) -> None:
    """Strip out scripts, nav, ads, and other non-article clutter in place."""
    for tag in soup(NOISE_TAGS):
        tag.decompose()

    # STEP 1: First just COLLECT which tags to remove — don't decompose yet.
    # (Decomposing while still iterating find_all() can crash when a later
    # tag in the list turns out to be a child of one we already removed.)
    tags_to_remove = []
    for tag in soup.find_all(True):
        identifiers = " ".join(tag.get("class", []) + [tag.get("id", "")])
        if identifiers and NOISE_PATTERNS.search(identifiers):
            tags_to_remove.append(tag)

    # STEP 2: Now decompose. If a tag's parent was already removed
    # (e.g. a widget inside a sidebar we just deleted), touching that
    # child raises AttributeError — we just skip it since it's already gone.
    for tag in tags_to_remove:
        try:
            if tag.parent is not None:
                tag.decompose()
        except AttributeError:
            pass  # already removed as part of an ancestor's cleanup

    # Remove HTML comments (often contain hidden/tracking clutter)
    for comment in soup.find_all(string=lambda s: s.__class__.__name__ == "Comment"):
        comment.extract()


def _extract_title(soup: BeautifulSoup) -> Optional[str]:
    og_title = soup.find("meta", property="og:title")
    if og_title and og_title.get("content"):
        return og_title["content"].strip()

    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        return h1.get_text(strip=True)

    if soup.title and soup.title.string:
        return soup.title.string.strip()

    return None


def _extract_author(soup: BeautifulSoup) -> Optional[str]:
    # Common meta tag conventions
    for attrs in (
        {"name": "author"},
        {"property": "article:author"},
        {"name": "byl"},  # some news sites (AP, etc.)
    ):
        tag = soup.find("meta", attrs=attrs)
        if tag and tag.get("content"):
            return tag["content"].strip()

    # Common markup patterns: <a rel="author">, class="author"/"byline"
    candidate = soup.find(attrs={"rel": "author"}) or soup.find(class_=re.compile(r"(author|byline)", re.I))
    if candidate:
        text = candidate.get_text(strip=True)
        if text and len(text) < 100:  # sanity check — avoid grabbing a whole paragraph
            return text

    return None


def _extract_publish_date(soup: BeautifulSoup) -> Optional[str]:
    meta_candidates = [
        {"property": "article:published_time"},
        {"name": "publish-date"},
        {"name": "date"},
        {"itemprop": "datePublished"},
    ]
    for attrs in meta_candidates:
        tag = soup.find("meta", attrs=attrs)
        if tag and tag.get("content"):
            return _normalize_date(tag["content"])

    time_tag = soup.find("time")
    if time_tag:
        raw = time_tag.get("datetime") or time_tag.get_text(strip=True)
        if raw:
            return _normalize_date(raw)

    return None


def _normalize_date(raw: str) -> Optional[str]:
    """Best-effort parse of a date string into ISO 8601 (YYYY-MM-DD)."""
    try:
        return date_parser.parse(raw, fuzzy=True).date().isoformat()
    except (ValueError, OverflowError, TypeError):
        return raw.strip()  # fall back to raw text rather than dropping it


def _extract_body(soup: BeautifulSoup) -> str:
    """
    Heuristic body extraction: prefer <article>, else the <div>/<section>
    with the most cumulative paragraph text (a simple density heuristic
    that works well without a heavy dependency like readability-lxml).
    """
    article_tag = soup.find("article")
    container = article_tag if article_tag else soup.find("body")
    if container is None:
        return ""

    best_container = container
    best_length = len(container.get_text(strip=True))

    if article_tag is None:
        for candidate in soup.find_all(["div", "section", "main"]):
            text_len = sum(
                len(p.get_text(strip=True)) for p in candidate.find_all("p")
            )
            if text_len > best_length:
                best_length = text_len
                best_container = candidate

    paragraphs = [
        p.get_text(" ", strip=True)
        for p in best_container.find_all("p")
        if len(p.get_text(strip=True)) > 30  # skip short caption/nav fragments
    ]

    return "\n\n".join(paragraphs).strip()


# ---------------------------------------------------------------------------
# API endpoint
# ---------------------------------------------------------------------------

@app.post("/extract", response_model=ExtractResponse, summary="Extract article content from a URL")
def extract_content(payload: ExtractRequest):
    """
    Scrape the given URL and return structured article data:
    title, author, publish date, and cleaned body text.
    """
    response = _safe_get(payload.url)

    soup = BeautifulSoup(response.content, "lxml")

    title = _extract_title(soup)
    author = _extract_author(soup)
    publish_date = _extract_publish_date(soup)

    _clean_soup(soup)  # strip clutter AFTER metadata extraction (meta tags live in <head>, which gets removed)
    body_text = _extract_body(soup)

    if not body_text:
        raise HTTPException(status_code=422, detail="Could not extract article body from this page")

    return ExtractResponse(
        url=payload.url,
        title=title,
        author=author,
        publish_date=publish_date,
        body_text=body_text,
        word_count=len(body_text.split()),
    )


@app.get("/health", summary="Health check")
def health_check():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}