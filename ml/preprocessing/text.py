"""HTML cleanup and text normalisation for Stack Exchange post bodies."""

from __future__ import annotations

import hashlib
import html
import re
import unicodedata
import warnings
from dataclasses import dataclass, field
from urllib.parse import urlparse

from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

ZERO_WIDTH_RE = re.compile("[\u200b\u200c\u200d\u2060\ufeff]")
WS_RE = re.compile(r"[ \t\r\f\v]+")
BLANK_LINES_RE = re.compile(r"\n\s*\n+")
QUESTION_LINK_RE = re.compile(
    r"(?:https?://(?:www\.)?apple\.stackexchange\.com)?/(?:questions|q)/(\d+)(?:[/?#]|$)",
    re.IGNORECASE,
)
SMART_CHARS = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u2026": "...",
    }
)
SITE_HOSTS = {"", "apple.stackexchange.com", "www.apple.stackexchange.com"}


@dataclass
class CleanedBody:
    text: str
    code: str
    question_links: list[int] = field(default_factory=list)


def normalize_text(text: str | None) -> str:
    """Unicode NFKC, unescape entities, unify quotes/dashes, strip zero-width chars, tidy spaces."""
    if not text:
        return ""
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = ZERO_WIDTH_RE.sub("", text).translate(SMART_CHARS)
    lines = [WS_RE.sub(" ", line).strip() for line in text.split("\n")]
    return BLANK_LINES_RE.sub("\n\n", "\n".join(lines)).strip()


def extract_question_links(hrefs: list[str], self_id: int | None = None) -> list[int]:
    """Ask Different question ids linked from a post. Answer links (/a/<id>) are ignored."""
    ids: list[int] = []
    for href in hrefs:
        href = (href or "").strip()
        if urlparse(href).netloc.lower() not in SITE_HOSTS:
            continue
        m = QUESTION_LINK_RE.search(href)
        if not m:
            continue
        qid = int(m.group(1))
        if qid != self_id and qid not in ids:
            ids.append(qid)
    return ids


def clean_html(body_html: str | None, self_id: int | None = None) -> CleanedBody:
    """Split an HTML body into prose text, code-block text and linked question ids.

    ``<pre>`` blocks go to ``code`` (they often hold logs and error output); inline ``<code>``
    stays in the prose. Images become ``[image]`` placeholders.
    """
    if not body_html:
        return CleanedBody("", "", [])
    soup = BeautifulSoup(body_html, "lxml")
    hrefs = [a.get("href", "") for a in soup.find_all("a")]
    code_blocks = []
    for pre in soup.find_all("pre"):
        code_blocks.append(pre.get_text("\n"))
        pre.decompose()
    for img in soup.find_all("img"):
        img.replace_with(" [image] ")
    for br in soup.find_all("br"):
        br.replace_with("\n")
    for block in soup.find_all(["p", "li", "h1", "h2", "h3", "h4", "blockquote", "div"]):
        block.insert_after("\n")
    return CleanedBody(
        text=normalize_text(soup.get_text()),
        code=normalize_text("\n\n".join(code_blocks)),
        question_links=extract_question_links(hrefs, self_id),
    )


def content_hash(title: str, body: str) -> str:
    """Hash of case- and whitespace-insensitive title+body, for exact-content duplicate detection."""
    key = re.sub(r"\s+", " ", f"{title}\n{body}".lower()).strip()
    return hashlib.sha1(key.encode("utf-8")).hexdigest()
