"""Safe, non-rendering HTML text extraction helpers."""

from __future__ import annotations

from html.parser import HTMLParser
from typing import ClassVar


class _EmailHTMLTextParser(HTMLParser):
    """Extract visible text and links without rendering or executing HTML."""

    _IGNORED_TAGS: ClassVar[set[str]] = {"script", "style", "noscript", "template"}
    _URL_ATTRIBUTES: ClassVar[set[str]] = {"href", "src", "action"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._ignored_depth = 0
        self.text_parts: list[str] = []
        self.urls: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in self._IGNORED_TAGS:
            self._ignored_depth += 1
            return
        if self._ignored_depth:
            return
        for name, value in attrs:
            if name.lower() in self._URL_ATTRIBUTES and value:
                self.urls.append(value)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self._IGNORED_TAGS and self._ignored_depth:
            self._ignored_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth:
            self.text_parts.append(data)


def extract_html_text_and_urls(html: str) -> tuple[str, tuple[str, ...]]:
    """Return visible HTML text and URL attributes without network access.

    ``HTMLParser`` only tokenizes the supplied string. It does not execute
    JavaScript, fetch remote resources, or render the message.
    """

    parser = _EmailHTMLTextParser()
    parser.feed(html)
    parser.close()
    return " ".join(parser.text_parts), tuple(parser.urls)
