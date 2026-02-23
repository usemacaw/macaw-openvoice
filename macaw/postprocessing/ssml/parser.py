"""SSML parser — extracts supported tags into engine-neutral directives.

Uses ``defusedxml`` for safe XML parsing. Unsupported
tags are stripped (fail-open), preserving their inner text content.

Supported subset (see ADR-009):
- ``<break time="500ms"/>``
- ``<prosody rate="slow" pitch="high">text</prosody>``
- ``<emphasis level="strong">text</emphasis>``
- ``<say-as interpret-as="cardinal">42</say-as>``
- ``<phoneme alphabet="ipa" ph="...">text</phoneme>``
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET  # nosec B405 — only Element/ParseError used, parsing via defusedxml

import defusedxml.ElementTree as SafeET

from macaw.postprocessing.ssml.directives import (
    BreakDirective,
    EmphasisDirective,
    PhonemeDirective,
    ProsodyDirective,
    SayAsDirective,
    SSMLDirective,
)

# Regex to parse time values like "500ms", "1.5s", "2s"
_TIME_PATTERN = re.compile(r"^(\d+(?:\.\d+)?)\s*(ms|s)$")

# Supported emphasis levels
_EMPHASIS_LEVELS = frozenset({"strong", "moderate", "reduced", "none"})

# Tags that are containers (have children that should be recursively walked)
_CONTAINER_TAGS = frozenset({"prosody", "emphasis", "say-as", "phoneme"})


class SSMLParseError(ValueError):
    """Raised when SSML input cannot be parsed."""


class SSMLParseResult:
    """Result of SSML parsing: plain text + extracted directives.

    Attributes:
        text: The plain text with SSML tags removed.
        directives: Ordered list of extracted SSML directives.
    """

    __slots__ = ("directives", "text")

    def __init__(self, text: str, directives: list[SSMLDirective]) -> None:
        self.text = text
        self.directives = directives


class SSMLParser:
    """Stateless SSML parser.

    Parses a subset of SSML tags into engine-neutral directives.
    Unsupported tags are stripped, their inner text preserved.

    Usage::

        parser = SSMLParser()
        result = parser.parse('<speak>Hello <break time="500ms"/> world</speak>')
        print(result.text)        # "Hello  world"
        print(result.directives)  # [BreakDirective(time_ms=500, position=6)]
    """

    def parse(self, ssml_input: str) -> SSMLParseResult:
        """Parse SSML string into plain text and directives.

        Args:
            ssml_input: SSML input text. May or may not be wrapped in
                ``<speak>`` tags.

        Returns:
            SSMLParseResult with extracted plain text and directives.

        Raises:
            SSMLParseError: If the input is not valid XML.
        """
        wrapped = self._ensure_speak_root(ssml_input)
        try:
            root = SafeET.fromstring(wrapped)
        except ET.ParseError as exc:
            raise SSMLParseError(f"Invalid SSML: {exc}") from exc

        directives: list[SSMLDirective] = []
        text_parts: list[str] = []

        self._walk_children(root, text_parts, directives)

        plain_text = "".join(text_parts)
        return SSMLParseResult(text=plain_text, directives=directives)

    def _ensure_speak_root(self, ssml_input: str) -> str:
        """Wrap input in ``<speak>`` if not already wrapped."""
        stripped = ssml_input.strip()
        if stripped.startswith("<speak"):
            return stripped
        return f"<speak>{stripped}</speak>"

    def _walk_children(
        self,
        element: ET.Element,
        text_parts: list[str],
        directives: list[SSMLDirective],
    ) -> None:
        """Walk an element's content: its .text and all children.

        Handles .text before children and .tail after each child.
        This is the standard ElementTree traversal pattern.
        """
        if element.text:
            text_parts.append(element.text)
        for child in element:
            self._walk_element(child, text_parts, directives)
            if child.tail:
                text_parts.append(child.tail)

    def _walk_element(
        self,
        element: ET.Element,
        text_parts: list[str],
        directives: list[SSMLDirective],
    ) -> None:
        """Process a single element (not the root <speak>)."""
        tag = _strip_namespace(element.tag)

        if tag == "break":
            _handle_break(element, text_parts, directives)
        elif tag in _CONTAINER_TAGS:
            # Container tags: record directive, then recurse into children
            self._handle_container(tag, element, text_parts, directives)
        else:
            # Unsupported tag: strip tag, recurse into children (fail-open)
            self._walk_children(element, text_parts, directives)

    def _handle_container(
        self,
        tag: str,
        element: ET.Element,
        text_parts: list[str],
        directives: list[SSMLDirective],
    ) -> None:
        """Handle a container tag that may have nested children.

        Records the directive using the flattened inner text, then
        recursively walks children to extract nested directives.
        """
        pos = _current_position(text_parts)

        # Collect inner text by walking children into text_parts directly.
        # This allows nested <break> etc. to emit their own directives.
        pre_len = len(text_parts)
        self._walk_children(element, text_parts, directives)
        inner_text = "".join(text_parts[pre_len:])

        # Now create the container's own directive
        if tag == "prosody":
            rate = element.get("rate")
            pitch = element.get("pitch")
            if rate is not None or pitch is not None:
                directives.append(
                    ProsodyDirective(text=inner_text, rate=rate, pitch=pitch, position=pos)
                )

        elif tag == "emphasis":
            level_str = element.get("level", "moderate")
            level = level_str if level_str in _EMPHASIS_LEVELS else "moderate"
            directives.append(
                EmphasisDirective(
                    text=inner_text,
                    level=level,  # type: ignore[arg-type]
                    position=pos,
                )
            )

        elif tag == "say-as":
            interpret_as = element.get("interpret-as")
            if interpret_as is not None:
                directives.append(
                    SayAsDirective(
                        text=inner_text,
                        interpret_as=interpret_as,
                        format_str=element.get("format"),
                        position=pos,
                    )
                )
            # If interpret-as is missing, text was already added by _walk_children

        elif tag == "phoneme":
            ph = element.get("ph")
            if ph is not None:
                directives.append(
                    PhonemeDirective(
                        text=inner_text,
                        alphabet=element.get("alphabet", "ipa"),
                        ph=ph,
                        position=pos,
                    )
                )
            # If ph is missing, text was already added by _walk_children


def _strip_namespace(tag: str) -> str:
    """Remove XML namespace prefix from tag name."""
    if tag.startswith("{"):
        return tag.split("}", 1)[1]
    return tag


def _current_position(text_parts: list[str]) -> int:
    """Calculate current character position from accumulated text parts."""
    return sum(len(p) for p in text_parts)


# ---------------------------------------------------------------------------
# Leaf tag handlers
# ---------------------------------------------------------------------------


def _handle_break(
    element: ET.Element,
    text_parts: list[str],
    directives: list[SSMLDirective],
) -> None:
    """Handle ``<break time="500ms"/>``."""
    time_attr = element.get("time")
    if time_attr is None:
        # Default break: 250ms (W3C SSML default for medium break)
        directives.append(BreakDirective(time_ms=250, position=_current_position(text_parts)))
        return

    time_ms = _parse_time_to_ms(time_attr)
    if time_ms is not None:
        directives.append(BreakDirective(time_ms=time_ms, position=_current_position(text_parts)))


def _parse_time_to_ms(value: str) -> int | None:
    """Parse a time string like '500ms' or '1.5s' to milliseconds."""
    match = _TIME_PATTERN.match(value.strip())
    if match is None:
        return None
    number = float(match.group(1))
    unit = match.group(2)
    if unit == "s":
        return int(number * 1000)
    return int(number)
