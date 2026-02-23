"""Tests for TTSTextBuffer — incremental text accumulation for TTS synthesis."""

from __future__ import annotations

from macaw.server.tts_text_buffer import TTSTextBuffer

# ---------------------------------------------------------------------------
# Basic append / flush / clear
# ---------------------------------------------------------------------------


class TestTTSTextBufferBasic:
    """Basic operations: append, flush, clear."""

    def test_empty_buffer_flush_returns_none(self) -> None:
        buf = TTSTextBuffer()
        assert buf.flush() is None

    def test_append_stores_text(self) -> None:
        buf = TTSTextBuffer(split_strategy="none")
        segments = buf.append("hello ", "req1")
        assert segments == []
        assert buf.pending_text == "hello "

    def test_append_then_flush(self) -> None:
        buf = TTSTextBuffer(split_strategy="none")
        buf.append("hello world", "req1")
        result = buf.flush()
        assert result == "hello world"
        assert buf.pending_text == ""

    def test_flush_strips_whitespace(self) -> None:
        buf = TTSTextBuffer(split_strategy="none")
        buf.append("  hello  ", "req1")
        result = buf.flush()
        assert result == "hello"

    def test_flush_empty_after_flush(self) -> None:
        buf = TTSTextBuffer(split_strategy="none")
        buf.append("text", "req1")
        buf.flush()
        assert buf.flush() is None

    def test_clear_empties_buffer(self) -> None:
        buf = TTSTextBuffer(split_strategy="none")
        buf.append("text", "req1")
        buf.clear()
        assert buf.pending_text == ""
        assert buf.request_id is None
        assert buf.flush() is None

    def test_clear_after_append_leaves_buffer_empty(self) -> None:
        buf = TTSTextBuffer(split_strategy="none")
        buf.append("first ", "req1")
        buf.append("second", "req1")
        buf.clear()
        assert buf.pending_text == ""

    def test_request_id_set_on_append(self) -> None:
        buf = TTSTextBuffer()
        buf.append("text", "req_42")
        assert buf.request_id == "req_42"

    def test_pending_text_reflects_buffer_state(self) -> None:
        buf = TTSTextBuffer(split_strategy="none")
        assert buf.pending_text == ""
        buf.append("chunk1", "req1")
        assert buf.pending_text == "chunk1"
        buf.append(" chunk2", "req1")
        assert buf.pending_text == "chunk1 chunk2"

    def test_multiple_appends_concatenate(self) -> None:
        buf = TTSTextBuffer(split_strategy="none")
        buf.append("Hello", "req1")
        buf.append(" world", "req1")
        assert buf.flush() == "Hello world"


# ---------------------------------------------------------------------------
# Sentence split strategy
# ---------------------------------------------------------------------------


class TestSentenceSplitStrategy:
    """Sentence split: splits on `.` `!` `?` followed by whitespace or end."""

    def test_no_split_on_incomplete_sentence(self) -> None:
        buf = TTSTextBuffer(split_strategy="sentence")
        segments = buf.append("Hello world", "req1")
        assert segments == []
        assert buf.pending_text == "Hello world"

    def test_split_on_period_space(self) -> None:
        buf = TTSTextBuffer(split_strategy="sentence")
        segments = buf.append("Hello world. How are you?", "req1")
        # "Hello world." is a complete sentence, "How are you?" stays in buffer
        assert len(segments) == 1
        assert segments[0] == "Hello world."
        assert buf.pending_text == "How are you?"

    def test_split_on_exclamation(self) -> None:
        buf = TTSTextBuffer(split_strategy="sentence")
        segments = buf.append("Wow! Amazing!", "req1")
        # "Wow!" splits off, but "Amazing!" has no trailing space -> stays
        # Actually the regex splits on punct followed by space, so:
        # "Wow! Amazing!" -> split after "!" space -> ["Wow!", "Amazing!"]
        # But "Amazing!" has no trailing space, it stays in buffer
        # Wait, let me re-check: `(?<=[.!?])\s+` applied to "Wow! Amazing!"
        # => splits at the whitespace after "!": ["Wow!", "Amazing!"]
        # parts[:-1] = ["Wow!"], parts[-1] = "Amazing!"
        assert len(segments) == 1
        assert segments[0] == "Wow!"
        assert buf.pending_text == "Amazing!"

    def test_split_on_question_mark(self) -> None:
        buf = TTSTextBuffer(split_strategy="sentence")
        segments = buf.append("Is it working? Yes it is.", "req1")
        assert len(segments) == 1
        assert segments[0] == "Is it working?"
        assert buf.pending_text == "Yes it is."

    def test_multiple_sentences_in_one_append(self) -> None:
        buf = TTSTextBuffer(split_strategy="sentence")
        segments = buf.append("First. Second. Third.", "req1")
        assert len(segments) == 2
        assert segments[0] == "First."
        assert segments[1] == "Second."
        assert buf.pending_text == "Third."

    def test_incremental_sentence_building(self) -> None:
        buf = TTSTextBuffer(split_strategy="sentence")
        # Token by token
        segments = buf.append("Hello", "req1")
        assert segments == []
        segments = buf.append(" world", "req1")
        assert segments == []
        segments = buf.append(".", "req1")
        assert segments == []  # No trailing space yet
        segments = buf.append(" How", "req1")
        assert len(segments) == 1
        assert segments[0] == "Hello world."
        assert buf.pending_text == "How"

    def test_period_without_space_does_not_split(self) -> None:
        buf = TTSTextBuffer(split_strategy="sentence")
        segments = buf.append("3.14 is pi", "req1")
        assert segments == []
        # The period is not followed by a space (it's followed by "1")
        # Actually "3.14 is pi" -> the regex finds "." followed by "1" which is \S, not \s
        # But wait, "." followed by "1" is not \s+, so no split. Correct.

    def test_default_strategy_is_sentence(self) -> None:
        buf = TTSTextBuffer()
        assert buf.split_strategy == "sentence"


# ---------------------------------------------------------------------------
# Paragraph split strategy
# ---------------------------------------------------------------------------


class TestParagraphSplitStrategy:
    """Paragraph split: splits on double newline."""

    def test_no_split_without_double_newline(self) -> None:
        buf = TTSTextBuffer(split_strategy="paragraph")
        segments = buf.append("Hello\nworld", "req1")
        assert segments == []

    def test_split_on_double_newline(self) -> None:
        buf = TTSTextBuffer(split_strategy="paragraph")
        segments = buf.append("First paragraph.\n\nSecond paragraph.", "req1")
        assert len(segments) == 1
        assert segments[0] == "First paragraph."
        assert buf.pending_text == "Second paragraph."

    def test_split_on_triple_newline(self) -> None:
        buf = TTSTextBuffer(split_strategy="paragraph")
        segments = buf.append("Para one.\n\n\nPara two.", "req1")
        assert len(segments) == 1
        assert segments[0] == "Para one."

    def test_multiple_paragraphs(self) -> None:
        buf = TTSTextBuffer(split_strategy="paragraph")
        segments = buf.append("One.\n\nTwo.\n\nThree.", "req1")
        assert len(segments) == 2
        assert segments[0] == "One."
        assert segments[1] == "Two."
        assert buf.pending_text == "Three."


# ---------------------------------------------------------------------------
# None split strategy
# ---------------------------------------------------------------------------


class TestNoneSplitStrategy:
    """None strategy: never auto-splits."""

    def test_no_auto_split(self) -> None:
        buf = TTSTextBuffer(split_strategy="none")
        segments = buf.append("Hello. World. Foo.", "req1")
        assert segments == []
        assert buf.pending_text == "Hello. World. Foo."

    def test_no_auto_split_paragraph(self) -> None:
        buf = TTSTextBuffer(split_strategy="none")
        segments = buf.append("Hello\n\nWorld", "req1")
        assert segments == []

    def test_manual_flush_still_works(self) -> None:
        buf = TTSTextBuffer(split_strategy="none")
        buf.append("Hello world", "req1")
        result = buf.flush()
        assert result == "Hello world"


# ---------------------------------------------------------------------------
# Request ID change auto-flush
# ---------------------------------------------------------------------------


class TestRequestIdAutoFlush:
    """Request ID change triggers auto-flush of previous buffer."""

    def test_request_id_change_auto_flushes(self) -> None:
        buf = TTSTextBuffer(split_strategy="none")
        buf.append("old text", "req1")
        segments = buf.append("new text", "req2")
        assert len(segments) == 1
        assert segments[0] == "old text"
        assert buf.request_id == "req2"
        assert buf.pending_text == "new text"

    def test_same_request_id_no_auto_flush(self) -> None:
        buf = TTSTextBuffer(split_strategy="none")
        buf.append("part1 ", "req1")
        segments = buf.append("part2", "req1")
        assert segments == []
        assert buf.pending_text == "part1 part2"

    def test_request_id_change_with_empty_buffer(self) -> None:
        buf = TTSTextBuffer(split_strategy="none")
        buf.append("text", "req1")
        buf.flush()
        # Buffer is now empty, change request_id
        segments = buf.append("new", "req2")
        # No auto-flush since buffer was empty (but request_id was reset by flush)
        assert segments == []
        assert buf.pending_text == "new"

    def test_request_id_change_whitespace_only_not_flushed(self) -> None:
        buf = TTSTextBuffer(split_strategy="none")
        buf.append("   ", "req1")
        segments = buf.append("new text", "req2")
        # Whitespace-only buffer flushes to None, so no segment returned
        assert segments == []
        assert buf.request_id == "req2"

    def test_request_id_change_combined_with_split(self) -> None:
        buf = TTSTextBuffer(split_strategy="sentence")
        buf.append("Old sentence", "req1")
        segments = buf.append("New sentence. More text.", "req2")
        # First: auto-flush "Old sentence" from req1
        # Then: append "New sentence. More text." and split
        # "New sentence." is a complete segment
        assert len(segments) >= 1
        assert segments[0] == "Old sentence"
        # The sentence split may also produce "New sentence."
        if len(segments) > 1:
            assert segments[1] == "New sentence."


# ---------------------------------------------------------------------------
# Properties and configuration
# ---------------------------------------------------------------------------


class TestProperties:
    """Properties and configurability."""

    def test_split_strategy_property(self) -> None:
        buf = TTSTextBuffer(split_strategy="paragraph")
        assert buf.split_strategy == "paragraph"

    def test_split_strategy_setter(self) -> None:
        buf = TTSTextBuffer(split_strategy="sentence")
        buf.split_strategy = "none"
        assert buf.split_strategy == "none"

    def test_flush_timeout_ms_property(self) -> None:
        buf = TTSTextBuffer(flush_timeout_ms=3000)
        assert buf.flush_timeout_ms == 3000

    def test_flush_timeout_ms_setter(self) -> None:
        buf = TTSTextBuffer()
        buf.flush_timeout_ms = 10000
        assert buf.flush_timeout_ms == 10000

    def test_default_flush_timeout(self) -> None:
        buf = TTSTextBuffer()
        assert buf.flush_timeout_ms == 5000

    def test_initial_request_id_is_none(self) -> None:
        buf = TTSTextBuffer()
        assert buf.request_id is None

    def test_initial_pending_text_is_empty(self) -> None:
        buf = TTSTextBuffer()
        assert buf.pending_text == ""


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and robustness."""

    def test_empty_segments_filtered(self) -> None:
        buf = TTSTextBuffer(split_strategy="paragraph")
        # Double newline at start creates empty first part
        segments = buf.append("\n\nHello", "req1")
        # parts would be ["", "Hello"], complete = [""], remainder = "Hello"
        # Empty string stripped -> filtered out
        assert all(s.strip() for s in segments)

    def test_flush_after_clear_returns_none(self) -> None:
        buf = TTSTextBuffer()
        buf.append("text", "req1")
        buf.clear()
        assert buf.flush() is None

    def test_append_after_flush_works(self) -> None:
        buf = TTSTextBuffer(split_strategy="none")
        buf.append("first", "req1")
        buf.flush()
        buf.append("second", "req1")
        assert buf.pending_text == "second"
        assert buf.flush() == "second"

    def test_strategy_change_mid_buffer(self) -> None:
        buf = TTSTextBuffer(split_strategy="none")
        buf.append("Hello. World. ", "req1")
        # No split with "none"
        assert buf.pending_text == "Hello. World. "
        # Change to sentence strategy
        buf.split_strategy = "sentence"
        # Next append triggers split evaluation
        segments = buf.append("More text.", "req1")
        # The buffer is now "Hello. World. More text."
        # Split at ". " boundaries: ["Hello.", "World.", "More text."]
        # complete = ["Hello.", "World."], remainder = "More text."
        assert len(segments) == 2
        assert segments[0] == "Hello."
        assert segments[1] == "World."

    def test_unicode_text(self) -> None:
        buf = TTSTextBuffer(split_strategy="sentence")
        segments = buf.append("Ola mundo. Como vai?", "req1")
        assert len(segments) == 1
        assert segments[0] == "Ola mundo."
