"""Tests for LocalAgreementPolicy -- token confirmation by agreement between passes.

Covers:
- Basic behavior: first pass, second pass, agreement, retention
- Monotonic growth of confirmed tokens
- Flush (speech_end) emitting all tokens
- State reset
- Edge cases: empty tokens, complete divergence, min_confirm_passes > 2
- AgreementResult immutability
"""

from __future__ import annotations

import pytest

from macaw.session.local_agreement import AgreementResult, LocalAgreementPolicy

# ---------------------------------------------------------------------------
# Basic agreement tests
# ---------------------------------------------------------------------------


class TestFirstPass:
    """First pass never confirms tokens (needs comparison)."""

    def test_first_pass_confirms_nothing(self) -> None:
        """Arrange: policy with min_confirm_passes=2.
        Act: process first pass with tokens.
        Assert: no tokens confirmed, all retained.
        """
        policy = LocalAgreementPolicy(min_confirm_passes=2)

        result = policy.process_pass(["hello", "how"])

        assert result.confirmed_text == ""
        assert result.retained_text == "hello how"
        assert result.confirmed_tokens == []
        assert result.retained_tokens == ["hello", "how"]
        assert result.is_new_confirmation is False

    def test_first_pass_sets_pass_count(self) -> None:
        """Arrange: new policy.
        Act: process one pass.
        Assert: pass_count is 1.
        """
        policy = LocalAgreementPolicy()

        policy.process_pass(["hello"])

        assert policy.pass_count == 1


class TestSecondPass:
    """Second pass confirms tokens that agree with the first."""

    def test_second_pass_confirms_agreeing_tokens(self) -> None:
        """Arrange: two passes with matching tokens.
        Act: process second pass.
        Assert: agreeing tokens are confirmed.
        """
        policy = LocalAgreementPolicy(min_confirm_passes=2)

        policy.process_pass(["hello", "how"])
        result = policy.process_pass(["hello", "how", "are"])

        assert result.confirmed_text == "hello how"
        assert result.retained_text == "are"
        assert result.confirmed_tokens == ["hello", "how"]
        assert result.retained_tokens == ["are"]
        assert result.is_new_confirmation is True

    def test_diverging_tokens_are_retained(self) -> None:
        """Arrange: second pass diverges at the second token.
        Act: process second pass.
        Assert: only first token confirmed, rest retained.
        """
        policy = LocalAgreementPolicy(min_confirm_passes=2)

        policy.process_pass(["hello", "how"])
        result = policy.process_pass(["hello", "world"])

        assert result.confirmed_text == "hello"
        assert result.retained_text == "world"
        assert result.confirmed_tokens == ["hello"]
        assert result.retained_tokens == ["world"]
        assert result.is_new_confirmation is True


class TestMonotonicGrowth:
    """Confirmed tokens grow monotonically (never decrease)."""

    def test_confirmed_tokens_grow_monotonically(self) -> None:
        """Arrange: several passes with progressive growth.
        Act: process 4 passes.
        Assert: confirmed never decreases between passes.
        """
        policy = LocalAgreementPolicy(min_confirm_passes=2)

        policy.process_pass(["a", "b", "c"])
        r1 = policy.process_pass(["a", "b", "c", "d"])
        assert len(r1.confirmed_tokens) == 3  # a, b, c confirmados

        r2 = policy.process_pass(["a", "b", "c", "d", "e"])
        assert len(r2.confirmed_tokens) == 4  # d agora confirmado tambem

        r3 = policy.process_pass(["a", "b", "c", "d", "e", "f"])
        assert len(r3.confirmed_tokens) == 5  # e agora confirmado tambem

        # Confirm monotonic growth
        assert r1.confirmed_tokens == ["a", "b", "c"]
        assert r2.confirmed_tokens == ["a", "b", "c", "d"]
        assert r3.confirmed_tokens == ["a", "b", "c", "d", "e"]

    def test_confirmed_never_shrink_on_divergence(self) -> None:
        """Arrange: tokens agree in pass 2, diverge in pass 3.
        Act: process 3 passes.
        Assert: already confirmed tokens remain confirmed.
        """
        policy = LocalAgreementPolicy(min_confirm_passes=2)

        policy.process_pass(["hello", "world"])
        r1 = policy.process_pass(["hello", "world"])
        assert r1.confirmed_tokens == ["hello", "world"]

        # Pass 3 completely diverges after the confirmed ones
        r2 = policy.process_pass(["hello", "world", "foo"])
        assert r2.confirmed_tokens == ["hello", "world"]  # did not shrink
        assert r2.retained_tokens == ["foo"]


class TestFlush:
    """Flush (speech_end) emits all tokens."""

    def test_flush_emits_all_tokens(self) -> None:
        """Arrange: policy with confirmed and retained tokens.
        Act: flush.
        Assert: confirmed + retained all become confirmed.
        """
        policy = LocalAgreementPolicy(min_confirm_passes=2)

        policy.process_pass(["hello", "how", "are"])
        policy.process_pass(["hello", "how", "are", "you"])

        result = policy.flush()

        assert result.confirmed_text == "hello how are you"
        assert result.retained_text == ""
        assert result.confirmed_tokens == ["hello", "how", "are", "you"]
        assert result.retained_tokens == []

    def test_flush_resets_state(self) -> None:
        """Arrange: policy with accumulated state.
        Act: flush.
        Assert: pass_count returns to 0, confirmed_text empty.
        """
        policy = LocalAgreementPolicy(min_confirm_passes=2)

        policy.process_pass(["a", "b"])
        policy.process_pass(["a", "b", "c"])
        policy.flush()

        assert policy.pass_count == 0
        assert policy.confirmed_text == ""

    def test_flush_empty_state(self) -> None:
        """Arrange: policy with no passes processed.
        Act: flush.
        Assert: empty result without errors.
        """
        policy = LocalAgreementPolicy()

        result = policy.flush()

        assert result.confirmed_text == ""
        assert result.retained_text == ""
        assert result.confirmed_tokens == []
        assert result.retained_tokens == []
        assert result.is_new_confirmation is False

    def test_flush_after_single_pass(self) -> None:
        """Arrange: flush after only 1 pass (nothing confirmed).
        Act: flush.
        Assert: all tokens from the single pass are emitted.
        """
        policy = LocalAgreementPolicy(min_confirm_passes=2)

        policy.process_pass(["hello", "world"])
        result = policy.flush()

        assert result.confirmed_text == "hello world"
        assert result.confirmed_tokens == ["hello", "world"]
        assert result.is_new_confirmation is True


class TestEdgeCases:
    """Edge cases: empty tokens, complete divergence, etc."""

    def test_empty_pass(self) -> None:
        """Arrange: pass with empty token list.
        Act: process empty pass.
        Assert: empty result without errors.
        """
        policy = LocalAgreementPolicy(min_confirm_passes=2)

        result = policy.process_pass([])

        assert result.confirmed_text == ""
        assert result.retained_text == ""
        assert result.confirmed_tokens == []
        assert result.retained_tokens == []
        assert result.is_new_confirmation is False

    def test_empty_second_pass(self) -> None:
        """Arrange: first pass with tokens, second empty.
        Act: process empty second pass.
        Assert: nothing confirmed (empty agrees with nothing).
        """
        policy = LocalAgreementPolicy(min_confirm_passes=2)

        policy.process_pass(["hello", "world"])
        result = policy.process_pass([])

        assert result.confirmed_text == ""
        assert result.confirmed_tokens == []
        assert result.retained_tokens == []
        assert result.is_new_confirmation is False

    def test_single_token_agreement(self) -> None:
        """Arrange: passes with a single identical token.
        Act: process 2 passes.
        Assert: single token confirmed.
        """
        policy = LocalAgreementPolicy(min_confirm_passes=2)

        policy.process_pass(["hello"])
        result = policy.process_pass(["hello"])

        assert result.confirmed_text == "hello"
        assert result.confirmed_tokens == ["hello"]
        assert result.retained_tokens == []
        assert result.is_new_confirmation is True

    def test_partial_agreement_prefix(self) -> None:
        """Arrange: first N tokens agree, rest diverges.
        Act: process 2 passes.
        Assert: only agreeing prefix is confirmed.
        """
        policy = LocalAgreementPolicy(min_confirm_passes=2)

        policy.process_pass(["the", "cat", "sat", "on"])
        result = policy.process_pass(["the", "cat", "stood", "up"])

        assert result.confirmed_text == "the cat"
        assert result.retained_text == "stood up"
        assert result.confirmed_tokens == ["the", "cat"]
        assert result.retained_tokens == ["stood", "up"]

    def test_tokens_change_completely_between_passes(self) -> None:
        """Arrange: all tokens diverge between passes.
        Act: process 2 passes.
        Assert: no new tokens confirmed.
        """
        policy = LocalAgreementPolicy(min_confirm_passes=2)

        policy.process_pass(["hello", "world"])
        result = policy.process_pass(["goodbye", "earth"])

        assert result.confirmed_text == ""
        assert result.confirmed_tokens == []
        assert result.retained_tokens == ["goodbye", "earth"]
        assert result.is_new_confirmation is False

    def test_shorter_second_pass(self) -> None:
        """Arrange: second pass shorter than the first.
        Act: process 2 passes.
        Assert: only confirms tokens that exist in both.
        """
        policy = LocalAgreementPolicy(min_confirm_passes=2)

        policy.process_pass(["hello", "how", "are", "you"])
        result = policy.process_pass(["hello", "how"])

        assert result.confirmed_text == "hello how"
        assert result.confirmed_tokens == ["hello", "how"]
        assert result.retained_tokens == []


class TestMinConfirmPasses:
    """Tests with min_confirm_passes different from 2."""

    def test_min_confirm_passes_3(self) -> None:
        """Arrange: policy with min_confirm_passes=3.
        Act: process 3 identical passes.
        Assert: tokens are only confirmed on the third pass.
        """
        policy = LocalAgreementPolicy(min_confirm_passes=3)

        r1 = policy.process_pass(["a", "b"])
        assert r1.confirmed_text == ""
        assert r1.is_new_confirmation is False

        r2 = policy.process_pass(["a", "b"])
        assert r2.confirmed_text == ""
        assert r2.is_new_confirmation is False

        r3 = policy.process_pass(["a", "b", "c"])
        assert r3.confirmed_text == "a b"
        assert r3.confirmed_tokens == ["a", "b"]
        assert r3.retained_tokens == ["c"]
        assert r3.is_new_confirmation is True

    def test_min_confirm_passes_1(self) -> None:
        """Arrange: policy with min_confirm_passes=1.
        Act: process single pass.
        Assert: nothing confirmed on first pass (no previous pass).
        """
        policy = LocalAgreementPolicy(min_confirm_passes=1)

        r1 = policy.process_pass(["hello"])
        # min_confirm_passes=1 means first pass can already confirm,
        # but there is no previous pass to compare, so nothing is confirmed
        assert r1.confirmed_text == ""
        assert r1.is_new_confirmation is False

        # On the second pass, compares with the previous one
        r2 = policy.process_pass(["hello", "world"])
        assert r2.confirmed_text == "hello"
        assert r2.is_new_confirmation is True

    def test_min_confirm_passes_invalid(self) -> None:
        """Arrange: min_confirm_passes < 1.
        Act: instantiate policy.
        Assert: ValueError raised.
        """
        with pytest.raises(ValueError, match="min_confirm_passes must be >= 1"):
            LocalAgreementPolicy(min_confirm_passes=0)

    def test_min_confirm_passes_negative(self) -> None:
        """Arrange: negative min_confirm_passes.
        Act: instantiate policy.
        Assert: ValueError raised.
        """
        with pytest.raises(ValueError, match="min_confirm_passes must be >= 1"):
            LocalAgreementPolicy(min_confirm_passes=-1)


class TestProperties:
    """Tests for properties and immutability."""

    def test_confirmed_text_property(self) -> None:
        """Arrange: policy with confirmed tokens.
        Act: access confirmed_text.
        Assert: returns string with tokens joined by space.
        """
        policy = LocalAgreementPolicy(min_confirm_passes=2)

        policy.process_pass(["hello", "world"])
        policy.process_pass(["hello", "world"])

        assert policy.confirmed_text == "hello world"

    def test_confirmed_text_empty_when_no_passes(self) -> None:
        """Arrange: new policy.
        Act: access confirmed_text.
        Assert: empty string.
        """
        policy = LocalAgreementPolicy()

        assert policy.confirmed_text == ""

    def test_agreement_result_is_frozen(self) -> None:
        """Arrange: create AgreementResult.
        Act: try to modify attribute.
        Assert: FrozenInstanceError raised.
        """
        result = AgreementResult(
            confirmed_text="hello",
            retained_text="world",
            confirmed_tokens=["hello"],
            retained_tokens=["world"],
            is_new_confirmation=True,
        )

        with pytest.raises(AttributeError):
            result.confirmed_text = "modified"  # type: ignore[misc]


class TestReset:
    """Tests for the reset method."""

    def test_reset_clears_all_state(self) -> None:
        """Arrange: policy with accumulated state.
        Act: reset.
        Assert: everything zeroed, ready for next segment.
        """
        policy = LocalAgreementPolicy(min_confirm_passes=2)

        policy.process_pass(["hello", "world"])
        policy.process_pass(["hello", "world", "foo"])

        assert policy.pass_count == 2
        assert policy.confirmed_text == "hello world"

        policy.reset()

        assert policy.pass_count == 0
        assert policy.confirmed_text == ""

    def test_reset_allows_reuse(self) -> None:
        """Arrange: reset policy.
        Act: process new passes.
        Assert: works as if it were a new policy.
        """
        policy = LocalAgreementPolicy(min_confirm_passes=2)

        # First segment
        policy.process_pass(["a", "b"])
        policy.process_pass(["a", "b"])
        assert policy.confirmed_text == "a b"

        policy.reset()

        # Second segment -- should not interfere with the first
        r1 = policy.process_pass(["x", "y"])
        assert r1.confirmed_text == ""
        assert r1.is_new_confirmation is False

        r2 = policy.process_pass(["x", "y", "z"])
        assert r2.confirmed_text == "x y"
        assert r2.is_new_confirmation is True


class TestThreePassesProgressive:
    """Tests for progressive confirmation in 3+ passes."""

    def test_three_passes_progressive_confirmation(self) -> None:
        """Arrange: 3 passes with growing and agreeing tokens.
        Act: process 3 passes.
        Assert: confirmed tokens grow with each pass.
        """
        policy = LocalAgreementPolicy(min_confirm_passes=2)

        # Pass 1: nada confirmado
        r1 = policy.process_pass(["the", "quick"])
        assert r1.confirmed_tokens == []

        # Pass 2: "the quick" concordam -> confirmados
        r2 = policy.process_pass(["the", "quick", "brown"])
        assert r2.confirmed_tokens == ["the", "quick"]
        assert r2.retained_tokens == ["brown"]

        # Pass 3: "brown" agora concorda -> confirmado
        r3 = policy.process_pass(["the", "quick", "brown", "fox"])
        assert r3.confirmed_tokens == ["the", "quick", "brown"]
        assert r3.retained_tokens == ["fox"]

    def test_multiple_segments_via_flush(self) -> None:
        """Arrange: two segments separated by flush.
        Act: process, flush, process again.
        Assert: second segment independent from the first.
        """
        policy = LocalAgreementPolicy(min_confirm_passes=2)

        # Segment 1
        policy.process_pass(["hello", "world"])
        policy.process_pass(["hello", "world"])
        flush_result = policy.flush()
        assert flush_result.confirmed_text == "hello world"

        # Segment 2 -- clean state
        r1 = policy.process_pass(["goodbye"])
        assert r1.confirmed_text == ""
        assert policy.pass_count == 1

        r2 = policy.process_pass(["goodbye", "moon"])
        assert r2.confirmed_text == "goodbye"
        assert r2.retained_text == "moon"
