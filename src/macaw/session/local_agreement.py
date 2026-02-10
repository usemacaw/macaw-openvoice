"""LocalAgreement -- token confirmation across passes for partial transcripts.

For encoder-decoder engines (Whisper), native partials do not exist.
LocalAgreement compares output across consecutive passes: tokens that
agree across 2+ passes are confirmed as partial; divergent tokens are
retained until the next pass.

Concept inspired by whisper-streaming (UFAL), with a custom implementation.

Algorithm:
    1. Receive token list (words) from each inference pass.
    2. Compare position-by-position with the previous pass.
    3. Tokens that agree in min_confirm_passes consecutive passes
       are promoted to confirmed (emitted as transcript.partial).
    4. Divergent tokens are retained (await next pass).
    5. Confirmed tokens are monotonically increasing (never retracted).
    6. Flush (VAD speech_end) emits all tokens as transcript.final.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class AgreementResult:
    """Result of an agreement comparison.

    Attributes:
        confirmed_text: Text confirmed so far (can be emitted as partial).
        retained_text: Retained text (waiting for next pass).
        confirmed_tokens: List of confirmed tokens.
        retained_tokens: List of retained tokens.
        is_new_confirmation: True if new tokens were confirmed in this pass.
    """

    confirmed_text: str
    retained_text: str
    confirmed_tokens: list[str] = field(default_factory=list)
    retained_tokens: list[str] = field(default_factory=list)
    is_new_confirmation: bool = False


class LocalAgreementPolicy:
    """Token confirmation policy via agreement across passes.

    Compares inference tokens between consecutive passes. Tokens that
    agree at the same position in min_confirm_passes consecutive passes are
    confirmed. Confirmed tokens are monotonically increasing.

    Typical usage::

        policy = LocalAgreementPolicy(min_confirm_passes=2)

        # Pass 1: first tokens from the worker
        result = policy.process_pass(["hello", "how"])
        # result.confirmed_text == ""  (first pass, nothing confirmed)

        # Pass 2: tokens agree
        result = policy.process_pass(["hello", "how", "are"])
        # result.confirmed_text == "hello how"
        # result.retained_text == "are"

        # Flush (speech_end): emit everything
        result = policy.flush()
        # result.confirmed_text == "hello how are"

    Args:
        min_confirm_passes: Minimum passes to confirm a token (default: 2).

    Raises:
        ValueError: If min_confirm_passes < 1.
    """

    __slots__ = (
        "_confirmed_tokens",
        "_min_confirm_passes",
        "_pass_count",
        "_previous_tokens",
    )

    def __init__(self, min_confirm_passes: int = 2) -> None:
        if min_confirm_passes < 1:
            msg = f"min_confirm_passes must be >= 1, got {min_confirm_passes}"
            raise ValueError(msg)
        self._min_confirm_passes = min_confirm_passes
        self._previous_tokens: list[str] = []
        self._confirmed_tokens: list[str] = []
        self._pass_count: int = 0

    @property
    def confirmed_text(self) -> str:
        """Text confirmed so far."""
        return " ".join(self._confirmed_tokens) if self._confirmed_tokens else ""

    @property
    def pass_count(self) -> int:
        """Number of processed passes."""
        return self._pass_count

    def process_pass(self, tokens: list[str]) -> AgreementResult:
        """Process a new pass of tokens from the worker.

        Compares tokens with the previous pass position-by-position.
        Tokens that agree in min_confirm_passes consecutive passes are confirmed.
        Tokens already confirmed are never retracted.

        Args:
            tokens: List of tokens (words) returned by the engine.

        Returns:
            AgreementResult with confirmed and retained tokens.
        """
        self._pass_count += 1

        if self._pass_count < self._min_confirm_passes:
            # Insufficient passes: nothing can be confirmed yet
            self._previous_tokens = list(tokens)
            retained = tokens[len(self._confirmed_tokens) :]
            return AgreementResult(
                confirmed_text=self.confirmed_text,
                retained_text=" ".join(retained) if retained else "",
                confirmed_tokens=list(self._confirmed_tokens),
                retained_tokens=list(retained),
                is_new_confirmation=False,
            )

        # Compare with previous pass: find matching prefix
        # starting after already confirmed tokens
        new_confirmed: list[str] = []
        confirmed_end = len(self._confirmed_tokens)
        prev = self._previous_tokens
        curr = tokens

        i = confirmed_end
        while i < len(prev) and i < len(curr):
            if prev[i] == curr[i]:
                new_confirmed.append(curr[i])
                i += 1
            else:
                break

        is_new = len(new_confirmed) > 0
        if is_new:
            self._confirmed_tokens.extend(new_confirmed)

        # Tokens after confirmed ones are retained
        retained = tokens[len(self._confirmed_tokens) :]

        self._previous_tokens = list(tokens)

        return AgreementResult(
            confirmed_text=self.confirmed_text,
            retained_text=" ".join(retained) if retained else "",
            confirmed_tokens=list(self._confirmed_tokens),
            retained_tokens=list(retained),
            is_new_confirmation=is_new,
        )

    def flush(self) -> AgreementResult:
        """Flush: emit ALL tokens (confirmed + retained) as confirmed.

        Called on VAD speech_end. Everything becomes transcript.final.
        Resets state for the next segment.

        Returns:
            AgreementResult with all tokens confirmed.
        """
        all_tokens = list(self._confirmed_tokens)
        # Add retained tokens from the previous pass that were not confirmed
        if self._previous_tokens:
            remaining = self._previous_tokens[len(self._confirmed_tokens) :]
            all_tokens.extend(remaining)

        result = AgreementResult(
            confirmed_text=" ".join(all_tokens) if all_tokens else "",
            retained_text="",
            confirmed_tokens=list(all_tokens),
            retained_tokens=[],
            is_new_confirmation=len(all_tokens) > len(self._confirmed_tokens),
        )

        # Reset for next segment
        self._previous_tokens = []
        self._confirmed_tokens = []
        self._pass_count = 0

        return result

    def reset(self) -> None:
        """Reset state completely (new speech segment)."""
        self._previous_tokens = []
        self._confirmed_tokens = []
        self._pass_count = 0
