"""Testes do LocalAgreementPolicy -- confirmacao de tokens por agreement entre passes.

Cobre:
- Comportamento basico: primeira pass, segunda pass, agreement, retencao
- Crescimento monotonico de tokens confirmados
- Flush (speech_end) emitindo todos os tokens
- Reset de estado
- Edge cases: tokens vazios, divergencia completa, min_confirm_passes > 2
- Imutabilidade de AgreementResult
"""

from __future__ import annotations

import pytest

from macaw.session.local_agreement import AgreementResult, LocalAgreementPolicy

# ---------------------------------------------------------------------------
# Testes basicos de agreement
# ---------------------------------------------------------------------------


class TestFirstPass:
    """Primeira pass nunca confirma tokens (precisa de comparacao)."""

    def test_first_pass_confirms_nothing(self) -> None:
        """Arrange: policy com min_confirm_passes=2.
        Act: processar primeira pass com tokens.
        Assert: nenhum token confirmado, todos retidos.
        """
        policy = LocalAgreementPolicy(min_confirm_passes=2)

        result = policy.process_pass(["hello", "how"])

        assert result.confirmed_text == ""
        assert result.retained_text == "hello how"
        assert result.confirmed_tokens == []
        assert result.retained_tokens == ["hello", "how"]
        assert result.is_new_confirmation is False

    def test_first_pass_sets_pass_count(self) -> None:
        """Arrange: policy nova.
        Act: processar uma pass.
        Assert: pass_count e 1.
        """
        policy = LocalAgreementPolicy()

        policy.process_pass(["hello"])

        assert policy.pass_count == 1


class TestSecondPass:
    """Segunda pass confirma tokens que concordam com a primeira."""

    def test_second_pass_confirms_agreeing_tokens(self) -> None:
        """Arrange: duas passes com tokens coincidentes.
        Act: processar segunda pass.
        Assert: tokens que concordam sao confirmados.
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
        """Arrange: segunda pass diverge no segundo token.
        Act: processar segunda pass.
        Assert: apenas primeiro token confirmado, restante retido.
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
    """Tokens confirmados crescem monotonicamente (nunca diminuem)."""

    def test_confirmed_tokens_grow_monotonically(self) -> None:
        """Arrange: varias passes com crescimento progressivo.
        Act: processar 4 passes.
        Assert: confirmed nunca diminui entre passes.
        """
        policy = LocalAgreementPolicy(min_confirm_passes=2)

        policy.process_pass(["a", "b", "c"])
        r1 = policy.process_pass(["a", "b", "c", "d"])
        assert len(r1.confirmed_tokens) == 3  # a, b, c confirmados

        r2 = policy.process_pass(["a", "b", "c", "d", "e"])
        assert len(r2.confirmed_tokens) == 4  # d agora confirmado tambem

        r3 = policy.process_pass(["a", "b", "c", "d", "e", "f"])
        assert len(r3.confirmed_tokens) == 5  # e agora confirmado tambem

        # Confirma crescimento monotonico
        assert r1.confirmed_tokens == ["a", "b", "c"]
        assert r2.confirmed_tokens == ["a", "b", "c", "d"]
        assert r3.confirmed_tokens == ["a", "b", "c", "d", "e"]

    def test_confirmed_never_shrink_on_divergence(self) -> None:
        """Arrange: tokens concordam em pass 2, divergem em pass 3.
        Act: processar 3 passes.
        Assert: tokens ja confirmados permanecem confirmados.
        """
        policy = LocalAgreementPolicy(min_confirm_passes=2)

        policy.process_pass(["hello", "world"])
        r1 = policy.process_pass(["hello", "world"])
        assert r1.confirmed_tokens == ["hello", "world"]

        # Pass 3 diverge completamente apos os confirmados
        r2 = policy.process_pass(["hello", "world", "foo"])
        assert r2.confirmed_tokens == ["hello", "world"]  # nao diminuiu
        assert r2.retained_tokens == ["foo"]


class TestFlush:
    """Flush (speech_end) emite todos os tokens."""

    def test_flush_emits_all_tokens(self) -> None:
        """Arrange: policy com tokens confirmados e retidos.
        Act: flush.
        Assert: confirmed + retained viram todos confirmados.
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
        """Arrange: policy com estado acumulado.
        Act: flush.
        Assert: pass_count volta a 0, confirmed_text vazio.
        """
        policy = LocalAgreementPolicy(min_confirm_passes=2)

        policy.process_pass(["a", "b"])
        policy.process_pass(["a", "b", "c"])
        policy.flush()

        assert policy.pass_count == 0
        assert policy.confirmed_text == ""

    def test_flush_empty_state(self) -> None:
        """Arrange: policy sem nenhuma pass processada.
        Act: flush.
        Assert: resultado vazio sem erros.
        """
        policy = LocalAgreementPolicy()

        result = policy.flush()

        assert result.confirmed_text == ""
        assert result.retained_text == ""
        assert result.confirmed_tokens == []
        assert result.retained_tokens == []
        assert result.is_new_confirmation is False

    def test_flush_after_single_pass(self) -> None:
        """Arrange: flush apos apenas 1 pass (nada confirmado).
        Act: flush.
        Assert: todos os tokens da unica pass sao emitidos.
        """
        policy = LocalAgreementPolicy(min_confirm_passes=2)

        policy.process_pass(["hello", "world"])
        result = policy.flush()

        assert result.confirmed_text == "hello world"
        assert result.confirmed_tokens == ["hello", "world"]
        assert result.is_new_confirmation is True


class TestEdgeCases:
    """Edge cases: tokens vazios, divergencia completa, etc."""

    def test_empty_pass(self) -> None:
        """Arrange: pass com lista vazia de tokens.
        Act: processar pass vazia.
        Assert: resultado vazio sem erros.
        """
        policy = LocalAgreementPolicy(min_confirm_passes=2)

        result = policy.process_pass([])

        assert result.confirmed_text == ""
        assert result.retained_text == ""
        assert result.confirmed_tokens == []
        assert result.retained_tokens == []
        assert result.is_new_confirmation is False

    def test_empty_second_pass(self) -> None:
        """Arrange: primeira pass com tokens, segunda vazia.
        Act: processar segunda pass vazia.
        Assert: nada confirmado (vazio nao concorda com nada).
        """
        policy = LocalAgreementPolicy(min_confirm_passes=2)

        policy.process_pass(["hello", "world"])
        result = policy.process_pass([])

        assert result.confirmed_text == ""
        assert result.confirmed_tokens == []
        assert result.retained_tokens == []
        assert result.is_new_confirmation is False

    def test_single_token_agreement(self) -> None:
        """Arrange: passes com um unico token identico.
        Act: processar 2 passes.
        Assert: token unico confirmado.
        """
        policy = LocalAgreementPolicy(min_confirm_passes=2)

        policy.process_pass(["hello"])
        result = policy.process_pass(["hello"])

        assert result.confirmed_text == "hello"
        assert result.confirmed_tokens == ["hello"]
        assert result.retained_tokens == []
        assert result.is_new_confirmation is True

    def test_partial_agreement_prefix(self) -> None:
        """Arrange: primeiros N tokens concordam, restante diverge.
        Act: processar 2 passes.
        Assert: apenas prefixo concordante e confirmado.
        """
        policy = LocalAgreementPolicy(min_confirm_passes=2)

        policy.process_pass(["the", "cat", "sat", "on"])
        result = policy.process_pass(["the", "cat", "stood", "up"])

        assert result.confirmed_text == "the cat"
        assert result.retained_text == "stood up"
        assert result.confirmed_tokens == ["the", "cat"]
        assert result.retained_tokens == ["stood", "up"]

    def test_tokens_change_completely_between_passes(self) -> None:
        """Arrange: todos os tokens divergem entre passes.
        Act: processar 2 passes.
        Assert: nenhum token novo confirmado.
        """
        policy = LocalAgreementPolicy(min_confirm_passes=2)

        policy.process_pass(["hello", "world"])
        result = policy.process_pass(["goodbye", "earth"])

        assert result.confirmed_text == ""
        assert result.confirmed_tokens == []
        assert result.retained_tokens == ["goodbye", "earth"]
        assert result.is_new_confirmation is False

    def test_shorter_second_pass(self) -> None:
        """Arrange: segunda pass mais curta que a primeira.
        Act: processar 2 passes.
        Assert: confirma apenas tokens que existem em ambas.
        """
        policy = LocalAgreementPolicy(min_confirm_passes=2)

        policy.process_pass(["hello", "how", "are", "you"])
        result = policy.process_pass(["hello", "how"])

        assert result.confirmed_text == "hello how"
        assert result.confirmed_tokens == ["hello", "how"]
        assert result.retained_tokens == []


class TestMinConfirmPasses:
    """Testes com min_confirm_passes diferente de 2."""

    def test_min_confirm_passes_3(self) -> None:
        """Arrange: policy com min_confirm_passes=3.
        Act: processar 3 passes identicas.
        Assert: tokens so sao confirmados na terceira pass.
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
        """Arrange: policy com min_confirm_passes=1.
        Act: processar unica pass.
        Assert: nada confirmado na primeira pass (nao ha pass anterior).
        """
        policy = LocalAgreementPolicy(min_confirm_passes=1)

        r1 = policy.process_pass(["hello"])
        # min_confirm_passes=1 significa que na primeira pass ja pode confirmar,
        # mas nao ha pass anterior para comparar, entao nada e confirmado
        assert r1.confirmed_text == ""
        assert r1.is_new_confirmation is False

        # Na segunda pass, compara com a anterior
        r2 = policy.process_pass(["hello", "world"])
        assert r2.confirmed_text == "hello"
        assert r2.is_new_confirmation is True

    def test_min_confirm_passes_invalid(self) -> None:
        """Arrange: min_confirm_passes < 1.
        Act: instanciar policy.
        Assert: ValueError levantado.
        """
        with pytest.raises(ValueError, match="min_confirm_passes deve ser >= 1"):
            LocalAgreementPolicy(min_confirm_passes=0)

    def test_min_confirm_passes_negative(self) -> None:
        """Arrange: min_confirm_passes negativo.
        Act: instanciar policy.
        Assert: ValueError levantado.
        """
        with pytest.raises(ValueError, match="min_confirm_passes deve ser >= 1"):
            LocalAgreementPolicy(min_confirm_passes=-1)


class TestProperties:
    """Testes de properties e imutabilidade."""

    def test_confirmed_text_property(self) -> None:
        """Arrange: policy com tokens confirmados.
        Act: acessar confirmed_text.
        Assert: retorna string com tokens unidos por espaco.
        """
        policy = LocalAgreementPolicy(min_confirm_passes=2)

        policy.process_pass(["hello", "world"])
        policy.process_pass(["hello", "world"])

        assert policy.confirmed_text == "hello world"

    def test_confirmed_text_empty_when_no_passes(self) -> None:
        """Arrange: policy nova.
        Act: acessar confirmed_text.
        Assert: string vazia.
        """
        policy = LocalAgreementPolicy()

        assert policy.confirmed_text == ""

    def test_agreement_result_is_frozen(self) -> None:
        """Arrange: criar AgreementResult.
        Act: tentar modificar atributo.
        Assert: FrozenInstanceError levantado.
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
    """Testes do metodo reset."""

    def test_reset_clears_all_state(self) -> None:
        """Arrange: policy com estado acumulado.
        Act: reset.
        Assert: tudo zerado, pronto para proximo segmento.
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
        """Arrange: policy resetada.
        Act: processar novas passes.
        Assert: funciona como se fosse policy nova.
        """
        policy = LocalAgreementPolicy(min_confirm_passes=2)

        # Primeiro segmento
        policy.process_pass(["a", "b"])
        policy.process_pass(["a", "b"])
        assert policy.confirmed_text == "a b"

        policy.reset()

        # Segundo segmento -- nao deve interferir com o primeiro
        r1 = policy.process_pass(["x", "y"])
        assert r1.confirmed_text == ""
        assert r1.is_new_confirmation is False

        r2 = policy.process_pass(["x", "y", "z"])
        assert r2.confirmed_text == "x y"
        assert r2.is_new_confirmation is True


class TestThreePassesProgressive:
    """Testes de confirmacao progressiva em 3+ passes."""

    def test_three_passes_progressive_confirmation(self) -> None:
        """Arrange: 3 passes com tokens crescentes e concordantes.
        Act: processar 3 passes.
        Assert: tokens confirmados crescem a cada pass.
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
        """Arrange: dois segmentos separados por flush.
        Act: processar, flush, processar novamente.
        Assert: segundo segmento independente do primeiro.
        """
        policy = LocalAgreementPolicy(min_confirm_passes=2)

        # Segmento 1
        policy.process_pass(["hello", "world"])
        policy.process_pass(["hello", "world"])
        flush_result = policy.flush()
        assert flush_result.confirmed_text == "hello world"

        # Segmento 2 -- estado limpo
        r1 = policy.process_pass(["goodbye"])
        assert r1.confirmed_text == ""
        assert policy.pass_count == 1

        r2 = policy.process_pass(["goodbye", "moon"])
        assert r2.confirmed_text == "goodbye"
        assert r2.retained_text == "moon"
