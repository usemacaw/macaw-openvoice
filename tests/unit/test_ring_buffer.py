"""Testes unitarios para RingBuffer — buffer circular pre-alocado.

Cobre: pre-alocacao, write, read, wrap-around, offsets absolutos,
BufferOverrunError, configuracao, edge cases.

Nota: testes de wrap-around comitam dados antes de sobrescrever (read fence).
Testes especificos de read fence e force commit estao em test_ring_buffer_fence.py.
"""

from __future__ import annotations

import pytest

from macaw.exceptions import BufferOverrunError
from macaw.session.ring_buffer import RingBuffer

# --- Construcao e pre-alocacao ---


class TestRingBufferInit:
    """Testes de inicializacao e pre-alocacao do buffer."""

    def test_default_capacity_is_60s_16khz_pcm16(self) -> None:
        """Buffer default: 60s * 16000 * 2 = 1,920,000 bytes."""
        rb = RingBuffer()
        assert rb.capacity_bytes == 1_920_000

    def test_custom_duration_creates_correct_capacity(self) -> None:
        """Buffer de 30s deve ter metade da capacidade do default."""
        rb = RingBuffer(duration_s=30.0)
        assert rb.capacity_bytes == 960_000

    def test_custom_sample_rate_creates_correct_capacity(self) -> None:
        """Buffer com 8kHz (telefonia) deve ter capacidade ajustada."""
        rb = RingBuffer(duration_s=10.0, sample_rate=8000, bytes_per_sample=2)
        assert rb.capacity_bytes == 160_000

    def test_initial_state_is_empty(self) -> None:
        """Buffer recem-criado tem zero bytes escritos e 0% de uso."""
        rb = RingBuffer(duration_s=1.0)
        assert rb.total_written == 0
        assert rb.used_bytes == 0
        assert rb.usage_percent == 0.0

    def test_buffer_is_preallocated_bytearray(self) -> None:
        """Buffer interno e um bytearray pre-alocado com tamanho correto."""
        rb = RingBuffer(duration_s=1.0, sample_rate=16000, bytes_per_sample=2)
        assert isinstance(rb._buffer, bytearray)
        assert len(rb._buffer) == rb.capacity_bytes

    def test_zero_duration_raises_value_error(self) -> None:
        """Duracao zero deve levantar ValueError."""
        with pytest.raises(ValueError, match="positive"):
            RingBuffer(duration_s=0.0)

    def test_negative_duration_raises_value_error(self) -> None:
        """Duracao negativa deve levantar ValueError."""
        with pytest.raises(ValueError, match="positive"):
            RingBuffer(duration_s=-1.0)


# --- Write ---


class TestRingBufferWrite:
    """Testes de escrita no buffer."""

    def test_write_returns_absolute_offset(self) -> None:
        """Write retorna o offset absoluto do inicio da escrita."""
        rb = RingBuffer(duration_s=1.0, sample_rate=100, bytes_per_sample=1)
        # capacity = 100 bytes
        offset = rb.write(b"\x01" * 10)
        assert offset == 0

        offset2 = rb.write(b"\x02" * 20)
        assert offset2 == 10

    def test_write_updates_total_written(self) -> None:
        """total_written incrementa corretamente a cada escrita."""
        rb = RingBuffer(duration_s=1.0, sample_rate=100, bytes_per_sample=1)
        rb.write(b"\x01" * 10)
        assert rb.total_written == 10

        rb.write(b"\x02" * 20)
        assert rb.total_written == 30

    def test_write_empty_data_is_noop(self) -> None:
        """Escrita de dados vazios nao altera estado."""
        rb = RingBuffer(duration_s=1.0, sample_rate=100, bytes_per_sample=1)
        offset = rb.write(b"")
        assert offset == 0
        assert rb.total_written == 0

    def test_write_updates_used_bytes(self) -> None:
        """used_bytes reflete dados escritos ate atingir capacidade."""
        rb = RingBuffer(duration_s=1.0, sample_rate=100, bytes_per_sample=1)
        rb.write(b"\x01" * 50)
        assert rb.used_bytes == 50

    def test_write_data_is_stored_correctly(self) -> None:
        """Dados escritos podem ser lidos corretamente."""
        rb = RingBuffer(duration_s=1.0, sample_rate=100, bytes_per_sample=1)
        data = b"\x01\x02\x03\x04\x05"
        offset = rb.write(data)
        result = rb.read(offset, len(data))
        assert result == data


# --- Write com wrap-around ---


class TestRingBufferWrapAround:
    """Testes de wrap-around (escrita circular).

    Todos os testes comitam dados antes de sobrescrever (read fence).
    """

    def test_wrap_around_overwrites_from_beginning(self) -> None:
        """Quando buffer enche e dados sao commitados, sobrescreve do inicio."""
        rb = RingBuffer(duration_s=1.0, sample_rate=10, bytes_per_sample=1)
        # capacity = 10 bytes

        # Preencher buffer inteiro
        rb.write(b"\x01" * 10)
        assert rb.used_bytes == 10
        assert rb.usage_percent == 100.0

        # Comitar tudo antes de sobrescrever
        rb.commit(rb.total_written)

        # Sobrescrever primeiros 5 bytes
        rb.write(b"\x02" * 5)
        assert rb.total_written == 15
        assert rb.used_bytes == 10  # Limitado pela capacidade

    def test_wrap_around_data_is_readable(self) -> None:
        """Dados apos wrap-around sao legiveis com offset correto."""
        rb = RingBuffer(duration_s=1.0, sample_rate=10, bytes_per_sample=1)
        # capacity = 10 bytes

        # Preencher com 10 bytes (offsets 0-9)
        rb.write(b"\x01" * 10)

        # Comitar para permitir sobrescrita
        rb.commit(rb.total_written)

        # Sobrescrever com 5 bytes (offsets 10-14)
        rb.write(b"\x02" * 5)

        # Os 5 novos bytes sao legiveis
        result = rb.read(10, 5)
        assert result == b"\x02" * 5

        # Os ultimos 5 do primeiro write ainda estao la
        result = rb.read(5, 5)
        assert result == b"\x01" * 5

    def test_multiple_wrap_arounds_work_correctly(self) -> None:
        """Buffer funciona apos multiplos wrap-arounds."""
        rb = RingBuffer(duration_s=1.0, sample_rate=10, bytes_per_sample=1)
        # capacity = 10 bytes

        # 3 voltas completas (comitando a cada volta)
        for i in range(3):
            rb.commit(rb.total_written)
            data = bytes([i + 1]) * 10
            rb.write(data)

        assert rb.total_written == 30
        assert rb.used_bytes == 10

        # Ultimos 10 bytes devem ser do terceiro write
        result = rb.read(20, 10)
        assert result == b"\x03" * 10

    def test_wrap_around_read_spans_boundary(self) -> None:
        """Leitura que cruza a borda do buffer funciona corretamente."""
        rb = RingBuffer(duration_s=1.0, sample_rate=10, bytes_per_sample=1)
        # capacity = 10 bytes

        # Escrever 8 bytes (posicao 0-7)
        rb.write(b"\x01" * 8)

        # Comitar para permitir wrap-around
        rb.commit(rb.total_written)

        # Escrever 5 bytes (posicao 8-9, wrap para 0-2)
        rb.write(b"\x02" * 5)

        # Ler os 5 bytes que cruzam a borda (offsets 8-12)
        result = rb.read(8, 5)
        assert result == b"\x02" * 5

    def test_write_data_larger_than_capacity(self) -> None:
        """Escrita de dados maiores que a capacidade retem apenas os ultimos capacity bytes."""
        rb = RingBuffer(duration_s=1.0, sample_rate=10, bytes_per_sample=1)
        # capacity = 10 bytes

        # Escrever 25 bytes (tudo commitado, fence = total_written = 0, available = 10)
        data = bytes(range(25))
        offset = rb.write(data)
        assert offset == 0
        assert rb.total_written == 25

        # Apenas ultimos 10 bytes sao retidos (bytes 15-24)
        result = rb.read(15, 10)
        assert result == bytes(range(15, 25))


# --- Read ---


class TestRingBufferRead:
    """Testes de leitura do buffer."""

    def test_read_returns_correct_data(self) -> None:
        """Leitura basica retorna dados corretos."""
        rb = RingBuffer(duration_s=1.0, sample_rate=100, bytes_per_sample=1)
        data = b"hello world!"
        rb.write(data)
        result = rb.read(0, len(data))
        assert result == data

    def test_read_returns_bytes_copy(self) -> None:
        """Leitura retorna copia, nao view do buffer interno."""
        rb = RingBuffer(duration_s=1.0, sample_rate=100, bytes_per_sample=1)
        rb.write(b"\x01\x02\x03")
        result = rb.read(0, 3)
        assert isinstance(result, bytes)

    def test_read_empty_length_returns_empty_bytes(self) -> None:
        """Leitura de length=0 retorna bytes vazio."""
        rb = RingBuffer(duration_s=1.0, sample_rate=100, bytes_per_sample=1)
        rb.write(b"\x01\x02\x03")
        result = rb.read(0, 0)
        assert result == b""

    def test_read_negative_offset_raises_value_error(self) -> None:
        """Offset negativo levanta ValueError."""
        rb = RingBuffer(duration_s=1.0, sample_rate=100, bytes_per_sample=1)
        rb.write(b"\x01\x02\x03")
        with pytest.raises(ValueError, match="negative"):
            rb.read(-1, 1)

    def test_read_negative_length_raises_value_error(self) -> None:
        """Length negativo levanta ValueError."""
        rb = RingBuffer(duration_s=1.0, sample_rate=100, bytes_per_sample=1)
        rb.write(b"\x01\x02\x03")
        with pytest.raises(ValueError, match="negative"):
            rb.read(0, -1)

    def test_read_beyond_total_written_raises_buffer_overrun(self) -> None:
        """Leitura alem do total escrito levanta BufferOverrunError."""
        rb = RingBuffer(duration_s=1.0, sample_rate=100, bytes_per_sample=1)
        rb.write(b"\x01\x02\x03")
        with pytest.raises(BufferOverrunError, match="Read beyond"):
            rb.read(0, 10)

    def test_read_overwritten_data_raises_buffer_overrun(self) -> None:
        """Leitura de dados ja sobrescritos levanta BufferOverrunError."""
        rb = RingBuffer(duration_s=1.0, sample_rate=10, bytes_per_sample=1)
        # capacity = 10 bytes

        # Escrever 10 bytes (preenche buffer)
        rb.write(b"\x01" * 10)

        # Comitar para permitir sobrescrita
        rb.commit(rb.total_written)

        # Sobrescrever com mais 5 bytes
        rb.write(b"\x02" * 5)

        # Offset 0 foi sobrescrito (oldest_available = 15 - 10 = 5)
        with pytest.raises(BufferOverrunError, match="overwritten"):
            rb.read(0, 5)

    def test_read_partial_data_from_middle(self) -> None:
        """Leitura parcial do meio do buffer funciona."""
        rb = RingBuffer(duration_s=1.0, sample_rate=100, bytes_per_sample=1)
        rb.write(b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a")
        result = rb.read(3, 4)
        assert result == b"\x04\x05\x06\x07"


# --- read_from_offset ---


class TestRingBufferReadFromOffset:
    """Testes de read_from_offset (leitura desde offset ate total_written)."""

    def test_read_from_offset_returns_all_data_since_offset(self) -> None:
        """read_from_offset le tudo de offset ate total_written."""
        rb = RingBuffer(duration_s=1.0, sample_rate=100, bytes_per_sample=1)
        rb.write(b"\x01\x02\x03\x04\x05")
        result = rb.read_from_offset(2)
        assert result == b"\x03\x04\x05"

    def test_read_from_offset_at_total_written_returns_empty(self) -> None:
        """read_from_offset no offset = total_written retorna vazio."""
        rb = RingBuffer(duration_s=1.0, sample_rate=100, bytes_per_sample=1)
        rb.write(b"\x01\x02\x03")
        result = rb.read_from_offset(3)
        assert result == b""

    def test_read_from_offset_beyond_total_written_returns_empty(self) -> None:
        """read_from_offset alem do total_written retorna vazio."""
        rb = RingBuffer(duration_s=1.0, sample_rate=100, bytes_per_sample=1)
        rb.write(b"\x01\x02\x03")
        result = rb.read_from_offset(100)
        assert result == b""

    def test_read_from_offset_after_wrap_around(self) -> None:
        """read_from_offset funciona apos wrap-around."""
        rb = RingBuffer(duration_s=1.0, sample_rate=10, bytes_per_sample=1)
        # capacity = 10 bytes

        # Preencher e dar uma volta (comitando antes)
        rb.write(b"\x01" * 10)
        rb.commit(rb.total_written)
        rb.write(b"\x02" * 5)

        # Ler a partir do offset 10 (inicio dos novos dados)
        result = rb.read_from_offset(10)
        assert result == b"\x02" * 5

    def test_read_from_offset_zero_reads_all_available(self) -> None:
        """read_from_offset(0) le tudo quando buffer nao deu volta."""
        rb = RingBuffer(duration_s=1.0, sample_rate=100, bytes_per_sample=1)
        data = b"abcdefghij"
        rb.write(data)
        result = rb.read_from_offset(0)
        assert result == data

    def test_read_from_offset_overwritten_raises_buffer_overrun(self) -> None:
        """read_from_offset de offset sobrescrito levanta BufferOverrunError."""
        rb = RingBuffer(duration_s=1.0, sample_rate=10, bytes_per_sample=1)
        # capacity = 10

        rb.write(b"\x01" * 10)
        rb.commit(rb.total_written)
        rb.write(b"\x02" * 10)
        # total_written=20, oldest_available=10

        with pytest.raises(BufferOverrunError, match="overwritten"):
            rb.read_from_offset(5)

    def test_read_from_offset_negative_raises_value_error(self) -> None:
        """Offset negativo levanta ValueError."""
        rb = RingBuffer(duration_s=1.0, sample_rate=100, bytes_per_sample=1)
        rb.write(b"\x01\x02\x03")
        with pytest.raises(ValueError, match="negative"):
            rb.read_from_offset(-1)


# --- Propriedades ---


class TestRingBufferProperties:
    """Testes de propriedades (capacity, used_bytes, usage_percent, total_written)."""

    def test_usage_percent_before_wrap(self) -> None:
        """usage_percent correto antes do buffer dar volta."""
        rb = RingBuffer(duration_s=1.0, sample_rate=10, bytes_per_sample=1)
        rb.write(b"\x01" * 5)
        assert rb.usage_percent == pytest.approx(50.0)

    def test_usage_percent_at_full(self) -> None:
        """usage_percent = 100% quando buffer esta cheio."""
        rb = RingBuffer(duration_s=1.0, sample_rate=10, bytes_per_sample=1)
        rb.write(b"\x01" * 10)
        assert rb.usage_percent == pytest.approx(100.0)

    def test_usage_percent_after_wrap_stays_100(self) -> None:
        """usage_percent permanece 100% apos wrap-around."""
        rb = RingBuffer(duration_s=1.0, sample_rate=10, bytes_per_sample=1)
        rb.write(b"\x01" * 10)
        rb.commit(rb.total_written)
        rb.write(b"\x02" * 5)
        assert rb.usage_percent == pytest.approx(100.0)

    def test_used_bytes_capped_at_capacity(self) -> None:
        """used_bytes nunca excede capacity_bytes."""
        rb = RingBuffer(duration_s=1.0, sample_rate=10, bytes_per_sample=1)
        # Escrever 25 bytes de uma vez (initial fence=0, available=10, 25>10).
        # Para escrever mais que capacity, fence deve igualar total_written.
        # Mas 25 > 10 (capacity), entao precisamos primeiro escrever 10, comitar,
        # escrever 10, comitar, e escrever 5.
        rb.write(b"\x01" * 10)
        rb.commit(rb.total_written)
        rb.write(b"\x01" * 10)
        rb.commit(rb.total_written)
        rb.write(b"\x01" * 5)
        assert rb.used_bytes == 10
        assert rb.total_written == 25

    def test_total_written_is_monotonically_increasing(self) -> None:
        """total_written nunca diminui, independente de wrap-around."""
        rb = RingBuffer(duration_s=1.0, sample_rate=10, bytes_per_sample=1)
        values = []
        for _ in range(5):
            rb.commit(rb.total_written)
            rb.write(b"\x01" * 7)
            values.append(rb.total_written)

        # Verificar monotonicamente crescente
        for i in range(1, len(values)):
            assert values[i] > values[i - 1]


# --- Edge cases ---


class TestRingBufferEdgeCases:
    """Testes de edge cases."""

    def test_write_exactly_capacity_fills_buffer(self) -> None:
        """Escrita exata da capacidade preenche buffer sem wrap."""
        rb = RingBuffer(duration_s=1.0, sample_rate=10, bytes_per_sample=1)
        data = bytes(range(10))
        rb.write(data)
        assert rb.total_written == 10
        assert rb.used_bytes == 10
        result = rb.read(0, 10)
        assert result == data

    def test_single_byte_writes(self) -> None:
        """Escritas de 1 byte funcionam corretamente com commits incrementais."""
        rb = RingBuffer(duration_s=1.0, sample_rate=10, bytes_per_sample=1)
        for i in range(15):
            # Comitar antes de sobrescrever dados nao commitados
            if rb.available_for_write_bytes < 1:
                rb.commit(rb.total_written)
            rb.write(bytes([i % 256]))

        assert rb.total_written == 15
        # Ultimos 10 bytes
        result = rb.read(5, 10)
        assert result == bytes([i % 256 for i in range(5, 15)])

    def test_read_at_exact_boundary_of_availability(self) -> None:
        """Leitura no limite exato de disponibilidade funciona."""
        rb = RingBuffer(duration_s=1.0, sample_rate=10, bytes_per_sample=1)
        rb.write(b"\x01" * 10)
        rb.commit(rb.total_written)
        rb.write(b"\x02" * 5)
        # oldest_available = 15 - 10 = 5
        # Ler exatamente a partir do mais antigo disponivel
        result = rb.read(5, 5)
        assert result == b"\x01" * 5

    def test_concurrent_small_writes_and_reads(self) -> None:
        """Sequencia de pequenas escritas e leituras intercaladas."""
        rb = RingBuffer(duration_s=1.0, sample_rate=10, bytes_per_sample=1)
        # capacity = 10

        for i in range(20):
            data = bytes([i % 256]) * 3
            # Comitar se necessario para abrir espaco
            if rb.available_for_write_bytes < 3:
                rb.commit(rb.total_written)
            offset = rb.write(data)
            result = rb.read(offset, 3)
            assert result == data

    def test_large_buffer_configuration(self) -> None:
        """Buffer grande (120s) funciona corretamente."""
        rb = RingBuffer(duration_s=120.0, sample_rate=16000, bytes_per_sample=2)
        assert rb.capacity_bytes == 3_840_000

        # Escrever 1MB e ler
        data = b"\xaa" * (1024 * 1024)
        offset = rb.write(data)
        result = rb.read(offset, len(data))
        assert result == data

    def test_no_internal_list_or_deque_used(self) -> None:
        """Buffer interno e bytearray, nao list ou deque."""
        rb = RingBuffer(duration_s=1.0, sample_rate=10, bytes_per_sample=1)

        # Verificar que o buffer e bytearray
        assert type(rb._buffer) is bytearray

        # Verificar que nao ha atributos list ou deque
        for attr_name in dir(rb):
            if attr_name.startswith("_") and not attr_name.startswith("__"):
                attr = getattr(rb, attr_name)
                assert not isinstance(attr, list), f"Atributo {attr_name} e list"

    def test_write_and_read_with_realistic_audio_sizes(self) -> None:
        """Simula escrita de frames de audio reais (20ms de PCM 16kHz)."""
        rb = RingBuffer(duration_s=5.0, sample_rate=16000, bytes_per_sample=2)
        # capacity = 160,000 bytes

        # 20ms de audio a 16kHz = 320 samples = 640 bytes
        frame_size = 640
        frames_written = 0

        for i in range(300):  # 300 frames = 6s (excede 5s de buffer)
            frame = bytes([i % 256]) * frame_size
            # Comitar antes se nao ha espaco
            if rb.available_for_write_bytes < frame_size:
                rb.commit(rb.total_written)
            rb.write(frame)
            frames_written += 1

        assert rb.total_written == 300 * frame_size
        assert rb.used_bytes == rb.capacity_bytes

        # Ultimos frames sao legiveis
        last_frame_offset = (frames_written - 1) * frame_size
        result = rb.read(last_frame_offset, frame_size)
        assert len(result) == frame_size
        assert result == bytes([299 % 256]) * frame_size
