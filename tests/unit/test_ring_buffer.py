"""Unit tests for RingBuffer -- pre-allocated circular buffer.

Covers: pre-allocation, write, read, wrap-around, absolute offsets,
BufferOverrunError, configuration, edge cases.

Note: wrap-around tests commit data before overwriting (read fence).
Specific tests for read fence and force commit are in test_ring_buffer_fence.py.
"""

from __future__ import annotations

import pytest

from macaw.exceptions import BufferOverrunError
from macaw.session.ring_buffer import RingBuffer

# --- Construction and pre-allocation ---


class TestRingBufferInit:
    """Tests for buffer initialization and pre-allocation."""

    def test_default_capacity_is_60s_16khz_pcm16(self) -> None:
        """Buffer default: 60s * 16000 * 2 = 1,920,000 bytes."""
        rb = RingBuffer()
        assert rb.capacity_bytes == 1_920_000

    def test_custom_duration_creates_correct_capacity(self) -> None:
        """30s buffer should have half the default capacity."""
        rb = RingBuffer(duration_s=30.0)
        assert rb.capacity_bytes == 960_000

    def test_custom_sample_rate_creates_correct_capacity(self) -> None:
        """Buffer with 8kHz (telephony) should have adjusted capacity."""
        rb = RingBuffer(duration_s=10.0, sample_rate=8000, bytes_per_sample=2)
        assert rb.capacity_bytes == 160_000

    def test_initial_state_is_empty(self) -> None:
        """Freshly created buffer has zero bytes written and 0% usage."""
        rb = RingBuffer(duration_s=1.0)
        assert rb.total_written == 0
        assert rb.used_bytes == 0
        assert rb.usage_percent == 0.0

    def test_buffer_is_preallocated_bytearray(self) -> None:
        """Internal buffer is a pre-allocated bytearray with correct size."""
        rb = RingBuffer(duration_s=1.0, sample_rate=16000, bytes_per_sample=2)
        assert isinstance(rb._buffer, bytearray)
        assert len(rb._buffer) == rb.capacity_bytes

    def test_zero_duration_raises_value_error(self) -> None:
        """Zero duration should raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            RingBuffer(duration_s=0.0)

    def test_negative_duration_raises_value_error(self) -> None:
        """Negative duration should raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            RingBuffer(duration_s=-1.0)


# --- Write ---


class TestRingBufferWrite:
    """Tests for buffer writes."""

    def test_write_returns_absolute_offset(self) -> None:
        """Write returns the absolute offset of the write start."""
        rb = RingBuffer(duration_s=1.0, sample_rate=100, bytes_per_sample=1)
        # capacity = 100 bytes
        offset = rb.write(b"\x01" * 10)
        assert offset == 0

        offset2 = rb.write(b"\x02" * 20)
        assert offset2 == 10

    def test_write_updates_total_written(self) -> None:
        """total_written increments correctly on each write."""
        rb = RingBuffer(duration_s=1.0, sample_rate=100, bytes_per_sample=1)
        rb.write(b"\x01" * 10)
        assert rb.total_written == 10

        rb.write(b"\x02" * 20)
        assert rb.total_written == 30

    def test_write_empty_data_is_noop(self) -> None:
        """Writing empty data does not alter state."""
        rb = RingBuffer(duration_s=1.0, sample_rate=100, bytes_per_sample=1)
        offset = rb.write(b"")
        assert offset == 0
        assert rb.total_written == 0

    def test_write_updates_used_bytes(self) -> None:
        """used_bytes reflects written data up to capacity."""
        rb = RingBuffer(duration_s=1.0, sample_rate=100, bytes_per_sample=1)
        rb.write(b"\x01" * 50)
        assert rb.used_bytes == 50

    def test_write_data_is_stored_correctly(self) -> None:
        """Written data can be read back correctly."""
        rb = RingBuffer(duration_s=1.0, sample_rate=100, bytes_per_sample=1)
        data = b"\x01\x02\x03\x04\x05"
        offset = rb.write(data)
        result = rb.read(offset, len(data))
        assert result == data


# --- Write with wrap-around ---


class TestRingBufferWrapAround:
    """Tests for wrap-around (circular write).

    All tests commit data before overwriting (read fence).
    """

    def test_wrap_around_overwrites_from_beginning(self) -> None:
        """When buffer fills and data is committed, overwrites from the beginning."""
        rb = RingBuffer(duration_s=1.0, sample_rate=10, bytes_per_sample=1)
        # capacity = 10 bytes

        # Fill entire buffer
        rb.write(b"\x01" * 10)
        assert rb.used_bytes == 10
        assert rb.usage_percent == 100.0

        # Commit everything before overwriting
        rb.commit(rb.total_written)

        # Overwrite first 5 bytes
        rb.write(b"\x02" * 5)
        assert rb.total_written == 15
        assert rb.used_bytes == 10  # Capped at capacity

    def test_wrap_around_data_is_readable(self) -> None:
        """Data after wrap-around is readable with correct offset."""
        rb = RingBuffer(duration_s=1.0, sample_rate=10, bytes_per_sample=1)
        # capacity = 10 bytes

        # Fill with 10 bytes (offsets 0-9)
        rb.write(b"\x01" * 10)

        # Commit to allow overwriting
        rb.commit(rb.total_written)

        # Overwrite with 5 bytes (offsets 10-14)
        rb.write(b"\x02" * 5)

        # The 5 new bytes are readable
        result = rb.read(10, 5)
        assert result == b"\x02" * 5

        # The last 5 from the first write are still there
        result = rb.read(5, 5)
        assert result == b"\x01" * 5

    def test_multiple_wrap_arounds_work_correctly(self) -> None:
        """Buffer works after multiple wrap-arounds."""
        rb = RingBuffer(duration_s=1.0, sample_rate=10, bytes_per_sample=1)
        # capacity = 10 bytes

        # 3 full rotations (committing each rotation)
        for i in range(3):
            rb.commit(rb.total_written)
            data = bytes([i + 1]) * 10
            rb.write(data)

        assert rb.total_written == 30
        assert rb.used_bytes == 10

        # Last 10 bytes should be from the third write
        result = rb.read(20, 10)
        assert result == b"\x03" * 10

    def test_wrap_around_read_spans_boundary(self) -> None:
        """Read spanning the buffer boundary works correctly."""
        rb = RingBuffer(duration_s=1.0, sample_rate=10, bytes_per_sample=1)
        # capacity = 10 bytes

        # Write 8 bytes (position 0-7)
        rb.write(b"\x01" * 8)

        # Commit to allow wrap-around
        rb.commit(rb.total_written)

        # Write 5 bytes (position 8-9, wraps to 0-2)
        rb.write(b"\x02" * 5)

        # Read the 5 bytes that span the boundary (offsets 8-12)
        result = rb.read(8, 5)
        assert result == b"\x02" * 5

    def test_write_data_larger_than_capacity(self) -> None:
        """Writing data larger than capacity retains only the last capacity bytes."""
        rb = RingBuffer(duration_s=1.0, sample_rate=10, bytes_per_sample=1)
        # capacity = 10 bytes

        # Write 25 bytes (all committed, fence = total_written = 0, available = 10)
        data = bytes(range(25))
        offset = rb.write(data)
        assert offset == 0
        assert rb.total_written == 25

        # Only last 10 bytes are retained (bytes 15-24)
        result = rb.read(15, 10)
        assert result == bytes(range(15, 25))


# --- Read ---


class TestRingBufferRead:
    """Tests for buffer reads."""

    def test_read_returns_correct_data(self) -> None:
        """Basic read returns correct data."""
        rb = RingBuffer(duration_s=1.0, sample_rate=100, bytes_per_sample=1)
        data = b"hello world!"
        rb.write(data)
        result = rb.read(0, len(data))
        assert result == data

    def test_read_returns_bytes_copy(self) -> None:
        """Read returns a copy, not a view of the internal buffer."""
        rb = RingBuffer(duration_s=1.0, sample_rate=100, bytes_per_sample=1)
        rb.write(b"\x01\x02\x03")
        result = rb.read(0, 3)
        assert isinstance(result, bytes)

    def test_read_empty_length_returns_empty_bytes(self) -> None:
        """Read with length=0 returns empty bytes."""
        rb = RingBuffer(duration_s=1.0, sample_rate=100, bytes_per_sample=1)
        rb.write(b"\x01\x02\x03")
        result = rb.read(0, 0)
        assert result == b""

    def test_read_negative_offset_raises_value_error(self) -> None:
        """Negative offset raises ValueError."""
        rb = RingBuffer(duration_s=1.0, sample_rate=100, bytes_per_sample=1)
        rb.write(b"\x01\x02\x03")
        with pytest.raises(ValueError, match="negative"):
            rb.read(-1, 1)

    def test_read_negative_length_raises_value_error(self) -> None:
        """Negative length raises ValueError."""
        rb = RingBuffer(duration_s=1.0, sample_rate=100, bytes_per_sample=1)
        rb.write(b"\x01\x02\x03")
        with pytest.raises(ValueError, match="negative"):
            rb.read(0, -1)

    def test_read_beyond_total_written_raises_buffer_overrun(self) -> None:
        """Read beyond total written raises BufferOverrunError."""
        rb = RingBuffer(duration_s=1.0, sample_rate=100, bytes_per_sample=1)
        rb.write(b"\x01\x02\x03")
        with pytest.raises(BufferOverrunError, match="Read beyond"):
            rb.read(0, 10)

    def test_read_overwritten_data_raises_buffer_overrun(self) -> None:
        """Read of already overwritten data raises BufferOverrunError."""
        rb = RingBuffer(duration_s=1.0, sample_rate=10, bytes_per_sample=1)
        # capacity = 10 bytes

        # Write 10 bytes (fills buffer)
        rb.write(b"\x01" * 10)

        # Commit to allow overwriting
        rb.commit(rb.total_written)

        # Overwrite with 5 more bytes
        rb.write(b"\x02" * 5)

        # Offset 0 was overwritten (oldest_available = 15 - 10 = 5)
        with pytest.raises(BufferOverrunError, match="overwritten"):
            rb.read(0, 5)

    def test_read_partial_data_from_middle(self) -> None:
        """Partial read from the middle of the buffer works."""
        rb = RingBuffer(duration_s=1.0, sample_rate=100, bytes_per_sample=1)
        rb.write(b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a")
        result = rb.read(3, 4)
        assert result == b"\x04\x05\x06\x07"


# --- read_from_offset ---


class TestRingBufferReadFromOffset:
    """Tests for read_from_offset (read from offset up to total_written)."""

    def test_read_from_offset_returns_all_data_since_offset(self) -> None:
        """read_from_offset reads everything from offset to total_written."""
        rb = RingBuffer(duration_s=1.0, sample_rate=100, bytes_per_sample=1)
        rb.write(b"\x01\x02\x03\x04\x05")
        result = rb.read_from_offset(2)
        assert result == b"\x03\x04\x05"

    def test_read_from_offset_at_total_written_returns_empty(self) -> None:
        """read_from_offset at offset = total_written returns empty."""
        rb = RingBuffer(duration_s=1.0, sample_rate=100, bytes_per_sample=1)
        rb.write(b"\x01\x02\x03")
        result = rb.read_from_offset(3)
        assert result == b""

    def test_read_from_offset_beyond_total_written_returns_empty(self) -> None:
        """read_from_offset beyond total_written returns empty."""
        rb = RingBuffer(duration_s=1.0, sample_rate=100, bytes_per_sample=1)
        rb.write(b"\x01\x02\x03")
        result = rb.read_from_offset(100)
        assert result == b""

    def test_read_from_offset_after_wrap_around(self) -> None:
        """read_from_offset works after wrap-around."""
        rb = RingBuffer(duration_s=1.0, sample_rate=10, bytes_per_sample=1)
        # capacity = 10 bytes

        # Fill and wrap around (committing first)
        rb.write(b"\x01" * 10)
        rb.commit(rb.total_written)
        rb.write(b"\x02" * 5)

        # Read from offset 10 (start of new data)
        result = rb.read_from_offset(10)
        assert result == b"\x02" * 5

    def test_read_from_offset_zero_reads_all_available(self) -> None:
        """read_from_offset(0) reads everything when buffer has not wrapped."""
        rb = RingBuffer(duration_s=1.0, sample_rate=100, bytes_per_sample=1)
        data = b"abcdefghij"
        rb.write(data)
        result = rb.read_from_offset(0)
        assert result == data

    def test_read_from_offset_overwritten_raises_buffer_overrun(self) -> None:
        """read_from_offset of overwritten offset raises BufferOverrunError."""
        rb = RingBuffer(duration_s=1.0, sample_rate=10, bytes_per_sample=1)
        # capacity = 10

        rb.write(b"\x01" * 10)
        rb.commit(rb.total_written)
        rb.write(b"\x02" * 10)
        # total_written=20, oldest_available=10

        with pytest.raises(BufferOverrunError, match="overwritten"):
            rb.read_from_offset(5)

    def test_read_from_offset_negative_raises_value_error(self) -> None:
        """Negative offset raises ValueError."""
        rb = RingBuffer(duration_s=1.0, sample_rate=100, bytes_per_sample=1)
        rb.write(b"\x01\x02\x03")
        with pytest.raises(ValueError, match="negative"):
            rb.read_from_offset(-1)


# --- Properties ---


class TestRingBufferProperties:
    """Tests for properties (capacity, used_bytes, usage_percent, total_written)."""

    def test_usage_percent_before_wrap(self) -> None:
        """usage_percent is correct before the buffer wraps around."""
        rb = RingBuffer(duration_s=1.0, sample_rate=10, bytes_per_sample=1)
        rb.write(b"\x01" * 5)
        assert rb.usage_percent == pytest.approx(50.0)

    def test_usage_percent_at_full(self) -> None:
        """usage_percent = 100% when buffer is full."""
        rb = RingBuffer(duration_s=1.0, sample_rate=10, bytes_per_sample=1)
        rb.write(b"\x01" * 10)
        assert rb.usage_percent == pytest.approx(100.0)

    def test_usage_percent_after_wrap_stays_100(self) -> None:
        """usage_percent stays at 100% after wrap-around."""
        rb = RingBuffer(duration_s=1.0, sample_rate=10, bytes_per_sample=1)
        rb.write(b"\x01" * 10)
        rb.commit(rb.total_written)
        rb.write(b"\x02" * 5)
        assert rb.usage_percent == pytest.approx(100.0)

    def test_used_bytes_capped_at_capacity(self) -> None:
        """used_bytes never exceeds capacity_bytes."""
        rb = RingBuffer(duration_s=1.0, sample_rate=10, bytes_per_sample=1)
        # Write 25 bytes at once (initial fence=0, available=10, 25>10).
        # To write more than capacity, fence must equal total_written.
        # But 25 > 10 (capacity), so we need to first write 10, commit,
        # write 10, commit, and write 5.
        rb.write(b"\x01" * 10)
        rb.commit(rb.total_written)
        rb.write(b"\x01" * 10)
        rb.commit(rb.total_written)
        rb.write(b"\x01" * 5)
        assert rb.used_bytes == 10
        assert rb.total_written == 25

    def test_total_written_is_monotonically_increasing(self) -> None:
        """total_written never decreases, regardless of wrap-around."""
        rb = RingBuffer(duration_s=1.0, sample_rate=10, bytes_per_sample=1)
        values = []
        for _ in range(5):
            rb.commit(rb.total_written)
            rb.write(b"\x01" * 7)
            values.append(rb.total_written)

        # Verify monotonically increasing
        for i in range(1, len(values)):
            assert values[i] > values[i - 1]


# --- Edge cases ---


class TestRingBufferEdgeCases:
    """Tests for edge cases."""

    def test_write_exactly_capacity_fills_buffer(self) -> None:
        """Write exactly at capacity fills buffer without wrap."""
        rb = RingBuffer(duration_s=1.0, sample_rate=10, bytes_per_sample=1)
        data = bytes(range(10))
        rb.write(data)
        assert rb.total_written == 10
        assert rb.used_bytes == 10
        result = rb.read(0, 10)
        assert result == data

    def test_single_byte_writes(self) -> None:
        """Single byte writes work correctly with incremental commits."""
        rb = RingBuffer(duration_s=1.0, sample_rate=10, bytes_per_sample=1)
        for i in range(15):
            # Commit before overwriting uncommitted data
            if rb.available_for_write_bytes < 1:
                rb.commit(rb.total_written)
            rb.write(bytes([i % 256]))

        assert rb.total_written == 15
        # Last 10 bytes
        result = rb.read(5, 10)
        assert result == bytes([i % 256 for i in range(5, 15)])

    def test_read_at_exact_boundary_of_availability(self) -> None:
        """Read at the exact boundary of availability works."""
        rb = RingBuffer(duration_s=1.0, sample_rate=10, bytes_per_sample=1)
        rb.write(b"\x01" * 10)
        rb.commit(rb.total_written)
        rb.write(b"\x02" * 5)
        # oldest_available = 15 - 10 = 5
        # Read exactly from the oldest available
        result = rb.read(5, 5)
        assert result == b"\x01" * 5

    def test_concurrent_small_writes_and_reads(self) -> None:
        """Sequence of small interleaved writes and reads."""
        rb = RingBuffer(duration_s=1.0, sample_rate=10, bytes_per_sample=1)
        # capacity = 10

        for i in range(20):
            data = bytes([i % 256]) * 3
            # Commit if needed to free up space
            if rb.available_for_write_bytes < 3:
                rb.commit(rb.total_written)
            offset = rb.write(data)
            result = rb.read(offset, 3)
            assert result == data

    def test_large_buffer_configuration(self) -> None:
        """Large buffer (120s) works correctly."""
        rb = RingBuffer(duration_s=120.0, sample_rate=16000, bytes_per_sample=2)
        assert rb.capacity_bytes == 3_840_000

        # Write 1MB and read
        data = b"\xaa" * (1024 * 1024)
        offset = rb.write(data)
        result = rb.read(offset, len(data))
        assert result == data

    def test_no_internal_list_or_deque_used(self) -> None:
        """Internal buffer is bytearray, not list or deque."""
        rb = RingBuffer(duration_s=1.0, sample_rate=10, bytes_per_sample=1)

        # Verify buffer is bytearray
        assert type(rb._buffer) is bytearray

        # Verify no list or deque attributes
        for attr_name in dir(rb):
            if attr_name.startswith("_") and not attr_name.startswith("__"):
                attr = getattr(rb, attr_name)
                assert not isinstance(attr, list), f"Attribute {attr_name} is list"

    def test_write_and_read_with_realistic_audio_sizes(self) -> None:
        """Simulates writing real audio frames (20ms of PCM 16kHz)."""
        rb = RingBuffer(duration_s=5.0, sample_rate=16000, bytes_per_sample=2)
        # capacity = 160,000 bytes

        # 20ms de audio a 16kHz = 320 samples = 640 bytes
        frame_size = 640
        frames_written = 0

        for i in range(300):  # 300 frames = 6s (excede 5s de buffer)
            frame = bytes([i % 256]) * frame_size
            # Commit first if no space available
            if rb.available_for_write_bytes < frame_size:
                rb.commit(rb.total_written)
            rb.write(frame)
            frames_written += 1

        assert rb.total_written == 300 * frame_size
        assert rb.used_bytes == rb.capacity_bytes

        # Last frames are readable
        last_frame_offset = (frames_written - 1) * frame_size
        result = rb.read(last_frame_offset, frame_size)
        assert len(result) == frame_size
        assert result == bytes([299 % 256]) * frame_size
