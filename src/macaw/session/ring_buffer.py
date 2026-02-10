"""Ring Buffer — pre-allocated circular buffer for audio storage.

Fixed-size circular buffer that stores recent session audio frames.
Pre-allocates a bytearray in __init__ and performs zero allocations during streaming.

Essential for:
- Recovery after worker crash (reprocess uncommitted audio)
- LocalAgreement (accumulate 3-5s windows for comparison between passes)

The absolute offset (total_written) is monotonically increasing and never resets.
Allows precise tracking of positions in the WAL.

Read fence (last_committed_offset) protects uncommitted data from overwrite.
Data before the fence can be overwritten by wrap-around; data between the
fence and total_written is protected. If uncommitted_bytes / capacity > 90%
after a write, the on_force_commit callback is invoked to notify the session.

No threading/locking — single-threaded in the asyncio event loop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from macaw.exceptions import BufferOverrunError

if TYPE_CHECKING:
    from collections.abc import Callable

# Uncommitted usage threshold that triggers force commit (90%)
_FORCE_COMMIT_THRESHOLD = 0.90


class RingBuffer:
    """Pre-allocated circular buffer for PCM audio storage.

    Parameters:
        duration_s: Buffer duration in seconds (default: 60s).
        sample_rate: Sample rate in Hz (default: 16000).
        bytes_per_sample: Bytes per sample (default: 2 for 16-bit PCM).
        on_force_commit: Optional callback invoked when uncommitted_bytes
            exceeds 90% capacity. Receives total_written as argument.
            Synchronous callback (not async) — called from write().

    The buffer is pre-allocated in __init__ and no allocation occurs during
    write/read operations. Default size (60s * 16000 * 2) = 1,920,000 bytes.
    """

    __slots__ = (
        "_buffer",
        "_capacity_bytes",
        "_on_force_commit",
        "_read_fence",
        "_total_written",
        "_write_pos",
    )

    def __init__(
        self,
        duration_s: float = 60.0,
        sample_rate: int = 16000,
        bytes_per_sample: int = 2,
        on_force_commit: Callable[[int], None] | None = None,
    ) -> None:
        self._capacity_bytes: int = int(duration_s * sample_rate * bytes_per_sample)
        if self._capacity_bytes <= 0:
            msg = f"Capacidade do buffer deve ser positiva, valor atual: {self._capacity_bytes}"
            raise ValueError(msg)
        self._buffer: bytearray = bytearray(self._capacity_bytes)
        self._write_pos: int = 0
        self._total_written: int = 0
        self._read_fence: int = 0
        self._on_force_commit = on_force_commit

    @property
    def capacity_bytes(self) -> int:
        """Total buffer size in bytes."""
        return self._capacity_bytes

    @property
    def total_written(self) -> int:
        """Total absolute written offset (monotonically increasing, never resets)."""
        return self._total_written

    @property
    def used_bytes(self) -> int:
        """Bytes currently used in the buffer (min of total_written and capacity)."""
        return min(self._total_written, self._capacity_bytes)

    @property
    def usage_percent(self) -> float:
        """Buffer usage percentage (0.0 to 100.0)."""
        return self.used_bytes / self._capacity_bytes * 100.0

    @property
    def read_fence(self) -> int:
        """Absolute offset of last_committed_offset (read fence).

        Data before this offset may be overwritten by wrap-around.
        Data between read_fence and total_written is protected.
        """
        return self._read_fence

    @property
    def uncommitted_bytes(self) -> int:
        """Bytes between read_fence and total_written (uncommitted data)."""
        return self._total_written - self._read_fence

    @property
    def available_for_write_bytes(self) -> int:
        """Bytes that can be written without overwriting uncommitted data.

        Calculation: total capacity minus uncommitted bytes occupying space
        in the circular buffer. If the buffer has not wrapped, all remaining
        space is available.
        """
        uncommitted = self.uncommitted_bytes
        if uncommitted >= self._capacity_bytes:
            return 0
        return self._capacity_bytes - uncommitted

    def commit(self, offset: int) -> None:
        """Advance the read fence to the given offset.

        Marks data before the offset as "safe to overwrite".
        Offset must be between the current fence and total_written (inclusive).

        Args:
            offset: New read fence offset (last_committed_offset).

        Raises:
            ValueError: If offset < current read_fence or offset > total_written.
        """
        if offset < self._read_fence:
            msg = (
                f"Offset de commit ({offset}) nao pode ser menor que o read_fence atual "
                f"({self._read_fence})"
            )
            raise ValueError(msg)

        if offset > self._total_written:
            msg = (
                f"Offset de commit ({offset}) nao pode ser maior que total_written "
                f"({self._total_written})"
            )
            raise ValueError(msg)

        self._read_fence = offset

    def write(self, data: bytes) -> int:
        """Write data to the buffer at the current circular position.

        Protects uncommitted data: if the write would overwrite bytes between
        read_fence and total_written, raises BufferOverrunError.

        After a successful write, checks if uncommitted_bytes > 90% capacity
        and invokes on_force_commit if configured.

        Args:
            data: Bytes to write to the buffer.

        Returns:
            Absolute offset of the write start (total_written before the operation).

        Raises:
            BufferOverrunError: If the write would overwrite uncommitted data.
        """
        data_len = len(data)
        if data_len == 0:
            return self._total_written

        # Check if the write would overwrite uncommitted data.
        # Only check when there is uncommitted data (read_fence < total_written).
        # When read_fence == total_written (all committed), writing is free
        # because there is nothing to protect — even if data_len > capacity.
        uncommitted = self.uncommitted_bytes
        if uncommitted > 0:
            available = self._capacity_bytes - uncommitted
            if available < 0:
                available = 0
            if data_len > available:
                msg = (
                    f"Escrita de {data_len} bytes sobrescreveria dados nao comitados. "
                    f"Disponivel: {available} bytes, "
                    f"read_fence={self._read_fence}, total_written={self._total_written}"
                )
                raise BufferOverrunError(msg)

        start_offset = self._total_written
        mv = memoryview(self._buffer)

        if data_len >= self._capacity_bytes:
            # Data larger than buffer: keep only the last capacity_bytes.
            # This branch is reached only when available_for_write >= capacity,
            # i.e., when read_fence == total_written (all committed).
            trimmed = data[-self._capacity_bytes :]
            new_total = self._total_written + data_len
            retained_start_offset = new_total - self._capacity_bytes
            retained_start_pos = retained_start_offset % self._capacity_bytes
            if retained_start_pos == 0:
                mv[:] = trimmed
            else:
                first_part = self._capacity_bytes - retained_start_pos
                mv[retained_start_pos:] = trimmed[:first_part]
                mv[:retained_start_pos] = trimmed[first_part:]
            self._write_pos = new_total % self._capacity_bytes
            self._total_written = new_total
            self._check_force_commit()
            return start_offset

        # Space to the end of the circular buffer
        space_to_end = self._capacity_bytes - self._write_pos

        if data_len <= space_to_end:
            # Fits without wrap-around
            mv[self._write_pos : self._write_pos + data_len] = data
        else:
            # Wrap-around: write in two parts
            first_part = space_to_end
            mv[self._write_pos : self._write_pos + first_part] = data[:first_part]
            second_part = data_len - first_part
            mv[:second_part] = data[first_part:]

        self._write_pos = (self._write_pos + data_len) % self._capacity_bytes
        self._total_written += data_len

        self._check_force_commit()
        return start_offset

    def _check_force_commit(self) -> None:
        """Check if uncommitted_bytes exceeded 90% and notify."""
        if self._on_force_commit is None:
            return

        uncommitted = self.uncommitted_bytes
        if uncommitted <= 0:
            return

        if uncommitted / self._capacity_bytes > _FORCE_COMMIT_THRESHOLD:
            self._on_force_commit(self._total_written)

    def read(self, offset: int, length: int) -> bytes:
        """Read data from the buffer starting at an absolute offset.

        Converts the absolute offset to a circular position and reads the data.
        If data crosses the buffer boundary (wrap-around), reads in two parts.

        Args:
            offset: Absolute read start offset.
            length: Number of bytes to read.

        Returns:
            Copy of the read bytes.

        Raises:
            BufferOverrunError: If the offset is too old (data already overwritten)
                or if offset + length exceeds total_written.
            ValueError: If length is negative or offset is negative.
        """
        if length == 0:
            return b""

        if offset < 0:
            msg = f"Offset nao pode ser negativo: {offset}"
            raise ValueError(msg)

        if length < 0:
            msg = f"Comprimento nao pode ser negativo: {length}"
            raise ValueError(msg)

        end_offset = offset + length

        # Check if data is still available in the buffer
        if end_offset > self._total_written:
            msg = (
                f"Leitura alem do escrito: offset={offset}, length={length}, "
                f"total_written={self._total_written}"
            )
            raise BufferOverrunError(msg)

        # Oldest available data: total_written - capacity (or 0 if buffer has not wrapped)
        oldest_available = max(0, self._total_written - self._capacity_bytes)
        if offset < oldest_available:
            msg = (
                f"Dados ja foram sobrescritos: offset={offset}, "
                f"oldest_available={oldest_available}"
            )
            raise BufferOverrunError(msg)

        # Convert absolute offset to circular position
        circular_pos = offset % self._capacity_bytes
        mv = memoryview(self._buffer)

        space_to_end = self._capacity_bytes - circular_pos

        if length <= space_to_end:
            # Read without wrap-around
            return bytes(mv[circular_pos : circular_pos + length])

        # Wrap-around: read in two parts
        first_part = bytes(mv[circular_pos : circular_pos + space_to_end])
        second_part = bytes(mv[: length - space_to_end])
        return first_part + second_part

    def read_from_offset(self, offset: int) -> bytes:
        """Read all data from offset to the current total_written.

        Useful for recovery: read everything after the last commit.

        Args:
            offset: Absolute read start offset.

        Returns:
            Copy of the read bytes.

        Raises:
            BufferOverrunError: If the offset is too old (data already overwritten).
            ValueError: If offset is negative.
        """
        if offset < 0:
            msg = f"Offset nao pode ser negativo: {offset}"
            raise ValueError(msg)

        if offset >= self._total_written:
            return b""

        length = self._total_written - offset
        return self.read(offset, length)
