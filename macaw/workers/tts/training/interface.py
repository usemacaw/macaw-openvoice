"""Abstract interface for voice training (fine-tuning) backends.

Every voice training engine must implement this interface. Voice training
accepts audio samples of a target speaker and produces a fine-tuned
voice model usable via normal TTS synthesis endpoints.

Adding an **external** engine:
1. Implement VoiceTrainingBackend in your own package
2. Create a macaw.yaml manifest with ``python_package: your_module``
   and ``type: voice_training``
3. Install your package in the same environment
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class VoiceTrainingBackend(ABC):
    """Contract that every voice training engine must implement."""

    @abstractmethod
    async def create_project(
        self,
        name: str,
        description: str = "",
    ) -> str:
        """Create a new voice training project.

        Args:
            name: Human-readable project name.
            description: Optional project description.

        Returns:
            Project ID string.
        """
        ...

    @abstractmethod
    async def add_samples(
        self,
        project_id: str,
        audio_samples: list[bytes],
    ) -> None:
        """Add audio samples to a training project.

        Args:
            project_id: ID of the training project.
            audio_samples: List of audio file bytes (WAV/MP3/FLAC).

        Raises:
            InvalidRequestError: If samples are invalid.
        """
        ...

    @abstractmethod
    async def train(
        self,
        project_id: str,
        config: dict[str, object] | None = None,
    ) -> str:
        """Start async training job for the project.

        Args:
            project_id: ID of the training project.
            config: Optional training configuration.

        Returns:
            Training job ID.
        """
        ...

    @abstractmethod
    async def get_training_status(
        self,
        job_id: str,
    ) -> str:
        """Get training job status.

        Args:
            job_id: Training job ID.

        Returns:
            Status string: "pending", "training", "completed", "error".
        """
        ...

    @abstractmethod
    async def get_trained_voice_id(
        self,
        job_id: str,
    ) -> str | None:
        """Get the voice ID of a completed training job.

        Args:
            job_id: Training job ID.

        Returns:
            Voice ID if training is complete, None otherwise.
        """
        ...

    async def post_load_hook(self) -> None:  # noqa: B027
        """Optional hook called after initialization."""
