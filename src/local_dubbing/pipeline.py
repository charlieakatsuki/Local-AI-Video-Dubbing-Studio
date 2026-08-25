"""Future orchestration interface for the local dubbing workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class DubbingRequest:
    """Input specification for a future dubbing job."""

    video_path: Path
    source_language: str
    target_language: str


class DubbingPipeline:
    """Placeholder orchestration boundary; processing is intentionally absent."""

    def run(self, request: DubbingRequest) -> Path:
        """Raise a clear error until pipeline components have been implemented."""
        if not request.video_path:
            raise ValueError("A video path is required for a dubbing request.")
        raise NotImplementedError("The local AI dubbing pipeline has not been implemented yet.")
