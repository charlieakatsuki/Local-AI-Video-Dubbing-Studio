"""Plain-text and SRT formatting for transcription results."""

from __future__ import annotations

from .models import TranscriptionResult, TranscriptionSegment


def format_srt_timestamp(seconds: float) -> str:
    """Convert seconds to the ``HH:MM:SS,mmm`` format required by SRT."""
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d},{milliseconds:03d}"


def format_srt(segments: tuple[TranscriptionSegment, ...] | list[TranscriptionSegment]) -> str:
    """Format timestamped segments as an SRT subtitle file."""
    blocks = []
    for index, segment in enumerate(segments, start=1):
        blocks.append(
            f"{index}\n"
            f"{format_srt_timestamp(segment.start)} --> {format_srt_timestamp(segment.end)}\n"
            f"{segment.text.strip()}"
        )
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def format_transcript(result: TranscriptionResult) -> str:
    """Format the full transcript as a plain UTF-8 text document."""
    return result.full_text + ("\n" if result.full_text else "")
