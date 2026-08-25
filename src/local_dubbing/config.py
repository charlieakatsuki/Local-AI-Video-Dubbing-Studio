"""Configuration models for the application."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Project-relative paths used by future local processing components."""

    project_root: Path
    models_dir: Path
    outputs_dir: Path

    @classmethod
    def from_project_root(cls, project_root: Path) -> "AppConfig":
        """Create configuration without relying on a machine-specific user path."""
        resolved_root = project_root.resolve()
        return cls(
            project_root=resolved_root,
            models_dir=resolved_root / "models",
            outputs_dir=resolved_root / "outputs",
        )
