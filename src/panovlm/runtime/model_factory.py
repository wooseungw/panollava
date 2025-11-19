"""Thin factory for PanoramaVLM instantiation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from panovlm.config import ModelConfig
from panovlm.models.model import PanoramaVLM


@dataclass
class ModelFactory:
    """Keeps all PanoramaVLM construction paths in one place."""

    model_config: ModelConfig

    def build(self) -> PanoramaVLM:
        """Fresh initialization (no checkpoint)."""
        return PanoramaVLM(config=self.model_config)

    def load_checkpoint(
        self,
        checkpoint_path: str,
        *,
        device: str = "auto",
        **kwargs,
    ) -> PanoramaVLM:
        """Load a Lightning checkpoint."""
        return PanoramaVLM.from_checkpoint(
            checkpoint_path,
            device=device,
            model_config=self.model_config,
            **kwargs,
        )

    def load_pretrained_dir(
        self,
        pretrained_dir: str,
        *,
        device: str = "auto",
        strict_loading: bool = False,
        **kwargs,
    ) -> PanoramaVLM:
        """Load a HuggingFace-style directory."""
        return PanoramaVLM.from_pretrained_dir(
            pretrained_dir,
            device=device,
            strict_loading=strict_loading,
            model_config=self.model_config,
            **kwargs,
        )

    def build_from_sources(
        self,
        *,
        pretrained_dir: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        device: str = "auto",
    ) -> PanoramaVLM:
        """Utility used by scripts that accept multiple input types."""
        if checkpoint_path:
            return self.load_checkpoint(checkpoint_path, device=device)
        if pretrained_dir:
            return self.load_pretrained_dir(pretrained_dir, device=device)
        return self.build()
