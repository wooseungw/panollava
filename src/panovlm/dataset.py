"""Backward-compatible dataset module shim."""

from panovlm.data import (
    BaseChatPanoDataset,
    ChatPanoDataset,
    ChatPanoTestDataset,
    VLMDataModule,
)

__all__ = [
    "BaseChatPanoDataset",
    "ChatPanoDataset",
    "ChatPanoTestDataset",
    "VLMDataModule",
]
