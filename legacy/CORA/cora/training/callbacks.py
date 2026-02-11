"""PyTorch Lightning Callbacks for CORA."""

import logging
import json
import time
import shutil
import lightning as pl
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class MetadataCallback(pl.Callback):
    """
    Saves metadata alongside checkpoints.
    Metadata includes config, training metrics, and timestamps.
    """
    def __init__(self, ckpt_dir: str, metadata: Dict[str, Any], config_path: Optional[str] = None):
        self.ckpt_dir = Path(ckpt_dir)
        self.metadata = metadata
        self.meta_path = self.ckpt_dir / "checkpoint_metadata.json"
        
        # Determine config path
        self.config_path = Path(config_path).resolve() if config_path else None
        self._config_copied = False
        
        if self.config_path:
            self.metadata.setdefault("config", {})
            self.metadata["config"]["source_path"] = str(self.config_path)

        # Create dir if needed
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_config_copy()

    def _ensure_config_copy(self):
        if not self.config_path or self._config_copied:
            return
            
        try:
            if not self.config_path.exists():
                return
                
            target = self.ckpt_dir / "config.yaml"
            if not target.exists():
                shutil.copy2(self.config_path, target)
                logger.info(f"Copied config to {target}")
            
            self._config_copied = True
        except Exception as e:
            logger.warning(f"Failed to copy config: {e}")

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        # Update metadata with current state
        current_meta = self.metadata.copy()
        current_meta.update({
            "epoch": trainer.current_epoch,
            "global_step": trainer.global_step,
            "val_loss": float(trainer.callback_metrics.get("val_loss", 0.0)),
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        })
        
        # Save JSON
        try:
            with open(self.meta_path, "w") as f:
                json.dump(current_meta, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save metadata json: {e}")
        
        self._ensure_config_copy()
