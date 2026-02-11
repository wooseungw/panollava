import json
from types import SimpleNamespace
from pathlib import Path

import pytest
import torch

import sys

project_root = Path(__file__).resolve().parents[2]
src_path = project_root / "src"
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from panovlm.runtime import StageManager, load_config_dict as runtime_load_config_dict

import scripts.train as train_mod
import scripts.eval as eval_mod


@pytest.fixture(scope="session")
def default_bundle():
    raw_cfg = runtime_load_config_dict("configs/default.yaml")
    model_cfg = train_mod._derive_model_config_from_cfg(raw_cfg)
    return SimpleNamespace(raw=raw_cfg, model=model_cfg)


def test_stage_manager_preview_default_yaml(default_bundle):
    stage_manager = StageManager(default_bundle.raw)
    stages = stage_manager.available_stage_names()
    preview = stage_manager.preview()
    assert preview, "Stage preview should not be empty"
    assert len(preview) == len(stages)
    for summary in preview:
        assert "stage" in summary
        assert summary["epochs"] is not None


def test_run_stage_with_stubs(default_bundle, monkeypatch, tmp_path):
    stage_manager = StageManager(default_bundle.raw)
    stage = stage_manager.available_stage_names()[0]

    # Stub datamodule creation
    def dummy_build_datamodule(cfg, stage_cfg):
        return SimpleNamespace()

    # Stub model creation
    def dummy_build_model(cfg, stage_name, stage_cfg, pretrained_dir_override=None):
        class DummyLitModel:
            def __init__(self):
                self.model_config = default_bundle.model
                self.vision_trainable_blocks = stage_cfg.get("vision_trainable_blocks", 0)
                self.hparams = SimpleNamespace(lr=stage_cfg.get("lr", 1e-5))
                self._stage_key = stage_name
                self.use_lora = False
                self.model = SimpleNamespace(
                    save_lora_weights=lambda path: False,
                    eval=lambda: self,
                    to=lambda device: self,
                )

            def to(self, device):
                return self

        return DummyLitModel()

    # Stub logger/callback builder
    class DummyCheckpoint:
        def __init__(self, best_model_path, last_model_path):
            self.best_model_path = best_model_path
            self.last_model_path = last_model_path

    monkeypatch.setattr(train_mod, "ModelCheckpoint", DummyCheckpoint)

    def dummy_build_logger_and_callbacks(cfg, stage_name, stage_cfg, dm, lit_model):
        ckpt_dir = tmp_path / "ckpts"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        callbacks = [
            train_mod.ModelCheckpoint(
                best_model_path=str(ckpt_dir / "best.ckpt"),
                last_model_path=str(ckpt_dir / "last.ckpt"),
            )
        ]
        return (None, callbacks, str(ckpt_dir))

    # Stub Trainer
    class DummyTrainer:
        def __init__(self, **kwargs):
            self.callbacks = kwargs.get("callbacks", [])

        def fit(self, model, **fit_kwargs):
            self.model = model
            self.fit_kwargs = fit_kwargs

    monkeypatch.setattr(train_mod, "build_datamodule", dummy_build_datamodule)
    monkeypatch.setattr(train_mod, "build_model", dummy_build_model)
    monkeypatch.setattr(train_mod, "build_logger_and_callbacks", dummy_build_logger_and_callbacks)
    monkeypatch.setattr(train_mod.pl, "Trainer", DummyTrainer)

    result = train_mod.run_stage(default_bundle.raw, stage, stage_manager)
    assert result.status == "completed"
    assert result.best_checkpoint is not None
    assert result.artifact_dir is not None


def test_eval_uses_model_factory(default_bundle, monkeypatch, tmp_path):
    ckpt_path = tmp_path / "dummy.ckpt"
    ckpt_path.write_text("")

    class FakeModel:
        def __init__(self):
            self.config = None

        def eval(self):
            return self

        def to(self, device):
            return self

    class TrackingFactory:
        last_call = None

        def __init__(self, config):
            self.config = config

        def load_checkpoint(self, path, device, **kwargs):
            TrackingFactory.last_call = ("ckpt", path, device, kwargs)
            return FakeModel()

        def load_pretrained_dir(self, path, device, **kwargs):
            TrackingFactory.last_call = ("dir", path, device, kwargs)
            return FakeModel()

    monkeypatch.setattr(eval_mod, "ModelFactory", TrackingFactory)

    model = eval_mod.load_model_and_lora(
        str(ckpt_path),
        lora_weights_path=None,
        device=torch.device("cpu"),
        config_data=default_bundle.raw.get("models", {}),
    )

    assert TrackingFactory.last_call is not None
    kind, path, device_str, _ = TrackingFactory.last_call
    assert kind == "ckpt"
    assert Path(path) == ckpt_path
    assert model is not None
