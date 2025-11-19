#!/usr/bin/env python3
"""
Measure FLOPs for a PanoramaVLM checkpoint.

Usage examples:
  python tools/measure_flops.py runs/.../siglip2_...ckpt --device cuda --batch 1 --views 6 --h 224 --w 224

Notes:
 - This script will try fvcore.nn.FlopCountAnalysis first, then thop.profile.
 - If neither is installed, it will print parameter counts and prompt to install one of these packages.
"""
import argparse
import inspect
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import List, Tuple

import torch

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency when using --config
    yaml = None

from panovlm.runtime import ModelFactory
from panovlm.config.loader import load_config_dict
from panovlm.config import ModelConfig, ConfigManager
from panovlm.models.model import PanoramaVLM


def humanize_flops(flops: float) -> str:
    # flops is number of floating point operations
    for unit in ["", "K", "M", "G", "T"]:
        if flops < 1000.0:
            return f"{flops:0.3f} {unit}FLOP"
        flops /= 1000.0
    return f"{flops:0.3f} PFLOP"


def _ensure_projection_defaults(model_config: ModelConfig) -> ModelConfig:
    """
    PanoramaProjector requires projection_intermediate_size but many older
    configs (and checkpoints) never set it explicitly. For FLOP inspection we
    can safely fall back to a 4x expansion of the latent dimension so that the
    model can be instantiated.
    """
    existing = getattr(model_config, "projection_intermediate_size", None)
    if existing is None or (isinstance(existing, (int, float)) and int(existing) <= 0):
        latent_dim = getattr(model_config, "latent_dimension", None)
        if latent_dim is None or int(latent_dim) <= 0:
            latent_dim = 768  # final fallback
        default_size = int(latent_dim) * 4
        setattr(model_config, "projection_intermediate_size", default_size)
        print(
            f"[FLOPs] projection_intermediate_size가 없어 기본값 {default_size} "
            f"(latent_dimension={latent_dim} × 4)로 대체합니다."
        )
    # ensure bool flag exists for PanoramaProjector._require call
    if not hasattr(model_config, "use_projection_positional_encoding"):
        setattr(model_config, "use_projection_positional_encoding", False)
    return model_config


def _clear_thop_attributes(module: torch.nn.Module) -> None:
    """Remove thop profiling attributes if they exist to avoid duplicate attachment errors."""
    if module is None:
        return
    for child in module.modules():
        for attr in ("total_ops", "total_params"):
            if hasattr(child, attr):
                try:
                    delattr(child, attr)
                except AttributeError:
                    pass
            buffers = getattr(child, "_buffers", None)
            if isinstance(buffers, dict):
                buffers.pop(attr, None)


def _find_thop_conflict_modules(module: torch.nn.Module) -> List[Tuple[str, torch.nn.Module]]:
    """
    Identify modules that already expose attributes named total_ops/total_params
    that are not buffers. thop.profile registers buffers with these names and
    will crash with KeyError if they already exist (e.g., custom properties).
    """
    conflicts: List[Tuple[str, torch.nn.Module]] = []
    if module is None:
        return conflicts
    for name, child in module.named_modules():
        buffers = getattr(child, "_buffers", None)
        has_conflict = False
        for attr in ("total_ops", "total_params"):
            if isinstance(buffers, dict) and attr in buffers:
                continue
            try:
                static_attr = inspect.getattr_static(child, attr)
            except AttributeError:
                static_attr = None
            if static_attr is not None:
                has_conflict = True
                break
        if has_conflict:
            conflicts.append((name or child.__class__.__name__, child))
    return conflicts


@contextmanager
def _safe_thop_register_buffer():
    """
    Temporarily patch nn.Module.register_buffer so thop can re-register
    its profiling buffers even if a previous run left them attached, and to
    tolerate modules that already implement attributes named total_ops/total_params.
    """
    from torch.nn.modules.module import _global_buffer_registration_hooks

    original_register_buffer = torch.nn.Module.register_buffer
    special_names = {"total_ops", "total_params"}

    def patched_register_buffer(self, name, tensor, persistent=True):  # type: ignore[override]
        if name in special_names and hasattr(self, name):
            try:
                delattr(self, name)
            except AttributeError:
                buffers = getattr(self, "_buffers", None)
                if isinstance(buffers, dict):
                    buffers.pop(name, None)
        try:
            return original_register_buffer(self, name, tensor, persistent=persistent)
        except KeyError as err:
            if name not in special_names:
                raise
            # Allow thop to register profiling buffers even if a module already exposes
            # descriptors named total_ops/total_params (properties, methods, etc.).
            buffers = getattr(self, "_buffers", None)
            if buffers is None:
                raise err
            for hook in _global_buffer_registration_hooks.values():
                output = hook(self, name, tensor)
                if output is not None:
                    tensor = output
            buffers[name] = tensor
            if persistent:
                self._non_persistent_buffers_set.discard(name)
            else:
                self._non_persistent_buffers_set.add(name)
            return None

    torch.nn.Module.register_buffer = patched_register_buffer  # type: ignore[assignment]
    try:
        yield
    finally:
        torch.nn.Module.register_buffer = original_register_buffer  # type: ignore[assignment]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", nargs="?", default=None, help="Path to .ckpt checkpoint file (optional if --config is provided)")
    parser.add_argument("--device", type=str, default="auto", help="cuda or cpu or auto")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--views", type=int, default=9)
    parser.add_argument("--c", type=int, default=3)
    parser.add_argument("--h", type=int, default=256)
    parser.add_argument("--w", type=int, default=256)
    parser.add_argument("--prompt", type=str, default="Describe the panoramic scene:")
    parser.add_argument("--max-text-len", type=int, default=128)
    parser.add_argument("--mode", type=str, default="forward", choices=["forward", "generate"],
                        help="FLOPs 모드: forward (학습/finetune 경로) 또는 generate (디코딩 루프 근사)")
    parser.add_argument("--config", type=str, default=None,
                        help="YAML config path for scratch model instantiation (use instead of checkpoint)")
    # Optional: use a lightweight single-projection wrapper instead of full PanoramaVLM pipeline for forward mode
    parser.add_argument("--single-projection", action="store_true",
                        help="Approximate old single-projection architecture for FLOPs in forward mode (vision->resampler->single linear)")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint) if args.checkpoint else None
    config_path = Path(args.config) if args.config else None

    if ckpt_path is None and config_path is None:
        print("Please provide either a checkpoint path or --config <yaml> for scratch models.")
        sys.exit(2)

    if ckpt_path is not None and not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        sys.exit(2)

    if config_path is not None and not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(2)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    device_obj = torch.device(device)

    def _derive_model_config(cfg: dict) -> ModelConfig:
        pseudo_json = {
            "models": cfg.get("models", {}),
            "data": cfg.get("data", {}),
            "image_processing": cfg.get("image_processing", {}),
            "training": cfg.get("training", {}),
            "lora": cfg.get("lora", {}),
        }
        flat = ConfigManager._flatten_json_config(pseudo_json)
        return ModelConfig.from_dict(flat)

    def _ensure_projection_defaults_for_factory(cfg_obj: ModelConfig) -> ModelConfig:
        updates = {}
        if getattr(cfg_obj, "vicreg_projector_dim", None) is None:
            updates["vicreg_projector_dim"] = int(getattr(cfg_obj, "latent_dimension", 768) * 2)
        if getattr(cfg_obj, "vicreg_projector_depth", None) is None:
            updates["vicreg_projector_depth"] = 2
        if getattr(cfg_obj, "vicreg_projector_hidden", None) is None:
            updates["vicreg_projector_hidden"] = updates.get("vicreg_projector_dim", cfg_obj.latent_dimension * 2)
        if getattr(cfg_obj, "projection_intermediate_size", None) is None:
            updates["projection_intermediate_size"] = cfg_obj.latent_dimension * 2
        return cfg_obj.model_copy(update=updates) if updates else cfg_obj

    def _load_model_from_checkpoint(path: Path) -> PanoramaVLM:
        print(f"Loading pretrained PanoramaVLM checkpoint: {path} (device={device})")
        return PanoramaVLM.from_checkpoint(str(path), device=device)

    def _build_model_from_config(path: Path) -> PanoramaVLM:
        print(f"Building PanoramaVLM from config: {path} (device={device})")
        cfg_dict = load_config_dict(str(path))
        model_config = _ensure_projection_defaults_for_factory(_derive_model_config(cfg_dict))
        factory = ModelFactory(model_config)
        return factory.build().to(device_obj)

    if ckpt_path is not None:
        base_model = _load_model_from_checkpoint(ckpt_path)
    else:
        if config_path is None:
            print("Unable to resolve ModelConfig for scratch build. Provide --config.")
            sys.exit(2)
        base_model = _build_model_from_config(config_path)

    base_model.eval()

    # Build dummy inputs
    B = args.batch
    V = args.views
    C = args.c
    H = args.h
    W = args.w

    pixel_values = torch.randn(B, V, C, H, W, device=device_obj)

    # Build a short prompt using the model tokenizer if available
    input_ids = None
    attention_mask = None
    labels = None
    try:
        if hasattr(base_model, 'tokenizer') and base_model.tokenizer is not None:
            toks = base_model.tokenizer(args.prompt, return_tensors='pt', padding='max_length', truncation=True, max_length=args.max_text_len)
            input_ids = toks.input_ids.to(device_obj)
            attention_mask = toks.attention_mask.to(device_obj)
            labels = input_ids.clone()
        else:
            # fallback synthetic input ids
            input_ids = torch.zeros(B, args.max_text_len, dtype=torch.long, device=device_obj)
            attention_mask = torch.ones_like(input_ids)
            labels = input_ids.clone()
    except Exception as e:
        print(f"Warning: tokenizer usage failed: {e}. Using synthetic token ids.")
        input_ids = torch.zeros(B, args.max_text_len, dtype=torch.long, device=device_obj)
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()

    # forward 모드: 학습/finetune 경로 FLOPs 측정
    if args.mode == "forward":
        # Optional lightweight single-projection wrapper
        if args.single_projection:
            import torch.nn as nn

            class SingleProjectionWrapper(nn.Module):
                def __init__(self, pano_model: PanoramaVLM):
                    super().__init__()
                    self.vision_backbone = pano_model.vision_backbone
                    self.resampler_module = pano_model.resampler_module
                    # use the existing projector's linear as single projection layer
                    self.projector = pano_model.projector.linear

                def forward(self, pixel_values: torch.Tensor, input_ids, attention_mask=None, labels=None, stage="finetune"):
                    # stage 인자는 무시하고 purely vision+resampler+projection만 수행
                    vision_result = self.vision_backbone(pixel_values)
                    vision_features = vision_result["vision_features"]  # [B*V, S, D_vision]
                    batch_size = vision_result["batch_size"]
                    num_views = vision_result["num_views"]

                    resampled = self.resampler_module(vision_features, batch_size, num_views)  # [B*V, S, D_latent]
                    B, V = batch_size, num_views
                    resampled = resampled.view(B, V, -1, resampled.size(-1))  # [B,V,S,D_latent]
                    tokens = resampled.flatten(1, 2)  # [B, V*S, D_latent]
                    out = self.projector(tokens)  # [B, V*S, D_lm]
                    return {"vision_tokens": out}

            model = SingleProjectionWrapper(base_model).to(device_obj)
        else:
            model = base_model

        sample_args = (pixel_values, input_ids, attention_mask, labels, 'finetune')

        # Try fvcore first
        try:
            from fvcore.nn import FlopCountAnalysis

            print("Using fvcore.nn.FlopCountAnalysis to compute FLOPs (forward mode)...")
            flops = FlopCountAnalysis(model, sample_args)
            total = flops.total()
            total_flops = sum(total.values()) if isinstance(total, dict) else float(total)
            print(f"Total FLOPs (fvcore, forward): {total_flops:,}")
            print(f"Human: {humanize_flops(total_flops)}")
            return
        except Exception:
            print("fvcore not available or failed - falling back to thop if present.")

        try:
            from thop import profile

            print("Using thop.profile to compute FLOPs (forward mode, single pass)...")

            import torch.nn as nn

            class ThopWrapper(nn.Module):
                def __init__(self, inner: torch.nn.Module):
                    super().__init__()
                    self.inner = inner
                    self._thop_conflicts = _find_thop_conflict_modules(inner)
                    if self._thop_conflicts:
                        print(
                            "[thop] Skipping buffer registration for modules that already "
                            "define total_ops/total_params attributes:"
                        )
                        for mod_name, mod in self._thop_conflicts:
                            print(f"  - {mod_name or mod.__class__.__name__} ({mod.__class__.__name__})")
                    self._thop_conflict_ids = {id(mod) for _, mod in self._thop_conflicts}

                def forward(self, pixel_values, input_ids, attention_mask, labels):
                    return self.inner(
                        pixel_values,
                        input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        stage="finetune",
                    )

                def apply(self, fn):  # type: ignore[override]
                    """
                    Override nn.Module.apply to ensure thop only registers hooks once per nn.Module.

                    PanoramaVLM re-exports several inner modules under multiple attribute names for
                    backward compatibility. On some PyTorch builds this can make Module.apply visit
                    the same module object more than once, causing thop.profile's buffer registration
                    to raise "attribute 'total_ops' already exists". We traverse the module tree
                    manually while keeping a seen-set to guard against duplicate visits.
                    """
                    visited = set()
                    skip_ids = getattr(self, "_thop_conflict_ids", set())

                    def _apply(module: torch.nn.Module):
                        if id(module) in visited:
                            return module
                        visited.add(id(module))
                        for child in module.children():
                            _apply(child)
                        if skip_ids and id(module) in skip_ids:
                            return module
                        fn(module)
                        return module

                    _apply(self)
                    return self

            wrapped = ThopWrapper(model)

            _clear_thop_attributes(wrapped)

            with _safe_thop_register_buffer():
                macs, params = profile(
                    wrapped,
                    inputs=(pixel_values, input_ids, attention_mask, labels),
                    verbose=False,
                )
            flops = 2 * macs
            print(f"MACs: {macs:,}")
            print(f"Estimated FLOPs (2*MACs, forward): {flops:,}")
            print(f"Params: {params:,}")
            print(f"Human: {humanize_flops(flops)}")
            return
        except Exception as e:
            print(f"thop not available or failed: {e}")

    # generate 모드: vision 파이프라인 + LM generate FLOPs 근사
    if args.mode == "generate":
        # 1) 비전 파이프라인 FLOPs 측정용 래퍼 (vision_encoder+resampler+projection)
        import torch.nn as nn

        class VisionPipelineWrapper(nn.Module):
            def __init__(self, pano_model: PanoramaVLM):
                super().__init__()
                self.model = pano_model

            def forward(self, pixel_values: torch.Tensor):
                vision_result = self.model._process_vision_encoder(pixel_values)
                vision_features = vision_result["vision_features"]
                batch_size = vision_result["batch_size"]
                num_views = vision_result["num_views"]
                resampled = self.model._process_resampler(vision_features, batch_size, num_views)
                vision_tokens = self.model._process_projection_layer(resampled, batch_size, num_views)
                return vision_tokens

        vision_wrapper = VisionPipelineWrapper(base_model).to(device_obj)

        # 1-1) Vision FLOPs
        try:
            from fvcore.nn import FlopCountAnalysis

            print("Using fvcore.nn.FlopCountAnalysis to compute FLOPs (vision pipeline for generate)...")
            flops_vision = FlopCountAnalysis(vision_wrapper, (pixel_values,))
            total_v = flops_vision.total()
            total_v_flops = sum(total_v.values()) if isinstance(total_v, dict) else float(total_v)
        except Exception:
            total_v_flops = None

        # 2) LM generate FLOPs 근사: 작은 max_new_tokens로 generate 실행 후 선형 확장
        #    여기서는 실제 generate(max_new_tokens=k)를 프로파일해서 k-step FLOPs를 얻은 뒤, 원하는 길이에 맞게 scaling
        small_tokens = min(8, args.max_text_len)  # 너무 크지 않게 작은 길이로 측정
        max_new_tokens = small_tokens

        # dummy generate inputs
        gen_pixel = pixel_values
        gen_input_ids = input_ids
        gen_attention_mask = attention_mask

        def generate_wrapper(pv, ids, attn):
            return base_model.generate(pv, input_ids=ids, attention_mask=attn, max_new_tokens=max_new_tokens)

        total_gen_flops = None
        try:
            from fvcore.nn import FlopCountAnalysis

            print("Using fvcore.nn.FlopCountAnalysis to compute FLOPs (generate, small max_new_tokens)...")
            flops_gen = FlopCountAnalysis(generate_wrapper, (gen_pixel, gen_input_ids, gen_attention_mask))
            total_g = flops_gen.total()
            total_gen_flops = sum(total_g.values()) if isinstance(total_g, dict) else float(total_g)
        except Exception:
            print("fvcore for generate failed or unavailable - LM generate FLOPs will not be profiled.")

        # 결과 요약
        if total_v_flops is not None:
            print(f"Vision pipeline FLOPs (fvcore, once): {total_v_flops:,} [{humanize_flops(total_v_flops)}]")
        else:
            print("Vision pipeline FLOPs could not be computed (fvcore unavailable).")

        if total_gen_flops is not None:
            per_step = total_gen_flops / max_new_tokens
            print(f"Generate FLOPs for max_new_tokens={max_new_tokens}: {total_gen_flops:,} [{humanize_flops(total_gen_flops)}]")
            print(f"Approx per-step FLOPs: {per_step:,} [{humanize_flops(per_step)}]")
        else:
            print("Generate FLOPs could not be computed (fvcore unavailable).")

        return

    # 공통 Fallback: fvcore/thop 둘 다 실패했을 때 파라미터 수만 출력
    try:
        total_params = sum(p.numel() for p in base_model.parameters())
        trainable = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
        print("Could not compute FLOPs (fvcore/thop not fully available).\nShowing parameter counts instead:")
        print(f"Total params: {total_params:,}")
        print(f"Trainable params: {trainable:,}")
        print("Install `fvcore` or `thop` to compute FLOPs: pip install fvcore thop")
    except Exception as e:
        print(f"Failed to compute parameter counts: {e}")


if __name__ == '__main__':
    main()
