"""Generic evaluation utility for Hugging Face vision-language models (VLMs).

This script extends the earlier Qwen-specific evaluator so that it can score
any LoRA-tuned VLM that lives under ``results/vlm_lora_ablation``.  It reuses
``scripts.eval`` for prediction logging and metric computation, making the
output consistent with the PanoramaVLM pipeline.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence
import sys

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor

# Ensure repository root is importable so we can reuse scripts.eval utilities
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SRC_DIR = _REPO_ROOT / "src"
if _SRC_DIR.is_dir() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

try:  # Optional specialised model heads
    from transformers import (
        Blip2ForConditionalGeneration,
        LlavaForConditionalGeneration,
        LlavaProcessor,
        Qwen2_5_VLForConditionalGeneration,
    )
except ImportError:  # pragma: no cover - optional
    Blip2ForConditionalGeneration = None
    LlavaForConditionalGeneration = None
    LlavaProcessor = None
    Qwen2_5_VLForConditionalGeneration = None

try:  # LoRA support
    from peft import PeftModel

    PEFT_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    PEFT_AVAILABLE = False

# Shared persistence helpers from scripts/eval.py
from scripts.eval import (  # type: ignore
    calculate_evaluation_metrics,
    save_and_log_results,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_PROMPT_TEMPLATE = (
    "In this panoramic image, please provide a concise but detailed description of \"{query}\"."
)
DEFAULT_SYSTEM_MESSAGE = (
    "You are an expert assistant specialized in analyzing panoramic images. "
    "Please provide detailed, accurate, and helpful responses about what you observe."
)
DEFAULT_RESULTS_ROOT = Path("results/vlm_lora_ablation")


def _init_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    LOGGER.setLevel(level)


@dataclass
class RunArtifacts:
    run_dir: Path
    lora_path: Optional[Path]
    base_model: Optional[str]
    processor_id: Optional[str]
    prompt_template: Optional[str] = None
    system_message: Optional[str] = None


@dataclass
class EvalConfig:
    model_id: str
    processor_id: str
    csv_path: Path
    image_column: str
    query_column: str
    reference_column: str
    output_dir: Path
    output_prefix: str
    batch_size: int
    num_workers: int
    max_new_tokens: int
    min_new_tokens: int
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float
    device: str
    dtype: str
    prompt_template: str
    system_message: Optional[str]
    lora_path: Optional[Path]
    trust_remote_code: bool
    seed: Optional[int]
    model_type: str


class SafeDict(dict):
    """`dict` that returns empty string for missing keys (for str.format_map)."""

    def __missing__(self, key: str) -> str:  # pragma: no cover - defensive
        return ""


class VisionLanguageEvalDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        processor: AutoProcessor,
        image_column: str,
        query_column: str,
        reference_column: str,
        prompt_template: str,
        system_message: Optional[str],
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.image_column = image_column
        self.query_column = query_column
        self.reference_column = reference_column
        self.prompt_template = prompt_template
        self.system_message = system_message

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, path_str: str) -> Image.Image:
        image_path = Path(path_str)
        if not image_path.is_file():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        return Image.open(image_path).convert("RGB")

    def _build_messages(self, prompt_text: str) -> List[Dict[str, object]]:
        messages: List[Dict[str, object]] = []
        if self.system_message:
            messages.append(
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": str(self.system_message)},
                    ],
                }
            )
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_text},
                ],
            }
        )
        return messages

    def __getitem__(self, index: int) -> Dict[str, object]:
        row = self.df.iloc[index]
        context = {
            col: "" if pd.isna(row[col]) else str(row[col])
            for col in self.df.columns
        }
        query_value = context.get(self.query_column, "")
        reference_value = context.get(self.reference_column, "")
        context.setdefault("query", query_value)
        context.setdefault("instruction", query_value)
        context.setdefault("reference", reference_value)

        prompt_text = self.prompt_template.format_map(SafeDict(context))
        messages = self._build_messages(prompt_text)

        if hasattr(self.processor, "apply_chat_template"):
            prompt_chat = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt_chat = prompt_text

        image_path = context.get(self.image_column, "")
        sample = {
            "prompt_chat": prompt_chat,
            "prompt_text": prompt_text,
            "query": query_value,
            "reference": reference_value,
            "image_path": image_path,
            "image": self._load_image(image_path),
        }
        return sample


def _collate_fn(processor: AutoProcessor) -> Callable[[Sequence[Dict[str, object]]], Dict[str, object]]:
    def collate(features: Sequence[Dict[str, object]]) -> Dict[str, object]:
        texts = [f["prompt_chat"] for f in features]
        images = [f["image"] for f in features]
        processor_inputs = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
        )
        processor_inputs["references"] = [f["reference"] for f in features]
        processor_inputs["image_paths"] = [f["image_path"] for f in features]
        processor_inputs["queries"] = [f["query"] for f in features]
        processor_inputs["prompt_texts"] = [f["prompt_text"] for f in features]
        return processor_inputs

    return collate


def _resolve_dtype(dtype_name: str, device: str) -> torch.dtype:
    dtype_name = dtype_name.lower()
    if dtype_name == "auto":
        if device.startswith("cuda"):
            return torch.float16
        return torch.float32
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    resolved = mapping[dtype_name]
    if device == "cpu" and resolved in {torch.float16, torch.bfloat16}:
        LOGGER.warning("Precision %s not supported on CPU; falling back to float32", dtype_name)
        return torch.float32
    return resolved


def _prepare_processor(processor_id: str, trust_remote_code: bool, model_type: str) -> AutoProcessor:
    if model_type == "llava" and LlavaProcessor is not None:
        processor = LlavaProcessor.from_pretrained(
            processor_id,
            trust_remote_code=trust_remote_code,
        )
    else:
        processor = AutoProcessor.from_pretrained(
            processor_id,
            trust_remote_code=trust_remote_code,
        )

    if hasattr(processor, "image_processor"):
        image_proc = processor.image_processor
        if model_type == "qwen_vl":
            if hasattr(image_proc, "min_pixels"):
                image_proc.min_pixels = 224 * 224
            if hasattr(image_proc, "max_pixels"):
                image_proc.max_pixels = 224 * 224

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None and hasattr(processor, "tokenizer_class"):  # pragma: no cover - defensive
        tokenizer = processor.tokenizer_class.from_pretrained(processor_id)

    if tokenizer is not None and getattr(tokenizer, "pad_token_id", None) is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            LOGGER.info("Tokenizer pad token set to eos token")
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
            LOGGER.info("Tokenizer pad token '<pad>' added")

    # Set padding_side to left for decoder-only models during generation
    if tokenizer is not None:
        tokenizer.padding_side = "left"
        LOGGER.info("Tokenizer padding_side set to 'left' for generation")

    return processor


def _prepare_model(config: EvalConfig, dtype: torch.dtype) -> torch.nn.Module:
    loader_map = {
        "qwen_vl": Qwen2_5_VLForConditionalGeneration,
        "llava": LlavaForConditionalGeneration,
        "blip2": Blip2ForConditionalGeneration,
    }
    loader = loader_map.get(config.model_type)
    if loader is None:
        LOGGER.info("Using AutoModelForCausalLM for model type '%s'", config.model_type)
        base_model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            dtype=dtype,
            trust_remote_code=config.trust_remote_code,
        )
    else:
        if loader is None:  # pragma: no cover - optional
            raise RuntimeError(
                f"Model type '{config.model_type}' requires transformers extras that are not installed."
            )
        base_model = loader.from_pretrained(
            config.model_id,
            dtype=dtype,
            trust_remote_code=config.trust_remote_code,
        )

    model: torch.nn.Module = base_model
    if config.lora_path is not None:
        if not PEFT_AVAILABLE:
            raise RuntimeError("peft is required to load LoRA weights")
        LOGGER.info("Loading LoRA adapter from %s", config.lora_path)
        model = PeftModel.from_pretrained(base_model, str(config.lora_path))

    if config.device != "auto":
        LOGGER.info("Moving model to device %s", config.device)
        model = model.to(config.device)

    model.eval()
    return model


def _set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _infer_model_type(model_type_arg: str, model_id: str) -> str:
    if model_type_arg and model_type_arg.lower() != "auto":
        return model_type_arg.lower()
    lower_id = model_id.lower()
    if "qwen" in lower_id and "vl" in lower_id:
        return "qwen_vl"
    if "llava" in lower_id:
        return "llava"
    if "blip2" in lower_id or "blip-2" in lower_id:
        return "blip2"
    return "auto"


def _discover_run_artifacts(run_name: str, results_root: Path) -> RunArtifacts:
    run_dir = results_root / run_name
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    candidates: Iterable[Path] = (
        run_dir / "lora_adapter",
        run_dir / "final",
    )

    lora_path: Optional[Path] = None
    adapter_config: Optional[Dict[str, object]] = None
    for candidate in candidates:
        config_path = candidate / "adapter_config.json"
        if config_path.is_file():
            lora_path = candidate
            adapter_config = json.loads(config_path.read_text(encoding="utf-8"))
            break

    if lora_path is None:
        # search checkpoints/*/adapter_config.json as fallback
        for config_path in run_dir.rglob("adapter_config.json"):
            lora_path = config_path.parent
            adapter_config = json.loads(config_path.read_text(encoding="utf-8"))
            LOGGER.warning(
                "Using adapter at %s (discovered recursively). Consider placing it in 'lora_adapter' for faster lookup.",
                lora_path,
            )
            break

    base_model = None
    if adapter_config:
        base_model = adapter_config.get("base_model_name_or_path") or adapter_config.get("base_model")

    return RunArtifacts(
        run_dir=run_dir,
        lora_path=lora_path,
        base_model=str(base_model) if base_model else None,
        processor_id=str(base_model) if base_model else None,
    )


def run_evaluation(cfg: EvalConfig) -> None:
    _set_seed(cfg.seed)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    processor = _prepare_processor(cfg.processor_id, cfg.trust_remote_code, cfg.model_type)
    dtype = _resolve_dtype(cfg.dtype, cfg.device)
    model = _prepare_model(cfg, dtype)

    LOGGER.info("Loading evaluation data from %s", cfg.csv_path)
    df = pd.read_csv(cfg.csv_path)
    missing_columns = [
        col
        for col in [cfg.image_column, cfg.query_column, cfg.reference_column]
        if col not in df.columns
    ]
    if missing_columns:
        raise ValueError(
            f"CSV missing required columns: {missing_columns}. Available columns: {list(df.columns)}"
        )

    dataset = VisionLanguageEvalDataset(
        df=df,
        processor=processor,
        image_column=cfg.image_column,
        query_column=cfg.query_column,
        reference_column=cfg.reference_column,
        prompt_template=cfg.prompt_template,
        system_message=cfg.system_message,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=_collate_fn(processor),
    )

    device = cfg.device
    if device == "auto":
        device = "cuda" if next(model.parameters()).is_cuda else "cpu"
    LOGGER.info("Using device: %s", device)

    predictions: List[str] = []
    references: List[str] = []
    image_paths: List[str] = []
    query_texts: List[str] = []

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("Processor does not expose a tokenizer for decoding outputs")

    LOGGER.info("Starting generation loop over %d batches", len(dataloader))
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generating")):
        references.extend(batch.pop("references"))
        image_paths.extend(batch.pop("image_paths"))
        query_texts.extend(batch.pop("queries"))
        prompt_texts = batch.pop("prompt_texts")

        model_inputs: Dict[str, torch.Tensor] = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                model_inputs[key] = value.to(device)

        if "input_ids" not in model_inputs or "attention_mask" not in model_inputs:
            raise RuntimeError("Processor did not return input_ids/attention_mask")

        prompt_lengths = model_inputs["attention_mask"].sum(dim=1).tolist()

        try:
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=cfg.max_new_tokens,
                min_new_tokens=cfg.min_new_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                top_k=cfg.top_k,
                repetition_penalty=cfg.repetition_penalty,
                do_sample=cfg.temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        except Exception as exc:  # pragma: no cover - generation failures are logged
            LOGGER.error("Generation failed for batch %d: %s", batch_idx, exc, exc_info=True)
            batch_size = len(prompt_texts)
            predictions.extend([f"[generation_error_{batch_idx}_{i}]" for i in range(batch_size)])
            continue

        for i, output_ids in enumerate(generated_ids):
            prompt_len = int(prompt_lengths[i])
            generated = output_ids[prompt_len:]
            decoded = tokenizer.decode(
                generated,
                skip_special_tokens=True,
            )
            prediction = decoded.strip() or "[빈 응답]"
            predictions.append(prediction)

    LOGGER.info("Generation finished. Processed %d samples", len(predictions))

    results_df = save_and_log_results(
        predictions=predictions,
        references=references,
        image_paths=image_paths,
        input_texts=query_texts,
        output_dir=cfg.output_dir,
        timestamp=timestamp,
        prefix=cfg.output_prefix,
    )

    metrics = calculate_evaluation_metrics(
        results_df,
        output_dir=cfg.output_dir,
        timestamp=timestamp,
        prefix=cfg.output_prefix,
    )

    if metrics:
        LOGGER.info("Evaluation metrics:")
        for key, value in metrics.items():
            LOGGER.info("  %s: %.4f", key, value)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Hugging Face VLM LoRA runs")
    parser.add_argument("--csv", required=True, help="CSV file with evaluation data")
    parser.add_argument("--image-column", default="url")
    parser.add_argument("--query-column", default="query")
    parser.add_argument("--reference-column", default="annotation")

    parser.add_argument(
        "--run",
        help="Run directory name under results/vlm_lora_ablation (e.g. qwen_vl_chat__lora_r16)",
    )
    parser.add_argument(
        "--results-root",
        default=str(DEFAULT_RESULTS_ROOT),
        help="Root directory that stores VLM LoRA runs",
    )

    parser.add_argument("--model-id", help="Base model id/path (defaults to run metadata)")
    parser.add_argument("--processor-id", help="Processor id (defaults to model id)")
    parser.add_argument("--lora-path", help="Path to LoRA adapter (defaults to run metadata)")
    parser.add_argument(
        "--model-type",
        default="auto",
        help="Model family hint (auto, qwen_vl, llava, blip2, ...)",
    )

    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--min-new-tokens", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--dtype",
        default="auto",
        help="Precision to use (auto, float16, bfloat16, float32)",
    )

    parser.add_argument("--prompt-template", default=DEFAULT_PROMPT_TEMPLATE)
    parser.add_argument("--system-message", default=DEFAULT_SYSTEM_MESSAGE)
    parser.add_argument("--no-system-message", action="store_true", help="Disable system message")
    parser.add_argument("--output-dir", default="eval_results/vlm")
    parser.add_argument(
        "--output-prefix",
        help="Prefix used for saved CSV/JSON (defaults to run name or sanitized model id)",
    )
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--no-remote-code", action="store_true", help="Disable trust_remote_code")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser


def _sanitize_prefix(value: str) -> str:
    for ch in ("/", "\\", " "):
        value = value.replace(ch, "_")
    return value


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    _init_logging(args.verbose)

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    run_artifacts = None
    if args.run:
        run_artifacts = _discover_run_artifacts(args.run, Path(args.results_root))
        LOGGER.info("Discovered run '%s' (LoRA: %s, base model: %s)", args.run, run_artifacts.lora_path, run_artifacts.base_model)

    model_id = args.model_id or (run_artifacts.base_model if run_artifacts else None)
    if model_id is None:
        raise ValueError("model-id is required (either explicitly or via --run metadata)")

    processor_id = args.processor_id or (run_artifacts.processor_id if run_artifacts else None) or model_id

    lora_path = args.lora_path
    if lora_path is None and run_artifacts and run_artifacts.lora_path:
        lora_path = str(run_artifacts.lora_path)
    lora_path_path = Path(lora_path) if lora_path else None

    output_dir = Path(args.output_dir)
    if run_artifacts:
        default_prefix = run_artifacts.run_dir.name
    else:
        default_prefix = Path(model_id).name
    output_prefix = _sanitize_prefix(args.output_prefix or default_prefix)

    system_message = None if args.no_system_message else args.system_message

    model_type = _infer_model_type(args.model_type, model_id)

    config = EvalConfig(
        model_id=model_id,
        processor_id=processor_id,
        csv_path=csv_path,
        image_column=args.image_column,
        query_column=args.query_column,
        reference_column=args.reference_column,
        output_dir=output_dir,
        output_prefix=output_prefix,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        device=args.device,
        dtype=args.dtype,
        prompt_template=args.prompt_template,
        system_message=system_message,
        lora_path=lora_path_path,
        trust_remote_code=not args.no_remote_code,
        seed=args.seed,
        model_type=model_type,
    )

    run_evaluation(config)


if __name__ == "__main__":
    main()
