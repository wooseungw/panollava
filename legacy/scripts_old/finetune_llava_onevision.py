#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLaVA-OneVision-4B LoRA Fine-tuning & Evaluation Script

이 스크립트는 다음을 수행합니다:
1. LLaVA-OneVision-4B를 LoRA로 fine-tuning
2. Fine-tuned 모델로 평가 수행
3. 평가 결과 시각화

사용법:
    python scripts/finetune_llava_onevision.py \
        --train_csv data/quic360/train.csv \
        --val_csv data/quic360/valid.csv \
        --test_csv data/quic360/test.csv \
        --output_dir ablation/finetuning/llava-onevision-4b \
        --epochs 3 \
        --batch_size 4 \
        --learning_rate 2e-4 \
        --lora_rank 16
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
# LoRA 관련 임포트 (선택적)
try:
    from peft import LoraConfig, get_peft_model, PeftModel
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False
    logging.warning("⚠️ peft not found. Please install with: pip install peft")

try:
    from datasets import Dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    logging.warning("⚠️ datasets not found. Please install with: pip install datasets")

# 프로젝트 루트 추가
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# 평가 메트릭 임포트 (eval.py 사용)
try:
    from scripts.eval import calculate_evaluation_metrics, basic_cleanup
    USE_EVAL_METRICS = True
    logging.info("✓ Using eval.py metrics")
except ImportError:
    USE_EVAL_METRICS = False
    basic_cleanup = None
    logging.warning("⚠️ Could not import eval.py metrics, metrics calculation will be limited")

# 시각화 도구 임포트 (선택적)
try:
    from scripts.visualize import visualize_sample_predictions
    USE_VISUALIZATION = True
except ImportError:
    USE_VISUALIZATION = False
    logging.warning("⚠️ Could not import visualization tools")

# qwen_vl_utils for image processing
try:
    from qwen_vl_utils import process_vision_info
    HAS_QWEN_VL_UTILS = True
except ImportError:
    HAS_QWEN_VL_UTILS = False
    logging.warning("⚠️ qwen_vl_utils not found. Please install it.")

# Allow large images
Image.MAX_IMAGE_PIXELS = None

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# ============================================================
# 데이터셋 준비
# ============================================================

class LLaVAOneVisionDataset(torch.utils.data.Dataset):
    """LLaVA-OneVision용 데이터셋"""

    def __init__(
        self,
        csv_path: str,
        processor,
        max_length: int = 512,
        image_column: str = "url",
        instruction_column: str = "query",
        response_column: str = "annotation",
    ):
        self.df = pd.read_csv(csv_path)
        self.processor = processor
        self.max_length = max_length
        self.image_column = image_column
        self.instruction_column = instruction_column
        self.response_column = response_column

        logging.info(f"Loaded {len(self.df)} samples from {csv_path}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 이미지 로드
        image_path = row[self.image_column]
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logging.warning(f"Failed to load image {image_path}: {e}")
            # Return a dummy black image
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        # 텍스트 준비
        instruction = str(row.get(self.instruction_column, "Describe the image."))
        response = str(row.get(self.response_column, ""))

        # 메시지 포맷
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": instruction}
            ]
        }]

        # Chat template 적용
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Add response for training
        full_text = text + response

        # Process vision info
        if HAS_QWEN_VL_UTILS:
            image_inputs, video_inputs = process_vision_info(messages)
        else:
            image_inputs = [image]
            video_inputs = None

        # Tokenize
        inputs = self.processor(
            text=[full_text],
            images=image_inputs if image_inputs else None,
            videos=video_inputs if video_inputs else None,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

        # Prepare labels (mask prompt part)
        labels = inputs["input_ids"].clone()

        # Find where the assistant response starts (after generation prompt)
        prompt_inputs = self.processor(
            text=[text],
            images=image_inputs if image_inputs else None,
            videos=video_inputs if video_inputs else None,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        prompt_length = (prompt_inputs["input_ids"] != self.processor.tokenizer.pad_token_id).sum()
        labels[0, :prompt_length] = -100  # Ignore prompt tokens

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "pixel_values": inputs.get("pixel_values", None).squeeze(0) if "pixel_values" in inputs else None,
            "labels": labels.squeeze(0),
        }


def collate_fn(batch):
    """Custom collate function"""
    input_ids = torch.stack([x["input_ids"] for x in batch])
    attention_mask = torch.stack([x["attention_mask"] for x in batch])
    labels = torch.stack([x["labels"] for x in batch])

    result = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

    # Add pixel_values if present
    if batch[0]["pixel_values"] is not None:
        pixel_values = torch.stack([x["pixel_values"] for x in batch])
        result["pixel_values"] = pixel_values

    return result


# ============================================================
# Fine-tuning
# ============================================================

def setup_lora_model(
    model,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    target_modules: Optional[List[str]] = None,
):
    """Setup LoRA for the model"""

    if not HAS_PEFT:
        raise ImportError("peft is required for LoRA. Install with: pip install peft")

    if target_modules is None:
        # Default target modules for LLaVA-OneVision (Qwen2.5 based)
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model


def train_model(
    model,
    processor,
    train_csv: str,
    val_csv: str,
    output_dir: str,
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    max_length: int = 512,
    save_steps: int = 500,
    logging_steps: int = 10,
):
    """Train the model with LoRA"""

    logging.info("=" * 60)
    logging.info("Setting up training datasets...")
    logging.info("=" * 60)

    # 데이터셋 준비
    train_dataset = LLaVAOneVisionDataset(
        train_csv, processor, max_length=max_length
    )
    val_dataset = LLaVAOneVisionDataset(
        val_csv, processor, max_length=max_length
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=1,
        learning_rate=learning_rate,
        warmup_steps=100,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=2,
        evaluation_strategy="steps",
        eval_steps=save_steps,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to="none",  # Disable wandb/tensorboard
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
    )

    logging.info("=" * 60)
    logging.info("Starting training...")
    logging.info("=" * 60)

    # Train
    train_result = trainer.train()

    # Save final model
    logging.info("=" * 60)
    logging.info(f"Saving model to {output_dir}")
    logging.info("=" * 60)

    trainer.save_model(output_dir)

    # Save training metrics
    metrics_file = Path(output_dir) / "training_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(train_result.metrics, f, indent=2)

    logging.info(f"Training metrics saved to {metrics_file}")

    return trainer


# ============================================================
# Evaluation
# ============================================================

def evaluate_model(
    model,
    processor,
    test_csv: str,
    output_dir: str,
    batch_size: int = 1,
    max_new_tokens: int = 128,
    image_column: str = "url",
    instruction_column: str = "query",
    response_column: str = "annotation",
):
    """Evaluate the fine-tuned model"""

    logging.info("=" * 60)
    logging.info("Starting evaluation...")
    logging.info("=" * 60)

    # Load test data
    df = pd.read_csv(test_csv)
    logging.info(f"Loaded {len(df)} test samples")

    predictions = []
    references = []
    instructions = []
    image_paths = []

    model.eval()
    device = next(model.parameters()).device

    with torch.inference_mode():
        for idx in tqdm(range(len(df)), desc="Evaluating"):
            row = df.iloc[idx]

            # Load image
            image_path = row[image_column]
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                logging.warning(f"Failed to load image {image_path}: {e}")
                continue

            # Prepare prompt
            instruction = str(row.get(instruction_column, "Describe the image."))
            reference = str(row.get(response_column, ""))

            # Message format
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": instruction}
                ]
            }]

            # Chat template
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Process vision info
            if HAS_QWEN_VL_UTILS:
                image_inputs, video_inputs = process_vision_info(messages)
            else:
                image_inputs = [image]
                video_inputs = None

            # Prepare inputs
            inputs = processor(
                text=[text],
                images=image_inputs if image_inputs else None,
                videos=video_inputs if video_inputs else None,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

            # Generate
            with torch.cuda.amp.autocast(enabled=True):
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=processor.tokenizer.pad_token_id,
                )

            # Decode (remove prompt)
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = generated_ids[0, input_length:].cpu()
            prediction = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

            predictions.append(prediction)
            references.append(reference)
            instructions.append(instruction)
            image_paths.append(image_path)

    # Calculate metrics using eval.py
    logging.info("=" * 60)
    logging.info("Calculating metrics with eval.py...")
    logging.info("=" * 60)

    # Apply basic_cleanup to predictions if available
    if USE_EVAL_METRICS and basic_cleanup is not None:
        logging.info("Applying basic_cleanup to predictions...")
        predictions_cleaned = [basic_cleanup(pred) for pred in predictions]
        references_cleaned = [basic_cleanup(ref) for ref in references]
    else:
        predictions_cleaned = predictions
        references_cleaned = references

    if USE_EVAL_METRICS:
        # Prepare DataFrame for eval.py
        eval_df = pd.DataFrame({
            'sample_id': list(range(len(predictions_cleaned))),
            'image_path': image_paths,
            'prediction': predictions_cleaned,
            'reference': references_cleaned,
            'original_query': instructions,
        })

        # Save temporary CSV for eval.py
        temp_csv = Path(output_dir) / "temp_predictions.csv"
        eval_df.to_csv(temp_csv, index=False, encoding='utf-8')

        # Calculate metrics using eval.py
        try:
            metrics = calculate_evaluation_metrics(
                data_input=temp_csv,
                output_dir=Path(output_dir),
                timestamp=time.strftime('%Y%m%d_%H%M%S'),
                prefix='llava_onevision_4b'
            )
            logging.info("✓ Metrics calculated successfully using eval.py")
        except Exception as e:
            logging.error(f"Failed to calculate metrics with eval.py: {e}")
            metrics = {}

        # Clean up temp file
        if temp_csv.exists():
            temp_csv.unlink()
    else:
        # Fallback: simple exact match
        exact_matches = sum([p.lower() == r.lower() for p, r in zip(predictions_cleaned, references_cleaned)])
        metrics = {
            "exact_match": exact_matches / len(predictions_cleaned) if predictions_cleaned else 0.0
        }
        logging.warning("Using fallback metrics (exact match only)")

    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics
    metrics_file = output_dir / "eval_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    logging.info(f"Metrics saved to {metrics_file}")

    # Save predictions (both cleaned and original)
    predictions_df = pd.DataFrame({
        "sample_id": list(range(len(image_paths))),
        "image_path": image_paths,
        "original_query": instructions,
        "prediction": predictions_cleaned,
        "reference": references_cleaned,
        "prediction_raw": predictions,
        "reference_raw": references,
        "pred_length": [len(p.split()) for p in predictions_cleaned],
        "ref_length": [len(r.split()) for r in references_cleaned],
    })
    predictions_file = output_dir / "predictions.csv"
    predictions_df.to_csv(predictions_file, index=False, encoding="utf-8")
    logging.info(f"Predictions saved to {predictions_file}")

    # Print metrics
    logging.info("\n" + "=" * 60)
    logging.info("Evaluation Results:")
    logging.info("=" * 60)
    for key, value in metrics.items():
        logging.info(f"{key}: {value:.4f}")
    logging.info("=" * 60)

    return metrics, predictions_df


# ============================================================
# Visualization
# ============================================================

def create_visualizations(
    predictions_df: pd.DataFrame,
    output_dir: str,
    num_samples: int = 10,
):
    """Create visualization of sample predictions"""

    logging.info("=" * 60)
    logging.info("Creating visualizations...")
    logging.info("=" * 60)

    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend

    output_dir = Path(output_dir)
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Sample random predictions
    sample_indices = np.random.choice(
        len(predictions_df),
        size=min(num_samples, len(predictions_df)),
        replace=False
    )

    for idx in sample_indices:
        row = predictions_df.iloc[idx]

        # Load image
        try:
            image = Image.open(row["image_path"]).convert("RGB")
        except Exception as e:
            logging.warning(f"Failed to load image: {e}")
            continue

        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image)
        ax.axis("off")

        # Add text
        title = f"Instruction: {row['instruction'][:100]}"
        prediction_text = f"Prediction: {row['prediction'][:200]}"
        reference_text = f"Reference: {row['reference'][:200]}"

        fig.suptitle(title, fontsize=12, fontweight='bold')
        plt.figtext(0.5, 0.05, prediction_text, ha='center', fontsize=10, color='blue', wrap=True)
        plt.figtext(0.5, 0.01, reference_text, ha='center', fontsize=10, color='green', wrap=True)

        # Save
        output_file = viz_dir / f"sample_{idx:03d}.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        logging.info(f"Saved visualization: {output_file}")

    logging.info(f"All visualizations saved to {viz_dir}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune LLaVA-OneVision-4B with LoRA on Quic360 dataset"
    )

    # Data arguments
    parser.add_argument("--train_csv", type=str, default="data/quic360/train.csv", help="Training data CSV")
    parser.add_argument("--val_csv", type=str, default="data/quic360/valid.csv", help="Validation data CSV")
    parser.add_argument("--test_csv", type=str, default="data/quic360/test.csv", help="Test data CSV")
    parser.add_argument("--image_column", type=str, default="url", help="Image path column name")
    parser.add_argument("--instruction_column", type=str, default="query", help="Instruction column name")
    parser.add_argument("--response_column", type=str, default="annotation", help="Response column name")

    # Model arguments
    parser.add_argument("--model_id", type=str, default="lmms-lab/LLaVA-OneVision-1.5-4B-Instruct", help="Model ID")
    parser.add_argument("--output_dir", type=str, default="ablation/finetuning/llava-onevision-4b", help="Output directory")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every N steps")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log every N steps")

    # LoRA arguments
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")

    # Evaluation arguments
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Max tokens for generation")
    parser.add_argument("--num_viz_samples", type=int, default=10, help="Number of samples to visualize")

    # Other arguments
    parser.add_argument("--skip_training", action="store_true", help="Skip training and only evaluate")
    parser.add_argument("--skip_evaluation", action="store_true", help="Skip evaluation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save arguments
    args_file = output_dir / "args.json"
    with open(args_file, "w") as f:
        json.dump(vars(args), f, indent=2)
    logging.info(f"Arguments saved to {args_file}")

    # ============================================================
    # Load model and processor
    # ============================================================

    logging.info("=" * 60)
    logging.info(f"Loading model: {args.model_id}")
    logging.info("=" * 60)

    # Load processor
    processor = AutoProcessor.from_pretrained(
        args.model_id,
        trust_remote_code=True,
    )

    # Ensure pad token is set
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    if not args.skip_training:
        # Load model for training
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=args.device,
            trust_remote_code=True,
        )

        # Setup LoRA
        logging.info("=" * 60)
        logging.info("Setting up LoRA...")
        logging.info("=" * 60)

        model = setup_lora_model(
            model,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )

        # ============================================================
        # Training
        # ============================================================

        trainer = train_model(
            model=model,
            processor=processor,
            train_csv=args.train_csv,
            val_csv=args.val_csv,
            output_dir=str(output_dir / "checkpoints"),
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
            save_steps=args.save_steps,
            logging_steps=args.logging_steps,
        )

        # Save final LoRA weights
        model.save_pretrained(output_dir / "lora_weights")
        logging.info(f"LoRA weights saved to {output_dir / 'lora_weights'}")

    # ============================================================
    # Evaluation
    # ============================================================

    if not args.skip_evaluation:
        # Load fine-tuned model for evaluation
        logging.info("=" * 60)
        logging.info("Loading fine-tuned model for evaluation...")
        logging.info("=" * 60)

        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=args.device,
            trust_remote_code=True,
        )

        # Load LoRA weights
        model = PeftModel.from_pretrained(base_model, str(output_dir / "lora_weights"))
        model.eval()

        # Evaluate
        metrics, predictions_df = evaluate_model(
            model=model,
            processor=processor,
            test_csv=args.test_csv,
            output_dir=str(output_dir / "evaluation"),
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            image_column=args.image_column,
            instruction_column=args.instruction_column,
            response_column=args.response_column,
        )

        # ============================================================
        # Visualization
        # ============================================================

        create_visualizations(
            predictions_df=predictions_df,
            output_dir=str(output_dir / "evaluation"),
            num_samples=args.num_viz_samples,
        )

    logging.info("\n" + "=" * 60)
    logging.info("✅ All done!")
    logging.info(f"Results saved to: {output_dir}")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
