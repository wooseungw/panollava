"""LLM-as-a-judge scoring for CORA evaluation outputs."""

import argparse
import base64
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm import tqdm


LOGGER = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "You are a strict evaluator for panorama VLM answers. "
    "Score prediction quality against reference on a 1-10 integer scale. "
    "Return JSON only."
)

USER_PROMPT_TEMPLATE = """Evaluate this sample.

Query: {query}
Prediction: {prediction}
Reference: {reference}

Scoring rubric:
- semantic_similarity (0-4): core meaning match
- detail_accuracy (0-3): objects/counts/attributes correctness
- hallucination_penalty (0-2): fabricated content penalty
- panorama_boundary_penalty (0-1): 360 wrap-around misunderstanding penalty

Output JSON schema:
{{
  "score": <int 1-10>,
  "semantic_similarity": <int 0-4>,
  "detail_accuracy": <int 0-3>,
  "hallucination_penalty": <int 0-2>,
  "panorama_boundary_penalty": <int 0-1>,
  "reason": "<short Korean explanation>"
}}
"""


def _encode_image_data_url(image_path: Path) -> Optional[str]:
    if not image_path.exists() or not image_path.is_file():
        return None
    ext = image_path.suffix.lower()
    mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(ext, "image/jpeg")
    raw = image_path.read_bytes()
    b64 = base64.b64encode(raw).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _safe_int(value: Any, default: int, low: int, high: int) -> int:
    try:
        iv = int(value)
    except Exception:
        return default
    return max(low, min(high, iv))


def _parse_json_response(text: str) -> Dict[str, Any]:
    payload = text.strip()
    if "```json" in payload:
        payload = payload.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in payload:
        payload = payload.split("```", 1)[1].split("```", 1)[0].strip()

    start = payload.find("{")
    end = payload.rfind("}")
    if start != -1 and end != -1 and end > start:
        payload = payload[start : end + 1]

    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return {
            "score": 5,
            "semantic_similarity": 2,
            "detail_accuracy": 1,
            "hallucination_penalty": 1,
            "panorama_boundary_penalty": 0,
            "reason": "judge response parse failed",
            "parse_error": True,
        }

    return {
        "score": _safe_int(data.get("score", 5), 5, 1, 10),
        "semantic_similarity": _safe_int(data.get("semantic_similarity", 2), 2, 0, 4),
        "detail_accuracy": _safe_int(data.get("detail_accuracy", 1), 1, 0, 3),
        "hallucination_penalty": _safe_int(data.get("hallucination_penalty", 1), 1, 0, 2),
        "panorama_boundary_penalty": _safe_int(data.get("panorama_boundary_penalty", 0), 0, 0, 1),
        "reason": str(data.get("reason", "")),
    }


def _normalize_records(input_path: Path) -> List[Dict[str, Any]]:
    if input_path.suffix.lower() == ".json":
        data = json.loads(input_path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("predictions"), list):
            rows = data["predictions"]
        elif isinstance(data, list):
            rows = data
        else:
            raise ValueError("Unsupported JSON format. Expect list or {'predictions': [...]}.")
        df = pd.DataFrame(rows)
    else:
        df = pd.read_csv(input_path)

    records: List[Dict[str, Any]] = []
    for i, row in df.iterrows():
        query = row.get("query", None)
        if query is None:
            query = row.get("prompt", row.get("original_query", ""))

        prediction = row.get("prediction", None)
        if prediction is None:
            prediction = row.get("caption", row.get("output", ""))

        reference = row.get("reference", row.get("annotation", ""))

        image_path = ""
        for key in ("image_path", "file_name", "url", "image"):
            value = row.get(key, None)
            if value is not None and not pd.isna(value):
                image_path = str(value)
                break

        if reference is None or pd.isna(reference) or str(reference).strip() == "":
            continue

        sample_id = row.get("sample_id", row.get("image_id", i))
        try:
            normalized_sample_id = int(str(sample_id))
        except Exception:
            normalized_sample_id = i

        records.append(
            {
                "sample_id": normalized_sample_id,
                "query": str(query),
                "prediction": str(prediction),
                "reference": str(reference),
                "image_path": image_path,
            }
        )
    return records


class LLMJudge:
    def __init__(
        self,
        model: str,
        api_key: str,
        include_image: bool,
        base_path: Path,
        max_retries: int = 3,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise SystemExit("Install dependency first: pip install openai") from exc

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.include_image = include_image
        self.base_path = base_path
        self.max_retries = max_retries

    def _resolve_image(self, raw_path: str) -> Optional[Path]:
        if not raw_path:
            return None
        p = Path(raw_path)
        if p.is_absolute():
            return p
        return self.base_path / p

    def evaluate(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        user_prompt = USER_PROMPT_TEMPLATE.format(
            query=sample["query"],
            prediction=sample["prediction"],
            reference=sample["reference"],
        )

        content: List[Dict[str, Any]] = []
        if self.include_image:
            img_path = self._resolve_image(sample.get("image_path", ""))
            if img_path is not None:
                data_url = _encode_image_data_url(img_path)
                if data_url:
                    content.append({"type": "input_image", "image_url": data_url})
        content.append({"type": "input_text", "text": user_prompt})

        last_error = ""
        for attempt in range(1, self.max_retries + 1):
            try:
                input_messages: Any = [{"role": "user", "content": content}]
                response = self.client.responses.create(
                    model=self.model,
                    instructions=SYSTEM_PROMPT,
                    input=input_messages,
                    reasoning={"effort": "low"},
                    text={"verbosity": "low"},
                    max_output_tokens=500,
                )
                text = response.output_text or ""
                parsed = _parse_json_response(text)
                parsed["sample_id"] = sample["sample_id"]
                return parsed
            except Exception as exc:  # pragma: no cover
                last_error = str(exc)
                time.sleep(0.8 * attempt)

        return {
            "sample_id": sample["sample_id"],
            "score": 5,
            "semantic_similarity": 2,
            "detail_accuracy": 1,
            "hallucination_penalty": 1,
            "panorama_boundary_penalty": 0,
            "reason": f"api_error: {last_error[:120]}",
            "api_error": True,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-as-a-judge for CORA eval outputs")
    parser.add_argument("--input", required=True, help="Path to eval JSON or CSV")
    parser.add_argument("--output", default=None, help="Output CSV path")
    parser.add_argument("--model", default="gpt-5-mini", help="OpenAI model name")
    parser.add_argument("--api-key", default=None, help="OpenAI API key (optional if env set)")
    parser.add_argument("--max-samples", type=int, default=None, help="Evaluate first N samples")
    parser.add_argument("--base-path", default=None, help="Base path to resolve relative image paths")
    parser.add_argument("--no-image", action="store_true", help="Disable image input to judge")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is required")

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    output_path = Path(args.output) if args.output else input_path.with_name(f"{input_path.stem}_judge_scores.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    base_path = Path(args.base_path) if args.base_path else input_path.parent
    records = _normalize_records(input_path)
    if args.max_samples:
        records = records[: args.max_samples]
    if not records:
        raise SystemExit("No valid records with reference found")

    judge = LLMJudge(
        model=args.model,
        api_key=api_key,
        include_image=not args.no_image,
        base_path=base_path,
    )

    scores: List[Dict[str, Any]] = []
    for sample in tqdm(records, desc="LLM judge"):
        scores.append(judge.evaluate(sample))

    merged = pd.DataFrame(records).merge(pd.DataFrame(scores), on="sample_id", how="left")
    merged.to_csv(output_path, index=False, encoding="utf-8-sig")

    mean_score = float(merged["score"].mean())
    LOGGER.info("Saved judge scores to %s", output_path)
    LOGGER.info("Mean score: %.3f", mean_score)

    stats_path = output_path.with_suffix(".stats.json")
    stats = {
        "samples": int(len(merged)),
        "model": args.model,
        "mean_score": mean_score,
        "std_score": float(merged["score"].std(ddof=0)),
    }
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("Saved stats to %s", stats_path)


if __name__ == "__main__":
    main()
