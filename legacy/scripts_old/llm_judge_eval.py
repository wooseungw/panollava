#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM-as-a-Judge Evaluation Script for Panorama VLM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GPT-5.2 Responses API를 사용하여 파노라마 이미지 VLM 예측 결과를 평가합니다.

평가 기준:
1. 파노라마 분절 오류: 360° 경계에서 객체 분리 오인식
2. 할루시네이션: 이미지에 없는 정보 생성
3. 의미적 유사성: Reference와의 핵심 내용 일치도 (가중치 높음)
4. 세부 정확성: 색상, 위치, 수량 등 구체적 정보 (가중치 높음)

사용법:
    python scripts/llm_judge_eval.py \
        --csv-input results/eval_results/.../predictions.csv \
        --output results/judge_results/scores.csv \
        --model gpt-5-mini
"""

import os
import sys
import json
import time
import base64
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

# dotenv는 선택적 의존성
try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv():
        pass

# OpenAI 패키지 임포트
try:
    from openai import OpenAI
except ImportError:
    print("Error: openai 패키지가 설치되어 있지 않습니다.")
    print("설치: pip install openai")
    sys.exit(1)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Judge 프롬프트 정의
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for panoramic image descriptions.
You evaluate predictions from Vision-Language Models (VLMs) that analyze 360° panoramic images.

Consider the unique characteristics of panoramic images:
- Objects may appear distorted or stretched at the edges
- The image wraps around 360°, so objects at the left/right edges may be the same object
- Spatial relationships may differ from standard perspective images

Evaluate fairly and consistently, providing scores from 1-10.
Always respond with valid JSON only."""

JUDGE_USER_PROMPT = """## Evaluation Task
Compare the model's Prediction against the Reference answer for the given panoramic image and query.
Score from 1 to 10 based on the criteria below.

## Input Information
- **Query**: {query}
- **Prediction**: {prediction}
- **Reference**: {reference}

## Evaluation Criteria (Total: 10 points)

### 1. Panorama Segmentation Errors (Deduct 0-1 point)
- Check if the model incorrectly interpreted objects split at 360° image boundaries as separate objects

### 2. Hallucination Detection (Deduct 0-2 points)
- Check if the prediction mentions objects, properties, or situations NOT present in the image

### 3. Semantic Similarity (Award 0-4 points) ⭐ HIGH WEIGHT
- How well does the prediction capture the core meaning and content of the reference?
- 4: Excellent, 3: Good, 2: Partial, 1: Minimal, 0: None

### 4. Detail Accuracy (Award 0-3 points) ⭐ HIGH WEIGHT
- Accuracy of specific details: colors, positions, quantities, sizes
- 3: All accurate, 2: Most accurate, 1: Some accurate, 0: Major errors

## Response Format (JSON only)
{{"score": <1-10>, "segmentation_issue": <true/false>, "hallucination_detected": <true/false>, "semantic_similarity_score": <0-4>, "detail_accuracy_score": <0-3>, "reasoning": "<Brief explanation in Korean>"}}"""

# 배치 평가용 프롬프트 (여러 샘플을 한 번에 평가)
JUDGE_BATCH_PROMPT = """## Batch Evaluation Task
Evaluate ALL the following {count} predictions for the same panoramic image.
For each sample, compare the Prediction against the Reference.

{samples}

## Evaluation Criteria (Total: 10 points per sample)
1. Panorama Segmentation Errors (Deduct 0-1 point)
2. Hallucination Detection (Deduct 0-2 points) 
3. Semantic Similarity (Award 0-4 points) ⭐ HIGH WEIGHT
4. Detail Accuracy (Award 0-3 points) ⭐ HIGH WEIGHT

## Response Format (JSON array, one object per sample)
[
  {{"sample_id": <id>, "score": <1-10>, "segmentation_issue": <true/false>, "hallucination_detected": <true/false>, "semantic_similarity_score": <0-4>, "detail_accuracy_score": <0-3>, "reasoning": "<brief>"}},
  ...
]"""


def encode_image_to_base64(image_path: str) -> Optional[str]:
    """이미지 파일을 base64로 인코딩"""
    try:
        path = Path(image_path)
        if not path.exists():
            logger.warning(f"이미지 파일 없음: {image_path}")
            return None
        
        with open(path, "rb") as f:
            encoded = base64.standard_b64encode(f.read()).decode("utf-8")
        
        ext = path.suffix.lower()
        mime_types = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', '.webp': 'image/webp'}
        mime_type = mime_types.get(ext, 'image/jpeg')
        
        return f"data:{mime_type};base64,{encoded}"
    except Exception as e:
        logger.error(f"이미지 인코딩 실패 {image_path}: {e}")
        return None


def parse_judge_response(response_text: str) -> Dict[str, Any]:
    """Judge 응답 JSON 파싱 (잘린 응답도 처리)"""
    import re
    
    text = response_text.strip() if response_text else ""
    
    # JSON 코드 블록 추출
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        parts = text.split("```")
        if len(parts) > 1:
            text = parts[1].strip()
    
    # JSON 객체 추출
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        text = text[start:end]
    
    # 먼저 정상 JSON 파싱 시도
    try:
        result = json.loads(text)
        score = result.get("score", 5)
        if not isinstance(score, (int, float)) or score < 1 or score > 10:
            score = max(1, min(10, int(score) if score else 5))
        result["score"] = int(score)
        return result
    except json.JSONDecodeError:
        pass
    
    # 잘린 JSON에서 score 추출 시도
    result = {
        "score": 5,
        "segmentation_issue": False,
        "hallucination_detected": False,
        "semantic_similarity_score": 2,
        "detail_accuracy_score": 1,
        "reasoning": "Partial parse",
        "parse_error": True
    }
    
    # 정규식으로 값 추출
    score_match = re.search(r'"score"\s*:\s*(\d+)', text)
    if score_match:
        result["score"] = max(1, min(10, int(score_match.group(1))))
        result["parse_error"] = False
    
    seg_match = re.search(r'"segmentation_issue"\s*:\s*(true|false)', text, re.I)
    if seg_match:
        result["segmentation_issue"] = seg_match.group(1).lower() == "true"
    
    hal_match = re.search(r'"hallucination_detected"\s*:\s*(true|false)', text, re.I)
    if hal_match:
        result["hallucination_detected"] = hal_match.group(1).lower() == "true"
    
    sem_match = re.search(r'"semantic_similarity_score"\s*:\s*(\d+)', text)
    if sem_match:
        result["semantic_similarity_score"] = min(4, int(sem_match.group(1)))
    
    det_match = re.search(r'"detail_accuracy_score"\s*:\s*(\d+)', text)
    if det_match:
        result["detail_accuracy_score"] = min(3, int(det_match.group(1)))
    
    reason_match = re.search(r'"reasoning"\s*:\s*"([^"]*)', text)
    if reason_match:
        result["reasoning"] = reason_match.group(1)
    
    if result.get("parse_error") and result["score"] == 5:
        logger.warning(f"JSON 파싱 실패, 기본값 사용. 원본: {text[:100]}...")
    
    return result


class LLMJudge:
    """OpenAI GPT-5.2 Responses API 기반 LLM Judge 클래스"""
    
    # GPT-5.2 모델 (Responses API 사용)
    GPT5_MODELS = [
        "gpt-5.2",           # 복잡한 추론
        "gpt-5.2-pro",       # 더 깊은 사고
        "gpt-5-mini",        # 비용 효율적 (권장)
        "gpt-5-nano",        # 고처리량
    ]
    
    # GPT-4 모델 (Chat Completions API 사용)
    GPT4_MODELS = [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-4.1",
        "gpt-4.1-mini",
    ]
    
    SUPPORTED_MODELS = GPT5_MODELS + GPT4_MODELS
    
    def __init__(
        self,
        model: str = "gpt-5-mini",
        api_key: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        include_image: bool = True,
        base_path: Optional[str] = None,
        reasoning_effort: str = "low",  # none, low, medium, high, xhigh
        verbosity: str = "low",  # low, medium, high
    ):
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.include_image = include_image
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self.reasoning_effort = reasoning_effort
        self.verbosity = verbosity
        
        # GPT-5 모델인지 확인
        self.use_responses_api = model in self.GPT5_MODELS
        
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API Key가 필요합니다.\n"
                "환경변수 OPENAI_API_KEY를 설정하거나 --api-key 옵션을 사용하세요."
            )
        
        self.client = OpenAI(api_key=api_key)
        api_type = "Responses API" if self.use_responses_api else "Chat Completions API"
        logger.info(f"LLM Judge 초기화 - 모델: {self.model}, API: {api_type}, 이미지: {self.include_image}")
    
    def _resolve_image_path(self, image_path: str) -> Path:
        """이미지 경로 해결"""
        path = Path(image_path)
        if path.is_absolute() and path.exists():
            return path
        resolved = self.base_path / image_path
        return resolved if resolved.exists() else path
    
    def evaluate_sample(
        self,
        query: str,
        prediction: str,
        reference: str,
        image_path: Optional[str] = None,
        sample_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """단일 샘플 평가 - 모델에 따라 적절한 API 사용"""
        
        if self.use_responses_api:
            return self._evaluate_with_responses_api(
                query, prediction, reference, image_path, sample_id
            )
        else:
            return self._evaluate_with_chat_completions(
                query, prediction, reference, image_path, sample_id
            )
    
    def _evaluate_with_responses_api(
        self,
        query: str,
        prediction: str,
        reference: str,
        image_path: Optional[str] = None,
        sample_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """GPT-5.2 Responses API를 사용한 평가"""
        
        user_prompt = JUDGE_USER_PROMPT.format(
            query=query,
            prediction=prediction,
            reference=reference
        )
        
        # Responses API input 구성 (문서 기반)
        content = []
        
        # 이미지 포함
        if self.include_image and image_path:
            resolved_path = self._resolve_image_path(image_path)
            image_data = encode_image_to_base64(str(resolved_path))
            if image_data:
                content.append({
                    "type": "input_image",
                    "image_url": image_data,
                })
        
        # 텍스트 프롬프트
        content.append({
            "type": "input_text",
            "text": user_prompt
        })
        
        # Input 구성
        input_messages = [{
            "role": "user",
            "content": content
        }]
        
        for attempt in range(self.max_retries):
            try:
                # GPT-5.2 Responses API 호출
                response = self.client.responses.create(
                    model=self.model,
                    instructions=JUDGE_SYSTEM_PROMPT,
                    input=input_messages,
                    reasoning={"effort": self.reasoning_effort},
                    text={"verbosity": self.verbosity},
                    max_output_tokens=1000,  # 응답 잘림 방지
                )
                
                # 응답 텍스트 추출 (GPT-5.2 Responses API)
                response_text = ""
                
                # 1. output_text 속성 우선 사용 (가장 간단)
                if hasattr(response, 'output_text') and response.output_text:
                    response_text = response.output_text
                # 2. output 리스트에서 message 타입 찾아서 추출
                elif hasattr(response, 'output') and response.output:
                    for item in response.output:
                        # message 타입에서 텍스트 추출
                        if hasattr(item, 'type') and item.type == 'message':
                            if hasattr(item, 'content') and item.content:
                                for content_item in item.content:
                                    # ResponseOutputText의 text 속성
                                    if hasattr(content_item, 'text') and content_item.text:
                                        response_text += content_item.text
                
                result = parse_judge_response(response_text)
                result["sample_id"] = sample_id
                result["model_used"] = self.model
                result["api_used"] = "responses"
                
                return result
                
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"Responses API 호출 실패 (시도 {attempt + 1}/{self.max_retries}): {error_msg}")
                
                # 400 에러는 API 형식 문제 - Chat Completions로 폴백
                if "400" in error_msg:
                    logger.info("Responses API 형식 오류, Chat Completions로 폴백")
                    return self._evaluate_with_chat_completions(
                        query, prediction, reference, image_path, sample_id
                    )
                
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    return {
                        "score": 5,
                        "segmentation_issue": False,
                        "hallucination_detected": False,
                        "reasoning": f"API error: {error_msg[:100]}",
                        "api_error": True,
                        "sample_id": sample_id,
                        "model_used": self.model
                    }
    
    def _evaluate_with_chat_completions(
        self,
        query: str,
        prediction: str,
        reference: str,
        image_path: Optional[str] = None,
        sample_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Chat Completions API를 사용한 평가"""
        
        user_prompt = JUDGE_USER_PROMPT.format(
            query=query,
            prediction=prediction,
            reference=reference
        )
        
        messages = [{"role": "system", "content": JUDGE_SYSTEM_PROMPT}]
        
        # 이미지 포함
        if self.include_image and image_path:
            resolved_path = self._resolve_image_path(image_path)
            image_data = encode_image_to_base64(str(resolved_path))
            if image_data:
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_data, "detail": "low"}},
                        {"type": "text", "text": user_prompt}
                    ]
                })
            else:
                messages.append({"role": "user", "content": user_prompt})
        else:
            messages.append({"role": "user", "content": user_prompt})
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=500,
                    temperature=0.1,
                )
                
                response_text = response.choices[0].message.content
                result = parse_judge_response(response_text)
                result["sample_id"] = sample_id
                result["model_used"] = self.model
                result["api_used"] = "chat_completions"
                
                return result
            except Exception as e:
                logger.warning(f"Chat Completions 호출 실패 (시도 {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    return {
                        "score": 5,
                        "segmentation_issue": False,
                        "hallucination_detected": False,
                        "reasoning": f"API error: {str(e)[:100]}",
                        "api_error": True,
                        "sample_id": sample_id,
                        "model_used": self.model
                    }
    
    def evaluate_image_batch(
        self,
        image_path: str,
        samples: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """같은 이미지의 여러 샘플을 한 번에 평가"""
        
        if not samples:
            return []
        
        # 샘플 텍스트 구성
        samples_text = ""
        for i, sample in enumerate(samples):
            samples_text += f"""
### Sample {i+1} (ID: {sample['sample_id']})
- **Query**: {sample['query']}
- **Prediction**: {sample['prediction']}
- **Reference**: {sample['reference']}
"""
        
        batch_prompt = JUDGE_BATCH_PROMPT.format(
            count=len(samples),
            samples=samples_text
        )
        
        # Responses API input 구성
        content = []
        
        # 이미지 포함
        if self.include_image and image_path:
            resolved_path = self._resolve_image_path(image_path)
            image_data = encode_image_to_base64(str(resolved_path))
            if image_data:
                content.append({
                    "type": "input_image",
                    "image_url": image_data,
                })
        
        content.append({
            "type": "input_text",
            "text": batch_prompt
        })
        
        input_messages = [{"role": "user", "content": content}]
        
        # max_output_tokens 계산 (샘플당 ~200토큰)
        max_tokens = min(4000, len(samples) * 250 + 200)
        
        for attempt in range(self.max_retries):
            try:
                if self.use_responses_api:
                    response = self.client.responses.create(
                        model=self.model,
                        instructions=JUDGE_SYSTEM_PROMPT,
                        input=input_messages,
                        reasoning={"effort": self.reasoning_effort},
                        text={"verbosity": self.verbosity},
                        max_output_tokens=max_tokens,
                    )
                    response_text = response.output_text if hasattr(response, 'output_text') else ""
                else:
                    messages = [{"role": "system", "content": JUDGE_SYSTEM_PROMPT}]
                    # Chat Completions 형식
                    chat_content = []
                    if self.include_image and image_path:
                        resolved_path = self._resolve_image_path(image_path)
                        image_data = encode_image_to_base64(str(resolved_path))
                        if image_data:
                            chat_content.append({"type": "image_url", "image_url": {"url": image_data, "detail": "low"}})
                    chat_content.append({"type": "text", "text": batch_prompt})
                    messages.append({"role": "user", "content": chat_content})
                    
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=0.1,
                    )
                    response_text = response.choices[0].message.content
                
                # 배치 응답 파싱
                results = self._parse_batch_response(response_text, samples)
                return results
                
            except Exception as e:
                logger.warning(f"배치 평가 실패 (시도 {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
        
        # 실패 시 기본값 반환
        return [
            {
                "score": 5, "segmentation_issue": False, "hallucination_detected": False,
                "semantic_similarity_score": 2, "detail_accuracy_score": 1,
                "reasoning": "Batch API error", "api_error": True,
                "sample_id": s["sample_id"], "model_used": self.model
            }
            for s in samples
        ]
    
    def _parse_batch_response(
        self,
        response_text: str,
        samples: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """배치 응답 JSON 배열 파싱"""
        import re
        
        text = response_text.strip() if response_text else ""
        
        # JSON 배열 추출
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            parts = text.split("```")
            if len(parts) > 1:
                text = parts[1].strip()
        
        # 배열 시작/끝 찾기
        start = text.find("[")
        end = text.rfind("]") + 1
        if start != -1 and end > start:
            text = text[start:end]
        
        results = []
        sample_id_map = {s["sample_id"]: s for s in samples}
        
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                for item in parsed:
                    sid = item.get("sample_id")
                    result = {
                        "sample_id": sid,
                        "score": max(1, min(10, int(item.get("score", 5)))),
                        "segmentation_issue": item.get("segmentation_issue", False),
                        "hallucination_detected": item.get("hallucination_detected", False),
                        "semantic_similarity_score": min(4, int(item.get("semantic_similarity_score", 2))),
                        "detail_accuracy_score": min(3, int(item.get("detail_accuracy_score", 1))),
                        "reasoning": item.get("reasoning", ""),
                        "model_used": self.model,
                    }
                    results.append(result)
        except json.JSONDecodeError:
            # 정규식으로 개별 객체 추출
            pattern = r'\{[^{}]*"sample_id"\s*:\s*(\d+)[^{}]*\}'
            matches = re.findall(pattern, text)
            
            for i, sample in enumerate(samples):
                # 해당 샘플의 JSON 객체 찾기
                sample_pattern = rf'\{{[^{{}}]*"sample_id"\s*:\s*{sample["sample_id"]}[^{{}}]*\}}'
                match = re.search(sample_pattern, text)
                
                if match:
                    obj_text = match.group(0)
                    result = parse_judge_response(obj_text)
                    result["sample_id"] = sample["sample_id"]
                else:
                    result = {
                        "score": 5, "segmentation_issue": False,
                        "hallucination_detected": False, "parse_error": True,
                        "sample_id": sample["sample_id"], "model_used": self.model
                    }
                results.append(result)
        
        # 누락된 샘플 기본값 추가
        result_ids = {r["sample_id"] for r in results}
        for sample in samples:
            if sample["sample_id"] not in result_ids:
                results.append({
                    "score": 5, "segmentation_issue": False,
                    "hallucination_detected": False, "parse_error": True,
                    "sample_id": sample["sample_id"], "model_used": self.model
                })
        
        return results
    
    def evaluate_batch(
        self,
        df: pd.DataFrame,
        max_samples: Optional[int] = None,
        num_workers: int = 1,
        progress_bar: bool = True,
        batch_by_image: bool = True,  # 이미지별 배치 평가
    ) -> pd.DataFrame:
        """배치 평가 수행 (이미지별 그룹화 지원)"""
        
        if max_samples:
            df = df.head(max_samples)
        
        results = []
        total = len(df)
        
        if batch_by_image and "image_path" in df.columns:
            # 이미지별로 그룹화
            grouped = df.groupby("image_path")
            unique_images = len(grouped)
            logger.info(f"평가 시작: {total}개 샘플 ({unique_images}개 이미지)")
            
            pbar = tqdm(total=total, desc="Evaluating", disable=not progress_bar)
            
            for image_path, group_df in grouped:
                samples = []
                for idx, row in group_df.iterrows():
                    samples.append({
                        "sample_id": row.get("sample_id", idx),
                        "query": str(row.get("original_query", "")),
                        "prediction": str(row.get("prediction", "")),
                        "reference": str(row.get("reference", "")),
                    })
                
                batch_results = self.evaluate_image_batch(str(image_path), samples)
                results.extend(batch_results)
                pbar.update(len(samples))
            
            pbar.close()
        else:
            # 기존 방식: 샘플별 평가
            logger.info(f"평가 시작: {total}개 샘플")
            iterator = tqdm(df.iterrows(), total=total, desc="Evaluating", disable=not progress_bar)
            for idx, row in iterator:
                result = self.evaluate_sample(
                    query=str(row.get("original_query", "")),
                    prediction=str(row.get("prediction", "")),
                    reference=str(row.get("reference", "")),
                    image_path=str(row.get("image_path", "")),
                    sample_id=row.get("sample_id", idx),
                )
                results.append(result)
        
        results_df = pd.DataFrame(results)
        if "sample_id" in results_df.columns:
            results_df = results_df.sort_values("sample_id").reset_index(drop=True)
        
        return results_df


def compute_statistics(results_df: pd.DataFrame) -> Dict[str, Any]:
    """평가 결과 통계 계산"""
    scores = results_df["score"].dropna()
    
    stats = {
        "total_samples": len(results_df),
        "evaluated_samples": len(scores),
        "mean_score": float(scores.mean()),
        "std_score": float(scores.std()),
        "min_score": int(scores.min()),
        "max_score": int(scores.max()),
        "median_score": float(scores.median()),
    }
    
    score_counts = scores.value_counts().sort_index()
    stats["score_distribution"] = {int(k): int(v) for k, v in score_counts.items()}
    
    if "hallucination_detected" in results_df.columns:
        halluc = results_df["hallucination_detected"].sum()
        stats["hallucination_rate"] = float(halluc / len(results_df))
    
    if "segmentation_issue" in results_df.columns:
        seg_issue = results_df["segmentation_issue"].sum()
        stats["segmentation_issue_rate"] = float(seg_issue / len(results_df))
    
    if "semantic_similarity_score" in results_df.columns:
        stats["mean_semantic_similarity"] = float(results_df["semantic_similarity_score"].mean())
    
    if "detail_accuracy_score" in results_df.columns:
        stats["mean_detail_accuracy"] = float(results_df["detail_accuracy_score"].mean())
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="LLM-as-a-Judge Evaluation for Panorama VLM (GPT-5.2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # GPT-5-mini (권장, 비용 효율적)
  python scripts/llm_judge_eval.py --csv-input predictions.csv --model gpt-5-mini

  # GPT-5.2 (고성능)
  python scripts/llm_judge_eval.py --csv-input predictions.csv --model gpt-5.2 --reasoning medium
        """
    )
    
    parser.add_argument("--csv-input", "-i", type=str, required=True,
                       help="입력 CSV 파일 경로 (predictions.csv)")
    parser.add_argument("--output", "-o", type=str, default=None,
                       help="출력 CSV 파일 경로")
    parser.add_argument("--model", "-m", type=str, default="gpt-5-mini",
                       choices=LLMJudge.SUPPORTED_MODELS,
                       help="사용할 OpenAI 모델 (기본: gpt-5-mini)")
    parser.add_argument("--api-key", type=str, default=None,
                       help="OpenAI API Key")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="평가할 최대 샘플 수")
    parser.add_argument("--no-image", action="store_true",
                       help="이미지 없이 텍스트만으로 평가")
    parser.add_argument("--base-path", type=str, default=None,
                       help="이미지 경로의 기준 디렉토리")
    parser.add_argument("--num-workers", type=int, default=1,
                       help="병렬 처리 워커 수")
    parser.add_argument("--reasoning", type=str, default="low",
                       choices=["none", "low", "medium", "high", "xhigh"],
                       help="GPT-5.2 reasoning effort (기본: low)")
    parser.add_argument("--verbosity", type=str, default="low",
                       choices=["low", "medium", "high"],
                       help="출력 상세도 (기본: low)")
    parser.add_argument("--save-stats", action="store_true",
                       help="통계를 별도 JSON 파일로 저장")
    parser.add_argument("--batch-by-image", action="store_true",
                       help="같은 이미지의 샘플들을 배치로 묶어 평가 (API 호출 감소)")
    
    args = parser.parse_args()
    load_dotenv()
    
    csv_path = Path(args.csv_input)
    if not csv_path.exists():
        logger.error(f"입력 파일을 찾을 수 없습니다: {csv_path}")
        sys.exit(1)
    
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = csv_path.parent / f"judge_scores_{timestamp}.csv"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    base_path = args.base_path
    if not base_path:
        current = csv_path.parent
        while current != current.parent:
            if (current / "data").exists():
                base_path = str(current)
                break
            current = current.parent
        if not base_path:
            base_path = str(Path.cwd())
    
    logger.info(f"Base path: {base_path}")
    
    logger.info(f"CSV 로드: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"샘플 수: {len(df)}")
    
    required_columns = ["prediction", "reference"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        logger.error(f"필수 컬럼 누락: {missing}")
        sys.exit(1)
    
    try:
        judge = LLMJudge(
            model=args.model,
            api_key=args.api_key,
            include_image=not args.no_image,
            base_path=base_path,
            reasoning_effort=args.reasoning,
            verbosity=args.verbosity,
        )
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)
    
    start_time = time.time()
    results_df = judge.evaluate_batch(
        df, 
        max_samples=args.max_samples, 
        num_workers=args.num_workers,
        batch_by_image=args.batch_by_image
    )
    elapsed = time.time() - start_time
    
    results_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info(f"결과 저장: {output_path}")
    
    stats = compute_statistics(results_df)
    stats["elapsed_seconds"] = elapsed
    stats["model"] = args.model
    stats["reasoning_effort"] = args.reasoning
    
    print("\n" + "="*60)
    print("📊 LLM Judge 평가 결과 (GPT-5.2)")
    print("="*60)
    print(f"  총 샘플 수: {stats['total_samples']}")
    print(f"  소요 시간: {elapsed:.1f}초")
    print("-"*60)
    print(f"  ⭐ 평균 점수: {stats['mean_score']:.2f} / 10")
    print(f"  📈 표준편차: {stats['std_score']:.2f}")
    print(f"  🔽 최소: {stats['min_score']} | 🔼 최대: {stats['max_score']} | 📍 중앙: {stats['median_score']:.1f}")
    print("-"*60)
    
    if "mean_semantic_similarity" in stats:
        print(f"  📝 의미적 유사성: {stats['mean_semantic_similarity']:.2f} / 4")
    if "mean_detail_accuracy" in stats:
        print(f"  🔍 세부 정확성: {stats['mean_detail_accuracy']:.2f} / 3")
    if "hallucination_rate" in stats:
        print(f"  ⚠️ 할루시네이션: {stats['hallucination_rate']*100:.1f}%")
    
    print("="*60)
    
    print("\n📊 점수 분포:")
    max_count = max(stats["score_distribution"].values()) if stats["score_distribution"] else 1
    for score in range(1, 11):
        count = stats["score_distribution"].get(score, 0)
        bar = "█" * int(count / max_count * 20) if max_count else ""
        print(f"  {score:2d}점: {count:4d} {bar}")
    
    if args.save_stats:
        stats_path = output_path.with_suffix(".stats.json")
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        logger.info(f"통계 저장: {stats_path}")
    
    print(f"\n✅ 완료! 결과: {output_path}")


if __name__ == "__main__":
    main()
