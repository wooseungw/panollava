#!/usr/bin/env python3
"""
데이터셋 라벨링 및 마스킹 디버깅 스크립트
"""

import pandas as pd
from PIL import Image
from transformers import AutoTokenizer
from panovlm.processors.pano_llava_processor import PanoLLaVAProcessor
from panovlm.processors.builder import ConversationPromptBuilder
from panovlm.processors.image import PanoramaImageProcessor
from panovlm.processors.text import TextTokenizer
import torch

def debug_single_sample():
    # 설정
    csv_path = "data/quic360/valid.csv"
    tokenizer_name = "Qwen/Qwen2.5-0.5B"
    system_msg = "You are an expert assistant specialized in analyzing panoramic images. Please provide detailed, accurate, and helpful responses about what you observe in the panoramic view."
    
    # 데이터 로드
    df = pd.read_csv(csv_path)
    row = df.iloc[0]
    
    print("=" * 80)
    print("데이터셋 디버깅")
    print("=" * 80)
    print(f"Image path: {row.url}")
    print(f"Query: {row.query}")
    print(f"Annotation: {row.annotation}")
    print()
    
    # 프로세서 초기화
    img_proc = PanoramaImageProcessor(image_size=(224, 224), crop_strategy="e2p")
    txt_tok = TextTokenizer(tokenizer_name, max_len=128)
    processor = PanoLLaVAProcessor(img_proc, txt_tok, max_length=128)
    tokenizer = txt_tok.tok
    
    # 이미지 로드
    pil = Image.open(row.url).convert("RGB")
    
    # 대화 템플릿 구성
    builder = ConversationPromptBuilder(tokenizer, system_msg=system_msg)
    builder.push("user", str(row.query))
    builder.push("assistant", str(row.annotation))
    
    # 포맷된 텍스트 확인
    formatted_text = builder.formatted()
    print("포맷된 전체 텍스트:")
    print("-" * 40)
    print(formatted_text)
    print("-" * 40)
    print()
    
    # 토큰화
    batch = processor(pil, builder)
    input_ids = batch["input_ids"].flatten()
    
    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Total tokens: {len(input_ids)}")
    print()
    
    # 토큰 디코딩으로 확인
    decoded_text = tokenizer.decode(input_ids.tolist(), skip_special_tokens=False)
    print("디코딩된 텍스트:")
    print("-" * 40)
    print(decoded_text)
    print("-" * 40)
    print()
    
    # Assistant 패턴 찾기
    assistant_pattern = "<|im_start|>assistant"
    assistant_pos = formatted_text.find(assistant_pattern)
    
    if assistant_pos != -1:
        print(f"Assistant 패턴 위치: {assistant_pos}")
        
        # Assistant 이후 텍스트
        after_assistant = formatted_text[assistant_pos + len(assistant_pattern):].strip()
        if after_assistant.startswith('\n'):
            after_assistant = after_assistant[1:]
        
        # 응답 끝 찾기
        end_pos = after_assistant.find("<|im_end|>")
        if end_pos != -1:
            response_text = after_assistant[:end_pos].strip()
        else:
            response_text = after_assistant.strip()
        
        print(f"추출된 응답: '{response_text}'")
        print()
        
        # 토큰 분석
        prefix_text = formatted_text[:assistant_pos + len(assistant_pattern)].strip() + "\n"
        
        print("접두사 텍스트:")
        print(f"'{prefix_text}'")
        
        # 토큰화 테스트
        prefix_tokens = tokenizer.encode(prefix_text, add_special_tokens=False)
        response_tokens = tokenizer.encode(response_text, add_special_tokens=False)
        
        print(f"접두사 토큰 수: {len(prefix_tokens)}")
        print(f"응답 토큰 수: {len(response_tokens)}")
        print()
        
        # 라벨 마스킹 시뮬레이션
        IGNORE_INDEX = -100
        labels = input_ids.clone()
        labels.fill_(IGNORE_INDEX)
        
        prefix_len = len(prefix_tokens)
        response_len = len(response_tokens)
        total_len = len(input_ids)
        end_idx = min(prefix_len + response_len, total_len)
        
        if prefix_len < total_len and end_idx > prefix_len:
            labels[prefix_len:end_idx] = input_ids[prefix_len:end_idx]
        
        valid_labels = (labels != IGNORE_INDEX).sum().item()
        total_labels = len(labels)
        
        print(f"라벨 통계:")
        print(f"- 유효 라벨: {valid_labels}/{total_labels} ({valid_labels/total_labels*100:.1f}%)")
        print(f"- 접두사 길이: {prefix_len}")
        print(f"- 응답 길이: {response_len}")
        print(f"- 마스킹 범위: {prefix_len}:{end_idx}")
        print()
        
        # 실제 라벨이 설정된 토큰들 디코딩
        valid_token_ids = input_ids[labels != IGNORE_INDEX]
        if len(valid_token_ids) > 0:
            valid_text = tokenizer.decode(valid_token_ids.tolist(), skip_special_tokens=False)
            print(f"라벨이 설정된 토큰들의 텍스트:")
            print(f"'{valid_text}'")
        else:
            print("라벨이 설정된 토큰이 없습니다!")
        
    else:
        print("Assistant 패턴을 찾을 수 없습니다!")

if __name__ == "__main__":
    debug_single_sample()
