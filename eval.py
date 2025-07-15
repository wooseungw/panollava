# coding: utf-8
"""
Panorama-VLM Checkpoint Evaluation Script
- 저장된 체크포인트(.ckpt)로부터 모델 가중치만 불러와서 평가(evaluation)만 수행
- 사용 예시:
    python eval.py --ckpt runs/vlm_vision/checkpoints/epoch=02-val_loss=0.123.ckpt --csv-val data/quic360/downtest.csv --stage vision
"""
import argparse, torch, lightning as pl
from pathlib import Path
from panovlm.model import PanoramaVLM
from train import VLMModule, VLMDataModule

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True, help='평가할 체크포인트 경로(.ckpt)')
    p.add_argument('--csv-val', required=True, help='평가용 CSV 파일')
    p.add_argument('--vision-name', default='google/siglip-base-patch16-224')
    p.add_argument('--lm-name', default='Qwen/Qwen3-0.6B')
    p.add_argument('--resampler', default='mlp')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--max-txt-len', type=int, default=256)
    p.add_argument('--max-new-tokens', type=int, default=64)
    args = p.parse_args()

    # 데이터 준비
    dm = VLMDataModule(
        csv_train=args.csv_val,  # dummy
        csv_val=args.csv_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        tokenizer_name=args.lm_name,
        max_txt_len=args.max_txt_len
    )
    dm.setup()

    # 모델 가중치만 불러오기 (stage 인자 없이)
    model = VLMModule.load_from_checkpoint(
        args.ckpt,
        vision_name=args.vision_name,
        lm_name=args.lm_name,
        resampler=args.resampler,
        stage="finetune",  # dummy, generate만 쓸 것임
        lr=1e-5,
        map_location='cpu'
    )
    model.eval()

    val_loader = dm.val_dataloader()
    all_preds, all_gts = [], []
    with torch.no_grad():
        for batch in val_loader:
            pixel = batch["pixel_values"].to(model.device)
            gt_text = batch["text"] if "text" in batch else None
            out = model.model(stage="generate", pixel_values=pixel, max_new_tokens=args.max_new_tokens, temperature=0.7)
            preds = out["text"]
            all_preds.extend(preds)
            if gt_text is not None:
                all_gts.extend(gt_text)

    # 결과 출력 및 저장
    print(f"[Eval] {len(all_preds)} samples")
    for i in range(min(5, len(all_preds))):
        print(f"[Sample {i}]\n  GT: {all_gts[i] if i < len(all_gts) else ''}\n  PR: {all_preds[i]}")

    # 예측 결과를 jsonl로 저장
    import json
    output_path = "eval_outputs.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for gt, pr in zip(all_gts, all_preds):
            f.write(json.dumps({"gt": gt, "pred": pr}, ensure_ascii=False) + "\n")
    print(f"[Eval] Saved outputs to {output_path}")

    # ---- 텍스트 메트릭 ----
    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        from nltk.translate.meteor_score import meteor_score
        from rouge_score import rouge_scorer
    except ImportError:
        print("[WARN] nltk/rouge_score 패키지가 필요합니다. pip install nltk rouge-score")
        return

    # BLEU-4
    bleu4 = corpus_bleu([[gt] for gt in all_gts], all_preds, smoothing_function=SmoothingFunction().method1)
    # METEOR
    meteor = sum(meteor_score([gt], pr) for gt, pr in zip(all_gts, all_preds)) / len(all_preds)
    # ROUGE-L
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_l = sum(scorer.score(gt, pr)['rougeL'].fmeasure for gt, pr in zip(all_gts, all_preds)) / len(all_preds)

    print(f"[Metrics] BLEU-4: {bleu4:.4f}  METEOR: {meteor:.4f}  ROUGE-L: {rouge_l:.4f}")

    # ---- CIDEr, SPICE, CLIP score (optional) ----
    try:
        from pycocoevalcap.cider.cider import Cider
        cider_scorer = Cider()
        gts = {str(i): [gt] for i, gt in enumerate(all_gts)}
        res = {str(i): [pr] for i, pr in enumerate(all_preds)}
        cider, _ = cider_scorer.compute_score(gts, res)
        print(f"[Metrics] CIDEr: {cider:.4f}")
    except ImportError:
        print("[INFO] pycocoevalcap 설치 시 CIDEr, SPICE 등 추가 메트릭 사용 가능")

    try:
        import open_clip
        import torchvision.transforms as T
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
        clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')
        clip_model.eval()
        clip_model = clip_model.to(model.device)
        # 이미지-텍스트 CLIP score 계산 (이미지 경로가 필요할 경우 추가 구현 필요)
        # 예시: CLIP score = 평균 max(similarity)
        # (여기서는 dummy 예시, 실제 이미지를 batch["image_path"] 등에서 불러와야 함)
        # clip_scores = ...
        # print(f"[Metrics] CLIP score: {sum(clip_scores)/len(clip_scores):.4f}")
    except ImportError:
        print("[INFO] open_clip 설치 시 CLIP score 사용 가능")

if __name__ == '__main__':
    main()

# python eval.py --ckpt runs/vlm_vision/checkpoints/epoch=02-val_loss=0.123.ckpt --csv-val data/quic360/downtest.csv --stage vision