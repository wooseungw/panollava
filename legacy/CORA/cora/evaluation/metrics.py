"""Evaluation Metrics for CORA."""

import logging
import json
import os
import tempfile
from typing import List, Dict, Any, Union
from pathlib import Path

# Try importing pycocoevalcap
try:
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap
    _HAS_COCO = True
except ImportError:
    _HAS_COCO = False

logger = logging.getLogger(__name__)

class CORAEvaluator:
    """
    Wrapper for standard VLM evaluation metrics (COCO Captioning Metrics).
    Computes BLEU-1..4, METEOR, ROUGE_L, CIDEr, SPICE.
    """
    def __init__(self):
        if not _HAS_COCO:
            logger.warning("pycocoevalcap or pycocotools not installed. Metrics will be unavailable.")

    def compute_metrics(
        self, 
        predictions: List[Dict[str, Any]], 
        references: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Compute metrics given predictions and references.
        
        Args:
            predictions: List of dicts [{"image_id": str, "caption": str}, ...]
            references: List of dicts [{"image_id": str, "caption": str}, ...]
            
        Returns:
            Dict of metric scores.
        """
        if not _HAS_COCO:
            return {}

        # 1. Format for COCO
        # COCO expects specific JSON structure
        # Images: [{"id": id}]
        # Annotations: [{"image_id": id, "id": ann_id, "caption": str}]
        
        # Create temp files
        with tempfile.TemporaryDirectory() as tmp_dir:
            res_file = os.path.join(tmp_dir, 'res.json')
            ann_file = os.path.join(tmp_dir, 'ann.json')
            
            # Save predictions
            with open(res_file, 'w') as f:
                json.dump(predictions, f)
                
            # Save references (Annotations)
            # Need to reshape references to COCO format
            # references arg is simple list of dicts, but COCO needs image/annotation structure
            
            images = []
            annotations = []
            img_ids = set()
            
            for idx, ref in enumerate(references):
                img_id = ref.get("image_id", idx)
                if img_id not in img_ids:
                    images.append({"id": img_id})
                    img_ids.add(img_id)
                
                annotations.append({
                    "image_id": img_id,
                    "id": idx,
                    "caption": ref["caption"]
                })
                
            coco_dict = {
                "images": images,
                "annotations": annotations,
                "type": "captions",
                "info": {},
                "licenses": []
            }
            
            with open(ann_file, 'w') as f:
                json.dump(coco_dict, f)

            # 2. Run Evaluation
            try:
                coco = COCO(ann_file)
                coco_res = coco.loadRes(res_file)
                
                coco_eval = COCOEvalCap(coco, coco_res)
                coco_eval.params['image_id'] = coco_res.getImgIds()
                coco_eval.evaluate()
                
                # 3. Collect Results
                return coco_eval.eval
                
            except Exception as e:
                logger.error(f"Error during COCO evaluation: {e}")
                return {}

    @staticmethod
    def simple_accuracy(predictions: List[str], references: List[str]) -> float:
        """Simple exact match accuracy."""
        correct = sum([1 for p, r in zip(predictions, references) if p.strip() == r.strip()])
        return correct / len(predictions) if predictions else 0.0
