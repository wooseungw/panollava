# coding: utf-8

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .utils import safe_save_pretrained
from .language_fusion import LanguageFusion
from ..losses import VicRegLoss
from ..losses.vicreg_overlap import compute_vicreg_overlap_loss
from ..losses.vicreg_projector import VICRegProjector
from .vision import VisionBackbone, PanoramaProjector, ResamplerModule
from .vision.utils import resolve_module_dtype

# LoRA 지원을 위한 PEFT import (선택적)
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: PEFT not available. LoRA fine-tuning will not be supported.")

# ---------------------------------------------------------------------------
# ‣ VICReg Loss (moved to panovlm/utils/losses.py)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# ‣ PanoramaVLM (VICReg + AR)
# ---------------------------------------------------------------------------
class PanoramaVLM(nn.Module):
    """
    3단계 학습:
    1) stage="vision": VICReg (인접 뷰 겹치는 패치 구간만)
       - NEW: VICReg projector로 차원 축소/정규화 후 손실 계산
    2) stage="resampler": Resampler + LM AR-loss (resampler와 proj만 학습 가능)
    3) stage="finetune": 전체 미세조정
    """
    def __init__(self, config=None, **kwargs):
        super().__init__()

        # 설정 ------------------------------------------------
        if config is not None:
            self.config = config
        else:
            # 간단한 dict 기반 설정 폴백
            class _Cfg:
                def __init__(self, **kw):
                    for k, v in kw.items():
                        setattr(self, k, v)
            self.config = _Cfg(**kwargs)

        # 비전 인코더 ------------------------------------------
        vision_name = getattr(self.config, 'vision_name', 'google/siglip-base-patch16-224')
        use_vicreg_norm = bool(getattr(self.config, 'use_vicreg_norm', False))
        self.vision_backbone = VisionBackbone(vision_name=vision_name, use_vicreg_norm=use_vicreg_norm)
        vision_hidden_size = self.vision_backbone.hidden_size

        # NEW: VICReg 전용 Projector ---------------------------
        self.use_vicreg_projector = getattr(self.config, 'use_vicreg_projector', True)
        self.vicreg_projector_dim = int(getattr(self.config, 'vicreg_projector_dim', 768))
        self.vicreg_projector_depth = int(getattr(self.config, 'vicreg_projector_depth', 2))
        self.vicreg_projector_hidden = int(getattr(self.config, 'vicreg_projector_hidden', self.vicreg_projector_dim))
        self.vicreg_projector_ln = bool(getattr(self.config, 'vicreg_projector_ln', True))

        if self.use_vicreg_projector:
            self.vicreg_projector = VICRegProjector(
                in_dim=vision_hidden_size,
                out_dim=self.vicreg_projector_dim,
                hidden_dim=self.vicreg_projector_hidden,
                depth=self.vicreg_projector_depth,
                use_ln=self.vicreg_projector_ln,
            )
            self._vicreg_feat_dim = self.vicreg_projector_dim
        else:
            self.vicreg_projector = nn.Identity()
            self._vicreg_feat_dim = vision_hidden_size

        # 리샘플러 ----------------------------------------------
        self.resampler_module = ResamplerModule(self.config, vision_hidden_size)
        latent_dimension = self.resampler_module.output_dim
        self.resampler = self.resampler_module.resampler  # Backward compatibility alias

        # 언어 모델 및 투영 ------------------------------------
        lm_name = getattr(self.config, 'language_model_name', 'Qwen/Qwen2.5-0.5B-Instruct')
        self.language_model = AutoModelForCausalLM.from_pretrained(
            lm_name,
            attn_implementation="sdpa",
        )
        # Reduce GPU footprint during training
        if hasattr(self.language_model, "config"):
            self.language_model.config.use_cache = False
        self._gradient_checkpointing = False

        # 프로젝터 ----------------------------------------------
        self.projector = PanoramaProjector(self.config, latent_dimension, self.language_model.config.hidden_size)
        self.vision_to_language_projection = self.projector.linear  # Backward compatibility alias

        # 텍스트 프로젝터 (2단계 학습용) -------------------------
        self.use_text_projection = getattr(self.config, 'use_text_projection', False)
        if self.use_text_projection:
            # 자동으로 LLM의 입력 차원으로 projection (항상 동일한 차원 유지)
            llm_hidden_size = self.language_model.config.hidden_size
            self.text_projection = nn.Linear(llm_hidden_size, llm_hidden_size)
            print(f"✓ Text projection enabled: {llm_hidden_size} -> {llm_hidden_size} (auto-aligned with LLM input)")
        else:
            self.text_projection = nn.Identity()

        # 토크나이저 -------------------------------------------
        self.tokenizer = AutoTokenizer.from_pretrained(lm_name)
        self._setup_tokenizer()
        self.vision_token_id = self.tokenizer.convert_tokens_to_ids("<|vision|>")
        if self.vision_token_id == self.tokenizer.unk_token_id:
            self.vision_token_id = self.tokenizer.bos_token_id

        # VICReg 손실 및 하이퍼 --------------------------------
        self.vicreg_loss = VicRegLoss(
            similarity_weight=float(getattr(self.config, 'vicreg_similarity_weight', 25.0)),
            variance_weight=float(getattr(self.config, 'vicreg_variance_weight', 25.0)),
            covariance_weight=float(getattr(self.config, 'vicreg_covariance_weight', 1.0)),
            gamma=float(getattr(self.config, 'vicreg_gamma', 1.0)),
            use_ddp_gather=bool(getattr(self.config, 'vicreg_use_ddp_gather', False)),
        )
        self.vicreg_loss_weight = float(getattr(self.config, 'vicreg_loss_weight', 1.0))
        self.overlap_ratio = float(getattr(self.config, 'overlap_ratio', 0.5))

        self.skip_first_view_in_vision_stage = bool(getattr(self.config, 'skip_first_view_in_vision_stage', False))
        expected_views_cfg = getattr(self.config, 'vision_stage_expected_views', None)
        if expected_views_cfg is None:
            expected_views_cfg = getattr(self.config, 'resampler_num_views', None)
        try:
            self.vision_stage_expected_views = int(expected_views_cfg) if expected_views_cfg is not None else None
        except (TypeError, ValueError):
            self.vision_stage_expected_views = None
        if self.vision_stage_expected_views is not None and self.vision_stage_expected_views <= 0:
            self.vision_stage_expected_views = None

        # 텍스트 설정 ------------------------------------------
        self.max_text_length = int(getattr(self.config, 'max_text_length', 512))
        self.ignore_index = -100
        system_prompt = getattr(self.config, 'system_prompt', None)
        default_generation_prompt = getattr(self.config, 'default_generation_prompt', None)
        fallback_insert_position = getattr(self.config, 'vision_token_fallback_position', 'prefix')
        if isinstance(fallback_insert_position, str):
            fallback_insert_position = fallback_insert_position.lower().strip()
        else:
            fallback_insert_position = 'prefix'

        self.language_fusion = LanguageFusion(
            language_model=self.language_model,
            tokenizer=self.tokenizer,
            vision_token_id=self.vision_token_id,
            ignore_index=self.ignore_index,
            max_text_length=self.max_text_length,
            system_prompt=system_prompt,
            default_generation_prompt=default_generation_prompt,
            fallback_insert_position=fallback_insert_position,
        )
        self.language_fusion.update_vision_token_id(self.vision_token_id)
        self.language_fusion.update_max_text_length(self.max_text_length)

        # 디버그 플래그
        self._warned_single_view = False
        self._debug_loss_verification = False
        self._cached_module_dtypes: Dict[str, torch.dtype] = {}
        self._vision_stage_trim_warned = False

    def _setup_tokenizer(self):
        tokens_added = False
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                print(f"[Tokenizer Setup] Set pad_token = eos_token: '{self.tokenizer.eos_token}'")
            else:
                new_tokens = {'eos_token': '<|endoftext|>', 'pad_token': '<|endoftext|>'}
                self.tokenizer.add_special_tokens(new_tokens)
                tokens_added = True
                print(f"[Tokenizer Setup] Added eos_token and pad_token: '<|endoftext|>'")
        special_tokens_to_add = []
        vision_token = '<|vision|>'
        if not any(vision_token in str(token) for token in self.tokenizer.additional_special_tokens):
            special_tokens_to_add.append(vision_token)
        if self.tokenizer.eos_token != '<|endoftext|>':
            if not any('<|endoftext|>' in str(token) for token in self.tokenizer.additional_special_tokens):
                special_tokens_to_add.append('<|endoftext|>')
        if special_tokens_to_add:
            added_tokens = self.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens_to_add})
            if added_tokens > 0:
                tokens_added = True
                print(f"[Tokenizer Setup] Added {added_tokens} special tokens: {special_tokens_to_add}")
        if tokens_added:
            old_embeddings = self.language_model.get_input_embeddings().weight.size(0)
            self.language_model.resize_token_embeddings(len(self.tokenizer))
            new_embeddings = self.language_model.get_input_embeddings().weight.size(0)
            print(f"[Tokenizer Setup] Resized embeddings: {old_embeddings} -> {new_embeddings}")
        self.tokenizer.padding_side = "right"
        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.bos_token_id = self.tokenizer.bos_token_id
        if self.pad_token_id == self.eos_token_id:
            print(f"[Tokenizer Setup] ✓ pad_token_id == eos_token_id (consistent with loss masking)")

    # ==================== 공통 처리 함수들 ====================
    def _process_vision_encoder(self, pixel_values: torch.Tensor) -> Dict[str, Any]:
        """
        이미지 B,V,3,H,W를 B*V,3,H,W로 변환하여 비전 인코더를 통과시킴
        
        Returns:
            Dict with vision_features: [B*V, S, D_vision], batch_size, num_views
        """
        target_dtype = resolve_module_dtype(
            self._cached_module_dtypes,
            "vision_encoder",
            getattr(self.vision_backbone, 'encoder', None),
            pixel_values.dtype,
        )
        batch_size, num_views, normalized_pixels = self.vision_backbone.normalize_inputs(pixel_values, target_dtype)
        vision_features = self.vision_backbone.extract_features(normalized_pixels)  # [B*V, S, D_vision]

        return {
            "vision_features": vision_features,
            "batch_size": batch_size,
            "num_views": num_views,
            "device": normalized_pixels.device
        }
    
    def _process_resampler(self, vision_features: torch.Tensor, batch_size: int, num_views: int) -> torch.Tensor:
        """
        Resampler를 통한 vision feature 변환

        Args:
            vision_features: [B*V, S, D_vision] - Vision encoder의 출력

        Returns:
            resampled_features: [B*V, S, D_latent] - Resampler 출력
        """
        target_dtype = resolve_module_dtype(
            self._cached_module_dtypes,
            "resampler",
            getattr(self.resampler_module, 'resampler', None),
            vision_features.dtype,
        )
        return self.resampler_module(vision_features, target_dtype)
    
    def _process_projection_layer(self, resampled_features: torch.Tensor, batch_size: int, num_views: int) -> torch.Tensor:
        """
        Projection layer 처리: B*V,L,D -> B,V*L,D 변환
        
        Args:
            resampled_features: [B*V, S, D_latent] - Resampler 출력
            
        Returns:
            projected_features: [B, V*S, D_lm] - 투영된 특징
        """
        return self.projector(resampled_features, batch_size, num_views, self._cached_module_dtypes, self.language_model)
    
    
    def _fuse_text_image_embeddings(self, vision_tokens: torch.Tensor, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None, stage: str = "finetune") -> Dict[str, torch.Tensor]:
        """
        텍스트 토큰을 임베딩하고 이미지 토큰 자리에 vision tokens 추가
        
        Args:
            vision_tokens: [B, V*S, D_lm] - 투영된 vision tokens
            input_ids: [B, L] - 텍스트 토큰 ID
            
        Returns:
            Dict with inputs_embeds: [B, L'-1+V*S, D_lm], attention_mask, labels
        """
        text_inputs = self._prepare_text_inputs(input_ids, attention_mask, labels, stage)
        return self._create_combined_inputs(vision_tokens, text_inputs=text_inputs)
    
    # ==================== 메인 순전파 및 생성 ====================
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        stage: str = "vision",
        **kwargs
    ):
        """
        Args:
            pixel_values: [B,V,C,H,W] 또는 [B,C,H,W]
            stage: "vision" | "resampler" | "finetune"
            **kwargs: 추가 파라미터들 (미래 확장성을 위해 보존)
        """
        # kwargs에서 추가 설정 추출 (필요시)
        debug_mode = kwargs.get('debug_mode', False)
        if debug_mode:
            print(f"[DEBUG] Forward pass - stage: {stage}, batch_size: {pixel_values.size(0)}")
        if stage == "vision":
            # Vision-only stage uses raw vision hidden states (no resampler/projection)
            batch_size, num_views, normalized_pixels = self._normalize_pixel_values(pixel_values)
            batch_size, num_views, normalized_pixels = self._maybe_trim_views_for_stage(
                normalized_pixels, stage, batch_size, num_views
            )
            vision_hidden_states = self._extract_vision_features(normalized_pixels, batch_size, num_views)
            if num_views <= 1 and not self._warned_single_view:
                print("[VICReg] Warning: num_views <= 1, VICReg 손실이 0이 됩니다.")
                self._warned_single_view = True

            if self.vicreg_loss_weight == 0.0:
                if not hasattr(self, '_warned_zero_vicreg_weight'):
                    print("[VICReg] Warning: VICReg 가중치 0.0 → 계산 스킵")
                    self._warned_zero_vicreg_weight = True
                zero = torch.zeros((), device=vision_hidden_states.device)
                return {"loss": zero, "vicreg_loss": zero, "vicreg_raw": zero, "vicreg_weight": 0.0}

            # NEW: VICReg projector 적용 (vision stage에서만)
            vicreg_feats = self.vicreg_projector(vision_hidden_states)

            vicreg_raw = self._compute_vicreg_overlap_loss(
                vicreg_feats, batch_size, num_views, overlap_ratio=self.overlap_ratio
            )
            vicreg_loss = vicreg_raw * self.vicreg_loss_weight
            total_loss = vicreg_loss
            return {
                "loss": total_loss,
                "vicreg_loss": vicreg_loss,
                "vicreg_raw": vicreg_raw.detach(),
                "vicreg_weight": self.vicreg_loss_weight,
                "vicreg_dim": self._vicreg_feat_dim,
            }

        # vision 외 단계: 공통 처리 파이프라인 사용
        if stage in ("resampler", "finetune"):
            if input_ids is None or labels is None:
                raise ValueError(f"'{stage}' 단계에서는 input_ids와 labels가 반드시 필요합니다.")
            
            # 1. Vision encoder 처리
            vision_result = self._process_vision_encoder(pixel_values)
            vision_features = vision_result["vision_features"]  # [B*V, S, D_vision]
            batch_size = vision_result["batch_size"]
            num_views = vision_result["num_views"]
            
            # 2. Resampler 처리
            resampled_features = self._process_resampler(vision_features, batch_size, num_views)  # [B*V, S, D_latent]
            
            # 3. Projection layer 처리
            vision_tokens = self._process_projection_layer(resampled_features, batch_size, num_views)  # [B, V*S, D_lm]
            
            # 4. 텍스트-이미지 융합 및 LM forward
            return self._compute_autoregressive_loss(
                vision_tokens, input_ids, attention_mask, labels, stage
            )

        raise ValueError("stage는 'vision', 'resampler', 'finetune' 중 하나여야 합니다.")

    @torch.inference_mode()
    def generate(self, pixel_values: torch.Tensor, input_ids: Optional[torch.Tensor] = None,
                 attention_mask: Optional[torch.Tensor] = None,
                 max_new_tokens: int = 32, temperature: float = 0.7, **kwargs):
        """forward와 동일한 처리 과정을 사용하는 생성 함수"""
        self.eval()
        try:
            # 1. Vision encoder 처리 (forward와 동일)
            vision_result = self._process_vision_encoder(pixel_values)
            vision_features = vision_result["vision_features"]  # [B*V, S, D_vision]
            batch_size = vision_result["batch_size"]
            num_views = vision_result["num_views"]
            device = vision_result["device"]
            
            # 2. Resampler 처리 (forward와 동일)
            resampled_features = self._process_resampler(vision_features, batch_size, num_views)  # [B*V, S, D_latent]
            
            # 3. Projection layer 처리 (forward와 동일)
            vision_tokens = self._process_projection_layer(resampled_features, batch_size, num_views)  # [B, V*S, D_lm]
            
            # 4. 생성용 입력 준비
            input_ids, attention_mask = self._prepare_generation_inputs(
                input_ids, attention_mask, batch_size, device
            )
            
            # 5. 텍스트-이미지 융합 (forward와 동일, 단 labels 없음)
            combined_inputs = self._fuse_text_image_embeddings(
                vision_tokens, input_ids=input_ids, attention_mask=attention_mask, stage="finetune"
            )
            
            # 6. LM generate 호출
            generation_kwargs = self._build_lm_generate_kwargs(combined_inputs, max_new_tokens, temperature, **kwargs)
            generated_ids = self.language_model.generate(**generation_kwargs)
            return self._decode_generated_text(generated_ids)
        except Exception as e:
            print(f"[Generate] Error in generation: {e}")
            import traceback; traceback.print_exc()
            return self._fallback_generation_output(pixel_values.size(0) if pixel_values.ndim in (4,5) else 1,
                                                        pixel_values.device if isinstance(pixel_values, torch.Tensor) else 'cpu')

    # ==================== 기본 유틸리티 헬퍼 ====================
    def _normalize_pixel_values(self, pixel_values):
        target_dtype = resolve_module_dtype(
            self._cached_module_dtypes,
            "vision_encoder",
            getattr(self.vision_backbone, 'encoder', None),
            pixel_values.dtype,
        )
        return self.vision_backbone.normalize_inputs(pixel_values, target_dtype)

    def _maybe_trim_views_for_stage(
        self,
        pixel_values: torch.Tensor,
        stage: str,
        batch_size: int,
        num_views: int,
    ) -> tuple[int, int, torch.Tensor]:
        if stage != "vision":
            return batch_size, num_views, pixel_values
        if pixel_values.ndim != 5 or num_views <= 0:
            return batch_size, num_views, pixel_values

        drop_count = 0
        if self.skip_first_view_in_vision_stage:
            drop_count = 1
        else:
            expected = getattr(self, "vision_stage_expected_views", None)
            if expected is not None and expected > 0 and num_views > expected:
                drop_count = num_views - int(expected)

        if drop_count <= 0:
            return batch_size, num_views, pixel_values

        drop_count = min(drop_count, num_views - 1)
        if drop_count <= 0:
            return batch_size, num_views, pixel_values

        trimmed = pixel_values[:, drop_count:, ...].contiguous()
        if not getattr(self, "_vision_stage_trim_warned", False):
            try:
                print(f"[Vision Stage] Dropping {drop_count} leading view(s) prior to VICReg ({num_views}->{trimmed.size(1)})")
            except Exception:
                pass
            self._vision_stage_trim_warned = True

        return batch_size, trimmed.size(1), trimmed

    def _extract_vision_features(self, pixel_values, batch_size, num_views):
        return self.vision_backbone.extract_features(pixel_values)

    def _prepare_generation_inputs(self, input_ids, attention_mask, batch_size, device):
        if input_ids is None:
            print("[Generate] Warning: input_ids not provided, creating default prompt for captioning")
        return self.language_fusion.prepare_generation_inputs(input_ids, attention_mask, batch_size, device)

    def _create_default_prompt(self, batch_size, device):
        return self.language_fusion.create_default_prompt(batch_size, device)

    def _adjust_batch_size(self, input_ids, attention_mask, target_batch_size):
        return self.language_fusion.adjust_batch_size(input_ids, attention_mask, target_batch_size)

    def _get_tokenizer_info(self):
        try:
            if hasattr(self.language_model, 'config'):
                config = self.language_model.config
            elif hasattr(self.language_model, 'base_model') and hasattr(self.language_model.base_model, 'config'):
                config = self.language_model.base_model.config
            else:
                config = None
            if config:
                return {
                    'pad_token_id': getattr(config, 'pad_token_id', None),
                    'eos_token_id': getattr(config, 'eos_token_id', None),
                    'bos_token_id': getattr(config, 'bos_token_id', None),
                }
            return {}
        except Exception:
            return {}

    def _build_generation_kwargs(self, combined_inputs, max_new_tokens, temperature, **kwargs):
        temperature = max(0.1, min(temperature, 1.0))
        MIN_NEW = 5
        DEFAULTS = dict(top_p=0.9, top_k=50, repetition_penalty=1.1, length_penalty=1.0)
        tk = self._get_tokenizer_info()
        out = {
            **combined_inputs,
            'max_new_tokens': max_new_tokens,
            'min_new_tokens': kwargs.get('min_new_tokens', MIN_NEW),
            'temperature': temperature,
            'do_sample': temperature > 0.1,
            'top_p': kwargs.get('top_p', DEFAULTS['top_p']),
            'top_k': kwargs.get('top_k', DEFAULTS['top_k']),
            'repetition_penalty': kwargs.get('repetition_penalty', DEFAULTS['repetition_penalty']),
            'length_penalty': kwargs.get('length_penalty', DEFAULTS['length_penalty']),
        }
        if tk.get('pad_token_id') is not None: out['pad_token_id'] = tk['pad_token_id']
        if tk.get('eos_token_id') is not None: out['eos_token_id'] = tk['eos_token_id']
        for key in ['pad_token_id', 'eos_token_id', 'bos_token_id']:
            if key in kwargs:
                out[key] = kwargs[key]
        return out

    def _build_lm_generate_kwargs(self, combined_inputs, max_new_tokens, temperature, **kwargs):
        return self._build_generation_kwargs(combined_inputs, max_new_tokens, temperature, **kwargs)

    def _postprocess_generated_text(self, generated_ids):
        generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        cleaned_texts = [t.strip() for t in generated_texts]
        return {"generated_ids": generated_ids, "text": cleaned_texts}

    def _decode_generated_text(self, generated_ids):
        return self._postprocess_generated_text(generated_ids)

    def _get_fallback_generation_result(self, batch_size, device):
        fallback_text = ["a panoramic view"] * batch_size
        fallback_ids = torch.ones((batch_size, 5), dtype=torch.long, device=device)
        return {"generated_ids": fallback_ids, "text": fallback_text}

    def _fallback_generation_output(self, batch_size, device):
        return self._get_fallback_generation_result(batch_size, device)

    # ==================== VICReg & AR 헬퍼 ====================
    def _compute_vicreg_overlap_loss(
        self,
        vision_output: torch.Tensor,   # [B*V, S, D_vicreg]
        batch_size: int,
        num_views: int,
        overlap_ratio: float = 0.5,
        pair_chunk: Optional[int] = 8,  # 메모리 절약용 청킹 (None이면 full-batch)
    ):
        """
        벡터화된 VICReg overlap loss 계산
        - 원래 구현과 동일하게 (v, v+1 mod V) 페어별로 VICReg을 구해 평균냄
        - 파이썬 루프 제거 → 큰 폭의 속도 개선
        - 공분산 항은 [P, D, D]를 만들기 때문에 D가 아주 큰 경우 pair_chunk로 나눠 계산 권장

        Args:
            pair_chunk: 한 번에 처리할 페어 수(P=B*V). 메모리 피크 낮추고 싶을 때 설정.
        """
        if num_views <= 1:
            return torch.zeros((), device=vision_output.device)

        return compute_vicreg_overlap_loss(
            vision_output,
            batch_size=batch_size,
            num_views=num_views,
            overlap_ratio=overlap_ratio,
            pair_chunk=pair_chunk,
            vicreg_loss_module=self.vicreg_loss,
            has_cls_token=self.vision_backbone.has_cls_token(vision_output),
        )


    def _prepare_text_inputs(self, input_ids, attention_mask, labels=None, stage="finetune"):
        """텍스트 입력 준비 (projection 적용 포함)"""
        text_inputs = self.language_fusion.prepare_text_inputs(input_ids, attention_mask, labels)
        
        # 2단계(resampler) 학습에서만 텍스트 projection 적용
        if self.use_text_projection and hasattr(self, 'text_projection') and stage == "resampler":
            text_inputs["inputs_embeds"] = self.text_projection(text_inputs["inputs_embeds"])
        
        return text_inputs

    def _create_combined_inputs(self, vision_tokens, input_ids=None, attention_mask=None, labels=None, text_inputs=None):
        """
        학습 시 결합 로직을 LanguageFusion 서브모듈에 위임합니다.
        """
        return self.language_fusion.fuse(
            vision_tokens,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            text_inputs=text_inputs,
        )

    def _compose_multimodal_embeddings(self, vision_tokens, input_ids=None, attention_mask=None, labels=None, text_inputs=None):
        return self._create_combined_inputs(vision_tokens, input_ids=input_ids, attention_mask=attention_mask, labels=labels, text_inputs=text_inputs)

    # Gradient checkpointing utilities -------------------------------------------------
    def gradient_checkpointing_enable(self, use_reentrant: bool | None = None):
        """Enable gradient checkpointing for submodules that support it."""
        if hasattr(self.language_model, "gradient_checkpointing_enable"):
            try:
                kwargs = {}
                if use_reentrant is not None:
                    kwargs["use_reentrant"] = use_reentrant
                self.language_model.gradient_checkpointing_enable(**kwargs)
            except TypeError:
                self.language_model.gradient_checkpointing_enable()
        encoder = getattr(self.vision_backbone, 'encoder', None)
        if hasattr(encoder, "gradient_checkpointing_enable"):
            try:
                encoder.gradient_checkpointing_enable()
            except TypeError:
                encoder.gradient_checkpointing_enable()
        self._gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        if hasattr(self.language_model, "gradient_checkpointing_disable"):
            self.language_model.gradient_checkpointing_disable()
        encoder = getattr(self.vision_backbone, 'encoder', None)
        if hasattr(encoder, "gradient_checkpointing_disable"):
            encoder.gradient_checkpointing_disable()
        self._gradient_checkpointing = False

    def _compute_autoregressive_loss(self, vision_tokens, input_ids, attention_mask, labels, stage="finetune"):
        text_inputs = self._prepare_text_inputs(input_ids, attention_mask, labels, stage)
        combined_inputs = self._create_combined_inputs_for_training(vision_tokens, text_inputs)
        try:
            outputs = self.language_model(
                inputs_embeds=combined_inputs['inputs_embeds'],
                attention_mask=combined_inputs['attention_mask'],
                labels=combined_inputs['labels'],
                return_dict=True
            )
            if not torch.isfinite(outputs.loss):
                manual = self._compute_manual_cross_entropy_loss(outputs.logits, combined_inputs['labels'])
                manual.setdefault('ar_loss', manual['loss'])
                return manual
            if hasattr(self, '_debug_loss_verification') and self._debug_loss_verification:
                manual = self._compute_manual_cross_entropy_loss(outputs.logits, combined_inputs['labels'])
                diff = abs(outputs.loss.item() - manual['loss'].item())
                if diff > 0.01:
                    print(f"[Loss Debug] HF loss: {outputs.loss.item():.6f}, Manual loss: {manual['loss'].item():.6f}, Diff: {diff:.6f}")
            return {"loss": outputs.loss, "ar_loss": outputs.loss, "logits": outputs.logits}
        except Exception:
            fallback_loss = torch.tensor(0.0, device=vision_tokens.device, requires_grad=True)
            return {"loss": fallback_loss, "ar_loss": fallback_loss,
                    "logits": torch.zeros((text_inputs['batch_size'], combined_inputs['inputs_embeds'].size(1),
                                           self.language_model.config.vocab_size), device=vision_tokens.device)}

    def _create_combined_inputs_for_training(self, vision_tokens, text_inputs):
        return self._create_combined_inputs(vision_tokens, text_inputs=text_inputs)

    def _compute_manual_cross_entropy_loss(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        if not hasattr(self, '_debug_shift_logged'):
            valid_tokens = (shift_labels != self.ignore_index).sum()
            total_tokens = shift_labels.numel()
            print(f"[Loss Debug] logits: {logits.shape} -> {shift_logits.shape}, labels: {labels.shape} -> {shift_labels.shape}")
            print(f"[Loss Debug] Valid tokens: {valid_tokens.item()}/{total_tokens} ({valid_tokens.item()/total_tokens*100:.1f}%)")
            self._debug_shift_logged = True
        return {"loss": loss, "ar_loss": loss, "perplexity": torch.exp(loss) if torch.isfinite(loss) else torch.tensor(float('inf'))}


    # ==================== LoRA 유틸 (생략 없이 유지) ====================
    def setup_lora_for_finetune(self, lora_r: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.1, target_modules: Optional[list] = None) -> bool:
        if not PEFT_AVAILABLE:
            print("Warning: PEFT not available. LoRA setup skipped.")
            return False
        try:
            if target_modules is None:
                target_modules = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
            lora_config = LoraConfig(r=lora_r, lora_alpha=lora_alpha, target_modules=target_modules,
                                     lora_dropout=lora_dropout, bias="none", task_type=TaskType.CAUSAL_LM)
            self.language_model = get_peft_model(self.language_model, lora_config)
            if hasattr(self.language_model, 'enable_input_require_grads'):
                self.language_model.enable_input_require_grads()
            trainable_params = sum(p.numel() for p in self.language_model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.language_model.parameters())
            print(f"✓ LoRA setup: trainable {trainable_params:,}/{total_params:,} ({trainable_params/total_params*100:.2f}%)")
            return True
        except Exception as e:
            print(f"Error setting up LoRA: {e}")
            return False

    def load_lora_weights(self, load_path: str):
        if not PEFT_AVAILABLE:
            print("Warning: PEFT not available. Cannot load LoRA weights.")
            return False
        try:
            from peft import PeftModel
            is_already_peft = hasattr(self.language_model, 'peft_config')
            if is_already_peft:
                print("Model already PEFT-enabled. Trying to load adapter...")
                if hasattr(self.language_model, 'load_adapter'):
                    adapter_name = "eval_adapter"
                    try:
                        self.language_model.load_adapter(load_path, adapter_name=adapter_name)
                        self.language_model.set_adapter(adapter_name)
                        print(f"✓ LoRA adapter '{adapter_name}' loaded")
                        return True
                    except Exception as e:
                        print(f"load_adapter failed, fallback to manual state_dict load: {e}")
                import os
                state_dict = None
                # safetensors 우선 -> 실패 시 PyTorch bin 시도
                st_path = os.path.join(load_path, "adapter_model.safetensors")
                if os.path.exists(st_path):
                    try:
                        from safetensors.torch import load_file as load_sft
                        state_dict = load_sft(st_path)
                    except Exception as _e:
                        print(f"Warning: failed to load safetensors adapter: {_e}")
                        state_dict = None
                if state_dict is None:
                    bin_path = os.path.join(load_path, "adapter_model.bin")
                    if os.path.exists(bin_path):
                        try:
                            import torch
                            state_dict = torch.load(bin_path, map_location='cpu')
                        except Exception as _e:
                            print(f"Warning: failed to load bin adapter: {_e}")
                            state_dict = None
                if state_dict is None:
                    print(f"Error: No adapter weights found in {load_path} (adapter_model.safetensors or adapter_model.bin)")
                    return False
                cleaned = {}
                for k, v in state_dict.items():
                    ck = k
                    for prefix in ('base_model.model.', 'base_model.'):
                        if ck.startswith(prefix):
                            ck = ck[len(prefix):]
                            break
                    cleaned[ck] = v
                missing, unexpected = self.language_model.load_state_dict(cleaned, strict=False)
                loaded_cnt = len(cleaned) - len(missing)
                if loaded_cnt == 0:
                    print("Error: No LoRA weights matched.")
                    return False
                print(f"✓ LoRA weights loaded: {loaded_cnt}/{len(cleaned)} (missing {len(missing)}, unexpected {len(unexpected)})")
                return True
            else:
                print("Converting to PeftModel.from_pretrained...")
                self.language_model = PeftModel.from_pretrained(self.language_model, load_path, is_trainable=False)
                print("✓ LoRA weights loaded via PeftModel")
                return True
        except Exception as e:
            import traceback
            print(f"Error loading LoRA weights: {e}")
            print(f"Detailed error: {traceback.format_exc()}")
            return False

    def merge_lora_weights(self):
        if not PEFT_AVAILABLE:
            print("Warning: PEFT not available. Cannot merge LoRA weights.")
            return False
        if hasattr(self.language_model, 'merge_and_unload'):
            try:
                self.language_model = self.language_model.merge_and_unload()
                print("✓ LoRA weights merged into base model")
                return True
            except Exception as e:
                print(f"Error merging LoRA weights: {e}")
                return False
        else:
            print("Warning: Language model does not support LoRA weight merging.")
            return False

    def get_lora_info(self) -> Dict[str, Any]:
        if not PEFT_AVAILABLE:
            return {"peft_available": False}
        info = {"peft_available": True}
        if hasattr(self.language_model, 'peft_config'):
            peft_config = getattr(self.language_model, 'peft_config', {})
            if peft_config:
                adapter_name = list(peft_config.keys())[0]
                config = peft_config.get(adapter_name, None)
                if config is not None:
                    info.update({
                        "is_lora_enabled": True,
                        "lora_r": getattr(config, 'r', None),
                        "lora_alpha": getattr(config, 'lora_alpha', None),
                        "lora_dropout": getattr(config, 'lora_dropout', None),
                        "target_modules": getattr(config, 'target_modules', None),
                        "adapter_name": adapter_name
                    })
                else:
                    info["is_lora_enabled"] = False
            else:
                info["is_lora_enabled"] = False
        else:
            info["is_lora_enabled"] = False
        return info

    def save_lora_weights(self, save_path: str) -> bool:
        if not PEFT_AVAILABLE:
            print("Warning: PEFT not available. Cannot save LoRA weights.")
            return False
        try:
            from pathlib import Path
            lora_info = self.get_lora_info()
            if not lora_info.get("is_lora_enabled", False):
                print("Warning: LoRA is not enabled. No weights to save.")
                return False
            save_dir = Path(save_path)
            save_dir.mkdir(parents=True, exist_ok=True)
            # safetensors를 사용하지 않도록 safe_serialization=False로 강제
            if safe_save_pretrained(self.language_model, str(save_dir), safe_serialization=False):
                print(f"✓ LoRA weights saved: {save_dir}")
                return True
            else:
                print("Error: Failed to save LoRA weights")
                return False
        except Exception as e:
            print(f"Error saving LoRA weights: {e}")
            import traceback; traceback.print_exc()
            return False

    # (저장 메서드 삭제: 체크포인트 저장은 Lightning의 ModelCheckpoint에 위임)

    @classmethod
    def from_checkpoint(cls, 
                       checkpoint_path: str, 
                       lora_weights_path: Optional[str] = None,
                       device: str = "auto",
                       auto_detect_lora: bool = True,
                       strict_loading: bool = False,
                       **model_kwargs) -> 'PanoramaVLM':
        """
        통합된 체크포인트 로딩 메서드 - Lightning 체크포인트와 LoRA 가중치를 자동으로 처리
        
        Args:
            checkpoint_path (str): Lightning 체크포인트 파일 경로 (.ckpt)
            lora_weights_path (str, optional): LoRA 가중치 디렉토리 경로
            device (str): 모델을 로드할 디바이스 ("auto", "cuda", "cpu")
            auto_detect_lora (bool): LoRA 가중치 자동 감지 여부
            strict_loading (bool): 엄격한 가중치 로딩 여부
            **model_kwargs: 모델 생성에 필요한 추가 파라미터들
            
        Returns:
            PanoramaVLM: 로드된 모델 인스턴스
            
        Example:
            # 기본 사용법
            model = PanoramaVLM.from_checkpoint("runs/best.ckpt")
            
            # LoRA 경로 직접 지정
            model = PanoramaVLM.from_checkpoint(
                "runs/best.ckpt", 
                lora_weights_path="runs/lora_weights"
            )
        """
        import torch
        from pathlib import Path
        
        print(f"🚀 PanoramaVLM 체크포인트 로딩: {checkpoint_path}")
        
        # 디바이스 설정
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        device_obj = torch.device(device)
        
        # 체크포인트 로드
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"체크포인트를 찾을 수 없습니다: {checkpoint_path}")
        
        print(f"📂 체크포인트 로딩 중...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device_obj, weights_only=False)
        except Exception as e:
            raise RuntimeError(f"체크포인트 로딩 실패: {e}")
        
        # 하이퍼파라미터 추출
        hparams = checkpoint.get('hyper_parameters', {})
        model_state_dict = checkpoint.get('state_dict', {})

        # 체크포인트로부터 투영 레이어 출력 차원 추정 (LM hidden size 유도)
        proj_out_dim = None
        try:
            for k in [
                'projector.linear.weight',
                'model.projector.linear.weight',
                'model.vision_to_language_projection.weight',
                'vision_to_language_projection.weight'
            ]:
                if k in model_state_dict:
                    w = model_state_dict[k]
                    # torch.Size([out_features, in_features])
                    if hasattr(w, 'shape') and len(w.shape) == 2:
                        proj_out_dim = int(w.shape[0])
                        break
        except Exception:
            proj_out_dim = None

        # Qwen2.5 계열의 hidden_size 추정 → 모델 이름 추론 (필요 시만 사용)
        def _infer_qwen_from_hidden_size(hs: Optional[int]) -> Optional[str]:
            if hs is None:
                return None
            # 최소한의 매핑만 제공 (현재 코드베이스에서 사용하는 모델들)
            mapping = {
                896:  'Qwen/Qwen2.5-0.5B-Instruct',
                1536: 'Qwen/Qwen2.5-1.5B-Instruct',
            }
            return mapping.get(int(hs))
        
        # 설정 시스템을 활용한 모델 파라미터 결정
        try:
            # 1. 설정 파일 자동 감지 시도
            from .config import ConfigManager, ModelConfig
            detected_config = ConfigManager.auto_detect_config(checkpoint_path)
            
            if detected_config:
                print(f"🔍 설정 파일 자동 감지 성공")
                model_config = detected_config
            else:
                print(f"🔍 설정 파일 감지 실패 - 하이퍼파라미터에서 생성")
                # 2. 하이퍼파라미터에서 설정 생성
                config_dict = {
                    'vision_name': hparams.get('vision_name', 'google/siglip-base-patch16-224'),
                    'language_model_name': hparams.get('language_model_name', 'Qwen/Qwen2.5-0.5B-Instruct'),
                    'resampler_type': hparams.get('resampler_type', 'mlp'),
                    'latent_dimension': hparams.get('latent_dimension', 768),
                    'vicreg_loss_weight': hparams.get('vicreg_loss_weight', 1.0),
                    'overlap_ratio': hparams.get('overlap_ratio', 0.5),
                    'max_text_length': hparams.get('max_text_length', 512),
                }
                model_config = ModelConfig.from_dict(config_dict)
            
            # 2.5 체크포인트 하이퍼파라미터를 우선 적용하여 구조적 불일치 최소화
            #     (auto_detect_config가 전역 config.json을 잡을 수 있으므로, hparams로 핵심 값 동기화)
            if isinstance(model_config, ModelConfig) and isinstance(hparams, dict) and len(hparams) > 0:
                override_keys = [
                    'vision_name', 'language_model_name', 'resampler_type',
                    'latent_dimension', 'max_text_length', 'vicreg_loss_weight', 'overlap_ratio',
                    'resampler_depth', 'resampler_hidden_dim', 'resampler_use_ln',
                    'resampler_num_latents', 'resampler_heads', 'resampler_dropout'
                ]
                hp_overrides = {k: v for k, v in hparams.items() if k in override_keys}
                if hp_overrides:
                    try:
                        model_config = model_config.update(**hp_overrides)
                        print(f"🧩 체크포인트 하이퍼파라미터로 핵심 설정 동기화: {list(hp_overrides.keys())[:5]}{'...' if len(hp_overrides)>5 else ''}")
                    except Exception as _e:
                        print(f"[from_checkpoint] Warning: failed to merge hparams into config: {_e}")

            # 투영 레이어로부터 유도된 LM 이름을 우선 고려 (체크포인트와 구조 일치 보장)
            inferred_lm_name = _infer_qwen_from_hidden_size(proj_out_dim)
            if inferred_lm_name:
                try:
                    model_config = model_config.update(language_model_name=inferred_lm_name)
                    print(f"🧭 체크포인트 투영 차원({proj_out_dim})에 맞춰 LM 고정: {inferred_lm_name}")
                except Exception as _e:
                    print(f"[from_checkpoint] Warning: failed to enforce LM from projection dim: {_e}")

            # 3. 사용자 지정 파라미터로 오버라이드 (허용된 키만)
            if model_kwargs:
                try:
                    allowed = set(ModelConfig().__dict__.keys())
                except Exception:
                    allowed = set()
                filtered = {k: v for k, v in model_kwargs.items() if k in allowed}
                # 만약 체크포인트에서 LM 차원을 유도했으면, 사용자 입력으로 인한 구조 불일치 방지
                if inferred_lm_name and 'language_model_name' in filtered and filtered['language_model_name'] != inferred_lm_name:
                    print(f"⚠️  사용자 LM({filtered['language_model_name']})가 체크포인트 투영 차원({proj_out_dim})과 불일치 → 무시하고 {inferred_lm_name} 사용")
                    filtered.pop('language_model_name', None)
                if filtered:
                    print(f"🛠️  사용자 파라미터로 설정 오버라이드: {list(filtered.keys())}")
                    try:
                        model_config = model_config.update(**filtered)
                    except Exception as _e:
                        print(f"[from_checkpoint] Warning: failed to apply user kwargs: {_e}")
            
            # 4. 모델 생성용 파라미터 추출
            model_params = model_config.get_model_kwargs()
            model_params['config'] = model_config  # config 객체도 전달

        except Exception as e:
            print(f"⚠️ 설정 시스템 사용 실패 ({e}) - 기존 방식 사용")
            # 폴백: 기존 방식
            model_params = {
                'vision_name': 'google/siglip-base-patch16-224',
                'language_model_name': 'Qwen/Qwen2.5-0.5B-Instruct',
                'resampler_type': 'mlp',
                'latent_dimension': 768,
                'vicreg_loss_weight': 1.0,
                'overlap_ratio': 0.5,
                'max_text_length': 512,
            }
            
            # 하이퍼파라미터에서 업데이트
            for key in model_params.keys():
                if key in hparams:
                    model_params[key] = hparams[key]
            
            # 사용자 지정 파라미터로 최종 업데이트
            model_params.update(model_kwargs)
        
        print(f"🛠️  모델 파라미터:")
        preview = {k: v for k, v in model_params.items() if k != 'config'}
        for i, (key, value) in enumerate(preview.items()):
            if i >= 12:
                print("   - ...")
                break
            print(f"   - {key}: {value}")
        
        # 모델 인스턴스 생성
        print(f"🏗️  모델 인스턴스 생성 중...")
        model = cls(**model_params)
        
        # Lightning wrapper에서 실제 모델 가중치 추출
        print(f"⚙️  가중치 로딩 중...")
        model_weights = {}
        for key, value in model_state_dict.items():
            if key.startswith('model.'):
                # 'model.' 접두어 제거
                clean_key = key[6:]  # len('model.') = 6
                model_weights[clean_key] = value
        
        # 가중치 로드
        if model_weights:
            missing_keys, unexpected_keys = model.load_state_dict(model_weights, strict=strict_loading)
            loaded_cnt = len(model_weights) - len(missing_keys)
            print(f"   - 로드된 키: {loaded_cnt}")
            if missing_keys:
                print(f"   - 누락된 키: {len(missing_keys)} (예: {missing_keys[:3]}{'...' if len(missing_keys) > 3 else ''})")
            if unexpected_keys:
                print(f"   - 예상치 못한 키: {len(unexpected_keys)} (예: {unexpected_keys[:3]}{'...' if len(unexpected_keys) > 3 else ''})")
        else:
            print("   ⚠️  모델 가중치를 찾을 수 없습니다. 기본 초기화된 모델을 사용합니다.")
        
        # LoRA 가중치 처리
        if auto_detect_lora and lora_weights_path is None:
            # 자동 감지: 체크포인트와 같은 디렉토리에서 lora_weights 폴더 찾기
            checkpoint_dir = checkpoint_path.parent
            potential_lora_path = checkpoint_dir / "lora_weights"
            if potential_lora_path.exists() and potential_lora_path.is_dir():
                lora_weights_path = str(potential_lora_path)
                print(f"🔍 LoRA 가중치 자동 감지: {lora_weights_path}")
        
        if lora_weights_path:
            lora_path = Path(lora_weights_path)
            if lora_path.exists():
                print(f"🔧 LoRA 가중치 로딩: {lora_weights_path}")
                success = model.load_lora_weights(str(lora_path))
                if success:
                    lora_info = model.get_lora_info()
                    if lora_info.get("is_lora_enabled", False):
                        print(f"   ✅ LoRA 로딩 성공 - Rank: {lora_info.get('lora_r')}, Alpha: {lora_info.get('lora_alpha')}")
                    else:
                        print(f"   ⚠️  LoRA 상태 확인 실패")
                else:
                    print(f"   ❌ LoRA 로딩 실패")
            else:
                print(f"   ⚠️  LoRA 경로가 존재하지 않습니다: {lora_weights_path}")
        
        # 모델을 지정된 디바이스로 이동
        model = model.to(device_obj)
        model.eval()  # 기본적으로 평가 모드
        
        # 토크나이저 정보 추가 (eval.py 호환성)
        if not hasattr(model, 'tokenizer'):
            try:
                from transformers import AutoTokenizer
                tokenizer_name = model_params.get('language_model_name', 'Qwen/Qwen2.5-0.5B-Instruct')
                model.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
                print(f"   ✅ 토크나이저 로드: {tokenizer_name}")
            except Exception as e:
                print(f"   ⚠️ 토크나이저 로드 실패: {e}")
        
        print(f"✅ 모델 로딩 완료 - Device: {device}")
        return model

    @classmethod
    def from_pretrained_dir(
        cls,
        pretrained_dir: str,
        device: str = "auto",
        strict_loading: bool = False,
        **model_kwargs
    ) -> 'PanoramaVLM':
        """
        HuggingFace-style로 저장된 디렉토리에서 모델 로드.
        - save_pretrained로 저장된 폴더 구조를 기대 (pytorch_model.bin, config.json)
        - config.json이 있으면 파라미터 추출, 전달된 model_kwargs가 우선 적용
        """
        from pathlib import Path
        import json

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        device_obj = torch.device(device)

        p = Path(pretrained_dir)
        if not p.exists() or not p.is_dir():
            raise FileNotFoundError(f"Pretrained directory not found: {pretrained_dir}")

        # 기본 파라미터
        params = {
            'vision_name': 'google/siglip-base-patch16-224',
            'language_model_name': 'Qwen/Qwen2.5-0.5B-Instruct',
            'resampler_type': 'mlp',
            'latent_dimension': 768,
            'vicreg_loss_weight': 1.0,
            'overlap_ratio': 0.5,
            'max_text_length': 512,
        }

        # config.json에서 보정
        cfg_path = p / "config.json"
        if cfg_path.exists():
            try:
                with open(cfg_path, 'r', encoding='utf-8') as f:
                    saved_cfg = json.load(f)
                params.update({
                    'vision_name': saved_cfg.get('vision_name', params['vision_name']),
                    'language_model_name': saved_cfg.get('language_model_name', params['language_model_name']),
                    'latent_dimension': saved_cfg.get('latent_dimension', params['latent_dimension']),
                    'max_text_length': saved_cfg.get('max_text_length', params['max_text_length']),
                    'vicreg_loss_weight': saved_cfg.get('vicreg_loss_weight', params['vicreg_loss_weight']),
                    'overlap_ratio': saved_cfg.get('overlap_ratio', params['overlap_ratio']),
                })
            except Exception as e:
                print(f"[from_pretrained_dir] Warning: failed to parse config.json: {e}")

        # 외부 전달 인자 우선하되 'config'/'model_config' 키는 무시하여 저장된 설정과의 구조 불일치를 방지
        if model_kwargs:
            filtered = {k: v for k, v in model_kwargs.items() if k not in ("config", "model_config")}
            if filtered:
                params.update(filtered)

        print(f"🏗️  모델 인스턴스 생성(from_pretrained_dir): {pretrained_dir}")
        model = cls(**params)

        # 가중치 파일 찾기 (PyTorch bin만 지원)
        state_path = None
        if (p/"pytorch_model.bin").exists():
            state_path = p/"pytorch_model.bin"
            try:
                state = torch.load(str(state_path), map_location='cpu')
            except Exception as e:
                print(f"[from_pretrained_dir] Failed to load torch model: {e}")
                state = None
        else:
            state = None

        if state is not None:
            try:
                missing_keys, unexpected_keys = model.load_state_dict(state, strict=strict_loading)
                print(f"   ✅ Weights loaded from {state_path}")
                if missing_keys:
                    print(f"   ⚠️ Missing keys: {len(missing_keys)} (showing first 5) {missing_keys[:5]}")
                if unexpected_keys:
                    print(f"   ⚠️ Unexpected keys: {len(unexpected_keys)} (showing first 5) {unexpected_keys[:5]}")
            except Exception as e:
                print(f"   ❌ Failed to load state dict: {e}")
        else:
            print(f"   ⚠️ No state dict found in {pretrained_dir}. Using randomly initialized weights.")

        model = model.to(device_obj)
        model.eval()
        # 토크나이저 보장
        if not hasattr(model, 'tokenizer'):
            try:
                model.tokenizer = AutoTokenizer.from_pretrained(params.get('language_model_name', 'Qwen/Qwen2.5-0.5B-Instruct'))
            except Exception:
                pass
        print(f"✅ 모델 로딩 완료 - Device: {device}")
        return model

# -------------------- 편의: combined inputs 생성 (공용) --------------------
# def _stack_pad_mask(mask_list):  # 사용되지 않음
#     return torch.cat(mask_list, dim=1)

# 클래스 메서드에 두기 애매한 보조 함수들 보완
def _create_combined_inputs_for_training(self, vision_tokens, text_inputs):
    return self._create_combined_inputs(vision_tokens, text_inputs=text_inputs)

# 바인딩
PanoramaVLM._create_combined_inputs_for_training = _create_combined_inputs_for_training
