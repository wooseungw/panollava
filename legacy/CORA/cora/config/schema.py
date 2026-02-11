from typing import Any, Dict, List, Optional, Union, Literal
from pydantic import BaseModel, Field, model_validator
import logging

logger = logging.getLogger(__name__)

# Resampler Specific Defaults (SOTA 2025-10-28)
RESAMPLER_CONFIGS = {
    "mlp": {
        "hidden_dim": None,
        "depth": 3,
        "use_ln": True,
        "pool_type": "avg",
    },
    "bimamba": {
        "hidden_dim": 1024,  # SOTA
        "depth": 4,          # SOTA
        "d_state": 64,       # SOTA
        "d_conv": 4,         # SOTA
        "expand": 2,         # SOTA
        "use_ln": True,
        "norm_first": True,
        "dropout": 0.05,
    },
    "qformer": {
        "num_query_tokens": 64,
        "num_hidden_layers": 6,
        "num_attention_heads": 8,
    },
    "perceiver": {
        "num_latents": 64,
        "heads": 8,
        "depth": 6,
    }
}

class ImageProcessingConfig(BaseModel):
    crop_strategy: Literal["sliding_window", "e2p", "cubemap", "anyres", "anyres_max", "anyres_e2p", "resize"] = "anyres_e2p"
    image_size: Optional[List[int]] = None  # If None, inferred from vision model
    overlap_ratio: float = 0.5
    stitching_mode: str = "concat"
    stitch_target_to_view_width: bool = True
    stitch_interp: str = "linear"
    fov_deg: float = 90.0
    use_vision_processor: bool = True
    anyres_max_patches: int = 9
    normalize: bool = True

class LoRAConfig(BaseModel):
    use_lora: bool = True
    rank: int = 32
    alpha: int = 64
    dropout: float = 0.1
    target_modules: List[str] = [
        "q_proj", "k_proj", "v_proj", "o_proj", 
        "gate_proj", "up_proj", "down_proj"
    ]
    save_lora_only: bool = False

class GenerationConfig(BaseModel):
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    repetition_penalty: float = 1.1

class ModelConfig(BaseModel):
    # Vision
    vision_name: str = "google/siglip-base-patch16-224"
    use_vicreg_norm: bool = False
    vision_backbone_type: str = "hf"
    
    # Language
    language_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    
    # Resampler
    resampler_type: Literal["mlp", "bimamba", "qformer", "perceiver"] = "bimamba"
    latent_dimension: int = 768
    resampler_config: Dict[str, Any] = Field(default_factory=dict)
    
    # Projector / Positional Encoding
    use_projection_positional_encoding: bool = True
    pe_view_encoding_type: str = "sinusoidal"
    pe_spatial_encoding_type: str = "sinusoidal"
    pe_enable_continuity: bool = True
    
    # VICReg Projector (SOTA: Default 2-layer MLP)
    use_vicreg_projector: bool = True
    vicreg_projector_dim: Optional[int] = None # Defaults to latent_dimension
    vicreg_projector_depth: int = 2
    vicreg_projector_ln: bool = True
    
    # Text Projection (for stage 2)
    use_text_projection: bool = False

    @model_validator(mode='before')
    def apply_resampler_defaults(cls, values):
        resampler_type = values.get('resampler_type', 'bimamba')
        user_config = values.get('resampler_config', {})
        
        defaults = RESAMPLER_CONFIGS.get(resampler_type, {}).copy()
        
        # Merge defaults with user config (user config takes precedence)
        merged_config = {**defaults, **user_config}
        values['resampler_config'] = merged_config
        
        return values

class StageConfig(BaseModel):
    epochs: int = 3
    lr: float = 1e-4
    batch_size: int = 1
    accumulate_grad_batches: int = 1
    vicreg_loss_weight: float = 0.0
    vicreg_mode: str = "pairwise"
    vicreg_similarity_weight: float = 25.0
    vicreg_variance_weight: float = 25.0
    vicreg_covariance_weight: float = 1.0
    vision_trainable_blocks: int = 0
    max_text_length: int = 128
    # Optional overrides for specific stages
    image_processing: Optional[ImageProcessingConfig] = None
    data: Optional[Dict[str, Any]] = None

class TrainingConfig(BaseModel):
    output_dir: str = "outputs"
    stages: List[str] = ["vision", "resampler", "finetune"]
    
    # Trainer Args
    max_epochs: int = 10
    limit_val_batches: Optional[float] = None
    devices: Union[int, List[int]] = 1
    strategy: str = "auto"
    precision: str = "16-mixed"
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 1
    val_check_interval: Union[float, int] = 1.0
    
    # Optimizer / Scheduler
    learning_rate: float = 1e-4
    batch_size: int = 4
    num_workers: int = 4
    vision_trainable_blocks: int = 0
    
    resume_from_checkpoint: Optional[str] = None
    trackers: bool = False
    
    # Eval / Misc
    eval_batch_size: int = 2
    system_msg: str = "You are a helpful assistant. Describe the panorama image."
    wandb_project: Optional[str] = None
    cache_cleanup_interval: int = 1000
    seed: int = 42
    
    stage_configs: Dict[str, StageConfig] = Field(default_factory=dict)
    
    # DeepSpeed (Optional)
    deepspeed: Dict[str, Any] = Field(default_factory=lambda: {"enabled": False})

class CORAConfig(BaseModel):
    experiment: Dict[str, str] = Field(default_factory=lambda: {"name": "auto", "version": "1.0"})
    environment: Dict[str, str] = Field(default_factory=lambda: {"cuda_visible_devices": "0"})
    paths: Dict[str, str] = Field(default_factory=dict)
    
    models: ModelConfig
    image_processing: ImageProcessingConfig
    training: TrainingConfig
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    data: Dict[str, Any] = Field(default_factory=dict)
