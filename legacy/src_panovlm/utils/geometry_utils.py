"""
Geometry utilities for panorama vision processing

This module provides utilities for handling geometric transformations,
coordinate systems, and alignment for panoramic vision models.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


def infer_hw(num_patches: int) -> Tuple[int, int]:
    """
    Infer height and width from number of patches.
    Assumes square grid layout.
    
    Args:
        num_patches: Total number of patches
        
    Returns:
        Tuple of (height, width)
    """
    h = w = int(np.sqrt(num_patches))
    # Ensure h * w == num_patches
    while h * w < num_patches:
        if h <= w:
            h += 1
        else:
            w += 1
    return h, w


def compute_effective_fov(original_fov_deg: float, crop_ratio: float) -> float:
    """
    Compute effective field of view after central cropping.
    
    Formula: FOV' = 2 * arctan(keep * tan(FOV/2))
    
    Args:
        original_fov_deg: Original field of view in degrees
        crop_ratio: Central crop ratio (e.g., 0.5 for keeping central 50%)
        
    Returns:
        Effective field of view in degrees
    """
    fov_rad = np.radians(original_fov_deg)
    effective_fov_rad = 2 * np.arctan(crop_ratio * np.tan(fov_rad / 2))
    return np.degrees(effective_fov_rad)


def create_erp_warp_grid(source_view: Dict, target_view: Dict, grid_size: Tuple[int, int]) -> torch.Tensor:
    """
    Create ERP (Equirectangular Projection) warp grid for aligning two views.
    
    Args:
        source_view: Source view metadata with keys 'yaw', 'pitch', 'effective_fov'
        target_view: Target view metadata with keys 'yaw', 'pitch', 'effective_fov'  
        grid_size: Output grid size (height, width)
        
    Returns:
        Warp grid tensor of shape [1, H, W, 2] for grid_sample
    """
    height, width = grid_size
    
    # Calculate relative transformation
    yaw_diff = source_view['yaw'] - target_view['yaw']
    pitch_diff = source_view.get('pitch', 0.0) - target_view.get('pitch', 0.0)
    source_fov = source_view.get('effective_fov', source_view.get('original_fov', 90.0))
    
    # Create normalized coordinate grid [-1, 1]
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, height),
        torch.linspace(-1, 1, width),
        indexing='ij'
    )
    
    # Convert to perspective coordinates
    fov_rad = np.radians(source_fov)
    tan_half_fov = np.tan(fov_rad / 2)
    
    sphere_x = x * tan_half_fov
    sphere_y = y * tan_half_fov
    sphere_z = torch.ones_like(x)
    
    # Apply rotations
    # Yaw rotation (around Y axis)
    yaw_rad = np.radians(yaw_diff)
    cos_yaw, sin_yaw = np.cos(yaw_rad), np.sin(yaw_rad)
    
    rotated_x = sphere_x * cos_yaw + sphere_z * sin_yaw
    rotated_z = -sphere_x * sin_yaw + sphere_z * cos_yaw
    
    # Pitch rotation (around X axis)  
    pitch_rad = np.radians(pitch_diff)
    cos_pitch, sin_pitch = np.cos(pitch_rad), np.sin(pitch_rad)
    
    final_y = sphere_y * cos_pitch - rotated_z * sin_pitch
    final_z = sphere_y * sin_pitch + rotated_z * cos_pitch
    
    # Convert to ERP coordinates
    longitude = torch.atan2(rotated_x, final_z)  # [-π, π]
    latitude = torch.atan2(final_y, torch.sqrt(rotated_x**2 + final_z**2))  # [-π/2, π/2]
    
    # Normalize to [-1, 1] for grid_sample
    erp_x = longitude / np.pi
    erp_y = latitude / (np.pi / 2)
    
    # Stack and add batch dimension
    grid = torch.stack([erp_x, erp_y], dim=-1).unsqueeze(0)  # [1, H, W, 2]
    
    return grid


def create_erp_weight_mask(height: int, width: int, use_cosine_weighting: bool = True) -> torch.Tensor:
    """
    Create ERP area weighting mask to account for latitude distortion.
    
    Args:
        height: Grid height
        width: Grid width  
        use_cosine_weighting: Whether to apply cos(φ) weighting for ERP area correction
        
    Returns:
        Weight mask tensor of shape [height, width]
    """
    if not use_cosine_weighting:
        return torch.ones(height, width)
        
    # Create latitude coordinates
    y_coords = torch.linspace(-np.pi/2, np.pi/2, height)  # From north to south pole
    
    # Compute cosine weights for each latitude
    cos_weights = torch.cos(y_coords)
    
    # Expand to full grid
    weight_mask = cos_weights.unsqueeze(1).repeat(1, width)
    
    return weight_mask


def compute_overlap_mask(grid_size: Tuple[int, int], overlap_ratio: float = 0.5) -> torch.Tensor:
    """
    Create overlap mask for valid comparison regions.
    
    Args:
        grid_size: Grid size (height, width)
        overlap_ratio: Ratio of central region considered valid for comparison
        
    Returns:
        Binary mask tensor of shape [height, width]
    """
    height, width = grid_size
    
    # Calculate central region bounds
    center_h, center_w = height // 2, width // 2
    half_overlap_h = int((height * overlap_ratio) // 2)
    half_overlap_w = int((width * overlap_ratio) // 2)
    
    # Create mask
    mask = torch.zeros(height, width)
    
    h_start = max(0, center_h - half_overlap_h)
    h_end = min(height, center_h + half_overlap_h)
    w_start = max(0, center_w - half_overlap_w)
    w_end = min(width, center_w + half_overlap_w)
    
    mask[h_start:h_end, w_start:w_end] = 1.0
    
    return mask


def warp_features_geometric_align(
    source_features: torch.Tensor,
    target_features: torch.Tensor, 
    source_metadata: Dict,
    target_metadata: Dict,
    overlap_ratio: float = 0.5,
    use_erp_weighting: bool = True
) -> Dict[str, float]:
    """
    Compute geometrically-aligned similarity between two feature maps.
    
    This function addresses the core geometric misalignment issues by:
    1. Warping source features to target coordinate system
    2. Applying proper ERP area weighting (cos φ)
    3. Using overlap mask for valid comparison regions
    
    Args:
        source_features: Source feature tensor [C, H, W] or [N, C]
        target_features: Target feature tensor [C, H, W] or [N, C]
        source_metadata: Source view metadata (yaw, pitch, fov, etc.)
        target_metadata: Target view metadata
        overlap_ratio: Central region overlap ratio for comparison
        use_erp_weighting: Whether to apply ERP cos(φ) weighting
        
    Returns:
        Dictionary containing:
        - 'ocs': Overall Cosine Similarity (weighted mean)
        - 'residual_mean': Mean residual (1 - cosine)
        - 'valid_ratio': Fraction of valid comparison pixels
    """
    # Convert patch features to spatial format if needed
    if source_features.ndim == 2:  # [N, C] -> [C, H, W]
        num_patches, channels = source_features.shape
        spatial_size = int(np.sqrt(num_patches))
        source_features = source_features.T.view(channels, spatial_size, spatial_size)
        target_features = target_features.T.view(channels, spatial_size, spatial_size)
        
    # Add batch dimension
    if source_features.ndim == 3:
        source_features = source_features.unsqueeze(0)  # [1, C, H, W]
        target_features = target_features.unsqueeze(0)
        
    _, channels, height, width = source_features.shape
    
    # Create warp grid from target to source coordinate system
    warp_grid = create_erp_warp_grid(
        target_metadata, source_metadata, (height, width)
    )
    
    # Warp target features to align with source
    target_warped = F.grid_sample(
        target_features, warp_grid, 
        mode='bilinear', padding_mode='border', align_corners=False
    )
    
    # Create masks and weights
    erp_weights = create_erp_weight_mask(height, width, use_erp_weighting)
    overlap_mask = compute_overlap_mask((height, width), overlap_ratio)
    
    # Combined weighting
    combined_weights = erp_weights * overlap_mask
    valid_mask = combined_weights > 0
    
    if not valid_mask.any():
        return {'ocs': 0.0, 'residual_mean': 1.0, 'valid_ratio': 0.0}
    
    # Flatten for computation
    source_flat = source_features.view(channels, -1)  # [C, HW]
    target_warped_flat = target_warped.view(channels, -1)
    weights_flat = combined_weights.view(-1)
    valid_mask_flat = valid_mask.view(-1)
    
    # Extract valid pixels
    source_valid = source_flat[:, valid_mask_flat].T  # [N_valid, C]
    target_valid = target_warped_flat[:, valid_mask_flat].T
    weights_valid = weights_flat[valid_mask_flat]
    
    # Normalize features
    source_norm = F.normalize(source_valid, dim=1)
    target_norm = F.normalize(target_valid, dim=1)
    
    # Compute cosine similarity
    cosine_sim = torch.sum(source_norm * target_norm, dim=1)
    
    # Weighted statistics
    weights_sum = torch.sum(weights_valid)
    ocs = torch.sum(cosine_sim * weights_valid) / weights_sum
    residual = 1.0 - cosine_sim
    residual_mean = torch.sum(residual * weights_valid) / weights_sum
    
    return {
        'ocs': float(ocs),
        'residual_mean': float(residual_mean), 
        'valid_ratio': float(valid_mask.sum()) / float(valid_mask.numel())
    }


def analyze_phi_bins(
    features_a: torch.Tensor,
    features_b: torch.Tensor,
    num_bins: int = 5,
    use_erp_weighting: bool = True
) -> List[Dict[str, float]]:
    """
    Analyze similarity by latitude bins to understand ERP distortion effects.
    
    Args:
        features_a: First feature map [C, H, W] or [N, C]
        features_b: Second feature map [C, H, W] or [N, C]
        num_bins: Number of latitude bins
        use_erp_weighting: Apply cos(φ) weighting
        
    Returns:
        List of statistics per bin containing lat_range, mean_similarity, etc.
    """
    # Convert to spatial format if needed
    if features_a.ndim == 2:
        num_patches, channels = features_a.shape
        spatial_size = int(np.sqrt(num_patches))
        features_a = features_a.T.view(channels, spatial_size, spatial_size)
        features_b = features_b.T.view(channels, spatial_size, spatial_size)
        
    channels, height, width = features_a.shape
    
    # Compute patch-wise cosine similarity
    feat_a_flat = features_a.view(channels, -1).T  # [HW, C]
    feat_b_flat = features_b.view(channels, -1).T
    
    feat_a_norm = F.normalize(feat_a_flat, dim=1)
    feat_b_norm = F.normalize(feat_b_flat, dim=1)
    
    cosine_sim = torch.sum(feat_a_norm * feat_b_norm, dim=1)
    cosine_map = cosine_sim.view(height, width)
    
    # Create latitude bins
    y_coords = torch.linspace(-np.pi/2, np.pi/2, height)
    lat_bins = torch.linspace(-np.pi/2, np.pi/2, num_bins + 1)
    
    bin_stats = []
    for b in range(num_bins):
        lat_min, lat_max = lat_bins[b], lat_bins[b + 1]
        
        # Find rows in this latitude range
        in_bin = (y_coords >= lat_min) & (y_coords < lat_max)
        row_indices = torch.where(in_bin)[0]
        
        if len(row_indices) > 0:
            bin_cosines = cosine_map[row_indices, :].flatten()
            cos_phi = np.cos((lat_min + lat_max) / 2) if use_erp_weighting else 1.0
            
            bin_stats.append({
                'lat_range_deg': (float(np.degrees(lat_min)), float(np.degrees(lat_max))),
                'mean_similarity': float(torch.mean(bin_cosines)),
                'std_similarity': float(torch.std(bin_cosines)),
                'cos_phi_weight': float(cos_phi),
                'weighted_mean': float(torch.mean(bin_cosines) * cos_phi),
                'num_pixels': int(bin_cosines.numel())
            })
            
    return bin_stats


def analyze_seam_effects(
    features_a: torch.Tensor,
    features_b: torch.Tensor, 
    seam_width: int = 2
) -> Dict[str, float]:
    """
    Analyze panorama seam effects by comparing left/right boundaries vs center.
    
    Args:
        features_a: First feature map [C, H, W] or [N, C]
        features_b: Second feature map [C, H, W] or [N, C]
        seam_width: Width of seam region to analyze
        
    Returns:
        Dictionary with seam analysis statistics
    """
    # Convert to spatial format if needed
    if features_a.ndim == 2:
        num_patches, channels = features_a.shape
        spatial_size = int(np.sqrt(num_patches))
        features_a = features_a.T.view(channels, spatial_size, spatial_size)
        features_b = features_b.T.view(channels, spatial_size, spatial_size)
        
    channels, height, width = features_a.shape
    
    # Compute cosine similarity map
    feat_a_flat = features_a.view(channels, -1).T
    feat_b_flat = features_b.view(channels, -1).T
    
    feat_a_norm = F.normalize(feat_a_flat, dim=1)
    feat_b_norm = F.normalize(feat_b_flat, dim=1)
    
    cosine_sim = torch.sum(feat_a_norm * feat_b_norm, dim=1)
    cosine_map = cosine_sim.view(height, width)
    
    # Define regions
    left_seam = cosine_map[:, :seam_width]
    right_seam = cosine_map[:, -seam_width:]
    center_region = cosine_map[:, seam_width:-seam_width] if seam_width < width // 2 else cosine_map
    
    # Compute statistics
    left_mean = float(torch.mean(left_seam)) if left_seam.numel() > 0 else 0.0
    right_mean = float(torch.mean(right_seam)) if right_seam.numel() > 0 else 0.0
    center_mean = float(torch.mean(center_region)) if center_region.numel() > 0 else 0.0
    
    return {
        'left_seam_mean': left_mean,
        'right_seam_mean': right_mean,
        'center_mean': center_mean,
        'left_center_diff': left_mean - center_mean,
        'right_center_diff': right_mean - center_mean,
        'seam_consistency': abs(left_mean - right_mean),
        'seam_width': seam_width
    }