"""
PBAI CUDA Vision Transformer - Three-Frame Identity System

Pure emergence vision processing on RTX 4070.
No pretraining - crash and burn til you don't.

════════════════════════════════════════════════════════════════════════════════
THREE-FRAME IDENTITY (Blender Model)
════════════════════════════════════════════════════════════════════════════════

Frame X: COLOR   - What IS        - Identity constraint
Frame Y: POSITION - Where IS      - Successor constraint
Frame Z: HEAT    - How much MATTERS - Multiplication constraint

These three frames are the fundamental representation of visual reality.
The transformer learns to attend across all three simultaneously.

════════════════════════════════════════════════════════════════════════════════
ARCHITECTURE
════════════════════════════════════════════════════════════════════════════════

Screen (H×W×3)
     ↓
Three-Frame Encoder
     ↓
┌─────────────────────────────────────┐
│  Frame X: Color field (RGB)         │
│  Frame Y: Position field (dx, dy)   │
│  Frame Z: Heat field (salience)     │
└─────────────────────────────────────┘
     ↓
Spiral Patch Embedding (center-out)
     ↓
Transformer Layers
     ↓
Attention = Heat (what matters)
     ↓
Compressed Output → Pi

════════════════════════════════════════════════════════════════════════════════
PLANCK GROUNDING
════════════════════════════════════════════════════════════════════════════════

- K × φ² = 4 (thermal quantum)
- Attention thresholds use 1/φⁿ ladder
- Fire zones in spiral: 6 (center) → 1 (peripheral)
- Learning rate scaled by K

════════════════════════════════════════════════════════════════════════════════
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# PLANCK CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

PHI = 1.618033988749895
K = 4 / (PHI ** 2)  # ≈ 1.528 (K × φ² = 4 exactly)

# Threshold ladder (1/φⁿ)
THRESHOLD_NOISE = 1 / (PHI ** 5)       # ≈ 0.090
THRESHOLD_MOVEMENT = 1 / (PHI ** 4)    # ≈ 0.146
THRESHOLD_EXISTENCE = 1 / (PHI ** 3)   # ≈ 0.236
THRESHOLD_ORDER = 1 / (PHI ** 2)       # ≈ 0.382
THRESHOLD_RIGHTEOUS = 1 / PHI          # ≈ 0.618

# Fire zone radii (as fraction of max distance from center)
FIRE_ZONES = {
    6: 0.10,  # Center - body temp zone
    5: 0.25,
    4: 0.40,
    3: 0.55,
    2: 0.75,
    1: 1.00,  # Peripheral
}


# ═══════════════════════════════════════════════════════════════════════════════
# SPIRAL GENERATOR (Center-Out Scan)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_spiral_indices(size: int, device: torch.device = None) -> torch.Tensor:
    """
    Generate spiral scan order from center outward.
    Returns tensor of (y, x) indices in scan order.
    
    This mimics foveal attention - center is processed first.
    """
    cx, cy = size // 2, size // 2
    
    indices = []
    visited = set()
    
    x, y = cx, cy
    indices.append((y, x))
    visited.add((x, y))
    
    step = 1
    while len(indices) < size * size:
        # Right
        for _ in range(step):
            x += 1
            if 0 <= x < size and 0 <= y < size and (x, y) not in visited:
                indices.append((y, x))
                visited.add((x, y))
        
        # Down
        for _ in range(step):
            y += 1
            if 0 <= x < size and 0 <= y < size and (x, y) not in visited:
                indices.append((y, x))
                visited.add((x, y))
        
        step += 1
        
        # Left
        for _ in range(step):
            x -= 1
            if 0 <= x < size and 0 <= y < size and (x, y) not in visited:
                indices.append((y, x))
                visited.add((x, y))
        
        # Up
        for _ in range(step):
            y -= 1
            if 0 <= x < size and 0 <= y < size and (x, y) not in visited:
                indices.append((y, x))
                visited.add((x, y))
        
        step += 1
    
    indices_tensor = torch.tensor(indices, dtype=torch.long)
    if device:
        indices_tensor = indices_tensor.to(device)
    
    return indices_tensor


def generate_fire_map(size: int, device: torch.device = None) -> torch.Tensor:
    """
    Generate fire zone map based on distance from center.
    
    Fire 6 (center) → Fire 1 (peripheral)
    Maps to K × φⁿ heat scaling.
    """
    cy, cx = size // 2, size // 2
    
    y_coords = torch.arange(size, dtype=torch.float32)
    x_coords = torch.arange(size, dtype=torch.float32)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
    
    distance = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    max_dist = math.sqrt(cx ** 2 + cy ** 2)
    norm_dist = distance / max_dist
    
    fire_map = torch.ones((size, size), dtype=torch.long)
    
    for fire_level in range(6, 0, -1):
        threshold = FIRE_ZONES[fire_level]
        fire_map[norm_dist <= threshold] = fire_level
    
    if device:
        fire_map = fire_map.to(device)
    
    return fire_map


# ═══════════════════════════════════════════════════════════════════════════════
# THREE-FRAME ENCODER
# ═══════════════════════════════════════════════════════════════════════════════

class ThreeFrameEncoder(nn.Module):
    """
    Encodes raw image into three-frame identity:
    
    Frame X: Color (what IS) - RGB values normalized
    Frame Y: Position (where IS) - normalized (dx, dy) from center  
    Frame Z: Heat (how much MATTERS) - computed salience
    
    Output: (B, H, W, 7) tensor
        - channels 0-2: Color (RGB normalized)
        - channels 3-4: Position (dx, dy normalized)
        - channels 5-6: Heat (raw salience, fire-weighted salience)
    """
    
    def __init__(self, resolution: int = 64):
        super().__init__()
        self.resolution = resolution
        self.center_x = resolution // 2
        self.center_y = resolution // 2
        
        # Precompute position field (static)
        self.register_buffer('position_field', self._create_position_field())
        
        # Precompute fire map (static)
        self.register_buffer('fire_map', generate_fire_map(resolution))
        
        # Learnable edge detection kernels
        self.edge_conv = nn.Conv2d(3, 2, kernel_size=3, padding=1, bias=False)
        
        # Initialize with Sobel-like kernels
        with torch.no_grad():
            # Horizontal edge
            self.edge_conv.weight[0, :, :, :] = torch.tensor([
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
            ]).float() / 8.0
            # Vertical edge
            self.edge_conv.weight[1, :, :, :] = torch.tensor([
                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
            ]).float() / 8.0
        
        # Previous frame for motion detection
        self.prev_frame = None
    
    def _create_position_field(self) -> torch.Tensor:
        """Create normalized position field (dx, dy from center)."""
        y_coords = torch.arange(self.resolution, dtype=torch.float32)
        x_coords = torch.arange(self.resolution, dtype=torch.float32)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        dx = (xx - self.center_x) / self.center_x  # -1 to 1
        dy = (yy - self.center_y) / self.center_y  # -1 to 1
        
        return torch.stack([dx, dy], dim=-1)  # (H, W, 2)
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Encode image to three-frame representation.
        
        Args:
            image: (B, 3, H, W) RGB image tensor, values 0-255 or 0-1
            
        Returns:
            (B, H, W, 7) three-frame tensor
        """
        B = image.shape[0]
        
        # Normalize to 0-1 if needed
        if image.max() > 1.0:
            image = image / 255.0
        
        # ─────────────────────────────────────────────────────────────────────
        # Frame X: COLOR (what IS)
        # ─────────────────────────────────────────────────────────────────────
        color = image.permute(0, 2, 3, 1)  # (B, H, W, 3)
        
        # ─────────────────────────────────────────────────────────────────────
        # Frame Y: POSITION (where IS)
        # ─────────────────────────────────────────────────────────────────────
        position = self.position_field.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, W, 2)
        
        # ─────────────────────────────────────────────────────────────────────
        # Frame Z: HEAT (how much MATTERS)
        # ─────────────────────────────────────────────────────────────────────
        heat = self._compute_heat(image)  # (B, H, W, 2)
        
        # Store for next frame's motion detection
        self.prev_frame = image.detach().clone()
        
        # Combine all frames
        three_frame = torch.cat([color, position, heat], dim=-1)  # (B, H, W, 7)
        
        return three_frame
    
    def _compute_heat(self, image: torch.Tensor) -> torch.Tensor:
        """
        Compute heat (salience) field.
        
        Heat sources:
        1. Center bias (fovea) - weighted by fire zone
        2. Edges (boundaries matter)
        3. Motion (change from previous frame)
        4. Color saturation (vivid = salient)
        
        Returns: (B, H, W, 2) - raw heat and fire-weighted heat
        """
        B = image.shape[0]
        device = image.device
        
        # Initialize heat
        heat = torch.zeros(B, self.resolution, self.resolution, device=device)
        
        # 1. Mild center bias via fire map (foveal hint, not dominant)
        fire_weight = self.fire_map.float() / 6.0  # Normalize to 0-1
        heat = heat + fire_weight.unsqueeze(0) * 0.05
        
        # 2. Edge detection
        edges = self.edge_conv(image)  # (B, 2, H, W)
        edge_magnitude = torch.sqrt(edges[:, 0] ** 2 + edges[:, 1] ** 2)
        edge_magnitude = edge_magnitude / (edge_magnitude.max() + 1e-6)
        heat = heat + edge_magnitude * 0.3
        
        # 3. Motion detection
        if self.prev_frame is not None and self.prev_frame.shape == image.shape:
            motion = torch.abs(image - self.prev_frame).mean(dim=1)  # (B, H, W)
            heat = heat + motion * 0.3
        
        # 4. Color saturation
        max_c = image.max(dim=1)[0]
        min_c = image.min(dim=1)[0]
        saturation = (max_c - min_c) / (max_c + 1e-6)
        heat = heat + saturation * 0.1
        
        # Clamp
        heat = torch.clamp(heat, 0, 1)
        
        # Fire-weighted heat (gentle center preference, not dominating)
        # Range: 0.67 (edge, zone 1) to 1.5 (center, zone 6)
        fire_weighted = heat * (self.fire_map.float() / 6.0 + 0.5).unsqueeze(0)
        
        # Stack raw and weighted
        return torch.stack([heat, fire_weighted], dim=-1)  # (B, H, W, 2)


# ═══════════════════════════════════════════════════════════════════════════════
# SPIRAL PATCH EMBEDDING
# ═══════════════════════════════════════════════════════════════════════════════

class SpiralPatchEmbedding(nn.Module):
    """
    Embeds the three-frame representation as patches in spiral order.
    
    Unlike standard ViT which uses raster order, we embed center-out
    to match foveal attention patterns.
    """
    
    def __init__(self, 
                 resolution: int = 64,
                 patch_size: int = 8,
                 embed_dim: int = 256,
                 in_channels: int = 7):  # 3 color + 2 position + 2 heat
        super().__init__()
        
        self.resolution = resolution
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (resolution // patch_size) ** 2
        self.grid_size = resolution // patch_size
        
        # Patch embedding projection
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )
        
        # Learnable position embedding (in spiral order)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )
        
        # Fire zone embedding (which fire zone each patch is in)
        self.fire_embed = nn.Embedding(7, embed_dim)  # 0-6 fire levels
        
        # Generate spiral order for patches
        self.register_buffer('spiral_order', self._create_patch_spiral())
        
        # Precompute patch fire zones
        self.register_buffer('patch_fire_zones', self._compute_patch_fire_zones())
        
        self._init_weights()
    
    def _init_weights(self):
        # Initialize position embedding with small values
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def _create_patch_spiral(self) -> torch.Tensor:
        """Create spiral order for patches (not pixels)."""
        spiral = generate_spiral_indices(self.grid_size)
        # Convert (y, x) to flat indices
        flat_indices = spiral[:, 0] * self.grid_size + spiral[:, 1]
        return flat_indices
    
    def _compute_patch_fire_zones(self) -> torch.Tensor:
        """Compute which fire zone each patch center falls into."""
        fire_map = generate_fire_map(self.resolution)
        
        # Sample at patch centers
        patch_fires = torch.zeros(self.num_patches, dtype=torch.long)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                cy = i * self.patch_size + self.patch_size // 2
                cx = j * self.patch_size + self.patch_size // 2
                patch_fires[i * self.grid_size + j] = fire_map[cy, cx]
        
        return patch_fires
    
    def forward(self, three_frame: torch.Tensor) -> torch.Tensor:
        """
        Embed three-frame as patches in spiral order.
        
        Args:
            three_frame: (B, H, W, 7) three-frame tensor
            
        Returns:
            (B, num_patches, embed_dim) embedded patches in spiral order
        """
        B = three_frame.shape[0]
        
        # Permute for conv2d: (B, H, W, C) → (B, C, H, W)
        x = three_frame.permute(0, 3, 1, 2)
        
        # Project to patches: (B, embed_dim, grid, grid)
        x = self.proj(x)
        
        # Flatten spatial: (B, embed_dim, num_patches)
        x = x.flatten(2)
        
        # Transpose: (B, num_patches, embed_dim)
        x = x.transpose(1, 2)
        
        # Reorder to spiral
        x = x[:, self.spiral_order, :]
        
        # Add position embedding (already in spiral order)
        x = x + self.pos_embed
        
        # Add fire zone embedding
        fire_embeds = self.fire_embed(self.patch_fire_zones)  # (num_patches, embed_dim)
        x = x + fire_embeds.unsqueeze(0)
        
        return x


# ═══════════════════════════════════════════════════════════════════════════════
# ATTENTION AS HEAT
# ═══════════════════════════════════════════════════════════════════════════════

class HeatAttention(nn.Module):
    """
    Multi-head attention where attention weights ARE the heat signal.
    
    The attention pattern directly represents what the system finds salient.
    No separate heat calculation needed - attention IS heat.
    
    Planck-grounded:
    - Uses φ-scaled attention temperature
    - Threshold ladder for attention gating
    """
    
    def __init__(self, 
                 embed_dim: int = 256,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Planck-grounded attention temperature
        # Using K as the base temperature
        self.register_buffer('attention_temp', torch.tensor(K))
        
        # Q, K, V projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # Store attention weights for heat extraction
        self.last_attention = None
    
    def forward(self, x: torch.Tensor, 
                focus_hint: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute attention (which IS the heat signal).
        
        Args:
            x: (B, N, D) input tensor
            focus_hint: Optional (B, N) tensor from Pi indicating where to focus
            
        Returns:
            (B, N, D) attended tensor
        """
        B, N, D = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Apply focus hint from Pi (if provided)
        if focus_hint is not None:
            # focus_hint: (B, N) → (B, 1, 1, N)
            focus_bias = focus_hint.unsqueeze(1).unsqueeze(2) * self.attention_temp
            attn = attn + focus_bias
        
        # Softmax with Planck temperature
        attn = F.softmax(attn / self.attention_temp, dim=-1)
        
        # Store for heat extraction
        self.last_attention = attn.detach()
        
        # Apply dropout
        attn = self.dropout(attn)
        
        # Attend to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        
        # Output projection
        x = self.out_proj(x)
        
        return x
    
    def get_heat(self) -> Optional[torch.Tensor]:
        """
        Get the heat signal (attention weights averaged across heads).
        
        Returns:
            (B, N, N) attention matrix or None if not computed
        """
        if self.last_attention is None:
            return None
        
        # Average across heads: (B, heads, N, N) → (B, N, N)
        return self.last_attention.mean(dim=1)


# ═══════════════════════════════════════════════════════════════════════════════
# TRANSFORMER BLOCK
# ═══════════════════════════════════════════════════════════════════════════════

class VisionTransformerBlock(nn.Module):
    """
    Single transformer block with HeatAttention.
    
    Architecture:
        x → LayerNorm → HeatAttention → + x
        x → LayerNorm → MLP → + x
    """
    
    def __init__(self,
                 embed_dim: int = 256,
                 num_heads: int = 8,
                 mlp_ratio: float = PHI,  # Use golden ratio!
                 dropout: float = 0.1):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = HeatAttention(embed_dim, num_heads, dropout)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # MLP with golden ratio expansion
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor, 
                focus_hint: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with optional focus hint."""
        x = x + self.attn(self.norm1(x), focus_hint)
        x = x + self.mlp(self.norm2(x))
        return x
    
    def get_heat(self) -> Optional[torch.Tensor]:
        """Get heat from attention layer."""
        return self.attn.get_heat()


# ═══════════════════════════════════════════════════════════════════════════════
# FULL VISION TRANSFORMER
# ═══════════════════════════════════════════════════════════════════════════════

class PBAIVisionTransformer(nn.Module):
    """
    PBAI's CUDA Vision Transformer.
    
    Three-frame identity → Spiral patches → Transformer → Compressed output
    
    No pretraining. Learns from heat/reward signals from Pi.
    Crash and burn til you don't.
    """
    
    def __init__(self,
                 resolution: int = 64,
                 patch_size: int = 8,
                 embed_dim: int = 256,
                 depth: int = 6,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        
        self.resolution = resolution
        self.patch_size = patch_size
        self.num_patches = (resolution // patch_size) ** 2
        
        # Three-frame encoder
        self.frame_encoder = ThreeFrameEncoder(resolution)
        
        # Spiral patch embedding
        self.patch_embed = SpiralPatchEmbedding(
            resolution, patch_size, embed_dim, in_channels=7
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            VisionTransformerBlock(embed_dim, num_heads, PHI, dropout)
            for _ in range(depth)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Output heads
        # 1. Heat head - predicts which patches matter
        self.heat_head = nn.Linear(embed_dim, 1)
        
        # 2. Feature head - compressed representation for Pi
        self.feature_head = nn.Linear(embed_dim, 32)  # Compress to 32-dim
        
        # Statistics
        self.t_K = 0
        self.total_heat = 0.0
    
    def forward(self, image: torch.Tensor,
                focus_hint: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Full forward pass.
        
        Args:
            image: (B, 3, H, W) or (B, H, W, 3) RGB image
            focus_hint: Optional (B, num_patches) focus weights from Pi
            
        Returns:
            Dict with:
                - 'patches': (B, N, D) patch embeddings
                - 'heat': (B, N) per-patch heat scores
                - 'features': (B, N, 32) compressed features
                - 'attention': (B, N, N) attention matrix from last layer
                - 'three_frame': (B, H, W, 7) three-frame representation
        """
        # Handle channel order
        if image.shape[-1] == 3:  # (B, H, W, 3)
            image = image.permute(0, 3, 1, 2)  # → (B, 3, H, W)
        
        # Resize if needed
        if image.shape[-1] != self.resolution:
            image = F.interpolate(image, size=(self.resolution, self.resolution), 
                                  mode='bilinear', align_corners=False)
        
        # Three-frame encoding
        three_frame = self.frame_encoder(image)
        
        # Spiral patch embedding
        x = self.patch_embed(three_frame)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, focus_hint)
        
        # Final norm
        x = self.norm(x)
        
        # Output heads
        heat = self.heat_head(x).squeeze(-1)  # (B, N)
        heat = torch.sigmoid(heat)  # 0-1 range
        
        features = self.feature_head(x)  # (B, N, 32)
        
        # Get attention from last block
        attention = self.blocks[-1].get_heat()
        
        # Update stats
        self.t_K += 1
        self.total_heat += heat.sum().item()
        
        return {
            'patches': x,
            'heat': heat,
            'features': features,
            'attention': attention,
            'three_frame': three_frame
        }
    
    def get_top_k_patches(self, output: Dict, k: int = 10) -> Dict:
        """
        Extract top-K patches by heat for sending to Pi.
        
        This is the compression step - only send what matters.
        """
        heat = output['heat']  # (B, N)
        features = output['features']  # (B, N, 32)
        
        B, N = heat.shape
        
        # Get top-K indices
        top_k_values, top_k_indices = torch.topk(heat, k=min(k, N), dim=1)
        
        # Gather features for top-K
        top_k_features = torch.gather(
            features, 1,
            top_k_indices.unsqueeze(-1).expand(-1, -1, features.shape[-1])
        )
        
        # Convert indices to (x, y) coordinates
        grid_size = self.resolution // self.patch_size
        spiral_order = self.patch_embed.spiral_order  # Flat indices in spiral order
        
        # Map back from spiral to grid positions
        top_k_positions = []
        for b in range(B):
            positions = []
            for idx in top_k_indices[b]:
                spiral_idx = spiral_order[idx].item()
                y = spiral_idx // grid_size
                x = spiral_idx % grid_size
                # Convert to pixel coordinates (patch center)
                px = x * self.patch_size + self.patch_size // 2
                py = y * self.patch_size + self.patch_size // 2
                positions.append((px, py))
            top_k_positions.append(positions)
        
        return {
            'indices': top_k_indices,
            'heat': top_k_values,
            'features': top_k_features,
            'positions': top_k_positions
        }


# ═══════════════════════════════════════════════════════════════════════════════
# HEAT BACKPROP - Learning from Outcomes
# ═══════════════════════════════════════════════════════════════════════════════

class HeatBackprop:
    """
    Learning mechanism that adjusts transformer based on outcomes.
    
    Good outcome → reinforce attention pattern
    Bad outcome → adjust attention pattern
    
    No labels, no pretraining. Pure reinforcement from heat signals.
    """
    
    def __init__(self, model: PBAIVisionTransformer, 
                 learning_rate: float = K * 1e-4):  # LR scaled by K
        self.model = model
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # Store recent outputs for credit assignment
        self.output_history = []
        self.max_history = 10
    
    def store_output(self, output: Dict):
        """Store output for later credit assignment."""
        # Detach but keep on GPU
        stored = {
            'heat': output['heat'].detach(),
            'patches': output['patches'].detach(),
        }
        self.output_history.append(stored)
        
        if len(self.output_history) > self.max_history:
            self.output_history.pop(0)
    
    def update(self, reward: float, current_output: Dict):
        """
        Update model based on reward signal.
        
        Args:
            reward: Heat reward from Pi (-1 to 1 range)
            current_output: Output from current forward pass
        """
        if not self.output_history:
            return
        
        self.optimizer.zero_grad()
        
        # Loss: encourage high heat on patches if reward positive,
        #       discourage if negative
        heat = current_output['heat']
        
        # Simple policy gradient-style loss
        # If reward > 0: maximize heat on attended patches
        # If reward < 0: minimize heat on attended patches
        loss = -reward * heat.mean()
        
        # Add entropy bonus to encourage exploration
        # (avoid collapsing to always-attend-nothing or always-attend-everything)
        entropy = -(heat * torch.log(heat + 1e-6) + 
                   (1 - heat) * torch.log(1 - heat + 1e-6)).mean()
        loss = loss - 0.01 * entropy
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        self.optimizer.step()
        
        return loss.item()


# ═══════════════════════════════════════════════════════════════════════════════
# WORLD STATE BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def build_world_state(model: PBAIVisionTransformer,
                      image: torch.Tensor,
                      focus_hint: Optional[torch.Tensor] = None,
                      top_k: int = 10,
                      output: Optional[Dict] = None) -> Dict:
    """
    Process image and build world state dict for Pi.

    This is what gets sent over WebSocket to the Pi.
    All values are converted to JSON-serializable Python types.
    """
    # Forward pass (skip if pre-computed output provided)
    if output is None:
        with torch.no_grad():
            output = model(image, focus_hint)
    
    # Get top-K patches
    top_k_data = model.get_top_k_patches(output, k=top_k)
    
    # Extract three-frame stats
    three_frame = output['three_frame']
    heat_field = three_frame[..., 5]  # Raw heat channel
    
    # Build world state
    B = image.shape[0]
    world_states = []
    
    for b in range(B):
        # Peaks (top-K patches)
        peaks = []
        for i in range(len(top_k_data['positions'][b])):
            px, py = top_k_data['positions'][b][i]
            heat_val = float(top_k_data['heat'][b, i].item())
            features = [float(f) for f in top_k_data['features'][b, i].cpu().numpy().tolist()]
            
            # Get color at this position from three_frame - ensure Python ints
            color_tensor = (three_frame[b, py, px, :3] * 255).int().cpu()
            color = [int(c) for c in color_tensor.numpy().tolist()]
            
            # Map pixel position to hypersphere angular coordinates
            # dx, dy normalized to [-1, 1] from center
            cx_half = model.resolution / 2
            dx = (px - cx_half) / cx_half  # -1 to 1
            dy = (py - cx_half) / cx_half  # -1 to 1
            dist = math.sqrt(dx * dx + dy * dy)
            angle = math.atan2(dy, dx)

            # theta: center (dist=0) → equator, edge (dist=1) → pole
            theta = math.pi / 2 + min(dist, 1.0) * math.pi / 2
            theta = min(theta, math.pi)
            # phi: angle in [0, 2π)
            phi = angle % (2 * math.pi)

            # Cube projection: X=Blue/Yellow, Y=Red/Green, Z=Past/Future
            cube_x = math.sin(theta) * math.cos(phi)
            cube_y = math.sin(theta) * math.sin(phi)
            cube_tau = math.cos(theta)

            peaks.append({
                'x': int(px),
                'y': int(py),
                'heat': round(heat_val, 4),
                'features': features,
                'color': color,
                'existence': 'actual' if heat_val > THRESHOLD_EXISTENCE else 'potential',
                # Hypersphere angular coordinates
                'theta': round(theta, 4),
                'phi': round(phi, 4),
                # Color Cube projection
                'cube_x': round(cube_x, 4),   # Blue(-1) / Yellow(+1)
                'cube_y': round(cube_y, 4),   # Red(-1) / Green(+1)
                'cube_tau': round(cube_tau, 4) # Past(-) / Future(+)
            })
        
        # Heat stats
        heat_np = heat_field[b].cpu().numpy()
        
        # Center point
        cy, cx = model.resolution // 2, model.resolution // 2
        center_heat = float(heat_field[b, cy, cx].item())
        center_color_tensor = (three_frame[b, cy, cx, :3] * 255).int().cpu()
        center_color = [int(c) for c in center_color_tensor.numpy().tolist()]
        
        # Motion delta (from frame encoder)
        if model.frame_encoder.prev_frame is not None:
            motion_delta = float(torch.abs(
                image[b] - model.frame_encoder.prev_frame[b]
            ).mean().item())
        else:
            motion_delta = 0.0
        
        # Kappa (accumulated attention energy)
        kappa = float(output['heat'][b].sum().item()) * K
        
        world_state = {
            't_K': int(model.t_K),
            'kappa': round(kappa, 4),
            'peaks': peaks,
            'center': {
                'x': int(cx),
                'y': int(cy),
                'heat': round(center_heat, 4),
                'color': center_color
            },
            'heat_stats': {
                'mean': round(float(heat_np.mean()), 4),
                'max': round(float(heat_np.max()), 4),
                'variance': round(float(heat_np.var()), 4)
            },
            'motion_delta': round(motion_delta, 4),
            'resolution': int(model.resolution),
            'thresholds': {
                'noise': round(float(THRESHOLD_NOISE), 4),
                'movement': round(float(THRESHOLD_MOVEMENT), 4),
                'existence': round(float(THRESHOLD_EXISTENCE), 4),
                'order': round(float(THRESHOLD_ORDER), 4),
                'righteous': round(float(THRESHOLD_RIGHTEOUS), 4)
            }
        }
        
        world_states.append(world_state)
    
    return world_states[0] if B == 1 else world_states


# ═══════════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════════

def test_vision_transformer():
    """Test the vision transformer."""
    print("=" * 60)
    print("PBAI CUDA VISION TRANSFORMER TEST")
    print("=" * 60)
    print()
    
    # Check CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()
    
    # Create model
    print("Creating model...")
    model = PBAIVisionTransformer(
        resolution=64,
        patch_size=8,
        embed_dim=256,
        depth=6,
        num_heads=8
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")
    print()
    
    # Test forward pass
    print("Testing forward pass...")
    dummy_image = torch.randn(1, 3, 64, 64).to(device)
    
    output = model(dummy_image)
    
    print(f"Patches shape: {output['patches'].shape}")
    print(f"Heat shape: {output['heat'].shape}")
    print(f"Features shape: {output['features'].shape}")
    print(f"Three-frame shape: {output['three_frame'].shape}")
    print()
    
    # Test top-K extraction
    print("Testing top-K extraction...")
    top_k = model.get_top_k_patches(output, k=5)
    print(f"Top-5 heat values: {top_k['heat'][0].cpu().numpy()}")
    print(f"Top-5 positions: {top_k['positions'][0]}")
    print()
    
    # Test world state building
    print("Testing world state builder...")
    world_state = build_world_state(model, dummy_image, top_k=5)
    print(f"World state keys: {list(world_state.keys())}")
    print(f"t_K: {world_state['t_K']}")
    print(f"kappa: {world_state['kappa']}")
    print(f"Peaks: {len(world_state['peaks'])}")
    print()
    
    # Test heat backprop
    print("Testing heat backprop...")
    learner = HeatBackprop(model)
    
    # Simulate a few learning steps
    for i in range(3):
        output = model(dummy_image)
        learner.store_output(output)
        
        # Fake reward
        reward = 0.5 if i % 2 == 0 else -0.3
        loss = learner.update(reward, output)
        print(f"  Step {i+1}: reward={reward:.1f}, loss={loss:.4f}")
    
    print()
    print("=" * 60)
    print("TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    test_vision_transformer()
