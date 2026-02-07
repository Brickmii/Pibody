"""
PBAI VisionStep Engine (Planck-Grounded)

Connects the visual scanner to the manifold.
Converts scan frames into manifold-compatible structures.

PIPELINE:
    Image → Scanner → ScanFrame → VisionStep → Manifold Nodes

════════════════════════════════════════════════════════════════════════════════
PLANCK GROUNDING
════════════════════════════════════════════════════════════════════════════════

CLOCK SYNC:
    Each VisionStep is timestamped with t_K (heat-time).
    Allows temporal correlation of visual events.

ROBINSON CONSTRAINTS:
    Visual features have constraint types:
    - position (x, y): successor constraint (spatial)
    - angle: addition constraint (angular/temporal)
    - heat, color: multiplication constraint (quantitative)

BODY TEMPERATURE REFERENCE:
    - Kappa accumulation scaled to K
    - Heat values in K units (thermal quantum)

THRESHOLD LADDER (1/φⁿ):
    φ = 1.618033988749895
    
    THRESHOLD_NOISE     = 1/φ⁵ ≈ 0.090  (below = noise)
    THRESHOLD_MOVEMENT  = 1/φ⁴ ≈ 0.146  (motion detection)
    THRESHOLD_EXISTENCE = 1/φ³ ≈ 0.236  (persistence gate)
    THRESHOLD_ORDER     = 1/φ² ≈ 0.382  (structure detection)
    THRESHOLD_RIGHTEOUS = 1/φ  ≈ 0.618  (alignment threshold)

════════════════════════════════════════════════════════════════════════════════

VISION STEP OUTPUTS:
    kappa:              Accumulated heat delta (motion energy)
    righteous_alignment: Distance from center (0 = perfect)
    proper_properties:   Color/position at peaks
    heat_peaks:         Candidate nodes
    persistence_flags:  What survived existence gate
    t_K:                Heat-time timestamp
"""

import numpy as np
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from time import time

# Try to import from core
try:
    from core.node_constants import (
        K, PHI,
        BODY_TEMPERATURE,
        ROBINSON_CONSTRAINTS,
    )
except ImportError:
    PHI = 1.618033988749895
    K = 4 / (PHI ** 2)  # ≈ 1.528
    BODY_TEMPERATURE = K * PHI ** 11
    ROBINSON_CONSTRAINTS = {
        'identity': 1.0,
        'successor': 4 * PHI / 7,
        'addition': 4/3,
        'multiplication': 13/10,
    }

# Threshold ladder based on 1/φⁿ
THRESHOLD_NOISE = 1 / (PHI ** 5)      # ≈ 0.090
THRESHOLD_MOVEMENT = 1 / (PHI ** 4)   # ≈ 0.146
THRESHOLD_EXISTENCE = 1 / (PHI ** 3)  # ≈ 0.236
THRESHOLD_ORDER = 1 / (PHI ** 2)      # ≈ 0.382
THRESHOLD_RIGHTEOUS = 1 / PHI         # ≈ 0.618


# ═══════════════════════════════════════════════════════════════════════════════
# VISUAL FEATURE - A point that might become a node (Planck-Grounded)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class VisualFeature:
    """
    A visual feature extracted from the scan.
    Candidate for manifold node creation.
    
    PLANCK GROUNDING:
        - constraint_type: Robinson constraint for this feature
        - heat scaled to K units
        - position uses successor constraint
    """
    # Position in scan space (successor constraint)
    x: int
    y: int
    
    # Properties (proper frame data)
    color: Tuple[int, int, int]  # multiplication constraint (quantitative)
    heat: float                   # multiplication constraint
    
    # Derived values
    distance_from_center: float  # For righteousness (successor)
    angle: float                 # For ordering (addition)
    
    # Persistence tracking
    first_seen: float = field(default_factory=time)
    last_seen: float = field(default_factory=time)
    times_seen: int = 1
    
    # State
    existence: str = "potential"  # potential → actual (after persistence gate)
    
    # Planck grounding
    constraint_type: str = "successor"  # spatial features are successor by default
    t_K_first: int = 0  # t_K when first seen
    t_K_last: int = 0   # t_K when last seen
    
    @property
    def righteous_alignment(self) -> float:
        """
        Righteousness = how close to center (0 = perfect).
        Inverted distance normalized to 0-1.
        Scaled by successor constraint (spatial precision).
        """
        # Assume max distance is ~45 for 64x64 grid (diagonal to corner)
        max_dist = 45.0
        raw_alignment = min(1.0, self.distance_from_center / max_dist)
        # Scale by successor constraint (tighter for spatial)
        return raw_alignment * ROBINSON_CONSTRAINTS['successor']
    
    @property
    def age(self) -> float:
        """How long this feature has been tracked."""
        return self.last_seen - self.first_seen
    
    @property
    def age_t_K(self) -> int:
        """Age in heat-time units."""
        return self.t_K_last - self.t_K_first
    
    @property
    def persistence(self) -> float:
        """Persistence score based on times seen and age."""
        if self.age < 0.1:
            return 0.0
        return min(1.0, self.times_seen / 5.0)  # Need 5 sightings for full persistence
    
    def update(self, heat: float, color: Tuple[int, int, int], t_K: int = 0):
        """Update feature with new observation."""
        self.heat = heat
        self.color = color
        self.last_seen = time()
        self.t_K_last = t_K
        self.times_seen += 1
        
        # Existence gate
        if self.persistence >= THRESHOLD_EXISTENCE:
            self.existence = "actual"
    
    def to_node_data(self) -> dict:
        """Convert to data suitable for Node creation.

        Maps scan-space (x, y, angle) to hypersphere angular coordinates:
            theta: distance_from_center mapped to [0, π] (center→equator, edge→poles)
            phi:   angle in scan space mapped to [0, 2π)
        """
        # Map distance_from_center to theta:
        #   center (dist=0) → equator (theta=π/2, present moment)
        #   edge (dist=max) → poles (theta→0 or π, past/future)
        max_dist = 45.0  # diagonal of 64x64 grid
        normalized_dist = min(1.0, self.distance_from_center / max_dist)
        theta = math.pi / 2 + (normalized_dist * math.pi / 2)  # equator → south pole
        theta = min(theta, math.pi)

        # Map angle directly to phi [0, 2π)
        phi = self.angle % (2 * math.pi)

        return {
            "concept": f"visual_{self.x}_{self.y}",
            "theta": theta,
            "phi": phi,
            "radius": 1.0,
            "heat": self.heat * K,  # Scale to K units
            "righteousness": self.righteous_alignment,
            "existence": self.existence,
            "constraint_type": self.constraint_type,
            "properties": {
                "color_r": self.color[0],
                "color_g": self.color[1],
                "color_b": self.color[2],
                "scan_x": self.x,
                "scan_y": self.y,
                "angle": self.angle,
                "t_K_first": self.t_K_first,
                "t_K_last": self.t_K_last,
            }
        }


# ═══════════════════════════════════════════════════════════════════════════════
# VISION STEP - Output of one scan cycle (Planck-Grounded)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class VisionStep:
    """
    Output of one visual scan cycle.
    Ready to feed into manifold.
    
    PLANCK GROUNDING:
        - t_K: Heat-time timestamp (clock sync)
        - kappa: Accumulated heat in K units
        - Features have Robinson constraint types
    """
    # Heat/motion
    kappa: float                          # Total accumulated heat delta
    delta_motion: float                   # Motion since last frame
    
    # Righteousness
    righteous_point: Optional[VisualFeature]  # Center point
    righteous_alignment: float            # Overall alignment score
    
    # Features
    heat_peaks: List[VisualFeature]       # High-heat points (candidates)
    proper_properties: Dict[str, any]     # Extracted properties
    
    # Persistence
    persistence_flags: Dict[str, bool]    # Which features passed existence gate
    new_actuals: List[VisualFeature]      # Features that just became actual
    
    # Metadata
    timestamp: float = field(default_factory=time)
    frame_number: int = 0
    
    # Planck grounding - heat-time timestamp
    t_K: int = 0
    
    def get_actual_features(self) -> List[VisualFeature]:
        """Get features that have passed existence gate."""
        return [f for f in self.heat_peaks if f.existence == "actual"]
    
    def get_motion_detected(self) -> bool:
        """Check if motion exceeds threshold."""
        return self.delta_motion >= THRESHOLD_MOVEMENT
    
    def get_ordered_peaks(self) -> List[VisualFeature]:
        """Get peaks ordered by angle (spatial structure)."""
        return sorted(self.heat_peaks, key=lambda f: f.angle)
    
    def to_perception_dict(self) -> dict:
        """Convert to dict suitable for EnvironmentCore Perception."""
        return {
            "t_K": self.t_K,
            "kappa": self.kappa,
            "motion_delta": self.delta_motion,
            "righteousness": self.righteous_alignment,
            "peaks": [
                {
                    "x": f.x,
                    "y": f.y,
                    "heat": f.heat,
                    "color": list(f.color),
                    "constraint_type": f.constraint_type,
                }
                for f in self.heat_peaks[:10]
            ],
            "center": {
                "x": self.righteous_point.x if self.righteous_point else 32,
                "y": self.righteous_point.y if self.righteous_point else 32,
                "heat": self.righteous_point.heat if self.righteous_point else 0,
                "color": list(self.righteous_point.color) if self.righteous_point else [128, 128, 128],
            },
            "heat_stats": {
                "mean": self.proper_properties.get("mean_heat", 0),
                "max": self.proper_properties.get("max_heat", 0),
            },
            "resolution": 64,  # Default
        }


# ═══════════════════════════════════════════════════════════════════════════════
# VISION STEP ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class VisionStepEngine:
    """
    Converts scan frames into VisionSteps for the manifold.
    
    Maintains:
    - Feature tracking (persistence)
    - Motion accumulation (kappa)
    - Existence gating
    """
    
    def __init__(self, resolution: int = 64):
        self.resolution = resolution
        self.center_x = resolution // 2
        self.center_y = resolution // 2
        
        # Feature tracking
        self.tracked_features: Dict[Tuple[int, int], VisualFeature] = {}
        
        # State
        self.prev_heat_field: Optional[np.ndarray] = None
        self.accumulated_kappa: float = 0.0
        self.frame_count: int = 0
        
        # Statistics
        self.total_features_created: int = 0
        self.total_actuals: int = 0
    
    def process_frame(self, 
                      color_field: np.ndarray,
                      position_field: np.ndarray,
                      heat_field: np.ndarray) -> VisionStep:
        """
        Process a scan frame and produce a VisionStep.
        
        Args:
            color_field: (H, W, 3) RGB values
            position_field: (H, W, 2) normalized x, y from center
            heat_field: (H, W) salience values
            
        Returns:
            VisionStep ready for manifold integration
        """
        self.frame_count += 1
        
        # ─────────────────────────────────────────────────────────────────────
        # 1. CALCULATE MOTION (delta from previous frame)
        # ─────────────────────────────────────────────────────────────────────
        
        if self.prev_heat_field is not None:
            delta = np.abs(heat_field - self.prev_heat_field)
            delta_motion = float(np.mean(delta))
        else:
            delta_motion = 0.0
        
        self.prev_heat_field = heat_field.copy()
        
        # ─────────────────────────────────────────────────────────────────────
        # 2. ACCUMULATE KAPPA (heat budget)
        # ─────────────────────────────────────────────────────────────────────
        
        # Kappa accumulates from motion above threshold
        if delta_motion >= THRESHOLD_MOVEMENT:
            self.accumulated_kappa += delta_motion * K
        
        # ─────────────────────────────────────────────────────────────────────
        # 3. EXTRACT RIGHTEOUS POINT (center)
        # ─────────────────────────────────────────────────────────────────────
        
        center_color = tuple(color_field[self.center_y, self.center_x])
        center_heat = float(heat_field[self.center_y, self.center_x])
        
        righteous_point = VisualFeature(
            x=self.center_x,
            y=self.center_y,
            color=center_color,
            heat=center_heat,
            distance_from_center=0.0,
            angle=0.0,
            existence="actual"  # Center is always actual
        )
        
        # ─────────────────────────────────────────────────────────────────────
        # 4. EXTRACT HEAT PEAKS (above order threshold)
        # ─────────────────────────────────────────────────────────────────────
        
        heat_peaks = []
        new_actuals = []
        persistence_flags = {}
        
        # Find all points above noise threshold
        peak_coords = np.argwhere(heat_field >= THRESHOLD_NOISE)
        
        for (y, x) in peak_coords:
            heat = float(heat_field[y, x])
            
            # Skip if below order threshold for peak consideration
            if heat < THRESHOLD_ORDER:
                continue
            
            color = tuple(color_field[y, x])
            dx = x - self.center_x
            dy = y - self.center_y
            distance = math.sqrt(dx**2 + dy**2)
            angle = math.atan2(dy, dx)
            
            pos = (x, y)
            
            # Track or update feature
            if pos in self.tracked_features:
                feature = self.tracked_features[pos]
                was_potential = feature.existence == "potential"
                feature.update(heat, color)
                
                if was_potential and feature.existence == "actual":
                    new_actuals.append(feature)
                    self.total_actuals += 1
            else:
                feature = VisualFeature(
                    x=x, y=y,
                    color=color,
                    heat=heat,
                    distance_from_center=distance,
                    angle=angle
                )
                self.tracked_features[pos] = feature
                self.total_features_created += 1
            
            heat_peaks.append(feature)
            persistence_flags[f"{x},{y}"] = feature.existence == "actual"
        
        # Sort peaks by heat (highest first)
        heat_peaks.sort(key=lambda f: f.heat, reverse=True)
        
        # ─────────────────────────────────────────────────────────────────────
        # 5. CALCULATE OVERALL RIGHTEOUSNESS
        # ─────────────────────────────────────────────────────────────────────
        
        if heat_peaks:
            # Weighted average of peak alignments
            total_heat = sum(f.heat for f in heat_peaks)
            if total_heat > 0:
                weighted_alignment = sum(f.righteous_alignment * f.heat for f in heat_peaks) / total_heat
            else:
                weighted_alignment = 1.0  # No peaks = maximally unaligned
        else:
            weighted_alignment = 0.0  # No peaks but center exists
        
        # ─────────────────────────────────────────────────────────────────────
        # 6. EXTRACT PROPER PROPERTIES
        # ─────────────────────────────────────────────────────────────────────
        
        proper_properties = {
            "mean_heat": float(np.mean(heat_field)),
            "max_heat": float(np.max(heat_field)),
            "peak_count": len(heat_peaks),
            "actual_count": len([f for f in heat_peaks if f.existence == "actual"]),
            "center_color": center_color,
            "center_heat": center_heat,
            "dominant_angle": heat_peaks[0].angle if heat_peaks else 0.0,
        }
        
        # ─────────────────────────────────────────────────────────────────────
        # 7. BUILD VISION STEP
        # ─────────────────────────────────────────────────────────────────────
        
        return VisionStep(
            kappa=self.accumulated_kappa,
            delta_motion=delta_motion,
            righteous_point=righteous_point,
            righteous_alignment=weighted_alignment,
            heat_peaks=heat_peaks[:20],  # Top 20 peaks
            proper_properties=proper_properties,
            persistence_flags=persistence_flags,
            new_actuals=new_actuals,
            frame_number=self.frame_count
        )
    
    def prune_stale_features(self, max_age: float = 5.0):
        """Remove features not seen recently."""
        now = time()
        stale = [pos for pos, f in self.tracked_features.items() 
                 if now - f.last_seen > max_age]
        for pos in stale:
            del self.tracked_features[pos]
    
    def get_actual_nodes(self) -> List[dict]:
        """Get all actual features as node data."""
        return [f.to_node_data() for f in self.tracked_features.values()
                if f.existence == "actual"]
    
    def reset_kappa(self):
        """Reset accumulated kappa (after action)."""
        self.accumulated_kappa = 0.0
    
    def get_stats(self) -> dict:
        """Get engine statistics."""
        return {
            "frames_processed": self.frame_count,
            "tracked_features": len(self.tracked_features),
            "actual_features": len([f for f in self.tracked_features.values() 
                                    if f.existence == "actual"]),
            "total_created": self.total_features_created,
            "total_actuals": self.total_actuals,
            "accumulated_kappa": self.accumulated_kappa,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION: Scanner + VisionStep → Manifold
# ═══════════════════════════════════════════════════════════════════════════════

def integrate_vision_step(manifold, vision_step: VisionStep) -> int:
    """
    Integrate a VisionStep into the manifold.
    
    Creates/updates nodes for actual features.
    Returns number of nodes affected.
    
    Args:
        manifold: The PBAI manifold
        vision_step: Output from VisionStepEngine
        
    Returns:
        Number of nodes created or updated
    """
    affected = 0
    
    for feature in vision_step.new_actuals:
        node_data = feature.to_node_data()
        
        # Check if node already exists
        existing = manifold.get_node_by_concept(node_data["concept"])
        
        if existing:
            # Update heat
            existing.add_heat(node_data["heat"])
            affected += 1
        else:
            # Create new node with angular coordinates
            try:
                from core.nodes import Node
                node = Node(
                    concept=node_data["concept"],
                    theta=node_data["theta"],
                    phi=node_data["phi"],
                    radius=node_data["radius"],
                    heat=node_data["heat"],
                    righteousness=node_data["righteousness"],
                    existence=node_data["existence"]
                )
                manifold.add_node(node)
                affected += 1
            except Exception as e:
                print(f"Could not create node: {e}")
    
    return affected


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN - Test the pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Test the VisionStep engine."""
    print("=" * 60)
    print("PBAI VisionStep Engine Test")
    print("=" * 60)
    print(f"K = {K:.6f}")
    print(f"φ = {PHI:.6f}")
    print()
    print("Thresholds (1/φⁿ ladder):")
    print(f"  NOISE:     {THRESHOLD_NOISE:.3f}")
    print(f"  MOVEMENT:  {THRESHOLD_MOVEMENT:.3f}")
    print(f"  EXISTENCE: {THRESHOLD_EXISTENCE:.3f}")
    print(f"  ORDER:     {THRESHOLD_ORDER:.3f}")
    print(f"  RIGHTEOUS: {THRESHOLD_RIGHTEOUS:.3f}")
    print()
    
    # Create test data
    resolution = 64
    engine = VisionStepEngine(resolution=resolution)
    
    # Simulate several frames
    for frame_num in range(10):
        # Generate synthetic scan data
        color_field = np.random.randint(0, 255, (resolution, resolution, 3), dtype=np.uint8)
        position_field = np.zeros((resolution, resolution, 2), dtype=np.float32)
        heat_field = np.random.random((resolution, resolution)).astype(np.float32) * 0.5
        
        # Add some hot spots
        heat_field[32, 32] = 0.9  # Center
        heat_field[20, 40] = 0.7 + frame_num * 0.02  # Persistent peak
        heat_field[45, 15] = 0.6  # Another peak
        
        # Add motion to one spot
        if frame_num > 3:
            heat_field[20, 40] = 0.8
        
        # Process
        step = engine.process_frame(color_field, position_field, heat_field)
        
        print(f"\nFrame {step.frame_number}:")
        print(f"  Motion: {step.delta_motion:.4f} (detected: {step.get_motion_detected()})")
        print(f"  Kappa:  {step.kappa:.4f}")
        print(f"  Peaks:  {len(step.heat_peaks)} (actual: {len(step.get_actual_features())})")
        print(f"  Righteousness: {step.righteous_alignment:.3f}")
        print(f"  New actuals: {len(step.new_actuals)}")
    
    print()
    print("=" * 60)
    print("Final Statistics:")
    print("=" * 60)
    stats = engine.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    print()
    print("Actual nodes ready for manifold:")
    for node_data in engine.get_actual_nodes()[:5]:
        print(f"  {node_data['concept']}: heat={node_data['heat']:.3f}, R={node_data['righteousness']:.3f}")


if __name__ == "__main__":
    main()
