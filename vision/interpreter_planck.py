"""
PBAI World State Interpreter (Planck-Grounded)

Interprets structured world state from PC into semantic perception.
Runs on Pi - lightweight, no heavy vision models needed.

════════════════════════════════════════════════════════════════════════════════
PLANCK GROUNDING
════════════════════════════════════════════════════════════════════════════════

CLOCK SYNC:
    WorldPerception includes t_K for heat-time synchronization.
    All interpretations are grounded in the manifold's time reference.

ROBINSON CONSTRAINTS:
    Different perception types have different constraint types:
    - motion: addition constraint (temporal)
    - position/attention: successor constraint (spatial)
    - heat/kappa: multiplication constraint (quantitative)
    - objects: identity constraint (categorical)

BODY TEMPERATURE REFERENCE:
    - Kappa scaled to K units
    - High kappa (> K) indicates action potential

════════════════════════════════════════════════════════════════════════════════

USAGE:
    from vision import WorldPerception, create_interpreter
    
    interpreter = create_interpreter()
    perception = interpreter.interpret(world_state)
    
    # Access data
    print(perception.motion_level)   # "still", "moving", "fast"
    print(perception.attention_focus)  # "center", "left", "right"
    print(perception.kappa)          # Heat budget
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# Import constants from core if available
try:
    from core.node_constants import (
        K, PHI,
        BODY_TEMPERATURE,
        ROBINSON_CONSTRAINTS,
    )
except ImportError:
    PHI = 1.618033988749895
    K = 4 / (PHI ** 2)
    BODY_TEMPERATURE = K * PHI ** 11
    ROBINSON_CONSTRAINTS = {
        'identity': 1.0,
        'successor': 4 * PHI / 7,
        'addition': 4/3,
        'multiplication': 13/10,
    }

# Thresholds
THRESHOLD_MOVEMENT = 1 / (PHI ** 4)   # ≈ 0.146


@dataclass
class WorldPerception:
    """
    Structured perception from world model state (Planck-Grounded).
    
    This is what PBAI drivers use to make decisions.
    
    PLANCK GROUNDING:
        - t_K: Heat-time timestamp for clock sync
        - kappa: Heat budget in K units
        - constraint_types: Robinson constraints for each property
    """
    # Time/energy (Planck-grounded)
    t_K: int = 0
    kappa: float = 0.0
    
    # Motion (addition constraint - temporal)
    motion_delta: float = 0.0
    motion_level: str = "still"  # still, moving, fast
    
    # Attention (successor constraint - spatial)
    righteousness: float = 0.0
    attention_focus: str = "center"  # center, left, right, top, bottom
    
    # Peaks (salient points)
    peaks: List[Dict] = field(default_factory=list)
    center: Dict = field(default_factory=dict)
    
    # Heat statistics (multiplication constraint - quantitative)
    mean_heat: float = 0.0
    max_heat: float = 0.0
    
    # Detected objects (identity constraint - categorical)
    objects: List[Dict] = field(default_factory=list)
    
    # History
    recent_actions: List[str] = field(default_factory=list)
    recent_deltas: List[float] = field(default_factory=list)
    
    # Scene analysis
    scene_complexity: str = "simple"  # simple, moderate, complex
    
    # Summary
    summary: str = ""
    
    # Raw world state
    raw: Dict = field(default_factory=dict)
    
    # Planck grounding - constraint types for properties
    constraint_types: Dict[str, str] = field(default_factory=lambda: {
        "motion_delta": "addition",
        "attention_focus": "successor",
        "mean_heat": "multiplication",
        "max_heat": "multiplication",
        "kappa": "multiplication",
        "objects": "identity",
    })
    
    @classmethod
    def from_world_state(cls, ws: dict) -> 'WorldPerception':
        """
        Parse world state dict from PC into perception.
        
        This is the main entry point for drivers.
        """
        p = cls(raw=ws)
        
        # Basic fields
        p.t_K = ws.get("t_K", 0)
        p.kappa = ws.get("kappa", 0.0)
        p.motion_delta = ws.get("motion_delta", 0.0)
        p.righteousness = ws.get("righteousness", 0.0)
        
        # Peaks and center
        p.peaks = ws.get("peaks", [])
        p.center = ws.get("center", {})
        
        # Heat stats
        stats = ws.get("heat_stats", {})
        p.mean_heat = stats.get("mean", 0.0)
        p.max_heat = stats.get("max", 0.0)
        
        # History
        p.recent_actions = [a.get("action", "") for a in ws.get("recent_actions", [])]
        p.recent_deltas = ws.get("recent_deltas", [])
        
        # Interpret motion level (addition constraint - temporal)
        if p.motion_delta < 0.05:
            p.motion_level = "still"
        elif p.motion_delta < 0.15:
            p.motion_level = "moving"
        else:
            p.motion_level = "fast"
        
        # Interpret attention focus (successor constraint - spatial)
        if p.peaks:
            top = p.peaks[0]
            res = ws.get("resolution", 64)
            cx = res // 2
            cy = res // 2
            x = top.get("x", cx)
            y = top.get("y", cy)
            
            dx = x - cx
            dy = y - cy
            
            if abs(dx) < res // 4 and abs(dy) < res // 4:
                p.attention_focus = "center"
            elif abs(dx) > abs(dy):
                p.attention_focus = "left" if dx < 0 else "right"
            else:
                p.attention_focus = "top" if dy < 0 else "bottom"
        
        # Scene complexity
        if len(p.peaks) <= 3:
            p.scene_complexity = "simple"
        elif len(p.peaks) <= 7:
            p.scene_complexity = "moderate"
        else:
            p.scene_complexity = "complex"
        
        # Build summary
        p.summary = f"{p.motion_level} scene, attention {p.attention_focus}, κ={p.kappa:.1f} @ t_K={p.t_K}"
        
        return p
    
    @classmethod
    def from_vision_step(cls, step) -> 'WorldPerception':
        """
        Create WorldPerception from a VisionStep.
        
        This bridges the visual cortex to the interpreter.
        """
        ws = step.to_perception_dict()
        return cls.from_world_state(ws)
    
    def can_act(self) -> bool:
        """Check if enough kappa for action (body temp grounded)."""
        return self.kappa >= K
    
    def to_dict(self) -> dict:
        """Convert to dict for logging/serialization."""
        return {
            "t_K": self.t_K,
            "kappa": self.kappa,
            "motion_level": self.motion_level,
            "attention_focus": self.attention_focus,
            "righteousness": self.righteousness,
            "scene_complexity": self.scene_complexity,
            "num_peaks": len(self.peaks),
            "summary": self.summary,
            "constraint_types": self.constraint_types,
        }


class Interpreter:
    """
    Interprets world state into perception.
    
    Currently rule-based. Can be extended with small Qwen for
    more sophisticated interpretation.
    """
    
    def __init__(self, use_qwen: bool = False):
        self.use_qwen = use_qwen
        self._qwen_loaded = False
    
    def interpret(self, world_state: dict) -> WorldPerception:
        """
        Interpret world state into perception.
        
        Args:
            world_state: Dict from PC world model server
            
        Returns:
            WorldPerception with semantic interpretation
        """
        # Parse basic perception
        perception = WorldPerception.from_world_state(world_state)
        
        # Classify objects by color
        perception.objects = self._classify_peaks(perception.peaks)
        
        # Optional: Enhance with Qwen
        if self.use_qwen:
            perception = self._qwen_enhance(world_state, perception)
        
        return perception
    
    def _classify_peaks(self, peaks: List[Dict]) -> List[Dict]:
        """
        Classify peaks by color into object types.
        
        Simple heuristic - can be improved per game.
        """
        objects = []
        
        for peak in peaks[:5]:  # Top 5 only
            color = peak.get("color", [128, 128, 128])
            heat = peak.get("heat", 0.0)
            x = peak.get("x", 0)
            y = peak.get("y", 0)
            
            if len(color) < 3:
                continue
                
            r, g, b = color[0], color[1], color[2]
            
            obj = {
                "x": x,
                "y": y,
                "heat": heat,
                "color": color,
                "type": "unknown"
            }
            
            # Green-dominant = vegetation
            if g > r + 30 and g > b + 30:
                obj["type"] = "vegetation"
            
            # Blue-dominant = sky/water
            elif b > r + 30 and b > g:
                obj["type"] = "sky_water"
            
            # Brown (r > g > b) = earth/wood
            elif r > g > b and r - b < 100:
                obj["type"] = "earth_wood"
            
            # Red-dominant = danger
            elif r > g + 50 and r > b + 50:
                obj["type"] = "danger"
            
            # Bright = highlight/UI
            elif r > 200 and g > 200 and b > 200:
                obj["type"] = "highlight"
            
            # Dark = shadow/cave
            elif r < 50 and g < 50 and b < 50:
                obj["type"] = "shadow"
            
            # Gray = stone/metal
            elif abs(r - g) < 30 and abs(g - b) < 30:
                obj["type"] = "stone_metal"
            
            objects.append(obj)
        
        return objects
    
    def _qwen_enhance(self, ws: dict, base: WorldPerception) -> WorldPerception:
        """
        Enhance perception with small Qwen (optional).
        
        Requires: pip install transformers torch
        Uses: Qwen2-0.5B-Instruct (small enough for Pi)
        """
        if not self._qwen_loaded:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                import torch
                
                logger.info("Loading Qwen2-0.5B...")
                self._tokenizer = AutoTokenizer.from_pretrained(
                    "Qwen/Qwen2-0.5B-Instruct",
                    trust_remote_code=True
                )
                self._model = AutoModelForCausalLM.from_pretrained(
                    "Qwen/Qwen2-0.5B-Instruct",
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
                self._qwen_loaded = True
                logger.info("Qwen loaded")
            except Exception as e:
                logger.warning(f"Could not load Qwen: {e}")
                self.use_qwen = False
                return base
        
        # Build prompt
        prompt = f"""Visual scan data:
- Motion: {base.motion_level}
- Attention: {base.attention_focus}  
- Objects: {[o['type'] for o in base.objects]}

Describe in one sentence:"""
        
        try:
            inputs = self._tokenizer(prompt, return_tensors="pt")
            inputs = inputs.to(self._model.device)
            
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False
                )
            
            response = self._tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).strip()
            
            base.summary = response[:100]
        except Exception as e:
            logger.warning(f"Qwen inference failed: {e}")
        
        return base


def create_interpreter(use_qwen: bool = False) -> Interpreter:
    """
    Create an interpreter.
    
    Args:
        use_qwen: If True, use small Qwen for enhanced interpretation
                  (requires transformers + torch, slower but smarter)
    """
    return Interpreter(use_qwen=use_qwen)


# Test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test world state
    test_ws = {
        "t_K": 100,
        "kappa": 3.5,
        "peaks": [
            {"x": 32, "y": 30, "heat": 0.8, "color": [100, 200, 100]},
            {"x": 45, "y": 32, "heat": 0.6, "color": [139, 90, 43]},
            {"x": 20, "y": 40, "heat": 0.5, "color": [135, 206, 235]},
        ],
        "center": {"x": 32, "y": 32, "heat": 0.5, "color": [128, 128, 128]},
        "heat_stats": {"mean": 0.4, "max": 0.8, "variance": 0.05},
        "motion_delta": 0.12,
        "righteousness": 0.3,
        "recent_actions": [],
        "recent_deltas": [0.1, 0.15, 0.08],
        "resolution": 64
    }
    
    interpreter = create_interpreter(use_qwen=False)
    perception = interpreter.interpret(test_ws)
    
    print("World Perception:")
    print(f"  t_K: {perception.t_K}")
    print(f"  κ: {perception.kappa}")
    print(f"  Motion: {perception.motion_level}")
    print(f"  Attention: {perception.attention_focus}")
    print(f"  Objects: {[o['type'] for o in perception.objects]}")
    print(f"  Summary: {perception.summary}")
