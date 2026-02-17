"""
PBAI Visual Cortex - Complete Vision Pipeline (Planck-Grounded)

Combines:
    - scan_simulator.py: Image → ScanFrame (color, position, heat)
    - vision_step.py: ScanFrame → VisionStep (kappa, features, persistence)
    - Manifold integration: VisionStep → Nodes

════════════════════════════════════════════════════════════════════════════════
PLANCK GROUNDING
════════════════════════════════════════════════════════════════════════════════

CLOCK SYNC:
    Visual Cortex synchronizes with EnvironmentCore's t_K:
    - Each process_* call can trigger a clock tick
    - VisionSteps are timestamped with t_K
    - Heat accumulation (kappa) is grounded in K units

BODY TEMPERATURE REFERENCE:
    - Kappa thresholds scaled to BODY_TEMPERATURE (K × φ¹¹ ≈ 304 K)
    - Action cost = accumulated kappa reaches body-temp threshold

STRUCTURE DETECTION (44/45):
    - When visual entropy > 44/45 of max, there's unrecognized pattern
    - Triggers enhanced feature extraction
    - The 1/45 gap is where visual structure lives

ROBINSON CONSTRAINTS:
    - Position features: successor constraint (spatial)
    - Motion features: addition constraint (temporal)
    - Color/heat features: multiplication constraint (quantitative)

════════════════════════════════════════════════════════════════════════════════

USAGE:
    # Standalone test
    python visual_cortex.py --test
    
    # Screen capture loop
    python visual_cortex.py --capture --loop
    
    # With manifold integration
    python visual_cortex.py --capture --loop --manifold

PIPELINE:
    Image/Screen → Scanner → ScanFrame → VisionEngine → VisionStep → Manifold
         ↓              ↓           ↓            ↓            ↓
      pixels      color/pos/heat  3 frames   thresholds    nodes
"""

import sys
import os
import time
import numpy as np
from typing import Optional

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision.scan_simulator import VisualScanner, capture_screen, ScanFrame
from vision.vision_step import VisionStepEngine, VisionStep, K, PHI, THRESHOLD_MOVEMENT

# Import Planck constants
try:
    from core.node_constants import (
        BODY_TEMPERATURE, 
        MAX_ENTROPIC_PROBABILITY,
        EMERGENCE_THRESHOLD,
    )
except ImportError:
    BODY_TEMPERATURE = K * PHI ** 11  # ≈ 304 K
    MAX_ENTROPIC_PROBABILITY = 44/45
    EMERGENCE_THRESHOLD = 45/44


# ═══════════════════════════════════════════════════════════════════════════════
# VISUAL CORTEX (Planck-Grounded)
# ═══════════════════════════════════════════════════════════════════════════════

class VisualCortex:
    """
    Complete visual processing pipeline for PBAI (Planck-Grounded).
    
    Scanner (retina) → VisionEngine (V1) → Manifold integration
    
    CLOCK SYNC:
        The cortex can be connected to an EnvironmentCore for t_K synchronization.
        Each frame is timestamped with the current heat-time.
    
    STRUCTURE DETECTION:
        When visual entropy exceeds 44/45 threshold, enhanced pattern-seeking
        is triggered to find the hidden structure.
    """
    
    def __init__(self, resolution: int = 64, manifold=None, environment=None):
        """
        Initialize the visual cortex.
        
        Args:
            resolution: Scan resolution (default 64x64)
            manifold: Optional PBAI manifold for node integration
            environment: Optional EnvironmentCore for clock sync
        """
        self.resolution = resolution
        self.manifold = manifold
        self.environment = environment  # For clock sync
        
        # Components
        self.scanner = VisualScanner(width=resolution, height=resolution)
        self.engine = VisionStepEngine(resolution=resolution)
        
        # State
        self.last_step: Optional[VisionStep] = None
        self.frame_count = 0
        self.total_kappa_spent = 0.0
        
        # Planck grounding
        self._t_K = 0  # Local t_K if no environment connected
        self._structure_detected = False
        self._pattern_seek_mode = False
        
        # Kappa threshold for action (body temperature grounded)
        # Need to accumulate kappa = K before action is warranted
        self.kappa_action_threshold = K
        
        # Callbacks
        self.on_motion_detected = None
        self.on_new_actual = None
        self.on_structure_detected = None
    
    def get_t_K(self) -> int:
        """Get current t_K from environment or local counter."""
        if self.environment:
            return self.environment.get_t_K()
        return self._t_K
    
    def _check_structure_detection(self) -> bool:
        """
        Check if visual field has unrecognized structure (entropy > 44/45).
        
        Returns True if pattern-seeking should be triggered.
        """
        if not self.last_step:
            return False
        
        # Calculate visual entropy from heat distribution
        heat_field = self.engine.prev_heat_field
        if heat_field is None:
            return False
        
        # Normalize heat to probability distribution
        heat_sum = heat_field.sum()
        if heat_sum <= 0:
            return False
        
        probs = heat_field.flatten() / heat_sum
        probs = probs[probs > 0]  # Remove zeros for log
        
        # Calculate entropy
        entropy = -np.sum(probs * np.log(probs))
        
        # Max entropy for uniform distribution
        max_entropy = np.log(len(probs))
        
        if max_entropy <= 0:
            return False
        
        # Ratio of entropy to max
        ratio = entropy / max_entropy
        
        # Structure detected if entropy > 44/45 (there's hidden order)
        # Actually, LOWER entropy means MORE structure
        # So we detect structure when entropy is LOW (< 44/45 of max)
        structure_present = ratio < MAX_ENTROPIC_PROBABILITY
        
        return structure_present
    
    def process_image(self, image: np.ndarray, tick_clock: bool = True) -> VisionStep:
        """
        Process a single image through the full pipeline.
        
        Args:
            image: Input image (H, W, 3) RGB
            tick_clock: If True, advance t_K (clock sync)
            
        Returns:
            VisionStep with all extracted data
        """
        # Clock sync - advance t_K
        if tick_clock:
            if self.environment:
                # Environment handles clock
                pass
            else:
                self._t_K += 1
        
        # Scan image
        frame = self.scanner.scan(image)
        
        # Process through vision engine
        step = self.engine.process_frame(
            frame.color_field,
            frame.position_field,
            frame.heat_field
        )
        
        # Timestamp with t_K
        step.t_K = self.get_t_K()
        
        self.last_step = step
        self.frame_count += 1
        
        # Check structure detection
        structure_now = self._check_structure_detection()
        if structure_now and not self._structure_detected:
            self._pattern_seek_mode = True
            if self.on_structure_detected:
                self.on_structure_detected(step)
        elif not structure_now and self._structure_detected:
            self._pattern_seek_mode = False
        self._structure_detected = structure_now
        
        # Callbacks
        if self.on_motion_detected and step.get_motion_detected():
            self.on_motion_detected(step)
        
        if self.on_new_actual and step.new_actuals:
            for feature in step.new_actuals:
                self.on_new_actual(feature)
        
        # Manifold integration
        if self.manifold and step.new_actuals:
            self._integrate_to_manifold(step)
        
        return step
    
    def process_world_state(self, world_state: dict) -> Optional[VisionStep]:
        """Process a world_state dict from the PC vision transformer.

        Reconstructs color/position/heat fields from sparse peaks
        and feeds them through the VisionStepEngine pipeline for
        persistence tracking, existence gating, and manifold integration.

        Args:
            world_state: Dict with peaks, center, heat_stats, resolution

        Returns:
            VisionStep with tracked features, or None on error
        """
        if not world_state or not world_state.get('peaks'):
            return None

        res = self.resolution  # 64
        ws_res = world_state.get('resolution', 512)
        scale = res / ws_res

        # Background from center and mean heat
        center = world_state.get('center', {})
        bg_color = center.get('color', [128, 128, 128])
        bg_heat = world_state.get('heat_stats', {}).get('mean', 0.0)

        # Initialize fields
        color_field = np.full((res, res, 3), bg_color, dtype=np.uint8)
        heat_field = np.full((res, res), bg_heat, dtype=np.float32)

        # Position field via meshgrid
        xs = np.linspace(-1.0, 1.0, res, dtype=np.float32)
        ys = np.linspace(-1.0, 1.0, res, dtype=np.float32)
        gx, gy = np.meshgrid(xs, ys)
        position_field = np.stack([gx, gy], axis=-1)

        # Place peaks as small splats
        for peak in world_state.get('peaks', []):
            px = int(peak['x'] * scale)
            py = int(peak['y'] * scale)
            px = min(max(0, px), res - 1)
            py = min(max(0, py), res - 1)

            color = peak.get('color', bg_color)
            heat = peak.get('heat', 0.0)

            # Splat a small region around the peak
            r = 2
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    nx, ny = px + dx, py + dy
                    if 0 <= nx < res and 0 <= ny < res:
                        falloff = 1.0 / (1 + dx * dx + dy * dy)
                        heat_field[ny, nx] = max(heat_field[ny, nx], heat * falloff)
                        if dx == 0 and dy == 0:
                            color_field[ny, nx] = color

        # Feed through engine
        step = self.engine.process_frame(color_field, position_field, heat_field)
        step.t_K = self.get_t_K()

        self.last_step = step
        self.frame_count += 1

        # Structure detection
        structure_now = self._check_structure_detection()
        if structure_now and not self._structure_detected:
            self._pattern_seek_mode = True
            if self.on_structure_detected:
                self.on_structure_detected(step)
        elif not structure_now and self._structure_detected:
            self._pattern_seek_mode = False
        self._structure_detected = structure_now

        # Callbacks
        if self.on_motion_detected and step.get_motion_detected():
            self.on_motion_detected(step)

        if self.on_new_actual and step.new_actuals:
            for feature in step.new_actuals:
                self.on_new_actual(feature)

        # Manifold integration
        if self.manifold and step.new_actuals:
            self._integrate_to_manifold(step)

        return step

    def process_screen(self, tick_clock: bool = True) -> Optional[VisionStep]:
        """
        Capture and process the screen.

        Returns:
            VisionStep or None if capture failed
        """
        image = capture_screen()
        if image is None:
            return None
        return self.process_image(image, tick_clock=tick_clock)
    
    def _integrate_to_manifold(self, step: VisionStep):
        """Integrate vision step into manifold using driver's compendium.

        For each newly-actual visual feature:
        1. Ask the active driver to identify the color → block name
        2. Create/update a manifold node named after the block (not grid coords)
        3. Connect the identified node to the driver node
        """
        import logging
        _logger = logging.getLogger(__name__)
        if not self.manifold or not step.new_actuals:
            return

        # Get active driver for color identification
        driver = None
        if self.environment:
            driver = self.environment.get_active_driver()

        affected = 0
        try:
            from core.nodes import Node

            for feature in step.new_actuals:
                node_data = feature.to_node_data()
                concept = node_data["concept"]  # default: visual_X_Y

                # Ask driver's compendium to identify the color
                if driver and hasattr(driver, 'identify_color'):
                    r, g, b = feature.color
                    block_name, confidence = driver.identify_color(r, g, b)
                    if block_name and confidence > 0.3:
                        concept = block_name

                # Create or reinforce the node
                existing = self.manifold.get_node_by_concept(concept)
                if existing:
                    existing.add_heat(node_data["heat"])
                    affected += 1
                else:
                    node = Node(
                        concept=concept,
                        theta=node_data["theta"],
                        phi=node_data["phi"],
                        radius=node_data["radius"],
                        heat=node_data["heat"],
                        righteousness=node_data["righteousness"],
                        existence=node_data["existence"]
                    )
                    self.manifold.add_node(node)
                    affected += 1

                    # Connect to driver node so the block is in the action graph
                    if driver and hasattr(driver, 'driver_node') and driver.driver_node:
                        dn = driver.driver_node
                        if dn.node:
                            self.manifold.add_axis_safe(
                                dn.node, concept[:20], node.id
                            )

            if affected > 0:
                _logger.info(f"Vision integrated {affected} nodes to manifold @ t_K={step.t_K}")
        except Exception as e:
            _logger.debug(f"Manifold integration error: {e}")
    
    def spend_kappa(self, amount: float) -> bool:
        """
        Spend kappa for an action.
        
        Returns True if enough kappa was available.
        """
        if self.engine.accumulated_kappa >= amount:
            self.engine.accumulated_kappa -= amount
            self.total_kappa_spent += amount
            return True
        return False
    
    def can_act(self) -> bool:
        """Check if enough kappa accumulated for action (body temp grounded)."""
        return self.engine.accumulated_kappa >= self.kappa_action_threshold
    
    def get_kappa(self) -> float:
        """Get available kappa."""
        return self.engine.accumulated_kappa
    
    def get_righteous_alignment(self) -> float:
        """Get current righteousness alignment (lower = better)."""
        if self.last_step:
            return self.last_step.righteous_alignment
        return 1.0
    
    def get_motion_delta(self) -> float:
        """Get motion delta from last frame."""
        if self.last_step:
            return self.last_step.delta_motion
        return 0.0
    
    def get_actual_features(self):
        """Get all actual (persisted) features."""
        return self.engine.get_actual_nodes()
    
    def is_structure_detected(self) -> bool:
        """Check if visual structure has been detected."""
        return self._structure_detected
    
    def is_pattern_seeking(self) -> bool:
        """Check if in enhanced pattern-seeking mode."""
        return self._pattern_seek_mode
    
    def prune(self, max_age: float = 5.0):
        """Prune stale features."""
        self.engine.prune_stale_features(max_age)
    
    def get_stats(self) -> dict:
        """Get combined statistics."""
        engine_stats = self.engine.get_stats()
        return {
            **engine_stats,
            "total_kappa_spent": self.total_kappa_spent,
            "current_righteousness": self.get_righteous_alignment(),
            "t_K": self.get_t_K(),
            "structure_detected": self._structure_detected,
            "pattern_seek_mode": self._pattern_seek_mode,
            "can_act": self.can_act(),
        }
    
    def sync_status(self) -> dict:
        """Get clock sync status."""
        return {
            "t_K": self.get_t_K(),
            "frame_count": self.frame_count,
            "synced": self.frame_count == self.get_t_K() or self.get_t_K() == 0,
            "environment_connected": self.environment is not None,
            "manifold_connected": self.manifold is not None,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="PBAI Visual Cortex")
    parser.add_argument("--image", type=str, help="Process single image")
    parser.add_argument("--capture", action="store_true", help="Capture screen")
    parser.add_argument("--loop", action="store_true", help="Continuous processing")
    parser.add_argument("--resolution", type=int, default=64, help="Scan resolution")
    parser.add_argument("--manifold", action="store_true", help="Enable manifold integration")
    parser.add_argument("--test", action="store_true", help="Run test pattern")
    parser.add_argument("--interval", type=float, default=0.5, help="Loop interval")
    args = parser.parse_args()
    
    print("=" * 60)
    print("PBAI VISUAL CORTEX")
    print("=" * 60)
    print(f"Resolution: {args.resolution}x{args.resolution}")
    print(f"K = {K:.6f}")
    print(f"Motion threshold = {THRESHOLD_MOVEMENT:.4f}")
    print()
    
    # Load manifold if requested
    manifold = None
    if args.manifold:
        try:
            from core.manifold import get_pbai_manifold
            manifold = get_pbai_manifold()
            print(f"Manifold loaded: {len(manifold.nodes)} nodes")
        except Exception as e:
            print(f"Could not load manifold: {e}")
    
    # Create cortex
    cortex = VisualCortex(resolution=args.resolution, manifold=manifold)
    
    # Set up callbacks
    def on_motion(step):
        print(f"  ⚡ MOTION: delta={step.delta_motion:.4f}")
    
    def on_actual(feature):
        print(f"  ✓ NEW ACTUAL: ({feature.x}, {feature.y}) heat={feature.heat:.3f}")
    
    cortex.on_motion_detected = on_motion
    cortex.on_new_actual = on_actual
    
    # Process based on mode
    if args.test:
        print("Running test pattern...")
        for i in range(10):
            # Generate test image
            image = np.zeros((256, 256, 3), dtype=np.uint8)
            # Moving bright spot
            cx = 128 + int(40 * np.sin(i * 0.5))
            cy = 128 + int(40 * np.cos(i * 0.5))
            image[cy-10:cy+10, cx-10:cx+10] = [255, 200, 100]
            # Static spot
            image[100:120, 100:120] = [100, 255, 100]
            
            step = cortex.process_image(image)
            
            print(f"\nFrame {step.frame_number}:")
            print(f"  κ={step.kappa:.3f}, R={step.righteous_alignment:.3f}")
            print(f"  Peaks: {len(step.heat_peaks)}, Actuals: {len(step.get_actual_features())}")
            
            time.sleep(0.2)
    
    elif args.image:
        from PIL import Image
        image = np.array(Image.open(args.image).convert('RGB'))
        step = cortex.process_image(image)
        
        print(f"Processed {args.image}:")
        print(f"  κ={step.kappa:.3f}")
        print(f"  Righteousness: {step.righteous_alignment:.3f}")
        print(f"  Peaks: {len(step.heat_peaks)}")
        print(f"  Actual features: {len(step.get_actual_features())}")
    
    elif args.capture:
        if args.loop:
            print("Starting continuous capture (Ctrl+C to stop)...")
            try:
                while True:
                    step = cortex.process_screen()
                    if step:
                        print(f"\rFrame {step.frame_number}: κ={step.kappa:.2f} R={step.righteous_alignment:.2f} peaks={len(step.heat_peaks)}", end="")
                    time.sleep(args.interval)
            except KeyboardInterrupt:
                print("\n\nStopped.")
        else:
            step = cortex.process_screen()
            if step:
                print(f"Screen capture:")
                print(f"  κ={step.kappa:.3f}")
                print(f"  Righteousness: {step.righteous_alignment:.3f}")
                print(f"  Peaks: {len(step.heat_peaks)}")
    
    else:
        print("No input mode specified. Use --test, --image, or --capture")
        return
    
    # Final stats
    print()
    print("=" * 60)
    print("Final Statistics:")
    print("=" * 60)
    for k, v in cortex.get_stats().items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
