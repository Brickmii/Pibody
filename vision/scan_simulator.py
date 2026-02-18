"""
PBAI Visual Scan Simulator (Planck-Grounded)

Three-frame scan system reflecting the Blender model structure:
    Frame X: Color (what it IS) - identity/categorical
    Frame Y: Position (where it IS) - successor/spatial
    Frame Z: Heat (how much it MATTERS) - multiplication/quantitative

════════════════════════════════════════════════════════════════════════════════
PLANCK GROUNDING & BLENDER MODEL
════════════════════════════════════════════════════════════════════════════════

BLENDER STRUCTURE:
    The scan reflects PBAI's 12-direction manifold:
    
    Physical Directions (6):
        n, s, e, w, u, d - cubic lattice
        
    Self Directions (6):
        forward, back, left, right, up, down - relative to view
    
    The righteous point (center) is where Self looks.
    Heat radiates outward in spiral pattern (fovea → peripheral).

ROBINSON CONSTRAINTS ON FRAMES:
    Frame X (Color):    identity constraint - what exists
    Frame Y (Position): successor constraint - spatial stepping  
    Frame Z (Heat):     multiplication constraint - quantitative

CLOCK SYNC:
    ScanFrames are timestamped with t_K for manifold integration.
    Each scan can advance the visual cortex's local clock.

FIRE HEAT SCALING:
    Heat calculation uses K × φⁿ scaling:
    - Center (fovea): highest heat (Fire 6 = body temp)
    - Edges: lower heat (Fire 1-5 scaling)

════════════════════════════════════════════════════════════════════════════════

Scan pattern: Center-out spiral (fovea → peripheral)
Righteous point: Center of frame
Order: Correlation between frames

Usage:
    python scan_simulator.py --image screenshot.png
    python scan_simulator.py --capture  # live screen capture
"""

import numpy as np
import time
from dataclasses import dataclass, field
from typing import Tuple, List, Optional
import math

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import mss
    HAS_MSS = True
except ImportError:
    HAS_MSS = False

# Import Planck constants
try:
    from core.node_constants import (
        K, PHI,
        FIRE_HEAT,
        BODY_TEMPERATURE,
        ROBINSON_CONSTRAINTS,
    )
except ImportError:
    PHI = 1.618033988749895
    K = 4 / (PHI ** 2)
    BODY_TEMPERATURE = K * PHI ** 11
    FIRE_HEAT = {
        1: K * PHI ** 1,
        2: K * PHI ** 2,
        3: K * PHI ** 3,
        4: K * PHI ** 4,
        5: K * PHI ** 5,
        6: BODY_TEMPERATURE,
    }
    ROBINSON_CONSTRAINTS = {
        'identity': 1.0,
        'successor': 4 * PHI / 7,
        'addition': 4/3,
        'multiplication': 13/10,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SCAN POINT (Planck-Grounded)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ScanPoint:
    """
    A single point in the scan.
    
    PLANCK GROUNDING:
        - heat scaled to K units
        - constraint_type based on what this point represents
    """
    x: int              # Position X (successor constraint)
    y: int              # Position Y (successor constraint)
    color: Tuple[int, int, int]  # RGB (identity constraint)
    heat: float         # Salience 0-1 (multiplication constraint)
    distance: float     # Distance from center (successor)
    angle: float        # Angle from center (addition)
    
    # Planck grounding
    constraint_type: str = "successor"  # spatial points use successor
    fire_level: int = 1  # Which fire level (1-6) based on distance


# ═══════════════════════════════════════════════════════════════════════════════
# SCAN FRAME (Planck-Grounded)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ScanFrame:
    """
    One complete scan of the visual field.
    
    BLENDER MODEL STRUCTURE:
        Reflects the 12-direction manifold with righteous center.
        Three correlated frames matching Robinson constraints:
        - color_field: identity (what exists)
        - position_field: successor (where things are)
        - heat_field: multiplication (how much they matter)
    
    PLANCK GROUNDING:
        - t_K timestamp for clock sync
        - Fire heat scaling from center outward
    """
    
    # Dimensions
    width: int
    height: int
    center_x: int
    center_y: int
    
    # Three channels (Robinson-constrained)
    color_field: np.ndarray     # (H, W, 3) - RGB values (identity)
    position_field: np.ndarray  # (H, W, 2) - normalized x, y from center (successor)
    heat_field: np.ndarray      # (H, W) - salience values (multiplication)
    
    # Scan order (spiral from center)
    scan_order: List[Tuple[int, int]] = None
    
    # Timestamps
    timestamp: float = 0.0      # Wall clock
    t_K: int = 0                # Heat-time (Planck-grounded)
    
    # Fire level map (which fire zone each pixel is in)
    fire_map: np.ndarray = None  # (H, W) - fire level 1-6 based on distance


# ═══════════════════════════════════════════════════════════════════════════════
# SPIRAL GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

def generate_spiral(width: int, height: int) -> List[Tuple[int, int]]:
    """
    Generate spiral scan order from center outward.
    
    Returns list of (x, y) coordinates in scan order.
    """
    cx, cy = width // 2, height // 2
    
    points = []
    visited = set()
    
    # Start at center
    x, y = cx, cy
    points.append((x, y))
    visited.add((x, y))
    
    # Spiral outward
    step = 1
    while len(points) < width * height:
        # Right
        for _ in range(step):
            x += 1
            if 0 <= x < width and 0 <= y < height and (x, y) not in visited:
                points.append((x, y))
                visited.add((x, y))
        
        # Down
        for _ in range(step):
            y += 1
            if 0 <= x < width and 0 <= y < height and (x, y) not in visited:
                points.append((x, y))
                visited.add((x, y))
        
        step += 1
        
        # Left
        for _ in range(step):
            x -= 1
            if 0 <= x < width and 0 <= y < height and (x, y) not in visited:
                points.append((x, y))
                visited.add((x, y))
        
        # Up
        for _ in range(step):
            y -= 1
            if 0 <= x < width and 0 <= y < height and (x, y) not in visited:
                points.append((x, y))
                visited.add((x, y))
        
        step += 1
    
    return points


# ═══════════════════════════════════════════════════════════════════════════════
# HEAT CALCULATION (Planck-Grounded with Fire Scaling)
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_fire_map(width: int, height: int) -> np.ndarray:
    """
    Calculate fire level map based on distance from center.
    
    Fire zones (reflecting Blender model):
        Fire 6 (center): 0-10% of max distance - body temperature zone
        Fire 5: 10-25% - order zone
        Fire 4: 25-40% - righteousness zone  
        Fire 3: 40-55% - existence zone
        Fire 2: 55-75% - polarity zone
        Fire 1: 75-100% - heat zone (peripheral)
    
    Returns (H, W) array of fire levels 1-6.
    """
    h, w = height, width
    cy, cx = h // 2, w // 2
    
    y_coords, x_coords = np.ogrid[:h, :w]
    distance = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
    max_dist = np.sqrt(cx**2 + cy**2)
    
    # Normalize distance to 0-1
    norm_dist = distance / max_dist
    
    # Assign fire levels (6 = center, 1 = peripheral)
    fire_map = np.ones((h, w), dtype=np.int32)  # Default to Fire 1
    fire_map[norm_dist < 0.75] = 2
    fire_map[norm_dist < 0.55] = 3
    fire_map[norm_dist < 0.40] = 4
    fire_map[norm_dist < 0.25] = 5
    fire_map[norm_dist < 0.10] = 6  # Center = Fire 6 (body temp)

    # Hotbar boost: bottom strip gets Fire 4 so item icons register as peaks
    # At 512x512, hotbar spans roughly y=483..510 (bottom ~5% of screen)
    hotbar_top = int(h * 0.945)  # ~484 at 512
    fire_map[hotbar_top:, :] = np.maximum(fire_map[hotbar_top:, :], 4)

    return fire_map


def calculate_heat(color_field: np.ndarray, position_field: np.ndarray, 
                   prev_color_field: Optional[np.ndarray] = None,
                   fire_map: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Calculate heat (salience) for each pixel.
    
    PLANCK GROUNDING:
        Heat sources are weighted by Fire level (K × φⁿ scaling):
        - Center (Fire 6): body temperature weight
        - Peripheral (Fire 1): minimal weight
    
    Heat sources:
    - Center bias (fovea attention) - Fire scaled
    - Edge detection (boundaries matter)
    - Motion (if previous frame provided)
    - Color saturation (vivid = salient)
    """
    h, w = color_field.shape[:2]
    heat = np.zeros((h, w), dtype=np.float32)
    
    # 1. Center bias - Gaussian falloff from center (Fire-scaled)
    cy, cx = h // 2, w // 2
    y_coords, x_coords = np.ogrid[:h, :w]
    distance = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
    max_dist = np.sqrt(cx**2 + cy**2)
    
    # Fire-scaled center heat
    if fire_map is not None:
        # Weight by fire level (normalized to 0-1)
        fire_weight = fire_map / 6.0
        center_heat = fire_weight * 0.4
    else:
        center_heat = (1.0 - (distance / max_dist)) * 0.3
    heat += center_heat
    
    # 2. Edge detection - Sobel-like gradient
    gray = np.mean(color_field, axis=2)
    gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    edges = np.sqrt(gx**2 + gy**2)
    edges = edges / (edges.max() + 1e-6)
    heat += edges * 0.3
    
    # 3. Motion detection - difference from previous frame
    if prev_color_field is not None:
        diff = np.abs(color_field.astype(float) - prev_color_field.astype(float))
        motion = np.mean(diff, axis=2) / 255.0
        heat += motion * 0.3
    
    # 4. Color saturation - vivid colors are salient
    max_c = np.max(color_field, axis=2)
    min_c = np.min(color_field, axis=2)
    saturation = (max_c - min_c) / (max_c + 1e-6)
    heat += saturation * 0.1
    
    # Normalize
    heat = np.clip(heat, 0, 1)
    
    return heat


# ═══════════════════════════════════════════════════════════════════════════════
# SCANNER (Planck-Grounded with Blender Model Structure)
# ═══════════════════════════════════════════════════════════════════════════════

class VisualScanner:
    """
    PBAI Visual Scanner (Planck-Grounded)
    
    Scans images center-out, building three correlated frames:
    - X (color): what things are (identity constraint)
    - Y (position): where things are (successor constraint)
    - Z (heat): how much they matter (multiplication constraint)
    
    BLENDER MODEL:
        Reflects the 12-direction manifold:
        - Center (righteous point) = Self's view direction
        - Spiral outward = 6 fire zones
        - Each zone has K × φⁿ heat scaling
    """
    
    def __init__(self, width: int = 64, height: int = 64):
        """
        Initialize scanner.
        
        Args:
            width: Scan resolution width
            height: Scan resolution height
        """
        self.width = width
        self.height = height
        self.center_x = width // 2
        self.center_y = height // 2
        
        # Generate spiral scan order
        print(f"Generating spiral scan order for {width}x{height}...")
        self.scan_order = generate_spiral(width, height)
        print(f"Scan order: {len(self.scan_order)} points")
        
        # Planck grounding - fire map
        print("Calculating fire map (K × φⁿ zones)...")
        self.fire_map = calculate_fire_map(width, height)
        print(f"Fire zones: {np.bincount(self.fire_map.flatten(), minlength=7)[1:]}")
        
        # Previous frame for motion detection
        self.prev_color_field = None
        
        # Frame counter
        self.frame_count = 0
        
        # t_K counter (local clock)
        self._t_K = 0
    
    def scan(self, image: np.ndarray, t_K: int = None) -> ScanFrame:
        """
        Scan an image and return the three-frame representation.
        
        Args:
            image: Input image (H, W, 3) RGB
            t_K: Optional t_K timestamp (uses internal counter if not provided)
            
        Returns:
            ScanFrame with color, position, heat fields, and fire map
        """
        # Advance local clock if no t_K provided
        if t_K is None:
            self._t_K += 1
            t_K = self._t_K
        
        # Resize to scan resolution
        if HAS_PIL:
            pil_img = Image.fromarray(image)
            pil_img = pil_img.resize((self.width, self.height), Image.LANCZOS)
            image = np.array(pil_img)
        else:
            # Simple resize without PIL
            h, w = image.shape[:2]
            y_indices = (np.arange(self.height) * h / self.height).astype(int)
            x_indices = (np.arange(self.width) * w / self.width).astype(int)
            image = image[y_indices][:, x_indices]
        
        # Frame X: Color field (identity constraint)
        color_field = image.copy()
        
        # Frame Y: Position field (successor constraint - normalized from center)
        position_field = np.zeros((self.height, self.width, 2), dtype=np.float32)
        for y in range(self.height):
            for x in range(self.width):
                dx = (x - self.center_x) / self.center_x
                dy = (y - self.center_y) / self.center_y
                position_field[y, x] = [dx, dy]
        
        # Frame Z: Heat field (multiplication constraint, fire-scaled)
        heat_field = calculate_heat(color_field, position_field, 
                                    self.prev_color_field, self.fire_map)
        
        # Store for next frame's motion detection
        self.prev_color_field = color_field.copy()
        
        # Build frame
        frame = ScanFrame(
            width=self.width,
            height=self.height,
            center_x=self.center_x,
            center_y=self.center_y,
            color_field=color_field,
            position_field=position_field,
            heat_field=heat_field,
            scan_order=self.scan_order,
            timestamp=time.time(),
            t_K=t_K,
            fire_map=self.fire_map,
        )
        
        self.frame_count += 1
        
        return frame
    
    def scan_progressive(self, image: np.ndarray, steps: int = 100) -> List[ScanPoint]:
        """
        Scan progressively in spiral order, yielding points.
        
        Args:
            image: Input image
            steps: Number of points to scan (for partial scan)
            
        Returns:
            List of ScanPoints in scan order
        """
        frame = self.scan(image)
        
        points = []
        for i, (x, y) in enumerate(self.scan_order[:steps]):
            color = tuple(frame.color_field[y, x])
            heat = frame.heat_field[y, x]
            
            dx = x - self.center_x
            dy = y - self.center_y
            distance = math.sqrt(dx**2 + dy**2)
            angle = math.atan2(dy, dx)
            
            point = ScanPoint(
                x=x, y=y,
                color=color,
                heat=float(heat),
                distance=distance,
                angle=angle
            )
            points.append(point)
        
        return points
    
    def get_righteous_point(self, frame: ScanFrame) -> ScanPoint:
        """Get the center (righteous) point of a frame."""
        x, y = self.center_x, self.center_y
        return ScanPoint(
            x=x, y=y,
            color=tuple(frame.color_field[y, x]),
            heat=float(frame.heat_field[y, x]),
            distance=0.0,
            angle=0.0
        )
    
    def get_heat_peaks(self, frame: ScanFrame, threshold: float = 0.7) -> List[ScanPoint]:
        """Get points where heat exceeds threshold."""
        points = []
        for y in range(frame.height):
            for x in range(frame.width):
                heat = frame.heat_field[y, x]
                if heat >= threshold:
                    dx = x - self.center_x
                    dy = y - self.center_y
                    points.append(ScanPoint(
                        x=x, y=y,
                        color=tuple(frame.color_field[y, x]),
                        heat=float(heat),
                        distance=math.sqrt(dx**2 + dy**2),
                        angle=math.atan2(dy, dx)
                    ))
        
        # Sort by heat descending
        points.sort(key=lambda p: p.heat, reverse=True)
        return points


# ═══════════════════════════════════════════════════════════════════════════════
# SCREEN CAPTURE
# ═══════════════════════════════════════════════════════════════════════════════

def capture_screen() -> Optional[np.ndarray]:
    """Capture the screen and return as numpy array."""
    if not HAS_MSS:
        return None
    
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        screenshot = sct.grab(monitor)
        image = np.array(screenshot)
        # Convert BGRA to RGB
        image = image[:, :, [2, 1, 0]]
        return image


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def visualize_frame(frame: ScanFrame) -> np.ndarray:
    """
    Create a visualization of the three frames side by side.
    
    Returns (H, W*3, 3) image showing:
    - Color field
    - Position field (as color)
    - Heat field (grayscale)
    """
    h, w = frame.height, frame.width
    
    # Color field (as-is)
    color_vis = frame.color_field
    
    # Position field (map x,y to color)
    pos_vis = np.zeros((h, w, 3), dtype=np.uint8)
    pos_vis[:, :, 0] = ((frame.position_field[:, :, 0] + 1) * 127).astype(np.uint8)  # R = x
    pos_vis[:, :, 1] = ((frame.position_field[:, :, 1] + 1) * 127).astype(np.uint8)  # G = y
    pos_vis[:, :, 2] = 128  # B = constant
    
    # Heat field (grayscale → RGB)
    heat_vis = (frame.heat_field * 255).astype(np.uint8)
    heat_vis = np.stack([heat_vis, heat_vis, heat_vis], axis=2)
    
    # Combine
    combined = np.concatenate([color_vis, pos_vis, heat_vis], axis=1)
    
    return combined


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="PBAI Visual Scan Simulator")
    parser.add_argument("--image", type=str, help="Image file to scan")
    parser.add_argument("--capture", action="store_true", help="Capture screen")
    parser.add_argument("--resolution", type=int, default=64, help="Scan resolution")
    parser.add_argument("--loop", action="store_true", help="Continuous scan loop")
    parser.add_argument("--output", type=str, help="Output visualization to file")
    args = parser.parse_args()
    
    # Create scanner
    scanner = VisualScanner(width=args.resolution, height=args.resolution)
    
    # Get image
    if args.image:
        if not HAS_PIL:
            print("PIL required for image loading")
            return
        image = np.array(Image.open(args.image).convert('RGB'))
    elif args.capture:
        image = capture_screen()
        if image is None:
            print("mss required for screen capture")
            return
    else:
        # Generate test pattern
        print("No input specified, generating test pattern...")
        image = np.zeros((256, 256, 3), dtype=np.uint8)
        # Red center
        image[112:144, 112:144] = [255, 0, 0]
        # Green top-right
        image[32:64, 192:224] = [0, 255, 0]
        # Blue bottom-left
        image[192:224, 32:64] = [0, 0, 255]
        # White edges
        image[0:8, :] = [255, 255, 255]
        image[-8:, :] = [255, 255, 255]
        image[:, 0:8] = [255, 255, 255]
        image[:, -8:] = [255, 255, 255]
    
    def do_scan(img):
        # Scan
        frame = scanner.scan(img)
        
        # Report
        print(f"\n{'='*50}")
        print(f"SCAN {scanner.frame_count}")
        print(f"{'='*50}")
        
        # Righteous point
        rp = scanner.get_righteous_point(frame)
        print(f"RIGHTEOUS (center): color={rp.color}, heat={rp.heat:.3f}")
        
        # Heat peaks
        peaks = scanner.get_heat_peaks(frame, threshold=0.5)[:5]
        print(f"\nHEAT PEAKS (top 5):")
        for p in peaks:
            print(f"  ({p.x}, {p.y}): heat={p.heat:.3f}, dist={p.distance:.1f}, color={p.color}")
        
        # Stats
        print(f"\nSTATS:")
        print(f"  Mean heat: {frame.heat_field.mean():.3f}")
        print(f"  Max heat:  {frame.heat_field.max():.3f}")
        print(f"  Hot pixels (>0.5): {(frame.heat_field > 0.5).sum()}")
        
        # Visualization
        vis = visualize_frame(frame)
        if args.output:
            Image.fromarray(vis).save(args.output)
            print(f"\nVisualization saved to {args.output}")
        
        return frame
    
    if args.loop:
        print("Starting continuous scan loop (Ctrl+C to stop)...")
        try:
            while True:
                if args.capture:
                    image = capture_screen()
                do_scan(image)
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        do_scan(image)


if __name__ == "__main__":
    main()
