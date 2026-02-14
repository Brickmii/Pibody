"""
PBAI HUD Reader — Extract game state from raw Minecraft screenshots.

Pure numpy module. Reads health, hunger, hotbar selection, and
slot occupancy from pixel analysis of the Bedrock Edition HUD.

Performance: ~1ms per frame on CPU (no GPU needed).

HUD LAYOUT (Bedrock Edition, relative to screen bottom-center):
    Hotbar: 9 slots centered at bottom of screen
    Slot size: 20px * gui_scale, gap: 2px * gui_scale
    Hearts: 10 icons above hotbar left side, spaced 8px * gui_scale
    Hunger: 10 icons above hotbar right side, same spacing

USAGE:
    reader = HUDReader(1920, 1080, gui_scale=2)
    hud = reader.read_hud(screenshot_rgb)
    # hud = {"health": 18.0, "hunger": 16.0, "hotbar_selected": 1,
    #        "hotbar_slots": [True, True, False, ...]}
"""

import numpy as np
from typing import List, Dict, Any


class HUDReader:
    """Extract Minecraft HUD data from raw screenshots."""

    def __init__(self, screen_width: int, screen_height: int, gui_scale: int = 2):
        self.sw = screen_width
        self.sh = screen_height
        self.gs = gui_scale

        # Pre-compute pixel regions
        self._compute_regions()

    def _compute_regions(self):
        """Pre-compute HUD element pixel coordinates."""
        sw, sh, gs = self.sw, self.sh, self.gs

        # ── Hotbar ──
        slot_size = 20 * gs       # 40px at scale 2
        slot_gap = 2 * gs         # 4px at scale 2
        slot_stride = slot_size + slot_gap
        hotbar_width = 9 * slot_size + 8 * slot_gap
        hotbar_left = sw // 2 - hotbar_width // 2
        hotbar_top = sh - 22 * gs  # 44px from bottom at scale 2

        # Center of each slot (x, y)
        self._slot_centers = []
        for i in range(9):
            cx = hotbar_left + i * slot_stride + slot_size // 2
            cy = hotbar_top + slot_size // 2
            self._slot_centers.append((cx, cy))

        # Top edge of each slot (for selection detection)
        self._slot_top_edges = []
        for i in range(9):
            x = hotbar_left + i * slot_stride + slot_size // 2
            y = hotbar_top - gs  # Just above the slot (selection border)
            self._slot_top_edges.append((x, y))

        # ── Hearts (health) ──
        # 10 heart icons above hotbar left side
        heart_spacing = 8 * gs
        heart_y = hotbar_top - 10 * gs  # Above hotbar
        heart_start_x = hotbar_left + 8 * gs
        self._heart_centers = []
        for i in range(10):
            hx = heart_start_x + i * heart_spacing
            self._heart_centers.append((hx, heart_y))

        # ── Hunger (drumsticks) ──
        # 10 drumstick icons above hotbar right side
        hunger_start_x = hotbar_left + hotbar_width - 8 * gs
        hunger_y = heart_y
        self._hunger_centers = []
        for i in range(10):
            hx = hunger_start_x - i * heart_spacing
            self._hunger_centers.append((hx, hunger_y))

        # Sample radius for pixel checks
        self._sample_r = max(1, gs)

    def read_hud(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract all HUD data from a raw RGB screenshot.

        Args:
            image: (H, W, 3) uint8 RGB array

        Returns:
            dict with health, hunger, hotbar_selected, hotbar_slots
        """
        return {
            "health": self._read_health(image),
            "hunger": self._read_hunger(image),
            "hotbar_selected": self._read_hotbar_selection(image),
            "hotbar_slots": self._read_hotbar_slots(image),
        }

    def _sample_region(self, image: np.ndarray, cx: int, cy: int) -> np.ndarray:
        """Sample a small square region around (cx, cy). Returns (N, 3) RGB."""
        r = self._sample_r
        h, w = image.shape[:2]
        y0 = max(0, cy - r)
        y1 = min(h, cy + r + 1)
        x0 = max(0, cx - r)
        x1 = min(w, cx + r + 1)
        return image[y0:y1, x0:x1].reshape(-1, 3)

    def _read_health(self, image: np.ndarray) -> float:
        """Count red hearts → health (0-20 half-hearts).

        Full heart = R>180, G<60, B<60.
        Half heart: same but fewer red pixels in sample region.
        """
        count = 0.0
        for cx, cy in self._heart_centers:
            pixels = self._sample_region(image, cx, cy)
            if len(pixels) == 0:
                continue
            # Red pixel mask
            red_mask = (pixels[:, 0] > 180) & (pixels[:, 1] < 60) & (pixels[:, 2] < 60)
            red_ratio = red_mask.sum() / len(pixels)
            if red_ratio > 0.5:
                count += 2.0   # Full heart = 2 half-hearts
            elif red_ratio > 0.15:
                count += 1.0   # Half heart
        return min(count, 20.0)

    def _read_hunger(self, image: np.ndarray) -> float:
        """Count drumsticks → hunger (0-20 half-shanks).

        Drumstick brown: R 140-200, G 90-140, B 20-70.
        """
        count = 0.0
        for cx, cy in self._hunger_centers:
            pixels = self._sample_region(image, cx, cy)
            if len(pixels) == 0:
                continue
            # Brown pixel mask (drumstick color)
            brown_mask = (
                (pixels[:, 0] > 140) & (pixels[:, 0] < 200) &
                (pixels[:, 1] > 90) & (pixels[:, 1] < 140) &
                (pixels[:, 2] > 20) & (pixels[:, 2] < 70)
            )
            brown_ratio = brown_mask.sum() / len(pixels)
            if brown_ratio > 0.4:
                count += 2.0
            elif brown_ratio > 0.1:
                count += 1.0
        return min(count, 20.0)

    def _read_hotbar_selection(self, image: np.ndarray) -> int:
        """Find selected slot (1-9) by white border brightness.

        Selected slot has a bright white/light gray border at its top edge.
        """
        best_brightness = 0
        best_slot = 1
        for i, (x, y) in enumerate(self._slot_top_edges):
            pixels = self._sample_region(image, x, y)
            if len(pixels) == 0:
                continue
            brightness = pixels.mean()
            if brightness > best_brightness:
                best_brightness = brightness
                best_slot = i + 1  # 1-indexed
        return best_slot

    def _read_hotbar_slots(self, image: np.ndarray) -> List[bool]:
        """Detect occupied vs empty slots.

        High pixel variance in slot center = item present.
        Uniform dark gray = empty slot.
        """
        result = []
        # Empty slot background is uniform dark gray (~50-80 luminance)
        variance_threshold = 200  # Empirical: items have colorful pixels
        for cx, cy in self._slot_centers:
            # Sample a larger region inside the slot
            r = max(3, self.gs * 3)
            h, w = image.shape[:2]
            y0 = max(0, cy - r)
            y1 = min(h, cy + r + 1)
            x0 = max(0, cx - r)
            x1 = min(w, cx + r + 1)
            region = image[y0:y1, x0:x1].reshape(-1, 3).astype(float)
            if len(region) == 0:
                result.append(False)
                continue
            # Variance across all channels
            var = region.var()
            result.append(var > variance_threshold)
        return result
