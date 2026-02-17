"""
PBAI HUD Reader — Extract game state from raw Minecraft screenshots.

Pure numpy module. Reads health, hunger, air, hotbar selection, slot
occupancy, and player coordinates from pixel analysis of the Bedrock
Edition HUD.

Performance: ~1ms per frame on CPU (no GPU needed).
Coordinate OCR requires easyocr (optional, graceful fallback).

HUD LAYOUT (Bedrock Edition, relative to screen bottom-center):
    Hotbar: 9 slots centered at bottom of screen
    Slot size: 20px * gui_scale, gap: 2px * gui_scale
    Hearts: 10 icons above hotbar left side, spaced 8px * gui_scale
    Hunger: 10 icons above hotbar right side, same spacing
    Air bubbles: 10 icons above hunger bar (only when underwater)

USAGE:
    reader = HUDReader(1920, 1080, gui_scale=2)
    hud = reader.read_hud(screenshot_rgb)
    # hud = {"health": 18.0, "hunger": 16.0, "air": -1,
    #        "hotbar_selected": 1, "hotbar_slots": [True, ...],
    #        "coordinates": {"x": 100.0, "y": 64.0, "z": -200.0}}
"""

import logging
import re
import time
import numpy as np
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

try:
    import easyocr
    HAS_EASYOCR = True
except ImportError:
    HAS_EASYOCR = False


class HUDReader:
    """Extract Minecraft HUD data from raw screenshots."""

    # Bedrock HUD layout constants (GUI-pixels, multiply by gui_scale)
    HOTBAR_BOTTOM_OFFSET = 22   # hotbar top = sh - this * gs
    HEART_ROW_OFFSET = 10       # hearts above hotbar top
    HEART_SPACING = 8           # pixels between heart left edges
    HEART_LEFT_INSET = 9        # first heart center from hotbar left
    HUNGER_RIGHT_INSET = 8      # first hunger icon from hotbar right
    AIR_ROW_OFFSET = 10         # air bubbles above hunger row

    def __init__(self, screen_width: int, screen_height: int, gui_scale: int = 2):
        self.sw = screen_width
        self.sh = screen_height
        self.gs = gui_scale

        # Pre-compute pixel regions
        self._compute_regions()

        # Auto-calibration state
        self._calibrated = False
        self._default_gs = gui_scale

        # Lazy-init easyocr reader
        self._ocr_reader = None

        # Coordinate OCR cache (easyocr is ~200-500ms, only re-run every 5s)
        self._coord_cache: Dict[str, float] = {}
        self._coord_cache_time: float = 0.0

    def _compute_regions(self):
        """Pre-compute HUD element pixel coordinates."""
        sw, sh, gs = self.sw, self.sh, self.gs

        # ── Hotbar ──
        slot_size = 20 * gs       # 40px at scale 2
        slot_gap = 2 * gs         # 4px at scale 2
        slot_stride = slot_size + slot_gap
        hotbar_width = 9 * slot_size + 8 * slot_gap
        hotbar_left = sw // 2 - hotbar_width // 2
        hotbar_top = sh - self.HOTBAR_BOTTOM_OFFSET * gs

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
        heart_spacing = self.HEART_SPACING * gs
        heart_y = hotbar_top - self.HEART_ROW_OFFSET * gs
        heart_start_x = hotbar_left + self.HEART_LEFT_INSET * gs
        self._heart_centers = []
        for i in range(10):
            hx = heart_start_x + i * heart_spacing
            self._heart_centers.append((hx, heart_y))

        # ── Hunger (drumsticks) ──
        hunger_start_x = hotbar_left + hotbar_width - self.HUNGER_RIGHT_INSET * gs
        hunger_y = heart_y
        self._hunger_centers = []
        for i in range(10):
            hx = hunger_start_x - i * heart_spacing
            self._hunger_centers.append((hx, hunger_y))

        # ── Air bubbles (above hunger) ──
        bubble_y = hunger_y - self.AIR_ROW_OFFSET * gs
        self._bubble_centers = []
        for i in range(10):
            bx = hunger_start_x - i * heart_spacing
            self._bubble_centers.append((bx, bubble_y))

        # Sample radius for pixel checks (large enough for gs=4 icons ~32px)
        self._sample_r = max(3, gs * 2)

    def _auto_calibrate(self, image: np.ndarray) -> None:
        """Try gui_scale values and pick the one that finds the most hearts/hunger."""
        best_score = 0
        best_gs = self._default_gs
        scores: Dict[int, int] = {}

        for try_gs in [4, 3, 2, 1]:
            self.gs = try_gs
            self._compute_regions()

            score = 0
            # Check first 5 heart positions for red pixels
            for cx, cy in self._heart_centers[:5]:
                pixels = self._sample_region(image, cx, cy)
                if len(pixels) == 0:
                    continue
                red = (pixels[:, 0] > 150) & (pixels[:, 1] < 80) & (pixels[:, 2] < 80)
                if red.sum() / len(pixels) > 0.15:
                    score += 1

            # Check first 5 hunger positions for brown pixels
            for cx, cy in self._hunger_centers[:5]:
                pixels = self._sample_region(image, cx, cy)
                if len(pixels) == 0:
                    continue
                brown = (
                    (pixels[:, 0] > 130) & (pixels[:, 0] < 210) &
                    (pixels[:, 1] > 80) & (pixels[:, 1] < 150) &
                    (pixels[:, 2] > 15) & (pixels[:, 2] < 80)
                )
                if brown.sum() / len(pixels) > 0.1:
                    score += 1

            scores[try_gs] = score
            if score > best_score:
                best_score = score
                best_gs = try_gs

        # Need at least 3 hits (out of 10) to trust the result
        if best_score >= 3:
            self.gs = best_gs
            self._compute_regions()
            self._calibrated = True
            logger.info("Auto-calibrated gui_scale=%d (score=%d/10, all scores: %s)",
                        best_gs, best_score, scores)

            # Diagnostic: log RGB at first 3 heart positions
            h, w = image.shape[:2]
            for i, (cx, cy) in enumerate(self._heart_centers[:3]):
                if 0 <= cy < h and 0 <= cx < w:
                    r, g, b = image[cy, cx]
                    logger.info("  heart[%d] (%d,%d) RGB=(%d,%d,%d)", i, cx, cy, r, g, b)

            logger.info("  image shape: %s", image.shape)
            self._force_debug_resave = True
        else:
            # Keep default, retry next frame
            self.gs = self._default_gs
            self._compute_regions()
            logger.warning("Auto-calibrate inconclusive (best=%d, score=%d/10, scores=%s) — retrying next frame",
                           best_gs, best_score, scores)

    def read_hud(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract all HUD data from a raw RGB screenshot.

        Args:
            image: (H, W, 3) uint8 RGB array

        Returns:
            dict with health, hunger, air, hotbar_selected, hotbar_slots,
            coordinates
        """
        if not self._calibrated:
            self._auto_calibrate(image)

        return {
            "health": self._read_health(image),
            "hunger": self._read_hunger(image),
            "air": self._read_air(image),
            "hotbar_selected": self._read_hotbar_selection(image),
            "hotbar_slots": self._read_hotbar_slots(image),
            "coordinates": self._read_coordinates(image),
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
        """Count red hearts -> health (0-20 half-hearts).

        Full heart = R>150, G<80, B<80 (widened for half hearts, damage flash).
        """
        count = 0.0
        for cx, cy in self._heart_centers:
            pixels = self._sample_region(image, cx, cy)
            if len(pixels) == 0:
                continue
            # Red pixel mask (widened thresholds)
            red_mask = (pixels[:, 0] > 150) & (pixels[:, 1] < 80) & (pixels[:, 2] < 80)
            red_ratio = red_mask.sum() / len(pixels)
            if red_ratio > 0.5:
                count += 2.0   # Full heart = 2 half-hearts
            elif red_ratio > 0.15:
                count += 1.0   # Half heart
        return min(count, 20.0)

    def _read_hunger(self, image: np.ndarray) -> float:
        """Count drumsticks -> hunger (0-20 half-shanks).

        Drumstick brown: R 130-210, G 80-150, B 15-80.
        """
        count = 0.0
        for cx, cy in self._hunger_centers:
            pixels = self._sample_region(image, cx, cy)
            if len(pixels) == 0:
                continue
            # Brown pixel mask (widened thresholds)
            brown_mask = (
                (pixels[:, 0] > 130) & (pixels[:, 0] < 210) &
                (pixels[:, 1] > 80) & (pixels[:, 1] < 150) &
                (pixels[:, 2] > 15) & (pixels[:, 2] < 80)
            )
            brown_ratio = brown_mask.sum() / len(pixels)
            if brown_ratio > 0.4:
                count += 2.0
            elif brown_ratio > 0.1:
                count += 1.0
        return min(count, 20.0)

    def _read_air(self, image: np.ndarray) -> int:
        """Count air bubbles -> air ticks (0-300, or -1 if not underwater).

        Air bubbles are blue: R<100, G>130, B>170.
        10 bubbles max, each = 30 air ticks. Pop from right to left.
        Returns -1 if no bubbles detected (not underwater).
        """
        count = 0
        any_blue = False
        for cx, cy in self._bubble_centers:
            pixels = self._sample_region(image, cx, cy)
            if len(pixels) == 0:
                continue
            blue_mask = (pixels[:, 0] < 100) & (pixels[:, 1] > 130) & (pixels[:, 2] > 170)
            blue_ratio = blue_mask.sum() / len(pixels)
            if blue_ratio > 0.3:
                count += 1
                any_blue = True
        if not any_blue:
            return -1
        return count * 30

    def _read_coordinates(self, image: np.ndarray) -> Dict[str, float]:
        """OCR the coordinate overlay from the top-left of the screen.

        Bedrock "Show Coordinates" displays white text:
            Position: X, Y, Z

        Returns {"x": float, "y": float, "z": float} or empty dict on failure.
        """
        if not HAS_EASYOCR:
            return {}

        # Return cached result if within TTL (easyocr is ~200-500ms)
        now = time.monotonic()
        if self._coord_cache and (now - self._coord_cache_time) < 5.0:
            return dict(self._coord_cache)

        try:
            # Lazy-init the reader (heavy first-time load)
            if self._ocr_reader is None:
                self._ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)

            # Crop top-left region (generous for 1440p scale 2)
            crop_h = min(100, image.shape[0])
            crop_w = min(400, image.shape[1])
            crop = image[:crop_h, :crop_w]

            results = self._ocr_reader.readtext(crop, detail=0)
            text = " ".join(results)

            match = re.search(r'Position:\s*([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)', text)
            if match:
                self._coord_cache = {
                    "x": float(match.group(1)),
                    "y": float(match.group(2)),
                    "z": float(match.group(3)),
                }
                self._coord_cache_time = now
                return dict(self._coord_cache)
        except Exception:
            pass
        return {}

    def _read_hotbar_selection(self, image: np.ndarray) -> int:
        """Find selected slot (1-9) by white border brightness."""
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
        """
        result = []
        variance_threshold = 200
        for cx, cy in self._slot_centers:
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
            var = region.var()
            result.append(bool(var > variance_threshold))
        return result

    def debug_save(self, image: np.ndarray, path: str = "hud_debug.png"):
        """Save a debug overlay showing sample points on the screenshot.

        Green dots  = heart centers
        Blue dots   = hunger centers
        Yellow dots = slot centers
        Cyan dots   = slot top edges
        Magenta dots = air bubble centers
        """
        from PIL import Image, ImageDraw

        debug = Image.fromarray(image.copy())
        draw = ImageDraw.Draw(debug)
        dot_r = max(6, self.gs * 3)

        def dot(x, y, color):
            draw.ellipse([x - dot_r, y - dot_r, x + dot_r, y + dot_r], fill=color)

        for cx, cy in self._heart_centers:
            dot(cx, cy, (0, 255, 0))       # Green

        for cx, cy in self._hunger_centers:
            dot(cx, cy, (0, 100, 255))      # Blue

        for cx, cy in self._slot_centers:
            dot(cx, cy, (255, 255, 0))      # Yellow

        for cx, cy in self._slot_top_edges:
            dot(cx, cy, (0, 255, 255))      # Cyan

        for cx, cy in self._bubble_centers:
            dot(cx, cy, (255, 0, 255))      # Magenta

        debug.save(path)
