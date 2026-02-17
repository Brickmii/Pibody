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
        """Find hearts/hunger by scanning for red pixel clusters in the HUD.

        The Bedrock Edition HUD constants vary with resolution and GUI slider,
        so instead of guessing gui_scale we scan the bottom of the screen for
        red pixel clusters (hearts on the left, drumstick red on the right),
        derive gui_scale from the measured spacing, and set positions directly.
        """
        h, w = image.shape[:2]

        # Scan bottom 25% for red pixels (hearts + drumstick outlines)
        scan_top = int(h * 0.75)
        bottom = image[scan_top:]
        red_mask = (bottom[:, :, 0] > 150) & (bottom[:, :, 1] < 80) & (bottom[:, :, 2] < 80)
        ys, xs = np.where(red_mask)

        if len(ys) < 50:
            logger.warning("Auto-calibrate: only %d red pixels in bottom 25%%, retry next frame", len(ys))
            return

        # Find the row with the most red pixels (peak of the heart row)
        unique_ys, counts = np.unique(ys, return_counts=True)
        peak_local_y = int(unique_ys[counts.argmax()])
        peak_y = peak_local_y + scan_top

        # Get X positions on that row and cluster them (gap > 3px = new icon)
        peak_xs = sorted(xs[ys == peak_local_y].tolist())
        clusters: List[int] = []
        start = peak_xs[0]
        prev = peak_xs[0]
        for x in peak_xs[1:]:
            if x - prev > 3:
                clusters.append((start + prev) // 2)
                start = x
            prev = x
        clusters.append((start + prev) // 2)

        if len(clusters) < 10:
            logger.warning("Auto-calibrate: only %d red clusters (need >=10), retry", len(clusters))
            return

        # Split into hearts (left) and hunger (right) at the biggest gap
        gaps = [(clusters[i + 1] - clusters[i], i) for i in range(len(clusters) - 1)]
        max_gap_val, max_gap_idx = max(gaps, key=lambda g: g[0])

        heart_xs = clusters[:max_gap_idx + 1]
        hunger_xs = clusters[max_gap_idx + 1:]

        if len(heart_xs) < 3:
            logger.warning("Auto-calibrate: only %d heart clusters, retry", len(heart_xs))
            return

        # Derive gui_scale from heart spacing  (HEART_SPACING = 8 gui-px)
        spacings = [heart_xs[i + 1] - heart_xs[i] for i in range(len(heart_xs) - 1)]
        avg_spacing = sum(spacings) / len(spacings)
        gs = max(1, round(avg_spacing / self.HEART_SPACING))
        self.gs = gs

        # ── Set heart centres directly from scan ──
        self._heart_centers = [(x, peak_y) for x in heart_xs[:10]]

        # ── Set hunger centres directly from scan (rightmost first) ──
        if len(hunger_xs) >= 3:
            self._hunger_centers = [(x, peak_y) for x in reversed(hunger_xs[:10])]
        else:
            # Fallback: mirror hearts across screen centre
            cx = w // 2
            self._hunger_centers = [(2 * cx - x, peak_y) for x in reversed(heart_xs[:10])]

        # ── Recompute hotbar slots from gs (HOTBAR_BOTTOM_OFFSET=22 is reliable) ──
        sw, sh = self.sw, self.sh
        slot_size = 20 * gs
        slot_gap = 2 * gs
        slot_stride = slot_size + slot_gap
        hotbar_width = 9 * slot_size + 8 * slot_gap
        hotbar_left = sw // 2 - hotbar_width // 2
        hotbar_top = sh - self.HOTBAR_BOTTOM_OFFSET * gs

        self._slot_centers = []
        for i in range(9):
            sx = hotbar_left + i * slot_stride + slot_size // 2
            sy = hotbar_top + slot_size // 2
            self._slot_centers.append((sx, sy))

        self._slot_top_edges = []
        for i in range(9):
            sx = hotbar_left + i * slot_stride + slot_size // 2
            sy = hotbar_top - gs
            self._slot_top_edges.append((sx, sy))

        # ── Air bubbles: same X as hunger, offset upward ──
        bubble_y = peak_y - self.AIR_ROW_OFFSET * gs
        self._bubble_centers = [(bx, bubble_y) for bx, _ in self._hunger_centers]

        self._sample_r = max(3, gs * 2)
        self._calibrated = True
        self._force_debug_resave = True

        logger.info("Auto-calibrated gui_scale=%d (spacing=%.1fpx, hearts=%d, hunger=%d)",
                    gs, avg_spacing, len(heart_xs), len(hunger_xs))
        logger.info("  heart_y=%d, first_heart_x=%d, image=%dx%d",
                    peak_y, heart_xs[0], w, h)
        for i, (cx, cy) in enumerate(self._heart_centers[:3]):
            if 0 <= cy < h and 0 <= cx < w:
                r, g, b = image[cy, cx]
                logger.info("  heart[%d] (%d,%d) RGB=(%d,%d,%d)", i, cx, cy, r, g, b)

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

        Drumstick has a red cap (R>150, G<80, B<80) and a brown/dark-brown
        body.  We detect both since the sample region straddles the cap and
        body at large GUI scales.
        """
        count = 0.0
        for cx, cy in self._hunger_centers:
            pixels = self._sample_region(image, cx, cy)
            if len(pixels) == 0:
                continue
            # Red cap of drumstick
            red_mask = (pixels[:, 0] > 150) & (pixels[:, 1] < 80) & (pixels[:, 2] < 80)
            # Brown body (original range + darker browns at high GUI scales)
            brown_mask = (
                (pixels[:, 0] > 50) & (pixels[:, 0] < 220) &
                (pixels[:, 1] > 25) & (pixels[:, 1] < 160) &
                (pixels[:, 2] > 10) & (pixels[:, 2] < 80) &
                ~red_mask  # avoid double-counting
            )
            food_ratio = (red_mask | brown_mask).sum() / len(pixels)
            if food_ratio > 0.25:
                count += 2.0   # Full drumstick
            elif food_ratio > 0.08:
                count += 1.0   # Half drumstick
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
