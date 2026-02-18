"""
PBAI Region Reader — MasterFrame-driven HUD extraction.

Receives a MasterFrame (JSON from Pi on connect) and applies it to
native-resolution screenshots to extract structured game state.

Works on the NATIVE resolution image (not 512x512 vision transformer input).
The masterframe's normalized coordinates map to whatever resolution the
image is, keeping hearts at 16x16 pixels (readable) instead of 3x6 (not).

Read dispatch by type:
    pixel    → sample region, compute color ratios / variance / brightness
    ocr      → crop region, run easyocr, parse with expected pattern
    template → future: match against known item textures
    viewport → skip (vision transformer handles this separately)
    composite → skip (aggregated from children)

The reader logic for pixel analysis (red ratio for hearts, food ratio for
hunger) lives HERE, not in masterframe metadata. The masterframe only says
"this region is pixel type." The reader knows how to interpret that.
"""

import logging
import re
import time
import numpy as np
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

# Import MasterFrame types — these come from the Pi codebase but we
# reconstruct from dict, so we only need them for type hints.
# The actual Region/MasterFrame objects are rebuilt from JSON.
try:
    import sys, os
    # Add parent of Windows Client to path for imports
    _parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _parent not in sys.path:
        sys.path.insert(0, _parent)
    from drivers.minecraft.masterframe import MasterFrame, Region, ReadType
    HAS_MASTERFRAME = True
except ImportError:
    HAS_MASTERFRAME = False

try:
    import easyocr
    HAS_EASYOCR = True
except ImportError:
    HAS_EASYOCR = False


class RegionReader:
    """Applies a MasterFrame to native-resolution screenshots.

    Extracts health, hunger, air, hotbar state, and coordinates
    using the region layout from the Pi's MasterFrame.
    """

    def __init__(self, masterframe_dict: dict):
        """Initialize from a masterframe dict (received over WebSocket).

        Args:
            masterframe_dict: MasterFrame.to_dict() output from Pi.
        """
        if HAS_MASTERFRAME:
            self.masterframe = MasterFrame.from_dict(masterframe_dict)
        else:
            # Lightweight fallback: store raw dict and extract regions manually
            self.masterframe = None
            self._raw = masterframe_dict
            self._regions = self._parse_regions(masterframe_dict)

        self._ocr_reader = None
        self._coord_cache: Dict[str, float] = {}
        self._coord_cache_time: float = 0.0

        region_count = len(self._get_all_regions())
        logger.info(f"RegionReader initialized: {region_count} regions")

    def _parse_regions(self, mf_dict: dict) -> Dict[str, dict]:
        """Parse regions from raw masterframe dict (fallback when no import)."""
        regions = {}
        active = mf_dict.get("active_mode", "hud")
        modes = mf_dict.get("modes", {})
        mode_data = modes.get(active, {})
        for name, rd in mode_data.get("regions", {}).items():
            regions[name] = rd
        return regions

    def _get_all_regions(self) -> Dict[str, Any]:
        """Get all regions regardless of import availability."""
        if self.masterframe:
            return self.masterframe.get_all_regions()
        return self._regions

    def _get_region(self, name: str):
        """Get a single region by name."""
        if self.masterframe:
            return self.masterframe.get_region(name)
        return self._regions.get(name)

    def _crop_region(self, image: np.ndarray, region) -> np.ndarray:
        """Crop image to a region's bounds."""
        h, w = image.shape[:2]
        if self.masterframe and hasattr(region, 'crop'):
            return region.crop(image)
        # Dict-based fallback
        rd = region if isinstance(region, dict) else region
        rx = rd["x"] if isinstance(rd, dict) else rd.x
        ry = rd["y"] if isinstance(rd, dict) else rd.y
        rw = rd["w"] if isinstance(rd, dict) else rd.w
        rh = rd["h"] if isinstance(rd, dict) else rd.h
        px = max(0, int(rx * w))
        py = max(0, int(ry * h))
        pw = max(1, int(rw * w))
        ph = max(1, int(rh * h))
        return image[py:min(h, py+ph), px:min(w, px+pw)]

    def _region_read_type(self, region) -> str:
        """Get the read_type string from a region."""
        if hasattr(region, 'read_type'):
            return region.read_type.value if hasattr(region.read_type, 'value') else str(region.read_type)
        if isinstance(region, dict):
            return region.get("read_type", "pixel")
        return "pixel"

    # ═══════════════════════════════════════════════════════════════════════════
    # MAIN API
    # ═══════════════════════════════════════════════════════════════════════════

    def read_all(self, image: np.ndarray) -> dict:
        """Extract all region data from a native-resolution screenshot.

        Returns dict with:
            health: float (0-20, half-hearts)
            hunger: float (0-20, half-shanks)
            air: int (-1 if not underwater, 0-300 otherwise)
            hotbar_selected: int (1-9)
            hotbar_slots: list[bool] (9 elements, True if occupied)
            coordinates: dict with x, y, z (or empty)
            screen_mode: str ("hud")
        """
        result = {}

        # ── Health (hearts) ──
        result["health"] = self._read_health(image)

        # ── Hunger (drumsticks) ──
        result["hunger"] = self._read_hunger(image)

        # ── Air (bubbles) ──
        result["air"] = self._read_air(image)

        # ── Hotbar ──
        result["hotbar_selected"] = self._read_hotbar_selection(image)
        result["hotbar_slots"] = self._read_hotbar_slots(image)

        # ── Coordinates (OCR, cached) ──
        result["coordinates"] = self._read_coordinates(image)

        # ── Screen mode (always HUD for now) ──
        result["screen_mode"] = "hud"

        return result

    # ═══════════════════════════════════════════════════════════════════════════
    # PIXEL READERS
    # ═══════════════════════════════════════════════════════════════════════════

    def _read_health(self, image: np.ndarray) -> float:
        """Count red hearts → health (0-20 half-hearts).

        Full heart = dominant red channel (R>150, G<80, B<80).
        """
        count = 0.0
        for i in range(10):
            region = self._get_region(f"heart_{i}")
            if region is None:
                continue
            crop = self._crop_region(image, region)
            if crop.size == 0:
                continue
            pixels = crop.reshape(-1, 3)
            red_mask = (pixels[:, 0] > 150) & (pixels[:, 1] < 80) & (pixels[:, 2] < 80)
            red_ratio = red_mask.sum() / len(pixels)
            if red_ratio > 0.15:
                count += 2.0 if red_ratio > 0.5 else 1.0
        return min(count, 20.0)

    def _read_hunger(self, image: np.ndarray) -> float:
        """Count drumsticks → hunger (0-20 half-shanks).

        Drumstick: red cap (R>150, G<80, B<80) + brown body.
        """
        count = 0.0
        for i in range(10):
            region = self._get_region(f"hunger_{i}")
            if region is None:
                continue
            crop = self._crop_region(image, region)
            if crop.size == 0:
                continue
            pixels = crop.reshape(-1, 3)
            red_mask = (pixels[:, 0] > 150) & (pixels[:, 1] < 80) & (pixels[:, 2] < 80)
            brown_mask = (
                (pixels[:, 0] > 50) & (pixels[:, 0] < 220) &
                (pixels[:, 1] > 25) & (pixels[:, 1] < 160) &
                (pixels[:, 2] > 10) & (pixels[:, 2] < 80) &
                ~red_mask
            )
            food_ratio = (red_mask | brown_mask).sum() / len(pixels)
            if food_ratio > 0.08:
                count += 2.0 if food_ratio > 0.25 else 1.0
        return min(count, 20.0)

    def _read_air(self, image: np.ndarray) -> int:
        """Count air bubbles → air ticks (0-300, or -1 if not underwater).

        Bubbles: R<100, G>130, B>170. 10 max, each = 30 ticks.
        """
        count = 0
        any_blue = False
        for i in range(10):
            region = self._get_region(f"air_{i}")
            if region is None:
                continue
            crop = self._crop_region(image, region)
            if crop.size == 0:
                continue
            pixels = crop.reshape(-1, 3)
            blue_mask = (pixels[:, 0] < 100) & (pixels[:, 1] > 130) & (pixels[:, 2] > 170)
            blue_ratio = blue_mask.sum() / len(pixels)
            if blue_ratio > 0.3:
                count += 1
                any_blue = True
        return count * 30 if any_blue else -1

    def _read_hotbar_selection(self, image: np.ndarray) -> int:
        """Find selected slot (1-9) by brightness at slot top edge."""
        best_brightness = 0.0
        best_slot = 1
        for i in range(9):
            region = self._get_region(f"hotbar_slot_{i+1}")
            if region is None:
                continue
            crop = self._crop_region(image, region)
            if crop.size == 0:
                continue
            # Sample top 2 rows for selection highlight
            top_strip = crop[:max(1, crop.shape[0] // 6), :]
            if top_strip.size == 0:
                continue
            brightness = float(top_strip.mean())
            if brightness > best_brightness:
                best_brightness = brightness
                best_slot = i + 1
        return best_slot

    def _read_hotbar_slots(self, image: np.ndarray) -> List[bool]:
        """Detect occupied vs empty slots by pixel variance."""
        result = []
        for i in range(9):
            region = self._get_region(f"hotbar_slot_{i+1}")
            if region is None:
                result.append(False)
                continue
            crop = self._crop_region(image, region)
            if crop.size == 0:
                result.append(False)
                continue
            pixels = crop.reshape(-1, 3).astype(float)
            var = pixels.var()
            result.append(bool(var > 200))
        return result

    # ═══════════════════════════════════════════════════════════════════════════
    # OCR READERS
    # ═══════════════════════════════════════════════════════════════════════════

    def _read_coordinates(self, image: np.ndarray) -> Dict[str, float]:
        """OCR the coordinate overlay from the coordinates region.

        Bedrock "Show Coordinates" displays: Position: X, Y, Z
        Returns {"x": float, "y": float, "z": float} or empty dict.
        """
        if not HAS_EASYOCR:
            return {}

        # Cache: easyocr is ~200-500ms, only re-run every 5s
        now = time.monotonic()
        if self._coord_cache and (now - self._coord_cache_time) < 5.0:
            return dict(self._coord_cache)

        region = self._get_region("coordinates")
        if region is None:
            return {}

        try:
            if self._ocr_reader is None:
                self._ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)

            crop = self._crop_region(image, region)
            if crop.size == 0:
                return {}

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
        except Exception as e:
            logger.debug(f"Coordinate OCR error: {e}")

        return {}

    # ═══════════════════════════════════════════════════════════════════════════
    # DEBUG
    # ═══════════════════════════════════════════════════════════════════════════

    def debug_overlay(self, image: np.ndarray, path: str = "masterframe_debug.png"):
        """Draw region rectangles on a screenshot for visual verification.

        Color code by read_type:
            pixel    → green
            ocr      → cyan
            composite → yellow
            viewport → dim gray (outline only)
        """
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            logger.warning("PIL required for debug_overlay")
            return

        debug = Image.fromarray(image.copy())
        draw = ImageDraw.Draw(debug)
        h, w = image.shape[:2]

        colors = {
            "pixel": (0, 255, 0, 128),
            "ocr": (0, 255, 255, 128),
            "composite": (255, 255, 0, 80),
            "viewport": (100, 100, 100, 40),
            "template": (255, 128, 0, 128),
        }

        regions = self._get_all_regions()
        for name, region in regions.items():
            rt = self._region_read_type(region)
            color = colors.get(rt, (255, 255, 255, 128))

            if hasattr(region, 'to_pixels'):
                px, py, pw, ph = region.to_pixels(w, h)
            else:
                rd = region
                px = int(rd["x"] * w)
                py = int(rd["y"] * h)
                pw = max(1, int(rd["w"] * w))
                ph = max(1, int(rd["h"] * h))

            # Skip viewport (covers whole screen)
            if rt == "viewport":
                continue

            outline_color = color[:3]
            draw.rectangle([px, py, px + pw, py + ph], outline=outline_color, width=2)

            # Label (small text)
            try:
                draw.text((px + 2, py + 2), name, fill=outline_color)
            except Exception:
                pass

        debug.save(path)
        logger.info(f"Saved debug overlay: {path} ({len(regions)} regions)")
