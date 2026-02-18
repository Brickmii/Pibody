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

        # Auto-calibration: MasterFrame positions assume gs=2 but actual
        # GUI scale varies. On first frame we scan for hearts to find the
        # real positions and rescale all HUD regions.
        self._calibrated = False
        self._force_debug_resave = False

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
    # AUTO-CALIBRATION
    # ═══════════════════════════════════════════════════════════════════════════

    def _update_region(self, name: str, x: float, y: float, w: float, h: float):
        """Update a region's normalized bounds (works with both backends)."""
        if self.masterframe:
            region = self.masterframe.get_region(name)
            if region:
                region.x, region.y, region.w, region.h = x, y, w, h
        elif name in self._regions:
            r = self._regions[name]
            r["x"], r["y"], r["w"], r["h"] = x, y, w, h

    def _auto_calibrate(self, image: np.ndarray) -> None:
        """Scan for red pixel clusters to find actual heart/hunger positions.

        The MasterFrame ships with positions measured at gs=2 on 2560x1440,
        but the actual GUI scale varies. We scan for red pixel clusters
        (heart icons + drumstick red caps), derive the real scale, and
        update every HUD region to match.
        """
        h, w = image.shape[:2]

        # Scan bottom 25% for red pixels
        scan_top = int(h * 0.75)
        bottom = image[scan_top:]
        red_mask = (bottom[:, :, 0] > 150) & (bottom[:, :, 1] < 80) & (bottom[:, :, 2] < 80)
        ys, xs = np.where(red_mask)

        if len(ys) < 50:
            logger.warning("RegionReader calibrate: only %d red pixels, retry next frame", len(ys))
            return

        # Peak red row
        unique_ys, counts = np.unique(ys, return_counts=True)
        peak_local_y = int(unique_ys[counts.argmax()])
        peak_y = peak_local_y + scan_top

        # Cluster X positions on peak row
        peak_xs = sorted(xs[ys == peak_local_y].tolist())
        clusters: List[int] = []
        start = peak_xs[0]
        prev = peak_xs[0]
        for x_val in peak_xs[1:]:
            if x_val - prev > 3:
                clusters.append((start + prev) // 2)
                start = x_val
            prev = x_val
        clusters.append((start + prev) // 2)

        if len(clusters) < 10:
            logger.warning("RegionReader calibrate: only %d clusters, retry", len(clusters))
            return

        # Split hearts (left) / hunger (right) at biggest gap
        gaps = [(clusters[i + 1] - clusters[i], i) for i in range(len(clusters) - 1)]
        _, max_gap_idx = max(gaps, key=lambda g: g[0])
        heart_xs = clusters[:max_gap_idx + 1]
        hunger_xs = clusters[max_gap_idx + 1:]

        if len(heart_xs) < 3:
            logger.warning("RegionReader calibrate: only %d heart clusters, retry", len(heart_xs))
            return

        # Derive gui_scale from heart spacing (HEART_SPACING = 8 gui-px)
        spacings = [heart_xs[i + 1] - heart_xs[i] for i in range(len(heart_xs) - 1)]
        avg_spacing = sum(spacings) / len(spacings)
        gs = max(1, round(avg_spacing / 8))

        # Icon size in pixels: 9 gui-px * gs
        icon_px = 9 * gs
        icon_w = icon_px / w
        icon_h = icon_px / h
        spacing_norm = avg_spacing / w

        # Normalized heart Y (icon top = peak center - half icon height)
        heart_y_norm = (peak_y - icon_px // 2) / h

        # ── Update heart regions ──
        for i in range(10):
            if i < len(heart_xs):
                hx_norm = (heart_xs[i] - icon_px // 2) / w
            else:
                # Extrapolate from spacing
                hx_norm = (heart_xs[0] - icon_px // 2) / w + i * spacing_norm
            self._update_region(f"heart_{i}", hx_norm, heart_y_norm, icon_w, icon_h)

        # Health bar composite
        bar_x = (heart_xs[0] - icon_px // 2) / w
        bar_w = ((heart_xs[min(9, len(heart_xs) - 1)] + icon_px // 2) / w) - bar_x
        self._update_region("health_bar", bar_x, heart_y_norm, bar_w, icon_h)

        # ── Update hunger regions (rightmost first) ──
        if len(hunger_xs) >= 3:
            hunger_rev = list(reversed(hunger_xs[:10]))  # hunger_0 = rightmost
            for i in range(10):
                if i < len(hunger_rev):
                    hx_norm = (hunger_rev[i] - icon_px // 2) / w
                else:
                    hx_norm = (hunger_rev[0] - icon_px // 2) / w - i * spacing_norm
                self._update_region(f"hunger_{i}", hx_norm, heart_y_norm, icon_w, icon_h)
            # Hunger bar composite
            leftmost = min(hunger_xs)
            rightmost = max(hunger_xs)
            hbar_x = (leftmost - icon_px // 2) / w
            hbar_w = ((rightmost + icon_px // 2) / w) - hbar_x
            self._update_region("hunger_bar", hbar_x, heart_y_norm, hbar_w, icon_h)
        else:
            # Mirror hearts across screen centre
            for i in range(10):
                if i < len(heart_xs):
                    mirrored_px = w - heart_xs[i]
                else:
                    mirrored_px = w - (heart_xs[0] + i * int(avg_spacing))
                hx_norm = (mirrored_px - icon_px // 2) / w
                self._update_region(f"hunger_{i}", hx_norm, heart_y_norm, icon_w, icon_h)

        # ── Update air bubble regions (same X as hunger, offset upward) ──
        air_offset = 10 * gs  # AIR_ROW_OFFSET * gs
        air_y_norm = (peak_y - air_offset - icon_px // 2) / h
        for i in range(10):
            hunger_reg = self._get_region(f"hunger_{i}")
            if hunger_reg:
                hx = hunger_reg.x if hasattr(hunger_reg, 'x') else hunger_reg["x"]
                self._update_region(f"air_{i}", hx, air_y_norm, icon_w, icon_h)
        # Air bar composite
        hunger_bar = self._get_region("hunger_bar")
        if hunger_bar:
            hbx = hunger_bar.x if hasattr(hunger_bar, 'x') else hunger_bar["x"]
            hbw = hunger_bar.w if hasattr(hunger_bar, 'w') else hunger_bar["w"]
            self._update_region("air_bar", hbx, air_y_norm, hbw, icon_h)

        # ── Update hotbar regions ──
        slot_size = 20 * gs
        slot_gap = 2 * gs
        slot_stride = slot_size + slot_gap
        hotbar_width = 9 * slot_size + 8 * slot_gap
        hotbar_left = w // 2 - hotbar_width // 2
        hotbar_top = h - 22 * gs  # HOTBAR_BOTTOM_OFFSET * gs

        hotbar_x_norm = hotbar_left / w
        hotbar_y_norm = hotbar_top / h
        hotbar_w_norm = hotbar_width / w
        slot_w_norm = slot_size / w
        slot_h_norm = slot_size / h

        self._update_region("hotbar", hotbar_x_norm, hotbar_y_norm,
                            hotbar_w_norm, slot_h_norm)
        for i in range(9):
            sx = hotbar_left + i * slot_stride
            self._update_region(f"hotbar_slot_{i + 1}",
                                sx / w, hotbar_y_norm, slot_w_norm, slot_h_norm)

        self._calibrated = True
        self._force_debug_resave = True

        logger.info("RegionReader auto-calibrated: gui_scale=%d (spacing=%.1fpx, hearts=%d, hunger=%d)",
                    gs, avg_spacing, len(heart_xs), len(hunger_xs))
        logger.info("  heart_y=%d, first_heart_x=%d, image=%dx%d",
                    peak_y, heart_xs[0], w, h)
        for i, hx in enumerate(heart_xs[:3]):
            if 0 <= peak_y < h and 0 <= hx < w:
                r, g, b = image[peak_y, hx]
                logger.info("  heart[%d] (%d,%d) RGB=(%d,%d,%d)", i, hx, peak_y, r, g, b)

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
        if not self._calibrated:
            self._auto_calibrate(image)

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
        The heart shape fills ~40% of its bounding box, so thresholds
        are lower than a centre-point sample.
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
            if red_ratio > 0.30:
                count += 2.0   # Full heart (~40% of bounding box)
            elif red_ratio > 0.12:
                count += 1.0   # Half heart (~20%)
        return min(count, 20.0)

    def _read_hunger(self, image: np.ndarray) -> float:
        """Count drumsticks → hunger (0-20 half-shanks).

        Drumstick: red cap (R>150, G<80, B<80) + brown body.
        The drumstick shape fills ~19% of its bounding box, so
        thresholds are lower than a centre-point sample.
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
            if food_ratio > 0.12:
                count += 2.0   # Full drumstick (~19% of bounding box)
            elif food_ratio > 0.05:
                count += 1.0   # Half drumstick (~10%)
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
