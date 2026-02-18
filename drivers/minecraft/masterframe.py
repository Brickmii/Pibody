"""
MasterFrame — Spatial Knowledge of the Minecraft Screen

The game board: pure geometry of where every UI element lives on screen.
No behavioral logic, no detection algorithms. Just "hearts are HERE,
hotbar is HERE, coordinates are HERE."

Normalized coordinates (0.0-1.0) so it scales to any resolution.
HUD mode only for now.

Measured from Bedrock Edition at 2560x1440, GUI scale 2.

COORDINATE SYSTEM:
    (0.0, 0.0) = top-left of screen
    (1.0, 1.0) = bottom-right of screen
    All regions stored as (x, y, w, h) normalized

READ TYPES:
    pixel    — Color-based reading (hearts, hunger, slot occupancy)
    ocr      — Text reading (coordinates, chat)
    template — Match known textures (future: item identification)
    viewport — 3D world (vision transformer handles this)
    composite — Contains sub-regions, not read directly
"""

import json
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)


class ReadType(Enum):
    PIXEL = "pixel"
    OCR = "ocr"
    TEMPLATE = "template"
    VIEWPORT = "viewport"
    COMPOSITE = "composite"


class ScreenMode(Enum):
    HUD = "hud"


@dataclass
class Region:
    """A named rectangular area on the screen."""
    name: str
    x: float       # Left edge, normalized 0.0-1.0
    y: float       # Top edge, normalized 0.0-1.0
    w: float       # Width, normalized
    h: float       # Height, normalized
    read_type: ReadType
    parent: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        return (self.x, self.y, self.w, self.h)

    def to_pixels(self, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
        """Map normalized coords to pixel coordinates.

        Returns (px, py, pw, ph) — top-left corner + size in pixels.
        """
        px = int(self.x * img_w)
        py = int(self.y * img_h)
        pw = max(1, int(self.w * img_w))
        ph = max(1, int(self.h * img_h))
        return (px, py, pw, ph)

    def crop(self, image) -> Any:
        """Extract this region from an image array (H, W, 3).

        Returns the cropped sub-array.
        """
        h, w = image.shape[:2]
        px, py, pw, ph = self.to_pixels(w, h)
        # Clamp to image bounds
        x0 = max(0, px)
        y0 = max(0, py)
        x1 = min(w, px + pw)
        y1 = min(h, py + ph)
        return image[y0:y1, x0:x1]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "x": self.x,
            "y": self.y,
            "w": self.w,
            "h": self.h,
            "read_type": self.read_type.value,
            "parent": self.parent,
            "metadata": self.metadata,
        }

    @staticmethod
    def from_dict(d: dict) -> 'Region':
        return Region(
            name=d["name"],
            x=d["x"],
            y=d["y"],
            w=d["w"],
            h=d["h"],
            read_type=ReadType(d["read_type"]),
            parent=d.get("parent"),
            metadata=d.get("metadata", {}),
        )


@dataclass
class ModeLayout:
    """All regions for a given screen mode."""
    mode: ScreenMode
    regions: Dict[str, Region] = field(default_factory=dict)

    def add(self, region: Region):
        self.regions[region.name] = region

    def to_dict(self) -> dict:
        return {
            "mode": self.mode.value,
            "regions": {k: v.to_dict() for k, v in self.regions.items()},
        }

    @staticmethod
    def from_dict(d: dict) -> 'ModeLayout':
        layout = ModeLayout(mode=ScreenMode(d["mode"]))
        for name, rd in d.get("regions", {}).items():
            layout.regions[name] = Region.from_dict(rd)
        return layout


class MasterFrame:
    """Spatial knowledge of the Minecraft screen. The board, not the rulebook.

    Contains named regions with normalized (0-1) coordinates for every
    UI element. Modes define different layouts (HUD, inventory, etc).
    Currently only HUD mode is implemented.
    """

    def __init__(self):
        self.modes: Dict[ScreenMode, ModeLayout] = {}
        self._active_mode = ScreenMode.HUD
        self._build_hud_mode()

    @property
    def active_mode(self) -> ScreenMode:
        return self._active_mode

    @active_mode.setter
    def active_mode(self, mode: ScreenMode):
        if mode in self.modes:
            self._active_mode = mode
        else:
            logger.warning(f"Mode {mode} not defined, keeping {self._active_mode}")

    def get_region(self, name: str) -> Optional[Region]:
        """Look up a region by name in the active mode."""
        layout = self.modes.get(self._active_mode)
        if layout:
            return layout.regions.get(name)
        return None

    def get_regions_by_type(self, read_type: ReadType) -> List[Region]:
        """All regions needing a specific read type in the active mode."""
        layout = self.modes.get(self._active_mode)
        if not layout:
            return []
        return [r for r in layout.regions.values() if r.read_type == read_type]

    def get_all_regions(self) -> Dict[str, Region]:
        """All regions in the active mode."""
        layout = self.modes.get(self._active_mode)
        if layout:
            return dict(layout.regions)
        return {}

    # ═══════════════════════════════════════════════════════════════════════════
    # HUD MODE LAYOUT — Bedrock Edition
    # ═══════════════════════════════════════════════════════════════════════════
    #
    # Reference: 2560x1440, GUI scale 2
    #
    # Bedrock uses gui_scale multiplier for all HUD elements.
    # At gs=2 on 2560x1440:
    #   Slot size = 20 * 2 = 40px native → 40/2560 = 0.015625 normalized width
    #   Slot gap  = 2 * 2  = 4px  native
    #   Slot stride = 44px → 44/2560 = 0.017188
    #   Hotbar width = 9*40 + 8*4 = 392px → 392/2560 = 0.153125
    #   Hotbar left = (2560 - 392) / 2 = 1084 → 1084/2560 = 0.423438
    #   Hotbar top = (1440 - 22*2) / 1440 = 1396/1440 = 0.969444
    #   Heart spacing = 8 * 2 = 16px → 16/2560 = 0.00625
    #   Heart icon ~= 9 * 2 = 18px wide → 18/2560 = 0.007031 (w), 18/1440 = 0.0125 (h)
    #   Heart row y = hotbar_top - 10*2 = 1396 - 20 = 1376 → 1376/1440 = 0.955556
    #   First heart x = hotbar_left + 9*2 = 1084 + 18 = 1102 → 1102/2560 = 0.430469

    def _build_hud_mode(self):
        """Build HUD mode with all Bedrock Edition regions."""
        layout = ModeLayout(mode=ScreenMode.HUD)

        # ── Viewport (full screen, vision transformer) ──
        layout.add(Region("viewport", 0.0, 0.0, 1.0, 1.0, ReadType.VIEWPORT))

        # ── Crosshair (dead center) ──
        layout.add(Region("crosshair", 0.4922, 0.4931, 0.0156, 0.0139, ReadType.PIXEL))

        # ── Hotbar ──
        # Reference: 2560x1440, gs=2
        # hotbar_left = 1084, hotbar_top = 1396
        # 9 slots, each 40px wide, 4px gap
        hotbar_x = 0.4234
        hotbar_y = 0.9694
        hotbar_w = 0.1531
        hotbar_h = 0.0278  # 40/1440
        slot_w = 0.015625  # 40/2560
        slot_h = 0.027778  # 40/1440
        slot_stride = 0.017188  # 44/2560

        layout.add(Region("hotbar", hotbar_x, hotbar_y, hotbar_w, hotbar_h,
                          ReadType.COMPOSITE, metadata={"slot_count": 9}))

        for i in range(9):
            sx = hotbar_x + i * slot_stride
            layout.add(Region(f"hotbar_slot_{i+1}", sx, hotbar_y, slot_w, slot_h,
                              ReadType.PIXEL, parent="hotbar",
                              metadata={"slot_index": i}))

        # ── Health bar (hearts) ──
        # Heart row: 10 hearts above hotbar left
        # heart_y = hotbar_top - 10*gs = 1396 - 20 = 1376 → 0.9556
        # first_heart_x = hotbar_left + 9*gs = 1084 + 18 = 1102 → 0.4305
        heart_spacing = 0.00625   # 16px / 2560
        heart_w = 0.007031        # ~18px / 2560
        heart_h = 0.012500        # ~18px / 1440
        heart_y = 0.9556
        first_heart_x = 0.4305

        # Composite for the whole bar
        health_bar_w = 9 * heart_spacing + heart_w
        layout.add(Region("health_bar", first_heart_x, heart_y,
                          health_bar_w, heart_h, ReadType.COMPOSITE,
                          metadata={"icon_count": 10}))

        for i in range(10):
            hx = first_heart_x + i * heart_spacing
            layout.add(Region(f"heart_{i}", hx, heart_y, heart_w, heart_h,
                              ReadType.PIXEL, parent="health_bar",
                              metadata={"heart_index": i}))

        # ── Hunger bar (drumsticks) ──
        # Mirrored from hearts on the right side of hotbar
        # hunger starts from hotbar right edge minus inset, goes right-to-left
        # hotbar_right = hotbar_x + hotbar_w = 0.4234 + 0.1531 = 0.5765
        # first_hunger_x (rightmost) = hotbar_right - 8*gs/2560 = 0.5765 - 0.00625 = 0.5703
        # But hunger icons go right-to-left, so hunger_0 is rightmost
        hotbar_right = hotbar_x + hotbar_w
        first_hunger_x = hotbar_right - 0.00625  # inset from right edge

        hunger_bar_w = 9 * heart_spacing + heart_w
        hunger_bar_x = first_hunger_x - 9 * heart_spacing
        layout.add(Region("hunger_bar", hunger_bar_x, heart_y,
                          hunger_bar_w, heart_h, ReadType.COMPOSITE,
                          metadata={"icon_count": 10}))

        for i in range(10):
            # hunger_0 = rightmost, hunger_9 = leftmost
            hx = first_hunger_x - i * heart_spacing
            layout.add(Region(f"hunger_{i}", hx, heart_y, heart_w, heart_h,
                              ReadType.PIXEL, parent="hunger_bar",
                              metadata={"hunger_index": i}))

        # ── Air bar (bubbles above hunger, only visible underwater) ──
        # air_y = heart_y - 10*gs/1440 = 0.9556 - 0.01389 = 0.9417
        air_y = heart_y - 0.01389
        air_bar_x = hunger_bar_x
        layout.add(Region("air_bar", air_bar_x, air_y,
                          hunger_bar_w, heart_h, ReadType.COMPOSITE,
                          metadata={"icon_count": 10}))

        for i in range(10):
            bx = first_hunger_x - i * heart_spacing
            layout.add(Region(f"air_{i}", bx, air_y, heart_w, heart_h,
                              ReadType.PIXEL, parent="air_bar",
                              metadata={"bubble_index": i}))

        # ── Coordinates (top-left "Position: X, Y, Z") ──
        layout.add(Region("coordinates", 0.0, 0.0, 0.156, 0.069,
                          ReadType.OCR, metadata={"pattern": "Position: X, Y, Z"}))

        # ── Chat history (left side) ──
        layout.add(Region("chat_history", 0.0, 0.556, 0.313, 0.278,
                          ReadType.OCR, metadata={"multiline": True}))

        self.modes[ScreenMode.HUD] = layout
        logger.info(f"MasterFrame HUD mode: {len(layout.regions)} regions")

    # ═══════════════════════════════════════════════════════════════════════════
    # SERIALIZATION
    # ═══════════════════════════════════════════════════════════════════════════

    def to_dict(self) -> dict:
        return {
            "active_mode": self._active_mode.value,
            "modes": {m.value: l.to_dict() for m, l in self.modes.items()},
        }

    @staticmethod
    def from_dict(d: dict) -> 'MasterFrame':
        mf = MasterFrame.__new__(MasterFrame)
        mf.modes = {}
        for mode_str, layout_dict in d.get("modes", {}).items():
            mode = ScreenMode(mode_str)
            mf.modes[mode] = ModeLayout.from_dict(layout_dict)
        mf._active_mode = ScreenMode(d.get("active_mode", "hud"))
        return mf

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @staticmethod
    def from_json(s: str) -> 'MasterFrame':
        return MasterFrame.from_dict(json.loads(s))
