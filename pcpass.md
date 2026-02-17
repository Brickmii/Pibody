# PC Pass: What the Pi Brain Needs from the PC Client

## Overview

The PBAI daemon on the Pi runs a thermal manifold cognitive architecture that makes decisions for a Minecraft agent. The PC client (`Windows Client/pbai_client.py`) is the daemon's eyes and hands — it captures screenshots, runs the vision transformer, reads the HUD, and sends `world_state` to the Pi over WebSocket. The Pi then sends back actions for the motor controller to execute.

This document tells PC Claude what the Pi side expects and what's currently broken.

---

## What the Pi Reads from `world_state["player"]`

File: `drivers/minecraft/minecraft_driver.py:956-966`

```python
player = gs.get("player", {})
px = player.get("x", 0.0)
py = player.get("y", 62.0)    # defaults to sea level
pz = player.get("z", 0.0)
yaw = player.get("yaw", 0.0)
health = player.get("health", 20.0)
hunger = player.get("hunger", 20.0)
air = player.get("air", player.get("oxygen", -1))
```

### Required fields in `world_state["player"]`:

| Key | Type | Range | Default | Status |
|-----|------|-------|---------|--------|
| `health` | float | 0-20 (half-hearts) | 20.0 | **BROKEN** — reading 0.0, #1 priority |
| `hunger` | float | 0-20 (half-shanks) | 20.0 | **BROKEN** — reading 0.0, #1 priority |
| `air` | int | 0-300 ticks, -1=not underwater | -1 | DONE — working correctly |
| `x` | float | world coordinate | 0.0 | DONE — easyocr OCR with 5s cache |
| `y` | float | world coordinate | 62.0 | DONE — easyocr OCR with 5s cache |
| `z` | float | world coordinate | 0.0 | DONE — easyocr OCR with 5s cache |
| `yaw` | float | 0-360 degrees | 0.0 | DONE — tracked from mouse deltas in MotorExecutor |
| `pitch` | float | -90 to 90 degrees | 0.0 | DONE — tracked from mouse deltas in MotorExecutor |

### What the Pi reads from `world_state["hotbar"]`:

| Key | Type | Status |
|-----|------|--------|
| `selected` | int (1-9) | Working |
| `slots` | List[bool] (9 items) | Working |

---

## STILL BROKEN: Health and Hunger (Priority #1)

The state key on Pi currently shows `h0` — meaning `int(player.get("health", 20))` is getting `0` from the client. The HUD reader's `_read_health()` and `_read_hunger()` are returning 0.0. This is the last critical fix.

### Why this matters to the Pi brain:

1. **State key**: `ow_plains_h20_e0p0` — health is encoded directly. `h0` means the brain thinks the player is dead every tick. Every decision is made under a false premise.
2. **Drowning detection** (line 976-986): Uses `health < self._last_health` to detect damage underwater. With health stuck at 0, this can't fire.
3. **Conscience reward system**: The brain rewards/penalizes its planning node based on outcomes. Accurate health change = "that was a bad plan" signal. Without it, the conscience can't learn survival.

### Root cause: sample points miss the HUD

The `_compute_regions()` math calculates pixel coordinates for hearts/hunger based on `screen_width`, `screen_height`, and `gui_scale`. If ANY of these don't match the actual game window, every sample lands on background pixels and returns 0.

### Debugging — check `hud_debug.png`

The `debug_save()` overlay should have been saved on first frame. Open it and check:
- **Green dots** = heart sample points. Do they land ON the heart icons?
- **Blue dots** = hunger sample points. Do they land ON the drumstick icons?
- If they're offset, the coordinate math needs adjusting.

### Options to fix:

**Option A: Manual calibration (quickest)**
Open `hud_debug.png`. If dots are close but offset, adjust the constants in `_compute_regions()`:
- `HOTBAR_BOTTOM_OFFSET = 22` — try 21, 23, 24
- `HEART_ROW_OFFSET = 10` — try 9, 11, 12
- `HEART_LEFT_INSET = 9` — try 8, 10
- `HUNGER_RIGHT_INSET = 8` — try 7, 9

Verify the constructor is called with the correct values:
- `screen_width` and `screen_height` must match the actual capture resolution
- `gui_scale` must match Minecraft's GUI scale setting (check Settings > Video > GUI Scale)

**Option B: Auto-calibration (robust)**
Instead of fixed offsets, scan for the hotbar programmatically:
1. The hotbar has a distinctive dark border (~RGB 0,0,0) with grey interior (~RGB 139,139,139)
2. Search the bottom 100px of the screen for a horizontal band matching this pattern
3. Once hotbar is found, hearts are always a fixed offset above it
4. Cache the found positions — they don't change unless resolution changes

**Option C: Template matching (most reliable)**
Use a small template image of a full heart (9x9 px at gui_scale=1) and `numpy` cross-correlation to find all 10 heart positions in the bottom quarter of the screen. Same for hunger drumsticks. This is resolution/scale independent.

### Key questions to answer:

1. What does `hud_debug.png` look like? Are the dots close or way off?
2. What resolution is the game capture returning? (log `image.shape` in `read_hud()`)
3. What `gui_scale` is the `HUDReader` being constructed with? What does Minecraft have set?
4. Is the game fullscreen or windowed? (windowed may have title bar offset)

---

## DONE: Position Data (x, y, z, yaw, pitch)

All resolved in commit `ece64dd`:
- **x/y/z**: easyocr OCR on "Show Coordinates" overlay, cached 5s. Working if easyocr is installed and coordinates are enabled in Minecraft settings.
- **yaw/pitch**: Estimated from mouse movement deltas in `MotorExecutor` (`dx/7.0` sensitivity). Not pixel-perfect but gives the Pi real directional awareness.

Just confirm:
1. easyocr is installed (`pip install easyocr`)
2. "Show Coordinates" is ON in Minecraft Bedrock settings (Game > Show Coordinates)
3. Check the client log for `HUD:` lines — do x/y/z appear?

---

## What's Working

- **Air bubbles**: Returns -1 when not underwater (correct), 0-300 when submerged
- **Hotbar selected**: Reads which slot (1-9) is active
- **Hotbar slots**: Detects which slots have items
- **Position x/y/z**: easyocr coordinate OCR with 5s cache (commit `ece64dd`)
- **Yaw/pitch**: Mouse-delta tracking in MotorExecutor (commit `ece64dd`)
- **Vision transformer**: Scene classification, entity detection, peaks — all flowing correctly
- **Action execution**: Motor controller executing moves, attacks, jumps, mining

---

## Client Code Reference

The HUD merge happens in two places (both now include yaw/pitch):

### Request path (`pbai_client.py:536-560`):
```python
world_state["player"] = {
    "health": hud_data["health"],      # <-- returns 0.0 (BROKEN)
    "hunger": hud_data["hunger"],      # <-- returns 0.0 (BROKEN)
    "air": hud_data.get("air", -1),    # working
    "yaw": self.motor.yaw,             # working (mouse-delta)
    "pitch": self.motor.pitch,         # working (mouse-delta)
    **hud_data.get("coordinates", {}), # x, y, z from easyocr
}
```

### Stream path (`pbai_client.py:659-665`):
Same structure, in `_transform_image()`.

### Quick diagnostic:
Add a one-time log of the raw pixel values at heart positions to confirm the sample points are wrong:
```python
# In _read_health(), add temporarily:
if not hasattr(self, '_health_debug_logged'):
    for i, (cx, cy) in enumerate(self._heart_centers[:3]):
        pixels = self._sample_region(image, cx, cy)
        print(f"Heart {i} at ({cx},{cy}): mean RGB = {pixels.mean(axis=0)}")
    print(f"Image shape: {image.shape}")
    self._health_debug_logged = True
```
If mean RGB is all grey/dark/sky-colored, the sample points are in the wrong place.

---

## Priority

1. **Health + Hunger** (HIGH) — The only remaining broken thing. Fix the HUD coordinate calibration so `_read_health()` and `_read_hunger()` return real values. See options A/B/C above.
2. ~~**Position x/y/z**~~ DONE
3. ~~**Yaw/pitch**~~ DONE

---

## Verification

After fixes, the Pi logs should show:
```
state=ow_plains_h20_e0p0    (full health, not h0)
```

Check from Pi:
```bash
curl -s localhost:8420/status | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['psychology'])"
journalctl -u pbai-daemon -f | grep state=
```

The HUD log line in the client should show non-zero health:
```
HUD: health=20.0 hunger=20.0 air=-1
```
