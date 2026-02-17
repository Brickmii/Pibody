# Vision Pass: Fix HUD Reader — Health, Hunger, Oxygen

## Problem

The Pi daemon's state key shows `h0` (health=0) on every tick. The `HUDReader` in `Windows Client/hud_reader.py` is either failing silently or returning 0 for health/hunger. The driver defaults to `player.get("health", 20.0)` but gets `0` from the client, meaning HUDReader runs but reads wrong values.

The daemon uses health for:
- State key encoding (`h{health}` in decision chain)
- Drowning detection (health drop + below sea level)
- Future: survival decisions (eat, flee, heal)

## Root Cause (Likely)

`HUDReader._read_health()` checks for red pixels (`R>180, G<60, B<60`) in 10 heart icon positions. If the pixel coordinates are wrong for the current resolution/GUI scale, it samples background pixels and returns 0.

The coordinate math in `_compute_regions()` assumes Bedrock Edition HUD layout with specific offsets. If GUI scale doesn't match, or the game window isn't fullscreen, every sample misses the hearts entirely.

## Files

| File | Role |
|------|------|
| `Windows Client/hud_reader.py` | HUD pixel reader (health, hunger, hotbar) |
| `Windows Client/pbai_client.py:532-545` | Request path — merges HUD into world_state |
| `Windows Client/pbai_client.py:604-639` | Stream path — merges HUD into world_state |

## What To Fix

### 1. Debug the HUD coordinates

Add a one-time diagnostic that saves a screenshot with the sample points overlaid:

```python
def debug_save(self, image: np.ndarray, path: str = "hud_debug.png"):
    """Save screenshot with sample points marked for calibration."""
    import copy
    debug = image.copy()
    # Mark heart positions in green
    for cx, cy in self._heart_centers:
        debug[max(0,cy-2):cy+3, max(0,cx-2):cx+3] = [0, 255, 0]
    # Mark hunger positions in blue
    for cx, cy in self._hunger_centers:
        debug[max(0,cy-2):cy+3, max(0,cx-2):cx+3] = [0, 0, 255]
    # Mark hotbar slot centers in yellow
    for cx, cy in self._slot_centers:
        debug[max(0,cy-2):cy+3, max(0,cx-2):cx+3] = [255, 255, 0]
    from PIL import Image
    Image.fromarray(debug).save(path)
```

Call it once on first frame in `pbai_client.py` to verify coordinates land on the HUD elements. If they don't, adjust the offsets in `_compute_regions()`.

### 2. Fix coordinate math if needed

The current HUD layout constants:
- `hotbar_top = sh - 22 * gs` (44px from bottom at scale 2)
- `heart_y = hotbar_top - 10 * gs` (20px above hotbar)
- `heart_spacing = 8 * gs`
- `heart_start_x = hotbar_left + 8 * gs`

These may not match the actual Bedrock Edition layout at the current resolution and GUI scale. Cross-reference against an actual screenshot.

Key variables to check:
- What resolution is the game running at?
- What is `--gui-scale` set to? (default 2)
- Is the game fullscreen or windowed?

### 3. Fix color thresholds

Heart detection: `R>180, G<60, B<60` — this is strict. Minecraft hearts can have:
- Full heart: bright red
- Half heart: partial red
- Damaged/flashing: lighter red, pink tint
- Absorption (golden): yellow hearts
- Poison: green-tinted hearts
- Wither: dark hearts

Consider loosening thresholds or adding a broader red-ish detection.

Hunger detection: `R 140-200, G 90-140, B 20-70` — drumstick brown. Same issue with hunger effects (hunger effect makes them green-tinted).

### 4. Add air/oxygen reading

The daemon already handles oxygen at `minecraft_driver.py:978`:
```python
air = player.get("air", player.get("oxygen", -1))
```

But the client never sends it. Add bubble detection to HUDReader:
- Air bubbles appear above hunger bar when underwater
- Blue bubble color: approximately `R<100, G<150, B>180`
- 10 bubbles max, same spacing as hearts
- Pop from right to left as air decreases
- Each bubble = 30 air ticks, max 300

Add to `world_state["player"]`:
```python
"air": hud_data.get("air", 300)  # 300 = full, -1 = not underwater
```

### 5. Parse and send player position (x, y, z, yaw, pitch)

The on-screen position data (F3 debug or overlay) needs to be read and included in the `player` dict. The Pi driver already expects it at `minecraft_driver.py:956-966`:
```python
px = player.get("x", 0.0)
py = player.get("y", MC_Y_SEA)
pz = player.get("z", 0.0)
yaw = player.get("yaw", 0.0)
pitch = player.get("pitch", 0.0)
```

Currently the client never sends these — they all fall back to defaults (x=0, y=62, z=0, yaw=0). The driver uses position for:
- Hypersphere mapping (`mc_altitude_to_theta`, `mc_yaw_to_phi`)
- Drowning detection (y < sea level)
- State key biome/dimension encoding

Options:
- **OCR the debug overlay** — if coordinates are displayed on screen, use OCR (pytesseract or easyocr) to read the x/y/z text
- **Read from game memory/API** — if a mod or companion app exposes coordinates
- **Parse from F3 screen** — if F3 is toggled on, the coordinates are in a known screen position

Add to `world_state["player"]` alongside health/hunger:
```python
world_state["player"] = {
    "health": hud_data["health"],
    "hunger": hud_data["hunger"],
    "x": position_x,
    "y": position_y,
    "z": position_z,
    "yaw": yaw,
    "pitch": pitch,
}
```

### 6. Log HUD values for verification

In `pbai_client.py`, change the HUD debug log (currently `logger.debug`) to `logger.info` temporarily so you can see what values are being read:

```python
except Exception as e:
    logger.info(f"HUD read: health={hud_data.get('health', '?')} hunger={hud_data.get('hunger', '?')}")
```

## Verification

After fixing, the Pi daemon logs should show state keys with real health:
```
state=ow_plains_h20_e0p0   (full health)
state=ow_plains_h16_e1p0   (took some damage, hostile nearby)
```

Check from Pi side:
```bash
curl -s localhost:8420/status | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['psychology'])"
journalctl -u pbai-daemon -f | grep state=
```

## Priority

Medium — the bot functions without health data but can't make survival decisions (eat food, flee when low, surface when drowning). The drowning detection partially works via position heuristic but real health/air data would make it reliable.
