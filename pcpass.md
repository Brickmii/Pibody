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
| `health` | float | 0-20 (half-hearts) | 20.0 | **BROKEN** — reading 0.0 |
| `hunger` | float | 0-20 (half-shanks) | 20.0 | **BROKEN** — reading 0.0 |
| `air` | int | 0-300 ticks, -1=not underwater | -1 | Working (-1 correct when on land) |
| `x` | float | world coordinate | 0.0 | **MISSING** — never sent |
| `y` | float | world coordinate | 62.0 | **MISSING** — never sent |
| `z` | float | world coordinate | 0.0 | **MISSING** — never sent |
| `yaw` | float | 0-360 degrees | 0.0 | **MISSING** — never sent |
| `pitch` | float | -90 to 90 degrees | 0.0 | **MISSING** — never sent |

### What the Pi reads from `world_state["hotbar"]`:

| Key | Type | Status |
|-----|------|--------|
| `selected` | int (1-9) | Working |
| `slots` | List[bool] (9 items) | Working |

---

## What's Broken: Health and Hunger

The state key on Pi currently shows `h0` — meaning `int(player.get("health", 20))` is getting `0` from the client. The HUD reader's `_read_health()` and `_read_hunger()` are returning 0.0.

### How health flows into decisions:

1. **State key**: `ow_plains_h20_e0p0` — health is encoded directly. `h0` means the brain thinks the player is dead every tick.
2. **Drowning detection** (line 976-986): Uses `health < self._last_health` to detect damage underwater. With health stuck at 0, this can't work.
3. **Future survival**: The conscience reward system uses action outcomes to learn. Accurate health lets it learn "taking damage = bad plan."

### Debugging approach:

The `debug_save()` overlay is good — check `hud_debug.png` to verify the green dots (heart centers) actually land on the heart icons. If they don't, the coordinate math in `_compute_regions()` needs adjustment for the current resolution/GUI scale.

Key questions:
- What resolution is Minecraft running at? (1920x1080? 2560x1440?)
- What is the GUI scale in Minecraft settings? (1, 2, 3, or Auto?)
- Is the game fullscreen or windowed?

The `HUDReader` constructor takes `(screen_width, screen_height, gui_scale)`. If any of these don't match the actual game, every sample point will miss.

---

## What's Missing: Position Data (x, y, z, yaw, pitch)

Bedrock Edition "Show Coordinates" displays position in the top-left:
```
Position: X, Y, Z
```

The current `hud_reader.py` has `_read_coordinates()` using easyocr, and the client spreads the result:
```python
**hud_data.get("coordinates", {})
```

This would put `x`, `y`, `z` into `player` dict — which is exactly right. But:

1. **easyocr must be installed** on the PC (`pip install easyocr`)
2. **"Show Coordinates" must be ON** in Minecraft Bedrock settings (Game > Show Coordinates)
3. The OCR crop region (top-left 400x100px) must contain the position text
4. The regex `Position:\s*([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)` must match the text

### What position data enables on Pi:

- **Hypersphere mapping**: `mc_altitude_to_theta(y)` and `mc_yaw_to_phi(yaw)` map the player to the manifold's coordinate system. Without real y, it defaults to sea level and the altitude dimension of the manifold is dead.
- **Drowning detection**: `y < sea_level - 1` is used as a heuristic for underwater. Without real y, this is always false (default y=62 = sea level).
- **Biome/dimension encoding**: State key includes biome from `gs.get("biome", "unknown")`. This comes from the vision transformer, not coordinates, but position helps contextualize it.

### Yaw and Pitch

The Pi uses yaw for `mc_yaw_to_phi()` to map facing direction to the manifold. Without it, the agent can't reason about which direction it's facing. Currently defaults to 0.

These are NOT in the Bedrock coordinate overlay. Options:
- Parse from Bedrock F3-like debug screen (if available)
- Infer from mouse movement tracking in the motor controller
- Skip for now — the manifold still functions, just with less spatial awareness

---

## What's Working

- **Air bubbles**: Returns -1 when not underwater (correct), should return 0-300 when submerged
- **Hotbar selected**: Reads which slot (1-9) is active
- **Hotbar slots**: Detects which slots have items
- **Vision transformer**: Scene classification, entity detection, peaks — all flowing correctly
- **Action execution**: Motor controller executing moves, attacks, jumps, mining

---

## Client Code Reference

The HUD merge happens in two places:

### Request path (`pbai_client.py:536-555`):
```python
hud_data = self.hud_reader.read_hud(image)
world_state["player"] = {
    "health": hud_data["health"],
    "hunger": hud_data["hunger"],
    "air": hud_data.get("air", -1),
    **hud_data.get("coordinates", {}),  # spreads x, y, z if OCR succeeds
}
```

### Stream path (`pbai_client.py:640-647`):
Same structure, in `_transform_image()`.

Both paths should produce identical `player` dicts.

---

## Priority

1. **Health + Hunger** (HIGH) — Fix the HUD coordinate calibration so `_read_health()` and `_read_hunger()` return real values. This is the most impactful fix.
2. **Position x/y/z** (MEDIUM) — Get easyocr reading coordinates. Enables drowning detection and spatial mapping.
3. **Yaw/pitch** (LOW) — Nice to have but not critical right now.

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
