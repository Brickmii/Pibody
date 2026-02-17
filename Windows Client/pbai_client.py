"""
PBAI Client v6.0 - CUDA Vision Transformer

PC-side client with Three-Frame Identity vision system.

════════════════════════════════════════════════════════════════════════════════
ARCHITECTURE
════════════════════════════════════════════════════════════════════════════════

Screen Capture
     ↓
CUDA Vision Transformer
├── Three-Frame Encoder (Color/Position/Heat)
├── Spiral Patch Embedding (center-out)
├── Transformer Blocks (attention = heat)
└── Top-K Compression
     ↓
World State → WebSocket → Pi
     ↓
Pi makes decision
     ↓
Action ← WebSocket ← Pi
     ↓
Motor Executor
     ↓
Outcome → Heat Backprop → Learn

════════════════════════════════════════════════════════════════════════════════

USAGE:
    python pbai_client.py --host 192.168.5.46
    python pbai_client.py --host 192.168.5.46 --kill-key F12
    python pbai_client.py --host 192.168.5.46 --no-learn

REQUIREMENTS:
    pip install torch mss pydirectinput pyautogui websockets numpy keyboard

════════════════════════════════════════════════════════════════════════════════
"""

import asyncio
import json
import logging
import time
import sys
import os
from typing import Optional, Dict
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ═══════════════════════════════════════════════════════════════════════════════

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("ERROR: PyTorch required. Install: pip install torch")

try:
    import mss
    HAS_MSS = True
except ImportError:
    HAS_MSS = False

try:
    import pydirectinput
    pydirectinput.PAUSE = 0.02
    pydirectinput.FAILSAFE = True
    HAS_DIRECTINPUT = True
except ImportError:
    HAS_DIRECTINPUT = False

try:
    import pyautogui
    pyautogui.PAUSE = 0.02
    pyautogui.FAILSAFE = True
    HAS_PYAUTOGUI = True
except ImportError:
    HAS_PYAUTOGUI = False

try:
    import websockets
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False

try:
    import keyboard
    HAS_KEYBOARD = True
except ImportError:
    HAS_KEYBOARD = False

# Fallback: try pynput
if not HAS_KEYBOARD:
    try:
        from pynput import keyboard as pynput_keyboard
        HAS_PYNPUT = True
    except ImportError:
        HAS_PYNPUT = False
else:
    HAS_PYNPUT = False

try:
    from hud_reader import HUDReader
    HAS_HUD_READER = True
except ImportError:
    HAS_HUD_READER = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Planck constants
PHI = 1.618033988749895
K = 4 / (PHI ** 2)


# ═══════════════════════════════════════════════════════════════════════════════
# SCREEN CAPTURE
# ═══════════════════════════════════════════════════════════════════════════════

class ScreenCapture:
    """Captures screen using mss."""
    
    def __init__(self, monitor: int = 1):
        self.monitor = monitor
        self.sct = mss.mss() if HAS_MSS else None
    
    def capture(self) -> Optional[np.ndarray]:
        """Capture screen as RGB numpy array."""
        if not self.sct:
            return None
        try:
            mon = self.sct.monitors[self.monitor]
            shot = self.sct.grab(mon)
            img = np.array(shot)
            return img[:, :, [2, 1, 0]]  # BGRA -> RGB
        except Exception as e:
            logger.error(f"Capture failed: {e}")
            return None


# ═══════════════════════════════════════════════════════════════════════════════
# MOTOR EXECUTOR  
# ═══════════════════════════════════════════════════════════════════════════════

class MotorExecutor:
    """
    Executes motor commands from Pi.
    Mode agnostic - Pi's driver determines key mappings.
    """
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled and (HAS_DIRECTINPUT or HAS_PYAUTOGUI)
        self.use_directinput = HAS_DIRECTINPUT
        self._held_keys = set()
        self._held_buttons = set()
        self._killed = False
        
        if self.enabled:
            logger.info(f"Motor: {'pydirectinput' if self.use_directinput else 'pyautogui'}")
        else:
            logger.info("Motor: disabled")
    
    def kill(self):
        """Emergency kill."""
        self._killed = True
        self.release_all()
        logger.warning("MOTOR KILLED")
    
    def unkill(self):
        """Re-enable."""
        self._killed = False
    
    def _press(self, key): 
        (pydirectinput if self.use_directinput else pyautogui).press(key)
    
    def _keyDown(self, key):
        (pydirectinput if self.use_directinput else pyautogui).keyDown(key)
    
    def _keyUp(self, key):
        (pydirectinput if self.use_directinput else pyautogui).keyUp(key)
    
    def _moveRel(self, dx, dy):
        if self.use_directinput:
            pydirectinput.moveRel(dx, dy, relative=True)
        else:
            pyautogui.moveRel(dx, dy, _pause=False)
    
    def _click(self, button="left"):
        (pydirectinput if self.use_directinput else pyautogui).click(button=button)

    def _moveTo(self, x, y):
        if self.use_directinput:
            pydirectinput.moveTo(x, y)
        else:
            pyautogui.moveTo(x, y)
    
    def _mouseDown(self, button="left"):
        (pydirectinput if self.use_directinput else pyautogui).mouseDown(button=button)
    
    def _mouseUp(self, button="left"):
        (pydirectinput if self.use_directinput else pyautogui).mouseUp(button=button)
    
    def execute(self, action: dict) -> bool:
        """Execute action from Pi."""
        if self._killed:
            return False
        
        mt = action.get("motor_type", "")
        
        if not self.enabled:
            logger.debug(f"[SIM] {action}")
            return True
        
        try:
            if mt == "key_press":
                self._press(action["key"])
            
            elif mt == "key_hold":
                key = action["key"]
                self._keyDown(key)
                self._held_keys.add(key)
                if action.get("duration"):
                    time.sleep(action["duration"])
                    self._keyUp(key)
                    self._held_keys.discard(key)
            
            elif mt == "key_release":
                key = action["key"]
                self._keyUp(key)
                self._held_keys.discard(key)
            
            elif mt == "look":
                d = action.get("direction", [0, 0])
                self._moveRel(int(d[0]) if d else 0, int(d[1]) if len(d) > 1 else 0)
            
            elif mt == "mouse_click":
                x, y = action.get("x"), action.get("y")
                if x is not None and y is not None:
                    self._moveTo(int(x), int(y))
                self._click(action.get("button", "left"))
            
            elif mt == "mouse_hold":
                btn = action.get("button", "left")
                self._mouseDown(btn)
                self._held_buttons.add(btn)
                if action.get("duration"):
                    time.sleep(action["duration"])
                    self._mouseUp(btn)
                    self._held_buttons.discard(btn)
            
            elif mt == "mouse_release":
                btn = action.get("button", "left")
                self._mouseUp(btn)
                self._held_buttons.discard(btn)
            
            elif mt == "wait":
                time.sleep(action.get("duration", 0.1))
            
            elif mt == "sequence":
                for sub in action.get("sequence", []):
                    if self._killed: break
                    self.execute(sub)
            
            return True
        except Exception as e:
            logger.error(f"Motor error: {e}")
            return False
    
    def release_all(self):
        """Release all held keys/buttons."""
        for k in list(self._held_keys):
            try: self._keyUp(k)
            except: pass
        for b in list(self._held_buttons):
            try: self._mouseUp(b)
            except: pass
        self._held_keys.clear()
        self._held_buttons.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# PBAI CLIENT
# ═══════════════════════════════════════════════════════════════════════════════

class PBAIClient:
    """PBAI Client with CUDA Vision Transformer."""
    
    def __init__(self,
                 host: str = "localhost",
                 port: int = 8421,
                 resolution: int = 64,
                 motor_enabled: bool = True,
                 learn: bool = True,
                 kill_key: str = "F12",
                 checkpoint: Optional[str] = None,
                 stream_fps: float = 2.0,
                 gui_scale: int = 2):

        self.host = host
        self.port = port
        self.resolution = resolution
        self.learn = learn
        self.kill_key = kill_key.lower()
        self._stream_fps = stream_fps
        self._streaming = False
        self._gui_scale = gui_scale

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Components
        self.screen = ScreenCapture()
        self.motor = MotorExecutor(enabled=motor_enabled)

        # HUD reader — extract health/hunger/hotbar from raw screenshots
        self.hud_reader = None
        if HAS_HUD_READER and HAS_MSS:
            try:
                mon = self.screen.sct.monitors[1] if self.screen.sct else None
                if mon:
                    self.hud_reader = HUDReader(mon['width'], mon['height'], gui_scale)
                    logger.info(f"HUD reader: {mon['width']}x{mon['height']} scale={gui_scale}")
            except Exception as e:
                logger.warning(f"HUD reader init failed: {e}")
        
        # One-shot debug overlay (saves hud_debug.png on first HUD read)
        self._hud_debug_saved = False

        # Vision transformer
        self._init_vision(checkpoint)
        
        # Connection
        self.connected = False
        self._running = False
        self._ws = None
        
        # Focus hint from Pi
        self._focus_hint = None
        
        # Stats
        self.stats = {
            "world_requests": 0,
            "actions": 0,
            "errors": 0,
            "heat_received": 0.0
        }
        
        # Kill key
        self._setup_kill_key()
    
    def _init_vision(self, checkpoint: Optional[str]):
        """Initialize vision transformer."""
        # Import the transformer
        from vision_transformer import (
            PBAIVisionTransformer, 
            HeatBackprop, 
            build_world_state
        )
        
        self.build_world_state = build_world_state
        
        # Create model
        self.model = PBAIVisionTransformer(
            resolution=self.resolution,
            patch_size=8,
            embed_dim=256,
            depth=6,
            num_heads=8
        ).to(self.device)
        
        # Learner
        self.learner = HeatBackprop(self.model) if self.learn else None
        
        # Load checkpoint
        if checkpoint and os.path.exists(checkpoint):
            state = torch.load(checkpoint, map_location=self.device)
            self.model.load_state_dict(state['model'])
            if self.learner and 'optimizer' in state:
                self.learner.optimizer.load_state_dict(state['optimizer'])
            logger.info(f"Loaded checkpoint: {checkpoint}")
        
        params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Vision Transformer: {params:,} params on {self.device}")
    
    def _setup_kill_key(self):
        """Set up kill key handler."""
        if HAS_KEYBOARD:
            keyboard.on_press_key(self.kill_key, lambda _: self.motor.kill())
            logger.info(f"Kill key: {self.kill_key.upper()} (keyboard)")
        elif HAS_PYNPUT:
            # Use pynput listener
            def on_press(key):
                try:
                    if hasattr(key, 'name') and key.name.lower() == self.kill_key.lower():
                        self.motor.kill()
                    elif hasattr(key, 'char') and key.char and key.char.lower() == self.kill_key.lower():
                        self.motor.kill()
                except:
                    pass
            
            self._pynput_listener = pynput_keyboard.Listener(on_press=on_press)
            self._pynput_listener.start()
            logger.info(f"Kill key: {self.kill_key.upper()} (pynput)")
        else:
            logger.warning("No kill key - install 'keyboard' or 'pynput' package")
            logger.warning("  pip install keyboard   (needs admin on Windows)")
            logger.warning("  pip install pynput     (cross-platform)")
            logger.info("You can still stop with Ctrl+C")
    
    async def connect(self) -> bool:
        """Connect to Pi."""
        uri = f"ws://{self.host}:{self.port}"
        logger.info(f"Connecting to {uri}...")
        
        try:
            self._ws = await websockets.connect(uri, ping_interval=30, ping_timeout=30)
            
            await self._ws.send(json.dumps({
                "type": "hello",
                "client": "pbai_cuda_vision",
                "version": "6.0.0",
                "device": str(self.device),
                "resolution": self.resolution,
                "learning": self.learn,
                "features": ["three_frame", "spiral_attention", "heat_backprop"]
            }))
            
            response = await asyncio.wait_for(self._ws.recv(), timeout=5.0)
            msg = json.loads(response)
            
            if msg.get("type") == "hello":
                self.connected = True
                logger.info("✓ Connected to Pi")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect."""
        self.motor.release_all()
        if self._ws:
            await self._ws.close()
        self.connected = False
    
    async def handle_message(self, msg: dict):
        """Handle message from Pi."""
        msg_type = msg.get("type")
        logger.debug(f"Received message type: {msg_type}")
        
        if msg_type == "request_world":
            await self._handle_world_request()
        
        elif msg_type == "action":
            await self._handle_action(msg)
        
        elif msg_type == "heat_feedback":
            await self._handle_heat(msg)
        
        elif msg_type == "focus_hint":
            self._focus_hint = msg.get("focus")
        
        elif msg_type == "release_all":
            self.motor.release_all()
        
        elif msg_type == "save_checkpoint":
            self._save_checkpoint(msg.get("path", "pbai_checkpoint.pt"))
        
        elif msg_type == "stop":
            self._running = False
        
        elif msg_type == "ping":
            await self._ws.send(json.dumps({"type": "pong"}))
    
    async def _handle_world_request(self):
        """Process world request."""
        logger.info("Received request_world from Pi")
        
        image = self.screen.capture()
        if image is None:
            logger.error("Screen capture failed!")
            await self._ws.send(json.dumps({"type": "world_state", "error": True}))
            self.stats["errors"] += 1
            return
        
        logger.info(f"Captured image: {image.shape}")
        
        # Convert to tensor
        img_tensor = torch.from_numpy(image).float().to(self.device)
        if img_tensor.max() > 1:
            img_tensor = img_tensor / 255.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        
        # Resize if needed
        if img_tensor.shape[-1] != self.resolution:
            img_tensor = torch.nn.functional.interpolate(
                img_tensor, 
                size=(self.resolution, self.resolution),
                mode='bilinear',
                align_corners=False
            )
        
        logger.info(f"Tensor shape: {img_tensor.shape}")
        
        # Focus hint
        focus = None
        if self._focus_hint:
            # Convert (x, y) to patch indices
            # This would need proper implementation
            self._focus_hint = None
        
        # Forward pass
        logger.info("Running vision transformer...")
        with torch.no_grad():
            output = self.model(img_tensor, focus)
        
        logger.info(f"Transformer output keys: {output.keys()}")
        
        # Store for learning
        if self.learner:
            self.learner.store_output(output)
        
        # Build world state
        world_state = self.build_world_state(self.model, img_tensor, top_k=10, output=output)

        # HUD reading on raw full-resolution image
        if self.hud_reader is not None:
            try:
                hud_data = self.hud_reader.read_hud(image)
                world_state["player"] = {
                    "health": hud_data["health"],
                    "hunger": hud_data["hunger"],
                    "air": hud_data.get("air", -1),
                    **hud_data.get("coordinates", {}),
                }
                world_state["hotbar"] = {
                    "selected": hud_data["hotbar_selected"],
                    "slots": hud_data["hotbar_slots"],
                }
                logger.info(f"HUD: health={hud_data['health']} hunger={hud_data['hunger']} air={hud_data.get('air', '?')}")
                if not self._hud_debug_saved:
                    self.hud_reader.debug_save(image)
                    self._hud_debug_saved = True
                    logger.info("Saved hud_debug.png")
            except Exception as e:
                logger.warning(f"HUD read error: {e}")

        logger.info(f"World state: t_K={world_state['t_K']}, kappa={world_state['kappa']:.4f}, peaks={len(world_state['peaks'])}")

        # Send
        await self._ws.send(json.dumps({
            "type": "world_state",
            "data": world_state
        }))
        
        logger.info("Sent world_state to Pi")
        
        self.stats["world_requests"] += 1
    
    async def _handle_action(self, msg: dict):
        """Execute action."""
        action = msg.get("data", {})
        success = self.motor.execute(action)
        self.stats["actions"] += 1
        
        await self._ws.send(json.dumps({
            "type": "action_result",
            "success": success
        }))
    
    async def _handle_heat(self, msg: dict):
        """Handle heat feedback and learn."""
        heat = msg.get("heat", 0.0)
        self.stats["heat_received"] += heat
        
        if self.learner:
            # Need current output for learning
            img = self.screen.capture()
            if img is not None:
                img_tensor = torch.from_numpy(img).float().to(self.device)
                if img_tensor.max() > 1:
                    img_tensor = img_tensor / 255.0
                img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
                
                if img_tensor.shape[-1] != self.resolution:
                    img_tensor = torch.nn.functional.interpolate(
                        img_tensor,
                        size=(self.resolution, self.resolution),
                        mode='bilinear',
                        align_corners=False
                    )
                
                output = self.model(img_tensor)
                loss = self.learner.update(heat, output)
                
                if abs(heat) > 0.5:
                    logger.info(f"Heat feedback: {heat:.2f}, loss: {loss:.4f}")
    
    def _transform_image(self, image):
        """Synchronous GPU inference (runs in thread executor).

        Only the heavy GPU work is offloaded — screen capture stays in
        the main thread since it uses thread-local Windows GDI resources.
        """
        # HUD reading on raw full-resolution image (before downscaling)
        hud_data = None
        if self.hud_reader is not None:
            try:
                hud_data = self.hud_reader.read_hud(image)
            except Exception as e:
                logger.warning(f"HUD read error: {e}")

        img_tensor = torch.from_numpy(image).float().to(self.device)
        if img_tensor.max() > 1:
            img_tensor = img_tensor / 255.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)

        if img_tensor.shape[-1] != self.resolution:
            img_tensor = torch.nn.functional.interpolate(
                img_tensor,
                size=(self.resolution, self.resolution),
                mode='bilinear',
                align_corners=False
            )

        with torch.no_grad():
            output = self.model(img_tensor, self._focus_hint)

        world_state = self.build_world_state(self.model, img_tensor, top_k=10, output=output)

        # Merge HUD data into world state
        if hud_data:
            world_state["player"] = {
                "health": hud_data["health"],
                "hunger": hud_data["hunger"],
                "air": hud_data.get("air", -1),
                **hud_data.get("coordinates", {}),
            }
            world_state["hotbar"] = {
                "selected": hud_data["hotbar_selected"],
                "slots": hud_data["hotbar_slots"],
            }
            logger.info(f"HUD: health={hud_data['health']} hunger={hud_data['hunger']} air={hud_data.get('air', '?')}")
            if not self._hud_debug_saved:
                self.hud_reader.debug_save(image)
                self._hud_debug_saved = True
                logger.info("Saved hud_debug.png")

        return world_state

    async def _continuous_capture_loop(self):
        """Background loop: capture + transform + send at configured FPS.

        Sends unsolicited world_state frames so the Pi always has fresh
        vision data in its buffer without explicit request_world round-trips.
        Screen capture runs in the main thread (needs Windows GDI context),
        GPU inference runs in a thread executor to keep pings responsive.
        """
        self._streaming = True
        interval = 1.0 / self._stream_fps
        loop = asyncio.get_event_loop()
        logger.info(f"Starting continuous capture at {self._stream_fps} FPS (interval={interval:.3f}s)")

        while self._streaming and self._running and self._ws:
            try:
                # Capture in main thread (Windows GDI is thread-local)
                image = self.screen.capture()
                if image is None:
                    await asyncio.sleep(interval)
                    continue

                # Offload heavy GPU inference to thread executor
                world_state = await loop.run_in_executor(
                    None, self._transform_image, image
                )

                await self._ws.send(json.dumps({
                    "type": "world_state",
                    "data": world_state,
                    "stream": True
                }))

            except websockets.ConnectionClosed:
                break
            except Exception as e:
                logger.error(f"Stream capture error: {e}")

            await asyncio.sleep(interval)

        self._streaming = False
        logger.info("Continuous capture stopped")

    def _save_checkpoint(self, path: str):
        """Save model checkpoint."""
        state = {
            'model': self.model.state_dict(),
            't_K': self.model.t_K,
            'stats': self.stats
        }
        if self.learner:
            state['optimizer'] = self.learner.optimizer.state_dict()
        
        torch.save(state, path)
        logger.info(f"Saved checkpoint: {path}")
    
    async def run(self):
        """Main loop."""
        if not await self.connect():
            return

        self._running = True
        logger.info(f"Running (learning={'ON' if self.learn else 'OFF'})")

        # Start continuous capture in background (if enabled)
        stream_task = None
        if self._stream_fps > 0:
            stream_task = asyncio.create_task(self._continuous_capture_loop())

        try:
            while self._running and self.connected:
                try:
                    raw = await asyncio.wait_for(self._ws.recv(), timeout=30.0)
                    msg = json.loads(raw)
                    await self.handle_message(msg)
                except asyncio.TimeoutError:
                    await self._ws.send(json.dumps({"type": "heartbeat"}))
                except websockets.ConnectionClosed:
                    self.connected = False
                    break
        except Exception as e:
            logger.error(f"Error: {e}")
        finally:
            self._streaming = False
            if stream_task:
                stream_task.cancel()
            await self.disconnect()
    
    def stop(self):
        """Stop client."""
        self._running = False
        self.motor.release_all()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="PBAI CUDA Vision Client")
    parser.add_argument("--host", default="192.168.5.46")
    parser.add_argument("--port", type=int, default=8421)
    parser.add_argument("--resolution", type=int, default=512  )
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--no-learn", action="store_true")
    parser.add_argument("--kill-key", default="F12")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--stream-fps", type=float, default=2.0,
                        help="Continuous vision stream rate (frames/sec, 0=disabled)")
    parser.add_argument("--gui-scale", type=int, default=2,
                        help="Minecraft GUI scale (default 2)")
    args = parser.parse_args()
    
    # Check requirements
    missing = []
    if not HAS_TORCH: missing.append("torch")
    if not HAS_MSS: missing.append("mss")
    if not HAS_WEBSOCKETS: missing.append("websockets")
    if not HAS_DIRECTINPUT and not HAS_PYAUTOGUI:
        missing.append("pydirectinput or pyautogui")
    
    if missing:
        print(f"Missing: {', '.join(missing)}")
        return
    
    print("=" * 60)
    print("PBAI CUDA VISION CLIENT v6.0")
    print("=" * 60)
    print()
    print("Three-Frame Identity:")
    print("  X: Color (what IS)")
    print("  Y: Position (where IS)")
    print("  Z: Heat (how much MATTERS)")
    print()
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Pi: {args.host}:{args.port}")
    print(f"Resolution: {args.resolution}x{args.resolution}")
    print(f"Learning: {'OFF' if args.no_learn else 'ON'}")
    print(f"Stream: {args.stream_fps} FPS" if args.stream_fps > 0 else "Stream: OFF")
    print(f"Kill key: {args.kill_key}")
    print()
    print(f"K = {K:.6f}, phi = {PHI:.6f}")
    print("=" * 60)
    print()
    
    client = PBAIClient(
        host=args.host,
        port=args.port,
        resolution=args.resolution,
        motor_enabled=not args.simulate,
        learn=not args.no_learn,
        kill_key=args.kill_key,
        checkpoint=args.checkpoint,
        stream_fps=args.stream_fps,
        gui_scale=args.gui_scale,
    )
    
    import signal
    signal.signal(signal.SIGINT, lambda s, f: client.stop())
    
    await client.run()
    
    print(f"\nStats: {client.stats}")


if __name__ == "__main__":
    asyncio.run(main())
