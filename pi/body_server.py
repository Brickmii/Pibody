"""
PBAI Body Server - Pi-Driven WebSocket Bridge (Planck-Grounded)

Pi brain controls when to see and act. PC body just responds.

════════════════════════════════════════════════════════════════════════════════
PLANCK GROUNDING
════════════════════════════════════════════════════════════════════════════════

CLOCK SYNC:
    All world state includes t_K timestamp for heat-time synchronization.
    Vision requests and action results are timestamped with t_K.

CONSTRAINT TYPES:
    World state properties include Robinson constraint types:
    - motion: addition (temporal)
    - position: successor (spatial)
    - heat/kappa: multiplication (quantitative)

STRUCTURE DETECTION:
    When world state entropy > 44/45, pattern-seeking is triggered.

════════════════════════════════════════════════════════════════════════════════

PROTOCOL (Pi-Driven):
    Server → Client:
        {"type": "request_vision", "prompt": "...", "t_K": ...}
        {"type": "request_world", "t_K": ...}
        {"type": "action", "data": {...}, "t_K": ...}
        {"type": "stop"}
        {"type": "release_all"}
    
    Client → Server:
        {"type": "hello", "client": "pbai_body", "version": "..."}
        {"type": "vision_result", "data": "...", "timestamp": ..., "t_K": ...}
        {"type": "world_state", "data": {...}, "t_K": ...}
        {"type": "action_result", "success": true/false, "t_K": ...}
        {"type": "heartbeat"}

ALSO SUPPORTS (for diagnostics/testing):
        {"type": "status"}
        {"type": "test_constraint", "action": "...", ...}
        {"type": "input", "action": "press/release", ...}

DRIVER INTERFACE:
    body_server.request_vision(prompt) -> str   # Get vision from PC
    body_server.request_world() -> dict         # Get world state from PC (fast)
    body_server.send_action(action) -> bool     # Send action to PC
    body_server.release_all()                   # Emergency stop
"""

import asyncio
import collections
import json
import logging
import time
import threading
from typing import Optional, Dict, Any, TYPE_CHECKING
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from .daemon import PBAIDaemon

try:
    import websockets
    from websockets.server import serve, WebSocketServerProtocol
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False
    WebSocketServerProtocol = None  # Placeholder for type hints

# Import Planck constants
try:
    from core.node_constants import K, PHI, BODY_TEMPERATURE
except ImportError:
    PHI = 1.618033988749895
    K = 4 / (PHI ** 2)
    BODY_TEMPERATURE = K * PHI ** 11

logger = logging.getLogger(__name__)


@dataclass
class BodyConnection:
    """A connected body client."""
    websocket: 'WebSocketServerProtocol'
    client_type: str = "unknown"
    version: str = "unknown"
    connected_at: float = field(default_factory=time.time)
    vision_requests: int = 0
    actions_sent: int = 0
    
    # For request/response tracking
    pending_vision: Optional[asyncio.Future] = None


@dataclass
class InputState:
    """Track timed input state for diagnostic tests."""
    button: str
    pressed_at: float
    released_at: Optional[float] = None
    
    @property
    def duration(self) -> float:
        if self.released_at:
            return self.released_at - self.pressed_at
        return time.time() - self.pressed_at


class BodyServer:
    """
    Pi-driven body server (Planck-Grounded).
    
    Drivers call request_vision() and send_action() to interact with PC.
    PC does nothing unless Pi asks.
    
    PLANCK GROUNDING:
        - All requests include t_K timestamp
        - World state includes constraint types
        - Vision requests sync with clock
    """
    
    def __init__(self, daemon: 'PBAIDaemon' = None, port: int = 8421):
        self.daemon = daemon
        self.port = port
        
        self.bodies: Dict[str, BodyConnection] = {}
        self._primary_body: Optional[str] = None  # First connected body
        self._running = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        
        # Diagnostic test state
        self._test_specs: Dict[str, Any] = {}
        self._input_states: Dict[str, InputState] = {}
        
        # Vision prompt (driver can set this)
        self.vision_prompt = "Describe what you see on screen."
        
        # World model support (Planck-grounded)
        self.last_world_state: Optional[dict] = None
        self.world_requests: int = 0
        self.last_world_t_K: int = 0  # t_K of last world state

        # Persistent vision buffer (fed by continuous stream from PC)
        self._world_buffer: collections.deque = collections.deque(maxlen=10)
        self._world_buffer_lock: threading.Lock = threading.Lock()
    
    @property
    def has_body(self) -> bool:
        """Check if any body is connected."""
        return len(self.bodies) > 0
    
    @property
    def primary_body(self) -> Optional[BodyConnection]:
        """Get the primary connected body."""
        if self._primary_body and self._primary_body in self.bodies:
            return self.bodies[self._primary_body]
        return None
    
    # ═══════════════════════════════════════════════════════════════════════════
    # DRIVER INTERFACE - Call these from drivers
    # ═══════════════════════════════════════════════════════════════════════════
    
    def request_vision(self, prompt: str = None, timeout: float = 10.0) -> Optional[str]:
        """
        Request vision from PC body.
        
        Blocks until PC responds or timeout.
        Returns vision text or None on error.
        
        Called by drivers to see the screen.
        """
        if not self.has_body or not self._loop:
            logger.warning("No body connected, cannot request vision")
            return None
        
        prompt = prompt or self.vision_prompt
        
        # Run async request in the server's event loop
        future = asyncio.run_coroutine_threadsafe(
            self._async_request_vision(prompt, timeout),
            self._loop
        )
        
        try:
            return future.result(timeout=timeout + 1)
        except Exception as e:
            logger.error(f"Vision request failed: {e}")
            return None
    
    async def _async_request_vision(self, prompt: str, timeout: float) -> Optional[str]:
        """Async implementation of vision request."""
        body = self.primary_body
        if not body:
            return None
        
        # Create future for response
        body.pending_vision = self._loop.create_future()
        
        # Send request
        try:
            await body.websocket.send(json.dumps({
                "type": "request_vision",
                "prompt": prompt
            }))
            body.vision_requests += 1
            
            # Wait for response
            result = await asyncio.wait_for(body.pending_vision, timeout=timeout)
            return result
        
        except asyncio.TimeoutError:
            logger.warning("Vision request timed out")
            return None
        except Exception as e:
            logger.error(f"Vision request error: {e}")
            return None
        finally:
            body.pending_vision = None
    
    def request_world(self, include_image: bool = False, timeout: float = 5.0) -> Optional[dict]:
        """
        Request world model state from PC body.
        
        This is a structured, low-latency alternative to VLM screen description.
        
        Args:
            include_image: If True, the body may include an encoded image payload.
            timeout: Seconds to wait for a response.
        
        Returns:
            Dict world state, or None on error/timeout. Also updates self.last_world_state.
        """
        if not self.has_body or not self._loop:
            logger.warning("No body connected, cannot request world")
            return None
        
        future = asyncio.run_coroutine_threadsafe(
            self._async_request_world(include_image, timeout),
            self._loop
        )
        
        try:
            result = future.result(timeout=timeout + 1)
            self.last_world_state = result
            self.world_requests += 1
            return result
        except Exception as e:
            logger.error(f"World request failed: {e}")
            return None
    
    async def _async_request_world(self, include_image: bool, timeout: float) -> Optional[dict]:
        """Async implementation of world request."""
        body = self.primary_body
        if not body:
            return None
        
        # Reuse the existing pending future slot
        body.pending_vision = self._loop.create_future()
        
        try:
            await body.websocket.send(json.dumps({
                "type": "request_world",
                "include_image": include_image
            }))
            
            result = await asyncio.wait_for(body.pending_vision, timeout=timeout)
            return result
        
        except asyncio.TimeoutError:
            logger.warning("World request timed out")
            return None
        except Exception as e:
            logger.error(f"World request error: {e}")
            return None
        finally:
            body.pending_vision = None
    
    def get_latest_world(self) -> Optional[dict]:
        """Get most recent world state from buffer (non-blocking).

        Returns None if no frames have been received yet.
        Used by drivers that want continuous vision without blocking.
        """
        with self._world_buffer_lock:
            if self._world_buffer:
                return self._world_buffer[-1]
        return None

    def send_action(self, action: dict, timeout: float = 5.0) -> bool:
        """
        Send action to PC body.
        
        Called by drivers to execute motor commands.
        """
        if not self.has_body or not self._loop:
            logger.warning("No body connected, cannot send action")
            return False
        
        future = asyncio.run_coroutine_threadsafe(
            self._async_send_action(action, timeout),
            self._loop
        )
        
        try:
            return future.result(timeout=timeout + 1)
        except Exception as e:
            logger.error(f"Action send failed: {e}")
            return False
    
    async def _async_send_action(self, action: dict, timeout: float) -> bool:
        """Async implementation of action send."""
        body = self.primary_body
        if not body:
            return False
        
        try:
            await body.websocket.send(json.dumps({
                "type": "action",
                "data": action
            }))
            body.actions_sent += 1
            return True
        except Exception as e:
            logger.error(f"Action send error: {e}")
            return False
    
    def send_stop(self):
        """Tell PC to stop (session over)."""
        if self.has_body and self._loop:
            asyncio.run_coroutine_threadsafe(
                self._async_send_stop(),
                self._loop
            )
    
    async def _async_send_stop(self):
        """Async stop."""
        body = self.primary_body
        if body:
            try:
                await body.websocket.send(json.dumps({"type": "stop"}))
            except:
                pass
    
    def release_all(self):
        """Emergency release all keys/buttons on PC."""
        if self._loop:
            for body in self.bodies.values():
                try:
                    asyncio.run_coroutine_threadsafe(
                        body.websocket.send(json.dumps({"type": "release_all"})),
                        self._loop
                    )
                except:
                    pass
        logger.warning("Sent release_all to all bodies")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # WEBSOCKET HANDLING
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def _handle_client(self, websocket: WebSocketServerProtocol, path: str = None):
        """Handle a body connection."""
        client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        body = BodyConnection(websocket=websocket)
        self.bodies[client_id] = body
        
        # First body is primary
        if self._primary_body is None:
            self._primary_body = client_id
        
        logger.info(f"Body connected: {client_id}")
        
        try:
            async for message in websocket:
                response = await self._route_message(message, body)
                if response:
                    await websocket.send(json.dumps(response))
        except websockets.ConnectionClosed:
            logger.info(f"Body disconnected: {client_id}")
        except Exception as e:
            logger.error(f"Body error: {e}")
        finally:
            del self.bodies[client_id]
            if self._primary_body == client_id:
                self._primary_body = next(iter(self.bodies), None)
    
    async def _route_message(self, raw: str, body: BodyConnection) -> Optional[dict]:
        """Route incoming message to handler."""
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return {"type": "error", "message": "Invalid JSON"}

        msg_type = msg.get("type")
        
        # ─────────────────────────────────────────────────────────────────────
        # BODY CLIENT MESSAGES
        # ─────────────────────────────────────────────────────────────────────
        
        if msg_type == "hello":
            body.client_type = msg.get("client", "unknown")
            body.version = msg.get("version", "unknown")
            logger.info(f"Body hello: {body.client_type} v{body.version}")

            # Send masterframe so client knows the screen layout
            masterframe_data = None
            if self.daemon and self.daemon.env_core:
                driver = self.daemon.env_core.get_active_driver()
                if driver and hasattr(driver, 'masterframe') and driver.masterframe:
                    masterframe_data = driver.masterframe.to_dict()
                    logger.info(f"Sending masterframe to client ({len(masterframe_data.get('modes', {}).get('hud', {}).get('regions', {}))} regions)")

            return {"type": "hello", "status": "ok", "masterframe": masterframe_data}
        
        elif msg_type == "vision_result":
            # Response to our vision request
            if body.pending_vision and not body.pending_vision.done():
                data = msg.get("data", "")
                if msg.get("error"):
                    body.pending_vision.set_result(None)
                else:
                    body.pending_vision.set_result(data)
            return None  # No response needed
        
        elif msg_type == "world_state":
            data = msg.get("data", {})

            # Always buffer incoming world state (streamed or requested)
            if data and not msg.get("error"):
                with self._world_buffer_lock:
                    self._world_buffer.append(data)
                self.last_world_state = data

                # Heat feed: vision frames sustain psychology against existence tax
                # Identity sees, so it gets the most from perception
                # All three get enough to survive tax between frames (~5s gap)
                if self.daemon and self.daemon.manifold:
                    m = self.daemon.manifold
                    if m.identity_node:
                        m.identity_node.add_heat_unchecked(0.382)
                    if m.ego_node:
                        m.ego_node.add_heat_unchecked(0.146)
                    if m.conscience_node:
                        m.conscience_node.add_heat_unchecked(0.146)

                # Feed to visual cortex for persistence tracking
                # and manifold integration (throttled to every 3rd frame)
                if self.daemon and self.daemon.visual_cortex:
                    self._vision_frame_count = getattr(self, '_vision_frame_count', 0) + 1
                    if self._vision_frame_count % 3 == 0:
                        try:
                            self.daemon.visual_cortex.process_world_state(data)
                        except Exception as e:
                            logger.debug(f"Visual cortex process error: {e}")

            # If there's a pending request, resolve it too
            if body.pending_vision and not body.pending_vision.done():
                if msg.get("error"):
                    body.pending_vision.set_result(None)
                else:
                    body.pending_vision.set_result(data)
            return None  # No response needed
        
        elif msg_type == "action_result":
            # Acknowledgment of action (ignore for now)
            return None
        
        elif msg_type == "heartbeat":
            return {"type": "pong"}
        
        elif msg_type == "pong":
            return None
        
        # ─────────────────────────────────────────────────────────────────────
        # DIAGNOSTIC / TEST MESSAGES
        # ─────────────────────────────────────────────────────────────────────
        
        elif msg_type == "status":
            return self._handle_status(msg)
        
        elif msg_type == "test_constraint":
            return self._handle_test_constraint(msg)
        
        elif msg_type == "input":
            return self._handle_input(msg)
        
        return {"type": "error", "message": f"Unknown type: {msg_type}"}
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STATUS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _handle_status(self, msg: dict) -> dict:
        """Return status including t_K for clock sync (Planck-Grounded)."""
        t_K = 0
        nodes = 0
        total_heat = 0.0
        structure_detected = False
        
        if self.daemon and self.daemon.manifold:
            manifold = self.daemon.manifold
            t_K = manifold.self_node.t_K if manifold.self_node else 0
            nodes = len(manifold.nodes)
            total_heat = manifold.total_heat()
            structure_detected = manifold.structure_detected()
        
        temp = 0.0
        zone = "unknown"
        fire_level = 1
        if self.daemon:
            temp = self.daemon.stats.current_temp
            zone = self.daemon.stats.current_zone
            fire_level = getattr(self.daemon.stats, 'current_fire_level', 1)
        
        return {
            "type": "status",
            "t_K": t_K,
            "nodes": nodes,
            "total_heat": total_heat,
            "temperature": temp,
            "thermal_zone": zone,
            "fire_level": fire_level,
            "structure_detected": structure_detected,
            "bodies": len(self.bodies),
            "running": self._running,
            "test_specs": len(self._test_specs),
            "world_requests": self.world_requests,
            "last_world_t_K": self.last_world_t_K,
        }
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CONSTRAINT TESTING (for manifold_test.py)
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _handle_test_constraint(self, msg: dict) -> dict:
        """Handle constraint/spec testing commands."""
        action = msg.get("action")
        
        try:
            if action == "create_robinson_spec":
                return self._test_create_robinson_spec(msg)
            elif action == "create_defined_spec":
                return self._test_create_defined_spec(msg)
            elif action == "get_spec":
                return self._test_get_spec(msg)
            elif action == "measure":
                return self._test_measure(msg)
            elif action == "verify":
                return self._test_verify(msg)
            elif action == "learn":
                return self._test_learn(msg)
            elif action == "clear_specs":
                self._test_specs.clear()
                return {"success": True, "message": "All test specs cleared"}
            else:
                return {"success": False, "error": f"Unknown action: {action}"}
        except Exception as e:
            logger.error(f"Constraint test error: {e}")
            return {"success": False, "error": str(e)}
    
    def _test_create_robinson_spec(self, msg: dict) -> dict:
        from core.constraints import robinson_spec
        name = msg.get("name", "test")
        observed = msg.get("observed", 1.0)
        unit = msg.get("unit", "")
        spec = robinson_spec(observed, unit)
        self._test_specs[name] = spec
        return {
            "success": True,
            "spec": spec.to_dict(),
            "message": f"Created Robinson spec '{name}'"
        }
    
    def _test_create_defined_spec(self, msg: dict) -> dict:
        from core.constraints import Spec, SpecSource
        from core.node_constants import K
        name = msg.get("name", "test")
        nominal = msg.get("nominal", 1.0)
        tolerance = msg.get("tolerance")
        tolerance_low = msg.get("tolerance_low")
        tolerance_high = msg.get("tolerance_high")
        unit = msg.get("unit", "")
        if tolerance is not None:
            tolerance_low = nominal - tolerance
            tolerance_high = nominal + tolerance
        spec = Spec(
            nominal=nominal,
            tolerance_low=tolerance_low if tolerance_low is not None else nominal,
            tolerance_high=tolerance_high if tolerance_high is not None else nominal,
            unit=unit,
            source=SpecSource.DEFINED,
            heat=5 * K,
        )
        self._test_specs[name] = spec
        return {"success": True, "spec": spec.to_dict()}
    
    def _test_get_spec(self, msg: dict) -> dict:
        name = msg.get("name", "test")
        if name not in self._test_specs:
            return {"success": False, "error": f"Spec '{name}' not found"}
        return {"success": True, "spec": self._test_specs[name].to_dict()}
    
    def _test_measure(self, msg: dict) -> dict:
        name = msg.get("name", "test")
        value = msg.get("value", 0.0)
        if name not in self._test_specs:
            return {"success": False, "error": f"Spec '{name}' not found"}
        result = self._test_specs[name].measure(value)
        return {"success": True, "result": result}
    
    def _test_verify(self, msg: dict) -> dict:
        name = msg.get("name", "test")
        value = msg.get("value", 0.0)
        success = msg.get("success", True)
        if name not in self._test_specs:
            return {"success": False, "error": f"Spec '{name}' not found"}
        heat_added = self._test_specs[name].verify(value, success)
        return {"success": True, "heat_added": heat_added, "spec": self._test_specs[name].to_dict()}
    
    def _test_learn(self, msg: dict) -> dict:
        from core.constraints import tighten_spec
        name = msg.get("name", "test")
        value = msg.get("value", 0.0)
        success = msg.get("success", True)
        if name not in self._test_specs:
            return {"success": False, "error": f"Spec '{name}' not found"}
        tighten_spec(self._test_specs[name], value, success)
        return {"success": True, "spec": self._test_specs[name].to_dict()}
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TIMED INPUT TESTING
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _handle_input(self, msg: dict) -> dict:
        action = msg.get("action")
        button = msg.get("button", "unknown")
        
        if action == "press":
            self._input_states[button] = InputState(button=button, pressed_at=time.time())
            return {"success": True, "action": "press", "button": button}
        
        elif action == "release":
            if button not in self._input_states:
                return {"success": False, "error": f"Button '{button}' not pressed"}
            state = self._input_states[button]
            state.released_at = time.time()
            duration = state.duration
            
            # Learn timing
            spec_name = f"input_{button}_duration"
            if spec_name not in self._test_specs:
                from core.constraints import robinson_spec
                self._test_specs[spec_name] = robinson_spec(duration, "seconds")
            else:
                from core.constraints import tighten_spec
                tighten_spec(self._test_specs[spec_name], duration, True)
            
            del self._input_states[button]
            return {
                "success": True,
                "action": "release",
                "button": button,
                "duration": duration,
                "spec": self._test_specs[spec_name].to_dict()
            }
        
        return {"success": False, "error": f"Unknown input action: {action}"}
    
    # ═══════════════════════════════════════════════════════════════════════════
    # LIFECYCLE
    # ═══════════════════════════════════════════════════════════════════════════
    
    def start(self):
        """Start server in background thread."""
        if self._running or not HAS_WEBSOCKETS:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_server, daemon=True)
        self._thread.start()
        logger.info(f"Body server starting on port {self.port}")
    
    def _run_server(self):
        """Run async server in new event loop."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._async_run())
        finally:
            self._loop.close()
    
    async def _async_run(self):
        """Async server loop."""
        async with serve(self._handle_client, "0.0.0.0", self.port, ping_interval=60, ping_timeout=120):
            logger.info(f"✓ Body server on ws://0.0.0.0:{self.port}")
            while self._running:
                await asyncio.sleep(1)
    
    def stop(self):
        """Stop server."""
        self._running = False
        self.release_all()
        for body in list(self.bodies.values()):
            try:
                asyncio.run_coroutine_threadsafe(body.websocket.close(), self._loop)
            except:
                pass
        self.bodies.clear()
        logger.info("Body server stopped")
    
    def get_status(self) -> dict:
        """Get server status (Planck-Grounded)."""
        with self._world_buffer_lock:
            buffer_len = len(self._world_buffer)
        return {
            "running": self._running,
            "port": self.port,
            "bodies": len(self.bodies),
            "primary_body": self._primary_body,
            "test_specs": len(self._test_specs),
            "world_requests": self.world_requests,
            "has_last_world_state": self.last_world_state is not None,
            "last_world_t_K": self.last_world_t_K,
            "world_buffer_depth": buffer_len,
        }


# Standalone test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Body server requires a daemon. Use:")
    print("  python -m pi.daemon --body-port 8421")
