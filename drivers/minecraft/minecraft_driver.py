"""
Minecraft Bedrock Driver - PBAI plays Minecraft through manifold

ARCHITECTURE:
    Windows Client (vision_transformer + pbai_client)
        ↕ WebSocket
    MinecraftPort (this file)
        ↕ PortMessage
    MinecraftDriver (this file)
        ↕ Perception / Action
    EnvironmentCore → Manifold

COORDINATE MAPPING:
    Minecraft world (x, y, z) → Hypersphere (theta, phi, radius)

    Player heading (yaw 0-360°) → phi [0, 2π)
    Player altitude (y 0-320)   → theta: sea level (y=64) maps to equator (π/2),
                                   higher y → north pole (0), lower y → south pole (π)
    Entities/blocks near player → placed near driver node on hypersphere

HEAT ECONOMY:
    HEAT_SCALE = 2.0 (rich environment, many interactions)
    - Discovery (new biome, structure): K * 2.0
    - Combat (kill mob): K * 1.5
    - Crafting/building: K * 1.0
    - Movement to new area: K * 0.5
    - Idle observation: K * 0.2
    - Death: 0.0 (failure)

SUPPORTED ACTIONS:
    Movement: move_forward, move_backward, strafe_left, strafe_right, jump, sneak, sprint
    Combat:   attack, use
    Camera:   look_up, look_down, look_left, look_right
    Cardinal: turn_north, turn_south, turn_east, turn_west, turn_up, turn_down
    Other:    wait, open_inventory

CLOCK SYNC:
    Each game tick routed through this driver = one tick of Self time.
    Perceptions route through Clock, decisions through DecisionNode.
"""

import logging
import math
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from time import time

import sys
import os
# Go up two levels: minecraft/ → drivers/ → project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.node_constants import K
from core.driver_node import (
    SensorReport, MotorAction, MotorType, ActionPlan,
    press, hold_key, release_key, mouse_click, look, wait as motor_wait,
    create_plan
)
from drivers.environment import (
    Driver, Port, PortState, PortMessage,
    Perception, Action, ActionResult, NullPort
)

# Game knowledge (SpockBotMC/python-minecraft-data wrapping PrismarineJS)
try:
    import minecraft_data as _minecraft_data
    HAS_MC_DATA = True
except ImportError:
    HAS_MC_DATA = False

logger = logging.getLogger(__name__)

# Bedrock entity data lacks type/category — hardcode hostile mob names
_HOSTILE_MOBS = frozenset({
    "blaze", "cave_spider", "creeper", "drowned", "elder_guardian",
    "enderman", "endermite", "evoker", "ghast", "guardian", "hoglin",
    "husk", "magma_cube", "phantom", "piglin_brute", "pillager",
    "ravager", "shulker", "silverfish", "skeleton", "slime", "spider",
    "stray", "vex", "vindicator", "warden", "witch", "wither",
    "wither_skeleton", "zoglin", "zombie", "zombie_villager",
})
_NEUTRAL_MOBS = frozenset({
    "bee", "dolphin", "enderman", "goat", "iron_golem", "llama",
    "piglin", "polar_bear", "spider", "wolf", "zombified_piglin",
})


# ═══════════════════════════════════════════════════════════════════════════════
# COORDINATE MAPPING: Minecraft → Hypersphere
# ═══════════════════════════════════════════════════════════════════════════════

# Minecraft altitude range
MC_Y_MIN = -64       # Deepslate bottom (1.18+)
MC_Y_MAX = 320       # Build limit
MC_Y_SEA = 64        # Sea level → maps to equator

# View distance for normalizing entity/block distances
MC_VIEW_DISTANCE = 128.0  # ~8 chunks


def mc_altitude_to_theta(y: float) -> float:
    """
    Map Minecraft Y altitude to theta [0, π].

    Sea level (y=64) → equator (π/2)
    Build limit (y=320) → north pole (0)
    Bedrock (y=-64) → south pole (π)

    Piecewise linear: centers sea level at equator regardless of range asymmetry.
    """
    y = max(MC_Y_MIN, min(MC_Y_MAX, y))

    if y >= MC_Y_SEA:
        # Above sea level: y=64→π/2, y=320→0
        t = (y - MC_Y_SEA) / (MC_Y_MAX - MC_Y_SEA)  # 0→1
        theta = math.pi / 2 * (1.0 - t)               # π/2→0
    else:
        # Below sea level: y=64→π/2, y=-64→π
        t = (MC_Y_SEA - y) / (MC_Y_SEA - MC_Y_MIN)   # 0→1
        theta = math.pi / 2 + (math.pi / 2) * t       # π/2→π

    return theta


def mc_yaw_to_phi(yaw: float) -> float:
    """
    Map Minecraft yaw (0-360°, 0=south, 90=west) to phi [0, 2π).

    Minecraft yaw is clockwise from south.
    We map to standard mathematical angle (counterclockwise from east).
    """
    # Convert: MC south=0° → math east=0
    # MC: 0=S, 90=W, 180=N, 270=E
    # Math: 0=E, π/2=N, π=W, 3π/2=S
    phi = math.radians((270.0 - yaw) % 360.0)
    return phi % (2 * math.pi)


def mc_relative_to_sphere(dx: float, dy: float, dz: float,
                           player_theta: float, player_phi: float) -> Tuple[float, float]:
    """
    Map a relative Minecraft position (entity/block offset from player)
    to angular coordinates near the player on the hypersphere.

    Args:
        dx, dy, dz: Offset from player in Minecraft coords
        player_theta, player_phi: Player's position on hypersphere

    Returns:
        (theta, phi) for the entity on the hypersphere surface
    """
    # Horizontal distance and angle
    horiz_dist = math.sqrt(dx * dx + dz * dz)
    horiz_angle = math.atan2(dx, dz)  # Minecraft: +x=east, +z=south

    # Normalize distance to angular offset (max ~0.3 radians for view distance)
    dist_norm = min(horiz_dist / MC_VIEW_DISTANCE, 1.0)
    angular_offset = dist_norm * 0.3  # Small region near player

    # Vertical offset → theta shift
    vert_norm = dy / (MC_Y_MAX - MC_Y_MIN)
    theta_offset = -vert_norm * 0.2  # Up = lower theta

    # Place near player
    theta = max(0.0, min(math.pi, player_theta + theta_offset + angular_offset * 0.5))
    phi = (player_phi + horiz_angle) % (2 * math.pi)

    return theta, phi


# ═══════════════════════════════════════════════════════════════════════════════
# CUBE PROJECTION: Hypersphere → Color Cube
# ═══════════════════════════════════════════════════════════════════════════════

def sphere_to_cube(theta: float, phi: float) -> Tuple[float, float, float]:
    """
    Project hypersphere coords to Color Cube (x, y, tau).

    X = sin(θ)cos(φ)  → Blue(-1)/Yellow(+1)
    Y = sin(θ)sin(φ)  → Red(-1)/Green(+1)
    τ = cos(θ)        → Past(-τ)/Future(+τ)
    """
    x = math.sin(theta) * math.cos(phi)
    y = math.sin(theta) * math.sin(phi)
    tau = math.cos(theta)
    return x, y, tau


# ═══════════════════════════════════════════════════════════════════════════════
# MINECRAFT PORT: WebSocket communication with Windows Client
# ═══════════════════════════════════════════════════════════════════════════════

class MinecraftPort(Port):
    """
    WebSocket port for communicating with the Windows Client.

    The Windows Client runs vision_transformer + pbai_client,
    captures the Minecraft screen, and sends game state.
    This port receives that state and sends back commands.

    Protocol:
        Receive: {"type": "perception", "payload": {game_state}}
        Send:    {"type": "action", "payload": {command}}
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 9876,
                 config: Dict[str, Any] = None):
        super().__init__("minecraft_ws", config)
        self.host = host
        self.port_num = port
        self._ws_server = None
        self._client_connection = None
        self._message_queue: List[PortMessage] = []

    def connect(self) -> bool:
        """
        Start WebSocket server and wait for Windows Client connection.

        In practice, the actual WebSocket setup happens externally
        (pbai_client.py connects to us). This marks the port ready.
        """
        self.state = PortState.CONNECTED
        logger.info(f"MinecraftPort ready on {self.host}:{self.port_num}")
        return True

    def disconnect(self) -> bool:
        """Close WebSocket connection."""
        self.state = PortState.DISCONNECTED
        self._client_connection = None
        logger.info("MinecraftPort disconnected")
        return True

    def send(self, message: PortMessage) -> bool:
        """Send command to Windows Client."""
        if self.state != PortState.CONNECTED:
            logger.warning("MinecraftPort not connected, cannot send")
            return False

        self.last_message = message
        # In production, this would send via WebSocket:
        # await self._client_connection.send(json.dumps(message.to_dict()))
        logger.debug(f"MinecraftPort send: {message.msg_type}")
        return True

    def receive(self, timeout: float = 1.0) -> Optional[PortMessage]:
        """Receive game state from Windows Client."""
        if self._message_queue:
            return self._message_queue.pop(0)
        return None

    def feed_game_state(self, game_state: Dict[str, Any]) -> None:
        """
        External entry point: feed game state from Windows Client.

        Called by the WebSocket handler when data arrives from pbai_client.
        """
        msg = PortMessage(
            msg_type="perception",
            payload=game_state,
            source="minecraft_client",
            sequence=self.next_sequence()
        )
        self._message_queue.append(msg)


# ═══════════════════════════════════════════════════════════════════════════════
# MINECRAFT DRIVER: Translates between PBAI and Minecraft Bedrock
# ═══════════════════════════════════════════════════════════════════════════════

# Action → key/mouse mapping for Minecraft Bedrock
MC_ACTION_MAP = {
    # Single-key actions
    "move_forward":   {"motor": MotorType.KEY_HOLD, "key": "w", "duration": 1.0},
    "move_backward":  {"motor": MotorType.KEY_HOLD, "key": "s", "duration": 0.8},
    "strafe_left":    {"motor": MotorType.KEY_HOLD, "key": "a", "duration": 0.6},
    "strafe_right":   {"motor": MotorType.KEY_HOLD, "key": "d", "duration": 0.6},
    "jump":           {"motor": MotorType.KEY_PRESS, "key": "space"},
    "sneak":          {"motor": MotorType.KEY_HOLD, "key": "shift", "duration": 0.5},
    "sprint":         {"motor": MotorType.KEY_HOLD, "key": "ctrl", "duration": 1.5},
    "attack":         {"motor": MotorType.MOUSE_CLICK, "button": "left"},
    "use":            {"motor": MotorType.MOUSE_CLICK, "button": "right"},
    "look_up":        {"motor": MotorType.LOOK, "direction": (0.0, -210.0)},
    "look_down":      {"motor": MotorType.LOOK, "direction": (0.0, 210.0)},
    "look_left":      {"motor": MotorType.LOOK, "direction": (-280.0, 0.0)},
    "look_right":     {"motor": MotorType.LOOK, "direction": (280.0, 0.0)},
    "open_inventory": {"motor": MotorType.KEY_PRESS, "key": "e"},
    "wait":           {"motor": MotorType.WAIT, "duration": 0.5},
    # Combo actions — multiple keys held simultaneously
    "sprint_forward":  {"combo": ["ctrl", "w"], "duration": 1.5},
    "jump_forward":    {"combo": ["space", "w"], "duration": 0.8},
    "sprint_jump":     {"combo": ["ctrl", "w", "space"], "duration": 1.0},
    "strafe_left_fwd": {"combo": ["a", "w"], "duration": 0.8},
    "strafe_right_fwd":{"combo": ["d", "w"], "duration": 0.8},
    # Sequence actions — look-move combos (look where you're going)
    "explore_left": {"sequence": [
        {"motor_type": "look", "direction": (-210, 0)},
        {"motor_type": "wait", "duration": 0.15},
        {"motor_type": "key_hold", "key": "a"},
        {"motor_type": "key_hold", "key": "w"},
        {"motor_type": "wait", "duration": 0.8},
        {"motor_type": "key_release", "key": "w"},
        {"motor_type": "key_release", "key": "a"},
    ]},
    "explore_right": {"sequence": [
        {"motor_type": "look", "direction": (210, 0)},
        {"motor_type": "wait", "duration": 0.15},
        {"motor_type": "key_hold", "key": "d"},
        {"motor_type": "key_hold", "key": "w"},
        {"motor_type": "wait", "duration": 0.8},
        {"motor_type": "key_release", "key": "w"},
        {"motor_type": "key_release", "key": "d"},
    ]},
    "scout_ahead": {"sequence": [
        {"motor_type": "look", "direction": (0, -105)},
        {"motor_type": "wait", "duration": 0.1},
        {"motor_type": "key_hold", "key": "ctrl"},
        {"motor_type": "key_hold", "key": "w"},
        {"motor_type": "wait", "duration": 1.2},
        {"motor_type": "key_release", "key": "w"},
        {"motor_type": "key_release", "key": "ctrl"},
    ]},
    "watch_step": {"sequence": [
        {"motor_type": "look", "direction": (0, 140)},
        {"motor_type": "wait", "duration": 0.1},
        {"motor_type": "key_hold", "key": "w"},
        {"motor_type": "wait", "duration": 0.6},
        {"motor_type": "key_release", "key": "w"},
    ]},
    # Mining: look down + hold left click to break block below/ahead
    "mine_block": {"sequence": [
        {"motor_type": "look", "direction": (0, 280)},
        {"motor_type": "wait", "duration": 0.1},
        {"motor_type": "mouse_hold", "button": "left", "duration": 3.0},
    ]},
    # Mine forward: look straight ahead + hold attack to break block in front
    "mine_forward": {"sequence": [
        {"motor_type": "look", "direction": (0, 0)},
        {"motor_type": "wait", "duration": 0.1},
        {"motor_type": "mouse_hold", "button": "left", "duration": 3.0},
    ]},
    # Cardinal turning — yaw-aware, delta computed at act() time
    "turn_north":  {"motor": MotorType.LOOK, "cardinal": True},
    "turn_south":  {"motor": MotorType.LOOK, "cardinal": True},
    "turn_east":   {"motor": MotorType.LOOK, "cardinal": True},
    "turn_west":   {"motor": MotorType.LOOK, "cardinal": True},
    "turn_up":     {"motor": MotorType.LOOK, "cardinal": True},
    "turn_down":   {"motor": MotorType.LOOK, "cardinal": True},
    # Close inventory / escape UI — press Escape to close any open UI
    "close_ui": {"motor": MotorType.KEY_PRESS, "key": "escape"},
    # Swim up: hold space + w to surface when underwater
    "swim_up": {"combo": ["space", "w"], "duration": 1.5},
}

# Exploration weights — higher = more likely to be chosen during exploration.
# Favors deliberate movement over disruptive/stationary actions.
MC_ACTION_WEIGHTS = {
    "move_forward":    5.0,
    "move_backward":   2.0,
    "strafe_left":     0.5,
    "strafe_right":    0.5,
    "jump":            3.0,
    "sneak":           1.0,
    "sprint":          1.5,
    "attack":          3.0,
    "use":             2.5,
    "look_up":         3.0,
    "look_down":       3.0,
    "look_left":       3.5,
    "look_right":      3.5,
    "open_inventory":  0.1,
    "wait":            0.5,
    "sprint_forward":  4.0,
    "jump_forward":    3.5,
    "sprint_jump":     3.0,
    "strafe_left_fwd": 1.0,
    "strafe_right_fwd":1.0,
    "explore_left":    4.0,
    "explore_right":   4.0,
    "scout_ahead":     3.5,
    "watch_step":      2.0,
    "mine_block":      3.5,
    "mine_forward":    3.0,
    "turn_north":      2.0,
    "turn_south":      1.5,
    "turn_east":       2.0,
    "turn_west":       2.0,
    "turn_up":         1.5,
    "turn_down":       1.5,
    "close_ui":        0.0,    # Only used when UI is open (forced)
    "swim_up":         0.0,    # Only used when drowning (boosted dynamically)
}


# ═══════════════════════════════════════════════════════════════════════════════
# PERCEPTION-DRIVEN TARGETING: Vision peaks → camera movement
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# CARDINAL TURNING: Yaw-aware direction actions (NSEWUD)
# ═══════════════════════════════════════════════════════════════════════════════

# Cardinal yaw targets (Minecraft convention: clockwise from south)
MC_CARDINAL_YAW = {
    "turn_north": 180.0,
    "turn_south": 0.0,
    "turn_east":  270.0,
    "turn_west":  90.0,
}

# Vertical pitch targets
MC_CARDINAL_PITCH = {
    "turn_up":   -45.0,   # Look toward sky
    "turn_down":  45.0,   # Look toward ground
}

# Conversion: degrees of rotation → pixel delta for pydirectinput
# Baseline: 40px ≈ 40° at default MC sensitivity. Tunable.
MC_PIXELS_PER_DEGREE = 7.0

# Max single-frame turn (pixels). Allows up to ~160° turns.
MC_MAX_TURN_DELTA = 1260

# Dead zone: if already within this many degrees, skip the turn
MC_TURN_DEAD_ZONE = 5.0


MC_SCREEN_CENTER = 32           # Vision grid is 64x64
MC_LOOK_SENSITIVITY = 10.0     # Pixels → mouse delta multiplier
MC_TARGET_HEAT_THRESHOLD = 0.3  # Min peak heat to target
MC_TARGET_DEAD_ZONE = 3        # Cells from center to skip targeting
MAX_LOOK_DELTA = 350           # Clamp mouse movement


class MinecraftDriver(Driver):
    """
    PBAI Minecraft Bedrock driver.

    Inherits from Driver (environment.py) for EnvironmentCore integration.
    Translates Minecraft game state to normalized Perceptions and
    Actions to Minecraft commands via the Windows Client.

    COORDINATE SYSTEM:
        All node positions use hypersphere angular coordinates:
        - theta [0, π]: altitude-mapped (high=north, low=south)
        - phi [0, 2π): heading-mapped (player yaw)
        - radius = 1.0 (surface)

    HEAT ECONOMY:
        Rich environment with many interaction types.
        HEAT_SCALE = 2.0 to reflect complexity.
    """

    DRIVER_ID = "minecraft"
    DRIVER_NAME = "Minecraft Bedrock Driver"
    DRIVER_VERSION = "1.0.0"
    SUPPORTED_ACTIONS = list(MC_ACTION_MAP.keys())
    HEAT_SCALE = 2.0

    def __init__(self, port: Port = None, config: Dict[str, Any] = None,
                 manifold=None, mc_version: str = "1.19.50",
                 body_server=None):
        # Game state tracking
        self._game_state: Dict[str, Any] = {}
        self._seen_biomes: set = set()
        self._seen_entities: set = set()
        self._seen_structures: set = set()
        self._last_position: Optional[Tuple[float, float, float]] = None
        self._death_count: int = 0
        self._kill_count: int = 0
        self._blocks_mined: int = 0
        self._items_crafted: int = 0

        # Player state on hypersphere
        self._player_theta: float = math.pi / 2  # Equator (sea level)
        self._player_phi: float = 0.0             # Facing east

        # Raw MC yaw/pitch for cardinal turning
        self._player_yaw: float = 0.0    # Raw MC yaw (0-360)
        self._player_pitch: float = 0.0  # Raw MC pitch (-90 to 90)

        # Body server reference for streaming vision
        self._body_server = body_server

        # Targeting state (Layer 2: perception-driven camera)
        self._current_target: Optional[Dict[str, Any]] = None

        # UI state tracking — when inventory/chest/crafting UI is open,
        # movement actions don't work. Must close UI first.
        self._ui_open: bool = False

        # Drowning detection — tracks health trend + position_y
        self._last_health: float = 20.0
        self._underwater_ticks: int = 0

        # Load game knowledge (minecraft-data from SpockBotMC)
        self._mc_data = None
        if HAS_MC_DATA:
            try:
                self._mc_data = _minecraft_data(mc_version, "bedrock")
                logger.info(f"Loaded minecraft-data: {len(self._mc_data.blocks_list)} blocks, "
                           f"{len(self._mc_data.items_list)} items, "
                           f"{len(self._mc_data.entities_list)} entities")
            except Exception as e:
                logger.warning(f"Failed to load minecraft-data: {e}")

        # Initialize with MinecraftPort if none provided
        if port is None:
            port = MinecraftPort(config=config)

        super().__init__(port, config, manifold=manifold)

        # Register motor patterns
        self._register_motors()

    def _register_motors(self):
        """Register Minecraft motor actions with the DriverNode."""
        if not self.driver_node:
            return

        for action_name, mapping in MC_ACTION_MAP.items():
            if "sequence" in mapping:
                # Sequence actions have no single motor — skip registration
                continue
            elif "combo" in mapping:
                # Combo actions register as KEY_HOLD (primary key = first in combo)
                motor = MotorAction(
                    motor_type=MotorType.KEY_HOLD,
                    key=mapping["combo"][0],
                    duration=mapping.get("duration"),
                    heat_cost=1.0,
                    name=action_name,
                    description=f"Minecraft combo: {'+'.join(mapping['combo'])}"
                )
            else:
                motor_type = mapping["motor"]
                motor = MotorAction(
                    motor_type=motor_type,
                    key=mapping.get("key"),
                    button=mapping.get("button"),
                    direction=mapping.get("direction"),
                    duration=mapping.get("duration"),
                    heat_cost=1.0 if motor_type != MotorType.WAIT else 0.1,
                    name=action_name,
                    description=f"Minecraft: {action_name}"
                )
            self.driver_node.register_motor(action_name, motor)

    def get_actions(self, state: dict = None) -> List[str]:
        """Return valid actions for current state.

        Filters actions based on UI state and danger:
        - UI open: only close_ui is valid
        - UI closed: close_ui is excluded, open_inventory available
        """
        if self._ui_open:
            return ["close_ui"]

        # Normal play: exclude close_ui (only valid when UI is open)
        return [a for a in self.SUPPORTED_ACTIONS if a != "close_ui"]

    def get_action_weights(self, actions: list = None) -> Dict[str, float]:
        """Return exploration weights for actions (higher = more likely).

        Used by EnvironmentCore.decide() for weighted random exploration
        instead of uniform random, producing more deliberate movement.

        State-aware:
        - When drowning: swim_up and jump get massive boost
        - When target exists: look_at_target gets high weight
        """
        if actions is None:
            actions = self.get_actions()
        weights = {a: MC_ACTION_WEIGHTS.get(a, 1.0) for a in actions}

        # DROWNING: massively boost swim_up and jump
        if self._underwater_ticks >= 2:
            weights["swim_up"] = 15.0
            weights["jump"] = 10.0
            weights["jump_forward"] = 8.0
            # Suppress actions that keep you underwater
            weights["sneak"] = 0.0
            weights["mine_block"] = 0.0
            weights["wait"] = 0.0
            logger.info(f"DROWNING! Boosting swim_up/jump (underwater {self._underwater_ticks} ticks)")

        # If targeting found a valid peak, add look_at_target with high weight
        if self._current_target and "look_at_target" in MC_ACTION_MAP:
            weights["look_at_target"] = 6.0

        return weights

    # ═══════════════════════════════════════════════════════════════════════════
    # GAME KNOWLEDGE (minecraft-data)
    # ═══════════════════════════════════════════════════════════════════════════

    def get_block_info(self, name: str) -> Optional[dict]:
        """Look up block by name (e.g. 'dirt', 'stone', 'oak_log')."""
        if not self._mc_data:
            return None
        return self._mc_data.blocks_name.get(name)

    def get_item_info(self, name: str) -> Optional[dict]:
        """Look up item by name."""
        if not self._mc_data:
            return None
        return self._mc_data.items_name.get(name)

    def get_entity_info(self, name: str) -> Optional[dict]:
        """Look up entity type by name (e.g. 'zombie', 'cow')."""
        if not self._mc_data:
            return None
        return self._mc_data.entities_name.get(name)

    def get_recipes_for(self, item_name: str) -> list:
        """Get crafting recipes that produce this item."""
        if not self._mc_data or not hasattr(self._mc_data, 'recipes'):
            return []
        item = self._mc_data.items_name.get(item_name)
        if not item:
            return []
        return self._mc_data.recipes.get(str(item["id"]), [])

    def classify_entity_threat(self, entity_name: str) -> str:
        """Classify entity as hostile/neutral/passive using game data."""
        name = entity_name.lower()
        if name in _HOSTILE_MOBS:
            return "hostile"
        if name in _NEUTRAL_MOBS:
            return "neutral"
        # If we have mc_data, check if it's a known entity at all
        if self._mc_data and self._mc_data.entities_name.get(name):
            return "passive"
        return "unknown"

    @property
    def has_game_knowledge(self) -> bool:
        return self._mc_data is not None

    # ═══════════════════════════════════════════════════════════════════════════
    # DRIVER LIFECYCLE
    # ═══════════════════════════════════════════════════════════════════════════

    def initialize(self) -> bool:
        """Initialize driver and connect port."""
        if self.port:
            self.port.connect()
        self.active = True
        logger.info("MinecraftDriver initialized")
        return True

    def shutdown(self) -> bool:
        """Shutdown driver and disconnect port."""
        if self.driver_node:
            self.driver_node.save()
        if self.port:
            self.port.disconnect()
        self.active = False
        logger.info("MinecraftDriver shutdown")
        return True

    # ═══════════════════════════════════════════════════════════════════════════
    # PERCEPTION: Minecraft game state → normalized Perception
    # ═══════════════════════════════════════════════════════════════════════════

    def perceive(self) -> Perception:
        """
        Get current perception from Minecraft via Windows Client.

        Reads game state from port (fed by vision_transformer output),
        maps to hypersphere coordinates, and returns normalized Perception.
        """
        # Get latest game state: port message first, then stream buffer
        msg = self.port.receive() if self.port else None
        if msg and msg.payload:
            self._game_state = msg.payload
        elif self._body_server:
            streamed = self._body_server.get_latest_world()
            if streamed:
                self._game_state = streamed

        gs = self._game_state

        # Extract player position
        player = gs.get("player", {})
        px = player.get("x", 0.0)
        py = player.get("y", MC_Y_SEA)
        pz = player.get("z", 0.0)
        yaw = player.get("yaw", 0.0)
        health = player.get("health", 20.0)
        hunger = player.get("hunger", 20.0)

        # Store raw yaw/pitch for cardinal turning
        self._player_yaw = yaw % 360.0
        self._player_pitch = player.get("pitch", 0.0)

        # Map player to hypersphere
        self._player_theta = mc_altitude_to_theta(py)
        self._player_phi = mc_yaw_to_phi(yaw)

        # Cube projection
        cube_x, cube_y, cube_tau = sphere_to_cube(self._player_theta, self._player_phi)

        # ── Drowning detection ──
        # MC sea level is ~62. If player is below sea level AND health is dropping,
        # they're likely drowning. Also detect via oxygen/air if available.
        air = player.get("air", player.get("oxygen", -1))
        if air >= 0 and air < 150:
            # Direct air level detection (normal max is 300)
            self._underwater_ticks += 1
        elif py < MC_Y_SEA - 1 and health < self._last_health:
            # Below sea level + taking damage = probably drowning
            self._underwater_ticks += 1
        else:
            self._underwater_ticks = max(0, self._underwater_ticks - 1)
        self._last_health = health

        # ── Entities (enriched with game knowledge) ──
        entities = []
        entity_data = gs.get("entities", [])
        hostile_count = 0
        for ent in entity_data:
            ent_type = ent.get("type", "unknown")
            entities.append(ent_type)
            self._seen_entities.add(ent_type)
            # Classify threat using game knowledge
            threat = self.classify_entity_threat(ent_type)
            ent["_threat"] = threat
            if threat == "hostile":
                hostile_count += 1

        # ── Locations ──
        locations = []
        biome = gs.get("biome", "")
        if biome:
            locations.append(biome)
            self._seen_biomes.add(biome)
        structure = gs.get("structure", "")
        if structure:
            locations.append(structure)
            self._seen_structures.add(structure)
        dimension = gs.get("dimension", "overworld")
        locations.append(dimension)

        # ── Events ──
        events = gs.get("events", [])

        # ── Properties ──
        properties = {
            "health": health,
            "hunger": hunger,
            "position_x": px,
            "position_y": py,
            "position_z": pz,
            "yaw": yaw,
            "pitch": player.get("pitch", 0.0),
            "dimension": dimension,
            "biome": biome,
            "time_of_day": gs.get("time_of_day", 0),
            "is_raining": gs.get("is_raining", False),
            "theta": round(self._player_theta, 4),
            "phi": round(self._player_phi, 4),
            "cube_x": round(cube_x, 4),
            "cube_y": round(cube_y, 4),
            "cube_tau": round(cube_tau, 4),
            "state_key": self._build_state_key(gs),
        }

        # Add inventory summary if present
        inventory = gs.get("inventory", {})
        if inventory:
            properties["inventory_slots_used"] = inventory.get("slots_used", 0)
            properties["has_weapon"] = inventory.get("has_weapon", False)
            properties["has_food"] = inventory.get("has_food", False)

        # Game knowledge metadata
        if self._mc_data:
            properties["game_knowledge"] = True
            properties["hostile_count"] = hostile_count

        # Danger states
        properties["is_drowning"] = self._underwater_ticks >= 2
        properties["ui_open"] = self._ui_open

        # ── Heat: novelty/discovery ──
        novelty_heat = self._calculate_novelty_heat(gs)

        # ── Build perception ──
        perception = Perception(
            entities=entities,
            locations=locations,
            properties=properties,
            events=events,
            heat_value=novelty_heat,
            raw=gs
        )

        # Targeting: extract vision peaks and store for action selection
        targets = self._extract_targets()
        if targets:
            properties["has_target"] = True
            properties["target_heat"] = targets[0]["heat"]
        else:
            properties["has_target"] = False
            properties["target_heat"] = 0.0

        # Feed to DriverNode for learning
        self.feed_perception(perception)

        # Track position for movement heat
        self._last_position = (px, py, pz)

        return perception

    # ═══════════════════════════════════════════════════════════════════════════
    # TARGETING: Vision peaks → camera movement
    # ═══════════════════════════════════════════════════════════════════════════

    def _extract_targets(self) -> List[Dict[str, Any]]:
        """
        Extract targeting candidates from vision peaks.

        Reads peaks from game state, filters by heat threshold,
        computes screen-space offsets from center.

        Returns:
            List of target dicts sorted by heat descending
        """
        peaks = self._game_state.get("peaks", [])
        targets = []

        for peak in peaks:
            heat = peak.get("heat", 0.0)
            if heat < MC_TARGET_HEAT_THRESHOLD:
                continue

            px = peak.get("x", MC_SCREEN_CENTER)
            py = peak.get("y", MC_SCREEN_CENTER)
            dx = px - MC_SCREEN_CENTER
            dy = py - MC_SCREEN_CENTER

            # Get angular coords if available
            theta = peak.get("theta", self._player_theta)
            phi = peak.get("phi", self._player_phi)
            cube_x, cube_y, cube_tau = sphere_to_cube(theta, phi)

            targets.append({
                "heat": heat,
                "dx": dx,
                "dy": dy,
                "theta": theta,
                "phi": phi,
                "cube_x": round(cube_x, 4),
                "cube_y": round(cube_y, 4),
                "cube_tau": round(cube_tau, 4),
            })

        targets.sort(key=lambda t: t["heat"], reverse=True)
        return targets

    def _get_target_action(self) -> Optional[str]:
        """
        Build a look_at_target action from the best vision peak.

        Returns:
            "look_at_target" if a valid target exists, None otherwise
        """
        targets = self._extract_targets()
        if not targets:
            self._current_target = None
            return None

        best = targets[0]

        # Skip if already centered (within dead zone)
        if abs(best["dx"]) < MC_TARGET_DEAD_ZONE and abs(best["dy"]) < MC_TARGET_DEAD_ZONE:
            self._current_target = None
            return None

        # Compute mouse delta, clamped
        mouse_dx = max(-MAX_LOOK_DELTA, min(MAX_LOOK_DELTA,
                       best["dx"] * MC_LOOK_SENSITIVITY))
        mouse_dy = max(-MAX_LOOK_DELTA, min(MAX_LOOK_DELTA,
                       best["dy"] * MC_LOOK_SENSITIVITY))

        # Dynamically register look_at_target in MC_ACTION_MAP
        MC_ACTION_MAP["look_at_target"] = {
            "motor": MotorType.LOOK,
            "direction": (mouse_dx, mouse_dy),
        }

        self._current_target = best
        return "look_at_target"

    def get_targeting_context(self) -> Optional[Dict[str, Any]]:
        """Return current target info for Introspector to read."""
        return self._current_target

    def _compute_cardinal_direction(self, action_type: str) -> Tuple[float, float]:
        """Compute pixel delta to face a cardinal direction.

        Returns (dx, dy) in pixels for pydirectinput.moveRel().
        Finds shortest rotation path (never turns more than 180°).
        """
        if action_type in MC_CARDINAL_YAW:
            target_yaw = MC_CARDINAL_YAW[action_type]
            # Shortest rotation: delta in [-180, 180]
            delta_deg = (target_yaw - self._player_yaw + 180) % 360 - 180
            # MC yaw is clockwise, mouse right is positive → same sign
            dx = delta_deg * MC_PIXELS_PER_DEGREE
            dx = max(-MC_MAX_TURN_DELTA, min(MC_MAX_TURN_DELTA, dx))
            return (dx, 0.0)

        elif action_type in MC_CARDINAL_PITCH:
            target_pitch = MC_CARDINAL_PITCH[action_type]
            delta_deg = target_pitch - self._player_pitch
            # MC pitch: positive = down, mouse dy positive = down → same sign
            dy = delta_deg * MC_PIXELS_PER_DEGREE
            dy = max(-MC_MAX_TURN_DELTA, min(MC_MAX_TURN_DELTA, dy))
            return (0.0, dy)

        return (0.0, 0.0)

    def get_domain_context(self, perception) -> Dict[str, Any]:
        """Minecraft-scoped context for introspection."""
        # Domain prefix for this dimension
        dim = self._game_state.get("dimension", "ow")[:3]

        # Domain-scoped hot nodes: only concepts from this domain
        domain_nodes = []
        if self.manifold:
            for node in self.manifold.nodes.values():
                if node.concept in ('identity', 'ego', 'conscience'):
                    continue
                if node.concept.startswith('bootstrap'):
                    continue
                if node.heat == float('inf') or node.existence != 'actual':
                    continue
                # Include: domain state keys OR action names (no digits = action)
                is_domain_state = node.concept.startswith(dim + '_')
                is_action = '_' in node.concept and not any(c.isdigit() for c in node.concept)
                is_general = '_' not in node.concept  # abstract concepts
                if is_domain_state or is_action or is_general:
                    domain_nodes.append(node)
            domain_nodes.sort(key=lambda n: n.heat, reverse=True)

        # Targeting
        targets = self._extract_targets()
        target_action = self._get_target_action() if targets else None

        # Available plans from driver state
        reactive_plans = self._get_reactive_plans(perception)

        return {
            "domain_prefix": dim,
            "hot_nodes": domain_nodes[:10],
            "targets": targets,
            "target_action": target_action,
            "reactive_plans": reactive_plans,
            "action_durations": {a: MC_ACTION_MAP[a].get("duration", 0.2) for a in MC_ACTION_MAP if "duration" in MC_ACTION_MAP[a]},
        }

    def _get_reactive_plans(self, perception) -> List[List[str]]:
        """State-driven plans — driver decides when to trigger."""
        plans = []
        props = perception.properties if perception else {}
        health = props.get("health", 20)
        hostile_count = props.get("hostile_count", 0)

        # Flee: low health + hostiles
        if health < 8 and hostile_count > 0:
            plans.append(["sprint_forward", "sprint_jump", "sprint_forward"])

        # Eat: low hunger (when we can detect food in inventory)
        hunger = props.get("hunger", 20)
        if hunger < 12:
            plans.append(["use"])  # Eating with food selected

        return plans

    def _build_state_key(self, gs: Dict[str, Any]) -> str:
        """Build a concise state key from game state."""
        player = gs.get("player", {})
        health = int(player.get("health", 20))
        biome = gs.get("biome", "unknown")[:10]
        dim = gs.get("dimension", "ow")[:3]

        # Nearby entities summary (use enriched _threat if available, fallback to hostile flag)
        entities = gs.get("entities", [])
        hostile_count = sum(1 for e in entities
                           if e.get("_threat") == "hostile" or e.get("hostile", False))
        passive_count = len(entities) - hostile_count

        return f"{dim}_{biome}_h{health}_e{hostile_count}p{passive_count}"

    def _calculate_novelty_heat(self, gs: Dict[str, Any]) -> float:
        """Calculate heat from novelty/discovery."""
        heat = 0.0

        # New biome discovery
        biome = gs.get("biome", "")
        if biome and biome not in self._seen_biomes:
            heat += self.scale_heat(K * 2.0)

        # New structure discovery
        structure = gs.get("structure", "")
        if structure and structure not in self._seen_structures:
            heat += self.scale_heat(K * 2.0)

        # New entity type
        for ent in gs.get("entities", []):
            ent_type = ent.get("type", "")
            if ent_type and ent_type not in self._seen_entities:
                heat += self.scale_heat(K * 0.5)

        # Movement to new area
        player = gs.get("player", {})
        px, py, pz = player.get("x", 0), player.get("y", 0), player.get("z", 0)
        if self._last_position:
            lx, ly, lz = self._last_position
            dist = math.sqrt((px-lx)**2 + (py-ly)**2 + (pz-lz)**2)
            if dist > 16:  # Moved more than a chunk
                heat += self.scale_heat(K * 0.5)

        # Base observation heat
        if heat == 0.0:
            heat = self.scale_heat(K * 0.2)

        return heat

    # ═══════════════════════════════════════════════════════════════════════════
    # ACTION: normalized Action → Minecraft command
    # ═══════════════════════════════════════════════════════════════════════════

    def act(self, action: Action) -> ActionResult:
        """
        Execute action in Minecraft via Windows Client.

        Translates normalized Action to Minecraft-specific MotorAction
        and sends through the port to the Windows Client.
        """
        action_type = action.action_type

        # Track UI state: open_inventory opens UI, close_ui/escape closes it
        if action_type == "open_inventory":
            self._ui_open = True
        elif action_type == "close_ui":
            self._ui_open = False

        mapping = MC_ACTION_MAP.get(action_type)

        if not mapping:
            return ActionResult(
                success=False,
                outcome=f"Unknown Minecraft action: {action_type}",
                heat_value=0.0
            )

        # Cardinal turning: compute direction dynamically from current yaw/pitch
        if mapping.get("cardinal"):
            direction = self._compute_cardinal_direction(action_type)
            # Skip if already facing target (within dead zone)
            if abs(direction[0]) < MC_TURN_DEAD_ZONE and abs(direction[1]) < MC_TURN_DEAD_ZONE:
                return ActionResult(
                    success=True,
                    outcome=f"Already facing {action_type.replace('turn_', '')}",
                    heat_value=self.scale_heat(K * 0.05),
                )
            command = {
                "action": action_type,
                "motor_type": "look",
                "direction": direction,
            }
        # Build command for Windows Client
        elif "sequence" in mapping:
            # Sequence action: multi-step look+move combo
            command = {
                "action": action_type,
                "motor_type": "sequence",
                "sequence": mapping["sequence"],
            }
        elif "combo" in mapping:
            # Combo action: hold multiple keys simultaneously via sequence
            keys = mapping["combo"]
            duration = mapping.get("duration", 0.8)
            seq = []
            for k in keys:
                seq.append({"motor_type": "key_hold", "key": k})
            seq.append({"motor_type": "wait", "duration": duration})
            for k in reversed(keys):
                seq.append({"motor_type": "key_release", "key": k})
            command = {
                "action": action_type,
                "motor_type": "sequence",
                "sequence": seq,
            }
        else:
            command = {
                "action": action_type,
                "motor_type": mapping["motor"].value,
                "key": mapping.get("key"),
                "button": mapping.get("button"),
                "direction": mapping.get("direction"),
                "duration": mapping.get("duration", 0.2),
                "target": action.target,
                "parameters": action.parameters,
            }

        # Send via body server (real WebSocket to PC) or fall back to port
        sent = False
        if self._body_server:
            sent = self._body_server.send_action(command)
        elif self.port:
            msg = PortMessage(
                msg_type="action",
                payload=command,
                source="minecraft_driver",
                sequence=self.port.next_sequence() if hasattr(self.port, 'next_sequence') else 0
            )
            sent = self.port.send(msg)

        # Determine heat from action type
        heat = self._action_heat(action_type)

        result = ActionResult(
            success=sent,
            outcome=f"Executed {action_type}" if sent else f"Failed to send {action_type}",
            heat_value=heat if sent else 0.0,
            changes={"action": action_type, "command": command}
        )

        # Feed to DriverNode for learning
        self.feed_result(result, action)

        return result

    def _action_heat(self, action_type: str) -> float:
        """Determine heat reward for action type."""
        heat_map = {
            "move_forward": K * 0.3,
            "move_backward": K * 0.2,
            "strafe_left": K * 0.2,
            "strafe_right": K * 0.2,
            "jump": K * 0.3,
            "sneak": K * 0.1,
            "sprint": K * 0.4,
            "attack": K * 0.8,
            "use": K * 0.6,
            "look_up": K * 0.1,
            "look_down": K * 0.1,
            "look_left": K * 0.1,
            "look_right": K * 0.1,
            "open_inventory": K * 0.2,
            "wait": K * 0.05,
            "look_at_target": K * 0.15,
            "mine_block": K * 1.0,
            "mine_forward": K * 1.0,
            "turn_north": K * 0.15,
            "turn_south": K * 0.15,
            "turn_east": K * 0.15,
            "turn_west": K * 0.15,
            "turn_up": K * 0.15,
            "turn_down": K * 0.15,
            "close_ui": K * 0.1,
            "swim_up": K * 0.5,
        }
        return self.scale_heat(heat_map.get(action_type, K * 0.1))

    # ═══════════════════════════════════════════════════════════════════════════
    # ACTION PLANS: Pre-built sequences for common goals
    # ═══════════════════════════════════════════════════════════════════════════

    def register_default_plans(self):
        """Register default action plans for common Minecraft activities."""
        if not self.driver_node:
            return

        # Mine tree: look up, walk forward, hold attack
        mine_tree = create_plan(
            name="mine_tree",
            goal="Break a tree trunk",
            steps=[
                look(0.0, -30.0),           # Look up at tree
                hold_key("w", duration=1.0), # Walk to tree
                MotorAction(MotorType.MOUSE_HOLD, button="left",
                           duration=3.0, heat_cost=2.0, name="hold_attack"),
            ],
            requires=["near_tree"],
            provides=["wood"]
        )
        self.driver_node.add_plan(mine_tree)

        # Flee: sprint away
        flee = create_plan(
            name="flee",
            goal="Run away from danger",
            steps=[
                look(180.0, 0.0),            # Turn around
                hold_key("w", duration=3.0),  # Run
                press("ctrl"),                # Sprint
            ],
            requires=["hostile_nearby"],
            provides=["safety"]
        )
        self.driver_node.add_plan(flee)

        # Eat: open inventory, find food, eat
        eat = create_plan(
            name="eat",
            goal="Restore hunger",
            steps=[
                press("e"),                   # Open inventory
                motor_wait(0.5),              # Wait for UI
                mouse_click(button="right"),  # Use food
            ],
            requires=["has_food", "hunger_low"],
            provides=["hunger_restored"]
        )
        self.driver_node.add_plan(eat)

        logger.info("Registered default Minecraft action plans")

    # ═══════════════════════════════════════════════════════════════════════════
    # GAME STATE HELPERS
    # ═══════════════════════════════════════════════════════════════════════════

    def get_player_sphere_position(self) -> Tuple[float, float, float]:
        """Get player's current position on hypersphere (theta, phi, radius)."""
        return self._player_theta, self._player_phi, 1.0

    def get_player_cube_position(self) -> Tuple[float, float, float]:
        """Get player's current Color Cube projection (x, y, tau)."""
        return sphere_to_cube(self._player_theta, self._player_phi)

    def get_nearby_entity_positions(self) -> List[Dict[str, Any]]:
        """Get nearby entities with their hypersphere positions."""
        entities = []
        for ent in self._game_state.get("entities", []):
            # Relative position from player
            dx = ent.get("dx", ent.get("x", 0))
            dy = ent.get("dy", ent.get("y", 0))
            dz = ent.get("dz", ent.get("z", 0))

            theta, phi = mc_relative_to_sphere(
                dx, dy, dz,
                self._player_theta, self._player_phi
            )
            cube_x, cube_y, cube_tau = sphere_to_cube(theta, phi)

            entities.append({
                "type": ent.get("type", "unknown"),
                "hostile": ent.get("hostile", False),
                "theta": round(theta, 4),
                "phi": round(phi, 4),
                "cube_x": round(cube_x, 4),
                "cube_y": round(cube_y, 4),
                "cube_tau": round(cube_tau, 4),
                "distance": math.sqrt(dx*dx + dy*dy + dz*dz),
            })

        return entities

    def is_in_danger(self) -> bool:
        """Check if player is in immediate danger."""
        gs = self._game_state
        player = gs.get("player", {})
        health = player.get("health", 20)

        # Low health
        if health <= 6:
            return True

        # Hostile entities nearby
        for ent in gs.get("entities", []):
            if ent.get("hostile", False):
                dist = math.sqrt(
                    ent.get("dx", 100)**2 +
                    ent.get("dy", 100)**2 +
                    ent.get("dz", 100)**2
                )
                if dist < 8:  # Within attack range
                    return True

        return False

    def get_info(self) -> dict:
        """Get driver information with Minecraft-specific stats."""
        info = super().get_info()
        info.update({
            "biomes_discovered": len(self._seen_biomes),
            "entity_types_seen": len(self._seen_entities),
            "structures_found": len(self._seen_structures),
            "player_theta": round(self._player_theta, 4),
            "player_phi": round(self._player_phi, 4),
            "game_knowledge": self._mc_data is not None,
        })
        if self._mc_data:
            info["known_blocks"] = len(self._mc_data.blocks_list)
            info["known_items"] = len(self._mc_data.items_list)
            info["known_entities"] = len(self._mc_data.entities_list)
        return info


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY
# ═══════════════════════════════════════════════════════════════════════════════

def create_minecraft_driver(manifold=None, config: Dict[str, Any] = None,
                            body_server=None) -> MinecraftDriver:
    """
    Create a MinecraftDriver instance.

    Args:
        manifold: PBAI manifold (optional, can connect later)
        config: Driver configuration
        body_server: BodyServer for streaming vision (optional)

    Returns:
        Configured MinecraftDriver
    """
    port = MinecraftPort(config=config)
    driver = MinecraftDriver(port=port, config=config, manifold=manifold,
                             body_server=body_server)
    driver.register_default_plans()
    return driver


# ═══════════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    print("=== Minecraft Bedrock Driver Self-Test ===\n")
    errors = 0

    def check(cond, msg):
        global errors
        if cond:
            print(f"  PASS: {msg}")
        else:
            print(f"  FAIL: {msg}")
            errors += 1

    # ── Coordinate mapping ──
    print("1. Coordinate Mapping")

    # Sea level → equator
    theta_sea = mc_altitude_to_theta(MC_Y_SEA)
    check(abs(theta_sea - math.pi / 2) < 0.1, f"Sea level y={MC_Y_SEA} → theta≈π/2 ({theta_sea:.3f})")

    # Build limit → near north pole
    theta_high = mc_altitude_to_theta(MC_Y_MAX)
    check(theta_high < 0.1, f"Build limit y={MC_Y_MAX} → theta≈0 ({theta_high:.3f})")

    # Bedrock → near south pole
    theta_low = mc_altitude_to_theta(MC_Y_MIN)
    check(theta_low > math.pi - 0.1, f"Bedrock y={MC_Y_MIN} → theta≈π ({theta_low:.3f})")

    # Yaw mapping
    phi_south = mc_yaw_to_phi(0.0)
    phi_west = mc_yaw_to_phi(90.0)
    phi_north = mc_yaw_to_phi(180.0)
    phi_east = mc_yaw_to_phi(270.0)
    check(0 <= phi_south < 2 * math.pi, f"Yaw 0° (south) → phi={phi_south:.3f}")
    check(0 <= phi_east < 2 * math.pi, f"Yaw 270° (east) → phi={phi_east:.3f}")

    # Cube projection
    cx, cy, ctau = sphere_to_cube(math.pi / 2, 0.0)
    check(abs(cx - 1.0) < 0.01, f"Equator, phi=0 → cube_x≈1.0 ({cx:.3f})")
    check(abs(ctau) < 0.01, f"Equator → cube_tau≈0 ({ctau:.3f})")

    # ── Driver instantiation ──
    print("\n2. Driver Instantiation")

    driver = MinecraftDriver()
    check(driver.DRIVER_ID == "minecraft", f"DRIVER_ID = {driver.DRIVER_ID}")
    check(driver.HEAT_SCALE == 2.0, f"HEAT_SCALE = {driver.HEAT_SCALE}")
    check(len(driver.SUPPORTED_ACTIONS) == len(MC_ACTION_MAP),
          f"SUPPORTED_ACTIONS count = {len(driver.SUPPORTED_ACTIONS)}")
    check(driver.born, "Driver born after __init__")

    # ── Port ──
    print("\n3. Port Communication")

    port = MinecraftPort()
    check(port.state == PortState.DISCONNECTED, "Port starts disconnected")
    port.connect()
    check(port.state == PortState.CONNECTED, "Port connected")

    # Feed game state
    test_state = {
        "player": {"x": 100, "y": 64, "z": 200, "yaw": 90, "health": 18, "hunger": 15},
        "biome": "plains",
        "dimension": "overworld",
        "entities": [
            {"type": "cow", "dx": 5, "dy": 0, "dz": 3},
            {"type": "zombie", "dx": 10, "dy": 0, "dz": -5, "hostile": True},
        ],
        "events": ["zombie spotted"],
        "time_of_day": 6000,
    }
    port.feed_game_state(test_state)
    msg = port.receive()
    check(msg is not None, "Received game state message")
    check(msg.msg_type == "perception", f"Message type = {msg.msg_type}")

    port.disconnect()
    check(port.state == PortState.DISCONNECTED, "Port disconnected")

    # ── Perception ──
    print("\n4. Perception")

    driver2 = MinecraftDriver(port=MinecraftPort())
    driver2.port.connect()
    driver2.port.feed_game_state(test_state)

    perception = driver2.perceive()
    check(len(perception.entities) == 2, f"Entities count = {len(perception.entities)}")
    check("cow" in perception.entities, "Cow in entities")
    check("plains" in perception.locations, "Plains in locations")
    check(perception.properties["health"] == 18, f"Health = {perception.properties['health']}")
    check("theta" in perception.properties, "Theta in properties")
    check("phi" in perception.properties, "Phi in properties")
    check("cube_x" in perception.properties, "Cube X in properties")
    check(perception.heat_value > 0, f"Heat > 0 ({perception.heat_value:.3f})")

    # ── Action ──
    print("\n5. Action")

    driver2.active = True
    result = driver2.act(Action(action_type="move_forward"))
    check(result.success, "Move forward succeeded")
    check(result.heat_value > 0, f"Move heat > 0 ({result.heat_value:.3f})")

    result2 = driver2.act(Action(action_type="attack"))
    check(result2.success, "Attack succeeded")
    check(result2.heat_value > result.heat_value, "Attack heat > move heat")

    result3 = driver2.act(Action(action_type="invalid_action"))
    check(not result3.success, "Invalid action failed")

    # Combo actions
    result4 = driver2.act(Action(action_type="sprint_forward"))
    check(result4.success, "sprint_forward combo succeeded")
    result5 = driver2.act(Action(action_type="jump_forward"))
    check(result5.success, "jump_forward combo succeeded")
    result6 = driver2.act(Action(action_type="sprint_jump"))
    check(result6.success, "sprint_jump combo succeeded")

    # Verify combo count in SUPPORTED_ACTIONS
    combos = [a for a in driver.SUPPORTED_ACTIONS if "combo" in MC_ACTION_MAP.get(a, {})]
    check(len(combos) == 6, f"6 combo actions defined ({len(combos)})")

    # ── Entity positions ──
    print("\n6. Entity Positions")

    entities = driver2.get_nearby_entity_positions()
    check(len(entities) == 2, f"Entity positions count = {len(entities)}")
    for ent in entities:
        check(0 <= ent["theta"] <= math.pi, f"{ent['type']} theta={ent['theta']:.3f} in range")
        check(0 <= ent["phi"] < 2 * math.pi, f"{ent['type']} phi={ent['phi']:.3f} in range")

    # ── Danger detection ──
    print("\n7. Danger Detection")

    # Zombie at distance 10 is not immediately dangerous
    check(not driver2.is_in_danger(), "Not in danger (zombie at dist ~11)")

    # Feed close hostile
    close_state = dict(test_state)
    close_state["entities"] = [{"type": "creeper", "dx": 3, "dy": 0, "dz": 2, "hostile": True}]
    driver2.port.feed_game_state(close_state)
    driver2.perceive()
    check(driver2.is_in_danger(), "In danger (creeper at dist ~3.6)")

    # ── Info ──
    print("\n8. Driver Info")

    info = driver2.get_info()
    check(info["id"] == "minecraft", f"Info id = {info['id']}")
    check(info["biomes_discovered"] >= 1, f"Biomes discovered = {info['biomes_discovered']}")
    check("player_theta" in info, "Player theta in info")

    # ── Game Knowledge ──
    print("\n9. Game Knowledge (minecraft-data)")

    if HAS_MC_DATA:
        gk = MinecraftDriver()
        check(gk.has_game_knowledge, "Game knowledge loaded")

        # Block lookup
        dirt = gk.get_block_info("dirt")
        check(dirt is not None, f"Block 'dirt' found: {dirt.get('displayName', 'N/A')}")
        stone = gk.get_block_info("stone")
        check(stone is not None, f"Block 'stone' found: {stone.get('displayName', 'N/A')}")
        check(gk.get_block_info("not_a_block") is None, "Unknown block returns None")

        # Item lookup
        diamond = gk.get_item_info("diamond")
        check(diamond is not None, f"Item 'diamond' found: {diamond.get('displayName', 'N/A')}")
        check(gk.get_item_info("not_an_item") is None, "Unknown item returns None")

        # Entity lookup
        zombie = gk.get_entity_info("zombie")
        check(zombie is not None, f"Entity 'zombie' found: {zombie.get('displayName', 'N/A')}")

        # Threat classification
        check(gk.classify_entity_threat("zombie") == "hostile", "Zombie is hostile")
        check(gk.classify_entity_threat("creeper") == "hostile", "Creeper is hostile")
        check(gk.classify_entity_threat("cow") == "passive", "Cow is passive")
        check(gk.classify_entity_threat("wolf") == "neutral", "Wolf is neutral")
        check(gk.classify_entity_threat("xyzzy") == "unknown", "Unknown entity = unknown")

        # Info includes game knowledge
        gk_info = gk.get_info()
        check(gk_info["game_knowledge"] is True, "Info reports game_knowledge=True")
        check(gk_info["known_blocks"] > 700, f"Known blocks = {gk_info['known_blocks']}")
        check(gk_info["known_items"] > 1000, f"Known items = {gk_info['known_items']}")

        # Perceive with game knowledge enrichment
        gk2 = MinecraftDriver(port=MinecraftPort())
        gk2.port.connect()
        gk2.port.feed_game_state(test_state)
        p = gk2.perceive()
        check(p.properties.get("game_knowledge") is True, "Perception has game_knowledge=True")
        check(p.properties.get("hostile_count", -1) >= 0, f"hostile_count in properties = {p.properties.get('hostile_count')}")
    else:
        print("  SKIP: minecraft-data not installed")

    # ── Sequence Actions (Layer 1) ──
    print("\n10. Sequence Actions (Look-Move Combos)")

    sequences = [a for a in driver.SUPPORTED_ACTIONS if "sequence" in MC_ACTION_MAP.get(a, {})]
    check(len(sequences) == 6, f"6 sequence actions defined ({len(sequences)})")
    check("explore_left" in driver.SUPPORTED_ACTIONS, "explore_left in SUPPORTED_ACTIONS")
    check("scout_ahead" in driver.SUPPORTED_ACTIONS, "scout_ahead in SUPPORTED_ACTIONS")
    check("watch_step" in driver.SUPPORTED_ACTIONS, "watch_step in SUPPORTED_ACTIONS")

    # Execute sequence actions
    for seq_name in ["explore_left", "explore_right", "scout_ahead", "watch_step"]:
        r = driver2.act(Action(action_type=seq_name))
        check(r.success, f"{seq_name} sequence succeeded")

    # Verify sequence structure
    explore_map = MC_ACTION_MAP["explore_left"]
    check("sequence" in explore_map, "explore_left has 'sequence' key")
    check(len(explore_map["sequence"]) == 7, f"explore_left has 7 steps ({len(explore_map['sequence'])})")
    check(explore_map["sequence"][0]["motor_type"] == "look", "explore_left starts with look")

    # Weights exist for sequence actions
    w = driver.get_action_weights()
    check("explore_left" in w, "explore_left has weight")
    check("scout_ahead" in w, "scout_ahead has weight")
    check(w.get("scout_ahead", 0) == 3.5, f"scout_ahead weight = 3.5 ({w.get('scout_ahead')})")

    # Total action count: 17 single + 6 cardinal + 5 combo + 6 sequence = 34
    total_actions = len(MC_ACTION_MAP)
    check(total_actions == 34, f"Total actions = 34 ({total_actions})")

    # ── Targeting (Layer 2) ──
    print("\n11. Perception-Driven Targeting")

    # No peaks → no target
    driver3 = MinecraftDriver(port=MinecraftPort())
    driver3.port.connect()
    no_peak_state = dict(test_state)
    no_peak_state["peaks"] = []
    driver3.port.feed_game_state(no_peak_state)
    driver3.perceive()
    check(driver3._get_target_action() is None, "No peaks → no target action")
    check(driver3.get_targeting_context() is None, "No peaks → no targeting context")

    # Peaks with heat → targeting
    peak_state = dict(test_state)
    peak_state["peaks"] = [
        {"x": 48, "y": 20, "heat": 0.8, "theta": 1.2, "phi": 0.5},
        {"x": 33, "y": 33, "heat": 0.1},  # Below threshold
    ]
    driver3.port.feed_game_state(peak_state)
    driver3.perceive()

    targets = driver3._extract_targets()
    check(len(targets) == 1, f"1 target above threshold ({len(targets)})")
    check(targets[0]["heat"] == 0.8, f"Target heat = 0.8 ({targets[0]['heat']})")
    check(targets[0]["dx"] == 16, f"Target dx = 16 ({targets[0]['dx']})")

    target_action = driver3._get_target_action()
    check(target_action == "look_at_target", f"Target action = look_at_target ({target_action})")
    check(driver3.get_targeting_context() is not None, "Targeting context exists")
    check("look_at_target" in MC_ACTION_MAP, "look_at_target registered in MC_ACTION_MAP")

    # look_at_target has correct direction
    lat_map = MC_ACTION_MAP.get("look_at_target", {})
    check("direction" in lat_map, "look_at_target has direction")
    lat_dir = lat_map.get("direction", (0, 0))
    check(abs(lat_dir[0] - 160.0) < 0.1, f"look_at_target dx = 160.0 ({lat_dir[0]})")

    # get_action_weights includes look_at_target when target exists
    w_target = driver3.get_action_weights()
    check("look_at_target" in w_target, "look_at_target in action weights")
    check(w_target["look_at_target"] == 6.0, f"look_at_target weight = 6.0 ({w_target.get('look_at_target')})")

    # Dead zone: peak at center → no target
    center_state = dict(test_state)
    center_state["peaks"] = [{"x": 33, "y": 31, "heat": 0.9}]
    driver3.port.feed_game_state(center_state)
    driver3.perceive()
    check(driver3._get_target_action() is None, "Dead zone peak → no target action")

    # Perception includes targeting properties
    check("has_target" in perception.properties, "has_target in perception properties")
    check("target_heat" in perception.properties, "target_heat in perception properties")

    # ── Cardinal Turning (NSEWUD) ──
    print("\n12. Cardinal Turning (NSEWUD)")

    driver4 = MinecraftDriver(port=MinecraftPort())
    driver4.port.connect()
    driver4._player_yaw = 0.0  # Facing south
    driver4._player_pitch = 0.0  # Looking ahead

    # turn_north: south(0) → north(180) = ±180° (ambiguous), clamped to ±1260
    dx, dy = driver4._compute_cardinal_direction("turn_north")
    check(abs(dx) == MC_MAX_TURN_DELTA, f"South→North |dx|=1260 (clamped, {dx:.1f})")
    check(abs(dy) < 0.01, f"South→North dy≈0 ({dy:.1f})")

    # turn_east: south(0) → east(270) = shortest is -90° → -630px
    dx, dy = driver4._compute_cardinal_direction("turn_east")
    check(abs(dx - (-630.0)) < 1.0, f"South→East dx≈-630 ({dx:.1f})")

    # turn_west: south(0) → west(90) = +90° → +630px
    dx, dy = driver4._compute_cardinal_direction("turn_west")
    check(abs(dx - 630.0) < 1.0, f"South→West dx≈630 ({dx:.1f})")

    # Already facing south → turn_south is dead zone
    dx, dy = driver4._compute_cardinal_direction("turn_south")
    check(abs(dx) < 1.0, f"South→South dx≈0 ({dx:.1f})")

    # Pitch: turn_up from pitch=0 → pitch=-45 = -315px
    dx, dy = driver4._compute_cardinal_direction("turn_up")
    check(abs(dy - (-315.0)) < 1.0, f"Ahead→Up dy≈-315 ({dy:.1f})")

    # Pitch: turn_down from pitch=0 → pitch=45 = +315px
    dx, dy = driver4._compute_cardinal_direction("turn_down")
    check(abs(dy - 315.0) < 1.0, f"Ahead→Down dy≈315 ({dy:.1f})")

    # Total action count: 28 + 6 = 34 static (+1 dynamic look_at_target from section 11)
    # Remove dynamic entry before counting
    had_lat = "look_at_target" in MC_ACTION_MAP
    if had_lat:
        del MC_ACTION_MAP["look_at_target"]
    total = len(MC_ACTION_MAP)
    check(total == 34, f"Total static actions = 34 ({total})")
    if had_lat:
        MC_ACTION_MAP["look_at_target"] = {"motor": MotorType.LOOK, "direction": (0, 0)}

    # Cardinal actions in SUPPORTED_ACTIONS
    for d in ["turn_north", "turn_south", "turn_east", "turn_west", "turn_up", "turn_down"]:
        check(d in driver.SUPPORTED_ACTIONS, f"{d} in SUPPORTED_ACTIONS")

    # Execute cardinal turn
    result = driver4.act(Action(action_type="turn_north"))
    check(result.success, "turn_north action succeeded")

    # Dead zone: already facing south → turn_south returns success with low heat
    result_dz = driver4.act(Action(action_type="turn_south"))
    check(result_dz.success, "turn_south dead zone returns success")
    check("Already facing" in result_dz.outcome, f"Dead zone outcome: {result_dz.outcome}")

    print(f"\n{'='*40}")
    if errors == 0:
        print("ALL TESTS PASSED")
    else:
        print(f"{errors} TESTS FAILED")
    sys.exit(errors)
