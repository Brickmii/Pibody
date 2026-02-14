"""
PBAI Thermal Manifold - Universal Environment Core

This core isolates PBAI from the environments it interacts with.
Environments connect through drivers and ports.

Architecture:
    PBAI <---> Environment Core <---> Driver <---> Port <---> External Environment

The core handles:
- Driver loading and management
- Port communication protocols
- Perception/Action translation
- Environment state normalization
"""

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Tuple
from time import time
from enum import Enum
import importlib
import os

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# PORT PROTOCOL
# Ports define how data flows between drivers and external environments
# ═══════════════════════════════════════════════════════════════════════════════

class PortState(Enum):
    """Port connection states."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass
class PortMessage:
    """
    Standardized message format for port communication.
    All environment data flows through this structure.
    """
    msg_type: str                    # "perception", "action", "event", "error"
    payload: Dict[str, Any]          # The actual data
    timestamp: float = field(default_factory=time)
    source: str = ""                 # Origin identifier
    sequence: int = 0                # Message sequence number
    
    def to_dict(self) -> dict:
        return {
            "msg_type": self.msg_type,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "source": self.source,
            "sequence": self.sequence
        }


class Port(ABC):
    """
    Abstract port interface.
    Ports handle the actual communication with external environments.
    """
    
    def __init__(self, port_id: str, config: Dict[str, Any] = None):
        self.port_id = port_id
        self.config = config or {}
        self.state = PortState.DISCONNECTED
        self.sequence = 0
        self.last_message: Optional[PortMessage] = None
    
    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to external environment."""
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """Close connection to external environment."""
        pass
    
    @abstractmethod
    def send(self, message: PortMessage) -> bool:
        """Send message to external environment."""
        pass
    
    @abstractmethod
    def receive(self, timeout: float = 1.0) -> Optional[PortMessage]:
        """Receive message from external environment."""
        pass
    
    def is_connected(self) -> bool:
        return self.state == PortState.CONNECTED
    
    def next_sequence(self) -> int:
        self.sequence += 1
        return self.sequence


# ═══════════════════════════════════════════════════════════════════════════════
# DRIVER PROTOCOL
# Drivers translate between PBAI's normalized format and environment specifics
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Perception:
    """
    Normalized perception from any environment.
    This is what PBAI sees, regardless of source.
    """
    entities: List[str] = field(default_factory=list)      # Things that exist
    locations: List[str] = field(default_factory=list)     # Places that exist
    properties: Dict[str, Any] = field(default_factory=dict)  # State properties
    events: List[str] = field(default_factory=list)        # Recent happenings
    heat_value: float = 0.0                                 # Heat from perception (discovery, novelty)
    timestamp: float = field(default_factory=time)
    source_driver: str = ""
    raw: Any = None                                         # Original data if needed
    
    def to_dict(self) -> dict:
        return {
            "entities": self.entities,
            "locations": self.locations,
            "properties": self.properties,
            "events": self.events,
            "heat_value": self.heat_value,
            "timestamp": self.timestamp,
            "source_driver": self.source_driver
        }


@dataclass 
class Action:
    """
    Normalized action for any environment.
    This is what PBAI does, regardless of target.
    """
    action_type: str                                        # "move", "interact", "observe", etc.
    target: Optional[str] = None                            # What to act on
    parameters: Dict[str, Any] = field(default_factory=dict)  # Action-specific params
    
    def to_dict(self) -> dict:
        return {
            "action_type": self.action_type,
            "target": self.target,
            "parameters": self.parameters
        }


@dataclass
class ActionResult:
    """
    Normalized result of an action.
    
    HEAT ECONOMY:
        The driver determines heat_value based on task-specific semantics:
        - Blackjack win: high heat
        - Gym positive reward: proportional heat
        - Failed action: zero heat
        
        EnvironmentCore uses this to reward/punish psychology.
    """
    success: bool
    outcome: str                                            # Description of what happened
    heat_value: float = 0.0                                 # Heat returned by this outcome
    changes: Dict[str, Any] = field(default_factory=dict)   # State changes
    timestamp: float = field(default_factory=time)
    
    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "outcome": self.outcome,
            "heat_value": self.heat_value,
            "changes": self.changes,
            "timestamp": self.timestamp
        }


@dataclass
class ActionSuggestion:
    """An action with optional duration override from Introspector."""
    action: str
    duration: Optional[float] = None  # None = use default
    score: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# MOTION BUS — Shared verb activation vector
# ═══════════════════════════════════════════════════════════════════════════════

class MotionBus:
    """Shared verb activation vector — 20 base motions as cognitive bus.

    Multiple sources write activations [0,1]; decide() reads them as
    action weight boosts.  Activations decay each tick.

    Sources:
        Chat text  ────→  activate_from_text()   (strength 0.7)
        Vision ctx ────→  activate_from_perception()  (0.2–0.5)
        Introspector ──→  activate()              (0.4)

    Sink:
        decide() reads get_weight_boosts() → {action: multiplier}
        tick() decays activations (called after read)
    """

    DECAY_FACTOR = 0.85
    ACTIVATION_THRESHOLD = 0.1
    BOOST_SCALE = 3.0

    def __init__(self, manifold=None):
        from core.node_constants import ALL_BASE_MOTIONS, BASE_MOTION_PREFIX
        self._activations: Dict[str, float] = {
            f"{BASE_MOTION_PREFIX}{v}": 0.0 for v in ALL_BASE_MOTIONS
        }
        self._manifold = manifold
        self._verbs = list(ALL_BASE_MOTIONS)  # flat list of verb stems
        self._prefix = BASE_MOTION_PREFIX

    # ───────────────────────── write interface ─────────────────────────

    def activate(self, concept: str, strength: float, source: str = "unknown") -> None:
        """Max-merge activation (strongest signal wins, no runaway stacking).

        Also injects heat into the manifold bm_* node so Conscience
        can learn verb co-activation patterns.
        """
        if concept not in self._activations:
            return
        strength = max(0.0, min(1.0, strength))
        self._activations[concept] = max(self._activations[concept], strength)

        # Heat injection: bm_* node accumulates heat → manifold learns
        if self._manifold and strength > 0:
            node = self._manifold.get_node_by_concept(concept)
            if node:
                from core.node_constants import K
                node.add_heat(strength * K * 0.1)

        logger.debug(f"MotionBus activate: {concept}={strength:.2f} (src={source})")

    # Words to exclude from noun extraction
    STOP_WORDS = frozenset({
        'a', 'an', 'the', 'to', 'from', 'in', 'on', 'at', 'by', 'for',
        'with', 'of', 'and', 'or', 'but', 'is', 'are', 'was', 'go',
        'some', 'please', 'now', 'then', 'here', 'there', 'it', 'its',
        'i', 'me', 'my', 'up', 'down', 'left', 'right', 'around', 'out',
        'over', 'can', 'you', 'your', 'this', 'that', 'not', 'all', 'more',
    })

    def activate_from_text(self, text: str, base_strength: float = 0.7) -> List[str]:
        """Word-boundary match verbs in text → activate.

        Returns list of activated verb concepts (e.g. ['bm_explore', 'bm_find']).
        Also extracts noun hints for driver targeting.
        """
        activated = []
        text_lower = text.lower()
        for verb in self._verbs:
            if re.search(r'\b' + re.escape(verb) + r'\b', text_lower):
                concept = f"{self._prefix}{verb}"
                self.activate(concept, base_strength, "chat")
                activated.append(concept)
        # Extract nouns
        self._extract_nouns(text_lower)
        return activated

    def _extract_nouns(self, text: str):
        """Extract remaining words as noun/target hints after verb extraction."""
        words = re.findall(r'[a-z_]+', text)
        verb_set = set(self._verbs)
        self._target_hints = [
            w for w in words
            if w not in self.STOP_WORDS and w not in verb_set and len(w) >= 3
        ]

    @property
    def target_hints(self) -> List[str]:
        """Return extracted noun hints from last chat text."""
        return getattr(self, '_target_hints', [])

    def activate_from_perception(self, props: Dict[str, Any]) -> None:
        """Vision/state context → verb activation.

        Reads standard perception properties and activates appropriate verbs.
        """
        # bm_see always active when perceiving (we ARE seeing)
        self.activate(f"{self._prefix}see", 0.2, "perception")

        # Hostile entities nearby → bm_find, bm_quickly
        hostile_count = props.get("hostile_count", 0)
        if hostile_count and hostile_count > 0:
            self.activate(f"{self._prefix}find", 0.4, "perception")
            self.activate(f"{self._prefix}quickly", 0.3, "perception")

        # Has vision target → bm_identify
        if props.get("has_target"):
            self.activate(f"{self._prefix}identify", 0.35, "perception")

        # Drowning → bm_quickly
        if props.get("is_drowning"):
            self.activate(f"{self._prefix}quickly", 0.5, "perception")

        # High novelty (heat_value > K) → bm_discover, bm_explore
        heat_value = props.get("heat_value", 0)
        if heat_value and heat_value > 1.5:
            self.activate(f"{self._prefix}discover", 0.3, "perception")
            self.activate(f"{self._prefix}explore", 0.25, "perception")

        # Open terrain → bm_explore
        if props.get("open_terrain"):
            self.activate(f"{self._prefix}explore", 0.3, "perception")

        # Has resources nearby → bm_get, bm_take
        if props.get("has_resources"):
            self.activate(f"{self._prefix}get", 0.3, "perception")
            self.activate(f"{self._prefix}take", 0.25, "perception")

    # ───────────────────────── read interface ──────────────────────────

    def get_weight_boosts(self) -> Dict[str, float]:
        """Convert activations to {action: multiplier} via BASE_MOTION_ACTION_MAP.

        boost = 1.0 + activation * BOOST_SCALE
        Multiple verbs mapping to same action: take max boost.
        """
        from core.introspector import Introspector
        action_map = Introspector.BASE_MOTION_ACTION_MAP

        boosts: Dict[str, float] = {}
        for concept, activation in self._activations.items():
            if activation < self.ACTIVATION_THRESHOLD:
                continue
            mapped_actions = action_map.get(concept, [])
            boost = 1.0 + activation * self.BOOST_SCALE
            for action in mapped_actions:
                boosts[action] = max(boosts.get(action, 1.0), boost)

        return boosts

    def tick(self) -> None:
        """Decay all activations. Zero out anything below threshold."""
        for concept in self._activations:
            self._activations[concept] *= self.DECAY_FACTOR
            if self._activations[concept] < self.ACTIVATION_THRESHOLD:
                self._activations[concept] = 0.0

    def get_active(self) -> Dict[str, float]:
        """Return {concept: activation} for active verbs only."""
        return {c: a for c, a in self._activations.items()
                if a >= self.ACTIVATION_THRESHOLD}


class Driver(ABC):
    """
    Abstract driver interface.
    Drivers translate between PBAI and specific environments.
    
    DRIVER NODE INTEGRATION:
        Each driver can optionally create a DriverNode for learning.
        The DriverNode stores:
        - Learned state patterns (what the driver has seen)
        - Motor patterns (what actions work in what states)
        - Plans (sequences of actions for goals)
        
        Data persists to drivers/{DRIVER_ID}/ folder.
    
    HEAT ECONOMY:
        Drivers are the metabolic membrane - they translate task-specific
        outcomes into universal heat currency.
        
        Each driver defines:
        - HEAT_SCALE: Base multiplier for heat (task difficulty/richness)
        - Heat values in ActionResult based on outcome semantics
        - Heat values in Perception for discovery/novelty
    """
    
    # Driver metadata - override in subclasses
    DRIVER_ID: str = "base"
    DRIVER_NAME: str = "Base Driver"
    DRIVER_VERSION: str = "1.0.0"
    SUPPORTED_ACTIONS: List[str] = []
    
    # Heat economy - override in subclasses
    HEAT_SCALE: float = 1.0  # Base heat multiplier for this environment
    
    def __init__(self, port: Port, config: Dict[str, Any] = None, manifold=None):
        self.port = port
        self.config = config or {}
        self.active = False
        self.manifold = manifold
        self.born = False
        
        # DriverNode integration (PBAI's learned knowledge about this driver)
        self.driver_node = None
        
        # Birth
        self._birth()
    
    def _birth(self):
        """Birth this driver - create DriverNode if manifold provided."""
        if self.born:
            logger.warning(f"Driver {self.DRIVER_ID} already born, skipping")
            return
        
        # Create DriverNode for learning (if manifold provided)
        if self.manifold:
            from core.driver_node import DriverNode
            self.driver_node = DriverNode(self.DRIVER_ID, self.manifold)
            logger.info(f"Driver {self.DRIVER_ID} connected to DriverNode")
        
        self.born = True
        logger.debug(f"Driver {self.DRIVER_ID} born")
    
    def connect_manifold(self, manifold) -> None:
        """Connect manifold after initialization (creates DriverNode)."""
        if self.manifold is not None:
            logger.warning(f"Driver {self.DRIVER_ID} already has manifold")
            return
        
        self.manifold = manifold
        from core.driver_node import DriverNode
        self.driver_node = DriverNode(self.DRIVER_ID, self.manifold)
        logger.info(f"Driver {self.DRIVER_ID} connected to DriverNode")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TRANSLATION METHODS - Perception ↔ SensorReport, Action ↔ MotorAction
    # ═══════════════════════════════════════════════════════════════════════════
    
    def perception_to_sensor(self, perception: 'Perception') -> 'SensorReport':
        """Translate Perception to SensorReport for DriverNode."""
        from core.driver_node import SensorReport
        
        # Build objects list from entities
        objects = [{"type": e, "source": "entity"} for e in perception.entities]
        
        # Add locations as objects
        for loc in perception.locations:
            objects.append({"type": loc, "source": "location"})
        
        # Build description from events and properties
        desc_parts = []
        if perception.events:
            desc_parts.extend(perception.events)
        for key, val in perception.properties.items():
            desc_parts.append(f"{key}: {val}")
        description = "; ".join(desc_parts) if desc_parts else "observing environment"
        
        return SensorReport(
            timestamp=perception.timestamp,
            sensor_type="vision",
            description=description,
            objects=objects,
            status=perception.properties
        )
    
    # Keep old name for backwards compatibility
    def perception_to_vision(self, perception: 'Perception') -> 'SensorReport':
        """Alias for perception_to_sensor (backwards compatibility)."""
        return self.perception_to_sensor(perception)
    
    def motor_to_action(self, motor: 'MotorAction') -> 'Action':
        """Translate MotorAction to Action for environment."""
        from core.driver_node import MotorType
        
        # Map motor types to action types
        type_map = {
            MotorType.KEY_PRESS: "interact",
            MotorType.KEY_HOLD: "interact",
            MotorType.MOUSE_CLICK: "interact",
            MotorType.MOUSE_MOVE: "observe",
            MotorType.WAIT: "wait",
        }
        
        action_type = type_map.get(motor.motor_type, "interact")
        
        return Action(
            action_type=action_type,
            target=motor.target,
            parameters={
                "motor_type": motor.motor_type.value,
                "key": motor.key,
                "duration": motor.duration,
                "original_motor": motor.to_dict()
            }
        )
    
    def action_to_motor(self, action: 'Action') -> 'MotorAction':
        """Translate Action to MotorAction for DriverNode."""
        from core.driver_node import MotorAction, MotorType
        
        # Map action types to motor types
        if action.action_type in ["move", "interact"]:
            motor_type = MotorType.KEY_PRESS
        elif action.action_type == "observe":
            motor_type = MotorType.MOUSE_MOVE
        elif action.action_type == "wait":
            motor_type = MotorType.WAIT
        else:
            motor_type = MotorType.KEY_PRESS
        
        return MotorAction(
            motor_type=motor_type,
            key=action.parameters.get("key", ""),
            target=action.target,
            duration=action.parameters.get("duration", 0.0),
            description=f"{action.action_type} {action.target or ''}"
        )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # DRIVER NODE INTEGRATION HELPERS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def feed_perception(self, perception: 'Perception') -> None:
        """Feed perception to DriverNode for learning."""
        if self.driver_node:
            vision = self.perception_to_vision(perception)
            self.driver_node.see(vision)
    
    def feed_result(self, result: 'ActionResult', action: 'Action') -> None:
        """Feed action result to DriverNode for learning."""
        # Learning from results - track in action history
        # Future: update Order success rates based on outcomes
        pass
    
    def get_motor_suggestion(self, goal: str = None) -> Optional['Action']:
        """Get action suggestion from DriverNode based on learned patterns."""
        # Future: use collapse/correlate to find best action
        return None
    
    def save_learning(self) -> None:
        """Save DriverNode state (learned patterns, plans)."""
        if self.driver_node:
            self.driver_node.save()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ABSTRACT METHODS
    # ═══════════════════════════════════════════════════════════════════════════
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the driver and connect port."""
        pass
    
    @abstractmethod
    def shutdown(self) -> bool:
        """Shutdown driver and disconnect port."""
        pass
    
    @abstractmethod
    def perceive(self) -> Perception:
        """
        Get current perception from environment.
        Translates environment-specific data to normalized Perception.
        
        Should set perception.heat_value for discovery/novelty heat.
        
        NOTE: Call self.feed_perception(perception) to feed to DriverNode.
        """
        pass
    
    @abstractmethod
    def act(self, action: Action) -> ActionResult:
        """
        Execute action in environment.
        Translates normalized Action to environment-specific commands.
        
        MUST set result.heat_value based on outcome:
        - Positive outcomes: heat > 0 (reward)
        - Neutral outcomes: heat = 0
        - This replaces the fixed reward system
        
        NOTE: Call self.feed_result(result, action) to feed to DriverNode.
        """
        pass
    
    def scale_heat(self, base_heat: float) -> float:
        """Scale heat by driver's heat scale."""
        return base_heat * self.HEAT_SCALE
    
    def supports_action(self, action_type: str) -> bool:
        """Check if driver supports an action type."""
        return action_type in self.SUPPORTED_ACTIONS

    def get_domain_context(self, perception: 'Perception') -> Dict[str, Any]:
        """Return domain-scoped context for introspection.

        Override in subclasses to provide domain-specific data.
        Default: empty context (introspector gets nothing).
        """
        return {}
    
    def get_info(self) -> dict:
        """Get driver information."""
        return {
            "id": self.DRIVER_ID,
            "name": self.DRIVER_NAME,
            "version": self.DRIVER_VERSION,
            "supported_actions": self.SUPPORTED_ACTIONS,
            "heat_scale": self.HEAT_SCALE,
            "active": self.active,
            "port_state": self.port.state.value if self.port else "no_port",
            "has_driver_node": self.driver_node is not None,
            "manifold_connected": self.manifold is not None
        }


# ═══════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT CORE
# The central manager that PBAI interacts with
# ═══════════════════════════════════════════════════════════════════════════════

class EnvironmentCore:
    """
    Universal Environment Core.
    
    This is the CONNECTION POINT between the Manifold and external environments.
    This is the ENTRY point into the PBAI system.
    
    PBAI (Manifold) <---> EnvironmentCore <---> Driver <---> External Environment
    
    The core handles:
    - Driver management
    - Automatic perception → manifold integration
    - Automatic action result → psychology updates
    - Single source of truth for environment interaction
    """
    
    def __init__(self, manifold=None):
        """
        Initialize EnvironmentCore.
        
        Args:
            manifold: The PBAI manifold to integrate with (optional, can set later)
        """
        self.manifold = manifold
        self.born = False  # Birth tracking
        self.drivers: Dict[str, Driver] = {}
        self.active_driver: Optional[str] = None
        self.perception_history: List[Perception] = []
        self.action_history: List[tuple] = []  # (Action, ActionResult)
        self.max_history = 100
        self.loop_count = 0

        # Action queue (Introspector plans)
        self._action_queue: List[str] = []
        # Weight boosts from Introspector suggestions
        self._weight_boosts: Dict[str, float] = {}

        # Motion bus (shared verb activation vector)
        self._motion_bus: Optional[MotionBus] = None
        if self.manifold:
            self._motion_bus = MotionBus(manifold=self.manifold)

        # Birth
        self._birth()
    
    def _birth(self):
        """Birth this environment core - ENTRY point exists."""
        if self.born:
            logger.warning("EnvironmentCore already born, skipping")
            return
        
        self.born = True
        logger.info("EnvironmentCore born (ENTRY point ready)")
    
    def set_manifold(self, manifold) -> None:
        """Connect a manifold to this environment core."""
        self.manifold = manifold
        if manifold and self._motion_bus is None:
            self._motion_bus = MotionBus(manifold=manifold)
        logger.info("Manifold connected to EnvironmentCore")

    def activate_verbs(self, text: str) -> List[str]:
        """Activate motion verbs from chat text. Returns list of activated concepts."""
        if self._motion_bus:
            return self._motion_bus.activate_from_text(text)
        return []

    def get_motion_bus_state(self) -> Dict[str, Any]:
        """Return current motion bus state for API/debug."""
        if not self._motion_bus:
            return {"active": False, "verbs": {}}
        return {
            "active": True,
            "verbs": self._motion_bus.get_active(),
            "boosts": self._motion_bus.get_weight_boosts(),
        }

    def connect_visual_cortex(self, visual_cortex) -> None:
        """Connect a VisualCortex for vision integration."""
        self.visual_cortex = visual_cortex
        logger.info("Visual cortex connected to EnvironmentCore")

    def step_with_vision(self, image=None):
        """Process one vision frame through the visual cortex.

        Args:
            image: Optional image array (H, W, 3) RGB

        Returns:
            VisionStep result, or None if no visual cortex
        """
        vc = getattr(self, 'visual_cortex', None)
        if not vc:
            return None
        if image is not None:
            return vc.process_image(image)
        return vc.process_screen()
    
    # ───────────────────────────────────────────────────────────────────────────
    # INTROSPECTION BRIDGE (Environment delegates to driver + introspector)
    # ───────────────────────────────────────────────────────────────────────────

    def introspect(self, introspector, perception: Perception) -> Optional[List[str]]:
        """Bridge introspection through the environment.

        Gathers domain context from driver, passes to introspector,
        handles targeting and plan enqueuing. The daemon calls this
        instead of talking to introspector/driver directly.
        """
        if not introspector:
            return None

        driver = self.get_active_driver()
        if not driver:
            return None

        # 1. TARGETING — independent of introspector, runs every cycle
        target_action = None
        if hasattr(driver, '_get_target_action'):
            target_action = driver._get_target_action()
            if target_action and not self.has_plan():
                # Inject target into weight boosts for next decide()
                self._weight_boosts["look_at_target"] = 6.0

        # 1b. NOUN HINTS — thread from motion bus to driver, then clear
        if self._motion_bus and self._motion_bus.target_hints:
            if hasattr(driver, '_target_hints'):
                driver._target_hints = self._motion_bus.target_hints
                self._motion_bus._target_hints = []  # Consume: don't re-send stale hints

        # 2. DOMAIN CONTEXT — driver provides scoped data
        domain_ctx = {}
        if hasattr(driver, 'get_domain_context'):
            domain_ctx = driver.get_domain_context(perception)

        # 3. REACTIVE PLANS — driver state triggers plans directly
        reactive_plans = domain_ctx.get("reactive_plans", [])
        if reactive_plans and not self.has_plan():
            # First reactive plan wins (flee > eat > etc.)
            self.enqueue_plan(reactive_plans[0])
            logger.info(f"Reactive plan: {reactive_plans[0]}")
            return None  # Reactive plan overrides introspection

        # 4. INTROSPECTOR SANDBOX — scoped by domain context
        if self.has_plan():
            return None  # Already have a plan, skip introspection

        if not introspector.should_think():
            return None

        # Inject active verbs into domain context for query builder
        if self._motion_bus:
            domain_ctx["active_verbs"] = self._motion_bus.get_active()

        # Pass domain context to introspector (not raw perception)
        suggestions = introspector.suggest(domain_ctx)

        if not suggestions:
            return None

        # Feed bm_* suggestions back onto the bus
        if self._motion_bus and suggestions:
            for s in suggestions:
                if s.startswith("bm_"):
                    self._motion_bus.activate(s, 0.4, "introspector")

        # 5. WEIGHT BOOSTS from suggestions
        available = self.get_supported_actions()
        boosts = introspector.get_weight_boosts(suggestions, available)
        self._weight_boosts.update(boosts)

        # 6. DELIBERATIVE PLAN — if targeting + suggestions, interleave
        if target_action and len(suggestions) >= 1:
            plan = [target_action, suggestions[0]]
            if len(suggestions) > 1:
                plan.extend([target_action, suggestions[1]])
            self.enqueue_plan(plan)
            logger.info(f"Introspector plan ({len(plan)} steps): {plan}")

        logger.info(f"Introspector suggests: {suggestions[:3]} (boosted {len(boosts)} actions)")
        return suggestions

    # ───────────────────────────────────────────────────────────────────────────
    # DRIVER MANAGEMENT
    # ───────────────────────────────────────────────────────────────────────────

    def register_driver(self, driver: Driver) -> bool:
        """Register a driver with the core."""
        driver_id = driver.DRIVER_ID
        
        if driver_id in self.drivers:
            logger.warning(f"Driver '{driver_id}' already registered, replacing")
        
        self.drivers[driver_id] = driver
        logger.info(f"Registered driver: {driver.DRIVER_NAME} ({driver_id})")
        return True
    
    def unregister_driver(self, driver_id: str) -> bool:
        """Unregister a driver from the core."""
        if driver_id not in self.drivers:
            logger.warning(f"Driver '{driver_id}' not found")
            return False
        
        driver = self.drivers[driver_id]
        if driver.active:
            driver.shutdown()
        
        if self.active_driver == driver_id:
            self.active_driver = None
        
        del self.drivers[driver_id]
        logger.info(f"Unregistered driver: {driver_id}")
        return True
    
    def activate_driver(self, driver_id: str) -> bool:
        """Activate a driver for use."""
        if driver_id not in self.drivers:
            logger.error(f"Driver '{driver_id}' not found")
            return False
        
        driver = self.drivers[driver_id]
        
        if not driver.initialize():
            logger.error(f"Failed to initialize driver '{driver_id}'")
            return False
        
        self.active_driver = driver_id
        
        # Set active task for heat isolation
        # Heat should only flow within this task's frame hierarchy
        clock = self._get_clock()
        if clock:
            # GymDriver has game_handler.task_node
            task_node = None
            if hasattr(driver, 'game_handler') and driver.game_handler:
                task_node = driver.game_handler.task_node
            elif hasattr(driver, 'task_node'):
                task_node = driver.task_node
            
            clock.set_active_task(task_node)
        
        logger.info(f"Activated driver: {driver_id}")
        return True
    
    def deactivate_driver(self) -> bool:
        """Deactivate the current driver."""
        if not self.active_driver:
            return True
        
        driver = self.drivers[self.active_driver]
        driver.shutdown()
        
        # Clear active task
        clock = self._get_clock()
        if clock:
            clock.set_active_task(None)
        
        logger.info(f"Deactivated driver: {self.active_driver}")
        self.active_driver = None
        return True
    
    def get_active_driver(self) -> Optional[Driver]:
        """Get the currently active driver."""
        if self.active_driver:
            return self.drivers.get(self.active_driver)
        return None
    
    def list_drivers(self) -> List[dict]:
        """List all registered drivers."""
        return [d.get_info() for d in self.drivers.values()]
    
    # ───────────────────────────────────────────────────────────────────────────
    # PBAI INTERFACE (What PBAI/Manifold calls)
    # ───────────────────────────────────────────────────────────────────────────
    
    def perceive(self) -> Perception:
        """
        Get current perception from the active environment.
        Automatically integrates into manifold if connected.
        """
        driver = self.get_active_driver()
        
        if not driver:
            logger.warning("No active driver, returning empty perception")
            return Perception()
        
        perception = driver.perceive()
        perception.source_driver = driver.DRIVER_ID
        
        # Store in history
        self.perception_history.append(perception)
        if len(self.perception_history) > self.max_history:
            self.perception_history.pop(0)
        
        # AUTO-INTEGRATE: Update manifold with perception
        if self.manifold:
            self._integrate_perception(perception)

        # MOTION BUS: perception context → verb activations
        if self._motion_bus and perception.properties:
            self._motion_bus.activate_from_perception(perception.properties)

        logger.debug(f"Perception from {driver.DRIVER_ID}: {len(perception.entities)} entities")
        return perception
    
    def act(self, action: Action) -> ActionResult:
        """
        Execute an action in the active environment.
        Automatically updates psychology based on result.
        
        HEAT COST: Actions cost COST_ACTION from Ego (the doer).
        If Ego can't afford it, action fails.
        """
        from core.node_constants import COST_ACTION, PSYCHOLOGY_MIN_HEAT
        
        driver = self.get_active_driver()
        
        if not driver:
            logger.warning("No active driver, action failed")
            return ActionResult(
                success=False,
                outcome="No active environment driver"
            )
        
        if not driver.supports_action(action.action_type):
            logger.warning(f"Driver does not support action: {action.action_type}")
            return ActionResult(
                success=False,
                outcome=f"Action '{action.action_type}' not supported by {driver.DRIVER_ID}"
            )
        
        # HEAT ECONOMY: Ego must pay to act
        # For gym environments, use reduced cost (they step much faster than normal)
        is_gym = hasattr(driver, 'gym_env')
        action_cost = COST_ACTION * 0.01 if is_gym else COST_ACTION  # 1% cost for gym
        
        if self.manifold and self.manifold.ego_node:
            ego = self.manifold.ego_node
            if ego.heat <= PSYCHOLOGY_MIN_HEAT + action_cost:
                # Instead of failing, just log and continue (gym envs need to run)
                if is_gym:
                    logger.debug(f"Ego low heat ({ego.heat:.3f}), continuing anyway for gym")
                else:
                    logger.warning(f"Ego exhausted - cannot afford action (heat={ego.heat:.3f})")
                    return ActionResult(
                        success=False,
                        outcome="Insufficient heat for action (Ego exhausted)"
                    )
            else:
                # Spend heat to output to environment
                spent = ego.spend_heat(action_cost, minimum=PSYCHOLOGY_MIN_HEAT)
                logger.debug(f"Ego spent {spent:.3f} heat on action")
        
        result = driver.act(action)
        
        # Store in history
        self.action_history.append((action, result))
        if len(self.action_history) > self.max_history:
            self.action_history.pop(0)
        
        # AUTO-INTEGRATE: Update psychology based on result
        if self.manifold:
            self._integrate_action_result(action, result)
        
        logger.debug(f"Action {action.action_type}: {result.outcome}")
        return result
    
    def get_supported_actions(self) -> List[str]:
        """Get list of actions supported by active driver."""
        driver = self.get_active_driver()
        if driver:
            return driver.SUPPORTED_ACTIONS
        return []

    def enqueue_plan(self, plan: List[str]):
        """Queue a sequence of actions (from Introspector)."""
        self._action_queue = list(plan)

    def clear_plan(self):
        """Clear queued action plan."""
        self._action_queue.clear()

    def has_plan(self) -> bool:
        """Check if there's a queued action plan."""
        return len(self._action_queue) > 0
    
    # ───────────────────────────────────────────────────────────────────────────
    # MANIFOLD INTEGRATION (Routes to proper nodes)
    # ───────────────────────────────────────────────────────────────────────────
    
    def _integrate_perception(self, perception: Perception) -> None:
        """
        Route perception INPUT to Clock (Self).
        
        Environment.py is the membrane - it routes, doesn't process.
        Clock.receive() handles the actual integration on tick.
        
        DEFENSIVE: Validates state_key, never passes None.
        """
        if not self.manifold:
            return
        
        # Extract state and context for decision phase
        state_key = perception.properties.get("state_key")
        
        # DEFENSIVE: Validate state_key - never allow None or "None" patterns
        if state_key is None or state_key == "None" or "None" in str(state_key):
            state_key = f"state_{hash(str(perception.raw)) % 10000}" if perception.raw else "unknown_state"
            perception.properties["state_key"] = state_key
            logger.debug(f"Generated fallback state_key: {state_key}")
        
        # Build context from active features
        context = {}
        feature_threshold = 0.5
        skip_keys = {'state_key', 'step', 'episode', 'episode_reward', 'last_reward',
                     'row', 'col', 'row_norm', 'col_norm', 'dist_to_goal'}
        
        for key, value in perception.properties.items():
            if key in skip_keys:
                continue
            if isinstance(value, (int, float)) and value > feature_threshold:
                context[key] = True
        
        # Store for decision phase
        self._current_state_key = state_key
        self._current_context = context
        
        # DEFENSIVE: Filter entities - remove None values
        entities = [e for e in perception.entities if e and e != "None" and "None" not in str(e)]
        if not entities:
            entities = [state_key]
        
        # Route to Clock (INPUT)
        clock = self._get_clock()
        if clock:
            clock.receive({
                "state_key": state_key,
                "context": context,
                "heat_value": perception.heat_value,
                "entities": entities,
                "locations": perception.locations,
                "events": perception.events,
                "properties": perception.properties
            })
        
        # Ego sustain: perceiving the world maintains minimal ability to act
        if self.manifold.ego_node:
            from core.node_constants import K as _K, PSYCHOLOGY_MIN_HEAT
            # Small sustain: enough to rebuild capacity between actions, not enough to sustain indefinitely
            ego_sustain = PSYCHOLOGY_MIN_HEAT if perception.heat_value > 0 else PSYCHOLOGY_MIN_HEAT * 0.5
            self.manifold.ego_node.add_heat(ego_sustain)
            logger.debug(f"Ego perception sustain: +{ego_sustain:.3f}")

        logger.debug(f"Routed perception to Clock: {state_key}")
    
    def _get_clock(self):
        """Get or create the Clock."""
        from core.clock_node import Clock
        
        if not hasattr(self, '_clock') or self._clock is None:
            self._clock = Clock(self.manifold)
            # Don't auto-start - environment controls timing
        return self._clock
    
    def _integrate_action_result(self, action: Action, result: ActionResult) -> None:
        """
        Route action result OUTPUT to DecisionNode.
        
        Environment.py is the membrane - it routes, doesn't process.
        DecisionNode.complete_decision() handles recording with context.
        """
        if not self.manifold:
            return
        
        action_name = action.target or action.action_type
        success = result.success
        heat_value = result.heat_value
        
        # For gym environments, reward Ego with heat on positive outcomes
        if heat_value > 0 and self.manifold.ego_node:
            # Give Ego some heat back (capped)
            from core.node_constants import K
            heat_gain = min(heat_value * 0.1, K * 0.1)  # Max 10% of K
            self.manifold.ego_node.add_heat(heat_gain)
            logger.debug(f"Ego gained {heat_gain:.3f} heat from positive outcome")
        
        # Route to DecisionNode (OUTPUT)
        decision_node = self._get_decision_node()
        if decision_node and decision_node.pending_choice:
            outcome = f"{action_name}_{'success' if success else 'failure'}"
            decision_node.complete_decision(outcome, success, heat_value)
        
        logger.debug(f"Routed result to DecisionNode: {action_name} success={success}")
    
    def _get_decision_node(self):
        """Get or create the DecisionNode."""
        from core.decision_node import DecisionNode
        
        if not hasattr(self, '_decision_node') or self._decision_node is None:
            self._decision_node = DecisionNode(self.manifold)
        return self._decision_node
    
    # ───────────────────────────────────────────────────────────────────────────
    # DECISION CYCLE (Routes to DecisionNode)
    # ───────────────────────────────────────────────────────────────────────────
    
    def decide(self, perception: Perception = None) -> Action:
        """
        Route decision OUTPUT through DecisionNode.

        Uses learned action scores from GameHandler when available.
        Drains action queue first if Introspector has queued a plan.

        Args:
            perception: Current perception (if None, calls perceive())

        Returns:
            Action selected based on learned outcomes
        """
        import random

        # Get perception if not provided
        if perception is None:
            perception = self.perceive()

        # Get available actions (state-aware if driver supports it)
        driver = self.get_active_driver()
        if driver and hasattr(driver, 'get_actions'):
            available_actions = driver.get_actions(getattr(driver, '_current_hand_state', None))
        else:
            available_actions = self.get_supported_actions()
        if not available_actions:
            logger.warning("No available actions - returning wait")
            return Action(action_type="wait")

        # Drain action queue first (Introspector plans)
        if self._action_queue:
            next_action = self._action_queue.pop(0)
            # Validate: action must still be available (or be a dynamic action like look_at_target)
            valid_set = set(available_actions)
            # Also accept dynamic targeting actions
            if driver and hasattr(driver, '_current_target') and driver._current_target:
                valid_set.add("look_at_target")
            if next_action in valid_set:
                logger.info(f"Plan queue: {next_action} ({len(self._action_queue)} remaining)")
                return Action(action_type=next_action, target=next_action)
            else:
                logger.warning(f"Queued action {next_action} no longer valid — clearing plan")
                self._action_queue.clear()

        # Get state and context
        state_key = getattr(self, '_current_state_key', perception.properties.get("state_key", "unknown"))
        context = getattr(self, '_current_context', {})

        # Inject driver action weights into context for DecisionNode fallback
        driver = self.get_active_driver()
        if hasattr(driver, 'get_action_weights'):
            context["action_weights"] = driver.get_action_weights(available_actions)

        # Route to DecisionNode (OUTPUT)
        decision_node = self._get_decision_node()

        # MOTION BUS: merge verb boosts into weight boosts, then decay
        if self._motion_bus:
            bus_boosts = self._motion_bus.get_weight_boosts()
            for action, boost in bus_boosts.items():
                self._weight_boosts[action] = max(
                    self._weight_boosts.get(action, 1.0), boost
                )
            self._motion_bus.tick()  # Decay after read

        # Get exploration rate (decays as we learn more)
        exploration_rate = self.manifold.get_exploration_rate() if self.manifold else 0.3

        if random.random() < exploration_rate:
            # Explore: weighted random if driver provides weights, else uniform
            driver = self.get_active_driver()
            if hasattr(driver, 'get_action_weights'):
                weights_map = driver.get_action_weights(available_actions)
                # Apply Introspector weight boosts if available
                if self._weight_boosts:
                    for action in available_actions:
                        if action in self._weight_boosts:
                            weights_map[action] = weights_map.get(action, 1.0) * self._weight_boosts[action]
                    self._weight_boosts = {}  # Clear after use
                weights = [weights_map.get(a, 1.0) for a in available_actions]
                chosen = random.choices(available_actions, weights=weights, k=1)[0]
            else:
                chosen = random.choice(available_actions)
            logger.info(f"Explore: {chosen} (rate={exploration_rate:.2f})")
        else:
            # Exploit: Use learned action scores from driver
            driver = self.get_active_driver()
            
            # Check if driver has learned action scores (GymDriver with GameHandler)
            if hasattr(driver, 'get_action_scores'):
                scores = driver.get_action_scores()
                
                if scores and any(s != 0.5 for s in scores.values()):
                    # We have learned preferences - pick best action
                    # Add small random tiebreaker for equal scores
                    best_action = max(scores.keys(), key=lambda a: scores[a] + random.random() * 0.01)
                    chosen = best_action
                    logger.info(f"Exploit: {chosen} (score={scores[chosen]:.2f}, scores={{{', '.join(f'{k}:{v:.2f}' for k,v in scores.items())}}})")
                else:
                    # No learned data yet - use DecisionNode
                    chosen = decision_node.decide(
                        state_key=state_key,
                        options=available_actions,
                        context=context
                    )
                    logger.info(f"Exploit (no data): {chosen}")
            else:
                # No GameHandler — weighted sampling with driver weights
                if hasattr(driver, 'get_action_weights'):
                    weights_map = driver.get_action_weights(available_actions)
                    weights = [weights_map.get(a, 1.0) for a in available_actions]
                    chosen = random.choices(available_actions, weights=weights, k=1)[0]
                    logger.debug(f"Decision: {chosen} (weighted)")
                else:
                    chosen = decision_node.decide(
                        state_key=state_key,
                        options=available_actions,
                        context=context
                    )
                logger.info(f"Exploit (weighted): {chosen}")
        
        # Record decision for learning
        decision_node.begin_decision(state_key, available_actions, 
                                     self.manifold.get_confidence() if self.manifold else 1.0, 
                                     context)
        decision_node.commit_decision(chosen)
        
        return Action(action_type=chosen, target=chosen)
    
    def _decide_via_psychology(self, perception: Perception) -> Action:
        """
        Fallback decision making through manifold psychology.
        
        Uses the 5/6 confidence threshold:
        - confidence > 5/6: EXPLOIT (use learned pattern)
        - confidence < 5/6: EXPLORE (try options)
        """
        from core.node_constants import K, CONFIDENCE_EXPLOIT_THRESHOLD
        
        # Get available actions from driver
        available_actions = self.get_supported_actions()
        if not available_actions:
            logger.warning("No available actions - returning wait")
            return Action(action_type="wait")
        
        # Without manifold, pick first action
        if not self.manifold:
            return Action(action_type=available_actions[0], target=available_actions[0])
        
        # Build state key from perception
        state_key = perception.properties.get("state_key", "unknown")
        if not state_key or state_key == "unknown":
            state_key = "_".join(perception.entities[:3]) if perception.entities else "empty"
        
        # Get confidence from manifold (Conscience mediation)
        confidence = self.manifold.get_confidence(state_key)
        should_exploit = confidence > CONFIDENCE_EXPLOIT_THRESHOLD
        
        # Update Identity's awareness
        heat = perception.heat_value if perception.heat_value > 0 else K
        self.manifold.update_identity(state_key, heat_delta=heat * 0.1)
        
        if should_exploit:
            # EXPLOIT: Use decision_node's history
            from core.decision_node import DecisionNode
            if not hasattr(self, '_decision_node') or self._decision_node is None:
                self._decision_node = DecisionNode(self.manifold)
            
            best = self._decision_node.get_best_choice(state_key, available_actions)
            if best:
                logger.info(f"EXPLOIT decision: {best} (confidence={confidence:.3f})")
                return Action(action_type=best, target=best)
        
        # EXPLORE: Random selection
        import random
        selected = random.choice(available_actions)
        logger.info(f"EXPLORE decision: {selected} (confidence={confidence:.3f})")
        return Action(action_type=selected, target=selected)
    
    def feedback(self, result: ActionResult, action: Action = None) -> Dict[str, float]:
        """
        Process feedback through manifold psychology.
        
        Updates:
        - Conscience validation (confirms/denies the action worked)
        - Heat distribution to Identity/Ego/Conscience
        
        Heat distribution depends on outcome type:
        - SUCCESS:  Ego 60%, Identity 30%, Conscience 10% (pattern worked!)
        - FAILURE:  Identity 60%, Ego 20%, Conscience 20% (needs to learn)
        - NEUTRAL:  Equal small amounts to all (just maintenance)
        
        Args:
            result: ActionResult from act()
            action: The action that was taken (optional, for logging)
            
        Returns:
            Heat changes: {"identity": delta, "ego": delta, "conscience": delta}
        """
        from core.node_constants import K
        
        if not self.manifold:
            return {"identity": 0, "ego": 0, "conscience": 0}
        
        # Build outcome concept
        action_name = action.action_type if action else "action"
        outcome_concept = f"{action_name}_result"
        
        # Check for success_type in changes (game-specific success detection)
        success_type = result.changes.get("success_type", "success" if result.success else "failure")
        
        # Distribute heat based on outcome
        changes = {"identity": 0.0, "ego": 0.0, "conscience": 0.0}
        
        if success_type == "neutral":
            # ═══════════════════════════════════════════════════════════════
            # NEUTRAL: Small maintenance heat, equally distributed
            # No learning signal - just keep the system alive
            # ═══════════════════════════════════════════════════════════════
            heat = K * 0.05  # Small maintenance amount
            
            if self.manifold.identity_node:
                self.manifold.identity_node.add_heat_unchecked(heat)
                changes["identity"] = heat
            if self.manifold.ego_node:
                self.manifold.ego_node.add_heat_unchecked(heat)
                changes["ego"] = heat
            if self.manifold.conscience_node:
                self.manifold.conscience_node.add_heat_unchecked(heat)
                changes["conscience"] = heat
            
            logger.debug(f"Feedback: {outcome_concept} [NEUTRAL] "
                        f"I={changes['identity']:+.2f} "
                        f"E={changes['ego']:+.2f} "
                        f"C={changes['conscience']:+.2f}")
            return changes
        
        # Calculate heat magnitude for success/failure
        heat = abs(result.heat_value) * K if result.heat_value != 0 else K * 0.1
        
        # Validate through Conscience (builds Ego's confidence)
        # Only validate on actual success/failure, not neutral
        self.manifold.validate_conscience(outcome_concept, confirmed=(success_type == "success"))

        # CONFIDENCE BOOST: Successful actions return 0.618 to Ego
        # Matches COST_ACTION — successful moves are cost-neutral
        if success_type == "success" and self.manifold.ego_node:
            from core.node_constants import COST_ACTION
            confidence_boost = COST_ACTION  # 0.618
            self.manifold.ego_node.add_heat_unchecked(confidence_boost)
            changes["ego"] += confidence_boost
            logger.debug(f"Ego confidence boost: +{confidence_boost:.3f}")

        if success_type == "success":
            # ═══════════════════════════════════════════════════════════════
            # SUCCESS: Ego gets more (pattern worked!)
            # Ego 60%, Identity 30%, Conscience 10%
            # ═══════════════════════════════════════════════════════════════
            if self.manifold.ego_node:
                self.manifold.ego_node.add_heat_unchecked(heat * 0.6)
                changes["ego"] = heat * 0.6
            if self.manifold.identity_node:
                self.manifold.identity_node.add_heat_unchecked(heat * 0.3)
                changes["identity"] = heat * 0.3
            if self.manifold.conscience_node:
                self.manifold.conscience_node.add_heat_unchecked(heat * 0.1)
                changes["conscience"] = heat * 0.1
        else:
            # ═══════════════════════════════════════════════════════════════
            # FAILURE: Identity gets more (needs to learn)
            # Identity 60%, Ego 20%, Conscience 20%
            # ═══════════════════════════════════════════════════════════════
            if self.manifold.identity_node:
                self.manifold.identity_node.add_heat_unchecked(heat * 0.6)
                changes["identity"] = heat * 0.6
            if self.manifold.ego_node:
                self.manifold.ego_node.add_heat_unchecked(heat * 0.2)
                changes["ego"] = heat * 0.2
            if self.manifold.conscience_node:
                self.manifold.conscience_node.add_heat_unchecked(heat * 0.2)
                changes["conscience"] = heat * 0.2
        
        logger.debug(f"Feedback: {outcome_concept} [{success_type.upper()}] "
                   f"I={changes['identity']:+.2f} "
                   f"E={changes['ego']:+.2f} "
                   f"C={changes['conscience']:+.2f}")
        
        return changes
    
    def step(self, perception: Perception = None) -> Tuple[Action, ActionResult, Dict[str, float]]:
        """
        Complete one step: perceive → tick → decide → act → feedback.
        
        Environment time syncs with Clock (Self) time.
        Each step = one tick of existence.
        
        Flow:
            1. perceive() → routes INPUT to Clock
            2. tick() → Clock processes perception
            3. decide() → routes OUTPUT from DecisionNode
            4. act() → executes action
            5. feedback() → routes result to DecisionNode
        
        Args:
            perception: Optional perception (if None, calls perceive())
            
        Returns:
            Tuple of (action_taken, result, heat_changes)
        """
        # 1. PERCEIVE - routes INPUT to Clock
        if perception is None:
            perception = self.perceive()
        
        # 2. TICK - Clock processes perception (sync environment time with system time)
        clock = self._get_clock()
        if clock:
            clock.tick()  # Process inputs, existence tax, redistribution
        
        # 3. DECIDE - routes OUTPUT from DecisionNode
        action = self.decide(perception)
        
        # 4. ACT - executes action
        result = self.act(action)
        
        # 5. FEEDBACK - routes result to DecisionNode
        heat_changes = self.feedback(result, action)
        
        self.loop_count += 1
        
        return action, result, heat_changes
    
    # ───────────────────────────────────────────────────────────────────────────
    # HISTORY ACCESS
    # ───────────────────────────────────────────────────────────────────────────
    
    def get_recent_perceptions(self, count: int = 10) -> List[Perception]:
        """Get recent perceptions."""
        return self.perception_history[-count:]
    
    def get_recent_actions(self, count: int = 10) -> List[tuple]:
        """Get recent action/result pairs."""
        return self.action_history[-count:]
    
    # ───────────────────────────────────────────────────────────────────────────
    # CONVENIENCE: Full experience loop
    # ───────────────────────────────────────────────────────────────────────────
    
    def experience_loop(self, decide_action: Callable[[Perception], Action]) -> ActionResult:
        """
        Run one full experience cycle:
        1. Perceive
        2. Decide (via callback)
        3. Act
        
        Args:
            decide_action: Function that takes Perception and returns Action
            
        Returns:
            Result of the action
        """
        self.loop_count += 1
        
        # Perceive (auto-integrates)
        perception = self.perceive()
        
        # Decide (caller's responsibility)
        action = decide_action(perception)
        
        # Act (auto-integrates)
        result = self.act(action)
        
        # Save manifold state
        if self.manifold:
            self.manifold.loop_number = self.loop_count
            self.manifold.save_growth_map()
        
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# DRIVER LOADER
# Dynamically loads drivers from files
# ═══════════════════════════════════════════════════════════════════════════════

class DriverLoader:
    """
    Loads driver modules from the drivers directory.
    """
    
    def __init__(self, drivers_path: str = None):
        # Default to the directory this file is in
        if drivers_path is None:
            self.drivers_path = os.path.dirname(os.path.abspath(__file__))
        else:
            self.drivers_path = drivers_path
    
    def discover_drivers(self) -> List[str]:
        """Find all driver files in the drivers directory."""
        if not os.path.exists(self.drivers_path):
            return []
        
        drivers = []
        for filename in os.listdir(self.drivers_path):
            if filename.endswith("_driver.py") and not filename.startswith("_"):
                driver_name = filename[:-10]  # Remove _driver.py
                drivers.append(driver_name)
        
        return drivers
    
    def load_driver(self, driver_name: str, config: Dict[str, Any] = None) -> Optional['Driver']:
        """
        Load and instantiate a driver from the drivers directory.
        
        Args:
            driver_name: Name of driver (e.g., "minecraft" loads minecraft_driver.py)
            config: Configuration dict to pass to the driver
            
        Returns:
            Instantiated Driver, or None if loading fails
        """
        driver_file = os.path.join(self.drivers_path, f"{driver_name}_driver.py")
        
        if not os.path.exists(driver_file):
            logger.error(f"Driver file not found: {driver_file}")
            return None
        
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(f"{driver_name}_driver", driver_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Look for create_driver function first
            if hasattr(module, 'create_driver'):
                return module.create_driver(config)
            
            # Otherwise find the Driver subclass
            for name in dir(module):
                obj = getattr(module, name)
                if (isinstance(obj, type) and 
                    issubclass(obj, Driver) and 
                    obj is not Driver and
                    not name.startswith('_')):
                    return obj(config)
            
            logger.warning(f"No Driver subclass found in {driver_name}_driver.py")
            return None
            
        except Exception as e:
            logger.error(f"Failed to load driver {driver_name}: {e}")
            return None


# ═══════════════════════════════════════════════════════════════════════════════
# NULL PORT & DRIVER (For testing without external environments)
# ═══════════════════════════════════════════════════════════════════════════════

class NullPort(Port):
    """A port that doesn't connect to anything - for testing."""
    
    def connect(self) -> bool:
        self.state = PortState.CONNECTED
        return True
    
    def disconnect(self) -> bool:
        self.state = PortState.DISCONNECTED
        return True
    
    def send(self, message: PortMessage) -> bool:
        self.last_message = message
        return True
    
    def receive(self, timeout: float = 1.0) -> Optional[PortMessage]:
        return None


class MockDriver(Driver):
    """
    Mock driver for testing PBAI without a real environment.
    Simulates a simple world with entities, locations, and basic interactions.
    
    HEAT ECONOMY:
        - Observe: small heat (0.5 * K) - passive
        - Move (success): medium heat (1.0 * K) - achieved goal
        - Move (fail): zero heat
        - Interact: medium heat (1.0 * K)
        - Explore: variable heat based on discovery
        - Wait: tiny heat (0.1 * K) - just existing
    
    DRIVER NODE INTEGRATION:
        When manifold is provided, creates DriverNode for learning.
        Learned patterns persist to drivers/mock/
    """
    
    DRIVER_ID = "mock"
    DRIVER_NAME = "Mock Environment Driver"
    DRIVER_VERSION = "2.0.0"
    SUPPORTED_ACTIONS = ["observe", "move", "interact", "wait", "explore"]
    HEAT_SCALE = 1.0  # Standard heat scale
    
    def __init__(self, port: Port = None, config: Dict[str, Any] = None, manifold=None):
        super().__init__(port or NullPort("null"), config, manifold=manifold)
        
        # Simulated world state
        self.world = {
            "entities": ["tree", "rock", "river", "bird", "flower"],
            "locations": ["forest", "mountain", "cave", "meadow", "lake"],
            "current_location": "forest",
            "time_of_day": "day",
            "weather": "clear",
            "inventory": []
        }
        self.events = []
        self._seen_events = set()  # Track novelty
    
    def initialize(self) -> bool:
        if self.port:
            self.port.connect()
        self.active = True
        logger.info(f"MockDriver initialized: {self.world['current_location']}")
        return True
    
    def shutdown(self) -> bool:
        if self.port:
            self.port.disconnect()
        self.active = False
        return True
    
    def perceive(self) -> Perception:
        """
        Get perception from mock environment.
        Also feeds to DriverNode for learning.
        """
        from core.node_constants import K
        
        # Simulate some environmental variation
        import random
        
        novelty_heat = 0.0
        
        # Occasionally add events
        if random.random() < 0.3:
            event = random.choice([
                "A bird sings in the distance",
                "Wind rustles the leaves",
                "A small animal scurries by",
                "Clouds drift overhead"
            ])
            self.events.append(event)
            if len(self.events) > 5:
                self.events.pop(0)
            
            # NOVELTY: New events give heat
            if event not in self._seen_events:
                self._seen_events.add(event)
                novelty_heat = self.scale_heat(K * 0.3)  # Discovery!
        
        perception = Perception(
            entities=self.world["entities"][:],
            locations=self.world["locations"][:],
            properties={
                "current_location": self.world["current_location"],
                "time_of_day": self.world["time_of_day"],
                "weather": self.world["weather"],
                "inventory": self.world["inventory"][:]
            },
            events=self.events[:],
            heat_value=novelty_heat
        )
        
        # Feed to DriverNode for learning
        self.feed_perception(perception)
        
        return perception
    
    def act(self, action: Action) -> ActionResult:
        """
        Execute action in mock environment.
        Also feeds results to DriverNode for learning.
        """
        from core.node_constants import K
        
        action_type = action.action_type
        target = action.target
        result = None
        
        if action_type == "observe":
            result = ActionResult(
                success=True,
                outcome=f"Observed the {self.world['current_location']}",
                heat_value=self.scale_heat(K * 0.5),  # Passive observation
                changes={}
            )
        
        elif action_type == "wait":
            result = ActionResult(
                success=True,
                outcome="Waited patiently",
                heat_value=self.scale_heat(K * 0.1),  # Minimal heat for just existing
                changes={}
            )
        
        elif action_type == "move":
            if target in self.world["locations"]:
                old_loc = self.world["current_location"]
                self.world["current_location"] = target
                result = ActionResult(
                    success=True,
                    outcome=f"Moved from {old_loc} to {target}",
                    heat_value=self.scale_heat(K * 1.0),  # Goal achieved
                    changes={"current_location": target}
                )
            else:
                result = ActionResult(
                    success=False,
                    outcome=f"Cannot move to unknown location: {target}",
                    heat_value=0.0,  # Failed - no heat
                    changes={}
                )
        
        elif action_type == "interact":
            if target in self.world["entities"]:
                result = ActionResult(
                    success=True,
                    outcome=f"Interacted with {target}",
                    heat_value=self.scale_heat(K * 1.0),  # Successful interaction
                    changes={"last_interaction": target}
                )
            else:
                result = ActionResult(
                    success=False,
                    outcome=f"Cannot interact with unknown entity: {target}",
                    heat_value=0.0,  # Failed
                    changes={}
                )
        
        elif action_type == "explore":
            import random
            discoveries = [
                ("nothing special", 0.2),
                ("an interesting rock", 0.5),
                ("a hidden path", 1.5),      # Rare = high heat
                ("animal tracks", 0.8)
            ]
            discovery, heat_mult = random.choice(discoveries)
            result = ActionResult(
                success=True,
                outcome=f"Explored and found: {discovery}",
                heat_value=self.scale_heat(K * heat_mult),  # Variable based on discovery
                changes={"discovery": discovery}
            )
        
        else:
            result = ActionResult(
                success=False,
                outcome=f"Unknown action: {action_type}",
                heat_value=0.0,
                changes={}
            )
        
        # Feed to DriverNode for learning
        self.feed_result(result, action)
        
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def create_environment_core(manifold=None, use_mock: bool = True) -> EnvironmentCore:
    """
    Create and configure an EnvironmentCore.
    
    Args:
        manifold: The PBAI manifold to connect (optional)
        use_mock: If True, registers and activates a MockDriver
    
    Returns:
        Configured EnvironmentCore connected to manifold
    
    Note:
        If manifold is provided, drivers will create DriverNodes for learning.
        Learned patterns persist to drivers/{driver_id}/
    """
    core = EnvironmentCore(manifold=manifold)
    
    if use_mock:
        # Pass manifold so MockDriver creates DriverNode for learning
        mock_driver = MockDriver(manifold=manifold)
        core.register_driver(mock_driver)
        core.activate_driver("mock")
    
    return core
