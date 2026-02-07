"""
PBAI Driver Components - Sensors, Motors, Plans

Three core dataclasses for driver interaction:

SENSOR REPORT
    Input to manifold from any sensor:
    - Vision (Sonnet describes screen)
    - Thermal (Pi temperature sensors)
    - Audio, touch, or any other sensor
    
MOTOR ACTION
    Individual potential action a driver can take:
    - Key press, mouse move, API call
    - Tagged with heat cost
    - Driver clusters these as options
    
ACTION PLAN
    Sequence of MotorActions toward a goal:
    - Directed by capabilities
    - Has prerequisites and expected outcomes
    - Learns success/failure rates

Hierarchy:
    Capability (clusters drivers, directs plans)
        └── Driver node (has motor actions as axes)
                └── MotorAction (individual choice)
        └── ActionPlan (sequence toward goal)
    
    SensorReport → Manifold (input from world)
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from time import time
from enum import Enum

from .node_constants import K

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# SENSOR REPORT - Input from any sensor
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SensorReport:
    """
    Report from any sensor to the manifold.
    
    Sensors include:
    - Vision (Sonnet/camera describing what it sees)
    - Thermal (Pi temperature sensors)
    - Audio (microphone input)
    - Touch (physical sensors)
    - API responses (external data)
    
    The manifold processes these to update its state.
    """
    timestamp: float = field(default_factory=time)
    sensor_type: str = "unknown"          # "vision", "thermal", "audio", etc.
    
    # Core data
    description: str = ""                  # Natural language description
    raw_data: Any = None                   # Raw sensor data (bytes, numbers, etc.)
    
    # Structured interpretation
    objects: List[Dict[str, Any]] = field(default_factory=list)   # Detected objects
    measurements: Dict[str, float] = field(default_factory=dict)  # Numeric readings
    status: Dict[str, Any] = field(default_factory=dict)          # Status flags
    
    # Alerts
    threats: List[Dict[str, Any]] = field(default_factory=list)   # Danger signals
    anomalies: List[str] = field(default_factory=list)            # Unusual readings
    
    def to_state_key(self) -> str:
        """Create a hashable key for this sensor state."""
        # Try description first
        if self.description and "None" not in self.description:
            return f"{self.sensor_type}:{self.description[:50]}"
        
        # Build from measurements
        if self.measurements:
            parts = [f"{k}={v:.1f}" for k, v in sorted(self.measurements.items())]
            return f"{self.sensor_type}:{','.join(parts[:5])}"
        
        # Build from objects
        if self.objects:
            obj_types = [str(o.get("type", "obj")) for o in self.objects[:3]]
            return f"{self.sensor_type}:{','.join(obj_types)}"
        
        return f"{self.sensor_type}:{self.timestamp:.0f}"
    
    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "sensor_type": self.sensor_type,
            "description": self.description,
            "objects": self.objects,
            "measurements": self.measurements,
            "status": self.status,
            "threats": self.threats,
            "anomalies": self.anomalies
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'SensorReport':
        return cls(
            timestamp=data.get("timestamp", time()),
            sensor_type=data.get("sensor_type", "unknown"),
            description=data.get("description", ""),
            raw_data=data.get("raw_data"),
            objects=data.get("objects", []),
            measurements=data.get("measurements", {}),
            status=data.get("status", {}),
            threats=data.get("threats", []),
            anomalies=data.get("anomalies", [])
        )
    
    # Convenience constructors
    @classmethod
    def vision(cls, description: str, objects: List[Dict] = None, 
               threats: List[Dict] = None) -> 'SensorReport':
        """Create a vision sensor report (from Sonnet/camera)."""
        return cls(
            sensor_type="vision",
            description=description,
            objects=objects or [],
            threats=threats or []
        )
    
    @classmethod
    def thermal(cls, temperatures: Dict[str, float], 
                anomalies: List[str] = None) -> 'SensorReport':
        """Create a thermal sensor report (from Pi sensors)."""
        return cls(
            sensor_type="thermal",
            measurements=temperatures,
            anomalies=anomalies or [],
            description=f"Thermal: {temperatures}"
        )
    
    @classmethod
    def audio(cls, description: str, raw_data: Any = None) -> 'SensorReport':
        """Create an audio sensor report."""
        return cls(
            sensor_type="audio",
            description=description,
            raw_data=raw_data
        )


# ═══════════════════════════════════════════════════════════════════════════════
# MOTOR ACTION - Individual potential action
# ═══════════════════════════════════════════════════════════════════════════════

class MotorType(Enum):
    """Types of motor actions."""
    # Keyboard
    KEY_PRESS = "key_press"       # Tap a key
    KEY_HOLD = "key_hold"         # Hold key down
    KEY_RELEASE = "key_release"   # Release held key
    
    # Mouse
    MOUSE_MOVE = "mouse_move"     # Move mouse
    MOUSE_CLICK = "mouse_click"   # Click
    MOUSE_HOLD = "mouse_hold"     # Hold button
    MOUSE_RELEASE = "mouse_release"
    MOUSE_SCROLL = "mouse_scroll"
    
    # Control
    WAIT = "wait"                 # Pause
    LOOK = "look"                 # Camera/view direction
    
    # API/External
    API_CALL = "api_call"         # External API
    SEND_MESSAGE = "send_message" # Communication


@dataclass
class MotorAction:
    """
    Individual motor action a driver can take.
    
    This is a potential choice - drivers cluster these as options.
    Heat cost indicates energy required.
    
    Examples:
        press("w")           - Move forward
        mouse_click(x, y)    - Click at position
        api_call("/status")  - Check API
    """
    motor_type: MotorType
    heat_cost: float = 1.0
    
    # Type-specific parameters
    key: Optional[str] = None              # For KEY_* types
    button: Optional[str] = None           # For MOUSE_* types ("left", "right")
    position: Optional[Tuple[int, int]] = None  # For MOUSE_MOVE/CLICK
    direction: Optional[Tuple[float, float]] = None  # For LOOK (dx, dy)
    duration: Optional[float] = None       # For HOLD/WAIT types
    until: Optional[str] = None            # Condition to stop
    
    # For API/message types
    endpoint: Optional[str] = None
    payload: Optional[Dict] = None
    
    # Metadata
    name: Optional[str] = None             # Human-readable name
    description: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "motor_type": self.motor_type.value,
            "heat_cost": self.heat_cost,
            "key": self.key,
            "button": self.button,
            "position": self.position,
            "direction": self.direction,
            "duration": self.duration,
            "until": self.until,
            "endpoint": self.endpoint,
            "payload": self.payload,
            "name": self.name,
            "description": self.description
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'MotorAction':
        return cls(
            motor_type=MotorType(data["motor_type"]),
            heat_cost=data.get("heat_cost", 1.0),
            key=data.get("key"),
            button=data.get("button"),
            position=tuple(data["position"]) if data.get("position") else None,
            direction=tuple(data["direction"]) if data.get("direction") else None,
            duration=data.get("duration"),
            until=data.get("until"),
            endpoint=data.get("endpoint"),
            payload=data.get("payload"),
            name=data.get("name"),
            description=data.get("description")
        )
    
    def __repr__(self) -> str:
        if self.name:
            return f"MotorAction({self.name})"
        return f"MotorAction({self.motor_type.value})"


# Convenience constructors
def press(key: str, heat_cost: float = 1.0) -> MotorAction:
    """Create key press action."""
    return MotorAction(MotorType.KEY_PRESS, key=key, heat_cost=heat_cost, 
                       name=f"press_{key}")

def hold_key(key: str, duration: float = None, until: str = None) -> MotorAction:
    """Create key hold action."""
    return MotorAction(MotorType.KEY_HOLD, key=key, duration=duration, 
                       until=until, heat_cost=2.0, name=f"hold_{key}")

def release_key(key: str) -> MotorAction:
    """Create key release action."""
    return MotorAction(MotorType.KEY_RELEASE, key=key, heat_cost=0.5,
                       name=f"release_{key}")

def mouse_move(x: int, y: int) -> MotorAction:
    """Create mouse move action."""
    return MotorAction(MotorType.MOUSE_MOVE, position=(x, y), heat_cost=1.0,
                       name=f"move_to_{x}_{y}")

def mouse_click(x: int = None, y: int = None, button: str = "left") -> MotorAction:
    """Create mouse click action."""
    pos = (x, y) if x is not None and y is not None else None
    return MotorAction(MotorType.MOUSE_CLICK, position=pos, button=button,
                       heat_cost=1.0, name=f"click_{button}")

def mouse_hold(button: str = "left", until: str = None) -> MotorAction:
    """Create mouse hold action."""
    return MotorAction(MotorType.MOUSE_HOLD, button=button, until=until,
                       heat_cost=2.0, name=f"hold_{button}")

def mouse_release(button: str = "left") -> MotorAction:
    """Create mouse release action."""
    return MotorAction(MotorType.MOUSE_RELEASE, button=button, heat_cost=0.5,
                       name=f"release_{button}")

def look(dx: float, dy: float) -> MotorAction:
    """Create look/camera action."""
    return MotorAction(MotorType.LOOK, direction=(dx, dy), heat_cost=1.0,
                       name=f"look_{dx}_{dy}")

def wait(duration: float) -> MotorAction:
    """Create wait action."""
    return MotorAction(MotorType.WAIT, duration=duration, heat_cost=0.1,
                       name=f"wait_{duration}s")

def api_call(endpoint: str, payload: Dict = None, heat_cost: float = 1.0) -> MotorAction:
    """Create API call action."""
    return MotorAction(MotorType.API_CALL, endpoint=endpoint, payload=payload,
                       heat_cost=heat_cost, name=f"api_{endpoint}")


# ═══════════════════════════════════════════════════════════════════════════════
# ACTION PLAN - Sequence of MotorActions directed by capabilities
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ActionPlan:
    """
    Ordered sequence of MotorActions to achieve a goal.
    
    Directed by capabilities - a capability knows which plans
    achieve its goals.
    
    Examples:
        "mine_tree":
            1. look(up)
            2. hold_key("w") until close
            3. mouse_hold("left") until break
            4. release all
            
        "respond_to_user":
            1. api_call("/get_context")
            2. api_call("/generate_response")
            3. send_message(response)
    """
    name: str
    goal: str                                    # What this achieves
    steps: List[MotorAction] = field(default_factory=list)
    
    # Prerequisites
    requires: List[str] = field(default_factory=list)   # Required state/items
    provides: List[str] = field(default_factory=list)   # What it produces
    
    # Learning stats
    heat_cost: float = K                         # Total heat required
    executions: int = 0
    successes: int = 0
    avg_duration: float = 0.0
    
    @property
    def success_rate(self) -> float:
        if self.executions == 0:
            return 0.0
        return self.successes / self.executions
    
    def record_execution(self, success: bool, duration: float):
        """Record an execution attempt."""
        self.executions += 1
        if success:
            self.successes += 1
        # Running average
        self.avg_duration = (self.avg_duration * (self.executions - 1) + duration) / self.executions
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "goal": self.goal,
            "steps": [s.to_dict() for s in self.steps],
            "requires": self.requires,
            "provides": self.provides,
            "heat_cost": self.heat_cost,
            "executions": self.executions,
            "successes": self.successes,
            "avg_duration": self.avg_duration
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ActionPlan':
        steps = [MotorAction.from_dict(s) for s in data.get("steps", [])]
        return cls(
            name=data["name"],
            goal=data.get("goal", ""),
            steps=steps,
            requires=data.get("requires", []),
            provides=data.get("provides", []),
            heat_cost=data.get("heat_cost", K),
            executions=data.get("executions", 0),
            successes=data.get("successes", 0),
            avg_duration=data.get("avg_duration", 0.0)
        )
    
    def __repr__(self) -> str:
        return f"ActionPlan({self.name}: {len(self.steps)} steps, {self.success_rate:.0%} success)"


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def create_plan(name: str, goal: str, steps: List[MotorAction], 
                requires: List[str] = None, provides: List[str] = None) -> ActionPlan:
    """Create an action plan."""
    heat_cost = sum(s.heat_cost for s in steps)
    return ActionPlan(
        name=name,
        goal=goal,
        steps=steps,
        requires=requires or [],
        provides=provides or [],
        heat_cost=heat_cost
    )


# ═══════════════════════════════════════════════════════════════════════════════
# DRIVER NODE - Environment interface with tag-based integration
# ═══════════════════════════════════════════════════════════════════════════════

import os
import json
from .node_constants import K, THRESHOLD_ORDER, THRESHOLD_RIGHTEOUSNESS, get_growth_path


class DriverNode:
    """
    A driver node interfaces with an environment.
    
    The DriverNode:
    - Tags itself as driver:{name} in manifold
    - Tags task nodes as task:{name}
    - Stores states/motors/plans in drivers/{name}/
    - Has .see() to process SensorReports
    - Has .register_motor() to add motor actions
    - Has .act() to execute actions
    - Has .save()/.load() for persistence
    
    Filesystem structure:
        drivers/{name}/
        ├── frame.json      # Node data + metadata
        ├── states/         # Sensor states seen
        ├── motors/         # Available motor actions
        └── plans/          # Learned action plans
    """
    
    def __init__(self, name: str, manifold: 'Manifold' = None):
        """
        Initialize or load a driver node.
        
        Args:
            name: Driver name (e.g., "minecraft", "browser")
            manifold: The PBAI manifold
        """
        self.name = name
        self.manifold = manifold
        self.born = False
        
        # Core node (will be tagged driver:{name})
        self.node = None
        
        # Learned structures
        self.states: Dict[str, dict] = {}              # state_key → state data
        self.motor_patterns: Dict[str, MotorAction] = {}  # pattern_name → motor
        self.plans: Dict[str, ActionPlan] = {}         # plan_name → plan
        
        # Runtime state
        self.current_sensor: Optional[SensorReport] = None
        self.current_state_key: Optional[str] = None
        self.current_plan: Optional[ActionPlan] = None
        self.current_step: int = 0
        
        # Execution history
        self.action_history: List[Tuple[str, MotorAction, float]] = []
        
        # Motor execution callback
        self._motor_executor: Optional[callable] = None
        
        # Filesystem path
        project_root = get_growth_path("").replace("/growth", "")
        self.driver_dir = os.path.join(project_root, "drivers", name)
        
        # Birth
        self._birth()
    
    def _birth(self):
        """Birth this driver - load if exists, create if not."""
        if self.born:
            logger.warning(f"DriverNode {self.name} already born")
            return
        
        if os.path.exists(os.path.join(self.driver_dir, "frame.json")):
            self._load()
        else:
            self._create()
        
        self.born = True
        logger.debug(f"DriverNode {self.name} born")
    
    def _create(self):
        """Create a new driver node."""
        logger.info(f"Creating new driver: {self.name}")
        
        from .nodes import Node
        
        # Create node
        self.node = Node(
            concept=self.name,
            position="u",  # Drivers go "up" (task level)
            heat=K,
            polarity=1,
            existence="actual",
            righteousness=1.0,
            order=1
        )
        
        # Tag as driver
        self.node.add_tag(f"driver:{self.name}")
        
        # Add to manifold
        if self.manifold:
            existing = self.manifold.get_node_by_concept(self.name)
            if existing:
                self.node = existing
                self.node.add_tag(f"driver:{self.name}")
            else:
                self.manifold.add_node(self.node)
        
        # Create directory structure
        os.makedirs(self.driver_dir, exist_ok=True)
        os.makedirs(os.path.join(self.driver_dir, "states"), exist_ok=True)
        os.makedirs(os.path.join(self.driver_dir, "motors"), exist_ok=True)
        os.makedirs(os.path.join(self.driver_dir, "plans"), exist_ok=True)
        
        self.save()
    
    def _load(self):
        """Load existing driver node."""
        logger.info(f"Loading driver: {self.name}")
        
        from .nodes import Node
        
        # Load frame
        frame_path = os.path.join(self.driver_dir, "frame.json")
        with open(frame_path, "r") as f:
            data = json.load(f)
        
        self.node = Node.from_dict(data.get("node", {}))
        self.node.add_tag(f"driver:{self.name}")
        
        # Add to manifold if provided
        if self.manifold:
            existing = self.manifold.get_node_by_concept(self.name)
            if existing:
                self.node = existing
                self.node.add_tag(f"driver:{self.name}")
            else:
                self.manifold.add_node(self.node)
        
        # Load states
        states_dir = os.path.join(self.driver_dir, "states")
        if os.path.exists(states_dir):
            for filename in os.listdir(states_dir):
                if filename.endswith(".json"):
                    with open(os.path.join(states_dir, filename), "r") as f:
                        state_data = json.load(f)
                    state_key = state_data.get("state_key", filename[:-5])
                    self.states[state_key] = state_data
        
        # Load motors
        motors_dir = os.path.join(self.driver_dir, "motors")
        if os.path.exists(motors_dir):
            for filename in os.listdir(motors_dir):
                if filename.endswith(".json"):
                    with open(os.path.join(motors_dir, filename), "r") as f:
                        motor_data = json.load(f)
                    motor = MotorAction.from_dict(motor_data)
                    self.motor_patterns[motor.name or filename[:-5]] = motor
        
        # Load plans
        plans_dir = os.path.join(self.driver_dir, "plans")
        if os.path.exists(plans_dir):
            for filename in os.listdir(plans_dir):
                if filename.endswith(".json"):
                    with open(os.path.join(plans_dir, filename), "r") as f:
                        plan_data = json.load(f)
                    plan = ActionPlan.from_dict(plan_data)
                    self.plans[plan.name] = plan
        
        logger.info(f"Loaded driver {self.name}: {len(self.states)} states, "
                   f"{len(self.motor_patterns)} motors, {len(self.plans)} plans")
    
    def save(self):
        """Save driver state to filesystem."""
        os.makedirs(self.driver_dir, exist_ok=True)
        
        # Save frame
        frame_path = os.path.join(self.driver_dir, "frame.json")
        data = {
            "name": self.name,
            "node": self.node.to_dict() if self.node else {},
            "state_count": len(self.states),
            "motor_count": len(self.motor_patterns),
            "plan_count": len(self.plans),
            "saved_at": time()
        }
        with open(frame_path, "w") as f:
            json.dump(data, f, indent=2)
        
        # Save states
        states_dir = os.path.join(self.driver_dir, "states")
        os.makedirs(states_dir, exist_ok=True)
        for state_key, state_data in self.states.items():
            safe_key = state_key.replace("/", "_").replace(":", "_")[:50]
            with open(os.path.join(states_dir, f"{safe_key}.json"), "w") as f:
                json.dump(state_data, f, indent=2)
        
        # Save motors
        motors_dir = os.path.join(self.driver_dir, "motors")
        os.makedirs(motors_dir, exist_ok=True)
        for motor_name, motor in self.motor_patterns.items():
            safe_name = motor_name.replace("/", "_")[:50]
            with open(os.path.join(motors_dir, f"{safe_name}.json"), "w") as f:
                json.dump(motor.to_dict(), f, indent=2)
        
        # Save plans
        plans_dir = os.path.join(self.driver_dir, "plans")
        os.makedirs(plans_dir, exist_ok=True)
        for plan_name, plan in self.plans.items():
            safe_name = plan_name.replace("/", "_")[:50]
            with open(os.path.join(plans_dir, f"{safe_name}.json"), "w") as f:
                json.dump(plan.to_dict(), f, indent=2)
        
        logger.debug(f"Saved driver {self.name}")
    
    # ═══════════════════════════════════════════════════════════════════════
    # SENSOR PROCESSING
    # ═══════════════════════════════════════════════════════════════════════
    
    def see(self, report: SensorReport) -> str:
        """
        Process a sensor report.
        
        Creates or updates state node tagged task:{driver_name}.
        Returns state key.
        """
        self.current_sensor = report
        state_key = report.to_state_key()
        self.current_state_key = state_key
        
        # Store state
        self.states[state_key] = {
            "state_key": state_key,
            "sensor_type": report.sensor_type,
            "description": report.description,
            "objects": report.objects,
            "measurements": report.measurements,
            "status": report.status,
            "threats": report.threats,
            "seen_count": self.states.get(state_key, {}).get("seen_count", 0) + 1,
            "last_seen": time()
        }
        
        # Create/update task node in manifold
        if self.manifold:
            from .nodes import Node
            task_node = self.manifold.get_node_by_concept(state_key)
            if not task_node:
                task_node = Node(
                    concept=state_key,
                    position=self.node.position + "n" if self.node else "un",
                    heat=K * 0.5,
                    righteousness=0.5
                )
                self.manifold.add_node(task_node)
            
            # Tag as task for this driver
            task_node.add_tag(f"task:{self.name}")
            
            # Connect driver to task
            if self.node:
                self.node.add_axis(state_key[:20], task_node.id)
        
        logger.debug(f"Driver {self.name} saw: {state_key}")
        return state_key
    
    # ═══════════════════════════════════════════════════════════════════════
    # MOTOR MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════════
    
    def register_motor(self, name: str, motor: MotorAction) -> None:
        """Register a motor action pattern."""
        motor.name = name
        self.motor_patterns[name] = motor
        
        # Add as axis on driver node
        if self.node and self.manifold:
            # Motors are potential actions - store as axis
            self.node.add_axis(f"motor_{name}", self.node.id, polarity=1)
        
        logger.debug(f"Registered motor: {name}")
    
    def get_motor(self, name: str) -> Optional[MotorAction]:
        """Get a registered motor action."""
        return self.motor_patterns.get(name)
    
    def set_motor_executor(self, executor: callable):
        """Set the function that executes motor actions."""
        self._motor_executor = executor
    
    # ═══════════════════════════════════════════════════════════════════════
    # ACTION EXECUTION
    # ═══════════════════════════════════════════════════════════════════════
    
    def act(self, motor_or_name) -> bool:
        """
        Execute a motor action.
        
        Args:
            motor_or_name: MotorAction instance or name of registered motor
        
        Returns:
            True if successful
        """
        # Get motor
        if isinstance(motor_or_name, str):
            motor = self.motor_patterns.get(motor_or_name)
            if not motor:
                logger.warning(f"Unknown motor: {motor_or_name}")
                return False
        else:
            motor = motor_or_name
        
        # Record history
        self.action_history.append((
            self.current_state_key or "unknown",
            motor,
            time()
        ))
        
        # Execute if executor set
        if self._motor_executor:
            try:
                return self._motor_executor(motor)
            except Exception as e:
                logger.error(f"Motor execution failed: {e}")
                return False
        
        logger.debug(f"Motor action (no executor): {motor}")
        return True
    
    # ═══════════════════════════════════════════════════════════════════════
    # PLAN MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════════
    
    def add_plan(self, plan: ActionPlan) -> None:
        """Add an action plan."""
        self.plans[plan.name] = plan
        logger.debug(f"Added plan: {plan.name}")
    
    def get_plan(self, name: str) -> Optional[ActionPlan]:
        """Get a plan by name."""
        return self.plans.get(name)
    
    def execute_plan(self, plan_or_name) -> bool:
        """
        Execute an action plan.
        
        Args:
            plan_or_name: ActionPlan instance or name
        
        Returns:
            True if all steps succeeded
        """
        if isinstance(plan_or_name, str):
            plan = self.plans.get(plan_or_name)
            if not plan:
                logger.warning(f"Unknown plan: {plan_or_name}")
                return False
        else:
            plan = plan_or_name
        
        self.current_plan = plan
        self.current_step = 0
        start_time = time()
        
        success = True
        for i, step in enumerate(plan.steps):
            self.current_step = i
            if not self.act(step):
                success = False
                break
        
        duration = time() - start_time
        plan.record_execution(success, duration)
        
        self.current_plan = None
        self.current_step = 0
        
        return success
    
    def __repr__(self) -> str:
        return (f"DriverNode({self.name}: {len(self.states)} states, "
                f"{len(self.motor_patterns)} motors, {len(self.plans)} plans)")
