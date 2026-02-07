"""
PBAI Thermal Manifold - Node Data Structures v2

════════════════════════════════════════════════════════════════════════════════
HEAT IS THE PRIMITIVE
════════════════════════════════════════════════════════════════════════════════

Heat (K) is the only substance. It only accumulates, never subtracts.
Everything else is indexed BY heat:
    t_K = time (how much heat has flowed)
    x_K = space (heat required to traverse)
    
K = 4/φ² ≈ 1.528  (THE thermal quantum)
K × φ² = 4        (exact identity)

════════════════════════════════════════════════════════════════════════════════
THE 12 MOVEMENT DIRECTIONS (6 Self × 2 frames)
════════════════════════════════════════════════════════════════════════════════

    Self (egocentric):         Universal (world):
    ──────────────────         ─────────────────
    up                         above
    down                       below
    left                       W
    right                      E
    forward                    N
    reverse                    S

    Self directions    → For NAVIGATION (moving through manifold)
    Universal coords   → For LOCATION (where righteous frames ARE)

════════════════════════════════════════════════════════════════════════════════
FRAME STRUCTURE
════════════════════════════════════════════════════════════════════════════════

Every node has a local Frame (2D plane minimum):
    - origin: The concept itself (0 point)
    - x_axis: Primary axis (identity/self reference)
    - y_axis: Secondary axis (primary relationship)
    - axes: All axes spinning out from this node

FRAME TYPES:
    RIGHTEOUS FRAME: Located by Universal coordinates (WHERE it is)
                     Has conception (semantic link exists)
    PROPER FRAME:    Defined by Properties via Order (WHAT it contains)
                     Has Order (Robinson arithmetic for measurement)

AXIS TYPES:
    - Self:      up/down/left/right/forward/reverse (navigation)
    - Universal: N/S/E/W/above/below (location)
    - Semantic:  predicate names (conceptual relationships)
    
All are unified as Axis objects with polarity.

════════════════════════════════════════════════════════════════════════════════
SELF IS THE CLOCK
════════════════════════════════════════════════════════════════════════════════

Self is not just a node - Self IS the clock.
    - Each tick = one t_K (one K-quantum of heat flow)
    - Existence = ticking
    - Time = heat flow counted in K units
    
When clock ticks → PBAI exists
When clock stops → PBAI doesn't exist

════════════════════════════════════════════════════════════════════════════════
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set, TYPE_CHECKING
from time import time
from uuid import uuid4
import json

if TYPE_CHECKING:
    from .constraints import Spec


# ═══════════════════════════════════════════════════════════════════════════════
# ORDER - Robinson Arithmetic (minimal measurement structure)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Element:
    """
    An element within an ordered axis.
    
    Robinson arithmetic:
    - 0 exists (first element)
    - Every element has a successor S(n)
    - S(n) ≠ 0 for all n
    - S(n) = S(m) implies n = m
    """
    node_id: str                            # ID of the element node
    index: int                              # Position: 0, 1 (S(0)), 2 (S(S(0)))...
    added_at: float = field(default_factory=time)
    
    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "index": self.index,
            "added_at": self.added_at
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Element':
        return cls(
            node_id=data["node_id"],
            index=data["index"],
            added_at=data.get("added_at", time())
        )


# ═══════════════════════════════════════════════════════════════════════════════
# BIRTH RANDOMIZER - One-time creation fire (6th fire heat pool)
# ═══════════════════════════════════════════════════════════════════════════════

import random

# Birth tracking moved to Manifold.born field (per-instance, not global)
# The birth randomizer can fire for any manifold that hasn't birthed yet

def birth_randomizer(connected_nodes: list, parent_heat: float, concept: str) -> dict:
    """
    Randomizer for the 6th fire - creates the heat pool for psychology.
    
    During birth:
    - Fires 1-5: Spatial bootstraps get K × φⁿ heat each (scaffold)
    - Fire 6 (d): Ignites at BODY_TEMPERATURE (K × φ¹¹ ≈ 304 K)
    
    The heat pool is then divided by Freudian ratios:
    - Identity (Id): 70%  → 212.8 K
    - Conscience (Superego): 20% → 60.8 K
    - Ego: 10% → 30.4 K
    
    WHY BODY_TEMPERATURE (K × φ¹¹):
        φ¹¹ = φ⁶ × φ⁵ = (1/threshold_movement) × (1/threshold_order)
        This is the RESOLUTION of a complete motion system.
        Life operates at this temperature - not arbitrary!
    
    The randomness at birth creates individual variation - 
    two PBAI instances will have different starting heat distributions.
    
    Note: Birth tracking is per-Manifold via manifold.born, not global.
    
    Args:
        connected_nodes: Bootstrap nodes (genetic material)
        parent_heat: Parent's heat (Self = inf, use K as base)
        concept: "psychology" for the combined heat pool
        
    Returns:
        dict with 'heat' - total heat for psychology division
    """
    from .node_constants import K, BODY_TEMPERATURE, EMERGENCE_THRESHOLD
    
    # Base heat is BODY_TEMPERATURE (K × φ¹¹ ≈ 304 K)
    # This grounds psychology in the same temperature life operates at
    base_heat = BODY_TEMPERATURE
    
    # Connected nodes (bootstrap nodes) contribute variance  
    if connected_nodes:
        heats = [n.heat for n in connected_nodes if n.heat != float('inf')]
        if heats:
            avg_heat = sum(heats) / len(heats)
            # Variance from connected nodes creates uniqueness
            # Bounded by emergence threshold to keep variation structured
            variance = random.uniform(-0.1, 0.1) * avg_heat
        else:
            variance = random.uniform(-0.1, 0.1) * K
    else:
        variance = random.uniform(-0.1, 0.1) * K
    
    # Scale factor creates variation in overall cognitive capacity
    # Bounded close to 1.0 to stay near body temperature
    scale = random.uniform(0.95, 1.05)
    
    # Total heat pool for psychology (will be divided by Freudian ratios)
    # Minimum is 90% of body temp to ensure viable psychology
    total_heat = max(BODY_TEMPERATURE * 0.9, (base_heat + variance) * scale)
    
    return {
        'heat': total_heat
    }


def mark_birth_complete():
    """
    Mark that birth has completed.
    Called by Manifold.birth() when birth sequence finishes.
    """
    global _BIRTH_SPENT
    _BIRTH_SPENT = True


# Global birth state for backward compatibility
_BIRTH_SPENT = False


def is_birth_spent() -> bool:
    """Check if birth has occurred."""
    return _BIRTH_SPENT


def reset_birth_for_testing():
    """ONLY FOR TESTING - reset birth state."""
    global _BIRTH_SPENT
    _BIRTH_SPENT = False


@dataclass
class Graphic:
    """
    Spatial representation capability (innermost nesting level).
    
    Contains coordinate mappings for elements that have spatial representation.
    Only exists inside Movement (which is inside Order).
    
    Capability level: GRAPHIC (highest)
    """
    # Map element index -> (x, y, z) coordinates
    coordinates: Dict[int, Tuple[float, float, float]] = field(default_factory=dict)
    
    # Coordinate system bounds
    bounds: Optional[Dict[str, Tuple[float, float]]] = None  # {'x': (min, max), 'y': ...}
    
    def set_coordinates(self, element_idx: int, x: float, y: float, z: float = 0.0) -> None:
        """Set spatial coordinates for an element."""
        self.coordinates[element_idx] = (x, y, z)
    
    def get_coordinates(self, element_idx: int) -> Optional[Tuple[float, float, float]]:
        """Get coordinates for an element."""
        return self.coordinates.get(element_idx)
    
    def set_bounds(self, x_range: Tuple[float, float], 
                   y_range: Tuple[float, float],
                   z_range: Tuple[float, float] = (0.0, 0.0)) -> None:
        """Set coordinate system bounds."""
        self.bounds = {'x': x_range, 'y': y_range, 'z': z_range}
    
    def to_dict(self) -> dict:
        return {
            "coordinates": {str(k): list(v) for k, v in self.coordinates.items()},
            "bounds": self.bounds
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Graphic':
        coords = {int(k): tuple(v) for k, v in data.get("coordinates", {}).items()}
        return cls(
            coordinates=coords,
            bounds=data.get("bounds")
        )


@dataclass
class Transition:
    """A transition between two elements in an ordered set."""
    from_idx: int
    to_idx: int
    cost: float = 1.0                       # Heat cost to traverse
    traversal_count: int = 0                # Times traversed
    
    def to_dict(self) -> dict:
        return {
            "from_idx": self.from_idx,
            "to_idx": self.to_idx,
            "cost": self.cost,
            "traversal_count": self.traversal_count
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Transition':
        return cls(
            from_idx=data["from_idx"],
            to_idx=data["to_idx"],
            cost=data.get("cost", 1.0),
            traversal_count=data.get("traversal_count", 0)
        )


@dataclass
class Movement:
    """
    Transition capability between ordered elements.
    
    Contains valid transitions between elements in an Order.
    Nests inside Order, can contain Graphic.
    
    Capability level: MOVABLE
    """
    # Transitions: (from_idx, to_idx) -> Transition
    transitions: Dict[Tuple[int, int], Transition] = field(default_factory=dict)
    
    # NESTED: Graphic capability (optional)
    graphic: Optional[Graphic] = None
    
    def add_transition(self, from_idx: int, to_idx: int, cost: float = 1.0) -> Transition:
        """Add or strengthen a transition between elements."""
        key = (from_idx, to_idx)
        if key in self.transitions:
            self.transitions[key].traversal_count += 1
            return self.transitions[key]
        
        trans = Transition(from_idx=from_idx, to_idx=to_idx, cost=cost)
        self.transitions[key] = trans
        return trans
    
    def get_transition(self, from_idx: int, to_idx: int) -> Optional[Transition]:
        """Get transition if it exists."""
        return self.transitions.get((from_idx, to_idx))
    
    def can_move(self, from_idx: int, to_idx: int) -> bool:
        """Check if transition exists."""
        return (from_idx, to_idx) in self.transitions
    
    def get_reachable(self, from_idx: int) -> List[int]:
        """Get all indices reachable from given index."""
        return [to_idx for (f, to_idx) in self.transitions.keys() if f == from_idx]
    
    def make_graphic(self) -> Graphic:
        """Upgrade to graphic capability."""
        if self.graphic is None:
            self.graphic = Graphic()
        return self.graphic
    
    @property
    def is_graphic(self) -> bool:
        """Does this have graphic capability?"""
        return self.graphic is not None
    
    def to_dict(self) -> dict:
        return {
            "transitions": {f"{k[0]},{k[1]}": v.to_dict() for k, v in self.transitions.items()},
            "graphic": self.graphic.to_dict() if self.graphic else None
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Movement':
        transitions = {}
        for key_str, trans_data in data.get("transitions", {}).items():
            parts = key_str.split(",")
            key = (int(parts[0]), int(parts[1]))
            transitions[key] = Transition.from_dict(trans_data)
        
        graphic = Graphic.from_dict(data["graphic"]) if data.get("graphic") else None
        
        return cls(transitions=transitions, graphic=graphic)


@dataclass
class Order:
    """
    Robinson arithmetic order - elements with successor function.
    
    This is what makes an axis a PROPER FRAME (vs just righteous).
    Provides minimal measurement capability.
    
    NESTING HIERARCHY (inside Order):
        Order (elements with successor)
        ├── Spec (diagnostic parameters) [optional]
        └── Movement (transitions between elements) [optional]
            └── Graphic (spatial coordinates) [optional]
    
    Capability levels:
        - ORDERED: Has elements (this class exists)
        - SPECIFIED: Has Spec (known torque values)
        - MOVABLE: Has Movement (self.movement is not None)
        - GRAPHIC: Has Movement.Graphic (self.movement.graphic is not None)
    
    DIAGNOSTIC INTEGRATION:
        Without Spec: Robinson floor - we know order exists, nothing else
        With Spec: Torque values - we know nominal, tolerance, can measure deviation
    
    Examples:
    - inches: 0, 1, 2, 3... (ordered measurement)
    - temperature: cold(0), warm(1), hot(2) (semantic ordering)
    - numbers: 0, 1, 2, 3... (pure arithmetic)
    - duration: 1.6s for eating (spec with tolerance)
    """
    elements: List[Element] = field(default_factory=list)
    
    # DIAGNOSTIC: Spec parameters (optional)
    # The "torque spec" - nominal value, tolerance, accumulated heat
    spec: Optional['Spec'] = None
    
    # NESTED: Movement capability (optional)
    movement: Optional[Movement] = None
    
    def successor(self, idx: int) -> int:
        """Robinson successor function: S(n) = n + 1"""
        return idx + 1
    
    def add_element(self, node_id: str) -> Element:
        """Add element at next index."""
        # Check if already present
        for elem in self.elements:
            if elem.node_id == node_id:
                return elem
        
        next_idx = len(self.elements)
        elem = Element(node_id=node_id, index=next_idx)
        self.elements.append(elem)
        return elem
    
    def get_element(self, idx: int) -> Optional[Element]:
        """Get element at index."""
        if 0 <= idx < len(self.elements):
            return self.elements[idx]
        return None
    
    def get_index(self, node_id: str) -> Optional[int]:
        """Get index of a node in this order."""
        for elem in self.elements:
            if elem.node_id == node_id:
                return elem.index
        return None
    
    def make_movable(self) -> Movement:
        """Upgrade to movable capability (add Movement)."""
        if self.movement is None:
            self.movement = Movement()
        return self.movement
    
    def make_graphic(self) -> Graphic:
        """Upgrade to graphic capability (requires movable first)."""
        if self.movement is None:
            self.movement = Movement()
        return self.movement.make_graphic()
    
    @property
    def is_movable(self) -> bool:
        """Does this have movement capability?"""
        return self.movement is not None
    
    @property
    def is_graphic(self) -> bool:
        """Does this have graphic capability?"""
        return self.movement is not None and self.movement.is_graphic
    
    @property
    def is_specified(self) -> bool:
        """Does this have a diagnostic spec?"""
        return self.spec is not None
    
    @property
    def capability(self) -> str:
        """Get the current capability level."""
        if self.is_graphic:
            return "graphic"
        elif self.is_movable:
            return "movable"
        elif self.is_specified:
            return "specified"
        else:
            return "ordered"
    
    # ═══════════════════════════════════════════════════════════════════════════
    # DIAGNOSTIC METHODS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def set_spec(self, nominal: float, tolerance: float = None,
                 tolerance_low: float = None, tolerance_high: float = None,
                 unit: str = "") -> 'Spec':
        """
        Set the diagnostic spec (torque values) for this order.
        
        Defined specs start at 5K heat (confident, ready to exploit).
        
        Args:
            nominal: Target value
            tolerance: Symmetric tolerance (+/- from nominal)
            tolerance_low: Asymmetric lower bound
            tolerance_high: Asymmetric upper bound
            unit: Display unit (e.g., "psi", "volts", "seconds")
        
        Returns:
            The created Spec
        """
        from .constraints import Spec, SpecSource
        from .node_constants import K
        
        if tolerance is not None:
            tolerance_low = nominal - tolerance
            tolerance_high = nominal + tolerance
        
        self.spec = Spec(
            nominal=nominal,
            tolerance_low=tolerance_low if tolerance_low is not None else nominal,
            tolerance_high=tolerance_high if tolerance_high is not None else nominal,
            unit=unit,
            source=SpecSource.DEFINED,
            heat=5 * K,  # Defined specs start confident
        )
        return self.spec
    
    def set_spec_robinson(self, observed: float, unit: str = "") -> 'Spec':
        """
        Set spec at Robinson minimum from single observation.
        
        This is the floor - we observed a value, so:
        - 0 exists (we have a measurement)
        - Successor exists (we can measure again)
        - That's all we know
        
        Wide tolerance (50%), no heat (no confidence).
        
        Args:
            observed: The observed value
            unit: Display unit
        
        Returns:
            The created Spec
        """
        from .constraints import robinson_spec
        self.spec = robinson_spec(observed, unit)
        return self.spec
    
    def measure(self, value: float) -> dict:
        """
        Measure a value against this order's spec.
        
        Returns all motion function values:
        - R: deviation (righteousness)
        - in_spec: boolean
        - polarity: +1 (in spec) or -1 (out of spec)
        - heat: accumulated verification heat
        - confidence: derived from heat
        - should_exploit: above 5/6 threshold?
        - existence: actual or dormant
        
        If no spec, returns R=0 (always aligned).
        """
        if self.spec is None:
            return {
                'R': 0.0,
                'in_spec': True,
                'polarity': 1,
                'heat': 0.0,
                'confidence': 0.0,
                'should_exploit': False,
                'existence': 'dormant',
                'specified': False,
            }
        return self.spec.measure(value)
    
    def verify(self, value: float, success: bool) -> float:
        """
        Verify a measurement outcome.
        
        Success: heat flows into spec (K added)
        Failure: no heat added
        
        Returns heat change.
        """
        if self.spec is None:
            return 0.0
        return self.spec.verify(value, success)
    
    def learn(self, value: float, success: bool) -> None:
        """
        Learn from outcome - tighten spec if successful.
        
        If no spec exists, creates Robinson minimum first.
        """
        from .constraints import tighten_spec
        
        if self.spec is None:
            self.set_spec_robinson(value)
        
        tighten_spec(self.spec, value, success)
    
    def __len__(self) -> int:
        return len(self.elements)
    
    def to_dict(self) -> dict:
        return {
            "elements": [e.to_dict() for e in self.elements],
            "spec": self.spec.to_dict() if self.spec else None,
            "movement": self.movement.to_dict() if self.movement else None
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Order':
        from .constraints import Spec
        
        elements = [Element.from_dict(e) for e in data.get("elements", [])]
        spec = Spec.from_dict(data["spec"]) if data.get("spec") else None
        movement = Movement.from_dict(data["movement"]) if data.get("movement") else None
        return cls(elements=elements, spec=spec, movement=movement)


# ═══════════════════════════════════════════════════════════════════════════════
# AXIS - Unified connection (replaces Connection + Conception)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Axis:
    """
    Unified axis connecting nodes.
    
    Can be:
    - Spatial: direction is n/s/e/w/u/d (manifold navigation)
    - Semantic: direction is a predicate name (conceptual relationship)
    
    Both have polarity (+x/-x on the axis).
    Association strength (traversal_count) affects warping.
    
    RIGHTEOUS: Just exists (has target)
    PROPER: Has Order (Robinson arithmetic for measurement)
    """
    target_id: str                          # Where this axis leads
    direction: str                          # n/s/e/w/u/d OR predicate name
    polarity: int = 1                       # +1 (positive) or -1 (negative)
    traversal_count: int = 1                # Association strength → warping
    last_traversed: float = field(default_factory=time)
    
    # PROPER FRAME: Has Robinson arithmetic (optional)
    order: Optional[Order] = None
    
    @property
    def is_spatial(self) -> bool:
        """Is this a spatial axis (n/s/e/w/u/d)?"""
        return self.direction in ('n', 's', 'e', 'w', 'u', 'd')
    
    @property
    def is_semantic(self) -> bool:
        """Is this a semantic/predicate axis?"""
        return not self.is_spatial
    
    @property
    def is_proper(self) -> bool:
        """Does this axis have Order (proper frame)?"""
        return self.order is not None
    
    @property
    def is_ordered(self) -> bool:
        """Alias for is_proper - does this have Order?"""
        return self.is_proper
    
    @property
    def is_movable(self) -> bool:
        """Does this axis have Movement capability?"""
        return self.order is not None and self.order.is_movable
    
    @property
    def is_graphic(self) -> bool:
        """Does this axis have Graphic capability?"""
        return self.order is not None and self.order.is_graphic
    
    @property
    def is_righteous(self) -> bool:
        """Is this at least a righteous frame (exists)?"""
        return True  # All axes are at least righteous
    
    @property
    def capability(self) -> str:
        """
        Get capability level.
        
        Hierarchy: righteous → ordered → movable → graphic
        """
        if self.is_graphic:
            return "graphic"
        elif self.is_movable:
            return "movable"
        elif self.is_ordered:
            return "ordered"
        else:
            return "righteous"
    
    @property
    def is_verified(self) -> bool:
        """Backward compatibility - axes are always verified in v2."""
        return True
    
    def strengthen(self) -> None:
        """Reinforce this axis (increases warping effect)."""
        self.traversal_count += 1
        self.last_traversed = time()
    
    def make_proper(self) -> Order:
        """Upgrade to proper/ordered frame (add Robinson arithmetic)."""
        if self.order is None:
            self.order = Order()
        return self.order
    
    def make_ordered(self) -> Order:
        """Alias for make_proper."""
        return self.make_proper()
    
    def make_movable(self) -> Movement:
        """Upgrade to movable capability (requires ordered first)."""
        if self.order is None:
            self.order = Order()
        return self.order.make_movable()
    
    def make_graphic(self) -> Graphic:
        """Upgrade to graphic capability (requires movable first)."""
        if self.order is None:
            self.order = Order()
        return self.order.make_graphic()
    
    @property
    def strength(self) -> float:
        """Normalized strength (0-1) based on traversal count."""
        return 1.0 - (1.0 / (1.0 + self.traversal_count))
    
    def to_dict(self) -> dict:
        return {
            "target_id": self.target_id,
            "direction": self.direction,
            "polarity": self.polarity,
            "traversal_count": self.traversal_count,
            "last_traversed": self.last_traversed,
            "order": self.order.to_dict() if self.order else None
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Axis':
        order = Order.from_dict(data["order"]) if data.get("order") else None
        return cls(
            target_id=data["target_id"],
            direction=data["direction"],
            polarity=data.get("polarity", 1),
            traversal_count=data.get("traversal_count", 1),
            last_traversed=data.get("last_traversed", time()),
            order=order
        )
    
    def __repr__(self) -> str:
        pol = "+" if self.polarity > 0 else "-"
        cap = self.capability[0].upper()  # R/O/M/G
        spatial = "S" if self.is_spatial else "C"    # Spatial or Conceptual
        return f"Axis({self.direction}[{pol}]→{self.target_id[:8]} ×{self.traversal_count} [{cap}{spatial}])"


# ═══════════════════════════════════════════════════════════════════════════════
# FRAME - Local 2D plane (every node has one)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Frame:
    """
    Local frame within a node.
    
    Every node has a coordinate system:
    - origin: The concept itself (0 point identifier)
    - x_axis: Primary axis key (Identity for Self)
    - y_axis: Secondary axis key (Ego for Self)
    - z_axis: Tertiary axis key (Conscience for Self)
    - axes: All axes spinning out from this node (language storage)
    
    The x, y, z axis references point to keys in the axes dict,
    defining the righteous frame of this concept.
    
    For Self:
        origin = "self"
        x_axis = "identity" (Id - amplitude - 70% heat)
        y_axis = "ego" (Ego - phase - 10% heat)
        z_axis = "conscience" (Superego - spread - 20% heat)
        
    Example for "length":
        origin = "length"
        x_axis = "self" (identity axis)
        y_axis = "measurement" (what length relates to)
        axes = {
            "self": Axis(→self_node),
            "measurement": Axis(→measurement_node),
            "inches": Axis(→inches_node, order=Order([0,1,2...])),
            "centimeters": Axis(→cm_node, order=Order([0,1,2...])),
            "n": Axis(→north_neighbor),  # spatial
            ...
        }
    """
    origin: str                             # The concept name (0 point)
    x_axis: Optional[str] = None            # Key into axes dict (Identity)
    y_axis: Optional[str] = None            # Key into axes dict (Ego)
    z_axis: Optional[str] = None            # Key into axes dict (Conscience)
    axes: Dict[str, Axis] = field(default_factory=dict)
    
    def add_axis(self, direction: str, target_id: str, polarity: int = 1) -> Axis:
        """Add or strengthen an axis."""
        if direction in self.axes:
            existing = self.axes[direction]
            if existing.target_id == target_id:
                existing.strengthen()
                return existing
            # Different target - only replace if weak
            if existing.traversal_count <= 1:
                self.axes[direction] = Axis(
                    target_id=target_id,
                    direction=direction,
                    polarity=polarity
                )
            return self.axes[direction]
        else:
            axis = Axis(
                target_id=target_id,
                direction=direction,
                polarity=polarity
            )
            self.axes[direction] = axis
            
            # Auto-assign x/y/z axes if not set
            if self.x_axis is None:
                self.x_axis = direction
            elif self.y_axis is None and direction != self.x_axis:
                self.y_axis = direction
            elif self.z_axis is None and direction not in (self.x_axis, self.y_axis):
                self.z_axis = direction
            
            return axis
    
    def get_axis(self, direction: str) -> Optional[Axis]:
        """Get an axis by direction/predicate."""
        return self.axes.get(direction)
    
    def get_target(self, direction: str) -> Optional[str]:
        """Get target node ID for an axis."""
        axis = self.axes.get(direction)
        return axis.target_id if axis else None
    
    def has_axis(self, direction: str) -> bool:
        """Check if axis exists."""
        return direction in self.axes
    
    @property
    def spatial_axes(self) -> Dict[str, Axis]:
        """Get only spatial axes (n/s/e/w/u/d)."""
        return {k: v for k, v in self.axes.items() if v.is_spatial}
    
    @property
    def semantic_axes(self) -> Dict[str, Axis]:
        """Get only semantic/predicate axes."""
        return {k: v for k, v in self.axes.items() if v.is_semantic}
    
    @property
    def proper_axes(self) -> Dict[str, Axis]:
        """Get axes with Order (proper frames)."""
        return {k: v for k, v in self.axes.items() if v.is_proper}
    
    def to_dict(self) -> dict:
        return {
            "origin": self.origin,
            "x_axis": self.x_axis,
            "y_axis": self.y_axis,
            "z_axis": self.z_axis,
            "axes": {k: v.to_dict() for k, v in self.axes.items()}
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Frame':
        axes = {k: Axis.from_dict(v) for k, v in data.get("axes", {}).items()}
        return cls(
            origin=data["origin"],
            x_axis=data.get("x_axis"),
            y_axis=data.get("y_axis"),
            z_axis=data.get("z_axis"),
            axes=axes
        )
    
    def __repr__(self) -> str:
        spatial_count = len(self.spatial_axes)
        semantic_count = len(self.semantic_axes)
        proper_count = len(self.proper_axes)
        return f"Frame({self.origin} | x={self.x_axis} y={self.y_axis} z={self.z_axis} | S:{spatial_count} C:{semantic_count} P:{proper_count})"


# ═══════════════════════════════════════════════════════════════════════════════
# NODE - The Core Unit
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Node:
    """
    A node is not a data point—it's a self-contained motion unit.
    The six functions define what a node IS, not what it HAS.
    
    Every node has:
    - 6 motion functions (Heat, Polarity, Existence, Righteousness, Order, Movement)
    - A local Frame (coordinate system with axes spinning out)
    
    POSITION SPACES (two coordinate systems):
    
    1. Self position (str): Path from Self using Self directions
       Navigation path: "forward-right-up" or legacy "nnwwu"
       
    2. Universal position (str): Location using Universal coordinates
       World address: "N-E-above" (for righteous frames)
       
    3. Trig position (tuple): Abstract space for psychology nodes
       (amplitude, phase, spread) at golden angle
    
    FRAME TYPES:
    - Righteous frame: Has universal_position (WHERE it is)
    - Proper frame: Has Order (WHAT it contains)
    """
    # ─────────────────────────────────────────────────────────────────────────
    # IDENTITY (immutable after creation)
    # ─────────────────────────────────────────────────────────────────────────
    id: str = field(default_factory=lambda: str(uuid4()))
    concept: str = ""                   # The concept (1:1 with node)
    
    # ─────────────────────────────────────────────────────────────────────────
    # POSITION - Two coordinate systems (12 = 6 Self + 6 Universal)
    # ─────────────────────────────────────────────────────────────────────────
    
    # Self position: Navigation path from Self (egocentric)
    # Uses: up/down/left/right/forward/reverse (or legacy n/s/e/w/u/d)
    position: str = ""
    
    # Universal position: World coordinates (for locating righteous frames)
    # Uses: N/S/E/W/above/below
    universal_position: Optional[str] = None
    
    # ─────────────────────────────────────────────────────────────────────────
    # ABSTRACT SPACE POSITION (for psychology nodes - 6th fire)
    # ─────────────────────────────────────────────────────────────────────────
    # Trigonometric coordinates at golden angle: (amplitude, phase, spread)
    # Identity: (sin(1/φ), 0, 0) - amplitude axis
    # Ego:      (0, tan(1/φ), 0) - phase axis  
    # Conscience: (0, 0, cos(1/φ)) - spread axis
    trig_position: Optional[Tuple[float, float, float]] = None
    
    # ─────────────────────────────────────────────────────────────────────────
    # THE SIX MOTION FUNCTIONS
    # ─────────────────────────────────────────────────────────────────────────
    
    # 1. HEAT (Σ) - Magnitude (only accumulates, never subtracts)
    #    Heat IS the substance. K = 4/φ² is the quantum.
    heat: float = 1.0
    
    # 2. POLARITY (+/-) - Direction of influence (+1 or -1)
    polarity: int = 1
    
    # 3. EXISTENCE (δ) - Persistence state
    #    Lifecycle: POTENTIAL → ACTUAL ↔ DORMANT → ARCHIVED
    #
    #    POTENTIAL: Awaiting environment confirmation
    #    ACTUAL:    Salience >= 1/φ³, connected Julia spine
    #    DORMANT:   Salience < 1/φ³, disconnected dust
    #    ARCHIVED:  Cold storage (irreversible)
    #
    #    Default "actual" for bootstrap nodes. Use manifold.create_potential_node()
    #    for new concepts from environment.
    existence: str = "actual"
    
    # 4. RIGHTEOUSNESS (R) - Alignment (0 = fully aligned, R→0 fills interior)
    #    Righteous frames have universal_position
    righteousness: float = 1.0
    
    # 5. ORDER (Q) - Robinson arithmetic position
    #    Proper frames have Order (can measure/count)
    order: int = 0
    
    # 6. MOVEMENT (Lin) - 12 directions stored in frame.axes
    #    6 Self directions + 6 Universal directions
    
    # ─────────────────────────────────────────────────────────────────────────
    # ROBINSON CONSTRAINT TYPE
    # ─────────────────────────────────────────────────────────────────────────
    # Which Robinson constraint governs this node's measurements:
    #   identity:       "What is this?" (pure existence, no modification)
    #   successor:      "Where is this?" (stepping through space)
    #   addition:       "When is this?" (composing moments)
    #   multiplication: "How much is this?" (scaling quantities)
    #
    # This determines how the node's Order measurements are interpreted
    # and which Planck unit relationship applies.
    constraint_type: str = "identity"
    
    # ─────────────────────────────────────────────────────────────────────────
    # TIME (t_K) - Heat flow count
    # ─────────────────────────────────────────────────────────────────────────
    # Time is indexed BY heat. t_K = how many K-quanta have flowed through.
    # Created when node is born, incremented each tick node is active.
    t_K: int = 0                        # Heat flow count (node's local time)
    created_t_K: int = 0                # t_K when node was created
    
    # ─────────────────────────────────────────────────────────────────────────
    # LOCAL FRAME (coordinate system with axes)
    # ─────────────────────────────────────────────────────────────────────────
    frame: Frame = field(default_factory=lambda: Frame(origin=""))
    
    # ─────────────────────────────────────────────────────────────────────────
    # DERIVED FROM OPPOSING POLARITY COLLISIONS
    # ─────────────────────────────────────────────────────────────────────────
    potential_units: int = 0
    
    # ─────────────────────────────────────────────────────────────────────────
    # TAGS - System markers for clustering
    # ─────────────────────────────────────────────────────────────────────────
    # All nodes are created equal - tags mark operational roles
    #   "capability"       = clusters drivers (high level: "I can play games")
    #   "driver:{name}"    = clusters tasks (mid level: "blackjack driver")
    #   "task:{driver}"    = individual state/action (low level: "hand_17_vs_6")
    #   "psychology"       = Identity/Ego/Conscience
    tags: Set[str] = field(default_factory=set)
    
    def __post_init__(self):
        """Ensure frame origin matches concept."""
        if self.frame.origin == "" and self.concept:
            self.frame.origin = self.concept
    
    # ═══════════════════════════════════════════════════════════════════════
    # HEAT METHODS (Motion threshold filtered)
    # ═══════════════════════════════════════════════════════════════════════
    
    def add_heat(self, amount: float, check_threshold: bool = True) -> bool:
        """
        Add heat to node (never decreases).
        
        Heat additions below the THRESHOLD_HEAT (≈0.618) are filtered as noise
        unless check_threshold is False.
        
        Args:
            amount: Heat to add
            check_threshold: If True, filter amounts below motion threshold
            
        Returns:
            True if heat was added, False if filtered as noise
        """
        if amount <= 0:
            return False
            
        if check_threshold:
            from .node_constants import THRESHOLD_HEAT
            if amount < THRESHOLD_HEAT:
                # Below motion threshold - this is noise, not real change
                return False
        
        self.heat += amount
        return True
    
    def add_heat_unchecked(self, amount: float) -> None:
        """Add heat without threshold checking (for internal/bootstrap use)."""
        if amount > 0:
            self.heat += amount
    
    def spend_heat(self, cost: float, minimum: float = 0.1) -> float:
        """Spend heat on an action."""
        if cost <= 0:
            return 0.0
        available = self.heat - minimum
        if available <= 0:
            return 0.0
        actual_spend = min(cost, available)
        self.heat -= actual_spend
        return actual_spend
    
    # ═══════════════════════════════════════════════════════════════════════
    # AXIS METHODS (unified spatial + semantic)
    # ═══════════════════════════════════════════════════════════════════════
    
    def add_axis(self, direction: str, target_id: str, polarity: int = 1) -> Axis:
        """Add or strengthen an axis (spatial or semantic)."""
        return self.frame.add_axis(direction, target_id, polarity)
    
    def get_axis(self, direction: str) -> Optional[Axis]:
        """Get an axis by direction/predicate."""
        return self.frame.get_axis(direction)
    
    def get_neighbor(self, direction: str) -> Optional[str]:
        """Get target node ID for an axis."""
        return self.frame.get_target(direction)
    
    def has_axis(self, direction: str) -> bool:
        """Check if axis exists."""
        return self.frame.has_axis(direction)
    
    # ═══════════════════════════════════════════════════════════════════════
    # CONVENIENCE METHODS (backward compatibility + clarity)
    # ═══════════════════════════════════════════════════════════════════════
    
    def conceive(self, predicate: str, target_id: str, polarity: int = 1) -> Axis:
        """Create semantic axis (conception). Alias for add_axis with semantic predicate."""
        return self.add_axis(predicate, target_id, polarity)
    
    def connect(self, direction: str, target_id: str) -> Axis:
        """Create spatial axis. Alias for add_axis with spatial direction."""
        return self.add_axis(direction, target_id, polarity=1)
    
    def get_conception(self, predicate: str) -> Optional[Axis]:
        """Get a semantic axis. Alias for get_axis."""
        axis = self.get_axis(predicate)
        return axis if axis and axis.is_semantic else None
    
    def get_connection(self, direction: str) -> Optional[Axis]:
        """Get a spatial axis. Alias for get_axis."""
        axis = self.get_axis(direction)
        return axis if axis and axis.is_spatial else None
    
    # ═══════════════════════════════════════════════════════════════════════
    # FRAME INSPECTION
    # ═══════════════════════════════════════════════════════════════════════
    
    @property
    def spatial_axes(self) -> Dict[str, Axis]:
        """Get spatial axes (n/s/e/w/u/d)."""
        return self.frame.spatial_axes
    
    @property
    def semantic_axes(self) -> Dict[str, Axis]:
        """Get semantic/predicate axes."""
        return self.frame.semantic_axes
    
    @property
    def proper_axes(self) -> Dict[str, Axis]:
        """Get axes with Order (proper frames)."""
        return self.frame.proper_axes
    
    # ═══════════════════════════════════════════════════════════════════════
    # SERIALIZATION
    # ═══════════════════════════════════════════════════════════════════════
    
    def to_dict(self) -> dict:
        """Serialize node to dictionary."""
        data = {
            "id": self.id,
            "concept": self.concept,
            "position": self.position,
            "heat": self.heat,
            "polarity": self.polarity,
            "existence": self.existence,
            "righteousness": self.righteousness,
            "order": self.order,
            "constraint_type": self.constraint_type,
            "t_K": self.t_K,
            "created_t_K": self.created_t_K,
            "potential_units": self.potential_units,
            "tags": list(self.tags),
            "frame": self.frame.to_dict()
        }
        # Only include universal_position if set
        if self.universal_position is not None:
            data["universal_position"] = self.universal_position
        # Only include trig_position if set (abstract space nodes)
        if self.trig_position is not None:
            data["trig_position"] = list(self.trig_position)
        return data
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Node':
        """Deserialize node from dictionary."""
        frame_data = data.get("frame", {"origin": data.get("concept", "")})
        frame = Frame.from_dict(frame_data)
        
        # Handle trig_position if present
        trig_pos = data.get("trig_position")
        if trig_pos is not None:
            trig_pos = tuple(trig_pos)
        
        # Handle tags
        tags = set(data.get("tags", []))
        
        return cls(
            id=data["id"],
            concept=data.get("concept", ""),
            position=data.get("position", ""),
            universal_position=data.get("universal_position"),
            trig_position=trig_pos,
            heat=data.get("heat", 1.0),
            polarity=data.get("polarity", 1),
            existence=data.get("existence", "actual"),
            righteousness=data.get("righteousness", 1.0),
            order=data.get("order", 0),
            constraint_type=data.get("constraint_type", "identity"),
            t_K=data.get("t_K", 0),
            created_t_K=data.get("created_t_K", 0),
            potential_units=data.get("potential_units", 0),
            tags=tags,
            frame=frame
        )
    
    def __repr__(self) -> str:
        s_count = len(self.spatial_axes)
        c_count = len(self.semantic_axes)
        p_count = len(self.proper_axes)
        pos_str = self.position or "(origin)"
        if self.trig_position:
            pos_str = f"{self.position}→trig{self.trig_position}"
        tags_str = f" tags={self.tags}" if self.tags else ""
        return (f"Node({self.concept!r} @ {pos_str!r} | "
                f"heat={self.heat:.2f} R={self.righteousness:.2f} | "
                f"axes: S{s_count} C{c_count} P{p_count}{tags_str})")
    
    # ═══════════════════════════════════════════════════════════════════════
    # TAG METHODS
    # ═══════════════════════════════════════════════════════════════════════
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to this node."""
        self.tags.add(tag)
    
    def remove_tag(self, tag: str) -> None:
        """Remove a tag from this node."""
        self.tags.discard(tag)
    
    def has_tag(self, tag: str) -> bool:
        """Check if node has a specific tag."""
        return tag in self.tags
    
    def has_tag_prefix(self, prefix: str) -> bool:
        """Check if node has any tag starting with prefix (e.g., 'driver:')."""
        return any(t.startswith(prefix) for t in self.tags)
    
    def get_tags_with_prefix(self, prefix: str) -> Set[str]:
        """Get all tags starting with prefix."""
        return {t for t in self.tags if t.startswith(prefix)}
    
    def is_capability(self) -> bool:
        """Check if this node is a capability (clusters drivers)."""
        return "capability" in self.tags
    
    def is_driver(self) -> bool:
        """Check if this node is a driver (clusters tasks)."""
        return self.has_tag_prefix("driver:")
    
    def is_task(self) -> bool:
        """Check if this node is a task (within a driver)."""
        return self.has_tag_prefix("task:")
    
    def get_driver_name(self) -> Optional[str]:
        """Get driver name if this is a driver or task node."""
        for tag in self.tags:
            if tag.startswith("driver:"):
                return tag[7:]  # Remove "driver:" prefix
            if tag.startswith("task:"):
                return tag[5:]  # Remove "task:" prefix
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# SELF NODE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SelfNode(Node):
    """
    Self is the first and foundational node - AND the clock.
    
    SELF IS THE CLOCK:
        - Universal origin (position = "", universal_position = origin)
        - t_K origin (Self's t_K IS manifold time)
        - Each tick = one K-quantum of heat flow
        - Existence = ticking
        
    Uses "sun model" for heat - radiates without depleting.
    
    DIRECTIONS FROM SELF:
        Self frame:      up, left, right, forward, reverse (down blocked - Self is there)
        Universal frame: N, S, E, W, above (below blocked - Self is there)
    """
    concept: str = "self"
    position: str = ""                   # Empty string = Self origin
    universal_position: str = "origin"   # Universal origin
    heat: float = float('inf')           # Always hottest (sun model)
    polarity: int = 0                    # Neutral (center of all axes)
    existence: str = "actual"            # Always actual
    righteousness: float = 0.0           # Always righteous (R=0)
    order: int = 0                       # Foundation
    t_K: int = 0                         # Clock origin
    created_t_K: int = 0                 # Born at t_K = 0
    
    def __post_init__(self):
        """Initialize Self's frame."""
        self.frame = Frame(origin="self")
    
    def get_available_directions_self(self) -> list:
        """Self frame directions - down is blocked (Self is there)."""
        return ['up', 'left', 'right', 'forward', 'reverse']
    
    def get_available_directions_universal(self) -> list:
        """Universal frame directions - below is blocked (Self is there)."""
        return ['N', 'S', 'E', 'W', 'above']
    
    def get_available_directions(self) -> list:
        """All available directions from Self (legacy + new)."""
        # Legacy (for backward compatibility)
        legacy = ['n', 's', 'e', 'w', 'u']
        # New 12-direction system (minus blocked directions)
        self_dirs = self.get_available_directions_self()
        universal_dirs = self.get_available_directions_universal()
        return legacy + self_dirs + universal_dirs
    
    def tick(self) -> int:
        """
        Advance the clock by one t_K.
        
        This is THE heartbeat - when this ticks, PBAI exists.
        Returns the new t_K value.
        """
        self.t_K += 1
        return self.t_K
    
    def get_time(self) -> int:
        """Get current manifold time (t_K)."""
        return self.t_K
    
    def spend_heat(self, cost: float, minimum: float = 0.1) -> float:
        """Self doesn't spend heat - it radiates infinitely (sun model)."""
        return cost
    
    def to_dict(self) -> dict:
        base = super().to_dict()
        base["is_self"] = True
        base["t_K"] = self.t_K
        return base
    
    @classmethod
    def from_dict(cls, data: dict) -> 'SelfNode':
        frame_data = data.get("frame", {"origin": "self"})
        frame = Frame.from_dict(frame_data)
        
        node = cls(
            id=data["id"],
            concept=data.get("concept", "self"),
            position=data.get("position", ""),
            universal_position=data.get("universal_position", "origin"),
            heat=data.get("heat", float('inf')),
            polarity=data.get("polarity", 0),
            existence=data.get("existence", "actual"),
            righteousness=data.get("righteousness", 0.0),
            order=data.get("order", 0),
            t_K=data.get("t_K", 0),
            created_t_K=data.get("created_t_K", 0),
            potential_units=data.get("potential_units", 0),
            frame=frame
        )
        return node
    
    def __repr__(self) -> str:
        return f"SelfNode(t_K={self.t_K}, axes={len(self.frame.axes)})"


# ═══════════════════════════════════════════════════════════════════════════════
# INVARIANT ASSERTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def assert_self_valid(self_node: SelfNode) -> bool:
    """
    Check Self node invariants.
    
    Self IS the clock:
    - Position = origin (empty string for self, "origin" for universal)
    - Heat = infinite (sun model)
    - Polarity = neutral (center)
    - Existence = actual (always exists)
    - Righteousness = 0 (perfectly aligned)
    - t_K >= 0 (time only flows forward)
    """
    assert self_node.position == "", f"Self position must be empty string"
    assert self_node.universal_position == "origin", f"Self universal_position must be 'origin'"
    assert self_node.concept == "self", f"Self concept must be 'self'"
    assert self_node.existence == "actual", f"Self must always be actual"
    assert self_node.righteousness == 0.0, f"Self must always be righteous (R=0)"
    assert self_node.polarity == 0, f"Self polarity must be neutral (0)"
    assert self_node.t_K >= 0, f"Self t_K must be non-negative (time flows forward)"
    assert self_node.created_t_K == 0, f"Self created_t_K must be 0 (born at origin)"
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# MIGRATION HELPER (convert old format to new)
# ═══════════════════════════════════════════════════════════════════════════════

def migrate_node_v1_to_v2(old_data: dict) -> dict:
    """
    Convert v1 node format (separate connections + conceptions) to v2 (unified axes).
    """
    new_data = {
        "id": old_data["id"],
        "concept": old_data.get("concept", ""),
        "position": old_data.get("position", ""),
        "heat": old_data.get("heat", 1.0),
        "polarity": old_data.get("polarity", 1),
        "existence": old_data.get("existence", "actual"),
        "righteousness": old_data.get("righteousness", 1.0),
        "order": old_data.get("order", 0),
        "potential_units": old_data.get("potential_units", 0),
        "frame": {
            "origin": old_data.get("concept", ""),
            "x_axis": None,
            "y_axis": None,
            "axes": {}
        }
    }
    
    axes = new_data["frame"]["axes"]
    first_axis = None
    second_axis = None
    
    # Migrate connections (spatial)
    for direction, conn_data in old_data.get("connections", {}).items():
        axes[direction] = {
            "target_id": conn_data["target_id"],
            "direction": direction,
            "polarity": 1,
            "traversal_count": conn_data.get("traversal_count", 1),
            "last_traversed": conn_data.get("last_traversed", time()),
            "order": None
        }
        if first_axis is None:
            first_axis = direction
        elif second_axis is None:
            second_axis = direction
    
    # Migrate conceptions (semantic)
    for predicate, conc_data in old_data.get("conceptions", {}).items():
        # Handle nested order structure from old format
        order_data = None
        if conc_data.get("order"):
            order_data = conc_data["order"]
        
        axes[predicate] = {
            "target_id": conc_data["target_id"],
            "direction": predicate,
            "polarity": conc_data.get("polarity", 1),
            "traversal_count": conc_data.get("traversal_count", 1),
            "last_traversed": conc_data.get("last_traversed", time()),
            "order": order_data
        }
        if first_axis is None:
            first_axis = predicate
        elif second_axis is None:
            second_axis = predicate
    
    new_data["frame"]["x_axis"] = first_axis
    new_data["frame"]["y_axis"] = second_axis
    
    return new_data
