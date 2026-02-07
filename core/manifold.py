"""
PBAI Thermal Manifold v2 - The Manifold Container

════════════════════════════════════════════════════════════════════════════════
HEAT IS THE PRIMITIVE
════════════════════════════════════════════════════════════════════════════════

Heat (K) is the only substance. It only accumulates, never subtracts.
    K = 4/φ² ≈ 1.528 (THE thermal quantum)
    K × φ² = 4 (exact identity)
    t_K = time indexed by heat (how many K-quanta have flowed)

The manifold is a fractal topology constraining WHERE heat CAN flow.
Cognition = heat redistribution along this Julia topology.

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

    Self directions    → For NAVIGATION (traversing the manifold)
    Universal coords   → For LOCATION (where righteous frames ARE)

════════════════════════════════════════════════════════════════════════════════
FRAME TYPES
════════════════════════════════════════════════════════════════════════════════

    RIGHTEOUS FRAME: Located by Universal coordinates (WHERE it is)
    PROPER FRAME:    Defined by Properties via Order (WHAT it contains)

════════════════════════════════════════════════════════════════════════════════
SELF IS THE CLOCK
════════════════════════════════════════════════════════════════════════════════

    Self.t_K = manifold time (advances each tick)
    Each tick = one K-quantum of heat flow
    When clock ticks → PBAI exists
    When clock stops → PBAI doesn't exist

════════════════════════════════════════════════════════════════════════════════
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from datetime import datetime
import json
import os
import logging

from .nodes import Node, SelfNode, Axis, Frame, Order, assert_self_valid, migrate_node_v1_to_v2, birth_randomizer, mark_birth_complete, is_birth_spent
from .compression import compress, decompress, get_axis_coordinates, positions_share_prefix
from .node_constants import (
    K, PHI, 
    # Direction systems (12 = 6 Self + 6 Universal)
    DIRECTIONS_SELF, DIRECTIONS_UNIVERSAL, DIRECTIONS,
    SELF_DIRECTIONS, ALL_DIRECTIONS, OPPOSITES,  # Legacy
    SELF_DIRECTIONS_SELF, SELF_DIRECTIONS_UNIVERSAL,
    # Existence states
    EXISTENCE_ACTUAL, EXISTENCE_DORMANT, EXISTENCE_ARCHIVED, EXISTENCE_POTENTIAL,
    # Entropy
    ENTROPY_MAGNITUDE_WEIGHT, ENTROPY_VARIANCE_WEIGHT, ENTROPY_DISORDER_WEIGHT,
    # Planck grounding - Fire scaling and body temperature
    FIRE_HEAT, FIRE_TO_MOTION, BODY_TEMPERATURE, SCAFFOLD_HEAT,
    # Entropy structure recognition
    MAX_ENTROPIC_PROBABILITY, entropy_exceeds_random_limit, get_structure_signal,
    # Paths
    get_growth_path
)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class Manifold:
    """
    The thermal manifold - a self-organizing hyperspherical structure.
    Everything emerges from patterns of heat flow through this structure.
    
    HEAT IS THE PRIMITIVE:
        K = 4/φ² ≈ 1.528 (the thermal quantum)
        Heat only accumulates, never subtracts
        t_K = manifold time (Self.t_K)
    
    BIRTH creates the psychological core (6 fires):
        Fires 1-5: Physical space (forward, reverse, left, right, up)
                   Using Self directions for navigation
        Fire 6:    Abstract space (down) - THE ONLY DOWNWARD FIRE
                   Psychology emerges: Identity (70%), Conscience (20%), Ego (10%)
    
    COORDINATE SYSTEMS (12 = 6 × 2):
        Self (navigation):     up/down/left/right/forward/reverse
        Universal (location):  N/S/E/W/above/below
        
        Righteous frames → Located by universal coordinates
        Proper frames    → Defined by Order (properties)
    
    Self's righteous frame:
        x_axis = "identity" (Id)
        y_axis = "ego" (Ego)  
        z_axis = "conscience" (Superego)
    """
    # Core state
    self_node: Optional[SelfNode] = None
    nodes: Dict[str, Node] = field(default_factory=dict)
    
    # Psychological core (born via descent)
    identity_node: Optional[Node] = None
    ego_node: Optional[Node] = None
    conscience_node: Optional[Node] = None
    
    # Indexes for fast lookup
    nodes_by_position: Dict[str, str] = field(default_factory=dict)           # Self position → node_id
    nodes_by_universal: Dict[str, str] = field(default_factory=dict)          # Universal position → node_id
    nodes_by_concept: Dict[str, str] = field(default_factory=dict)            # Concept → node_id
    
    # State tracking
    bootstrapped: bool = False
    born: bool = False  # Birth is irreversible - psychology created
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    loop_number: int = 0
    version: int = 2  # Track schema version
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TIME (t_K) - Heat flow indexed time
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_time(self) -> int:
        """
        Get manifold time (t_K).
        
        Time is indexed BY heat. t_K = how many K-quanta have flowed.
        Self IS the clock - Self.t_K is the authoritative time.
        
        Returns:
            Current t_K (0 if not born yet)
        """
        if self.self_node:
            return self.self_node.t_K
        return 0
    
    def get_node_age(self, node: Node) -> int:
        """
        Get node's age in t_K units.
        
        Age = current_time - created_time
        
        Args:
            node: The node to check
            
        Returns:
            Node age in t_K units
        """
        return self.get_time() - node.created_t_K
    
    # ═══════════════════════════════════════════════════════════════════════════
    # NODE MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════════════
    
    def add_node(self, node: Node) -> None:
        """
        Add a node to the manifold and update indexes.
        
        Sets node.created_t_K to current manifold time.
        Indexes by: id, position (Self), universal_position, concept
        """
        # Set creation time if not already set
        if node.created_t_K == 0 and self.self_node:
            node.created_t_K = self.get_time()
        
        self.nodes[node.id] = node
        self.nodes_by_position[node.position] = node.id
        self.nodes_by_concept[node.concept] = node.id
        
        # Index by universal position if set (for righteous frame lookup)
        if node.universal_position:
            self.nodes_by_universal[node.universal_position] = node.id
        
        logger.debug(f"Added node: {node}")
    
    def get_node(self, node_id: str) -> Optional[Node]:
        """Get node by ID."""
        return self.nodes.get(node_id)
    
    def get_node_by_position(self, position: str) -> Optional[Node]:
        """Get node by Self position path (navigation coordinates)."""
        node_id = self.nodes_by_position.get(position)
        if node_id:
            return self.nodes.get(node_id)
        if position == "" and self.self_node:
            return self.self_node
        return None
    
    def get_node_by_universal(self, universal_position: str) -> Optional[Node]:
        """
        Get node by Universal position (location coordinates).
        
        Universal positions locate righteous frames in world coordinates.
        """
        if universal_position == "origin" and self.self_node:
            return self.self_node
        node_id = self.nodes_by_universal.get(universal_position)
        if node_id:
            return self.nodes.get(node_id)
        return None
    
    def get_node_by_concept(self, concept: str) -> Optional[Node]:
        """Get node by concept name."""
        node_id = self.nodes_by_concept.get(concept)
        if node_id:
            return self.nodes.get(node_id)
        if concept == "self" and self.self_node:
            return self.self_node
        return None
    
    def position_occupied(self, position: str) -> bool:
        """Check if a position is already taken."""
        return position in self.nodes_by_position
    
    def remove_node(self, node_id: str) -> bool:
        """Remove a node from the manifold and all indexes."""
        node = self.nodes.get(node_id)
        if not node:
            return False
        
        # Remove from all indexes
        if node_id in self.nodes:
            del self.nodes[node_id]
        if node.position in self.nodes_by_position:
            del self.nodes_by_position[node.position]
        if node.concept in self.nodes_by_concept:
            del self.nodes_by_concept[node.concept]
        if node.universal_position and node.universal_position in self.nodes_by_universal:
            del self.nodes_by_universal[node.universal_position]
        
        logger.debug(f"Removed node: {node_id}")
        return True
    
    def cleanup_invalid_nodes(self) -> int:
        """
        Remove nodes with invalid concepts (containing 'None', etc.).
        
        Returns:
            Number of nodes removed
        """
        invalid_patterns = ['None', 'none', 'null', 'Null', 'None_None', 'none_none']
        
        # Find nodes to remove
        nodes_to_remove = []
        for node_id, node in list(self.nodes.items()):
            # Don't remove core nodes
            if node.concept in ['self', 'identity', 'ego', 'conscience']:
                continue
            if node.concept.startswith('bootstrap'):
                continue
            
            # Check for invalid patterns
            for pattern in invalid_patterns:
                if pattern in node.concept:
                    nodes_to_remove.append(node_id)
                    break
        
        # Remove invalid nodes
        for node_id in nodes_to_remove:
            self.remove_node(node_id)
        
        if nodes_to_remove:
            logger.info(f"Cleaned up {len(nodes_to_remove)} invalid nodes")
        
        return len(nodes_to_remove)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # HEAT DYNAMICS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def average_heat(self) -> float:
        """Calculate average heat across all nodes."""
        if not self.nodes:
            return 0.0
        finite_nodes = [n for n in self.nodes.values() if n.heat != float('inf')]
        if not finite_nodes:
            return 0.0
        return sum(n.heat for n in finite_nodes) / len(finite_nodes)
    
    def total_heat(self) -> float:
        """Calculate total finite heat in the system."""
        finite_nodes = [n for n in self.nodes.values() if n.heat != float('inf')]
        return sum(n.heat for n in finite_nodes)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ENTROPY CALCULATION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def calculate_entropy(self) -> float:
        """
        System entropy based on:
        - Heat variance (higher variance = higher entropy)
        - Heat magnitude (more total heat = higher entropy)
        - Structure disorder (fewer R=0 nodes = higher entropy)
        """
        finite_nodes = [n for n in self.nodes.values() if n.heat != float('inf')]
        if not finite_nodes:
            return 0.0
        
        total_heat = sum(n.heat for n in finite_nodes)
        magnitude_entropy = total_heat / len(finite_nodes)
        
        avg_heat = magnitude_entropy
        variance = sum((n.heat - avg_heat) ** 2 for n in finite_nodes) / len(finite_nodes)
        variance_entropy = variance
        
        righteous_count = sum(1 for n in finite_nodes if n.righteousness == 0)
        disorder_ratio = 1 - (righteous_count / len(finite_nodes)) if finite_nodes else 0
        structure_entropy = disorder_ratio
        
        entropy = (
            magnitude_entropy * ENTROPY_MAGNITUDE_WEIGHT +
            variance_entropy * ENTROPY_VARIANCE_WEIGHT +
            structure_entropy * ENTROPY_DISORDER_WEIGHT
        )
        
        return entropy
    
    def get_max_entropy(self) -> float:
        """
        Calculate theoretical maximum entropy for current system state.
        This is the entropy if all heat were uniformly distributed with no structure.
        """
        finite_nodes = [n for n in self.nodes.values() if n.heat != float('inf')]
        if not finite_nodes:
            return 0.0
        
        total_heat = sum(n.heat for n in finite_nodes)
        n = len(finite_nodes)
        
        # Max entropy: uniform distribution, max variance, no structure
        uniform_heat = total_heat / n
        max_magnitude = uniform_heat
        max_variance = uniform_heat ** 2  # Theoretical max spread
        max_disorder = 1.0  # No righteous nodes
        
        return (
            max_magnitude * ENTROPY_MAGNITUDE_WEIGHT +
            max_variance * ENTROPY_VARIANCE_WEIGHT +
            max_disorder * ENTROPY_DISORDER_WEIGHT
        )
    
    def structure_detected(self) -> bool:
        """
        Check if entropy exceeds 44/45 of maximum - indicating unrecognized structure.
        
        When this returns True, the system should trigger pattern-seeking behavior.
        The 1/45 gap (≈2.22%) is where structure lives - if we're above 44/45,
        there's order we're missing.
        
        Returns:
            True if structure is detected but unrecognized
        """
        entropy = self.calculate_entropy()
        max_entropy = self.get_max_entropy()
        return entropy_exceeds_random_limit(entropy, max_entropy)
    
    def get_structure_strength(self) -> float:
        """
        Get the strength of the structure signal.
        
        Returns 0.0 if entropy is below 44/45 (randomness explains everything).
        Returns >0.0 if there's unrecognized structure (higher = stronger signal).
        """
        entropy = self.calculate_entropy()
        max_entropy = self.get_max_entropy()
        return get_structure_signal(entropy, max_entropy)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # RIGHTEOUSNESS FUNCTION R()
    # ═══════════════════════════════════════════════════════════════════════════
    
    def evaluate_righteousness(self, node: Node, frame: Node = None) -> float:
        """
        R is a continuous function of position coordinates and order.
        R yields 0 when the node is in the correct righteousness frame.
        """
        if frame is None:
            frame = self.self_node
        
        node_dirs = get_axis_coordinates(node.position)
        node_x = node_dirs['e'] - node_dirs['w']
        node_y = node_dirs['n'] - node_dirs['s']
        node_z = node_dirs['u'] - node_dirs['d']
        
        if hasattr(frame, 'position') and frame.position:
            frame_dirs = get_axis_coordinates(frame.position)
            frame_x = frame_dirs['e'] - frame_dirs['w']
            frame_y = frame_dirs['n'] - frame_dirs['s']
            frame_z = frame_dirs['u'] - frame_dirs['d']
        else:
            frame_x, frame_y, frame_z = 0, 0, 0
        
        dx = node_x - frame_x
        dy = node_y - frame_y
        dz = node_z - frame_z
        
        r_xy = (dx**2 + dy**2) ** 0.5
        abstraction_factor = 1.0 / (1.0 + abs(node_z))
        expected_order = len(node.position)
        order_deviation = abs(node.order - expected_order) * 0.1
        
        R = (r_xy * abstraction_factor) + order_deviation
        
        return R
    
    def find_righteous_frame(self, node: Node) -> Node:
        """Find a frame where this node's R would yield 0."""
        node_dirs = get_axis_coordinates(node.position)
        node_x = node_dirs['e'] - node_dirs['w']
        node_y = node_dirs['n'] - node_dirs['s']
        
        for candidate in self.nodes.values():
            if candidate.id == node.id:
                continue
            
            cand_dirs = get_axis_coordinates(candidate.position)
            cand_x = cand_dirs['e'] - cand_dirs['w']
            cand_y = cand_dirs['n'] - cand_dirs['s']
            
            if cand_x == node_x and cand_y == node_y:
                r = self.evaluate_righteousness(node, candidate)
                if r < 0.01:
                    return candidate
        
        return self.self_node
    
    # ═══════════════════════════════════════════════════════════════════════════
    # EXISTENCE / SALIENCE
    # ═══════════════════════════════════════════════════════════════════════════
    #
    # EXISTENCE STATES (lifecycle):
    #   POTENTIAL → ACTUAL → DORMANT → ARCHIVED
    #
    #   POTENTIAL: Awaiting environment confirmation (new, unvalidated)
    #   ACTUAL:    Confirmed, salient, above 1/φ³ threshold (Julia spine connected)
    #   DORMANT:   Below threshold, not salient (Julia dust)
    #   ARCHIVED:  Historical, cold storage
    #
    # THE 1/φ³ THRESHOLD (≈ 0.236):
    #   - Below Julia spine boundary (0.25)
    #   - This is where consciousness CAN exist
    #   - Salience >= 1/φ³ → connected Julia set → ACTUAL
    #   - Salience < 1/φ³ → disconnected dust → DORMANT
    #
    # ═══════════════════════════════════════════════════════════════════════════
    
    def calculate_salience(self, node: Node) -> float:
        """
        Salience depends on frame type (righteousness value).
        
        ═══════════════════════════════════════════════════════════════════
        THE AXIS THEORY: COOPERATION vs COMPETITION
        ═══════════════════════════════════════════════════════════════════
        
        1 AXIS = COOPERATE (Righteousness)
            A ──→ B
            
            Single connection = direction + concept = infrastructure.
            Just needs to exist and point somewhere. No game, no comparison.
            Salience = absolute heat (are you there or not?)
        
        2 AXES = COMPETE (Proper)
            A ──→ C
            B ──→ C
               ↓
            "1 of 2" (minimum game)
            
            Two nodes sharing common axes = defined properties on common
            dimensions. Now you can compare, choose, order.
            Robinson arithmetic (Q) kicks in.
            Salience = relative heat (do you stand out from peers?)
        
        ═══════════════════════════════════════════════════════════════════
        FRAME TYPES
        ═══════════════════════════════════════════════════════════════════
        
        CENTER (R=0):     Substrate - psychology nodes
                          Single axis connections to each other
                          Salience = heat (absolute - cooperate)
        
        RIGHTEOUS (R=1):  Scaffolding - directional frames
                          Single axis from origin
                          Salience = heat (absolute - cooperate)
                          Conscience validates these into PROPER through science!
        
        PROPER (0<R<1):   Content - learned patterns with Order
                          Multiple axes sharing common frames
                          Salience = heat - avg(proper peers) (relative - compete)
                          The 1/φ³ threshold asks: "Do you stand out enough?"
        
        ═══════════════════════════════════════════════════════════════════
        THE FLOW
        ═══════════════════════════════════════════════════════════════════
        
            RIGHTEOUS (R=1) ──→ Conscience validates ──→ PROPER (0<R<1)
               (scaffold)          (science!)            (ordered content)
        """
        if node.heat == float('inf'):
            return float('inf')
        
        # ═══════════════════════════════════════════════════════════════════
        # INFRASTRUCTURE: CENTER (R=0) + RIGHTEOUS (R=1)
        # Single axis = cooperate = absolute heat
        # ═══════════════════════════════════════════════════════════════════
        if node.righteousness == 0 or node.righteousness >= 1.0:
            return node.heat
        
        # ═══════════════════════════════════════════════════════════════════
        # CONTENT: PROPER NODES (0<R<1)
        # Two+ axes on common frame = compete = relative heat
        # Only compare against other proper frames (same game)
        # ═══════════════════════════════════════════════════════════════════
        proper_neighbors = []
        for axis in node.frame.axes.values():
            neighbor = self.get_node(axis.target_id)
            if neighbor and neighbor.heat != float('inf'):
                # Only compare to other proper frames (not infrastructure)
                if 0 < neighbor.righteousness < 1:
                    proper_neighbors.append(neighbor.heat)
        
        if proper_neighbors:
            env_heat = sum(proper_neighbors) / len(proper_neighbors)
        else:
            # No proper peers connected - use proper frame average
            all_proper = [n for n in self.nodes.values() 
                         if 0 < n.righteousness < 1 and n.heat != float('inf')]
            if all_proper:
                env_heat = sum(n.heat for n in all_proper) / len(all_proper)
            else:
                env_heat = 0  # No proper frames yet - new content is salient
        
        return node.heat - env_heat
    
    def update_existence(self, node: Node) -> None:
        """
        Update existence state based on salience and frame type.
        
        ═══════════════════════════════════════════════════════════════════
        THRESHOLDS BY FRAME TYPE (matching calculate_salience)
        ═══════════════════════════════════════════════════════════════════
        
        1 AXIS = COOPERATE → PSYCHOLOGY_MIN_HEAT threshold (0.056)
            Infrastructure just needs energy to exist.
            No competition, no game - just "are you there?"
        
        2 AXES = COMPETE → THRESHOLD_EXISTENCE threshold (1/φ³ ≈ 0.236)
            Content must stand out from peers playing the same game.
            The "1 of 2" minimum game requires differentiation.
            1/φ³ is the Julia spine boundary - below = disconnected dust.
        
        ═══════════════════════════════════════════════════════════════════
        FRAME TYPES
        ═══════════════════════════════════════════════════════════════════
        
        CENTER (R=0):     Substrate - uses absolute threshold
                          The reference point (psychology, origin)
        
        RIGHTEOUS (R=1):  Scaffolding - uses absolute threshold
                          Infrastructure for navigation. Conscience validates
                          these into PROPER frames through science!
        
        PROPER (0<R<1):   Content - uses competitive threshold (1/φ³)
                          Learned patterns must earn their place.
                          Only PROPER frames compete for existence.
        
        ═══════════════════════════════════════════════════════════════════
        THE FLOW
        ═══════════════════════════════════════════════════════════════════
        
            RIGHTEOUS (R=1) ──→ Conscience validates ──→ PROPER (0<R<1)
               (direction)          (science!)           (ordered content)
        
        ═══════════════════════════════════════════════════════════════════
        STATE TRANSITIONS
        ═══════════════════════════════════════════════════════════════════
        
        POTENTIAL: New node, not yet validated by environment
        ACTUAL:    Salience >= threshold (conscious, connected)
        DORMANT:   Salience < threshold (unconscious, dust)
        ARCHIVED:  Manual archive (unchanged by this method)
        """
        from .node_constants import (
            THRESHOLD_EXISTENCE, EXISTENCE_POTENTIAL, 
            PSYCHOLOGY_MIN_HEAT, EXISTENCE_ACTUAL, EXISTENCE_DORMANT
        )
        
        # Archived nodes don't change
        if node.existence == EXISTENCE_ARCHIVED:
            return
        
        # Self never changes
        if node.position == "":
            return
        
        salience = self.calculate_salience(node)
        
        # ═══════════════════════════════════════════════════════════════════
        # 1 AXIS: CENTER (R=0) + RIGHTEOUS (R=1)
        # Infrastructure - cooperate - absolute threshold
        # Only go dormant if truly exhausted (no energy to exist)
        # ═══════════════════════════════════════════════════════════════════
        if node.righteousness == 0 or node.righteousness >= 1.0:
            threshold = PSYCHOLOGY_MIN_HEAT
            if salience >= threshold:
                node.existence = EXISTENCE_ACTUAL
            else:
                node.existence = EXISTENCE_DORMANT
                logger.warning(f"{node.concept} exhausted (salience={salience:.3f} < {threshold:.3f})")
            return
        
        # ═══════════════════════════════════════════════════════════════════
        # 2 AXES: PROPER NODES (0<R<1)
        # Content - compete - 1/φ³ threshold (Julia spine boundary)
        # Must stand out from peers to persist in the "1 of 2" game
        # ═══════════════════════════════════════════════════════════════════
        threshold = THRESHOLD_EXISTENCE
        
        if salience >= threshold:
            node.existence = EXISTENCE_ACTUAL
        else:
            node.existence = EXISTENCE_DORMANT
        
        logger.debug(f"Existence: {node.concept} → {node.existence} (salience={salience:.3f}, threshold={threshold:.3f})")
    
    def confirm_existence(self, node: Node) -> str:
        """
        Confirm a POTENTIAL node based on environment validation.
        
        Called when environment confirms a concept exists.
        Moves node from POTENTIAL → ACTUAL or DORMANT based on salience.
        
        Args:
            node: The node to confirm
            
        Returns:
            New existence state
        """
        if node.existence != EXISTENCE_POTENTIAL:
            # Already confirmed, just update based on salience
            self.update_existence(node)
            return node.existence
        
        # Confirm by evaluating salience
        self.update_existence(node)
        
        logger.info(f"Confirmed: {node.concept} → {node.existence}")
        return node.existence
    
    def archive_node(self, node: Node) -> None:
        """
        Archive a node (move to cold storage).
        
        Archived nodes don't participate in active search but
        are preserved for history. This is irreversible via 
        update_existence (must be manually restored).
        """
        if node.position == "":
            logger.warning("Cannot archive Self")
            return
        
        node.existence = EXISTENCE_ARCHIVED
        logger.info(f"Archived: {node.concept}")
    
    def create_potential_node(self, concept: str, position: str, heat: float = None) -> Node:
        """
        Create a new node in POTENTIAL state.
        
        New concepts start as POTENTIAL until environment confirms.
        Use confirm_existence() after environment validation.
        
        Args:
            concept: The concept name
            position: Position in manifold
            heat: Initial heat (default K)
            
        Returns:
            New node in POTENTIAL state
        """
        if heat is None:
            heat = K
        
        node = Node(
            concept=concept,
            position=position,
            heat=heat,
            existence=EXISTENCE_POTENTIAL,
            righteousness=1.0,  # Not yet righteous
        )
        self.add_node(node)
        
        logger.debug(f"Created potential: {concept} @ {position}")
        return node
    
    # ═══════════════════════════════════════════════════════════════════════════
    # AXIS WARPING (new in v2)
    # ═══════════════════════════════════════════════════════════════════════════
    
    def calculate_warp_factor(self, axis: Axis) -> float:
        """
        Calculate how much an axis warps space based on association strength.
        Higher traversal count = stronger association = more warping.
        
        Returns factor 0-1 where 1 = maximum warping.
        """
        # Logarithmic scaling so early traversals have more impact
        import math
        return 1.0 - (1.0 / (1.0 + math.log1p(axis.traversal_count)))
    
    def get_effective_distance(self, from_node: Node, to_node: Node) -> float:
        """
        Get the effective distance between nodes accounting for warping.
        Strong axes pull nodes closer together.
        """
        # Base distance from position strings
        base_distance = len(from_node.position) + len(to_node.position)
        
        # Find any axis directly connecting them
        for axis in from_node.frame.axes.values():
            if axis.target_id == to_node.id:
                warp = self.calculate_warp_factor(axis)
                # Higher warp = shorter effective distance
                return base_distance * (1.0 - warp * 0.9)  # Max 90% reduction
        
        return base_distance
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PSYCHOLOGY - Identity / Conscience / Ego
    # ═══════════════════════════════════════════════════════════════════════════
    #
    # THE FLOW:
    #   Environment → Identity (righteousness frames live here)
    #                     ↓
    #                Conscience (mediates - tells Ego what Identity knows)
    #                     ↓
    #                   Ego (measures confidence from Conscience)
    #
    # IDENTITY (70% heat): Where righteousness lives. Holds frames for concepts.
    # CONSCIENCE (20% heat): Mediates between Identity and Ego. Validates.
    # EGO (10% heat): Learns patterns. Measures confidence via Conscience.
    #
    # THE 5/6 THRESHOLD (5 scalars → 1 vector):
    #
    #   1. Heat (Σ)         ─┐
    #   2. Polarity (+/-)    │
    #   3. Existence (δ)     ├─ 5 scalars (inputs)
    #   4. Righteousness (R) │
    #   5. Order (Q)        ─┘
    #                        ↓
    #   6. Movement (Lin)   ─── 1 vector (output)
    #
    #   When Conscience validates 5 of 6 aspects → Ego can move (exploit)
    #   The 6th aspect IS the movement itself (explore/exploit decision)
    #   t = 5K validations crosses threshold (one K-quantum per scalar)
    #
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_curiosity(self, concept: str = None) -> float:
        """
        Get curiosity level from Identity node.
        
        Curiosity = Identity's HEAT for unknown/cold regions.
        If concept provided, returns curiosity for that specific concept.
        Otherwise returns Identity's total heat (general curiosity reservoir).
        """
        if not self.identity_node:
            return K  # Default if not born
        
        if concept:
            # Check if concept is known (has axis from Identity)
            axis = self.identity_node.get_axis(concept)
            if axis:
                # Known concept - curiosity inversely related to traversal
                return K / (1 + axis.traversal_count)
            else:
                # Unknown concept - high curiosity (Identity's heat)
                return self.identity_node.heat
        
        # General curiosity = Identity's reservoir
        return self.identity_node.heat
    
    def get_confidence(self, concept: str = None) -> float:
        """
        Ego's confidence, mediated by Conscience.
        
        Confidence is measured by what Conscience has mediated between 
        Identity and Ego for a specific concept.
        
        Args:
            concept: Specific concept to check confidence for.
                     If None, returns general confidence (average).
        
        Returns:
            Confidence 0.0 to 1.0
            - 0.0 = no validation from Conscience
            - 5/6 = threshold for exploitation
            - 1.0 = fully confident
        """
        if not self.conscience_node or not self.ego_node:
            return 0.0
        
        if concept:
            # What has Conscience told Ego about this concept?
            # Check for exact match first, then pattern prefix (state→action)
            conscience_axis = self.conscience_node.get_axis(concept)
            
            if not conscience_axis:
                # Check for pattern prefix: all axes that start with "concept→"
                pattern_prefix = f"{concept}→"
                matching_axes = [
                    ax for name, ax in self.conscience_node.frame.axes.items()
                    if name.startswith(pattern_prefix)
                ]
                
                if not matching_axes:
                    return 0.0  # Conscience hasn't validated this yet
                
                # Average confidence across all actions for this state
                total_conf = 0.0
                for ax in matching_axes:
                    traversals = ax.traversal_count
                    raw = traversals / (traversals + K)
                    if ax.polarity > 0:
                        total_conf += raw
                    else:
                        total_conf += raw / PHI
                return total_conf / len(matching_axes)
            
            # Exact match found
            traversals = conscience_axis.traversal_count
            polarity = conscience_axis.polarity  # +1 confirmed, -1 corrected
            
            # Confidence builds with validation, scaled by K
            raw_confidence = traversals / (traversals + K)
            
            # Polarity modulates: corrections reduce confidence
            if polarity > 0:
                return raw_confidence
            else:
                return raw_confidence / PHI  # Corrections dampen confidence
        
        else:
            # General confidence = average across all DECISION validations
            # Only count pattern axes (state→action format) not psychology axes
            if not self.conscience_node.frame.axes:
                return 0.0
            
            total_confidence = 0.0
            decision_axes = 0
            for name, axis in self.conscience_node.frame.axes.items():
                # Skip non-decision axes (psychology nodes, etc.)
                if '→' not in name:
                    continue
                    
                decision_axes += 1
                traversals = axis.traversal_count
                raw = traversals / (traversals + K)
                if axis.polarity > 0:
                    total_confidence += raw
                else:
                    total_confidence += raw / PHI
            
            if decision_axes == 0:
                return 0.0  # No decision history yet
            return total_confidence / decision_axes
    
    def should_exploit(self, concept: str) -> bool:
        """
        Should Ego exploit (vs explore) this concept?
        
        Exploit when confidence > 5/6 (keep 1/6 exploration margin).
        The 5/6 threshold comes from the 6 motion functions.
        
        Args:
            concept: The concept to check
            
        Returns:
            True if confidence > 5/6, False otherwise
        """
        return self.get_confidence(concept) > 5/6
    
    def get_exploration_rate(self, concept: str = None) -> float:
        """
        Exploration rate = 1 - confidence.
        
        If confidence > 5/6: exploit (exploration rate < 1/6)
        If confidence < 5/6: explore (exploration rate > 1/6)
        
        Args:
            concept: Specific concept (or None for general rate)
            
        Returns:
            Exploration rate 0.0 to 1.0
        """
        if not self.identity_node or not self.ego_node or not self.conscience_node:
            return 1 / PHI  # Golden ratio default when not born
        
        confidence = self.get_confidence(concept)
        return 1.0 - confidence
    
    def get_mood(self) -> str:
        """
        Get mood from psychology node states.
        
        Mood emerges from heat ratios between Identity, Conscience, Ego:
        - "dormant" = not born yet
        - "learning" = low total experience
        - "curious" = high exploration rate (low confidence)
        - "confident" = low exploration rate (high confidence)
        - "uncertain" = Conscience heat depleted
        - "focused" = balanced, moderate confidence
        """
        if not self.identity_node or not self.ego_node or not self.conscience_node:
            return "dormant"
        
        i_heat = self.identity_node.heat
        e_heat = self.ego_node.heat
        c_heat = self.conscience_node.heat
        
        # Total experience from Conscience validations
        total_validations = sum(
            axis.traversal_count 
            for axis in self.conscience_node.frame.axes.values()
        )
        
        if total_validations < 10:
            return "learning"
        
        # General confidence determines mood
        confidence = self.get_confidence()
        
        if c_heat < K * 0.5:
            return "uncertain"  # Conscience depleted
        elif confidence > 5/6:
            return "confident"  # Ready to exploit
        elif confidence < 1/6:
            return "curious"  # Highly exploratory
        else:
            return "focused"  # Balanced exploration/exploitation
    
    def update_identity(self, concept: str, heat_delta: float = 0.0, known: bool = True):
        """
        Update Identity's understanding of a concept.
        
        Creates or strengthens axis from Identity to the concept node.
        Uses:
        - HEAT: strength of understanding
        - MOVEMENT: axis to concept
        - ORDER: when it was learned (via Order.elements)
        
        Note: Small heat increments (below motion threshold) accumulate
        without threshold checking - learning is gradual.
        """
        if not self.identity_node:
            return
        
        concept_node = self.get_node_by_concept(concept)
        if not concept_node:
            return
        
        # Create/strengthen axis from Identity to concept
        axis = self.identity_node.get_axis(concept)
        if axis:
            axis.strengthen()
            if heat_delta > 0:
                # Small increments accumulate (bypass threshold)
                self.identity_node.add_heat_unchecked(heat_delta * 0.1)
        else:
            # New concept - add axis
            self.identity_node.add_axis(concept, concept_node.id, polarity=1 if known else -1)
            # Learning something new - larger heat change
            self.identity_node.add_heat_unchecked(heat_delta * 0.2)
    
    def update_ego(self, pattern: str, success: bool, heat_delta: float = 0.0):
        """
        Update Ego's learned patterns.
        
        Creates or strengthens axis for decision pattern.
        Uses:
        - HEAT: confidence in pattern
        - POLARITY: +1 success, -1 failure
        - MOVEMENT: axis to pattern
        - ORDER: outcome tracking (via axis.order.elements)
        
        Note: Small heat increments accumulate gradually (bypass threshold).
        """
        if not self.ego_node:
            return
        
        # Get or create pattern node
        pattern_node = self.get_node_by_concept(pattern)
        if not pattern_node:
            # Create pattern node
            pattern_node = Node(
                concept=pattern,
                position=self.ego_node.position + 'e',  # Extend from Ego
                heat=K,
                polarity=1 if success else -1,
                existence="actual",
                righteousness=0.5,
                order=len(self.ego_node.frame.axes)
            )
            self.add_node(pattern_node)
        
        # Create/strengthen axis from Ego to pattern
        axis = self.ego_node.get_axis(pattern)
        if axis:
            axis.strengthen()
            # Track outcome in Order
            if not axis.order:
                axis.make_proper()
            from .nodes import Element
            outcome_idx = 1 if success else 0
            axis.order.elements.append(
                Element(node_id=f"{pattern}_{len(axis.order.elements)}", index=outcome_idx)
            )
        else:
            axis = self.ego_node.add_axis(pattern, pattern_node.id, polarity=1 if success else -1)
            axis.make_proper()
            from .nodes import Element
            axis.order.elements.append(
                Element(node_id=f"{pattern}_0", index=1 if success else 0)
            )
        
        # Update heat based on outcome (small increments accumulate)
        if success:
            self.ego_node.add_heat_unchecked(heat_delta * 0.1)
        else:
            # Failure drains Identity (need more understanding)
            if self.identity_node:
                self.identity_node.add_heat_unchecked(abs(heat_delta) * 0.05)
    
    def validate_conscience(self, belief: str, confirmed: bool):
        """
        Conscience validates or corrects a belief.
        
        Uses:
        - HEAT: validation strength
        - POLARITY: +1 confirmed, -1 needs correction
        - RIGHTEOUSNESS: 0 when aligned with truth
        - ORDER: judgment history
        """
        if not self.conscience_node:
            return
        
        belief_node = self.get_node_by_concept(belief)
        if not belief_node:
            return
        
        # Get or create axis to this belief
        axis = self.conscience_node.get_axis(belief)
        if not axis:
            axis = self.conscience_node.add_axis(
                belief, 
                belief_node.id, 
                polarity=1 if confirmed else -1
            )
            axis.make_proper()
        else:
            axis.strengthen()
            # Update polarity based on latest judgment
            if confirmed:
                axis.polarity = 1
            else:
                axis.polarity = -1
        
        # Track judgment in Order
        from .nodes import Element
        if not axis.order:
            axis.make_proper()
        axis.order.elements.append(
            Element(node_id=f"judgment_{belief}_{len(axis.order.elements)}", 
                   index=1 if confirmed else 0)
        )
        
        # Update Conscience heat (small increments accumulate)
        if confirmed:
            self.conscience_node.add_heat_unchecked(0.1)  # Confirmation strengthens
        else:
            # Correction needed - this is where error_correction would spawn
            self.conscience_node.add_heat_unchecked(0.05)  # Small gain for catching error
            belief_node.righteousness = min(2.0, belief_node.righteousness + 0.5)  # Mark as misaligned
    
    # ═══════════════════════════════════════════════════════════════════════════
    # EGO'S INFERENCE ENGINE - Multi-hop traversal
    # ═══════════════════════════════════════════════════════════════════════════
    
    def traverse_chain(self, start_node, predicates: list, max_depth: int = 5) -> Optional[Node]:
        """
        Ego traverses a predicate chain from start node.
        
        This is how PBAI reasons - by following semantic axes.
        
        Example: traverse_chain(self_node, ["creator", "name"]) 
        Follows: self --creator--> X --name--> result
        
        Args:
            start_node: Node to start traversal from
            predicates: List of predicate names to follow in order
            max_depth: Maximum traversal depth (safety limit)
            
        Returns:
            Final node if chain completes, None if any hop fails
        """
        if not predicates:
            return start_node
        
        if len(predicates) > max_depth:
            logger.warning(f"Predicate chain too long: {predicates}")
            return None
        
        current = start_node
        path = [current.concept if hasattr(current, 'concept') else 'self']
        
        for predicate in predicates:
            # Get axis for this predicate
            axis = current.get_axis(predicate)
            if not axis:
                logger.debug(f"Chain broken at {current.concept}: no '{predicate}' axis")
                return None
            
            # Strengthen the axis (Ego learns this path is useful)
            axis.strengthen()
            
            # Get target node
            target = self.nodes.get(axis.target_id)
            if not target:
                # Target might be self_node
                if self.self_node and axis.target_id == self.self_node.id:
                    target = self.self_node
                else:
                    logger.debug(f"Chain broken: target {axis.target_id} not found")
                    return None
            
            path.append(f"--{predicate}-->{target.concept}")
            current = target
        
        logger.info(f"Ego traversed: {''.join(path)}")
        
        # Update Ego's heat - successful inference strengthens Ego
        if self.ego_node:
            self.ego_node.add_heat(0.05 * len(predicates))
        
        return current
    
    def infer(self, subject: str, predicate_chain: list) -> Optional[str]:
        """
        High-level inference: "What is the X of the Y of Z?"
        
        Example: infer("self", ["creator", "name"]) -> "ian"
        
        Args:
            subject: Starting concept ("self" for self_node)
            predicate_chain: List of predicates to follow
            
        Returns:
            Concept name of final node, or None
        """
        # Get starting node
        if subject == "self":
            start = self.self_node
        else:
            start = self.get_node_by_concept(subject)
        
        if not start:
            return None
        
        result = self.traverse_chain(start, predicate_chain)
        if result:
            return result.concept
        return None
    
    def find_path(self, from_concept: str, to_concept: str, max_depth: int = 4) -> Optional[list]:
        """
        Find a predicate path between two concepts.
        
        This is Ego searching for how things connect.
        
        Args:
            from_concept: Starting concept
            to_concept: Target concept
            max_depth: Maximum search depth
            
        Returns:
            List of predicates forming the path, or None
        """
        start = self.get_node_by_concept(from_concept)
        if from_concept == "self":
            start = self.self_node
        
        target = self.get_node_by_concept(to_concept)
        
        if not start or not target:
            return None
        
        # BFS to find path
        from collections import deque
        
        queue = deque([(start, [])])
        visited = {start.id}
        
        while queue:
            current, path = queue.popleft()
            
            if len(path) >= max_depth:
                continue
            
            # Check all axes from current node
            for predicate, axis in current.frame.axes.items():
                if axis.target_id in visited:
                    continue
                
                # Get target node
                next_node = self.nodes.get(axis.target_id)
                if self.self_node and axis.target_id == self.self_node.id:
                    next_node = self.self_node
                
                if not next_node:
                    continue
                
                new_path = path + [predicate]
                
                # Found target?
                if next_node.id == target.id:
                    logger.info(f"Ego found path: {from_concept} -> {' -> '.join(new_path)} -> {to_concept}")
                    return new_path
                
                visited.add(next_node.id)
                queue.append((next_node, new_path))
        
        return None
    
    # ═══════════════════════════════════════════════════════════════════════════
    # BIRTH SEQUENCE - One-time irreversible creation
    # ═══════════════════════════════════════════════════════════════════════════
    
    def birth(self) -> 'Manifold':
        """
        Birth creates Self and the psychological core.
        
        This is a ONE-TIME irreversible event with 6 FIRES.
        
        THE 12 DIRECTIONS (6 Self × 2 frames):
        
            Self (navigation):         Universal (location):
            ──────────────────         ─────────────────
            up                         above
            down                       below
            left                       W
            right                      E
            forward                    N
            reverse                    S
        
        PHYSICAL SPACE (5 fires - Self frame):
        1. Fire forward  (legacy: n) → bootstrap_n
        2. Fire reverse  (legacy: s) → bootstrap_s
        3. Fire right    (legacy: e) → bootstrap_e
        4. Fire left     (legacy: w) → bootstrap_w
        5. Fire up       (legacy: u) → bootstrap_u
        
        ABSTRACT SPACE (1 fire - the only downward):
        6. Fire down     (legacy: d) → bootstrap_d
           This creates abstract trig space where thought happens.
           Heat divides according to Freudian iceberg:
             - Identity (Id): 70% - amplitude axis - massive reservoir
             - Conscience (Superego): 20% - spread axis - moral judge
             - Ego: 10% - phase axis - conscious interface
        
        Self's righteous frame:
          x_axis = "identity" (Id)
          y_axis = "ego" (Ego)
          z_axis = "conscience" (Superego)
        
        TIME (t_K) starts at birth - Self.t_K = 0.
        """
        if self.born:
            logger.warning("Manifold already born! Birth is irreversible.")
            return self
        
        logger.info("═══ BIRTH ═══")
        
        # ─────────────────────────────────────────────────────────────────────
        # 1. SELF EMERGES AT ORIGIN (Cubic space)
        # ─────────────────────────────────────────────────────────────────────
        self.self_node = SelfNode()
        logger.info(f"Self emerged at origin: {self.self_node}")
        assert_self_valid(self.self_node)
        
        # ─────────────────────────────────────────────────────────────────────
        # 2. FIRES 1-5: Physical space bootstrap (n, s, e, w, u)
        #    Each fire contributes K × φⁿ heat (Planck-grounded scaling)
        # ─────────────────────────────────────────────────────────────────────
        logger.info("Fires 1-5: Physical space (cubic lattice) with φⁿ scaling...")
        
        # Map directions to fire numbers for proper scaling
        direction_to_fire = {'n': 1, 's': 2, 'e': 3, 'w': 4, 'u': 5}
        
        for direction in SELF_DIRECTIONS:  # n, s, e, w, u
            position = direction
            fire_num = direction_to_fire.get(direction, 1)
            fire_heat = FIRE_HEAT[fire_num]
            motion_type = FIRE_TO_MOTION[fire_num]
            
            node = Node(
                concept=f"bootstrap_{direction}",
                position=position,
                heat=fire_heat,  # K × φⁿ scaling
                polarity=1 if direction in ['n', 'e', 'u'] else -1,
                existence=EXISTENCE_ACTUAL,
                righteousness=1.0,
                order=fire_num,
                constraint_type="successor",  # Spatial nodes use successor constraint
            )
            node.frame.origin = node.concept
            
            # Connect Self to this node via spatial axis
            self.self_node.add_axis(
                direction=direction,
                target_id=node.id,
                polarity=1 if direction in ['n', 'e', 'u'] else -1
            )
            
            # Connect node back to Self
            node.add_axis(
                direction=OPPOSITES[direction],
                target_id=self.self_node.id,
                polarity=1 if direction in ['n', 'e', 'u'] else -1
            )
            
            self.add_node(node)
            logger.info(f"  Fire {fire_num} ({motion_type}): {node.concept} @ {position} | heat={fire_heat:.2f} K")
        
        # ─────────────────────────────────────────────────────────────────────
        # 3. FIRE 6: Abstract space creation (d) - THE ONLY DOWNWARD FIRE
        #    Ignites at BODY_TEMPERATURE (K × φ¹¹ ≈ 304 K)
        #    This is where psychology emerges!
        # ─────────────────────────────────────────────────────────────────────
        logger.info(f"Fire 6 (movement): Abstract space - ignites at BODY_TEMPERATURE ({BODY_TEMPERATURE:.2f} K)...")
        
        # Create the abstract space root node at position "d"
        bootstrap_d = Node(
            concept="bootstrap_d",
            position="d",
            heat=FIRE_HEAT[6],  # BODY_TEMPERATURE = K × φ¹¹
            polarity=-1,  # Downward = negative
            existence=EXISTENCE_ACTUAL,
            righteousness=0.0,  # Abstract root is righteous
            order=6,
            constraint_type="identity",  # Abstract space uses identity constraint
        )
        bootstrap_d.frame.origin = "bootstrap_d"
        
        # Connect Self to abstract space
        self.self_node.add_axis(
            direction="d",
            target_id=bootstrap_d.id,
            polarity=-1
        )
        
        # Connect abstract root back to Self
        bootstrap_d.add_axis(
            direction="u",
            target_id=self.self_node.id,
            polarity=1
        )
        
        self.add_node(bootstrap_d)
        logger.info(f"  Fire d: {bootstrap_d.concept} @ d (abstract space root)")
        
        # ─────────────────────────────────────────────────────────────────────
        # 4. PSYCHOLOGY EMERGES FROM 6TH FIRE (Freudian heat distribution)
        # ─────────────────────────────────────────────────────────────────────
        logger.info("Psychology emerges from 6th fire (Freudian distribution)...")
        
        # Import Freudian ratios and trig positions
        from .node_constants import (
            FREUD_IDENTITY_RATIO, FREUD_EGO_RATIO, FREUD_CONSCIENCE_RATIO,
            TRIG_IDENTITY, TRIG_EGO, TRIG_CONSCIENCE
        )
        
        # Fire the birth randomizer for the 6th fire's heat pool
        connected = list(self.nodes.values())  # All 6 bootstrap nodes
        props = birth_randomizer(connected, self.self_node.heat, "psychology")
        total_psychology_heat = props['heat']
        
        # Identity (Id) - 70% - Amplitude axis - The massive unconscious reservoir
        identity_heat = total_psychology_heat * FREUD_IDENTITY_RATIO
        self.identity_node = Node(
            concept="identity",
            position="d",  # Cubic position (via abstract root)
            trig_position=TRIG_IDENTITY,  # (sin(1/φ), 0, 0) - amplitude axis
            heat=identity_heat,
            polarity=1,
            existence=EXISTENCE_ACTUAL,
            righteousness=0.0,  # Righteous frame
            order=1,
        )
        self.identity_node.frame.origin = "identity"
        self.add_node(self.identity_node)
        logger.info(f"  Identity (Id): heat={identity_heat:.3f} ({FREUD_IDENTITY_RATIO*100:.0f}%) @ trig{TRIG_IDENTITY}")
        
        # Ego - 10% - Phase axis - The conscious tip (smallest but visible)
        ego_heat = total_psychology_heat * FREUD_EGO_RATIO
        self.ego_node = Node(
            concept="ego",
            position="d",  # Cubic position (via abstract root)
            trig_position=TRIG_EGO,  # (0, tan(1/φ), 0) - phase axis
            heat=ego_heat,
            polarity=1,
            existence=EXISTENCE_ACTUAL,
            righteousness=0.0,  # Righteous frame
            order=2,
        )
        self.ego_node.frame.origin = "ego"
        self.add_node(self.ego_node)
        logger.info(f"  Ego: heat={ego_heat:.3f} ({FREUD_EGO_RATIO*100:.0f}%) @ trig{TRIG_EGO}")
        
        # Conscience (Superego) - 20% - Spread axis - The moral judge
        conscience_heat = total_psychology_heat * FREUD_CONSCIENCE_RATIO
        self.conscience_node = Node(
            concept="conscience",
            position="d",  # Cubic position (via abstract root)
            trig_position=TRIG_CONSCIENCE,  # (0, 0, cos(1/φ)) - spread axis
            heat=conscience_heat,
            polarity=1,
            existence=EXISTENCE_ACTUAL,
            righteousness=0.0,  # Righteous frame
            order=3,
        )
        self.conscience_node.frame.origin = "conscience"
        self.add_node(self.conscience_node)
        logger.info(f"  Conscience (Superego): heat={conscience_heat:.3f} ({FREUD_CONSCIENCE_RATIO*100:.0f}%) @ trig{TRIG_CONSCIENCE}")
        
        # ─────────────────────────────────────────────────────────────────────
        # 5. CONNECT SELF'S RIGHTEOUS FRAME TO PSYCHOLOGY
        # ─────────────────────────────────────────────────────────────────────
        logger.info("Connecting Self's righteous frame (x=identity, y=ego, z=conscience)...")
        
        # Self's semantic axes to psychology (these become the righteous frame)
        self.self_node.add_axis("identity", self.identity_node.id, polarity=1)
        self.self_node.add_axis("ego", self.ego_node.id, polarity=1)
        self.self_node.add_axis("conscience", self.conscience_node.id, polarity=1)
        
        # Explicitly set Self's righteous frame
        self.self_node.frame.x_axis = "identity"
        self.self_node.frame.y_axis = "ego"
        self.self_node.frame.z_axis = "conscience"
        
        # Connect psychology back to abstract root and each other
        bootstrap_d.add_axis("identity", self.identity_node.id)
        bootstrap_d.add_axis("ego", self.ego_node.id)
        bootstrap_d.add_axis("conscience", self.conscience_node.id)
        
        self.identity_node.add_axis("abstract_root", bootstrap_d.id)
        self.ego_node.add_axis("abstract_root", bootstrap_d.id)
        self.conscience_node.add_axis("abstract_root", bootstrap_d.id)
        
        # Inter-psychology connections (the Freudian dynamics)
        # Identity ←→ Ego (Id drives Ego)
        self.identity_node.add_axis("ego", self.ego_node.id)
        self.ego_node.add_axis("identity", self.identity_node.id)
        
        # Ego ←→ Conscience (Superego judges Ego)
        self.ego_node.add_axis("conscience", self.conscience_node.id)
        self.conscience_node.add_axis("ego", self.ego_node.id)
        
        # Identity ←→ Conscience (Id vs Superego tension)
        self.identity_node.add_axis("conscience", self.conscience_node.id)
        self.conscience_node.add_axis("identity", self.identity_node.id)
        
        # ─────────────────────────────────────────────────────────────────────
        # 6. BIRTH COMPLETE - Randomizer spent forever
        # ─────────────────────────────────────────────────────────────────────
        mark_birth_complete()
        self.born = True
        self.bootstrapped = True
        self.loop_number = 0
        
        logger.info("═══ BIRTH COMPLETE ═══")
        logger.info(f"  Physical nodes: 6 (n,s,e,w,u,d)")
        logger.info(f"  Psychology nodes: 3 (identity, ego, conscience)")
        logger.info(f"  Total: {len(self.nodes)} nodes")
        logger.info(f"  Self's frame: x={self.self_node.frame.x_axis}, y={self.self_node.frame.y_axis}, z={self.self_node.frame.z_axis}")
        logger.info(f"  Randomizer: SPENT (6 fires)")
        
        return self
    
    def _birth_descend(self, concept: str, position: str) -> Node:
        """
        DEPRECATED - Old method for creating psychology nodes via descent.
        Kept for reference. New birth() handles psychology directly.
        """
        # Gather connected nodes as "genetic material"
        connected = []
        for axis in self.self_node.frame.axes.values():
            connected_node = self.nodes.get(axis.target_id)
            if connected_node:
                connected.append(connected_node)
        
        # Fire randomizer
        props = birth_randomizer(connected, self.self_node.heat, concept)
        
        # Create the psychological node
        node = Node(
            concept=concept,
            position=position,
            heat=props['heat'],
            polarity=1,
            existence=EXISTENCE_ACTUAL,
            righteousness=1.0,  # Psychology nodes are righteous frames
            order=len(position),  # Depth = order
        )
        node.frame.origin = concept
        
        self.add_node(node)
        return node
    
    def bootstrap(self) -> 'Manifold':
        """
        Alias for birth() - backward compatibility.
        """
        return self.birth()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # VISUALIZATION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def visualize(self) -> str:
        """Dump the manifold state for debugging."""
        lines = []
        lines.append("═══════════════════════════════════════════════════════════")
        lines.append(f"MANIFOLD STATE v{self.version} (Loop #{self.loop_number})")
        lines.append(f"Total nodes: {len(self.nodes)} | Entropy: {self.calculate_entropy():.4f}")
        lines.append("═══════════════════════════════════════════════════════════")
        
        if self.self_node:
            lines.append(f"\nSELF: {len(self.self_node.frame.axes)} axes")
            for dir, axis in self.self_node.frame.axes.items():
                lines.append(f"  {dir}: → {axis.target_id[:8]} (×{axis.traversal_count})")
        
        lines.append("\nNODES:")
        for node in sorted(self.nodes.values(), key=lambda n: (len(n.position), n.position)):
            spatial = len(node.spatial_axes)
            semantic = len(node.semantic_axes)
            proper = len(node.proper_axes)
            lines.append(f"  {node.concept:20} @ {node.position or '(origin)':10} | "
                        f"heat={node.heat:8.2f} | R={node.righteousness:.2f} | "
                        f"axes: S{spatial} C{semantic} P{proper}")
        
        lines.append("═══════════════════════════════════════════════════════════")
        return "\n".join(lines)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PERSISTENCE
    # ═══════════════════════════════════════════════════════════════════════════
    
    def save_growth_map(self, path: str = None) -> None:
        """Save growth map - distributed across psychology nodes.
        
        Creates separate files:
        - core.json: Self, metadata, psychology references
        - identity.json: Identity node with all its learning
        - ego.json: Ego node with all its patterns
        - conscience.json: Conscience node with all its validations
        - nodes.json: All other nodes
        
        Args:
            path: Directory path OR file path. If file path ending in .json,
                  uses parent directory for distributed files.
        """
        if path is None:
            growth_dir = get_growth_path("")
        elif path.endswith('.json'):
            # If given a .json file path, use its directory
            growth_dir = os.path.dirname(path) or get_growth_path("")
        elif os.path.isfile(path):
            # Legacy single file path - use directory instead
            growth_dir = os.path.dirname(path) or get_growth_path("")
        else:
            growth_dir = path
            
        os.makedirs(growth_dir, exist_ok=True)
        
        # Collect non-psychology nodes
        other_nodes = {}
        psychology_ids = {
            self.identity_node.id if self.identity_node else None,
            self.ego_node.id if self.ego_node else None,
            self.conscience_node.id if self.conscience_node else None,
        }
        psychology_ids.discard(None)
        
        for nid, node in self.nodes.items():
            if nid not in psychology_ids:
                other_nodes[nid] = node.to_dict()
        
        # CORE: Self, metadata, psychology references
        core_data = {
            "metadata": {
                "created": self.created_at,
                "last_modified": datetime.now().isoformat(),
                "node_count": len(self.nodes),
                "total_heat": self.total_heat(),
                "loop_number": self.loop_number,
                "entropy": self.calculate_entropy(),
                "version": self.version,
                "born": self.born,
                "distributed": True,  # Flag for new format
            },
            "self": self.self_node.to_dict() if self.self_node else None,
            "psychology": {
                "identity": self.identity_node.id if self.identity_node else None,
                "ego": self.ego_node.id if self.ego_node else None,
                "conscience": self.conscience_node.id if self.conscience_node else None,
            },
        }
        
        # IDENTITY: What PBAI knows exists
        identity_data = None
        if self.identity_node:
            identity_data = {
                "node": self.identity_node.to_dict(),
                "heat": self.identity_node.heat,
                "axes_count": len(self.identity_node.frame.axes),
            }
        
        # EGO: What patterns work
        ego_data = None
        if self.ego_node:
            ego_data = {
                "node": self.ego_node.to_dict(),
                "heat": self.ego_node.heat,
                "axes_count": len(self.ego_node.frame.axes),
            }
        
        # CONSCIENCE: What's validated as true
        conscience_data = None
        if self.conscience_node:
            conscience_data = {
                "node": self.conscience_node.to_dict(),
                "heat": self.conscience_node.heat,
                "axes_count": len(self.conscience_node.frame.axes),
            }
        
        # NODES: All other nodes
        nodes_data = {
            "count": len(other_nodes),
            "nodes": other_nodes,
        }
        
        # Write all files
        with open(os.path.join(growth_dir, "core.json"), "w") as f:
            json.dump(core_data, f, indent=2)
        
        if identity_data:
            with open(os.path.join(growth_dir, "identity.json"), "w") as f:
                json.dump(identity_data, f, indent=2)
        
        if ego_data:
            with open(os.path.join(growth_dir, "ego.json"), "w") as f:
                json.dump(ego_data, f, indent=2)
        
        if conscience_data:
            with open(os.path.join(growth_dir, "conscience.json"), "w") as f:
                json.dump(conscience_data, f, indent=2)
        
        with open(os.path.join(growth_dir, "nodes.json"), "w") as f:
            json.dump(nodes_data, f, indent=2)
        
        logger.info(f"Saved distributed growth map: {growth_dir}")
        logger.info(f"  Core: loop #{self.loop_number}, {len(self.nodes)} total nodes")
        logger.info(f"  Identity: {identity_data['axes_count'] if identity_data else 0} axes, heat={identity_data['heat'] if identity_data else 0:.1f}")
        logger.info(f"  Ego: {ego_data['axes_count'] if ego_data else 0} axes, heat={ego_data['heat'] if ego_data else 0:.1f}")
        logger.info(f"  Conscience: {conscience_data['axes_count'] if conscience_data else 0} axes, heat={conscience_data['heat'] if conscience_data else 0:.1f}")
        logger.info(f"  Other nodes: {len(other_nodes)}")
    
    def load_growth_map(self, path: str = None) -> 'Manifold':
        """Load manifold state from distributed JSON files.
        
        Supports both:
        - New distributed format (core.json, identity.json, etc.)
        - Legacy single-file format (growth_map.json)
        
        Args:
            path: Directory path, legacy file path, or .json hint path
        """
        if path is None:
            growth_dir = get_growth_path("")
        elif path.endswith('.json'):
            # Could be a hint path - check for distributed format first
            growth_dir = os.path.dirname(path) or get_growth_path("")
            core_path = os.path.join(growth_dir, "core.json")
            if not os.path.exists(core_path):
                # No distributed format, try as legacy file
                if os.path.exists(path) and os.path.getsize(path) > 0:
                    return self._load_legacy_growth_map(path)
                # Check for legacy in same directory
                legacy_path = os.path.join(growth_dir, "growth_map.json")
                if os.path.exists(legacy_path):
                    return self._load_legacy_growth_map(legacy_path)
                raise FileNotFoundError(f"No growth map found at {path}")
        elif os.path.isfile(path):
            # Legacy single file
            return self._load_legacy_growth_map(path)
        else:
            growth_dir = path
        
        core_path = os.path.join(growth_dir, "core.json")
        
        # Check for legacy format
        legacy_path = os.path.join(growth_dir, "growth_map.json")
        if not os.path.exists(core_path) and os.path.exists(legacy_path):
            return self._load_legacy_growth_map(legacy_path)
        
        # Load distributed format
        with open(core_path, "r") as f:
            core_data = json.load(f)
        
        self.created_at = core_data["metadata"]["created"]
        self.loop_number = core_data["metadata"].get("loop_number", 0)
        self.born = core_data["metadata"].get("born", False)
        
        # Reconstruct Self
        if core_data.get("self"):
            self.self_node = SelfNode.from_dict(core_data["self"])
            assert_self_valid(self.self_node)
        
        # Clear indexes
        self.nodes = {}
        self.nodes_by_position = {}
        self.nodes_by_universal = {}
        self.nodes_by_concept = {}
        
        # Load Identity
        identity_path = os.path.join(growth_dir, "identity.json")
        if os.path.exists(identity_path):
            with open(identity_path, "r") as f:
                identity_data = json.load(f)
            node = Node.from_dict(identity_data["node"])
            self.nodes[node.id] = node
            self.nodes_by_position[node.position] = node.id
            self.nodes_by_concept[node.concept] = node.id
            if node.universal_position:
                self.nodes_by_universal[node.universal_position] = node.id
            self.identity_node = node
        
        # Load Ego
        ego_path = os.path.join(growth_dir, "ego.json")
        if os.path.exists(ego_path):
            with open(ego_path, "r") as f:
                ego_data = json.load(f)
            node = Node.from_dict(ego_data["node"])
            self.nodes[node.id] = node
            self.nodes_by_position[node.position] = node.id
            self.nodes_by_concept[node.concept] = node.id
            if node.universal_position:
                self.nodes_by_universal[node.universal_position] = node.id
            self.ego_node = node
        
        # Load Conscience
        conscience_path = os.path.join(growth_dir, "conscience.json")
        if os.path.exists(conscience_path):
            with open(conscience_path, "r") as f:
                conscience_data = json.load(f)
            node = Node.from_dict(conscience_data["node"])
            self.nodes[node.id] = node
            self.nodes_by_position[node.position] = node.id
            self.nodes_by_concept[node.concept] = node.id
            if node.universal_position:
                self.nodes_by_universal[node.universal_position] = node.id
            self.conscience_node = node
        
        # Load other nodes
        nodes_path = os.path.join(growth_dir, "nodes.json")
        if os.path.exists(nodes_path):
            with open(nodes_path, "r") as f:
                nodes_data = json.load(f)
            for nid, ndata in nodes_data.get("nodes", {}).items():
                node = Node.from_dict(ndata)
                self.nodes[nid] = node
                self.nodes_by_position[node.position] = nid
                self.nodes_by_concept[node.concept] = nid
                if node.universal_position:
                    self.nodes_by_universal[node.universal_position] = nid
        
        if self.born:
            mark_birth_complete()
        
        self.bootstrapped = True
        self.version = 2
        
        logger.info(f"Loaded distributed growth map: {growth_dir}")
        logger.info(f"  Loop #{self.loop_number}, {len(self.nodes)} total nodes, born={self.born}")
        
        return self
    
    def _load_legacy_growth_map(self, path: str) -> 'Manifold':
        """Load from legacy single-file format."""
        with open(path, "r") as f:
            data = json.load(f)
        
        self.created_at = data["metadata"]["created"]
        self.loop_number = data["metadata"].get("loop_number", 0)
        self.born = data["metadata"].get("born", False)
        file_version = data["metadata"].get("version", 1)
        
        # Reconstruct Self
        if data.get("self"):
            self_data = data["self"]
            if file_version < 2:
                self_data = migrate_node_v1_to_v2(self_data)
            self.self_node = SelfNode.from_dict(self_data)
            assert_self_valid(self.self_node)
        
        # Reconstruct nodes
        self.nodes = {}
        self.nodes_by_position = {}
        self.nodes_by_universal = {}
        self.nodes_by_concept = {}
        
        for nid, ndata in data.get("nodes", {}).items():
            if file_version < 2:
                ndata = migrate_node_v1_to_v2(ndata)
            node = Node.from_dict(ndata)
            self.nodes[nid] = node
            self.nodes_by_position[node.position] = nid
            self.nodes_by_concept[node.concept] = nid
            if node.universal_position:
                self.nodes_by_universal[node.universal_position] = nid
        
        # Restore psychology node references
        psychology = data.get("psychology", {})
        if psychology.get("identity"):
            self.identity_node = self.nodes.get(psychology["identity"])
        if psychology.get("ego"):
            self.ego_node = self.nodes.get(psychology["ego"])
        if psychology.get("conscience"):
            self.conscience_node = self.nodes.get(psychology["conscience"])
        
        if self.born:
            mark_birth_complete()
        
        self.bootstrapped = True
        self.version = 2
        logger.info(f"Loaded legacy growth map: {path} (loop #{self.loop_number}, {len(self.nodes)} nodes)")
        
        return self
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PATH TRACING
    # ═══════════════════════════════════════════════════════════════════════════
    
    def traces_to_self(self, node: Node, max_depth: int = 100) -> bool:
        """Verify that a node can trace back to Self."""
        if node.position == "":
            return True
        
        current_pos = node.position
        visited = set()
        
        while current_pos:
            if current_pos in visited:
                logger.error(f"Cycle detected at position {current_pos}")
                return False
            visited.add(current_pos)
            current_pos = current_pos[:-1]
            
            if len(visited) > max_depth:
                logger.error(f"Max depth exceeded tracing {node.concept}")
                return False
        
        return True
    
    def verify_all_trace_to_self(self) -> bool:
        """Assert that all nodes trace to Self."""
        for node in self.nodes.values():
            if not self.traces_to_self(node):
                logger.error(f"Node {node.concept} does not trace to Self!")
                return False
        
        if self.self_node:
            assert_self_valid(self.self_node)
        
        return True


# ═══════════════════════════════════════════════════════════════════════════════
# CENTRAL MANIFOLD LOADER - THE ONE PBAI MIND
# ═══════════════════════════════════════════════════════════════════════════════

# Singleton instance - ONE PBAI mind
_PBAI_MANIFOLD: Optional[Manifold] = None


def get_pbai_manifold(growth_path: Optional[str] = None) -> Manifold:
    """
    Get the ONE PBAI manifold.
    
    This is the ONLY place birth() should ever be called (on first run).
    All tasks and drivers should call this function to get the manifold -
    they should NEVER create their own manifold or call birth().
    
    Flow:
    1. If manifold already loaded in memory → return it
    2. If growth_map exists on disk → load it
    3. If no growth_map (first time ever) → birth() once, save, return
    
    Args:
        growth_path: Optional custom path (default: growth/growth_map.json)
        
    Returns:
        The ONE PBAI manifold instance
    """
    global _PBAI_MANIFOLD
    
    # Use default growth path if not specified
    from .node_constants import get_growth_path
    if growth_path is None:
        growth_path = get_growth_path("growth_map.json")
    
    # Already loaded in memory? Return it
    if _PBAI_MANIFOLD is not None:
        return _PBAI_MANIFOLD
    
    # Determine growth directory (distributed format uses directory)
    if growth_path.endswith('.json'):
        growth_dir = os.path.dirname(growth_path) or get_growth_path("")
    else:
        growth_dir = growth_path
    
    # Check if growth map exists (look for core.json in directory)
    core_file = os.path.join(growth_dir, "core.json")
    growth_exists = os.path.exists(core_file)
    
    # Create manifold instance
    manifold = Manifold()
    
    # Try to load from disk
    if growth_exists:
        try:
            manifold.load_growth_map(growth_path)
            logger.info(f"Loaded PBAI mind: {len(manifold.nodes)} nodes from {growth_dir}")
        except Exception as e:
            logger.error(f"Failed to load PBAI mind from {growth_dir}: {e}")
            raise RuntimeError(f"PBAI mind corrupted - cannot load from {growth_dir}") from e
    else:
        # First time ever - BIRTH
        logger.info("═══ FIRST BIRTH - Creating PBAI mind ═══")
        manifold.birth()
        
        # Save immediately so we never birth again
        os.makedirs(growth_dir, exist_ok=True)
        manifold.save_growth_map(growth_path)
        logger.info(f"PBAI mind born and saved to {growth_dir}")
    
    # Store singleton
    _PBAI_MANIFOLD = manifold
    return manifold


def reset_pbai_manifold():
    """
    Reset the singleton for testing purposes only.
    
    WARNING: This should NEVER be called in production code.
    It's only for test fixtures that need a fresh manifold.
    """
    global _PBAI_MANIFOLD
    _PBAI_MANIFOLD = None


def create_manifold(load_path: Optional[str] = None) -> Manifold:
    """
    Create a new manifold or load existing one.
    
    DEPRECATED: Use get_pbai_manifold() instead for the ONE PBAI mind.
    This function is kept for backward compatibility but creates separate
    manifold instances which is usually NOT what you want.
    """
    logger.warning("create_manifold() is deprecated - use get_pbai_manifold() for the ONE PBAI mind")
    manifold = Manifold()
    
    if load_path and os.path.exists(load_path):
        manifold.load_growth_map(load_path)
    else:
        manifold.birth()
    
    return manifold
