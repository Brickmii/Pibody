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

import math

from .nodes import Node, SelfNode, Axis, Frame, Order, assert_self_valid, migrate_node_v1_to_v2, birth_randomizer, mark_birth_complete, is_birth_spent
from .hypersphere import (
    SpherePosition, angular_distance, place_evenly, place_node_near,
    find_neighbors, self_position,
)
from .color_cube import CubePosition, evaluate_righteousness as cube_evaluate_righteousness
from .node_constants import (
    K, PHI, INV_PHI,
    # Direction systems (12 = 6 cardinal + 6 self-relative)
    DIRECTIONS_SELF, DIRECTIONS_UNIVERSAL, DIRECTIONS,
    SELF_DIRECTIONS, ALL_DIRECTIONS, OPPOSITES,  # Legacy
    SELF_DIRECTIONS_SELF, SELF_DIRECTIONS_UNIVERSAL,
    # Existence states
    EXISTENCE_ACTUAL, EXISTENCE_DORMANT, EXISTENCE_ARCHIVED, EXISTENCE_POTENTIAL,
    # Entropy
    ENTROPY_MAGNITUDE_WEIGHT, ENTROPY_VARIANCE_WEIGHT, ENTROPY_DISORDER_WEIGHT,
    # Planck grounding - Fire scaling and body temperature
    FIRE_HEAT, FIRE_TO_MOTION, BODY_TEMPERATURE, SCAFFOLD_HEAT,
    # Base motions
    BASE_MOTIONS, BASE_MOTION_PREFIX, BASE_MOTION_HEAT,
    # Entropy structure recognition
    MAX_ENTROPIC_PROBABILITY, entropy_exceeds_random_limit, get_structure_signal,
    # Emergence threshold
    MAX_ORDER_TOKENS,
    # Costs
    COST_ORDER,
    # Paths
    get_growth_path
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
        Indexes by: id, concept
        """
        # Set creation time if not already set
        if node.created_t_K == 0 and self.self_node:
            node.created_t_K = self.get_time()

        self.nodes[node.id] = node
        self.nodes_by_concept[node.concept] = node.id

        logger.debug(f"Added node: {node}")
    
    def get_node(self, node_id: str) -> Optional[Node]:
        """Get node by ID."""
        return self.nodes.get(node_id)
    
    def get_node_by_angular(self, theta: float, phi: float, tolerance: float = None) -> Optional[Node]:
        """Get the nearest node to an angular position within tolerance."""
        if tolerance is None:
            tolerance = INV_PHI ** 6  # Movement threshold ~0.056 rad
        target = SpherePosition(theta=theta, phi=phi)
        best_node = None
        best_dist = float('inf')
        for node in self.nodes.values():
            sp = SpherePosition(theta=node.theta, phi=node.phi, radius=node.radius)
            dist = angular_distance(target, sp)
            if dist < best_dist and dist < tolerance:
                best_dist = dist
                best_node = node
        return best_node
    
    def get_node_by_concept(self, concept: str) -> Optional[Node]:
        """Get node by concept name."""
        node_id = self.nodes_by_concept.get(concept)
        if node_id:
            return self.nodes.get(node_id)
        if concept == "self" and self.self_node:
            return self.self_node
        return None
    
    def position_occupied(self, theta: float, phi: float, min_separation: float = None) -> bool:
        """Check if an angular position is too close to an existing node."""
        if min_separation is None:
            min_separation = INV_PHI ** 6  # Movement threshold
        target = SpherePosition(theta=theta, phi=phi)
        for node in self.nodes.values():
            sp = SpherePosition(theta=node.theta, phi=node.phi, radius=node.radius)
            if angular_distance(target, sp) < min_separation:
                return True
        return False
    
    def remove_node(self, node_id: str) -> bool:
        """Remove a node from the manifold and all indexes."""
        node = self.nodes.get(node_id)
        if not node:
            return False

        # Remove from all indexes
        if node_id in self.nodes:
            del self.nodes[node_id]
        if node.concept in self.nodes_by_concept:
            del self.nodes_by_concept[node.concept]

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
        
        # Max entropy: all heat at one node = maximum variance, no structure
        uniform_heat = total_heat / n
        max_magnitude = uniform_heat
        max_variance = (total_heat ** 2 * (n - 1)) / (n ** 2) if n > 1 else 0.0
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
        R is angular deviation from the cube frame via Conscience (cos projection).
        R yields 0 when the node is perfectly aligned.

        Uses angular distance on the hypersphere between node and reference,
        projected through the Conscience (cos) function for cube alignment.
        """
        if frame is None:
            frame = self.self_node

        # Angular distance between node and reference frame
        node_sp = SpherePosition(theta=node.theta, phi=node.phi, radius=node.radius)

        if frame and frame.radius > 0:
            frame_sp = SpherePosition(theta=frame.theta, phi=frame.phi, radius=frame.radius)
        else:
            # Frame is at center (Self) — measure from origin
            frame_sp = self_position()

        angle = angular_distance(node_sp, frame_sp) if node.radius > 0 else 0.0

        # Conscience projection: cos maps alignment to cube frame
        # cos(0) = 1.0 (aligned), cos(π) = -1.0 (opposed)
        # R = 1 - cos(angle) → R=0 when aligned, R=2 when opposed
        R = 1.0 - math.cos(angle)

        return R

    def find_righteous_frame(self, node: Node) -> Node:
        """Find the nearest frame where this node's R would yield ~0."""
        node_sp = SpherePosition(theta=node.theta, phi=node.phi, radius=node.radius)
        best_r = float('inf')
        best_frame = self.self_node

        for candidate in self.nodes.values():
            if candidate.id == node.id:
                continue
            cand_sp = SpherePosition(theta=candidate.theta, phi=candidate.phi, radius=candidate.radius)
            angle = angular_distance(node_sp, cand_sp)
            r = 1.0 - math.cos(angle)
            if r < best_r:
                best_r = r
                best_frame = candidate

        return best_frame
    
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
        
        # Self never changes (radius=0 is the center)
        if node.radius == 0.0:
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
        if node.radius == 0.0:
            logger.warning("Cannot archive Self")
            return
        
        node.existence = EXISTENCE_ARCHIVED
        logger.info(f"Archived: {node.concept}")
    
    def create_potential_node(self, concept: str, theta: float = None, phi: float = None, heat: float = None) -> Node:
        """
        Create a new node in POTENTIAL state.

        New concepts start as POTENTIAL until environment confirms.
        Use confirm_existence() after environment validation.

        Args:
            concept: The concept name
            theta: Polar angle on hypersphere (default π/2 = equator)
            phi: Azimuthal angle on hypersphere (default 0.0)
            heat: Initial heat (default K)

        Returns:
            New node in POTENTIAL state
        """
        if heat is None:
            heat = K
        if theta is None:
            theta = math.pi / 2
        if phi is None:
            phi = 0.0

        node = Node(
            concept=concept,
            theta=theta,
            phi=phi,
            radius=1.0,
            heat=heat,
            existence=EXISTENCE_POTENTIAL,
            righteousness=1.0,  # Not yet righteous
        )
        self.add_node(node)

        logger.debug(f"Created potential: {concept} @ theta={theta:.3f},phi={phi:.3f}")
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
        # Base distance = angular distance on hypersphere
        from_sp = SpherePosition(theta=from_node.theta, phi=from_node.phi, radius=from_node.radius)
        to_sp = SpherePosition(theta=to_node.theta, phi=to_node.phi, radius=to_node.radius)
        base_distance = angular_distance(from_sp, to_sp)

        # Find any axis directly connecting them
        for axis in from_node.frame.axes.values():
            if axis.target_id == to_node.id:
                warp = self.calculate_warp_factor(axis)
                # Higher warp = shorter effective distance
                return base_distance * (1.0 - warp * 0.9)  # Max 90% reduction

        return base_distance
    
    def k_nearest(self, node: Node, k: int = 5, exclude_psychology: bool = True) -> list:
        """
        Find k nearest nodes by angular distance on the hypersphere.

        Args:
            node: Reference node
            k: Number of neighbors to return
            exclude_psychology: Skip identity/ego/conscience nodes

        Returns:
            List of (Node, float) tuples sorted by distance ascending
        """
        node_sp = SpherePosition(theta=node.theta, phi=node.phi, radius=node.radius)
        distances = []
        for candidate in self.nodes.values():
            if candidate.id == node.id:
                continue
            if exclude_psychology and candidate.concept in ('identity', 'ego', 'conscience'):
                continue
            if candidate.concept.startswith('bootstrap'):
                continue
            cand_sp = SpherePosition(theta=candidate.theta, phi=candidate.phi, radius=candidate.radius)
            dist = angular_distance(node_sp, cand_sp)
            distances.append((candidate, dist))
        distances.sort(key=lambda x: x[1])
        return distances[:k]

    # ═══════════════════════════════════════════════════════════════════════════
    # AXIS OVERFLOW — 44/45 Emergence Threshold
    # ═══════════════════════════════════════════════════════════════════════════

    def add_axis_safe(self, node: Node, direction: str, target_id: str, polarity: int = 1) -> Axis:
        """Add axis with 44-limit enforcement. Routes to children on overflow.

        Strengthening an existing axis is always OK (not a new token).
        Under limit: normal add. At limit: route to existing child with
        room, or birth a new one. Recurses so children that fill up
        overflow into grandchildren — building sequence depth.

        Args:
            node: The node to add the axis to
            direction: Axis direction/label
            target_id: Target node ID
            polarity: Axis polarity (+1 or -1)

        Returns:
            The created or strengthened Axis
        """
        # Strengthening existing axis is always OK (not a new token)
        if direction in node.frame.axes:
            existing = node.frame.axes[direction]
            existing.strengthen()
            return existing

        # Under limit: normal add
        if len(node.frame.axes) < MAX_ORDER_TOKENS:
            return node.add_axis(direction, target_id, polarity)

        # AT LIMIT: find existing child with room, or birth a new one.
        # Recurse so full children overflow into grandchildren (deeper sequences).
        child = self._get_or_create_overflow_child(node)
        return self.add_axis_safe(child, direction, target_id, polarity)

    def _get_or_create_overflow_child(self, parent: Node) -> Node:
        """Find existing child with room, create new, or promote at 44-limit.

        Children accumulate axes up to 44 before overflowing themselves.
        At 44 children, the 45th triggers promotion — ego selects the best
        child to become a sub-parent, routed through conscience.
        """
        children = self.get_overflow_children(parent)
        for child in children:
            if len(child.frame.axes) < MAX_ORDER_TOKENS:
                return child

        # Under child limit? Birth a new one
        if len(children) < MAX_ORDER_TOKENS:
            return self._create_overflow_child(parent)

        # AT 44 CHILDREN: promote one to sub-parent (internal ego action)
        return self._promote_overflow_child(parent, children)

    def _create_overflow_child(self, parent: Node) -> Node:
        """Birth a child node to carry overflow from a full parent.

        Parent keeps all 44 axes untouched. Child is placed near parent
        on the hypersphere. Relationship is positional (angular proximity)
        and by naming convention ({parent}_c{N}).
        """
        # Count existing children by concept naming convention
        prefix = f"{parent.concept}_c"
        child_count = sum(1 for c in self.nodes_by_concept if c.startswith(prefix))

        child_concept = f"{parent.concept}_c{child_count}"

        # Place child near parent on the hypersphere
        parent_sp = SpherePosition(theta=parent.theta, phi=parent.phi)
        existing = [SpherePosition(theta=n.theta, phi=n.phi) for n in self.nodes.values()]
        child_sp = place_node_near(parent_sp, existing)

        child = Node(
            concept=child_concept,
            theta=child_sp.theta,
            phi=child_sp.phi,
            radius=parent.radius,
            heat=K,
            polarity=parent.polarity,
            existence=parent.existence,
            righteousness=parent.righteousness,
        )
        self.add_node(child)

        logger.debug(f"Overflow child born: {child_concept} from {parent.concept} "
                      f"({len(parent.frame.axes)} axes full)")
        return child

    def _promote_overflow_child(self, parent: Node, children: list) -> Node:
        """Ego promotes the best child to sub-parent via conscience weighing.

        Identity perceives: each child's heat and traversal depth.
        Conscience weighs: validation confidence for each candidate.
        Ego executes: selects highest-scoring child as sub-parent.

        The promoted child becomes the overflow target. Its own overflow
        will create grandchildren naturally via recursive add_axis_safe.
        Ego pays COST_ORDER heat for the structural decision.
        """
        # ── Identity perceives: score each child's accumulated knowledge ──
        scores = []
        for child in children:
            # Heat = accumulated importance (primary signal)
            heat_score = child.heat

            # Traversal depth = how active this branch is
            traversal_total = sum(
                ax.traversal_count for ax in child.frame.axes.values()
            )

            # Righteousness alignment (closer to 0 = better aligned)
            r_weight = 1.0 / (1.0 + abs(child.righteousness))

            identity_score = heat_score * (1.0 + traversal_total * 0.01) * r_weight

            # ── Conscience weighs: validation confidence ──
            conscience_weight = 1.0
            if self.conscience_node:
                conscience_axis = self.conscience_node.get_axis(child.concept)
                if conscience_axis:
                    t = conscience_axis.traversal_count
                    raw_conf = t / (t + K)
                    # Positive polarity = confirmed, negative = corrected
                    if conscience_axis.polarity > 0:
                        conscience_weight = 1.0 + raw_conf  # up to 2.0 boost
                    else:
                        conscience_weight = max(0.1, 1.0 - raw_conf)  # penalty

            combined = identity_score * conscience_weight
            scores.append((combined, child))

        # ── Ego executes: select the highest-scoring child ──
        scores.sort(key=lambda x: x[0], reverse=True)
        promoted = scores[0][1]

        # Ego pays COST_ORDER for the structural decision
        if self.ego_node:
            self.ego_node.spend_heat(COST_ORDER)

        logger.info(
            f"Promoted {promoted.concept} to sub-parent of {parent.concept} "
            f"(score={scores[0][0]:.2f}, heat={promoted.heat:.2f}, "
            f"children={len(children)}/{MAX_ORDER_TOKENS})"
        )

        return promoted

    def get_overflow_children(self, node: Node) -> list:
        """Get direct overflow children of a node by naming convention.

        Only matches {concept}_cN where N is a number — not grandchildren
        like {concept}_cN_cM. This keeps each level's child count accurate
        so the 44-cap and promotion operate at the correct depth.
        """
        prefix = f"{node.concept}_c"
        children = []
        for concept, nid in self.nodes_by_concept.items():
            if concept.startswith(prefix):
                suffix = concept[len(prefix):]
                if suffix.isdigit():
                    child = self.nodes.get(nid)
                    if child:
                        children.append(child)
        return children

    def _cleanup_empty_children(self) -> int:
        """Remove overflow children that have no axes and minimal heat.

        Unlike pruning, this doesn't delete active children. Only removes
        nodes that were created but never used (empty shells).
        """
        removed = 0
        for concept, nid in list(self.nodes_by_concept.items()):
            if '_c' not in concept:
                continue
            node = self.nodes.get(nid)
            if not node:
                continue
            # Only remove if truly empty: no axes AND heat at or below starting K
            if len(node.frame.axes) == 0 and node.heat <= K:
                self.remove_node(nid)
                removed += 1

        if removed:
            logger.info(f"Cleaned up {removed} empty overflow children")
        return removed

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
            # New concept - add axis (with 44-limit enforcement)
            self.add_axis_safe(self.identity_node, concept, concept_node.id, polarity=1 if known else -1)
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
            # Place pattern node near Ego on the hypersphere
            ego_sp = SpherePosition(theta=self.ego_node.theta, phi=self.ego_node.phi)
            existing = [SpherePosition(theta=n.theta, phi=n.phi) for n in self.nodes.values()]
            new_sp = place_node_near(ego_sp, existing)
            pattern_node = Node(
                concept=pattern,
                theta=new_sp.theta,
                phi=new_sp.phi,
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
            axis = self.add_axis_safe(self.ego_node, pattern, pattern_node.id, polarity=1 if success else -1)
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
            axis = self.add_axis_safe(
                self.conscience_node,
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
        # 1. SELF EMERGES AT CENTER OF HYPERSPHERE
        # ─────────────────────────────────────────────────────────────────────
        self.self_node = SelfNode()
        logger.info(f"Self emerged at center: {self.self_node}")
        assert_self_valid(self.self_node)

        # ─────────────────────────────────────────────────────────────────────
        # 2. FIRES 1-5: Bootstrap nodes on hypersphere surface
        #    Each fire at a cardinal direction, K × φⁿ heat scaling
        #    Placed at the 6 cube poles projected onto the sphere
        # ─────────────────────────────────────────────────────────────────────
        logger.info("Fires 1-5: Bootstrap nodes on hypersphere with K*phi^n scaling...")

        # Map cardinal directions to angular positions and fire numbers
        # theta=π/2 is equator (present), phi determines chromatic direction
        fire_positions = {
            'N': {'fire': 1, 'theta': math.pi / 2, 'phi': math.pi / 2,     'polarity': +1},  # +Y (Green)
            'S': {'fire': 2, 'theta': math.pi / 2, 'phi': 3 * math.pi / 2, 'polarity': -1},  # -Y (Red)
            'E': {'fire': 3, 'theta': math.pi / 2, 'phi': 0.0,             'polarity': +1},  # +X (Yellow)
            'W': {'fire': 4, 'theta': math.pi / 2, 'phi': math.pi,         'polarity': -1},  # -X (Blue)
            'U': {'fire': 5, 'theta': 0.0,         'phi': 0.0,             'polarity': +1},  # +Z (Future)
        }

        for direction, info in fire_positions.items():
            fire_num = info['fire']
            fire_heat = FIRE_HEAT[fire_num]
            motion_type = FIRE_TO_MOTION[fire_num]
            opposite = OPPOSITES.get(direction, 'S')

            node = Node(
                concept=f"bootstrap_{direction}",
                theta=info['theta'],
                phi=info['phi'],
                heat=fire_heat,  # K × φⁿ scaling
                polarity=info['polarity'],
                existence=EXISTENCE_ACTUAL,
                righteousness=1.0,
                order=fire_num,
                constraint_type="successor",
            )
            node.frame.origin = node.concept

            # Connect Self to this node via cardinal axis
            self.self_node.add_axis(
                direction=direction,
                target_id=node.id,
                polarity=info['polarity']
            )

            # Connect node back to Self
            node.add_axis(
                direction=opposite,
                target_id=self.self_node.id,
                polarity=info['polarity']
            )

            self.add_node(node)
            logger.info(f"  Fire {fire_num} ({motion_type}): {node.concept} @ theta={info['theta']:.3f},phi={info['phi']:.3f} | heat={fire_heat:.2f} K")

        # ─────────────────────────────────────────────────────────────────────
        # 3. FIRE 6: Abstract space root (south pole — past direction)
        #    Ignites at BODY_TEMPERATURE (K × φ¹¹ ≈ 304 K)
        #    This is where psychology emerges!
        # ─────────────────────────────────────────────────────────────────────
        logger.info(f"Fire 6 (movement): Abstract space - ignites at BODY_TEMPERATURE ({BODY_TEMPERATURE:.2f} K)...")

        # Abstract space root at south pole (-Z / past)
        bootstrap_d = Node(
            concept="bootstrap_d",
            theta=math.pi,     # South pole (-Z)
            phi=0.0,
            heat=FIRE_HEAT[6],  # BODY_TEMPERATURE = K × φ¹¹
            polarity=-1,
            existence=EXISTENCE_ACTUAL,
            righteousness=0.0,  # Abstract root is righteous
            order=6,
            constraint_type="identity",
        )
        bootstrap_d.frame.origin = "bootstrap_d"

        # Connect Self to abstract space
        self.self_node.add_axis(direction="D", target_id=bootstrap_d.id, polarity=-1)

        # Connect abstract root back to Self
        bootstrap_d.add_axis(direction="U", target_id=self.self_node.id, polarity=1)

        self.add_node(bootstrap_d)
        logger.info(f"  Fire 6: {bootstrap_d.concept} @ south pole (abstract space root)")

        # ─────────────────────────────────────────────────────────────────────
        # 2.5. BASE MOTION TOKENS — Cognitive vocabulary spawns from bootstraps
        # ─────────────────────────────────────────────────────────────────────
        logger.info("Base motion tokens: spawning cognitive vocabulary from bootstraps...")

        # Build fire→bootstrap lookup from what we just created
        _bootstrap_map = {}
        for _dir, _info in fire_positions.items():
            _bnode = self.get_node_by_concept(f"bootstrap_{_dir}")
            if _bnode:
                _bootstrap_map[_info['fire']] = _bnode
        _bootstrap_map[6] = bootstrap_d

        # Collect existing positions for place_node_near
        existing_positions = [
            SpherePosition(theta=n.theta, phi=n.phi) for n in self.nodes.values()
        ]

        bm_count = 0
        for fire_num in sorted(BASE_MOTIONS.keys()):
            parent = _bootstrap_map.get(fire_num)
            if not parent:
                continue
            parent_sp = SpherePosition(theta=parent.theta, phi=parent.phi)
            verbs = BASE_MOTIONS[fire_num]

            for verb in verbs:
                concept = f"{BASE_MOTION_PREFIX}{verb}"
                child_sp = place_node_near(parent_sp, existing_positions)

                child = Node(
                    concept=concept,
                    theta=child_sp.theta,
                    phi=child_sp.phi,
                    radius=1.0,
                    heat=BASE_MOTION_HEAT[verb],
                    polarity=parent.polarity,
                    existence=EXISTENCE_ACTUAL,
                    righteousness=1.0,
                    order=fire_num,
                )
                child.tags.add("base_motion")
                child.frame.origin = concept

                self.add_node(child)
                existing_positions.append(child_sp)

                # Connect parent bootstrap → base motion
                parent.add_axis(verb, child.id, polarity=parent.polarity)
                # Connect base motion → parent bootstrap
                child.add_axis(f"bootstrap_{FIRE_TO_MOTION[fire_num]}", parent.id, polarity=parent.polarity)
                # Connect base motion → Self
                child.add_axis("self", self.self_node.id, polarity=1)

                bm_count += 1

        logger.info(f"  Base motions spawned: {bm_count} tokens across 6 fires")

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
        
        # Psychology nodes placed near south pole (abstract space)
        # Offset from bootstrap_d by golden angle increments
        golden_angle = 2 * math.pi / (PHI ** 2)
        psych_theta_base = math.pi * 0.85  # Near south pole but not on it

        # Identity (Id) - 70% - Amplitude axis - sin(1/φ)
        identity_heat = total_psychology_heat * FREUD_IDENTITY_RATIO
        self.identity_node = Node(
            concept="identity",
            theta=psych_theta_base,
            phi=0.0,
            trig_position=TRIG_IDENTITY,  # (sin(1/φ), 0, 0) - amplitude axis
            heat=identity_heat,
            polarity=1,
            existence=EXISTENCE_ACTUAL,
            righteousness=0.0,
            order=1,
        )
        self.identity_node.frame.origin = "identity"
        self.add_node(self.identity_node)
        logger.info(f"  Identity (Id): heat={identity_heat:.3f} ({FREUD_IDENTITY_RATIO*100:.0f}%) @ trig{TRIG_IDENTITY}")

        # Ego - 10% - Phase axis - tan(1/φ)
        ego_heat = total_psychology_heat * FREUD_EGO_RATIO
        self.ego_node = Node(
            concept="ego",
            theta=psych_theta_base,
            phi=golden_angle,
            trig_position=TRIG_EGO,  # (0, tan(1/φ), 0) - phase axis
            heat=ego_heat,
            polarity=1,
            existence=EXISTENCE_ACTUAL,
            righteousness=0.0,
            order=2,
        )
        self.ego_node.frame.origin = "ego"
        self.add_node(self.ego_node)
        logger.info(f"  Ego: heat={ego_heat:.3f} ({FREUD_EGO_RATIO*100:.0f}%) @ trig{TRIG_EGO}")

        # Conscience (Superego) - 20% - Spread axis - cos(1/φ)
        conscience_heat = total_psychology_heat * FREUD_CONSCIENCE_RATIO
        self.conscience_node = Node(
            concept="conscience",
            theta=psych_theta_base,
            phi=2 * golden_angle,
            trig_position=TRIG_CONSCIENCE,  # (0, 0, cos(1/φ)) - spread axis
            heat=conscience_heat,
            polarity=1,
            existence=EXISTENCE_ACTUAL,
            righteousness=0.0,
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
        logger.info(f"  Bootstrap nodes: 6 (N,S,E,W,U + abstract root)")
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
        for node in sorted(self.nodes.values(), key=lambda n: n.theta):
            spatial = len(node.spatial_axes)
            semantic = len(node.semantic_axes)
            proper = len(node.proper_axes)
            pos_str = f"θ={node.theta:.3f},φ={node.phi:.3f}" if node.radius > 0 else "(center)"
            lines.append(f"  {node.concept:20} @ {pos_str:20} | "
                        f"heat={node.heat:8.2f} | R={node.righteousness:.2f} Q={node.quadrant} | "
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

        # Clean up empty overflow children before save
        self._cleanup_empty_children()

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
        self.nodes_by_concept = {}
        
        # Load Identity
        identity_path = os.path.join(growth_dir, "identity.json")
        if os.path.exists(identity_path):
            with open(identity_path, "r") as f:
                identity_data = json.load(f)
            node = Node.from_dict(identity_data["node"])
            self.nodes[node.id] = node
            self.nodes_by_concept[node.concept] = node.id
            self.identity_node = node
        
        # Load Ego
        ego_path = os.path.join(growth_dir, "ego.json")
        if os.path.exists(ego_path):
            with open(ego_path, "r") as f:
                ego_data = json.load(f)
            node = Node.from_dict(ego_data["node"])
            self.nodes[node.id] = node
            self.nodes_by_concept[node.concept] = node.id
            self.ego_node = node
        
        # Load Conscience
        conscience_path = os.path.join(growth_dir, "conscience.json")
        if os.path.exists(conscience_path):
            with open(conscience_path, "r") as f:
                conscience_data = json.load(f)
            node = Node.from_dict(conscience_data["node"])
            self.nodes[node.id] = node
            self.nodes_by_concept[node.concept] = node.id
            self.conscience_node = node
        
        # Load other nodes
        nodes_path = os.path.join(growth_dir, "nodes.json")
        if os.path.exists(nodes_path):
            with open(nodes_path, "r") as f:
                nodes_data = json.load(f)
            for nid, ndata in nodes_data.get("nodes", {}).items():
                node = Node.from_dict(ndata)
                self.nodes[nid] = node
                self.nodes_by_concept[node.concept] = nid
        
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
        self.nodes_by_concept = {}

        for nid, ndata in data.get("nodes", {}).items():
            if file_version < 2:
                ndata = migrate_node_v1_to_v2(ndata)
            node = Node.from_dict(ndata)
            self.nodes[nid] = node
            self.nodes_by_concept[node.concept] = nid
        
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
        """Verify that a node can trace back to Self.

        On the hypersphere, every surface node (radius=1.0) traces to Self
        at the center (radius=0.0) by definition — Self is the origin.
        We verify the node has valid angular coordinates.
        """
        if node.radius == 0.0:
            return True  # This IS Self

        # All surface nodes on the hypersphere trace to Self at center
        # Verify valid angular coordinates
        if not (0.0 <= node.theta <= math.pi):
            logger.error(f"Node {node.concept} has invalid theta: {node.theta}")
            return False
        if not (0.0 <= node.phi < 2 * math.pi + 0.001):
            logger.error(f"Node {node.concept} has invalid phi: {node.phi}")
            return False

        # Verify radius is valid (0.0 = Self, 1.0 = surface)
        if not (0.0 <= node.radius <= 1.0):
            logger.error(f"Node {node.concept} has invalid radius: {node.radius}")
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
