"""
PBAI Introspector — Cube-Native Simulation + Short-Term Memory

Two-fold thinking:
    1. SIMULATION: Project options onto hypersphere, read cube coordinates
    2. SHORT-TERM MEMORY: Recent simulation results for decision context

For each option:
    1. Project option onto hypersphere surface (angular position)
    2. Get cube coordinates from projection (x, y, tau)
    3. Read heat magnitude, quadrant, tau position
    4. Evaluate Righteousness via Conscience (cos projection)
    5. Package as enriched context for decision pipeline

No hardcoded color mappings — cube quadrants drive everything.
No deepcopy per option — lightweight angular projection.
should_think gates on Conscience too, not just Ego.

Architecture:
    Introspector sits between perception and decision.
    It enriches options with simulated cube-native context
    before the beta->delta->Gamma->alpha->zeta pipeline runs.
"""

import math
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from .nodes import Node
from .hypersphere import SpherePosition, angular_distance, place_node_near
from .color_cube import CubePosition, evaluate_righteousness as cube_evaluate_R
from .node_constants import (
    K, PHI, INV_PHI,
    COST_EVALUATE, PSYCHOLOGY_MIN_HEAT,
    CONFIDENCE_EXPLOIT_THRESHOLD,
    EXISTENCE_ACTUAL,
)

logger = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    """Result of simulating one option through cube projection.

    Contains the cube-native coordinates and derived quantities
    that enrich decision context without copying the manifold.
    """
    option: str
    # Hypersphere position
    theta: float = 0.0
    phi: float = 0.0
    # Cube projection
    cube_x: float = 0.0       # Blue(-1) / Yellow(+1)
    cube_y: float = 0.0       # Red(-1) / Green(+1)
    cube_tau: float = 0.0     # Past(-) / Future(+)
    # Derived
    heat_magnitude: float = 0.0
    quadrant: str = "Q1"
    righteousness: float = 1.0  # R via Conscience cos projection
    # Conscience evaluation
    conscience_score: float = 0.0  # How righteous Conscience judges this

    def to_dict(self) -> dict:
        return {
            "option": self.option,
            "theta": self.theta, "phi": self.phi,
            "cube_x": self.cube_x, "cube_y": self.cube_y, "cube_tau": self.cube_tau,
            "heat_magnitude": self.heat_magnitude,
            "quadrant": self.quadrant,
            "righteousness": self.righteousness,
            "conscience_score": self.conscience_score,
        }


class Introspector:
    """
    Cube-native introspection engine.

    Enriches decision options by simulating their cube projections
    without copying or mutating the manifold.

    Usage:
        introspector = Introspector(manifold)
        if introspector.should_think():
            results = introspector.simulate(options, state_key)
            context = introspector.to_context(results)
            # Pass context to decision pipeline
    """

    def __init__(self, manifold):
        self.manifold = manifold

        # Short-term simulation memory — recent results for pattern detection
        self._stm: List[Dict[str, Any]] = []
        self._stm_capacity: int = 12  # Movement constant (12 directions)

    # ═══════════════════════════════════════════════════════════════════════════
    # GATING — Should we think about this?
    # ═══════════════════════════════════════════════════════════════════════════

    def should_think(self) -> bool:
        """
        Gate introspection on Ego energy + Conscience existence.

        Ego must have energy to think (above min heat + eval cost).
        Conscience must be actual (connected, conscious) to validate.

        NOTE: In mature manifolds, Conscience heat stays near PSYCHOLOGY_MIN_HEAT
        because the clock tick drains it through axis traversal. We only check
        Conscience existence (ACTUAL = not dormant), not heat level. The
        Introspector pays from Ego, not Conscience.
        """
        ego = self.manifold.ego_node
        conscience = self.manifold.conscience_node

        # Both psychology nodes must exist and be actual
        if not ego or ego.existence != EXISTENCE_ACTUAL:
            return False
        if not conscience or conscience.existence != EXISTENCE_ACTUAL:
            return False

        # Ego must have energy to afford evaluation
        if ego.heat < COST_EVALUATE + PSYCHOLOGY_MIN_HEAT:
            return False

        return True

    # ═══════════════════════════════════════════════════════════════════════════
    # SIMULATION — Project options through cube
    # ═══════════════════════════════════════════════════════════════════════════

    def simulate(self, options: List[str], state_key: str = "") -> List[SimulationResult]:
        """
        Simulate each option by projecting onto hypersphere → cube.

        For each option:
        1. Find or estimate angular position on hypersphere
        2. Project to cube coordinates (x, y, tau)
        3. Read heat magnitude and quadrant
        4. Evaluate Righteousness via Conscience (cos projection)
        5. Package as SimulationResult

        This is LIGHTWEIGHT — no deepcopy, no manifold mutation.

        Args:
            options: Available choices to simulate
            state_key: Current state for context

        Returns:
            List of SimulationResult, one per option
        """
        results = []

        # Get reference point: state node or Identity node
        ref_node = None
        if state_key:
            ref_node = self.manifold.get_node_by_concept(state_key)
        if not ref_node and self.manifold.identity_node:
            ref_node = self.manifold.identity_node

        for i, option in enumerate(options):
            result = self._simulate_option(option, i, len(options), ref_node)
            results.append(result)

        # Pay evaluation cost from Ego (thinking costs energy)
        if self.manifold.ego_node:
            self.manifold.ego_node.spend_heat(COST_EVALUATE, minimum=PSYCHOLOGY_MIN_HEAT)

        # Record to STM
        self._record_stm(state_key, results)

        return results

    def _simulate_option(self, option: str, index: int, total: int,
                         ref_node: Optional[Node]) -> SimulationResult:
        """
        Simulate a single option through cube projection.

        If the option already exists as a node, use its position.
        Otherwise, estimate position using golden-angle distribution
        around the reference node.
        """
        # Check if option already has a manifold node
        option_node = self.manifold.get_node_by_concept(option)

        if option_node:
            # Use existing node's angular position
            theta = option_node.theta
            phi = option_node.phi
        elif ref_node:
            # Estimate: distribute options around reference via golden angle
            golden_angle = 2 * math.pi * INV_PHI
            theta = ref_node.theta
            phi = (ref_node.phi + index * golden_angle) % (2 * math.pi)
        else:
            # Fallback: distribute evenly on equator
            theta = math.pi / 2
            phi = (2 * math.pi * index) / max(total, 1)

        # Project to cube coordinates
        cube_x = math.sin(theta) * math.cos(phi)
        cube_y = math.sin(theta) * math.sin(phi)
        cube_tau = math.cos(theta)

        # Derive heat magnitude from cube position
        heat_magnitude = math.sqrt(cube_x ** 2 + cube_y ** 2)

        # Determine quadrant from cube signs
        if cube_x >= 0 and cube_y >= 0:
            quadrant = "Q1"
        elif cube_x < 0 and cube_y >= 0:
            quadrant = "Q2"
        elif cube_x < 0 and cube_y < 0:
            quadrant = "Q3"
        else:
            quadrant = "Q4"

        # Evaluate Righteousness via Conscience (cos projection)
        # R = 1 - cos(angle from reference)
        # R -> 0 = righteous (aligned with reference)
        righteousness = 1.0
        if ref_node:
            ref_sp = SpherePosition(theta=ref_node.theta, phi=ref_node.phi)
            opt_sp = SpherePosition(theta=theta, phi=phi)
            angle = angular_distance(ref_sp, opt_sp)
            righteousness = 1.0 - math.cos(angle)

        # Conscience score: how well does this align with cube frame?
        # Use cube-space distance from origin (lower = more righteous)
        cube_pos = CubePosition(x=cube_x, y=cube_y, tau=cube_tau)
        conscience_score = 1.0 - min(cube_evaluate_R(cube_pos), 1.0)

        return SimulationResult(
            option=option,
            theta=theta, phi=phi,
            cube_x=cube_x, cube_y=cube_y, cube_tau=cube_tau,
            heat_magnitude=heat_magnitude,
            quadrant=quadrant,
            righteousness=righteousness,
            conscience_score=conscience_score,
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # CONTEXT — Package results for decision pipeline
    # ═══════════════════════════════════════════════════════════════════════════

    def to_context(self, results: List[SimulationResult]) -> Dict[str, Any]:
        """
        Package simulation results as decision context.

        Returns dict that can be merged into decision_node context,
        enriching the beta->delta->Gamma->alpha->zeta pipeline.
        """
        context = {
            "introspection_count": len(results),
            "simulated": True,
        }

        for result in results:
            prefix = f"sim_{result.option}"
            context[f"{prefix}_quadrant"] = result.quadrant
            context[f"{prefix}_heat"] = result.heat_magnitude
            context[f"{prefix}_R"] = result.righteousness
            context[f"{prefix}_conscience"] = result.conscience_score

        # Summary metrics
        if results:
            best = max(results, key=lambda r: r.conscience_score)
            worst = min(results, key=lambda r: r.conscience_score)
            context["sim_best_option"] = best.option
            context["sim_best_conscience"] = best.conscience_score
            context["sim_worst_option"] = worst.option
            context["sim_spread"] = best.conscience_score - worst.conscience_score

            # Quadrant distribution
            quadrants = [r.quadrant for r in results]
            context["sim_quadrant_diversity"] = len(set(quadrants))

        return context

    def rank_by_conscience(self, results: List[SimulationResult]) -> List[SimulationResult]:
        """Rank simulation results by Conscience score (highest first)."""
        return sorted(results, key=lambda r: r.conscience_score, reverse=True)

    def rank_by_heat(self, results: List[SimulationResult]) -> List[SimulationResult]:
        """Rank simulation results by heat magnitude (highest first)."""
        return sorted(results, key=lambda r: r.heat_magnitude, reverse=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # SHORT-TERM MEMORY — Recent simulation results
    # ═══════════════════════════════════════════════════════════════════════════

    def _record_stm(self, state_key: str, results: List[SimulationResult]) -> None:
        """Record simulation results to short-term memory."""
        self._stm.append({
            "state_key": state_key,
            "results": [r.to_dict() for r in results],
            "t_K": self.manifold.get_time() if self.manifold else 0,
        })
        if len(self._stm) > self._stm_capacity:
            self._stm.pop(0)

    def get_stm_pattern(self) -> Optional[str]:
        """
        Detect patterns in recent simulations.

        Looks for repeated quadrant preferences or consistent
        conscience scores across recent states.

        Returns:
            Pattern description string, or None if no pattern
        """
        if len(self._stm) < 3:
            return None

        # Check if same quadrant dominates recent best options
        recent_quadrants = []
        for entry in self._stm[-3:]:
            results = entry["results"]
            if results:
                best = max(results, key=lambda r: r.get("conscience_score", 0))
                recent_quadrants.append(best.get("quadrant", ""))

        if len(set(recent_quadrants)) == 1 and recent_quadrants[0]:
            return f"quadrant_preference:{recent_quadrants[0]}"

        return None

    def get_stm_summary(self) -> Dict[str, Any]:
        """Get summary of recent simulation memory."""
        return {
            "stm_entries": len(self._stm),
            "stm_capacity": self._stm_capacity,
            "stm_pattern": self.get_stm_pattern(),
        }

    # ═══════════════════════════════════════════════════════════════════════════
    # HEAT-PATTERN EXPLORER — Watches hot nodes, suggests cold neighbors
    # ═══════════════════════════════════════════════════════════════════════════

    def find_hot_nodes(self, n: int = 5) -> List[Node]:
        """
        Find the hottest nodes on the manifold (what's active right now).

        Skips psychology nodes (identity/ego/conscience), bootstraps, and Self.
        Only considers ACTUAL nodes (alive, connected).

        Args:
            n: Number of hot nodes to return

        Returns:
            Top N nodes sorted by heat descending
        """
        candidates = []
        for node in self.manifold.nodes.values():
            if node.concept in ('identity', 'ego', 'conscience'):
                continue
            if node.concept.startswith('bootstrap'):
                continue
            if node.heat == float('inf'):
                continue
            if node.existence != EXISTENCE_ACTUAL:
                continue
            candidates.append(node)

        candidates.sort(key=lambda n: n.heat, reverse=True)
        return candidates[:n]

    def find_cold_neighbors(self, hot_node: Node, k: int = 5) -> List[Node]:
        """
        Find cold neighbors of a hot node — nearby but underused connections.

        These are concepts that are topologically close on the hypersphere
        to something active, but aren't being used. The Introspector
        suggests these as potentially relevant actions or concepts.

        Args:
            hot_node: The active node to search near
            k: Max neighbors to return

        Returns:
            List of cold neighbor Nodes (heat below average)
        """
        # Get more candidates than needed, then filter for coldness
        neighbors = self.manifold.k_nearest(hot_node, k * 2)
        avg_heat = self.manifold.average_heat()

        cold = []
        for node, dist in neighbors:
            if node.heat < avg_heat:
                cold.append(node)
            if len(cold) >= k:
                break
        return cold

    def suggest(self, perception_props: Dict) -> Optional[List[str]]:
        """
        Main entry point: explore heat patterns and suggest actions.

        Called from daemon before decide(). Finds what's hot (active),
        looks for cold neighbors (unused but nearby), and matches
        cold neighbor concepts to available actions.

        The key insight: if "zombie" is hot and "sword" is a cold neighbor,
        the Introspector suggests "attack" because the manifold topology
        connects combat concepts near threat concepts.

        Args:
            perception_props: Current perception properties dict

        Returns:
            List of suggested action names (ordered by relevance), or None
        """
        if not self.should_think():
            return None

        hot_nodes = self.find_hot_nodes(5)
        if not hot_nodes:
            return None

        # Collect cold neighbors across all hot nodes
        # Track (concept, distance_to_hot) for ranking
        cold_suggestions = []
        seen_concepts = set()

        for hot_node in hot_nodes:
            cold_neighbors = self.find_cold_neighbors(hot_node, 3)
            for cold_node in cold_neighbors:
                if cold_node.concept not in seen_concepts:
                    seen_concepts.add(cold_node.concept)
                    cold_suggestions.append(cold_node.concept)

        if not cold_suggestions:
            return None

        # Pay evaluation cost from Ego
        if self.manifold.ego_node:
            self.manifold.ego_node.spend_heat(COST_EVALUATE, minimum=PSYCHOLOGY_MIN_HEAT)

        # Record to STM
        self._record_stm("introspect_suggest", [
            SimulationResult(option=c, heat_magnitude=0.0) for c in cold_suggestions[:5]
        ])

        logger.debug(f"Introspector found {len(cold_suggestions)} cold neighbors near hot nodes")
        return cold_suggestions

    def get_weight_boosts(self, suggestions: List[str], available_actions: List[str]) -> Dict[str, float]:
        """
        Convert Introspector suggestions to action weight boosts.

        Matches cold neighbor concepts against available action names:
        - Direct match: cold concept == action name → boost 4.0
        - Partial match: cold concept is substring of action name → boost 2.5
        - Axis match: cold node has axis pointing to node whose concept
          matches an action → boost 2.0

        Args:
            suggestions: Cold neighbor concepts from suggest()
            available_actions: Actions the driver currently supports

        Returns:
            Dict of {action_name: boost_multiplier}
        """
        boosts = {}
        action_set = set(available_actions)

        for concept in suggestions:
            # Direct match: concept name IS an action
            if concept in action_set:
                boosts[concept] = max(boosts.get(concept, 1.0), 4.0)
                continue

            # Partial match: concept appears in an action name
            for action in available_actions:
                if concept in action or action in concept:
                    boosts[action] = max(boosts.get(action, 1.0), 2.5)

            # Axis match: concept node has axes to nodes whose concepts match actions
            concept_node = self.manifold.get_node_by_concept(concept)
            if concept_node:
                for axis_name, axis in concept_node.frame.axes.items():
                    target_node = self.manifold.get_node(axis.target_id)
                    if target_node and target_node.concept in action_set:
                        boosts[target_node.concept] = max(
                            boosts.get(target_node.concept, 1.0), 2.0
                        )

        return boosts
