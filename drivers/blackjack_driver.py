"""
Blackjack Driver - Self plays blackjack through manifold

Inherits from Driver (environment.py). EnvironmentCore owns the Clock and
DecisionNode; this driver implements perceive()/act() plus blackjack-specific
helpers (card counting, bet sizing, outcome recording).

Architecture:
    blackjack.py ↔ BlackjackDriver ↔ EnvironmentCore → Manifold

FRAME HIERARCHY (on hypersphere):
    blackjack (R=1.0, righteous frame) — task level
    ├── tc_high (R=0.9, proper frame)  — True Count >= +2
    ├── tc_neutral (R=0.9, proper)     — -2 < TC < +2
    └── tc_low (R=0.9, proper)         — True Count <= -2

CARD COUNTING (Hi-Lo):
    - Running count: +1 for 2-6, 0 for 7-9, -1 for 10-A
    - True count: running count / decks remaining
    - Same situation in different count brackets learns INDEPENDENTLY
"""

import logging
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from time import time

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import Manifold, K
from core.nodes import Node, Axis, Order, Element
from core.hypersphere import SpherePosition, place_node_near
from drivers.environment import (
    Driver, Port, NullPort, Perception, Action, ActionResult
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# COUNT BRACKETS
# ═══════════════════════════════════════════════════════════════════════════════

COUNT_HIGH_THRESHOLD = 2      # TC >= +2 is high
COUNT_LOW_THRESHOLD = -2      # TC <= -2 is low

def get_count_bracket(true_count: float) -> str:
    """Get count bracket name for true count."""
    if true_count >= COUNT_HIGH_THRESHOLD:
        return "tc_high"
    elif true_count <= COUNT_LOW_THRESHOLD:
        return "tc_low"
    else:
        return "tc_neutral"


@dataclass
class HandState:
    """Current blackjack hand state."""
    player_value: int
    dealer_upcard: int
    is_soft: bool
    can_double: bool
    can_split: bool
    pair_value: Optional[int]
    num_cards: int


# Angular positions for blackjack frame hierarchy
BJ_TASK_THETA = math.pi / 5        # 36° from north pole
BJ_TASK_PHI = math.pi / 2          # +Y direction (green)
BJ_COUNT_OFFSETS = {
    "tc_high":    (0.15, 0.0),      # North of task frame
    "tc_neutral": (0.0,  0.15),     # East of task frame
    "tc_low":     (-0.15, 0.0),     # South of task frame
}


class BlackjackDriver(Driver):
    """
    PBAI blackjack driver — inherits Driver, routes through EnvironmentCore.

    EnvironmentCore provides Clock (perception routing) and DecisionNode
    (action selection). This driver provides perceive()/act() plus
    game-specific helpers for counting, betting, and outcome recording.
    """

    DRIVER_ID = "blackjack"
    DRIVER_NAME = "Blackjack Driver"
    DRIVER_VERSION = "1.0.0"
    SUPPORTED_ACTIONS = ['hit', 'stand', 'double', 'split']
    HEAT_SCALE = 1.5

    COUNT_VALUES = {
        '2': 1, '3': 1, '4': 1, '5': 1, '6': 1,
        '7': 0, '8': 0, '9': 0,
        '10': -1, 'J': -1, 'Q': -1, 'K': -1, 'A': -1
    }

    def __init__(self, port=None, config=None, manifold=None):
        # Game state
        self.num_decks = 6
        self.task_frame: Optional[Node] = None
        self.count_frames: Dict[str, Node] = {}  # tc_high, tc_neutral, tc_low
        self.conservation_weight = 0.5
        self.bet_weight = 0.5
        self._current_hand_state: Optional[HandState] = None
        self._current_bankroll: int = 0
        self._current_state_key: str = ""
        self._current_context: Dict = {}
        self._pending_action: Optional[str] = None

        super().__init__(port or NullPort("null"), config, manifold=manifold)
        self._init_frames()
        self._init_stat_nodes()

    # ═══════════════════════════════════════════════════════════════════════════
    # DRIVER INTERFACE (abstract methods from Driver)
    # ═══════════════════════════════════════════════════════════════════════════

    def initialize(self) -> bool:
        if self.port:
            self.port.connect()
        self.active = True
        return True

    def shutdown(self) -> bool:
        if self.port:
            self.port.disconnect()
        self.active = False
        return True

    def perceive(self) -> Perception:
        """
        Package current hand state as a Perception.
        Called by EnvironmentCore.perceive() which routes to Clock.
        """
        state = self._current_hand_state
        if not state:
            return Perception(
                entities=["blackjack_idle"],
                properties={"state_key": "idle"},
                heat_value=0.0
            )

        tc = self.get_true_count()
        bracket = get_count_bracket(tc)

        return Perception(
            entities=[self._current_state_key, bracket],
            locations=[bracket],
            properties={
                "state_key": self._current_state_key,
                "true_count": tc,
                "player_value": state.player_value,
                "dealer_upcard": state.dealer_upcard,
                "is_soft": 1.0 if state.is_soft else 0.0,
                "can_double": 1.0 if state.can_double else 0.0,
                "can_split": 1.0 if state.can_split else 0.0,
            },
            heat_value=self.scale_heat(K * 0.1),
            raw=state
        )

    def act(self, action: Action) -> ActionResult:
        """
        Record chosen action. Outcome unknown yet (two-phase game).
        Called by EnvironmentCore.act().
        """
        self._pending_action = action.action_type
        return ActionResult(
            success=True,
            outcome=f"chose_{action.action_type}",
            heat_value=0.0  # Real heat comes from record_outcome/record_push
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # FRAME INITIALIZATION (hypersphere-native)
    # ═══════════════════════════════════════════════════════════════════════════

    def _init_frames(self):
        """Initialize the righteous and proper frame hierarchy on the hypersphere."""
        if not self.manifold:
            return

        # 1. RIGHTEOUS FRAME: blackjack task (concept "bj_task" to avoid collision with DriverNode "blackjack")
        self.task_frame = self.manifold.get_node_by_concept("bj_task")
        if not self.task_frame:
            self.task_frame = Node(
                concept="bj_task",
                theta=BJ_TASK_THETA,
                phi=BJ_TASK_PHI,
                radius=1.0,
                heat=K,
                polarity=1,
                existence="actual",
                righteousness=1.0,
                order=1
            )
            self.manifold.add_node(self.task_frame)
            logger.info(f"Created righteous frame: bj_task @ theta={BJ_TASK_THETA:.2f}, phi={BJ_TASK_PHI:.2f}")

        # 2. PROPER FRAMES: count brackets (children of blackjack)
        count_configs = [
            ("tc_high", 1, "TC >= +2: Deck favors player"),
            ("tc_neutral", 0, "-2 < TC < +2: Neutral deck"),
            ("tc_low", -1, "TC <= -2: Deck favors dealer"),
        ]

        for concept, polarity, description in count_configs:
            node = self.manifold.get_node_by_concept(concept)
            if not node:
                d_theta, d_phi = BJ_COUNT_OFFSETS[concept]
                node = Node(
                    concept=concept,
                    theta=BJ_TASK_THETA + d_theta,
                    phi=BJ_TASK_PHI + d_phi,
                    radius=1.0,
                    heat=K,
                    polarity=polarity,
                    existence="actual",
                    righteousness=0.9,
                    order=2
                )
                self.manifold.add_node(node)

                # Connect task_frame → count_frame
                self.manifold.add_axis_safe(self.task_frame, concept, node.id)
                logger.info(f"Created proper frame: {concept}")

            self.count_frames[concept] = node

    def _init_stat_nodes(self):
        """Create stat tracking nodes near the blackjack task frame."""
        if not self.manifold:
            return

        stats = ["running_count", "cards_seen", "hands_played",
                 "hands_won", "hands_lost", "hands_pushed",
                 "total_wagered", "profit"]

        # Collect existing stat positions
        existing_positions = []
        for stat in stats:
            existing = self.manifold.get_node_by_concept(stat)
            if existing:
                existing_positions.append(SpherePosition(theta=existing.theta, phi=existing.phi))

        for i, stat in enumerate(stats):
            if not self.manifold.get_node_by_concept(stat):
                # Place stats in an arc west of the task frame
                target = SpherePosition(
                    theta=BJ_TASK_THETA + 0.05 * i,
                    phi=BJ_TASK_PHI - 0.3,
                    radius=1.0
                )
                pos = place_node_near(target, existing_positions)
                existing_positions.append(pos)

                node = Node(
                    concept=stat,
                    theta=pos.theta,
                    phi=pos.phi,
                    radius=pos.radius,
                    heat=0.0,
                    polarity=1,
                    existence="actual",
                    righteousness=0.0,
                    order=2
                )
                self.manifold.add_node(node)

    # ═══════════════════════════════════════════════════════════════════════════
    # STAT ACCESSORS
    # ═══════════════════════════════════════════════════════════════════════════

    def _get_stat(self, name: str) -> float:
        if not self.manifold:
            return 0.0
        node = self.manifold.get_node_by_concept(name)
        return node.heat if node else 0.0

    def _set_stat(self, name: str, value: float):
        if not self.manifold:
            return
        node = self.manifold.get_node_by_concept(name)
        if node:
            node.heat = value

    def _add_stat(self, name: str, delta: float):
        if not self.manifold:
            return
        node = self.manifold.get_node_by_concept(name)
        if node:
            node.heat += delta

    @property
    def running_count(self) -> int:
        return int(self._get_stat("running_count"))

    @property
    def cards_seen(self) -> int:
        return int(self._get_stat("cards_seen"))

    def count_card(self, rank: str):
        """Update count for card seen."""
        val = self.COUNT_VALUES.get(rank, 0)
        self._add_stat("running_count", val)
        self._add_stat("cards_seen", 1)

    def get_true_count(self) -> float:
        """True count = running count / decks remaining."""
        decks_left = max(1, (self.num_decks * 52 - self.cards_seen) / 52)
        return self.running_count / decks_left

    def reset_count(self):
        """Reset count for new shoe."""
        self._set_stat("running_count", 0)
        self._set_stat("cards_seen", 0)
        logger.info("Count reset for new shoe")

    # ═══════════════════════════════════════════════════════════════════════════
    # SITUATION KEYS (count-aware)
    # ═══════════════════════════════════════════════════════════════════════════

    def _get_count_frame(self) -> Optional[Node]:
        """Get the proper frame for current count bracket."""
        bracket = get_count_bracket(self.get_true_count())
        return self.count_frames.get(bracket)

    def _situation_key(self, state: HandState) -> str:
        """Generate situation key (hand description only, count is in frame)."""
        soft = "s" if state.is_soft else "h"
        if state.can_split and state.pair_value:
            return f"p{state.pair_value}v{state.dealer_upcard}"
        return f"{soft}{state.player_value}v{state.dealer_upcard}"

    def _full_situation_key(self, state: HandState) -> str:
        """Full key including count bracket for unique identification."""
        bracket = get_count_bracket(self.get_true_count())
        base = self._situation_key(state)
        return f"{bracket}_{base}"

    # ═══════════════════════════════════════════════════════════════════════════
    # SITUATION AND DECISION NODES (hypersphere-native)
    # ═══════════════════════════════════════════════════════════════════════════

    def _get_or_create_situation(self, state: HandState) -> Node:
        """Get or create situation node within current count bracket."""
        if not self.manifold:
            return None

        full_key = self._full_situation_key(state)

        node = self.manifold.get_node_by_concept(full_key)
        if node:
            return node

        # Create new situation near its count frame
        count_frame = self._get_count_frame()
        if not count_frame:
            count_frame = self.task_frame

        # Collect existing situation positions near this count frame
        existing_positions = []
        if count_frame and hasattr(count_frame, 'frame') and count_frame.frame:
            for axis_name, axis in count_frame.frame.axes.items():
                sit_node = self.manifold.get_node(axis.target_id)
                if sit_node:
                    existing_positions.append(SpherePosition(theta=sit_node.theta, phi=sit_node.phi))

        target = SpherePosition(
            theta=count_frame.theta + 0.1 if count_frame else BJ_TASK_THETA + 0.1,
            phi=count_frame.phi + 0.05 if count_frame else BJ_TASK_PHI + 0.05,
            radius=1.0
        )
        pos = place_node_near(target, existing_positions)

        node = Node(
            concept=full_key,
            theta=pos.theta,
            phi=pos.phi,
            radius=pos.radius,
            heat=K,
            polarity=1,
            existence="actual",
            righteousness=0.5,
            order=3
        )
        self.manifold.add_node(node)

        # Connect count_frame → situation via semantic axis
        base_key = self._situation_key(state)
        if count_frame:
            self.manifold.add_axis_safe(count_frame, base_key, node.id)

        logger.debug(f"Created situation: {full_key}")
        return node

    def _get_decision_axis(self, situation: Node, action: str) -> Optional[Axis]:
        """Get the axis for a decision on a situation."""
        return situation.get_axis(action)

    def _get_or_create_decision_axis(self, situation: Node, action: str) -> Axis:
        """Get or create decision axis on situation node."""
        axis = situation.get_axis(action)
        if axis:
            return axis

        decision_id = f"{situation.concept}_{action}"
        axis = self.manifold.add_axis_safe(situation, action, decision_id)
        axis.make_proper()

        logger.debug(f"Created decision axis: {situation.concept} --{action}-->")
        return axis

    # ═══════════════════════════════════════════════════════════════════════════
    # HAND STATE MANAGEMENT (for EnvironmentCore integration)
    # ═══════════════════════════════════════════════════════════════════════════

    def set_hand_state(self, state: HandState, bankroll: int):
        """
        Set current hand state before perceive cycle.
        Called by blackjack.py before env_core.perceive() → decide().
        """
        self._current_hand_state = state
        self._current_bankroll = bankroll

        state_key = self._full_situation_key(state)
        self._current_state_key = state_key

        tc = self.get_true_count()
        bracket = get_count_bracket(tc)

        self._current_context = {
            'count_bracket': bracket,
            'true_count': tc,
            'is_soft': state.is_soft,
            'can_double': state.can_double,
            'can_split': state.can_split,
            'player_value': state.player_value,
            'dealer_upcard': state.dealer_upcard,
        }

    def get_actions(self, state: HandState = None) -> List[str]:
        """Get available actions for current state."""
        if state is None:
            state = self._current_hand_state
        if state is None:
            return ["hit", "stand"]
        actions = ["hit", "stand"]
        if state.can_double:
            actions.append("double")
        if state.can_split:
            actions.append("split")
        return actions

    def get_action_scores(self) -> Dict[str, float]:
        """
        Return action scores for EnvironmentCore.decide() exploitation.
        Uses learned axis scores + basic strategy baseline.
        """
        state = self._current_hand_state
        if not state:
            return {}

        situation = self._get_or_create_situation(state)
        if not situation:
            return {}

        actions = self.get_actions(state)
        scores = {}

        for action in actions:
            scores[action] = self._score_action(situation, action, state)

        return scores

    # ═══════════════════════════════════════════════════════════════════════════
    # BET SIZING
    # ═══════════════════════════════════════════════════════════════════════════

    def get_bet_size(self, bankroll: int) -> int:
        """Determine bet size based on true count and manifold state."""
        tc = self.get_true_count()

        base_bet = max(10, bankroll // 100)

        if tc >= 2:
            multiplier = min(5, 1 + (tc - 1))
        elif tc <= -1:
            multiplier = 0.5
        else:
            multiplier = 1.0

        multiplier = 1 + (multiplier - 1) * self.bet_weight

        bet = int(base_bet * multiplier)
        bet = max(10, min(bet, bankroll))

        return bet

    # ═══════════════════════════════════════════════════════════════════════════
    # ACTION SCORING
    # ═══════════════════════════════════════════════════════════════════════════

    def _score_action(self, situation: Node, action: str, state: HandState) -> float:
        """Score an action based on manifold state."""
        score = 0.0

        # 1. Historical learning from axis
        axis = self._get_decision_axis(situation, action)
        if axis:
            experience = axis.traversal_count

            if axis.order and axis.order.elements:
                wins = len([e for e in axis.order.elements if e.index == 1])
                losses = len([e for e in axis.order.elements if e.index == 0])
                total = wins + losses
                if total > 0:
                    win_rate = wins / total
                    confidence = min(1.0, experience / 20)
                    score += (win_rate - 0.5) * 2 * confidence * self.conservation_weight
            else:
                score += 0.05 * min(experience, 5) * self.conservation_weight
        else:
            # Unexplored: curiosity bonus
            score += (1 - self.conservation_weight) * 0.3

        # 2. Basic strategy baseline
        score += self._basic_score(state, action) * 0.3

        return score

    def _basic_score(self, state: HandState, action: str) -> float:
        """Basic strategy heuristic score."""
        pv, dv = state.player_value, state.dealer_upcard

        if not state.is_soft:
            if pv >= 17:
                return 1.0 if action == "stand" else -0.5
            if pv <= 11:
                return 0.8 if action == "hit" else 0.0
            if pv == 11 and state.can_double:
                return 1.0 if action == "double" else 0.5
            if 12 <= pv <= 16:
                if dv >= 7:
                    return 0.6 if action == "hit" else 0.0
                else:
                    return 0.6 if action == "stand" else 0.2
        else:
            if pv >= 19:
                return 1.0 if action == "stand" else -0.3
            if pv == 18:
                if dv >= 9:
                    return 0.5 if action == "hit" else 0.4
                return 0.6 if action == "stand" else 0.3
            return 0.7 if action == "hit" else 0.1

        return 0.0

    # ═══════════════════════════════════════════════════════════════════════════
    # LEARNING (outcome recording — returns ActionResult for feedback)
    # ═══════════════════════════════════════════════════════════════════════════

    def record_decision(self, state: HandState, action: str) -> str:
        """
        Record that a decision was made (before outcome known).
        Returns the situation key so caller can pass it to record_outcome.
        """
        situation = self._get_or_create_situation(state)
        if situation:
            axis = self._get_or_create_decision_axis(situation, action)
            axis.strengthen()

        return situation.concept if situation else ""

    def record_outcome(self, state: HandState, action: str, won: bool, amount: float,
                       situation_key: str = None) -> ActionResult:
        """
        Record outcome — returns ActionResult for env_core.feedback().
        """
        # Get situation
        if situation_key and self.manifold:
            situation = self.manifold.get_node_by_concept(situation_key)
            if not situation:
                situation = self._get_or_create_situation(state)
        else:
            situation = self._get_or_create_situation(state)

        if situation:
            axis = self._get_or_create_decision_axis(situation, action)

            if not axis.order:
                axis.make_proper()

            outcome_idx = 1 if won else 0
            element_id = f"{situation.concept}_{action}_{len(axis.order.elements)}"
            axis.order.elements.append(
                Element(node_id=element_id, index=outcome_idx)
            )

            if won:
                situation.add_heat(K * 0.5)

        # Calculate heat value for feedback
        if won:
            heat_value = K * amount / 100
            outcome = f"{action}_win"
        else:
            heat_value = 0.0
            outcome = f"{action}_loss"

        # Update stats
        self._add_stat("hands_played", 1)
        if won:
            self._add_stat("hands_won", 1)
            self._add_stat("profit", amount)
        else:
            self._add_stat("hands_lost", 1)
            self._add_stat("profit", -amount)
        self._add_stat("total_wagered", amount)

        self._pending_action = None

        result_text = "WIN" if won else "LOSS"
        sit_key = situation.concept if situation else "unknown"
        logger.info(f"LEARN: {sit_key} {action} -> {result_text} (${amount})")

        return ActionResult(
            success=won,
            outcome=outcome,
            heat_value=self.scale_heat(heat_value)
        )

    def record_push(self, state: HandState, action: str, amount: float,
                    situation_key: str = None) -> ActionResult:
        """Record a push (tie) — returns ActionResult for env_core.feedback()."""
        self._add_stat("hands_played", 1)
        self._add_stat("hands_pushed", 1)
        self._pending_action = None

        sit_key = situation_key or self._full_situation_key(state)
        logger.info(f"PUSH: {sit_key} {action}")

        return ActionResult(
            success=True,
            outcome=f"{action}_push",
            heat_value=0.0,
            changes={"success_type": "neutral"}
        )

    def record_blackjack(self, won: bool, amount: float) -> ActionResult:
        """Record a blackjack (natural 21) — returns ActionResult for env_core.feedback()."""
        self._add_stat("hands_played", 1)
        if won:
            self._add_stat("hands_won", 1)
            self._add_stat("profit", amount * 1.5)
            heat_value = self.scale_heat(K * 0.5)
        else:
            self._add_stat("hands_lost", 1)
            heat_value = 0.0

        self._pending_action = None

        return ActionResult(
            success=won,
            outcome="blackjack_win" if won else "blackjack_push",
            heat_value=heat_value
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # WEIGHT ADJUSTMENT
    # ═══════════════════════════════════════════════════════════════════════════

    def adjust_weights(self, current_bankroll: int, starting_bankroll: int):
        """Adjust strategy weights based on performance."""
        profit_ratio = current_bankroll / starting_bankroll

        if profit_ratio > 1.2:
            self.conservation_weight = min(0.8, self.conservation_weight + 0.05)
        elif profit_ratio < 0.8:
            self.conservation_weight = max(0.3, self.conservation_weight - 0.05)

    # ═══════════════════════════════════════════════════════════════════════════
    # INTROSPECTION
    # ═══════════════════════════════════════════════════════════════════════════

    def get_stats(self) -> Dict[str, float]:
        """Get all statistics."""
        return {
            'running_count': self.running_count,
            'true_count': self.get_true_count(),
            'cards_seen': self.cards_seen,
            'hands_played': self._get_stat("hands_played"),
            'hands_won': self._get_stat("hands_won"),
            'hands_lost': self._get_stat("hands_lost"),
            'hands_pushed': self._get_stat("hands_pushed"),
            'profit': self._get_stat("profit"),
            'total_wagered': self._get_stat("total_wagered"),
            'conservation_weight': self.conservation_weight,
            'bet_weight': self.bet_weight,
        }

    def get_confidence(self) -> float:
        """Overall confidence based on experience."""
        total_experience = 0
        situations_with_data = 0

        for bracket, frame in self.count_frames.items():
            if hasattr(frame, 'frame') and frame.frame:
                for axis_name, axis in frame.frame.axes.items():
                    situation = self.manifold.get_node(axis.target_id) if self.manifold else None
                    if situation and hasattr(situation, 'frame') and situation.frame:
                        for dec_name, dec_axis in situation.frame.axes.items():
                            total_experience += dec_axis.traversal_count
                            if dec_axis.order and len(dec_axis.order.elements) >= 3:
                                situations_with_data += 1

        if total_experience > 0:
            confidence = min(1.0, math.log(1 + total_experience) / math.log(500))
        else:
            confidence = 0.0

        return confidence

    def get_mood(self) -> str:
        """Mood based on recent performance."""
        hands = self._get_stat("hands_played")
        if hands < 10:
            return "learning"

        won = self._get_stat("hands_won")
        lost = self._get_stat("hands_lost")

        if won + lost == 0:
            return "uncertain"

        win_rate = won / (won + lost)
        profit = self._get_stat("profit")

        if profit > 0 and win_rate > 0.45:
            return "confident"
        elif profit < -100:
            return "cautious"
        elif win_rate < 0.35:
            return "uncertain"
        else:
            return "focused"

    def get_situation_summary(self) -> str:
        """Get summary of learned situations per count bracket."""
        lines = []
        for bracket, frame in self.count_frames.items():
            if hasattr(frame, 'frame') and frame.frame:
                situations = len(frame.frame.axes)
            else:
                situations = 0
            lines.append(f"{bracket}: {situations} situations")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')

    from core import get_pbai_manifold, get_growth_path
    from drivers.environment import EnvironmentCore

    print("=== BlackjackDriver Self-Test ===\n")
    passed = 0
    failed = 0

    def check(name, condition):
        global passed, failed
        if condition:
            print(f"  ✓ {name}")
            passed += 1
        else:
            print(f"  ✗ {name}")
            failed += 1

    # 1. Create manifold and driver
    growth_path = get_growth_path("growth_map.json")
    manifold = get_pbai_manifold(growth_path)
    driver = BlackjackDriver(manifold=manifold)

    check("Driver inherits from Driver base", isinstance(driver, Driver))
    check("Driver ID is 'blackjack'", driver.DRIVER_ID == "blackjack")
    check("Task frame created", driver.task_frame is not None)
    check("Task frame concept is 'bj_task'", driver.task_frame.concept == "bj_task")
    check("Task frame on hypersphere", abs(driver.task_frame.theta - BJ_TASK_THETA) < 0.01)
    check("Count frames created", len(driver.count_frames) == 3)

    # 2. Initialize via EnvironmentCore
    env_core = EnvironmentCore(manifold=manifold)
    env_core.register_driver(driver)
    activated = env_core.activate_driver("blackjack")
    check("EnvironmentCore activates driver", activated)

    # 3. Card counting
    driver.count_card('5')
    driver.count_card('10')
    check("Running count after 5+10 = 0", driver.running_count == 0)
    driver.count_card('3')
    check("Running count after +3 = 1", driver.running_count == 1)
    driver.reset_count()
    check("Reset count works", driver.running_count == 0)

    # 4. Set hand state + perceive
    state = HandState(
        player_value=15,
        dealer_upcard=10,
        is_soft=False,
        can_double=True,
        can_split=False,
        pair_value=None,
        num_cards=2
    )
    driver.set_hand_state(state, 1000)
    check("Hand state set", driver._current_hand_state is not None)

    perception = env_core.perceive()
    check("Perception has entities", len(perception.entities) > 0)
    check("Perception has state_key", 'state_key' in perception.properties)

    # 5. Decide via EnvironmentCore
    action = env_core.decide(perception)
    check("Action returned", action is not None)
    check("Action type is valid", action.action_type in ['hit', 'stand', 'double', 'split'])

    # 6. get_action_scores
    scores = driver.get_action_scores()
    check("Action scores returned", len(scores) > 0)
    check("Scores are floats", all(isinstance(v, float) for v in scores.values()))

    # 7. Record outcome returns ActionResult
    result = driver.record_outcome(state, "hit", True, 50.0)
    check("record_outcome returns ActionResult", isinstance(result, ActionResult))
    check("Win result has heat", result.heat_value > 0)

    result_push = driver.record_push(state, "stand", 50.0)
    check("record_push returns ActionResult", isinstance(result_push, ActionResult))
    check("Push has neutral success_type", result_push.changes.get("success_type") == "neutral")

    result_bj = driver.record_blackjack(True, 50.0)
    check("record_blackjack returns ActionResult", isinstance(result_bj, ActionResult))

    # 8. Stats
    stats = driver.get_stats()
    check("Stats available", len(stats) > 0)
    check("Hands played tracked", stats['hands_played'] >= 2)

    # 9. Bet sizing
    bet = driver.get_bet_size(1000)
    check("Bet size reasonable", 10 <= bet <= 1000)

    print(f"\n{'='*40}")
    print(f"  Passed: {passed}  Failed: {failed}")
    print(f"{'='*40}")
    sys.exit(1 if failed > 0 else 0)
