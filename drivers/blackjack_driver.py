"""
Blackjack Driver - Self plays blackjack through manifold

FRAME HIERARCHY:
    blackjack (R=1.0, righteous frame) - "u"
    ├── tc_high (R=0.9, proper frame) - "un" - True Count >= +2
    │   └── situations learn here when deck favors player
    ├── tc_neutral (R=0.9, proper frame) - "ue" - -2 < TC < +2
    │   └── situations learn here in neutral conditions
    └── tc_low (R=0.9, proper frame) - "us" - True Count <= -2
        └── situations learn here when deck favors dealer

CARD COUNTING (Hi-Lo):
    - Running count: +1 for 2-6, 0 for 7-9, -1 for 10-A
    - True count: running count / decks remaining
    - Same situation in different count brackets learns INDEPENDENTLY

CLOCK SYNC:
    - Each hand = one tick of game time
    - Self time synchronizes with blackjack-world time
    - Perceptions route through Clock
    - Decisions route through DecisionNode
    - Outcomes route as feedback to psychology
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
from core.clock_node import Clock
from core.decision_node import DecisionNode

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

def get_count_position(true_count: float) -> str:
    """Get manifold position suffix for count bracket."""
    if true_count >= COUNT_HIGH_THRESHOLD:
        return "n"   # North = positive = high count
    elif true_count <= COUNT_LOW_THRESHOLD:
        return "s"   # South = negative = low count
    else:
        return "e"   # East = neutral


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


class BlackjackDriver:
    """
    PBAI blackjack - routes through Clock and DecisionNode.
    
    Same interface for blackjack.py, but internally syncs with manifold.
    """
    
    COUNT_VALUES = {
        '2': 1, '3': 1, '4': 1, '5': 1, '6': 1,
        '7': 0, '8': 0, '9': 0,
        '10': -1, 'J': -1, 'Q': -1, 'K': -1, 'A': -1
    }
    
    def __init__(self, manifold: Manifold):
        self.manifold = manifold
        self.num_decks = 6
        
        # Clock for perception routing
        self._clock: Optional[Clock] = None
        
        # DecisionNode for action routing
        self._decision_node: Optional[DecisionNode] = None
        
        # Frame references (cached after init)
        self.task_frame: Optional[Node] = None
        self.count_frames: Dict[str, Node] = {}  # tc_high, tc_neutral, tc_low
        
        # Weights for decision scoring
        self.conservation_weight = 0.5
        self.bet_weight = 0.5
        
        # Current decision tracking
        self._current_state_key: str = ""
        self._current_context: Dict = {}
        self._pending_action: Optional[str] = None
        
        # Initialize frame hierarchy
        self._init_frames()
        self._init_stat_nodes()
    
    def _get_clock(self) -> Clock:
        """Get or create Clock for perception routing."""
        if self._clock is None:
            self._clock = Clock(self.manifold)
        return self._clock
    
    def _get_decision_node(self) -> DecisionNode:
        """Get or create DecisionNode for action routing."""
        if self._decision_node is None:
            self._decision_node = DecisionNode(self.manifold)
        return self._decision_node
    
    # ═══════════════════════════════════════════════════════════════════════════
    # FRAME INITIALIZATION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _init_frames(self):
        """Initialize the righteous and proper frame hierarchy."""
        # 1. RIGHTEOUS FRAME: blackjack task
        self.task_frame = self.manifold.get_node_by_concept("blackjack")
        if not self.task_frame:
            self.task_frame = Node(
                concept="blackjack",
                position="u",
                heat=K,
                polarity=1,
                existence="actual",
                righteousness=1.0,  # Righteous frame
                order=1
            )
            self.manifold.add_node(self.task_frame)
            logger.info("Created righteous frame: blackjack @ u")
        
        # 2. PROPER FRAMES: count brackets (children of blackjack)
        count_configs = [
            ("tc_high", "un", 1, "TC >= +2: Deck favors player"),
            ("tc_neutral", "ue", 0, "-2 < TC < +2: Neutral deck"),
            ("tc_low", "us", -1, "TC <= -2: Deck favors dealer"),
        ]
        
        for concept, position, polarity, description in count_configs:
            node = self.manifold.get_node_by_concept(concept)
            if not node:
                node = Node(
                    concept=concept,
                    position=position,
                    heat=K,
                    polarity=polarity,
                    existence="actual",
                    righteousness=0.9,  # Proper frame (under righteous)
                    order=2
                )
                self.manifold.add_node(node)
                
                # Connect task_frame → count_frame
                direction = position[-1]  # n, e, or s
                self.task_frame.add_axis(direction, node.id)
                
                logger.info(f"Created proper frame: {concept} @ {position}")
            
            self.count_frames[concept] = node
    
    def _init_stat_nodes(self):
        """Create stat tracking nodes (under blackjack frame)."""
        stats = ["running_count", "cards_seen", "hands_played", 
                 "hands_won", "hands_lost", "hands_pushed", 
                 "total_wagered", "profit"]
        
        for i, stat in enumerate(stats):
            if not self.manifold.get_node_by_concept(stat):
                # Stats are at position "uw" (west of blackjack = data/stats)
                node = Node(
                    concept=stat,
                    position=f"uw{i}",
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
        node = self.manifold.get_node_by_concept(name)
        return node.heat if node else 0.0
    
    def _set_stat(self, name: str, value: float):
        node = self.manifold.get_node_by_concept(name)
        if node:
            node.heat = value
    
    def _add_stat(self, name: str, delta: float):
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
    
    def _get_count_frame(self) -> Node:
        """Get the proper frame for current count bracket."""
        bracket = get_count_bracket(self.get_true_count())
        return self.count_frames[bracket]
    
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
    # SITUATION AND DECISION NODES
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _get_or_create_situation(self, state: HandState) -> Node:
        """Get or create situation node within current count bracket."""
        full_key = self._full_situation_key(state)
        
        # Check if exists
        node = self.manifold.get_node_by_concept(full_key)
        if node:
            return node
        
        # Create new situation inside count frame
        count_frame = self._get_count_frame()
        
        # Position: count frame + down + index
        existing = [n for n in self.manifold.nodes.values() 
                   if n.position.startswith(count_frame.position + "d")]
        idx = len(existing)
        position = f"{count_frame.position}d{idx}"
        
        node = Node(
            concept=full_key,
            position=position,
            heat=K,
            polarity=1,
            existence="actual",
            righteousness=0.5,  # Inside proper frame
            order=3
        )
        self.manifold.add_node(node)
        
        # Connect count_frame → situation via semantic axis
        base_key = self._situation_key(state)
        count_frame.add_axis(base_key, node.id)
        
        logger.debug(f"Created situation: {full_key} @ {position}")
        return node
    
    def _get_decision_axis(self, situation: Node, action: str) -> Optional[Axis]:
        """Get the axis for a decision on a situation."""
        return situation.get_axis(action)
    
    def _get_or_create_decision_axis(self, situation: Node, action: str) -> Axis:
        """Get or create decision axis on situation node."""
        axis = situation.get_axis(action)
        if axis:
            return axis
        
        # Create new axis for this decision
        decision_id = f"{situation.concept}_{action}"
        axis = situation.add_axis(action, decision_id)
        
        # Initialize with Order (makes it a proper frame axis)
        axis.make_proper()
        
        logger.debug(f"Created decision axis: {situation.concept} --{action}-->")
        return axis
    
    # ═══════════════════════════════════════════════════════════════════════════
    # BET SIZING
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_bet_size(self, bankroll: int) -> int:
        """Determine bet size based on true count and manifold state."""
        tc = self.get_true_count()
        
        # Base bet: 1% of bankroll
        base_bet = max(10, bankroll // 100)
        
        # Scale by true count (Kelly-ish)
        if tc >= 2:
            multiplier = min(5, 1 + (tc - 1))
        elif tc <= -1:
            multiplier = 0.5  # Minimum bet when count is bad
        else:
            multiplier = 1.0
        
        # Modulate by bet weight
        multiplier = 1 + (multiplier - 1) * self.bet_weight
        
        bet = int(base_bet * multiplier)
        bet = max(10, min(bet, bankroll))  # Clamp
        
        return bet
    
    # ═══════════════════════════════════════════════════════════════════════════
    # DECISION MAKING (routes through manifold)
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_actions(self, state: HandState) -> List[str]:
        """Get available actions for current state."""
        actions = ["hit", "stand"]
        if state.can_double:
            actions.append("double")
        if state.can_split:
            actions.append("split")
        return actions
    
    def get_action(self, state: HandState, bankroll: int) -> str:
        """
        Choose action - routes through Clock and DecisionNode.
        
        1. Route perception to Clock (sync time)
        2. Get explore/exploit from manifold
        3. Route decision through DecisionNode
        """
        situation = self._get_or_create_situation(state)
        actions = self.get_actions(state)
        
        # ─────────────────────────────────────────────────────────────────────────
        # ROUTE TO CLOCK (Perception → Self)
        # ─────────────────────────────────────────────────────────────────────────
        
        state_key = self._full_situation_key(state)
        self._current_state_key = state_key
        
        tc = self.get_true_count()
        bracket = get_count_bracket(tc)
        
        context = {
            'count_bracket': bracket,
            'true_count': tc,
            'is_soft': state.is_soft,
            'can_double': state.can_double,
            'can_split': state.can_split,
            'player_value': state.player_value,
            'dealer_upcard': state.dealer_upcard,
        }
        self._current_context = context
        
        # Route to Clock
        clock = self._get_clock()
        clock.receive({
            "state_key": state_key,
            "context": context,
            "heat_value": K * 0.1,
            "entities": [state_key, bracket],
            "properties": {
                "state_key": state_key,
                "true_count": tc,
                "player_value": state.player_value,
                "dealer_upcard": state.dealer_upcard,
            }
        })
        
        # Tick Clock (sync Self time with game time)
        clock.tick()
        
        # ─────────────────────────────────────────────────────────────────────────
        # ROUTE TO DECISIONNODE (Self decides)
        # ─────────────────────────────────────────────────────────────────────────
        
        decision_node = self._get_decision_node()
        
        # Get exploration rate from manifold
        exploration_rate = self.manifold.get_exploration_rate()
        
        import random
        if random.random() < exploration_rate:
            # EXPLORE: try something, weighted by basic strategy
            weights = {a: max(0.1, self._basic_score(state, a) + 0.5) for a in actions}
            total = sum(weights.values())
            weights = {a: w/total for a, w in weights.items()}
            
            chosen = random.choices(list(weights.keys()), weights=list(weights.values()))[0]
            logger.info(f"Explore: {chosen} (rate={exploration_rate:.2f})")
        else:
            # EXPLOIT: use learned scores
            scores = {}
            for action in actions:
                score = self._score_action(situation, action, state)
                scores[action] = score
            
            chosen = max(scores, key=scores.get)
            logger.info(f"Exploit: {chosen} | {scores}")
        
        # Record decision start
        self._pending_action = chosen
        decision_node.begin_decision(state_key, actions, 
                                     self.manifold.get_confidence(), 
                                     context)
        decision_node.commit_decision(chosen)
        
        # Log decision
        logger.info(f"{self._situation_key(state)} [{bracket}, TC={tc:.1f}] -> {chosen}")
        
        return chosen
    
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
        
        # Hard hands
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
        
        # Soft hands
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
    # LEARNING (outcome recording with feedback routing)
    # ═══════════════════════════════════════════════════════════════════════════
    
    def record_decision(self, state: HandState, action: str) -> str:
        """
        Record that a decision was made (before outcome known).
        
        Returns the situation key so caller can pass it to record_outcome.
        """
        situation = self._get_or_create_situation(state)
        axis = self._get_or_create_decision_axis(situation, action)
        axis.strengthen()  # Increment traversal count
        
        return situation.concept
    
    def record_outcome(self, state: HandState, action: str, won: bool, amount: float, 
                       situation_key: str = None):
        """
        Record outcome - routes feedback through manifold.
        """
        # Get situation
        if situation_key:
            situation = self.manifold.get_node_by_concept(situation_key)
            if not situation:
                situation = self._get_or_create_situation(state)
        else:
            situation = self._get_or_create_situation(state)
        
        axis = self._get_or_create_decision_axis(situation, action)
        
        # Ensure axis has Order for tracking outcomes
        if not axis.order:
            axis.make_proper()
        
        # Add outcome to Order: index 0 = loss, index 1 = win
        outcome_idx = 1 if won else 0
        element_id = f"{situation.concept}_{action}_{len(axis.order.elements)}"
        axis.order.elements.append(
            Element(node_id=element_id, index=outcome_idx)
        )
        
        # Update situation heat
        if won:
            situation.add_heat(K * 0.5)
        
        # ─────────────────────────────────────────────────────────────────────────
        # ROUTE FEEDBACK TO DECISIONNODE AND PSYCHOLOGY
        # ─────────────────────────────────────────────────────────────────────────
        
        decision_node = self._get_decision_node()
        
        if won:
            outcome = f"{action}_win"
            heat_value = K * amount / 100  # Scale by bet size
        else:
            outcome = f"{action}_loss"
            heat_value = 0
        
        if decision_node.pending_choice:
            decision_node.complete_decision(outcome, won, heat_value)
        
        # Feed psychology
        self._feed_psychology(won, heat_value)
        
        # Log
        result = "WIN" if won else "LOSS"
        logger.info(f"LEARN: {situation.concept} {action} -> {result} (${amount})")
        
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
    
    def _feed_psychology(self, success: bool, heat_value: float):
        """Route feedback to psychology nodes."""
        if success and heat_value > 0:
            # WIN - boost all psychology
            if self.manifold.identity_node:
                self.manifold.identity_node.add_heat(heat_value * 0.5)
            if self.manifold.ego_node:
                self.manifold.ego_node.add_heat(heat_value)
            if self.manifold.conscience_node:
                self.manifold.conscience_node.add_heat(heat_value * 0.2)
        elif heat_value > 0:
            # Small positive
            if self.manifold.identity_node:
                self.manifold.identity_node.add_heat(heat_value * 0.1)
            if self.manifold.ego_node:
                self.manifold.ego_node.add_heat(heat_value * 0.1)
    
    def record_push(self, state: HandState, action: str, amount: float,
                    situation_key: str = None):
        """Record a push (tie) - neutral outcome."""
        if situation_key:
            situation = self.manifold.get_node_by_concept(situation_key)
            if not situation:
                situation = self._get_or_create_situation(state)
        else:
            situation = self._get_or_create_situation(state)
        
        # Push is neutral - just track it happened
        self._add_stat("hands_played", 1)
        self._add_stat("hands_pushed", 1)
        
        self._pending_action = None
        
        logger.info(f"PUSH: {situation.concept} {action}")
    
    def record_blackjack(self, won: bool, amount: float):
        """Record a blackjack (natural 21)."""
        self._add_stat("hands_played", 1)
        if won:
            self._add_stat("hands_won", 1)
            self._add_stat("profit", amount * 1.5)  # Blackjack pays 3:2
            self._feed_psychology(True, K * 0.5)  # Bonus for blackjack
        else:
            self._add_stat("hands_lost", 1)
        
        self._pending_action = None
    
    # ═══════════════════════════════════════════════════════════════════════════
    # WEIGHT ADJUSTMENT
    # ═══════════════════════════════════════════════════════════════════════════
    
    def adjust_weights(self, current_bankroll: int, starting_bankroll: int):
        """Adjust strategy weights based on performance."""
        profit_ratio = current_bankroll / starting_bankroll
        
        if profit_ratio > 1.2:
            # Winning big - trust learned strategy more
            self.conservation_weight = min(0.8, self.conservation_weight + 0.05)
        elif profit_ratio < 0.8:
            # Losing - be more conservative
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
                    situation = self.manifold.get_node(axis.target_id)
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
