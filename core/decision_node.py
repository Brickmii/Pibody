"""
PBAI Decision Node - Movement (Lin) - The 6th Motion Function

════════════════════════════════════════════════════════════════════════════════
THE 6 MOTION FUNCTIONS → THE 6 CORE FILES
════════════════════════════════════════════════════════════════════════════════

    1. Heat (Σ)           - psychology     - magnitude validator
    2. Polarity (+/-)     - node_constants - direction validator  
    3. Existence (δ)      - clock_node     - persistence validator (Self IS clock)
    4. Righteousness (R)  - nodes          - alignment validator (frames)
    5. Order (Q)          - manifold       - arithmetic validator
                            ↓
    6. Movement (Lin)     - decision_node  - VECTORIZED OUTPUT (this file)

The decision node is the EXIT point - where 5 scalar inputs become 1 vector output.

════════════════════════════════════════════════════════════════════════════════
5 SCALARS → 1 VECTOR
════════════════════════════════════════════════════════════════════════════════

Decision takes the 5 validated scalars and produces movement:

    Heat (Σ)              How much energy for this action?
    Polarity (+/-)        Which direction? (approach/avoid)
    Existence (δ)         Does this option exist/persist? (above 1/φ³?)
    Righteousness (R)     Is this option aligned? (R→0?)
    Order (Q)             What's the history? (success count)
                          ↓
    Movement (Lin)        THE DECISION (selected action vector)

════════════════════════════════════════════════════════════════════════════════
THE 5/6 CONFIDENCE THRESHOLD
════════════════════════════════════════════════════════════════════════════════

    Confidence = Conscience's mediation (Identity → Ego)
    
    When confidence > 5/6 (0.8333):
        - 5 scalar functions are validated
        - EXPLOIT: Use the learned pattern
        
    When confidence < 5/6:
        - Still gathering validation
        - EXPLORE: Try to learn more

    t = 5K validations crosses threshold (one K-quantum per scalar)

════════════════════════════════════════════════════════════════════════════════
DECISION PROCESS (Collapse → Correlate → Select)
════════════════════════════════════════════════════════════════════════════════

    1. COLLAPSE: Find CENTER node (R→0, most righteous)
    2. CORRELATE: Get CLUSTER from center (current + historical + novel)
    3. SELECT: 
       - If confidence > 5/6: EXPLOIT Order (use proven pattern)
       - If confidence < 5/6: EXPLORE (try options, learn)

════════════════════════════════════════════════════════════════════════════════
ARCHITECTURE
════════════════════════════════════════════════════════════════════════════════

    ┌─────────────────────────────────────────────────────────────┐
    │                      MANIFOLD                                │
    │                                                              │
    │   [environment.py]   ──────────▶  [decision_node.py]        │
    │      (ENTRY)            process       (EXIT)                 │
    │           │                              │                   │
    │           ▼                              ▼                   │
    │   ┌─────────────┐               ┌─────────────┐             │
    │   │  Identity   │ ────────────▶ │    Ego      │             │
    │   │  (observe)  │   Conscience  │  (decide)   │             │
    │   └─────────────┘   (mediate)   └─────────────┘             │
    └──────────┬───────────────────────────────┬──────────────────┘
               │                               │
               │ Perception                    │ Action (Movement)
               ▼                               ▼
    ┌─────────────────┐               ┌─────────────────┐
    │   driver node    │               │   driver node    │
    │   (states)      │               │   (plans)       │
    └─────────────────┘               └─────────────────┘

════════════════════════════════════════════════════════════════════════════════
"""

import math
import os
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from time import time

from .nodes import Node, Axis, Order, Element
from .hypersphere import SpherePosition, angular_distance, place_node_near
from .node_constants import (
    K, PHI, THRESHOLD_ORDER, THRESHOLD_EXISTENCE,
    CONFIDENCE_EXPLOIT_THRESHOLD,
    EXISTENCE_ACTUAL, EXISTENCE_DORMANT, EXISTENCE_POTENTIAL,
    get_growth_path
)
from .driver_node import MotorAction, ActionPlan, SensorReport

logger = logging.getLogger(__name__)


def get_decisions_path() -> str:
    """Get path to decisions folder."""
    project_root = get_growth_path("").replace("/growth", "")
    return os.path.join(project_root, "decisions")


@dataclass
class Choice:
    """
    A recorded choice with its outcome - the atomic unit of decision.
    
    Records the 5 scalar inputs and 1 vector output:
    
        INPUTS (5 scalars):
        - heat: Energy available (magnitude)
        - polarity: Direction preference (+1 approach, -1 avoid)
        - existence: Does option persist? (above 1/φ³?)
        - righteousness: Is option aligned? (R value)
        - order: Historical success count
        
        OUTPUT (1 vector):
        - selected: The chosen action (movement)
    
    Context enables generalization - features like "near_cliff" can be
    shared across multiple state_keys, allowing learning to transfer.
    """
    timestamp: float
    state_key: str                      # What state we were in
    options: List[str]                  # What options were available
    selected: str                       # OUTPUT: The movement vector
    confidence: float                   # Ego's confidence (via Conscience)
    
    # The 5 scalar inputs (optional, for detailed tracking)
    heat: float = 0.0                   # 1. Heat (Σ) - magnitude
    polarity: int = 1                   # 2. Polarity (+/-) - direction
    existence_valid: bool = True        # 3. Existence (δ) - above 1/φ³?
    righteousness: float = 1.0          # 4. Righteousness (R) - alignment
    order_count: int = 0                # 5. Order (Q) - success history
    
    # Context and outcome
    context: Dict[str, Any] = field(default_factory=dict)
    outcome: Optional[str] = None
    success: Optional[bool] = None
    heat_delta: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "state_key": self.state_key,
            "options": self.options,
            "selected": self.selected,
            "confidence": self.confidence,
            "heat": self.heat,
            "polarity": self.polarity,
            "existence_valid": self.existence_valid,
            "righteousness": self.righteousness,
            "order_count": self.order_count,
            "context": self.context,
            "outcome": self.outcome,
            "success": self.success,
            "heat_delta": self.heat_delta
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Choice':
        return cls(
            timestamp=data["timestamp"],
            state_key=data["state_key"],
            options=data["options"],
            selected=data["selected"],
            confidence=data["confidence"],
            heat=data.get("heat", 0.0),
            polarity=data.get("polarity", 1),
            existence_valid=data.get("existence_valid", True),
            righteousness=data.get("righteousness", 1.0),
            order_count=data.get("order_count", 0),
            context=data.get("context", {}),
            outcome=data.get("outcome"),
            success=data.get("success"),
            heat_delta=data.get("heat_delta", 0.0)
        )


class ChoiceNode:
    """
    A node that records choices over time.
    
    Base class for DecisionNode (exit) and EnvironmentNode (entry).
    Saves choice history to filesystem.
    
    Structure:
        Node (choice_node)
        ├── Axis: "history" with Order
        │   └── Elements: previous choices (with heat = success)
        ├── Axis: "patterns" 
        │   └── State → Choice mappings with traversal count
        └── Connection to driver node
    """
    
    def __init__(self, name: str, manifold: 'Manifold', driver: Optional[Node] = None,
                 save_dir: str = None):
        self.name = name
        self.manifold = manifold
        self.driver = driver
        self.born = False  # Birth tracking
        self.save_dir = save_dir or get_decisions_path()
        
        # Core node (will be set during birth)
        self.node = None
        
        # In-memory choice buffer
        self.choices: List[Choice] = []
        self.max_history = 1000
        
        # Birth
        self._birth()
    
    def _birth(self):
        """Birth this choice node - create node and load history."""
        if self.born:
            logger.warning(f"ChoiceNode {self.name} already born, skipping")
            return
        
        # Create or find node
        existing = self.manifold.get_node_by_concept(self.name) if self.manifold else None
        if existing:
            self.node = existing
        else:
            # Place on hypersphere near north pole (task-level, "above" the equator)
            self.node = Node(
                concept=self.name,
                theta=math.pi / 6,  # 30° from north pole (task level)
                phi=0.0,
                radius=1.0,
                heat=K,
                polarity=1,
                existence="actual",
                righteousness=1.0,  # Righteous frame
                order=1
            )
            if self.manifold:
                self.manifold.add_node(self.node)
        
        # Ensure history axis exists
        if not self.node.get_axis("history"):
            history_axis = self.node.add_axis("history", f"{self.name}_history")
            history_axis.make_proper()  # Has Order
        
        # Load from filesystem
        self._load()
        
        self.born = True
        logger.debug(f"ChoiceNode {self.name} born")
    
    def _get_filepath(self) -> str:
        """Get path for this choice node's data."""
        os.makedirs(self.save_dir, exist_ok=True)
        safe_name = self.name.replace("/", "_").replace("\\", "_")
        return os.path.join(self.save_dir, f"{safe_name}.json")
    
    def _load(self):
        """Load choices from filesystem."""
        filepath = self._get_filepath()
        if os.path.exists(filepath):
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                self.choices = [Choice.from_dict(c) for c in data.get("choices", [])]
                logger.debug(f"Loaded {len(self.choices)} choices for {self.name}")
            except Exception as e:
                logger.warning(f"Failed to load choices for {self.name}: {e}")
    
    def save(self):
        """Save choices to filesystem."""
        filepath = self._get_filepath()
        data = {
            "name": self.name,
            "choices": [c.to_dict() for c in self.choices],
            "total_choices": len(self.choices),
            "success_count": sum(1 for c in self.choices if c.success),
            "saved_at": time()
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        logger.debug(f"Saved {len(self.choices)} choices for {self.name}")
    
    def record(self, choice: Choice):
        """
        Record a choice.
        
        Creates axes for:
        1. state_key → tracks this specific state
        2. state_key_action → tracks this action in this state  
        3. context_item_action → tracks this action with this context (GENERALIZATION)
        """
        self.choices.append(choice)
        if len(self.choices) > self.max_history:
            self.choices.pop(0)
        
        # Add to node's history Order
        history_axis = self.node.get_axis("history")
        if history_axis and history_axis.order:
            element = Element(
                node_id=f"choice_{len(history_axis.order.elements)}",
                index=1 if choice.success else 0
            )
            history_axis.order.elements.append(element)
        
        # Update pattern strength (state → choice mapping)
        pattern_axis = self.node.get_axis(choice.state_key)
        if pattern_axis is None:
            pattern_axis = self.node.add_axis(choice.state_key, f"{self.name}_{choice.state_key}")
        
        # Track which choices work in this state
        choice_key = f"{choice.state_key}_{choice.selected}"
        choice_axis = self.node.get_axis(choice_key)
        if choice_axis is None:
            choice_axis = self.node.add_axis(choice_key, f"{self.name}_{choice_key}")
            choice_axis.make_proper()
        
        # Record outcome in state_action Order
        if choice_axis.order is None:
            choice_axis.make_proper()
        choice_axis.order.elements.append(
            Element(node_id=f"{choice_key}_{len(choice_axis.order.elements)}", 
                    index=1 if choice.success else 0)
        )
        
        # Adjust traversal_count based on outcome
        if choice.success is not None:
            if choice.success:
                choice_axis.traversal_count += 1
        
        # CONTEXT-BASED LEARNING (enables generalization)
        # Create axes for context_item + action combinations
        for ctx_key, ctx_value in choice.context.items():
            # Only track "active" context items (truthy values)
            if ctx_value:
                context_choice_key = f"ctx:{ctx_key}_{choice.selected}"
                ctx_axis = self.node.get_axis(context_choice_key)
                if ctx_axis is None:
                    ctx_axis = self.node.add_axis(context_choice_key, f"{self.name}_{context_choice_key}")
                    ctx_axis.make_proper()
                
                # Record outcome
                if ctx_axis.order is None:
                    ctx_axis.make_proper()
                ctx_axis.order.elements.append(
                    Element(node_id=f"{context_choice_key}_{len(ctx_axis.order.elements)}",
                            index=1 if choice.success else 0)
                )
                
                # Adjust traversal_count
                if choice.success is not None:
                    if choice.success:
                        ctx_axis.traversal_count += 1
                
                logger.debug(f"Context learning: {ctx_key}+{choice.selected} → {'✓' if choice.success else '✗'}")
        
        # Auto-save
        self.save()
    
    def get_best_choice(self, state_key: str, options: List[str], 
                        context: Dict[str, Any] = None) -> Optional[str]:
        """
        Get the historically best choice for a state.
        
        Considers:
        1. State-specific history (state_key + action)
        2. Context-based history (context_item + action) - enables generalization
        
        Args:
            state_key: Current state
            options: Available options
            context: Current context (features like near_cliff)
            
        Returns:
            Best option based on combined history, or None if no history
        """
        if context is None:
            context = {}
        
        # Score each option
        option_scores = {}
        
        for option in options:
            score = 0.0
            weight = 0.0
            
            # 1. State-specific score (weighted 0.6)
            state_axis = self.node.get_axis(f"{state_key}_{option}")
            if state_axis:
                state_score = self._score_from_axis(state_axis)
                if state_score is not None:
                    score += state_score * 0.6
                    weight += 0.6
            
            # 2. Context-based scores (weighted 0.4 total, split among active contexts)
            active_contexts = [k for k, v in context.items() if v]
            if active_contexts:
                ctx_weight_each = 0.4 / len(active_contexts)
                for ctx_key in active_contexts:
                    ctx_axis = self.node.get_axis(f"ctx:{ctx_key}_{option}")
                    if ctx_axis:
                        ctx_score = self._score_from_axis(ctx_axis)
                        if ctx_score is not None:
                            score += ctx_score * ctx_weight_each
                            weight += ctx_weight_each
            
            if weight > 0:
                option_scores[option] = score / weight
        
        if not option_scores:
            return None
        
        # Return best scoring option
        return max(option_scores, key=option_scores.get)
    
    def _score_from_axis(self, axis: Axis) -> Optional[float]:
        """
        Score an action from its axis history.
        
        Uses:
        - Traversal count (experience)
        - Success rate from Order elements
        
        Returns:
            Score 0.0-1.0, or None if no data
        """
        if not axis.order or not axis.order.elements:
            # Use traversal count as proxy if no Order
            if axis.traversal_count > 0:
                return min(axis.traversal_count / 10.0, 1.0)
            return None
        
        # Calculate success rate
        successes = sum(1 for e in axis.order.elements if e.index == 1)
        total = len(axis.order.elements)
        success_rate = successes / total if total > 0 else 0.5
        
        # Weight by confidence (more samples = more confident)
        confidence = min(total / 10.0, 1.0)
        
        # Blend success rate with prior (0.5) based on confidence
        return success_rate * confidence + 0.5 * (1 - confidence)
    
    def get_success_rate(self, state_key: str = None, choice: str = None) -> float:
        """Get success rate for a state or specific choice."""
        relevant = self.choices
        
        if state_key:
            relevant = [c for c in relevant if c.state_key == state_key]
        if choice:
            relevant = [c for c in relevant if c.selected == choice]
        
        if not relevant:
            return 0.0
        
        successes = sum(1 for c in relevant if c.success)
        return successes / len(relevant)


class DecisionNode(ChoiceNode):
    """
    Movement (Lin) - The 6th Motion Function - Vectorized Output.
    
    This is the EXIT point where 5 scalar inputs become 1 vector output:
    
        INPUTS (5 scalars, validated by other files):
        ─────────────────────────────────────────────
        1. Heat (Σ)           ← psychology (magnitude)
        2. Polarity (+/-)     ← node_constants (direction)
        3. Existence (δ)      ← clock_node (persistence, 1/φ³ threshold)
        4. Righteousness (R)  ← nodes (alignment, R→0)
        5. Order (Q)          ← manifold (history, success count)
        
        OUTPUT (1 vector):
        ──────────────────
        6. Movement (Lin)     → THE DECISION (selected action)
    
    THE 5/6 CONFIDENCE THRESHOLD:
        - confidence > 5/6: EXPLOIT (5 scalars validated → use pattern)
        - confidence < 5/6: EXPLORE (still gathering validation)
        - t = 5K validations crosses threshold
    
    Saves decision history to decisions/ folder.
    """
    
    def __init__(self, manifold: 'Manifold', driver: Optional[Node] = None):
        super().__init__("pbai_decisions", manifold, driver, get_decisions_path())
        
        # Current decision context
        self.pending_choice: Optional[Choice] = None
        
        # ═══════════════════════════════════════════════════════════════════════
        # SHORT-TERM MEMORY (Working Memory)
        # ═══════════════════════════════════════════════════════════════════════
        # Rolling buffer of recent decisions - "what just happened"
        # Each entry: (state, action, outcome, success, t_K)
        # Size: 12 (movement constant - one per direction)
        self.short_term_memory: List[Dict[str, Any]] = []
        self.stm_capacity: int = 12  # Movement constant
    
    def _record_to_stm(self, state: str, action: str, outcome: str, success: bool) -> None:
        """Record completed decision to short-term memory."""
        t_K = self.manifold.get_time() if self.manifold else 0
        
        self.short_term_memory.append({
            "state": state,
            "action": action,
            "outcome": outcome,
            "success": success,
            "t_K": t_K
        })
        
        # Maintain capacity (FIFO)
        if len(self.short_term_memory) > self.stm_capacity:
            self.short_term_memory.pop(0)
    
    def get_recent_actions(self, n: int = 3) -> List[str]:
        """Get last N actions taken."""
        return [m["action"] for m in self.short_term_memory[-n:]]
    
    def get_recent_states(self, n: int = 3) -> List[str]:
        """Get last N states visited."""
        return [m["state"] for m in self.short_term_memory[-n:]]
    
    def get_recent_outcomes(self, n: int = 3) -> List[bool]:
        """Get last N success/fail outcomes."""
        return [m["success"] for m in self.short_term_memory[-n:]]
    
    def get_stm_context(self) -> Dict[str, Any]:
        """Get short-term memory as context dict for decision-making."""
        if not self.short_term_memory:
            return {}
        
        recent = self.short_term_memory[-3:]  # Last 3
        return {
            "stm_actions": [m["action"] for m in recent],
            "stm_states": [m["state"] for m in recent],
            "stm_successes": [m["success"] for m in recent],
            "stm_streak": self._get_streak(),
            "stm_last_action": recent[-1]["action"] if recent else None,
            "stm_last_success": recent[-1]["success"] if recent else None,
        }
    
    def _get_streak(self) -> int:
        """Get current success/fail streak. Positive = successes, negative = failures."""
        if not self.short_term_memory:
            return 0
        
        streak = 0
        last_success = self.short_term_memory[-1]["success"]
        
        for m in reversed(self.short_term_memory):
            if m["success"] == last_success:
                streak += 1 if last_success else -1
            else:
                break
        
        return streak
    
    def begin_decision(self, state_key: str, options: List[str], confidence: float,
                       context: Dict[str, Any] = None) -> Choice:
        """
        Begin a decision - gather the 5 scalar inputs.
        
        Collects validation from all 5 scalar motion functions:
        1. Heat (Σ) - from psychology nodes
        2. Polarity (+/-) - from state context
        3. Existence (δ) - is state above 1/φ³?
        4. Righteousness (R) - state alignment
        5. Order (Q) - historical success count
        
        Args:
            state_key: Current state from perception
            options: Available choices
            confidence: Ego's confidence (via Conscience mediation)
            context: Features for generalization
        """
        # Gather the 5 scalar inputs
        heat = 0.0
        polarity = 1
        existence_valid = True
        righteousness = 1.0
        order_count = 0
        
        if self.manifold:
            # 1. Heat (Σ) - from Ego (decision-maker)
            if self.manifold.ego_node:
                heat = self.manifold.ego_node.heat
            
            # 2-5. From state node if it exists
            state_node = self.manifold.get_node_by_concept(state_key)
            if state_node:
                # 2. Polarity (+/-) - direction
                polarity = state_node.polarity
                
                # 3. Existence (δ) - above 1/φ³?
                existence_valid = state_node.existence == EXISTENCE_ACTUAL
                
                # 4. Righteousness (R) - alignment
                righteousness = state_node.righteousness
                
                # 5. Order (Q) - count choices for this state
                choice_axis = self.node.get_axis(state_key) if self.node else None
                if choice_axis:
                    order_count = choice_axis.traversal_count
        
        # Merge short-term memory context with provided context
        full_context = self.get_stm_context()
        if context:
            full_context.update(context)
        
        self.pending_choice = Choice(
            timestamp=time(),
            state_key=state_key,
            options=options,
            selected="",
            confidence=confidence,
            heat=heat,
            polarity=polarity,
            existence_valid=existence_valid,
            righteousness=righteousness,
            order_count=order_count,
            context=full_context
        )
        return self.pending_choice
    
    def commit_decision(self, selected: str) -> Choice:
        """
        Commit to a decision.
        
        Args:
            selected: The chosen option
        """
        if self.pending_choice is None:
            # Create choice retroactively
            self.pending_choice = Choice(
                timestamp=time(),
                state_key="unknown",
                options=[selected],
                selected=selected,
                confidence=K
            )
        else:
            self.pending_choice.selected = selected
        
        logger.debug(f"Decision: {selected} (confidence={self.pending_choice.confidence:.2f})")
        return self.pending_choice
    
    def complete_decision(self, outcome: str, success: bool, heat_delta: float = 0.0):
        """
        Complete a decision with its outcome.
        
        This is called after the action has been executed and we know the result.
        The outcome is propagated to Conscience to build confidence over time.
        """
        if self.pending_choice is None:
            logger.warning("No pending choice to complete")
            return
        
        self.pending_choice.outcome = outcome
        self.pending_choice.success = success
        self.pending_choice.heat_delta = heat_delta
        
        # Record in history (auto-saves)
        self.record(self.pending_choice)
        
        # Update driver if connected
        if self.driver and success:
            self.driver.reward(heat_delta, achieved=outcome)
        elif self.driver and not success:
            self.driver.punish(abs(heat_delta))
        
        # === CONSCIENCE PROPAGATION ===
        # Validated patterns (successes) become Conscience axes
        # This is how confidence rises with learning
        if self.manifold:
            conscience = self.manifold.conscience_node
            ego = self.manifold.ego_node
            
            if conscience and ego:
                # Pattern key: state→action (what Conscience validates)
                state_key = self.pending_choice.state_key
                action = self.pending_choice.selected
                pattern_key = f"{state_key}→{action}"
                
                # Get or create axis on Conscience
                axis = conscience.get_axis(pattern_key)
                
                if axis:
                    # Already have an axis - strengthen with success/failure
                    if success:
                        axis.strengthen()  # Reinforce validation
                        logger.debug(f"Conscience: reinforced {pattern_key} (traversals={axis.traversal_count})")
                    else:
                        # Failure: flip polarity to indicate correction needed
                        if axis.polarity > 0 and axis.traversal_count > 1:
                            axis.polarity = -1  # Now indicates "corrected" pattern
                        logger.debug(f"Conscience: corrected {pattern_key}")
                else:
                    # New pattern - create axis if successful
                    if success:
                        axis = conscience.add_axis(pattern_key, ego.id, polarity=1)
                        axis.strengthen()  # First validation
                        logger.debug(f"Conscience: new validation {pattern_key}")
        
        logger.info(f"Decision complete: {self.pending_choice.selected} → {outcome} ({'✓' if success else '✗'})")
        
        # === SHORT-TERM MEMORY ===
        # Record this decision to working memory
        self._record_to_stm(
            state=self.pending_choice.state_key,
            action=self.pending_choice.selected,
            outcome=outcome,
            success=success
        )
        
        self.pending_choice = None
    
    def decide(self, state_key: str, options: List[str], 
               confidence: float = None, goal: str = None,
               context: Dict[str, Any] = None) -> str:
        """
        Make a decision - the 6th motion function (vectorized movement).
        
        NOW WITH FULL β→δ→Γ→α→ζ PROCESSING CYCLE:
        
        "Life is motion that mixes paths (β), collapses events (δ), 
         multiplies structure (Γ), couples interaction (α), 
         normalizes meaning (ζ), and then chooses direction."
        
        Processing flow:
            1. Get confidence from Conscience (5/6 threshold)
            2. β processing (Identity): Create path superposition
            3. Γ processing (Conscience): Count arrangements
            4. α processing (Conscience): Get coupling strengths
            5. ζ processing (Conscience): Normalize for direction
            6. δ processing (Ego): Collapse to selection
            7. Movement (Lin): Output direction vector
        
        The 5/6 Confidence Threshold:
            confidence > 5/6 (0.8333) → EXPLOIT: Use validated pattern
            confidence < 5/6         → EXPLORE: Probabilistic collapse
        
        Args:
            state_key: Current state
            options: Available choices
            confidence: Ego's confidence (via Conscience mediation)
            goal: Optional goal to achieve
            context: Features for generalization
            
        Returns:
            Selected option (the movement vector)
        """
        # Get confidence from manifold if not provided
        if confidence is None:
            confidence = self.manifold.get_confidence(state_key) if self.manifold else 0.0
        if context is None:
            context = {}
        
        # Begin decision with context
        self.begin_decision(state_key, options, confidence, context)
        
        logger.debug(f"Decision: state={state_key}, options={options}, "
                    f"confidence={confidence:.3f}, threshold={CONFIDENCE_EXPLOIT_THRESHOLD:.3f}")
        
        # ═══════════════════════════════════════════════════════════════════════
        # FULL β→δ→Γ→α→ζ PROCESSING CYCLE
        # ═══════════════════════════════════════════════════════════════════════
        
        # Process through psychology nodes (Identity → Ego via Conscience)
        selected, details = self._process_through_psychology(state_key, options, confidence)
        
        if selected:
            logger.info(f"{details['mode'].upper()} decision: {selected} "
                       f"(confidence={confidence:.3f}, "
                       f"significance={details['significance'].get(selected, 0):.3f})")
            return self.commit_decision(selected).selected
        
        # ═══════════════════════════════════════════════════════════════════════
        # FALLBACK PATHS (if psychology processing returns None)
        # ═══════════════════════════════════════════════════════════════════════
        
        # 1. Check driver for goal-directed plan
        if self.driver and goal:
            decision = self.driver.think(goal)
            if isinstance(decision, ActionPlan):
                self.driver.start_plan(decision)
                decision = self.driver.think()
            if isinstance(decision, MotorAction):
                for opt in options:
                    if opt in str(decision):
                        return self.commit_decision(opt).selected
        
        # 2. Try wave function collapse with manifold nodes
        selected = self._decide_by_collapse(state_key, options, context)
        if selected:
            return self.commit_decision(selected).selected
        
        # 3. Check for best historical choice
        best = self.get_best_choice(state_key, options, context)
        if best:
            return self.commit_decision(best).selected
        
        # 4. Check driver patterns
        if self.driver:
            driver_decision = self.driver.think()
            if driver_decision:
                for opt in options:
                    if opt in str(driver_decision):
                        return self.commit_decision(opt).selected
        
        # 5. Default: Random exploration
        import random
        selected = random.choice(options) if len(options) > 1 else options[0]
        logger.info(f"RANDOM decision: {selected} (no pattern found, confidence={confidence:.3f})")
        return self.commit_decision(selected).selected
    
    def _decide_by_collapse(self, state_key: str, options: List[str],
                           context: Dict[str, Any] = None) -> Optional[str]:
        """
        Decide using wave function collapse and cluster correlation.
        
        1. Find candidate nodes (related to state_key)
        2. Collapse to find CENTER (R→0)
        3. Correlate cluster from center (current + historical + novel)
        4. Select option: exploit Order if exists, else explore
        
        Returns:
            Selected option, or None if no cluster context
        """
        from .node_constants import collapse_wave_function, correlate_cluster, select_from_cluster
        
        if not self.manifold:
            return None
        
        # Find candidate nodes related to current state
        candidates = self._find_candidate_nodes(state_key, context)
        
        if not candidates:
            return None
        
        # COLLAPSE: Find the center (node with R closest to 0)
        center_idx = collapse_wave_function(candidates, self.manifold)
        if center_idx < 0:
            return None
        
        center_node = candidates[center_idx]
        logger.debug(f"Collapse found center: {center_node.concept} (R={center_node.righteousness:.3f})")
        
        # CORRELATE: Get cluster (current + historical + novel)
        cluster = correlate_cluster(center_node, self.manifold, max_depth=3)
        
        if not cluster.get('all'):
            return None
        
        logger.debug(f"Cluster: {len(cluster['current'])} current, "
                    f"{len(cluster['historical'])} historical, "
                    f"{len(cluster['novel'])} novel")
        
        # SELECT: Exploit Order if exists, else explore
        selected_idx, reason = select_from_cluster(options, cluster, self.manifold)
        
        if selected_idx >= 0 and selected_idx < len(options):
            selected = options[selected_idx]
            logger.info(f"Collapse decision: {selected} ({reason}, cluster={len(cluster['all'])})")
            return selected
        
        return None
    
    def _find_candidate_nodes(self, state_key: str,
                              context: Dict[str, Any] = None) -> List[Node]:
        """
        Find nodes related to current state for collapse.

        Candidates are nodes that might be the CENTER:
        - The current state node (if exists)
        - Angular neighbors on the hypersphere (proximity-based)
        - Context-related nodes
        - Nodes connected to current decision node
        """
        if not self.manifold:
            return []

        candidates = []
        seen_ids = set()

        # 1. Current state node
        state_node = self.manifold.get_node_by_concept(state_key)
        if state_node:
            candidates.append(state_node)
            seen_ids.add(state_node.id)

        # 2. Angular proximity on hypersphere (replaces string prefix matching)
        # Find nodes near the state node's position
        if state_node:
            target_sp = SpherePosition(theta=state_node.theta, phi=state_node.phi)
            proximity_threshold = math.pi / 4  # 45° neighborhood
            for node in self.manifold.nodes.values():
                if node.id in seen_ids:
                    continue
                if node.existence == "archived":
                    continue
                node_sp = SpherePosition(theta=node.theta, phi=node.phi)
                if angular_distance(target_sp, node_sp) < proximity_threshold:
                    candidates.append(node)
                    seen_ids.add(node.id)

        # 3. Context-related nodes
        if context:
            for ctx_key, ctx_value in context.items():
                if ctx_value:  # Only active contexts
                    ctx_node = self.manifold.get_node_by_concept(f"ctx:{ctx_key}")
                    if ctx_node and ctx_node.id not in seen_ids:
                        candidates.append(ctx_node)
                        seen_ids.add(ctx_node.id)

        # 4. Nodes connected to current decision node
        for axis in self.node.frame.axes.values():
            if axis.target_id and axis.target_id not in seen_ids:
                target = self.manifold.get_node(axis.target_id)
                if target:
                    candidates.append(target)
                    seen_ids.add(target.id)

        return candidates
    
    # ═══════════════════════════════════════════════════════════════════════════
    # β→δ→Γ→α→ζ PROCESSING CYCLE - The "Life" Functions
    # ═══════════════════════════════════════════════════════════════════════════
    #
    # "Life is motion that mixes paths (β), collapses events (δ), 
    #  multiplies structure (Γ), couples interaction (α), 
    #  normalizes meaning (ζ), and then chooses direction."
    #
    # These functions process through psychology nodes before direction selection.
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _create_superposition(self, options: List[str], state_key: str) -> Dict[str, Any]:
        """
        β processing: Create weighted superposition of choice paths.
        
        This is Identity (Id) node processing - path mixing based on
        prior success/failure history using the Euler Beta function.
        
        B(a,b) = Γ(a)Γ(b) / Γ(a+b)
        
        Where:
            a = successes + 1 (Laplace smoothing)
            b = failures + 1
        
        Args:
            options: Available choice paths
            state_key: Current state for history lookup
            
        Returns:
            {
                'paths': List of option strings,
                'weights': Normalized β weights,
                'psi': Complex wave function amplitude
            }
        """
        from .node_constants import euler_beta, PHI
        import math
        
        paths = []
        weights = []
        
        for option in options:
            # Get history for this option in this state
            history = self._get_option_history(state_key, option)
            successes = sum(1 for h in history if h.get('success', False))
            failures = len(history) - successes
            
            # Euler beta weights by success/failure ratio
            # Laplace smoothing: add 1 to both to handle zero cases
            beta_weight = euler_beta(successes + 1, failures + 1)
            paths.append(option)
            weights.append(beta_weight)
        
        # Normalize weights to sum to 1
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]
        else:
            # Uniform if no history
            weights = [1.0 / len(options)] * len(options)
        
        # Create wave function (complex amplitude)
        # Each path contributes a phase-shifted component using golden angle
        psi = complex(0, 0)
        for i, w in enumerate(weights):
            phase = i * PHI  # Golden angle separation for optimal phase coverage
            psi += w * complex(math.cos(phase), math.sin(phase))
        
        return {
            'paths': paths,
            'weights': weights,
            'psi': psi,
            'state_key': state_key
        }
    
    def _get_option_history(self, state_key: str, option: str) -> List[Dict]:
        """Get history of outcomes for an option in a state."""
        history = []
        
        # Check local choice history
        for choice in self.choices:
            if choice.state_key == state_key and choice.selected == option:
                history.append({
                    'success': choice.success,
                    'heat_delta': choice.heat_delta,
                    'timestamp': choice.timestamp
                })
        
        # Also check manifold's decision history if available
        if self.manifold and hasattr(self.manifold, 'get_decision_history'):
            manifold_history = self.manifold.get_decision_history(state_key, option)
            history.extend(manifold_history)
        
        return history
    
    def _collapse_superposition(self, superposition: Dict[str, Any]) -> str:
        """
        δ processing: Collapse superposition to single outcome.
        
        This is Ego node processing - actualization from potential.
        Uses the Born rule: P(option_i) = |ψ_i|² / Σ|ψ_j|²
        
        The collapse is probabilistic but weighted by prior success.
        Higher weighted options are more likely to be selected.
        
        Args:
            superposition: Output from _create_superposition()
            
        Returns:
            Selected option string
        """
        import random
        
        paths = superposition['paths']
        weights = superposition['weights']
        
        if not paths:
            return None
        
        if len(paths) == 1:
            return paths[0]
        
        # Born rule: probability proportional to |weight|²
        # (weights are already normalized amplitudes)
        probabilities = [w * w for w in weights]
        total_prob = sum(probabilities)
        
        if total_prob > 0:
            probabilities = [p / total_prob for p in probabilities]
        else:
            probabilities = [1.0 / len(paths)] * len(paths)
        
        # Probabilistic collapse (quantum measurement)
        r = random.random()
        cumulative = 0.0
        for i, p in enumerate(probabilities):
            cumulative += p
            if r <= cumulative:
                return paths[i]
        
        # Fallback to last option
        return paths[-1]
    
    def _get_gamma_scores(self, options: List[str], state_key: str) -> Dict[str, float]:
        """
        Γ processing: Get arrangement counts for each option.
        
        This is part of Conscience processing - counting structural multiplicity.
        Options with more structural arrangements (higher Γ) have higher entropy.
        
        Args:
            options: Available choices
            state_key: Current state
            
        Returns:
            Dict mapping options to their Γ (arrangement count) scores
        """
        scores = {}
        
        for option in options:
            # Look for Order associated with this option
            gamma_score = 1  # Default: single arrangement
            
            # Check if option has an axis with Order
            if self.manifold:
                option_node = self.manifold.get_node_by_concept(f"{state_key}:{option}")
                if option_node:
                    # Sum Γ contributions from all ordered axes
                    for axis in option_node.frame.axes.values():
                        if axis.order:
                            gamma_score *= axis.order.count_arrangements()
            
            scores[option] = gamma_score
        
        return scores
    
    def _get_alpha_couplings(self, options: List[str], state_key: str) -> Dict[str, float]:
        """
        α processing: Get coupling strengths for each option.
        
        This is part of Conscience processing - how strongly each option
        is coupled to learned experience (memory influence).
        
        Args:
            options: Available choices
            state_key: Current state
            
        Returns:
            Dict mapping options to their α (coupling) scores
        """
        from .node_constants import FINE_STRUCTURE_CONSTANT
        
        couplings = {}
        
        for option in options:
            # Base coupling is fine-structure constant
            coupling = FINE_STRUCTURE_CONSTANT
            
            # Check if option has traversal history
            if self.manifold:
                option_node = self.manifold.get_node_by_concept(f"{state_key}:{option}")
                if option_node:
                    # Average coupling across all axes
                    axis_couplings = []
                    for axis in option_node.frame.axes.values():
                        axis_couplings.append(axis.get_coupling_strength())
                    if axis_couplings:
                        coupling = sum(axis_couplings) / len(axis_couplings)
            
            couplings[option] = coupling
        
        return couplings
    
    def _zeta_normalize(self, options: List[str], superposition: Dict[str, Any],
                        gamma_scores: Dict[str, float], 
                        alpha_couplings: Dict[str, float]) -> Dict[str, float]:
        """
        ζ processing: Normalize all inputs for direction selection.
        
        This completes Conscience processing before Movement (direction).
        Combines heat × γ × α for each option, then normalizes using
        zeta-like suppression to make infinities finite and rank significance.
        
        The formula: significance = raw_score / (order^1.5)
        This prevents high-frequency options from dominating while
        preserving meaningful ordering.
        
        Args:
            options: Available choices
            superposition: β output (weights)
            gamma_scores: Γ output (arrangement counts)
            alpha_couplings: α output (coupling strengths)
            
        Returns:
            Dict mapping options to normalized significance scores in [0,1]
        """
        import math
        
        raw_scores = {}
        
        for i, option in enumerate(options):
            # Get components
            beta_weight = superposition['weights'][i] if i < len(superposition['weights']) else 0.0
            gamma = gamma_scores.get(option, 1)
            alpha = alpha_couplings.get(option, 0.0)
            
            # Raw score: heat (from β) × structure (Γ) × coupling (α)
            raw_score = beta_weight * math.log(1 + gamma) * alpha
            raw_scores[option] = raw_score
        
        # ζ normalization: suppress high-order contributions
        # significance = raw / (rank^1.5)
        # This makes infinities finite while preserving relative ordering
        
        # Rank by raw score
        ranked = sorted(raw_scores.items(), key=lambda x: x[1], reverse=True)
        
        normalized = {}
        for rank, (option, raw) in enumerate(ranked, start=1):
            # ζ suppression factor
            suppression = rank ** 1.5
            normalized[option] = raw / suppression if suppression > 0 else raw
        
        # Scale to [0, 1]
        max_score = max(normalized.values()) if normalized else 1.0
        if max_score > 0:
            for option in normalized:
                normalized[option] /= max_score
        
        # === STM ADJUSTMENT ===
        # Penalize recently failed actions, boost recently successful ones
        normalized = self._apply_stm_adjustment(normalized)
        
        return normalized
    
    def _apply_stm_adjustment(self, significance: Dict[str, float]) -> Dict[str, float]:
        """
        Adjust significance based on short-term memory.
        
        - Recent failures: penalize that action (avoid immediate repetition)
        - Recent successes: slight boost (reinforce working patterns)
        - Streak multiplier: stronger effect with consecutive same outcomes
        
        This implements "don't repeat what just failed" intuition.
        """
        if not self.short_term_memory:
            return significance
        
        adjusted = dict(significance)
        
        # Look at last 3 decisions
        recent = self.short_term_memory[-3:]
        
        for memory in recent:
            action = memory["action"]
            success = memory["success"]
            
            if action in adjusted:
                if success:
                    # Slight boost for recent successes (1.1x)
                    adjusted[action] *= 1.1
                else:
                    # Stronger penalty for recent failures (0.5x)
                    adjusted[action] *= 0.5
        
        # Extra penalty for immediate repeat of last failure
        if recent and not recent[-1]["success"]:
            last_failed = recent[-1]["action"]
            if last_failed in adjusted:
                adjusted[last_failed] *= 0.3  # Strong penalty
        
        # Extra boost for streak
        streak = self._get_streak()
        if streak >= 2 and recent:
            # On success streak - boost the winning action
            streaking_action = recent[-1]["action"]
            if streaking_action in adjusted:
                adjusted[streaking_action] *= 1.2
        elif streak <= -2 and recent:
            # On failure streak - heavily penalize to force exploration
            failing_action = recent[-1]["action"]
            if failing_action in adjusted:
                adjusted[failing_action] *= 0.1
        
        # Re-normalize to [0, 1]
        max_score = max(adjusted.values()) if adjusted else 1.0
        if max_score > 0:
            for option in adjusted:
                adjusted[option] /= max_score
        
        return adjusted
    
    def _process_through_psychology(self, state_key: str, options: List[str],
                                     confidence: float) -> Tuple[str, Dict[str, Any]]:
        """
        Complete β→δ→Γ→α→ζ processing cycle through psychology nodes.
        
        This is the "Life" processing that distinguishes living motion
        from mere mechanical state changes.
        
        Flow:
            Identity (Id): β - mix paths by prior success
            Ego: δ - collapse to selection (exploit if confident)
            Conscience: Γ,α,ζ - validate, couple, normalize
            
        Args:
            state_key: Current state
            options: Available choices
            confidence: Conscience's validation level
            
        Returns:
            (selected_option, processing_details)
        """
        # β processing (Identity): Create superposition
        superposition = self._create_superposition(options, state_key)
        
        # Γ processing (Conscience): Count arrangements
        gamma_scores = self._get_gamma_scores(options, state_key)
        
        # α processing (Conscience): Get coupling strengths
        alpha_couplings = self._get_alpha_couplings(options, state_key)
        
        # ζ processing (Conscience): Normalize for direction
        significance = self._zeta_normalize(options, superposition, 
                                            gamma_scores, alpha_couplings)
        
        # δ processing (Ego): Collapse based on confidence
        if confidence > CONFIDENCE_EXPLOIT_THRESHOLD:
            # EXPLOIT: Choose highest significance (deterministic)
            selected = max(significance.items(), key=lambda x: x[1])[0]
            mode = "exploit"
        else:
            # EXPLORE: Probabilistic collapse weighted by significance
            # Modify superposition weights by significance
            modified_weights = []
            for option in superposition['paths']:
                beta_w = superposition['weights'][superposition['paths'].index(option)]
                sig = significance.get(option, 0.0)
                modified_weights.append(beta_w * (1 + sig))  # Bias toward significant
            
            # Normalize
            total = sum(modified_weights)
            if total > 0:
                modified_weights = [w / total for w in modified_weights]
            
            modified_super = {
                'paths': superposition['paths'],
                'weights': modified_weights,
                'psi': superposition['psi']
            }
            selected = self._collapse_superposition(modified_super)
            mode = "explore"
        
        details = {
            'mode': mode,
            'confidence': confidence,
            'superposition': superposition,
            'gamma_scores': gamma_scores,
            'alpha_couplings': alpha_couplings,
            'significance': significance,
            'selected': selected
        }
        
        logger.debug(f"Psychology processing: {mode} → {selected} "
                    f"(conf={confidence:.3f}, sig={significance.get(selected, 0):.3f})")
        
        return selected, details


class EnvironmentNode(ChoiceNode):
    """
    Records perceptions received through environment.py (the ENTRY point).
    
    Note: The actual ENTRY logic is in drivers/environment.py
    This class just records and manages perception history.
    
    Connects external perceptions to:
    1. Identity (what exists)
    2. driver node states (learned state patterns)
    3. Choice history (what's been seen before)
    """
    
    def __init__(self, manifold: 'Manifold', driver: Optional[Node] = None):
        # Perceptions also save to decisions/ folder (they're choices the environment made)
        super().__init__("pbai_perceptions", manifold, driver, get_decisions_path())
    
    def receive(self, state_key: str, description: str, novelty: float = 0.0) -> Choice:
        """
        Receive a perception from the environment.
        
        Records what was perceived as a "choice" (the environment chose to show us this).
        
        Args:
            state_key: Unique key for this state
            description: Human-readable description
            novelty: How novel this perception is (affects heat)
        """
        # Perception is like a choice the environment made
        choice = Choice(
            timestamp=time(),
            state_key=state_key,
            options=[state_key],  # Only one "option" - what we perceived
            selected=state_key,
            confidence=novelty,  # Novelty = confidence in importance
            success=True,  # Perception is always "successful"
            heat_delta=novelty * THRESHOLD_ORDER
        )
        
        self.record(choice)
        
        # Driver node handling - now just tags, no .see() method
        # The sensor report would be processed elsewhere
        
        # Update Identity in manifold
        if self.manifold and hasattr(self.manifold, 'update_identity'):
            is_novel = novelty > 0.5
            self.manifold.update_identity(state_key, heat_delta=novelty, known=not is_novel)
        
        logger.debug(f"Perception: {state_key} (novelty={novelty:.2f})")
        return choice
    
    def is_familiar(self, state_key: str) -> bool:
        """Check if we've seen this state before."""
        return any(c.state_key == state_key for c in self.choices)
    
    def get_familiarity(self, state_key: str) -> float:
        """Get how familiar a state is (0-1)."""
        count = sum(1 for c in self.choices if c.state_key == state_key)
        # Diminishing returns on familiarity
        return min(1.0, count / 10.0)


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION - Connecting Entry and Exit
# ═══════════════════════════════════════════════════════════════════════════════

class PBAILoop:
    """
    The complete perception → decision → action loop.
    
    Entry: drivers/environment.py (perceptions come in)
    Exit: DecisionNode (actions go out)
    
    Connects:
    - EnvironmentNode (perception recording)
    - DecisionNode (exit/action)
    - driver node (learning/execution)
    - Manifold (core psychology)
    """
    
    def __init__(self, manifold: 'Manifold', driver: Optional[Node]):
        self.manifold = manifold
        self.driver = driver
        self.born = False  # Birth tracking
        
        # Entry and exit nodes (will be set during birth)
        self.entry = None
        self.exit = None
        
        # Birth
        self._birth()
    
    def _birth(self):
        """Birth this PBAI loop - create entry and exit nodes."""
        if self.born:
            logger.warning("PBAILoop already born, skipping")
            return
        
        # Create entry and exit nodes (both save to decisions/)
        self.entry = EnvironmentNode(self.manifold, self.driver)
        self.exit = DecisionNode(self.manifold, self.driver)
        
        self.born = True
        logger.info(f"PBAILoop born with driver: {self.driver.name}")
    
    def step(self, perception_key: str, perception_desc: str,
             options: List[str], goal: str = None) -> Tuple[str, Choice]:
        """
        One complete loop iteration.
        
        Args:
            perception_key: State key for current perception
            perception_desc: Human description of perception
            options: Available action options
            goal: Optional goal to work toward
            
        Returns:
            (selected_action, decision_choice)
        """
        # ENTRY: Receive perception
        novelty = 0.0 if self.entry.is_familiar(perception_key) else 1.0
        self.entry.receive(perception_key, perception_desc, novelty)
        
        # Get confidence from manifold
        confidence = self.manifold.get_confidence() if self.manifold else K
        
        # EXIT: Make decision
        selected = self.exit.decide(perception_key, options, confidence, goal)
        
        return selected, self.exit.pending_choice
    
    def complete(self, outcome: str, success: bool, heat_delta: float = 0.0):
        """Complete the current decision with outcome."""
        self.exit.complete_decision(outcome, success, heat_delta)
    
    def save(self):
        """Save state."""
        if self.driver:
            self.driver.save()
        if self.manifold:
            self.manifold.save_growth_map()
        # Choice nodes auto-save on record
