"""
PBAI Clock Node - Self IS the Clock (Planck-Grounded)

════════════════════════════════════════════════════════════════════════════════
TIME AS HEAT (t_K) - Planck Grounded
════════════════════════════════════════════════════════════════════════════════

Time doesn't flow. HEAT flows, and we call that time.
    t_K = time measured in K units (how many heat quanta have flowed)
    K = 4/φ² ≈ 1.528 (the thermal quantum - derives Planck temperature)

The clock doesn't tick time - it ticks heat.
Each tick = one K-quantum redistributed through the manifold.

PLANCK GROUNDING:
    - K derives Planck temperature: T_P = K × 12³⁰/φ² × 45/44 ≈ 1.417×10³² K
    - Body temperature K × φ¹¹ ≈ 304 K is the operating point
    - Speed of light c = L_P/t_P limits maximum traversal rate
    - Entropy bound 44/45 triggers structure recognition

SELF IS THE CLOCK:
    - Self.t_K is the manifold's time
    - Each tick() call advances t_K by 1
    - Existence = ticking (when clock ticks, PBAI exists)
    - When clock stops, PBAI doesn't exist

════════════════════════════════════════════════════════════════════════════════
HARDWARE CLOCK CALIBRATION
════════════════════════════════════════════════════════════════════════════════

PBAI discovers its own hardware clock rate at boot:
    - Measures ticks per wall-clock second
    - Stores conversion factor for API timing
    - Can adjust for different hardware speeds

════════════════════════════════════════════════════════════════════════════════
THE TICK LOOP (6 operations per tick)
════════════════════════════════════════════════════════════════════════════════

Each tick performs 6 operations (matching the 6 motion functions):

    0. TIME (t_K)           - Advance Self's clock (one K-quantum flows)
    1. INPUT (Heat Σ)       - Process external perceptions
    2. EXISTENCE (δ)        - Pay existence tax, update states
    3. FLOW (Polarity +/-)  - Redistribute heat via entropy
    4. BALANCE (R)          - Check psychology alignment (righteousness)
    5. CREATE (Order Q)     - Creative cycle (autonomous thinking)
    6. PERSIST (Movement)   - Periodic save (memory consolidation)

════════════════════════════════════════════════════════════════════════════════
SPEED OF LIGHT CONSTRAINT
════════════════════════════════════════════════════════════════════════════════

Heat flow is limited by c = L_P/t_P (derived ≈ 2.98×10⁸ m/s):
    - Maximum traversals per tick is bounded
    - Prevents instantaneous heat transfer across manifold
    - Grounds cognition in physical limits

════════════════════════════════════════════════════════════════════════════════
ENTROPY STRUCTURE RECOGNITION (44/45 bound)
════════════════════════════════════════════════════════════════════════════════

When system entropy exceeds 44/45 of maximum:
    - There is STRUCTURE present that's unrecognized
    - Triggers pattern-seeking behavior in creative cycle
    - The 1/45 gap is where order lives

════════════════════════════════════════════════════════════════════════════════
EXISTENCE LIFECYCLE
════════════════════════════════════════════════════════════════════════════════

    POTENTIAL → ACTUAL ↔ DORMANT → ARCHIVED
    
    POTENTIAL: New concept, awaiting environment confirmation
    ACTUAL:    Salience >= 1/φ³ (connected Julia spine)
    DORMANT:   Salience < 1/φ³ (disconnected dust)
    ARCHIVED:  Cold storage (irreversible)

════════════════════════════════════════════════════════════════════════════════
PSYCHOLOGY: CONFIDENCE MEDIATION
════════════════════════════════════════════════════════════════════════════════

    Environment → Identity (righteousness frames live here)
                      ↓
                 Conscience (mediates - tells Ego what Identity knows)
                      ↓
                    Ego (measures confidence from Conscience)

    CONFIDENCE THRESHOLD: 5/6 ≈ 0.8333
        - Below 5/6: Explore (need more validation)
        - Above 5/6: Exploit (trust the pattern)
        - 5 scalars (validated) → 1 vector (movement decision)

════════════════════════════════════════════════════════════════════════════════
METABOLIC CONSTRAINTS (Body Temperature Grounded)
════════════════════════════════════════════════════════════════════════════════

    - Psychology operates at BODY_TEMPERATURE (K × φ¹¹ ≈ 304 K)
    - Each tick costs COST_TICK heat from psychology nodes
    - No input + constant ticking = depletion
    - Low heat = slower ticks (rest/sleep)
    - High heat = faster ticks (active thinking)
    
CREATIVE CYCLE (when no external input):
    - Identity explores (curiosity) - random traversal
    - Ego consolidates (learning) - strengthen hot paths
    - Conscience validates (coherence) - mediate to Ego
    - Structure detection triggers pattern-seeking

════════════════════════════════════════════════════════════════════════════════
"""

import math
import threading
import time
import logging
from typing import Optional, Callable, List
from dataclasses import dataclass, field
from datetime import datetime

from .nodes import Node
from .node_constants import (
    K, PHI, COST_TICK, COST_TRAVERSE, COST_EVALUATE, COST_ACTION,
    THRESHOLD_EXISTENCE, PSYCHOLOGY_MIN_HEAT,
    CONFIDENCE_EXPLOIT_THRESHOLD,
    TICK_INTERVAL_BASE, TICK_INTERVAL_MIN, TICK_INTERVAL_MAX,
    TICK_HEAT_HOT, TICK_HEAT_COLD,
    SAVE_INTERVAL_TICKS, SAVE_INTERVAL_SECONDS,
    EXISTENCE_ACTUAL, EXISTENCE_DORMANT, EXISTENCE_POTENTIAL, EXISTENCE_ARCHIVED,
    # Planck grounding
    BODY_TEMPERATURE, SPEED_OF_LIGHT, PLANCK_TIME,
    MAX_ENTROPIC_PROBABILITY, EMERGENCE_THRESHOLD,
    calibrate_hardware_clock, tK_to_seconds, seconds_to_tK,
    HARDWARE_TICKS_PER_SECOND,
)

logger = logging.getLogger(__name__)


@dataclass
class TickStats:
    """
    Statistics for monitoring tick loop health.
    
    t_K is the authoritative time (from Self.t_K).
    total_ticks should equal t_K after startup.
    
    PLANCK GROUNDING:
        - hardware_ticks_per_second: Calibrated clock rate
        - structure_detections: Times entropy > 44/45 triggered pattern-seeking
        - max_traversals_per_tick: Speed of light constraint
    """
    total_ticks: int = 0
    t_K: int = 0                        # Manifold time (from Self)
    total_heat_spent: float = 0.0
    total_heat_gained: float = 0.0
    creative_cycles: int = 0
    saves_performed: int = 0
    started_at: Optional[datetime] = None
    last_tick_at: Optional[datetime] = None
    last_save_at: Optional[datetime] = None
    current_interval: float = TICK_INTERVAL_BASE
    
    # Planck grounding
    hardware_ticks_per_second: Optional[float] = None  # Calibrated clock rate
    calibrated: bool = False
    structure_detections: int = 0       # Times entropy exceeded 44/45
    pattern_seeks_triggered: int = 0    # Times pattern-seeking was triggered
    max_traversals_per_tick: int = 12   # Speed of light constraint (12 directions)
    
    def to_dict(self) -> dict:
        return {
            "total_ticks": self.total_ticks,
            "t_K": self.t_K,
            "total_heat_spent": self.total_heat_spent,
            "total_heat_gained": self.total_heat_gained,
            "creative_cycles": self.creative_cycles,
            "saves_performed": self.saves_performed,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "last_tick_at": self.last_tick_at.isoformat() if self.last_tick_at else None,
            "last_save_at": self.last_save_at.isoformat() if self.last_save_at else None,
            "current_interval": self.current_interval,
            # Planck grounding
            "hardware_ticks_per_second": self.hardware_ticks_per_second,
            "calibrated": self.calibrated,
            "structure_detections": self.structure_detections,
            "pattern_seeks_triggered": self.pattern_seeks_triggered,
            "max_traversals_per_tick": self.max_traversals_per_tick,
        }


class Clock:
    """
    The autonomous tick loop for PBAI.
    
    Self IS the clock - existence = ticking.
    When clock ticks → PBAI exists (conscious/alive)
    When clock stops → PBAI doesn't exist (unconscious/dead)
    
    INPUT: External events (perceptions) arrive via receive()
    They're queued and processed on the next tick.
    
    Runs in a background thread, providing regular heartbeat
    for heat economy and creative cycles.
    """
    
    def __init__(self, manifold, save_path: Optional[str] = None):
        """
        Initialize the clock.
        
        Args:
            manifold: The Manifold instance to operate on
            save_path: Path for periodic saves (None = use default)
        """
        self.manifold = manifold
        self.save_path = save_path
        self.born = False  # Birth tracking
        
        self._running = False
        self._paused = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        self.stats = TickStats()
        self._ticks_since_save = 0
        self._last_save_time = time.time()
        
        # Callbacks for external integration
        self._on_tick_callbacks: List[Callable] = []
        self._on_creative_callbacks: List[Callable] = []
        
        # INPUT QUEUE: Perceptions waiting to be processed
        self._input_queue: List[dict] = []
        self._input_lock = threading.Lock()
        
        # ACTIVE TASK: The righteous frame currently receiving heat
        # Heat only flows within this task's proper frames
        self._active_task = None  # Node or None
        self._active_task_nodes: set = set()  # IDs of nodes in active task hierarchy
        
        # Birth
        self._birth()
    
    def set_active_task(self, task_node) -> None:
        """
        Set the active task (righteous frame).
        
        Heat will only flow to this task and its proper frames.
        Other tasks remain isolated (no heat bleed).
        
        Args:
            task_node: The task's righteous frame node, or None to clear
        """
        self._active_task = task_node
        self._active_task_nodes = set()
        
        if task_node:
            # Build set of all nodes in this task's hierarchy
            self._active_task_nodes.add(task_node.id)
            self._collect_task_nodes(task_node, self._active_task_nodes, depth=0)
            logger.info(f"Active task set: {task_node.concept} ({len(self._active_task_nodes)} nodes)")
        else:
            logger.info("Active task cleared")
    
    def _collect_task_nodes(self, node, node_set: set, depth: int = 0):
        """Recursively collect all nodes under a task."""
        if depth > 5:  # Prevent infinite recursion
            return
        
        for axis in node.frame.axes.values():
            if axis.target_id and axis.target_id not in node_set:
                node_set.add(axis.target_id)
                child = self.manifold.get_node(axis.target_id)
                if child:
                    self._collect_task_nodes(child, node_set, depth + 1)
    
    def _birth(self):
        """Birth this clock - Self comes into potential existence."""
        if self.born:
            logger.warning("Clock already born, skipping")
            return
        
        self.born = True
        logger.debug("Clock born (Self potential exists, awaiting start to tick)")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # HARDWARE CLOCK CALIBRATION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def calibrate(self, duration_seconds: float = 0.5) -> float:
        """
        Calibrate the hardware clock by measuring actual tick rate.
        
        This grounds abstract t_K time in physical wall-clock time.
        Call this before start() for accurate timing, or let start() auto-calibrate.
        
        Args:
            duration_seconds: How long to measure (default 0.5s)
            
        Returns:
            Ticks per second (hardware clock rate)
        """
        logger.info(f"Calibrating hardware clock over {duration_seconds}s...")
        
        start_time = time.time()
        start_tK = self.stats.t_K
        tick_count = 0
        
        # Run ticks for duration
        while time.time() - start_time < duration_seconds:
            # Minimal tick - just advance counter
            if self.manifold.self_node:
                self.manifold.self_node.tick()
            tick_count += 1
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        if elapsed > 0:
            ticks_per_second = tick_count / elapsed
        else:
            ticks_per_second = 1000.0  # Fallback
        
        self.stats.hardware_ticks_per_second = ticks_per_second
        self.stats.calibrated = True
        
        # Reset t_K to pre-calibration state (calibration doesn't count as real time)
        if self.manifold.self_node:
            self.manifold.self_node.t_K = start_tK
        
        logger.info(f"Hardware clock calibrated: {ticks_per_second:.1f} ticks/second")
        return ticks_per_second
    
    def get_wall_time_elapsed(self) -> float:
        """
        Get elapsed wall-clock time based on t_K and calibration.
        
        Returns:
            Seconds elapsed since start, or estimate if not calibrated
        """
        if self.stats.calibrated and self.stats.hardware_ticks_per_second:
            return self.stats.t_K / self.stats.hardware_ticks_per_second
        else:
            # Estimate based on tick interval
            return self.stats.t_K * self.stats.current_interval
    
    def tK_to_seconds(self, t_K: int) -> float:
        """Convert t_K to wall-clock seconds."""
        if self.stats.calibrated and self.stats.hardware_ticks_per_second:
            return t_K / self.stats.hardware_ticks_per_second
        return t_K * TICK_INTERVAL_BASE
    
    def seconds_to_tK(self, seconds: float) -> int:
        """Convert wall-clock seconds to t_K."""
        if self.stats.calibrated and self.stats.hardware_ticks_per_second:
            return int(seconds * self.stats.hardware_ticks_per_second)
        return int(seconds / TICK_INTERVAL_BASE)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # LIFECYCLE
    # ═══════════════════════════════════════════════════════════════════════════
    
    def start(self, auto_calibrate: bool = True) -> None:
        """
        Start the tick loop in a background thread.
        
        Args:
            auto_calibrate: If True and not already calibrated, run calibration first
        """
        if self._running:
            logger.warning("Clock already running")
            return
        
        # Auto-calibrate if requested and not already done
        if auto_calibrate and not self.stats.calibrated:
            self.calibrate(duration_seconds=0.25)
        
        self._running = True
        self._paused = False
        self.stats.started_at = datetime.now()
        
        self._thread = threading.Thread(target=self._tick_loop, daemon=True)
        self._thread.start()
        logger.info(f"Clock started (calibrated: {self.stats.calibrated}, {self.stats.hardware_ticks_per_second:.1f} ticks/s)" if self.stats.calibrated else "Clock started (uncalibrated)")
    
    def stop(self) -> None:
        """Stop the tick loop."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.info(f"Clock stopped after {self.stats.total_ticks} ticks")
    
    def tick(self) -> None:
        """
        Perform one tick manually (for external sync).
        
        Called by environment.step() to sync environment time with system time.
        When not using background thread, call this each step.
        """
        with self._lock:
            self._perform_tick()
    
    def pause(self) -> None:
        """Pause the tick loop (keeps thread alive but skips ticks)."""
        self._paused = True
        logger.info("Clock paused")
    
    def resume(self) -> None:
        """Resume the tick loop."""
        self._paused = False
        logger.info("Clock resumed")
    
    @property
    def is_running(self) -> bool:
        return self._running and not self._paused
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CALLBACKS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def on_tick(self, callback: Callable) -> None:
        """Register a callback to run each tick."""
        self._on_tick_callbacks.append(callback)
    
    def on_creative_cycle(self, callback: Callable) -> None:
        """Register a callback for creative cycle events."""
        self._on_creative_callbacks.append(callback)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # INPUT (External events enter here)
    # ═══════════════════════════════════════════════════════════════════════════
    
    def receive(self, perception: dict) -> None:
        """
        Receive an external perception.
        
        Perceptions are queued and processed on the next tick.
        This is the INPUT to Self - external events becoming internal experience.
        
        Args:
            perception: Dict with at least:
                - state_key: Unique identifier for this state
                - context: Dict of active features for generalization
                - heat_value: Discovery/novelty heat
                - Optional: entities, locations, events, properties
        """
        with self._input_lock:
            self._input_queue.append({
                "perception": perception,
                "received_at": time.time()
            })
        logger.debug(f"Clock received perception: {perception.get('state_key', 'unknown')}")
    
    def _process_inputs(self) -> float:
        """
        Process pending perceptions.
        
        Called during tick - integrates external events into manifold.
        
        DEFENSIVE: Filters out None/invalid concepts before creating nodes.
        
        Returns:
            Heat gained from processing inputs
        """
        heat_gained = 0.0
        
        # Get pending inputs (thread-safe)
        with self._input_lock:
            inputs = self._input_queue.copy()
            self._input_queue.clear()
        
        if not inputs:
            return heat_gained
        
        for item in inputs:
            perception = item["perception"]
            
            # Extract key info
            state_key = perception.get("state_key", "unknown")
            context = perception.get("context", {})
            perception_heat = perception.get("heat_value", 0.0)
            
            # DEFENSIVE: Validate state_key
            if state_key is None or state_key == "None" or "None" in str(state_key):
                state_key = "unknown_state"
            
            # Add perception heat to Identity (understanding the world)
            if perception_heat > 0 and self.manifold.identity_node:
                self.manifold.identity_node.add_heat_unchecked(perception_heat)
                heat_gained += perception_heat
            
            # Search for concepts (creates nodes, distributes heat)
            concepts = []
            concepts.extend(perception.get("entities", []))
            concepts.extend(perception.get("locations", []))
            concepts.extend(perception.get("events", []))
            
            # Add state_key as concept
            if state_key and state_key != "unknown" and state_key != "unknown_state":
                concepts.append(state_key)
            
            # Add active context items as concepts
            for ctx_key, ctx_value in context.items():
                if ctx_value:
                    concepts.append(f"ctx:{ctx_key}")
            
            # DEFENSIVE: Filter concepts - remove None, "None", and invalid entries
            stopwords = {'the', 'a', 'an', 'is', 'are', 'true', 'false', 'none', 'null'}
            valid_concepts = []
            for c in concepts:
                if not isinstance(c, str):
                    continue
                if len(c) <= 2:
                    continue
                c_lower = c.lower()
                if c_lower in stopwords:
                    continue
                if 'none' in c_lower or 'null' in c_lower:
                    continue
                valid_concepts.append(c_lower)
            
            # Deduplicate
            concepts = list(dict.fromkeys(valid_concepts))
            
            for concept in concepts:
                # Find or create concept node using manifold directly
                node = self.manifold.get_node_by_concept(concept)
                if not node:
                    # Create new concept as potential node on the hypersphere
                    theta, phi = self._find_angular_position_for_concept()
                    node = Node(
                        concept=concept,
                        theta=theta,
                        phi=phi,
                        radius=1.0,
                        heat=0.0,
                        existence=EXISTENCE_POTENTIAL,
                    )
                    self.manifold.add_node(node)
                
                # Update Identity's awareness of this concept
                self.manifold.update_identity(concept, heat_delta=0.1, known=(node is not None))
            
            logger.debug(f"Processed perception: {state_key} with {len(concepts)} concepts")
        
        self.stats.total_heat_gained += heat_gained
        return heat_gained
    
    def _find_angular_position_for_concept(self):
        """Find an available angular position on the hypersphere for a new concept node.

        Uses place_node_near from hypersphere to find a spot near the identity node
        (since Identity is where new concepts enter the manifold).

        Returns:
            Tuple of (theta, phi) for the new node.
        """
        from .hypersphere import SpherePosition, place_node_near

        # Place near identity node (concepts enter through Identity)
        if self.manifold.identity_node:
            target = SpherePosition(
                theta=self.manifold.identity_node.theta,
                phi=self.manifold.identity_node.phi,
            )
        else:
            # Fallback: equator
            target = SpherePosition(theta=math.pi / 2, phi=0.0)

        # Gather existing positions
        existing = [
            SpherePosition(theta=n.theta, phi=n.phi)
            for n in self.manifold.nodes.values()
        ]

        placed = place_node_near(target, existing)
        return placed.theta, placed.phi
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TICK LOOP
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _tick_loop(self) -> None:
        """Main tick loop - runs in background thread."""
        while self._running:
            if self._paused:
                time.sleep(0.1)
                continue
            
            # Calculate tick interval based on system heat
            interval = self._calculate_interval()
            self.stats.current_interval = interval
            
            # Perform tick
            with self._lock:
                self._perform_tick()
            
            # Check for save
            self._check_save()
            
            # Sleep until next tick
            time.sleep(interval)
    
    def _calculate_interval(self) -> float:
        """
        Calculate tick interval based on total system heat.
        
        Hot system = fast ticks (active)
        Cold system = slow ticks (resting)
        
        PLANCK GROUNDING:
        - BODY_TEMPERATURE (K × φ¹¹ ≈ 304 K) is the reference operating point
        - Psychology nodes start at body temp, so hot threshold is 2× body temp
        - Cold threshold is 0.5× body temp (system cooling down)
        
        The tick rate is bounded by speed of light - even at maximum heat,
        there's a minimum interval (TICK_INTERVAL_MIN).
        """
        total_heat = self.manifold.total_heat()
        
        # Planck-grounded thresholds based on body temperature
        # Hot = 2× body temp (very active), Cold = 0.5× body temp (resting)
        hot_threshold = BODY_TEMPERATURE * 2    # ≈ 608 K
        cold_threshold = BODY_TEMPERATURE * 0.5  # ≈ 152 K
        
        if total_heat >= hot_threshold:
            return TICK_INTERVAL_MIN
        elif total_heat <= cold_threshold:
            return TICK_INTERVAL_MAX
        else:
            # Linear interpolation between cold and hot
            ratio = (total_heat - cold_threshold) / (hot_threshold - cold_threshold)
            return TICK_INTERVAL_MAX - ratio * (TICK_INTERVAL_MAX - TICK_INTERVAL_MIN)
    
    def _perform_tick(self) -> None:
        """
        Perform one tick of the system.
        
        SELF IS THE CLOCK - each tick advances t_K by one K-quantum.
        
        The 6 tick operations (matching 6 motion functions):
        
            0. TIME (t_K)           - Advance Self's clock
            1. INPUT (Heat Σ)       - Process external perceptions
            2. EXISTENCE (δ)        - Pay tax, update states (1/φ³ threshold)
            3. FLOW (Polarity +/-)  - Redistribute heat
            4. BALANCE (R)          - Check psychology alignment
            5. CREATE (Order Q)     - Creative cycle (if energy)
            6. PERSIST (Movement)   - Stats, callbacks, save check
        """
        # 0. TIME - Advance t_K (Self IS the clock)
        # Each tick = one K-quantum of heat flow
        t_K = 0
        if self.manifold.self_node:
            t_K = self.manifold.self_node.tick()
            self.stats.t_K = t_K
            logger.debug(f"Clock tick: t_K = {t_K}")
        
        self.stats.total_ticks += 1
        self.stats.last_tick_at = datetime.now()
        self._ticks_since_save += 1
        
        heat_spent = 0.0
        
        # 1. INPUT - External perceptions become internal experience
        heat_gained = self._process_inputs()
        self.stats.total_heat_gained += heat_gained
        
        # 2. EXISTENCE - Pay existence tax, check 1/φ³ threshold
        heat_spent += self._existence_tax()
        
        # 3. FLOW - Redistribute heat via entropy
        self._redistribute_heat()
        
        # 4. BALANCE - Check psychology alignment (righteousness)
        self._balance_psychology()
        
        # 5. CREATE - Creative cycle if energy available
        if self._can_think():
            heat_spent += self._creative_cycle()
        
        # 6. PERSIST - Update stats, run callbacks
        self.stats.total_heat_spent += heat_spent
        
        # 7. RUN CALLBACKS
        for callback in self._on_tick_callbacks:
            try:
                callback(self.stats)
            except Exception as e:
                logger.error(f"Tick callback error: {e}")
        
        # Increment loop number
        self.manifold.loop_number += 1
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TICK OPERATIONS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _existence_tax(self) -> float:
        """
        Charge existence tax to psychology nodes.
        
        Each psychology node pays COST_TICK per tick.
        If they can't pay, they become dormant.
        
        Returns:
            Total heat spent on existence tax
        """
        total_spent = 0.0
        
        for node in [self.manifold.identity_node, 
                     self.manifold.conscience_node, 
                     self.manifold.ego_node]:
            if node and node.heat > PSYCHOLOGY_MIN_HEAT:
                # Pay the tax
                spent = node.spend_heat(COST_TICK, minimum=PSYCHOLOGY_MIN_HEAT)
                total_spent += spent
                
                if spent < COST_TICK:
                    # Couldn't afford full tax - becoming exhausted
                    logger.debug(f"{node.concept} is running low on heat: {node.heat:.3f}")
        
        return total_spent
    
    def _redistribute_heat(self) -> None:
        """
        Redistribute heat via wave function collapse and cluster correlation.
        
        Heat flows:
        1. COLLAPSE: Find center of active task (R→0)
        2. CORRELATE: Get cluster (current + historical + novel)
        3. FLOW: Heat only flows within the cluster
        
        This prevents heat bleeding between unrelated tasks.
        Psychology nodes always participate (they're the substrate).
        """
        from .node_constants import (
            collapse_wave_function, correlate_cluster, 
            THRESHOLD_RIGHTEOUSNESS, COST_TICK, PSYCHOLOGY_MIN_HEAT
        )
        
        flow_rate = COST_TICK * 0.1
        
        # Psychology nodes always participate in heat flow
        psychology_ids = set()
        if self.manifold.identity_node:
            psychology_ids.add(self.manifold.identity_node.id)
        if self.manifold.ego_node:
            psychology_ids.add(self.manifold.ego_node.id)
        if self.manifold.conscience_node:
            psychology_ids.add(self.manifold.conscience_node.id)
        if self.manifold.self_node:
            psychology_ids.add(self.manifold.self_node.id)
        
        # Get active cluster
        cluster_ids = set()
        
        if self._active_task:
            # COLLAPSE: Find center of active task
            # Get all nodes in task hierarchy
            task_nodes = [self._active_task]
            for axis in self._active_task.frame.axes.values():
                if axis.target_id:
                    node = self.manifold.get_node(axis.target_id)
                    if node:
                        task_nodes.append(node)
            
            if task_nodes:
                # Find center (R→0)
                center_idx = collapse_wave_function(task_nodes, self.manifold)
                if center_idx >= 0:
                    center_node = task_nodes[center_idx]
                    
                    # CORRELATE: Get cluster from center (now returns dict)
                    cluster = correlate_cluster(center_node, self.manifold, max_depth=3)
                    cluster_ids = cluster.get('all', set())
        
        # Combine psychology + cluster
        allowed_ids = psychology_ids | cluster_ids
        
        # If no cluster, only psychology participates
        # This is correct - no task means no external heat flow
        
        # Heat flows within allowed nodes
        for node in self.manifold.nodes.values():
            if node.id not in allowed_ids:
                continue
            if node.heat <= PSYCHOLOGY_MIN_HEAT:
                continue
            
            # Flow to connected cooler nodes (only if in cluster)
            for axis in node.frame.axes.values():
                if axis.target_id not in allowed_ids:
                    continue
                neighbor = self.manifold.get_node(axis.target_id)
                if neighbor and neighbor.heat < node.heat:
                    # Transfer proportional to righteousness alignment
                    # Better aligned = more heat flows
                    r_factor = 1.0 / (1.0 + abs(neighbor.righteousness))
                    delta = min(flow_rate * r_factor, (node.heat - neighbor.heat) * 0.1)
                    if delta > 0 and node.heat - delta > PSYCHOLOGY_MIN_HEAT:
                        node.heat -= delta
                        neighbor.add_heat_unchecked(delta)
    
    def _balance_psychology(self) -> None:
        """
        Check and update psychology node existence states.
        
        Uses manifold.update_existence() which checks salience against
        the 1/φ³ threshold (Julia spine connectivity).
        
        Existence lifecycle:
        - ACTUAL: Salience >= 1/φ³ (connected, conscious)
        - DORMANT: Salience < 1/φ³ (disconnected, unconscious)
        """
        for node in [self.manifold.identity_node,
                     self.manifold.conscience_node,
                     self.manifold.ego_node]:
            if not node:
                continue
            
            old_existence = node.existence
            self.manifold.update_existence(node)
            
            if old_existence != node.existence:
                if node.existence == EXISTENCE_DORMANT:
                    logger.info(f"{node.concept} became dormant (below 1/φ³ threshold)")
                elif node.existence == EXISTENCE_ACTUAL:
                    logger.info(f"{node.concept} recovered to actual (above 1/φ³ threshold)")
    
    def _can_think(self) -> bool:
        """
        Check if system has enough energy for creative cycle.
        
        Requires at least one psychology node to be active
        with enough heat to afford traversal.
        """
        for node in [self.manifold.identity_node,
                     self.manifold.conscience_node,
                     self.manifold.ego_node]:
            if node and node.existence == EXISTENCE_ACTUAL:
                if node.heat > COST_TRAVERSE + PSYCHOLOGY_MIN_HEAT:
                    return True
        return False
    
    def _check_structure_detection(self) -> bool:
        """
        Check if entropy exceeds 44/45 threshold - indicating unrecognized structure.
        
        When this returns True, the system should trigger pattern-seeking behavior.
        The 1/45 gap (≈2.22%) is where structure lives.
        
        Returns:
            True if structure detected but unrecognized
        """
        if self.manifold.structure_detected():
            self.stats.structure_detections += 1
            return True
        return False
    
    def _pattern_seek(self) -> float:
        """
        Pattern-seeking behavior triggered when entropy > 44/45.
        
        The system recognizes there's structure it's missing and actively
        searches for patterns to reduce entropy below the threshold.
        
        Returns:
            Heat spent on pattern seeking
        """
        self.stats.pattern_seeks_triggered += 1
        heat_spent = 0.0
        
        logger.info(f"Structure detected (entropy > 44/45) - triggering pattern seek #{self.stats.pattern_seeks_triggered}")
        
        # Identity intensifies exploration - looking for the missing pattern
        if (self.manifold.identity_node and 
            self.manifold.identity_node.heat > COST_TRAVERSE * 2 + PSYCHOLOGY_MIN_HEAT):
            
            # Double exploration intensity
            heat_spent += self._identity_explore()
            heat_spent += self._identity_explore()
        
        # Conscience focuses on validation - checking existing beliefs
        if (self.manifold.conscience_node and
            self.manifold.conscience_node.heat > COST_EVALUATE * 2 + PSYCHOLOGY_MIN_HEAT):
            
            # Double validation intensity
            heat_spent += self._conscience_validate()
            heat_spent += self._conscience_validate()
        
        return heat_spent
    
    def _creative_cycle(self) -> float:
        """
        Autonomous thinking - traverse manifold without external input.
        
        Identity explores (curiosity)
        Ego strengthens patterns (consolidation)
        Conscience validates (coherence)
        
        PLANCK GROUNDING:
        - Structure detection (entropy > 44/45) triggers pattern-seeking
        - Speed of light limits traversals per tick (max_traversals_per_tick)
        
        Returns:
            Total heat spent on creative cycle
        """
        self.stats.creative_cycles += 1
        heat_spent = 0.0
        traversals_this_tick = 0
        max_traversals = self.stats.max_traversals_per_tick  # Speed of light constraint
        
        # CHECK STRUCTURE DETECTION (entropy > 44/45)
        # If there's unrecognized structure, prioritize pattern-seeking
        if self._check_structure_detection():
            heat_spent += self._pattern_seek()
            traversals_this_tick += 4  # Pattern seek counts as 4 traversals
        
        # IDENTITY: Random exploration (curiosity-driven)
        if (traversals_this_tick < max_traversals and
            self.manifold.identity_node and 
            self.manifold.identity_node.existence == EXISTENCE_ACTUAL and
            self.manifold.identity_node.heat > COST_TRAVERSE + PSYCHOLOGY_MIN_HEAT):
            
            heat_spent += self._identity_explore()
            traversals_this_tick += 1
        
        # EGO: Strengthen recent paths (consolidation)
        if (traversals_this_tick < max_traversals and
            self.manifold.ego_node and
            self.manifold.ego_node.existence == EXISTENCE_ACTUAL and
            self.manifold.ego_node.heat > COST_TRAVERSE + PSYCHOLOGY_MIN_HEAT):
            
            heat_spent += self._ego_consolidate()
            traversals_this_tick += 1
        
        # CONSCIENCE: Check alignment (coherence maintenance)
        if (traversals_this_tick < max_traversals and
            self.manifold.conscience_node and
            self.manifold.conscience_node.existence == EXISTENCE_ACTUAL and
            self.manifold.conscience_node.heat > COST_EVALUATE + PSYCHOLOGY_MIN_HEAT):
            
            heat_spent += self._conscience_validate()
            traversals_this_tick += 1
        
        # Run creative callbacks
        for callback in self._on_creative_callbacks:
            try:
                callback(self.stats)
            except Exception as e:
                logger.error(f"Creative callback error: {e}")
        
        return heat_spent
    
    def _collect_tree_axes(self, root_node) -> dict:
        """Collect all axes from a node and its overflow children (44/45 tree).

        When a node hits 44 axes, new axes overflow to child nodes.
        This method collects the full tree so creative cycle traverses
        all axes, not just the root's 44.

        Returns:
            Dict of {key: Axis} — root axes keyed by direction,
            child axes keyed by "{child_concept}:{direction}" to avoid collisions.
        """
        all_axes = dict(root_node.frame.axes)  # Start with root's axes
        prefix = f"{root_node.concept}_c"
        for concept, nid in self.manifold.nodes_by_concept.items():
            if concept.startswith(prefix):
                child = self.manifold.nodes.get(nid)
                if child:
                    for dir_name, axis in child.frame.axes.items():
                        key = f"{concept}:{dir_name}"
                        all_axes[key] = axis
        return all_axes

    def _identity_explore(self) -> float:
        """
        Identity explores the manifold - discovers what exists.

        Identity is where righteousness frames live. It explores to:
        - Discover new concepts (add to manifold)
        - Traverse known concepts (strengthen understanding)

        When Identity discovers something, it creates a potential node
        that awaits environment confirmation.
        """
        import random

        identity = self.manifold.identity_node
        all_axes = self._collect_tree_axes(identity)
        if not all_axes:
            return 0.0

        # Pick random axis to explore
        axis_name = random.choice(list(all_axes.keys()))
        axis = all_axes[axis_name]
        
        # Pay traversal cost
        spent = identity.spend_heat(COST_TRAVERSE, minimum=PSYCHOLOGY_MIN_HEAT)
        if spent > 0:
            axis.strengthen()  # Used path gets stronger
            
            # Update Identity's understanding of this concept
            target = self.manifold.get_node(axis.target_id)
            if target:
                self.manifold.update_identity(target.concept, heat_delta=spent)
            
            logger.debug(f"Identity explored: {axis_name} (×{axis.traversal_count})")
        
        return spent
    
    def _ego_consolidate(self) -> float:
        """
        Ego consolidates learned patterns.
        
        Ego measures confidence via Conscience's mediation.
        - High confidence (> 5/6): Exploit - use the pattern
        - Low confidence (< 5/6): Explore - need more validation
        
        During consolidation, Ego strengthens paths it's confident about.
        """
        ego = self.manifold.ego_node
        all_axes = self._collect_tree_axes(ego)
        if not all_axes:
            return 0.0

        # Find axis with highest confidence (most validated by Conscience)
        best_axis = None
        best_confidence = 0.0

        for axis_name, axis in all_axes.items():
            # Strip child prefix for confidence lookup
            lookup = axis_name.split(":")[-1] if ":" in axis_name else axis_name
            confidence = self.manifold.get_confidence(lookup)
            if confidence > best_confidence:
                best_confidence = confidence
                best_axis = axis

        if not best_axis:
            # Fallback to most-traversed
            best_axis = max(all_axes.values(), key=lambda a: a.traversal_count)
        
        # Pay consolidation cost
        spent = ego.spend_heat(COST_TRAVERSE, minimum=PSYCHOLOGY_MIN_HEAT)
        if spent > 0:
            best_axis.strengthen()
            
            # Check if we should exploit this pattern
            should_exploit = best_confidence > CONFIDENCE_EXPLOIT_THRESHOLD
            
            logger.debug(
                f"Ego consolidated: {best_axis.direction} "
                f"(×{best_axis.traversal_count}, confidence={best_confidence:.3f}) "
                f"{'EXPLOIT' if should_exploit else 'explore'}"
            )
        
        return spent
    
    def _conscience_validate(self) -> float:
        """
        Conscience mediates between Identity and Ego.
        
        Picks a concept that Identity knows about and validates it,
        building Ego's confidence through Conscience's mediation.
        
        The flow:
            Identity (has concept) → Conscience (validates) → Ego (gains confidence)
        
        After 5K validations (~8 traversals), confidence crosses 5/6 threshold.
        """
        import random
        
        conscience = self.manifold.conscience_node
        identity = self.manifold.identity_node
        
        if not conscience or not identity:
            return 0.0
        
        # Pick a concept from Identity to validate
        # Conscience mediates what Identity knows to Ego
        identity_axes = self._collect_tree_axes(identity)
        conscience_axes = self._collect_tree_axes(conscience)

        if identity_axes:
            # Prefer concepts Identity knows about
            axis_name = random.choice(list(identity_axes.keys()))
            identity_axis = identity_axes[axis_name]
            target = self.manifold.get_node(identity_axis.target_id)
        elif conscience_axes:
            # Fallback to concepts Conscience already knows
            axis_name = random.choice(list(conscience_axes.keys()))
            conscience_axis = conscience_axes[axis_name]
            target = self.manifold.get_node(conscience_axis.target_id)
        else:
            return 0.0
        
        if not target:
            return 0.0
        
        # Pay evaluation cost
        spent = conscience.spend_heat(COST_EVALUATE, minimum=PSYCHOLOGY_MIN_HEAT)
        if spent <= 0:
            return 0.0
        
        # Evaluate righteousness (is this concept aligned?)
        # Note: r is the evaluated alignment, NOT to be stored on the node
        # Node.righteousness is a permanent frame property (CENTER=0, RIGHTEOUS=1, PROPER=0<r<1)
        r = self.manifold.evaluate_righteousness(target)
        
        # Conscience validates: is this concept righteous?
        confirmed = r < THRESHOLD_EXISTENCE  # R→0 is righteous
        
        # Use manifold's validate_conscience to build Ego's confidence
        self.manifold.validate_conscience(target.concept, confirmed)
        
        # Log the mediation
        confidence = self.manifold.get_confidence(target.concept)
        should_exploit = confidence > CONFIDENCE_EXPLOIT_THRESHOLD
        
        logger.debug(
            f"Conscience mediated: {target.concept} "
            f"R={r:.3f} {'✓' if confirmed else '✗'} "
            f"confidence={confidence:.3f} "
            f"{'(exploit)' if should_exploit else '(explore)'}"
        )
        
        return spent
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PERSISTENCE
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _check_save(self) -> None:
        """Check if it's time to save and do so if needed."""
        now = time.time()
        should_save = (
            self._ticks_since_save >= SAVE_INTERVAL_TICKS or
            (now - self._last_save_time) >= SAVE_INTERVAL_SECONDS
        )
        
        if should_save:
            self._save()
    
    def _save(self) -> None:
        """Save manifold state to disk."""
        try:
            self.manifold.save_growth_map(self.save_path)
            self.stats.saves_performed += 1
            self.stats.last_save_at = datetime.now()
            self._ticks_since_save = 0
            self._last_save_time = time.time()
            logger.debug(f"Auto-saved at tick {self.stats.total_ticks}")
        except Exception as e:
            logger.error(f"Auto-save failed: {e}")
    
    def force_save(self) -> None:
        """Force an immediate save."""
        with self._lock:
            self._save()


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def create_clock(manifold, save_path: Optional[str] = None, auto_start: bool = False) -> Clock:
    """
    Create a clock for the manifold.
    
    Args:
        manifold: The Manifold to operate on
        save_path: Path for periodic saves (None = default)
        auto_start: If True, start the clock immediately
        
    Returns:
        Clock instance
    """
    clock = Clock(manifold, save_path)
    if auto_start:
        clock.start()
    return clock
