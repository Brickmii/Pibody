"""
PBAI Daemon - Raspberry Pi 5 Deployment (Planck-Grounded)

The living PBAI system. Runs continuously, choosing between environments,
learning, resting - all regulated by physical thermal constraints.

════════════════════════════════════════════════════════════════════════════════
PLANCK GROUNDING
════════════════════════════════════════════════════════════════════════════════

CLOCK SYNC:
    The daemon's tick loop is the MASTER clock for the entire system:
    - Each tick advances t_K (heat-time)
    - Environment.step() syncs with this clock
    - Vision syncs via step_with_vision()
    - All timestamps use t_K reference

BODY TEMPERATURE:
    Psychology operates at BODY_TEMPERATURE (K × φ¹¹ ≈ 304 K = 31°C).
    This grounds the heat metaphor in biology.

THERMAL GROUNDING:
    CPU temperature maps to Fire heat zones (K × φⁿ):
    - Cool (Fire 1-2): Fast thinking
    - Warm (Fire 3-4): Normal operation
    - Hot (Fire 5): Slowed cognition
    - Danger (Fire 6): Forced rest

STRUCTURE DETECTION:
    When manifold entropy > 44/45, there's unrecognized structure.
    The daemon triggers pattern-seeking mode in this case.

════════════════════════════════════════════════════════════════════════════════

ARCHITECTURE:
    ┌─────────────────────────────────────────────────────────┐
    │                     PBAI DAEMON                         │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │
    │  │  Tick Loop  │  │   Chooser   │  │  Network API    │ │
    │  │  (Clock)    │  │             │  │  (Remote Ctrl)  │ │
    │  └──────┬──────┘  └──────┬──────┘  └────────┬────────┘ │
    │         │                │                   │          │
    │  ┌──────┴────────────────┴───────────────────┴───────┐ │
    │  │                    MANIFOLD                        │ │
    │  │   Identity ←→ Ego ←→ Conscience ←→ Nodes          │ │
    │  └───────────────────────┬───────────────────────────┘ │
    │                          │                              │
    │  ┌───────────────────────┴───────────────────────────┐ │
    │  │              ENVIRONMENT CORE                      │ │
    │  │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────────┐ │ │
    │  │  │ Maze │ │ BJ   │ │ Chat │ │Vision│ │ Minecraft│ │ │
    │  │  └──────┘ └──────┘ └──────┘ └──────┘ └──────────┘ │ │
    │  └───────────────────────────────────────────────────┘ │
    │                          │                              │
    │  ┌───────────────────────┴───────────────────────────┐ │
    │  │              THERMAL MANAGER                       │ │
    │  │         CPU Temp → Fire Zones → Tick Rate         │ │
    │  └───────────────────────────────────────────────────┘ │
    └─────────────────────────────────────────────────────────┘
              │              │              │
         [NVMe]       [Network]      [Sensors/GPIO]
"""

import logging
import threading
import time
import signal
import sys
import os
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import Manifold, K, create_clock, Clock
from core.node_constants import (
    TICK_INTERVAL_BASE, TICK_INTERVAL_MIN, TICK_INTERVAL_MAX,
    PSYCHOLOGY_MIN_HEAT, COST_ACTION,
    BODY_TEMPERATURE, PHI,
    MAX_ENTROPIC_PROBABILITY, EMERGENCE_THRESHOLD,
)
from drivers import EnvironmentCore, Driver, Action, ActionResult

from core.introspector import Introspector
from .thermal import ThermalManager, ThermalState, create_thermal_manager

logger = logging.getLogger(__name__)


class DaemonState(Enum):
    """States of the PBAI daemon."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"           # Manually paused
    THERMAL_PAUSE = "thermal"   # Paused due to heat
    RESTING = "resting"         # Voluntarily resting (low heat)
    STOPPING = "stopping"


@dataclass
class EnvironmentStats:
    """Statistics for an environment."""
    name: str
    sessions: int = 0
    total_heat_earned: float = 0.0
    total_heat_spent: float = 0.0
    successes: int = 0
    failures: int = 0
    last_used: Optional[datetime] = None
    
    @property
    def success_rate(self) -> float:
        total = self.successes + self.failures
        return self.successes / total if total > 0 else 0.5
    
    @property
    def net_heat(self) -> float:
        return self.total_heat_earned - self.total_heat_spent


@dataclass
class DaemonStats:
    """Statistics for the daemon (Planck-Grounded)."""
    started_at: Optional[datetime] = None
    total_ticks: int = 0
    total_choices: int = 0
    environment_stats: Dict[str, EnvironmentStats] = field(default_factory=dict)
    thermal_pauses: int = 0
    voluntary_rests: int = 0
    current_temp: float = 0.0
    current_zone: str = "unknown"
    
    # Planck grounding
    current_fire_level: int = 1
    structure_detections: int = 0
    pattern_seek_activations: int = 0
    t_K: int = 0  # Current heat-time


class EnvironmentChooser:
    """
    Chooses which environment to engage with based on psychology state.
    
    CHOICE FACTORS:
        - Ego heat (high = exploit known good environments)
        - Identity heat (high = explore uncertain environments)
        - Environment success history
        - Time since last use (novelty)
        - Available heat budget
    """
    
    def __init__(self, manifold: Manifold, env_core: EnvironmentCore):
        self.manifold = manifold
        self.env_core = env_core
        self.stats: Dict[str, EnvironmentStats] = {}
        
        # Rest is always available
        self.stats['rest'] = EnvironmentStats(name='rest')
    
    def register_environment(self, driver_id: str, name: str = None):
        """Register an environment for choice tracking."""
        name = name or driver_id
        if driver_id not in self.stats:
            self.stats[driver_id] = EnvironmentStats(name=name)
    
    def choose(self) -> str:
        """
        Choose the next environment to engage with.
        
        Returns:
            driver_id of chosen environment (or 'rest')
        """
        # Get psychology state
        ego_heat = self.manifold.ego_node.heat if self.manifold.ego_node else 0
        identity_heat = self.manifold.identity_node.heat if self.manifold.identity_node else 0
        total_heat = ego_heat + identity_heat
        
        # If too exhausted, must rest
        # Let act() handle the actual cost check — chooser just gates on minimum
        from core.node_constants import PSYCHOLOGY_MIN_HEAT
        if ego_heat <= PSYCHOLOGY_MIN_HEAT:
            logger.debug(f"Ego resting (heat={ego_heat:.3f})")
            return 'rest'
        
        # Calculate exploration rate (Identity / Total)
        exploration_rate = identity_heat / total_heat if total_heat > 0 else 0.5
        
        # Get available environments (registered drivers)
        available = [d for d in self.env_core.drivers.keys() if d != 'rest']
        
        if not available:
            return 'rest'
        
        # Score each environment
        scores = {}
        for env_id in available:
            stats = self.stats.get(env_id, EnvironmentStats(name=env_id))
            
            # Base score from success rate
            success_score = stats.success_rate
            
            # Novelty bonus (time since last use)
            novelty = 1.0
            if stats.last_used:
                hours_since = (datetime.now() - stats.last_used).total_seconds() / 3600
                novelty = min(2.0, 1.0 + hours_since * 0.1)
            
            # Net heat efficiency
            efficiency = 1.0
            if stats.sessions > 0:
                efficiency = max(0.5, 1.0 + stats.net_heat / (stats.sessions * K))
            
            # Combine based on exploration/exploitation
            exploit_score = success_score * efficiency
            explore_score = novelty * (1.0 - success_score + 0.5)  # Uncertainty bonus
            
            scores[env_id] = (
                exploit_score * (1 - exploration_rate) +
                explore_score * exploration_rate
            )
        
        # Choose (weighted random or argmax)
        import random
        if random.random() < 0.1:  # 10% pure random for diversity
            chosen = random.choice(available)
        else:
            chosen = max(scores, key=scores.get)
        
        logger.info(f"Chose environment: {chosen} (scores: {scores})")
        return chosen
    
    def record_outcome(self, driver_id: str, heat_earned: float, 
                       heat_spent: float, success: bool):
        """Record outcome of an environment session."""
        if driver_id not in self.stats:
            self.stats[driver_id] = EnvironmentStats(name=driver_id)
        
        stats = self.stats[driver_id]
        stats.sessions += 1
        stats.total_heat_earned += heat_earned
        stats.total_heat_spent += heat_spent
        if success:
            stats.successes += 1
        else:
            stats.failures += 1
        stats.last_used = datetime.now()


class PBAIDaemon:
    """
    The main PBAI daemon (Planck-Grounded).
    
    Runs continuously on the Pi, managing:
    - Tick loop (thermally regulated via Fire zones)
    - Environment choice
    - Network API (REST on :8420)
    - Body server (WebSocket on :8421)
    - Vision integration
    - Persistence
    
    PLANCK GROUNDING:
        - Clock sync: daemon.tick() advances t_K
        - Thermal: Fire heat zones regulate tick rate
        - Structure: 44/45 entropy detection triggers pattern-seeking
    """
    
    def __init__(self, 
                 save_path: str = None,
                 enable_api: bool = True,
                 api_port: int = 8420,
                 enable_body: bool = True,
                 body_port: int = 8421,
                 simulated_thermal: bool = False,
                 enable_vision: bool = False):
        """
        Initialize the daemon.
        
        Args:
            save_path: Path for manifold persistence
            enable_api: Enable REST API
            api_port: Port for REST API
            enable_body: Enable body server for remote bodies
            body_port: Port for body WebSocket server
            simulated_thermal: Use simulated thermal (for testing)
            enable_vision: Enable visual cortex integration
        """
        self.save_path = save_path  # None = use default growth/ directory
        self.enable_api = enable_api
        self.api_port = api_port
        self.enable_body = enable_body
        self.body_port = body_port
        self.enable_vision = enable_vision
        
        # Ensure save directory exists (only for custom paths)
        if self.save_path:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        
        # Core components (initialized in start())
        self.manifold: Optional[Manifold] = None
        self.clock: Optional[Clock] = None
        self.env_core: Optional[EnvironmentCore] = None
        self.chooser: Optional[EnvironmentChooser] = None
        self.thermal: Optional[ThermalManager] = None
        self.introspector: Optional[Introspector] = None  # Heat-pattern explorer
        self.body_server = None  # BodyServer for remote bodies
        self.visual_cortex = None  # VisualCortex for vision
        
        # State
        self.state = DaemonState.STOPPED
        self.stats = DaemonStats()
        self._simulated_thermal = simulated_thermal
        
        # Planck grounding - structure detection
        self._structure_detected = False
        self._pattern_seek_mode = False
        
        # Threading
        self._main_thread: Optional[threading.Thread] = None
        self._api_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        
        # Current activity
        self.current_environment: Optional[str] = None
        self.current_session_heat: float = 0.0
        
        # Signal handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # LIFECYCLE
    # ═══════════════════════════════════════════════════════════════════════════
    
    def start(self):
        """Start the daemon (Planck-Grounded)."""
        if self.state != DaemonState.STOPPED:
            logger.warning("Daemon already running")
            return
        
        self.state = DaemonState.STARTING
        logger.info("═══ PBAI DAEMON STARTING (Planck-Grounded) ═══")
        
        # Initialize manifold
        self._init_manifold()
        
        # Initialize thermal manager (Fire heat zones)
        self.thermal = create_thermal_manager(simulated=self._simulated_thermal)
        
        # Initialize environment core
        self.env_core = EnvironmentCore(manifold=self.manifold)
        
        # Initialize chooser
        self.chooser = EnvironmentChooser(self.manifold, self.env_core)
        
        # Initialize clock (but don't start its internal loop - we manage ticks)
        self.clock = Clock(self.manifold, save_path=self.save_path)

        # Initialize Introspector (heat-pattern explorer)
        self.introspector = Introspector(self.manifold)
        logger.info("Introspector initialized (heat-pattern explorer)")

        # Initialize visual cortex if enabled
        if self.enable_vision:
            self._init_visual_cortex()
        
        # Start main loop
        self._stop_event.clear()
        self._main_thread = threading.Thread(target=self._main_loop, daemon=True)
        self._main_thread.start()
        
        # Start API if enabled
        if self.enable_api:
            self._start_api()
        
        # Start body server if enabled
        if self.enable_body:
            self._start_body_server()

        # Register Minecraft driver (uses body server stream for vision)
        self._register_minecraft_driver()

        self.state = DaemonState.RUNNING
        self.stats.started_at = datetime.now()
        logger.info("═══ PBAI DAEMON RUNNING ═══")
        logger.info(f"  Body temp reference: {BODY_TEMPERATURE:.2f} K ({BODY_TEMPERATURE - 273.15:.1f}°C)")
        logger.info(f"  Structure detection: 44/45 = {MAX_ENTROPIC_PROBABILITY:.6f}")
    
    def stop(self):
        """Stop the daemon."""
        if self.state == DaemonState.STOPPED:
            return
        
        self.state = DaemonState.STOPPING
        logger.info("═══ PBAI DAEMON STOPPING ═══")
        
        # Signal threads to stop
        self._stop_event.set()
        
        # Wait for main thread
        if self._main_thread:
            self._main_thread.join(timeout=5.0)
        
        # Final save
        if self.manifold:
            self.manifold.save_growth_map(self.save_path)
            logger.info(f"Final save to {self.save_path}")
        
        # Stop body server
        if self.body_server:
            self.body_server.stop()
        
        # Cleanup thermal
        if self.thermal:
            self.thermal.cleanup()
        
        self.state = DaemonState.STOPPED
        logger.info("═══ PBAI DAEMON STOPPED ═══")
    
    def _init_manifold(self):
        """Initialize or load the manifold using the singleton loader."""
        from core import get_pbai_manifold
        self.manifold = get_pbai_manifold(self.save_path)
        logger.info(f"Manifold ready: {len(self.manifold.nodes)} nodes, born={self.manifold.born}")
    
    def _init_visual_cortex(self):
        """Initialize visual cortex for vision integration."""
        try:
            from vision.visual_cortex import VisualCortex
            self.visual_cortex = VisualCortex(
                resolution=64,
                manifold=self.manifold,
                environment=self.env_core
            )
            self.env_core.connect_visual_cortex(self.visual_cortex)
            logger.info("Visual cortex initialized and connected")
        except ImportError as e:
            logger.warning(f"Visual cortex not available: {e}")
            self.visual_cortex = None
        except Exception as e:
            logger.error(f"Failed to initialize visual cortex: {e}")
            self.visual_cortex = None
    
    def step_with_vision(self, image=None):
        """
        Perform one step using vision input.
        
        This is a convenience method for vision-driven operation.
        
        Args:
            image: Optional image array (H, W, 3) RGB
            
        Returns:
            (action, result, heat_changes) or None if vision not available
        """
        if not self.env_core:
            return None
        
        return self.env_core.step_with_vision(image)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # MAIN LOOP
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _main_loop(self):
        """
        Main daemon loop (Planck-Grounded).
        
        Each iteration:
        1. Check thermal state (Fire zones)
        2. Check structure detection (44/45 entropy)
        3. Perform tick (if not paused)
        4. Choose/engage environment (if active)
        5. Sleep based on thermal-adjusted interval
        """
        logger.info("Main loop started (Planck-grounded)")
        
        while not self._stop_event.is_set():
            try:
                # 1. Check thermal state (Fire heat zones)
                thermal_state = self.thermal.update()
                self.stats.current_temp = thermal_state.temperature
                self.stats.current_zone = thermal_state.zone
                self.stats.current_fire_level = thermal_state.fire_level
                
                # Thermal pause?
                if thermal_state.zone == 'critical':
                    if self.state != DaemonState.THERMAL_PAUSE:
                        self.state = DaemonState.THERMAL_PAUSE
                        self.stats.thermal_pauses += 1
                        logger.warning(f"THERMAL PAUSE: {thermal_state.temperature:.1f}°C (Fire {thermal_state.fire_level})")
                    time.sleep(5.0)  # Wait for cooldown
                    continue
                elif self.state == DaemonState.THERMAL_PAUSE:
                    self.state = DaemonState.RUNNING
                    logger.info(f"Thermal pause ended: {thermal_state.temperature:.1f}°C (Fire {thermal_state.fire_level})")
                
                # Manual pause?
                if self.state == DaemonState.PAUSED:
                    time.sleep(1.0)
                    continue
                
                # 2. Check structure detection (44/45 entropy)
                if self.manifold:
                    structure_now = self.manifold.structure_detected()
                    if structure_now and not self._structure_detected:
                        self.stats.structure_detections += 1
                        self._pattern_seek_mode = True
                        self.stats.pattern_seek_activations += 1
                        logger.info(f"Structure detected (#{self.stats.structure_detections}) - pattern-seeking enabled")
                    elif not structure_now and self._structure_detected:
                        self._pattern_seek_mode = False
                        logger.info("Structure resolved - normal mode")
                    self._structure_detected = structure_now
                
                # 3. Perform tick (advances t_K)
                with self._lock:
                    self.clock._perform_tick()
                    self.stats.total_ticks += 1
                    self.stats.t_K = self.clock.stats.t_K
                
                # 4. Environment activity (every N ticks)
                if self.stats.total_ticks % 10 == 0:
                    self._environment_cycle()
                
                # 5. Sleep (thermally adjusted via Fire zones)
                base_interval = self._calculate_interval()
                adjusted_interval = base_interval * thermal_state.tick_multiplier
                adjusted_interval = max(TICK_INTERVAL_MIN, 
                                       min(TICK_INTERVAL_MAX * 4, adjusted_interval))
                
                time.sleep(adjusted_interval)
                
            except Exception as e:
                logger.error(f"Main loop error: {e}")
                time.sleep(1.0)
        
        logger.info("Main loop ended")
    
    def _calculate_interval(self) -> float:
        """Calculate base tick interval from manifold heat."""
        return self.clock._calculate_interval()
    
    def _environment_cycle(self):
        """One environment engagement cycle via 7-step decision chain.

        Map → Plot → Weigh → Simulate → Decide → Execute → Evaluate
        """
        # Choose
        chosen = self.chooser.choose()
        self.stats.total_choices += 1

        if chosen == 'rest':
            self.state = DaemonState.RESTING
            self.stats.voluntary_rests += 1
            self.current_environment = None
            return

        self.state = DaemonState.RUNNING
        self.current_environment = chosen

        # Activate environment
        if chosen in self.env_core.drivers:
            self.env_core.activate_driver(chosen)
        else:
            logger.warning(f"Environment {chosen} not registered")
            return

        try:
            chain = self.env_core.run_decision_chain(self.introspector)

            # Record outcome for chooser
            self.chooser.record_outcome(
                chosen,
                heat_earned=chain.result.heat_value if chain.result else 0,
                heat_spent=COST_ACTION,
                success=chain.result.success if chain.result else False
            )

        except Exception as e:
            logger.error(f"Environment cycle error: {e}")
            self.chooser.record_outcome(chosen, 0, COST_ACTION, False)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # API (Network interface)
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _start_api(self):
        """Start the network API."""
        try:
            from .api import create_api_server, run_api_server
            self._api_thread = threading.Thread(
                target=run_api_server,
                args=(self, self.api_port),
                daemon=True
            )
            self._api_thread.start()
            logger.info(f"API server started on port {self.api_port}")
        except ImportError:
            logger.warning("API module not available")
    
    def _start_body_server(self):
        """Start the body server for remote body connections."""
        try:
            from .body_server import BodyServer
            self.body_server = BodyServer(daemon=self, port=self.body_port)
            self.body_server.start()
            logger.info(f"Body server started on port {self.body_port}")
        except ImportError:
            logger.warning("Body server module not available")
        except Exception as e:
            logger.error(f"Body server failed to start: {e}")

    def _register_minecraft_driver(self):
        """Register MinecraftDriver with body server stream wired in."""
        try:
            from drivers.minecraft import create_minecraft_driver
            mc_driver = create_minecraft_driver(
                manifold=self.manifold,
                body_server=self.body_server
            )
            self.register_driver(mc_driver, "Minecraft Bedrock")
        except Exception as e:
            logger.error(f"Failed to register Minecraft driver: {e}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PUBLIC INTERFACE (for API and direct control)
    # ═══════════════════════════════════════════════════════════════════════════
    
    def pause(self):
        """Pause the daemon."""
        if self.state == DaemonState.RUNNING:
            self.state = DaemonState.PAUSED
            logger.info("Daemon paused")
    
    def resume(self):
        """Resume the daemon."""
        if self.state == DaemonState.PAUSED:
            self.state = DaemonState.RUNNING
            logger.info("Daemon resumed")
    
    def force_save(self):
        """Force immediate save."""
        with self._lock:
            self.clock.force_save()
    
    def get_status(self) -> dict:
        """Get current daemon status (Planck-Grounded)."""
        return {
            "state": self.state.value,
            "uptime": (datetime.now() - self.stats.started_at).total_seconds() 
                      if self.stats.started_at else 0,
            "ticks": self.stats.total_ticks,
            "t_K": self.stats.t_K,
            "choices": self.stats.total_choices,
            "temperature": self.stats.current_temp,
            "thermal_zone": self.stats.current_zone,
            "fire_level": self.stats.current_fire_level,
            "current_environment": self.current_environment,
            "structure_detected": self._structure_detected,
            "pattern_seek_mode": self._pattern_seek_mode,
            "structure_detections": self.stats.structure_detections,
            "manifold": {
                "nodes": len(self.manifold.nodes) if self.manifold else 0,
                "loop": self.manifold.loop_number if self.manifold else 0,
                "total_heat": self.manifold.total_heat() if self.manifold else 0,
            },
            "psychology": {
                "identity": self.manifold.identity_node.heat if self.manifold and self.manifold.identity_node else 0,
                "ego": self.manifold.ego_node.heat if self.manifold and self.manifold.ego_node else 0,
                "conscience": self.manifold.conscience_node.heat if self.manifold and self.manifold.conscience_node else 0,
            },
            "environments": {
                k: {
                    "sessions": v.sessions,
                    "success_rate": v.success_rate,
                    "net_heat": v.net_heat
                }
                for k, v in self.chooser.stats.items()
            } if self.chooser else {},
            "vision_enabled": self.visual_cortex is not None,
        }
    
    def register_driver(self, driver: Driver, name: str = None):
        """Register a new environment driver."""
        with self._lock:
            self.env_core.register_driver(driver)
            self.chooser.register_environment(driver.DRIVER_ID, name)
            logger.info(f"Registered environment: {driver.DRIVER_ID}")
    
    def inject_perception(self, perception: dict):
        """Inject a perception from external source."""
        from drivers import Perception
        p = Perception(**perception)
        if self.env_core:
            self.env_core._integrate_perception(p)
    
    def inject_heat(self, amount: float, target: str = "identity"):
        """Inject heat into psychology (for external rewards)."""
        with self._lock:
            if target == "identity" and self.manifold.identity_node:
                self.manifold.identity_node.add_heat_unchecked(amount)
            elif target == "ego" and self.manifold.ego_node:
                self.manifold.ego_node.add_heat_unchecked(amount)
            elif target == "conscience" and self.manifold.conscience_node:
                self.manifold.conscience_node.add_heat_unchecked(amount)


def run_daemon(save_path: str = None, api_port: int = 8420, 
               body_port: int = 8421, simulated: bool = False,
               enable_vision: bool = False):
    """
    Run the PBAI daemon (Planck-Grounded).
    
    Args:
        save_path: Path for manifold persistence
        api_port: Port for REST API
        body_port: Port for body WebSocket server
        simulated: Use simulated thermal (for testing)
        enable_vision: Enable visual cortex integration
    """
    daemon = PBAIDaemon(
        save_path=save_path,
        enable_api=True,
        api_port=api_port,
        enable_body=True,
        body_port=body_port,
        simulated_thermal=simulated,
        enable_vision=enable_vision
    )
    
    daemon.start()
    
    # Keep running until stopped
    try:
        while daemon.state != DaemonState.STOPPED:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        daemon.stop()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    import argparse
    parser = argparse.ArgumentParser(description="PBAI Daemon (Planck-Grounded)")
    parser.add_argument("--save-path", default=None, help="Path for manifold persistence")
    parser.add_argument("--port", type=int, default=8420, help="REST API port")
    parser.add_argument("--body-port", type=int, default=8421, help="Body WebSocket port")
    parser.add_argument("--simulated", action="store_true", help="Use simulated thermal")
    parser.add_argument("--vision", action="store_true", help="Enable visual cortex")
    args = parser.parse_args()
    
    run_daemon(
        save_path=args.save_path,
        api_port=args.port,
        body_port=args.body_port,
        simulated=args.simulated,
        enable_vision=args.vision
    )
