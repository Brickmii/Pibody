"""
PBAI Thermal Manifold - Core Module

THE 6 CORE FILES (aligned with 6 motion functions):
- node_constants.py   : Polarity (+/-) - direction validator, all constants
- nodes.py            : Righteousness (R) - alignment validator, frames
- manifold.py         : Order (Q) + Heat (Σ) - arithmetic validator + psychology
- clock_node.py       : Existence (δ) - persistence validator (Self IS clock)
- decision_node.py    : Movement (Lin) - vectorized output (5 scalars → 1 vector)
- compression.py      : Utility - position encoding

SUPPORT FILE:
- driver_node.py      : IO dataclasses (SensorReport, MotorAction, ActionPlan)

ARCHITECTURE:
    Self (clock_node) IS existence - when it ticks, PBAI is conscious
    Psychology nodes ARE the mind - Identity/Ego/Conscience (in manifold.py)
    Decision nodes ARE the interface - Entry (perceive) / Exit (act)
    
THE 5/6 CONFIDENCE THRESHOLD:
    - confidence > 5/6 (0.8333): EXPLOIT (5 scalars validated → use pattern)
    - confidence < 5/6: EXPLORE (still gathering validation)
    - t = 5K validations crosses threshold (one K-quantum per scalar)
    
MOTION UNIT THEORY:
    1 motion unit ≥ 1/φ (golden ratio reciprocal)
    Each motion function has threshold 1/φⁿ:
    - Heat (n=1): 0.618
    - Polarity (n=2): 0.382
    - Existence (n=3): 0.236
    - Righteousness (n=4): 0.146
    - Order (n=5): 0.090
    - Movement (n=6): 0.056
    
    K = sum of all thresholds ≈ 1.528
    
HEAT ECONOMY:
    - Costs = Thresholds (symmetric)
    - Actions output heat to environment (COST_ACTION = 0.618)
    - Traversals spend heat (COST_TRAVERSE = 0.056)
    - Creation spends heat (COST_CREATE_NODE = 0.236)
    - Tick loop taxes existence (COST_TICK = 0.056)
"""

from .node_constants import (
    # Fundamental constants
    MOVEMENT_CONSTANT, SIX, NEGATIVE_ONE_TWELFTH, ENTROPIC_BOUND, I, PHI, INV_PHI, K,
    
    # Motion thresholds - Abstract layer (1-6)
    THRESHOLD_HEAT, THRESHOLD_POLARITY, THRESHOLD_EXISTENCE,
    THRESHOLD_RIGHTEOUSNESS, THRESHOLD_ORDER, THRESHOLD_MOVEMENT,
    
    # Motion thresholds - Physical layer (7-9)
    THRESHOLD_SPACE, THRESHOLD_TIME, THRESHOLD_MATTER,
    
    # Motion thresholds - Coupling layer (10-12)
    THRESHOLD_ALPHA, THRESHOLD_BETA, THRESHOLD_PSI, THRESHOLD_GAMMA,
    FINE_STRUCTURE_CONSTANT,
    
    # Coupling functions
    euler_beta, wave_function, gamma_function, entropy_count,
    collapse_wave_function, correlate_cluster, select_from_cluster,
    
    # Righteousness and Selection
    R_ALIGNED, R_PERFECT, R_MISALIGNED,
    righteousness_weight, selection_score,
    SELECTION_HEAT_FACTOR, SELECTION_RIGHTEOUSNESS_FACTOR, SELECTION_ENTROPY_FACTOR,
    
    # Threshold collections
    CORE_THRESHOLDS,
    MOTION_THRESHOLDS, ABSTRACT_THRESHOLDS, PHYSICAL_THRESHOLDS, COUPLING_THRESHOLDS,
    K_PHYSICAL, K_COUPLING, K_TOTAL, K_INFINITE, K_RESIDUE,
    
    # Motion costs (heat expenditure for operations)
    COST_HEAT, COST_POLARITY, COST_EXISTENCE,
    COST_RIGHTEOUSNESS, COST_ORDER, COST_MOVEMENT,
    COST_SPACE, COST_TIME, COST_MATTER,
    COST_ALPHA, COST_BETA, COST_GAMMA,
    MOTION_COSTS,
    COST_TRAVERSE, COST_CREATE_NODE, COST_EVALUATE, COST_ACTION, COST_TICK, COST_COLLAPSE,
    
    # Tick configuration
    TICK_INTERVAL_BASE, TICK_INTERVAL_MIN, TICK_INTERVAL_MAX,
    TICK_HEAT_HOT, TICK_HEAT_COLD,
    SAVE_INTERVAL_TICKS, SAVE_INTERVAL_SECONDS,
    PSYCHOLOGY_MIN_HEAT,
    
    # Direction/motion mappings
    DIRECTIONS, OPPOSITES, SELF_DIRECTIONS, ALL_DIRECTIONS,
    DIRECTION_TO_MOTION, MOTION_TO_DIRECTION,
    
    # Existence states
    EXISTENCE_ACTUAL, EXISTENCE_DORMANT, EXISTENCE_ARCHIVED, EXISTENCE_POTENTIAL,
    
    # Capability levels
    CAPABILITY_RIGHTEOUS, CAPABILITY_PROPER, CAPABILITY_ORDERED, CAPABILITY_MOVABLE, CAPABILITY_GRAPHIC,
    
    # Entropy weights (golden angle trigonometry)
    ENTROPY_MAGNITUDE_WEIGHT, ENTROPY_VARIANCE_WEIGHT, ENTROPY_DISORDER_WEIGHT,
    
    # Psychology trig positions (abstract space basis vectors)
    TRIG_IDENTITY, TRIG_EGO, TRIG_CONSCIENCE,
    trig_position_to_string, string_to_trig_position,
    is_trig_position, is_cubic_position,
    
    # Freudian heat distribution ratios
    FREUD_IDENTITY_RATIO, FREUD_CONSCIENCE_RATIO, FREUD_EGO_RATIO,
    
    # Threshold functions
    get_threshold, get_threshold_for_direction,
    exceeds_threshold, exceeds_any_threshold, exceeds_all_thresholds,
    quantize_to_threshold,
    
    # Cost functions
    get_cost, get_cost_for_direction,
    
    # Other utilities
    heat_required, get_opposite, direction_to_vector,
    get_motion_for_direction, get_direction_for_motion,
    get_project_root, get_growth_path, GROWTH_DEFAULT
)

from .compression import (
    compress, decompress, validate_position,
    get_axis_coordinates, get_depth, position_length,
    positions_share_prefix, run_compression_tests
)

from .nodes import (
    # v2 Architecture: Unified Axis with nested capability hierarchy
    Element, Graphic, Transition, Movement, Order, Axis, Frame,
    Node, SelfNode, assert_self_valid,
    migrate_node_v1_to_v2,
    # Birth system
    birth_randomizer, mark_birth_complete, is_birth_spent, reset_birth_for_testing
)

from .manifold import (
    Manifold, create_manifold, get_pbai_manifold, reset_pbai_manifold
)

from .driver_node import (
    SensorReport, MotorType, MotorAction, ActionPlan, DriverNode,
    press, hold_key, release_key, look, mouse_hold, mouse_release, wait,
    mouse_move, mouse_click, api_call, create_plan
)

from .decision_node import (
    Choice, ChoiceNode, DecisionNode, EnvironmentNode, PBAILoop,
    get_decisions_path
)

from .clock_node import (
    Clock, TickStats, create_clock
)

# Import environment components from drivers package
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drivers import (
    PortState, PortMessage, Port, NullPort,
    Perception, Action, ActionResult, Driver, MockDriver,
    EnvironmentCore, DriverLoader, create_environment_core
)

__all__ = [
    # Fundamental Constants
    'MOVEMENT_CONSTANT', 'SIX', 'NEGATIVE_ONE_TWELFTH', 'ENTROPIC_BOUND', 'I', 'PHI', 'INV_PHI', 'K',
    
    # Motion Thresholds - Abstract layer (1-6): Σ, Noether, δ, dim, Q, Lin
    'THRESHOLD_HEAT', 'THRESHOLD_POLARITY', 'THRESHOLD_EXISTENCE',
    'THRESHOLD_RIGHTEOUSNESS', 'THRESHOLD_ORDER', 'THRESHOLD_MOVEMENT',
    
    # Motion Thresholds - Physical layer (7-9): Space, Time, Matter
    'THRESHOLD_SPACE', 'THRESHOLD_TIME', 'THRESHOLD_MATTER',
    
    # Motion Thresholds - Coupling layer (10-12): α, Ψ, Γ
    'THRESHOLD_ALPHA', 'THRESHOLD_BETA', 'THRESHOLD_PSI', 'THRESHOLD_GAMMA',
    'FINE_STRUCTURE_CONSTANT',
    
    # Coupling functions
    'euler_beta', 'wave_function', 'gamma_function', 'entropy_count',
    
    # Threshold collections
    'MOTION_THRESHOLDS', 'ABSTRACT_THRESHOLDS', 'PHYSICAL_THRESHOLDS', 'COUPLING_THRESHOLDS',
    'K_PHYSICAL', 'K_COUPLING', 'K_TOTAL', 'K_INFINITE', 'K_RESIDUE',
    
    # Motion Costs (heat expenditure)
    'COST_HEAT', 'COST_POLARITY', 'COST_EXISTENCE',
    'COST_RIGHTEOUSNESS', 'COST_ORDER', 'COST_MOVEMENT',
    'COST_SPACE', 'COST_TIME', 'COST_MATTER',
    'COST_ALPHA', 'COST_BETA', 'COST_GAMMA',
    'MOTION_COSTS',
    'COST_TRAVERSE', 'COST_CREATE_NODE', 'COST_EVALUATE', 'COST_ACTION', 'COST_TICK', 'COST_COLLAPSE',
    
    # Tick Configuration
    'TICK_INTERVAL_BASE', 'TICK_INTERVAL_MIN', 'TICK_INTERVAL_MAX',
    'TICK_HEAT_HOT', 'TICK_HEAT_COLD',
    'SAVE_INTERVAL_TICKS', 'SAVE_INTERVAL_SECONDS',
    'PSYCHOLOGY_MIN_HEAT',
    
    # Direction/Motion Mappings
    'DIRECTIONS', 'OPPOSITES', 'SELF_DIRECTIONS', 'ALL_DIRECTIONS',
    'DIRECTION_TO_MOTION', 'MOTION_TO_DIRECTION',
    
    # Existence States
    'EXISTENCE_ACTUAL', 'EXISTENCE_DORMANT', 'EXISTENCE_ARCHIVED', 'EXISTENCE_POTENTIAL',
    
    # Capability Levels
    'CAPABILITY_RIGHTEOUS', 'CAPABILITY_PROPER', 'CAPABILITY_ORDERED', 'CAPABILITY_MOVABLE', 'CAPABILITY_GRAPHIC',
    
    # Entropy Weights (golden angle trigonometry)
    'ENTROPY_MAGNITUDE_WEIGHT', 'ENTROPY_VARIANCE_WEIGHT', 'ENTROPY_DISORDER_WEIGHT',
    
    # Psychology Trig Positions (abstract space basis vectors)
    'TRIG_IDENTITY', 'TRIG_EGO', 'TRIG_CONSCIENCE',
    'trig_position_to_string', 'string_to_trig_position',
    'is_trig_position', 'is_cubic_position',
    
    # Freudian Heat Distribution Ratios
    'FREUD_IDENTITY_RATIO', 'FREUD_CONSCIENCE_RATIO', 'FREUD_EGO_RATIO',
    
    # Threshold Functions
    'get_threshold', 'get_threshold_for_direction',
    'exceeds_threshold', 'exceeds_any_threshold', 'exceeds_all_thresholds',
    'quantize_to_threshold',
    
    # Cost Functions
    'get_cost', 'get_cost_for_direction',
    
    # Other Utilities
    'heat_required', 'get_opposite', 'direction_to_vector',
    'get_motion_for_direction', 'get_direction_for_motion',
    'get_project_root', 'get_growth_path', 'GROWTH_DEFAULT',
    
    # Compression
    'compress', 'decompress', 'validate_position',
    'get_axis_coordinates', 'get_depth', 'position_length',
    'positions_share_prefix', 'run_compression_tests',
    
    # Nodes & Axes (v2 unified architecture with nested capability)
    'Element', 'Graphic', 'Transition', 'Movement', 'Order', 'Axis', 'Frame',
    'Node', 'SelfNode', 'assert_self_valid',
    'migrate_node_v1_to_v2',
    # Birth system
    'birth_randomizer', 'mark_birth_complete', 'is_birth_spent', 'reset_birth_for_testing',
    
    # Manifold (includes psychology: get_confidence, should_exploit, validate_conscience)
    'Manifold', 'create_manifold', 'get_pbai_manifold', 'reset_pbai_manifold',
    
    # Clock Node (Self IS the clock - existence = ticking)
    'Clock', 'TickStats', 'create_clock',
    
    # Driver Components (sensors, motors, plans, driver node)
    'SensorReport', 'MotorType', 'MotorAction', 'ActionPlan', 'DriverNode',
    'press', 'hold_key', 'release_key', 'look', 'mouse_hold', 'mouse_release', 'wait',
    'mouse_move', 'mouse_click', 'api_call', 'create_plan',
    
    # Decision Node (EXIT point - 5 scalars → 1 vector - saves to decisions/)
    'Choice', 'ChoiceNode', 'DecisionNode', 'EnvironmentNode', 'PBAILoop',
    'get_decisions_path',
    
    # Environment (from drivers/)
    'PortState', 'PortMessage', 'Port', 'NullPort',
    'Perception', 'Action', 'ActionResult', 'Driver', 'MockDriver',
    'EnvironmentCore', 'DriverLoader', 'create_environment_core',
]
