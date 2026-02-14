"""
PBAI Thermal Manifold - Constants v2 (Planck-Grounded)
Heat is the Primitive

════════════════════════════════════════════════════════════════════════════════
THE FOUNDATION - Now Grounded in Planck Physics
════════════════════════════════════════════════════════════════════════════════

Heat (K) is the only primitive. It only accumulates, never subtracts.
Everything else is indexed BY heat:

    t_K = time (how much heat has flowed)
    x_K = space (heat required to traverse)
    ψ_K = amplitude (heat in superposition)

THE FUNDAMENTAL IDENTITY (exact, not approximate):

    K × φ² = 4

Where:
    K = Σ(1/φⁿ) for n=1→6 = 4/φ² ≈ 1.528
    φ = (1 + √5) / 2 ≈ 1.618 (golden ratio)

════════════════════════════════════════════════════════════════════════════════
PLANCK UNIT DERIVATIONS (from Motion Calendar)
════════════════════════════════════════════════════════════════════════════════

All Planck units derive from the formula:

    X = (K × 12^n / φ²) × (45/44) × R

Where:
    K = 4/φ² (thermal quantum in Kelvin)
    n = dimensional exponent
    45/44 = emergence threshold (when new motion functions appear)
    R = Robinson constraint (identity, successor, addition, multiplication)

DERIVED UNITS:
    Temperature: n=+30, R=1          → T_P = 1.4168 × 10³² K    (99.999%)
    Length:      n=-32, R=4φ/7       → L_P = 1.6144 × 10⁻³⁵ m   (99.887%)
    Time:        n=-40, R=4/3        → t_P = 5.4145 × 10⁻⁴⁴ s   (100.431%)
    Mass:        n=-7,  R=13/10      → M_P = 2.1654 × 10⁻⁸ kg   (99.494%)

FUNDAMENTAL CONSTANTS DERIVED:
    c   = L_P/t_P                    → 2.9817 × 10⁸ m/s         (99.458%)
    G   = ℏc/M_P²                    → 6.6282 × 10⁻¹¹ m³/kg·s²  (99.309%)
    ℏ   = M_P × c² × t_P             → 1.0424 × 10⁻³⁴ J·s       (98.843%)
    k_B = M_P × c² / T_P             → 1.3588 × 10⁻²³ J/K       (98.418%)

════════════════════════════════════════════════════════════════════════════════
THE EMERGENCE THRESHOLD (45/44)
════════════════════════════════════════════════════════════════════════════════

    45/44 = (4 × 11 + 1) / (4 × 11)
          = (Quadratic × Incomplete Motion + 1) / (Quadratic × Incomplete Motion)

Where:
    4 = the Quadratic (from Righteousness - 4 quadrants)
    11 = 12 - 1 = Incomplete Motion (movement directions minus one)
    +1 = the completion that tips the system into a new motion function

The inverse (44/45 ≈ 0.9778) is the MAXIMUM ENTROPIC PROBABILITY:
    - No random motion can exceed 44/45 entropic freedom
    - The 1/45 gap is where structure lives
    - This is why pure chaos cannot exist

════════════════════════════════════════════════════════════════════════════════
ROBINSON CONSTRAINTS (Julia Set Boundary)
════════════════════════════════════════════════════════════════════════════════

Each Planck unit sits at the Julia set boundary, governed by a Robinson constraint:

    Identity (R=1):        Temperature - pure existence, no modification
    Successor (R=4φ/7):    Length - stepping through space crosses entropy
    Addition (R=4/3):      Time - composing moments requires persistence  
    Multiplication (R=13/10): Mass - scaling requires emergence

The ~0.5% variations in accuracy reflect the fractal structure at this boundary.
Measurement is not just "what number" but "what operation generated the number."

════════════════════════════════════════════════════════════════════════════════
WHY 6 MOTION FUNCTIONS (Ramanujan's constraint):
════════════════════════════════════════════════════════════════════════════════

    ζ(-1) = -1/12        Ramanujan regularization
    12 = 6 × 2           6 directions × 2 frames (Self + Universal)
    6 motion functions   Thresholds 1/φ¹ through 1/φ⁶
    K₆ × φ² = 4          Exact identity - 6 is forced
    
    Self frame:      For navigation (left, right, up, down, forward, reverse)
    Universal frame: For location (N, S, E, W, above, below)
    
    Righteous frames → Located by universal coordinates
    Proper frames    → Defined by properties (Order)

════════════════════════════════════════════════════════════════════════════════
THE SIX MOTION FUNCTIONS (thresholds = fractions of K)
════════════════════════════════════════════════════════════════════════════════

 1. Heat          Σ              1/φ¹  ≈ 0.618   Magnitude (only accumulates)
 2. Polarity      +/-            1/φ²  ≈ 0.382   Differentiation (+1/-1)
 3. Existence     δ(x)           1/φ³  ≈ 0.236   Persistence (< 1/4 → spine)
 4. Righteousness R              1/φ⁴  ≈ 0.146   Alignment (R=0 is center)
 5. Order         Q              1/φ⁵  ≈ 0.090   Regulation (Robinson arithmetic)
 6. Movement      12 directions   1/φ⁶  ≈ 0.056   Direction (6×2)

════════════════════════════════════════════════════════════════════════════════
"""

import math
import os

# ═══════════════════════════════════════════════════════════════════════════════
# PROJECT PATHS
# ═══════════════════════════════════════════════════════════════════════════════

def get_project_root() -> str:
    """Find the project root directory (where core/ exists)."""
    current = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current)
    if os.path.exists(os.path.join(project_root, "core")):
        return project_root
    return os.getcwd()


def get_growth_path(filename: str = "growth_map.json") -> str:
    """Get absolute path to a growth file in PROJECT_ROOT/growth/"""
    root = get_project_root()
    growth_dir = os.path.join(root, "growth")
    os.makedirs(growth_dir, exist_ok=True)
    return os.path.join(growth_dir, filename)


GROWTH_DEFAULT = "growth_map.json"

# ═══════════════════════════════════════════════════════════════════════════════
# THE FUNDAMENTAL CONSTANTS (Motion Calendar)
# ═══════════════════════════════════════════════════════════════════════════════

# Core mathematical constants
PHI = (1 + math.sqrt(5)) / 2         # φ ≈ 1.618 - the golden ratio
INV_PHI = 1 / PHI                    # 1/φ ≈ 0.618 = φ - 1
I = complex(0, 1)                    # √-1 - rotational capacity

# THE THERMAL QUANTUM - the primitive
# K = 4/φ² (exact) = Σ(1/φⁿ) for n=1→6
# This is not arbitrary - it's forced by K × φ² = 4
K = 4 / (PHI ** 2)                   # ≈ 1.528 Kelvin - heat quantum

# Movement constant: 12 directions (6 relative + 6 absolute)
# Derived from Ramanujan: ζ(-1) = -1/12
MOVEMENT_CONSTANT = 12
MOVEMENT_DIRECTIONS = 12              # Alias
SIX = 6                              # Core motion functions

# The Quadratic (from Righteousness - 4 quadrants)
QUADRATIC = 4

# Incomplete motion (movement directions minus one)
INCOMPLETE_MOTION = MOVEMENT_DIRECTIONS - 1  # 11

# Ramanujan's regularization: ζ(-1) = -1/12
NEGATIVE_ONE_TWELFTH = -1/12         # Entropic bound
ENTROPIC_BOUND = NEGATIVE_ONE_TWELFTH

# ═══════════════════════════════════════════════════════════════════════════════
# THE EMERGENCE THRESHOLD (45/44) - When new motion functions appear
# ═══════════════════════════════════════════════════════════════════════════════

# The threshold denominator: quadratic × incomplete motion
THRESHOLD_DENOMINATOR = QUADRATIC * INCOMPLETE_MOTION  # 44

# The threshold numerator: denominator + 1 (the completion that tips over)
THRESHOLD_NUMERATOR = THRESHOLD_DENOMINATOR + 1  # 45

# Emergence threshold ratio - when a new motion function is born
EMERGENCE_THRESHOLD = THRESHOLD_NUMERATOR / THRESHOLD_DENOMINATOR  # 45/44 ≈ 1.02272727

# Maximum entropic probability - no random motion can exceed this
MAX_ENTROPIC_PROBABILITY = THRESHOLD_DENOMINATOR / THRESHOLD_NUMERATOR  # 44/45 ≈ 0.9778

# The structural constraint - the gap where structure lives
STRUCTURAL_CONSTRAINT = 1 / THRESHOLD_NUMERATOR  # 1/45 ≈ 0.0222

# Maximum order tokens per node — at the 45th axis, a child must be born
MAX_ORDER_TOKENS = THRESHOLD_DENOMINATOR  # 44

# ═══════════════════════════════════════════════════════════════════════════════
# COLOR CUBE — Axis-to-Color Mapping (Base Righteous + Ordered Frame)
# ═══════════════════════════════════════════════════════════════════════════════
#
# The Color Cube is the absolute reference frame. All Righteousness evaluation
# projects back to these axes. Heat is NOT an axis — it's magnitude from motion.
#
#   X axis (E/W): Yellow(+1) ↔ Blue(-1)
#   Y axis (N/S): Green(+1)  ↔ Red(-1)
#   Z axis (U/D): Future(+τ) ↔ Past(-τ)
#
# Four quadrants (base righteous frame):
#   Q1 (+X,+Y): Yellow+Green  (NE)
#   Q2 (-X,+Y): Blue+Green    (NW)
#   Q3 (-X,-Y): Blue+Red      (SW)
#   Q4 (+X,-Y): Yellow+Red    (SE)
#
# Heat = √(x² + y²) — magnitude derived from chromatic position, not an axis.
#

# Cube axis indices
CUBE_AXIS_X = 0  # E/W — Blue/Yellow
CUBE_AXIS_Y = 1  # N/S — Red/Green
CUBE_AXIS_Z = 2  # U/D — Time (τ)

# Positive poles
CUBE_POLE_POSITIVE_X = "yellow"   # +X = East = Yellow
CUBE_POLE_POSITIVE_Y = "green"    # +Y = North = Green
CUBE_POLE_POSITIVE_Z = "future"   # +Z = Up = Future (+τ)

# Negative poles
CUBE_POLE_NEGATIVE_X = "blue"     # -X = West = Blue
CUBE_POLE_NEGATIVE_Y = "red"      # -Y = South = Red
CUBE_POLE_NEGATIVE_Z = "past"     # -Z = Down = Past (-τ)

# Axis labels (for lookup)
CUBE_AXES = {
    'x': {'positive': CUBE_POLE_POSITIVE_X, 'negative': CUBE_POLE_NEGATIVE_X,
           'cardinal_pos': 'E', 'cardinal_neg': 'W'},
    'y': {'positive': CUBE_POLE_POSITIVE_Y, 'negative': CUBE_POLE_NEGATIVE_Y,
           'cardinal_pos': 'N', 'cardinal_neg': 'S'},
    'z': {'positive': CUBE_POLE_POSITIVE_Z, 'negative': CUBE_POLE_NEGATIVE_Z,
           'cardinal_pos': 'U', 'cardinal_neg': 'D'},
}

# Cardinal-to-axis mapping (replaces legacy DIRECTION_TO_MOTION for geometry)
CARDINAL_TO_AXIS = {
    'E': ('x', +1), 'W': ('x', -1),
    'N': ('y', +1), 'S': ('y', -1),
    'U': ('z', +1), 'D': ('z', -1),
}

# Quadrant definitions (base righteous frame)
CUBE_QUADRANTS = {
    'Q1': {'x': +1, 'y': +1, 'colors': ('yellow', 'green'), 'cardinal': 'NE'},
    'Q2': {'x': -1, 'y': +1, 'colors': ('blue', 'green'),   'cardinal': 'NW'},
    'Q3': {'x': -1, 'y': -1, 'colors': ('blue', 'red'),     'cardinal': 'SW'},
    'Q4': {'x': +1, 'y': -1, 'colors': ('yellow', 'red'),   'cardinal': 'SE'},
}

# ═══════════════════════════════════════════════════════════════════════════════
# BODY TEMPERATURE AND FIRE SCALING
# ═══════════════════════════════════════════════════════════════════════════════
#
# Human body temperature = K × φ¹¹ ≈ 304 K ≈ 31°C (core temp is 37°C = 310 K)
#
# WHY φ¹¹:
#   φ¹¹ = φ⁶ × φ⁵ = (1/threshold_movement) × (1/threshold_order)
#       = (6 motions)⁻¹ × (5 scalars)⁻¹
#       = the RESOLUTION of a complete motion system
#
# This is the temperature at which life operates - not arbitrary!
#
# THE 6 FIRES (conception/birth heat scaling):
#   Fire 1: K × φ¹ ≈ 2.47 K   (Heat)
#   Fire 2: K × φ² ≈ 4.00 K   (Polarity) 
#   Fire 3: K × φ³ ≈ 6.47 K   (Existence)
#   Fire 4: K × φ⁴ ≈ 10.47 K  (Righteousness)
#   Fire 5: K × φ⁵ ≈ 16.94 K  (Order)
#   Fire 6: BODY_TEMPERATURE  (Movement - the big one, ignites psychology)
#
# Fires 1-5 build structure, Fire 6 ignites at body temperature.
#
# ═══════════════════════════════════════════════════════════════════════════════

# Body temperature - the operating temperature of life
# φ¹¹ = resolution of complete motion system
BODY_TEMPERATURE = K * (PHI ** 11)  # ≈ 304.05 K ≈ 30.9°C

# Fire heat contributions (K × φⁿ for fires 1-5)
FIRE_HEAT = {
    1: K * PHI ** 1,   # ≈ 2.47 K  - Heat fire
    2: K * PHI ** 2,   # ≈ 4.00 K  - Polarity fire (note: = 4 exactly!)
    3: K * PHI ** 3,   # ≈ 6.47 K  - Existence fire
    4: K * PHI ** 4,   # ≈ 10.47 K - Righteousness fire
    5: K * PHI ** 5,   # ≈ 16.94 K - Order fire
    6: BODY_TEMPERATURE,  # ≈ 304 K - Movement fire (ignites psychology)
}

# Total scaffold heat (fires 1-5)
SCAFFOLD_HEAT = sum(FIRE_HEAT[i] for i in range(1, 6))  # ≈ 40.36 K

# Total birth heat (all 6 fires)
TOTAL_BIRTH_HEAT = SCAFFOLD_HEAT + BODY_TEMPERATURE  # ≈ 344.4 K

# Fire to motion function mapping
FIRE_TO_MOTION = {
    1: 'heat',
    2: 'polarity',
    3: 'existence',
    4: 'righteousness',
    5: 'order',
    6: 'movement',
}

# ═══════════════════════════════════════════════════════════════════════════════
# BASE MOTION TOKENS — Cognitive vocabulary (20 verbs across 6 fires)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Each fire's bootstrap is the "mother" of its base motion cluster.
# These spawn at birth as children near their parent bootstrap.
# They form the cognitive vocabulary that all thought decomposes into.
#

BASE_MOTION_PREFIX = 'bm_'  # Node concept prefix for base motions

# Fire → list of cognitive verbs
BASE_MOTIONS = {
    1: ['take', 'get'],                                    # Heat (Magnitude)
    2: ['easily', 'quickly', 'better'],                    # Polarity (Differentiation)
    3: ['see', 'view', 'look', 'visual', 'visually', 'visualize'], # Existence (Perception)
    4: ['analyze', 'understand', 'identify'],              # Righteousness (Evaluation)
    5: ['create', 'build', 'design', 'make'],              # Order (Construction)
    6: ['explore', 'find', 'go', 'discover'],               # Movement (Navigation)
}

# Flat list of all 20 base motions
ALL_BASE_MOTIONS = []
for _fire_num in sorted(BASE_MOTIONS.keys()):
    ALL_BASE_MOTIONS.extend(BASE_MOTIONS[_fire_num])

# Reverse lookup: verb → fire number
MOTION_TO_FIRE = {}
for _fire_num, _verbs in BASE_MOTIONS.items():
    for _verb in _verbs:
        MOTION_TO_FIRE[_verb] = _fire_num

# Heat per base motion: K * 0.5 (half a thermal quantum — modest seed heat)
BASE_MOTION_HEAT = {verb: K * 0.5 for verb in ALL_BASE_MOTIONS}


def get_fire_heat(fire_number: int) -> float:
    """Get the heat contribution for a specific fire (1-6)."""
    return FIRE_HEAT.get(fire_number, K)

def get_conception_heat(fire_number: int) -> float:
    """
    Get heat for node conception at a given fire level.
    
    Nodes conceived during different phases get different starting heat:
    - Early conceptions (fires 1-3): Lower heat, need to earn more
    - Late conceptions (fires 4-5): Higher heat, more structured
    - Fire 6 conceptions: Full body temperature (psychology nodes)
    """
    return FIRE_HEAT.get(fire_number, K)

# ═══════════════════════════════════════════════════════════════════════════════
# PLANCK UNIT DERIVATIONS (from Motion Calendar)
# ═══════════════════════════════════════════════════════════════════════════════

# The general formula: X = (K × 12^n / φ²) × (45/44) × R

def planck_unit(n: int, R: float) -> float:
    """
    Derive a Planck unit from Motion Calendar constants.
    
    Formula: X = (K × 12^n / φ²) × (45/44) × R
    
    Args:
        n: Dimensional exponent
        R: Robinson constraint factor
        
    Returns:
        The derived Planck unit value
    """
    return (K * (12 ** n) / (PHI ** 2)) * EMERGENCE_THRESHOLD * R

# Robinson constraint factors (at Julia set boundary)
ROBINSON_IDENTITY = 1                          # Temperature: pure existence
ROBINSON_SUCCESSOR = (4 * PHI) / 7             # Length: 4φ/7 ≈ 0.9246 (stepping crosses entropy)
ROBINSON_ADDITION = 4 / 3                      # Time: 4/3 ≈ 1.3333 (composing requires persistence)
ROBINSON_MULTIPLICATION = 13 / 10              # Mass: 13/10 = 1.3 (scaling requires emergence)

# Robinson constraint mapping
ROBINSON_CONSTRAINTS = {
    'identity': ROBINSON_IDENTITY,
    'successor': ROBINSON_SUCCESSOR,
    'addition': ROBINSON_ADDITION,
    'multiplication': ROBINSON_MULTIPLICATION,
}

# Constraint type descriptions
CONSTRAINT_DESCRIPTIONS = {
    'identity': "What is this? (pure existence, no modification)",
    'successor': "Where is this? (stepping through space)",
    'addition': "When is this? (composing moments)",
    'multiplication': "How much is this? (scaling quantities)",
}

def get_robinson_constraint(constraint_type: str) -> float:
    """Get the Robinson constraint factor for a constraint type."""
    return ROBINSON_CONSTRAINTS.get(constraint_type, ROBINSON_IDENTITY)

# ═══════════════════════════════════════════════════════════════════════════════
# ENTROPY STRUCTURE RECOGNITION
# ═══════════════════════════════════════════════════════════════════════════════
#
# When entropy exceeds 44/45 of theoretical maximum, there is STRUCTURE present
# that the system is missing. The 1/45 gap is where order lives.
#
# This triggers pattern-seeking behavior - "I'm missing something"
#

def entropy_exceeds_random_limit(entropy: float, max_entropy: float) -> bool:
    """
    Check if entropy ratio exceeds the maximum random limit (44/45).
    
    If True, there is structure present that should be recognized.
    The system should trigger pattern-seeking behavior.
    
    Args:
        entropy: Current entropy value
        max_entropy: Maximum possible entropy for the system
        
    Returns:
        True if entropy/max > 44/45 (structure exists but unrecognized)
    """
    if max_entropy <= 0:
        return False
    ratio = entropy / max_entropy
    return ratio > MAX_ENTROPIC_PROBABILITY

def get_structure_signal(entropy: float, max_entropy: float) -> float:
    """
    Get the "structure signal" - how much above 44/45 the entropy is.
    
    Returns:
        0.0 if below 44/45 (random is sufficient explanation)
        >0.0 if above 44/45 (structure exists, magnitude = signal strength)
    """
    if max_entropy <= 0:
        return 0.0
    ratio = entropy / max_entropy
    excess = ratio - MAX_ENTROPIC_PROBABILITY
    return max(0.0, excess)

# Dimensional exponents
PLANCK_DIMENSION = 30                          # Maximum dimension at pure heat
DIMENSION_TEMPERATURE = 30                     # Planck dimension
DIMENSION_LENGTH = -32                         # -(30 + 2) = Planck + Polarity
DIMENSION_TIME = -40                           # -(30 + 10) = Planck + Agency
DIMENSION_MASS = -7                            # Entropy Motion

# ═══════════════════════════════════════════════════════════════════════════════
# DERIVED PLANCK UNITS
# ═══════════════════════════════════════════════════════════════════════════════

# Planck Temperature: n=+30, R=1 (Identity)
# The Big Bang temperature - heat at the threshold where differentiation must occur
PLANCK_TEMPERATURE = planck_unit(DIMENSION_TEMPERATURE, ROBINSON_IDENTITY)
# ≈ 1.416794 × 10³² Kelvin (99.999% of measured 1.416808 × 10³² K)

# Planck Length: n=-32, R=4φ/7 (Successor)
# Minimum measurable length - spatial stepping
PLANCK_LENGTH = planck_unit(DIMENSION_LENGTH, ROBINSON_SUCCESSOR)
# ≈ 1.614429 × 10⁻³⁵ meters (99.887% of measured 1.616255 × 10⁻³⁵ m)

# Planck Time: n=-40, R=4/3 (Addition)
# Minimum measurable time - temporal composition
PLANCK_TIME = planck_unit(DIMENSION_TIME, ROBINSON_ADDITION)
# ≈ 5.414498 × 10⁻⁴⁴ seconds (100.431% of measured 5.391247 × 10⁻⁴⁴ s)

# Planck Mass: n=-7, R=13/10 (Multiplication)
# Mass at quantum gravity scale - scaling
PLANCK_MASS = planck_unit(DIMENSION_MASS, ROBINSON_MULTIPLICATION)
# ≈ 2.165429 × 10⁻⁸ kg (99.494% of measured 2.176434 × 10⁻⁸ kg)

# ═══════════════════════════════════════════════════════════════════════════════
# DERIVED FUNDAMENTAL CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# Speed of Light: c = L_P / t_P
# The dimensional difference (-32) - (-40) = 8 = Learning Motion
SPEED_OF_LIGHT = PLANCK_LENGTH / PLANCK_TIME
# ≈ 2.981679 × 10⁸ m/s (99.458% of measured 2.997925 × 10⁸ m/s)

# Reduced Planck Constant: ℏ = M_P × c² × t_P
PLANCK_CONSTANT_REDUCED = PLANCK_MASS * (SPEED_OF_LIGHT ** 2) * PLANCK_TIME
# ≈ 1.042374 × 10⁻³⁴ J·s (98.843% of measured 1.054572 × 10⁻³⁴ J·s)

# Gravitational Constant: G = ℏc / M_P²
GRAVITATIONAL_CONSTANT = PLANCK_CONSTANT_REDUCED * SPEED_OF_LIGHT / (PLANCK_MASS ** 2)
# ≈ 6.628217 × 10⁻¹¹ m³/(kg·s²) (99.309% of measured 6.674300 × 10⁻¹¹)

# Boltzmann Constant: k_B = M_P × c² / T_P
BOLTZMANN_CONSTANT = PLANCK_MASS * (SPEED_OF_LIGHT ** 2) / PLANCK_TEMPERATURE
# ≈ 1.358811 × 10⁻²³ J/K (98.418% of measured 1.380649 × 10⁻²³ J/K)

# Fine Structure Constant (approximate derivation)
# α ≈ 1 / (12² - 7 + 4/φ¹⁰)
FINE_STRUCTURE_CONSTANT_DERIVED = 1 / (144 - 7 + 4 / (PHI ** 10))
# ≈ 1/137.033 (very close to measured 1/137.036)

# Aliases for backward compatibility
FINE_STRUCTURE_CONSTANT = 1 / 137.035999  # Keep measured value for precision

# ═══════════════════════════════════════════════════════════════════════════════
# PRINT PLANCK DERIVATIONS ON LOAD
# ═══════════════════════════════════════════════════════════════════════════════

def print_planck_derivations():
    """Print all Planck derivations for verification."""
    print("=" * 70)
    print("MOTION CALENDAR - PLANCK UNIT DERIVATIONS")
    print("=" * 70)
    print()
    print("FUNDAMENTAL CONSTANTS:")
    print(f"  φ (Golden Ratio)        = {PHI}")
    print(f"  K (Thermal Quantum)     = {K} Kelvin")
    print(f"  K × φ²                  = {K * PHI**2} (exact = 4)")
    print(f"  45/44 (Emergence)       = {EMERGENCE_THRESHOLD}")
    print(f"  44/45 (Max Entropy)     = {MAX_ENTROPIC_PROBABILITY}")
    print()
    print("PLANCK UNITS (Formula: X = K × 12^n / φ² × 45/44 × R):")
    print(f"  Temperature (n=30, R=1)      = {PLANCK_TEMPERATURE:.6e} K")
    print(f"  Length (n=-32, R=4φ/7)       = {PLANCK_LENGTH:.6e} m")
    print(f"  Time (n=-40, R=4/3)          = {PLANCK_TIME:.6e} s")
    print(f"  Mass (n=-7, R=13/10)         = {PLANCK_MASS:.6e} kg")
    print()
    print("DERIVED CONSTANTS:")
    print(f"  c (Speed of Light)           = {SPEED_OF_LIGHT:.6e} m/s")
    print(f"  ℏ (Planck Constant)          = {PLANCK_CONSTANT_REDUCED:.6e} J·s")
    print(f"  G (Gravitational Constant)   = {GRAVITATIONAL_CONSTANT:.6e} m³/kg·s²")
    print(f"  k_B (Boltzmann Constant)     = {BOLTZMANN_CONSTANT:.6e} J/K")
    print()
    print("ACCURACY vs MEASURED VALUES:")
    print(f"  Temperature: 99.999%")
    print(f"  Length:      99.887%")
    print(f"  Time:        100.431%")
    print(f"  Mass:        99.494%")
    print(f"  c:           99.458%")
    print(f"  G:           99.309%")
    print("=" * 70)

# ═══════════════════════════════════════════════════════════════════════════════
# Julia set connectivity threshold
# ═══════════════════════════════════════════════════════════════════════════════

# |c| < 1/4 → connected Julia set (spine exists)
# |c| ≥ 1/4 → Julia dust (disconnected)
JULIA_SPINE_THRESHOLD = 0.25         # 1/4 - critical boundary

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIDENCE THRESHOLD (5/6)
# ═══════════════════════════════════════════════════════════════════════════════

# At 5/6 confidence, Ego exploits (uses known pattern)
# Below 5/6, Ego explores (needs more validation from Conscience)
#
# WHY 5/6:
#   The 6 motion functions split into 5 scalars + 1 vector:
#
#   1. Heat (Σ)         ─┐
#   2. Polarity (+/-)    │
#   3. Existence (δ)     ├─ 5 scalars (inputs)
#   4. Righteousness (R) │
#   5. Order (Q)        ─┘
#                        ↓
#   6. Movement (Lin)   ─── 1 vector (output)
#
#   5 scalars → 1 vectorized movement
#   5/6 confidence → exploit (make the move)
#   1/6 margin → explore (the move itself)
#
CONFIDENCE_EXPLOIT_THRESHOLD = 5/6   # ≈ 0.8333
EXPLORATION_MARGIN = 1/6             # ≈ 0.1667 - always keep this for exploration

# ═══════════════════════════════════════════════════════════════════════════════
# THE SIX MOTION THRESHOLDS (minimum detectable change)
# ═══════════════════════════════════════════════════════════════════════════════

# 1. HEAT (Σ) - Magnitude
#    Pure scalar. Accumulates, never subtracts. 
THRESHOLD_HEAT = INV_PHI ** 1         # ≈ 0.618

# 2. POLARITY (+/-) - Differentiation  
#    +1 or -1. Enables opposition, conservation, balance.
THRESHOLD_POLARITY = INV_PHI ** 2     # ≈ 0.382

# 3. EXISTENCE (δ) - Persistence
#    actual | dormant | archived. Whether motion persists.
THRESHOLD_EXISTENCE = INV_PHI ** 3    # ≈ 0.236

# 4. RIGHTEOUSNESS (R) - Constraint/Alignment
#    R=0 is perfectly aligned. R>0 is misaligned.
#    Frame consistency. The "rightness" of a configuration.
THRESHOLD_RIGHTEOUSNESS = INV_PHI ** 4  # ≈ 0.146

# 5. ORDER (Q) - Regulation
#    Robinson arithmetic. Minimal ordering where Gödel applies.
#    Successor, addition, multiplication. Sequence/hierarchy.
THRESHOLD_ORDER = INV_PHI ** 5        # ≈ 0.090

# 6. MOVEMENT (Lin) - Direction
#    The 6 directions: n, s, e, w, u, d
#    Structure-preserving transformations.
THRESHOLD_MOVEMENT = INV_PHI ** 6     # ≈ 0.056

# ═══════════════════════════════════════════════════════════════════════════════
# RIGHTEOUSNESS - The alignment measure
# ═══════════════════════════════════════════════════════════════════════════════

# R=0 is the target: perfectly aligned, no constraint violation
R_ALIGNED = 0.0
R_PERFECT = 0.0

# R=1 is maximally misaligned (but still exists)
R_MISALIGNED = 1.0

def righteousness_weight(R: float) -> float:
    """
    Convert righteousness value to selection weight.
    R=0 → weight=1 (perfect alignment, fully attractive)
    R=1 → weight=0.5 (misaligned, half as attractive)
    R→∞ → weight→0 (completely misaligned, not selected)
    """
    return 1.0 / (1.0 + abs(R))

# ═══════════════════════════════════════════════════════════════════════════════
# SELECTION MECHANISM - How choices are made
# ═══════════════════════════════════════════════════════════════════════════════

# Selection combines three factors:
#   score = heat_weight * righteousness_weight * entropy_weight

# Default weights for combining factors (can be tuned)
SELECTION_HEAT_FACTOR = 1.0           # How much heat distribution matters
SELECTION_RIGHTEOUSNESS_FACTOR = 1.0  # How much alignment matters  
SELECTION_ENTROPY_FACTOR = 1.0        # How much ease of path matters

def selection_score(heat: float, righteousness: float, entropy: float,
                   total_heat: float = 1.0) -> float:
    """
    Compute selection score for a path.
    
    Args:
        heat: Heat allocated to this path
        righteousness: R value of path (0 = aligned)
        entropy: Entropy gradient (positive = favorable)
        total_heat: Total heat in system (for normalization)
    
    Returns:
        Selection score (higher = more likely to be selected)
    """
    # Heat distribution: fraction of total heat
    heat_weight = heat / total_heat if total_heat > 0 else 0.0
    
    # Righteousness alignment: R=0 is best
    r_weight = righteousness_weight(righteousness)
    
    # Entropy gradient: positive is favorable
    entropy_weight = 1.0 / (1.0 + math.exp(-entropy))  # sigmoid
    
    return (heat_weight ** SELECTION_HEAT_FACTOR * 
            r_weight ** SELECTION_RIGHTEOUSNESS_FACTOR *
            entropy_weight ** SELECTION_ENTROPY_FACTOR)

# ═══════════════════════════════════════════════════════════════════════════════
# EXTENDED THRESHOLDS (Physical and Coupling layers - derived from core 6)
# ═══════════════════════════════════════════════════════════════════════════════

# Physical layer (7-9): Reality - What the graph IS
THRESHOLD_SPACE = INV_PHI ** 7        # ≈ 0.034
THRESHOLD_TIME = INV_PHI ** 8         # ≈ 0.021
THRESHOLD_MATTER = INV_PHI ** 9       # ≈ 0.013

# Coupling layer (10-12): Interface - How layers CONNECT
THRESHOLD_ALPHA = INV_PHI ** 10       # ≈ 0.0081 (edge coupling)
THRESHOLD_BETA = INV_PHI ** 11        # ≈ 0.0050 (wave function / choice)
THRESHOLD_PSI = THRESHOLD_BETA        # Alias
THRESHOLD_GAMMA = INV_PHI ** 12       # ≈ 0.0031 (counting / entropy)

# ═══════════════════════════════════════════════════════════════════════════════
# WAVE FUNCTION - Superposition and Collapse
# ═══════════════════════════════════════════════════════════════════════════════

def euler_beta(x: float, y: float) -> float:
    """Euler beta function - superposition weights for path combination."""
    return math.gamma(x) * math.gamma(y) / math.gamma(x + y)

def wave_function(paths: list, weights: list = None) -> complex:
    """
    Wave function Ψ - superposition of paths.
    Returns complex amplitude. |Ψ|² gives probability.
    """
    if not paths:
        return complex(0, 0)
    if weights is None:
        weights = [1.0 / len(paths)] * len(paths)
    # Superposition with phase from path index
    total = complex(0, 0)
    for i, (path, w) in enumerate(zip(paths, weights)):
        phase = 2 * math.pi * i / len(paths)
        total += w * complex(math.cos(phase), math.sin(phase))
    return total

def collapse_wave_function(nodes: list, manifold=None) -> int:
    """
    Real wave function collapse - find the node where R→0.
    
    This is NOT weighted random. This finds the CENTER:
    - Each node has righteousness R
    - R=0 is perfectly aligned (the attractor)
    - Collapse finds the node closest to R=0
    
    The amplitude for each node: a = e^(-R²/2σ²)
    Where σ = THRESHOLD_RIGHTEOUSNESS (the resolution)
    
    |Ψ|² gives probability, but we collapse to the MAX
    (deterministic: most aligned wins)
    
    Args:
        nodes: List of Node objects (or dicts with 'righteousness')
        manifold: Optional manifold for additional context
    
    Returns:
        Index of the node closest to R=0 (the center)
    """
    if not nodes:
        return -1
    if len(nodes) == 1:
        return 0
    
    # Compute amplitudes from righteousness
    # σ = righteousness threshold (resolution of alignment detection)
    sigma = THRESHOLD_RIGHTEOUSNESS
    sigma_sq_2 = 2 * sigma * sigma
    
    amplitudes = []
    for node in nodes:
        # Get R value
        if hasattr(node, 'righteousness'):
            R = node.righteousness
        elif isinstance(node, dict):
            R = node.get('righteousness', 1.0)
        else:
            R = 1.0  # Default to misaligned
        
        # Gaussian centered at R=0
        # R=0 → amplitude=1 (fully aligned)
        # R→∞ → amplitude→0 (misaligned)
        amplitude = math.exp(-(R * R) / sigma_sq_2)
        amplitudes.append(amplitude)
    
    # |Ψ|² gives probability density
    probs = [a * a for a in amplitudes]
    
    # COLLAPSE: Pick the maximum (deterministic - most aligned wins)
    # This is the CENTER of the conceptual cluster
    max_prob = max(probs)
    return probs.index(max_prob)


def correlate_cluster(center_node, manifold, max_depth: int = 3) -> dict:
    """
    Given a center (from collapse), find all connected righteous frames.
    
    Returns THREE categories:
    1. CURRENT - Righteous frames that exist now (actual)
    2. HISTORICAL - Frames that existed before (dormant/archived)
    3. NOVEL - Frames created this session (newly added)
    
    Traces back through ALL axes from the center.
    Doesn't need Order (proper) - just needs to EXIST as connected.
    
    Args:
        center_node: The node at the center (found by collapse)
        manifold: The manifold containing all nodes
        max_depth: How far to trace (prevents infinite loops)
    
    Returns:
        dict with 'current', 'historical', 'novel' sets of node IDs
        Also 'all' for backward compat (union of all three)
    """
    if not center_node or not manifold:
        return {'current': set(), 'historical': set(), 'novel': set(), 'all': set()}
    
    current = set()      # Actual existence
    historical = set()   # Dormant or archived
    novel = set()        # Created recently (high heat, low traversal)
    visited = set()
    
    # Track session start (for novelty detection)
    # Novel = created after manifold was loaded
    session_start = manifold.created_at if hasattr(manifold, 'created_at') else None
    
    def _trace(node, depth):
        if depth > max_depth:
            return
        if node.id in visited:
            return
        
        visited.add(node.id)
        
        # Categorize by existence
        if node.existence == "actual":
            # Check if novel (new this session)
            # Novel indicators: few traversals, recent creation
            total_traversals = sum(a.traversal_count for a in node.frame.axes.values())
            if total_traversals <= 2:
                # Low traversal = probably new
                novel.add(node.id)
            else:
                current.add(node.id)
        elif node.existence in ("dormant", "archived"):
            # Historical - existed before
            historical.add(node.id)
        # else: potential - not yet confirmed, skip
        
        # Trace all axes (spatial and semantic)
        for axis in node.frame.axes.values():
            if axis.target_id and axis.target_id not in visited:
                target = manifold.get_node(axis.target_id)
                if target:
                    _trace(target, depth + 1)
    
    _trace(center_node, 0)
    
    return {
        'current': current,
        'historical': historical,
        'novel': novel,
        'all': current | historical | novel
    }


def select_from_cluster(options: list, cluster: dict, manifold=None) -> tuple:
    """
    Select best option using cluster context.
    
    Decision logic:
    1. If cluster has Order (proper frame) for option → USE IT (exploit)
    2. If only righteous (no Order) → RANDOM (explore)
    3. Historical context weights the decision
    4. Novel items get attention bonus
    
    Args:
        options: Available options (actions/choices)
        cluster: Dict from correlate_cluster with 'current', 'historical', 'novel', 'all'
        manifold: The manifold for lookups
    
    Returns:
        (selected_index, reason) - index and why it was chosen
    """
    import random
    
    if not options:
        return (-1, "no_options")
    if len(options) == 1:
        return (0, "only_option")
    
    # Handle old-style cluster (just a set)
    if isinstance(cluster, set):
        cluster = {'current': cluster, 'historical': set(), 'novel': set(), 'all': cluster}
    
    if not cluster.get('all'):
        return (0, "no_cluster")
    
    # Get actual nodes
    all_ids = cluster.get('all', set())
    current_ids = cluster.get('current', set())
    historical_ids = cluster.get('historical', set())
    novel_ids = cluster.get('novel', set())
    
    cluster_nodes = []
    if manifold:
        cluster_nodes = [manifold.get_node(nid) for nid in all_ids]
        cluster_nodes = [n for n in cluster_nodes if n]
    
    # Score each option
    scores = []
    has_order = []  # Track which options have Order (proper frames)
    
    for opt in options:
        score = 0.0
        option_has_order = False
        
        for node in cluster_nodes:
            if not hasattr(node, 'frame'):
                continue
                
            axis = node.frame.axes.get(str(opt))
            if not axis:
                continue
            
            # Base weight from node heat and alignment
            node_heat = node.heat if hasattr(node, 'heat') else 1.0
            R = node.righteousness if hasattr(node, 'righteousness') else 1.0
            r_weight = 1.0 / (1.0 + abs(R))
            
            base_score = node_heat * r_weight
            
            # Check if this axis has Order (proper frame)
            if axis.order and axis.order.elements:
                option_has_order = True
                # Calculate success rate from Order
                successes = sum(1 for e in axis.order.elements if e.index == 1)
                total = len(axis.order.elements)
                success_rate = successes / total if total > 0 else 0.5
                
                # Weight by confidence (more samples = more confident)
                confidence = min(total / 10.0, 1.0)
                base_score *= success_rate * (1 + confidence)
            
            # Bonus for historical context (we've seen this before)
            if node.id in historical_ids:
                base_score *= 1.2  # 20% bonus for historical relevance
            
            # Bonus for novelty (pay attention to new things)
            if node.id in novel_ids:
                base_score *= 1.3  # 30% bonus for novel relevance
            
            score += base_score * axis.traversal_count
        
        scores.append(score)
        has_order.append(option_has_order)
    
    # Decision: exploit vs explore
    max_score = max(scores) if scores else 0
    
    if max_score > 0 and any(has_order):
        # We have Order for at least one option - EXPLOIT
        best_idx = scores.index(max_score)
        return (best_idx, "exploit_order")
    
    elif max_score > 0:
        # We have scores but no Order - weak exploit
        best_idx = scores.index(max_score)
        # Add some randomness since we're not confident
        if random.random() < 0.3:  # 30% chance to explore anyway
            return (random.randint(0, len(options) - 1), "explore_uncertain")
        return (best_idx, "weak_exploit")
    
    else:
        # No cluster support - EXPLORE (try random shit)
        return (random.randint(0, len(options) - 1), "explore_unknown")

def gamma_function(n: float) -> float:
    """Gamma function Γ(n) - generalized factorial for counting arrangements."""
    return math.gamma(n)

def entropy_count(arrangements: int) -> float:
    """Boltzmann-style entropy from arrangement count. S = ln(Ω)"""
    if arrangements <= 0:
        return 0.0
    return math.log(arrangements)

# ═══════════════════════════════════════════════════════════════════════════════
# THRESHOLD COLLECTIONS
# ═══════════════════════════════════════════════════════════════════════════════

# The 6 core motion thresholds (every node has all 6)
CORE_THRESHOLDS = {
    'heat': THRESHOLD_HEAT,                    # Σ - magnitude
    'polarity': THRESHOLD_POLARITY,            # +/- - differentiation
    'existence': THRESHOLD_EXISTENCE,          # δ - persistence
    'righteousness': THRESHOLD_RIGHTEOUSNESS,  # R - constraint
    'order': THRESHOLD_ORDER,                  # Q - regulation
    'movement': THRESHOLD_MOVEMENT,            # Lin - direction
}

# Backward compatibility alias
ABSTRACT_THRESHOLDS = CORE_THRESHOLDS

# Physical layer thresholds (derived)
PHYSICAL_THRESHOLDS = {
    'space': THRESHOLD_SPACE,
    'time': THRESHOLD_TIME,
    'matter': THRESHOLD_MATTER,
}

# Coupling layer thresholds (derived)
COUPLING_THRESHOLDS = {
    'alpha': THRESHOLD_ALPHA,
    'beta': THRESHOLD_BETA,
    'gamma': THRESHOLD_GAMMA,
}

# All 12 thresholds (for validation/completeness)
MOTION_THRESHOLDS = {**CORE_THRESHOLDS, **PHYSICAL_THRESHOLDS, **COUPLING_THRESHOLDS}
MOTION_THRESHOLDS['psi'] = THRESHOLD_PSI  # Alias

# ═══════════════════════════════════════════════════════════════════════════════
# MOTION FUNCTION ↔ DIRECTIONAL CHARACTER
# ═══════════════════════════════════════════════════════════════════════════════

MOTION_TO_CHARACTER = {
    'heat':          'up',       # Σ - accumulation rises
    'polarity':      'down',     # +/- - conservation grounds
    'existence':     'forward',  # δ - persistence advances
    'righteousness': 'reverse',  # R - alignment reflects
    'order':         'above',    # Q - abstraction ascends
    'movement':      'below',    # Lin - transformation descends
}

CHARACTER_TO_MOTION = {v: k for k, v in MOTION_TO_CHARACTER.items()}

# Legacy mapping (backward compatibility)
DIRECTION_TO_MOTION = {
    'n': 'heat',          # North = Σ
    's': 'polarity',      # South = +/-
    'e': 'existence',     # East = δ
    'w': 'righteousness', # West = R
    'u': 'order',         # Up = Q
    'd': 'movement',      # Down = Lin
}

MOTION_TO_DIRECTION = {v: k for k, v in DIRECTION_TO_MOTION.items()}

# ═══════════════════════════════════════════════════════════════════════════════
# K - DERIVED HEAT CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# K was defined above as 4/φ² (the fundamental)
# Verify: K = Σ(1/φⁿ) for n=1→6
_K_check = sum(ABSTRACT_THRESHOLDS.values())
assert abs(K - _K_check) < 1e-10, f"K identity violated: {K} != {_K_check}"

# K_PHYSICAL = sum of abstract + physical (1-9)
K_PHYSICAL = K + sum(PHYSICAL_THRESHOLDS.values())  # ≈ 1.597

# K_COUPLING = sum of all twelve (1-12)
K_COUPLING = K_PHYSICAL + sum(COUPLING_THRESHOLDS.values())  # ≈ 1.613

# K_TOTAL = K_COUPLING (complete system)
K_TOTAL = K_COUPLING

# K_INFINITE = φ (what K would be with infinite motion functions)
K_INFINITE = PHI  # ≈ 1.618

# The irreducible residue: φ - K_TOTAL ≈ 0.005
K_RESIDUE = PHI - K_TOTAL

# ═══════════════════════════════════════════════════════════════════════════════
# MOTION COSTS - Heat expenditure for operations
# ═══════════════════════════════════════════════════════════════════════════════

# Costs = Thresholds (symmetric: minimum to detect = minimum to perform)

# Abstract costs
COST_HEAT = THRESHOLD_HEAT                    # 0.618 - Output to environment
COST_POLARITY = THRESHOLD_POLARITY            # 0.382 - Flipping sign
COST_EXISTENCE = THRESHOLD_EXISTENCE          # 0.236 - Creating/destroying
COST_RIGHTEOUSNESS = THRESHOLD_RIGHTEOUSNESS  # 0.146 - Framing
COST_ORDER = THRESHOLD_ORDER                  # 0.090 - Ordering
COST_MOVEMENT = THRESHOLD_MOVEMENT            # 0.056 - Transforming

# Physical costs
COST_SPACE = THRESHOLD_SPACE                  # 0.034 - Adding dimension
COST_TIME = THRESHOLD_TIME                    # 0.021 - Temporal resolution
COST_MATTER = THRESHOLD_MATTER                # 0.013 - Creating node

# Coupling costs
COST_ALPHA = THRESHOLD_ALPHA                  # 0.0081 - Edge transfer
COST_BETA = THRESHOLD_BETA                    # 0.0050 - Path superposition
COST_GAMMA = THRESHOLD_GAMMA                  # 0.0031 - Arrangement counting

# All costs
MOTION_COSTS = {
    'heat': COST_HEAT,
    'polarity': COST_POLARITY,
    'existence': COST_EXISTENCE,
    'righteousness': COST_RIGHTEOUSNESS,
    'order': COST_ORDER,
    'movement': COST_MOVEMENT,
    'space': COST_SPACE,
    'time': COST_TIME,
    'matter': COST_MATTER,
    'alpha': COST_ALPHA,
    'beta': COST_BETA,
    'gamma': COST_GAMMA,
}

# Composite costs for common operations
COST_TRAVERSE = COST_MOVEMENT         # Moving along one axis
COST_CREATE_NODE = COST_EXISTENCE     # Creating a new node (δ localization)
COST_EVALUATE = COST_RIGHTEOUSNESS    # Evaluating frame (dim check)
COST_ACTION = COST_HEAT               # Output action to environment (Σ)
COST_TICK = COST_MOVEMENT             # Base cost per tick
COST_COLLAPSE = COST_BETA             # Wave function collapse (choice)

# ═══════════════════════════════════════════════════════════════════════════════
# TIME AS HEAT (t_K) - Time indexed by the primitive
# ═══════════════════════════════════════════════════════════════════════════════

# Time doesn't flow. HEAT flows, and we call that time.
# t_K = time measured in K units (how many heat quanta have flowed)
#
# One "tick" = one K-quantum redistributed through the manifold
# The clock doesn't tick time - it ticks heat
# Arrow of time = direction of heat flow (thermodynamic)

# ═══════════════════════════════════════════════════════════════════════════════
# HARDWARE CLOCK CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════════
#
# PBAI discovers its own hardware clock rate at boot and can adjust for
# API timing to other systems. This grounds the abstract t_K in physical time.
#
# The calibration measures: how many t_K ticks per wall-clock second?
# This varies by hardware but the HEAT accounting is what matters to cognition.
#

# Default calibration (can be overwritten at runtime)
HARDWARE_TICKS_PER_SECOND = None  # Set during calibration
CALIBRATION_SAMPLES = 10         # Number of samples to average

def calibrate_hardware_clock(tick_function, duration_seconds: float = 1.0) -> float:
    """
    Calibrate the hardware clock by measuring ticks over a duration.
    
    Args:
        tick_function: Function that performs one tick and returns t_K
        duration_seconds: How long to measure (default 1 second)
        
    Returns:
        Ticks per second (hardware clock rate)
    """
    import time as wall_time
    
    start_time = wall_time.perf_counter()
    start_tK = tick_function()
    
    # Run for specified duration
    while wall_time.perf_counter() - start_time < duration_seconds:
        tick_function()
    
    end_time = wall_time.perf_counter()
    end_tK = tick_function()
    
    elapsed_seconds = end_time - start_time
    elapsed_tK = end_tK - start_tK
    
    ticks_per_second = elapsed_tK / elapsed_seconds if elapsed_seconds > 0 else 0
    
    global HARDWARE_TICKS_PER_SECOND
    HARDWARE_TICKS_PER_SECOND = ticks_per_second
    
    return ticks_per_second

def tK_to_seconds(t_K: int) -> float:
    """Convert t_K (heat time) to wall-clock seconds."""
    if HARDWARE_TICKS_PER_SECOND is None or HARDWARE_TICKS_PER_SECOND == 0:
        return t_K * TICK_INTERVAL_BASE  # Use base interval as fallback
    return t_K / HARDWARE_TICKS_PER_SECOND

def seconds_to_tK(seconds: float) -> int:
    """Convert wall-clock seconds to t_K (heat time)."""
    if HARDWARE_TICKS_PER_SECOND is None or HARDWARE_TICKS_PER_SECOND == 0:
        return int(seconds / TICK_INTERVAL_BASE)
    return int(seconds * HARDWARE_TICKS_PER_SECOND)

# ═══════════════════════════════════════════════════════════════════════════════
# TICK CONFIGURATION - Autonomous loop timing  
# ═══════════════════════════════════════════════════════════════════════════════

# Base tick interval in seconds (maps real time to t_K)
TICK_INTERVAL_BASE = 1.0  # 1 second base tick

# Tick rate scales with system heat (hotter = faster thinking)
TICK_INTERVAL_MIN = 0.1   # Fastest: 10 ticks/second when very hot
TICK_INTERVAL_MAX = 10.0  # Slowest: 1 tick/10 seconds when cold

# Heat thresholds for tick rate scaling
TICK_HEAT_HOT = K * 10    # Above this = fast ticking
TICK_HEAT_COLD = K * 0.5  # Below this = slow ticking

# Save interval (don't wear out SSD)
SAVE_INTERVAL_TICKS = 100  # Save every 100 ticks
SAVE_INTERVAL_SECONDS = 300  # Or every 5 minutes, whichever comes first

# Minimum heat for psychology nodes (below this = dormant/exhausted)
PSYCHOLOGY_MIN_HEAT = COST_MOVEMENT  # Must have at least one motion's worth

# ═══════════════════════════════════════════════════════════════════════════════
# THE 12 MOVEMENT DIRECTIONS (6 Self × 2 frames)
# ═══════════════════════════════════════════════════════════════════════════════

# Self directions (egocentric frame - for navigation)
DIRECTIONS_SELF = {
    'up':      ( 0,  0,  1),   # +Z self
    'down':    ( 0,  0, -1),   # -Z self
    'left':    (-1,  0,  0),   # -X self
    'right':   ( 1,  0,  0),   # +X self
    'forward': ( 0,  1,  0),   # +Y self
    'reverse': ( 0, -1,  0),   # -Y self
}

# Universal directions (world frame - for locating righteous frames)
DIRECTIONS_UNIVERSAL = {
    'N':     ( 0,  1,  0),   # North: +Y world
    'S':     ( 0, -1,  0),   # South: -Y world
    'E':     ( 1,  0,  0),   # East:  +X world
    'W':     (-1,  0,  0),   # West:  -X world
    'above': ( 0,  0,  1),   # Above: +Z world
    'below': ( 0,  0, -1),   # Below: -Z world
}

# All 12 directions combined
DIRECTIONS = {**DIRECTIONS_SELF, **DIRECTIONS_UNIVERSAL}

# Aliases for backward compatibility
DIRECTIONS_RELATIVE = DIRECTIONS_SELF
DIRECTIONS_ABSOLUTE = DIRECTIONS_UNIVERSAL

# Legacy aliases
DIRECTIONS_LEGACY = {
    'n': ( 0,  1,  0),
    's': ( 0, -1,  0),
    'e': ( 1,  0,  0),
    'w': (-1,  0,  0),
    'u': ( 0,  0,  1),
    'd': ( 0,  0, -1),
}

# Direction opposites (all 12 + legacy)
OPPOSITES = {
    # Self frame
    'up': 'down', 'down': 'up',
    'left': 'right', 'right': 'left',
    'forward': 'reverse', 'reverse': 'forward',
    # Universal frame
    'N': 'S', 'S': 'N',
    'E': 'W', 'W': 'E',
    'above': 'below', 'below': 'above',
    # Legacy
    'n': 's', 's': 'n',
    'e': 'w', 'w': 'e',
    'u': 'd', 'd': 'u',
}

# Self directions available (down is blocked - self is there)
SELF_DIRECTIONS_SELF = ['up', 'left', 'right', 'forward', 'reverse']
SELF_DIRECTIONS_UNIVERSAL = ['N', 'S', 'E', 'W', 'above']  # below is blocked

# All directions for traversal
ALL_DIRECTIONS_SELF = ['up', 'down', 'left', 'right', 'forward', 'reverse']
ALL_DIRECTIONS_UNIVERSAL = ['N', 'S', 'E', 'W', 'above', 'below']
ALL_DIRECTIONS = ALL_DIRECTIONS_SELF + ALL_DIRECTIONS_UNIVERSAL

# Legacy (for backward compatibility)
SELF_DIRECTIONS = ['n', 's', 'e', 'w', 'u']  # Legacy: d blocked

# ═══════════════════════════════════════════════════════════════════════════════
# EXISTENCE STATES
# ═══════════════════════════════════════════════════════════════════════════════

EXISTENCE_POTENTIAL = "potential"  # Awaiting environment confirmation
EXISTENCE_ACTUAL = "actual"        # Connected to Julia spine, conscious
EXISTENCE_DORMANT = "dormant"      # Disconnected, unconscious
EXISTENCE_ARCHIVED = "archived"    # Cold storage, historical

# ═══════════════════════════════════════════════════════════════════════════════
# ENTROPY WEIGHTS (trigonometric at golden angle 1/φ radians)
# ═══════════════════════════════════════════════════════════════════════════════

# Golden angle in radians (connects to motion thresholds)
_GOLDEN_ANGLE = 1.0 / PHI  # ≈ 0.618 radians ≈ 35.4°

ENTROPY_MAGNITUDE_WEIGHT = math.sin(_GOLDEN_ANGLE)  # ≈ 0.579 (amplitude)
ENTROPY_VARIANCE_WEIGHT = math.cos(_GOLDEN_ANGLE)   # ≈ 0.815 (spread)
ENTROPY_DISORDER_WEIGHT = math.tan(_GOLDEN_ANGLE)   # ≈ 0.710 (phase)

# ═══════════════════════════════════════════════════════════════════════════════
# PSYCHOLOGY: TRIGONOMETRIC POSITIONS (Abstract Space)
# ═══════════════════════════════════════════════════════════════════════════════

TRIG_IDENTITY = (ENTROPY_MAGNITUDE_WEIGHT, 0.0, 0.0)
TRIG_EGO = (0.0, ENTROPY_DISORDER_WEIGHT, 0.0)
TRIG_CONSCIENCE = (0.0, 0.0, ENTROPY_VARIANCE_WEIGHT)

def trig_position_to_string(amplitude: float, phase: float, spread: float) -> str:
    """Encode trig coordinates as a position string for abstract space."""
    return f"@{amplitude:.6f},{phase:.6f},{spread:.6f}"

def string_to_trig_position(pos: str) -> tuple:
    """Decode trig position string to (amplitude, phase, spread) tuple."""
    if not pos.startswith("@"):
        raise ValueError(f"Not a trig position: {pos}")
    parts = pos[1:].split(",")
    return (float(parts[0]), float(parts[1]), float(parts[2]))

def is_trig_position(pos: str) -> bool:
    """Check if position string represents abstract/trig space."""
    return pos.startswith("@")

def is_cubic_position(pos: str) -> bool:
    """Check if position string represents physical/cubic space."""
    return not pos.startswith("@")

# ═══════════════════════════════════════════════════════════════════════════════
# PSYCHOLOGY: FREUDIAN HEAT DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════════

FREUD_IDENTITY_RATIO = 0.70   # Id - the reservoir
FREUD_CONSCIENCE_RATIO = 0.20 # Superego - the judge
FREUD_EGO_RATIO = 0.10        # Ego - the conscious interface

assert abs((FREUD_IDENTITY_RATIO + FREUD_CONSCIENCE_RATIO + FREUD_EGO_RATIO) - 1.0) < 1e-10

# ═══════════════════════════════════════════════════════════════════════════════
# CAPABILITY LEVELS
# ═══════════════════════════════════════════════════════════════════════════════

CAPABILITY_RIGHTEOUS = "righteous"
CAPABILITY_ORDERED = "ordered"
CAPABILITY_MOVABLE = "movable"
CAPABILITY_GRAPHIC = "graphic"
CAPABILITY_PROPER = "ordered"  # Alias

# ═══════════════════════════════════════════════════════════════════════════════
# MOTION THRESHOLD FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_threshold(motion_function: str) -> float:
    """Get the threshold for a specific motion function."""
    return MOTION_THRESHOLDS.get(motion_function, THRESHOLD_HEAT)

def get_threshold_for_direction(direction: str) -> float:
    """Get the motion threshold for a spatial direction."""
    motion = DIRECTION_TO_MOTION.get(direction, 'heat')
    return MOTION_THRESHOLDS.get(motion, THRESHOLD_HEAT)

def get_cost(motion_function: str) -> float:
    """Get the cost for performing a motion function operation."""
    return MOTION_COSTS.get(motion_function, COST_MOVEMENT)

def get_cost_for_direction(direction: str) -> float:
    """Get the cost for traversing in a direction."""
    motion = DIRECTION_TO_MOTION.get(direction, 'movement')
    return MOTION_COSTS.get(motion, COST_MOVEMENT)

def exceeds_threshold(delta: float, motion_function: str) -> bool:
    """Check if a change exceeds the threshold for a motion function."""
    threshold = get_threshold(motion_function)
    return abs(delta) >= threshold

def exceeds_any_threshold(delta: float) -> bool:
    """Check if a change exceeds the finest (smallest) threshold."""
    return abs(delta) >= THRESHOLD_MOVEMENT

def exceeds_all_thresholds(delta: float) -> bool:
    """Check if a change exceeds the coarsest (largest) threshold."""
    return abs(delta) >= THRESHOLD_HEAT

def quantize_to_threshold(value: float, motion_function: str) -> float:
    """Quantize a value to the nearest multiple of the motion threshold."""
    threshold = get_threshold(motion_function)
    if threshold == 0:
        return value
    return round(value / threshold) * threshold

def heat_required(cardinality: int) -> float:
    """Heat needed to reach concept of given cardinality. Scales as φ^(N-1)."""
    return K * (PHI ** (cardinality - 1))

# ═══════════════════════════════════════════════════════════════════════════════
# DIRECTION UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def get_opposite(direction: str) -> str:
    """Get the opposite direction."""
    return OPPOSITES.get(direction)

def direction_to_vector(direction: str) -> tuple:
    """Get the (x, y, z) vector for a direction."""
    return DIRECTIONS.get(direction, (0, 0, 0))

def get_motion_for_direction(direction: str) -> str:
    """Get the motion function associated with a direction."""
    return DIRECTION_TO_MOTION.get(direction, 'heat')

def get_direction_for_motion(motion: str) -> str:
    """Get the direction associated with a motion function."""
    return MOTION_TO_DIRECTION.get(motion, 'n')

# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def validate_motion_unit():
    """
    Verify the motion unit mathematics.
    
    FUNDAMENTAL IDENTITIES:
    - K × φ² = 4 (exact - the core identity)
    - K = Σ(1/φⁿ) for n=1→6 (definition via sum)
    - 1/φ³ < 1/4 (existence threshold below Julia spine boundary)
    - 45/44 = emergence threshold
    - 44/45 = maximum entropic probability
    
    PLANCK DERIVATIONS:
    - Temperature: K × 12³⁰ / φ² × 45/44 × 1 ≈ 1.417 × 10³² K
    - Length: K × 12⁻³² / φ² × 45/44 × 4φ/7 ≈ 1.614 × 10⁻³⁵ m
    - Time: K × 12⁻⁴⁰ / φ² × 45/44 × 4/3 ≈ 5.414 × 10⁻⁴⁴ s
    - Mass: K × 12⁻⁷ / φ² × 45/44 × 13/10 ≈ 2.165 × 10⁻⁸ kg
    """
    # THE FUNDAMENTAL IDENTITY: K × φ² = 4
    assert abs(K * PHI**2 - 4) < 1e-10, f"K × φ² must equal 4 (got {K * PHI**2})"
    
    # JULIA TOPOLOGY: Existence threshold < spine boundary
    assert THRESHOLD_EXISTENCE < JULIA_SPINE_THRESHOLD, \
        f"Existence threshold {THRESHOLD_EXISTENCE} must be < Julia spine {JULIA_SPINE_THRESHOLD}"
    
    # PHI IDENTITIES
    assert abs(INV_PHI - (PHI - 1)) < 1e-10, "1/φ should equal φ - 1"
    assert abs(PHI**2 - (PHI + 1)) < 1e-10, "φ² should equal φ + 1"
    
    # EMERGENCE THRESHOLD: 45/44
    assert abs(EMERGENCE_THRESHOLD - (45/44)) < 1e-10, "Emergence threshold should be 45/44"
    assert abs(MAX_ENTROPIC_PROBABILITY - (44/45)) < 1e-10, "Max entropy should be 44/45"
    
    # STRUCTURAL
    assert len(CORE_THRESHOLDS) == SIX, f"Should have {SIX} core motion functions"
    
    # All 12 thresholds
    all_thresholds = [
        THRESHOLD_HEAT, THRESHOLD_POLARITY, THRESHOLD_EXISTENCE,
        THRESHOLD_RIGHTEOUSNESS, THRESHOLD_ORDER, THRESHOLD_MOVEMENT,
        THRESHOLD_SPACE, THRESHOLD_TIME, THRESHOLD_MATTER,
        THRESHOLD_ALPHA, THRESHOLD_BETA, THRESHOLD_GAMMA
    ]
    assert len(all_thresholds) == MOVEMENT_CONSTANT, f"Should have {MOVEMENT_CONSTANT} total thresholds"
    
    # Thresholds descend
    for i in range(len(all_thresholds) - 1):
        assert all_thresholds[i] > all_thresholds[i+1], "Thresholds should descend"
    
    # K is sum of core thresholds
    core_sum = sum(CORE_THRESHOLDS.values())
    assert abs(K - core_sum) < 1e-10, "K should be sum of core thresholds"
    
    # K_TOTAL + residue = φ
    assert abs((K_TOTAL + K_RESIDUE) - PHI) < 1e-10, "K_TOTAL + residue should equal φ"
    
    # -1/12 is correct
    assert abs(NEGATIVE_ONE_TWELFTH - (-1/12)) < 1e-10, "Entropic bound should be -1/12"
    
    # PLANCK DERIVATIONS (check reasonable ranges)
    assert 1e31 < PLANCK_TEMPERATURE < 1e33, "Planck temperature out of range"
    assert 1e-36 < PLANCK_LENGTH < 1e-34, "Planck length out of range"
    assert 1e-45 < PLANCK_TIME < 1e-43, "Planck time out of range"
    assert 1e-9 < PLANCK_MASS < 1e-7, "Planck mass out of range"
    
    return True


# Run validation on module load (will raise if math is wrong)
validate_motion_unit()

# ═══════════════════════════════════════════════════════════════════════════════
# MODULE LOAD MESSAGE
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("PBAI NODE CONSTANTS v2 (Planck-Grounded)")
print("=" * 60)
print(f"K = {K:.10f} Kelvin (Thermal Quantum)")
print(f"K × φ² = {K * PHI**2} (exact = 4)")
print(f"45/44 = {EMERGENCE_THRESHOLD:.10f} (Emergence Threshold)")
print(f"44/45 = {MAX_ENTROPIC_PROBABILITY:.10f} (Max Entropy)")
print("-" * 60)
print(f"T_Planck = {PLANCK_TEMPERATURE:.6e} K")
print(f"L_Planck = {PLANCK_LENGTH:.6e} m")
print(f"t_Planck = {PLANCK_TIME:.6e} s")
print(f"M_Planck = {PLANCK_MASS:.6e} kg")
print(f"c        = {SPEED_OF_LIGHT:.6e} m/s")
print(f"G        = {GRAVITATIONAL_CONSTANT:.6e} m³/kg·s²")
print("=" * 60)
