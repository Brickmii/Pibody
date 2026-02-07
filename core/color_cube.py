"""
PBAI Color Cube — Base Righteous Frame + Base Ordered Frame

The Color Cube is the absolute reference frame of the entire system.
It does NOT hold nodes. It holds the STANDARD.

    "Is this node righteous?" → How does it align with the Color Cube?
    "Is this ordered?"        → Does its sequence respect Robinson constraints?

Axis Definitions (locked, consistent everywhere):
    X axis (E/W): Yellow(+1) ↔ Blue(-1)
    Y axis (N/S): Green(+1)  ↔ Red(-1)
    Z axis (U/D): Future(+τ) ↔ Past(-τ)

Heat is NOT an axis. Heat = √(x² + y²) — magnitude from chromatic position.
Side view of any axis = wave function.

Four Quadrants (base righteous frame):
    Q1 (+X,+Y): Yellow+Green  (NE)
    Q2 (-X,+Y): Blue+Green    (NW)
    Q3 (-X,-Y): Blue+Red      (SW)
    Q4 (+X,-Y): Yellow+Red    (SE)

Robinson Constraints operate on ALL THREE axes simultaneously.
Every position (x, y, τ) has ordering in all three directions.
"""

import math
from dataclasses import dataclass, field
from typing import Tuple, Optional

from .node_constants import (
    PHI, INV_PHI, K,
    ROBINSON_IDENTITY, ROBINSON_SUCCESSOR, ROBINSON_ADDITION, ROBINSON_MULTIPLICATION,
    CUBE_AXES, CUBE_QUADRANTS, CARDINAL_TO_AXIS,
    CUBE_AXIS_X, CUBE_AXIS_Y, CUBE_AXIS_Z,
    CUBE_POLE_POSITIVE_X, CUBE_POLE_NEGATIVE_X,
    CUBE_POLE_POSITIVE_Y, CUBE_POLE_NEGATIVE_Y,
    CUBE_POLE_POSITIVE_Z, CUBE_POLE_NEGATIVE_Z,
)


# ═══════════════════════════════════════════════════════════════════════════════
# CUBE POSITION — A point in the Color Cube reference frame
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CubePosition:
    """
    A position projected onto the Color Cube axes.

    x: Blue(-1) ↔ Yellow(+1)   — chromatic axis E/W
    y: Red(-1)  ↔ Green(+1)    — chromatic axis N/S
    tau: Past(-) ↔ Future(+)   — time axis U/D

    Coordinates are normalized to [-1, +1] on the chromatic axes.
    tau is unbounded (time flows).
    """
    x: float = 0.0    # Blue/Yellow balance
    y: float = 0.0    # Red/Green balance
    tau: float = 0.0   # Time position

    @property
    def heat(self) -> float:
        """Heat magnitude — derived from chromatic position, NOT an axis."""
        return math.sqrt(self.x ** 2 + self.y ** 2)

    @property
    def quadrant(self) -> str:
        """Which of the 4 righteous quadrants this position falls in."""
        if self.x >= 0 and self.y >= 0:
            return "Q1"  # Yellow+Green (NE)
        elif self.x < 0 and self.y >= 0:
            return "Q2"  # Blue+Green (NW)
        elif self.x < 0 and self.y < 0:
            return "Q3"  # Blue+Red (SW)
        else:
            return "Q4"  # Yellow+Red (SE)

    @property
    def polarity_x(self) -> int:
        """Polarity on the X axis: +1 (yellow) or -1 (blue)."""
        return +1 if self.x >= 0 else -1

    @property
    def polarity_y(self) -> int:
        """Polarity on the Y axis: +1 (green) or -1 (red)."""
        return +1 if self.y >= 0 else -1

    @property
    def polarity_z(self) -> int:
        """Polarity on the Z axis: +1 (future) or -1 (past)."""
        return +1 if self.tau >= 0 else -1

    @property
    def color_name(self) -> str:
        """Dominant color description based on quadrant and magnitude."""
        q = self.quadrant
        colors = CUBE_QUADRANTS[q]['colors']
        return f"{colors[0]}+{colors[1]}"

    def as_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.tau)

    def distance_to(self, other: 'CubePosition') -> float:
        """Euclidean distance in cube space."""
        return math.sqrt(
            (self.x - other.x) ** 2 +
            (self.y - other.y) ** 2 +
            (self.tau - other.tau) ** 2
        )


# ═══════════════════════════════════════════════════════════════════════════════
# ROBINSON OPERATIONS — All three axes simultaneously
# ═══════════════════════════════════════════════════════════════════════════════

# Robinson arithmetic operates on ALL THREE axes.
# Every position (x, y, τ) has ordering in all three directions.

ROBINSON_OPERATIONS = {
    'identity':       {'factor': ROBINSON_IDENTITY,       'question': "What is this?",    'function': "Pole itself"},
    'successor':      {'factor': ROBINSON_SUCCESSOR,      'question': "What's next?",     'function': "Step toward opponent"},
    'addition':       {'factor': ROBINSON_ADDITION,       'question': "What combines?",   'function': "Compose positions"},
    'multiplication': {'factor': ROBINSON_MULTIPLICATION, 'question': "How much?",        'function': "Scale along axis"},
}


def robinson_identity(value: float) -> float:
    """R=1: What is this? The pole itself."""
    return value * ROBINSON_IDENTITY


def robinson_successor(value: float) -> float:
    """R=4phi/7: What's next? Step toward the opponent pole."""
    return value * ROBINSON_SUCCESSOR


def robinson_addition(a: float, b: float) -> float:
    """R=4/3: What combines? Compose two positions."""
    return (a + b) * ROBINSON_ADDITION


def robinson_multiplication(value: float, scale: float) -> float:
    """R=13/10: How much? Scale along axis."""
    return value * scale * ROBINSON_MULTIPLICATION


def apply_robinson(value: float, operation: str, operand: float = 0.0) -> float:
    """
    Apply a Robinson constraint operation to a value.

    Args:
        value: The position value on any axis
        operation: 'identity', 'successor', 'addition', 'multiplication'
        operand: Second operand for addition/multiplication
    """
    if operation == 'identity':
        return robinson_identity(value)
    elif operation == 'successor':
        return robinson_successor(value)
    elif operation == 'addition':
        return robinson_addition(value, operand)
    elif operation == 'multiplication':
        return robinson_multiplication(value, operand)
    else:
        raise ValueError(f"Unknown Robinson operation: {operation}")


# ═══════════════════════════════════════════════════════════════════════════════
# RIGHTEOUSNESS — Alignment with the Color Cube
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_righteousness(position: CubePosition, reference: CubePosition = None) -> float:
    """
    Evaluate how righteous a position is relative to the cube frame.

    Righteousness R measures angular deviation from the reference frame.
    R → 0 means perfectly aligned (righteous).
    R > 0 means deviation exists.

    If no reference is given, evaluates against the cube origin (0,0,0).

    Args:
        position: The position to evaluate
        reference: Reference position (default: cube center)

    Returns:
        R value (0 = perfectly righteous, higher = more deviation)
    """
    if reference is None:
        reference = CubePosition(0.0, 0.0, 0.0)

    # Angular deviation on chromatic plane
    dx = position.x - reference.x
    dy = position.y - reference.y
    dtau = position.tau - reference.tau

    # R is the magnitude of deviation
    return math.sqrt(dx ** 2 + dy ** 2 + dtau ** 2)


def get_quadrant_alignment(position: CubePosition, target_quadrant: str) -> float:
    """
    How well does a position align with a target quadrant?

    Returns a value in [0, 1] where 1 = perfectly aligned.
    """
    if target_quadrant not in CUBE_QUADRANTS:
        return 0.0

    q = CUBE_QUADRANTS[target_quadrant]
    target_x = q['x']
    target_y = q['y']

    # Dot product of signs (normalized)
    sign_x = 1 if position.x >= 0 else -1
    sign_y = 1 if position.y >= 0 else -1

    alignment = (sign_x * target_x + sign_y * target_y) / 2.0
    # Map from [-1, 1] to [0, 1]
    return (alignment + 1.0) / 2.0


# ═══════════════════════════════════════════════════════════════════════════════
# HEAT — Derived from motion, not an axis
# ═══════════════════════════════════════════════════════════════════════════════

def heat_from_position(position: CubePosition) -> float:
    """
    Heat magnitude derived from chromatic position.

    Heat = √(x² + y²)
    - Center (0,0) = no heat, achromatic, neutral
    - Edge = single-axis heat (max 1.0)
    - Corner = maximum heat (√2 ≈ 1.414)
    """
    return position.heat


def heat_zone(heat_magnitude: float) -> str:
    """
    Classify heat magnitude into zones.

    Returns: 'achromatic', 'warm', 'hot', 'maximum'
    """
    if heat_magnitude < INV_PHI ** 3:    # < ~0.236 (existence threshold)
        return 'achromatic'
    elif heat_magnitude < INV_PHI:        # < ~0.618 (heat threshold)
        return 'warm'
    elif heat_magnitude < 1.0:
        return 'hot'
    else:
        return 'maximum'


# ═══════════════════════════════════════════════════════════════════════════════
# WAVE FUNCTION — Side view of any axis
# ═══════════════════════════════════════════════════════════════════════════════

def wave_amplitude(position: CubePosition, axis: str = 'x') -> float:
    """
    View position as wave amplitude on a given axis over tau.

    Side view of any axis = wave function.
    Amplitude on the chromatic axis varying over τ.

    Args:
        position: The cube position
        axis: 'x' (Blue/Yellow) or 'y' (Red/Green)

    Returns:
        Amplitude value on the specified axis
    """
    if axis == 'x':
        return position.x
    elif axis == 'y':
        return position.y
    else:
        raise ValueError(f"Wave amplitude axis must be 'x' or 'y', got: {axis}")


# ═══════════════════════════════════════════════════════════════════════════════
# CUBE UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def clamp_chromatic(value: float) -> float:
    """Clamp a value to the chromatic range [-1, +1]."""
    return max(-1.0, min(1.0, value))


def normalize_to_cube(x: float, y: float, tau: float = 0.0) -> CubePosition:
    """
    Create a CubePosition with chromatic axes clamped to [-1, +1].
    tau is unbounded.
    """
    return CubePosition(
        x=clamp_chromatic(x),
        y=clamp_chromatic(y),
        tau=tau
    )


def cube_origin() -> CubePosition:
    """The center of the Color Cube — achromatic, zero heat, present moment."""
    return CubePosition(0.0, 0.0, 0.0)


def cube_pole(direction: str) -> CubePosition:
    """
    Get the CubePosition for a pure pole.

    Args:
        direction: 'E', 'W', 'N', 'S', 'U', 'D' or color name
    """
    poles = {
        # Cardinal directions
        'E': CubePosition(+1.0, 0.0, 0.0),   # Yellow
        'W': CubePosition(-1.0, 0.0, 0.0),    # Blue
        'N': CubePosition(0.0, +1.0, 0.0),    # Green
        'S': CubePosition(0.0, -1.0, 0.0),    # Red
        'U': CubePosition(0.0, 0.0, +1.0),    # Future
        'D': CubePosition(0.0, 0.0, -1.0),    # Past
        # Color names
        'yellow': CubePosition(+1.0, 0.0, 0.0),
        'blue':   CubePosition(-1.0, 0.0, 0.0),
        'green':  CubePosition(0.0, +1.0, 0.0),
        'red':    CubePosition(0.0, -1.0, 0.0),
        'future': CubePosition(0.0, 0.0, +1.0),
        'past':   CubePosition(0.0, 0.0, -1.0),
    }
    if direction not in poles:
        raise ValueError(f"Unknown pole direction: {direction}")
    return poles[direction]


def quadrant_center(quadrant: str) -> CubePosition:
    """
    Get the center position of a quadrant (at tau=0).

    Args:
        quadrant: 'Q1', 'Q2', 'Q3', or 'Q4'
    """
    if quadrant not in CUBE_QUADRANTS:
        raise ValueError(f"Unknown quadrant: {quadrant}")
    q = CUBE_QUADRANTS[quadrant]
    # Quadrant centers at (±0.5, ±0.5) — midpoint of the quadrant
    return CubePosition(
        x=q['x'] * 0.5,
        y=q['y'] * 0.5,
        tau=0.0
    )


def opponent_color(color: str) -> str:
    """Get the opponent color on the same axis."""
    opponents = {
        'yellow': 'blue', 'blue': 'yellow',
        'green': 'red', 'red': 'green',
        'future': 'past', 'past': 'future',
    }
    if color not in opponents:
        raise ValueError(f"Unknown color: {color}")
    return opponents[color]


def color_to_axis(color: str) -> Tuple[str, int]:
    """
    Map a color name to its axis and polarity.

    Returns: (axis_name, polarity) e.g., ('x', +1) for yellow
    """
    mapping = {
        'yellow': ('x', +1), 'blue': ('x', -1),
        'green': ('y', +1), 'red': ('y', -1),
        'future': ('z', +1), 'past': ('z', -1),
    }
    if color not in mapping:
        raise ValueError(f"Unknown color: {color}")
    return mapping[color]