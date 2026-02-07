"""
PBAI Hypersphere — Node Topology

The coordinate space where nodes actually live.
Embedded within the Color Cube.

Why Hypersphere:
    - Nodes distribute on SURFACE, not volume
    - Scaling: n² (surface area) vs n³ (volume)
    - Relationships via angular distance, not grid adjacency
    - Traversal paths wrap naturally (no edge dead-ends)
    - Every node equidistant from center (Self)

Node Position:
    Every node has angular position (theta, phi) on the hypersphere surface.
    That position projects onto the Color Cube's axes:

        Node at angular position (θ, φ) on sphere
          → projects to (x, y, τ) in cube space
          → x gives Blue/Yellow balance
          → y gives Red/Green balance
          → τ gives time position
          → √(x² + y²) gives heat magnitude

Self sits at the CENTER of the hypersphere (the origin).
Not on the surface — at the center. All nodes are equidistant from Self.
Self is the bridge: environment pushes perception IN, decisions push action OUT.
"""

import math
from dataclasses import dataclass, field
from typing import Tuple, List, Optional

from .node_constants import PHI, INV_PHI, K
from .color_cube import CubePosition


# ═══════════════════════════════════════════════════════════════════════════════
# SPHERE POSITION — Angular coordinates on the hypersphere surface
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SpherePosition:
    """
    Angular position on the hypersphere surface.

    theta: polar angle [0, π]     — from +Z (future) pole down to -Z (past)
    phi:   azimuthal angle [0, 2π) — rotation in the X-Y (chromatic) plane

    Convention:
        theta=0       → north pole (+Z, future)
        theta=π       → south pole (-Z, past)
        theta=π/2     → equator (present moment, τ=0)
        phi=0         → +X direction (yellow/east)
        phi=π/2       → +Y direction (green/north)
        phi=π         → -X direction (blue/west)
        phi=3π/2      → -Y direction (red/south)

    radius: distance from center. For surface nodes this is 1.0.
            Self is at radius=0 (the center).
    """
    theta: float = math.pi / 2   # Default: equator (present moment)
    phi: float = 0.0              # Default: +X (yellow/east)
    radius: float = 1.0           # Default: on surface (1.0), Self at 0.0

    def __post_init__(self):
        # Normalize angles
        self.theta = max(0.0, min(math.pi, self.theta))
        self.phi = self.phi % (2 * math.pi)

    def to_cube(self) -> CubePosition:
        """
        Project this sphere position onto the Color Cube.

        Standard spherical → Cartesian:
            x = r * sin(θ) * cos(φ)    → Blue/Yellow balance
            y = r * sin(θ) * sin(φ)    → Red/Green balance
            τ = r * cos(θ)              → Time position
        """
        r = self.radius
        sin_theta = math.sin(self.theta)
        return CubePosition(
            x=r * sin_theta * math.cos(self.phi),
            y=r * sin_theta * math.sin(self.phi),
            tau=r * math.cos(self.theta)
        )

    @staticmethod
    def from_cube(pos: CubePosition) -> 'SpherePosition':
        """
        Convert a cube position back to sphere coordinates.

        Inverse of to_cube().
        """
        r = math.sqrt(pos.x ** 2 + pos.y ** 2 + pos.tau ** 2)
        if r < 1e-12:
            # At center (Self position)
            return SpherePosition(theta=math.pi / 2, phi=0.0, radius=0.0)

        theta = math.acos(max(-1.0, min(1.0, pos.tau / r)))
        phi = math.atan2(pos.y, pos.x)
        if phi < 0:
            phi += 2 * math.pi

        return SpherePosition(theta=theta, phi=phi, radius=r)

    def as_tuple(self) -> Tuple[float, float, float]:
        return (self.theta, self.phi, self.radius)


# ═══════════════════════════════════════════════════════════════════════════════
# ANGULAR DISTANCE — Relationship strength between nodes
# ═══════════════════════════════════════════════════════════════════════════════

def angular_distance(a: SpherePosition, b: SpherePosition) -> float:
    """
    Angular distance between two positions on the hypersphere.

    Uses the haversine formula for great-circle distance.
    Returns angle in radians [0, π].

    Angular distance IS relationship strength:
        0     = perfect alignment (same position)
        π     = maximum opposition (antipodal)
        π/2   = orthogonal (independent)
    """
    # Great-circle distance via dot product of unit vectors
    ax = math.sin(a.theta) * math.cos(a.phi)
    ay = math.sin(a.theta) * math.sin(a.phi)
    az = math.cos(a.theta)

    bx = math.sin(b.theta) * math.cos(b.phi)
    by = math.sin(b.theta) * math.sin(b.phi)
    bz = math.cos(b.theta)

    dot = ax * bx + ay * by + az * bz
    # Clamp for numerical safety
    dot = max(-1.0, min(1.0, dot))
    return math.acos(dot)


def relationship_strength(a: SpherePosition, b: SpherePosition) -> float:
    """
    Relationship strength from angular distance.

    Returns value in [0, 1]:
        1.0 = same position (angular distance = 0)
        0.0 = antipodal (angular distance = π)
    """
    angle = angular_distance(a, b)
    return 1.0 - (angle / math.pi)


def are_aligned(a: SpherePosition, b: SpherePosition,
                threshold: float = None) -> bool:
    """
    Check if two positions are aligned within a threshold.

    Default threshold: 1/φ⁴ (Righteousness threshold) in radians.
    """
    if threshold is None:
        threshold = INV_PHI ** 4  # ~0.146 radians (~8.4 degrees)
    return angular_distance(a, b) < threshold


def are_opposed(a: SpherePosition, b: SpherePosition,
                threshold: float = None) -> bool:
    """
    Check if two positions are near-antipodal (opponent positions).
    """
    if threshold is None:
        threshold = INV_PHI ** 4
    return angular_distance(a, b) > (math.pi - threshold)


# ═══════════════════════════════════════════════════════════════════════════════
# SURFACE DISTRIBUTION — n² scaling
# ═══════════════════════════════════════════════════════════════════════════════

def surface_area(radius: float = 1.0) -> float:
    """Surface area of the hypersphere: 4πr²."""
    return 4 * math.pi * radius ** 2


def max_nodes_at_resolution(resolution: float, radius: float = 1.0) -> int:
    """
    Maximum number of nodes that can fit on the surface at a given angular resolution.

    Each node occupies a solid angle of ~resolution² steradians.
    Total solid angle = 4π steradians.
    Max nodes ≈ 4π / resolution²

    Args:
        resolution: Minimum angular separation in radians
        radius: Sphere radius (default 1.0)

    Returns:
        Maximum number of well-separated nodes
    """
    if resolution <= 0:
        return 0
    solid_angle_per_node = resolution ** 2
    total_solid_angle = 4 * math.pi
    return int(total_solid_angle / solid_angle_per_node)


def n_squared_capacity(n: int) -> float:
    """
    Demonstrate n² scaling: surface nodes scale as n², not n³.

    For n nodes, the average angular separation is ~√(4π/n).
    Doubling n → separation shrinks by √2 (area scales quadratically).
    """
    if n <= 0:
        return 0.0
    return math.sqrt(4 * math.pi / n)


# ═══════════════════════════════════════════════════════════════════════════════
# NODE PLACEMENT — Finding positions on the sphere
# ═══════════════════════════════════════════════════════════════════════════════

def place_node_near(
    target: SpherePosition,
    existing: List[SpherePosition],
    min_separation: float = None
) -> SpherePosition:
    """
    Place a new node near a target position, respecting minimum separation
    from existing nodes.

    Uses a golden-angle spiral offset to find non-colliding positions.

    Args:
        target: Desired position
        existing: List of existing node positions
        min_separation: Minimum angular distance from any existing node
                       (default: 1/φ⁶ — movement threshold)

    Returns:
        A valid SpherePosition near the target
    """
    if min_separation is None:
        min_separation = INV_PHI ** 6  # ~0.056 radians (~3.2 degrees)

    # Check if target itself is valid
    if _is_valid_placement(target, existing, min_separation):
        return target

    # Spiral outward from target using golden angle increments
    golden_angle = 2 * math.pi / (PHI ** 2)  # ~2.4 radians

    for i in range(1, 100):
        # Offset in both theta and phi using golden spiral
        offset_magnitude = min_separation * math.sqrt(i)
        offset_phi = golden_angle * i

        new_theta = target.theta + offset_magnitude * math.cos(offset_phi)
        new_phi = target.phi + offset_magnitude * math.sin(offset_phi)

        candidate = SpherePosition(theta=new_theta, phi=new_phi, radius=target.radius)

        if _is_valid_placement(candidate, existing, min_separation):
            return candidate

    # Fallback: return target anyway (rare — sphere is very full)
    return target


def _is_valid_placement(
    candidate: SpherePosition,
    existing: List[SpherePosition],
    min_separation: float
) -> bool:
    """Check if a candidate position is far enough from all existing nodes."""
    for pos in existing:
        if angular_distance(candidate, pos) < min_separation:
            return False
    return True


def place_evenly(n: int, radius: float = 1.0) -> List[SpherePosition]:
    """
    Distribute n nodes approximately evenly on the sphere surface.

    Uses the Fibonacci sphere algorithm (golden angle spiral)
    for near-uniform distribution.

    Args:
        n: Number of nodes to place
        radius: Sphere radius

    Returns:
        List of n SpherePositions
    """
    if n <= 0:
        return []
    if n == 1:
        return [SpherePosition(theta=math.pi / 2, phi=0.0, radius=radius)]

    positions = []
    golden_angle = math.pi * (3 - math.sqrt(5))  # ~2.4 radians

    for i in range(n):
        # Uniform distribution along Z axis
        z = 1 - (2 * i + 1) / n  # Range: ~1 to ~-1
        theta = math.acos(max(-1.0, min(1.0, z)))
        phi = (golden_angle * i) % (2 * math.pi)
        positions.append(SpherePosition(theta=theta, phi=phi, radius=radius))

    return positions


# ═══════════════════════════════════════════════════════════════════════════════
# QUADRANT PROJECTION — Sphere position to cube quadrant
# ═══════════════════════════════════════════════════════════════════════════════

def sphere_to_quadrant(pos: SpherePosition) -> str:
    """
    Which Color Cube quadrant does this sphere position project into?

    Returns: 'Q1', 'Q2', 'Q3', or 'Q4'
    """
    cube = pos.to_cube()
    return cube.quadrant


def nodes_in_quadrant(
    positions: List[SpherePosition],
    quadrant: str
) -> List[int]:
    """
    Find indices of all nodes that project into a given quadrant.

    Args:
        positions: List of sphere positions
        quadrant: 'Q1', 'Q2', 'Q3', or 'Q4'

    Returns:
        List of indices into the positions list
    """
    return [
        i for i, pos in enumerate(positions)
        if sphere_to_quadrant(pos) == quadrant
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# NEIGHBORHOOD — Angular proximity queries
# ═══════════════════════════════════════════════════════════════════════════════

def find_neighbors(
    target: SpherePosition,
    candidates: List[SpherePosition],
    max_angle: float = None
) -> List[Tuple[int, float]]:
    """
    Find all nodes within angular distance of target.

    Args:
        target: The position to search around
        candidates: List of positions to check
        max_angle: Maximum angular distance (default: 1/φ² — polarity threshold)

    Returns:
        List of (index, angular_distance) tuples, sorted by distance
    """
    if max_angle is None:
        max_angle = INV_PHI ** 2  # ~0.382 radians (~21.9 degrees)

    neighbors = []
    for i, pos in enumerate(candidates):
        dist = angular_distance(target, pos)
        if dist <= max_angle:
            neighbors.append((i, dist))

    neighbors.sort(key=lambda x: x[1])
    return neighbors


def k_nearest(
    target: SpherePosition,
    candidates: List[SpherePosition],
    k: int = 6
) -> List[Tuple[int, float]]:
    """
    Find the k nearest nodes to target by angular distance.

    Default k=6 (one per cardinal direction).

    Returns:
        List of (index, angular_distance) tuples, sorted by distance
    """
    distances = [
        (i, angular_distance(target, pos))
        for i, pos in enumerate(candidates)
    ]
    distances.sort(key=lambda x: x[1])
    return distances[:k]


# ═══════════════════════════════════════════════════════════════════════════════
# GREAT CIRCLE — Traversal paths on the sphere
# ═══════════════════════════════════════════════════════════════════════════════

def great_circle_path(
    start: SpherePosition,
    end: SpherePosition,
    steps: int = 10
) -> List[SpherePosition]:
    """
    Generate points along the great circle arc from start to end.

    This is the shortest path on the sphere surface — natural traversal.

    Args:
        start: Starting position
        end: Ending position
        steps: Number of intermediate points

    Returns:
        List of positions along the arc (including start and end)
    """
    # Convert to Cartesian for SLERP
    def to_cart(sp):
        st = math.sin(sp.theta)
        return (
            st * math.cos(sp.phi),
            st * math.sin(sp.phi),
            math.cos(sp.theta)
        )

    def from_cart(x, y, z, r):
        length = math.sqrt(x * x + y * y + z * z)
        if length < 1e-12:
            return SpherePosition(math.pi / 2, 0.0, r)
        x, y, z = x / length, y / length, z / length
        theta = math.acos(max(-1.0, min(1.0, z)))
        phi = math.atan2(y, x)
        if phi < 0:
            phi += 2 * math.pi
        return SpherePosition(theta=theta, phi=phi, radius=r)

    a = to_cart(start)
    b = to_cart(end)

    # Angle between
    dot = sum(ai * bi for ai, bi in zip(a, b))
    dot = max(-1.0, min(1.0, dot))
    omega = math.acos(dot)

    if omega < 1e-12:
        # Same point — just return start
        return [SpherePosition(theta=start.theta, phi=start.phi, radius=start.radius)]

    r = start.radius
    path = []

    for i in range(steps + 1):
        t = i / steps
        # SLERP (spherical linear interpolation)
        sin_omega = math.sin(omega)
        if sin_omega < 1e-12:
            # Degenerate — linear interpolation
            cx = a[0] * (1 - t) + b[0] * t
            cy = a[1] * (1 - t) + b[1] * t
            cz = a[2] * (1 - t) + b[2] * t
        else:
            sa = math.sin((1 - t) * omega) / sin_omega
            sb = math.sin(t * omega) / sin_omega
            cx = a[0] * sa + b[0] * sb
            cy = a[1] * sa + b[1] * sb
            cz = a[2] * sa + b[2] * sb

        path.append(from_cart(cx, cy, cz, r))

    return path


# ═══════════════════════════════════════════════════════════════════════════════
# SELF POSITION — Center of the sphere
# ═══════════════════════════════════════════════════════════════════════════════

def self_position() -> SpherePosition:
    """
    Self sits at the center of the hypersphere.
    Radius = 0. Not on the surface.
    All nodes are equidistant from Self.
    """
    return SpherePosition(theta=math.pi / 2, phi=0.0, radius=0.0)


def distance_from_self(pos: SpherePosition) -> float:
    """
    Distance from Self (center) to a surface node.
    For a unit sphere, this is always the radius.
    Self is at r=0, surface nodes at r=1.
    """
    return pos.radius