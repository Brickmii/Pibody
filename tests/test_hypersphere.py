"""
Tests for Hypersphere — Node Topology

Validates:
    1. Angular coordinates (theta, phi) on sphere surface
    2. Projection to Color Cube (sphere → cube → sphere round-trip)
    3. Angular distance (relationship strength)
    4. n² surface scaling
    5. Node placement (even distribution, collision avoidance)
    6. Quadrant projection (sphere position → cube quadrant)
    7. Neighborhood queries (angular proximity)
    8. Great circle traversal paths
    9. Self at center (radius = 0)
"""

import math
import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.hypersphere import (
    SpherePosition, angular_distance, relationship_strength,
    are_aligned, are_opposed,
    surface_area, max_nodes_at_resolution, n_squared_capacity,
    place_node_near, place_evenly,
    sphere_to_quadrant, nodes_in_quadrant,
    find_neighbors, k_nearest,
    great_circle_path,
    self_position, distance_from_self,
)
from core.color_cube import CubePosition
from core.node_constants import PHI, INV_PHI


class TestSpherePosition(unittest.TestCase):
    """Angular coordinates on the hypersphere surface."""

    def test_default_position(self):
        """Default: equator, phi=0 (present moment, yellow/east direction)."""
        sp = SpherePosition()
        self.assertAlmostEqual(sp.theta, math.pi / 2)
        self.assertAlmostEqual(sp.phi, 0.0)
        self.assertAlmostEqual(sp.radius, 1.0)

    def test_theta_clamped(self):
        """Theta clamped to [0, π]."""
        sp = SpherePosition(theta=-0.5)
        self.assertEqual(sp.theta, 0.0)
        sp2 = SpherePosition(theta=4.0)
        self.assertEqual(sp2.theta, math.pi)

    def test_phi_normalized(self):
        """Phi normalized to [0, 2π)."""
        sp = SpherePosition(phi=7.0)
        self.assertGreaterEqual(sp.phi, 0.0)
        self.assertLess(sp.phi, 2 * math.pi)

        sp2 = SpherePosition(phi=-1.0)
        self.assertGreaterEqual(sp2.phi, 0.0)

    def test_as_tuple(self):
        """as_tuple returns (theta, phi, radius)."""
        sp = SpherePosition(theta=1.0, phi=2.0, radius=1.0)
        self.assertEqual(sp.as_tuple(), (1.0, 2.0, 1.0))


class TestCubeProjection(unittest.TestCase):
    """Sphere ↔ Cube projection round-trips."""

    def test_north_pole_projects_to_future(self):
        """theta=0 (north pole) → +Z (future)."""
        sp = SpherePosition(theta=0.0, phi=0.0)
        cube = sp.to_cube()
        self.assertAlmostEqual(cube.x, 0.0, places=10)
        self.assertAlmostEqual(cube.y, 0.0, places=10)
        self.assertAlmostEqual(cube.tau, 1.0)

    def test_south_pole_projects_to_past(self):
        """theta=π (south pole) → -Z (past)."""
        sp = SpherePosition(theta=math.pi, phi=0.0)
        cube = sp.to_cube()
        self.assertAlmostEqual(cube.x, 0.0, places=10)
        self.assertAlmostEqual(cube.y, 0.0, places=10)
        self.assertAlmostEqual(cube.tau, -1.0)

    def test_equator_east_projects_to_yellow(self):
        """theta=π/2, phi=0 → +X (yellow/east)."""
        sp = SpherePosition(theta=math.pi / 2, phi=0.0)
        cube = sp.to_cube()
        self.assertAlmostEqual(cube.x, 1.0)
        self.assertAlmostEqual(cube.y, 0.0, places=10)
        self.assertAlmostEqual(cube.tau, 0.0, places=10)

    def test_equator_north_projects_to_green(self):
        """theta=π/2, phi=π/2 → +Y (green/north)."""
        sp = SpherePosition(theta=math.pi / 2, phi=math.pi / 2)
        cube = sp.to_cube()
        self.assertAlmostEqual(cube.x, 0.0, places=10)
        self.assertAlmostEqual(cube.y, 1.0)
        self.assertAlmostEqual(cube.tau, 0.0, places=10)

    def test_equator_west_projects_to_blue(self):
        """theta=π/2, phi=π → -X (blue/west)."""
        sp = SpherePosition(theta=math.pi / 2, phi=math.pi)
        cube = sp.to_cube()
        self.assertAlmostEqual(cube.x, -1.0)
        self.assertAlmostEqual(cube.y, 0.0, places=10)
        self.assertAlmostEqual(cube.tau, 0.0, places=10)

    def test_equator_south_projects_to_red(self):
        """theta=π/2, phi=3π/2 → -Y (red/south)."""
        sp = SpherePosition(theta=math.pi / 2, phi=3 * math.pi / 2)
        cube = sp.to_cube()
        self.assertAlmostEqual(cube.x, 0.0, places=10)
        self.assertAlmostEqual(cube.y, -1.0)
        self.assertAlmostEqual(cube.tau, 0.0, places=10)

    def test_round_trip(self):
        """Sphere → Cube → Sphere preserves position."""
        for theta in [0.3, 0.8, 1.2, 1.8, 2.5]:
            for phi in [0.0, 0.5, 1.5, 3.0, 5.0]:
                sp = SpherePosition(theta=theta, phi=phi)
                cube = sp.to_cube()
                sp2 = SpherePosition.from_cube(cube)
                self.assertAlmostEqual(sp.theta, sp2.theta, places=8,
                    msg=f"theta mismatch at ({theta}, {phi})")
                self.assertAlmostEqual(sp.phi, sp2.phi, places=8,
                    msg=f"phi mismatch at ({theta}, {phi})")

    def test_center_projects_to_origin(self):
        """radius=0 (Self) projects to cube origin."""
        sp = SpherePosition(theta=1.0, phi=1.0, radius=0.0)
        cube = sp.to_cube()
        self.assertAlmostEqual(cube.x, 0.0)
        self.assertAlmostEqual(cube.y, 0.0)
        self.assertAlmostEqual(cube.tau, 0.0)

    def test_cube_heat_from_sphere(self):
        """Sphere position gives correct heat magnitude when projected."""
        # Equator position → max chromatic, zero tau
        sp = SpherePosition(theta=math.pi / 2, phi=math.pi / 4)  # 45° in XY
        cube = sp.to_cube()
        # At equator: x = cos(π/4) ≈ 0.707, y = sin(π/4) ≈ 0.707
        self.assertAlmostEqual(cube.heat, 1.0, places=5)

        # Pole → zero heat
        sp_pole = SpherePosition(theta=0.0, phi=0.0)
        cube_pole = sp_pole.to_cube()
        self.assertAlmostEqual(cube_pole.heat, 0.0, places=10)


class TestAngularDistance(unittest.TestCase):
    """Angular distance = relationship strength between nodes."""

    def test_same_position_zero_distance(self):
        """Same position → distance = 0."""
        sp = SpherePosition(theta=1.0, phi=1.0)
        self.assertAlmostEqual(angular_distance(sp, sp), 0.0)

    def test_antipodal_pi_distance(self):
        """Opposite positions → distance = π."""
        a = SpherePosition(theta=0.0, phi=0.0)          # North pole
        b = SpherePosition(theta=math.pi, phi=0.0)      # South pole
        self.assertAlmostEqual(angular_distance(a, b), math.pi)

    def test_orthogonal_half_pi(self):
        """90° apart → distance = π/2."""
        a = SpherePosition(theta=math.pi / 2, phi=0.0)        # East
        b = SpherePosition(theta=math.pi / 2, phi=math.pi / 2)  # North
        self.assertAlmostEqual(angular_distance(a, b), math.pi / 2, places=5)

    def test_symmetry(self):
        """Distance is symmetric: d(a,b) = d(b,a)."""
        a = SpherePosition(theta=0.5, phi=1.0)
        b = SpherePosition(theta=1.5, phi=2.5)
        self.assertAlmostEqual(angular_distance(a, b), angular_distance(b, a))

    def test_relationship_strength(self):
        """Relationship strength: 1.0 for same, 0.0 for antipodal."""
        same = SpherePosition(theta=1.0, phi=1.0)
        self.assertAlmostEqual(relationship_strength(same, same), 1.0)

        a = SpherePosition(theta=0.0, phi=0.0)
        b = SpherePosition(theta=math.pi, phi=0.0)
        self.assertAlmostEqual(relationship_strength(a, b), 0.0)

    def test_are_aligned(self):
        """Aligned positions within threshold."""
        a = SpherePosition(theta=1.0, phi=1.0)
        b = SpherePosition(theta=1.001, phi=1.001)
        self.assertTrue(are_aligned(a, b))

    def test_are_opposed(self):
        """Opposed positions near antipodal."""
        a = SpherePosition(theta=0.01, phi=0.0)
        b = SpherePosition(theta=math.pi - 0.01, phi=math.pi)
        self.assertTrue(are_opposed(a, b))


class TestN2Scaling(unittest.TestCase):
    """Surface distribution scales as n², not n³."""

    def test_surface_area_formula(self):
        """4πr² for unit sphere."""
        self.assertAlmostEqual(surface_area(1.0), 4 * math.pi)
        self.assertAlmostEqual(surface_area(2.0), 16 * math.pi)

    def test_max_nodes_increases_with_finer_resolution(self):
        """More nodes fit at finer resolution."""
        coarse = max_nodes_at_resolution(0.5)
        fine = max_nodes_at_resolution(0.1)
        self.assertGreater(fine, coarse)

    def test_n_squared_scaling(self):
        """Average separation scales as √(4π/n) — confirms n² not n³."""
        sep_10 = n_squared_capacity(10)
        sep_40 = n_squared_capacity(40)
        # Doubling nodes by 4x → separation halved (√(1/4) = 0.5)
        ratio = sep_40 / sep_10
        self.assertAlmostEqual(ratio, 0.5, places=5)


class TestNodePlacement(unittest.TestCase):
    """Placing nodes on the sphere surface."""

    def test_place_evenly_count(self):
        """place_evenly returns correct count."""
        for n in [1, 6, 12, 50]:
            positions = place_evenly(n)
            self.assertEqual(len(positions), n)

    def test_place_evenly_zero(self):
        """place_evenly(0) returns empty list."""
        self.assertEqual(place_evenly(0), [])

    def test_place_evenly_on_surface(self):
        """All placed nodes are on the unit sphere (radius=1)."""
        for pos in place_evenly(20):
            self.assertAlmostEqual(pos.radius, 1.0)

    def test_place_evenly_spread(self):
        """Evenly placed nodes have reasonable minimum separation."""
        positions = place_evenly(12)
        min_dist = float('inf')
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                d = angular_distance(positions[i], positions[j])
                min_dist = min(min_dist, d)
        # 12 nodes on sphere → minimum separation should be > 0.3 radians
        self.assertGreater(min_dist, 0.3)

    def test_place_node_near_respects_separation(self):
        """place_node_near avoids collision with existing nodes."""
        existing = [
            SpherePosition(theta=math.pi / 2, phi=0.0),
            SpherePosition(theta=math.pi / 2, phi=1.0),
        ]
        target = SpherePosition(theta=math.pi / 2, phi=0.0)  # Same as existing[0]
        min_sep = 0.1

        result = place_node_near(target, existing, min_separation=min_sep)

        # Should not be too close to any existing node
        for ex in existing:
            self.assertGreaterEqual(angular_distance(result, ex), min_sep * 0.9)

    def test_place_node_near_returns_target_if_valid(self):
        """If target is valid, it's returned unchanged."""
        target = SpherePosition(theta=1.0, phi=1.0)
        result = place_node_near(target, [], min_separation=0.1)
        self.assertAlmostEqual(result.theta, target.theta)
        self.assertAlmostEqual(result.phi, target.phi)


class TestQuadrantProjection(unittest.TestCase):
    """Sphere position → cube quadrant projection."""

    def test_equator_quadrants(self):
        """Equator positions map to expected quadrants."""
        # phi=0 → +X → Q1 or Q4 (y≈0, x>0 → Q1)
        q = sphere_to_quadrant(SpherePosition(theta=math.pi / 2, phi=0.0))
        self.assertEqual(q, 'Q1')  # +X, y=0 → Q1 by convention

        # phi=π/4 → Q1 (both x,y > 0)
        q = sphere_to_quadrant(SpherePosition(theta=math.pi / 2, phi=math.pi / 4))
        self.assertEqual(q, 'Q1')

        # phi=3π/4 → Q2 (x<0, y>0)
        q = sphere_to_quadrant(SpherePosition(theta=math.pi / 2, phi=3 * math.pi / 4))
        self.assertEqual(q, 'Q2')

        # phi=5π/4 → Q3 (x<0, y<0)
        q = sphere_to_quadrant(SpherePosition(theta=math.pi / 2, phi=5 * math.pi / 4))
        self.assertEqual(q, 'Q3')

        # phi=7π/4 → Q4 (x>0, y<0)
        q = sphere_to_quadrant(SpherePosition(theta=math.pi / 2, phi=7 * math.pi / 4))
        self.assertEqual(q, 'Q4')

    def test_nodes_in_quadrant(self):
        """Filter nodes by quadrant."""
        positions = place_evenly(20)
        for q_name in ['Q1', 'Q2', 'Q3', 'Q4']:
            indices = nodes_in_quadrant(positions, q_name)
            # Each quadrant should have some nodes
            self.assertGreater(len(indices), 0,
                msg=f"No nodes in {q_name}")


class TestNeighborhood(unittest.TestCase):
    """Angular proximity queries."""

    def test_find_neighbors(self):
        """find_neighbors returns nodes within threshold."""
        positions = place_evenly(20)
        target = positions[0]
        neighbors = find_neighbors(target, positions, max_angle=1.0)
        # Should include self (distance=0) and some nearby nodes
        self.assertGreater(len(neighbors), 0)
        # All within threshold
        for idx, dist in neighbors:
            self.assertLessEqual(dist, 1.0)
        # Sorted by distance
        for i in range(len(neighbors) - 1):
            self.assertLessEqual(neighbors[i][1], neighbors[i + 1][1])

    def test_k_nearest(self):
        """k_nearest returns exactly k results."""
        positions = place_evenly(20)
        target = positions[0]
        nearest = k_nearest(target, positions, k=5)
        self.assertEqual(len(nearest), 5)
        # First should be self (distance ≈ 0)
        self.assertAlmostEqual(nearest[0][1], 0.0, places=5)
        # Sorted by distance
        for i in range(len(nearest) - 1):
            self.assertLessEqual(nearest[i][1], nearest[i + 1][1])

    def test_k_nearest_default_six(self):
        """Default k=6 (one per cardinal direction)."""
        positions = place_evenly(20)
        nearest = k_nearest(positions[0], positions)
        self.assertEqual(len(nearest), 6)


class TestGreatCircle(unittest.TestCase):
    """Traversal paths on the sphere surface."""

    def test_great_circle_endpoints(self):
        """Path starts and ends at correct positions."""
        a = SpherePosition(theta=1.0, phi=0.0)
        b = SpherePosition(theta=1.0, phi=2.0)
        path = great_circle_path(a, b, steps=10)

        self.assertEqual(len(path), 11)  # steps + 1
        self.assertAlmostEqual(path[0].theta, a.theta, places=5)
        self.assertAlmostEqual(path[0].phi, a.phi, places=5)
        self.assertAlmostEqual(path[-1].theta, b.theta, places=5)
        self.assertAlmostEqual(path[-1].phi, b.phi, places=5)

    def test_great_circle_stays_on_surface(self):
        """All path points remain on sphere surface."""
        a = SpherePosition(theta=0.5, phi=0.0)
        b = SpherePosition(theta=2.0, phi=3.0)
        path = great_circle_path(a, b, steps=20)

        for point in path:
            self.assertAlmostEqual(point.radius, a.radius, places=5)

    def test_great_circle_same_point(self):
        """Path from point to itself returns single point."""
        a = SpherePosition(theta=1.0, phi=1.0)
        path = great_circle_path(a, a, steps=10)
        self.assertEqual(len(path), 1)

    def test_great_circle_monotonic_distance(self):
        """Distance from start increases monotonically along path."""
        a = SpherePosition(theta=0.5, phi=0.0)
        b = SpherePosition(theta=2.0, phi=2.0)
        path = great_circle_path(a, b, steps=20)

        prev_dist = 0.0
        for point in path[1:]:
            dist = angular_distance(a, point)
            self.assertGreaterEqual(dist, prev_dist - 1e-10)
            prev_dist = dist


class TestSelfPosition(unittest.TestCase):
    """Self at the center of the hypersphere."""

    def test_self_at_center(self):
        """Self has radius = 0."""
        s = self_position()
        self.assertAlmostEqual(s.radius, 0.0)

    def test_self_projects_to_cube_origin(self):
        """Self at center projects to (0, 0, 0) in cube."""
        s = self_position()
        cube = s.to_cube()
        self.assertAlmostEqual(cube.x, 0.0)
        self.assertAlmostEqual(cube.y, 0.0)
        self.assertAlmostEqual(cube.tau, 0.0)

    def test_distance_from_self(self):
        """All surface nodes are equidistant from self (radius)."""
        positions = place_evenly(10)
        for pos in positions:
            self.assertAlmostEqual(distance_from_self(pos), 1.0)

    def test_self_distance_zero(self):
        """Self's distance from itself is 0."""
        s = self_position()
        self.assertAlmostEqual(distance_from_self(s), 0.0)


if __name__ == '__main__':
    unittest.main()
