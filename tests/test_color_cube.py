"""
Tests for Color Cube — Base Righteous Frame + Base Ordered Frame

Validates:
    1. Axis definitions (X=Blue/Yellow, Y=Red/Green, Z=Time)
    2. Quadrant geometry (4 righteous quadrants)
    3. Robinson operations on all 3 axes
    4. Heat derivation (magnitude from chromatic position, not an axis)
    5. Righteousness evaluation (alignment with cube frame)
    6. Wave function (side view of any axis)
    7. Pole/opponent relationships
"""

import math
import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.color_cube import (
    CubePosition, evaluate_righteousness, get_quadrant_alignment,
    robinson_identity, robinson_successor, robinson_addition, robinson_multiplication,
    apply_robinson,
    heat_from_position, heat_zone, wave_amplitude,
    clamp_chromatic, normalize_to_cube, cube_origin, cube_pole, quadrant_center,
    opponent_color, color_to_axis,
)
from core.node_constants import (
    PHI, INV_PHI, K,
    ROBINSON_IDENTITY, ROBINSON_SUCCESSOR, ROBINSON_ADDITION, ROBINSON_MULTIPLICATION,
    CUBE_AXES, CUBE_QUADRANTS, CARDINAL_TO_AXIS,
)


class TestAxisDefinitions(unittest.TestCase):
    """Axis definitions are locked and consistent everywhere."""

    def test_x_axis_colors(self):
        """X axis: Yellow(+1) ↔ Blue(-1)."""
        self.assertEqual(CUBE_AXES['x']['positive'], 'yellow')
        self.assertEqual(CUBE_AXES['x']['negative'], 'blue')

    def test_y_axis_colors(self):
        """Y axis: Green(+1) ↔ Red(-1)."""
        self.assertEqual(CUBE_AXES['y']['positive'], 'green')
        self.assertEqual(CUBE_AXES['y']['negative'], 'red')

    def test_z_axis_time(self):
        """Z axis: Future(+τ) ↔ Past(-τ)."""
        self.assertEqual(CUBE_AXES['z']['positive'], 'future')
        self.assertEqual(CUBE_AXES['z']['negative'], 'past')

    def test_cardinal_directions(self):
        """E/W → X, N/S → Y, U/D → Z."""
        self.assertEqual(CUBE_AXES['x']['cardinal_pos'], 'E')
        self.assertEqual(CUBE_AXES['x']['cardinal_neg'], 'W')
        self.assertEqual(CUBE_AXES['y']['cardinal_pos'], 'N')
        self.assertEqual(CUBE_AXES['y']['cardinal_neg'], 'S')
        self.assertEqual(CUBE_AXES['z']['cardinal_pos'], 'U')
        self.assertEqual(CUBE_AXES['z']['cardinal_neg'], 'D')

    def test_cardinal_to_axis_mapping(self):
        """Cardinal directions map to correct axis and polarity."""
        self.assertEqual(CARDINAL_TO_AXIS['E'], ('x', +1))
        self.assertEqual(CARDINAL_TO_AXIS['W'], ('x', -1))
        self.assertEqual(CARDINAL_TO_AXIS['N'], ('y', +1))
        self.assertEqual(CARDINAL_TO_AXIS['S'], ('y', -1))
        self.assertEqual(CARDINAL_TO_AXIS['U'], ('z', +1))
        self.assertEqual(CARDINAL_TO_AXIS['D'], ('z', -1))

    def test_poles(self):
        """cube_pole returns correct positions."""
        yellow = cube_pole('yellow')
        self.assertEqual(yellow.x, +1.0)
        self.assertEqual(yellow.y, 0.0)

        blue = cube_pole('blue')
        self.assertEqual(blue.x, -1.0)
        self.assertEqual(blue.y, 0.0)

        green = cube_pole('green')
        self.assertEqual(green.x, 0.0)
        self.assertEqual(green.y, +1.0)

        red = cube_pole('red')
        self.assertEqual(red.x, 0.0)
        self.assertEqual(red.y, -1.0)

    def test_pole_cardinals(self):
        """Cardinal poles match color poles."""
        self.assertEqual(cube_pole('E').x, cube_pole('yellow').x)
        self.assertEqual(cube_pole('W').x, cube_pole('blue').x)
        self.assertEqual(cube_pole('N').y, cube_pole('green').y)
        self.assertEqual(cube_pole('S').y, cube_pole('red').y)

    def test_opponent_colors(self):
        """Opponent pairs are correct."""
        self.assertEqual(opponent_color('yellow'), 'blue')
        self.assertEqual(opponent_color('blue'), 'yellow')
        self.assertEqual(opponent_color('green'), 'red')
        self.assertEqual(opponent_color('red'), 'green')
        self.assertEqual(opponent_color('future'), 'past')
        self.assertEqual(opponent_color('past'), 'future')

    def test_color_to_axis(self):
        """Colors map to correct axes and polarities."""
        self.assertEqual(color_to_axis('yellow'), ('x', +1))
        self.assertEqual(color_to_axis('blue'), ('x', -1))
        self.assertEqual(color_to_axis('green'), ('y', +1))
        self.assertEqual(color_to_axis('red'), ('y', -1))
        self.assertEqual(color_to_axis('future'), ('z', +1))
        self.assertEqual(color_to_axis('past'), ('z', -1))


class TestQuadrantGeometry(unittest.TestCase):
    """Four quadrants form the base righteous frame."""

    def test_four_quadrants_exist(self):
        """Exactly 4 quadrants."""
        self.assertEqual(len(CUBE_QUADRANTS), 4)
        self.assertIn('Q1', CUBE_QUADRANTS)
        self.assertIn('Q2', CUBE_QUADRANTS)
        self.assertIn('Q3', CUBE_QUADRANTS)
        self.assertIn('Q4', CUBE_QUADRANTS)

    def test_q1_yellow_green(self):
        """Q1 = (+X, +Y) = Yellow+Green = NE."""
        q = CUBE_QUADRANTS['Q1']
        self.assertEqual(q['x'], +1)
        self.assertEqual(q['y'], +1)
        self.assertEqual(q['colors'], ('yellow', 'green'))
        self.assertEqual(q['cardinal'], 'NE')

    def test_q2_blue_green(self):
        """Q2 = (-X, +Y) = Blue+Green = NW."""
        q = CUBE_QUADRANTS['Q2']
        self.assertEqual(q['x'], -1)
        self.assertEqual(q['y'], +1)
        self.assertEqual(q['colors'], ('blue', 'green'))

    def test_q3_blue_red(self):
        """Q3 = (-X, -Y) = Blue+Red = SW."""
        q = CUBE_QUADRANTS['Q3']
        self.assertEqual(q['x'], -1)
        self.assertEqual(q['y'], -1)
        self.assertEqual(q['colors'], ('blue', 'red'))

    def test_q4_yellow_red(self):
        """Q4 = (+X, -Y) = Yellow+Red = SE."""
        q = CUBE_QUADRANTS['Q4']
        self.assertEqual(q['x'], +1)
        self.assertEqual(q['y'], -1)
        self.assertEqual(q['colors'], ('yellow', 'red'))

    def test_position_quadrant_classification(self):
        """CubePosition correctly identifies its quadrant."""
        self.assertEqual(CubePosition(0.5, 0.5, 0.0).quadrant, 'Q1')
        self.assertEqual(CubePosition(-0.5, 0.5, 0.0).quadrant, 'Q2')
        self.assertEqual(CubePosition(-0.5, -0.5, 0.0).quadrant, 'Q3')
        self.assertEqual(CubePosition(0.5, -0.5, 0.0).quadrant, 'Q4')

    def test_origin_is_q1(self):
        """Origin (0,0) falls in Q1 by convention (>= 0)."""
        origin = CubePosition(0.0, 0.0, 0.0)
        self.assertEqual(origin.quadrant, 'Q1')

    def test_quadrant_alignment(self):
        """Quadrant alignment scoring works."""
        pos_q1 = CubePosition(0.5, 0.5, 0.0)
        self.assertEqual(get_quadrant_alignment(pos_q1, 'Q1'), 1.0)
        self.assertEqual(get_quadrant_alignment(pos_q1, 'Q3'), 0.0)

    def test_quadrant_centers(self):
        """Quadrant centers are at the midpoints."""
        c1 = quadrant_center('Q1')
        self.assertEqual(c1.x, 0.5)
        self.assertEqual(c1.y, 0.5)

        c3 = quadrant_center('Q3')
        self.assertEqual(c3.x, -0.5)
        self.assertEqual(c3.y, -0.5)


class TestRobinsonOperations(unittest.TestCase):
    """Robinson arithmetic on all three axes simultaneously."""

    def test_identity_factor(self):
        """R=1: Identity preserves value."""
        self.assertEqual(ROBINSON_IDENTITY, 1)
        self.assertAlmostEqual(robinson_identity(0.5), 0.5)
        self.assertAlmostEqual(robinson_identity(1.0), 1.0)

    def test_successor_factor(self):
        """R=4φ/7: Successor steps toward opponent."""
        expected = (4 * PHI) / 7
        self.assertAlmostEqual(ROBINSON_SUCCESSOR, expected)
        result = robinson_successor(1.0)
        self.assertAlmostEqual(result, expected)

    def test_addition_factor(self):
        """R=4/3: Addition composes positions."""
        self.assertAlmostEqual(ROBINSON_ADDITION, 4 / 3)
        result = robinson_addition(0.3, 0.4)
        self.assertAlmostEqual(result, (0.3 + 0.4) * (4 / 3))

    def test_multiplication_factor(self):
        """R=13/10: Multiplication scales along axis."""
        self.assertAlmostEqual(ROBINSON_MULTIPLICATION, 13 / 10)
        result = robinson_multiplication(0.5, 2.0)
        self.assertAlmostEqual(result, 0.5 * 2.0 * 1.3)

    def test_apply_robinson_dispatch(self):
        """apply_robinson correctly dispatches all operations."""
        self.assertAlmostEqual(apply_robinson(0.5, 'identity'), 0.5)
        self.assertAlmostEqual(apply_robinson(0.5, 'successor'), 0.5 * ROBINSON_SUCCESSOR)
        self.assertAlmostEqual(apply_robinson(0.3, 'addition', 0.4), (0.3 + 0.4) * ROBINSON_ADDITION)
        self.assertAlmostEqual(apply_robinson(0.5, 'multiplication', 2.0), 0.5 * 2.0 * ROBINSON_MULTIPLICATION)

    def test_apply_robinson_invalid(self):
        """Invalid operation raises ValueError."""
        with self.assertRaises(ValueError):
            apply_robinson(1.0, 'invalid_op')


class TestHeatDerivation(unittest.TestCase):
    """Heat is magnitude from motion, NOT an axis."""

    def test_center_no_heat(self):
        """Center (0,0) = no heat, achromatic."""
        origin = CubePosition(0.0, 0.0, 0.0)
        self.assertAlmostEqual(origin.heat, 0.0)

    def test_single_axis_heat(self):
        """Edge = single-axis heat, max 1.0."""
        pos = CubePosition(1.0, 0.0, 0.0)
        self.assertAlmostEqual(pos.heat, 1.0)

        pos2 = CubePosition(0.0, -1.0, 0.0)
        self.assertAlmostEqual(pos2.heat, 1.0)

    def test_corner_max_heat(self):
        """Corner = maximum heat = √2."""
        corner = CubePosition(1.0, 1.0, 0.0)
        self.assertAlmostEqual(corner.heat, math.sqrt(2))

    def test_heat_independent_of_tau(self):
        """Heat is √(x²+y²) — tau does NOT contribute."""
        pos_a = CubePosition(0.5, 0.5, 0.0)
        pos_b = CubePosition(0.5, 0.5, 100.0)
        self.assertAlmostEqual(pos_a.heat, pos_b.heat)

    def test_heat_zones(self):
        """Heat zone classification."""
        self.assertEqual(heat_zone(0.0), 'achromatic')
        self.assertEqual(heat_zone(0.1), 'achromatic')
        self.assertEqual(heat_zone(0.3), 'warm')
        self.assertEqual(heat_zone(0.7), 'hot')
        self.assertEqual(heat_zone(1.2), 'maximum')

    def test_heat_formula(self):
        """Heat = √(x² + y²) exactly."""
        pos = CubePosition(0.3, 0.4, 0.0)
        expected = math.sqrt(0.3 ** 2 + 0.4 ** 2)
        self.assertAlmostEqual(heat_from_position(pos), expected)
        self.assertAlmostEqual(pos.heat, 0.5)  # 3-4-5 triangle


class TestRighteousness(unittest.TestCase):
    """Righteousness measures alignment with the Color Cube frame."""

    def test_origin_is_perfectly_righteous(self):
        """R=0 at the origin (center of cube)."""
        origin = CubePosition(0.0, 0.0, 0.0)
        self.assertAlmostEqual(evaluate_righteousness(origin), 0.0)

    def test_deviation_increases_r(self):
        """Moving away from reference increases R."""
        near = CubePosition(0.1, 0.0, 0.0)
        far = CubePosition(0.9, 0.0, 0.0)
        r_near = evaluate_righteousness(near)
        r_far = evaluate_righteousness(far)
        self.assertGreater(r_far, r_near)

    def test_r_with_custom_reference(self):
        """R can be evaluated against any reference point."""
        pos = CubePosition(0.5, 0.5, 0.0)
        ref = CubePosition(0.5, 0.5, 0.0)
        self.assertAlmostEqual(evaluate_righteousness(pos, ref), 0.0)

    def test_r_is_euclidean_distance(self):
        """R = euclidean distance from reference."""
        pos = CubePosition(0.3, 0.4, 0.0)
        r = evaluate_righteousness(pos)
        self.assertAlmostEqual(r, 0.5)  # 3-4-5 triangle


class TestCubePosition(unittest.TestCase):
    """CubePosition properties and utilities."""

    def test_polarity_signs(self):
        """Polarity correctly identifies axis signs."""
        pos = CubePosition(0.5, -0.3, 1.0)
        self.assertEqual(pos.polarity_x, +1)
        self.assertEqual(pos.polarity_y, -1)
        self.assertEqual(pos.polarity_z, +1)

    def test_color_name(self):
        """Color name reflects quadrant."""
        self.assertEqual(CubePosition(0.5, 0.5, 0.0).color_name, 'yellow+green')
        self.assertEqual(CubePosition(-0.5, -0.5, 0.0).color_name, 'blue+red')

    def test_as_tuple(self):
        """as_tuple returns (x, y, tau)."""
        pos = CubePosition(0.1, 0.2, 0.3)
        self.assertEqual(pos.as_tuple(), (0.1, 0.2, 0.3))

    def test_distance_to(self):
        """Euclidean distance between positions."""
        a = CubePosition(0.0, 0.0, 0.0)
        b = CubePosition(1.0, 0.0, 0.0)
        self.assertAlmostEqual(a.distance_to(b), 1.0)

    def test_clamp_chromatic(self):
        """Chromatic values clamped to [-1, +1]."""
        self.assertEqual(clamp_chromatic(1.5), 1.0)
        self.assertEqual(clamp_chromatic(-1.5), -1.0)
        self.assertEqual(clamp_chromatic(0.5), 0.5)

    def test_normalize_to_cube(self):
        """normalize_to_cube clamps chromatic, keeps tau."""
        pos = normalize_to_cube(2.0, -2.0, 50.0)
        self.assertEqual(pos.x, 1.0)
        self.assertEqual(pos.y, -1.0)
        self.assertEqual(pos.tau, 50.0)

    def test_cube_origin(self):
        """Origin is (0, 0, 0)."""
        o = cube_origin()
        self.assertEqual(o.x, 0.0)
        self.assertEqual(o.y, 0.0)
        self.assertEqual(o.tau, 0.0)


class TestWaveFunction(unittest.TestCase):
    """Side view of any axis = wave function."""

    def test_x_axis_wave(self):
        """Wave amplitude on X axis."""
        pos = CubePosition(0.7, 0.3, 0.0)
        self.assertAlmostEqual(wave_amplitude(pos, 'x'), 0.7)

    def test_y_axis_wave(self):
        """Wave amplitude on Y axis."""
        pos = CubePosition(0.7, 0.3, 0.0)
        self.assertAlmostEqual(wave_amplitude(pos, 'y'), 0.3)

    def test_invalid_axis_raises(self):
        """Only x and y are valid wave axes."""
        pos = CubePosition(0.7, 0.3, 0.0)
        with self.assertRaises(ValueError):
            wave_amplitude(pos, 'z')


if __name__ == '__main__':
    unittest.main()
