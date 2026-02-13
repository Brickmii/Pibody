"""
PBAI Axis Overflow Test — 44/45 Emergence Threshold

Tests the MAX_ORDER_TOKENS (44) limit enforcement:
    - add_axis_safe creates child on 45th axis
    - Parent keeps all 44 axes untouched
    - Strengthening existing axis at 44 doesn't overflow
    - Child concept follows naming convention: {parent}_c{N}
    - Multiple overflows create sequential children
    - Psychology nodes overflow correctly (tree formation)
    - Clock _collect_tree_axes traverses full tree

Run: python3 -m unittest tests.test_axis_overflow -v
"""

import math
import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.node_constants import K, MAX_ORDER_TOKENS, THRESHOLD_DENOMINATOR
from core.nodes import Node, reset_birth_for_testing
from core.manifold import Manifold, reset_pbai_manifold
from core.hypersphere import angular_distance, SpherePosition


def fresh_manifold():
    """Create a fresh born manifold for testing."""
    reset_pbai_manifold()
    reset_birth_for_testing()
    m = Manifold()
    m.birth()
    return m


class TestMaxOrderTokensConstant(unittest.TestCase):
    """Test that the constant is correctly defined."""

    def test_constant_value(self):
        self.assertEqual(MAX_ORDER_TOKENS, 44)

    def test_matches_threshold_denominator(self):
        self.assertEqual(MAX_ORDER_TOKENS, THRESHOLD_DENOMINATOR)


class TestAddAxisSafeUnderLimit(unittest.TestCase):
    """Test add_axis_safe when node is under the 44-axis limit."""

    def setUp(self):
        self.m = fresh_manifold()
        self.node = Node(concept="test_node", theta=1.0, phi=1.0, radius=1.0, heat=K)
        self.m.add_node(self.node)
        # Create some target nodes
        for i in range(5):
            t = Node(concept=f"target_{i}", theta=1.0 + i * 0.1, phi=1.0, radius=1.0, heat=K)
            self.m.add_node(t)

    def test_adds_normally_under_limit(self):
        target = self.m.get_node_by_concept("target_0")
        axis = self.m.add_axis_safe(self.node, "test_dir", target.id)
        self.assertIn("test_dir", self.node.frame.axes)
        self.assertEqual(axis.target_id, target.id)

    def test_no_child_created_under_limit(self):
        target = self.m.get_node_by_concept("target_0")
        self.m.add_axis_safe(self.node, "test_dir", target.id)
        children = self.m.get_overflow_children(self.node)
        self.assertEqual(len(children), 0)


class TestAddAxisSafeStrengthen(unittest.TestCase):
    """Test that strengthening existing axis at limit doesn't overflow."""

    def setUp(self):
        self.m = fresh_manifold()
        self.node = Node(concept="full_node", theta=1.0, phi=1.0, radius=1.0, heat=K)
        self.m.add_node(self.node)
        # Fill to exactly 44 axes
        for i in range(44):
            t = Node(concept=f"fill_{i}", theta=1.0 + i * 0.05, phi=1.0 + i * 0.05,
                     radius=1.0, heat=K)
            self.m.add_node(t)
            self.node.add_axis(f"dir_{i}", t.id)

    def test_node_at_limit(self):
        self.assertEqual(len(self.node.frame.axes), 44)

    def test_strengthen_existing_no_overflow(self):
        """Strengthening an existing axis when at 44 should NOT create a child."""
        target = self.m.get_node_by_concept("fill_0")
        old_count = self.node.frame.axes["dir_0"].traversal_count
        axis = self.m.add_axis_safe(self.node, "dir_0", target.id)
        # Should strengthen, not create child
        self.assertEqual(axis.traversal_count, old_count + 1)
        self.assertEqual(len(self.node.frame.axes), 44)
        children = self.m.get_overflow_children(self.node)
        self.assertEqual(len(children), 0)


class TestAddAxisSafeOverflow(unittest.TestCase):
    """Test overflow at the 45th axis — child creation."""

    def setUp(self):
        self.m = fresh_manifold()
        self.parent = Node(concept="parent_node", theta=1.0, phi=1.0, radius=1.0, heat=K)
        self.m.add_node(self.parent)
        # Fill to exactly 44 axes
        for i in range(44):
            t = Node(concept=f"filler_{i}", theta=1.0 + i * 0.05, phi=1.0 + i * 0.05,
                     radius=1.0, heat=K)
            self.m.add_node(t)
            self.parent.add_axis(f"dir_{i}", t.id)
        # Create the 45th target
        self.overflow_target = Node(concept="overflow_target", theta=2.0, phi=2.0,
                                     radius=1.0, heat=K)
        self.m.add_node(self.overflow_target)

    def test_parent_stays_at_44(self):
        """Parent must keep all 44 axes untouched after overflow."""
        self.m.add_axis_safe(self.parent, "overflow_dir", self.overflow_target.id)
        self.assertEqual(len(self.parent.frame.axes), 44)
        # Original 44 axes still present
        for i in range(44):
            self.assertIn(f"dir_{i}", self.parent.frame.axes)

    def test_child_gets_overflow_axis(self):
        """Child should get the 45th axis."""
        axis = self.m.add_axis_safe(self.parent, "overflow_dir", self.overflow_target.id)
        children = self.m.get_overflow_children(self.parent)
        self.assertEqual(len(children), 1)
        child = children[0]
        self.assertIn("overflow_dir", child.frame.axes)
        self.assertEqual(axis.target_id, self.overflow_target.id)

    def test_child_concept_naming(self):
        """Child concept should follow {parent}_c{N} convention."""
        self.m.add_axis_safe(self.parent, "overflow_dir", self.overflow_target.id)
        children = self.m.get_overflow_children(self.parent)
        self.assertEqual(children[0].concept, "parent_node_c0")

    def test_child_near_parent_on_sphere(self):
        """Child should be placed near parent on the hypersphere."""
        self.m.add_axis_safe(self.parent, "overflow_dir", self.overflow_target.id)
        children = self.m.get_overflow_children(self.parent)
        child = children[0]
        parent_sp = SpherePosition(theta=self.parent.theta, phi=self.parent.phi)
        child_sp = SpherePosition(theta=child.theta, phi=child.phi)
        dist = angular_distance(parent_sp, child_sp)
        # Child should be within a reasonable angular distance
        self.assertLess(dist, math.pi / 2)

    def test_child_inherits_parent_properties(self):
        """Child inherits polarity, existence, righteousness from parent."""
        self.parent.polarity = -1
        self.parent.existence = "actual"
        self.parent.righteousness = 0.5
        self.m.add_axis_safe(self.parent, "overflow_dir", self.overflow_target.id)
        children = self.m.get_overflow_children(self.parent)
        child = children[0]
        self.assertEqual(child.polarity, -1)
        self.assertEqual(child.existence, "actual")
        self.assertEqual(child.righteousness, 0.5)


class TestMultipleOverflows(unittest.TestCase):
    """Test sequential overflow creates multiple children."""

    def setUp(self):
        self.m = fresh_manifold()
        self.parent = Node(concept="multi_parent", theta=1.5, phi=1.5, radius=1.0, heat=K)
        self.m.add_node(self.parent)
        # Fill to 44
        for i in range(44):
            t = Node(concept=f"mt_{i}", theta=1.5 + i * 0.03, phi=1.5 + i * 0.03,
                     radius=1.0, heat=K)
            self.m.add_node(t)
            self.parent.add_axis(f"md_{i}", t.id)

    def test_sequential_children(self):
        """Multiple overflows create _c0, _c1, _c2..."""
        for i in range(3):
            t = Node(concept=f"extra_{i}", theta=2.5 + i * 0.1, phi=2.5,
                     radius=1.0, heat=K)
            self.m.add_node(t)
            self.m.add_axis_safe(self.parent, f"extra_dir_{i}", t.id)

        children = self.m.get_overflow_children(self.parent)
        concepts = sorted([c.concept for c in children])
        # First overflow goes to _c0, subsequent ones might go to _c0 (if it has room)
        # or to _c1 etc.
        self.assertGreaterEqual(len(children), 1)
        self.assertTrue(concepts[0].startswith("multi_parent_c"))

    def test_parent_untouched_after_multiple(self):
        """Parent still has exactly 44 axes after 3 overflows."""
        for i in range(3):
            t = Node(concept=f"extra_{i}", theta=2.5 + i * 0.1, phi=2.5,
                     radius=1.0, heat=K)
            self.m.add_node(t)
            self.m.add_axis_safe(self.parent, f"extra_dir_{i}", t.id)
        self.assertEqual(len(self.parent.frame.axes), 44)


class TestPsychologyOverflow(unittest.TestCase):
    """Test overflow on psychology nodes (Identity, Ego, Conscience)."""

    def setUp(self):
        self.m = fresh_manifold()

    def test_identity_overflow(self):
        """Identity node should overflow at 44 axes."""
        identity = self.m.identity_node
        self.assertIsNotNone(identity)
        # Fill identity to 44 axes
        current_count = len(identity.frame.axes)
        for i in range(44 - current_count):
            t = Node(concept=f"id_fill_{i}", theta=1.0 + i * 0.05, phi=1.0,
                     radius=1.0, heat=K)
            self.m.add_node(t)
            identity.add_axis(f"id_dir_{i}", t.id)
        self.assertEqual(len(identity.frame.axes), 44)

        # 45th should overflow
        t = Node(concept="id_overflow", theta=2.5, phi=2.5, radius=1.0, heat=K)
        self.m.add_node(t)
        self.m.add_axis_safe(identity, "id_overflow_dir", t.id)

        # Parent still has 44
        self.assertEqual(len(identity.frame.axes), 44)
        # Child was created
        children = self.m.get_overflow_children(identity)
        self.assertGreaterEqual(len(children), 1)
        self.assertTrue(children[0].concept.startswith("identity_c"))


class TestCollectTreeAxes(unittest.TestCase):
    """Test _collect_tree_axes in clock_node.py."""

    def setUp(self):
        self.m = fresh_manifold()

    def test_collect_root_only(self):
        """When no children, returns root's axes only."""
        from core.clock_node import Clock
        clock = Clock(self.m)
        identity = self.m.identity_node
        all_axes = clock._collect_tree_axes(identity)
        self.assertEqual(len(all_axes), len(identity.frame.axes))

    def test_collect_with_children(self):
        """Includes child axes with prefixed keys."""
        from core.clock_node import Clock
        clock = Clock(self.m)
        identity = self.m.identity_node

        # Fill identity to 44
        current_count = len(identity.frame.axes)
        for i in range(44 - current_count):
            t = Node(concept=f"tc_fill_{i}", theta=1.0 + i * 0.05, phi=1.0,
                     radius=1.0, heat=K)
            self.m.add_node(t)
            identity.add_axis(f"tc_dir_{i}", t.id)

        # Overflow 2 axes
        for i in range(2):
            t = Node(concept=f"tc_over_{i}", theta=2.5 + i * 0.1, phi=2.5,
                     radius=1.0, heat=K)
            self.m.add_node(t)
            self.m.add_axis_safe(identity, f"tc_over_dir_{i}", t.id)

        all_axes = clock._collect_tree_axes(identity)
        # Should have 44 (root) + 2 (child) = 46
        self.assertEqual(len(all_axes), 46)
        # Child axes have ":" in their key
        child_keys = [k for k in all_axes if ":" in k]
        self.assertEqual(len(child_keys), 2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
