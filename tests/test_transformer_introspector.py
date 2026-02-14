"""
PBAI Transformer Introspector Test — ManifoldAttention + Suggest

Tests the transformer-based introspection engine:
    - Axis token extraction shape and normalization
    - Weighted pool produces (1, 32) per node
    - Cross-node attention shape and determinism
    - PBAI-constant initialization (not random)
    - suggest() returns concept strings
    - get_weight_boosts() returns Dict[str, float]
    - Torch fallback works
    - Performance < 50ms per forward pass
    - Param count < 100K

Run: python3 -m unittest tests.test_transformer_introspector -v
"""

import math
import sys
import os
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.node_constants import (
    K, PHI, INV_PHI, MAX_ORDER_TOKENS,
    COST_EVALUATE, PSYCHOLOGY_MIN_HEAT,
    EXISTENCE_ACTUAL,
    BASE_MOTION_PREFIX, ALL_BASE_MOTIONS,
)
from core.nodes import Node, reset_birth_for_testing
from core.manifold import Manifold, reset_pbai_manifold
from core.introspector import (
    Introspector, SimulationResult, ManifoldAttention,
    D_TOKEN, D_NODE, D_EMB, N_HEADS, _HAS_TORCH,
)

# Try to import torch — tests that need it are skipped if unavailable
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def fresh_manifold():
    """Create a fresh born manifold for testing."""
    reset_pbai_manifold()
    reset_birth_for_testing()
    m = Manifold()
    m.birth()
    return m


def populated_manifold():
    """Create a manifold with some concept nodes for testing."""
    m = fresh_manifold()
    # Pump psychology nodes so should_think() passes
    if m.ego_node:
        m.ego_node.heat = 50.0
        m.ego_node.existence = EXISTENCE_ACTUAL
    if m.conscience_node:
        m.conscience_node.heat = 5.0
        m.conscience_node.existence = EXISTENCE_ACTUAL
    if m.identity_node:
        m.identity_node.heat = 20.0
        m.identity_node.existence = EXISTENCE_ACTUAL

    # Add some concept nodes
    concepts = [
        "move_forward", "jump", "attack", "look_left", "look_right",
        "ow_plains_h20_e0p0", "ow_forest_h15_e1p0", "zombie", "pig", "sword",
    ]
    for i, concept in enumerate(concepts):
        n = Node(
            concept=concept,
            theta=0.5 + i * 0.2,
            phi=0.3 + i * 0.5,
            radius=1.0,
            heat=K * (i + 1),
            existence=EXISTENCE_ACTUAL,
            polarity=1,
            order=i,
        )
        m.add_node(n)
        # Give each node a couple of axes pointing to other nodes
        if i > 0:
            prev = m.get_node_by_concept(concepts[i - 1])
            if prev:
                n.add_axis(f"rel_{concepts[i-1]}", prev.id)

    # Add axes from identity to some concepts
    for concept in ["move_forward", "zombie", "sword"]:
        cn = m.get_node_by_concept(concept)
        if cn and m.identity_node:
            m.add_axis_safe(m.identity_node, concept, cn.id)

    return m


class TestConstants(unittest.TestCase):
    """Test introspector constants."""

    def test_d_token(self):
        self.assertEqual(D_TOKEN, 14)

    def test_d_node(self):
        self.assertEqual(D_NODE, 10)

    def test_d_emb(self):
        self.assertEqual(D_EMB, 32)

    def test_n_heads(self):
        self.assertEqual(N_HEADS, 2)

    def test_has_torch(self):
        """Torch should be available on Pi 5."""
        self.assertTrue(_HAS_TORCH)


@unittest.skipUnless(HAS_TORCH, "PyTorch not available")
class TestManifoldAttention(unittest.TestCase):
    """Test the ManifoldAttention module."""

    def setUp(self):
        self.model = ManifoldAttention(D_TOKEN, D_NODE, D_EMB, N_HEADS)
        self.model.eval()

    def test_param_count(self):
        """Total params should be < 100K (actually ~3,872)."""
        total = sum(p.numel() for p in self.model.parameters())
        self.assertLess(total, 100_000)
        self.assertGreater(total, 0)

    def test_param_count_exact(self):
        """Verify expected param count: ~3,872."""
        total = sum(p.numel() for p in self.model.parameters())
        # W_token_proj: 32*14=448, W_node_proj: 32*10=320
        # W_Q: 32*32=1024, W_K: 32*32=1024, W_V: 32*32=1024
        # W_out: 32*1=32
        expected = 448 + 320 + 1024 + 1024 + 1024 + 32
        self.assertEqual(total, expected)

    def test_attention_temp_is_K(self):
        """Attention temperature buffer should equal K."""
        self.assertAlmostEqual(self.model.attention_temp.item(), K, places=3)

    def test_forward_shape(self):
        """Forward pass should return (N, 1) scores."""
        N = 10
        node_embs = torch.randn(N, D_EMB)
        query = torch.randn(1, D_EMB)
        with torch.no_grad():
            scores = self.model.forward(node_embs, query)
        self.assertEqual(scores.shape, (N, 1))

    def test_forward_empty(self):
        """Forward with 0 nodes returns empty tensor."""
        node_embs = torch.zeros(0, D_EMB)
        query = torch.randn(1, D_EMB)
        with torch.no_grad():
            scores = self.model.forward(node_embs, query)
        self.assertEqual(scores.shape, (0, 1))

    def test_forward_deterministic(self):
        """Same input should produce same output (eval mode)."""
        node_embs = torch.randn(5, D_EMB)
        query = torch.randn(1, D_EMB)
        with torch.no_grad():
            s1 = self.model.forward(node_embs, query)
            s2 = self.model.forward(node_embs, query)
        self.assertTrue(torch.allclose(s1, s2))

    def test_forward_scores_nonnegative(self):
        """Scores should be non-negative (attention × sigmoid)."""
        node_embs = torch.randn(8, D_EMB)
        query = torch.randn(1, D_EMB)
        with torch.no_grad():
            scores = self.model.forward(node_embs, query)
        self.assertTrue((scores >= 0).all())

    def test_pool_axis_tokens_shape(self):
        """pool_axis_tokens should return (1, D_EMB)."""
        T = 10
        tokens = torch.randn(T, D_TOKEN)
        weights = torch.rand(T)
        pooled = self.model.pool_axis_tokens(tokens, weights)
        self.assertEqual(pooled.shape, (1, D_EMB))

    def test_pool_axis_tokens_empty(self):
        """Empty tokens should return zeros."""
        tokens = torch.zeros(0, D_TOKEN)
        weights = torch.zeros(0)
        pooled = self.model.pool_axis_tokens(tokens, weights)
        self.assertEqual(pooled.shape, (1, D_EMB))
        self.assertTrue(torch.allclose(pooled, torch.zeros(1, D_EMB)))

    def test_project_node_motion_shape(self):
        """project_node_motion should return (1, D_EMB)."""
        motion = torch.randn(D_NODE)
        projected = self.model.project_node_motion(motion)
        self.assertEqual(projected.shape, (1, D_EMB))

    def test_weights_not_random(self):
        """PBAI-constant init: weights should not be uniform/zero."""
        # W_V should be identity
        identity = torch.eye(D_EMB)
        self.assertTrue(torch.allclose(self.model.W_V.weight, identity))

        # W_Q should not be identity (Givens-rotated)
        self.assertFalse(torch.allclose(self.model.W_Q.weight, identity))

    def test_performance(self):
        """Forward pass should complete in < 50ms on Pi 5."""
        N = 100  # 100 nodes
        node_embs = torch.randn(N, D_EMB)
        query = torch.randn(1, D_EMB)
        # Warm up
        with torch.no_grad():
            self.model.forward(node_embs, query)
        # Time it
        start = time.time()
        with torch.no_grad():
            for _ in range(10):
                self.model.forward(node_embs, query)
        elapsed = (time.time() - start) / 10
        self.assertLess(elapsed, 0.05, f"Forward pass took {elapsed*1000:.1f}ms")


@unittest.skipUnless(HAS_TORCH, "PyTorch not available")
class TestFeatureExtraction(unittest.TestCase):
    """Test axis token and node motion extraction."""

    def setUp(self):
        self.m = populated_manifold()
        self.intro = Introspector(self.m)

    def test_eligible_nodes(self):
        """Should exclude psychology, bootstrap, Self, non-ACTUAL."""
        nodes = self.intro._eligible_nodes()
        concepts = [n.concept for n in nodes]
        self.assertNotIn("identity", concepts)
        self.assertNotIn("ego", concepts)
        self.assertNotIn("conscience", concepts)
        # Should include our test concepts
        self.assertIn("move_forward", concepts)
        self.assertIn("zombie", concepts)

    def test_extract_axis_tokens_shape(self):
        """Token shape should be (T, 14) where T <= 44."""
        node = self.m.get_node_by_concept("zombie")
        tokens, weights = self.intro._extract_axis_tokens(node)
        self.assertEqual(tokens.shape[1], D_TOKEN)
        self.assertLessEqual(tokens.shape[0], MAX_ORDER_TOKENS)
        self.assertEqual(tokens.shape[0], weights.shape[0])

    def test_extract_axis_tokens_empty_node(self):
        """Node with no axes returns empty tensors."""
        empty_node = Node(concept="empty", theta=1.0, phi=1.0, radius=1.0,
                         heat=K, existence=EXISTENCE_ACTUAL)
        self.m.add_node(empty_node)
        tokens, weights = self.intro._extract_axis_tokens(empty_node)
        self.assertEqual(tokens.shape[0], 0)
        self.assertEqual(weights.shape[0], 0)

    def test_extract_axis_tokens_normalization(self):
        """Token values should be in [0, 1] range (normalized)."""
        node = self.m.get_node_by_concept("zombie")
        tokens, weights = self.intro._extract_axis_tokens(node)
        if tokens.shape[0] > 0:
            # Most features are normalized to [0,1] except existence one-hot is {0,1}
            self.assertTrue((tokens >= -0.01).all(), f"Negative values found: {tokens.min()}")
            self.assertTrue((tokens <= 1.01).all(), f"Values > 1 found: {tokens.max()}")

    def test_extract_node_motion_shape(self):
        """Node motion features should be (10,)."""
        node = self.m.get_node_by_concept("zombie")
        motion = self.intro._extract_node_motion(node)
        self.assertEqual(motion.shape, (D_NODE,))

    def test_extract_node_motion_normalization(self):
        """Node motion values should be in [0, 1] range."""
        node = self.m.get_node_by_concept("move_forward")
        motion = self.intro._extract_node_motion(node)
        self.assertTrue((motion >= -0.01).all())
        self.assertTrue((motion <= 1.01).all())

    def test_pool_node_embedding_shape(self):
        """Node embedding should be (1, 32)."""
        node = self.m.get_node_by_concept("zombie")
        emb = self.intro._pool_node_embedding(node)
        self.assertEqual(emb.shape, (1, D_EMB))

    def test_embed_all_nodes_shape(self):
        """All node embeddings should be (N, 32)."""
        nodes = self.intro._eligible_nodes()
        embs = self.intro._embed_all_nodes(nodes)
        self.assertEqual(embs.shape, (len(nodes), D_EMB))


@unittest.skipUnless(HAS_TORCH, "PyTorch not available")
class TestQueryBuilding(unittest.TestCase):
    """Test query construction from domain context."""

    def setUp(self):
        self.m = populated_manifold()
        self.intro = Introspector(self.m)
        self.nodes = self.intro._eligible_nodes()
        self.embeddings = self.intro._embed_all_nodes(self.nodes)

    def test_query_shape(self):
        """Query should be (1, 32)."""
        hot_nodes = [self.m.get_node_by_concept("zombie")]
        domain_ctx = {"hot_nodes": hot_nodes, "domain_prefix": "ow"}
        query = self.intro._build_query(domain_ctx, self.embeddings, self.nodes)
        self.assertEqual(query.shape, (1, D_EMB))

    def test_query_with_vision_hint(self):
        """Vision features should modify query."""
        hot_nodes = [self.m.get_node_by_concept("zombie")]
        features = [0.1] * D_EMB
        domain_ctx = {
            "hot_nodes": hot_nodes,
            "domain_prefix": "ow",
            "targets": [{"features": features}],
        }
        query_with = self.intro._build_query(domain_ctx, self.embeddings, self.nodes)
        domain_ctx_no = {"hot_nodes": hot_nodes, "domain_prefix": "ow"}
        query_without = self.intro._build_query(domain_ctx_no, self.embeddings, self.nodes)
        # Should differ (vision hint adds to query)
        self.assertFalse(torch.allclose(query_with, query_without))

    def test_query_empty_context(self):
        """Empty context should still produce a query (from Ego+Identity)."""
        domain_ctx = {"hot_nodes": [], "domain_prefix": ""}
        query = self.intro._build_query(domain_ctx, self.embeddings, self.nodes)
        self.assertEqual(query.shape, (1, D_EMB))


@unittest.skipUnless(HAS_TORCH, "PyTorch not available")
class TestRighteousnessGate(unittest.TestCase):
    """Test the righteousness gate."""

    def setUp(self):
        self.m = populated_manifold()
        self.intro = Introspector(self.m)

    def test_gate_preserves_shape(self):
        """Gated scores should have same shape as input."""
        nodes = self.intro._eligible_nodes()
        scores = torch.rand(len(nodes), 1)
        gated = self.intro._righteousness_gate(scores, nodes)
        self.assertEqual(gated.shape, scores.shape)

    def test_gate_reduces_scores(self):
        """Gate should reduce or preserve scores (never increase beyond input)."""
        nodes = self.intro._eligible_nodes()
        scores = torch.ones(len(nodes), 1)
        gated = self.intro._righteousness_gate(scores, nodes)
        self.assertTrue((gated <= scores + 0.001).all())


class TestSuggest(unittest.TestCase):
    """Test suggest() with and without transformer."""

    def setUp(self):
        self.m = populated_manifold()
        self.intro = Introspector(self.m)

    def test_suggest_returns_concepts(self):
        """suggest() should return a list of concept strings."""
        domain_ctx = {
            "hot_nodes": [self.m.get_node_by_concept("zombie")],
            "domain_prefix": "ow",
        }
        result = self.intro.suggest(domain_ctx)
        if result is not None:
            self.assertIsInstance(result, list)
            for c in result:
                self.assertIsInstance(c, str)

    def test_suggest_domain_filter(self):
        """Suggestions should respect domain prefix filtering."""
        domain_ctx = {
            "hot_nodes": [self.m.get_node_by_concept("zombie")],
            "domain_prefix": "ow",
        }
        result = self.intro.suggest(domain_ctx)
        if result is not None:
            for c in result:
                # Should not contain foreign domain state keys
                if '_' in c and any(ch.isdigit() for ch in c):
                    self.assertTrue(c.startswith("ow"),
                                   f"Foreign domain concept leaked: {c}")

    def test_suggest_gates_on_ego(self):
        """suggest() should return None when Ego can't afford it."""
        self.m.ego_node.heat = PSYCHOLOGY_MIN_HEAT  # Too low
        domain_ctx = {"hot_nodes": [], "domain_prefix": ""}
        result = self.intro.suggest(domain_ctx)
        self.assertIsNone(result)

    def test_suggest_gates_on_conscience(self):
        """suggest() should return None when Conscience is dormant."""
        self.m.conscience_node.existence = "dormant"
        domain_ctx = {"hot_nodes": [], "domain_prefix": ""}
        result = self.intro.suggest(domain_ctx)
        self.assertIsNone(result)

    def test_suggest_pays_ego_cost(self):
        """suggest() should deduct COST_EVALUATE from Ego."""
        initial_heat = self.m.ego_node.heat
        domain_ctx = {
            "hot_nodes": [self.m.get_node_by_concept("zombie")],
            "domain_prefix": "ow",
        }
        result = self.intro.suggest(domain_ctx)
        if result is not None:
            self.assertLess(self.m.ego_node.heat, initial_heat)


class TestSuggestFallback(unittest.TestCase):
    """Test fallback suggest (no transformer)."""

    def setUp(self):
        self.m = populated_manifold()
        self.intro = Introspector(self.m)
        # Force fallback
        self.intro._transformer = None

    def test_fallback_returns_concepts(self):
        """Fallback should return concept list."""
        domain_ctx = {
            "hot_nodes": [self.m.get_node_by_concept("zombie"),
                          self.m.get_node_by_concept("pig")],
            "domain_prefix": "ow",
        }
        result = self.intro.suggest(domain_ctx)
        if result is not None:
            self.assertIsInstance(result, list)
            for c in result:
                self.assertIsInstance(c, str)


class TestGetWeightBoosts(unittest.TestCase):
    """Test get_weight_boosts() conversion of suggestions to action weights."""

    def setUp(self):
        self.m = populated_manifold()
        self.intro = Introspector(self.m)

    def test_direct_match(self):
        """Direct match (concept == action) should give boost 4.0."""
        suggestions = ["move_forward", "jump"]
        actions = ["move_forward", "jump", "attack", "look_left"]
        boosts = self.intro.get_weight_boosts(suggestions, actions)
        self.assertAlmostEqual(boosts.get("move_forward", 0), 4.0)
        self.assertAlmostEqual(boosts.get("jump", 0), 4.0)

    def test_partial_match(self):
        """Partial match should give boost 2.5."""
        suggestions = ["forward"]
        actions = ["move_forward", "sprint_forward", "jump"]
        boosts = self.intro.get_weight_boosts(suggestions, actions)
        self.assertAlmostEqual(boosts.get("move_forward", 0), 2.5)
        self.assertAlmostEqual(boosts.get("sprint_forward", 0), 2.5)

    def test_no_match(self):
        """No match should result in empty boosts."""
        suggestions = ["ow_unknown_h20_e0p0"]
        actions = ["move_forward", "jump"]
        boosts = self.intro.get_weight_boosts(suggestions, actions)
        # State key shouldn't directly match any action
        self.assertNotIn("move_forward", boosts)
        self.assertNotIn("jump", boosts)

    def test_returns_dict(self):
        """get_weight_boosts should return Dict[str, float]."""
        boosts = self.intro.get_weight_boosts(["zombie"], ["attack", "jump"])
        self.assertIsInstance(boosts, dict)
        for k, v in boosts.items():
            self.assertIsInstance(k, str)
            self.assertIsInstance(v, float)


class TestShouldThink(unittest.TestCase):
    """Test the should_think() gate."""

    def setUp(self):
        self.m = populated_manifold()
        self.intro = Introspector(self.m)

    def test_should_think_normal(self):
        """With pumped psychology, should_think() is True."""
        self.assertTrue(self.intro.should_think())

    def test_ego_too_low(self):
        """Ego below threshold should gate off."""
        self.m.ego_node.heat = PSYCHOLOGY_MIN_HEAT
        self.assertFalse(self.intro.should_think())

    def test_conscience_dormant(self):
        """Dormant Conscience should gate off."""
        self.m.conscience_node.existence = "dormant"
        self.assertFalse(self.intro.should_think())


class TestSimulation(unittest.TestCase):
    """Test simulation is unchanged (regression)."""

    def setUp(self):
        self.m = populated_manifold()
        self.intro = Introspector(self.m)

    def test_simulate_returns_results(self):
        results = self.intro.simulate(["move_forward", "jump"])
        self.assertEqual(len(results), 2)
        for r in results:
            self.assertIsInstance(r, SimulationResult)

    def test_simulate_cube_projection(self):
        """Results should have valid cube coordinates."""
        results = self.intro.simulate(["zombie"])
        r = results[0]
        self.assertTrue(-1.0 <= r.cube_x <= 1.0)
        self.assertTrue(-1.0 <= r.cube_y <= 1.0)
        self.assertTrue(-1.0 <= r.cube_tau <= 1.0)

    def test_to_context(self):
        """to_context should package results as dict."""
        results = self.intro.simulate(["move_forward", "jump"])
        ctx = self.intro.to_context(results)
        self.assertIn("introspection_count", ctx)
        self.assertIn("sim_best_option", ctx)


class TestSTM(unittest.TestCase):
    """Test short-term memory is unchanged."""

    def setUp(self):
        self.m = populated_manifold()
        self.intro = Introspector(self.m)

    def test_stm_summary(self):
        summary = self.intro.get_stm_summary()
        self.assertIn("stm_entries", summary)
        self.assertIn("stm_capacity", summary)
        self.assertEqual(summary["stm_capacity"], 12)

    def test_stm_pattern_needs_data(self):
        """Pattern detection requires >= 3 entries."""
        self.assertIsNone(self.intro.get_stm_pattern())


@unittest.skipUnless(HAS_TORCH, "PyTorch not available")
class TestBaseMotionEmbeddings(unittest.TestCase):
    """Test base motion token embedding and scoring."""

    def setUp(self):
        self.m = fresh_manifold()
        # Pump psychology for should_think
        if self.m.ego_node:
            self.m.ego_node.heat = 50.0
        if self.m.conscience_node:
            self.m.conscience_node.heat = 5.0
        self.intro = Introspector(self.m)

    def test_base_motions_in_eligible(self):
        """Base motion nodes (bm_*) should be included in _eligible_nodes()."""
        nodes = self.intro._eligible_nodes()
        concepts = [n.concept for n in nodes]
        # At least some bm_ concepts should be present
        bm_in_eligible = [c for c in concepts if c.startswith(BASE_MOTION_PREFIX)]
        self.assertEqual(len(bm_in_eligible), len(ALL_BASE_MOTIONS),
                         f"Expected {len(ALL_BASE_MOTIONS)} base motions in eligible, got {len(bm_in_eligible)}")

    def test_bootstraps_still_excluded(self):
        """Bootstrap nodes should still be excluded from eligible."""
        nodes = self.intro._eligible_nodes()
        concepts = [n.concept for n in nodes]
        bootstrap_in = [c for c in concepts if c.startswith('bootstrap')]
        self.assertEqual(len(bootstrap_in), 0)

    def test_get_base_motion_embeddings_shape(self):
        """_get_base_motion_embeddings should return (N, D_EMB) tensor."""
        embs = self.intro._get_base_motion_embeddings()
        self.assertIsNotNone(embs)
        self.assertEqual(embs.shape, (len(ALL_BASE_MOTIONS), D_EMB))

    def test_get_base_motion_embeddings_cached(self):
        """Second call should return cached result."""
        embs1 = self.intro._get_base_motion_embeddings()
        embs2 = self.intro._get_base_motion_embeddings()
        self.assertTrue(torch.equal(embs1, embs2))
        # Check it's the same object (cached)
        self.assertIs(embs1, embs2)

    def test_score_against_base_motions_returns_20(self):
        """score_against_base_motions should return N scores."""
        node = self.m.get_node_by_concept("bm_explore")
        self.assertIsNotNone(node)
        scores = self.intro.score_against_base_motions(node)
        self.assertEqual(len(scores), len(ALL_BASE_MOTIONS))
        for concept, score in scores.items():
            self.assertTrue(concept.startswith(BASE_MOTION_PREFIX))
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_score_against_base_motions_nonzero(self):
        """Scores should not all be zero."""
        node = self.m.get_node_by_concept("bm_see")
        scores = self.intro.score_against_base_motions(node)
        total = sum(scores.values())
        self.assertGreater(total, 0.0)


class TestBaseMotionWeightBoosts(unittest.TestCase):
    """Test base motion → action weight boosts."""

    def setUp(self):
        self.m = populated_manifold()
        self.intro = Introspector(self.m)

    def test_bm_explore_boosts_move_forward(self):
        """bm_explore should boost move_forward at 3.0."""
        suggestions = ["bm_explore"]
        actions = ["move_forward", "jump", "attack", "look_left"]
        boosts = self.intro.get_weight_boosts(suggestions, actions)
        self.assertAlmostEqual(boosts.get("move_forward", 0), 3.0)

    def test_bm_see_boosts_look_actions(self):
        """bm_see should boost look_left and look_right at 3.0."""
        suggestions = ["bm_see"]
        actions = ["move_forward", "look_left", "look_right", "attack"]
        boosts = self.intro.get_weight_boosts(suggestions, actions)
        self.assertAlmostEqual(boosts.get("look_left", 0), 3.0)
        self.assertAlmostEqual(boosts.get("look_right", 0), 3.0)

    def test_bm_mixed_with_regular(self):
        """Mix of bm_ and regular suggestions should both produce boosts."""
        suggestions = ["bm_explore", "move_forward"]
        actions = ["move_forward", "jump", "look_left"]
        boosts = self.intro.get_weight_boosts(suggestions, actions)
        # move_forward gets 4.0 (direct) which is > 3.0 (bm), so 4.0 wins
        self.assertAlmostEqual(boosts.get("move_forward", 0), 4.0)

    def test_bm_unmapped_action(self):
        """bm_ suggestion shouldn't boost actions not in its map."""
        suggestions = ["bm_take"]
        actions = ["move_forward", "jump"]
        boosts = self.intro.get_weight_boosts(suggestions, actions)
        # bm_take maps to attack/use, not move_forward/jump
        self.assertNotIn("move_forward", boosts)
        self.assertNotIn("jump", boosts)


if __name__ == '__main__':
    unittest.main(verbosity=2)
