"""
PBAI Introspector — Transformer Attention over Manifold Axes

Two blenders mirror each other:
    PC Blender (vision_transformer.py): Processes visual input → peaks with features
    Pi Blender (this file): Processes manifold axes → relevance scores for decision

Architecture:
    Each node's axes are its TOKENS (up to 44 = MAX_ORDER_TOKENS).
    Level 1: Intra-node weighted pool (axes → 32-dim node embedding)
    Level 2: Cross-node attention (query attends over all node embeddings)
    Temperature = K (attention IS heat, same as PC blender)

    No backprop — the manifold's thermal dynamics ARE the learning.
    Fixed weights initialized from PBAI constants (golden angle, motion thresholds).

    Introspector sits between perception and decision.
    It enriches options with transformer-scored relevance context
    before the beta->delta->Gamma->alpha->zeta pipeline runs.
"""

import math
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

from .nodes import Node
from .hypersphere import SpherePosition, angular_distance, place_node_near, relationship_strength
from .color_cube import CubePosition, evaluate_righteousness as cube_evaluate_R
from .node_constants import (
    K, PHI, INV_PHI,
    COST_EVALUATE, PSYCHOLOGY_MIN_HEAT,
    CONFIDENCE_EXPLOIT_THRESHOLD,
    EXISTENCE_ACTUAL,
    MAX_ORDER_TOKENS,
    THRESHOLD_HEAT, THRESHOLD_POLARITY, THRESHOLD_EXISTENCE,
    THRESHOLD_RIGHTEOUSNESS, THRESHOLD_ORDER, THRESHOLD_MOVEMENT,
    BASE_MOTION_PREFIX, ALL_BASE_MOTIONS,
)

logger = logging.getLogger(__name__)

# Transformer dimensions
D_TOKEN = 14    # Per-axis token: 6 motion functions of target + axis properties
D_NODE = 10     # Per-node: its own 6 motion functions in cube coords
D_EMB = 32      # Embedding dimension — matches PC's feature_head output
N_HEADS = 2     # Attention heads

# Try to import torch — graceful fallback if unavailable
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    torch = None
    nn = None
    F = None


# ═══════════════════════════════════════════════════════════════════════════════
# MANIFOLD ATTENTION — HeatAttention pattern from the Pi blender
# ═══════════════════════════════════════════════════════════════════════════════

if _HAS_TORCH:
    class ManifoldAttention(nn.Module):
        """Transformer attention over manifold node embeddings.

        Mirrors the PC's HeatAttention — attention weights ARE the heat signal.
        Uses K as temperature, same as vision_transformer.py.

        Level 1: Project each node's axis-tokens to D_EMB, weighted pool
        Level 2: Cross-node attention with query from state context

        Total params: ~3,872 (~15 KB) — trivial for Pi 5.
        """

        def __init__(self, d_token: int = D_TOKEN, d_node: int = D_NODE,
                     d_emb: int = D_EMB, n_heads: int = N_HEADS):
            super().__init__()
            self.d_emb = d_emb
            self.n_heads = n_heads
            self.d_head = d_emb // n_heads

            # Level 1: Axis token → embedding
            self.W_token_proj = nn.Linear(d_token, d_emb, bias=False)
            # Node's own motion functions → embedding
            self.W_node_proj = nn.Linear(d_node, d_emb, bias=False)

            # Level 2: Cross-node attention
            self.W_Q = nn.Linear(d_emb, d_emb, bias=False)
            self.W_K = nn.Linear(d_emb, d_emb, bias=False)
            self.W_V = nn.Linear(d_emb, d_emb, bias=False)
            self.W_out = nn.Linear(d_emb, 1, bias=False)

            # Attention temperature = K (registered buffer, not a parameter)
            self.register_buffer('attention_temp', torch.tensor(K, dtype=torch.float32))

            # Scale factor for attention
            self.scale = 1.0 / math.sqrt(self.d_head)

            # Initialize with PBAI constants
            self._init_weights()

        @torch.no_grad()
        def _init_weights(self):
            """Initialize weights from PBAI constants (not random).

            W_token_proj, W_node_proj: golden angle spiral weighted by motion thresholds
            W_Q, W_K: Givens-rotated identity at golden angle spacing
            W_V: identity (pass-through initially)
            W_out: motion function thresholds distributed across D_EMB dims
            """
            golden_angle = 2 * math.pi * INV_PHI
            thresholds = [
                THRESHOLD_HEAT, THRESHOLD_POLARITY, THRESHOLD_EXISTENCE,
                THRESHOLD_RIGHTEOUSNESS, THRESHOLD_ORDER, THRESHOLD_MOVEMENT,
            ]

            # W_token_proj: golden angle spiral weighted by 1/phi^n thresholds
            w = torch.zeros_like(self.W_token_proj.weight)
            for i in range(min(self.d_emb, D_TOKEN)):
                angle = golden_angle * i
                threshold = thresholds[i % len(thresholds)]
                for j in range(min(self.d_emb, D_TOKEN)):
                    w[i, j] = threshold * math.cos(angle * (j + 1))
            # Scale to reasonable magnitude
            w = w / (w.norm() + 1e-8) * math.sqrt(self.d_emb)
            self.W_token_proj.weight.copy_(w[:self.d_emb, :D_TOKEN])

            # W_node_proj: same pattern
            w_node = torch.zeros_like(self.W_node_proj.weight)
            for i in range(min(self.d_emb, D_NODE)):
                angle = golden_angle * i
                threshold = thresholds[i % len(thresholds)]
                for j in range(min(self.d_emb, D_NODE)):
                    w_node[i, j] = threshold * math.cos(angle * (j + 1))
            w_node = w_node / (w_node.norm() + 1e-8) * math.sqrt(self.d_emb)
            self.W_node_proj.weight.copy_(w_node[:self.d_emb, :D_NODE])

            # W_Q, W_K: Givens-rotated identity at golden angle spacing
            for W in [self.W_Q, self.W_K]:
                w_qk = torch.eye(self.d_emb)
                for i in range(0, self.d_emb - 1, 2):
                    angle = golden_angle * (i // 2)
                    c, s = math.cos(angle), math.sin(angle)
                    w_qk[i, i] = c
                    w_qk[i, i + 1] = -s
                    w_qk[i + 1, i] = s
                    w_qk[i + 1, i + 1] = c
                W.weight.copy_(w_qk)

            # W_V: identity (pass-through initially)
            self.W_V.weight.copy_(torch.eye(self.d_emb))

            # W_out: motion thresholds distributed across D_EMB dims
            w_out = torch.zeros(1, self.d_emb)
            for i in range(self.d_emb):
                w_out[0, i] = thresholds[i % len(thresholds)] / K
            self.W_out.weight.copy_(w_out)

        def forward(self, node_embeddings: 'torch.Tensor',
                    query: 'torch.Tensor') -> 'torch.Tensor':
            """Cross-node attention. Returns (N, 1) relevance scores.

            Args:
                node_embeddings: (N, D_EMB) — all eligible node embeddings
                query: (1, D_EMB) — state context query

            Returns:
                (N, 1) attention-weighted relevance scores
            """
            N = node_embeddings.shape[0]
            if N == 0:
                return torch.zeros(0, 1)

            # Project Q, K, V
            Q = self.W_Q(query)    # (1, D_EMB)
            K_mat = self.W_K(node_embeddings)  # (N, D_EMB)
            V = self.W_V(node_embeddings)      # (N, D_EMB)

            # Multi-head reshape: (batch, heads, seq, d_head)
            Q = Q.view(1, self.n_heads, self.d_head).unsqueeze(0)       # (1, 1, heads, d_head)
            K_mat = K_mat.view(N, self.n_heads, self.d_head).unsqueeze(0)  # (1, N, heads, d_head)
            V = V.view(N, self.n_heads, self.d_head).unsqueeze(0)          # (1, N, heads, d_head)

            # Transpose for attention: (1, heads, seq, d_head)
            Q = Q.permute(0, 2, 1, 3)      # (1, heads, 1, d_head)
            K_mat = K_mat.permute(0, 2, 1, 3)  # (1, heads, N, d_head)
            V = V.permute(0, 2, 1, 3)          # (1, heads, N, d_head)

            # Attention: softmax(QK^T * scale / K_temp)
            attn = torch.matmul(Q, K_mat.transpose(-2, -1))  # (1, heads, 1, N)
            attn = attn * self.scale / self.attention_temp
            attn = F.softmax(attn, dim=-1)

            # Weighted values
            out = torch.matmul(attn, V)  # (1, heads, 1, d_head)
            # But we want per-node scores, not aggregated output
            # Use attention weights directly as relevance
            attn_weights = attn.squeeze(0).mean(dim=0).squeeze(0)  # (N,)

            # Also compute per-node output scores via W_out
            node_scores = self.W_out(node_embeddings)  # (N, 1)

            # Combine: attention weight * output score
            combined = attn_weights.unsqueeze(1) * torch.sigmoid(node_scores)

            return combined  # (N, 1)

        def pool_axis_tokens(self, token_features: 'torch.Tensor',
                             weights: 'torch.Tensor') -> 'torch.Tensor':
            """Weighted pool of axis tokens into single node embedding.

            Args:
                token_features: (T, D_TOKEN) — up to 44 axis tokens
                weights: (T,) — traversal counts as weights

            Returns:
                (1, D_EMB) — single node embedding
            """
            if token_features.shape[0] == 0:
                return torch.zeros(1, self.d_emb)

            projected = self.W_token_proj(token_features)  # (T, D_EMB)

            # Normalize weights
            w = weights / (weights.sum() + 1e-8)
            w = w.unsqueeze(1)  # (T, 1)

            pooled = (projected * w).sum(dim=0, keepdim=True)  # (1, D_EMB)
            return pooled

        def project_node_motion(self, node_features: 'torch.Tensor') -> 'torch.Tensor':
            """Project node's own motion functions to embedding space.

            Args:
                node_features: (D_NODE,) — node's 6 motion functions

            Returns:
                (1, D_EMB)
            """
            return self.W_node_proj(node_features.unsqueeze(0))  # (1, D_EMB)


# ═══════════════════════════════════════════════════════════════════════════════
# SIMULATION RESULT (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SimulationResult:
    """Result of simulating one option through cube projection.

    Contains the cube-native coordinates and derived quantities
    that enrich decision context without copying the manifold.
    """
    option: str
    # Hypersphere position
    theta: float = 0.0
    phi: float = 0.0
    # Cube projection
    cube_x: float = 0.0       # Blue(-1) / Yellow(+1)
    cube_y: float = 0.0       # Red(-1) / Green(+1)
    cube_tau: float = 0.0     # Past(-) / Future(+)
    # Derived
    heat_magnitude: float = 0.0
    quadrant: str = "Q1"
    righteousness: float = 1.0  # R via Conscience cos projection
    # Conscience evaluation
    conscience_score: float = 0.0  # How righteous Conscience judges this

    def to_dict(self) -> dict:
        return {
            "option": self.option,
            "theta": self.theta, "phi": self.phi,
            "cube_x": self.cube_x, "cube_y": self.cube_y, "cube_tau": self.cube_tau,
            "heat_magnitude": self.heat_magnitude,
            "quadrant": self.quadrant,
            "righteousness": self.righteousness,
            "conscience_score": self.conscience_score,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# INTROSPECTOR
# ═══════════════════════════════════════════════════════════════════════════════

class Introspector:
    """
    Transformer-based introspection engine.

    Reads each node's axes as tokens (up to 44 per node), embeds them via
    ManifoldAttention, then uses cross-node attention to find relevant concepts
    for the current state context.

    Falls back to heat-sorted domain concepts if PyTorch is unavailable.

    Usage:
        introspector = Introspector(manifold)
        if introspector.should_think():
            suggestions = introspector.suggest(domain_ctx)
            boosts = introspector.get_weight_boosts(suggestions, actions)
    """

    def __init__(self, manifold):
        self.manifold = manifold

        # Short-term simulation memory — recent results for pattern detection
        self._stm: List[Dict[str, Any]] = []
        self._stm_capacity: int = 48  # Extended for deliberation window

        # Base motion embedding cache (invalidated on manifold reload)
        self._bm_cache: Optional['torch.Tensor'] = None
        self._bm_concepts: List[str] = []

        # Transformer (None if torch unavailable)
        self._transformer = None
        if _HAS_TORCH:
            try:
                self._transformer = ManifoldAttention(D_TOKEN, D_NODE, D_EMB, N_HEADS)
                self._transformer.eval()  # Inference only, no training
                logger.info(f"ManifoldAttention initialized: "
                            f"{sum(p.numel() for p in self._transformer.parameters())} params")
            except Exception as e:
                logger.warning(f"ManifoldAttention init failed, using fallback: {e}")
                self._transformer = None

    # ═══════════════════════════════════════════════════════════════════════════
    # GATING — Should we think about this?
    # ═══════════════════════════════════════════════════════════════════════════

    def should_think(self) -> bool:
        """
        Gate introspection on Ego energy + Conscience existence.

        Ego must have energy to think (above min heat + eval cost).
        Conscience must be actual (connected, conscious) to validate.

        NOTE: In mature manifolds, Conscience heat stays near PSYCHOLOGY_MIN_HEAT
        because the clock tick drains it through axis traversal. We only check
        Conscience existence (ACTUAL = not dormant), not heat level. The
        Introspector pays from Ego, not Conscience.
        """
        ego = self.manifold.ego_node
        conscience = self.manifold.conscience_node

        # Both psychology nodes must exist and be actual
        if not ego or ego.existence != EXISTENCE_ACTUAL:
            return False
        if not conscience or conscience.existence != EXISTENCE_ACTUAL:
            return False

        # Ego must have energy to afford evaluation
        if ego.heat < COST_EVALUATE + PSYCHOLOGY_MIN_HEAT:
            return False

        return True

    # ═══════════════════════════════════════════════════════════════════════════
    # BASE MOTION TOKENS — Cognitive vocabulary embedding + scoring
    # ═══════════════════════════════════════════════════════════════════════════

    def _get_base_motion_embeddings(self) -> Optional['torch.Tensor']:
        """Get (M, D_EMB) embeddings for all base motion nodes.

        Caches result — invalidate by setting self._bm_cache = None.

        Returns:
            (M, D_EMB) tensor where M <= 20 (only nodes present in manifold),
            or None if torch unavailable or no base motions found.
        """
        if not _HAS_TORCH or self._transformer is None:
            return None

        if self._bm_cache is not None:
            return self._bm_cache

        bm_nodes = []
        bm_concepts = []
        for verb in ALL_BASE_MOTIONS:
            concept = f"{BASE_MOTION_PREFIX}{verb}"
            node = self.manifold.get_node_by_concept(concept)
            if node and node.existence == EXISTENCE_ACTUAL:
                bm_nodes.append(node)
                bm_concepts.append(concept)

        if not bm_nodes:
            return None

        with torch.no_grad():
            self._bm_cache = self._embed_all_nodes(bm_nodes)
            self._bm_concepts = bm_concepts

        return self._bm_cache

    def score_against_base_motions(self, node: Node) -> Dict[str, float]:
        """Compute cosine similarity between a node and all base motions.

        Returns dict: {bm_concept: score} with scores in [0, 1].
        This is the node's "motion decomposition" — which verbs it activates.
        """
        if not _HAS_TORCH or self._transformer is None:
            return {}

        bm_embs = self._get_base_motion_embeddings()
        if bm_embs is None or bm_embs.shape[0] == 0:
            return {}

        with torch.no_grad():
            node_emb = self._pool_node_embedding(node)  # (1, D_EMB)
            # Cosine similarity
            node_norm = torch.nn.functional.normalize(node_emb, dim=1)  # (1, D_EMB)
            bm_norm = torch.nn.functional.normalize(bm_embs, dim=1)    # (M, D_EMB)
            sims = torch.matmul(node_norm, bm_norm.T).squeeze(0)       # (M,)
            # Clamp to [0, 1]
            sims = sims.clamp(0.0, 1.0)

        result = {}
        for i, concept in enumerate(self._bm_concepts):
            result[concept] = sims[i].item()
        return result

    # ═══════════════════════════════════════════════════════════════════════════
    # SIMULATION — Project options through cube (unchanged)
    # ═══════════════════════════════════════════════════════════════════════════

    def simulate(self, options: List[str], state_key: str = "") -> List[SimulationResult]:
        """
        Simulate each option by projecting onto hypersphere → cube.

        For each option:
        1. Find or estimate angular position on hypersphere
        2. Project to cube coordinates (x, y, tau)
        3. Read heat magnitude and quadrant
        4. Evaluate Righteousness via Conscience (cos projection)
        5. Package as SimulationResult

        This is LIGHTWEIGHT — no deepcopy, no manifold mutation.
        """
        results = []

        ref_node = None
        if state_key:
            ref_node = self.manifold.get_node_by_concept(state_key)
        if not ref_node and self.manifold.identity_node:
            ref_node = self.manifold.identity_node

        for i, option in enumerate(options):
            result = self._simulate_option(option, i, len(options), ref_node)
            results.append(result)

        if self.manifold.ego_node:
            self.manifold.ego_node.spend_heat(COST_EVALUATE, minimum=PSYCHOLOGY_MIN_HEAT)

        self._record_stm(state_key, results)
        return results

    def _simulate_option(self, option: str, index: int, total: int,
                         ref_node: Optional[Node]) -> SimulationResult:
        """Simulate a single option through cube projection."""
        option_node = self.manifold.get_node_by_concept(option)

        if option_node:
            theta = option_node.theta
            phi = option_node.phi
        elif ref_node:
            golden_angle = 2 * math.pi * INV_PHI
            theta = ref_node.theta
            phi = (ref_node.phi + index * golden_angle) % (2 * math.pi)
        else:
            theta = math.pi / 2
            phi = (2 * math.pi * index) / max(total, 1)

        cube_x = math.sin(theta) * math.cos(phi)
        cube_y = math.sin(theta) * math.sin(phi)
        cube_tau = math.cos(theta)
        heat_magnitude = math.sqrt(cube_x ** 2 + cube_y ** 2)

        if cube_x >= 0 and cube_y >= 0:
            quadrant = "Q1"
        elif cube_x < 0 and cube_y >= 0:
            quadrant = "Q2"
        elif cube_x < 0 and cube_y < 0:
            quadrant = "Q3"
        else:
            quadrant = "Q4"

        righteousness = 1.0
        if ref_node:
            ref_sp = SpherePosition(theta=ref_node.theta, phi=ref_node.phi)
            opt_sp = SpherePosition(theta=theta, phi=phi)
            angle = angular_distance(ref_sp, opt_sp)
            righteousness = 1.0 - math.cos(angle)

        cube_pos = CubePosition(x=cube_x, y=cube_y, tau=cube_tau)
        conscience_score = 1.0 - min(cube_evaluate_R(cube_pos), 1.0)

        return SimulationResult(
            option=option,
            theta=theta, phi=phi,
            cube_x=cube_x, cube_y=cube_y, cube_tau=cube_tau,
            heat_magnitude=heat_magnitude,
            quadrant=quadrant,
            righteousness=righteousness,
            conscience_score=conscience_score,
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # CONTEXT — Package results for decision pipeline (unchanged)
    # ═══════════════════════════════════════════════════════════════════════════

    def to_context(self, results: List[SimulationResult]) -> Dict[str, Any]:
        """Package simulation results as decision context."""
        context = {
            "introspection_count": len(results),
            "simulated": True,
        }

        for result in results:
            prefix = f"sim_{result.option}"
            context[f"{prefix}_quadrant"] = result.quadrant
            context[f"{prefix}_heat"] = result.heat_magnitude
            context[f"{prefix}_R"] = result.righteousness
            context[f"{prefix}_conscience"] = result.conscience_score

        if results:
            best = max(results, key=lambda r: r.conscience_score)
            worst = min(results, key=lambda r: r.conscience_score)
            context["sim_best_option"] = best.option
            context["sim_best_conscience"] = best.conscience_score
            context["sim_worst_option"] = worst.option
            context["sim_spread"] = best.conscience_score - worst.conscience_score
            quadrants = [r.quadrant for r in results]
            context["sim_quadrant_diversity"] = len(set(quadrants))

        return context

    def rank_by_conscience(self, results: List[SimulationResult]) -> List[SimulationResult]:
        """Rank simulation results by Conscience score (highest first)."""
        return sorted(results, key=lambda r: r.conscience_score, reverse=True)

    def rank_by_heat(self, results: List[SimulationResult]) -> List[SimulationResult]:
        """Rank simulation results by heat magnitude (highest first)."""
        return sorted(results, key=lambda r: r.heat_magnitude, reverse=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # SHORT-TERM MEMORY (unchanged)
    # ═══════════════════════════════════════════════════════════════════════════

    def _record_stm(self, state_key: str, results: List[SimulationResult]) -> None:
        """Record simulation results to short-term memory."""
        self._stm.append({
            "state_key": state_key,
            "results": [r.to_dict() for r in results],
            "t_K": self.manifold.get_time() if self.manifold else 0,
        })
        if len(self._stm) > self._stm_capacity:
            self._stm.pop(0)

    def get_stm_pattern(self) -> Optional[str]:
        """Detect patterns in recent simulations."""
        if len(self._stm) < 3:
            return None

        recent_quadrants = []
        for entry in self._stm[-3:]:
            results = entry["results"]
            if results:
                best = max(results, key=lambda r: r.get("conscience_score", 0))
                recent_quadrants.append(best.get("quadrant", ""))

        if len(set(recent_quadrants)) == 1 and recent_quadrants[0]:
            return f"quadrant_preference:{recent_quadrants[0]}"

        return None

    def get_stm_summary(self) -> Dict[str, Any]:
        """Get summary of recent simulation memory."""
        return {
            "stm_entries": len(self._stm),
            "stm_capacity": self._stm_capacity,
            "stm_pattern": self.get_stm_pattern(),
        }

    # ═══════════════════════════════════════════════════════════════════════════
    # TRANSFORMER FEATURE EXTRACTION — Cube-mapped axis tokens
    # ═══════════════════════════════════════════════════════════════════════════

    def _eligible_nodes(self) -> List[Node]:
        """Get all nodes eligible for transformer attention.

        Skips psychology nodes, bootstraps, Self (inf heat), and non-ACTUAL.
        """
        eligible = []
        for node in self.manifold.nodes.values():
            if node.concept in ('identity', 'ego', 'conscience'):
                continue
            if node.concept.startswith('bootstrap'):
                continue
            if node.heat == float('inf'):
                continue
            if node.existence != EXISTENCE_ACTUAL:
                continue
            eligible.append(node)
        return eligible

    def _extract_axis_tokens(self, node: Node) -> Tuple['torch.Tensor', 'torch.Tensor']:
        """Extract axis tokens from a node as a (T, D_TOKEN) tensor + weights.

        Each axis is one token. Features are the 6 motion functions of the
        axis's target node, plus axis-specific properties.

        Token layout (14 dims):
            0: target.heat_magnitude / sqrt(2)
            1: (target.cube_x + 1) / 2
            2: (target.cube_y + 1) / 2
            3: (target.polarity + 1) / 2
            4-7: target.existence one-hot (potential/actual/dormant/archived)
            8: min(target.R, 2) / 2
            9: log(1 + target.order) / log(1 + max_order)
            10: (target.cube_tau + 1) / 2
            11: (axis.polarity + 1) / 2
            12: log(1 + axis.traversal_count) / log(1 + max_tc)
            13: 1.0 if axis.is_proper else 0.0

        Returns up to MAX_ORDER_TOKENS tokens. For nodes with >44 axes
        (loaded from mature manifold), takes top 44 by traversal_count.
        """
        axes = list(node.frame.axes.values())
        if not axes:
            return torch.zeros(0, D_TOKEN), torch.zeros(0)

        # Cap at 44 tokens — take strongest if over limit
        if len(axes) > MAX_ORDER_TOKENS:
            axes = sorted(axes, key=lambda a: a.traversal_count, reverse=True)[:MAX_ORDER_TOKENS]

        # Compute max values for normalization
        max_order = max((n.order for n in self.manifold.nodes.values()), default=1)
        max_tc = max((a.traversal_count for a in axes), default=1)

        existence_map = {'potential': 0, 'actual': 1, 'dormant': 2, 'archived': 3}

        tokens = []
        weights = []
        for axis in axes:
            target = self.manifold.get_node(axis.target_id)
            if target is None:
                # Target not found — use zeros
                tokens.append(torch.zeros(D_TOKEN))
                weights.append(float(axis.traversal_count))
                continue

            # 6 motion functions of the target
            t_heat_mag = target.heat_magnitude / math.sqrt(2)
            t_cube_x = (target.cube_x + 1.0) / 2.0
            t_cube_y = (target.cube_y + 1.0) / 2.0
            t_polarity = (target.polarity + 1.0) / 2.0

            # Existence one-hot
            exist_idx = existence_map.get(target.existence, 0)
            exist_oh = [0.0, 0.0, 0.0, 0.0]
            exist_oh[exist_idx] = 1.0

            t_R = min(target.righteousness, 2.0) / 2.0
            t_order = math.log(1 + target.order) / math.log(1 + max(max_order, 1)) if max_order > 0 else 0.0
            t_tau = (target.cube_tau + 1.0) / 2.0

            # Axis properties
            a_pol = (axis.polarity + 1.0) / 2.0
            a_tc = math.log(1 + axis.traversal_count) / math.log(1 + max(max_tc, 1)) if max_tc > 0 else 0.0
            a_proper = 1.0 if axis.is_proper else 0.0

            token = torch.tensor([
                t_heat_mag, t_cube_x, t_cube_y, t_polarity,
                exist_oh[0], exist_oh[1], exist_oh[2], exist_oh[3],
                t_R, t_order, t_tau,
                a_pol, a_tc, a_proper,
            ], dtype=torch.float32)

            tokens.append(token)
            weights.append(float(axis.traversal_count))

        return torch.stack(tokens), torch.tensor(weights, dtype=torch.float32)

    def _extract_node_motion(self, node: Node) -> 'torch.Tensor':
        """Extract a node's own 6 motion functions as (D_NODE,) tensor.

        Layout (10 dims):
            0: heat_magnitude / sqrt(2)
            1: (cube_x + 1) / 2
            2: (cube_y + 1) / 2
            3: (polarity + 1) / 2
            4-7: existence one-hot
            8: min(R, 2) / 2
            9: (cube_tau + 1) / 2
        """
        existence_map = {'potential': 0, 'actual': 1, 'dormant': 2, 'archived': 3}
        exist_idx = existence_map.get(node.existence, 0)
        exist_oh = [0.0, 0.0, 0.0, 0.0]
        exist_oh[exist_idx] = 1.0

        return torch.tensor([
            node.heat_magnitude / math.sqrt(2),
            (node.cube_x + 1.0) / 2.0,
            (node.cube_y + 1.0) / 2.0,
            (node.polarity + 1.0) / 2.0,
            exist_oh[0], exist_oh[1], exist_oh[2], exist_oh[3],
            min(node.righteousness, 2.0) / 2.0,
            (node.cube_tau + 1.0) / 2.0,
        ], dtype=torch.float32)

    def _pool_node_embedding(self, node: Node) -> 'torch.Tensor':
        """Get (1, D_EMB) embedding for a node: weighted axis pool + own motion.

        A node's embedding = what it KNOWS (axis pool) + what it IS (motion functions).
        """
        # Axis pool
        token_feats, weights = self._extract_axis_tokens(node)
        if token_feats.shape[0] > 0:
            axis_emb = self._transformer.pool_axis_tokens(token_feats, weights)
        else:
            axis_emb = torch.zeros(1, D_EMB)

        # Node's own motion functions
        node_motion = self._extract_node_motion(node)
        node_emb = self._transformer.project_node_motion(node_motion)

        # Sum: what it KNOWS + what it IS
        return axis_emb + node_emb

    def _embed_all_nodes(self, nodes: List[Node]) -> 'torch.Tensor':
        """Embed all eligible nodes into (N, D_EMB) tensor."""
        if not nodes:
            return torch.zeros(0, D_EMB)

        embeddings = []
        for node in nodes:
            emb = self._pool_node_embedding(node)
            embeddings.append(emb)

        return torch.cat(embeddings, dim=0)  # (N, D_EMB)

    def _build_query(self, domain_ctx: Dict, node_embeddings: 'torch.Tensor',
                     nodes: List[Node]) -> 'torch.Tensor':
        """Build (1, D_EMB) query from state context.

        Query = weighted average of:
            State node (0.5) + Ego (0.2) + Identity (0.2) + hot domain nodes (0.1)

        When vision targets exist, their 32-dim features are added as focus hint.
        """
        query = torch.zeros(1, D_EMB)
        total_weight = 0.0

        # Helper to get embedding for a specific node by concept
        def get_emb(concept: str, fallback_node=None) -> Optional['torch.Tensor']:
            node = self.manifold.get_node_by_concept(concept) if concept else fallback_node
            if node is None:
                return None
            # Check if it's in our eligible list
            for i, n in enumerate(nodes):
                if n.id == node.id:
                    return node_embeddings[i:i+1]
            # Not in eligible list — compute directly
            if self._transformer:
                return self._pool_node_embedding(node)
            return None

        # State node (weight 0.5)
        hot_nodes = domain_ctx.get("hot_nodes", [])
        if hot_nodes:
            # First hot node is typically the state node
            state_emb = get_emb(None, hot_nodes[0])
            if state_emb is not None:
                query += 0.5 * state_emb
                total_weight += 0.5

        # Ego (weight 0.2)
        ego_emb = get_emb(None, self.manifold.ego_node)
        if ego_emb is not None:
            query += 0.2 * ego_emb
            total_weight += 0.2

        # Identity (weight 0.2)
        id_emb = get_emb(None, self.manifold.identity_node)
        if id_emb is not None:
            query += 0.2 * id_emb
            total_weight += 0.2

        # Hot domain nodes top-3 (weight 0.1 shared)
        if len(hot_nodes) > 1:
            hot_embs = []
            for h in hot_nodes[1:4]:
                h_emb = get_emb(None, h)
                if h_emb is not None:
                    hot_embs.append(h_emb)
            if hot_embs:
                hot_avg = torch.stack(hot_embs).mean(dim=0)
                query += 0.1 * hot_avg
                total_weight += 0.1

        # Normalize by total weight
        if total_weight > 0:
            query = query / total_weight

        # Vision focus hint: inject PC's 32-dim features if available
        targets = domain_ctx.get("targets", [])
        if targets and len(targets) > 0:
            best_target = targets[0]
            features = best_target.get("features")
            if features and len(features) == D_EMB:
                focus = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
                query = query + 0.1 * focus  # Additive hint (same as HeatAttention)

        # Base motion context: score primary hot node against base motions
        # and blend weighted base motion embeddings into query (0.15 weight)
        bm_embs = self._get_base_motion_embeddings()
        if bm_embs is not None and bm_embs.shape[0] > 0 and hot_nodes:
            primary = hot_nodes[0]
            bm_scores = self.score_against_base_motions(primary)
            if bm_scores:
                score_tensor = torch.tensor(
                    [bm_scores.get(c, 0.0) for c in self._bm_concepts],
                    dtype=torch.float32,
                )
                score_sum = score_tensor.sum()
                if score_sum > 0:
                    weights_bm = score_tensor / score_sum  # (M,)
                    bm_blend = (weights_bm.unsqueeze(1) * bm_embs).sum(dim=0, keepdim=True)
                    query = query + 0.15 * bm_blend

        # Active verb enrichment: MotionBus-activated verbs weight the BM embeddings
        # stronger than passive scoring (0.2 vs 0.15) because these are explicitly activated
        active_verbs = domain_ctx.get("active_verbs", {})
        if active_verbs and bm_embs is not None and bm_embs.shape[0] > 0:
            verb_weights = torch.tensor(
                [active_verbs.get(c, 0.0) for c in self._bm_concepts],
                dtype=torch.float32,
            )
            verb_sum = verb_weights.sum()
            if verb_sum > 0:
                verb_norm = verb_weights / verb_sum  # (M,)
                verb_blend = (verb_norm.unsqueeze(1) * bm_embs).sum(dim=0, keepdim=True)
                query = query + 0.2 * verb_blend

        return query

    def _righteousness_gate(self, scores: 'torch.Tensor',
                            nodes: List[Node]) -> 'torch.Tensor':
        """Gate scores by evaluate_righteousness — suppress misaligned nodes.

        Uses the actual cube measurement, not a learned gate.
        """
        gated = scores.clone()
        for i, node in enumerate(nodes):
            cube_pos = CubePosition(x=node.cube_x, y=node.cube_y, tau=node.cube_tau)
            R = cube_evaluate_R(cube_pos)
            # R=0 → gate=1.0, R≥2 → gate→0
            gate = 1.0 / (1.0 + R)
            gated[i] *= gate
        return gated

    # ═══════════════════════════════════════════════════════════════════════════
    # SUGGEST — Transformer-powered exploration
    # ═══════════════════════════════════════════════════════════════════════════

    def suggest(self, domain_ctx: Dict) -> Optional[List[str]]:
        """Sandbox exploration within domain-scoped context.

        Uses ManifoldAttention to score all eligible nodes against the current
        state context, then filters by domain and righteousness.

        Falls back to heat-sorted domain concepts if torch unavailable.

        Args:
            domain_ctx: Domain context from environment, containing:
                - hot_nodes: List[Node] — driver-scoped hot nodes
                - domain_prefix: str — domain ID (ow, the, end, tc, etc.)
                - action_durations: Dict[str, float] — default durations
                - targets: List[Dict] — vision targets

        Returns:
            List of suggested concept/action names, or None
        """
        if not self.should_think():
            return None

        domain_prefix = domain_ctx.get("domain_prefix", "")

        # Fallback path: no transformer
        if self._transformer is None:
            return self._suggest_fallback(domain_ctx)

        # Get eligible nodes
        nodes = self._eligible_nodes()
        if not nodes:
            return None

        with torch.no_grad():
            # Embed all nodes
            node_embeddings = self._embed_all_nodes(nodes)
            if node_embeddings.shape[0] == 0:
                return None

            # Build query from state context
            query = self._build_query(domain_ctx, node_embeddings, nodes)

            # Forward pass — cross-node attention
            scores = self._transformer.forward(node_embeddings, query)  # (N, 1)

            # Righteousness gate
            scores = self._righteousness_gate(scores, nodes)

        # Convert to list of (concept, score)
        scored = []
        for i, node in enumerate(nodes):
            scored.append((node.concept, scores[i].item()))

        # Domain post-filter — exclude foreign domain concepts
        if domain_prefix:
            _KNOWN_PREFIXES = ("ow_", "the_", "end_", "tc_", "bj_")
            def _is_own_domain(c):
                # Concepts starting with our prefix are always OK
                if c.startswith(domain_prefix):
                    return True
                # Concepts starting with a DIFFERENT known prefix are foreign
                for pfx in _KNOWN_PREFIXES:
                    if c.startswith(pfx) and not pfx.startswith(domain_prefix):
                        return False
                # State keys with digits from unknown prefix are foreign
                if '_' in c and any(ch.isdigit() for ch in c):
                    return False
                # Everything else (actions, abstract concepts) is OK
                return True
            scored = [(c, s) for c, s in scored if _is_own_domain(c)]
            if not scored:
                return None

        # Sort by score descending, take top 15
        scored.sort(key=lambda x: x[1], reverse=True)
        concepts = [c for c, s in scored[:15]]

        if not concepts:
            return None

        # Pay evaluation cost
        if self.manifold.ego_node:
            self.manifold.ego_node.spend_heat(COST_EVALUATE, minimum=PSYCHOLOGY_MIN_HEAT)

        # Record to STM
        self._record_stm("introspect_suggest", [
            SimulationResult(option=c, heat_magnitude=0.0) for c in concepts[:5]
        ])

        logger.debug(f"Introspector transformer: {len(concepts)} concepts, "
                      f"top={concepts[0] if concepts else 'none'}")
        return concepts

    def _suggest_fallback(self, domain_ctx: Dict) -> Optional[List[str]]:
        """Fallback suggest when torch unavailable — heat-sorted domain concepts."""
        hot_nodes = domain_ctx.get("hot_nodes", [])
        if not hot_nodes:
            nodes = self._eligible_nodes()
            if not nodes:
                return None
            nodes.sort(key=lambda n: n.heat, reverse=True)
            hot_nodes = nodes[:10]

        domain_prefix = domain_ctx.get("domain_prefix", "")
        concepts = []
        for node in hot_nodes:
            c = node.concept
            if domain_prefix:
                if '_' in c and any(ch.isdigit() for ch in c) and not c.startswith(domain_prefix):
                    continue
            concepts.append(c)

        if not concepts:
            return None

        # Pay evaluation cost
        if self.manifold.ego_node:
            self.manifold.ego_node.spend_heat(COST_EVALUATE, minimum=PSYCHOLOGY_MIN_HEAT)

        self._record_stm("introspect_fallback", [
            SimulationResult(option=c, heat_magnitude=0.0) for c in concepts[:5]
        ])

        return concepts

    # ═══════════════════════════════════════════════════════════════════════════
    # DELIBERATE — Multi-pass transformer reasoning (replaces single-shot suggest)
    # ═══════════════════════════════════════════════════════════════════════════

    def deliberate(self, domain_ctx: Dict, max_passes: int = 3) -> Optional[List[str]]:
        """Multi-pass reasoning: each pass refines the previous, narrowing focus.

        Pass 1: Run suggest() — broad top-15 concepts.
        Pass 2+: Inject prior results as weighted signal into query, re-score.
        Each pass pays COST_EVALUATE from Ego — natural energy budget limits depth.

        Returns a ranked concept sequence (up to ~10 actions) or None.
        """
        # Pass 1: broad exploration via suggest()
        concepts = self.suggest(domain_ctx)
        if not concepts:
            return None

        # No transformer → can't refine, return single-pass result
        if self._transformer is None:
            return concepts

        # Refinement passes (2..max_passes), gated by Ego energy
        nodes = self._eligible_nodes()
        if not nodes:
            return concepts

        for pass_num in range(2, max_passes + 1):
            # Energy gate: each refinement costs COST_EVALUATE
            if not self.should_think():
                break

            with torch.no_grad():
                node_embeddings = self._embed_all_nodes(nodes)
                if node_embeddings.shape[0] == 0:
                    break

                # Build refinement query: base query + prior pass signal
                query = self._build_refinement_query(
                    domain_ctx, node_embeddings, nodes, concepts
                )

                scores = self._transformer.forward(node_embeddings, query)
                scores = self._righteousness_gate(scores, nodes)

            # Re-score and merge with previous concepts
            scored = []
            for i, node in enumerate(nodes):
                scored.append((node.concept, scores[i].item()))

            # Domain filter (same as suggest)
            domain_prefix = domain_ctx.get("domain_prefix", "")
            if domain_prefix:
                _KNOWN_PREFIXES = ("ow_", "the_", "end_", "tc_", "bj_")
                scored = [(c, s) for c, s in scored
                          if c.startswith(domain_prefix)
                          or not any(c.startswith(p) for p in _KNOWN_PREFIXES
                                     if not p.startswith(domain_prefix))]

            scored.sort(key=lambda x: x[1], reverse=True)
            refined = [c for c, s in scored[:15]]

            if not refined:
                break

            # Merge: refined pass concepts take priority, append novel from prior
            seen = set(refined)
            for c in concepts:
                if c not in seen:
                    refined.append(c)
                    seen.add(c)
            concepts = refined[:15]

            # Pay for this pass
            if self.manifold.ego_node:
                self.manifold.ego_node.spend_heat(COST_EVALUATE, minimum=PSYCHOLOGY_MIN_HEAT)

            logger.debug(f"Deliberate pass {pass_num}: top={concepts[0] if concepts else 'none'}")

        return concepts

    def _build_refinement_query(self, domain_ctx: Dict,
                                 node_embeddings: 'torch.Tensor',
                                 nodes: List[Node],
                                 prior_concepts: List[str]) -> 'torch.Tensor':
        """Build query for refinement pass, injecting prior suggestions as signal.

        Same as _build_query, but adds prior concept embeddings as extra context.
        """
        # Start with standard query
        query = self._build_query(domain_ctx, node_embeddings, nodes)

        # Add prior concept embeddings as refinement signal (weight 0.25)
        prior_embs = []
        for c in prior_concepts[:5]:
            node = self.manifold.get_node_by_concept(c)
            if node is not None:
                emb = self._pool_node_embedding(node)
                if emb is not None:
                    prior_embs.append(emb)

        if prior_embs:
            prior_avg = torch.stack(prior_embs).mean(dim=0)
            query = query + 0.25 * prior_avg

        return query

    # ═══════════════════════════════════════════════════════════════════════════
    # MONITOR — Lightweight execution check during plan execution
    # ═══════════════════════════════════════════════════════════════════════════

    _MONITOR_DIVERGENCE_THRESHOLD = 0.3

    def monitor(self, domain_ctx: Dict, remaining_plan: List[str]) -> Optional[List[str]]:
        """Lightweight check: is the current state still consistent with the plan?

        Embeds current state query and computes cosine similarity against
        remaining plan action embeddings. If similarity drops below threshold,
        signals replan needed.

        Returns None (plan still valid) or a replacement concept list.
        Cost: ~1/3 of suggest() — just query embedding + dot products.
        """
        if not remaining_plan:
            return None

        if self._transformer is None:
            return None  # Can't monitor without embeddings

        # Energy gate: monitoring is cheap but not free
        if not self.should_think():
            return None

        nodes = self._eligible_nodes()
        if not nodes:
            return None

        with torch.no_grad():
            node_embeddings = self._embed_all_nodes(nodes)
            if node_embeddings.shape[0] == 0:
                return None

            # Build current state query
            query = self._build_query(domain_ctx, node_embeddings, nodes)  # (1, D_EMB)

            # Embed plan actions: get embeddings for concepts in remaining plan
            plan_embs = []
            for action_name in remaining_plan:
                node = self.manifold.get_node_by_concept(action_name)
                if node is not None:
                    emb = self._pool_node_embedding(node)
                    if emb is not None:
                        plan_embs.append(emb)

            if not plan_embs:
                return None  # Can't evaluate plan — let it run

            # Average plan embedding
            plan_avg = torch.stack(plan_embs).mean(dim=0)  # (1, D_EMB)

            # Cosine similarity between current state and plan direction
            cos_sim = torch.nn.functional.cosine_similarity(query, plan_avg, dim=1)
            similarity = cos_sim.item()

        if similarity < self._MONITOR_DIVERGENCE_THRESHOLD:
            # State has diverged — trigger replan via deliberate
            logger.info(f"Monitor: diverged (sim={similarity:.3f} < {self._MONITOR_DIVERGENCE_THRESHOLD}), replanning")
            return self.deliberate(domain_ctx)

        logger.debug(f"Monitor: on track (sim={similarity:.3f})")
        return None

    # ═══════════════════════════════════════════════════════════════════════════
    # WEIGHT BOOSTS — Convert suggestions to action weights
    # ═══════════════════════════════════════════════════════════════════════════

    def get_weight_boosts(self, suggestions: List[str], available_actions: List[str],
                          verb_action_map: Dict[str, List[str]] = None) -> Dict[str, float]:
        """
        Convert Introspector suggestions to action weight boosts.

        Matches suggested concepts against available action names:
        - Direct match: concept == action name → boost 4.0
        - Partial match: concept is substring of action name → boost 2.5
        - Axis match: concept node has axis to node matching action → boost 2.0
        - Base motion match: bm_* concept maps to actions via verb_action_map → boost 3.0

        verb_action_map is provided by the active driver (domain-specific).
        """
        boosts = {}
        action_set = set(available_actions)
        vmap = verb_action_map or {}

        for concept in suggestions:
            # Base motion match: bm_* concept → mapped actions
            if concept.startswith(BASE_MOTION_PREFIX):
                mapped = vmap.get(concept, [])
                for action in mapped:
                    if action in action_set:
                        boosts[action] = max(boosts.get(action, 1.0), 3.0)
                continue

            # Direct match: concept name IS an action
            if concept in action_set:
                boosts[concept] = max(boosts.get(concept, 1.0), 4.0)
                continue

            # Partial match: concept appears in an action name
            for action in available_actions:
                if concept in action or action in concept:
                    boosts[action] = max(boosts.get(action, 1.0), 2.5)

            # Axis match: concept node has axes to nodes whose concepts match actions
            concept_node = self.manifold.get_node_by_concept(concept)
            if concept_node:
                for axis_name, axis in concept_node.frame.axes.items():
                    target_node = self.manifold.get_node(axis.target_id)
                    if target_node and target_node.concept in action_set:
                        boosts[target_node.concept] = max(
                            boosts.get(target_node.concept, 1.0), 2.0
                        )

        return boosts
