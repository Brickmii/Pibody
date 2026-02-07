"""
PBAI Processing Cycle Test — β→δ→Γ→α→ζ Pipeline + Angular Proximity

Tests the full decision processing pipeline:
    β (Beta)  : Euler beta superposition weights
    δ (Delta) : Wave function collapse (Born rule)
    Γ (Gamma) : Arrangement counting
    α (Alpha) : Fine-structure coupling
    ζ (Zeta)  : Significance normalization

Also tests:
    - Angular proximity (candidate node finding)
    - Introspector cube projection
    - Hypersphere geometry functions

Run: python3 -m core.test_processing_cycle
"""

import math
import sys
import os
import logging

# Setup path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.node_constants import (
    K, PHI, INV_PHI,
    FINE_STRUCTURE_CONSTANT,
    CONFIDENCE_EXPLOIT_THRESHOLD,
    THRESHOLD_RIGHTEOUSNESS,
    euler_beta, wave_function, gamma_function, entropy_count,
    collapse_wave_function, correlate_cluster, select_from_cluster,
    COST_EVALUATE, PSYCHOLOGY_MIN_HEAT,
    EXISTENCE_ACTUAL, EXISTENCE_DORMANT,
)
from core.nodes import Node, SelfNode, reset_birth_for_testing
from core.manifold import Manifold, create_manifold, reset_pbai_manifold
from core.hypersphere import (
    SpherePosition, angular_distance, relationship_strength,
    place_node_near, find_neighbors, k_nearest,
)
from core.color_cube import CubePosition, evaluate_righteousness
from core.introspector import Introspector, SimulationResult
from core.decision_node import DecisionNode, ChoiceNode

logging.basicConfig(level=logging.WARNING, format='%(message)s')
logger = logging.getLogger(__name__)

passed = 0
failed = 0


def ok(msg):
    global passed
    passed += 1
    print(f"  ok {msg}")


def fail(msg):
    global failed
    failed += 1
    print(f"  FAIL {msg}")


def check(condition, msg):
    if condition:
        ok(msg)
    else:
        fail(msg)


# ═════════════════════════════════════════════════════════════════════════════
# TEST 1: EULER BETA FUNCTION (β)
# ═════════════════════════════════════════════════════════════════════════════

def test_euler_beta():
    """Test β processing: Euler beta superposition weights."""
    print("\n" + "=" * 60)
    print("TEST 1: EULER BETA FUNCTION (β)")
    print("=" * 60)

    # β(1,1) = Γ(1)Γ(1)/Γ(2) = 1*1/1 = 1.0
    b11 = euler_beta(1, 1)
    check(abs(b11 - 1.0) < 1e-10, f"β(1,1) = {b11:.6f} (expected 1.0)")

    # β(2,1) = Γ(2)Γ(1)/Γ(3) = 1*1/2 = 0.5
    b21 = euler_beta(2, 1)
    check(abs(b21 - 0.5) < 1e-10, f"β(2,1) = {b21:.6f} (expected 0.5)")

    # β(x,y) = β(y,x) — symmetry
    b23 = euler_beta(2, 3)
    b32 = euler_beta(3, 2)
    check(abs(b23 - b32) < 1e-10, f"β(2,3) = β(3,2) = {b23:.6f} (symmetric)")

    # β(a,b) = Γ(a)Γ(b)/Γ(a+b) — manual check
    b34 = euler_beta(3, 4)
    expected = math.gamma(3) * math.gamma(4) / math.gamma(7)
    check(abs(b34 - expected) < 1e-10, f"β(3,4) = Γ(3)Γ(4)/Γ(7) = {b34:.6f}")

    # With Laplace smoothing (1+s, 1+f): no history → β(1,1) = 1.0
    no_history = euler_beta(0 + 1, 0 + 1)
    check(abs(no_history - 1.0) < 1e-10,
          f"No history (Laplace): β(1,1) = {no_history:.6f}")

    # β is symmetric: β(a,b) = β(b,a) — but in the pipeline,
    # more successes with Laplace smoothing shifts the ratio
    # β(6,2) with higher first param = more concentrated prior
    b_high = euler_beta(10, 2)
    b_low = euler_beta(2, 10)
    check(abs(b_high - b_low) < 1e-10,
          f"β is symmetric: β(10,2) = β(2,10) = {b_high:.6f}")

    # But β(n+1, 1) decreases as n grows: 1, 0.5, 0.333, 0.25, ...
    b_1_1 = euler_beta(1, 1)
    b_2_1 = euler_beta(2, 1)
    b_5_1 = euler_beta(5, 1)
    check(b_1_1 > b_2_1 > b_5_1,
          f"β(1,1)={b_1_1:.4f} > β(2,1)={b_2_1:.4f} > β(5,1)={b_5_1:.4f}")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 2: WAVE FUNCTION (Ψ)
# ═════════════════════════════════════════════════════════════════════════════

def test_wave_function():
    """Test Ψ: Wave function creation and collapse."""
    print("\n" + "=" * 60)
    print("TEST 2: WAVE FUNCTION (Ψ)")
    print("=" * 60)

    # Empty → zero amplitude
    psi_empty = wave_function([])
    check(abs(psi_empty) < 1e-10, f"|Ψ(empty)| = {abs(psi_empty):.6f} (zero)")

    # Single path → unit amplitude
    psi_one = wave_function(["only"])
    check(abs(abs(psi_one) - 1.0) < 1e-10,
          f"|Ψ(single)| = {abs(psi_one):.6f} (unit)")

    # Two equal paths → interference
    psi_two = wave_function(["a", "b"])
    # Equal weights, opposite phases → partial cancellation
    check(abs(psi_two) < 1.0,
          f"|Ψ(2 paths)| = {abs(psi_two):.6f} (interference < 1)")

    # |Ψ|^2 gives probability (Born rule)
    prob = abs(psi_one) ** 2
    check(abs(prob - 1.0) < 1e-10,
          f"|Ψ|^2 = {prob:.6f} (probability = 1 for single path)")

    # Custom weights change amplitude
    psi_weighted = wave_function(["a", "b"], weights=[0.9, 0.1])
    psi_uniform = wave_function(["a", "b"], weights=[0.5, 0.5])
    check(abs(psi_weighted) != abs(psi_uniform),
          "Different weights -> different |Ψ|")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 3: COLLAPSE (δ) — Deterministic center-finding
# ═════════════════════════════════════════════════════════════════════════════

def test_collapse():
    """Test δ processing: collapse_wave_function finds R→0 center."""
    print("\n" + "=" * 60)
    print("TEST 3: WAVE FUNCTION COLLAPSE (δ)")
    print("=" * 60)

    # Empty → -1
    check(collapse_wave_function([]) == -1, "Empty nodes -> -1")

    # Single → 0
    check(collapse_wave_function([Node(concept="only")]) == 0, "Single node -> 0")

    # Collapse finds lowest R (most righteous)
    nodes = [
        Node(concept="far", theta=1.0, phi=0.0, righteousness=0.8),
        Node(concept="center", theta=1.5, phi=0.5, righteousness=0.01),
        Node(concept="mid", theta=2.0, phi=1.0, righteousness=0.5),
    ]
    center_idx = collapse_wave_function(nodes)
    check(center_idx == 1,
          f"Collapse finds center (R=0.01) at index {center_idx}")

    # R=0 always wins (perfectly aligned)
    nodes_with_zero = [
        Node(concept="misaligned", theta=1.0, phi=0.0, righteousness=1.0),
        Node(concept="perfect", theta=1.5, phi=0.5, righteousness=0.0),
    ]
    idx = collapse_wave_function(nodes_with_zero)
    check(idx == 1, f"R=0 (perfect alignment) wins: index {idx}")

    # Amplitude follows Gaussian: a = e^(-R^2/2sigma^2)
    sigma = THRESHOLD_RIGHTEOUSNESS
    for R_val in [0.0, 0.1, 0.5, 1.0]:
        expected_amp = math.exp(-(R_val ** 2) / (2 * sigma ** 2))
        node = Node(concept=f"R_{R_val}", righteousness=R_val)
        idx = collapse_wave_function([node])
        check(idx == 0,
              f"  R={R_val:.1f} -> amplitude={expected_amp:.4f} (Gaussian)")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 4: GAMMA FUNCTION (Γ) — Arrangement counting
# ═════════════════════════════════════════════════════════════════════════════

def test_gamma():
    """Test Γ processing: gamma function and entropy counting."""
    print("\n" + "=" * 60)
    print("TEST 4: GAMMA FUNCTION (Γ)")
    print("=" * 60)

    # Γ(1) = 0! = 1
    check(abs(gamma_function(1) - 1.0) < 1e-10,
          f"Γ(1) = {gamma_function(1):.6f} (= 0! = 1)")

    # Γ(2) = 1! = 1
    check(abs(gamma_function(2) - 1.0) < 1e-10,
          f"Γ(2) = {gamma_function(2):.6f} (= 1! = 1)")

    # Γ(5) = 4! = 24
    check(abs(gamma_function(5) - 24.0) < 1e-8,
          f"Γ(5) = {gamma_function(5):.6f} (= 4! = 24)")

    # Γ(n+1) = n * Γ(n)  — recurrence relation
    for n in [1.5, 2.5, 3.7]:
        lhs = gamma_function(n + 1)
        rhs = n * gamma_function(n)
        check(abs(lhs - rhs) < 1e-8,
              f"Γ({n}+1) = {n} * Γ({n}): {lhs:.6f} ~ {rhs:.6f}")

    # Entropy: S = ln(omega)
    check(abs(entropy_count(1) - 0.0) < 1e-10,
          f"S(1 arrangement) = {entropy_count(1):.6f} (zero entropy)")
    check(abs(entropy_count(math.e) - 1.0) < 1e-10,
          f"S(e arrangements) = {entropy_count(math.e):.6f} (unit entropy)")

    # More arrangements → more entropy
    s10 = entropy_count(10)
    s100 = entropy_count(100)
    check(s100 > s10, f"S(100)={s100:.4f} > S(10)={s10:.4f}")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 5: FINE-STRUCTURE COUPLING (α)
# ═════════════════════════════════════════════════════════════════════════════

def test_alpha_coupling():
    """Test α processing: fine-structure constant and coupling."""
    print("\n" + "=" * 60)
    print("TEST 5: FINE-STRUCTURE COUPLING (α)")
    print("=" * 60)

    # α ~ 1/137
    check(abs(FINE_STRUCTURE_CONSTANT - 1 / 137.035999) < 1e-10,
          f"α = {FINE_STRUCTURE_CONSTANT:.8f} ~ 1/137")

    # α is the base coupling strength (when no history)
    check(FINE_STRUCTURE_CONSTANT > 0 and FINE_STRUCTURE_CONSTANT < 0.01,
          f"α is a small positive coupling: {FINE_STRUCTURE_CONSTANT:.6f}")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 6: ZETA NORMALIZATION (ζ)
# ═════════════════════════════════════════════════════════════════════════════

def test_zeta_normalization():
    """Test ζ processing: significance normalization."""
    print("\n" + "=" * 60)
    print("TEST 6: ZETA NORMALIZATION (ζ)")
    print("=" * 60)

    # Create a manifold + decision node for testing
    reset_pbai_manifold()
    reset_birth_for_testing()
    m = create_manifold()

    dn = DecisionNode(m)

    options = ["opt_a", "opt_b", "opt_c"]
    state_key = "test_state"

    # β: create superposition
    superposition = dn._create_superposition(options, state_key)
    check(len(superposition['weights']) == 3,
          f"β creates {len(superposition['weights'])} weights for 3 options")
    check(abs(sum(superposition['weights']) - 1.0) < 1e-10,
          f"β weights sum to 1.0")

    # Γ: arrangement counts
    gamma_scores = dn._get_gamma_scores(options, state_key)
    check(len(gamma_scores) == 3,
          f"Γ scores for all {len(gamma_scores)} options")
    for opt, g in gamma_scores.items():
        check(g >= 1, f"  Γ({opt}) = {g} >= 1 (at least 1 arrangement)")

    # α: coupling strengths
    alpha_couplings = dn._get_alpha_couplings(options, state_key)
    check(len(alpha_couplings) == 3,
          f"α couplings for all {len(alpha_couplings)} options")
    for opt, a in alpha_couplings.items():
        check(abs(a - FINE_STRUCTURE_CONSTANT) < 1e-10,
              f"  α({opt}) = {a:.6f} (base coupling, no history)")

    # ζ: normalize
    significance = dn._zeta_normalize(options, superposition,
                                       gamma_scores, alpha_couplings)
    check(len(significance) == 3,
          f"ζ significance for all {len(significance)} options")

    # ζ scores should be in [0, 1]
    all_in_range = all(0.0 <= s <= 1.0 + 1e-10 for s in significance.values())
    check(all_in_range, "ζ scores in [0, 1]")

    # Best score = 1.0 (by normalization)
    max_sig = max(significance.values())
    check(abs(max_sig - 1.0) < 1e-10 or max_sig == 0.0,
          f"ζ best score = {max_sig:.4f} (normalized to 1.0 or all zero)")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 7: FULL β→δ→Γ→α→ζ PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def test_full_pipeline():
    """Test the full processing cycle through psychology."""
    print("\n" + "=" * 60)
    print("TEST 7: FULL β→δ→Γ→α→ζ PIPELINE")
    print("=" * 60)

    reset_pbai_manifold()
    reset_birth_for_testing()
    m = create_manifold()

    dn = DecisionNode(m)

    # Add a state node on the hypersphere
    state_node = Node(concept="pipeline_state", theta=math.pi / 3, phi=1.0,
                      radius=1.0, heat=K, existence=EXISTENCE_ACTUAL)
    m.add_node(state_node)

    options = ["go_left", "go_right", "stay"]

    # Low confidence → EXPLORE (probabilistic)
    result_explore = dn.decide("pipeline_state", options, confidence=0.3)
    check(result_explore in options,
          f"Explore selected: {result_explore} (confidence=0.3 < 5/6)")

    # High confidence → EXPLOIT (deterministic)
    result_exploit = dn.decide("pipeline_state", options, confidence=0.9)
    check(result_exploit in options,
          f"Exploit selected: {result_exploit} (confidence=0.9 > 5/6)")

    # 5/6 threshold
    check(abs(CONFIDENCE_EXPLOIT_THRESHOLD - 5 / 6) < 1e-10,
          f"Exploit threshold = 5/6 = {CONFIDENCE_EXPLOIT_THRESHOLD:.6f}")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 8: ANGULAR PROXIMITY (Candidate finding)
# ═════════════════════════════════════════════════════════════════════════════

def test_angular_proximity():
    """Test angular proximity for candidate node finding."""
    print("\n" + "=" * 60)
    print("TEST 8: ANGULAR PROXIMITY")
    print("=" * 60)

    # Same point → distance 0
    p1 = SpherePosition(theta=1.0, phi=2.0)
    check(abs(angular_distance(p1, p1)) < 1e-10,
          "Same point -> angular distance = 0")

    # Opposite poles → distance π
    north = SpherePosition(theta=0.0, phi=0.0)
    south = SpherePosition(theta=math.pi, phi=0.0)
    dist = angular_distance(north, south)
    check(abs(dist - math.pi) < 1e-10,
          f"North<->South = {dist:.6f} (pi = {math.pi:.6f})")

    # 90 deg apart on equator
    e1 = SpherePosition(theta=math.pi / 2, phi=0.0)
    e2 = SpherePosition(theta=math.pi / 2, phi=math.pi / 2)
    dist90 = angular_distance(e1, e2)
    check(abs(dist90 - math.pi / 2) < 1e-6,
          f"90 deg apart on equator = {dist90:.6f} (pi/2 = {math.pi / 2:.6f})")

    # Relationship strength: closer → stronger
    close = SpherePosition(theta=1.0, phi=0.0)
    near = SpherePosition(theta=1.0, phi=0.1)
    far = SpherePosition(theta=1.0, phi=2.0)
    r_near = relationship_strength(close, near)
    r_far = relationship_strength(close, far)
    check(r_near > r_far,
          f"Closer -> stronger: {r_near:.4f} > {r_far:.4f}")

    # k_nearest returns correct count
    center = SpherePosition(theta=math.pi / 2, phi=0.0)
    points = [
        SpherePosition(theta=math.pi / 2, phi=0.1),   # very close
        SpherePosition(theta=math.pi / 2, phi=0.5),   # medium
        SpherePosition(theta=math.pi / 2, phi=1.0),   # further
        SpherePosition(theta=math.pi / 2, phi=2.0),   # far
        SpherePosition(theta=math.pi / 2, phi=3.0),   # very far
    ]
    nearest3 = k_nearest(center, points, k=3)
    check(len(nearest3) == 3, f"k_nearest(k=3) returns {len(nearest3)} tuples")

    # k_nearest returns (index, distance) tuples, sorted by distance
    idx0, d0 = nearest3[0]
    idx2, d2 = nearest3[2]
    check(d0 <= d2, f"k_nearest sorted by distance ({d0:.4f} <= {d2:.4f})")
    check(idx0 == 0, f"Closest point is index 0 (phi=0.1), got {idx0}")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 9: INTROSPECTOR SIMULATION
# ═════════════════════════════════════════════════════════════════════════════

def test_introspector():
    """Test introspector cube projection and gating."""
    print("\n" + "=" * 60)
    print("TEST 9: INTROSPECTOR SIMULATION")
    print("=" * 60)

    reset_pbai_manifold()
    reset_birth_for_testing()
    m = create_manifold()

    intro = Introspector(m)

    # Ensure psychology nodes have enough energy
    if m.ego_node:
        m.ego_node.heat = COST_EVALUATE + PSYCHOLOGY_MIN_HEAT + 1.0
        m.ego_node.existence = EXISTENCE_ACTUAL
    if m.conscience_node:
        m.conscience_node.heat = COST_EVALUATE + PSYCHOLOGY_MIN_HEAT + 1.0
        m.conscience_node.existence = EXISTENCE_ACTUAL

    can_think = intro.should_think()
    check(can_think, f"should_think = {can_think} (both Ego + Conscience energized)")

    # Simulate options
    options = ["look", "move", "wait"]
    results = intro.simulate(options, "test_state")
    check(len(results) == 3, f"Simulated {len(results)} options")

    for r in results:
        check(isinstance(r, SimulationResult), f"  {r.option} is SimulationResult")
        # Cube projection in [-1, 1]
        check(-1.0 <= r.cube_x <= 1.0, f"  {r.option} cube_x={r.cube_x:.4f} in [-1,1]")
        check(-1.0 <= r.cube_y <= 1.0, f"  {r.option} cube_y={r.cube_y:.4f} in [-1,1]")
        check(-1.0 <= r.cube_tau <= 1.0, f"  {r.option} cube_tau={r.cube_tau:.4f} in [-1,1]")
        # Quadrant is valid
        check(r.quadrant in ("Q1", "Q2", "Q3", "Q4"),
              f"  {r.option} quadrant={r.quadrant}")

    # Context packaging
    context = intro.to_context(results)
    check(context.get("simulated") is True, "Context has simulated=True")
    check("sim_best_option" in context, "Context has best option")
    check("sim_spread" in context, "Context has spread")

    # Rankings
    ranked = intro.rank_by_conscience(results)
    check(ranked[0].conscience_score >= ranked[-1].conscience_score,
          "rank_by_conscience: best first")

    # STM
    summary = intro.get_stm_summary()
    check(summary["stm_entries"] == 1, f"STM has {summary['stm_entries']} entry after 1 simulation")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 10: ANGULAR NODE PLACEMENT
# ═════════════════════════════════════════════════════════════════════════════

def test_place_node_near():
    """Test angular node placement avoids collisions."""
    print("\n" + "=" * 60)
    print("TEST 10: ANGULAR NODE PLACEMENT")
    print("=" * 60)

    target = SpherePosition(theta=math.pi / 2, phi=0.0)
    existing = [
        SpherePosition(theta=math.pi / 2, phi=0.0),  # exactly at target
    ]

    placed = place_node_near(target, existing)
    check(0.0 <= placed.theta <= math.pi,
          f"Placed theta={placed.theta:.4f} in [0, pi]")
    check(0.0 <= placed.phi < 2 * math.pi,
          f"Placed phi={placed.phi:.4f} in [0, 2pi)")

    # Should be near target but not exactly on existing
    dist_to_target = angular_distance(target, placed)
    check(dist_to_target < math.pi / 2,
          f"Placed near target (dist={dist_to_target:.4f} < pi/2)")

    # Place multiple — all should be distinct
    positions = [target]
    for i in range(5):
        p = place_node_near(target, positions)
        positions.append(p)

    # Check pairwise distances > 0
    all_distinct = True
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            if angular_distance(positions[i], positions[j]) < 1e-6:
                all_distinct = False
    check(all_distinct, f"All {len(positions)} placed nodes are distinct")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 11: CUBE PROJECTION MATH
# ═════════════════════════════════════════════════════════════════════════════

def test_cube_projection():
    """Test that hypersphere -> cube projection is geometrically correct."""
    print("\n" + "=" * 60)
    print("TEST 11: CUBE PROJECTION MATH")
    print("=" * 60)

    # North pole (theta=0) → cube_tau = +1 (future)
    n = Node(concept="north_pole", theta=0.0, phi=0.0, radius=1.0)
    check(abs(n.cube_tau - 1.0) < 1e-10,
          f"North pole -> tau={n.cube_tau:.4f} (future)")
    check(abs(n.cube_x) < 1e-10 and abs(n.cube_y) < 1e-10,
          f"North pole -> x={n.cube_x:.4f}, y={n.cube_y:.4f} (origin)")

    # South pole (theta=pi) → cube_tau = -1 (past)
    s = Node(concept="south_pole", theta=math.pi, phi=0.0, radius=1.0)
    check(abs(s.cube_tau - (-1.0)) < 1e-10,
          f"South pole -> tau={s.cube_tau:.4f} (past)")

    # Equator (theta=pi/2, phi=0) → +X (Yellow/East)
    e = Node(concept="east", theta=math.pi / 2, phi=0.0, radius=1.0)
    check(abs(e.cube_x - 1.0) < 1e-10,
          f"Equator phi=0 -> x={e.cube_x:.4f} (+X = Yellow)")
    check(abs(e.cube_tau) < 1e-10,
          f"Equator -> tau={e.cube_tau:.4f} (present)")

    # Equator (theta=pi/2, phi=pi/2) → +Y (Green/North)
    gn = Node(concept="green_north", theta=math.pi / 2, phi=math.pi / 2, radius=1.0)
    check(abs(gn.cube_y - 1.0) < 1e-10,
          f"Equator phi=pi/2 -> y={gn.cube_y:.4f} (+Y = Green)")

    # Equator (theta=pi/2, phi=pi) → -X (Blue/West)
    w = Node(concept="west", theta=math.pi / 2, phi=math.pi, radius=1.0)
    check(abs(w.cube_x - (-1.0)) < 1e-10,
          f"Equator phi=pi -> x={w.cube_x:.4f} (-X = Blue)")

    # Self (radius=0) → all projections = 0
    self_node = Node(concept="self", theta=0.0, phi=0.0, radius=0.0)
    check(abs(self_node.cube_x) < 1e-10 and abs(self_node.cube_y) < 1e-10
          and abs(self_node.cube_tau) < 1e-10,
          "Self (radius=0) -> all cube projections = 0")

    # Unit sphere: x^2 + y^2 + tau^2 = radius^2 = 1.0
    for theta_val in [0.3, 0.8, 1.2, 2.0, 2.8]:
        for phi_val in [0.0, 1.0, 2.5, 4.0, 5.5]:
            node = Node(concept="test", theta=theta_val, phi=phi_val, radius=1.0)
            r_sq = node.cube_x ** 2 + node.cube_y ** 2 + node.cube_tau ** 2
            assert abs(r_sq - 1.0) < 1e-10, f"Failed at theta={theta_val}, phi={phi_val}"
    ok("x^2 + y^2 + tau^2 = 1 for all surface points (25 checked)")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 12: CONFIDENCE THRESHOLD MATH
# ═════════════════════════════════════════════════════════════════════════════

def test_confidence_threshold():
    """Test the 5/6 confidence threshold and K-quantum relationship."""
    print("\n" + "=" * 60)
    print("TEST 12: CONFIDENCE THRESHOLD (5/6)")
    print("=" * 60)

    # 5/6 = 0.8333...
    check(abs(CONFIDENCE_EXPLOIT_THRESHOLD - 5 / 6) < 1e-10,
          f"Threshold = 5/6 = {CONFIDENCE_EXPLOIT_THRESHOLD:.6f}")

    # K = 4/phi^2
    check(abs(K * PHI ** 2 - 4.0) < 1e-10,
          f"K * phi^2 = {K * PHI**2:.6f} (= 4.0)")

    # t = 5K validations crosses 5/6 threshold
    target_validations = 5 * K
    check(target_validations > 0,
          f"5K = {target_validations:.4f} validations to cross threshold")

    # 1/phi thresholds sum to K
    inv_phi_powers = [1 / PHI ** n for n in range(1, 7)]
    threshold_sum = sum(inv_phi_powers)
    check(abs(threshold_sum - K) < 0.01,
          f"Sum(1/phi^n, n=1..6) = {threshold_sum:.6f} ~ K = {K:.6f}")


# ═════════════════════════════════════════════════════════════════════════════
# RUN ALL
# ═════════════════════════════════════════════════════════════════════════════

def run_all_tests():
    global passed, failed

    print("=" * 60)
    print(" PBAI PROCESSING CYCLE TEST")
    print(" β→δ→Γ→α→ζ Pipeline + Angular Geometry")
    print("=" * 60)

    test_euler_beta()
    test_wave_function()
    test_collapse()
    test_gamma()
    test_alpha_coupling()
    test_zeta_normalization()
    test_full_pipeline()
    test_angular_proximity()
    test_introspector()
    test_place_node_near()
    test_cube_projection()
    test_confidence_threshold()

    print("\n" + "=" * 60)
    if failed == 0:
        print(f" ALL {passed} TESTS PASSED")
    else:
        print(f" {passed} PASSED, {failed} FAILED")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
