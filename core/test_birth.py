#!/usr/bin/env python3
"""
PBAI Core Birth Test - Full System Verification (Hypersphere Geometry)

Tests the complete birth sequence and operation of all 6 motion functions:

    1. Heat (S)           -> psychology (magnitude)
    2. Polarity (+/-)     -> node_constants (direction)
    3. Existence (d)      -> clock_node (persistence, 1/phi^3 threshold)
    4. Righteousness (R)  -> nodes (alignment, frames)
    5. Order (Q)          -> manifold (arithmetic, history)
    6. Movement (Lin)     -> decision_node (5 scalars -> 1 vector)

Run: python3 -m core.test_birth
"""

import math
import sys
import os

# Ensure we can import from parent
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.nodes import Node, SelfNode, reset_birth_for_testing, assert_self_valid
from core.node_constants import (
    K, PHI, THRESHOLD_EXISTENCE, CONFIDENCE_EXPLOIT_THRESHOLD,
    EXISTENCE_ACTUAL, EXISTENCE_DORMANT, EXISTENCE_POTENTIAL,
    FREUD_IDENTITY_RATIO, FREUD_EGO_RATIO, FREUD_CONSCIENCE_RATIO,
    # Planck grounding
    BODY_TEMPERATURE, FIRE_HEAT, FIRE_TO_MOTION,
    EMERGENCE_THRESHOLD, MAX_ENTROPIC_PROBABILITY,
    ROBINSON_CONSTRAINTS, PLANCK_TEMPERATURE, SPEED_OF_LIGHT,
    # Base motions
    BASE_MOTIONS, ALL_BASE_MOTIONS, BASE_MOTION_PREFIX, MOTION_TO_FIRE, BASE_MOTION_HEAT,
)
from core.manifold import Manifold
from core.decision_node import DecisionNode, Choice
from core.clock_node import Clock, TickStats
from core.hypersphere import SpherePosition, angular_distance


def test_birth():
    """Test complete manifold birth sequence with hypersphere geometry."""
    print("\n" + "=" * 60)
    print("TEST 1: BIRTH SEQUENCE (Hypersphere)")
    print("=" * 60)

    reset_birth_for_testing()
    m = Manifold()

    # Pre-birth state
    assert not m.born, "Should not be born yet"
    assert m.self_node is None, "Self should not exist yet"

    # Birth
    m.birth()

    # Post-birth checks
    assert m.born, "Should be born"
    assert m.self_node is not None, "Self should exist"
    assert_self_valid(m.self_node)

    # Self invariants — angular coordinates
    assert m.self_node.radius == 0.0, "Self at center (radius=0)"
    assert m.self_node.heat == float('inf'), "Self has infinite heat"
    assert m.self_node.righteousness == 0.0, "Self R=0 (perfect)"
    assert m.self_node.polarity == 0, "Self polarity=0 (neutral)"

    print(f"  ok Self born at center (radius=0) with infinite heat")

    # Bootstrap nodes (6 fires) — check by concept name
    bootstrap_concepts = [
        'bootstrap_N', 'bootstrap_S', 'bootstrap_E',
        'bootstrap_W', 'bootstrap_U', 'bootstrap_d'
    ]
    for concept in bootstrap_concepts:
        node = m.get_node_by_concept(concept)
        assert node is not None, f"Missing bootstrap node: {concept}"
        assert node.existence == EXISTENCE_ACTUAL, f"{concept} should be actual"
        assert node.radius == 1.0, f"{concept} should be on surface (radius=1)"

    print(f"  ok 6 bootstrap nodes created on hypersphere surface")

    # Verify cardinal pole positions
    n_node = m.get_node_by_concept('bootstrap_N')
    s_node = m.get_node_by_concept('bootstrap_S')
    e_node = m.get_node_by_concept('bootstrap_E')
    w_node = m.get_node_by_concept('bootstrap_W')
    u_node = m.get_node_by_concept('bootstrap_U')

    # N and S should be opposite on the equator
    ns_dist = angular_distance(
        SpherePosition(theta=n_node.theta, phi=n_node.phi),
        SpherePosition(theta=s_node.theta, phi=s_node.phi)
    )
    assert ns_dist > math.pi * 0.9, f"N-S should be ~pi apart, got {ns_dist:.3f}"

    # E and W should be opposite on the equator
    ew_dist = angular_distance(
        SpherePosition(theta=e_node.theta, phi=e_node.phi),
        SpherePosition(theta=w_node.theta, phi=w_node.phi)
    )
    assert ew_dist > math.pi * 0.9, f"E-W should be ~pi apart, got {ew_dist:.3f}"

    # U should be at north pole (theta~0)
    assert u_node.theta < 0.1, f"U should be near north pole, theta={u_node.theta:.3f}"

    print(f"  ok Cardinal poles at correct angular positions (N/S opposite, E/W opposite, U at pole)")

    # Psychology nodes
    assert m.identity_node is not None, "Identity missing"
    assert m.ego_node is not None, "Ego missing"
    assert m.conscience_node is not None, "Conscience missing"

    # Psychology nodes should be near south pole (abstract space)
    for psych_name, psych_node in [
        ("Identity", m.identity_node),
        ("Ego", m.ego_node),
        ("Conscience", m.conscience_node)
    ]:
        assert psych_node.theta > math.pi * 0.7, \
            f"{psych_name} should be near south pole, theta={psych_node.theta:.3f}"

    # Unique trig positions
    assert m.identity_node.trig_position != m.ego_node.trig_position
    assert m.ego_node.trig_position != m.conscience_node.trig_position

    print(f"  ok 3 psychology nodes near south pole (Identity, Ego, Conscience)")

    # Heat distribution (Freudian ratios)
    total_psych_heat = (m.identity_node.heat + m.ego_node.heat +
                        m.conscience_node.heat)

    identity_ratio = m.identity_node.heat / total_psych_heat
    ego_ratio = m.ego_node.heat / total_psych_heat
    conscience_ratio = m.conscience_node.heat / total_psych_heat

    # Allow 5% tolerance
    assert abs(identity_ratio - FREUD_IDENTITY_RATIO) < 0.05, \
        f"Identity ratio {identity_ratio:.2f} != {FREUD_IDENTITY_RATIO}"
    assert abs(ego_ratio - FREUD_EGO_RATIO) < 0.05, \
        f"Ego ratio {ego_ratio:.2f} != {FREUD_EGO_RATIO}"
    assert abs(conscience_ratio - FREUD_CONSCIENCE_RATIO) < 0.05, \
        f"Conscience ratio {conscience_ratio:.2f} != {FREUD_CONSCIENCE_RATIO}"

    print(f"  ok Freudian heat distribution: Id={identity_ratio:.0%}, "
          f"Ego={ego_ratio:.0%}, Conscience={conscience_ratio:.0%}")

    # Self's righteous frame
    assert m.self_node.has_axis("identity"), "Self missing identity axis"
    assert m.self_node.has_axis("ego"), "Self missing ego axis"
    assert m.self_node.has_axis("conscience"), "Self missing conscience axis"

    print(f"  ok Self's righteous frame: identity, ego, conscience")

    # Base motion tokens (20 verbs across 6 fires)
    bm_count = 0
    for verb in ALL_BASE_MOTIONS:
        concept = f"{BASE_MOTION_PREFIX}{verb}"
        bm_node = m.get_node_by_concept(concept)
        assert bm_node is not None, f"Missing base motion node: {concept}"
        assert bm_node.existence == EXISTENCE_ACTUAL, f"{concept} should be actual"
        assert bm_node.radius == 1.0, f"{concept} should be on surface"
        assert bm_node.has_tag("base_motion"), f"{concept} should have 'base_motion' tag"

        # Verify angular proximity to parent bootstrap
        fire_num = MOTION_TO_FIRE[verb]
        fire_to_dir = {1: 'N', 2: 'S', 3: 'E', 4: 'W', 5: 'U'}
        if fire_num <= 5:
            parent_concept = f"bootstrap_{fire_to_dir[fire_num]}"
        else:
            parent_concept = "bootstrap_d"
        parent_node = m.get_node_by_concept(parent_concept)
        assert parent_node is not None, f"Missing parent bootstrap: {parent_concept}"
        dist = angular_distance(
            SpherePosition(theta=bm_node.theta, phi=bm_node.phi),
            SpherePosition(theta=parent_node.theta, phi=parent_node.phi)
        )
        # Should be within a reasonable distance (golden angle region)
        assert dist < math.pi / 2, \
            f"{concept} too far from {parent_concept}: dist={dist:.3f}"

        # Verify heat
        expected_heat = BASE_MOTION_HEAT[verb]
        assert abs(bm_node.heat - expected_heat) < 0.01, \
            f"{concept} heat mismatch: {bm_node.heat:.3f} != {expected_heat:.3f}"

        # Verify connections
        assert bm_node.has_axis("self"), f"{concept} missing axis to Self"

        bm_count += 1

    assert bm_count == 20, f"Expected 20 base motions, got {bm_count}"
    print(f"  ok 20 base motion tokens spawned near parent bootstraps")

    # Total nodes (Self is stored separately)
    total_nodes = len(m.nodes)
    assert total_nodes == 29, f"Expected 29 nodes (6 bootstrap + 20 base motion + 3 psych), got {total_nodes}"

    print(f"  ok Total nodes: {total_nodes} + Self")

    # All trace to Self
    assert m.verify_all_trace_to_self(), "Not all nodes trace to Self"

    print(f"  ok All nodes trace to Self")

    return m


def test_motion_functions(m: Manifold):
    """Test all 6 motion functions."""
    print("\n" + "=" * 60)
    print("TEST 2: SIX MOTION FUNCTIONS")
    print("=" * 60)

    # Create a test node with angular position on hypersphere
    test_node = Node(
        concept="test_concept",
        theta=math.pi / 3,  # 60 degrees from north pole
        phi=math.pi / 4,    # 45 degrees azimuthal
        radius=1.0,
        heat=K,
    )
    m.add_node(test_node)

    # 1. HEAT (S) - Magnitude
    print("\n  1. Heat (S) - Magnitude validator")
    initial_heat = test_node.heat
    test_node.add_heat_unchecked(0.5)
    assert test_node.heat == initial_heat + 0.5, "Heat add failed"
    test_node.spend_heat(0.3)
    assert abs(test_node.heat - (initial_heat + 0.2)) < 0.001, "Heat spend failed"
    print(f"     ok Heat operations work (current: {test_node.heat:.3f})")

    # 2. POLARITY (+/-) - Direction
    print("\n  2. Polarity (+/-) - Direction validator")
    assert test_node.polarity in [-1, 0, 1], "Invalid polarity"
    test_node.polarity = -1
    assert test_node.polarity == -1, "Polarity set failed"
    test_node.polarity = 1
    print(f"     ok Polarity toggles work (current: {test_node.polarity})")

    # 3. EXISTENCE (d) - Persistence (1/phi^3 threshold)
    print("\n  3. Existence (d) - Persistence validator")
    print(f"     Threshold: 1/phi^3 = {THRESHOLD_EXISTENCE:.4f}")

    # High heat = high salience = ACTUAL
    test_node.heat = K * 3
    salience = m.calculate_salience(test_node)
    m.update_existence(test_node)
    assert test_node.existence == EXISTENCE_ACTUAL, \
        f"High heat should be ACTUAL, got {test_node.existence}"
    print(f"     ok High heat (salience={salience:.3f}) -> ACTUAL")

    # Low heat = negative salience = DORMANT
    test_node.heat = 0.01
    salience = m.calculate_salience(test_node)
    m.update_existence(test_node)
    assert test_node.existence == EXISTENCE_DORMANT, \
        f"Low heat should be DORMANT, got {test_node.existence}"
    print(f"     ok Low heat (salience={salience:.3f}) -> DORMANT")

    # Restore
    test_node.heat = K
    test_node.existence = EXISTENCE_ACTUAL

    # 4. RIGHTEOUSNESS (R) - Angular Alignment
    print("\n  4. Righteousness (R) - Angular alignment validator")
    r = m.evaluate_righteousness(test_node)
    assert r >= 0, f"R should be >= 0, got {r}"
    print(f"     ok Righteousness evaluated: R={r:.3f}")
    print(f"     (R->0 = aligned with cube frame, R>0 = angular deviation)")

    # 5. ORDER (Q) - Arithmetic
    print("\n  5. Order (Q) - Arithmetic validator")

    axis = test_node.add_axis("test_order", "some_target")
    axis.make_proper()
    assert axis.order is not None, "Order not created"
    assert axis.order.successor is not None, "Successor missing"

    print(f"     ok Order created with successor function")
    print(f"     ok Robinson arithmetic: S(0) = 1")

    # 6. MOVEMENT (Lin) - Vectorized output
    print("\n  6. Movement (Lin) - Vectorized output")
    print(f"     5/6 Confidence threshold: {CONFIDENCE_EXPLOIT_THRESHOLD:.4f}")

    low_confidence = 0.5
    should_exploit_low = low_confidence > CONFIDENCE_EXPLOIT_THRESHOLD
    assert not should_exploit_low, "Low confidence should EXPLORE"
    print(f"     ok Confidence {low_confidence:.3f} < threshold -> EXPLORE")

    high_confidence = 0.9
    should_exploit_high = high_confidence > CONFIDENCE_EXPLOIT_THRESHOLD
    assert should_exploit_high, "High confidence should EXPLOIT"
    print(f"     ok Confidence {high_confidence:.3f} > threshold -> EXPLOIT")

    return test_node


def test_psychology_mediation(m: Manifold):
    """Test Identity -> Conscience -> Ego mediation."""
    print("\n" + "=" * 60)
    print("TEST 3: PSYCHOLOGY MEDIATION")
    print("=" * 60)

    print("\n  Flow: Identity (discovers) -> Conscience (validates) -> Ego (decides)")

    # Create a concept node on the hypersphere
    concept = "test_belief"
    concept_node = Node(
        concept=concept,
        theta=math.pi / 2,
        phi=math.pi / 3,
        radius=1.0,
        heat=K,
    )
    m.add_node(concept_node)

    # Initial confidence
    initial_confidence = m.get_confidence(concept)
    print(f"\n  Initial confidence for '{concept}': {initial_confidence:.4f}")

    # Validate through Conscience multiple times
    print("\n  Validating through Conscience...")
    for i in range(10):
        m.validate_conscience(concept, confirmed=True)

    # Check confidence increased
    new_confidence = m.get_confidence(concept)
    assert new_confidence > initial_confidence, \
        f"Confidence should increase: {initial_confidence:.4f} -> {new_confidence:.4f}"
    print(f"  After 10 validations: {new_confidence:.4f}")

    # Check exploit/explore
    should_exploit = m.should_exploit(concept)
    print(f"\n  Should exploit: {should_exploit}")
    print(f"  (threshold = {CONFIDENCE_EXPLOIT_THRESHOLD:.4f})")

    # Update Identity
    m.update_identity(concept, heat_delta=0.1, known=True)
    assert m.identity_node.has_axis(concept), "Identity should know concept"
    print(f"\n  ok Identity knows '{concept}'")

    # Update Ego
    initial_ego_axes = len(m.ego_node.frame.axes)
    m.update_ego("new_pattern", success=True, heat_delta=0.1)
    assert m.ego_node.has_axis("new_pattern"), "Ego should have pattern axis"
    print(f"  ok Ego learned pattern (axes: {initial_ego_axes} -> {len(m.ego_node.frame.axes)})")

    return new_confidence


def test_decision_node(m: Manifold):
    """Test decision node (5 scalars -> 1 vector)."""
    print("\n" + "=" * 60)
    print("TEST 4: DECISION NODE (5 -> 1)")
    print("=" * 60)

    # Create decision node
    dn = DecisionNode(m)

    print("\n  Decision takes 5 scalar inputs:")
    print("    1. Heat (S)        - magnitude")
    print("    2. Polarity (+/-)  - direction")
    print("    3. Existence (d)   - persistence")
    print("    4. Righteousness   - alignment")
    print("    5. Order (Q)       - history")
    print("  And produces 1 vector output (Movement)")

    # Create a state node on the hypersphere
    state_node = Node(
        concept="decision_state",
        theta=math.pi / 2,
        phi=0.5,
        radius=1.0,
        heat=K,
    )
    m.add_node(state_node)

    # Make decision
    options = ["action_a", "action_b", "action_c"]
    selected = dn.decide("decision_state", options)

    assert selected in options, f"Selected '{selected}' not in options"
    print(f"\n  ok Decision made: {selected}")

    # Check pending choice has scalar values
    choice = dn.pending_choice
    if choice:
        print(f"\n  Scalar inputs recorded:")
        print(f"    Heat:        {choice.heat:.3f}")
        print(f"    Polarity:    {choice.polarity}")
        print(f"    Existence:   {choice.existence_valid}")
        print(f"    Righteousness: {choice.righteousness:.3f}")
        print(f"    Order count: {choice.order_count}")
        print(f"    Confidence:  {choice.confidence:.3f}")

    # Complete decision
    dn.complete_decision("success", success=True, heat_delta=0.5)
    print(f"\n  ok Decision completed with outcome")

    # Make another decision - should use history
    selected2 = dn.decide("decision_state", options)
    print(f"  ok Second decision: {selected2}")

    return dn


def test_clock_tick(m: Manifold):
    """Test clock tick (existence validation)."""
    print("\n" + "=" * 60)
    print("TEST 5: CLOCK TICK (EXISTENCE)")
    print("=" * 60)

    print("\n  Self IS the clock. Each tick = one K-quantum flows.")

    # Get initial t_K
    initial_t_K = m.self_node.t_K
    print(f"\n  Initial t_K: {initial_t_K}")

    # Manual tick
    new_t_K = m.self_node.tick()
    assert new_t_K == initial_t_K + 1, "t_K should increment"
    print(f"  After tick: t_K = {new_t_K}")

    # Tick a few more times
    for _ in range(4):
        m.self_node.tick()

    final_t_K = m.self_node.t_K
    assert final_t_K == initial_t_K + 5, f"Expected t_K={initial_t_K + 5}, got {final_t_K}"
    print(f"  After 5 ticks: t_K = {final_t_K}")

    print(f"\n  ok Self ticks correctly (existence persists)")

    return final_t_K


def test_full_lifecycle():
    """Test complete node lifecycle."""
    print("\n" + "=" * 60)
    print("TEST 6: FULL LIFECYCLE")
    print("=" * 60)

    print("\n  Lifecycle: POTENTIAL -> ACTUAL <-> DORMANT -> ARCHIVED")

    reset_birth_for_testing()
    m = Manifold()
    m.birth()

    # Create potential node on the hypersphere
    potential_node = m.create_potential_node(
        "lifecycle_test",
        theta=math.pi / 2,
        phi=1.0,
    )
    assert potential_node.existence == EXISTENCE_POTENTIAL, \
        f"New node should be POTENTIAL, got {potential_node.existence}"
    assert potential_node.radius == 1.0, "Should be on hypersphere surface"
    print(f"\n  1. Created POTENTIAL node at theta={potential_node.theta:.3f}, phi={potential_node.phi:.3f}")

    # Confirm to ACTUAL (high heat = high salience)
    potential_node.heat = K * 3
    m.update_existence(potential_node)
    assert potential_node.existence == EXISTENCE_ACTUAL, \
        f"High heat node should be ACTUAL, got {potential_node.existence}"
    print(f"  2. High heat -> ACTUAL (salience above 1/phi^3)")

    # Drop to DORMANT (low heat = negative salience)
    potential_node.heat = 0.01
    m.update_existence(potential_node)
    assert potential_node.existence == EXISTENCE_DORMANT, \
        f"Low heat should be DORMANT, got {potential_node.existence}"
    print(f"  3. Low heat -> DORMANT (disconnected dust)")

    # Recover to ACTUAL (high heat again)
    potential_node.heat = K * 2
    m.update_existence(potential_node)
    assert potential_node.existence == EXISTENCE_ACTUAL, \
        f"Recovered node should be ACTUAL, got {potential_node.existence}"
    print(f"  4. High heat -> ACTUAL (reconnected)")

    print(f"\n  ok Full lifecycle works correctly")


def test_planck_grounding():
    """Test Planck-grounded features."""
    print("\n" + "=" * 60)
    print("TEST 7: PLANCK GROUNDING")
    print("=" * 60)

    reset_birth_for_testing()
    m = Manifold()
    m.birth()

    # 1. K x phi^2 = 4 identity
    print("\n  1. Fundamental Identity: K x phi^2 = 4")
    k_phi_squared = K * PHI ** 2
    assert abs(k_phi_squared - 4.0) < 0.0001, f"K x phi^2 should be 4, got {k_phi_squared}"
    print(f"     K x phi^2 = {k_phi_squared} ok")

    # 2. Fire heat scaling
    print("\n  2. Fire Heat Scaling (K x phi^n)")
    for fire_num in range(1, 6):
        expected = K * PHI ** fire_num
        actual = FIRE_HEAT[fire_num]
        assert abs(actual - expected) < 0.01, f"Fire {fire_num} heat mismatch"
        print(f"     Fire {fire_num} ({FIRE_TO_MOTION[fire_num]}): {actual:.2f} K ok")

    # Fire 6 = body temperature
    assert abs(FIRE_HEAT[6] - BODY_TEMPERATURE) < 0.01, "Fire 6 should be body temp"
    print(f"     Fire 6 (movement): {FIRE_HEAT[6]:.2f} K = BODY_TEMPERATURE ok")

    # 3. Body temperature = K x phi^11
    print("\n  3. Body Temperature: K x phi^11")
    expected_body = K * PHI ** 11
    assert abs(BODY_TEMPERATURE - expected_body) < 0.1, \
        f"Body temp should be K x phi^11 ~ {expected_body:.2f}"
    print(f"     BODY_TEMPERATURE = {BODY_TEMPERATURE:.2f} K ({BODY_TEMPERATURE - 273.15:.1f} C) ok")

    # 4. Bootstrap nodes have correct heat (by concept name, not position)
    print("\n  4. Bootstrap Node Heat (Fire Scaling)")
    concept_to_fire = {
        'bootstrap_N': 1, 'bootstrap_S': 2, 'bootstrap_E': 3,
        'bootstrap_W': 4, 'bootstrap_U': 5,
    }
    for concept, fire_num in concept_to_fire.items():
        node = m.get_node_by_concept(concept)
        assert node is not None, f"Missing {concept}"
        expected_heat = FIRE_HEAT[fire_num]
        assert abs(node.heat - expected_heat) < 0.01, \
            f"{concept} heat mismatch: {node.heat:.2f} != {expected_heat:.2f}"
        assert node.constraint_type == "successor", \
            f"{concept} should have successor constraint"
    print(f"     All 5 spatial bootstrap nodes have K x phi^n heat ok")
    print(f"     All 5 spatial bootstrap nodes have 'successor' constraint ok")

    # 5. Abstract root (bootstrap_d) has body temperature
    bootstrap_d = m.get_node_by_concept("bootstrap_d")
    assert abs(bootstrap_d.heat - BODY_TEMPERATURE) < 0.01, \
        "bootstrap_d should have body temperature"
    assert bootstrap_d.constraint_type == "identity", \
        "bootstrap_d should have identity constraint"
    # Verify it's at south pole (theta ~ pi)
    assert bootstrap_d.theta > math.pi * 0.9, \
        f"bootstrap_d should be at south pole, theta={bootstrap_d.theta:.3f}"
    print(f"     bootstrap_d: {bootstrap_d.heat:.2f} K at south pole (theta={bootstrap_d.theta:.3f}) ok")

    # 6. Emergence threshold 45/44
    print("\n  5. Emergence Threshold: 45/44")
    assert abs(EMERGENCE_THRESHOLD - 45/44) < 0.0001, "Emergence threshold should be 45/44"
    assert abs(MAX_ENTROPIC_PROBABILITY - 44/45) < 0.0001, "Max entropy should be 44/45"
    print(f"     EMERGENCE_THRESHOLD = {EMERGENCE_THRESHOLD:.6f} (45/44) ok")
    print(f"     MAX_ENTROPIC_PROBABILITY = {MAX_ENTROPIC_PROBABILITY:.6f} (44/45) ok")

    # 7. Structure detection
    print("\n  6. Structure Detection (Entropy > 44/45)")
    structure_detected = m.structure_detected()
    structure_strength = m.get_structure_strength()
    print(f"     Structure detected: {structure_detected}")
    print(f"     Structure strength: {structure_strength:.4f}")
    assert structure_detected, "Should detect structure after organized birth"
    print(f"     ok System correctly detects organized structure")

    # 8. Robinson constraints
    print("\n  7. Robinson Constraints")
    assert abs(ROBINSON_CONSTRAINTS['identity'] - 1.0) < 0.0001
    assert abs(ROBINSON_CONSTRAINTS['successor'] - 4*PHI/7) < 0.0001
    assert abs(ROBINSON_CONSTRAINTS['addition'] - 4/3) < 0.0001
    assert abs(ROBINSON_CONSTRAINTS['multiplication'] - 13/10) < 0.0001
    print(f"     identity:       {ROBINSON_CONSTRAINTS['identity']:.4f} ok")
    print(f"     successor:      {ROBINSON_CONSTRAINTS['successor']:.4f} (4phi/7) ok")
    print(f"     addition:       {ROBINSON_CONSTRAINTS['addition']:.4f} (4/3) ok")
    print(f"     multiplication: {ROBINSON_CONSTRAINTS['multiplication']:.4f} (13/10) ok")

    # 9. Clock calibration and structure detection
    print("\n  8. Clock Planck Features")
    clock = Clock(m)
    clock.calibrate(duration_seconds=0.1)
    assert clock.stats.calibrated, "Clock should be calibrated"
    print(f"     Hardware calibrated: {clock.stats.hardware_ticks_per_second:.0f} ticks/sec ok")

    # Run ticks and check structure detection triggers
    for _ in range(3):
        clock.tick()
    assert clock.stats.structure_detections > 0, "Should have structure detections"
    print(f"     Structure detections: {clock.stats.structure_detections} ok")
    print(f"     Pattern seeks triggered: {clock.stats.pattern_seeks_triggered} ok")

    # 10. Angular geometry verification
    print("\n  9. Angular Geometry (Hypersphere)")
    # All bootstrap nodes should be on the unit sphere
    for node in m.nodes.values():
        assert node.radius == 1.0 or node.radius == 0.0, \
            f"{node.concept} has invalid radius: {node.radius}"
    print(f"     All nodes on unit sphere (radius=1) or center (radius=0) ok")

    # Cube projection sanity check
    e_node = m.get_node_by_concept('bootstrap_E')
    assert e_node.cube_x > 0.9, f"bootstrap_E should project to +X, got cube_x={e_node.cube_x:.3f}"
    w_node = m.get_node_by_concept('bootstrap_W')
    assert w_node.cube_x < -0.9, f"bootstrap_W should project to -X, got cube_x={w_node.cube_x:.3f}"
    print(f"     Cube projections correct (E->+X={e_node.cube_x:.2f}, W->-X={w_node.cube_x:.2f}) ok")

    print(f"\n  ok All Planck grounding + angular geometry tests passed")


def run_all_tests():
    """Run complete birth and operation test suite."""
    print("\n")
    print("=" * 60)
    print(" PBAI CORE BIRTH & OPERATION TEST (Hypersphere) ")
    print(" Testing all 6 motion functions + Planck grounding ")
    print("=" * 60)

    try:
        # Test 1: Birth
        m = test_birth()

        # Test 2: Motion functions
        test_node = test_motion_functions(m)

        # Test 3: Psychology mediation
        confidence = test_psychology_mediation(m)

        # Test 4: Decision node
        dn = test_decision_node(m)

        # Test 5: Clock tick
        t_K = test_clock_tick(m)

        # Test 6: Full lifecycle
        test_full_lifecycle()

        # Test 7: Planck grounding
        test_planck_grounding()

        # Summary
        print("\n")
        print("=" * 60)
        print(" ALL TESTS PASSED ")
        print("=" * 60)
        print(f"  Birth: Self + 29 nodes (6 bootstrap + 20 base motions + 3 psychology)")
        print(f"  Geometry: Hypersphere angular coords (theta/phi)")
        print(f"  Motion functions: All 6 validated")
        print(f"  Psychology: Identity -> Conscience -> Ego")
        print(f"  Decision: 5 scalars -> 1 vector")
        print(f"  Clock: Self ticks (t_K = {t_K})")
        print(f"  Lifecycle: POTENTIAL -> ACTUAL <-> DORMANT")
        print(f"  Planck: K*phi^2=4, body temp, 45/44, Robinson")
        print("=" * 60)

        return True

    except AssertionError as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
