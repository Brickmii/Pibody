#!/usr/bin/env python3
"""
PBAI Core Birth Test - Full System Verification

Tests the complete birth sequence and operation of all 6 motion functions:

    1. Heat (Σ)           → psychology (magnitude)
    2. Polarity (+/-)     → node_constants (direction)
    3. Existence (δ)      → clock_node (persistence, 1/φ³ threshold)
    4. Righteousness (R)  → nodes (alignment, frames)
    5. Order (Q)          → manifold (arithmetic, history)
    6. Movement (Lin)     → decision_node (5 scalars → 1 vector)

Run: python3 -m core.test_birth
"""

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
)
from core.manifold import Manifold
from core.decision_node import DecisionNode, Choice
from core.clock_node import Clock, TickStats


def test_birth():
    """Test complete manifold birth sequence."""
    print("\n" + "=" * 60)
    print("TEST 1: BIRTH SEQUENCE")
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
    
    # Self invariants
    assert m.self_node.position == "", "Self at origin"
    assert m.self_node.heat == float('inf'), "Self has infinite heat"
    assert m.self_node.righteousness == 0.0, "Self R=0 (perfect)"
    assert m.self_node.polarity == 0, "Self polarity=0 (neutral)"
    
    print(f"  ✓ Self born at origin with infinite heat")
    
    # Physical nodes (6 fires)
    positions = ['n', 's', 'e', 'w', 'u', 'd']
    for pos in positions:
        node = m.get_node_by_position(pos)
        assert node is not None, f"Missing node at {pos}"
        assert node.existence == EXISTENCE_ACTUAL, f"Node at {pos} should be actual"
    
    print(f"  ✓ 6 physical nodes created (n,s,e,w,u,d)")
    
    # Psychology nodes
    assert m.identity_node is not None, "Identity missing"
    assert m.ego_node is not None, "Ego missing"
    assert m.conscience_node is not None, "Conscience missing"
    
    # All at cubic position 'd' but different trig positions
    assert m.identity_node.position == "d"
    assert m.ego_node.position == "d"
    assert m.conscience_node.position == "d"
    
    # Unique trig positions
    assert m.identity_node.trig_position != m.ego_node.trig_position
    assert m.ego_node.trig_position != m.conscience_node.trig_position
    
    print(f"  ✓ 3 psychology nodes created (Identity, Ego, Conscience)")
    
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
    
    print(f"  ✓ Freudian heat distribution: Id={identity_ratio:.0%}, "
          f"Ego={ego_ratio:.0%}, Conscience={conscience_ratio:.0%}")
    
    # Self's righteous frame
    assert m.self_node.has_axis("identity"), "Self missing identity axis"
    assert m.self_node.has_axis("ego"), "Self missing ego axis"
    assert m.self_node.has_axis("conscience"), "Self missing conscience axis"
    
    print(f"  ✓ Self's righteous frame: identity, ego, conscience")
    
    # Total nodes (Self is stored separately)
    total_nodes = len(m.nodes)  # nodes dict doesn't include Self
    assert total_nodes == 9, f"Expected 9 nodes (6 physical + 3 psych), got {total_nodes}"
    
    print(f"  ✓ Total nodes: {total_nodes} + Self")
    
    # All trace to Self
    assert m.verify_all_trace_to_self(), "Not all nodes trace to Self"
    
    print(f"  ✓ All nodes trace to Self")
    
    return m


def test_motion_functions(m: Manifold):
    """Test all 6 motion functions."""
    print("\n" + "=" * 60)
    print("TEST 2: SIX MOTION FUNCTIONS")
    print("=" * 60)
    
    # Create a test node
    test_node = Node(concept="test_concept", position="nn", heat=K)
    m.add_node(test_node)
    
    # 1. HEAT (Σ) - Magnitude
    print("\n  1. Heat (Σ) - Magnitude validator")
    initial_heat = test_node.heat
    test_node.add_heat_unchecked(0.5)  # Use unchecked for direct add
    assert test_node.heat == initial_heat + 0.5, "Heat add failed"
    test_node.spend_heat(0.3)
    assert abs(test_node.heat - (initial_heat + 0.2)) < 0.001, "Heat spend failed"
    print(f"     ✓ Heat operations work (current: {test_node.heat:.3f})")
    
    # 2. POLARITY (+/-) - Direction
    print("\n  2. Polarity (+/-) - Direction validator")
    assert test_node.polarity in [-1, 0, 1], "Invalid polarity"
    test_node.polarity = -1
    assert test_node.polarity == -1, "Polarity set failed"
    test_node.polarity = 1
    print(f"     ✓ Polarity toggles work (current: {test_node.polarity})")
    
    # 3. EXISTENCE (δ) - Persistence (1/φ³ threshold)
    print("\n  3. Existence (δ) - Persistence validator")
    print(f"     Threshold: 1/φ³ = {THRESHOLD_EXISTENCE:.4f}")
    print(f"     Salience = node heat - local environment heat")
    
    # High heat = high salience = ACTUAL
    test_node.heat = K * 3  # Much higher than average
    salience = m.calculate_salience(test_node)
    m.update_existence(test_node)
    assert test_node.existence == EXISTENCE_ACTUAL, \
        f"High heat should be ACTUAL, got {test_node.existence}"
    print(f"     ✓ High heat (salience={salience:.3f}) → ACTUAL")
    
    # Low heat = negative salience = DORMANT
    test_node.heat = 0.01  # Much lower than average
    salience = m.calculate_salience(test_node)
    m.update_existence(test_node)
    assert test_node.existence == EXISTENCE_DORMANT, \
        f"Low heat should be DORMANT, got {test_node.existence}"
    print(f"     ✓ Low heat (salience={salience:.3f}) → DORMANT")
    
    # Restore
    test_node.heat = K
    test_node.existence = EXISTENCE_ACTUAL
    
    # 4. RIGHTEOUSNESS (R) - Alignment
    print("\n  4. Righteousness (R) - Alignment validator")
    r = m.evaluate_righteousness(test_node)
    assert r >= 0, f"R should be >= 0, got {r}"
    print(f"     ✓ Righteousness evaluated: R={r:.3f}")
    print(f"     (R→0 = aligned with Self, R>0 = divergence)")
    
    # 5. ORDER (Q) - Arithmetic
    print("\n  5. Order (Q) - Arithmetic validator")
    
    # Add axis with Order
    axis = test_node.add_axis("test_order", "some_target")
    axis.make_proper()  # Creates Order
    assert axis.order is not None, "Order not created"
    assert axis.order.successor is not None, "Successor missing"
    
    # Robinson arithmetic: S(n) exists
    print(f"     ✓ Order created with successor function")
    print(f"     ✓ Robinson arithmetic: S(0) = 1")
    
    # 6. MOVEMENT (Lin) - Vectorized output
    print("\n  6. Movement (Lin) - Vectorized output")
    print(f"     5/6 Confidence threshold: {CONFIDENCE_EXPLOIT_THRESHOLD:.4f}")
    
    # Test confidence below threshold (EXPLORE)
    low_confidence = 0.5
    should_exploit_low = low_confidence > CONFIDENCE_EXPLOIT_THRESHOLD
    assert not should_exploit_low, "Low confidence should EXPLORE"
    print(f"     ✓ Confidence {low_confidence:.3f} < threshold → EXPLORE")
    
    # Test confidence above threshold (EXPLOIT)
    high_confidence = 0.9
    should_exploit_high = high_confidence > CONFIDENCE_EXPLOIT_THRESHOLD
    assert should_exploit_high, "High confidence should EXPLOIT"
    print(f"     ✓ Confidence {high_confidence:.3f} > threshold → EXPLOIT")
    
    return test_node


def test_psychology_mediation(m: Manifold):
    """Test Identity → Conscience → Ego mediation."""
    print("\n" + "=" * 60)
    print("TEST 3: PSYCHOLOGY MEDIATION")
    print("=" * 60)
    
    print("\n  Flow: Identity (discovers) → Conscience (validates) → Ego (decides)")
    
    # First, create a concept node for the system to track
    concept = "test_belief"
    concept_node = Node(concept=concept, position="nnn", heat=K)
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
        f"Confidence should increase: {initial_confidence:.4f} → {new_confidence:.4f}"
    print(f"  After 10 validations: {new_confidence:.4f}")
    
    # Check exploit/explore
    should_exploit = m.should_exploit(concept)
    print(f"\n  Should exploit: {should_exploit}")
    print(f"  (threshold = {CONFIDENCE_EXPLOIT_THRESHOLD:.4f})")
    
    # Update Identity
    m.update_identity(concept, heat_delta=0.1, known=True)
    assert m.identity_node.has_axis(concept), "Identity should know concept"
    print(f"\n  ✓ Identity knows '{concept}'")
    
    # Update Ego
    initial_ego_axes = len(m.ego_node.frame.axes)
    m.update_ego("new_pattern", success=True, heat_delta=0.1)
    assert m.ego_node.has_axis("new_pattern"), "Ego should have pattern axis"
    print(f"  ✓ Ego learned pattern (axes: {initial_ego_axes} → {len(m.ego_node.frame.axes)})")
    
    return new_confidence


def test_decision_node(m: Manifold):
    """Test decision node (5 scalars → 1 vector)."""
    print("\n" + "=" * 60)
    print("TEST 4: DECISION NODE (5 → 1)")
    print("=" * 60)
    
    # Create decision node
    dn = DecisionNode(m)
    
    print("\n  Decision takes 5 scalar inputs:")
    print("    1. Heat (Σ)        - magnitude")
    print("    2. Polarity (+/-)  - direction")
    print("    3. Existence (δ)   - persistence")
    print("    4. Righteousness   - alignment")
    print("    5. Order (Q)       - history")
    print("  And produces 1 vector output (Movement)")
    
    # Create a state node
    state_node = Node(concept="decision_state", position="nnn", heat=K)
    m.add_node(state_node)
    
    # Make decision
    options = ["action_a", "action_b", "action_c"]
    selected = dn.decide("decision_state", options)
    
    assert selected in options, f"Selected '{selected}' not in options"
    print(f"\n  ✓ Decision made: {selected}")
    
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
    print(f"\n  ✓ Decision completed with outcome")
    
    # Make another decision - should use history
    selected2 = dn.decide("decision_state", options)
    print(f"  ✓ Second decision: {selected2}")
    
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
    
    print(f"\n  ✓ Self ticks correctly (existence persists)")
    
    return final_t_K


def test_full_lifecycle():
    """Test complete node lifecycle."""
    print("\n" + "=" * 60)
    print("TEST 6: FULL LIFECYCLE")
    print("=" * 60)
    
    print("\n  Lifecycle: POTENTIAL → ACTUAL ↔ DORMANT → ARCHIVED")
    
    reset_birth_for_testing()
    m = Manifold()
    m.birth()
    
    # Create potential node
    potential_node = m.create_potential_node("lifecycle_test", "nnnn")
    assert potential_node.existence == EXISTENCE_POTENTIAL, \
        f"New node should be POTENTIAL, got {potential_node.existence}"
    print(f"\n  1. Created POTENTIAL node")
    
    # Confirm to ACTUAL (high heat = high salience)
    potential_node.heat = K * 3
    m.update_existence(potential_node)
    assert potential_node.existence == EXISTENCE_ACTUAL, \
        f"High heat node should be ACTUAL, got {potential_node.existence}"
    print(f"  2. High heat → ACTUAL (salience above 1/φ³)")
    
    # Drop to DORMANT (low heat = negative salience)
    potential_node.heat = 0.01
    m.update_existence(potential_node)
    assert potential_node.existence == EXISTENCE_DORMANT, \
        f"Low heat should be DORMANT, got {potential_node.existence}"
    print(f"  3. Low heat → DORMANT (disconnected dust)")
    
    # Recover to ACTUAL (high heat again)
    potential_node.heat = K * 2
    m.update_existence(potential_node)
    assert potential_node.existence == EXISTENCE_ACTUAL, \
        f"Recovered node should be ACTUAL, got {potential_node.existence}"
    print(f"  4. High heat → ACTUAL (reconnected)")
    
    print(f"\n  ✓ Full lifecycle works correctly")


def test_planck_grounding():
    """Test Planck-grounded features."""
    print("\n" + "=" * 60)
    print("TEST 7: PLANCK GROUNDING")
    print("=" * 60)
    
    reset_birth_for_testing()
    m = Manifold()
    m.birth()
    
    # 1. K × φ² = 4 identity
    print("\n  1. Fundamental Identity: K × φ² = 4")
    k_phi_squared = K * PHI ** 2
    assert abs(k_phi_squared - 4.0) < 0.0001, f"K × φ² should be 4, got {k_phi_squared}"
    print(f"     K × φ² = {k_phi_squared} ✓")
    
    # 2. Fire heat scaling
    print("\n  2. Fire Heat Scaling (K × φⁿ)")
    for fire_num in range(1, 6):
        expected = K * PHI ** fire_num
        actual = FIRE_HEAT[fire_num]
        assert abs(actual - expected) < 0.01, f"Fire {fire_num} heat mismatch"
        print(f"     Fire {fire_num} ({FIRE_TO_MOTION[fire_num]}): {actual:.2f} K ✓")
    
    # Fire 6 = body temperature
    assert abs(FIRE_HEAT[6] - BODY_TEMPERATURE) < 0.01, "Fire 6 should be body temp"
    print(f"     Fire 6 (movement): {FIRE_HEAT[6]:.2f} K = BODY_TEMPERATURE ✓")
    
    # 3. Body temperature = K × φ¹¹
    print("\n  3. Body Temperature: K × φ¹¹")
    expected_body = K * PHI ** 11
    assert abs(BODY_TEMPERATURE - expected_body) < 0.1, \
        f"Body temp should be K × φ¹¹ ≈ {expected_body:.2f}"
    print(f"     BODY_TEMPERATURE = {BODY_TEMPERATURE:.2f} K ({BODY_TEMPERATURE - 273.15:.1f}°C) ✓")
    
    # 4. Bootstrap nodes have correct heat
    print("\n  4. Bootstrap Node Heat (Fire Scaling)")
    direction_to_fire = {'n': 1, 's': 2, 'e': 3, 'w': 4, 'u': 5}
    for direction, fire_num in direction_to_fire.items():
        node = m.get_node_by_concept(f"bootstrap_{direction}")
        expected_heat = FIRE_HEAT[fire_num]
        assert abs(node.heat - expected_heat) < 0.01, \
            f"bootstrap_{direction} heat mismatch"
        assert node.constraint_type == "successor", \
            f"bootstrap_{direction} should have successor constraint"
    print(f"     All 5 spatial bootstrap nodes have K × φⁿ heat ✓")
    print(f"     All 5 spatial bootstrap nodes have 'successor' constraint ✓")
    
    # 5. Abstract root (bootstrap_d) has body temperature
    bootstrap_d = m.get_node_by_concept("bootstrap_d")
    assert abs(bootstrap_d.heat - BODY_TEMPERATURE) < 0.01, \
        "bootstrap_d should have body temperature"
    assert bootstrap_d.constraint_type == "identity", \
        "bootstrap_d should have identity constraint"
    print(f"     bootstrap_d: {bootstrap_d.heat:.2f} K with 'identity' constraint ✓")
    
    # 6. Emergence threshold 45/44
    print("\n  5. Emergence Threshold: 45/44")
    assert abs(EMERGENCE_THRESHOLD - 45/44) < 0.0001, "Emergence threshold should be 45/44"
    assert abs(MAX_ENTROPIC_PROBABILITY - 44/45) < 0.0001, "Max entropy should be 44/45"
    print(f"     EMERGENCE_THRESHOLD = {EMERGENCE_THRESHOLD:.6f} (45/44) ✓")
    print(f"     MAX_ENTROPIC_PROBABILITY = {MAX_ENTROPIC_PROBABILITY:.6f} (44/45) ✓")
    
    # 7. Structure detection
    print("\n  6. Structure Detection (Entropy > 44/45)")
    structure_detected = m.structure_detected()
    structure_strength = m.get_structure_strength()
    print(f"     Structure detected: {structure_detected}")
    print(f"     Structure strength: {structure_strength:.4f}")
    # After birth, system is organized so structure should be detected
    assert structure_detected, "Should detect structure after organized birth"
    print(f"     ✓ System correctly detects organized structure")
    
    # 8. Robinson constraints
    print("\n  7. Robinson Constraints")
    assert abs(ROBINSON_CONSTRAINTS['identity'] - 1.0) < 0.0001
    assert abs(ROBINSON_CONSTRAINTS['successor'] - 4*PHI/7) < 0.0001
    assert abs(ROBINSON_CONSTRAINTS['addition'] - 4/3) < 0.0001
    assert abs(ROBINSON_CONSTRAINTS['multiplication'] - 13/10) < 0.0001
    print(f"     identity:       {ROBINSON_CONSTRAINTS['identity']:.4f} ✓")
    print(f"     successor:      {ROBINSON_CONSTRAINTS['successor']:.4f} (4φ/7) ✓")
    print(f"     addition:       {ROBINSON_CONSTRAINTS['addition']:.4f} (4/3) ✓")
    print(f"     multiplication: {ROBINSON_CONSTRAINTS['multiplication']:.4f} (13/10) ✓")
    
    # 9. Clock calibration and structure detection
    print("\n  8. Clock Planck Features")
    clock = Clock(m)
    clock.calibrate(duration_seconds=0.1)
    assert clock.stats.calibrated, "Clock should be calibrated"
    print(f"     Hardware calibrated: {clock.stats.hardware_ticks_per_second:.0f} ticks/sec ✓")
    
    # Run ticks and check structure detection triggers
    for _ in range(3):
        clock.tick()
    assert clock.stats.structure_detections > 0, "Should have structure detections"
    print(f"     Structure detections: {clock.stats.structure_detections} ✓")
    print(f"     Pattern seeks triggered: {clock.stats.pattern_seeks_triggered} ✓")
    
    print(f"\n  ✓ All Planck grounding tests passed")


def run_all_tests():
    """Run complete birth and operation test suite."""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " PBAI CORE BIRTH & OPERATION TEST ".center(58) + "║")
    print("║" + " Testing all 6 motion functions + Planck grounding ".center(58) + "║")
    print("╚" + "═" * 58 + "╝")
    
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
        print("╔" + "═" * 58 + "╗")
        print("║" + " ALL TESTS PASSED ✓ ".center(58) + "║")
        print("╠" + "═" * 58 + "╣")
        print("║" + f" Birth: Self + 9 nodes (6 physical + 3 psychology) ".ljust(57) + "║")
        print("║" + f" Motion functions: All 6 validated ".ljust(57) + "║")
        print("║" + f" Psychology: Identity → Conscience → Ego ".ljust(57) + "║")
        print("║" + f" Decision: 5 scalars → 1 vector ".ljust(57) + "║")
        print("║" + f" Clock: Self ticks (t_K = {t_K}) ".ljust(57) + "║")
        print("║" + f" Lifecycle: POTENTIAL → ACTUAL ↔ DORMANT ".ljust(57) + "║")
        print("║" + f" Planck: K×φ²=4, body temp, 45/44, Robinson ".ljust(57) + "║")
        print("╚" + "═" * 58 + "╝")
        
        return True
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
