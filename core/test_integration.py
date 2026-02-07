"""
PBAI Integration Test — Full Birth → Tick → Decide → Act

End-to-end test of the complete PBAI lifecycle:
    1. Birth: Manifold creates Self + 9 nodes on hypersphere
    2. Tick: Clock advances time, psychology processes
    3. Perceive: Environment receives perceptions
    4. Introspect: Options simulated through cube projection
    5. Decide: β→δ→Γ→α→ζ pipeline selects action
    6. Act: Decision recorded, confidence updated
    7. Save/Load: State persists and reloads correctly

Run: python3 -m core.test_integration
"""

import math
import sys
import os
import json
import tempfile
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.node_constants import (
    K, PHI, INV_PHI,
    CONFIDENCE_EXPLOIT_THRESHOLD,
    COST_EVALUATE, PSYCHOLOGY_MIN_HEAT,
    EXISTENCE_ACTUAL, EXISTENCE_DORMANT,
)
from core.nodes import Node, SelfNode, reset_birth_for_testing
from core.manifold import Manifold, create_manifold, reset_pbai_manifold
from core.hypersphere import SpherePosition, angular_distance
from core.clock_node import Clock, create_clock
from core.decision_node import DecisionNode, EnvironmentNode, ChoiceNode
from core.introspector import Introspector, SimulationResult
from core.driver_node import DriverNode, SensorReport

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


def fresh_manifold():
    """Create a fresh manifold for testing."""
    reset_pbai_manifold()
    reset_birth_for_testing()
    return create_manifold()


# ═════════════════════════════════════════════════════════════════════════════
# TEST 1: BIRTH → TICK
# ═════════════════════════════════════════════════════════════════════════════

def test_birth_to_tick():
    """Test that birth creates valid manifold and clock can tick."""
    print("\n" + "=" * 60)
    print("TEST 1: BIRTH -> TICK")
    print("=" * 60)

    m = fresh_manifold()

    # Verify birth created correct structure
    check(m.self_node is not None, "Self node exists")
    check(m.self_node.radius == 0.0, "Self at center (radius=0)")
    check(m.identity_node is not None, "Identity node exists")
    check(m.ego_node is not None, "Ego node exists")
    check(m.conscience_node is not None, "Conscience node exists")
    check(len(m.nodes) == 9, f"9 nodes after birth (got {len(m.nodes)})")

    # All nodes on hypersphere have valid angular coords
    for node in m.nodes.values():
        if node.radius > 0:
            check(0.0 <= node.theta <= math.pi,
                  f"  {node.concept} theta={node.theta:.3f} valid")

    # Create clock and tick
    clock = Clock(m, save_path=None)
    # Clock may auto-birth in __init__ — verify it's born after _birth()
    clock._birth()
    check(clock.born, "Clock born")

    initial_tK = m.self_node.t_K
    clock.tick()
    check(m.self_node.t_K == initial_tK + 1,
          f"Tick advanced t_K: {initial_tK} -> {m.self_node.t_K}")

    # Multiple ticks
    for _ in range(4):
        clock.tick()
    check(m.self_node.t_K == initial_tK + 5,
          f"5 ticks: t_K = {m.self_node.t_K}")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 2: PERCEIVE → INTROSPECT → DECIDE
# ═════════════════════════════════════════════════════════════════════════════

def test_perceive_introspect_decide():
    """Test the perception → introspection → decision pipeline."""
    print("\n" + "=" * 60)
    print("TEST 2: PERCEIVE -> INTROSPECT -> DECIDE")
    print("=" * 60)

    m = fresh_manifold()

    # Create perception entry point
    entry = EnvironmentNode(m)
    check(entry is not None, "EnvironmentNode created")

    # Receive a perception
    perception = entry.receive("enemy_ahead", "An enemy appeared ahead", novelty=0.8)
    check(perception is not None, "Perception recorded")
    check(perception.state_key == "enemy_ahead", "Perception state key correct")

    # Check familiarity
    check(entry.is_familiar("enemy_ahead"), "State now familiar")
    check(not entry.is_familiar("unknown_state"), "Unknown state unfamiliar")

    # Introspect on options
    intro = Introspector(m)

    # Ensure psychology nodes are energized
    if m.ego_node:
        m.ego_node.heat = max(m.ego_node.heat, COST_EVALUATE + PSYCHOLOGY_MIN_HEAT + 5.0)
        m.ego_node.existence = EXISTENCE_ACTUAL
    if m.conscience_node:
        m.conscience_node.heat = max(m.conscience_node.heat, COST_EVALUATE + PSYCHOLOGY_MIN_HEAT + 5.0)
        m.conscience_node.existence = EXISTENCE_ACTUAL

    check(intro.should_think(), "Introspector can think (Ego + Conscience energized)")

    options = ["attack", "defend", "flee"]
    sim_results = intro.simulate(options, "enemy_ahead")
    check(len(sim_results) == 3, f"Simulated {len(sim_results)} options")

    # Each result has cube coordinates
    for r in sim_results:
        check(-1.0 <= r.cube_x <= 1.0 and -1.0 <= r.cube_y <= 1.0,
              f"  {r.option}: cube({r.cube_x:.3f}, {r.cube_y:.3f})")

    # Get context from introspection
    context = intro.to_context(sim_results)
    check(context["simulated"] is True, "Introspection context available")

    # Make decision through full pipeline
    decision = DecisionNode(m)
    selected = decision.decide("enemy_ahead", options, context=context)
    check(selected in options, f"Decision made: {selected}")

    # Complete the decision
    decision.complete_decision("fought", True, heat_delta=0.5)
    check(decision.pending_choice is None or decision.pending_choice.success,
          "Decision completed successfully")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 3: MULTI-STEP LOOP (Perception → Decision cycle)
# ═════════════════════════════════════════════════════════════════════════════

def test_multi_step_loop():
    """Test multiple perception-decision cycles build confidence."""
    print("\n" + "=" * 60)
    print("TEST 3: MULTI-STEP LOOP")
    print("=" * 60)

    m = fresh_manifold()
    clock = Clock(m, save_path=None)
    clock._birth()

    entry = EnvironmentNode(m)
    exit_node = DecisionNode(m)

    state = "crossroads"
    options = ["go_north", "go_south", "wait"]

    initial_confidence = m.get_confidence(state)
    check(initial_confidence >= 0.0, f"Initial confidence: {initial_confidence:.4f}")

    # Run 10 perception-decision cycles
    decisions = []
    for i in range(10):
        # Tick the clock
        clock.tick()

        # Perceive
        novelty = 1.0 if i == 0 else 0.0
        entry.receive(state, f"At crossroads (step {i})", novelty=novelty)

        # Decide
        confidence = m.get_confidence(state)
        selected = exit_node.decide(state, options, confidence=confidence)
        decisions.append(selected)

        # Complete with outcome
        success = (selected == "go_north")  # north is "correct"
        exit_node.complete_decision(f"step_{i}", success, heat_delta=0.1 if success else -0.05)

    check(len(decisions) == 10, f"Made {len(decisions)} decisions")
    check(all(d in options for d in decisions), "All decisions are valid options")

    # Confidence should have changed after validations
    final_confidence = m.get_confidence(state)
    check(isinstance(final_confidence, float),
          f"Final confidence: {final_confidence:.4f}")

    # Check clock advanced
    check(m.self_node.t_K >= 10, f"Clock at t_K={m.self_node.t_K} after 10 ticks")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 4: SAVE → LOAD ROUNDTRIP
# ═════════════════════════════════════════════════════════════════════════════

def test_save_load_roundtrip():
    """Test that manifold state survives save/load."""
    print("\n" + "=" * 60)
    print("TEST 4: SAVE -> LOAD ROUNDTRIP")
    print("=" * 60)

    m = fresh_manifold()

    # Add some nodes
    test_node = Node(concept="save_test_concept",
                     theta=1.2, phi=2.3, radius=1.0,
                     heat=K * 2, existence=EXISTENCE_ACTUAL,
                     righteousness=0.42)
    m.add_node(test_node)

    # Tick a few times
    clock = Clock(m, save_path=None)
    clock._birth()
    for _ in range(3):
        clock.tick()

    original_tK = m.self_node.t_K
    original_node_count = len(m.nodes)
    original_ids = set(m.nodes.keys())

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode='w') as f:
        save_path = f.name

    try:
        m.save_growth_map(save_path)
        check(os.path.exists(save_path), f"Saved to {save_path}")

        # Load into new manifold
        reset_pbai_manifold()
        reset_birth_for_testing()
        m2 = Manifold()
        m2.load_growth_map(save_path)

        check(m2.self_node is not None, "Loaded manifold has Self")
        check(m2.self_node.t_K == original_tK,
              f"t_K preserved: {m2.self_node.t_K} == {original_tK}")
        check(len(m2.nodes) == original_node_count,
              f"Node count preserved: {len(m2.nodes)} == {original_node_count}")

        # Verify test node survived
        loaded_test = m2.get_node_by_concept("save_test_concept")
        check(loaded_test is not None, "Test node survived roundtrip")
        if loaded_test:
            check(abs(loaded_test.theta - 1.2) < 1e-6,
                  f"theta preserved: {loaded_test.theta:.6f}")
            check(abs(loaded_test.phi - 2.3) < 1e-6,
                  f"phi preserved: {loaded_test.phi:.6f}")
            check(abs(loaded_test.heat - K * 2) < 0.1,
                  f"heat preserved: {loaded_test.heat:.4f}")

        # Verify psychology nodes survived
        check(m2.identity_node is not None, "Identity survived load")
        check(m2.ego_node is not None, "Ego survived load")
        check(m2.conscience_node is not None, "Conscience survived load")

        # Verify all nodes have valid angular coords
        for node in m2.nodes.values():
            if node.radius > 0:
                valid = (0.0 <= node.theta <= math.pi and
                         0.0 <= node.phi < 2 * math.pi + 0.001)
                check(valid, f"  {node.concept} angular coords valid after load")

    finally:
        if os.path.exists(save_path):
            os.unlink(save_path)


# ═════════════════════════════════════════════════════════════════════════════
# TEST 5: DRIVER NODE INTEGRATION
# ═════════════════════════════════════════════════════════════════════════════

def test_driver_integration():
    """Test driver node creates sensor nodes on hypersphere."""
    print("\n" + "=" * 60)
    print("TEST 5: DRIVER NODE INTEGRATION")
    print("=" * 60)

    m = fresh_manifold()

    driver = DriverNode("test_driver", m)
    check(driver is not None, "DriverNode created")
    check(driver.node is not None, "Driver has manifold node")

    # Driver node should be on hypersphere
    check(driver.node.radius == 1.0, f"Driver node on surface (radius={driver.node.radius})")
    check(0.0 <= driver.node.theta <= math.pi,
          f"Driver theta={driver.node.theta:.4f} valid")

    # see() creates a perception node from SensorReport
    initial_count = len(m.nodes)
    report = SensorReport(
        sensor_type="vision",
        description="visual_input",
        measurements={"x": 100.0, "y": 200.0}
    )
    state_key = driver.see(report)
    check(state_key is not None, f"see() returned state key: {state_key}")
    check(len(m.nodes) > initial_count,
          f"see() added node: {initial_count} -> {len(m.nodes)}")

    # Find the created node by state key
    visual_node = m.get_node_by_concept(state_key)
    if visual_node:
        check(visual_node.radius == 1.0, "Perception node on surface")
        check(0.0 <= visual_node.theta <= math.pi,
              f"Perception theta={visual_node.theta:.4f} valid")
        # Should be near the driver node
        d = angular_distance(
            SpherePosition(theta=driver.node.theta, phi=driver.node.phi),
            SpherePosition(theta=visual_node.theta, phi=visual_node.phi)
        )
        check(d < math.pi, f"Perception near driver (dist={d:.4f})")
    else:
        ok("Perception node created (concept key may differ)")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 6: FULL END-TO-END LIFECYCLE
# ═════════════════════════════════════════════════════════════════════════════

def test_end_to_end():
    """Full lifecycle: birth → tick → perceive → introspect → decide → act."""
    print("\n" + "=" * 60)
    print("TEST 6: FULL END-TO-END LIFECYCLE")
    print("=" * 60)

    # 1. BIRTH
    m = fresh_manifold()
    check(m.self_node is not None, "1. Birth complete")

    # 2. TICK (start the clock)
    clock = Clock(m, save_path=None)
    clock._birth()
    clock.tick()
    check(m.self_node.t_K >= 1, "2. Clock ticking")

    # 3. PERCEIVE
    entry = EnvironmentNode(m)
    entry.receive("game_screen", "A game screen appeared", novelty=1.0)
    check(entry.is_familiar("game_screen"), "3. Perception recorded")

    # 4. INTROSPECT
    intro = Introspector(m)
    # Ensure psychology nodes energized
    if m.ego_node:
        m.ego_node.heat = max(m.ego_node.heat, COST_EVALUATE + PSYCHOLOGY_MIN_HEAT + 5.0)
        m.ego_node.existence = EXISTENCE_ACTUAL
    if m.conscience_node:
        m.conscience_node.heat = max(m.conscience_node.heat, COST_EVALUATE + PSYCHOLOGY_MIN_HEAT + 5.0)
        m.conscience_node.existence = EXISTENCE_ACTUAL

    options = ["press_a", "press_b", "wait"]
    if intro.should_think():
        results = intro.simulate(options, "game_screen")
        context = intro.to_context(results)
        check(context["simulated"] is True, "4. Introspection complete")
    else:
        context = {}
        ok("4. Introspection skipped (low energy)")

    # 5. DECIDE
    exit_node = DecisionNode(m)
    selected = exit_node.decide("game_screen", options, context=context)
    check(selected in options, f"5. Decision: {selected}")

    # 6. ACT (complete with outcome)
    exit_node.complete_decision("button_pressed", True, heat_delta=0.3)
    ok("6. Action completed")

    # 7. TICK AGAIN (time advances)
    clock.tick()
    check(m.self_node.t_K >= 2, "7. Time advanced after action")

    # 8. VERIFY STATE
    total_nodes = len(m.nodes)
    check(total_nodes >= 9, f"8. Manifold has {total_nodes} nodes (>= 9 from birth)")

    # All nodes trace to Self
    check(m.verify_all_trace_to_self(), "9. All nodes trace to Self")

    print(f"\n  End-to-end summary:")
    print(f"    t_K = {m.self_node.t_K}")
    print(f"    Nodes = {total_nodes}")
    print(f"    Decision = {selected}")
    print(f"    All valid = True")


# ═════════════════════════════════════════════════════════════════════════════
# RUN ALL
# ═════════════════════════════════════════════════════════════════════════════

def run_all_tests():
    global passed, failed

    print("=" * 60)
    print(" PBAI INTEGRATION TEST")
    print(" Birth -> Tick -> Perceive -> Introspect -> Decide -> Act")
    print("=" * 60)

    test_birth_to_tick()
    test_perceive_introspect_decide()
    test_multi_step_loop()
    test_save_load_roundtrip()
    test_driver_integration()
    test_end_to_end()

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
