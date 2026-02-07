#!/usr/bin/env python3
"""
Test the β→δ→Γ→α→ζ Processing Cycle

Tests Functions 7-11 of Motion Calendar:
- β (Euler Beta): Path superposition
- δ (Dirac Delta): Wave function collapse  
- Γ (Gamma): Entropy counting
- α (Fine-structure): Coupling strength
- ζ (Zeta): Significance normalization
"""

import sys
import os
import cmath
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.manifold import Manifold
from core.decision_node import DecisionNode
from core.node_constants import K, PHI, CONFIDENCE_EXPLOIT_THRESHOLD

def test_processing_cycle():
    """Test the full β→δ→Γ→α→ζ processing cycle."""
    
    print("=" * 60)
    print("TESTING β→δ→Γ→α→ζ PROCESSING CYCLE")
    print("=" * 60)
    
    # Create manifold with psychology nodes
    manifold = Manifold("test_manifold")
    manifold.birth()
    
    # Verify psychology nodes exist
    print("\n1. Psychology Nodes:")
    print(f"   Identity: heat={manifold.identity_node.heat:.2f}K")
    print(f"   Ego:      heat={manifold.ego_node.heat:.2f}K")
    print(f"   Conscience: heat={manifold.conscience_node.heat:.2f}K")
    
    # Create decision node
    decision = DecisionNode(manifold)
    print(f"\n2. Decision Node created: {decision.name}")
    
    # Test options
    state_key = "test_state"
    options = ["hit", "stand", "double"]
    
    print(f"\n3. Test State: {state_key}")
    print(f"   Options: {options}")
    
    # Test β: Create superposition
    print("\n4. β Processing (Identity) - Superposition:")
    superposition = decision._create_superposition(options, state_key)
    print(f"   Paths: {superposition['paths']}")
    print(f"   Weights: {[f'{w:.4f}' for w in superposition['weights']]}")
    psi = superposition['psi']
    print(f"   Ψ amplitude: |ψ|={abs(psi):.4f}, phase={cmath.phase(psi):.4f} rad")
    
    # Test Γ: Count arrangements
    print("\n5. Γ Processing (Conscience) - Arrangements:")
    gamma_scores = decision._get_gamma_scores(options, state_key)
    print(f"   Gamma scores: {gamma_scores}")
    
    # Test α: Coupling strength
    print("\n6. α Processing (Conscience) - Coupling:")
    alpha_couplings = decision._get_alpha_couplings(options, state_key)
    for opt, coupling in alpha_couplings.items():
        print(f"   {opt}: α_eff = {coupling:.6f}")
    
    # Test ζ: Significance normalization
    print("\n7. ζ Processing (Conscience) - Significance:")
    significance = decision._zeta_normalize(options, superposition, gamma_scores, alpha_couplings)
    for opt, sig in significance.items():
        print(f"   {opt}: significance = {sig:.4f}")
    
    # Test 5-scalar validation
    print("\n8. 5-Scalar Validation:")
    confidence = manifold.get_confidence()
    print(f"   Conscience confidence:   {confidence:.4f}")
    print(f"   5/6 threshold:           {CONFIDENCE_EXPLOIT_THRESHOLD:.4f}")
    print(f"   Mode: {'EXPLOIT' if confidence > CONFIDENCE_EXPLOIT_THRESHOLD else 'EXPLORE'}")
    
    # Test δ: Collapse
    print("\n9. δ Processing (Ego) - Collapse:")
    collapsed = decision._collapse_superposition(superposition)
    print(f"   Collapsed to: {collapsed}")
    
    # Test full decision with processing cycle
    print("\n10. Full Processing Cycle Decision:")
    for i in range(5):
        selected = decision.decide(state_key, options)
        print(f"    Trial {i+1}: {selected}")
    
    # Test main decide() uses processing cycle
    print("\n11. Main decide() with processing cycle:")
    selected = decision.decide(state_key, options)
    print(f"    Selected: {selected}")
    
    # Add some history and re-test
    print("\n12. After recording outcomes:")
    decision.complete_decision("won", True, heat_delta=0.1)
    
    # Make more decisions with history
    for i in range(3):
        decision.decide(state_key, options)
        decision.complete_decision("played", i % 2 == 0, heat_delta=0.05 if i % 2 == 0 else -0.02)
    
    # Check updated superposition
    superposition2 = decision._create_superposition(options, state_key)
    print(f"    Updated weights: {[f'{w:.4f}' for w in superposition2['weights']]}")
    
    # Check updated confidence
    confidence2 = manifold.get_confidence()
    print(f"    Updated confidence: {confidence2:.4f}")
    
    print("\n" + "=" * 60)
    print("✓ PROCESSING CYCLE TEST COMPLETE")
    print("=" * 60)
    
    return True


def test_explore_vs_exploit():
    """Test that explore and exploit modes behave differently."""
    
    print("\n" + "=" * 60)
    print("TESTING EXPLORE vs EXPLOIT MODES")
    print("=" * 60)
    
    manifold = Manifold("test_manifold_2")
    manifold.birth()
    decision = DecisionNode(manifold)
    
    options = ["A", "B", "C", "D", "E"]
    
    # Test EXPLORE mode (low confidence)
    print("\n1. EXPLORE Mode (low confidence):")
    print("   Middle-entropy selection expected")
    
    # Force low confidence by using new state
    explore_decisions = []
    for i in range(10):
        state_key = f"new_state_{i}"
        selected = decision.decide(state_key, options)
        explore_decisions.append(selected)
    
    print(f"   Decisions: {explore_decisions}")
    unique_explore = len(set(explore_decisions))
    print(f"   Unique choices: {unique_explore}/10")
    
    # Build up history for EXPLOIT mode
    print("\n2. Building history for EXPLOIT mode:")
    state_key = "learned_state"
    for i in range(20):
        decision.decide(state_key, options)
        # Mark 'B' as successful most of the time
        success = (decision.pending_choice and decision.pending_choice.selected == 'B') or (i % 5 == 0)
        decision.complete_decision("played", success, heat_delta=0.1 if success else -0.05)
    
    # Check confidence now
    confidence = manifold.get_confidence()
    print(f"   Confidence after learning: {confidence:.4f}")
    
    # Test EXPLOIT mode
    print("\n3. EXPLOIT Mode (high confidence):")
    exploit_decisions = []
    for i in range(10):
        selected = decision.decide(state_key, options)
        exploit_decisions.append(selected)
    
    print(f"   Decisions: {exploit_decisions}")
    unique_exploit = len(set(exploit_decisions))
    print(f"   Unique choices: {unique_exploit}/10")
    
    # EXPLOIT should be less diverse (converging on learned pattern)
    print(f"\n4. Comparison:")
    print(f"   EXPLORE diversity: {unique_explore}/10 unique")
    print(f"   EXPLOIT diversity: {unique_exploit}/10 unique")
    if unique_exploit <= unique_explore:
        print("   ✓ EXPLOIT is less diverse (converging)")
    else:
        print("   ⚠ EXPLOIT more diverse than EXPLORE")
    
    return True


if __name__ == "__main__":
    test_processing_cycle()
    test_explore_vs_exploit()
