"""
PBAI Thermal Manifold - Compression System d(n)
Position strings compressed for storage at Self.

Full position: "nnnnnnnnnneeeeewwwwuuuuu"
Compressed:    n(10)e(5)w(4)u(5)

This is touched everywhere - it must be rock solid.

LEGACY DIRECTION USAGE:
    This module uses legacy single-character directions (n, s, e, w, u, d)
    for position encoding. These are the CUBIC LATTICE directions used
    internally for spatial positioning.
    
    The system also supports:
    - SELF_DIRECTIONS: Relative navigation (forward, back, left, right, up, down)
    - UNIVERSAL_DIRECTIONS: Absolute positioning (N, S, E, W, above, below)
    
    But for POSITION STRINGS (paths from Self), we always use the legacy
    single-character format for compactness and compatibility.
    
    Valid position characters: n, s, e, w, u, d
    Example: "nnwwu" = north, north, west, west, up from Self
"""

import re
from typing import Optional


def compress(position: str) -> str:
    """
    Compress position string to n(count) format.
    
    Examples:
        "" -> ""
        "n" -> "n(1)"
        "nn" -> "n(2)"
        "nne" -> "n(2)e(1)"
        "nnnnnnnnnneeeeewwwwuuuuu" -> "n(10)e(5)w(4)u(5)"
    """
    if not position:
        return ""
    
    # Validate input - only valid directions
    valid_dirs = set('nsewud')
    for char in position:
        if char not in valid_dirs:
            raise ValueError(f"Invalid direction character: '{char}'. Must be one of: n, s, e, w, u, d")
    
    result = []
    current = position[0]
    count = 1
    
    for char in position[1:]:
        if char == current:
            count += 1
        else:
            result.append(f"{current}({count})")
            current = char
            count = 1
    
    # Don't forget the last run
    result.append(f"{current}({count})")
    
    return "".join(result)


def decompress(compressed: str) -> str:
    """
    Expand n(count) format to full position string.
    
    Examples:
        "" -> ""
        "n(1)" -> "n"
        "n(2)" -> "nn"
        "n(2)e(1)" -> "nne"
        "n(10)e(5)w(4)u(5)" -> "nnnnnnnnnneeeeewwwwuuuuu"
    """
    if not compressed:
        return ""
    
    result = []
    pattern = r'([nsewud])\((\d+)\)'
    
    for match in re.finditer(pattern, compressed):
        direction, count_str = match.groups()
        count = int(count_str)
        if count < 1:
            raise ValueError(f"Invalid count {count} for direction '{direction}'")
        result.append(direction * count)
    
    full = "".join(result)
    
    # Verify we consumed the entire input (check if pattern matched anything)
    if not result and compressed:
        raise ValueError(f"Invalid compressed format: '{compressed}'")
    
    return full


def validate_position(position: str) -> bool:
    """
    Check if a position string is valid.
    Valid positions contain only: n, s, e, w, u, d
    """
    if not position:
        return True  # Empty string (Self's position) is valid
    
    valid_dirs = set('nsewud')
    return all(char in valid_dirs for char in position)


def get_axis_coordinates(position: str) -> dict:
    """
    Derive axis view from path.
    Returns count of each direction in the position.
    
    Example:
        "nnwwu" -> {'n': 2, 's': 0, 'e': 0, 'w': 2, 'u': 1, 'd': 0}
    """
    coords = {'n': 0, 's': 0, 'e': 0, 'w': 0, 'u': 0, 'd': 0}
    for char in position:
        if char in coords:
            coords[char] += 1
    return coords


def get_depth(position: str) -> int:
    """Get the abstraction depth (count of 'u' minus count of 'd')."""
    coords = get_axis_coordinates(position)
    return coords['u'] - coords['d']


def position_length(position: str) -> int:
    """Get the total path length from Self."""
    return len(position)


def positions_share_prefix(pos1: str, pos2: str) -> int:
    """
    Find length of common prefix between two positions.
    Returns the number of characters that match from the start.
    """
    common = 0
    for i in range(min(len(pos1), len(pos2))):
        if pos1[i] == pos2[i]:
            common += 1
        else:
            break
    return common


# ═══════════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════════

def run_compression_tests():
    """
    Comprehensive tests for compression/decompression.
    Run these before building anything on top!
    """
    test_cases = [
        # (raw, compressed)
        ("", ""),
        ("n", "n(1)"),
        ("nn", "n(2)"),
        ("nnn", "n(3)"),
        ("nne", "n(2)e(1)"),
        ("nnee", "n(2)e(2)"),
        ("nneew", "n(2)e(2)w(1)"),
        ("nnnnnnnnnneeeeewwwwuuuuu", "n(10)e(5)w(4)u(5)"),
        ("nsewud", "n(1)s(1)e(1)w(1)u(1)d(1)"),
        ("uuuuuuuuuu", "u(10)"),
        ("nusd", "n(1)u(1)s(1)d(1)"),
    ]
    
    print("Running compression tests...")
    all_passed = True
    
    for raw, expected_compressed in test_cases:
        # Test compress
        actual_compressed = compress(raw)
        if actual_compressed != expected_compressed:
            print(f"  FAIL compress: '{raw}' -> '{actual_compressed}' (expected '{expected_compressed}')")
            all_passed = False
        
        # Test decompress (only if we have a compressed form)
        if expected_compressed:
            actual_raw = decompress(expected_compressed)
            if actual_raw != raw:
                print(f"  FAIL decompress: '{expected_compressed}' -> '{actual_raw}' (expected '{raw}')")
                all_passed = False
        
        # Test roundtrip: compress then decompress
        if raw:
            roundtrip = decompress(compress(raw))
            if roundtrip != raw:
                print(f"  FAIL roundtrip: '{raw}' -> '{compress(raw)}' -> '{roundtrip}'")
                all_passed = False
    
    # Test axis coordinates
    coords = get_axis_coordinates("nnwwu")
    expected_coords = {'n': 2, 's': 0, 'e': 0, 'w': 2, 'u': 1, 'd': 0}
    if coords != expected_coords:
        print(f"  FAIL axis_coordinates: {coords} != {expected_coords}")
        all_passed = False
    
    # Test shared prefix
    if positions_share_prefix("nnee", "nnes") != 3:
        print(f"  FAIL shared_prefix: expected 3")
        all_passed = False
    
    if positions_share_prefix("nnn", "sss") != 0:
        print(f"  FAIL shared_prefix: expected 0")
        all_passed = False
    
    # Test invalid input handling
    try:
        compress("xyz")
        print("  FAIL: compress should reject invalid characters")
        all_passed = False
    except ValueError:
        pass  # Expected
    
    if all_passed:
        print("  All compression tests PASSED ✓")
    else:
        print("  Some tests FAILED!")
    
    return all_passed


if __name__ == "__main__":
    run_compression_tests()
