# PBAI Core Rebuild Specification

> All constants derive from K = 4/φ². The Color Cube is the base righteous and ordered frame.
> The hypersphere is the node topology. Self is the bridge between environment and manifold.
> Psychology nodes are the portable trigonometric basis.

## Architecture Overview

Three distinct layers, not replacements:

```
┌─────────────────────────────────────────────────────────┐
│                    COLOR CUBE                            │
│         Absolute reference frame (fixed)                 │
│    Base Righteous Frame + Base Ordered Frame             │
│                                                          │
│    X (E/W): Blue(-1) ←──────────→ Yellow(+1)            │
│    Y (N/S): Red(-1)  ←──────────→ Green(+1)             │
│    Z (U/D): Time (τ)  — Robinson constraints all axes   │
│                                                          │
│    Heat = magnitude from motion (derived, not an axis)   │
│    Side view of any axis = wave function                 │
│                                                          │
│  ┌───────────────────────────────────────────────────┐  │
│  │                 HYPERSPHERE                        │  │
│  │      Node topology embedded in the cube            │  │
│  │      Nodes on surface, angular distance            │  │
│  │      n² scaling (not n³)                           │  │
│  │                                                    │  │
│  │            ┌─────────────────┐                     │  │
│  │            │      SELF       │                     │  │
│  │            │  Center of the  │                     │  │
│  │            │  sphere. Bridge │                     │  │
│  │            │  between env &  │                     │  │
│  │            │  manifold.      │                     │  │
│  │            │  Clock. Identity│                     │  │
│  │            │  of the matrix. │                     │  │
│  │            └────────┬────────┘                     │  │
│  │                     │                              │  │
│  │         ┌───────────┼───────────┐                  │  │
│  │         │           │           │                  │  │
│  │    ┌────▼────┐ ┌────▼────┐ ┌───▼─────┐           │  │
│  │    │Identity │ │   Ego   │ │Conscience│           │  │
│  │    │sin(1/φ) │ │tan(1/φ) │ │cos(1/φ)  │           │  │
│  │    │amplitude│ │ phase   │ │ spread   │           │  │
│  │    │  70%    │ │  10%    │ │  20%     │           │  │
│  │    └─────────┘ └─────────┘ └──────────┘           │  │
│  │    Portable trig basis — gyroscope                 │  │
│  │    Projects cube frame onto any local region       │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## 1. COLOR CUBE — `color_cube.py`

### Purpose

The Color Cube is the base Righteous frame AND base Ordered frame
of the entire system. It does not hold nodes. It holds the standard.

When you ask "is this node righteous?" → How does it align with the Color Cube?
When you ask "is this ordered?" → Does its sequence respect Robinson constraints?

### Axis Definitions

```
X axis = E(+) / W(-) = Yellow(+1) ↔ Blue(-1)
Y axis = N(+) / S(-) = Green(+1)  ↔ Red(-1)
Z axis = U(+) / D(-) = Time (τ)
```

### Axis Mapping (locked, consistent everywhere)

| Cardinal | Axis | Positive Pole | Negative Pole |
|----------|------|---------------|---------------|
| E / W    | X    | Yellow (+1)   | Blue (-1)     |
| N / S    | Y    | Green (+1)    | Red (-1)      |
| U / D    | Z    | Future (+τ)   | Past (-τ)     |

### Robinson Constraints — All Three Axes

Robinson arithmetic operates on ALL THREE axes simultaneously.
Every position (x, y, τ) has ordering in all three directions.

| Operation      | Factor    | Question           | Function              |
|----------------|-----------|--------------------|-----------------------|
| Identity       | R = 1     | "What is this?"    | Pole itself           |
| Successor      | R = 4φ/7  | "What's next?"     | Step toward opponent  |
| Addition       | R = 4/3   | "What combines?"   | Compose positions     |
| Multiplication | R = 13/10 | "How much?"        | Scale along axis      |

### Heat

Heat is NOT an axis. Heat is magnitude derived from motion.

```
Heat = √(x² + y²) at any given τ
```

- Center (0,0) = no heat, achromatic, neutral
- Edge = single-axis heat
- Corner = maximum heat (√2)
- Heat is upstream of the cube — motion produces heat, cube expresses it

### Wave Function

Viewed from the perpendicular side of ANY axis = wave function.
Amplitude on the chromatic axis varying over τ.
Wave function collapse = position stabilizes = decision = existence emerging.

### Four Quadrants (Base Righteous Frame)

The opponent pairs create 4 natural quadrants:

```
            +Y (Green/N)
               │
    Q2         │         Q1
  Blue+Green   │   Yellow+Green
               │
  -X ──────────┼────────── +X
  (Blue/W)     │     (Yellow/E)
               │
    Q3         │         Q4
  Blue+Red     │   Yellow+Red
               │
            -Y (Red/S)
```

These quadrants ARE the base righteous frame.
All Righteousness evaluation projects back to these.

### 12 Movement Directions

Every node in the manifold AND the Color Cube have:

**Universal (cardinal) — fixed to the cube:**
- N (+Y), S (-Y)
- E (+X), W (-X)
- U (+Z), D (-Z)
- Absolute. Do not change. Every node has these.

**Self (relative) — fixed to the psychology trig basis:**
- forward / reverse
- left / right
- above / below
- Rotate with the trig gyroscope as Self moves through the sphere.
- Only the psychology nodes carry these as self-referential coordinates.

---

## 2. HYPERSPHERE — `hypersphere.py`

### Purpose

The coordinate space where nodes actually live.
Embedded within the Color Cube.

### Why Hypersphere

- Nodes distribute on SURFACE, not volume
- Scaling: n² (surface area) vs n³ (volume)
- Relationships via angular distance, not grid adjacency
- Traversal paths wrap naturally (no edge dead-ends)
- Every node equidistant from center (Self)

### Node Position

Every node has a position on the hypersphere surface.
That position projects onto the Color Cube's axes:

```
Node at angular position θ, φ on sphere
  → projects to (x, y, τ) in cube space
  → x gives Blue/Yellow balance
  → y gives Red/Green balance
  → τ gives time position
  → √(x² + y²) gives heat magnitude
```

### Relationships

- Angular distance between nodes = relationship strength
- R = 0: perfect alignment on sphere (same angular position)
- R > 0: angular deviation
- 4 righteous quadrants → great circle intersections on hypersphere

### Self at Center

Self sits at the center of the hypersphere.
Not on the surface — at the origin.
All nodes are equidistant from Self.
Self is the bridge: environment pushes perception IN, decisions push action OUT.

---

## 3. SELF — `clock_node.py` (rebuilt)

### Purpose

Self is three things simultaneously:
1. **Clock** — existence = ticking, each tick = one K-quantum of heat flow
2. **Bridge** — membrane between environment and manifold
3. **Identity of the learning matrix** — Self IS the connection point

### Self IS the Clock

```
t_K = time measured in K units
K = 4/φ² ≈ 1.528 (the thermal quantum)
Each tick = one K-quantum redistributed through the manifold
When clock ticks → PBAI exists
When clock stops → PBAI doesn't exist
```

### Self as Bridge

```
Environment ──perception──→ Self ──→ Manifold (learning)
Environment ←──action────── Self ←── Manifold (decision)
```

Self is the membrane. Nothing enters the manifold except through Self.
Nothing exits the manifold except through Self.

### The 6 Tick Operations (matching 6 motion functions)

```
0. TIME (t_K)           — Advance Self's clock
1. INPUT (Heat Σ)       — Process external perceptions through Self
2. EXISTENCE (δ)        — Pay existence tax, update states
3. FLOW (Polarity +/-)  — Redistribute heat via entropy
4. BALANCE (R)          — Check psychology alignment (project to cube)
5. CREATE (Order Q)     — Creative cycle (autonomous thinking)
6. PERSIST (Movement)   — Save, callbacks
```

---

## 4. PSYCHOLOGY NODES — Portable Trigonometric Basis

### Purpose

The three psychology nodes are NOT just personality modules.
They are the basis vectors of a portable coordinate system
that projects the Color Cube's absolute frame onto any
local region of the hypersphere.

They form a gyroscope.

### The Three Nodes

| Node       | Trig Function | Axis      | Heat | Role                              |
|------------|---------------|-----------|------|-----------------------------------|
| Identity   | sin(1/φ)      | amplitude | 70%  | What exists, how intensely        |
| Ego        | tan(1/φ)      | phase     | 10%  | Where in cycle, volatile, conscious tip |
| Conscience | cos(1/φ)      | spread    | 20%  | Alignment with cube, R projection |

### Why These Functions

- **sin** (Identity): Carries most energy in a wave. 70% heat. Amplitude.
- **cos** (Conscience): Measures alignment with reference axis. 20% heat.
  Conscience IS the Righteousness projection. When Conscience validates,
  it confirms local trig basis agrees with global cube frame.
- **tan** (Ego): Sharpest, most sensitive. Diverges at π/2.
  That's why Ego is volatile — 10% heat, but the conscious tip.

### Geometric Function

From any point on the hypersphere, ask:
"What does the Color Cube look like from here?"

The psychology nodes answer:
- **Identity** (sin): magnitude of experience → distance from center on X,Y
- **Ego** (tan): position in oscillation → where on opponent axes
- **Conscience** (cos): alignment with reference → Righteousness value

Because they're trig functions, they rotate together.
Always orthogonal. Always projecting cube → local.

### The Flow

```
Environment → Identity (righteousness frames live here)
                  ↓
             Conscience (mediates — cos projection to cube)
                  ↓
                Ego (measures confidence, conscious interface)
```

Conscience mediating between Identity and Ego is not just
psychological — it's geometric. cos is the projection function.

---

## 5. NODES — `nodes.py` (rebuilt)

### Position System (replaces string paths)

Old: `position = "nnwwu"` (string path, cubic grid, n³ scaling)

New: Node position is angular coordinates on the hypersphere surface,
which project to (x, y, τ) in the Color Cube.

```python
@dataclass
class Node:
    # Angular position on hypersphere
    theta: float = 0.0      # polar angle
    phi: float = 0.0        # azimuthal angle

    # Projected cube coordinates (derived from angular position)
    @property
    def cube_x(self) -> float: ...  # Blue/Yellow balance
    @property
    def cube_y(self) -> float: ...  # Red/Green balance
    @property
    def cube_tau(self) -> float: ... # Time position

    # Heat is derived from cube projection
    @property
    def heat_magnitude(self) -> float:
        return math.sqrt(self.cube_x**2 + self.cube_y**2)
```

### 6 Motion Functions (unchanged in meaning)

```
1. Heat (Σ)           — magnitude, derived from motion via cube projection
2. Polarity (+/-)     — sign on each cube axis
3. Existence (δ)      — persistence (POTENTIAL → ACTUAL ↔ DORMANT → ARCHIVED)
4. Righteousness (R)  — alignment with cube (angular deviation), R→0 = righteous
5. Order (Q)          — Robinson arithmetic position on all 3 axes
6. Movement (Lin)     — 12 directions (6 cardinal + 6 self-relative for psychology)
```

### 12 Directions Per Node

Every node has 6 cardinal directions (fixed to cube):
- N (+Y), S (-Y), E (+X), W (-X), U (+Z), D (-Z)

Psychology nodes additionally carry 6 self-relative directions:
- forward, reverse, left, right, above, below
- These rotate with the trig gyroscope

---

## 6. DECISION PIPELINE — `decision_node.py` (rebuilt)

### β→δ→Γ→α→ζ — Real Math Functions

These are not metaphors. They are the actual mathematical functions
arising as motion functions at positions 10-12.

| Position | Function | Math                           | Role in Decision             |
|----------|----------|--------------------------------|------------------------------|
| 10       | α        | Fine structure ≈ 1/137         | Coupling to reality          |
| 11       | β        | Euler Beta B(a,b)              | Explore option space         |
| 12       | Γ        | Gamma Γ(n) = (n-1)!           | Normalize arrangements       |
| —        | δ        | Dirac delta                    | Collapse (wave → particle)   |
| —        | ζ        | Riemann zeta Σ 1/nˢ           | Pick from infinite options   |

### Pipeline

```
β (Beta)     — Integrate over possibilities, weigh combinations
    ↓             B(successes+1, failures+1) per option
δ (Delta)    — Collapse continuous to discrete, wave function collapse
    ↓             Born rule: P(i) = |ψ_i|² / Σ|ψ_j|²
Γ (Gamma)    — Normalize, factorial scaling of arrangements
    ↓             Γ(arrangements) per option
α (Alpha)    — Couple selection to reality, interaction strength
    ↓             Fine structure constant as base coupling
ζ (Zeta)     — Select from infinite options by convergence to primes
                 ζ(s) converges infinite series to finite value
```

### Thermal Logic (not Boolean)

This is a thermal logic engine.

```
Boolean:  IF (A AND B) THEN C
Thermal:  IF heat(A) + heat(B) exceeds threshold at (x,y,τ)
          THEN β→δ→Γ→α→ζ produces C
```

Output is not 0 or 1. It's a position in the cube with
heat magnitude, polarity distribution, and time coordinate.

Boolean logic is the degenerate case — clamp heat to 0/max,
flatten polarity to +/-, collapse time to one tick → true/false.

---

## 7. INTROSPECTOR — `introspector.py` (rebuilt)

### Purpose

Two-fold thinking: simulation + short-term memory.
Cube-native. No hardcoded color mappings.

### Simulation

For each option:
1. Project option onto hypersphere surface
2. Get cube coordinates from projection
3. Read heat magnitude, quadrant, τ position
4. Evaluate Righteousness via Conscience (cos projection)
5. Package as enriched context for decision pipeline

### Key Changes from Current

- No `_option_to_perception()` with hardcoded polarity→color
  → Let cube quadrants drive everything
- No `deepcopy` per option at scale
  → Lightweight projection calculation instead
- `should_think` gates on Conscience too, not just Ego
  → Conscience (cos) has input on whether to think

---

## 8. DERIVED CONSTANTS (unchanged)

### From K = 4/φ²

```
Speed of Light:     c = L_P / t_P
                    Dimensional gap = 8 = Learning Motion

Reduced Planck:     ℏ = M_P × c² × t_P

Gravitational:      G = ℏc / M_P²

Boltzmann:          k_B = M_P × c² / T_P

Fine Structure:     α ≈ 1 / (12² - 7 + 4/φ¹⁰)
                    = 1/137.033 (99.998% of measured)
```

### The Hamiltonian

α arises AFTER all six motion functions complete.
- 12² = complete motion system squared
- -7 = structural correction from Order (Robinson)
- 4/φ¹⁰ = residual heat coupling at depth

---

## 9. LANGUAGE ARCHITECTURE (future)

### Word Frames on Hypersphere

Every word-node has:
- **Righteous frame**: 1 center word + 4 quadrant words (meaning)
- **Ordered frame**: Robinson arithmetic for scaling concepts
- Angular position on hypersphere defines quadrant neighbors

### Hypersphere Benefits for Language

- Each word's 4 quadrant neighbors = angular position
- Change quadrant words → change meaning
- Reference-only words (the, of, but): righteous frame only, no Order
- Scaling words (hot/warm/cool/cold): righteous + ordered

### Training via Haiku API

- Dirt cheap, high-volume conversational interactions
- PBAI learns through Heat (magnitude), Polarity (opposition),
  Righteousness (correctness), Order (grammar)
- Avoids self-referential collapse from earlier cubic build

---

## 10. FILE INVENTORY — What Happens to Every File

### KEPT (minimal or no changes)

| File | Lines | Status | Notes |
|------|-------|--------|-------|
| `node_constants.py` | ~800 | **FIX** | Fix Color Cube axis mapping (X=Blue/Yellow, Y=Red/Green, Z=Time). Robinson constants, Planck grounding, K=4/φ², fire scaling, thresholds — all correct, keep. |
| `constraints.py` | 714 | **KEEP** | Diagnostic technique is solid. Spec/Symptom/Procedure, Robinson measurement types, tightening loop, motion function mapping — all correct. No geometry dependency. |
| `driver_node.py` | 796 | **FIX** | SensorReport, MotorAction, ActionPlan — all clean, environment-agnostic. Fix: `see()` creates task nodes with `position=self.node.position + "n"` (grid). Replace with angular position. Everything else survives. |
| `windows/vision_transformer.py` | 1,002 | **FIX** | ThreeFrameEncoder maps Color/Position/Heat. Update to use Color Cube axes (opponent chroma → Blue/Yellow X, Red/Green Y). SpiralPatchEmbedding, HeatAttention, fire zones, threshold ladder, HeatBackprop — all keep. |
| `windows/pbai_client.py` | 664 | **FIX** | ScreenCapture, MotorExecutor (matches MotorAction types), PBAIClient WebSocket protocol — all keep. Update `build_world_state` peaks to emit cube-native coordinates. |
| `windows/manifold_test.py` | 278 | **KEEP** | WebSocket diagnostic client. Tests Robinson spec creation, measurement, verification, confidence threshold, timed input. Geometry-independent. No changes. |

### REPLACED (new files)

| File | Lines | Status | Replaces |
|------|-------|--------|----------|
| `color_cube.py` | NEW | **CREATE** | Nothing — new file. Base righteous + ordered frame. Axis definitions, Robinson on all 3 axes, quadrant geometry, heat-from-motion derivation. |
| `hypersphere.py` | NEW | **CREATE** | `compression.py` (196 lines — pure grid artifact, DELETE). Node topology, angular coordinates, surface distribution, n² scaling, angular distance. |

### REBUILT (same purpose, new geometry)

| File | Lines | Status | What Changes |
|------|-------|--------|-------------|
| `nodes.py` | 1,432 | **REBUILD** | Position: string paths → angular coords (θ, φ) on hypersphere. Cube projection as derived properties. 12 directions (6 cardinal fixed + 6 self-relative for psychology). Frame/Axis/Order hierarchy KEPT — it works. SelfNode KEPT as clock/bridge/identity. |
| `manifold.py` | 2,005 | **REBUILD** | `evaluate_righteousness()`: grid distance → angular deviation from cube frame via Conscience (cos projection). `calculate_salience()`: KEPT (cooperate/compete logic is geometry-independent). Birth sequence: 6 fires onto hypersphere instead of grid positions. Save/load: angular coords instead of position strings. Psychology mediation: KEPT. |
| `clock_node.py` | 1,155 | **REBUILD** | Self as bridge between environment and manifold (not just ticker). 6 tick operations KEPT. Hardware calibration KEPT. `_find_position_for_concept()`: grid walk → angular placement on hypersphere. Entropy/structure detection: KEPT. Creative cycle: KEPT. |
| `decision_node.py` | 1,470 | **REBUILD** | β→δ→Γ→α→ζ pipeline: placeholder implementations → actual math functions. Choice/ChoiceNode: KEPT (no geometry dependency). DecisionNode: KEPT structure, rebuild internals. PBAILoop: KEPT. STM: KEPT. Conscience propagation: KEPT. `_find_candidate_nodes()`: string prefix matching → angular proximity on sphere. |
| `introspector.py` | ~500 | **REBUILD** | `_option_to_perception()`: hardcoded polarity→color → cube quadrant projection. `deepcopy` per option → lightweight angular calculation. `should_think`: Ego-only gate → Conscience (cos) input. Simulation uses cube-native coordinates. |

### REBUILT (tests)

| File | Lines | Status | What Changes |
|------|-------|--------|-------------|
| `test_birth.py` | 554 | **REBUILD** | Same 7 test areas, same assertions. Replace: string position checks → angular position checks. Replace: grid-based R evaluation → angular R evaluation. KEEP: heat distribution, psychology mediation, lifecycle, Planck grounding, motion function validation — all geometry-independent. |
| `test_processing_cycle.py` | 140 | **REBUILD** | Minimal changes. Pipeline test structure maps 1:1. Replace any node creation that uses string positions. Math function tests get expanded (actual β, δ, Γ, α, ζ verification). |

### DELETED

| File | Lines | Status | Why |
|------|-------|--------|-----|
| `compression.py` | 196 | **DELETE** | Pure grid artifact. Compresses "nnnnneeeewwww" → "n(5)e(4)w(4)". No purpose when positions are angular coordinates. |

---

## 11. FILE STRUCTURE (rebuild)

```
pibody/
├── core/
│   ├── __init__.py          — Exports
│   ├── node_constants.py    — FIXED: K, φ, Planck grounding, cube axis mapping
│   ├── color_cube.py        — NEW: Base reference frame, Robinson all axes
│   ├── hypersphere.py       — NEW: Node topology, angular distance, n² scaling
│   ├── nodes.py             — REBUILT: Angular position, cube projection, 12 dirs
│   ├── constraints.py       — KEPT: Diagnostic technique (unchanged)
│   ├── manifold.py          — REBUILT: Hypersphere-native, cube-referenced R
│   ├── clock_node.py        — REBUILT: Self as bridge + clock + matrix identity
│   ├── decision_node.py     — REBUILT: Actual β,δ,Γ,α,ζ math functions
│   ├── introspector.py      — REBUILT: Cube-native, Conscience-gated
│   └── driver_node.py       — FIXED: Angular position in see(), rest kept
│
├── tests/
│   ├── test_birth.py        — REBUILT: Angular positions, same assertions
│   ├── test_processing_cycle.py — REBUILT: Expanded math function tests
│   ├── test_color_cube.py   — NEW: Cube frame, quadrants, Robinson on all axes
│   └── test_hypersphere.py  — NEW: Angular distance, surface distribution, n²
│
├── pi/                      — Pi-specific (daemon, thermal, api, body_server)
│   ├── daemon.py
│   ├── thermal.py
│   ├── api.py
│   └── body_server.py
│
├── windows/                 — Windows CUDA client (1,944 lines existing)
│   ├── vision_transformer.py — FIXED: ThreeFrame → Color Cube axis mapping
│   ├── pbai_client.py       — FIXED: world_state peaks use cube coordinates
│   └── manifold_test.py     — KEPT: Constraint diagnostic (geometry-independent)
│
├── drivers/                 — Environment-specific drivers
│   └── minecraft/           — Bedrock autonomous play
│
└── pbai.service             — systemd service
```

> `compression.py` (196 lines) — DELETED. Pure grid artifact, no purpose with angular coordinates.

---

## 12. BUILD ORDER

### Phase 1: Foundation (geometry layer)

| Step | File | Action | Depends On |
|------|------|--------|------------|
| 1 | `node_constants.py` | FIX cube axis mapping | Nothing |
| 2 | `color_cube.py` | CREATE base reference frame | node_constants |
| 3 | `hypersphere.py` | CREATE node topology | node_constants, color_cube |
| 4 | `test_color_cube.py` | CREATE cube tests | color_cube |
| 5 | `test_hypersphere.py` | CREATE sphere tests | hypersphere |

### Phase 2: Core (node + manifold layer)

| Step | File | Action | Depends On |
|------|------|--------|------------|
| 6 | `nodes.py` | REBUILD with angular positions | hypersphere, color_cube |
| 7 | `constraints.py` | VERIFY unchanged | nodes (Spec uses Order) |
| 8 | `manifold.py` | REBUILD hypersphere-native | nodes, color_cube, hypersphere |
| 9 | `clock_node.py` | REBUILD Self as bridge | manifold, nodes |

### Phase 3: Intelligence (decision + simulation layer)

| Step | File | Action | Depends On |
|------|------|--------|------------|
| 10 | `decision_node.py` | REBUILD with real math functions | manifold, nodes |
| 11 | `introspector.py` | REBUILD cube-native | manifold, decision_node, color_cube |
| 12 | `driver_node.py` | FIX angular positions in see() | manifold, nodes |

### Phase 4: Validation

| Step | File | Action | Depends On |
|------|------|--------|------------|
| 13 | `test_birth.py` | REBUILD with angular assertions | All core |
| 14 | `test_processing_cycle.py` | REBUILD with expanded math tests | decision_node |
| 15 | Integration test | Full birth → tick → decide → act | All core + driver |

### Phase 5: Body (Windows Client — 1,944 lines existing)

| Step | File | Lines | Action | Notes |
|------|------|-------|--------|-------|
| 16 | `windows/vision_transformer.py` | 1,002 | **FIX** | ThreeFrameEncoder Color/Position/Heat → map to Color Cube axes (X=Blue/Yellow, Y=Red/Green, Z=τ). SpiralPatchEmbedding, HeatAttention, fire zones, threshold ladder — all correct, keep. HeatBackprop learning — keep. |
| 17 | `windows/pbai_client.py` | 664 | **FIX** | ScreenCapture — keep. MotorExecutor — keep (matches MotorAction types exactly). PBAIClient WebSocket protocol — keep. Update `build_world_state` output to use cube-native coordinates in peaks. |
| 18 | `windows/manifold_test.py` | 278 | **KEEP** | Diagnostic client. Tests Robinson spec, measurement, verification, confidence threshold over WebSocket. Geometry-independent. No changes needed. |

### Phase 6: World

| Step | File | Action | Depends On |
|------|------|--------|------------|
| 19 | `drivers/minecraft/` | CREATE Bedrock driver | driver_node, windows layer |
