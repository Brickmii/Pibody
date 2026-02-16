# Memory Pass: Universal mine_block + Core Decontamination

## Problem 1: Behavioral
- `look_at_target` (weight 6.0) constantly chases the sun (hottest peak) instead of letting task verbs drive behavior
- `mine_block` forces camera to look down (+280) before mining — overrides any prior targeting
- `mine_forward` forces camera straight (0,0) — also overrides targeting
- No way to "mine what you're currently looking at"
- Can't build natural chains: look at tree → approach → mine it → collect → craft

## Problem 2: Core Contamination
- `core/introspector.py` has `BASE_MOTION_ACTION_MAP` hardcoded with 25+ Minecraft action names:
  mine_block, mine_forward, attack, sprint_forward, sprint_jump, jump_forward,
  move_forward, explore_left, explore_right, scout_ahead, watch_step,
  select_slot_1, select_slot_2, open_inventory, use, look_left, look_right,
  look_up, look_down
- `drivers/environment.py` MotionBus.get_weight_boosts() imports this map from core (line 333)
- Core should be domain-agnostic — it should know NOTHING about Minecraft

## Fix: Move verb→action map to driver

### Step 1: Driver owns its verb→action map
- Add `get_verb_action_map()` to Driver ABC in `drivers/environment.py` (returns {})
- Create `MC_VERB_ACTION_MAP` in `drivers/minecraft/minecraft_driver.py` with all 25 verb→action mappings
- MinecraftDriver overrides `get_verb_action_map()` to return MC_VERB_ACTION_MAP
- Delete `BASE_MOTION_ACTION_MAP` from `core/introspector.py` entirely

### Step 2: Thread the map through the system
- `introspector.get_weight_boosts()` receives verb→action map as parameter
- `MotionBus.get_weight_boosts()` receives verb→action map as parameter
- `environment.introspect()` passes driver's map to both
- `environment.decide()` passes driver's map to motion bus

### Step 3: Redefine mine_block as universal
- Remove look direction from mine_block sequence
- Before: look down(0,+280) → wait 0.1s → mouse_hold left 3.0s
- After: mouse_hold left 3.0s (mines whatever you're facing)

### Step 4: Remove mine_forward
- Delete from MC_ACTION_MAP, MC_ACTION_WEIGHTS, _action_heat()
- Remove from all verb mappings
- Action count: 43 → 42, sequences: 6 → 5

### Step 5: Rebalance look_at_target (observation is a valid action)
- Lower standalone weight from 6.0 → 2.0 in get_action_weights()
- Lower standalone weight boost from 6.0 → 2.0 in environment.introspect()
- Looking at something IS a Fire 3 (Existence/Perception) action — don't cripple observation
- At weight 2.0 it's one option among many, not dominant
- When in a deliberative plan chain, the plan queue takes priority anyway
- Sun-tracking still possible during exploration, just not overwhelming during tasks

### Step 6: Use available memory — bigger manifold, longer plans
- System has 16GB RAM, daemon uses only 434MB, 10GB available
- Growth files are only 60MB on disk — manifold can be 10-20x bigger
- Consider: increase max plan length for deliberative interleaving
- Consider: increase conscience axes limit (currently 90, could grow much more)
- Consider: increase node retention (more learned state patterns persist)
- This is about letting the thermal manifold grow to fill the space, not constraining it

## Verb→Action Map (MC_VERB_ACTION_MAP, moving to driver)

### Fire 1 — Heat (Magnitude)
- bm_take → [attack, use, select_slot_1, select_slot_2]
- bm_get → [attack, use, mine_block, select_slot_1]
- bm_mine → [mine_block, attack, select_slot_1]
- bm_dig → [mine_block, attack]

### Fire 2 — Polarity (Differentiation)
- bm_easily → [sprint_forward, sprint_jump]
- bm_quickly → [sprint_forward, sprint_jump, jump_forward]
- bm_better → [move_forward, sprint_forward]

### Fire 3 — Existence (Perception)
- bm_see → [look_left, look_right, look_up, look_down]
- bm_view → [look_left, look_right, look_up, look_down]
- bm_look → [look_left, look_right, look_up, look_down, explore_left, explore_right]
- bm_visual → [look_left, look_right, look_up, look_down]
- bm_visually → [explore_left, explore_right, scout_ahead]
- bm_visualize → [explore_left, explore_right, scout_ahead, watch_step]

### Fire 4 — Righteousness (Evaluation)
- bm_analyze → [look_left, look_right, look_up, look_down, wait]
- bm_understand → [look_left, look_right, wait]
- bm_identify → [look_left, look_right, look_up, look_down]

### Fire 5 — Order (Construction)
- bm_create → [mine_block, use, select_slot_1]
- bm_build → [mine_block, use, select_slot_1]
- bm_design → [look_left, look_right, look_up, look_down]
- bm_make → [mine_block, use, attack, select_slot_1]
- bm_craft → [use, open_inventory, select_slot_1]

### Fire 6 — Movement (Navigation)
- bm_explore → [move_forward, explore_left, explore_right, sprint_forward]
- bm_find → [move_forward, sprint_forward, scout_ahead, explore_left]
- bm_go → [move_forward, sprint_forward, sprint_jump, jump_forward]
- bm_discover → [move_forward, explore_left, explore_right, jump_forward]

## After This Pass
- Core (introspector, manifold, nodes) knows zero Minecraft actions
- Driver owns ALL domain-specific verb→action mappings
- mine_block is universal (mines what you're facing)
- look_at_target balanced: available for observation (weight 2.0), prioritized through plan chains for tasks
- "mine birch trees" → deliberative plan: [look_at_target, mine_block] chain
- Passive observation (sun-tracking, scanning) still works during idle exploration
- Manifold has room to grow — 10GB available for richer node structure
- Any future driver (browser, robot, etc.) provides its own verb→action map

## Files
1. `core/introspector.py` — delete BASE_MOTION_ACTION_MAP, update get_weight_boosts() signature
2. `drivers/minecraft/minecraft_driver.py` — add MC_VERB_ACTION_MAP, redefine mine_block, remove mine_forward, remove look_at_target weight, add get_verb_action_map()
3. `drivers/environment.py` — add get_verb_action_map() to Driver ABC, thread map through MotionBus and introspect()