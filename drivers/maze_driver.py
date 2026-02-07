"""
Maze Driver - Routes maze interaction through the manifold

Inherits from Driver (environment.py). EnvironmentCore owns the Clock and
DecisionNode; this driver just implements perceive()/act() plus maze-specific
helpers that bigmaze.py calls directly.

Architecture:
    bigmaze.py ↔ MazeDriver ↔ EnvironmentCore → Manifold
"""

import logging
import math
from typing import Dict, List, Tuple, Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import Manifold, K
from core.nodes import Node
from core.hypersphere import SpherePosition, place_node_near
from drivers.environment import (
    Driver, Port, NullPort, Perception, Action, ActionResult
)

logger = logging.getLogger(__name__)

DIRECTIONS = {'N': (-1, 0), 'S': (1, 0), 'E': (0, 1), 'W': (0, -1)}
OPPOSITES = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}

# Angular position for the maze task frame on the hypersphere
MAZE_TASK_THETA = math.pi / 6      # 30° from north pole
MAZE_TASK_PHI = 0.0                 # +X direction


class MazeDriver(Driver):
    """
    PBAI driver that routes maze interaction through the manifold.

    Inherits Driver base class. EnvironmentCore provides the Clock
    (perception routing) and DecisionNode (action selection).
    """

    DRIVER_ID = "maze"
    DRIVER_NAME = "Maze Driver"
    DRIVER_VERSION = "1.0.0"
    SUPPORTED_ACTIONS = ['N', 'S', 'E', 'W']
    HEAT_SCALE = 1.0

    def __init__(self, port=None, config=None, manifold=None):
        # Game state (set by task file before each step)
        self.grid: List[List[int]] = []
        self.size: int = 0
        self.goals: List[Tuple[int, int]] = []
        self.start_pos: Tuple[int, int] = (1, 1)
        self.current_pos: Tuple[int, int] = (1, 1)
        self.maze_count = 0
        self.total_steps = 0
        self.total_backtracks = 0
        self.observed_cells: Dict[Tuple[int, int], int] = {}
        self.path_history: List[Tuple[int, int]] = []
        self._cell_node_ids: set = set()  # Track cell node IDs for wipe
        self._current_obs: Dict = {}
        self._current_state_key: str = ""
        self._current_context: Dict = {}

        # Task frame (created in _init_task_frame after super().__init__)
        self.task_frame: Optional[Node] = None

        super().__init__(port or NullPort("null"), config, manifold=manifold)
        self._init_task_frame()

    # ═══════════════════════════════════════════════════════════════════════════
    # DRIVER INTERFACE (abstract methods from Driver)
    # ═══════════════════════════════════════════════════════════════════════════

    def initialize(self) -> bool:
        if self.port:
            self.port.connect()
        self.active = True
        return True

    def shutdown(self) -> bool:
        if self.port:
            self.port.disconnect()
        self.active = False
        return True

    def perceive(self) -> Perception:
        """
        Package current maze state as a Perception.
        Called by EnvironmentCore.perceive() which routes to Clock.
        """
        obs = self._current_obs
        if not obs:
            return Perception(
                entities=["maze_empty"],
                properties={"state_key": "empty"},
                heat_value=0.0
            )

        # Open directions
        open_dirs = self.get_open_dirs(obs)
        row, col = obs.get('position', self.current_pos)

        # Novelty heat
        is_new = self.observed_cells.get((row, col), 0) == 1
        heat_value = K * 0.2 if is_new else K * 0.05

        return Perception(
            entities=[self._current_state_key],
            locations=open_dirs,
            properties={
                "state_key": self._current_state_key,
                "row": row,
                "col": col,
                "open_count": len(open_dirs),
                "unvisited_count": self._current_context.get('unvisited_count', 0),
                **{f"{d}_open": 1.0 if obs['directions'].get(d, {}).get('open') else 0.0
                   for d in DIRECTIONS},
                **{f"{d}_visited": 1.0 if obs['directions'].get(d, {}).get('visited') else 0.0
                   for d in DIRECTIONS},
            },
            events=["at_goal"] if obs.get('at_goal') else [],
            heat_value=self.scale_heat(heat_value),
            raw=obs
        )

    def act(self, action: Action) -> ActionResult:
        """
        Record outcome of a move. Called by EnvironmentCore.act().
        The task file has already moved the player; this reports what happened.
        """
        direction = action.action_type
        at_goal = self.current_pos in self.goals
        is_new = self.observed_cells.get(self.current_pos, 0) <= 1

        if at_goal:
            heat = K
            outcome = f"{direction}_goal"
            success = True
        elif is_new:
            heat = K * 0.1
            outcome = f"{direction}_new"
            success = True           # Discovery IS success
        else:
            heat = 0.0
            outcome = f"{direction}_revisit"
            success = False          # Only revisits are failure

        return ActionResult(
            success=success,
            outcome=outcome,
            heat_value=self.scale_heat(heat)
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # TASK FRAME AND CELL MANAGEMENT (hypersphere-native)
    # ═══════════════════════════════════════════════════════════════════════════

    def _init_task_frame(self):
        """Create the maze task frame on the hypersphere (once)."""
        if self.task_frame:
            return
        if not self.manifold:
            return

        existing = self.manifold.get_node_by_concept("bigmaze")
        if existing:
            self.task_frame = existing
            return

        self.task_frame = Node(
            concept="bigmaze",
            theta=MAZE_TASK_THETA,
            phi=MAZE_TASK_PHI,
            radius=1.0,
            heat=K,
            polarity=1,
            existence="actual",
            righteousness=1.0,
            order=1
        )
        self.manifold.add_node(self.task_frame)
        logger.info(f"Task frame created: bigmaze (theta={MAZE_TASK_THETA:.2f}, phi={MAZE_TASK_PHI:.2f})")

    def _get_cell(self, row: int, col: int) -> Optional[Node]:
        """Get cell node by concept name."""
        if not self.manifold:
            return None
        return self.manifold.get_node_by_concept(f"maze_{row}_{col}")

    def _get_or_create_cell(self, row: int, col: int) -> Node:
        """Get or create cell node on the hypersphere near the task frame."""
        concept = f"maze_{row}_{col}"
        node = self.manifold.get_node_by_concept(concept)
        if node:
            return node

        # Collect existing cell positions for placement
        existing_positions = []
        for nid in self._cell_node_ids:
            n = self.manifold.get_node(nid)
            if n:
                existing_positions.append(SpherePosition(theta=n.theta, phi=n.phi, radius=n.radius))

        # Place near task frame with angular offset based on grid coords
        # Use grid position to generate a deterministic target angle
        dr = row - self.start_pos[0]
        dc = col - self.start_pos[1]
        offset_theta = dr * 0.05  # ~2.9° per row
        offset_phi = dc * 0.05    # ~2.9° per col

        target = SpherePosition(
            theta=MAZE_TASK_THETA + offset_theta,
            phi=MAZE_TASK_PHI + offset_phi,
            radius=1.0
        )

        pos = place_node_near(target, existing_positions)

        node = Node(
            concept=concept,
            theta=pos.theta,
            phi=pos.phi,
            radius=pos.radius,
            heat=K,
            polarity=1,
            existence="actual",
            righteousness=0.0,  # Proper frame (inside righteous)
            order=max(2, abs(dr) + abs(dc) + 1)
        )
        self.manifold.add_node(node)
        self._cell_node_ids.add(node.id)

        logger.debug(f"Discovered cell: {concept} at theta={pos.theta:.3f} phi={pos.phi:.3f}")
        return node

    def _wipe_proper_frame(self):
        """Remove all cell nodes for new maze."""
        if not self.manifold:
            return

        removed = 0
        for node_id in list(self._cell_node_ids):
            if self.manifold.remove_node(node_id):
                removed += 1

        self._cell_node_ids.clear()

        # Clear task frame's connections
        if self.task_frame and hasattr(self.task_frame, 'frame') and self.task_frame.frame:
            self.task_frame.frame.axes.clear()

        logger.debug(f"Wiped proper frame: {removed} cells")

    # ═══════════════════════════════════════════════════════════════════════════
    # GAME-SPECIFIC HELPERS (called directly by bigmaze.py)
    # ═══════════════════════════════════════════════════════════════════════════

    def new_maze(self, start: Tuple[int, int], goal1: Tuple[int, int], goal2: Tuple[int, int]):
        """Start new maze - wipe proper frame, reset state."""
        self.maze_count += 1
        self.start_pos = start
        self.current_pos = start
        self.goals = [goal1, goal2]
        self.observed_cells = {}
        self.path_history = [start]

        # Wipe cell nodes from previous maze
        self._wipe_proper_frame()

        logger.info(f"═══ MAZE {self.maze_count} ═══")

    def observe(self, grid_pos: Tuple[int, int], grid: List[List[int]],
                size: int, goals: List[Tuple[int, int]]) -> Dict:
        """
        Observe current position - builds internal map.

        Creates cell nodes on the hypersphere and records passages.
        Does NOT route to Clock — that's EnvironmentCore's job via perceive().

        Returns obs dict for bigmaze.py compatibility.
        """
        # Update maze state
        self.grid = grid
        self.size = size
        self.goals = goals
        self.current_pos = grid_pos

        row, col = grid_pos

        # Track observation count
        self.observed_cells[grid_pos] = self.observed_cells.get(grid_pos, 0) + 1

        # Build internal map (discover cell on hypersphere)
        current_cell = self._get_or_create_cell(row, col)
        current_cell.add_heat(K * 0.1)

        # Build obs dict
        obs = {
            'position': grid_pos,
            'at_goal': grid_pos in goals,
            'directions': {}
        }

        open_dirs = []
        for d, (dr, dc) in DIRECTIONS.items():
            nr, nc = row + dr, col + dc

            if 0 <= nr < size and 0 <= nc < size and grid[nr][nc] == 0:
                target_cell = self._get_or_create_cell(nr, nc)
                current_cell.add_axis(d.lower(), target_cell.id)

                is_visited = (nr, nc) in self.observed_cells
                obs['directions'][d] = {
                    'open': True,
                    'visited': is_visited,
                    'visits': self.observed_cells.get((nr, nc), 0),
                    'is_goal': (nr, nc) in goals,
                }
                open_dirs.append(d)
            else:
                obs['directions'][d] = {'open': False}

        self._current_obs = obs

        # Build state key and context for perceive()
        unvisited_dirs = ''.join(sorted([
            d for d in open_dirs
            if (row + DIRECTIONS[d][0], col + DIRECTIONS[d][1]) not in self.observed_cells
        ]))
        visited_dirs = ''.join(sorted([
            d for d in open_dirs
            if (row + DIRECTIONS[d][0], col + DIRECTIONS[d][1]) in self.observed_cells
        ]))

        state_key = f"{current_cell.concept}|u:{unvisited_dirs}" if unvisited_dirs else f"{current_cell.concept}|done"
        self._current_state_key = state_key

        self._current_context = {
            'at_junction': len(open_dirs) > 2,
            'at_deadend': len(open_dirs) == 1,
            'at_corridor': len(open_dirs) == 2,
            'near_goal': any(obs['directions'].get(d, {}).get('is_goal') for d in open_dirs),
            'unvisited_count': len(unvisited_dirs),
            'visited_count': len(visited_dirs),
        }

        return obs

    def get_open_dirs(self, obs: Dict) -> List[str]:
        """Get list of open directions from obs."""
        return [d for d, info in obs['directions'].items() if info.get('open')]

    def is_dead_end(self, obs: Dict) -> bool:
        """All open paths already visited?"""
        for d in self.get_open_dirs(obs):
            if not obs['directions'][d].get('visited'):
                return False
        return True

    def get_backtrack_target(self) -> Optional[Tuple[int, int]]:
        """
        BFS from current position to find nearest observed cell
        that has an unvisited open neighbor.
        """
        from collections import deque

        start = self.current_pos
        queue = deque([start])
        visited = {start}

        while queue:
            pos = queue.popleft()
            row, col = pos

            # Does this cell have an unvisited open neighbor?
            for d, (dr, dc) in DIRECTIONS.items():
                nr, nc = row + dr, col + dc
                if (0 <= nr < self.size and 0 <= nc < self.size
                        and self.grid[nr][nc] == 0
                        and (nr, nc) not in self.observed_cells):
                    logger.info(f"Backtrack target: ({row},{col}) has unexplored {d} → ({nr},{nc})")
                    return pos

            # Expand BFS through observed cells
            for d, (dr, dc) in DIRECTIONS.items():
                npos = (row + dr, col + dc)
                if npos not in visited and npos in self.observed_cells:
                    if (0 <= npos[0] < self.size and 0 <= npos[1] < self.size
                            and self.grid[npos[0]][npos[1]] == 0):
                        visited.add(npos)
                        queue.append(npos)

        return None

    def get_backtrack_path(self, target: Tuple[int, int]) -> List[str]:
        """BFS shortest path from current_pos to target through observed cells."""
        from collections import deque

        start = self.current_pos
        if start == target:
            return []

        # BFS on observed grid
        queue = deque([(start, [])])
        visited = {start}

        while queue:
            pos, path = queue.popleft()
            for d, (dr, dc) in DIRECTIONS.items():
                npos = (pos[0] + dr, pos[1] + dc)
                if npos in visited:
                    continue
                # Only walk through cells we've already observed
                if npos not in self.observed_cells:
                    continue
                # Must be passable on the grid
                if not (0 <= npos[0] < self.size and 0 <= npos[1] < self.size
                        and self.grid[npos[0]][npos[1]] == 0):
                    continue
                new_path = path + [d]
                if npos == target:
                    return new_path
                visited.add(npos)
                queue.append((npos, new_path))

        logger.warning(f"BFS: no path from {start} to {target}")
        return []

    def record_move(self, new_pos: Tuple[int, int], direction: str = None, backtracking: bool = False):
        """
        Record movement - updates path history and step counts.
        Does NOT route to DecisionNode/psychology — that's EnvironmentCore's job.
        """
        self.current_pos = new_pos
        self.total_steps += 1

        if backtracking:
            self.total_backtracks += 1
            if new_pos in self.path_history:
                idx = self.path_history.index(new_pos)
                self.path_history = self.path_history[:idx + 1]
        else:
            self.path_history.append(new_pos)

    def record_completion(self):
        """Record maze completion."""
        cells = len(self.observed_cells)
        cell_nodes = len(self._cell_node_ids)
        total_nodes = len(self.manifold.nodes) if self.manifold else 0
        logger.info(f"Maze {self.maze_count} complete | Cells: {cells} | Cell nodes: {cell_nodes} | Total nodes: {total_nodes}")

    def get_action_scores(self) -> Dict[str, float]:
        """
        Return action scores for EnvironmentCore.decide() exploitation.
        Maze-specific: prefer unvisited directions, then least-visited.
        """
        obs = self._current_obs
        if not obs:
            return {}

        scores = {}
        open_dirs = self.get_open_dirs(obs)

        for d in open_dirs:
            info = obs['directions'].get(d, {})
            if info.get('is_goal'):
                scores[d] = 10.0  # Always go to goal
            elif not info.get('visited'):
                scores[d] = 1.0   # Prefer unvisited
            else:
                visits = info.get('visits', 1)
                scores[d] = max(0.01, 0.5 / visits)  # Least visited

        return scores

    # ═══════════════════════════════════════════════════════════════════════════
    # UI COMPATIBILITY
    # ═══════════════════════════════════════════════════════════════════════════

    @property
    def visited(self) -> set:
        return set(self.observed_cells.keys())

    @property
    def visit_count(self) -> Dict[Tuple[int, int], int]:
        return self.observed_cells.copy()


# ═══════════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')

    from core import get_pbai_manifold, get_growth_path
    from drivers.environment import EnvironmentCore

    print("=== MazeDriver Self-Test ===\n")
    passed = 0
    failed = 0

    def check(name, condition):
        global passed, failed
        if condition:
            print(f"  ✓ {name}")
            passed += 1
        else:
            print(f"  ✗ {name}")
            failed += 1

    # 1. Create manifold and driver
    growth_path = get_growth_path("growth_map.json")
    manifold = get_pbai_manifold(growth_path)
    driver = MazeDriver(manifold=manifold)

    check("Driver inherits from Driver base", isinstance(driver, Driver))
    check("Driver ID is 'maze'", driver.DRIVER_ID == "maze")
    check("Task frame created", driver.task_frame is not None)
    check("Task frame concept is 'bigmaze'", driver.task_frame.concept == "bigmaze")
    check("Task frame on hypersphere", driver.task_frame.theta == MAZE_TASK_THETA)

    # 2. Initialize via EnvironmentCore
    env_core = EnvironmentCore(manifold=manifold)
    env_core.register_driver(driver)
    activated = env_core.activate_driver("maze")
    check("EnvironmentCore activates driver", activated)

    # 3. New maze with simple 5x5 grid
    grid = [
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 1, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
    ]
    start = (1, 1)
    goal1 = (3, 3)
    goal2 = (1, 3)

    driver.new_maze(start, goal1, goal2)
    check("New maze resets state", driver.maze_count == 1)
    check("Goals set", driver.goals == [goal1, goal2])

    # 4. Observe
    obs = driver.observe(start, grid, 5, [goal1, goal2])
    check("Observe returns dict", isinstance(obs, dict))
    check("Position in obs", obs['position'] == start)
    check("Not at goal", not obs['at_goal'])
    open_dirs = driver.get_open_dirs(obs)
    check("Open directions found", len(open_dirs) > 0)

    # 5. Perceive via EnvironmentCore
    perception = env_core.perceive()
    check("Perception has entities", len(perception.entities) > 0)
    check("Perception has state_key", 'state_key' in perception.properties)

    # 6. Decide via EnvironmentCore
    action = env_core.decide(perception)
    check("Action returned", action is not None)
    check("Action type is a direction", action.action_type in ['N', 'S', 'E', 'W'])

    # 7. Cell nodes on hypersphere
    cell = driver._get_cell(1, 1)
    check("Cell node exists", cell is not None)
    check("Cell concept correct", cell.concept == "maze_1_1")
    check("Cell has angular position", hasattr(cell, 'theta') and hasattr(cell, 'phi'))

    # 8. Wipe works
    driver._wipe_proper_frame()
    check("Wipe clears cell IDs", len(driver._cell_node_ids) == 0)

    # 9. get_action_scores
    driver.new_maze(start, goal1, goal2)
    obs = driver.observe(start, grid, 5, [goal1, goal2])
    scores = driver.get_action_scores()
    check("Action scores returned", len(scores) > 0)
    check("Scores are floats", all(isinstance(v, float) for v in scores.values()))

    print(f"\n{'='*40}")
    print(f"  Passed: {passed}  Failed: {failed}")
    print(f"{'='*40}")
    sys.exit(1 if failed > 0 else 0)
