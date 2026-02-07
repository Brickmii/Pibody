"""
Maze Driver - Routes maze interaction through the manifold

Same interface bigmaze.py expects, but internally:
- observe() → routes perception to Clock
- get_direction() → routes decision through DecisionNode
- record_move() → routes result as feedback

Self synchronizes with maze-world through this driver.
"""

import logging
from typing import Dict, List, Tuple, Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import Manifold, K
from core.clock_node import Clock
from core.decision_node import DecisionNode
from core.nodes import Node

logger = logging.getLogger(__name__)

DIRECTIONS = {'N': (-1, 0), 'S': (1, 0), 'E': (0, 1), 'W': (0, -1)}
OPPOSITES = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}


class MazeDriver:
    """
    PBAI driver that routes maze interaction through the manifold.
    
    External interface unchanged for bigmaze.py compatibility.
    Internal routing goes through Clock (perception) and DecisionNode (action).
    """
    
    DRIVER_ID = "maze"
    SUPPORTED_ACTIONS = ['N', 'S', 'E', 'W']
    
    def __init__(self, manifold: Manifold):
        self.manifold = manifold
        
        # Clock for perception routing (syncs Self time with maze time)
        self._clock: Optional[Clock] = None
        
        # DecisionNode for action routing
        self._decision_node: Optional[DecisionNode] = None
        
        # Maze state (external world)
        self.grid: List[List[int]] = []
        self.size: int = 0
        self.goals: List[Tuple[int, int]] = []
        self.start_pos: Tuple[int, int] = (1, 1)
        self.current_pos: Tuple[int, int] = (1, 1)
        
        # Stats (for UI)
        self.maze_count = 0
        self.total_steps = 0
        self.total_backtracks = 0
        
        # Track observations (for UI and backtracking)
        self.observed_cells: Dict[Tuple[int, int], int] = {}
        self.path_history: List[Tuple[int, int]] = []
        
        # Current state for decision routing
        self._current_state_key: str = ""
        self._current_context: Dict = {}
        self._current_obs: Dict = {}
        self._pending_action: Optional[str] = None
        
        # Task frame (righteous) - Self's representation of "I am doing maze"
        self.task_frame: Optional[Node] = None
        self._init_task_frame()
    
    def _init_task_frame(self):
        """Create the maze task frame (righteous, R=1.0) - once."""
        if self.task_frame:
            return
        
        self.task_frame = Node(
            concept="bigmaze",
            position="u",  # One step up - task level
            heat=K,
            polarity=1,
            existence="actual",
            righteousness=1.0,  # Righteous frame (task level)
            order=1
        )
        self.manifold.add_node(self.task_frame)
        logger.info("Task frame created: bigmaze (R=1.0)")
    
    def _grid_to_manifold_pos(self, row: int, col: int) -> str:
        """Convert grid coordinates to manifold position relative to start."""
        dr = row - self.start_pos[0]
        dc = col - self.start_pos[1]
        
        pos = ""
        if dr > 0:
            pos += 's' * dr
        elif dr < 0:
            pos += 'n' * (-dr)
        if dc > 0:
            pos += 'e' * dc
        elif dc < 0:
            pos += 'w' * (-dc)
        
        return pos if pos else "o"  # "o" for origin
    
    def _full_position(self, row: int, col: int) -> str:
        """Get full manifold position (task frame + 'c' + grid position)."""
        local = self._grid_to_manifold_pos(row, col)
        # 'c' ensures cells are INSIDE task frame (child of)
        return self.task_frame.position + 'c' + local
    
    def _get_cell(self, row: int, col: int) -> Optional[Node]:
        """Get cell node at grid position if it exists."""
        pos = self._full_position(row, col)
        return self.manifold.get_node_by_position(pos)
    
    def _get_or_create_cell(self, row: int, col: int) -> Node:
        """Get or create cell node at grid position - Self discovers this cell."""
        pos = self._full_position(row, col)
        local = self._grid_to_manifold_pos(row, col)
        
        # Check if already exists
        node = self.manifold.get_node_by_position(pos)
        if node:
            return node
        
        # Create new cell - Self is building internal map
        name = f"cell_{local}"
        node = Node(
            concept=name,
            position=pos,
            heat=K,
            polarity=1,
            existence="actual",
            righteousness=0.0,  # Proper frame (inside righteous)
            order=len(local) + 1
        )
        self.manifold.add_node(node)
        logger.debug(f"Discovered cell: {name} at {pos}")
        return node
    
    def _wipe_proper_frame(self):
        """Wipe the proper frame (all cells) for new maze."""
        if not self.task_frame:
            return
            
        prefix = self.task_frame.position + 'c'
        to_remove = []
        
        for pos, node_id in list(self.manifold.nodes_by_position.items()):
            if pos.startswith(prefix):
                to_remove.append((pos, node_id))
        
        for pos, node_id in to_remove:
            node = self.manifold.nodes.get(node_id)
            if node:
                del self.manifold.nodes[node_id]
                del self.manifold.nodes_by_position[pos]
                if node.concept in self.manifold.nodes_by_concept:
                    del self.manifold.nodes_by_concept[node.concept]
        
        # Clear task frame's connections
        if hasattr(self.task_frame, 'frame') and self.task_frame.frame:
            self.task_frame.frame.axes.clear()
        
        logger.debug(f"Wiped proper frame: {len(to_remove)} cells")
    
    def _get_clock(self) -> Clock:
        """Get or create Clock for perception routing."""
        if self._clock is None:
            self._clock = Clock(self.manifold)
        return self._clock
    
    def _get_decision_node(self) -> DecisionNode:
        """Get or create DecisionNode for action routing."""
        if self._decision_node is None:
            self._decision_node = DecisionNode(self.manifold)
        return self._decision_node
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # BIGMAZE.PY INTERFACE (unchanged externally)
    # ═══════════════════════════════════════════════════════════════════════════════
    
    def new_maze(self, start: Tuple[int, int], goal1: Tuple[int, int], goal2: Tuple[int, int]):
        """Start new maze - wipe proper frame, reset state."""
        self.maze_count += 1
        self.start_pos = start
        self.current_pos = start
        self.goals = [goal1, goal2]
        self.observed_cells = {}
        self.path_history = [start]
        
        # Wipe the proper frame (cells from previous maze)
        # Self forgets the OLD maze structure, keeps task frame
        self._wipe_proper_frame()
        
        logger.info(f"═══ MAZE {self.maze_count} ═══")
    
    def observe(self, grid_pos: Tuple[int, int], grid: List[List[int]], 
                size: int, goals: List[Tuple[int, int]]) -> Dict:
        """
        Observe current position - Self discovers and maps this cell.
        
        1. Creates cell node in manifold (proper frame)
        2. Records passages as connections to neighbor cells
        3. Routes perception to Clock
        
        Returns obs dict for bigmaze.py compatibility.
        """
        # Update maze state
        self.grid = grid
        self.size = size
        self.goals = goals
        self.current_pos = grid_pos
        
        row, col = grid_pos
        
        # Track observation
        self.observed_cells[grid_pos] = self.observed_cells.get(grid_pos, 0) + 1
        
        # ─────────────────────────────────────────────────────────────────────────
        # BUILD INTERNAL MAP (Self discovers this cell)
        # ─────────────────────────────────────────────────────────────────────────
        
        current_cell = self._get_or_create_cell(row, col)
        current_cell.add_heat(K * 0.1)  # Heat from observation
        
        # Build obs dict (for bigmaze.py compatibility)
        obs = {
            'position': grid_pos,
            'at_goal': grid_pos in goals,
            'directions': {}
        }
        
        # Check each direction and record passages
        open_dirs = []
        for d, (dr, dc) in DIRECTIONS.items():
            nr, nc = row + dr, col + dc
            
            if 0 <= nr < size and 0 <= nc < size and grid[nr][nc] == 0:
                # OPEN passage - record in manifold
                target_cell = self._get_or_create_cell(nr, nc)
                
                # Add connection representing passage (Self learns topology)
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
                # Wall - no connection (implicit)
                obs['directions'][d] = {'open': False}
        
        self._current_obs = obs
        
        # ─────────────────────────────────────────────────────────────────────────
        # ROUTE TO CLOCK (Perception → Self)
        # ─────────────────────────────────────────────────────────────────────────
        
        # Build state key from cell node + exploration state
        # FIX: Include which directions are unvisited so state changes as we explore
        unvisited_dirs = ''.join(sorted([
            d for d in open_dirs 
            if (row + DIRECTIONS[d][0], col + DIRECTIONS[d][1]) not in self.observed_cells
        ]))
        visited_dirs = ''.join(sorted([
            d for d in open_dirs 
            if (row + DIRECTIONS[d][0], col + DIRECTIONS[d][1]) in self.observed_cells
        ]))
        
        # State key now reflects exploration progress
        # e.g., "cell_ne|u:NSW" means at cell_ne with N,S,W unexplored
        # After exploring N: "cell_ne|u:SW" - different state!
        state_key = f"{current_cell.concept}|u:{unvisited_dirs}" if unvisited_dirs else f"{current_cell.concept}|done"
        self._current_state_key = state_key
        
        # Build context for learning
        context = {
            'at_junction': len(open_dirs) > 2,
            'at_deadend': len(open_dirs) == 1,
            'at_corridor': len(open_dirs) == 2,
            'near_goal': any(obs['directions'].get(d, {}).get('is_goal') for d in open_dirs),
            'unvisited_count': len(unvisited_dirs),
            'visited_count': len(visited_dirs),
        }
        self._current_context = context
        
        # Heat for discovery
        is_new = self.observed_cells[grid_pos] == 1
        heat_value = K * 0.2 if is_new else K * 0.05
        
        # Route to Clock
        clock = self._get_clock()
        clock.receive({
            "state_key": state_key,
            "context": context,
            "heat_value": heat_value,
            "entities": [state_key],
            "locations": open_dirs,
            "properties": {
                "state_key": state_key,
                "row": row,
                "col": col,
                "open_count": len(open_dirs),
                "unvisited_count": len(unvisited_dirs),
                **{f"{d}_open": 1.0 if obs['directions'].get(d, {}).get('open') else 0.0 
                   for d in DIRECTIONS},
                **{f"{d}_visited": 1.0 if obs['directions'].get(d, {}).get('visited') else 0.0 
                   for d in DIRECTIONS},
            }
        })
        
        # Tick Clock (sync Self time with maze time)
        clock.tick()
        
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
        Find nearest cell with unexplored path using manifold structure.
        
        Self queries its internal map to find where to go.
        """
        # Walk through observed cells (most recent first)
        for (row, col) in reversed(list(self.observed_cells.keys())):
            cell = self._get_cell(row, col)
            if not cell:
                continue
            
            # Check each connection from this cell
            if hasattr(cell, 'frame') and cell.frame:
                for axis_name, conn in cell.frame.axes.items():
                    # Get target cell
                    target_node = self.manifold.get_node(conn.target_id)
                    if not target_node:
                        continue
                    
                    # Calculate grid position from manifold position
                    # Position is like 'uco' or 'ucne' -> local is 'o' or 'ne'
                    prefix = self.task_frame.position + 'c'
                    if not target_node.position.startswith(prefix):
                        continue
                        
                    local = target_node.position[len(prefix):]
                    
                    # Convert local position back to grid offset
                    if local == 'o':
                        dr, dc = 0, 0
                    else:
                        dr = local.count('s') - local.count('n')
                        dc = local.count('e') - local.count('w')
                    
                    target_grid = (self.start_pos[0] + dr, self.start_pos[1] + dc)
                    
                    # If this target hasn't been visited, this cell is a good backtrack point
                    if target_grid not in self.observed_cells:
                        logger.info(f"Backtrack target: {(row, col)} has unexplored {axis_name.upper()} → {target_grid}")
                        return (row, col)
        
        return None
    
    def get_backtrack_path(self, target: Tuple[int, int]) -> List[str]:
        """Get directions to backtrack to target."""
        if target not in self.path_history:
            logger.warning(f"Target {target} not in path history!")
            return []
        
        moves = []
        target_idx = self.path_history.index(target)
        
        for i in range(len(self.path_history) - 1, target_idx, -1):
            curr = self.path_history[i]
            prev = self.path_history[i - 1]
            
            dr = prev[0] - curr[0]
            dc = prev[1] - curr[1]
            
            if dr == -1:
                moves.append('N')
            elif dr == 1:
                moves.append('S')
            elif dc == -1:
                moves.append('W')
            elif dc == 1:
                moves.append('E')
        
        return moves
    
    def get_direction(self, obs: Dict) -> Optional[str]:
        """
        Choose direction - routes through DecisionNode.
        
        This is where Self decides, not custom thermal logic.
        """
        open_dirs = self.get_open_dirs(obs)
        if not open_dirs:
            return None
        
        # Goal visible? Always take it (this is domain knowledge, not learning)
        for d in open_dirs:
            if obs['directions'][d].get('is_goal'):
                logger.info(f"GOAL visible at {d}!")
                self._pending_action = d
                self._begin_decision(d)
                return d
        
        # Only one way? Take it
        if len(open_dirs) == 1:
            self._pending_action = open_dirs[0]
            self._begin_decision(open_dirs[0])
            return open_dirs[0]
        
        # ─────────────────────────────────────────────────────────────────────────
        # ROUTE TO DECISIONNODE (Self decides)
        # ─────────────────────────────────────────────────────────────────────────
        
        decision_node = self._get_decision_node()
        
        # Get exploration rate from manifold
        exploration_rate = self.manifold.get_exploration_rate()
        
        # Get unvisited directions
        unvisited = [d for d in open_dirs if not obs['directions'][d].get('visited')]
        
        import random
        if unvisited:
            # Always prefer unvisited when available - this is the key fix
            chosen = random.choice(unvisited)
            logger.info(f"Chose unvisited: {chosen} from {unvisited}")
        elif random.random() < exploration_rate:
            # EXPLORE: random when all visited
            chosen = random.choice(open_dirs)
            logger.info(f"Explore (all visited, random): {chosen} (rate={exploration_rate:.2f})")
        else:
            # EXPLOIT: use DecisionNode's learned history
            chosen = decision_node.decide(
                state_key=self._current_state_key,
                options=open_dirs,
                context=self._current_context
            )
            logger.info(f"Exploit: {chosen} (rate={exploration_rate:.2f})")
        
        self._pending_action = chosen
        self._begin_decision(chosen)
        
        if len(open_dirs) > 2:
            logger.info(f"Junction: chose {chosen} from {open_dirs}")
        
        return chosen
    
    def _begin_decision(self, chosen: str):
        """Record decision start for learning."""
        decision_node = self._get_decision_node()
        open_dirs = self.get_open_dirs(self._current_obs)
        confidence = self.manifold.get_confidence()
        
        decision_node.begin_decision(
            self._current_state_key, 
            open_dirs, 
            confidence, 
            self._current_context
        )
        decision_node.commit_decision(chosen)
    
    def record_move(self, new_pos: Tuple[int, int], direction: str = None, backtracking: bool = False):
        """
        Record movement - routes result as feedback.
        
        This completes the decision cycle.
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
        
        # ─────────────────────────────────────────────────────────────────────────
        # ROUTE TO DECISIONNODE (Feedback)
        # ─────────────────────────────────────────────────────────────────────────
        
        if self._pending_action:
            decision_node = self._get_decision_node()
            
            # Determine success
            at_goal = new_pos in self.goals
            is_new = self.observed_cells.get(new_pos, 0) <= 1
            
            if at_goal:
                # SUCCESS - reached goal
                outcome = f"{self._pending_action}_goal"
                success = True
                heat_value = K
            elif is_new:
                # Progress - found new cell (neutral-positive)
                outcome = f"{self._pending_action}_new"
                success = False  # Not goal yet
                heat_value = K * 0.1
            else:
                # Revisit - been here before (neutral)
                outcome = f"{self._pending_action}_revisit"
                success = False
                heat_value = 0
            
            # Complete decision with feedback
            if decision_node.pending_choice:
                decision_node.complete_decision(outcome, success, heat_value)
            
            # Feed psychology
            self._feed_psychology(success, heat_value)
            
            self._pending_action = None
    
    def _feed_psychology(self, success: bool, heat_value: float):
        """Route feedback to psychology nodes."""
        if success and heat_value > 0:
            # SUCCESS - boost all psychology
            if self.manifold.identity_node:
                self.manifold.identity_node.add_heat(heat_value * 0.5)
            if self.manifold.ego_node:
                self.manifold.ego_node.add_heat(heat_value)
            if self.manifold.conscience_node:
                self.manifold.conscience_node.add_heat(heat_value * 0.2)
        elif heat_value > 0:
            # Neutral-positive - small boost
            if self.manifold.identity_node:
                self.manifold.identity_node.add_heat(heat_value * 0.1)
            if self.manifold.ego_node:
                self.manifold.ego_node.add_heat(heat_value * 0.1)
    
    def record_completion(self):
        """Record maze completion."""
        cells = len(self.observed_cells)
        steps = self.total_steps
        
        # Count cell nodes in proper frame
        prefix = self.task_frame.position + 'c' if self.task_frame else ""
        cell_nodes = sum(1 for pos in self.manifold.nodes_by_position if pos.startswith(prefix))
        
        total_nodes = len(self.manifold.nodes)
        logger.info(f"Maze {self.maze_count} complete | Cells: {cells} | Cell nodes: {cell_nodes} | Total nodes: {total_nodes}")
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # UI COMPATIBILITY
    # ═══════════════════════════════════════════════════════════════════════════════
    
    @property
    def visited(self) -> set:
        return set(self.observed_cells.keys())
    
    @property
    def visit_count(self) -> Dict[Tuple[int, int], int]:
        return self.observed_cells.copy()
