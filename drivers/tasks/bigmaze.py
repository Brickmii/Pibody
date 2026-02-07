"""
PBAI Maze Task

Play manually with WASD/Arrows, or click "PBAI Auto" to watch PBAI explore.
PBAI maps the maze structure into the manifold as it explores.

Run:
    python drivers/tasks/bigmaze.py
"""

import tkinter as tk
import random
import sys
import os
import logging
from typing import List, Tuple, Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

DIRECTIONS = {'N': (-1, 0), 'S': (1, 0), 'E': (0, 1), 'W': (0, -1)}


class MazeGenerator:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = [[1 for _ in range(width)] for _ in range(height)]
        self.start = (1, 1)
        self.goal1 = None
        self.goal2 = None

    def generate(self):
        self.grid = [[1 for _ in range(self.width)] for _ in range(self.height)]
        stack = [self.start]
        self.grid[self.start[0]][self.start[1]] = 0
        directions = [(-2, 0), (0, 2), (2, 0), (0, -2)]

        while stack:
            current = stack[-1]
            row, col = current
            neighbors = []
            for dr, dc in directions:
                nr, nc = row + dr, col + dc
                if 1 <= nr < self.height - 1 and 1 <= nc < self.width - 1 and self.grid[nr][nc] == 1:
                    neighbors.append((nr, nc))
            if neighbors:
                next_cell = random.choice(neighbors)
                nr, nc = next_cell
                self.grid[row + (nr - row) // 2][col + (nc - col) // 2] = 0
                self.grid[nr][nc] = 0
                stack.append(next_cell)
            else:
                stack.pop()

        self.place_goals()
        self.grid[self.start[0]][self.start[1]] = 0

    def place_goals(self):
        open_cells = [(r, c) for r in range(1, self.height - 1) for c in range(1, self.width - 1)
                      if self.grid[r][c] == 0 and abs(r - self.start[0]) + abs(c - self.start[1]) >= 8]

        if len(open_cells) < 2:
            self.goal1, self.goal2 = (self.height - 4, self.width - 4), (self.height - 6, self.width - 7)
        else:
            self.goal1 = random.choice(open_cells)
            open_cells.remove(self.goal1)
            far = [c for c in open_cells if abs(c[0] - self.goal1[0]) + abs(c[1] - self.goal1[1]) >= 6]
            self.goal2 = random.choice(far) if far else random.choice(open_cells)

        self.grid[self.goal1[0]][self.goal1[1]] = 0
        self.grid[self.goal2[0]][self.goal2[1]] = 0


class MazeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PBAI Maze")
        self.root.geometry("700x800")
        self.root.configure(bg="#0f1923")

        self.maze_size = 20
        self.cell_size = 25
        self.maze = MazeGenerator(self.maze_size, self.maze_size)
        self.player_pos = self.maze.start
        self.won = False

        # PBAI
        self.pbai_on = False
        self.driver = None
        self.env_core = None
        self.delay = 100

        self.build_ui()
        self.new_maze()
        self.root.bind("<KeyPress>", self.key_press)
        self.root.focus_set()

    def build_ui(self):
        tk.Label(self.root, text="PBAI Maze", font=("Arial", 16, "bold"),
                 bg="#0f1923", fg="#e0e0e0").pack(pady=8)

        frame = tk.Frame(self.root, bg="#1d3557", bd=3, relief="sunken")
        frame.pack(padx=15, pady=8)
        self.canvas = tk.Canvas(frame, width=self.maze_size * self.cell_size,
                                height=self.maze_size * self.cell_size, bg="#0a1929")
        self.canvas.pack()

        btns = tk.Frame(self.root, bg="#0f1923")
        btns.pack(pady=8)
        tk.Button(btns, text="New Maze", font=("Arial", 11, "bold"), bg="#4ade80", fg="black",
                  width=10, command=self.new_maze).pack(side=tk.LEFT, padx=6)
        tk.Button(btns, text="Restart", font=("Arial", 11, "bold"), bg="#fb923c", fg="black",
                  width=10, command=self.restart).pack(side=tk.LEFT, padx=6)

        pbai = tk.Frame(self.root, bg="#0f1923")
        pbai.pack(pady=8)
        self.pbai_btn = tk.Button(pbai, text="▶ PBAI Auto", font=("Arial", 11, "bold"),
                                   bg="#22c55e", fg="white", width=12, command=self.toggle_pbai)
        self.pbai_btn.pack(side=tk.LEFT, padx=6)
        tk.Label(pbai, text="Delay:", bg="#0f1923", fg="#e0e0e0").pack(side=tk.LEFT)
        self.delay_var = tk.StringVar(value="100")
        tk.Entry(pbai, textvariable=self.delay_var, width=5).pack(side=tk.LEFT)
        tk.Label(pbai, text="ms", bg="#0f1923", fg="#e0e0e0").pack(side=tk.LEFT)

        leg = tk.Frame(self.root, bg="#0f1923")
        leg.pack(pady=6)
        for txt, col in [("You", "#fde047"), ("G1", "#f87171"), ("G2", "#60a5fa"), ("Mapped", "#3d5a7f")]:
            tk.Label(leg, text=f"■ {txt}", font=("Arial", 9), bg="#0f1923", fg=col).pack(side=tk.LEFT, padx=6)

        self.status = tk.Label(self.root, text="WASD/Arrows to play, or PBAI Auto",
                               font=("Arial", 11), bg="#0f1923", fg="#6ee7b7")
        self.status.pack(pady=4)
        self.stats = tk.Label(self.root, text="", font=("Arial", 10), bg="#0f1923", fg="#94a3b8")
        self.stats.pack(pady=4)

    def new_maze(self):
        self.stop_pbai()
        self.maze.generate()
        self.player_pos = self.maze.start
        self.won = False

        if self.driver:
            self.driver.new_maze(self.player_pos, self.maze.goal1, self.maze.goal2)
            # Initial observation at start
            goals = [self.maze.goal1, self.maze.goal2]
            self.driver.observe(self.player_pos, self.maze.grid, self.maze_size, goals)

        self.draw()
        self.status.config(text="WASD/Arrows or PBAI Auto", fg="#6ee7b7")

    def restart(self):
        self.stop_pbai()
        self.player_pos = self.maze.start
        self.won = False

        if self.driver:
            self.driver.new_maze(self.player_pos, self.maze.goal1, self.maze.goal2)
            goals = [self.maze.goal1, self.maze.goal2]
            self.driver.observe(self.player_pos, self.maze.grid, self.maze_size, goals)

        self.draw()

    def draw(self):
        self.canvas.delete("all")

        # Get mapped cells from driver
        mapped = self.driver.observed_cells if self.driver else {}

        for r in range(self.maze_size):
            for c in range(self.maze_size):
                x1, y1 = c * self.cell_size, r * self.cell_size
                x2, y2 = x1 + self.cell_size, y1 + self.cell_size

                if self.maze.grid[r][c] == 1:
                    color = "#374151"  # Wall
                elif (r, c) in mapped:
                    v = mapped[(r, c)]
                    color = "#1e3a5f" if v == 1 else "#2d4a6f" if v == 2 else "#3d5a7f"
                else:
                    color = "#0a1929"  # Unmapped

                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="#111827")

        # Goals
        for pos, col, lbl in [(self.maze.goal1, "#f87171", "G1"), (self.maze.goal2, "#60a5fa", "G2")]:
            if pos:
                r, c = pos
                x1, y1 = c * self.cell_size + 4, r * self.cell_size + 4
                x2, y2 = x1 + self.cell_size - 8, y1 + self.cell_size - 8
                self.canvas.create_oval(x1, y1, x2, y2, fill=col, outline="white", width=2)
                self.canvas.create_text((x1+x2)//2, (y1+y2)//2, text=lbl, fill="white", font=("Arial", 8, "bold"))

        # Player
        r, c = self.player_pos
        x1, y1 = c * self.cell_size + 5, r * self.cell_size + 5
        x2, y2 = x1 + self.cell_size - 10, y1 + self.cell_size - 10
        self.canvas.create_oval(x1, y1, x2, y2, fill="#fde047", outline="#facc15", width=2)

        self.update_stats()

    def update_stats(self):
        if self.driver:
            mapped = len(self.driver.observed_cells)
            nodes = len(self.driver.manifold.nodes)
            self.stats.config(text=f"Steps: {self.driver.total_steps} | Mapped: {mapped} | Nodes: {nodes}")
        else:
            self.stats.config(text="")

    def move(self, d: str) -> bool:
        if d not in DIRECTIONS:
            return False
        dr, dc = DIRECTIONS[d]
        nr, nc = self.player_pos[0] + dr, self.player_pos[1] + dc
        if 0 <= nr < self.maze_size and 0 <= nc < self.maze_size and self.maze.grid[nr][nc] == 0:
            self.player_pos = (nr, nc)
            return True
        return False

    def check_win(self) -> bool:
        if self.player_pos == self.maze.goal1:
            self.won = True
            self.status.config(text="🎉 Goal 1!", fg="#f87171")
            return True
        if self.player_pos == self.maze.goal2:
            self.won = True
            self.status.config(text="🎉 Goal 2!", fg="#60a5fa")
            return True
        return False

    def key_press(self, event):
        if self.won or self.pbai_on:
            return
        key = event.keysym.lower()
        d = {'up': 'N', 'w': 'N', 'down': 'S', 's': 'S', 'left': 'W', 'a': 'W', 'right': 'E', 'd': 'E'}.get(key)
        if d and self.move(d):
            if self.driver:
                self.driver.record_move(self.player_pos, d)
                # Observe new position (maps it)
                goals = [self.maze.goal1, self.maze.goal2]
                self.driver.observe(self.player_pos, self.maze.grid, self.maze_size, goals)
                # Route through EnvironmentCore for psychology
                if self.env_core:
                    from drivers.environment import Action
                    perception = self.env_core.perceive()
                    result = self.env_core.act(Action(action_type=d, target=d))
                    self.env_core.feedback(result, Action(action_type=d, target=d))
            self.draw()
            self.check_win()

    # ══════════════════════════════════════════════════════════════════════════
    # PBAI AUTO MODE
    # ══════════════════════════════════════════════════════════════════════════

    def init_driver(self):
        if self.driver:
            return
        from core import get_pbai_manifold, get_growth_path
        from drivers.maze_driver import MazeDriver
        from drivers.environment import EnvironmentCore

        # Get the ONE PBAI manifold (loads existing or births on first run)
        self.growth_path = get_growth_path("growth_map.json")
        manifold = get_pbai_manifold(self.growth_path)

        self.driver = MazeDriver(manifold=manifold)
        self.env_core = EnvironmentCore(manifold=manifold)
        self.env_core.register_driver(self.driver)
        self.env_core.activate_driver("maze")

        self.driver.new_maze(self.player_pos, self.maze.goal1, self.maze.goal2)

        # Initial observation
        goals = [self.maze.goal1, self.maze.goal2]
        self.driver.observe(self.player_pos, self.maze.grid, self.maze_size, goals)

    def save_growth(self):
        """Save to unified growth map."""
        if self.driver and hasattr(self, 'growth_path'):
            self.driver.manifold.save_growth_map(self.growth_path)

    def toggle_pbai(self):
        if self.pbai_on:
            self.stop_pbai()
        else:
            self.start_pbai()

    def start_pbai(self):
        if self.won:
            self.new_maze()
        self.init_driver()
        self.pbai_on = True
        self.pbai_btn.config(text="⏹ Stop", bg="#ef4444")
        self.status.config(text="PBAI mapping...", fg="#22c55e")
        self.pbai_step()

    def stop_pbai(self):
        self.pbai_on = False
        self.pbai_btn.config(text="▶ PBAI Auto", bg="#22c55e")

    def pbai_step(self):
        if not self.pbai_on:
            return

        goals = [self.maze.goal1, self.maze.goal2]

        # Observe current position (updates map, builds internal state)
        obs = self.driver.observe(self.player_pos, self.maze.grid, self.maze_size, goals)

        # Win?
        if obs['at_goal']:
            self.won = True
            self.driver.record_completion()
            self.save_growth()
            self.status.config(text=f"🎉 Maze {self.driver.maze_count} mapped! Next...", fg="#22c55e")
            self.draw()
            self.root.after(1500, self.pbai_next_maze)
            return

        # Dead end? Backtrack via BFS to nearest cell with unexplored neighbor.
        if self.driver.is_dead_end(obs):
            target = self.driver.get_backtrack_target()
            if target and target != self.player_pos:
                moves = self.driver.get_backtrack_path(target)
                if moves:
                    self.status.config(text="Backtracking...", fg="#fb923c")
                    self.do_backtrack(moves, 0)
                    return

        # Route through EnvironmentCore: perceive → tick → decide
        perception = self.env_core.perceive()  # Calls driver.perceive() → Clock
        action = self.env_core.decide(perception)  # DecisionNode picks direction
        d = action.action_type

        if d and self.move(d):
            self.driver.record_move(self.player_pos, d)
            result = self.env_core.act(action)  # Calls driver.act() → ActionResult
            self.env_core.feedback(result, action)  # Psychology updates

        self.draw()
        try:
            delay = int(self.delay_var.get())
        except:
            delay = 100
        self.root.after(delay, self.pbai_step)

    def do_backtrack(self, moves: List[str], i: int):
        if not self.pbai_on or i >= len(moves):
            self.root.after(50, self.pbai_step)
            return
        d = moves[i]
        if self.move(d):
            self.driver.record_move(self.player_pos, d, backtracking=True)
            # Observe during backtrack (reinforces map)
            goals = [self.maze.goal1, self.maze.goal2]
            self.driver.observe(self.player_pos, self.maze.grid, self.maze_size, goals)
            # Route backtrack through EnvironmentCore for psychology
            from drivers.environment import Action
            perception = self.env_core.perceive()
            result = self.env_core.act(Action(action_type=d, target=d))
            self.env_core.feedback(result, Action(action_type=d, target=d))
            self.draw()
        self.root.after(30, lambda: self.do_backtrack(moves, i + 1))

    def pbai_next_maze(self):
        if self.pbai_on:
            self.maze.generate()
            self.player_pos = self.maze.start
            self.won = False
            self.driver.new_maze(self.player_pos, self.maze.goal1, self.maze.goal2)
            # Initial observation
            goals = [self.maze.goal1, self.maze.goal2]
            self.driver.observe(self.player_pos, self.maze.grid, self.maze_size, goals)
            self.draw()
            self.pbai_step()


if __name__ == "__main__":
    root = tk.Tk()
    MazeApp(root)
    root.mainloop()
