import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.patches as mpatches
import random
from collections import deque
import shutil

class FrozenLakeEnv:
    def __init__(self, size=4, grid_input=None, plots_dir="plots"):
        # assuming always square grid
        self.LEFT = 0
        self.RIGHT = 1
        self.UP = 2
        self.DOWN = 3
        self.ACTIONS = { self.LEFT: (0, -1), self.RIGHT: (0, 1), self.UP: (-1, 0), self.DOWN: (1, 0) }
        self.ACTION_KEYS = list(self.ACTIONS.keys())
        self.ACTION_SIZE = len(self.ACTIONS)
        self.HOLE_STATE_RATIO = 0.25 # proportion between the number of holes and the number of states should be 25%
        self.HOLE = -1
        
        if grid_input:
            # grid_input in the format of a list of strings, each string representing a row of the grid, and (S: start, G: goal, H: hole)
            self.size = len(grid_input)
            if not self.size == len(grid_input[0]):
                raise ValueError("Map must be square in size")
            self._extract_map(grid_input)
            if not self._is_valid_map():
                raise ValueError("Invalid map")        
        else:
            self.size = size
            self.start = (0, 0)
            self.goal = (size-1, size-1)
            self._generate_random_map()

        self.state = self.start # initial state
        self.visited_states = set()

        self.plots_dir = plots_dir
        if not os.path.exists(self.plots_dir):
            os.makedirs(self.plots_dir)

    def reset(self):
        self.state = self.start
        self.visited_states.clear()
        return self.state

    def step(self, action):
        # return new_state, reward, terminated
        new_state = (self.state[0] + self.ACTIONS[action][0], self.state[1] + self.ACTIONS[action][1])

        if not (0 <= new_state[0] < self.size and 0 <= new_state[1] < self.size):
            return self.state, -1, True # penalty for out of bounds, terminate
        
        self.state = new_state # within bounds, update state
        
        if self.grid[self.state] == self.HOLE: 
            return self.state, -1, True # fall into a hole
        elif self.state == self.goal: 
            return self.state, 1, True # reach the goal

        # NOTE: IS THIS TOO CHEATING???
        # if new_state in self.visited_states:
        #     return new_state, -0.2, False  # penalty for revisiting a state
        # else:
        #     self.visited_states.add(new_state)

        return self.state, 0, False # default, TODO: small penalty for each step taken?

    def _extract_map(self, grid_input):
        self.grid = np.zeros((self.size, self.size))
        num_holes = 0
        for i in range(self.size):
            for j in range(self.size):
                if grid_input[i][j] == 'S':
                    self.start = (i, j)
                elif grid_input[i][j] == 'G':
                    self.goal = (i, j)
                    self.grid[i, j] = 1
                elif grid_input[i][j] == 'H':
                    self.grid[i, j] = -1
                    num_holes += 1

        self.HOLE_STATE_RATIO = num_holes / (self.size * self.size)
        if self.HOLE_STATE_RATIO < 0.25:
            print("Warning: proportion between the number of holes and the number of states is less than 25%")

    def _generate_random_map(self):
        self.grid = np.zeros((self.size, self.size))  # 0: empty, -1: hole, 1: goal
        self.grid[self.goal] = 1
        num_holes_needed = int((self.size * self.size) * self.HOLE_STATE_RATIO)
        
        # Generate all possible positions except start and goal
        possible_positions = [(x, y) for x in range(self.size) for y in range(self.size) 
                            if (x, y) != self.start and (x, y) != self.goal]
        
        hole_positions = random.sample(possible_positions, num_holes_needed)
        for hole in hole_positions:
            self.grid[hole] = self.HOLE

        if not self._is_valid_map():
            self._generate_random_map()  # Retry if invalid

    def _is_valid_map(self):
        # BFS to check if there is a valid path from start to goal
        frontier = deque([self.start])
        visited = set()
        while frontier:
            current = frontier.popleft()
            visited.add(current)
            for action in self.ACTIONS.values():
                neighbor = (current[0] + action[0], current[1] + action[1])
                if (0 <= neighbor[0] < self.size and 0 <= neighbor[1] < self.size 
                        and self.grid[neighbor] != self.HOLE and neighbor not in visited):
                    if neighbor == self.goal: # early goal test
                        return True
                    frontier.append(neighbor)
                    visited.add(neighbor)
        return False

    def visualize_map(self, show=True):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_xticks(np.arange(self.size + 1))
        ax.set_yticks(np.arange(self.size + 1))
        ax.grid()
        ax.invert_yaxis()
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i, j] == self.HOLE:
                    color = 'black'
                elif (i, j) == self.start:
                    color = '#D1FFBD' # light green
                elif (i, j) == self.goal:
                    color = 'green'
                else:
                    color = 'white'
                ax.add_patch(plt.Rectangle((j, i), 1, 1, color=color))
        ax.scatter(self.state[1] + 0.5, self.state[0] + 0.5, color='blue', marker='x', s=200)
        if show:
            plt.show()
        return fig, ax

    def visualize_deterministic_policy(self, policy, title="Policy Visualization", info=""):
        fig, ax = self.visualize_map(show=False)
        plt.suptitle(title, fontsize=16)
        plt.title(info, fontsize=12)
        directions = {self.RIGHT: '→', self.UP: '↑', self.LEFT: '←', self.DOWN: '↓'}
        for i in range(self.size):
            for j in range(self.size):
                ax.text(j + 0.5, i + 0.5, directions[policy[i, j]], ha='center', va='center', color='blue', fontsize=16)
        plt.show()

    def plot_heatmap(self, data, title="", show_plot=False, save_plot=True, folder_name=""):
        rows, cols, _ = data.shape
        fig, ax = plt.subplots(figsize=(cols * 2, rows * 2))
        ax.set_title(title, fontsize=16)
        ax.set_xlim(0, cols)
        ax.set_ylim(0, rows)
        ax.set_xticks(np.arange(cols))
        ax.set_yticks(np.arange(rows))
        ax.set_xticklabels(np.arange(cols))
        ax.set_yticklabels(np.arange(rows))
        ax.invert_yaxis() 

        # Normalize data to [0, 1] for the colormap
        data_min, data_max = data.min(), data.max()
        data_normalized = (data - data_min) / (data_max - data_min)

        edge_color = "#77DD77"
        text_color = "black"
        # define triangle vertices for each action
        for i in range(rows):
            for j in range(cols):
                x_center, y_center = j + 0.5, i + 0.5 # center of the cell

                # Triangle 1: LEFT
                color1 = plt.cm.YlGn(data_normalized[i, j, 0]) 
                triangle1 = Polygon([
                    (x_center - 0.5, y_center + 0.5),
                    (x_center - 0.5, y_center - 0.5),
                    (x_center, y_center)
                ], closed=True, facecolor=color1, edgecolor=edge_color, linewidth=1)
                ax.add_patch(triangle1)
                ax.text(x_center - 0.25, y_center, f"{data[i, j, 0]:.2f}", ha="center", va="center", fontsize=8, color=text_color)

                # Triangle 2: RIGHT
                color2 = plt.cm.YlGn(data_normalized[i, j, 1])
                triangle2 = Polygon([
                    (x_center + 0.5, y_center + 0.5),
                    (x_center + 0.5, y_center - 0.5),
                    (x_center, y_center)
                ], closed=True, facecolor=color2, edgecolor=edge_color, linewidth=1)
                ax.add_patch(triangle2)
                ax.text(x_center + 0.25, y_center, f"{data[i, j, 1]:.2f}", ha="center", va="center", fontsize=8, color=text_color)

                # Triangle 3: DOWN
                color3 = plt.cm.YlGn(data_normalized[i, j, 3]) 
                triangle3 = Polygon([
                    (x_center - 0.5, y_center + 0.5),
                    (x_center + 0.5, y_center + 0.5),
                    (x_center, y_center)
                ], closed=True, facecolor=color3, edgecolor=edge_color, linewidth=1)
                ax.add_patch(triangle3)
                ax.text(x_center, y_center + 0.25, f"{data[i, j, 3]:.2f}", ha="center", va="center", fontsize=8, color=text_color)

                # Triangle 4: UP
                color4 = plt.cm.YlGn(data_normalized[i, j, 2]) 
                triangle4 = Polygon([
                    (x_center - 0.5, y_center - 0.5),
                    (x_center + 0.5, y_center - 0.5),
                    (x_center, y_center)
                ], closed=True, facecolor=color4, edgecolor=edge_color, linewidth=1)
                ax.add_patch(triangle4)
                ax.text(x_center, y_center - 0.25, f"{data[i, j, 2]:.2f}", ha="center", va="center", fontsize=8, color=text_color)

                if self.grid[i, j] == self.HOLE:
                    ax.text(x_center, y_center, "H", ha="center", va="center", fontsize=12, color=text_color, weight="bold")
                elif (i, j) == self.start:
                    ax.text(x_center, y_center, "S", ha="center", va="center", fontsize=12, color=text_color, weight="bold")
                elif (i, j) == self.goal:
                    ax.text(x_center, y_center, "G", ha="center", va="center", fontsize=12, color=text_color, weight="bold")

        sm = plt.cm.ScalarMappable(cmap=plt.cm.YlGn, norm=plt.Normalize(vmin=data.min(), vmax=data.max()))
        sm.set_array([])
        plt.colorbar(sm, ax=ax)

        # legend_elements = [
        #     mpatches.Patch(color='#D1FFBD', label='Start (S)'),
        #     mpatches.Patch(color='black', label='Hole (H)'),
        #     mpatches.Patch(color='green', label='Goal (G)')
        # ]
        # ax.legend(handles=legend_elements, loc='upper right')

        if show_plot:
            plt.show()
        if save_plot:
            if not os.path.exists(os.path.join(self.plots_dir, str(folder_name))):
                os.makedirs(os.path.join(self.plots_dir, str(folder_name)))
            plt.savefig(os.path.join(self.plots_dir, str(folder_name), f"{title}.png"))
            plt.close()
        return fig, ax
        
if __name__ == "__main__":
    grid_input = [
                "S...",
                ".H.H",
                "...H",
                "H..G"
                ]
    # env = FrozenLakeEnv(grid_input=grid_input)
    env = FrozenLakeEnv(size=10)
    print(env.grid)
    env.visualize_map()