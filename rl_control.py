import os
import time
import numpy as np
import random
from plotter import Plotter

class RLControl:
    def __init__(self, env, alpha=0.1, epsilon=0.15, # epsilon_decay=0.995, min_epsilon=0.15,
                 gamma=0.95, num_of_episodes=1000, max_steps=100, plots_dir=None, algorithm_name="RL"):
        self.env = env
        self.alpha = alpha  # step size
        self.epsilon = epsilon  # exploration rate, for epsilon-greedy policy
        # self.epsilon_decay = epsilon_decay
        # self.min_epsilon = min_epsilon  # minimum value of epsilon after decay
        self.gamma = gamma  # discount factor
        self.num_of_episodes = num_of_episodes
        self.max_steps = max_steps
        self.plots_dir = plots_dir or os.path.join(os.getcwd(), "plots", algorithm_name.lower())
        self.algorithm_name = algorithm_name
        # ACTIONS = [0 (LEFT), 1 (RIGHT), 2 (UP), 3 (DOWN)]

        # initialize:
        # Q(s, a) arbitrarily for all non-terminal states s ∈ S and a ∈ A(s)
        # Q values are zero for all terminal states by definition
        self.Q = np.zeros((env.size, env.size, env.ACTION_SIZE))

        # counts accumulative number of times (over a set of episodes) a has been taken at s
        self.N = np.zeros((env.size, env.size, env.ACTION_SIZE))

        # folder to save plots for visualization
        self.info = f"num_of_episodes_{self.num_of_episodes}, max_steps_{self.max_steps}, epsilon_{self.epsilon}"

        # to store data for analysis
        self.plotter = Plotter(self, algorithm_name)
        self.episodes_length = []  # length of each episode
        self.success = []  # success (1) or failure (0) for each episode
        self.episodes_reward = []  # total reward accumulated by the agent for an episode

    def _epsilon_greedy_policy(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.env.ACTION_KEYS)  # explore: choose a random action
        else:
            return self._select_greedy_action(state)  # exploit: choose the best action

    def _select_greedy_action(self, state):
        q_values = self.Q[state[0], state[1], :]
        max_q_value = np.max(q_values)
        best_actions = np.where(q_values == max_q_value)[0]  # especially for multiple best actions (same highest q value)
        return int(random.choice(best_actions))

    def _train(self):
        raise NotImplementedError("Subclasses must implement this method")

    def extract_optimal_policy(self):
        start_time = time.time()
        self._train()
        policy = np.zeros((self.env.size, self.env.size))
        for i in range(self.env.size):
            for j in range(self.env.size):
                policy[i, j] = self._select_greedy_action((i, j))
        end_time = time.time()
        print(f"Time taken: {end_time - start_time} seconds")
        self.env.visualize_deterministic_policy(policy=policy, title=f"{self.algorithm_name.lower()}_optimal_policy", info=self.info, plots_dir=self.plots_dir)
        return policy

    def plot_q_values(self, title="Q(s,a)", info=""):
        self.env.plot_heatmap(data=self.Q, title=title, info=info, plots_dir=self.plots_dir)

    def plot_N_values(self, title="N(s,a)", info=""):
        self.env.plot_heatmap(data=self.N, title=title, info=info, plots_dir=self.plots_dir)