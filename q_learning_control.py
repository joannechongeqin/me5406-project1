from frozen_lake_env import FrozenLakeEnv
import os
import numpy as np
import random
import time
from gif_maker import create_gif_from_folder
from grid_inputs import grid_input_4x4, grid_input_8x8, grid_input_10x10

class QLearningControl:
    def __init__(self, env, alpha=0.1, epsilon=0.5, epsilon_decay=0.995, min_epsilon=0.15,
                 gamma=0.9, num_of_episodes=1000, max_steps=100, plots_dir=os.path.join(os.getcwd(), "plots", "qlearning")): 
        self.env = env
        self.alpha = alpha  # step size
        self.epsilon = epsilon  # exploration rate, for epsilon-greedy policy
        self.epsilon_decay = epsilon_decay 
        self.min_epsilon = min_epsilon  # minimum value of epsilon after decay
        self.gamma = gamma  # discount factor
        self.num_of_episodes = num_of_episodes
        self.max_steps = max_steps
        # ACTIONS = [0 (LEFT), 1 (RIGHT), 2 (UP), 3 (DOWN)]

        # initialize: Q(s, a) arbitrarily for all non-terminal states s ∈ S and a ∈ A(s)
        # Q values are zero for all terminal states by definition
        self.Q = np.zeros((env.size, env.size, env.ACTION_SIZE))

        # counts accumulative number of times (over a set of episodes) a has been taken at s
        self.N = np.zeros((env.size, env.size, env.ACTION_SIZE)) 

        self.plots_dir = plots_dir

    def _epsilon_greedy_policy(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.env.ACTION_KEYS)  # explore: choose a random action
        else:
            return self._select_greedy_action(state)  # exploit: choose the best action

    def _select_greedy_action(self, state):
        q_values = self.Q[state[0], state[1], :]
        max_q_value = np.max(q_values)
        best_actions = np.where(q_values == max_q_value)[0]  # especially for multiple best actions (same highest q value)
        chosen_action = random.choice(best_actions)
        return int(chosen_action)
    
    def _train(self, show_plot=False, save_plot=True, folder_name=""):
        print("Training for Q-Learning Control...")
        print(f"\rEpisode 0/{self.num_of_episodes} - 0.00% complete", end='')
        
        for i in range(self.num_of_episodes): # loop for each episode
            # initialize state
            state = self.env.reset()
            step_count = 0
            terminated = False

            # loop for each step of episode
            while not terminated and step_count < self.max_steps:
                # choose action A from state S using policy derived from Q (e.g., ϵ-greedy)
                action = self._epsilon_greedy_policy(state)
                # take action A; receive R and observe S'
                next_state, reward, terminated = self.env.step(action)
                # update Q(S, A) using Q-learning update rule: Q(S, A) ← Q(S, A) + α[R + γ max Q(S', A') - Q(S, A)]
                best_next_action = self._select_greedy_action(next_state)         
                self.Q[state[0], state[1], action] += self.alpha * (
                    reward + self.gamma * self.Q[next_state[0], next_state[1], best_next_action] - self.Q[state[0], state[1], action]
                )
                # Update the counter N(S, A)
                self.N[state[0], state[1], action] += 1
                # S ← S'
                state = next_state
                step_count += 1
            # until S is terminal state

            progress = (i + 1) / self.num_of_episodes * 100  
            if (i + 1) % int(self.num_of_episodes * 0.05) == 0:
                print(f"\rEpisode {i + 1}/{self.num_of_episodes} - {progress:.2f}% complete, epsilon={self.epsilon}", end='')
                self.plot_q_values(title=f"qlearning_q_episode_{i + 1}", info=f"Q_value_episode_{i + 1} (num_of_episodes_{self.num_of_episodes}, max_steps_{self.max_steps}, epsilon_{self.epsilon})")
                # self.plot_N_values(title=f"qlearning_N_episode_{i + 1}", info=f"Q_value_episode_{i + 1} (num_of_episodes_{self.num_of_episodes}, max_steps_{self.max_steps}, epsilon_{self.epsilon})")
            
            # decay
            # self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        create_gif_from_folder(self.plots_dir, f"{os.path.basename(self.plots_dir)}.gif")
        print("Q-Learning training complete!")

    def extract_optimal_policy(self):
        start_time = time.time()
        self._train()
        policy = np.zeros((self.env.size, self.env.size))
        for i in range(self.env.size):
            for j in range(self.env.size):
                policy[i, j] = self._select_greedy_action((i, j))
        end_time = time.time()
        print(f"Time taken: {end_time - start_time} seconds")
        self.env.reset()
        self.env.visualize_deterministic_policy(policy=policy, title="qlearning_optimal_policy", info=f"num_of_episodes_{self.num_of_episodes}, max_steps_{self.max_steps}, epsilon_{self.epsilon}", plots_dir=self.plots_dir)
        return policy

    def plot_q_values(self, title="Q(s,a) (QLearning)", info=""):
        self.env.plot_heatmap(data=self.Q, title=title, info=info, plots_dir=self.plots_dir)

    def plot_N_values(self, title="N(s,a) (QLearning)", info=""):
        self.env.plot_heatmap(data=self.N, title=title, info=info, plots_dir=self.plots_dir)


if __name__ == "__main__":
    
    # env_4x4 = FrozenLakeEnv(grid_input=grid_input_4x4)
    # qlearning_4x4 = QLearningControl(env=env_4x4, num_of_episodes=5000, max_steps=5000, epsilon=0.15, plots_dir=os.path.join(os.getcwd(), "plots", "qlearning_4x4"))
    # policy = qlearning_4x4.extract_optimal_policy()

    # env_8x8 = FrozenLakeEnv(grid_input=grid_input_8x8)
    # qlearning_8x8 = QLearningControl(env=env_8x8, num_of_episodes=50000, max_steps=15000, epsilon=0.5, plots_dir=os.path.join(os.getcwd(), "plots", "qlearning_8x8"))
    # policy = qlearning_8x8.extract_optimal_policy()

    env_10x10 = FrozenLakeEnv(grid_input=grid_input_10x10)
    qlearning_10x10 = QLearningControl(env=env_10x10, num_of_episodes=1000, max_steps=2000, epsilon=0.15, gamma=0.99, plots_dir=os.path.join(os.getcwd(), "plots", "qlearning_10x10"))
    policy = qlearning_10x10.extract_optimal_policy()
