from frozen_lake_env import FrozenLakeEnv
import os
import numpy as np
import random
import time
from gif_maker import create_gif_from_folder
from grid_inputs import grid_input_4x4, grid_input_8x8, grid_input_10x10
import matplotlib.pyplot as plt
from plotter import Plotter

class SarsaControl:
    def __init__(self, env, alpha=0.1, epsilon=0.15, epsilon_decay=0.995, min_epsilon=0.15,
                 gamma=0.95, num_of_episodes=1000, max_steps=100, plots_dir=os.path.join(os.getcwd(), "plots", "sarsa")): 
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

        # folder to save plots for visualization
        self.plots_dir = plots_dir
        self.info = f"num_of_episodes_{self.num_of_episodes}, max_steps_{self.max_steps}, epsilon_{self.epsilon}"

        # to store data for analysis
        self.plotter = Plotter(self, "SARSA")
        self.episodes_length = [] # length of each episode
        self.success_count = 0
        self.failure_count = 0
        self.episodes_reward = [] # total reward accumulated by the agent for an episode 
        self.mse_values = [] # TODO: mean squared error for convergence rate


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
    
    def _train(self):
        print("Training for SARSA Control...")
        print(f"\rEpisode 0/{self.num_of_episodes} - 0.00% complete", end='')

        for i in range(self.num_of_episodes): # loop for each episode
            # initialize state
            state = self.env.reset()

            # choose action A from state S using policy derived from Q (e.g., ϵ-greedy)
            action = self._epsilon_greedy_policy(state) 
            step_count = 0
            terminated = False
            episode_reward = 0 # its going to be only either 0 or 1 
        
            # repeat for each step of episode:
            while not terminated and step_count < self.max_steps:
                # take action A; receive R and observe S'
                next_state, reward, terminated = self.env.step(action)
                # choose A' from S' using policy derived from Q (e.g., ϵ-greedy)
                next_action = self._epsilon_greedy_policy(next_state)
                episode_reward += reward

                # update Q(S, A) using SARSA update rule: Q(S, A) ← Q(S, A) + α[R + γQ(S', A') - Q(S, A)]
                self.Q[state[0], state[1], action] += self.alpha * (reward + self.gamma * self.Q[next_state[0], next_state[1], next_action] - self.Q[state[0], state[1], action])
                # Update the counter N(S, A)
                self.N[state[0], state[1], action] += 1
                # S ← S'; A ← A'; 
                state = next_state
                action = next_action
                step_count += 1
            # until S is terminal state

            progress = (i + 1) / self.num_of_episodes * 100  # Percentage completion
            if (i + 1) % int(self.num_of_episodes * 0.025) == 0:                
                print(f"\rEpisode {i + 1}/{self.num_of_episodes} - {progress:.2f}% complete, epsilon={self.epsilon}", end='')
                self.plot_q_values(title=f"sarsa_q_episode_{i + 1}", info=f"Q_value_episode_{i + 1} ({self.info})")
                self.plot_N_values(title=f"sarsa_N_episode_{i + 1}", info=f"N_value_episode_{i + 1} ({self.info})")

            # decay
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            
            # data for analysis
            if reward == 1:
                self.success_count += 1
            else:
                self.failure_count += 1
            self.episodes_length.append(step_count)
            self.episodes_reward.append(episode_reward)

        create_gif_from_folder(self.plots_dir, f"{os.path.basename(self.plots_dir)}_q.gif")
        create_gif_from_folder(self.plots_dir, f"{os.path.basename(self.plots_dir)}_N.gif")
        print("SARSA training complete!")
      
    def extract_optimal_policy(self): # deterministic
        start_time = time.time()
        self._train()
        policy = np.zeros((self.env.size, self.env.size))
        for i in range(self.env.size):
            for j in range(self.env.size):
                policy[i, j] = self._select_greedy_action((i, j))
        end_time = time.time()
        print(f"Time taken: {end_time - start_time} seconds")
        self.env.visualize_deterministic_policy(policy=policy, title="sarsa_optimal_policy", info=self.info, plots_dir=self.plots_dir)
        return policy

    def plot_q_values(self, title="Q(s,a) (SARSA)", info=""):
        self.env.plot_heatmap(data=self.Q, title=title, info=info, plots_dir=self.plots_dir)

    def plot_N_values(self, title="N(s,a) (SARSA)", info=""):
        self.env.plot_heatmap(data=self.N, title=title, info=info, plots_dir=self.plots_dir)

if __name__ == "__main__":
    
    # env_4x4 = FrozenLakeEnv(grid_input=grid_input_4x4)
    # sarsa_4x4 = SarsaControl(env=env_4x4, num_of_episodes=5000, max_steps=5000, epsilon=0.15, plots_dir=os.path.join(os.getcwd(), "plots", "sarsa_4x4"))
    # policy = sarsa_4x4.extract_optimal_policy()

    # env_8x8 = FrozenLakeEnv(grid_input=grid_input_8x8)
    # sarsa_8x8 = SarsaControl(env=env_8x8, num_of_episodes=10000, max_steps=15000, epsilon=0.15, plots_dir=os.path.join(os.getcwd(), "plots", "sarsa_8x8"))
    # policy = sarsa_8x8.extract_optimal_policy()

    env_10x10 = FrozenLakeEnv(grid_input=grid_input_10x10)
    sarsa_10x10 = SarsaControl(env=env_10x10, 
                               num_of_episodes=10000, 
                               max_steps=500, 
                               epsilon=0.15, 
                               gamma=0.99, 
                               min_epsilon=0.05, 
                               plots_dir=os.path.join(os.getcwd(), "plots", "sarsa_10x10"))
    policy = sarsa_10x10.extract_optimal_policy()

    sarsa_10x10.plotter.plot_episode_lengths(show=True)
    sarsa_10x10.plotter.plot_cummulative_average_rewards()
    sarsa_10x10.plotter.plot_success_failure_rate()
    # sarsa_10x10.plot_convergence()
