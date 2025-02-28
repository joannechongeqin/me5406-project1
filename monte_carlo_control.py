from frozen_lake_env import FrozenLakeEnv
import time
import numpy as np
import random
import os
from gif_maker import create_gif_from_folder
from grid_inputs import grid_input_4x4, grid_input_8x8, grid_input_10x10

# first visit monte carlo without exploring starts
class MonteCarloControl:
    def __init__(self, env, epsilon=0.15, epsilon_decay=0.995, min_epsilon=0.15,
                    gamma=0.99, num_of_episodes=1000, max_steps=100, plots_dir=os.path.join(os.getcwd(), "plots", "mc")): 
        self.env = env
        self.epsilon = epsilon # exploration rate, for epsilon-greedy policy
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon # minimum value of epsilon after decay
        self.gamma = gamma # discount factor
        self.num_of_episodes = num_of_episodes
        self.max_steps = max_steps
        # ACTIONS = [ 0 (LEFT), 1 (RIGHT), 2 (UP), 3 (DOWN) ]

        # initialize:
        # (1) an arbitrary ε-soft policy π 
        # -> policy[i, j, a] = probability of taking action a at state (i, j) -> all actions are equally likely to be chosen at the start
        # self.policy = np.full((env.size, env.size, env.ACTION_SIZE), self.epsilon / env.ACTION_SIZE)
        # (2) Q(s,a), arbitrarily (usually 0), for all s ∈ S, a ∈ A(s)
        # -> Q[i, j, a] = Q value of taking action a at state (i, j)
        self.Q = np.zeros((env.size, env.size, env.ACTION_SIZE)) # self.Q = np.random.uniform(low=-0.1, high=0.1, size=(env.size, env.size, self.env.ACTION_SIZE))
        # (3) Returns(s,a) as an empty list, for all s ∈ S, a ∈ A(s)
        # -> returns[(i, j)][a] = list of returns for taking action a at state (i, j)
        self.returns = { (i, j): { a: [] for a in env.ACTION_KEYS } for i in range(env.size) for j in range(env.size) }

        self.plots_dir = plots_dir

    def _epsilon_greedy_policy(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.env.ACTION_KEYS)  # explore: choose a random action
        else:
            return self._select_greedy_action(state)  # exploit: choose the best action

    def _select_greedy_action(self, state):
        q_values = self.Q[state[0], state[1], :]
        max_q_value = np.max(q_values)
        best_actions = np.where(q_values == max_q_value)[0] # especially for multiple best actions (same highest q value)
        return int(random.choice(best_actions))
    
    def _generate_episode(self):
        episode = [] # (S0, A0, R1, S1, A1, R2, ..., ST-1, AT-1, RT)
        state = self.env.reset()
        step_count = 0
        terminated = False
        while not terminated and step_count < self.max_steps:
            # weights = self.policy[state[0], state[1], :]
            # normalized_weights = weights/ sum(weights)
            action = self._epsilon_greedy_policy(state) # random.choices(self.env.ACTION_KEYS, weights=normalized_weights)[0] # choose an action based on the probabilities
            next_state, reward, terminated = self.env.step(action)
            episode.append((state, action, reward))
            state = next_state
            step_count += 1
        state = self.env.reset()
        return episode

    # def _update_q(self, episode): # SUPER SLOW VERSION DO NOT USE
    #     G = 0 # return
    #     for t in range(len(episode) - 1, -1, -1): # Loop for each step of episode, t = T−1, T−2, ..., 0
    #         state, action, reward = episode[t]
    #         G = reward + self.gamma * G
    #         if (state, action) not in [x[:2] for x in episode[:t]]: # if state-action pair not in { S0,A0, S1,A1, ..., St−1,At−1 }  (aka first visit to this state)
    #             self.returns[state][action].append(G) # Append G to Return(St,At)
    #             self.Q[state[0], state[1], action] = np.mean(self.returns[state][action]) # Q(St,At) ← average(Return(St,At))

    def _update_q(self, episode):
        G = 0  # return
        first_occurrence = {} # first occurrence of each (state, action) in the episode
        for t, (state, action, _) in enumerate(episode):
            sa = (state, action)
            if sa not in first_occurrence:
                first_occurrence[sa] = t
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = reward + self.gamma * G
            if first_occurrence.get((state, action), -1) == t: # Check if this is the first occurrence in the episode
                self.returns[state][action].append(G) # Append G to returns and update Q with the new mean
                self.Q[state[0], state[1], action] = np.mean(self.returns[state][action])
    
    # def _update_policy(self):
    #     for i in range(self.env.size):
    #         for j in range(self.env.size):
    #             best_action = self._select_greedy_action((i, j))
    #             # epsilon greedy policy improvement
    #             for a in self.env.ACTION_KEYS:
    #                 if a == best_action: # exploit with probability 1−ε+ε/|A(s)|
    #                     self.policy[i, j, a] = 1 - self.epsilon + self.epsilon / self.env.ACTION_SIZE
    #                 else: # explore with probability ε/|A(s)|
    #                     self.policy[i, j, a] = self.epsilon / self.env.ACTION_SIZE

    def _train(self):
        print("Training for Monte Carlo Control...")
        print(f"\rEpisode 0/{self.num_of_episodes} - 0.00% complete", end='')
        for i in range(self.num_of_episodes):
            episode = self._generate_episode()
            self._update_q(episode)
            # self._update_policy()
            self.env.reset()

            progress = (i + 1) / self.num_of_episodes * 100  # Percentage completion
            if (i + 1) % int(self.num_of_episodes * 0.025) == 0:
                print(f"\rEpisode {i + 1}/{self.num_of_episodes} - {progress:.2f}% complete, epsilon={self.epsilon}", end='')
                self.plot_q_values(title=f"mc_q_episode_{i + 1}", info=f"Q_value_episode_{i + 1} (num_of_episodes_{self.num_of_episodes}, max_steps_{self.max_steps}, epsilon_{self.epsilon})")
            
            # decay
            # self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        
        create_gif_from_folder(self.plots_dir, f"{os.path.basename(self.plots_dir)}.gif")
        print("\nMonte Carlo training complete!")
        
    def extract_optimal_policy(self): # deterministic
        self._train()
        policy = np.zeros((self.env.size, self.env.size))
        for i in range(self.env.size):
            for j in range(self.env.size):
                policy[i, j] = self._select_greedy_action((i, j))
        self.env.visualize_deterministic_policy(policy=policy, title="mc_optimal_policy", info=f"num_of_episodes_{self.num_of_episodes}, max_steps_{self.max_steps}, epsilon_{self.epsilon}", plots_dir=self.plots_dir)
        return policy
    
    def plot_q_values(self, title="Q(s,a) (MC)", info=""):
        self.env.plot_heatmap(data=self.Q, title=title, info=info, plots_dir=self.plots_dir)


if __name__ == "__main__":
    
    # env_4x4 = FrozenLakeEnv(grid_input=grid_input_4x4)
    # mc_4x4 = MonteCarloControl(env=env_4x4, num_of_episodes=5000, max_steps=5000, epsilon=0.15, plots_dir=os.path.join(os.getcwd(), "plots", "mc_4x4"))
    # start_time_4x4 = time.time()
    # policy = mc_4x4.extract_optimal_policy()
    # end_time_4x4 = time.time()
    # time_taken_4x4 = end_time_4x4 - start_time_4x4
    # print(f"Time taken: {time_taken_4x4} seconds")
    # print("mc_4x4 policy:\n", policy)

    # env_8x8 = FrozenLakeEnv(grid_input=grid_input_8x8)
    # mc_8x8 = MonteCarloControl(env=env_8x8, num_of_episodes=50000, max_steps=15000, epsilon=0.5, plots_dir=os.path.join(os.getcwd(), "plots", "mc_8x8"))
    # start_time_8x8 = time.time()
    # policy = mc_8x8.extract_optimal_policy()
    # end_time_8x8 = time.time()
    # time_taken_8x8 = end_time_8x8 - start_time_8x8
    # print(f"Time taken: {time_taken_8x8} seconds")

    env_10x10 = FrozenLakeEnv(grid_input=grid_input_10x10)
    mc_10x10 = MonteCarloControl(env=env_10x10, num_of_episodes=100000, max_steps=50000, epsilon=0.3, gamma=0.99, plots_dir=os.path.join(os.getcwd(), "plots", "mc_10x10"))
    start_time_10x10 = time.time()
    policy = mc_10x10.extract_optimal_policy()
    end_time_10x10 = time.time()
    time_taken_10x10 = end_time_10x10 - start_time_10x10
    print(f"Time taken: {time_taken_10x10} seconds")