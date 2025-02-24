from frozen_lake_env import FrozenLakeEnv
import time
import numpy as np
import random

# first visit monte carlo without exploring starts
class MonteCarloControl:
    def __init__(self, env, epsilon=0.5, epsilon_decay=0.995, min_epsilon=0.15,
                    gamma=0.9, num_of_episodes=1000, max_steps=100): 
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
        self.policy = np.full((env.size, env.size, env.ACTION_SIZE), self.epsilon / env.ACTION_SIZE)
        # (2) Q(s,a), arbitrarily (usually 0), for all s ∈ S, a ∈ A(s)
        # -> Q[i, j, a] = Q value of taking action a at state (i, j)
        self.Q = np.zeros((env.size, env.size, env.ACTION_SIZE)) # self.Q = np.random.uniform(low=-0.1, high=0.1, size=(env.size, env.size, self.env.ACTION_SIZE))
        # (3) Returns(s,a) as an empty list, for all s ∈ S, a ∈ A(s)
        # -> returns[(i, j)][a] = list of returns for taking action a at state (i, j)
        self.returns = { (i, j): { a: [] for a in env.ACTION_KEYS } for i in range(env.size) for j in range(env.size) }

    def _generate_episode(self):
        episode = [] # (S0, A0, R1, S1, A1, R2, ..., ST-1, AT-1, RT)
        state = self.env.reset()
        step_count = 0
        terminated = False
        while not terminated and step_count < self.max_steps:
            weights = self.policy[state[0], state[1], :]
            normalized_weights = weights/ sum(weights)
            action = random.choices(self.env.ACTION_KEYS, weights=normalized_weights)[0] # choose an action based on the probabilities
            next_state, reward, terminated = self.env.step(action)
            episode.append((state, action, reward))
            state = next_state
            step_count += 1
        return episode

    def _update_q(self, episode):
        G = 0 # return
        for t in range(len(episode) - 1, -1, -1): # Loop for each step of episode, t = T−1, T−2, ..., 0
            state, action, reward = episode[t]
            G = reward + self.gamma * G
            if (state, action) not in [x[:2] for x in episode[:t]]: # if state-action pair not in { S0,A0, S1,A1, ..., St−1,At−1 }  (aka first visit to this state)
                self.returns[state][action].append(G) # Append G to Return(St,At)
                self.Q[state[0], state[1], action] = np.mean(self.returns[state][action]) # Q(St,At) ← average(Return(St,At))

    def _select_greedy_action(self, state):
        q_values = self.Q[state[0], state[1], :]
        max_q_value = np.max(q_values)
        best_actions = np.where(q_values == max_q_value)[0] # especially for multiple best actions (same highest q value)
        return int(random.choice(best_actions))
    
    def _update_policy(self):
        for i in range(self.env.size):
            for j in range(self.env.size):
                best_action = self._select_greedy_action((i, j))
                # epsilon greedy policy improvement
                for a in self.env.ACTION_KEYS:
                    if a == best_action: # exploit with probability 1−ε+ε/|A(s)|
                        self.policy[i, j, a] = 1 - self.epsilon + self.epsilon / self.env.ACTION_SIZE
                    else: # explore with probability ε/|A(s)|
                        self.policy[i, j, a] = self.epsilon / self.env.ACTION_SIZE

    def _train(self, show_plot=False, save_plot=True, folder_name=""):
        print("Training for Monte Carlo Control...")
        print(f"\rEpisode 0/{self.num_of_episodes} - 0.00% complete", end='')
        for i in range(self.num_of_episodes):
            episode = self._generate_episode()
            self._update_q(episode)
            self._update_policy()
            self.env.reset()

            progress = (i + 1) / self.num_of_episodes * 100  # Percentage completion
            if (i + 1) % int(self.num_of_episodes * 0.05) == 0:
                print(f"\rEpisode {i + 1}/{self.num_of_episodes} - {progress:.2f}% complete, epsilon={self.epsilon}", end='')
                self.plot_q_values(title=f"mc_q_value_episode_{i + 1}", show_plot=show_plot, save_plot=save_plot, folder_name=folder_name)
                # self.plot_probabilistic_policy(title=f"policy_episode_{i + 1}", folder_name={folder_name})

            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def extract_optimal_policy(self, show_plot=False, save_plot=True, folder_name=""): # deterministic
        self._train(show_plot=show_plot, save_plot=save_plot, folder_name=folder_name)
        policy = np.zeros((self.env.size, self.env.size))
        for i in range(self.env.size):
            for j in range(self.env.size):
                policy[i, j] = self._select_greedy_action((i, j))
        return policy
    
    def plot_q_values(self, title="Q(s,a) (MC)", show_plot=False, save_plot=True, folder_name=""):
        self.env.plot_heatmap(data=self.Q, title=title, show_plot=show_plot, save_plot=save_plot, folder_name=folder_name)
                              
    def plot_probabilistic_policy(self, title="Policy (MC)", show_plot=False, save_plot=True, folder_name=""):
        self.env.plot_heatmap(data=self.policy, title=title, show_plot=show_plot, save_plot=save_plot, folder_name=folder_name)



if __name__ == "__main__":
    grid_input_4x4 = [
            "S...",
            ".H.H",
            "...H",
            "H..G"
            ]
    
    grid_input_8x8 = [
        "S...HH..",
        "..H....H",
        "...H....",
        ".H...H..",
        "..HH....",
        ".HH....H",
        ".H..H..H",
        "...H...G"
    ]

    grid_input_10x10 = [
        "S....H....",
        "...H...H..",
        ".H.......H",
        "..HH..H.H.",
        "..H.H.....",
        "H......H.H",
        ".HH..H....",
        ".H...H...H",
        "...H.H....",
        ".H...HH..G"
    ]

    env = FrozenLakeEnv(grid_input=grid_input_8x8)

    # env = FrozenLakeEnv(size=10)   
    # print(env.grid)
    # env.visualize_map()

    mc = MonteCarloControl(env=env, num_of_episodes=35000, max_steps=1000)


    start_time = time.time()
    policy = mc.extract_optimal_policy(folder_name="mc_8x8")
    end_time = time.time()
    time_taken = end_time - start_time
    print(f"Time taken: {time_taken} seconds")
   
    print(policy)
    env.visualize_deterministic_policy(policy, 
            title=f"Optimal Policy from MC (epsilon={mc.epsilon}, decay={mc.epsilon_decay}, min_epsilon={mc.min_epsilon}, gamma={mc.gamma}, episodes={mc.num_of_episodes}, max_steps={mc.max_steps})")