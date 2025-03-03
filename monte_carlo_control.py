from frozen_lake_env import FrozenLakeEnv
from rl_control import RLControl
import numpy as np
import os
from gif_maker import create_gif_from_folder
from grid_inputs import grid_input_4x4, grid_input_8x8, grid_input_10x10


# first visit monte carlo without exploring starts

class MonteCarloControl(RLControl):
    def __init__(self, env, **kwargs):
        # initialize:
        # (1) an arbitrary ε-soft policy π 
        # -> policy[i, j, a] = probability of taking action a at state (i, j) -> all actions are equally likely to be chosen at the start
        # self.policy = np.full((env.size, env.size, env.ACTION_SIZE), self.epsilon / env.ACTION_SIZE)
        # (2) Q(s,a), arbitrarily (usually 0), for all s ∈ S, a ∈ A(s)
        # -> Q[i, j, a] = Q value of taking action a at state (i, j)

        super().__init__(env, algorithm_name="MC", **kwargs)
        # (3) Returns(s,a) as an empty list, for all s ∈ S, a ∈ A(s)
        # -> returns[(i, j)][a] = list of returns for taking action a at state (i, j)
        self.returns = {(i, j): {a: [] for a in env.ACTION_KEYS} for i in range(env.size) for j in range(env.size)}

    def _generate_episode(self):
        episode = []  # (S0, A0, R1, S1, A1, R2, ..., ST-1, AT-1, RT)
        state = self.env.reset()
        step_count = 0
        terminated = False
        episode_reward = 0
        while not terminated and step_count < self.max_steps:
            action = self._epsilon_greedy_policy(state)
            next_state, reward, terminated = self.env.step(action)
            episode.append((state, action, reward))
            state = next_state
            step_count += 1
            episode_reward += reward

        # data for analysis
        if reward == 1:  # final reward
            self.success.append(1)
        else:
            self.success.append(0)
        self.episodes_length.append(step_count)
        self.episodes_reward.append(episode_reward)
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
        first_occurrence = {}  # first occurrence of each (state, action) in the episode
        for t, (state, action, _) in enumerate(episode):
            sa = (state, action)
            if sa not in first_occurrence:
                first_occurrence[sa] = t
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = reward + self.gamma * G
            if first_occurrence.get((state, action), -1) == t:  # Check if this is the first occurrence in the episode
                self.returns[state][action].append(G)  # Append G to returns and update Q with the new mean
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
        print(f"Training for {self.algorithm_name} Control...")
        print(f"\rEpisode 0/{self.num_of_episodes} - 0.00% complete", end='')
        for i in range(self.num_of_episodes):
            episode = self._generate_episode()
            self._update_q(episode)
            # self._update_policy()
            self.env.reset()

            progress = (i + 1) / self.num_of_episodes * 100
            if (i + 1) % int(self.num_of_episodes * 0.025) == 0:
                print(f"\rEpisode {i + 1}/{self.num_of_episodes} - {progress:.2f}% complete, epsilon={self.epsilon}", end='')
                self.plot_q_values(title=f"{self.algorithm_name.lower()}_q_episode_{i + 1}", info=f"Q_value_episode_{i + 1} ({self.info})")
            
            # decay 
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        print(f"\n{self.algorithm_name} training complete!")
        create_gif_from_folder(self.plots_dir, f"{os.path.basename(self.plots_dir)}_q.gif", "q_episode")
        

if __name__ == "__main__":
    env_4x4 = FrozenLakeEnv(grid_input=grid_input_4x4)
    mc_4x4 = MonteCarloControl(env=env_4x4, 
                               num_of_episodes=5000, 
                               max_steps=100, 
                               epsilon=0.15, 
                               plots_dir=os.path.join(os.getcwd(), "plots", "mc_4x4"))
    policy = mc_4x4.extract_optimal_policy()

    mc_4x4.plotter.plot_episode_length_over_time(show=True)
    mc_4x4.plotter.plot_episodic_reward_over_time()
    mc_4x4.plotter.plot_success_rate_over_time()
    mc_4x4.plotter.plot_success_failure_bar()


    env_10x10 = FrozenLakeEnv(grid_input=grid_input_10x10)
    mc_10x10 = MonteCarloControl(env=env_10x10, 
                                 num_of_episodes=5000, 
                                 max_steps=100, 
                                 epsilon=0.15, 
                                 plots_dir=os.path.join(os.getcwd(), "plots", "mc_10x10"))
    policy = mc_10x10.extract_optimal_policy()

    mc_10x10.plotter.plot_episode_length_over_time(show=True)
    mc_10x10.plotter.plot_episodic_reward_over_time()
    mc_10x10.plotter.plot_success_rate_over_time()
    mc_10x10.plotter.plot_success_failure_bar()
