from frozen_lake_env import FrozenLakeEnv
from rl_control import RLControl
import os
from gif_maker import create_gif_from_folder
from grid_inputs import grid_input_4x4, grid_input_8x8, grid_input_10x10

class QLearningControl(RLControl):
    def __init__(self, env, **kwargs):
        super().__init__(env, algorithm_name="Q-Learning", **kwargs)

    def _train(self):
        print(f"Training for {self.algorithm_name} Control...")
        print(f"\rEpisode 0/{self.num_of_episodes} - 0.00% complete", end='')
        
        for i in range(self.num_of_episodes): # loop for each episode
            # initialize state
            state = self.env.reset()
            step_count = 0
            terminated = False
            episode_reward = 0

            # loop for each step of episode
            while not terminated and step_count < self.max_steps:
                # choose action A from state S using policy derived from Q (e.g., ϵ-greedy)
                action = self._epsilon_greedy_policy(state)
                # take action A; receive R and observe S'
                next_state, reward, terminated = self.env.step(action)
                episode_reward += reward

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
                self.plot_q_values(title=f"{self.algorithm_name.lower()}_q_episode_{i + 1}", info=f"Q_value_episode_{i + 1} ({self.info})")
                self.plot_N_values(title=f"{self.algorithm_name.lower()}_N_episode_{i + 1}", info=f"N_value_episode_{i + 1} ({self.info})")

            # decay
            # self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

            if reward == 1:
                self.success.append(1)
            else:
                self.success.append(0)
            self.episodes_length.append(step_count)
            self.episodes_reward.append(episode_reward)

        print(f"\n{self.algorithm_name} training complete!")
        create_gif_from_folder(self.plots_dir, f"{os.path.basename(self.plots_dir)}_q.gif", "q")
        create_gif_from_folder(self.plots_dir, f"{os.path.basename(self.plots_dir)}_N.gif", "N")
        

if __name__ == "__main__":
    env_4x4 = FrozenLakeEnv(grid_input=grid_input_4x4)
    q_learning_4x4 = QLearningControl(env=env_4x4, 
                                      num_of_episodes=5000, 
                                      max_steps=100, 
                                      epsilon=0.15, 
                                      plots_dir=os.path.join(os.getcwd(), "plots", "q_learning_4x4"))
    policy = q_learning_4x4.extract_optimal_policy()

    q_learning_4x4.plotter.plot_episode_length_over_time()
    q_learning_4x4.plotter.plot_episodic_average_reward_over_time()
    q_learning_4x4.plotter.plot_success_rate_over_time()
    q_learning_4x4.plotter.plot_success_failure_bar()


    env_10x10 = FrozenLakeEnv(grid_input=grid_input_10x10)
    q_learning_10x10 = QLearningControl(env=env_10x10, 
                                        num_of_episodes=5000, 
                                        max_steps=100, 
                                        epsilon=0.15, 
                                        plots_dir=os.path.join(os.getcwd(), "plots", "q_learning_10x10"))
    policy = q_learning_10x10.extract_optimal_policy()

    q_learning_10x10.plotter.plot_episode_length_over_time()
    q_learning_10x10.plotter.plot_episodic_average_reward_over_time()
    q_learning_10x10.plotter.plot_success_rate_over_time()
    q_learning_10x10.plotter.plot_success_failure_bar()