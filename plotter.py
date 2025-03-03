import matplotlib.pyplot as plt
import os
import numpy as np

class BasePlotter:
    def __init__(self, rl_agents, names):
        self.rl_agents = rl_agents
        self.names = names
        self.env_size = self.rl_agents[0].env.size

    def _moving_average(self, data, window_size=50):
        pad_size = (window_size - 1) // 2
        padded_data = np.pad(data, pad_size, mode='edge')
        return np.convolve(padded_data, np.ones(window_size)/window_size, mode='valid')

    def save_plot(self, filename, show=False):
        plt.savefig(os.path.join(self.rl_agents[0].plots_dir, filename))
        if show:
            plt.show()
        plt.close()

    def save_plot_to_parent_folder(self, filename, show=False):
        plt.savefig(os.path.join(self.rl_agents[0].plots_dir, "..", filename))
        if show:
            plt.show()
        plt.close()

class Plotter(BasePlotter):
    def __init__(self, rl_agent, name):
        super().__init__([rl_agent], [name])
        self.rl_agent = rl_agent
        self.name = name

    def plot_episodic_reward_over_time(self, window_size=50, show=False):
        smoothed_reward = self._moving_average(self.rl_agent.episodes_reward, window_size)
        plt.figure(figsize=(10, 6))
        plt.plot(smoothed_reward, label='Episodic Reward', color='green', linewidth=2)
        plt.xlabel('Episodes')
        plt.ylabel('Reward (Reward per Step for each Episode)')
        plt.title(f'Episodic Reward Over Time for {self.env_size}x{self.env_size} grid ({self.name}, smoothed with window size {window_size})')
        plt.axhline(y=np.mean(self.rl_agent.episodes_reward), color='r', linestyle='--', label=f'Average Episodic Reward: {np.mean(self.rl_agent.episodes_reward):.2f}')
        plt.legend()
        plt.grid(True)
        self.save_plot('episodic_reward.png', show)

    def plot_episode_length_over_time(self, window_size=50, show=False):
        average_length = np.mean(self.rl_agent.episodes_length)
        smoothed_length = self._moving_average(self.rl_agent.episodes_length, window_size=50)
        plt.figure(figsize=(10, 6))
        plt.plot(smoothed_length, label='Episode Length', color='blue', linewidth=2)
        plt.xlabel('Episode')
        plt.ylabel('Episode Length')
        plt.suptitle(f'Episode Length over Time for {self.env_size}x{self.env_size} grid ({self.name}, smoothed with window size {window_size})', fontsize=16)
        plt.title(self.rl_agent.info, fontsize=10)
        plt.axhline(y=average_length, color='r', linestyle='--', label=f'Average Episode Length: {average_length:.2f}')
        plt.legend()
        plt.grid(True)
        self.save_plot('episode_lengths.png', show)

    def plot_success_failure_bar(self, show=False):
        num_of_episodes = self.rl_agent.num_of_episodes
        success_count = sum(self.rl_agent.success)
        failure_count = num_of_episodes - success_count
        success_percentage = (success_count / num_of_episodes) * 100
        failure_percentage = ((failure_count) / num_of_episodes) * 100
        bars = plt.bar(['Success', 'Failure'], [success_percentage, failure_percentage])
        plt.xlabel('Result')
        plt.ylabel('Percentage')
        plt.suptitle(f'Success and Failure Rate (during Training) for {self.env_size}x{self.env_size} grid ({self.name})', fontsize=16)
        plt.title(self.rl_agent.info, fontsize=10)
        plt.bar_label(bars, labels=[f"{success_count} / {num_of_episodes} ({success_percentage:.2f}%)", 
                                    f"{failure_count} / {num_of_episodes} ({failure_percentage:.2f}%)"])
        self.save_plot('success_failure_rate.png', show)

    def plot_success_rate_over_time(self, show=False):
        success_rate = np.cumsum(self.rl_agent.success) / np.arange(1, len(self.rl_agent.success) + 1) * 100
        plt.figure(figsize=(10, 6))
        plt.plot(success_rate, label='Success Rate', color='blue', linewidth=2)
        plt.xlabel('Episodes')
        plt.ylabel('Success Rate (%)')
        plt.title(f'Success Rate Over Time for {self.env_size}x{self.env_size} grid ({self.name})')
        plt.legend()
        plt.grid(True)
        self.save_plot('success_rate_over_time.png', show)

class ComparePlotter(BasePlotter):
    def __init__(self, rl_agents, names):
        super().__init__(rl_agents, names)

    def plot_episode_lengths_over_time(self, window_size=50, show=False):
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, agent in enumerate(self.rl_agents):
            average_length = np.mean(agent.episodes_length)
            smoothed_length = self._moving_average(agent.episodes_length, window_size)
            line = ax.plot(smoothed_length, label=f'Episode Length ({self.names[i]})', linewidth=2)
            color = line[0].get_color()
            ax.axhline(y=average_length, linestyle='--', color=color, label=f'Avg Episode Length ({self.names[i]}): {average_length:.2f}')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Episode Length')
        ax.set_title(f'Episode Length over Time for {self.env_size}x{self.env_size} grid (smoothed with window size {window_size})')
        ax.legend()
        ax.grid(True)
        self.save_plot_to_parent_folder(f'compare_episode_lengths_{self.env_size}x{self.env_size}.png', show)

    def plot_success_failure_bars(self, show=False):
        fig, ax = plt.subplots(figsize=(10, 8))
        for i, agent in enumerate(self.rl_agents):
            num_of_episodes = agent.num_of_episodes
            success_count = sum(agent.success)
            failure_count = num_of_episodes - success_count
            success_percentage = (success_count / num_of_episodes) * 100
            failure_percentage = (failure_count / num_of_episodes) * 100
            bars = ax.bar([f'Success ({self.names[i]})', f'Failure ({self.names[i]})'], 
                        [success_percentage, failure_percentage], label=self.names[i])
            ax.bar_label(bars, labels=[f"{success_count} / {num_of_episodes} ({success_percentage:.2f}%)", 
                                    f"{failure_count} / {num_of_episodes} ({failure_percentage:.2f}%)"])
        ax.set_xlabel('Result')
        ax.set_ylabel('Percentage')
        ax.set_title(f'Success and Failure Rate during Training for {self.env_size}x{self.env_size} grid')
        ax.legend()
        self.save_plot_to_parent_folder(f'compare_success_failure_rate_{self.env_size}x{self.env_size}.png', show)

    def plot_success_rate_over_time(self, show=False):
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, agent in enumerate(self.rl_agents):
            success_rate = np.cumsum(agent.success) / np.arange(1, len(agent.success) + 1) * 100
            ax.plot(success_rate, label=f'Success Rate ({self.names[i]})', linewidth=2)
        ax.set_xlabel('Episodes')
        ax.set_ylabel('Success Rate (%)')
        ax.set_title(f'Success Rate Over Time for {self.env_size}x{self.env_size} grid')
        ax.legend()
        ax.grid(True)
        self.save_plot_to_parent_folder(f'compare_success_rate_over_time_{self.env_size}x{self.env_size}.png', show)

    def plot_episodic_reward_over_time(self, window_size=50, show=False):
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, agent in enumerate(self.rl_agents):
            smoothed_reward = self._moving_average(agent.episodes_reward, window_size)
            line = ax.plot(smoothed_reward, label=f'Episodic Reward ({self.names[i]})', linewidth=2)
            color = line[0].get_color()
            ax.axhline(y=np.mean(agent.episodes_reward), linestyle='--', color=color, label=f'Avg Reward ({self.names[i]}): {np.mean(agent.episodes_reward):.2f}')
        ax.set_xlabel('Episodes')
        ax.set_ylabel('Reward')
        ax.set_title(f'Episodic Reward Over Time for {self.env_size}x{self.env_size} grid (smoothed with window size {window_size})')
        ax.legend()
        ax.grid(True)
        self.save_plot_to_parent_folder(f'compare_episodic_reward_{self.env_size}x{self.env_size}.png', show)