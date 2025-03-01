import matplotlib.pyplot as plt
import os
import numpy as np

class Plotter:
    def __init__(self, rl_agent, name):
        self.rl_agent = rl_agent
        self.name = name

    def plot_success_failure_rate(self, show=False):
        success_percentage = (self.rl_agent.success_count / self.rl_agent.num_of_episodes) * 100
        failure_percentage = (self.rl_agent.failure_count / self.rl_agent.num_of_episodes) * 100
        bars = plt.bar(['Success', 'Failed'], [success_percentage, failure_percentage])
        plt.xlabel('Result')
        plt.ylabel('Percentage')
        plt.suptitle(f'Success and Failed Rate during Training ({self.name})', fontsize=16)
        plt.title(self.rl_agent.info, fontsize=10)
        plt.bar_label(bars, labels=[f"{self.rl_agent.success_count} / {self.rl_agent.num_of_episodes} ({success_percentage:.2f}%)", f"{self.rl_agent.failure_count} / {self.rl_agent.num_of_episodes} ({failure_percentage:.2f}%)"])
        plt.savefig(os.path.join(self.rl_agent.plots_dir, 'success_failure_rate.png'))
        if show:
            plt.show()
        plt.close()

    def plot_episode_lengths(self, show=False):
        average_length = np.mean(self.rl_agent.episodes_length)
        # plt.figure(figsize=(15, 5))
        # plt.plot(self.rl_agent.episodes_length)
        plt.scatter(range(len(self.rl_agent.episodes_length)), self.rl_agent.episodes_length, s=10)
        plt.xlabel('Episode')
        plt.ylabel('Episode Length')
        plt.suptitle(f'Episode Length over Time ({self.name})', fontsize=16)
        plt.title(self.rl_agent.info, fontsize=10)
        plt.axhline(y=average_length, color='r', linestyle='--', label=f'Average Episode Length: {average_length:.2f}')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.rl_agent.plots_dir, 'episode_lengths.png'))
        if show:
            plt.show()
        plt.close()
    
    def plot_cummulative_average_rewards(self, show=False):
        # average reward obtained by the agent up to the current episode
        cummulative_rewards = np.cumsum(self.rl_agent.episodes_reward)
        cummulative_average_rewards = cummulative_rewards / np.arange(1, len(cummulative_rewards) + 1)
        plt.plot(cummulative_average_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.suptitle(f'Average Reward over Time ({self.name})', fontsize=16)
        plt.title(self.rl_agent.info, fontsize=10)
        plt.grid(True)
        plt.savefig(os.path.join(self.rl_agent.plots_dir, 'cummulative_average_rewards.png'))
        if show:
            plt.show()
        plt.close()

class ComparePlotter:
    def __init__(self, rl_agents, names):
        self.rl_agents = rl_agents
        self.names = names

    def plot_success_failure_rate(self, show=False):
        success_percentages = [(agent.success_count / agent.num_of_episodes) * 100 for agent in self.rl_agents]
        failure_percentages = [(agent.failure_count / agent.num_of_episodes) * 100 for agent in self.rl_agents]
        
        x = np.arange(len(self.rl_agents))
        width = 0.35

        fig, ax = plt.subplots()
        bars1 = ax.bar(x - width/2, success_percentages, width, label='Success')
        bars2 = ax.bar(x + width/2, failure_percentages, width, label='Failed')

        ax.set_xlabel('Agents')
        ax.set_ylabel('Percentage')
        ax.set_title('Success and Failure Rate during Training')
        ax.set_xticks(x)
        ax.set_xticklabels(self.names)
        ax.legend()

        for bars in [bars1, bars2]:
            ax.bar_label(bars, padding=3)

        plt.savefig(os.path.join(self.rl_agents[0].plots_dir, 'compare_success_failure_rate.png'))
        if show:
            plt.show()
        plt.close()

    def plot_episode_lengths(self, show=False):
        fig, ax = plt.subplots()
        for i, agent in enumerate(self.rl_agents):
            average_length = np.mean(agent.episodes_length)
            ax.scatter(range(len(agent.episodes_length)), agent.episodes_length, s=10, label=f'{self.names[i]} (Avg: {average_length:.2f})')
            ax.axhline(y=average_length, linestyle='--')

        ax.set_xlabel('Episode')
        ax.set_ylabel('Episode Length')
        ax.set_title('Episode Length over Time')
        ax.legend()
        ax.grid(True)

        plt.savefig(os.path.join(self.rl_agents[0].plots_dir, 'compare_episode_lengths.png'))
        if show:
            plt.show()
        plt.close()

    def plot_cummulative_average_rewards(self, show=False):
        fig, ax = plt.subplots()
        for i, agent in enumerate(self.rl_agents):
            cummulative_rewards = np.cumsum(agent.episodes_reward)
            cummulative_average_rewards = cummulative_rewards / np.arange(1, len(cummulative_rewards) + 1)
            ax.plot(cummulative_average_rewards, label=self.names[i])

        ax.set_xlabel('Episode')
        ax.set_ylabel('Average Reward')
        ax.set_title('Average Reward over Time')
        ax.legend()
        ax.grid(True)

        plt.savefig(os.path.join(self.rl_agents[0].plots_dir, 'compare_cummulative_average_rewards.png'))
        if show:
            plt.show()
        plt.close()
        