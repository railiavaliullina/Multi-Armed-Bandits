import gym
import numpy as np

from utils.log_utils import Logger
from utils.visualization import plot_hist


class EGreedy(object):
    def __init__(self, cfg, env_type, seed, eps, init_type):
        self.cfg = cfg
        self.env_type = env_type
        self.seed = seed
        self.eps = eps
        self.init_type = init_type
        self.done = False
        self.cumulative_reward, self.average_reward, self.cumulative_accuracy = 0, 0, 0

        if self.init_type == 'optimistic':
            self.q = np.ones(self.cfg.arms_num) * self.cfg.optimistic_init_value
        else:
            self.q = np.zeros(self.cfg.arms_num)
        self.n = np.zeros(self.cfg.arms_num)

        self.env = gym.make(f'bandits:{self.env_type}', seed=self.seed)
        self.best_arm = self.env.get_best_arm()
        self.env.reset()

        self.experiment_name = f'e_greedy_{env_type}_eps_{eps}_init_{init_type}_seed_{seed}'
        self.logger = Logger(self.cfg, experiment_name=self.experiment_name)

    def pul_bandit_arm(self):
        choose_current_best_arm = np.random.choice([True, False], size=1, p=(1 - self.eps, self.eps))[0]
        arm = np.argmax(self.q) if choose_current_best_arm else np.random.randint(0, self.cfg.arms_num, size=1)[0]
        return arm, choose_current_best_arm

    def make_step(self, arm):
        balance, reward, done, _ = self.env.step(arm)
        self.n[arm] += 1
        self.q[arm] += (reward - self.q[arm]) / self.n[arm]
        self.done = done
        return balance, reward

    def update_metrics(self, arm, reward, step):
        self.cumulative_accuracy = self.cumulative_accuracy + 1 if arm == self.best_arm \
            else self.cumulative_accuracy
        self.cumulative_reward += reward
        cumulative_accuracy_out = (self.cumulative_accuracy / step) * 100 if step > 0 else 0
        cumulative_reward_out = (self.cumulative_reward / step) * 100 if step > 0 else 0
        return cumulative_accuracy_out, cumulative_reward_out

    def log_metrics(self, arm, cumulative_reward_out, reward, cumulative_accuracy_out, balance, step):
        self.logger.log_metrics(names=['reward/cumulative_average_reward', 'reward/cur_reward',
                                       'accuracy/cumulative_accuracy', 'balance/balance',
                                       'arms/chosen_arm', 'arms/best_arm',
                                       f'arms/cumulative_average_reward_arm_{arm}',
                                       f'arms/cur_reward_arm_{arm}', f'arms/cumulative_accuracy_arm_{arm}'],
                                metrics=[cumulative_reward_out, reward, cumulative_accuracy_out, balance,
                                         arm, self.best_arm, cumulative_reward_out, reward,
                                         cumulative_accuracy_out], step=step)

        for arm_ in range(self.cfg.arms_num):
            self.logger.log_metrics(names=[f'arms/q_{arm_}', f'arms/n_{arm_}'],
                                    metrics=[self.q[arm], self.n[arm]],
                                    step=step)

    def run(self):
        print(f'Starting exp: {self.experiment_name}')
        step = 0
        chosen_arms = []

        while step < self.cfg.steps_num and not self.done:
            arm, choose_current_best_arm = self.pul_bandit_arm()
            balance, reward = self.make_step(arm)
            cumulative_accuracy_out, cumulative_reward_out = self.update_metrics(arm, reward, step)

            if step % 100 == 0:
                print(f'step: {step}')
                print(f'is_best_strategy: {choose_current_best_arm}, chosen_arm: {arm}, '
                      f'reward: {reward}, balance: {balance}, '
                      f'cumulative_reward: {self.cumulative_reward}, '
                      f'cumulative_accuracy: {cumulative_accuracy_out}\n')

            chosen_arms.append(arm)
            step += 1

        plot_hist(self.cfg, chosen_arms, f'{self.experiment_name}', f'chosen_arms_hist\nexp: {self.experiment_name}')
