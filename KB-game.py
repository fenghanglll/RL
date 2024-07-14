#!/usr/bin/python3
# coding = UTF-8
# this is program for K摇臂赌博机
# @python version 3.7.9
# @code by va1id
import numpy as np
import matplotlib.pyplot as plt
class KB_Game(object):
    def __init__(self, *args, **kwargs):
        # 每个摇臂的平均收益
        self.q = np.array([0.0, 0.0, 0.0,0.0])
        # 每个摇臂摇动次数
        self.action_counts = np.array([0, 0, 0, 0])
        # 当前平均的累积回报
        self.current_cumulative_rewards = 0.0
        # 动作空间
        self.action = [1, 2, 3, 4]
        # 当前执行的次数
        self.counts = 0
        # 玩家玩的次数历史
        self.counts_history = []
        # 累积回报的历史
        self.cumulative_rewards_history = []
        # 当前的行为
        self.a = 2
        # 当前的收益
        self.reward = 0
 # 输入动作输出回报，就是模拟摇臂机给出奖励,动作1时输出回报均值为1，标准差为1的正态分布。 啊=2
    def step(self, a):
        r = 0
        if a == 1:
            r = np.random.normal(1.5, 1)
        if a == 2:
            r = np.random.normal(2, 2)
        if a == 3:
            r = np.random.normal(3, 10)
        if a == 4:
            r = np.random.normal(-1, 10)
        return r
    # 策略选择， 这里面有三个算法，1-e-greedy，2-ucb, 3-boltzman
    # 当选择e-greedy的时候字典当中输入的e
    def choose_action(self, policy, **kwargs):
        action = 0
        if policy == 'e':
            if np.random.random() < kwargs['epsilon']:
                action = np.random.randint(1, 5)
            else:
                # 如果是不随机的话，则选择每个摇臂平均收益最大的，找到索引，+1 就是这个action
                action = np.argmax(self.q) + 1

        if policy == 'ucb':
            # 这个参数是ucb策略公式里面的 c
            c_ratio = kwargs['c_ratio']
            if 0 in self.action_counts:
                # np.where返回一个元组，(array([1, 2], dtype=int64),)，里面传入的参数必须是ndarray
                # 如果里面还有没有执行过的摇臂，就选第一个没有摇过的摇臂
                action = np.where(self.action_counts==0)[0][0] + 1
            else:
                # 这个就是ucb的策略选择公式
                value = self.q + c_ratio * np.sqrt(np.log(self.counts) / self.action_counts)
                action = np.argmax(value) + 1

        if policy == 'boltzman':
            tau = kwargs['temperature']
            p = np.exp(self.q / tau) / (np.sum(np.exp(self.q / tau)))
            # random.choice  从一维数组中产生随机样本一个，a是要产生随机样本的列表，参数p为一维数组是描述与a中每一项相关联的概率，如果没有看作是随机分布
            action = np.random.choice(self.action, p=p.ravel())
        return action

    def trian(self, play_total, policy, **kwargs):
        reward_1 = []
        reward_2 = []
        reward_3 = []
        reward_4 = []
        # 进行训练
        for i in range(play_total):

            if policy == 'e':
                action = self.choose_action(policy, epsilon=kwargs['epsilon'])
            if policy == 'ucb':
                action = self.choose_action(policy, c_ratio=kwargs['c_ratio'])
            if policy == 'boltzman':
                action = self.choose_action(policy, temperature=kwargs['temperature'])
            # 实例拿到要执行的行为
            self.a = action
            # 与环境进行交互，通过step函数也就是在环境当中得到在该行为的奖励值 r
            self.reward = self.step(self.a)
            # 更新执行的次数
            self.counts += 1
            # 更新值函数，
            self.q[self.a - 1] = (self.q[self.a - 1] * self.action_counts[self.a - 1] + self.reward) / (
                        self.action_counts[self.a - 1] + 1)
            # 更新行为执行的次数
            self.action_counts[self.a - 1] += 1
            reward_1.append([self.q[0]])
            reward_2.append([self.q[1]])
            reward_3.append([self.q[2]])
            reward_4.append([self.q[3]])
            # 更新当前的收益
            self.current_cumulative_rewards += self.reward
            self.cumulative_rewards_history.append(self.current_cumulative_rewards)
            self.counts_history.append(i)

    def reset(self):
        self.q = np.array([0.0, 0.0, 0.0,0.0])
        self.action_counts = np.array([0, 0, 0,0])
        self.current_cumulative_rewards = 0.0
        self.counts = 0
        self.counts_history = []
        self.cumulative_rewards_history = []
        self.a = 2
        self.reward = 0

    def plot(self, colors, policy):
        plt.figure(1)
        plt.plot(self.counts_history, self.cumulative_rewards_history, colors, label=policy)
        plt.legend()
        plt.xlabel('n', fontsize=18)
        plt.ylabel('reward_history', fontsize=18)

if __name__ == '__main__':
    np.random.seed(0)
    k_gamble = KB_Game()
    total = 2000

    k_gamble.trian(total, policy='e', epsilon=0.05)
    # k_gamble.plot(colors='r', policy='e', style='-.')
    k_gamble.plot(colors='r', policy='e')
    print(k_gamble.action_counts)

    k_gamble.reset()
    k_gamble.trian(total, policy='ucb', c_ratio=0.5)
    # k_gamble.plot(colors='g', policy='ucb', style='-')
    k_gamble.plot(colors='g', policy='ucb')
    print(k_gamble.action_counts)

    k_gamble.reset()
    k_gamble.trian(total, policy='boltzman', temperature=1)
    # k_gamble.plot(colors='b', policy='boltzman', style='--')
    k_gamble.plot(colors='b', policy='boltzman')
    print(k_gamble.action_counts)

    plt.show()
