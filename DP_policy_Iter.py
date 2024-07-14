#!/usr/bin/python3
# coding = UTF-8
# this is program for DP_policy_iter
# @python version 3.7.9
# @code by va1id


import time
import random
from bird import YuanYangEnv
# from yuanyang import YuanYangEnv


class DP_Policy_Iter(object):
    def __init__(self, yuanyang:YuanYangEnv):
        self.states = yuanyang.states
        self.actions = yuanyang.actions
        # 值函数
        self.v = [0.0 for i in range(len(self.states))]   #TODO 有点问题感觉？
        self.pi = dict()
        self.yuanyang = yuanyang
        self.gamma = yuanyang.gamma
        # 初始化策略
        for state in self.states:
            #flag1 = 0  #TOdo 这块两个flag没用
            #flag2 = 0
            # 看有没有发生碰撞，若发生碰撞返回1
            #flag1 = yuanyang.collide(yuanyang.state_to_position(state))
            #flag2 = yuanyang.find(yuanyang.state_to_position(state))
            self.pi[state] = self.actions[int(random.random() * len(self.actions))]

    def policy_evaluate(self):
        for i in range(100):
        # for i in range(1):
            delta = 0.0
            for state in self.states:
                flag1 = 0
                flag2 = 0
                flag1 = self.yuanyang.collide(self.yuanyang.state_to_position(state))
                flag2 = self.yuanyang.find(self.yuanyang.state_to_position(state))
                if flag1 == 1 or flag2 == 1:
                    continue
                # 这里取出来的是此状态下要进行的行为
                action = self.pi[state]
                s, r, t = self.yuanyang.transform(state, action)
                # 更新值
                # try:
                new_v = r + self.gamma * self.v[s]
                # except Exception as e:
                #     print(e)
                #     print('list out of range in policy :', i)
                delta += abs(self.v[state] - new_v)
                self.v[state] = new_v
            if delta < 1e-6:
                print('策略评估迭代次数：', i)
                break

    def poliy_improve(self):
        for state in self.states:
            flag1 = 0
            flag2 = 0
            flag1 = self.yuanyang.collide(self.yuanyang.state_to_position(state))
            flag2 = self.yuanyang.find(self.yuanyang.state_to_position(state))

            if flag1 == 1 or flag2 == 1:
                continue

            a1 = self.actions[0]
            s, r, t = self.yuanyang.transform(state, a1)
            v1 = r + self.gamma * self.v[s]
            for action in self.actions:
                s, r, t = self.yuanyang.transform(state, action)
                if v1 < r + self.gamma * self.v[s]:
                    a1 = action
                    v1 = r + self.gamma * self.v[s]
            self.pi[state] = a1

    def policy_iterate(self):
        for i in range(100):
            # 策略评估，变的是v
            self.policy_evaluate()

            pi_old = self.pi.copy()
            # 变的pi
            self.poliy_improve()
            if self.pi == pi_old:
                print('策略改善次数：', i)
                break

if __name__ == '__main__':
    yuanyang = YuanYangEnv()
    policy_value = DP_Policy_Iter(yuanyang)
    policy_value.policy_iterate()
    flag = 1
    s = 0
    path = []
    for state in range(100):
        i = int(state / 10)
        j = state % 10
        yuanyang.value[j, i] = policy_value.v[state]
    step_num = 0

    while flag:
        path.append(s)
        yuanyang.path = path
        a = policy_value.pi[s]
        print('%d->%s\t'%(s, a))
        yuanyang.bird_male_position = yuanyang.state_to_position(s)
        yuanyang.render()
        time.sleep(0.5)
        step_num += 1
        s_, r, t = yuanyang.transform(s, a)
        if t == True or step_num > 200:
            flag = 0
        s = s_
    yuanyang.bird_male_position = yuanyang.state_to_position(s)
    path.append(s)
    yuanyang.render()
    time.sleep(3)
    exit()