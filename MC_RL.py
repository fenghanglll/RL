import time
import random
from MC_bird import YuanYangEnv
import pygame
import numpy as np
import matplotlib.pyplot as plt

class MC_RL:
    def __init__(self,yuanyang):
        #action——value初始化
        self.qvalue = np.zeros((len(yuanyang.states),len(yuanyang.actions)))*0.1
        #次数初始化
        self.n = 0.001*np.ones((len(yuanyang.states),len(yuanyang.actions)))
        self.actions = yuanyang.actions
        self.yuanyang = yuanyang
        self.gamma = yuanyang.gamma

    def greedy_policy(self,qfun,state):
        max_values = qfun[state, :].max()
        max_indices = np.where(qfun[state, :] == max_values)[0]
        if len(max_indices) > 1:
            amax = random.choice(max_indices)
        else:
            amax = max_indices[0]
        return self.actions[amax]

    def epsilon_greedy_policy(self,qfun,state,epsilon):
        #随机地找q值max的动作索引
        max_values = qfun[state, :].max()
        max_indices = np.where(qfun[state, :] == max_values)[0]
        if len(max_indices) > 1:
            amax = random.choice(max_indices)
        else:
            amax = max_indices[0]

        if np.random.uniform()<1-epsilon:
            return self.actions[amax]
        else:
            return self.actions[int(random.random()*len(self.actions))]

    def find_anum(self,a):
        for i in range(len(self.actions)):
            if a == self.actions[i]:
                return i

    def mc_learning_ei(self,num_iter):
        self.qvalue = np.zeros((len(self.yuanyang.states),len(self.yuanyang.actions)))
        self.n = np.zeros((len(self.yuanyang.states), len(self.yuanyang.actions)))
        #self.n  = 0.001*np.ones((len(self.yuanyang.states),len(self.yuanyang.actions)))
        for iter1 in range(num_iter):
            s_sample = []
            a_sample = []
            r_sample = []
            s = self.yuanyang.reset()
            a = self.actions[int(random.random()*len(self.actions))]
            done  = False
            step_num = 0

            if self.mc_test() ==1:
                print("探索初始化第1次完成任务需要的次数",iter1)
                break

            while False == done and step_num <50:
                s_next ,r,done =self.yuanyang.transform(s,a)
                a_num = self.find_anum(a)
                #往回走给予惩罚
                #if s_next in s_sample:
                #    r = -2
            #存储数据，采样数据
                s_sample.append(s)
                r_sample.append(r)
                a_sample.append(a_num)
                step_num+=1
            #转移到下一个状态
                s=s_next
                a = self.greedy_policy(self.qvalue,s)

            g = self.qvalue[s,self.find_anum(a)]
            for i in range(len(s_sample)-1,-1,-1):
                g*=self.gamma
                g+=r_sample[i]
            for i in range(len(s_sample)):
                self.n[s_sample[i],a_sample[i]] +=1.0
                self.qvalue[s_sample[i],a_sample[i]] = (self.qvalue[s_sample[i],a_sample[i]]*(self.n[s_sample[i],a_sample[i]]-1)+g)/self.n[s_sample[i],a_sample[i]]
                g-= r_sample[i]
                g/=self.gamma
        return self.qvalue

    def mc_test(self):
        s=0
        s_sample = []
        done=False
        flag = 0
        step_num = 0
        while False == done and step_num< 50:
            a = self.greedy_policy(self.qvalue,s)
            s_next,r,done = self.yuanyang.transform(s,a)
            s_sample.append(s)
            s = s_next
            step_num +=1
        if s==9:
            flag =1
        return flag

    def mc_learning_on_policy(self,num_iter,epsilon):
        self.qvalue = np.zeros((len(yuanyang.states),len(yuanyang.actions)))
        self.n = np.zeros((len(yuanyang.states),len(yuanyang.actions)))
        for iter1 in range(num_iter):
            s_sample = []
            a_sample = []
            r_sample = []
            s=0
            done= False
            step_num = 0
            epsilon = epsilon*np.exp(-iter1/1000) #epsilon逐渐变小,epsilon-贪婪策略逐渐随机性降低。

            while False ==done and step_num <50:
                a =self.epsilon_greedy_policy(self.qvalue,s,epsilon)
                s_next,r,done = self.yuanyang.transform(s,a)
                a_num = self.find_anum(a)
                # 往回走给予惩罚
                # if s_next in s_sample:
                #    r = -2
                s_sample.append(s)
                a_sample.append(a_num)
                r_sample.append(r)
                step_num +=1
                s = s_next

            if s==9:
                print("on_policy学习第1次完成任务所需次数：",iter1)
                break

            #a = self.epsilon_greedy_policy(self.qvalue,s,epsilon)
            g = self.qvalue[s, self.find_anum(a)]
            for i in range(len(s_sample) - 1, -1, -1):
                g *= self.gamma
                g += r_sample[i]
            for i in range(len(s_sample)):
                self.n[s_sample[i], a_sample[i]] += 1.0
                self.qvalue[s_sample[i], a_sample[i]] = (self.qvalue[s_sample[i], a_sample[i]] * (self.n[s_sample[i], a_sample[i]] - 1) + g) / self.n[s_sample[i], a_sample[i]]
                g -= r_sample[i]
                g /= self.gamma
        return self.qvalue

if  __name__ =="__main__":
    yuanyang = YuanYangEnv()
    brain = MC_RL(yuanyang)
    #qvalue1 = brain.mc_learning_ei(num_iter=10000)
    #print(qvalue1)
    #yuanyang.action_value = qvalue1

    qvalue2 = brain.mc_learning_on_policy(num_iter=10000,epsilon=0.2)
    print(qvalue2)
    yuanyang.action_value = qvalue2

    #测试学习的策略
    flag = 1
    s=0
    step_num = 0
    path=[]
    while flag:
        path.append(s)
        yuanyang.path =path    #这块传输回yuanyang.path是为了渲染出图像
        a = brain.greedy_policy(qvalue2,s)   #qvalue1与qvalue2可以切换去分别验证两种 MonteCarlo学习方法
        print("%d->%s\t"%(s,a),qvalue2[s,0],qvalue2[s,1],qvalue2[s,2],qvalue2[s,3])
        yuanyang.bird_male_position = yuanyang.state_to_position(s)
        yuanyang.render()
        time.sleep(0.25)
        step_num +=1
        s_,r,t = yuanyang.transform(s,a)
        if t==True or step_num>50:
            flag = 0
        s=s_
    while True:
        time.sleep(3)
        quit()
