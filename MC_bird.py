import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pygame
import load
import random


class YuanYangEnv(object):
    def __init__(self):
        # 状态空间
        self.states = [x for x in range(100)]

        # 动作空间
        self.actions = ['e', 's', 'w', 'n']
        # 每个状态处所对应的值函数
        self.value = np.zeros((10, 10))
        self.action_value = np.zeros((100,4))
        self.gamma = 0.95  # 此处gamma调大一些
        # 渲染窗口
        self.viewer = None
        self.FPSLOCK = pygame.time.Clock()
        # 屏幕大小
        self.screen_size = (1200, 900)
        self.bird_position = (0, 0)
        self.limit_distance_x = 120
        self.limit_distance_y = 90
        # 障碍物的大小
        self.obstacle_size = [120, 90]
        # 障碍物1
        self.obstacle1_x = []
        self.obstacle1_y = []
        # 障碍物2
        self.obstacle2_x = []
        self.obstacle2_y = []
        # 障碍物3
        self.obstacle3_x = []
        self.obstacle3_y = []

        # 障碍物
        for i in range(8):
            # obstical 1
            self.obstacle1_x.append(360)
            if i <= 3:
                self.obstacle1_y.append(90*i)
            else:
                self.obstacle1_y.append(90 * (i+2))
            # obstical 2
            self.obstacle2_x.append(720)
            if i <= 4:
                self.obstacle2_y.append(90 * i)
            else:
                self.obstacle2_y.append(90 * (i + 2))
            # obstical 3
            self.obstacle3_x.append(120)
            if i <= 6:
                self.obstacle3_y.append(90 * i)
            else:
                self.obstacle3_y.append(90 * (i + 2))


        self.bird_male_init_position = (0.0, 0.0)
        self.bird_male_position = [0, 0]
        self.bird_female_init_position = [1080, 0]

        self.path = []

    # 游戏渲染
    def render(self):
        if self.viewer is None:
            pygame.init()
            self.viewer = pygame.display.set_mode(self.screen_size, 0, 32)
            pygame.display.set_caption('yuanyang')
        # 下载图片
            self.bird_male = load.load_bird_male()
            self.bird_female = load.load_bird_female()
            self.background = load.load_background()
            self.obstacle = load.load_obstacle()
            self.font = pygame.font.SysFont('times', 15)

        self.viewer.blit(self.background, (0, 0))
        #self.viewer.blit(self.background, (0, 0))
        # 画直线
        for i in range(11):
            pygame.draw.lines(self.viewer, (255, 255, 255), True, ((120 * i, 0), (120 * i, 900)), 1)
            pygame.draw.lines(self.viewer, (255, 255, 255), True, ((0, 90 * i), (1200 , 90 * i)), 1)
        self.viewer.blit(self.bird_female, self.bird_female_init_position)
        for i in range(8):
            self.viewer.blit(self.obstacle, (self.obstacle1_x[i], self.obstacle1_y[i]))
            self.viewer.blit(self.obstacle, (self.obstacle2_x[i], self.obstacle2_y[i]))
            self.viewer.blit(self.obstacle, (self.obstacle3_x[i], self.obstacle3_y[i]))
        self.viewer.blit(self.bird_male, self.bird_male_position)
        # 画值函数
        for i in range(100):
            y=int(i/10)
            x=i % 10
            surface = self.font.render(str(round(float(self.action_value[i, 0]), 2)), True, (0, 0, 0))
            self.viewer.blit(surface, (120 * x + 80, 90 * y + 45))
            surface = self.font.render(str(round(float(self.action_value[i, 1]), 2)), True, (0, 0, 0))
            self.viewer.blit(surface, (120 * x + 50, 90 * y + 70))
            surface = self.font.render(str(round(float(self.action_value[i, 2]), 2)), True, (0, 0, 0))
            self.viewer.blit(surface, (120 * x + 10, 90 * y + 45))
            surface = self.font.render(str(round(float(self.action_value[i, 3]), 2)), True, (0, 0, 0))
            self.viewer.blit(surface, (120 * x + 50, 90 * y + 10))


        for i in range(len(self.path)):
            rec_position = self.state_to_position(self.path[i])
            pygame.draw.rect(self.viewer, [255, 0, 0],
                             [rec_position[0], rec_position[1], 120, 90], 3)
            surface = self.font.render(str(i), True, (255, 0, 0))
            self.viewer.blit(surface, (rec_position[0] + 5, rec_position[1] + 5))

        pygame.display.update()
        self.FPSLOCK.tick(30)
    # 状态转移函数
    def transform(self, state, action):
        current_position = self.state_to_position(state)
        next_position = [0, 0]
        flag_collide = 0
        flag_find = 0
        # 判断碰撞
        flag_collide = self.collide(current_position)
        # 判断终点
        flag_find = self.find(current_position)
        # if flag_find == 1 or flag_collide == 1:
        #     return state, 0, True
        if flag_find == 1:
            return state, 10, True
        if flag_collide == 1:
            return state, -10, True

        if action == 'e':
            next_position[0] = current_position[0] + 120
            next_position[1] = current_position[1]

        if action == 's':
            next_position[0] = current_position[0]
            next_position[1] = current_position[1] + 90

        if action == 'w':
            next_position[0] = current_position[0] - 120
            next_position[1] = current_position[1]

        if action == 'n':
            next_position[0] = current_position[0]
            next_position[1] = current_position[1] - 90
        flag_collide = self.collide(next_position)
        flag_find = self.find(next_position)
        if flag_collide == 1:
            return self.position_to_state(current_position), -10, True

        if flag_find == 1:
            return self.position_to_state(next_position), 10, True

        return self.position_to_state(next_position), -2, False

    # 状态转为像素坐标
    def state_to_position(self, state):
        i = int(state / 10)
        j = state % 10
        position = [0, 0]
        position[0] = 120 * j
        position[1] = 90 * i
        return position

    # 像素坐标转为状态
    def position_to_state(self, position):
        i = position[0] / 120
        j = position[1] / 90
        return int(i + 10 * j)

    # 雄鸟碰撞检测函数
    def collide(self, state_position):
        flag = 1
        flag1 = 1
        flag2 = 1
        flag3 = 1
        # 判断第三个 障碍物
        third_dx = []
        third_dy = []
        for i in range(8):
            dx1 = abs(self.obstacle3_x[i] - state_position[0])
            third_dx.append(dx1)
            dy1 = abs(self.obstacle3_y[i] - state_position[1])
            third_dy.append(dy1)
        mindx = min(third_dx)
        mindy = min(third_dy)
        if mindx >= self.limit_distance_x or mindy >= self.limit_distance_y:   #TODO 感觉这块以及后面应该是and
            flag3 = 0
        # 判断第一个 障碍物
        dx = []
        dy = []
        for i in range(8):
            dx1 = abs(self.obstacle1_x[i] - state_position[0])
            dx.append(dx1)
            dy1 = abs(self.obstacle1_y[i] - state_position[1])
            dy.append(dy1)
        mindx = min(dx)
        mindy = min(dy)
        if mindx >= self.limit_distance_x or mindy >= self.limit_distance_y:
            flag1 = 0

        # 判断第二个障碍
        second_dx = []
        second_dy = []
        for i in range(8):
            dx2 = abs(self.obstacle2_x[i] - state_position[0])
            second_dx.append(dx2)
            dy2 = abs(self.obstacle2_y[i] - state_position[1])
            second_dy.append(dy2)
        mindx = min(second_dx)
        mindy = min(second_dy)
        if mindx >= self.limit_distance_x or mindy >= self.limit_distance_y:
            flag2 = 0

        if flag1 == 0 and flag2 == 0 and flag3 == 0:
            flag = 0
        if state_position[0] > 1080 or\
                state_position[0] < 0 or\
                state_position[1] < 0 or state_position[1] > 810:
            flag = 1
        return flag

    # 雄鸟找到雌鸟
    def find(self, state_position):
        flag = 0
        if abs(state_position[0] - self.bird_female_init_position[0]) < self.limit_distance_x and abs(state_position[1] - self.bird_female_init_position[1]) < self.limit_distance_y:
            flag = 1
        return flag

    def reset(self):
        # 随机产生随机状态
        flag1 = 1
        flag2 = 1
        # 这里是找 flag1 与 flage2都为0的state？ 这样的话就是 不相撞的且是没有找到雌的这么一种状态
        while flag1 or flag2 == 1:
            state = self.states[int(random.random() * len(self.states))]
            state_position = self.state_to_position(state)
            flag1 = self.collide(state_position)
            flag2 = self.find(state_position)
        return state

if __name__ == '__main__':
    yy = YuanYangEnv()
    yy.render()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()