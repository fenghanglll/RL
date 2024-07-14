#!/usr/bin/python3
# coding = UTF-8
# this is program for
# @python version 3.7.9
# @code by va1id


import pygame
import os.path as osp

path_file = osp.abspath(__file__)
path_images = osp.join(osp.dirname(path_file), 'images')
print(path_images)

def load_bird_male():
    obj = 'bird_male.png'
    obj_path = osp.join(path_images, obj)
    return pygame.image.load(obj_path)


def load_bird_female():
    obj = 'bird_female.png'
    obj_path = osp.join(path_images, obj)
    return pygame.image.load(obj_path)


def load_background():
    obj = 'background.jpg'
    obj_path = osp.join(path_images, obj)
    return pygame.image.load(obj_path)


def load_obstacle():
    obj = 'obstacle.png'
    obj_path = osp.join(path_images, obj)
    return pygame.image.load(obj_path)