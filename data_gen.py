#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 16:52:05 2019

@author: akarsh
"""

import gym
import numpy as np



env = gym.make('Pendulum-v0')

def extended_state(state,action):
    train_state = np.array([state[0],state[1], state[2], action])
    return train_state

def create_data(env):
    state = env.reset()
    action = env.action_space.sample()
    train_state = extended_state(state,action)
    next_state = env.step(action)
    next_state = next_state[0]
    next_state = np.array([next_state[0],next_state[1],next_state[2]])
    return train_state, next_state


def train_data(env):
    inputs = np.empty([300000,4])
    labels = np.empty([300000,3])
    for i in range(300000):
        inp, label = create_data(env)
        inputs[i,:] = inp
        labels[i,:] = label
    
    return inputs, labels
            
            