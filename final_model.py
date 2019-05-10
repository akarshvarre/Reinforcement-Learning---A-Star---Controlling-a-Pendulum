#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 17:00:11 2019

@author: akarsh
"""


import gym


import torch

import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

import data_gen
import numpy as np



class TwoLayerNet(torch.nn.Module):
    def __init__(self):
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(4, 64)
        self.linear2 = torch.nn.Linear(64, 3)

    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        h_relu = F.sigmoid(h_relu)
        y_pred = self.linear2(h_relu)
        return y_pred



env = gym.make('Pendulum-v0')

x,y = data_gen.train_data(env)
x = torch.from_numpy(x).float()
y = torch.from_numpy(y).float()



model = TwoLayerNet()


criterion = torch.nn.MSELoss(reduction='mean')
optimizer = optim.Adam(model.parameters(), lr=1e-2)
loss_plot = []
for t in range(500):
  
    y_pred = model(x)

    # Compute and print loss
    loss = criterion(y_pred, y)
    loss_plot.append(loss.item())
    print(t, loss.item())
    
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
plt.plot(range(500), loss_plot)
plt.show()
torch.save(model.state_dict(), "./transitionmodel.pt")




    