#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 23:43:15 2019

@author: akarsh
"""

import numpy as np
from bisect import bisect_left
import torch

import gym

#import data_gen
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 3)
        
    def forward(self, x):
        out = torch.sigmoid(self.fc1(x))
        out = self.fc2(out)
        
        return out
env = gym.make('Pendulum-v0')

policy = Net()
policy.load_state_dict(torch.load("./transitionmodel.pt"))
policy.eval()



class Node():
    
    def __init__(self, parent,state,act):
        self.parent = parent
        self.child = state
        if self.parent is not None:
            self.action = act
            self.g = 1 + self.parent.g
            self.last_u = act
        else:
            self.action = 0
            self.g = 1
            self.last_u =0
            
        
        self.theta_temp = np.arctan2(self.child[1],self.child[0])
        self.theta = self.disc(list(theta_space),self.theta_temp)
        
        self.theta_dot_temp = self.child[2]
        self.theta_dot = self.disc(list(theta_dot_space),self.theta_dot_temp)
        
        
        self.h = self.theta**2 + self.theta_dot**2 + 0.001*self.action**2
        self.f = self.g + self.h
        self.viewer = None

    def getchildren(self):
        store = set()
        
        for action in action_space:
            go_state = np.array([np.cos(self.theta), np.sin(self.theta), self.theta_dot, action])
            in_state = torch.from_numpy(go_state).float().unsqueeze(0)
            
            out_state = (policy(in_state)).detach().numpy().reshape(3,)
            
            n = Node(self,out_state,action)
            #print(n)
            #print("before")
            flag = 1
            if self.theta == n.theta and self.theta_dot == n.theta_dot:
                flag = 0
            if flag:
                store.add(n)
            #print("after")
            
        return store

    def __lt__(self, other):
        return self.f<other.f
    def __gt__(self,other):
        return self.f>other.f
    def __eq__(self,other):
        #print("equal called")
        return (self.theta == other.theta) and (self.theta_dot == other.theta_dot)
    def __hash__(self,):
        #self.identity = [str(self.theta), str(self.theta_dot), str(self.g)]
        self.identity = [str(self.theta), str(self.theta_dot)]
        #print(("".join(self.identity)))
        return hash("".join(self.identity))
    def __str__(self):
        return "Theta: {} Theta_dot {}  g  {} h {}".format(self.theta,self.theta_dot,self.g,self.h)
    def __ne__(self, other):
        return not ((self.theta == other.theta) and (self.theta_dot == other.theta_dot))
    
    
    
    def disc(self, myList, myNumber):
        pos = bisect_left(myList, myNumber)
        if pos == 0:
            return myList[0]
        if pos == len(myList):
            return myList[-1]
        before = myList[pos - 1]
        after = myList[pos]
        if after - myNumber < myNumber - before:
           return after
        else:
           return before
      
    def render(self, mode='human'):

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500,500)
            self.viewer.set_bounds(-2.2,2.2,-2.2,2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0,0,0)
            self.viewer.add_geom(axle)
            #fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image('clockwise.png', 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.theta + np.pi/2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u/2, np.abs(self.last_u)/2)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')


def astar(start, goal):


    # Create start and end node
    start_node = start
    end_node = goal


    # Initialize both open and closed list
    open_list = []
    closed_list = []

    # Add the start node
    open_list.append(start_node)

    # Loop until you find the end

    count = 0
    while len(open_list) > 0:
        count+=1
       

        # Get the current node
        current_node = open_list[0]
        current_index = 0
        for index, item in enumerate(open_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index
        #print(current_node)
   

        # Pop current off open list, add to closed list
        open_list.pop(current_index)
        closed_list.append(current_node)

        # Found the goal
        #if current_node == end_node:
        if ((current_node.theta == end_node.theta) and (current_node.theta_dot == end_node.theta_dot)):
            path = []
            current = current_node
            
            while current is not None:
                path.append(current.parent)
                current = current.parent
            print("converged  ", count, "path len ", len(path[::-1]))
            return len(path[::-1])
            
            #return count

        # Generate children
        children = current_node.getchildren()
        
        # Loop through children
        for child in children:
            s = 1
            # Child is on the closed list
            for closed_child in closed_list:
                
                
                #if child == closed_child:
                if ((child.theta == closed_child.theta) and (child.theta_dot == closed_child.theta_dot)):
                    #print("here1")
                    s = 0

     
            if s==0:
                continue

            # Child is already in the open list
            for open_node in open_list:
           
                if ((child.theta == open_node.theta) and (child.theta_dot == open_node.theta_dot)) and child.g > open_node.g:
                    s = 0
                    
                elif ((child.theta == open_node.theta) and (child.theta_dot == open_node.theta_dot)) and child.g <= open_node.g:
                    open_list.remove(open_node)
            if s==0:
                continue
            # Add the child to the open list
            open_list.append(child)  




#path = astar(start_node, goal_node)


heap = []

start = np.array([-1, 0, 0])
start_node = Node(None, start[:3],None)


goal = np.array([1,0,0]).reshape(3,)

goal_node = Node(None, goal,None)

Z =[]
desc = [50]

acti = [10,15,20,25]
for ac in acti:
    for res in desc:
        theta_space = (np.linspace(-np.pi, np.pi, num=(res)))
        action_space = (np.linspace(-2.0, 2.0, num=ac))
        theta_dot_space = (np.linspace(-8.0, 8.0, num=(res)))
        
        steps = astar(start_node, goal_node)
        bluff = 0
        if steps is None:
            while bluff ==0:
                print("in while")
                steps = astar(start_node, goal_node)
                print(steps)
                if steps is not None:
                    bluff = 1
            
        Z.append(steps)
    


print(Z)
