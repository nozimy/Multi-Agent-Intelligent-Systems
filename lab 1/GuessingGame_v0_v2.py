# -*- coding: utf-8 -*-

# importing dependency libraries
from __future__ import print_function
import gym
import numpy as np
import time
import os


#Epsilon-Greedy approach for Exploration and Exploitation of the state-action spaces
def epsilon_greedy(q_table,state,env):
    epsilon = 0.3
    p = np.random.uniform(low=0,high=1)
    #print(p)
    if p > epsilon:
        sample = np.argmax(q_table[state,:])#say here,initial policy = for each state consider the action having highest else:
        return sample
    return (env.action_space.sample() + abs(env.action_space.low)).astype(np.int64)[0]

def loadEnv():
    #Load the environment
    env = gym.make('GuessingGame-v0')
    s = env.reset()
    print("initial state : ",s)
    print()
    #env.render()
    print()
    print(env.action_space) #number of actions
    print(env.observation_space) #number of states
    print()
    #print("Number of actions : ",env.action_space.n)
    print(env.action_space.shape[0])
    print(env.action_space.high)
    print(env.action_space.low)
    print("Number of states : ",env.observation_space.n)
    print()
    return env


def runQLearning(env):
    #set hyperparameters
    lr = 0.5 #learning rate
    y = 0.9 #discount factor lambda
    eps = 100000
    timeSteps = 200
    
    actionNum = abs(env.action_space.high) + abs(env.action_space.low)
    actionNum = actionNum.astype(np.int64)[0]
    
    #Initializing Q-table with zeros
    q_table = np.zeros([env.observation_space.n,actionNum], np.float32)
#    action = epsilon_greedy(q_table, 0, env)
#    print(action)
#    return env, q_table
    
    for epsIndex in range(eps):
        state = env.reset()
        for stepIndex in range(timeSteps):
            action = epsilon_greedy(q_table, state, env)
#            print(action)
            new_state,reward,done,_ = env.step(action + env.action_space.low)
            if (reward==0):
                if done==True:
                    r = -5 #to give negative rewards when holes turn up
                    q_table[new_state] = np.ones(actionNum)*reward #in terminal state Q value equals the reward
                else:
                    r = -1 #to give negative rewards to avoid long routes
            if (reward==1):
                reward = 100
                q_table[new_state] = np.ones(actionNum)*reward #in terminal state Q value equals the reward
            q_table[state,action] = q_table[state,action] + lr * (r + y*np.max(q_table[new_state,action]) - q_table[state,action])
            state = new_state
            if (done == True):
#                print("Episode finished after {} timesteps".format(t+1))
                break
    return env, q_table

def play(env, q_table):
    print("Q-table")
    print(q_table)
    print()
    print("Output after learning")
    print()
    #let's check how much our agent has learned
    state = env.reset()
    print("initial state : ",state)
    action = getAction(2,env)
    while(True):
        actionIndex = np.argmax(q_table[state])
        action = getAction(actionIndex, env, action)
        new_state,r,done,_ = env.step(action)
        print("===============")
        print("state : ",new_state)
        state = new_state
        if(done==True) :
            break

if __name__ == "__main__":
    env = loadEnv()
    env, Q = runQLearning(env)
    play(env, Q)

    