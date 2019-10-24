# importing dependency libraries
from __future__ import print_function
import gym
import numpy as np
import time
import os


#Epsilon-Greedy approach for Exploration and Exploitation of the state-action spaces
def epsilon_greedy(q_table,s,env,prevAction):
    epsilon = 0.3
    p = np.random.uniform(low=0,high=1)
    #print(p)
    if p > epsilon:
#        sample = np.empty(env.action_space.shape)
        sample = np.argmax(q_table[s,:])#say here,initial policy = for each state consider the action having highest else:
#        print("====================")
#        print(Q[s,:])
#        print(np.argmax(Q[s,:]))
#        return sample.astype(np.float32)
        return getAction(sample, env, prevAction), sample
    return env.action_space.sample(), 2

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
    
    actionNum = 3
    
    #Initializing Q-table with zeros
    q_table = np.zeros([env.observation_space.n,actionNum], np.float32)
    print(q_table)
    return env, q_table

    action = getAction(2,env)
    
    for epsIndex in range(eps):
        state = env.reset()
        for stepIndex in range(timeSteps):
            action, actionIndex = epsilon_greedy(q_table, state, env, action)
            print(action)
            new_state,r,done,_ = env.step(action)
            if (r==0):
                if done==True:
                    r = -5 #to give negative rewards when holes turn up
                    q_table[new_state] = np.ones(actionNum)*r #in terminal state Q value equals the reward
                else:
                    r = -1 #to give negative rewards to avoid long routes
            if (r==1):
                r = 100
                q_table[new_state] = np.ones(actionNum)*r #in terminal state Q value equals the reward
            q_table[state,actionIndex] = q_table[state,actionIndex] + lr * (r + y*np.max(q_table[new_state,actionIndex]) - q_table[state,actionIndex])
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

def binarySearch(env):
    _ = env.reset()
    right = env.action_space.high
    left = env.action_space.low
    total_steps = 0
    while left <= right:
        total_steps += 1
        mid = left + (right - left) / 2
        s_,r,t,_ = env.step(mid)
        if (r==1):
            break
        elif (r==0):
            if (s_ == 1):
                left = mid + 1
            elif (s_ == 2):
                print("Equal")
                break
            elif (s_ == 3):
                right = mid - 1
    print("Guessed number: ", np.asscalar(mid))
    print("Total steps: ", total_steps)

        
def actionToToRight(left, right, mid):
    left = mid + 1
    mid = left + (right - left) / 2
    return mid

def actionToToLeft(left, right, mid):
    right = mid - 1
    mid = left + (right - left) / 2
    return mid

def getAction(index, env, prevAction=0):
    right = env.action_space.high
    left = env.action_space.low
    if index == 0 :
        return actionToToRight(left,right,prevAction)
    elif index == 1:
        return actionToToLeft(left,right,prevAction)
    elif index == 2:
        return env.action_space.sample()

if __name__ == "__main__":
    env = loadEnv()
#    env, Q = runQLearning(env)
#    play(env, Q)
    binarySearch(env)

    