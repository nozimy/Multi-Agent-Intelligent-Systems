# importing dependency libraries
from __future__ import print_function
import gym
import numpy as np
import time
import os

def loadEnv():
    #Load the environment
    env = gym.make('FrozenLake-v0')
    s = env.reset()
    print("initial state : ",s)
    env.render()
    print()  
    print(env.action_space) #number of actions
    print(env.observation_space) #number of states
    print()
    print("Number of actions : ",env.action_space.n)
    print("Number of states : ",env.observation_space.n)
    print()
    return env

clear = lambda: os.system('clear')

#Epsilon-Greedy approach for Exploration and Exploitation of the state-action spaces
def epsilon_greedy(q_table,state,na, env):
    epsilon = 0.3
    p = np.random.uniform(low=0,high=1)
    if p > epsilon:
        # exploitation
        return np.argmax(q_table[state,:])#say here,initial policy = for each state consider the action having highest else:
    # exploration
    return env.action_space.sample()
    
def q_learning(env):
    # Q-Learning Implementation
    NUM_ACTIONS = env.action_space.n
    NUM_STATES = env.observation_space.n
    #Initializing Q-table with zeros
    q_table = np.zeros([NUM_STATES,NUM_ACTIONS])
    
    #set hyperparameters
    lr = 0.95 #learning rate
    y = 0.5 #discount factor lambda
    eps = 100000 #total episodes being 100000
    total_eps = 0
    
    for i in range(eps):
    #    print("===============")
    #    print("Episode", i)
    #    print("===============")
    #    time.sleep(0.05)
    #    clear()
        total_eps +=1
        done = False
        state = env.reset()
        while done is not True:
    #        env.render()
    #        clear()
            action = epsilon_greedy(q_table,state,env.action_space.n, env)
            new_state,reward,done,_ = env.step(action)
            if (reward==0):
                if done is True:
                    reward = -5 #to give negative rewards when holes turn up
                    q_table[new_state] = np.ones(env.action_space.n)*reward #in terminal state q_table value equals the reward
                else:
                    reward = -1 #to give negative rewards to avoid long routes
            if (reward==1):
                reward = 100
                q_table[new_state] = np.ones(env.action_space.n)*reward #in terminal state q_table value equals the reward
            q_table[state,action] += lr * (reward + y*np.max(q_table[new_state,action]) - q_table[state,action])
            state = new_state
            if (done is True):
                break
    print("Total episods: ", total_eps)
    return q_table
        
def check(env, q_table, show=False):
    if show:
        print("Q-table")
        print(q_table)
        print()
        print("Output after learning")
        print()
    #learning ends with the end of the above loop of several episodes above
    #let's check how much our agent has learned
    rew_tot=0.
    state = env.reset()
    if show:
        env.render()
    done = False
    while done is not True:
        action = np.argmax(q_table[state])
        new_state,reward,done,_ = env.step(action)
#        if show:
#            print(q_table[state,:])
#            print(new_state, reward, done, action)
#            print("===============")
        rew_tot += reward
        env.render()
        state = new_state
    if show:
        print("Reward:", rew_tot)
    
    if state == 15:
        return True
    return False

def play(env):
    q_table = q_learning(env)
    attempt_num = 0
    while not check(env, q_table):
        attempt_num += 1
        q_table = q_learning(env)
    
    print("Attempt num: ", attempt_num)
    check(env, q_table, True)
    
if __name__ == "__main__":
    env = loadEnv()
    play(env)
    