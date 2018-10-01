# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 18:25:59 2018

@author: Junaid.raza
"""

"""
Rinforcment Learnign is about a men, machine or a neural net to learning to navigate an
uncertain environment with the goal of maximizing numerical reward.
Sports is the best examples. Where your action changes the status of your game.
Each action is completed for a reward (0,1) 
Have to follow the policy or rule to maximze the score.

There is an Agent.
Agent have an Environmnent.
Agent knows the State (S) of Environment.
Agent takes any Action (A).
Environment updates with new State (S) and some Reward (R) in form of Binary (0,1).

Agent doesnt know about Environment. He is just experiencing the environment by actions , state and reward.

Gym is a toolkit for developing and comparing reinforcement learning algorithms.
"""
import gym

"""
Here’s a bare minimum example of getting something running. This will run an 
instance of the CartPole-v0 environment for 1000 timesteps, rendering the environment 
at each step. You should see a window pop up rendering the classic cart-pole problem:

"""
#This code is to test first form of our algorithm
#UnComment these four lines to see first shape and comment the below code
#env = gym.make('CartPole-v0')
#env.reset()
#for _ in range(1000):
#    env.render()
#    env.step(env.action_space.sample()) # take a random action
    
    
env = gym.make('CartPole-v0')
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

"""
try replacing CartPole-v0 above with something like MountainCar-v0, 
MsPacman-v0 (requires the Atari dependency), or Hopper-v1 (requires the MuJoCo dependencies). 
"""

"""
If we ever want to do better than take random actions at each step, 
it’d probably be good to actually know what our actions are doing to the environment.

The environment’s step function returns exactly what we need. In fact, step returns 
four values. These are:
    
    observation (object): 
        an environment-specific object representing your observation of the environment. 
        For example, pixel data from a camera, joint angles and joint velocities of a robot, 
        or the board state in a board game.
    reward (float): 
        amount of reward achieved by the previous action. The scale varies between 
        environments, but the goal is always to increase your total reward
    done (boolean): 
            whether it’s time to reset the environment again. Most (but not all) 
            tasks are divided up into well-defined episodes, and done being True indicates the 
            episode has terminated. (For example, perhaps the pole tipped too far, or you lost 
            your last life.)
    info (dict): 
        diagnostic information useful for debugging. It can sometimes be useful for 
        learning (for example, it might contain the raw probabilities behind the 
        environment’s last state change). However, official evaluations of your agent 
        are not allowed to use this for learning.
        
        
Spaces
In the examples above, we’ve been sampling random actions from the environment’s 
action space. But what actually are those actions? Every environment comes with an 
action_space and an observation_space. These attributes are of type Space, and they 
describe the format of valid actions and observations:
    
    import gym
    env = gym.make('CartPole-v0')
    print(env.action_space)
    #> Discrete(2)
    print(env.observation_space)
    #> Box(4,)
    
The Discrete space allows a fixed range of non-negative numbers, so in this case 
valid actions are either 0 or 1. The Box space represents an n-dimensional box, so 
valid observations will be an array of 4 numbers. We can also check the Box’s bounds:
   
    print(env.observation_space.high)
    #> array([ 2.4       ,         inf,  0.20943951,         inf])
    print(env.observation_space.low)
    #> array([-2.4       ,        -inf, -0.20943951,        -inf])
    
This introspection can be helpful to write generic code that works for many different 
environments. Box and Discrete are the most common Spaces. You can sample from a Space 
or check that something belongs to it:
    
    from gym import spaces
    space = spaces.Discrete(8) # Set with 8 elements {0, 1, 2, ..., 7}
    x = space.sample()
    assert space.contains(x)
    assert space.n == 8
"""















