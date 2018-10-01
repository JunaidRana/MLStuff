# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 19:13:22 2018

@author: Junaid.raza
"""

"""
Behind this strange and mysterious name hides pretty straightforward concept. 
Dynamic programming or DP, in short, is a collection of methods used calculate the 
optimal policies — solve the Bellman equations.



Model-Based RL: Policy and Value Iteration using Dynamic Programming

    Understand the difference between Policy Evaluation and Policy Improvement and how these processes interact
    Understand the Policy Iteration Algorithm
    Understand the Value Iteration Algorithm
    Understand the Limitations of Dynamic Programming Approaches
    
Summary
    1-Dynamic Programming (DP) methods assume that we have a perfect model of the 
        environment's Markov Decision Process (MDP). That's usually not the case in practice, 
        but it's important to study DP anyway.
    2-Policy Evaluation: Calculates the state-value function V(s) for a given policy. 
        In DP this is done using a "full backup". At each state, we look ahead one step at 
        each possible action and next state. We can only do this because we have a perfect 
        model of the environment.
    3-Full backups are basically the Bellman equations turned into updates.
    4-Policy Improvement: Given the correct state-value function for a policy we can act 
        greedily with respect to it (i.e. pick the best action at each state). Then we are 
        guaranteed to improve the policy or keep it fixed if it's already optimal.
        5-Policy Iteration: Iteratively perform Policy Evaluation and Policy Improvement until 
        we reach the optimal policy.
    6-Value Iteration: Instead of doing multiple steps of Policy Evaluation to find the 
        "correct" V(s) we only do a single step and improve the policy immediately. In practice, 
        this converges faster.
    7-Generalized Policy Iteration: The process of iteratively doing policy evaluation 
        and improvement. We can pick different algorithms for each of these steps but the 
        basic idea stays the same.
    8-DP methods bootstrap: They update estimates based on other estimates (one step ahead).


"""

def agent(policy, starting_position=None):
    l = list(range(16))
    state_to_state_prime = create_state_to_state_prime_verbose_map()
    agent_position = randint(1, 14) if starting_position is None else starting_position
        
    step_number = 1
    
    while not (agent_position == 0 or agent_position == 15):
        current_policy = policy[agent_position]
        next_move = random()
        lower_bound = 0
        for action, chance in current_policy.items():
            if chance == 0:
                continue
            if lower_bound <= next_move < lower_bound + chance:
                agent_position = state_to_state_prime[agent_position][action]
                break 
            lower_bound = lower_bound + chance
                
        step_number += 1   
    
    return step_number

#Link to Tutorial
#https://medium.com/harder-choices/dynamic-programming-in-python-reinforcement-learning-bb288d95288f













