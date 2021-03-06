#==============================================================================
# Exp 1 of "A Linear Reinforcement Learning Algorithm For Non Stationary Arms"
# Intuition on situation of interest for a long-term strategy
#==============================================================================

import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import ast
from itertools import product as itp
import time
from random import shuffle

red55 = [182/255,25/255,36/255]
blue55 = [83/255,173/255,180/255]
green55 = [89/255,178/255,107/255]

t0 = time.time()
colors = [blue55,green55,red55]

path = "/home/romain/Bureau/ongoingImages/"

#np.random.seed(123)

# params
#==============================================================================
# Loop over the two tests
#==============================================================================
def rounds(l,d=2):
    out = list(map(lambda x: round(x,d),l))
    return out

def indexes(l,elt):
    indices = [i for i, x in enumerate(l) if x == elt]
    return indices

def f(delta):
    if delta==0:
        out = 0
    else:
        out = 1/delta
    return out

# Transition probabilities
def next_state_(x,a):
    # x[j] contains the arm that have been played j+1 timesteps ago
    y = deepcopy(x)
    y = y[:-1]
    y = [a] + y
    return y
    
def reward(x,a):
    decay = [f(delta+1) for delta in indexes(x,a)] # compute decay
    decay = np.sum(decay)
    r = np.dot(phi[a],[decay**j for j in range(d+1)])
    return r

def update(a):
    r = reward(state,a) # we play a in this state
    next_state = next_state_(state,a)
    r+=gamma*V[str(next_state)]
    return r

def Bellman_a(a,gamma,state):
    # update for one state at random
    r = reward(state,a) # we play a in this state
    next_state = next_state_(state,a)
    r+=gamma*V[str(next_state)]
    return r
    
def Bellman(V,gamma,state):
    # update for one state at random
    tmp = np.zeros(K)
    for a in range(K):
        r = reward(state,a) # we play a in this state
        next_state = next_state_(state,a)
        r+=gamma*V[str(next_state)]
        tmp[a] = r
    V[str(state)] = max(tmp)
    return V

mu0 = 1
c1 = {0:0.3,1:2}
c2 = {0:0.4,1:0.01}
alpha = {0:1.5,1:2}

for _it in [0]: #range(2):
    phi = {0:[mu0,  -c1[_it]*mu0],
           1:[mu0/alpha[_it],  -c2[_it]*mu0],
          }
    d = len(phi[0])-1
    window = 8 
    
    K = len(phi)
    M = K**window
    print('Space state size:',M)
    
    description = ["Decaying arms with :"]
    description.append("- "+str(K)+" arms ")
    description.append("- a window of length "+str(window))
    for a in range(K):
        description.append("- arm "+str(a)+": "+str(phi[a]))
    
    mydir = 'README_param'+str(_it)+'.txt'
    np.savetxt(path+mydir,tuple(description),fmt="%s")
        
    V = {}
    S = itp(range(K), repeat=window) # all possibles states
    for state in S:
        state = list(state)
        V[str(state)] = 0
    stops = [1000] #,1000]
    gamma = 0.99
    n_iter = max(stops)
    policies = dict.fromkeys(stops)
    for it in range(n_iter):
        mem = deepcopy(V) # copy old value
        S = itp(range(K), repeat=window) # all possibles states
        S = list(S)
        shuffle(S)
        for state in S:
            state = list(state)
            V[str(state)] = max(list(map(lambda a: Bellman_a(a,gamma,state),range(K))))
        
        if (it+1) in stops:
            # compute optimal policy
            policies[it+1] = {}
            for state in V.keys():
                state = ast.literal_eval(state)
                tmp = np.zeros(K)
                for a in range(K):
                    r = reward(state,a) # we play a in this state
                    next_state = next_state_(state,a)
                    r+=gamma*V[str(next_state)]
                    tmp[a] = r
                policies[it+1][str(state)] = np.argmax(tmp)
    
    #==============================================================================
    # compare optimal policy to "play best current arm" policy
    #==============================================================================    
    regret = dict.fromkeys(stops)
    for stop in stops:
        regret[stop] = 0
    
    n_try = 1
    for _try in range(n_try):
        start = list(np.random.randint(0,K,window))
        T = 1000
        
        # VI policy obtained reward with the reward 
        # 1- computed on the last L steps as done in VI
        # 2- computed with the full history at it should be in reality
        VI_rew = dict.fromkeys(stops)
        VI_arms = dict.fromkeys(stops)
        for stop in stops:
            state = start
            VI_rew[stop] = []
            VI_arms[stop] = []
            for t in range(T):
                if str(state) not in policies[stop].keys():
                    arm = np.random.randint(0,K)
                else:
                    arm = int(policies[stop][str(state)])
                rew = reward(state,arm)
                state = next_state_(state,arm)
                VI_rew[stop].append(rew)
                VI_arms[stop].append(arm)
        
        # naive strategy that compute reward only on the last L steps
        naive_rew = []
        naive_arms = []
        state = start
        for t in range(T):
            rewards = []
            for a in range(K):
                rewards.append(reward(state,a))
            arm = int(np.argmax(rewards))
            rew = rewards[arm]
            state = next_state_(state,arm)
            naive_rew.append(rew)
            naive_arms.append(arm)
        
        for stop in stops:
            regret[stop] += sum(VI_rew[stop])-sum(naive_rew)
    
    for stop in stops:
        regret[stop] /= n_try
    
    #==============================================================================
    # Plot of the strategy for the VI and baseline
    #==============================================================================
    fig = plt.figure(figsize=(8, 8))
    
    sub1 = plt.subplot(2, 1, 1)
    sub1.spines["top"].set_visible(False) 
    sub1.spines["bottom"].set_visible(False)
    sub1.spines["right"].set_visible(False)
    sub1.spines["left"].set_visible(False)
    sub1.plot(range(60),[x+1 for x in VI_arms[n_iter][:60]],'o-',linewidth=3,
              markersize=10,label='optimal policy',color=blue55)
    lgd = plt.legend(loc=10,fontsize=25)
    plt.yticks([1,2],[1,2],fontsize=30)
    plt.xticks(fontsize=15)
    plt.ylim((0.8,2.2))
    
    sub2 = plt.subplot(2, 1, 2)
    sub2.spines["top"].set_visible(False) 
    sub2.spines["bottom"].set_visible(False)
    sub2.spines["right"].set_visible(False)
    sub2.spines["left"].set_visible(False)
    sub2.plot(range(60),[x+1 for x in naive_arms[:60]], 'o-',linewidth=3,
              markersize=10,label='greedy policy',color=red55)
    fig.tight_layout()
    lgd = plt.legend(loc=10,fontsize=25)
    plt.yticks([1,2],[1,2],fontsize=30)
    plt.xticks(fontsize=15)
    plt.ylim((0.8,2.2))
    plt.savefig(path+'pulled_arms_param'+str(_it)+'.pdf',
                bbox_extra_artists=(lgd,), bbox_inches='tight')
    #plt.show()
    
    #==============================================================================
    # Mean reward comparison from one random starting point
    #==============================================================================
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    mean0 = reward([0 for i in range(window)],0)
    mean1 = reward([1 for i in range(window)],1)
    bestMean = max(mean0,mean1)
    m = np.min([np.mean(VI_rew[n_iter]),np.mean(naive_rew),bestMean])
    M = np.max([np.mean(VI_rew[n_iter]),np.mean(naive_rew),bestMean])
    plt.bar(range(3),[np.mean(VI_rew[n_iter]),np.mean(naive_rew),bestMean],
            color=[blue55,red55,green55],width=1)
    plt.xticks(range(3),['optimal \npolicy','greedy\npolicy','best\nsingle arm'],fontsize=25)
    plt.yticks(fontsize=15)
    plt.xlim((-0.5,3))
    plt.ylim((0.95*m,1.01*M))
    plt.savefig(path+'pulled_arms_param'+str(_it)+'_rewards.pdf',
                bbox_inches='tight')
    
    t1 = time.time()
    
    print('total execution time:',str(t1-t0))
