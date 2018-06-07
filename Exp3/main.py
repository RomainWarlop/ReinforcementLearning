#==============================================================================
# Exp 2 of "A Linear Reinforcement Learning Algorithm For Non Stationary Arms"
# The parameters are choose based on movielens100k fitted curve which specific
# magnitude and intercept
#==============================================================================

from UCRL import UCRL, linUCRL
from UCB import linUCB
import numpy as np
import pandas as pd
from copy import deepcopy
from itertools import product as itp
import matplotlib.pyplot as plt
import ast
from random import shuffle

colors = ['blue','green','hotpink','darkcyan','goldenrod','grey','brown','purple','yellow','orange']

path = '/home/romain/Documents/BitBucket/PhD/linucrl/Exp3/'

data = pd.read_csv(path+'oracle_rewards.csv')
data.index = data['state']

# params
d = 10
window = 10
K = 2
arm_label = ['A','B']

M = K**window
print('Space state size:',M)


#==============================================================================
# functions 
#==============================================================================
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
    state = str(x)
    if a==0:
        p = data.loc[state,'mn_rewardA']
    else:
        p = data.loc[state,'mn_rewardB']
    
    # sample reward of 1 with probability p
    r = 1 if np.random.random()<=p else 0
    return r

def reward_oracle(x,a):
    state = str(x)
    if a==0:
        p = data.loc[state,'mn_rewardA']
    else:
        p = data.loc[state,'mn_rewardB']
    return p

#==============================================================================
#==============================================================================
# POLICIES
#==============================================================================
#==============================================================================
S = list(itp(range(K), repeat=window)) # all possibles states
S = list(map(lambda s: str(list(s)),S))

T = 5000#*M
start = S[0]
gamma = 0.99
n_iter = 200
n_repeat = 5

#==============================================================================
# UCRL
#==============================================================================
print('\n Start UCRL')
mod_ucrl = dict.fromkeys(range(n_repeat))
for i in range(n_repeat):
    mod_ucrl[i] = UCRL(alpha=2,states=S,s0=start,actions=range(K),
                    reward=reward,transition=next_state_)
    mod_ucrl[i].run(T=T,gamma=gamma,n_iter=n_iter)

#==============================================================================
# linUCRL
#==============================================================================
print('\n Start linUCRL')
mod_linucrl = dict.fromkeys(range(n_repeat))
for i in range(n_repeat):
    mod_linucrl[i] = linUCRL(alpha=1,d=d,states=S,s0=start,actions=range(K),
                    reward=reward,transition=next_state_)
    mod_linucrl[i].run(T=T,lbda=10.,gamma=gamma,n_iter=n_iter)

#==============================================================================
# linUCB
#==============================================================================
print('\n Start linUCB')
mod_linucb = dict.fromkeys(range(n_repeat))
for i in range(n_repeat):
    mod_linucb[i] = linUCB(alpha=2,d=d,s0=start,actions=range(K),
                    reward=reward,transition=next_state_)
    mod_linucb[i].run(T=T,lbda=10.)

#==============================================================================
# epsilon-greedy    
#==============================================================================
perfByState = dict.fromkeys(S)
for state in S:
    perfByState[state] = {}
    perfByState[state][0] = []
    perfByState[state][1] = []

eps = 1/100

epsgreedy_rewards = dict.fromkeys(range(n_repeat))
epsgreedy_arms = [] # same for all of the n_repeat tests

def mean_empty(l):
    if len(l)>0:
        return np.mean(l)
    else:
        # return something high because if (state,'A') received a feedback of 1
        # at the first pull, (state,'B') will always be lower and never pulled
        return 2 

for i in range(n_repeat):
    epsgreedy_rewards[i] = []
    state = start
    for t in range(T):
        eps_t = 1 if t<50 else eps # start to explore at random a little !
        if np.random.random()<eps_t:
            a = np.random.randint(0,2)
        else:
            rA, rB = mean_empty(perfByState[state][0]), mean_empty(perfByState[state][1])
            a = 1
            if rA>rB:
                a = 0
        r = reward(ast.literal_eval(state),a) # receive with noise
        perfByState[state][a].append(r)
        state = next_state_(ast.literal_eval(state),a)
        epsgreedy_rewards[i].append(r)
        epsgreedy_arms.append(a)
        state = str(state)

#==============================================================================
# Optimal policy
#==============================================================================
def Bellman_a(a,gamma,state):
    state = ast.literal_eval(state)
    # update for one state at random
    r = reward_oracle(state,a) # we play a in this state
    next_state = next_state_(state,a)
    r+=gamma*V[str(next_state)]
    return r

V = {}
for state in S:
    V[state] = 0

for it in range(n_iter):
    shuffle(S)
    for state in S:
        V[state] = max(list(map(lambda a: Bellman_a(a,gamma,state),range(K))))

policy = {}
for state in V.keys():
    state = ast.literal_eval(state)
    tmp = np.zeros(K)
    for a in range(K):
        r = reward_oracle(state,a) # compute without noise
        next_state = next_state_(state,a)
        r+=gamma*V[str(next_state)]
        tmp[a] = r
    policy[str(state)] = np.argmax(tmp)

optimal_rewards = dict.fromkeys(range(n_repeat))
optimal_arms = [] # same for all of the n_repeat tests

for i in range(n_repeat):
    optimal_rewards[i] = []
    state = ast.literal_eval(start)
    for t in range(T):
        a = policy[str(state)]
        r = reward(state,a) # receive with noise
        state = next_state_(state,a)
        optimal_rewards[i].append(r)
        optimal_arms.append(a)

#==============================================================================
# Greedy oracle
#==============================================================================
greedyOracle_rewards = dict.fromkeys(range(n_repeat))
greedyOracle_arms = []

for i in range(n_repeat):
    greedyOracle_rewards[i] = []
    state = ast.literal_eval(start)
    for t in range(T):
        rewards = []
        for a in range(K):
            rewards.append(reward_oracle(state,a)) # compute without noise
        arm = int(np.argmax(rewards))
        rew_hat = reward(state,arm) # receive with noise
        state = next_state_(state,arm)
        greedyOracle_rewards[i].append(rew_hat)
        greedyOracle_arms.append(arm)

print('greedy Oracle:',np.mean(greedyOracle_rewards[0]))

#==============================================================================
#==============================================================================
# PLOTS
#==============================================================================
#==============================================================================

# mean rewards among all repetition
ucrl_rewards_mean = np.array(mod_ucrl[0].reward_sequence)/n_repeat
linucrl_rewards_mean = np.array(mod_linucrl[0].reward_sequence)/n_repeat
linucb_rewards_mean = np.array(mod_linucb[0].reward_sequence)/n_repeat
epsgreedy_rewards_mean = np.array(epsgreedy_rewards[0])/n_repeat
optimal_rewards_mean = np.array(optimal_rewards[0])/n_repeat
greedyOracle_rewards_mean = np.array(greedyOracle_rewards[0])/n_repeat

for i in range(1,n_repeat):
    ucrl_rewards_mean += np.array(mod_ucrl[i].reward_sequence)/n_repeat
    linucrl_rewards_mean += np.array(mod_linucrl[i].reward_sequence)/n_repeat
    linucb_rewards_mean += np.array(mod_linucb[i].reward_sequence)/n_repeat
    epsgreedy_rewards_mean = np.array(epsgreedy_rewards[i])/n_repeat
    optimal_rewards_mean += np.array(optimal_rewards[i])/n_repeat
    greedyOracle_rewards_mean += np.array(greedyOracle_rewards[i])/n_repeat

#==============================================================================
# perf plot
#==============================================================================
plt.figure(figsize=(12,10))
plt.plot(range(T),np.cumsum(ucrl_rewards_mean),linewidth=3,c=colors[0],
             label='UCRL')
plt.scatter(mod_ucrl[0].update_time,np.cumsum(ucrl_rewards_mean)[mod_ucrl[0].update_time])
#
plt.plot(range(T),np.cumsum(linucrl_rewards_mean),linewidth=3,c=colors[1],
             label='linUCRL')
plt.scatter(mod_linucrl[0].update_time,np.cumsum(linucrl_rewards_mean)[mod_linucrl[0].update_time])
#
plt.plot(range(T),np.cumsum(optimal_rewards_mean),linewidth=3,c=colors[2],
             label='oracle optimal policy')
#
plt.plot(range(T),np.cumsum(greedyOracle_rewards_mean),linewidth=3,c=colors[3],
             label='oracle greedy policy')
#
plt.plot(range(T),np.cumsum(linucb_rewards_mean),linewidth=3,c=colors[4],
             label='linUCB')
#
plt.plot(range(T),np.cumsum(epsgreedy_rewards_mean),linewidth=3,c=colors[5],
             label='epsilon greedy policy')
#
plt.xlim((0,T))
plt.ylim((0,max(sum(mod_ucrl[0].reward_sequence),sum(optimal_rewards))))
lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#plt.title('Cumulated reward')
plt.savefig(path+'cumulative_rewards.png',bbox_inches='tight')
#plt.show()

#==============================================================================
# Plot of the strategies
#==============================================================================
fig = plt.figure(figsize=(8, 8))
#
sub1 = plt.subplot(6, 1, 1)
sub1.plot(range(40),[x for x in mod_ucrl[0].arm_sequence[-40:]],'o-',color=colors[0],linewidth=3,
          markersize=10,label='UCRL')
plt.yticks(range(K),arm_label)
lgd = plt.legend(loc=10,fontsize=15)
#
sub2 = plt.subplot(6, 1, 2)
sub2.plot(range(40),[x for x in mod_linucrl[2].arm_sequence[-40:]],'o-',color=colors[1],linewidth=3,
          markersize=10,label='linUCRL')
plt.yticks(range(K),arm_label)
lgd = plt.legend(loc=10,fontsize=15)
#
sub3 = plt.subplot(6, 1, 3)
sub3.plot(range(40),[x for x in optimal_arms[-40:]],'o-',color=colors[2],linewidth=3,
          markersize=10,label='oracle optimal')
plt.yticks(range(K),arm_label)
lgd = plt.legend(loc=10,fontsize=15)
#
sub4 = plt.subplot(6, 1, 4)
sub4.plot(range(40),[x for x in greedyOracle_arms[-40:]],'o-',color=colors[3],linewidth=3,
          markersize=10,label='oracle greedy')
plt.yticks(range(K),arm_label)
lgd = plt.legend(loc=10,fontsize=15)
#
sub5 = plt.subplot(6, 1, 5)
sub5.plot(range(40),[x for x in mod_linucb[0].arm_sequence[-40:]],'o-',color=colors[4],linewidth=3,
          markersize=10,label='linUCB')
plt.yticks(range(K),arm_label)
lgd = plt.legend(loc=10,fontsize=15)
#
sub6 = plt.subplot(6, 1, 6)
sub6.plot(range(40),[x for x in epsgreedy_arms[-40:]],'o-',color=colors[5],linewidth=3,
          markersize=10,label='eps greedy')
plt.yticks(range(K),arm_label)
lgd = plt.legend(loc=10,fontsize=15)
#fig.tight_layout()
plt.savefig(path+'pulled_arms.png',bbox_extra_artists=(lgd,), bbox_inches='tight')

#==============================================================================
# Mean reward comparison from one random starting point
#==============================================================================
res = pd.DataFrame({'model':['UCRL','linUCRL','oracle optimal','oracle greedy',
                             'linUCB','eps-greedy'],
                    'reward':[np.mean(ucrl_rewards_mean),np.mean(linucrl_rewards_mean),
         np.mean(optimal_rewards_mean),np.mean(greedyOracle_rewards_mean),
         np.mean(linucb_rewards_mean),np.mean(epsgreedy_rewards_mean)],
                    'color':colors[:6]})
res = res.sort_values(by='reward')

m = (int(res.reward.min()*10)*0.9)/10
M = (int(res.reward.max()*10)*1.1)/10

plt.figure(figsize=(8, 8))
plt.bar(range(6),
        res.reward,
        color=res.color,width=1)
plt.xticks([0.5,1.5,2.5,3.5,4.5,5.5],res.model)
for x in range(6):
    plt.annotate(xy=(x+0.2,(m+M)/2),s=round(list(res.reward)[x],3),fontsize=20)
plt.ylim((m,M))
plt.savefig(path+'all_rewards.png',bbox_inches='tight')

#==============================================================================
# Mean reward first steps
#==============================================================================
zoom = 200 #mod_linucrl[0].update_time[0]
res = pd.DataFrame({'model':['UCRL','linUCRL','oracle \noptimal','oracle \ngreedy',
                             'linUCB','$\epsilon$-greedy'],
                    'reward':[np.mean(ucrl_rewards_mean[:zoom]),
                             np.mean(linucrl_rewards_mean[:zoom]),
         np.mean(optimal_rewards_mean[:zoom]),np.mean(greedyOracle_rewards_mean[:zoom]),
        np.mean(linucb_rewards_mean[:zoom]),np.mean(epsgreedy_rewards_mean[:zoom])],
                 'color':colors[:6]})
res = res.sort_values(by='reward')

m = (int(res.reward.min()*10)*0.9)/10
M = (int(res.reward.max()*10)*1.1)/10

plt.figure(figsize=(8, 8))
plt.bar(range(6),
        res.reward,
        color=res.color,width=1)
plt.xticks([0.5,1.5,2.5,3.5,4.5,5.5],res.model,fontsize=15)
for x in range(6):
    plt.annotate(xy=(x+0.2,(m+M)/2),s=round(list(res.reward)[x],3),fontsize=20)
plt.ylim((m,M))
plt.savefig(path+'rewards_first.png',bbox_inches='tight')

#==============================================================================
# Mean reward last steps
#==============================================================================
Astate = list(np.repeat(0,window))
Bstate = list(np.repeat(1,window))
mnA = data.loc[str(Astate),'mn_rewardA'] 
mnB = data.loc[str(Bstate),'mn_rewardB'] 

max_last_update = 0
for i in range(n_repeat):
    if mod_linucrl[i].update_time[-1]>max_last_update:
        max_last_update = mod_linucrl[i].update_time[-1]

last = T-max_last_update
res = pd.DataFrame({'model':['UCRL','linUCRL','oracle \noptimal','oracle \ngreedy',
                             'linUCB','$\epsilon$-greedy','only A','only B'],
                    'reward':[np.mean(ucrl_rewards_mean[-last:]),
                              np.mean(linucrl_rewards_mean[-last:]),
         np.mean(optimal_rewards_mean[-last:]),np.mean(greedyOracle_rewards_mean[-last:]),
         np.mean(linucb_rewards_mean[-last:]),np.mean(epsgreedy_rewards_mean[-last:]),
         mnA,mnB],
                 'color':colors[:8]})
res = res.sort_values(by='reward')

m = (int(res.reward.min()*10)*0.9)/10
M = (int(res.reward.max()*10)*1.1)/10

plt.figure(figsize=(12, 8))
plt.bar(range(8),
        res.reward,
        color=res.color,width=1)
plt.xticks([0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5],res.model,fontsize=15)
for x in range(8):
    plt.annotate(xy=(x+0.2,(m+M)/2),s=round(list(res.reward)[x],3),fontsize=20)
plt.ylim((m,M))
plt.savefig(path+'rewards_last.png',bbox_inches='tight')

#==============================================================================
# final plot
#==============================================================================

res = pd.DataFrame({'model':['UCRL','linUCRL','oracle \noptimal','oracle \ngreedy',
                             'only B'],
                    'reward':[(np.mean(ucrl_rewards_mean[-last:])-mnA)/mnA*100,
                              (np.mean(linucrl_rewards_mean[-last:])-mnA)/mnA*100,
         (np.mean(optimal_rewards_mean[-last:])-mnA)/mnA*100,
         (np.mean(greedyOracle_rewards_mean[-last:])-mnA)/mnA*100,
         (mnB-mnA)/mnA*100],
                 'color':colors[:5]})
res = res.sort_values(by='reward')

m = (int(res.reward.min()*10)*0.9)/10
M = (int(res.reward.max()*10)*1.1)/10

plt.figure(figsize=(12, 8))
plt.bar(range(len(res)),
        res.reward,
        color=res.color,width=1)
plt.xticks([0.5,1.5,2.5,3.5,4.5],res.model,fontsize=15)
for x in range(len(res)):
    plt.annotate(xy=(x+0.2,(m+M)/2),s=str(round(list(res.reward)[x],1))+'%',fontsize=20)
plt.ylim((m,M))
plt.savefig(path+'improvement_last.pdf',bbox_inches='tight')

