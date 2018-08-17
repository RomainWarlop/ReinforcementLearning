from itertools import product as itp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np
from copy import deepcopy
from matplotlib import colors as mcolors
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

w = 5
m = 3
phi = {0:[3.4,0.08*m,-0.40*m,0.40*m,-0.15*m,0.01*m],
       1:[3.6,0.22*m,-1.16*m,1.20*m,-0.45*m,0.05*m],
       2:[3.8,0.20*m,-0.95*m,1.04*m,-0.54*m,0.10*m],
       3:[2.9,-0.1,-0.1,-0.01,-0.01,-0.01]
      }
arm_label = ['Action','Comedy','Thriller','Bad']
d = len(phi[0])-1
K = len(phi)

S = list(itp(range(K), repeat=w)) # all possibles states

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
    
def reward(decay,a):
    #decay = [f(delta+1) for delta in indexes(x,a)] # compute decay
    #decay = np.sum(decay)
    r = np.dot(phi[a],[decay**j for j in range(d+1)])
    return r

def recency(state,a):
    decay = np.sum([f(delta+1) for delta in indexes(state,a)])
    return decay

tab = pd.DataFrame(columns=['arm','recency','reward',
                            'next_recency_played','next_reward_played',
                            'next_recency_notplayed','next_reward_notplayed'])
recencies = {}
rewards = {}
for a in range(K):
    recencies[a] = []
    rewards[a] = []
    for state in S:
        state = list(state)
        recency_ = recency(state,a)
        ifPlayed_nextstate = next_state_(state,a)
        ifNotPlayed_nextstate = next_state_(state,(a+1)%K)
        nextPlayed_recency = recency(ifPlayed_nextstate,a)
        nextNotPlayed_recency = recency(ifNotPlayed_nextstate,a)
        recencies[a].append([recency_,nextPlayed_recency,nextNotPlayed_recency])
        rewards[a].append([reward(recency_,a),
                           reward(nextPlayed_recency,a),
                           reward(nextNotPlayed_recency,a)])
        tab.loc[len(tab)] = [a,recency_,reward(recency_,a),
                            nextPlayed_recency,reward(nextPlayed_recency,a),
                            nextNotPlayed_recency,reward(nextNotPlayed_recency,a)]

x = tab.loc[tab['arm']==1,'recency']
y = tab.loc[tab['arm']==1,'reward']
plt.scatter(x,y)

tab = tab.drop_duplicates()

style="Simple,tail_width=0.5,head_width=10,head_length=8"
kw = []
kw.append(dict(arrowstyle=style, color="blue"))
kw.append(dict(arrowstyle=style, color="red"))

path = "/home/romain/Bureau/ongoingImages/"

a = 1
tab_a = tab.loc[tab['arm']==a]
colors_a = np.random.choice(list(colors.keys()),len(tab_a),False)

plt.figure(figsize=(20,10))
ax = plt.subplot(111)
ax.spines["top"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
plt.ylim((2.4,3.8))
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
for i in range(len(tab_a)):
    ind = tab_a.index[i]
    kw = dict(arrowstyle=style, color=colors_a[i])
    ax.scatter(tab_a.loc[ind,'recency'],tab_a.loc[ind,'reward'],s=150,c=colors_a[i])
    
    arr = patches.FancyArrowPatch((tab_a.loc[ind,'recency'],tab_a.loc[ind,'reward']),
                             (tab_a.loc[ind,'next_recency_played'],tab_a.loc[ind,'next_reward_played']),
                             connectionstyle="arc3,rad=.5", **kw)
    plt.gca().add_patch(arr)

    arr = patches.FancyArrowPatch((tab_a.loc[ind,'recency'],tab_a.loc[ind,'reward']),
                             (tab_a.loc[ind,'next_recency_notplayed'],tab_a.loc[ind,'next_reward_notplayed']),
                             connectionstyle="arc3,rad=.5", **kw)
    plt.gca().add_patch(arr)
plt.savefig(path+'rewards_cycle_'+arm_label[a]+'.pdf',bbox_inches='tight')



