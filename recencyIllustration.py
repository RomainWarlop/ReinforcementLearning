from itertools import product as itp
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

w = 3
K = 3

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

def next_state_(x,a):
    # x[j] contains the arm that have been played j+1 timesteps ago
    y = deepcopy(x)
    y = y[:-1]
    y = [a] + y
    return y

def recency(state,a):
    decay = np.sum([f(delta+1) for delta in indexes(state,a)])
    return decay

recency_evolution = []
for state in S:
    state = list(state)
    now = recency(state,0)
    ifPlayed_nextstate = next_state_(state,0)
    ifNotPlayed_nextstate = next_state_(state,1)
    nextPlayed_recency = recency(ifPlayed_nextstate,0)
    nextNotPlayed_recency = recency(ifNotPlayed_nextstate,0)
    recency_evolution.append([now,nextPlayed_recency,nextNotPlayed_recency])

refs = [x[0] for x in recency_evolution]
up = [x[1] for x in recency_evolution]
down = [x[2] for x in recency_evolution]

refs_pos = np.argsort(refs)
refs = [refs[x] for x in refs_pos]
up = [up[x] for x in refs_pos]
down = [down[x] for x in refs_pos]

red55 = [182/255,25/255,36/255]
green55 = [89/255,178/255,56/255]
blue55 = [83/255,173/255,180/255]
lightred55 = [218/255,140/255,145/255]
lightgreen55 = [155/255,209/255,166/255]

head_length = 0.05

fig = plt.figure(figsize=(8, 6)) 
ax = plt.subplot(111)
ax.spines["top"].set_visible(False)    
ax.spines["bottom"].set_visible(False)
ax.spines["right"].set_visible(False)    
ax.spines["left"].set_visible(False) 
ax.scatter(range(1,len(refs)+1),refs,c=blue55,s=50)
ax.scatter(range(1,len(refs)+1),up,c=red55,s=50)
ax.scatter(range(1,len(refs)+1),down,c=green55,s=50)
for i in range(len(recency_evolution)):
    dy = up[i]-refs[i]
    if dy!=0:
        dy -= head_length
        ax.arrow(i+1,refs[i],
                  0,dy,color=lightred55,
                  head_width=0.3,head_length=head_length,fc='black')
    dy = down[i]-refs[i]
    if dy!=0:
        dy += head_length
        ax.arrow(i+1,refs[i],
                  0,dy,color=lightgreen55,
                  head_width=0.3,head_length=head_length,fc='black')







