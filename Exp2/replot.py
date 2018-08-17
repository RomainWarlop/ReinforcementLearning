import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

colors = ['blue','green','hotpink','darkcyan','goldenrod','grey','brown','black','purple','yellow','orange']

path = "/home/romain/Bureau/ongoingImages/"

# =============================================================================
# rewards functions
# =============================================================================
m = 3
phi = {0:[3.4,0.08*m,-0.40*m,0.40*m,-0.15*m,0.01*m],
       1:[3.6,0.22*m,-1.16*m,1.20*m,-0.45*m,0.05*m],
       2:[3.8,0.20*m,-0.95*m,1.04*m,-0.54*m,0.10*m],
       3:[2.9,-0.1,-0.1,-0.01,-0.01,-0.01]
      }
arm_label = ['Action','Comedy','Thriller','Bad']

d = len(phi[0])-1
K = len(phi)
window = 5

M = K**window
print('Space state size:',M)

x = np.arange(0,2.4,0.1)
#d = 1
X = []
for elt in x:
    X.append([elt**j for j in range(d+1)])

X = np.array(X)
plt.figure(figsize=(8, 8))
ax = plt.subplot(111)    
ax.spines["top"].set_visible(False) 
ax.spines["bottom"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
for a in phi.keys():
    ax.plot(x,np.dot(phi[a],X.T),label=arm_label[a],linewidth=3)
for y in np.arange(1, 4.5, .5):    
    ax.plot(np.arange(0,2.4,0.1), [y] * len(np.arange(0,2.4,0.1)), "--", lw=0.5, color="black", alpha=0.3)
ax.legend(loc=3,fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.savefig(path+'rewards_function.pdf',bbox_inches='tight')



# =============================================================================
# pulled arms
# =============================================================================
arm_label = ["Action","Comedy","Thriller","Bad"]

K = len(arm_label)

optimal_arms = [0]+list(np.tile([2,1,0],14))[:39]
greedyOracle_arms = list(np.tile([2,0,2,1,2,1],6))+[2,0,2,1]
UCRL_arms = list(np.tile([0,0,2,2,2,2,0,0,0,2,1,3,1,1,2],3))[:40]
linUCRL_arms = list(np.tile([2,0,1],14))[:40]
linUCB_arms = list(np.tile([1,2],20))[:40]

fig = plt.figure(figsize=(8, 10))
sub1 = plt.subplot(5, 1, 1)
sub1.spines["top"].set_visible(False) 
sub1.spines["bottom"].set_visible(False)
sub1.spines["right"].set_visible(False)
sub1.spines["left"].set_visible(False)
l1 = sub1.plot(range(40),[x for x in optimal_arms],'o-',color=colors[2],linewidth=3,
          markersize=10,label='oracle optimal')
plt.yticks(range(K),arm_label,fontsize=13)
plt.xticks(fontsize=15)
plt.ylim((-0.5,3.5))
lgd1 = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize=18,frameon=False)
#
sub2 = plt.subplot(5, 1, 2)
sub2.spines["top"].set_visible(False) 
sub2.spines["bottom"].set_visible(False)
sub2.spines["right"].set_visible(False)
sub2.spines["left"].set_visible(False)
l2 = sub2.plot(range(40),[x for x in greedyOracle_arms],'o-',color=colors[3],linewidth=3,
          markersize=10,label='oracle greedy')
plt.yticks(range(K),arm_label,fontsize=13)
plt.xticks(fontsize=15)
plt.ylim((-0.5,3.5))
lgd2 = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize=18,frameon=False)
#
sub3 = plt.subplot(5, 1, 3)
sub3.spines["top"].set_visible(False) 
sub3.spines["bottom"].set_visible(False)
sub3.spines["right"].set_visible(False)
sub3.spines["left"].set_visible(False)
l3 = sub3.plot(range(40),[x for x in UCRL_arms],'o-',color=colors[0],linewidth=3,
          markersize=10,label='UCRL')
plt.yticks(range(K),arm_label,fontsize=13)
plt.xticks(fontsize=15)
plt.ylim((-0.5,3.5))
lgd3 = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize=18,frameon=False)
#
sub4 = plt.subplot(5, 1, 4)
sub4.spines["top"].set_visible(False) 
sub4.spines["bottom"].set_visible(False)
sub4.spines["right"].set_visible(False)
sub4.spines["left"].set_visible(False)
l4 = sub4.plot(range(40),[x for x in linUCRL_arms],'o-',color=colors[1],linewidth=3,
          markersize=10,label='linUCRL')
plt.yticks(range(K),arm_label,fontsize=13)
plt.xticks(fontsize=15)
plt.ylim((-0.5,3.5))
lgd4 = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize=18,frameon=False)
#
sub5 = plt.subplot(5, 1, 5)
sub5.spines["top"].set_visible(False) 
sub5.spines["bottom"].set_visible(False)
sub5.spines["right"].set_visible(False)
sub5.spines["left"].set_visible(False)
l5 = sub5.plot(range(40),[x for x in linUCB_arms],'o-',color=colors[4],linewidth=3,
          markersize=10,label='linUCB')
plt.yticks(range(K),arm_label,fontsize=13)
plt.xticks(fontsize=15)
plt.ylim((-0.5,3.5))
lgd5 = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize=18,frameon=False)
#fig.tight_layout()
plt.savefig(path+'ML_pulled_arms.pdf',bbox_extra_artists=(lgd1,), bbox_inches='tight')



# =============================================================================
# all
# =============================================================================
res = pd.DataFrame({'model':['UCRL','linUCRL','oracle \noptimal','oracle \ngreedy','linUCB'],
                    'reward':[2.528,3.218,3.567,3.498,3.327],
                 'color':colors[:5]})
res = res.sort_values(by='reward')

m = (int(res.reward.min()*10)-1)/10
M = (int(res.reward.max()*10)+1)/10

plt.figure(figsize=(8, 8))
ax = plt.subplot(111)
ax.spines["top"].set_visible(False) 
ax.spines["bottom"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.bar(range(5),
        res.reward,
        color=res.color,width=1)
plt.xticks([0,1,2,3,4],res.model,fontsize=15)
plt.yticks(fontsize=15)
for x in range(5):
    plt.annotate(xy=(x-0.35,m+0.2),s=round(list(res.reward)[x],3),fontsize=20)
plt.ylim((m,M))
plt.savefig(path+'ML_rewards_all.pdf',bbox_inches='tight')

# =============================================================================
# last
# =============================================================================
res = pd.DataFrame({'model':['UCRL','linUCRL','oracle \noptimal','oracle \ngreedy','linUCB'],
                    'reward':[3.259,3.544,3.567,3.498,3.391],
                 'color':colors[:5]})
res = res.sort_values(by='reward')

m = (int(res.reward.min()*10)-1)/10
M = (int(res.reward.max()*10)+1)/10

plt.figure(figsize=(8, 8))
ax = plt.subplot(111)
ax.spines["top"].set_visible(False) 
ax.spines["bottom"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.bar(range(5),
        res.reward,
        color=res.color,width=1)
plt.xticks([0,1,2,3,4],res.model,fontsize=15)
plt.yticks(fontsize=15)
for x in range(5):
    plt.annotate(xy=(x-0.35,m+0.2),s=round(list(res.reward)[x],3),fontsize=20)
plt.ylim((m,M))
plt.savefig(path+'ML_rewards_last.pdf',bbox_inches='tight')


