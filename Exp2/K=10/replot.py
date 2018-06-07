import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

colors = ['blue','green','hotpink','darkcyan','goldenrod','grey','brown','black','purple','yellow','orange']

path = '/home/romain/Documents/PhD/linucrl/Exp2/K=10/'

# first
res = pd.DataFrame({'model':['linUCRL','oracle \noptimal','oracle \ngreedy','linUCB'],
                    'reward':[3.486,3.551,3.538,3.284],
                 'color':colors[1:5]})
res = res.sort_values(by='reward')

m = (int(res.reward.min()*10)-1)/10
M = (int(res.reward.max()*10)+1)/10

plt.figure(figsize=(8, 8))
plt.bar(range(4),
        res.reward,
        color=res.color,width=1)
plt.xticks([0,1,2,3],res.model,fontsize=15)
for x in range(4):
    plt.annotate(xy=(x-0.3,m+0.2),s=round(list(res.reward)[x],3),fontsize=20)
plt.ylim((m,M))
plt.savefig(path+'ML_rewards_first.pdf',bbox_inches='tight')


# all
res = pd.DataFrame({'model':['linUCRL','oracle \noptimal','oracle \ngreedy','linUCB'],
                    'reward':[3.54,3.555,3.536,3.43],
                 'color':colors[1:5]})
res = res.sort_values(by='reward')

m = (int(res.reward.min()*10)-1)/10
M = (int(res.reward.max()*10)+1)/10

plt.figure(figsize=(8, 8))
plt.bar(range(4),
        res.reward,
        color=res.color,width=1)
plt.xticks([0,1,2,3],res.model,fontsize=15)
for x in range(4):
    plt.annotate(xy=(x-0.3,m+0.2),s=round(list(res.reward)[x],3),fontsize=20)
plt.ylim((m,M))
plt.savefig(path+'ML_rewards_all.pdf',bbox_inches='tight')


# =============================================================================
# pulled arms
# =============================================================================
arm_label = ["Action","Comedy","Adventure","Thriller","Drama","Children","Crime",#"Documentary","Romance",
             "Horror",#"Musical",
             "SciFi",#"War",
             "Animation"]#,"FilmNoir","Mystery","Western"]

K = len(arm_label)

optimal_arms = list(np.tile([9,5,9],14))[:40]
greedyOracle_arms = list(np.tile([2,5,7,2,5,2,5],6))[:40]
linUCRL_arms = list(np.tile([4,7,4],14))[:40]
linUCB_arms = [2,4,2,5,4,4,4,7,4,7,8,8,7,7,7,8,6,8,6,6,6,0,0,5,5,0,7,5,7,4,4,4,4,3,4,4,4,7,4,4]

fig = plt.figure(figsize=(8, 8))
sub2 = plt.subplot(4, 1, 2)
sub2.plot(range(40),[x for x in greedyOracle_arms],'o-',color=colors[3],linewidth=3,
          markersize=10,label='oracle greedy')
plt.yticks(range(K),arm_label)
plt.ylim((-0.5,9.5))
lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
sub4 = plt.subplot(4, 1, 3)
sub4.plot(range(40),[x for x in linUCRL_arms],'o-',color=colors[1],linewidth=3,
          markersize=10,label='linUCRL')
plt.yticks(range(K),arm_label)
plt.ylim((-0.5,9.5))
lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
sub5 = plt.subplot(4, 1, 4)
sub5.plot(range(40),[x for x in linUCB_arms],'o-',color=colors[4],linewidth=3,
          markersize=10,label='linUCB')
plt.yticks(range(K),arm_label)
plt.ylim((-0.5,9.5))
lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
sub1 = plt.subplot(4, 1, 1)
sub1.plot(range(40),[x for x in optimal_arms],'o-',color=colors[2],linewidth=3,
          markersize=10,label='oracle optimal')
plt.yticks(range(K),arm_label)
plt.ylim((-0.5,9.5))
lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#fig.tight_layout()
plt.savefig(path+'ML_pulled_arms.pdf',bbox_extra_artists=(lgd,), bbox_inches='tight')
