import pandas as pd
import matplotlib.pyplot as plt

red55 = [182/255,25/255,36/255]
blue55 = [83/255,173/255,180/255]

path = "/home/romain/Bureau/ongoingImages/"

res = pd.DataFrame({'model':['only B','UCRL','linUCRL','oracle \noptimal','oracle \ngreedy'],
                    'reward':[46.0,46.5,66.7,95.2,61.3],
                 'color':colors[:5]})
res = res.sort_values(by='reward')

m = (int(res.reward.min()*10)*0.9)/10
M = (int(res.reward.max()*10)*1.1)/10

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
    plt.annotate(xy=(x-0.4,(m+M)/2),s=str(round(list(res.reward)[x],1))+'%',fontsize=20)
plt.ylim((m,M))
plt.savefig(path+'criteo_improvement_globalaverage.pdf',bbox_inches='tight')

# =============================================================================
# last
# =============================================================================
res = pd.DataFrame({'model':['only B','UCRL','linUCRL','oracle \noptimal','oracle \ngreedy'],
                    'reward':[46.0,46.0,75.8,95.2,61.3],
                 'color':colors[:5]})
res = res.sort_values(by='reward')

m = (int(res.reward.min()*10)*0.9)/10
M = (int(res.reward.max()*10)*1.1)/10

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
    plt.annotate(xy=(x-0.4,(m+M)/2),s=str(round(list(res.reward)[x],1))+'%',fontsize=20)
plt.ylim((m,M))
plt.savefig(path+'criteo_improvement_last.pdf',bbox_inches='tight')