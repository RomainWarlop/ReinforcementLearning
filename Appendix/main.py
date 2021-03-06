import pandas as pd
import matplotlib.pyplot as plt

path = '/home/romain/Documents/BitBucket/PhD/linucrl/Appendix/'

#data = pd.read_csv(path+'criteo-perfByFreq.csv')
data = pd.read_csv(path+'criteo-perfByFreq-userFreq50.csv')
#data = pd.read_csv(path+'criteo-truncperfByFreq-userFreq50.csv')
#data['freqA']/=100

data = data.fillna(0)
data = data.sort_values('freqA')

plt.figure(figsize=(8,6))
plt.plot(data['freqA'],data['mn_rewardA'],'b-',label='version A',linewidth=2)
plt.plot(data['freqA'],data['mn_rewardB'],'r-',label='version B',linewidth=2)
plt.xlabel('displayed frequency of version A',fontsize=15)
plt.title('Mean click rate as a function of A displayed frequency',fontsize=15)
plt.legend(loc='best')
plt.savefig(path+'criteo-truncperfByFreq-userFreq50.png')

zoomLowA = data['freqA']<0.2
plt.figure(figsize=(8,6))
plt.plot(data.loc[zoomLowA,'freqA'],data.loc[zoomLowA,'mn_rewardA'],'b-',label='version A',linewidth=2)
plt.plot(data.loc[zoomLowA,'freqA'],data.loc[zoomLowA,'mn_rewardB'],'r-',label='version B',linewidth=2)
plt.xlabel('displayed frequency of version A',fontsize=15)
plt.title('Mean click rate as a function of A displayed frequency',fontsize=15)
plt.legend(loc='best')
plt.savefig(path+'criteo-truncperfByFreq-userFreq50-zoomLowA.png')

zoomLowB = data['freqA']>0.8
plt.figure(figsize=(8,6))
plt.plot(data.loc[zoomLowB,'freqA'],data.loc[zoomLowB,'mn_rewardA'],'b-',label='version A',linewidth=2)
plt.plot(data.loc[zoomLowB,'freqA'],data.loc[zoomLowB,'mn_rewardB'],'r-',label='version B',linewidth=2)
plt.xlabel('displayed frequency of version A',fontsize=15)
plt.title('Mean click rate as a function of A displayed frequency',fontsize=15)
plt.legend(loc='best')
plt.savefig(path+'criteo-truncperfByFreq-userFreq50-zoomLowB.png')
