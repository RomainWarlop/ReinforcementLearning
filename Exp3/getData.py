import pandas as pd

path = '/home/romain/Documents/BitBucket/PhD/linucrl/Exp3/'

window = 10 

lag = """SELECT user_id, timestamp, IF(version='A',0,1) bin_version, reward"""
        
for i in range(1,window+1):
    lag += ", LAG(IF(version='A','0','1'),"+str(i)+") OVER (PARTITION BY user_id ORDER BY timestamp ASC) as version_"+str(i)

lag += """ FROM [data-science-55:criteo.user_frequent_50_50A_0p1A_0p1B]"""

state = "CONCAT('[',"
for i in range(1,window+1):
    state += "version_"+str(i)+",', ',"
state = state[:-5]
state += "']')"
    
output = "SELECT user_id, timestamp, bin_version, reward, "+state+" as state FROM ("+lag+")  HAVING state IS NOT NULL"

output = """SELECT state, count(*) as n, SUM(1-bin_version) nA, 
        SUM(bin_version) nB, SUM(IF(bin_version=0,reward,0)) rewardA,
        SUM(IF(bin_version=1,reward,0)) rewardB FROM ("""+output+""")
        GROUP BY 1"""
data = pd.read_gbq(output,project_id='data-science-55')

data['mn_rewardA'] = data['rewardA']/data['nA']
data['mn_rewardB'] = data['rewardB']/data['nB']

data.to_csv(path+'oracle_rewards.csv',index=False)