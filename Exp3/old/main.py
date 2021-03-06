#==============================================================================
# Exp 3 of "A Linear Reinforcement Learning Algorithm For Non Stationary Arms"
# The click rate are estimated using the Criteo AB test data and then 
# parameters are learned to fit the click. The goal is to check if the 
# learn click probability are good enough to find a better display strategy
#==============================================================================


import numpy as np
import pandas as pd
from copy import deepcopy
from itertools import product as itp
import ast
from random import shuffle
import matplotlib.pyplot as plt

nRepeat = 5

oracleImprovement = np.zeros((nRepeat,))
linUCRLImprovement = np.zeros((nRepeat,))
epsGreedyImprovement = np.zeros((nRepeat,))
randomImprovement = np.zeros((nRepeat,))

input_path = '/home/romain/Documents/BitBucket/PhD/linucrl/Exp3/'

window = 10   
fromBQ = True
for repeat_ in range(nRepeat):
    print(repeat_)
    #==============================================================================
    # check if when fitting a line, the model make the "right" choice
    #==============================================================================
    
    if fromBQ:
        lag = """SELECT user_id, timestamp, IF(version='A',0,1) bin_version, reward"""
            
        for i in range(1,window+1):
            lag += ", LAG(IF(version='A','0','1'),"+str(i)+") OVER (PARTITION BY user_id ORDER BY timestamp ASC) as version_"+str(i)
        
        lag += """ FROM (TABLE_QUERY([data-science-55:criteo],'table_id CONTAINS "frequent_user_50_details_0"'))"""
        
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
        
        #data.to_csv(input_path+'mean_reward_per_state_w'+str(window)+'.csv',index=False)
    
    else:
        data = pd.read_csv(input_path+'mean_reward_per_state_w'+str(window)+'.csv')
        
    data.loc[data.index,'best_choice'] = list(map(lambda i: 'A' if data.loc[i,'mn_rewardA']>data.loc[i,'mn_rewardB'] else 'B',data.index) )
    data.index = data['state']
    
    mnA = 0.09166072904895288 # computed from user that have a frequency of A of 1
    mnB = 0.09285164248577829 # computed from user that have a frequency of A of 0
    K = 2
    
    S = list(itp(range(K), repeat=window)) # all possibles states
    S = list(map(lambda s: str(list(s)),S))
    
    def next_state_(x,a):
        # x[j] contains the arm that have been played j+1 timesteps ago
        y = deepcopy(x)
        y = y[:-1]
        y = [a] + y
        return y
    
    def Bellman_a(a,gamma,state):
        # update for one state at random
        if a==0:
            r = deepcopy(data.loc[state,'mn_rewardA'])
        else:
            r = deepcopy(data.loc[state,'mn_rewardB'])
        
        state = ast.literal_eval(state)        
        next_state = next_state_(state,a)
        r+=gamma*V[str(next_state)]
        return r
    
    gamma = .99
    n_iter = 100
    #==============================================================================
    #     Oracle
    #==============================================================================
    V = {}
    newV = {}
    for state in S:
        V[state] = 0
        newV[state] = 0
    
    for it in range(n_iter):
        #shuffle(S)
        for state in S:
            newV[state] = max(list(map(lambda a: Bellman_a(a,gamma,state),range(K))))
        V = deepcopy(newV)
        
    policy = {}
    for state in V.keys():
        tmp = np.zeros(K)
        for a in range(K):
            if a==0:
                r = deepcopy(data.loc[state,'mn_rewardA'])
            else:
                r = deepcopy(data.loc[state,'mn_rewardB'])
            next_state = next_state_(ast.literal_eval(state),a)
            r+=gamma*V[str(next_state)]
            tmp[a] = r
        policy[str(state)] = np.argmax(tmp)
        
    # find the loop
    oracle_cyclePerf = np.zeros(len(S))
    
    #starts = np.random.choice(range(len(S)),10,False)
    for i in range(len(S)):
        #print(i)
        state = list(V.keys())[i]
        path = [state]
        actions = [policy[str(state)]]
        next_state = next_state_(ast.literal_eval(state),actions[0])
        while str(next_state) not in path:
            path.append(str(next_state))
            _state = next_state
            a = policy[str(_state)]
            actions.extend([a])
            next_state = next_state_(_state,a)
        first_visit = path.index(str(next_state))
        
        cycle = path[first_visit:]
        
        for state in cycle:
            a = policy[state]
            if a==0:
                r = deepcopy(data.loc[state,'mn_rewardA'])
            else:
                r = deepcopy(data.loc[state,'mn_rewardB'])
            oracle_cyclePerf[i] += r
    
        oracle_cyclePerf[i] /= len(cycle)
    print('*'*10,'Oracle','*'*10)
    print('mean cycle perf:',np.mean(oracle_cyclePerf))
    print('mean relative improvement over A:',(np.mean(oracle_cyclePerf)-mnA)/mnA*100)
    print('mean relative improvement over B:',(np.mean(oracle_cyclePerf)-mnB)/mnB*100)
    print('-'*20)
    
    #==============================================================================
    #     Model 
    #==============================================================================
    def weight(state,v,d):
        state = ast.literal_eval(state)
        indices = [1/(i+1) for i, x in enumerate(state) if x == v]
        tmp = np.sum(indices) # compute decay
        x = np.array([tmp**j for j in range(d+1)]).reshape(1,d+1)
        return x
    
    def prediction(index,d,version):
        state = data.loc[index,'state']
        if version=='A':
            xA = weight(state,0,d)
            p = w['A'].dot(xA.T)
        else:
            xB = weight(state,1,d)
            p = w['B'].dot(xB.T)
        return p
    
    D = 10
        
    data['weight_A'] = list(map(lambda x: weight(x,0,D),data.index))
    data['weight_B'] = list(map(lambda x: weight(x,1,D),data.index))
    X_A = np.vstack(data['weight_A'].tolist())
    X_B = np.vstack(data['weight_B'].tolist())
    
    w_A = np.linalg.solve(X_A.T.dot(X_A),X_A.T.dot(data['mn_rewardA']))
    w_B = np.linalg.solve(X_B.T.dot(X_B),X_B.T.dot(data['mn_rewardB']))
    
    w = {'A':None,'B':None}
    w['A'] = w_A.reshape(1,D+1)
    w['B'] = w_B.reshape(1,D+1)
    
    data.loc[data.index,'rhat_A'] = list(map(lambda i: prediction(i,D,'A'),data.index))
    data.loc[data.index,'rhat_B'] = list(map(lambda i: prediction(i,D,'B'),data.index))
    
    def Bellman_hat_a(a,gamma,state):
        if a==0:
            r = deepcopy(data.loc[state,'rhat_A'])
        else:
            r = deepcopy(data.loc[state,'rhat_B'])
        
        state = ast.literal_eval(state)        
        next_state = next_state_(state,a)
        r+=gamma*V[str(next_state)]
        return r
    
    V = {}
    newV = {}
    for state in S:
        V[state] = 0
        newV[state] = 0
    
    for it in range(n_iter):
        shuffle(S)
        for state in S:
            V[state] = max(list(map(lambda a: Bellman_hat_a(a,gamma,state),range(K))))
        #V = deepcopy(newV)
    
    policy = {}
    for state in V.keys():
        tmp = np.zeros(K)
        for a in range(K):
            if a==0:
                r = deepcopy(data.loc[state,'rhat_A']) # choose with approx reward
            else:
                r = deepcopy(data.loc[state,'rhat_B']) # choose with approx reward
            next_state = next_state_(ast.literal_eval(state),a)
            r+=gamma*V[str(next_state)]
            tmp[a] = r
        policy[str(state)] = np.argmax(tmp)
        
    # find the loop
    approx_cyclePerf = np.zeros(len(S))
    #starts = np.random.choice(range(len(S)),10,False)
    for i in range(len(S)):
        #print(i)
        state = list(V.keys())[i]
        path = [state]
        actions = [policy[str(state)]]
        next_state = next_state_(ast.literal_eval(state),actions[0])
        while str(next_state) not in path:
            path.append(str(next_state))
            _state = next_state
            a = policy[str(_state)]
            actions.extend([a])
            next_state = next_state_(_state,a)
        first_visit = path.index(str(next_state))
        
        cycle = path[first_visit:]
        
        for state in cycle:
            a = policy[state]
            if a==0:
                r = deepcopy(data.loc[state,'mn_rewardA']) # receive the true reward
            else:
                r = deepcopy(data.loc[state,'mn_rewardB']) # receive the true reward
            approx_cyclePerf[i] += r
        approx_cyclePerf[i] /= len(cycle)
    print('linUCRL')
    print('mean cycle perf:',np.mean(approx_cyclePerf))
    print('mean relative improvement over A:',(np.mean(approx_cyclePerf)-mnA)/mnA*100)
    print('mean relative improvement over B:',(np.mean(approx_cyclePerf)-mnB)/mnB*100)
    print('-'*20)
    
    #==============================================================================
    # epsilon greedy model
    #==============================================================================
    eps = 1/100
    
    tmp = []
    for _it in range(10):
        rewards_epsgreedy_estimated = []
        state = np.random.choice(S,1)[0]    
        actions = []
        for t in range(1000):
            if np.random.random()<eps: # choose between A and B at random
                if np.random.random()<0.5: # then A
                    r = data.loc[state,'mn_rewardA']
                    action = 0
                else:
                    r = data.loc[state,'mn_rewardB']
                    action = 1
            else:
                if data.loc[state,'rhat_A']>data.loc[state,'rhat_B']:
                    action = 0
                    r = data.loc[state,'mn_rewardA']
                else:
                    action = 1
                    r = data.loc[state,'mn_rewardB']
            
            actions.append(action)
            rewards_epsgreedy_estimated.append(r)
            state = str(next_state_(ast.literal_eval(state),action))
        
        meanrewards_epsgreedy_estimated = np.mean(rewards_epsgreedy_estimated)
        tmp.append(meanrewards_epsgreedy_estimated)
    
    meanrewards_epsgreedy_estimated = np.mean(tmp)
    print('mean reward of epsilon greedy on estimated reward:',meanrewards_epsgreedy_estimated)
    print('mean relative improvement over A:',(meanrewards_epsgreedy_estimated-mnA)/mnA*100)
    print('mean relative improvement over B:',(meanrewards_epsgreedy_estimated-mnB)/mnB*100)
    print('-'*20)
    
    #==============================================================================
    # random
    #==============================================================================
    random = ((data['nA']*data['mn_rewardA']).sum()+(data['nB']*data['mn_rewardB']).sum())/(data['nA']+data['nB']).sum()
    print('mean reward of random:',random)
    print('mean relative improvement over A:',(random-mnA)/mnA*100)
    print('mean relative improvement over B:',(random-mnB)/mnB*100)
    print('-'*20)
    
    #==============================================================================
    # plot
    #==============================================================================
    oracleImprovement[repeat_] = np.mean(oracle_cyclePerf)
    linUCRLImprovement[repeat_] = np.mean(approx_cyclePerf)
    epsGreedyImprovement[repeat_] = meanrewards_epsgreedy_estimated
    randomImprovement[repeat_] = random

perf = [np.mean(epsGreedyImprovement),
        np.mean(randomImprovement),
        np.mean(linUCRLImprovement),
        np.mean(oracleImprovement)]

conf = 1.96*np.array([np.std(epsGreedyImprovement),
        np.std(randomImprovement),
        np.std(linUCRLImprovement),
        np.std(oracleImprovement)])/np.sqrt(nRepeat)

print('#'*20)
print('#'*20)
print('Global results')
print('oracle over A:',(perf[3]-mnA)/mnA*100,'+/-',conf[3]/mnA*100)
print('linUCRL over A:',(perf[2]-mnA)/mnA*100,'+/-',conf[2]/mnA*100)
print('random over A:',(perf[1]-mnA)/mnA*100,'+/-',conf[1]/mnA*100)
print('eps greedy over A:',(perf[0]-mnA)/mnA*100,'+/-',conf[0]/mnA*100)

print('oracle over B:',(perf[3]-mnB)/mnB*100,'+/-',conf[3]/mnB*100)
print('linUCRL over B:',(perf[2]-mnB)/mnB*100,'+/-',conf[2]/mnB*100)
print('random over B:',(perf[1]-mnB)/mnB*100,'+/-',conf[1]/mnB*100)
print('eps greedy over B:',(perf[0]-mnB)/mnB*100,'+/-',conf[0]/mnB*100)

improvement = (np.array(perf)-mnB)/mnB*100

plt.figure(figsize=(8, 8))
plt.bar(range(4),improvement,yerr=conf/mnB*100,color=['blue','green','hotpink','darkcyan'],
        width=1,ecolor='black')
plt.xticks([0.5,1.5,2.5,3.5],['$\epsilon$-greedy \n(1%)','random','linUCRL','oracle'],fontsize=20)
for x in range(4):
    plt.annotate(xy=(x+0.2,6),s=str(round(improvement[x],2))+'%',fontsize=20)
    #if x!=1:
    #    plt.annotate(xy=(x+0.2,5.5),s='+/-'+str(round(conf[x]/mnB*100,2))+'%',fontsize=15)
plt.ylim((0,max(improvement)+1))
#plt.title('Relative improvement over the best single action',fontsize=15)
#plt.savefig(input_path+'t_300iters_relative_improvement_d='+str(D)+'.png',bbox_inches='tight')
