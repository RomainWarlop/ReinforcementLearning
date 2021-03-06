# =============================================================================
# updates
# change transition function to operate directly on string and avoid calling ast.literal_eval
# =============================================================================

import numpy as np
from itertools import product as itp
from copy import deepcopy
import ast
from random import shuffle
import time
import pandas as pd

def indexes(l,elt):
    indices = [i for i, x in enumerate(l) if x == elt]
    return indices

def decay(delta):
    if delta==0:
        out = 0
    else:
        out = 1/delta
    return out


# UCRL in the DMDP case
class UCRL(object):
    
    def __init__(self,alpha,states,s0,actions,reward,transition):
        self.alpha = alpha
        self.states = states
        self.actions = actions
        self.cardS = len(states)
        self.cardA = len(actions)
        self.round = 0
        self.reward = reward # true reward function
        self.transition = transition # deterministic transition
        self.policy = dict.fromkeys(states)
        self.conf = pd.DataFrame(columns=actions,index=states).fillna(0)
        self.t = 0
        self.reward_sequence = []
        self.arm_sequence = []

        self.N = pd.DataFrame(columns=actions,index=states).fillna(0)
        self.R = pd.DataFrame(columns=actions,index=states).fillna(0)
        self.r_hat = pd.DataFrame(columns=actions,index=states).fillna(0)
        for s in states:
            self.policy[s] = np.random.randint(0,self.cardA)

        # choose first state at random
        self.state = s0

    def conf_r(self,t,s,a,conf):
        if self.N.loc[s,a]>0:
           conf /= 2*self.N.loc[s,a]
        else:
            conf = 10
        conf = min(1,np.sqrt(conf))
        return conf
    
    def Bellman_a(self,a,V,gamma,str_state,rewards):
        # update for one state
        r = rewards[str_state][a]+self.conf[str_state][a] # UCB
        #next_state = self.transition(ast.literal_eval(state),a)
        #r+=gamma*V[str(next_state)]
        next_state = self.transition(str_state,a)
        r+=gamma*V[next_state]
        return r
        
    def policyUpdate(self,rewards):
        t0 = time.time()
        V = dict.fromkeys(self.states)
        for str_state in self.states:
            V[str_state] = 0
        
        for it in range(self.n_iter):
            shuffle(self.states)
            for str_state in self.states:
                V[str_state] = max(list(map(lambda a: self.Bellman_a(a,V,self.gamma,str_state,rewards),range(self.cardA))))
        
        for str_state in self.states:
            tmp = np.zeros(self.cardA)
            for a in range(self.cardA):
                r = rewards[str_state][a]+self.conf[str_state][a] # UCB
                next_state = self.transition(str_state,a)
                r+=self.gamma*V[next_state]
                tmp[a] = r
            self.policy[str_state] = np.argmax(tmp)
        t1 = time.time()
        tot = t1-t0
        print("Policy update time %f seconds" % tot)

    def run(self,T=1000,gamma=1,n_iter=100):
        self.gamma = gamma
        self.update_time = []
        self.n_iter = n_iter
        while self.t<T:
            print("iteration",self.t)
            # choose next action
            a = self.policy[self.state]
            self.arm_sequence.append(a)
            # Observe reward
            r = self.reward(ast.literal_eval(self.state),a)
            self.reward_sequence.append(r)
            # update      
            self.N.loc[self.state,a]+=1
            self.R[self.state,a]+=r
            self.state = self.transition(self.state,a) # next state
            update_policy = False
            _conf = np.log(2*(self.t+1**self.alpha)*self.cardS*self.cardA)
            new_conf = deepcopy(self.conf)
            
            for s_a in itp(*[self.states,self.actions]):
                s, a = s_a
                new_conf[s][a] = self.conf_r(self.t+1,s,a,_conf)
                if new_conf[s][a] < self.conf[s][a]/2:
                    update_policy = True
                if self.N[s][a]>0:
                    self.r_hat[s][a] = self.R[s][a]/self.N[s][a]
            
            # update policy ?
            if update_policy:
                self.round+=1
                self.update_time.append(self.t+1)
                print('update policy, round=',self.round)
                self.conf = deepcopy(new_conf)
                self.policyUpdate(self.r_hat)
            
            self.t+=1

class linUCRL(object):
    
    def __init__(self,alpha,d,states,s0,actions,reward,transition):
        self.alpha = alpha
        self.d = d # dimension -1
        l_w = 1+np.log(len(states[0]))
        self.L = np.sqrt((1-l_w**(d+1))/(1-l_w))
        self.states = states
        self.actions = actions
        self.cardS = len(states)
        self.cardA = len(actions)
        self.round = 0
        self.reward = reward # true reward function
        self.transition = transition # deterministic transition
        self.policy = dict.fromkeys(states)
        self.phi = dict.fromkeys(actions) # reward parameters
        self.conf = dict.fromkeys(actions)
        self.N = dict.fromkeys(actions) # time spend in a since begin
        self.roundN = dict.fromkeys(actions) # time spend in a in current round
        self.R = dict.fromkeys(actions) # reward sequence for each arm
        self.X = dict.fromkeys(actions) # observed context
        self.t = 0
        self.reward_sequence = []
        self.arm_sequence = []
        
        for s in states:
            self.policy[s] = np.random.randint(0,self.cardA)

        for a in actions:
            self.R[a] = []
            self.X[a] = []
            self.N[a] = 0
            self.roundN[a] = 0
            self.conf[a] = 1
            self.phi[a] = np.array([1.]+list(np.zeros(self.d))).reshape(1,self.d+1)

        # choose first state at random
        self.state = s0
    
    def conf_r(self,t,a,lbda):
        conf = np.log(self.cardA*t**self.alpha*(1+self.N[a]*self.L**2/lbda))
        conf = np.sqrt((self.d+1)*conf)+np.sqrt(lbda)
        #conf = min(1,conf)
        return 0.01*conf
    
    def Bellman_a(self,a,V,gamma,state,rewards):
        # update for one state
        r = rewards[state][a]
        next_state = self.transition(state,a)
        r+=gamma*V[str(next_state)]
        return r
        
    def policyUpdate(self):
        t0 = time.time()
        
        V = dict.fromkeys(self.states)
        for state in self.states:
            V[state] = 0
        
        # compute rewards once
        _rewards = dict.fromkeys(self.states)
        for str_state in self.states:
            _rewards[str_state] = dict.fromkeys(list(range(self.cardA)))
            for a in range(self.cardA):
                state = ast.literal_eval(str_state)
                tmp = np.sum([decay(delta+1) for delta in indexes(state,a)]) # compute decay
                x = np.array([tmp**j for j in range(self.d+1)]).reshape(1,self.d+1)
                r = self.phi[a].dot(x.T)+self.conf[a]*np.sqrt(x.dot(np.linalg.solve(self.V[a],x.T))) # UCB
                r = r[0][0]
                _rewards[str_state][a] = r
        
        for it in range(self.n_iter):
            shuffle(self.states)
            for str_state in self.states:
                V[str_state] = max(list(map(lambda a: self.Bellman_a(a,V,self.gamma,str_state,_rewards),range(self.cardA))))
        
        for str_state in self.states:
            state = ast.literal_eval(str_state)
            r_arms = np.zeros(self.cardA)
            for a in range(self.cardA):
                tmp = np.sum([decay(delta+1) for delta in indexes(state,a)]) # compute decay
                x = np.array([tmp**j for j in range(self.d+1)]).reshape(1,self.d+1)
                r = self.phi[a].dot(x.T)+self.conf[a]*np.sqrt(x.dot(np.linalg.solve(self.V[a],x.T))) # UCB
                r = r[0][0]
                next_state = self.transition(str_state,a)
                r+=self.gamma*V[next_state]
                r_arms[a] = r
            self.policy[str_state] = np.argmax(r_arms)
        t1 = time.time()
        tot = t1-t0
        print("Policy update time %f seconds" % tot)
    
    def run(self,T=1000,lbda=0.5,gamma=1,n_iter=100):
        self.V = dict.fromkeys(self.actions)
        for a in self.actions:
            self.V[a] = lbda*np.eye(self.d+1)
        
        self.gamma = gamma
        self.update_time = []
        self.n_iter = n_iter
        while self.t<T:
            # choose next action
            a = self.policy[self.state]
            self.arm_sequence.append(a)
            
            # Observe reward
            r = self.reward(ast.literal_eval(self.state),a)
            self.R[a].append(r)
            self.reward_sequence.append(r)
            
            # update V, X
            state = ast.literal_eval(self.state)
            tmp = np.sum([decay(delta+1) for delta in indexes(state,a)]) # compute decay
            x = np.array([tmp**j for j in range(self.d+1)]).reshape(1,self.d+1)
            self.V[a] += x.T.dot(x)
            self.X[a].append(x[0])
            
            # update      
            self.roundN[a]+=1
            self.state = self.transition(self.state,a) # next state
            update_policy = False
            for a in self.actions:
                if self.roundN[a] >= self.N[a]:
                    update_policy = True
            
            # update policy ?
            if update_policy:
                self.round+=1
                for a in self.actions:
                    self.N[a] += self.roundN[a]
                    self.roundN[a] = 0
                    XR = np.array(self.X[a]).T.dot(np.array(self.R[a]))
                    try:
                        self.phi[a] = np.linalg.solve(self.V[a],XR).reshape(1,self.d+1)
                    except:
                        pass # arm still not pulled
                    self.conf[a] = self.conf_r(self.t+1,a,lbda)
                self.update_time.append(self.t+1)
                print('update policy, round=',self.round)
                self.policyUpdate()
            
            self.t+=1
            
