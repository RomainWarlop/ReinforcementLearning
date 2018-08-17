import numpy as np
from itertools import product as itp
from copy import deepcopy
import ast
from random import shuffle

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
class linUCB(object):
    
    def __init__(self,alpha,d,s0,actions,reward,transition):
        self.alpha = alpha
        self.d = d # dimension -1
        l_w = 1+np.log(len(s0))
        self.L = np.sqrt((1-l_w**(d+1))/(1-l_w))
        self.actions = actions
        self.round = 0
        self.reward = reward # true reward function
        self.phi = dict.fromkeys(actions) # reward parameters
        self.R = dict.fromkeys(actions) # reward sequence for each arm
        self.X = dict.fromkeys(actions) # observed context
        self.N = dict.fromkeys(actions) # time spend in a since begin
        self.t = 0
        self.reward_sequence = []
        self.arm_sequence = []
        self.transition = transition # deterministic transition
        
        for a in actions:
            self.R[a] = []
            self.X[a] = []
            self.N[a] = 0
            self.phi[a] = np.array([1.]+list(np.zeros(self.d))).reshape(1,self.d+1)

        # choose first state at random
        self.state = ast.literal_eval(s0)
    
    def conf_r(self,a,lbda,x):
        conf = np.log(1+self.t*self.L**2/lbda)
        conf = np.sqrt((self.d+1)*conf)+np.sqrt(lbda)
        conf *= np.sqrt(x.dot(np.linalg.solve(self.V[a],x.T)))
        #conf = min(1,conf)
        return 0.5*conf
    

    def run(self,T=1000,lbda=0.5):
        self.V = dict.fromkeys(self.actions)
        for a in self.actions:
            self.V[a] = lbda*np.eye(self.d+1)
            
        while self.t<T:
            # choose next action
            r_s = []
            state = deepcopy(self.state)
            for a in self.actions:
                tmp = np.sum([decay(delta+1) for delta in indexes(state,a)]) # compute decay
                x = np.array([tmp**j for j in range(self.d+1)]).reshape(1,self.d+1)
                r_s.append(self.phi[a].dot(x.T)+self.conf_r(a,lbda,x))
            
            a = np.argmax(r_s)
            self.N[a] += 1
            self.arm_sequence.append(a)
            
            # Observe reward
            r = self.reward(self.state,a)
            self.R[a].append(r)
            self.reward_sequence.append(r)
            
            # update V, X
            tmp = np.sum([decay(delta+1) for delta in indexes(state,a)]) # compute decay
            x = np.array([tmp**j for j in range(self.d+1)]).reshape(1,self.d+1)
            self.V[a] += x.T.dot(x)
            self.X[a].append(x)
            XR = np.array(self.X[a]).T.dot(np.array(self.R[a]))
            self.phi[a] = np.linalg.solve(self.V[a],XR).reshape(1,self.d+1)
            
            self.state = self.transition(self.state,a) # next state
            
            self.t+=1