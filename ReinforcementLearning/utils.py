import numpy as np
from random import shuffle
from sklearn.preprocessing import MinMaxScaler

# Parameters
nStates 	= 100
nActions 	= 10
rewards 	= np.random.normal(size=(nStates,nActions))
transitions = np.random.normal(size=(nStates,nStates,nActions)) # P(s'|s,a), to scale s.t. sum_s' P(s'|s,a) = 1
gamma 		= 1.0
n_iter 		= 50

class MDP(object):

	def __init__(self,nStates,nActions,rewards,transitions,gamma):
		
		# make it a proba for sure !
		for a in range(nActions):
			scaler = MinMaxScaler()
			tmp = transitions[:,:,a].T
			tmp = scaler.fit_transform(tmp)
			tmp = tmp.T
			tmp /= tmp.sum(axis=0)
			transitions[:,:,a] = tmp

		self.nStates = nStates
		self.nActions = nActions
		self.rewards = rewards
		self.transitions = transitions
		self.gamma = gamma

		self.V = np.zeros(nStates)
		self.policy = np.random.choice(range(nActions),nStates)


	def Bellman_a(self,a,V,gamma,state):
	    """
	    Bellman update for a particular action,state
	    """
	    # update for one state
	    r = self.rewards[state,a]
	    next_state = np.random.choice(range(nStates),p=self.transitions[:,state,a])
	    r += self.gamma*self.V[next_state]
	    return r

	def ValueIteration(self,n_iter)
		# Value Iteration
		for it in range(n_iter):
		    states = list(range(self.nStates))
		    shuffle(states)
		    for state in states:
		        self.V[state] = max(list(map(lambda a: self.Bellman_a(a,self.V,self.gamma,state,self.rewards,self.transitions),range(self.nActions))))

	def updateGreedyPolicy(self)
		# Compute policy using value function
		for state in range(nStates):
		    tmp = np.zeros(nActions)
		    for a in range(nActions):
		        r = rewards[state,a]
		        next_state = np.random.choice(range(nStates),p=P[:,state,a])
		        r += gamma*V[next_state]
		        tmp[a] = r
		    self.policy[state] = np.argmax(tmp)

print(policy)