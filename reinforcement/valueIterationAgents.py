# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        self.actions = dict()
        for state in self.mdp.getStates():
          state_actions = self.mdp.getPossibleActions(state)
          if len(state_actions) > 0:
            action = state_actions[0]
            self.actions[state] = action
            
        for i in range(self.iterations):
          next = self.values.copy()
          for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
              
              action = self.getPolicy(state)
              self.actions[state] = action
              next[state] = self.getQValue(state, action)
          self.values.update(next)
          


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        total = 0
        for new_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):
          total += prob * (self.mdp.getReward(state,action,new_state) + (self.discount * self.values[new_state]))
        return total
        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if  self.mdp.isTerminal(state):
          return  None
        else :
          actions =  self.mdp.getPossibleActions(state)
          max_value =  self.getQValue(state, actions[0])
          max_action = actions[0]
          
          for action in actions:
            value =  self.getQValue (state, action)
            if max_value <= value:
              max_value = value
              max_action = action
              
          return max_action
        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        current_states = self.mdp.getStates()
        for iteration in range(self.iterations):
          workingState = current_states[iteration % len(current_states)] # Don't want OOB error
          if not self.mdp.isTerminal(workingState):
            q_v = []
            for action in self.mdp.getPossibleActions(workingState):
              q_v.append(self.computeQValueFromValues(workingState, action))
            self.values[workingState] = max(q_v)
        

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        predecessors = {}
        for workingState in self.mdp.getStates():
          if not self.mdp.isTerminal(workingState):
            for action in self.mdp.getPossibleActions(workingState):
              for new_state, problem in self.mdp.getTransitionStatesAndProbs(workingState, action):
                if new_state in predecessors:
                  predecessors[new_state].add(workingState)
                else:
                  predecessors[new_state] = {workingState}
                  
        queue = util.PriorityQueue()
      
        for s in self.mdp.getStates():
            if not self.mdp.isTerminal(s):
                q_v = []
                for action in self.mdp.getPossibleActions(s):
                    q_v.append(self.computeQValueFromValues(s, action))
                diff = abs(max(q_v) - self.values[s])
                queue.update(s, -diff)
          
        for iteration in range(self.iterations):
            if queue.isEmpty():
                break

            s = queue.pop()
            if not self.mdp.isTerminal(s):
                q_v = []
                for action in self.mdp.getPossibleActions(s):
                    q_v.append(self.computeQValueFromValues(s, action))
                self.values[s] = max(q_v)
              
            for p in predecessors[s]:
                if not self.mdp.isTerminal(p):
                    q_v = []
                    for action in self.mdp.getPossibleActions(p):
                        q_v.append(self.computeQValueFromValues(p, action))
                    diff = abs(max(q_v) - self.values[p])
                  
                    if diff > self.theta:
                        queue.update(p, -diff)

      