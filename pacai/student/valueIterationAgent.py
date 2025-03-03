import selectors
from pacai.agents.learning.value import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
    A value iteration agent.

    Make sure to read `pacai.agents.learning` before working on this class.

    A `ValueIterationAgent` takes a `pacai.core.mdp.MarkovDecisionProcess` on initialization,
    and runs value iteration for a given number of iterations using the supplied discount factor.

    Some useful mdp methods you will use:
    `pacai.core.mdp.MarkovDecisionProcess.getStates`,
    `pacai.core.mdp.MarkovDecisionProcess.getPossibleActions`,
    `pacai.core.mdp.MarkovDecisionProcess.getTransitionStatesAndProbs`,
    `pacai.core.mdp.MarkovDecisionProcess.getReward`.

    Additional methods to implement:

    `pacai.agents.learning.value.ValueEstimationAgent.getQValue`:
    The q-value of the state action pair (after the indicated number of value iteration passes).
    Note that value iteration does not necessarily create this quantity,
    and you may have to derive it on the fly.

    `pacai.agents.learning.value.ValueEstimationAgent.getPolicy`:
    The policy is the best action in the given state
    according to the values computed by value iteration.
    You may break ties any way you see fit.
    Note that if there are no legal actions, which is the case at the terminal state,
    you should return None.
    """

    def __init__(self, index, mdp, discountRate = 0.9, iters = 100, **kwargs):
        super().__init__(index, **kwargs)

        self.mdp = mdp
        self.discountRate = discountRate
        self.iters = iters
        self.values = {}  # A dictionary which holds the q-values for each state.

        # Compute the values here.
        states = self.mdp.getStates()
        for state in states:
            self.values[state] = 0.0  # Initialize all state values to 0

        for _ in range(self.iters):
            newValues = {}
            for state in states:
                possibleActions = self.mdp.getPossibleActions(state)
                if not possibleActions:
                    newValues[state] = self.getValue(state)
                else:
                    maxQValue = float('-inf')
                    for action in possibleActions:
                        qValue = self.getQValue(state, action)
                        if qValue > maxQValue:
                            maxQValue = qValue
                    newValues[state] = maxQValue
            self.values = newValues

    def getValue(self, state):
        """
        Return the value of the state (computed in __init__).
        """

        return self.values.get(state, 0.0)

    def getAction(self, state):
        """
        Returns the policy at the state (no exploration).
        """

        return self.getPolicy(state)
    
    def getQValue(self, state, action):
        """
        The q-value of the state action pair (after the indicated number of value iteration passes).
        Note that value iteration does not necessarily create this quantity,
        and you may have to derive it on the fly.
        """

        qVal = 0
        for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            qVal += prob * (self.mdp.getReward(state, action, nextState) + self.discountRate * self.getValue(nextState))
        return qVal
    
    def getPolicy(self, state):
        """
        The policy is the best action in the given state
        according to the values computed by value iteration.
        You may break ties any way you see fit.
        Note that if there are no legal actions, which is the case at the terminal state,
        you should return None.
        """
        
        actions = self.mdp.getPossibleActions(state)
        if actions:
            bestAction = actions[0]
            for action in actions:
                if self.getQValue(state, action) > self.getQValue(state, bestAction):
                    bestAction = action
        else:
            bestAction = None
        return bestAction
