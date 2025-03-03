import random

from pacai.agents.base import BaseAgent
from pacai.agents.search.multiagent import MultiAgentSearchAgent
from pacai.core import distance

class ReflexAgent(BaseAgent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.
    You are welcome to change it in any way you see fit,
    so long as you don't touch the method headers.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        `ReflexAgent.getAction` chooses among the best options according to the evaluation function.

        Just like in the previous project, this method takes a
        `pacai.core.gamestate.AbstractGameState` and returns some value from
        `pacai.core.directions.Directions`.
        """

        # Collect legal moves.
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions.
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best.

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current `pacai.bin.pacman.PacmanGameState`
        and an action, and returns a number, where higher numbers are better.
        Make sure to understand the range of different values before you combine them
        in your evaluation function.
        """

        successorGameState = currentGameState.generatePacmanSuccessor(action)

        newScore = successorGameState.getScore()
        newPosition = successorGameState.getPacmanPosition()
        oldFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()

        # Get all ghost positions
        ghostDistances = [ghostState.getPosition() for ghostState in newGhostStates]
        ghostHeuristics = [distance.manhattan(newPosition, ghostDistance)
        for ghostDistance in ghostDistances]

        ghostScore = 0
        for ghostDistance in ghostDistances:
            if newGhostStates[ghostDistances.index(ghostDistance)].getScaredTimer == 0:
                ghostScore -= 50
            else:
                ghostScore = 50

        ghostable = []
        # Iterate through all ghost states
        for newState in newGhostStates:
            ghostPosition = newState.getPosition()

            ghostable.append((ghostPosition[0], ghostPosition[1]))
            ghostable.append((ghostPosition[0], ghostPosition[1] - 1))
            ghostable.append((ghostPosition[0], ghostPosition[1] + 1))
            ghostable.append((ghostPosition[0] - 1, ghostPosition[1]))
            ghostable.append((ghostPosition[0] + 1, ghostPosition[1]))

        # If there is still food left, penalize
        if len(oldFood.asList()) > 0:
            newScore -= distance.manhattan(newPosition, oldFood.asList()[0])
        # If pacman is in danger proximity of ghost
        if newPosition in ghostable:
            newScore -= 750

        newScore += sum(ghostHeuristics) + ghostScore
        
        return newScore


class MinimaxAgent(MultiAgentSearchAgent):
    """
    A minimax agent.

    Here are some method calls that might be useful when implementing minimax.

    `pacai.core.gamestate.AbstractGameState.getNumAgents()`:
    Get the total number of agents in the game

    `pacai.core.gamestate.AbstractGameState.getLegalActions`:
    Returns a list of legal actions for an agent.
    Pacman is always at index 0, and ghosts are >= 1.

    `pacai.core.gamestate.AbstractGameState.generateSuccessor`:
    Get the successor game state after an agent takes an action.

    `pacai.core.directions.Directions.STOP`:
    The stop direction, which is always legal, but you may not want to include in your search.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getNumAgents(self, gameState):
        return gameState.getNumAgents()
    
    def getLegalActions(self, gameState, index):
        return gameState.getLegalActions(index)
    
    def generateSuccessor(self, gameState, legalActions):
        successors = []
        for action in legalActions:
            move = gameState.generatePacmanSuccessor(action)
            successors.append(move)
        return successors

    def getTreeDepth(self):
        return self._treeDepth
    
    def getEvaluationFunction(self, gameState):
        return self._evaluationFunction(gameState)
    
    def miniMax(self, gameState, index, curDepth):
        index = 0
        curDepth += 1

        # check if current depth is equal to the tree depth,
        # if so call eval function on current gamestate
        if curDepth == self.getTreeDepth():
            return self.getEvaluationFunction(gameState)
        
        legalActions = gameState.getLegalActions(index)
        # if there are no more legalActions left call the eval func on current gamestate
        if len(legalActions) == 0:
            return self.getEvaluationFunction(gameState)

        # recursive call to tree
        scores = [(self.miniMax(gameState.generateSuccessor(index, action), index + 1, curDepth))
                  for action in legalActions]
        
        if index == 0:
            return max(scores)
        else:
            return min(scores)

    def getAction(self, gameState):
        legalActions = gameState.getLegalActions(0)

        scores = [(self.miniMax(gameState.generateSuccessor(0, action), 1, 0), action)
                  for action in legalActions]

        scores = []
        for legalAction in gameState.getLegalActions(0):
            result = self.miniMax(gameState.generateSuccessor(0, legalAction), 1, 0)
            scores.append((result, legalAction))
        bestScore = max(scores)[0]
        bestIndices = [score[1] for score in scores if score[0] == bestScore]
        return bestIndices[0]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    A minimax agent with alpha-beta pruning.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

class ExpectimaxAgent(MultiAgentSearchAgent):
    
    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getTreeDepth(self, gameState):
        return self._treeDepth
    
    def getEvaluationFunction(self, gameState):
        return self._evaluationFunction(gameState)
    
    def getNumAgents(self, gameState):
        return gameState.getNumAgents()
    
    def generateSuccessor(self, gameState, legalActions):
        successors = []
        for action in legalActions:
            move = gameState.generatePacmanSuccessor(action)
            successors.append(move)
        return successors
    
    def expectiMax(self, gameState, index, depth):
        if index == self.getNumAgents(gameState):
            index = 0
            depth += 1

            if depth == self.getTreeDepth(gameState):
                return self.getEvaluationFunction(gameState)
        
        scores = []
        legalActions = gameState.getLegalActions(index)
        if len(legalActions) == 0:
            return self.getEvaluationFunction(gameState)
        for move in legalActions:
            result = self.expectiMax(gameState.generateSuccessor(index, move), index + 1, depth)
            scores.append(result)

        if index == 0:
            return max(scores)
        else:
            return sum(scores) / len(scores)
        
    def getAction(self, gameState):
        legalActions = gameState.getLegalActions(0)
        scores = []
        for move in legalActions:
            result = self.expectiMax(gameState.generateSuccessor(0, move), 1, 0)
            scores.append((result, move))

        bestScore = max(scores)[0]
        bestIndices = [score[1] for score in scores if score[0] == bestScore]
        return bestIndices[0]

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable evaluation function.

    DESCRIPTION: <write something here so we know what you did>
    """
    newScore = currentGameState.getScore()
    newPosition = currentGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhost = currentGameState.getGhostStates()
    
    ghostDist = [ghost.getPosition() for ghost in newGhost]
    ghostScore = 0
    for dist in ghostDist:
        if newGhost[ghostDist.index(dist)].getScaredTimer == 0:
            ghostScore -= 10
        else:
            ghostScore = 5

    # if food leftover then penalty
    ghostable = []
    # Iterate through all ghost states
    for newState in newGhost:
        ghostPosition = newState.getPosition()

        ghostable.append((ghostPosition[0], ghostPosition[1]))
        ghostable.append((ghostPosition[0], ghostPosition[1] - 1))
        ghostable.append((ghostPosition[0], ghostPosition[1] + 1))
        ghostable.append((ghostPosition[0] - 1, ghostPosition[1]))
        ghostable.append((ghostPosition[0] + 1, ghostPosition[1]))

    # If there is still food left, penalize
    if len(oldFood.asList()) > 0:
        newScore -= distance.manhattan(newPosition, oldFood.asList()[0])
    # If pacman is in danger proximity of ghost
    if newPosition in ghostable:
        newScore -= 500
        
    return newScore + currentGameState.getScore() + ghostScore

class ContestAgent(MultiAgentSearchAgent):
    """
    Your agent for the mini-contest.

    You can use any method you want and search to any depth you want.
    Just remember that the mini-contest is timed, so you have to trade off speed and computation.

    Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
    just make a beeline straight towards Pacman (or away if they're scared!)

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)