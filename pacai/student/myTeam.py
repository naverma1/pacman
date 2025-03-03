# myTeam.py
"""
An improved Pacman capture team using feature-based ReflexCaptureAgent for
offense and defense, without calling getNumCarrying().
"""

from pacai.agents.capture.reflex import ReflexCaptureAgent
from pacai.core.directions import Directions

def createTeam(firstIndex, secondIndex, isRed,
               first='pacai.student.myTeam.OffenseAgent',
               second='pacai.student.myTeam.DefenseAgent'):
    """
    Create a team of two agents: an OffenseAgent and a DefenseAgent.
    """
    return [
        OffenseAgent(firstIndex),
        DefenseAgent(secondIndex),
    ]

class OffenseAgent(ReflexCaptureAgent):
    """
    Offensive agent that tracks how many pellets it is carrying manually:
      - Compares last turn's enemy food set with current turn's enemy food set
        to see if we ate some pellets.
      - Resets `carriedFood` to 0 if we return to our side (no longer Pacman)
        or if we are eaten and respawn.
    """

    def registerInitialState(self, gameState):
        """
        Called at the start of the game; we initialize:
         - self.carriedFood = 0
         - self.prevFoodSet to the initial list of enemy food.
        """
        super().registerInitialState(gameState)
        self.carriedFood = 0

        # Store the initial set of enemy's food positions:
        self.prevFoodSet = set(self.getFood(gameState).asList())

    def chooseAction(self, gameState):
        """
        1. Evaluate all legal actions using self.evaluate.
        2. Pick the best one.
        3. Update self.carriedFood by checking how many pellets we just ate.
        4. Return the chosen action.
        """
        actions = gameState.getLegalActions(self.index)
        bestVal = float('-inf')
        bestAction = None

        for action in actions:
            val = self.evaluate(gameState, action)
            if val > bestVal:
                bestVal = val
                bestAction = action

        # After choosing our action, update carriedFood based on the resulting successor.
        successor = self.getSuccessor(gameState, bestAction)
        self.updateCarriedFood(gameState, successor)

        return bestAction

    def updateCarriedFood(self, currentGameState, successor):
        """
        Compares the enemy food set before and after to see how many pellets we ate.
        Resets to 0 if we're no longer Pacman (meaning we returned home or got eaten).
        """
        myState = successor.getAgentState(self.index)
        # myPos = myState.getPosition()

        # Compute the new food set on the enemy side:
        newFoodSet = set(self.getFood(successor).asList())

        # The difference indicates how many we *might* have eaten:
        eaten = self.prevFoodSet - newFoodSet
        if len(eaten) > 0:
            self.carriedFood += len(eaten)

        # If we are no longer Pacman, it means we've either crossed back to our side
        # or we've been eaten and respawned. In both cases, carriedFood resets to 0.
        if not myState.isPacman():
            self.carriedFood = 0

        # Update prevFoodSet to the new one:
        self.prevFoodSet = newFoodSet

    def getFeatures(self, gameState, action):
        """
        Returns a dict of features for this offense agent:
          - successorScore
          - distanceToFood
          - distanceToCapsule
          - distanceToBoundary
          - carrying (our internally tracked number of pellets)
          - returnDistance (if carrying is large)
          - runAway (if an enemy ghost is too close)
        """
        features = {
            'successorScore': 0,
            'distanceToFood': 0,
            'distanceToCapsule': 0,
            'distanceToBoundary': 0,
            'carrying': 0,
            'returnDistance': 0,
            'runAway': 0
        }

        successor = self.getSuccessor(gameState, action)
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # 1. Score feature.
        features['successorScore'] = self.getScore(successor)

        # 2. Distance to nearest food.
        foodList = self.getFood(successor).asList()
        if myPos and len(foodList) > 0:
            distFood = min(self.getMazeDistance(myPos, f) for f in foodList)
            features['distanceToFood'] = float(distFood)

        # 3. Distance to capsules.
        capsules = self.getCapsules(successor)
        if myPos and len(capsules) > 0:
            distCapsule = min(self.getMazeDistance(myPos, c) for c in capsules)
            features['distanceToCapsule'] = float(distCapsule)

        # 4. Distance to boundary: encourages crossing into enemy territory.
        walls = gameState.getWalls()
        width = walls.getWidth()
        if self.red:
            boundaryX = width // 2  # For red, crossing means x >= boundaryX
            if myPos:
                if myPos[0] >= boundaryX:
                    features['distanceToBoundary'] = 0.0
                else:
                    features['distanceToBoundary'] = float(boundaryX - myPos[0])
        else:
            boundaryX = (width // 2) - 1  # For blue, crossing means x <= boundaryX
            if myPos:
                if myPos[0] <= boundaryX:
                    features['distanceToBoundary'] = 0.0
                else:
                    features['distanceToBoundary'] = float(myPos[0] - boundaryX)

        # 5. Carrying: how many pellets we track manually.
        features['carrying'] = float(self.carriedFood)

        # 6. If carrying a lot, measure distance to return home safely.
        if self.carriedFood > 3 and myPos:
            # We can reuse the same boundary distance for simplicity:
            features['returnDistance'] = features['distanceToBoundary']

        # 7. If a non-scared ghost is near, runAway.
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        if myPos and myState.isPacman():
            for enemy in enemies:
                ePos = enemy.getPosition()
                if ePos and not enemy.isPacman():
                    # Check if the ghost is not effectively scared.
                    if enemy.getScaredTimer() < 2:
                        distGhost = self.getMazeDistance(myPos, ePos)
                        if distGhost < 2:
                            features['runAway'] = 1

        return features

    def getWeights(self, gameState, action):
        """
        Adjust these weights to fine-tune your offensive behavior.
        """
        return {
            'successorScore': 100,         # Reward scoring
            'distanceToFood': -4,         # Encourage getting food
            'distanceToCapsule': -3,      # Encourage picking up capsules
            'distanceToBoundary': -10,    # Encourage crossing to the enemy side
            'carrying': 2,               # Slight positive (we have something to lose)
            'returnDistance': -15,        # Big push to come home if carrying a lot
            'runAway': -50               # Strong penalty if near a dangerous ghost
        }

    def evaluate(self, gameState, action):
        feats = self.getFeatures(gameState, action)
        wts = self.getWeights(gameState, action)
        value = 0
        for f in feats:
            value += feats[f] * wts[f]
        return value

class DefenseAgent(ReflexCaptureAgent):
    """
    A defensive agent with patrolling behavior and typical reflex features:
      - numInvaders, invaderDistance, onDefense, etc.
      - If no invaders, we move toward a 'patrol position' on our boundary.
    """

    def getFeatures(self, gameState, action):
        features = {
            'numInvaders': 0,
            'onDefense': 0,
            'invaderDistance': 0,
            'stop': 0,
            'reverse': 0,
            'runAway': 0,
            'enemyDistance': 0,
            'distanceToPatrol': 0
        }

        successor = self.getSuccessor(gameState, action)
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # 1. Are we on defense (ghost) or offense (pacman)?
        features['onDefense'] = 1
        if myState.isPacman():
            features['onDefense'] = 0

        # 2. Visible enemies and invaders.
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [e for e in enemies if e.isPacman() and e.getPosition() is not None]
        features['numInvaders'] = len(invaders)

        # 3. Distance to invaders we can see.
        if len(invaders) > 0 and myPos is not None:
            dists = []
            for enemy in invaders:
                ePos = enemy.getPosition()
                if ePos is not None:
                    dists.append(self.getMazeDistance(myPos, ePos))
            if len(dists) > 0:
                features['invaderDistance'] = min(dists)

        # 4. Distance to any visible enemy.
        if myPos is not None:
            enemyDists = []
            for enemy in enemies:
                ePos = enemy.getPosition()
                if ePos is not None:
                    enemyDists.append(self.getMazeDistance(myPos, ePos))
            if len(enemyDists) > 0:
                features['enemyDistance'] = min(enemyDists)

        # 5. Discourage stopping or reversing.
        if action == Directions.STOP:
            features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(self.index).getDirection()]
        if action == rev:
            features['reverse'] = 1

        # 6. If we (the defender) are scared and an enemy is close, we might want to run away.
        if myState.getScaredTimer() > 0 and features['enemyDistance'] < 3:
            features['runAway'] = 1

        # 7. Patrol if there are no invaders.
        if features['numInvaders'] == 0 and myPos is not None:
            patrolPos = self.getPatrolPosition(successor)
            if patrolPos is not None:
                features['distanceToPatrol'] = float(self.getMazeDistance(myPos, patrolPos))

        return features

    def getWeights(self, gameState, action):
        return {
            'numInvaders': -1000,    # Big penalty for letting invaders exist
            'onDefense': 100,        # Reward staying on our side if no reason to leave
            'invaderDistance': -15,  # Closer to invader => better
            'stop': -100,            # Penalize stopping
            'reverse': -2,           # Slightly penalize reversing
            'runAway': -500,         # If we must run away, that's undesirable
            'enemyDistance': -5,     # Keep enemies at a safe distance
            'distanceToPatrol': -2   # Light incentive to hold a boundary position
        }

    def evaluate(self, gameState, action):
        feats = self.getFeatures(gameState, action)
        wts = self.getWeights(gameState, action)
        value = 0
        for f in feats:
            value += feats[f] * wts[f]
        return value

    def chooseAction(self, gameState):
        actions = gameState.getLegalActions(self.index)
        bestValue = float('-inf')
        bestAction = None

        for action in actions:
            val = self.evaluate(gameState, action)
            if val > bestValue:
                bestValue = val
                bestAction = action

        return bestAction

    def getPatrolPosition(self, gameState):
        """
        Returns a boundary position on our side to 'patrol' when there are no invaders.
        For red team, boundary is around x = width // 2 - 1
        For blue team, boundary is around x = width // 2
        """
        walls = gameState.getWalls()
        width = walls.getWidth()
        height = walls.getHeight()

        if self.red:
            boundaryX = width // 2 - 1
        else:
            boundaryX = width // 2

        # Collect all valid positions in that boundary column.
        boundaryPositions = []
        for y in range(height):
            if not walls[boundaryX][y]:  # Not a wall
                boundaryPositions.append((boundaryX, y))

        if not boundaryPositions:
            return None

        # Pick the closest boundary position to our current position.
        myPos = gameState.getAgentState(self.index).getPosition()
        if myPos is None:
            return None

        closestPos = min(boundaryPositions, key=lambda p: self.getMazeDistance(myPos, p))
        return closestPos
