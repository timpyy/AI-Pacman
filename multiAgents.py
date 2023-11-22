# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        capsules = currentGameState.getCapsules()
        score = successorGameState.getScore()
        ghostPosition = successorGameState.getGhostPosition(1)
        ghostDistance = util.manhattanDistance(ghostPosition, newPos)
        foods = newFood.asList()

        shortestDistance = 99999

        # Keep track of distance from ghosts
        score += max(ghostDistance, 3)

        # Manhattan distance to available food from successor state
        for food in foods:
            distance = util.manhattanDistance(food, newPos)
            # Find shortest distance to food
            if distance < shortestDistance:
                shortestDistance = distance

        # Increase score if new position is at a capsule
        if newPos in capsules:
            score += 100

        # Decrease score if Pacman has to stop
        if action == Directions.STOP:
            score -= 5

        # Take into account the shortest distance to food
        score += 50 / shortestDistance

        # Increase score if # of food in successor state is less than # of food in current state
        if len(foods) < len(currentGameState.getFood().asList()):
            score += 50

        return score

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & Expectimax PacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def minimax(depth, agents, gameState):
            # If game is in a losing or winning state, or if no change in depth
            if depth == self.depth or gameState.isLose() or gameState.isWin():
                return self.evaluationFunction(gameState)
            elif agents == 0:
                # Calculating max value for Pacman
                return max(minimax(depth, 1, gameState.generateSuccessor(agents, newState)) for newState in gameState.getLegalActions(agents))
            else:
                nextAgent = agents + 1
                if gameState.getNumAgents() == nextAgent:
                    nextAgent = 0
                if nextAgent == 0:
                    depth += 1
                # Calculating min value for ghosts
                return min(minimax(depth, nextAgent, gameState.generateSuccessor(agents, newState)) for newState in gameState.getLegalActions(agents))

        maximum = -99999
        action = Directions.SOUTH
        # Find action that maximizes utility for Pacman
        for agentState in gameState.getLegalActions(0):
            utility = minimax(0, 1, gameState.generateSuccessor(0, agentState))
            if maximum == -99999 or utility > maximum:
                action = agentState
                maximum = utility

        return action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        pacman = 0
        def alphaBeta(depth, gameState, player, alpha, beta):
            # If the game is in losing or winning state, return score
            if gameState.isLose() or gameState.isWin():
                return gameState.getScore()
            elif player == pacman:
                bestValue = -99999
                bestAction = Directions.SOUTH
                for action in gameState.getLegalActions(pacman):
                    # Depth cost of 1 means all moves have been played by Pacman and Ghosts
                    value = alphaBeta(depth, gameState.generateSuccessor(pacman, action), 1, alpha, beta)
                    if value > bestValue:
                        bestValue = value
                        bestAction = action
                        # Beta testing
                    if bestValue > beta:
                        return bestValue
                    alpha = max(alpha, bestValue)
                if depth == 0:
                    return bestAction
                return bestValue
            else:
                # Next player is Pacman if the current player is the last ghost
                if player == gameState.getNumAgents() - 1:
                    nextPlayer = pacman
                    tempDepth = depth + 1
                else:
                    nextPlayer = player + 1
                    tempDepth = depth

                bestValue = 99999
                for action in gameState.getLegalActions(player):
                    # Return evaluation if all the ghosts have played and the max depth has been reached
                    if tempDepth == self.depth:
                        value = self.evaluationFunction(gameState.generateSuccessor(player, action))
                    else:
                        value = alphaBeta(tempDepth, gameState.generateSuccessor(player, action), nextPlayer, alpha, beta)

                    bestValue = min(value, bestValue)
                    # Alpha testing
                    if bestValue < alpha:
                        return bestValue
                    beta = min(beta, bestValue)
                return bestValue
        return alphaBeta(0, gameState, pacman, -99999, 99999)

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # Very similar to alpha-beta, but minimizing player returns expected action
        pacman = 0
        def expectimax(depth, gameState, player):
            # If the game is in losing or winning state, return score
            if gameState.isLose() or gameState.isWin():
                return gameState.getScore()
            elif player == pacman:
                bestValue = -99999
                bestAction = Directions.SOUTH
                for action in gameState.getLegalActions(pacman):
                    # Depth cost of 1 means all moves have been played by Pacman and Ghosts
                    value = expectimax(depth, gameState.generateSuccessor(pacman, action), 1)
                    if value > bestValue:
                        bestValue = value
                        bestAction = action
                if depth == 0:
                    return bestAction
                return bestValue
            else:
                # Next player is Pacman if the current player is the last ghost
                if player == gameState.getNumAgents() - 1:
                    nextPlayer = pacman
                    tempDepth = depth + 1
                else:
                    nextPlayer = player + 1
                    tempDepth = depth

                value = 0
                for action in gameState.getLegalActions(player):
                    # Return evaluation if all the ghosts have played and the max depth has been reached
                    if tempDepth == self.depth:
                        value = self.evaluationFunction(gameState.generateSuccessor(player, action))
                    else:
                        value += 1/len(gameState.getLegalActions(player)) * expectimax(tempDepth, gameState.generateSuccessor(player, action), nextPlayer)
                return value
        return expectimax(0, gameState, pacman)

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: My evaluation function is based on scores that are calculated from
    distances between Pacman and the food dots / capsules / ghosts.
    Here are Pacman's priorities in order of importance:
    1. Stay alive and away from ghosts!
    2. Eat all the dots
    3. Go after capsules

    In my function, the ideal safe distance from a ghost is set to 4 (this yielded the best results).
    Further descriptions on how my scores are calculated and what they aim to achieve are annotated in the code below.
    """
    "*** YOUR CODE HERE ***"
    if currentGameState.isLose():
        return - 999999
    elif currentGameState.isWin():
        return 999999

    minDistanceFromDot = 999999
    newFood = currentGameState.getFood()
    score = scoreEvaluationFunction(currentGameState)

    # Calculate distances to all food dots and find the minimum distance
    for position in newFood.asList():
        distance = util.manhattanDistance(position, currentGameState.getPacmanPosition())
        if (distance < minDistanceFromDot):
            minDistanceFromDot = distance

    distanceFromGhost = 999999
    for x in range(1, currentGameState.getNumAgents()):
        nextDistance = util.manhattanDistance(currentGameState.getPacmanPosition(), currentGameState.getGhostPosition(x))
        distanceFromGhost = min(distanceFromGhost, nextDistance)

    # 2 is multiplied to the maximum distance from ghost in order to emphasize the importance of staying alive!
    score += max(distanceFromGhost, 4) * 2
    # Ensure that Pacman will try to get closer to the food dot in every move
    score -= minDistanceFromDot * 2
    # Add penalty for remaining food dots, this incentivizes Pacman to go after food dots that are further away as well
    score -= 3.5 * len(newFood.asList())

    # Add penalty for remaining capsules
    # This incentivizes Pacman to eat capsules so that it can go after ghosts
    capsulePositions = currentGameState.getCapsules()
    score -= 4 * len(capsulePositions)

    return score

# Abbreviation
better = betterEvaluationFunction
