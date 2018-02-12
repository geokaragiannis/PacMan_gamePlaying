# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

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
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.

    Print out these variables to see what you're getting, then combine them
    to create a masterful evaluation function.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPosition = successorGameState.getPacmanPosition()
    oldPosition = currentGameState.getPacmanPosition()
    oldFood = currentGameState.getFood().asList()
    newFood = successorGameState.getFood().asList()
    newGhostStates = successorGameState.getGhostStates()
    oldGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***"
    # print 'successor game state: ', successorGameState
    # print 'newPosition: ', newPosition
    # print 'oldFood: ', oldFood.asList()
    # # print 'newGhostStates: ', newGhostStates
    # print 'newScaredTimes: ', newScaredTimes
    score = successorGameState.getScore()
    newPacX = newPosition[0]
    newPacY = newPosition[1]
    oldPacX = oldPosition[0]
    oldPacY = oldPosition[1]
    newGhostPositions = [s.getPosition() for s in newGhostStates]
    oldGhostPositions = [s.getPosition() for s in oldGhostStates]
    newGhostX = newGhostPositions[0][0]
    newGhostY = newGhostPositions[0][1]
    oldGhostX = oldGhostPositions[0][0]
    oldGhostY = oldGhostPositions[0][1]
    oldDistanceToFood = 10000
    newDistanceToFood = 10000
    oldDistanceToGhost = 10000
    newDistanceToGhost = 10000
    oldFoodNum = currentGameState.getNumFood()
    newFoodNum = successorGameState.getNumFood()
    total = 0
    for foodx, foody in oldFood:
      d = util.manhattanDistance((oldPacX, oldPacY), (foodx, foody))
      total += d
    oldDistanceToFood = total/len(oldFood)
    total = 0
    for foodx, foody in oldFood:
      d = util.manhattanDistance((newPacY, newPacY), (foodx, foody))
      total += d
    newDistanceToFood = total/len(oldFood)

    oldDistanceToGhost = manhattanDistance((oldPacX,oldPacY), (oldGhostX, oldGhostY))
    newDistanceToGhost = manhattanDistance((newPacX,newPacY), (newGhostX, newGhostY))

    PacManWillDie = (newPacX, newPacY) == (newGhostX, newGhostY)

    if PacManWillDie:
      return -99999

    if newFoodNum < oldFoodNum:
      score += 99999
    elif newDistanceToFood < oldDistanceToFood:
      score += 20000
    else:
      score += -20000
    

    if action == 'Stop':
      score += -2000

    # if newDistanceToGhost > oldDistanceToGhost:
    #   score = (score+1) * 100
    # else:
    #   score = score * 1/20

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
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

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
    self.treeDepth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (question 2)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.treeDepth
      and self.evaluationFunction.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
    """
    "*** YOUR CODE HERE ***"
    # Pac-Man actions
    actions = gameState.getLegalActions(0)
    returnAction = None
    returnValue = -999999

    for action in actions:
      if action != 'Stop':
        successorState = gameState.generateSuccessor(0, action)
        # find the min value of the first Agent 
        minimaxValue = self.minValue(successorState, self.treeDepth, 1)
       
        # we need the max value that corresponds to the max action
        if minimaxValue > returnValue:
          print 'minimaxValue: ', minimaxValue
          returnValue = minimaxValue
          returnAction = action

    return returnAction
    # util.raiseNotDefined()

  def minValue(self, currentState, depth, AgentIndex):
    numAgents = currentState.getNumAgents()
    returnValue = 99999999
    # nextAgentIdex is 0 (Pac-Man) if there are no more ghosts. Else, it's the next ghost
    nextAgentIndex = 0 if AgentIndex + 1 >= numAgents else AgentIndex + 1
    actions = currentState.getLegalActions(AgentIndex)

    # if we hit a leaf node, use the evaluation function
    if depth == 0 or len(actions) == 0:
      return self.evaluationFunction(currentState)

    for action in actions:
      if action != 'Stop':
        successorState = currentState.generateSuccessor(AgentIndex, action)
        # if there is another Ghost, find its min Value. Else continue to Pac-Man's
        # next move
        if nextAgentIndex != 0:
          minimaxValue = self.minValue(currentState, depth, nextAgentIndex)
        else:
          minimaxValue = self.maxValue(successorState, depth-1, nextAgentIndex)

        if minimaxValue < returnValue:
          returnValue = minimaxValue

    return returnValue

  def maxValue(self, currentState, depth, AgentIndex):
    numAgents = currentState.getNumAgents()
    returnValue = -99999999
    # nextAgentIdex is 0 (Pac-Man) if there are no more ghosts. Else, it's the next ghost
    nextAgentIndex = 0 if AgentIndex + 1 >= numAgents else AgentIndex + 1
    actions = currentState.getLegalActions(AgentIndex)

    if depth == 0 or len(actions) == 0:
      return self.evaluationFunction(currentState)

    for action in actions:
      if action != 'Stop':
        successorState = currentState.generateSuccessor(AgentIndex, action)
        if nextAgentIndex != 0:
          minimaxValue = self.minValue(currentState, depth, nextAgentIndex)
        else:
          minimaxValue = self.maxValue(successorState, depth-1, nextAgentIndex)

        if minimaxValue > returnValue:
          returnValue = minimaxValue

    return returnValue

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.treeDepth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"
    # Pac-Man actions
    actions = gameState.getLegalActions(0)
    returnAction = None
    returnValue = -999999

    for action in actions:
      if action != 'Stop':
        successorState = gameState.generateSuccessor(0, action)
        # find the min value of the first Agent 
        minimaxValue = self.minValue(successorState, self.treeDepth, 1, -999999, 999999)
       
        # we need the max value that corresponds to the max action
        if minimaxValue > returnValue:
          print 'minimaxValue: ', minimaxValue
          returnValue = minimaxValue
          returnAction = action

    return returnAction
    # util.raiseNotDefined()

  def minValue(self, currentState, depth, AgentIndex, alpha, beta):
    numAgents = currentState.getNumAgents()
    returnValue = 99999999
    # nextAgentIdex is 0 (Pac-Man) if there are no more ghosts. Else, it's the next ghost
    nextAgentIndex = 0 if AgentIndex + 1 >= numAgents else AgentIndex + 1
    actions = currentState.getLegalActions(AgentIndex)

    # if we hit a leaf node, use the evaluation function
    if depth == 0 or len(actions) == 0:
      return self.evaluationFunction(currentState)

    for action in actions:
      if action != 'Stop':
        successorState = currentState.generateSuccessor(AgentIndex, action)
        # if there is another Ghost, find its min Value. Else continue to Pac-Man's
        # next move
        if nextAgentIndex != 0:
          minimaxValue = self.minValue(currentState, depth, nextAgentIndex, alpha, beta)
        else:
          minimaxValue = self.maxValue(successorState, depth-1, nextAgentIndex, alpha, beta)

        if minimaxValue < returnValue:
          returnValue = minimaxValue

        if returnValue <= alpha:
          return returnValue

        beta = min(beta, returnValue)

    return returnValue

  def maxValue(self, currentState, depth, AgentIndex, alpha, beta):
    numAgents = currentState.getNumAgents()
    returnValue = -99999999
    # nextAgentIdex is 0 (Pac-Man) if there are no more ghosts. Else, it's the next ghost
    nextAgentIndex = 0 if AgentIndex + 1 >= numAgents else AgentIndex + 1
    actions = currentState.getLegalActions(AgentIndex)

    if depth == 0 or len(actions) == 0:
      return self.evaluationFunction(currentState)

    for action in actions:
      if action != 'Stop':
        successorState = currentState.generateSuccessor(AgentIndex, action)
        if nextAgentIndex != 0:
          minimaxValue = self.minValue(currentState, depth, nextAgentIndex, alpha, beta)
        else:
          minimaxValue = self.maxValue(successorState, depth-1, nextAgentIndex, alpha, beta)

        if minimaxValue > returnValue:
          returnValue = minimaxValue

        if returnValue >= beta:
          return returnValue

        alpha = max(alpha, returnValue)
    return returnValue

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (question 4)
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.treeDepth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """
    "*** YOUR CODE HERE ***"
    actions = gameState.getLegalActions(0)
    returnAction = None
    returnValue = -999999

    for action in actions:
      if action != 'Stop':
        successorState = gameState.generateSuccessor(0, action)
        # find the min value of the first Agent 
        minimaxValue = self.expectiMax(successorState, self.treeDepth, 1)
       
        # we need the max value that corresponds to the max action
        if minimaxValue > returnValue:
          print 'minimaxValue: ', minimaxValue
          returnValue = minimaxValue
          returnAction = action

    return returnAction
    # util.raiseNotDefined()

  def expectiMax(self, currentState, depth, AgentIndex):
    numAgents = currentState.getNumAgents()
    returnValue = 0
    # nextAgentIdex is 0 (Pac-Man) if there are no more ghosts. Else, it's the next ghost
    nextAgentIndex = 0 if AgentIndex + 1 >= numAgents else AgentIndex + 1
    actions = currentState.getLegalActions(AgentIndex)

    # if we hit a leaf node, use the evaluation function
    if depth == 0 or len(actions) == 0:
      return self.evaluationFunction(currentState)

    for action in actions:
      if action != 'Stop':
        successorState = currentState.generateSuccessor(AgentIndex, action)
        # if there is another Ghost, find its min Value. Else continue to Pac-Man's
        # next move
        if nextAgentIndex != 0:
          expectiMaxValue = self.expectiMax(successorState, depth, nextAgentIndex)
        else:
          expectiMaxValue = self.maxValue(successorState, depth-1, nextAgentIndex)

        returnValue += expectiMaxValue

    return returnValue/len(actions)

  def maxValue(self, currentState, depth, AgentIndex):
    numAgents = currentState.getNumAgents()
    returnValue = -99999999
    # nextAgentIdex is 0 (Pac-Man) if there are no more ghosts. Else, it's the next ghost
    nextAgentIndex = 0 if AgentIndex + 1 >= numAgents else AgentIndex + 1
    actions = currentState.getLegalActions(AgentIndex)

    if depth == 0 or len(actions) == 0:
      return self.evaluationFunction(currentState)

    for action in actions:
      if action != 'Stop':
        successorState = currentState.generateSuccessor(AgentIndex, action)
        if nextAgentIndex != 0:
          expectiMaxValue = self.expectiMax(successorState, depth, nextAgentIndex)
        else:
          expectiMaxValue = self.maxValue(successorState, depth-1, nextAgentIndex)

        if expectiMaxValue > returnValue:
          returnValue = expectiMaxValue

    return returnValue

def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
  """
  "*** YOUR CODE HERE ***"
  util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
  """

  def getAction(self, gameState):
    """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.

      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

