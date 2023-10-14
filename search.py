# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    successors = []
    visited = []
    #Utilize stack for DFS
    stack = util.Stack()
    stack.push((problem.getStartState(), []))  # getStartState() returns (coordinates , path)
    while not stack.isEmpty():
        (currentState, path) = stack.pop()
        if currentState not in visited:
            # add current node to visited
            visited.append(currentState)
            # stop if goal is reached
            if problem.isGoalState(currentState):
                break
            # getSuccessors() returns possible next paths (coordinates, path)
            successors = problem.getSuccessors(currentState)
            for node in successors:
                # Don't add node again if it is already in the stack
                if node[0] not in visited:
                    stack.push((node[0], path + [node[1]]))
    return path

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    successors = []
    visited = []
    queue = util.Queue()
    queue.push((problem.getStartState(), []))  # getStartState() returns (coordinates , path)
    while not queue.isEmpty():
        (currentstate, path) = queue.pop()
        if currentstate not in visited:
            # add current node to visited
            visited.append(currentstate)
            # stop if goal is reached
            if problem.isGoalState(currentstate):
                break
            # getSuccessors() returns possible next paths (coordinates, path)
            successors = problem.getSuccessors(currentstate)
            for node in successors:
                # Don't add node again if it is already in the queue
                if node[0] not in visited:
                    queue.push((node[0], path + [node[1]]))

    return path

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    successors = []
    visited = {}
    priorityqueue = util.PriorityQueue()
    priorityqueue.push((problem.getStartState(), [], 0), 0)  # Coordinates, path, and cost

    while not priorityqueue.isEmpty():
        # choose a leaf node for expansion
        currentstate, path, cost = priorityqueue.pop()
        if currentstate not in visited or cost < visited[currentstate]:
            # update visited list with minimum cost
            visited[currentstate] = cost
            # stop if goal is reached
            if problem.isGoalState(currentstate):
                break
            # getSuccessors() returns possible next paths (coordinates, path)
            successors = problem.getSuccessors(currentstate)
            for node in successors:
                # Update priority queue
                if node in successors:
                    priorityqueue.push((node[0], path + [node[1]], cost + node[2]), cost + node[2])
    return path

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    visited, path = ([], [])
    priorityqueue = util.PriorityQueue()
    priorityqueue.push((problem.getStartState(), [], 0), 0)  # Coordinates, path, actual cost, heuristic cost

    while not priorityqueue.isEmpty():
        # Since we utilize a priority queue, smallest cost object will be popped
        currentstate, path, cost = priorityqueue.pop()

        # Check for goal state
        if problem.isGoalState(currentstate):
            return path
        # Else, expand and add to priority queue
        elif currentstate not in visited:
            visited.append(currentstate)
            successors = problem.getSuccessors(currentstate)
            for nextstate, newpath, stepcost in successors:
                if nextstate not in visited:
                    forwardcost = heuristic(nextstate, problem)
                    priorityqueue.push((nextstate, path + [newpath], cost + stepcost), cost + stepcost + forwardcost)

    return []
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
