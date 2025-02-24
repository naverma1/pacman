"""
In this file, you will implement generic search algorithms which are called by Pacman agents.
"""

from pacai.util.stack import Stack
from pacai.util.queue import Queue
from pacai.util.priorityQueue import PriorityQueue

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first [p 85].

    Your search algorithm needs to return a list of actions that reaches the goal.
    Make sure to implement a graph search algorithm [Fig. 3.7].

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    ```
    print("Start: %s" % (str(problem.startingState())))
    print("Is the start a goal?: %s" % (problem.isGoal(problem.startingState())))
    print("Start's successors: %s" % (problem.successorStates(problem.startingState())))
    ```
    """

    #print("Start: %s" % (str(problem.startingState())))
    #print("Is the start a goal?: %s" % (problem.isGoal(problem.startingState())))
    #print("Start's successors: %s" % (problem.successorStates(problem.startingState())))

    # *** Your Code Here ***
    # Init lists for visited and neighboring nodes
    # init set for visited nodes
    visited_nodes = ()

    # Init stack to push nodes from starting state
    node_stack = Stack()

    # push initial starting node into stack
    node_stack.push(((problem.startingState()), []))

    # iterate through each node dfs
    while not node_stack.isEmpty():
        # pop the top node in stack and store it in current_node
        # store the path of the dfs search in path
        current_node, path = node_stack.pop()
        
        #print("next node: ", path) -- debug statement

        # if the goal state is the current node then return the node
        if problem.isGoal(current_node):
            return path


        if current_node not in visited_nodes:
            visited_nodes.add(current_node)
            
            # Push successors onto the stack
            for successor, action, _ in problem.successorStates(current_node):
                if successor not in visited_nodes:
                    current_node.push((successor, path + [action]))
    
    return []

    raise NotImplementedError()

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first. [p 81]
    """

    # *** Your Code Here ***
    # Init lists for visited and neighboring nodes
    visited_nodes = []
    neighbor_nodes = []

    # Init queue to append nodes from starting state
    node_queue = Queue()

    node_queue.push((problem.startingState(), []))

    while not node_queue.isEmpty():
        current_node, node_path = node_queue.pop()

        if (problem.isGoal(current_node) is True):
            return node_path
        
        if current_node in visited_nodes:
            continue
    
        visited_nodes.append(current_node)
        neighbor_nodes = problem.successorStates(current_node)

        for i in neighbor_nodes:
            if i[0] in visited_nodes:
                continue
            else:
                # print(leaf)
                node_queue.push((i[0], node_path + [i[1]]))

    raise NotImplementedError()

def uniformCostSearch(problem):
    """
    Search the node of least total cost first.
    """

    # *** Your Code Here ***
    visited_nodes = []
    neighbor_nodes = []

    # Init queue to append nodes from starting state
    node_pqueue = PriorityQueue()

    node_pqueue.push((problem.startingState(), []), 0)

    while not node_pqueue.isEmpty():
        current_node, node_path = node_pqueue.pop()

        if (problem.isGoal(current_node) is True):
            return node_path
        
        if current_node in visited_nodes:
            continue

        visited_nodes.append(current_node)
        neighbor_nodes = problem.successorStates(current_node)

        for i in neighbor_nodes:
            if i[0] in visited_nodes:
                continue
            else:
                cost = problem.actionsCost(node_path + [i[1]]) + i[2]
                node_pqueue.push((i[0], node_path + [i[1]]), cost)

    raise NotImplementedError()

def aStarSearch(problem, heuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """

    # *** Your Code Here ***

    visited_nodes = []
    neighbor_nodes = []

    node_pqueue = PriorityQueue()

    node_pqueue.push(((problem.startingState()), []), 0)

    while not node_pqueue.isEmpty():
        current_node, node_path = node_pqueue.pop()

        if (problem.isGoal(current_node) is True):
            return node_path
        
        if current_node in visited_nodes:
            continue

        visited_nodes.append(current_node)
        neighbor_nodes = problem.successorStates(current_node)

        for i in neighbor_nodes:
            if i[0] in visited_nodes:
                continue
            else:
                cost = problem.actionsCost(node_path + [i[1]]) + heuristic(i[0], problem)
                node_pqueue.push((i[0], node_path + [i[1]]), cost)

    raise NotImplementedError()
