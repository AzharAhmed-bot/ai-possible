
# Implementing the frontier that follows the procedure:
# Repeate:
# 1. If the frontier is empty then no solution is possible
# 2. Remove a node from the frontier and stage it
#   1.1 If the node is the goal state then end the process, solution found
#   1.2 Else expand on the state to get all possible states from the staged state
#   1.3 Store the new states in the empty set of explored nodes
# We only need to add nodes in the frontier if they are not in the explored state
# and not in the frontier itself.

import sys

# A node is a data structure that contains : parent node, the state,
# path cost from the initial state and the action made to get from its parent node
class Node():
    def __init__(self,state,action,parent) -> None:
        self.state=state
        self.state=action
        self.parent=parent


class StackFrontier():
    def __init__(self):
        self.frontier=[]
    # Add node to the frontier
    def add(self,node):
        self.frontier.append(node)
    # Check if the frontier is already empty
    def empty(self):
        return len(self.frontier)==0
    # Check for a particular state in the frontier
    def contains_particular_state(self,state):
        return any(node.state==state for node in self.frontier)
    # Stage a particulat node
    def remove(self):
        if self.empty():
            raise Exception("The frontier is empty")
        else:
            node=self.frontier[-1]
            self.frontier=self.frontier[:-1]
            return node 


class QueueFrontier(StackFrontier):
    def remove(self):
        if self.empty():
            raise Exception("The queue is empty")
        else:
            node=self.frontier[0]
            self.frontier=self.frontier[1:]
            return node