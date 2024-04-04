# Search problem
This are some of the terms used in this topic:
 - Agent:This is an entity in an environment that perceives its environment.
 - State:This is the configuration of the agent in its environment.
 - Action:This are actions made in a state,its a function that takes in a state and returns a set of actions that can be performed on the state.
 - Transitional model:A description of what state results when performing any action in any state,its a function that takes in a state and an action returns a new state.
 - State space:Its a set of all states achievable from the inital state by any sequence of actions.
 - Goal test:This is a test to determine whether the state is our goal or not.
 - Path cost: This is a nuemerical value associated with any path between nodes.

## Solving a search problem
When solving a search problem we achieve a solution, which is a sequence of actions that lead from inital state to goal state
An optimal solution is the solution that has the lowest **path cost**.
In a search problem we solve solutions using nodes,which is a data structure that contains:
- The actions that lead from the parent node to the current node
- Its parent node
- Its path cost from the parent node 
- Its current state
The nodes hold information but do not provide the solution. So we use a frontier which is a mechanism that "manages" the nodes. A frontier consists of the **initial node** and **a set of empty explored sets**. To achieve its solution a frontier optimally goes through the following steps repeatedly:
1. If the frontier is empty end the algorithm, no solution.
2. Else, remove the node from the frontier and stage it:
    - If the node is the goal, end the algorithm, the solution is found
    - Else, expand on the node to achieve all sets of reachable nodes from the current node adding the result to the empty set of explored nodes **if**:
        - The explored node is not in the frontier.
        - The explored node is not in the explored set.

There are 2 types of search **informed search**,which is a search that uses knowledge specific to the problem and **uniformed search** which is a search that has no knowledge specific to the problem.

 ## Uninformed Search

 ### Depth First Search
Dfs is an algorithm that exhausts the search in a single direction before proceeding to the next direction. Dfs uses a mechanism known as "first in last out". This is implemented in a data structure called **stack**. This results in a search that goes deep in a single direction that comes in its way while leaving the other direction for later.
The algorithm has the advantage that, at its very **best**, the algorithm may choose the right path and achieve the solution optimally.
However, at its very **worst**, the algorithm may choose the wrong path and provide a solution that is not optimal.

 ### Breadth First Search
 Bfs is an algorithm that operates shallowly by exhausting each direction at the same time, taking one step in every direction.Bfs uses a mechanism called "first in first out". This is implemented by the data structure called **queue**.
 The algorithm has the advantage that its guaranteed to find the optimal solution. However, the algorithm may take longer than the minimal run time.

## Informed Search

### Greedy best first search
This algorithm expands only on the nodes that is closest to the goal state using a heuristic function,h(n). The function takes in a state and returns the estimate distance between the node and the goal state. The function ignores walls or obstructions in its way. In the case of a maze puzzle the distance is called **manhattan distance**. The efficiency of gbfs is determined by its h(n) thus its name **greedy** which will always try to take the path that will cost less based on the (x,y) co-ordinates. However, its important to note that in some cases the heuristic function can make the algorithm slower, and it'll be better to use uninformed search.

###  A* Search
Pronounced as 'A star seach', the algorithm expands on the nodes that has the lowest value of h(n),**estimated cost to the goal** and g(n),**the cost to reach the node** ie g(n)+h(n). If the algorithm finds a node with the lowest value of h(n)+g(n), it'll ditch the other path with a higher value. However, for A* to be optimal, the heuristic function should be:
  - Admissible.In other words, its should never overestimate the true cost.
  - Consistent.In other words, the estimated cost of the successive node plus the cost to reach the node should always be greater that or equal to the the cost of the current node to the goal state ie h(n)<= h(n')+g(n).

### Adversarial search 
Ths