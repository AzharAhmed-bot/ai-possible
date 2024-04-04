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
This type of algorithm the agent attempts to reach a certain goal to win but is being opposed by an adversary who wants it to lose. This is exhibited in games such as chess, tic tac toe among others.
The algorithm has the following functions:
1. The S_o: This is the initial state; in the case of a game its the empty board.
2. Player: This function given a state returns which players turn it is to play.
3. Result: This function given a state and an action returns a new state.
4. Terminal: This function given a state checks if the game is finished and returns true if win or tie and false otherwise.
5. Utility: This function given a state returns the utilty value of the state ie if -1,0,1.

## Minimax
This is a type of Adversarial search algorithm that is represented by 1 as a winning condition and -1 as losing.
![Screenshot 2024-04-04 154229](https://github.com/AzharAhmed-bot/cs50-ai-course/assets/126657393/91cd16c0-7bef-41e8-9d61-c1b060423172)
**How the algorithm works**:
Minimax simulates all the possible games that can be played from the current state upto the terminal. Each state states its value as whether its -1,0 or 1. The maximizing player always tries to maximize the value to 1 while the minimizing player tries to minimize the value to -1. At a worst case scenario both players opt for 0 rather than let the other win. 
Given a state s;
- The maximizing player picks an action a from the set of Actions(a) that produces the highest value of Min-Value(Result(s,a)) ie it picks the highest value from the result of the previous player who tried to minimize the value.
- The minimizing player picks an action a from a set of Action(a) that produces the lowest value of Max_Value(Result(s,a)) ie it picks the lowest value from the result of the previous player who tried to maximize the value.

So What are this 2 functions that are called recursively by each other?
**Psuedocode**
- Max_Value(state) function:
  - if terminal(state): 
    - return Utility 
  - else:
    - let v=-∞ 
    - for action in Action(state)
    - v=Max(v,Min_Value(Result(s,a)))
    - return v
- Min_Value(state) function:
  - if terminal(state):
    - return Utility
  - else:
    - v=∞
    - for action in Action(state)
    - v=Min(v,Max_Value(Result(s,a)))
    - return v

The value of v in both cases is opposite of the expected value; this is because each function is always trying to achieve the opposite value of the other.


## Alpha beta prunning

This is a way of optimizing minimax algorithms.After establishing the initial value of one action,if there's is evidence that the following action can bring the opponent to get a better score than the one already established action,there is no need to further investigate this action because its least favorable.
- There are a total of 255,168 tic tac toe games that can be played from the inital empty state. And 10^29000 games of chess that can be played. 
- Sometimes simulating all possible games can be impossible due to limited computation power.Thus:

## Depth Limited Minimax
Depth limited minimax considers only a pre-defined number of games before it stops without reaching the terminal. However, we don't get the precise values from each state, thus we have the **evaluation function** that estimates the utility of a given state.
![Screenshot 2024-04-04 161027](https://github.com/AzharAhmed-bot/cs50-ai-course/assets/126657393/d1301f2d-50c6-4a80-807a-e2296cab17b1)

# Conclusion
In conclusion, the study of search algorithms illuminates the intricate dynamics between problem exploration and exploitation, offering insights into how agents navigate complex solution spaces. From the foundational concepts of states, actions, and transitional models to the diverse strategies of uninformed and informed search algorithms, each facet contributes to our understanding of efficient problem-solving. Techniques such as Alpha-Beta Pruning and Depth Limited Minimax further refine our approaches, emphasizing the ongoing pursuit of optimization and scalability in computational problem-solving. As we continue to delve deeper into search algorithms, we not only refine our ability to solve diverse challenges but also advance the broader landscape of artificial intelligence and intelligent systems.
