"""
Tic Tac Toe Player
"""
import copy
import math

X = "X"
O = "O"
EMPTY = None
INFINITY = 1e3


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    x_counter=0
    o_counter=0

    for row in board:
        x_counter+=row.count(X)
        o_counter+=row.count(O)
    if o_counter<x_counter:
        return O
    
    return X
    

    
def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    
    An available in this game refers to a position on board where no X nor O is 
    chosen yet.
    """
    possible_actions=set()
    for r_index, row in enumerate(board):
        for c_index,col in enumerate(row):
            if  col == EMPTY:
                possible_actions.add((r_index,c_index))
    
    return possible_actions



def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.

    Cannot overwrite values.
    """
    if action not in actions(board):
        raise Exception("Invalid action")
    
    new_board = copy.deepcopy(board)
    current_player = player(board)

    # Unpack the action tuple into row and column indices
    i, j = action  
    # Update the cell at (i, j) with the current player's symbol
    new_board[i][j] = current_player

    return new_board


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    # Square box it'll return 3
    board_size=len(board)

    # Horizontal winner
    for row in board:
        if row.count(X)==board_size:
            return X
        if row.count(O)==board_size:
            return O
        
    # Vertical winner
    for c_index in range(board_size):
        col=[row[c_index] for row in board]
        if col.count(X)==board_size:
            return X
        if col.count(O)==board_size:
            return O
        
            
    # Diagonal winner
    main_diagonal=[board[i][i] for i in range(board_size)]
    anti_diagonal=[board[i][board_size-1-i] for i in range(board_size)]
            
    if main_diagonal.count(X)==board_size  or anti_diagonal.count(X)==board_size:
        return X
    if main_diagonal.count(O)==board_size or anti_diagonal.count(O)==board_size:
        return O
    

    return None
            


    
    
    


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    player_won=winner(board)

    # X, O who won
    if player_won==X or player_won==O:
        return True
    
    # Check if the board is full
    if not any(map(lambda x: EMPTY in x,board)):
        return True

    return False


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    player_won=winner(board)

    if player_won==X:
        return 1
    if player_won==O:
        return -1
    
    return 0
    




def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if terminal(board):
        return None
    
    current_player=player(board)

    if current_player==X:
        player_value=-math.inf
        for action in actions(board):
            new_board=result(board,action)
            opponent_value=min_value(new_board)
            if opponent_value>player_value:
                player_value=opponent_value
                optimal_action=action

    if current_player==O:
        player_value=math.inf
        for action in actions(board):
            new_board=result(board,action)
            opponent_value=max_value(new_board)
            if opponent_value<player_value:
                player_value=opponent_value
                optimal_action=action

    return optimal_action

   
    




def max_value(board):
    if terminal(board):
        return utility(board)
    v=-math.inf

    for action in actions(board):
        new_baord=result(board,action)
        v=max(v,min_value(new_baord))
    return v
 

def min_value(board):
    if terminal(board):
        return utility(board)

    v=math.inf
    for action in actions(board):
        new_board=result(board,action)
        v=min(v,max_value(new_board))
    
    return v
    

                








    

