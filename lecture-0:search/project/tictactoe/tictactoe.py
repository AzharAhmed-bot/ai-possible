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
    action_set = set()
    for row_index, row in enumerate(board):
        for col_index, value in enumerate(row):
            if value not in (X, O):
                action_set.add((row_index, col_index))

    return action_set


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.

    Cannot overwrite values.
    """
    # check if action is a valid action
    if action not in actions(board):
        raise Exception("You have performed an invalid action.")

    # whose turn it is now? result will be either X or O
    current_player = player(board)

    # create a deep copy so as not to substitute values
    new_board = copy.deepcopy(board)
    
    row_index, col_index = action 
    new_board[row_index][col_index] = current_player

    return new_board


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    for i in range(len(board)):
        if board[i][0]==board[i][1]==board[i][2] != EMPTY:
            return board[i][0]
        if board[0][i]==board[1][i]==board[2][i] != EMPTY:
            return board[0][i]
    if board[0][0]==board[1][1]==board[2][2]!=EMPTY:
        return board[0][0]
    if board[0][2]==board[1][1]==board[2][0] !=EMPTY:
        return board[0][2]
    return None
    
    


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    game_winner=winner(board)
    if game_winner:
        return True
    
    for  row in board:
        for cell in row:
            if cell is None:
                return False
            
    return False


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    winner_player=winner(board)
    if winner_player==X:
        return 1
    elif winner_player==O:
        return -1
    
    return 0




def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if terminal(board):
        return None
    
    current_player = player(board)
    if current_player== X:
        candidate_outcome=-math.inf
        for a in actions(board):
            possible_board=result(board,a)
            opponent_value=min_value(possible_board)
            if opponent_value> candidate_outcome:
                candidate_outcome=opponent_value
                candidate_action=a
    
    if current_player==O:
        candidate_outcome=math.inf
        for a in actions(board):
            possible_board=result(board,a)
            opponent_value=max_value(possible_board)
            if opponent_value < candidate_outcome:
                candidate_outcome=opponent_value
                candidate_action=a

    return candidate_action

    
    

            
def max_value(board):
    if terminal(board):
        return utility(board)
    v=-math.inf
    for a in actions(board):
        new_board=result(board,a)
        v=max(v,min_value(new_board))
    return v


def min_value(board):
    if terminal(board):
        return utility(board)
    
    v=math.inf
    for a in actions(board):
        new_board=result(board,a)
        v=min(v,max_value(new_board))
    return v

    

