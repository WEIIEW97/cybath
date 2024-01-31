import numpy as np
import cv2

LABEL_SCALAR = 50

def generate_case1_cheat_board():
    sp = np.array([0, 0])
    initial_offset_y = 1060+1200
    # this offset x is not the distance from the boundary
    initial_offset_x = 1610-1475
    dx, dy = 550, 550
    board = np.empty((5, 5), dtype=object)
    for i in range(4, -1, -1):
        begin_board_offset = 2
        if i == 4:
            board[i][begin_board_offset] = sp+np.array([initial_offset_x, initial_offset_y])
        else:
            board[i][begin_board_offset] = board[i+1][begin_board_offset] + np.array([0, dy])
        for j in range(1, 3):
            board[i][begin_board_offset+j] = board[i][begin_board_offset+j-1] + np.array([dx, 0])
            board[i][begin_board_offset-j] = board[i][begin_board_offset-j+1] + np.array([-dx, 0])
    return board

def generate_case1_cheat_board_plus():
    sp = np.array([0, 0])
    dx, dy = 550, 550
    initial_offset_y = 1060+1200+dy//2
    initial_offset_x = 1610-1475+dx//2
    board = np.empty((4, 4), dtype=object)
    for i in range(3, -1, -1):
        begin_board_offset = 2
        if i == 3:
            board[i][begin_board_offset] = sp+np.array([initial_offset_x, initial_offset_y])
        else:
            board[i][begin_board_offset] = board[i+1][begin_board_offset] + np.array([0, dy])
        board[i][begin_board_offset+1] = board[i][begin_board_offset+1-1] + np.array([dx, 0])
        for j in range(1, 3):
            board[i][begin_board_offset-j] = board[i][begin_board_offset-j+1] + np.array([-dx, 0])
    return board



if __name__ == "__main__":
   print(generate_case1_cheat_board_plus())
    