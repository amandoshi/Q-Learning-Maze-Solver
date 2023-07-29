import numpy as np
from q_learning import Q_Learning

# constants
maze_filename = "mazes/maze_21.csv"

def main():
    maze = load_maze()
    option = choose_option()
    if option == 't':
        ql = Q_Learning(maze)
        ql.train()

def load_maze():
    maze_rows = open(maze_filename).read().split("\n")
    maze_arr = []
    for row in maze_rows:
        if row:
            row = list(map(int, row.split(",")))
            maze_arr.append(row)
    return np.array(maze_arr)

def choose_option():
    option = input("1.(T)rain\n2.(L)oad\n")
    return option.lower()

if __name__ == "__main__":
    main()
