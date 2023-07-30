import numpy as np
from q_learning import Q_Learning

# constants
maze_filename = "mazes/maze_5.csv"
q_table_filename = "q_table.csv"

def main():
    maze = load_maze()
    option = choose_option()
    if option == 't':
        ql = Q_Learning(maze)
        ql.train()
        ql.save_plot()
        store_q_table(ql.get_q_table())
    elif option == "l":
        qt = load_q_table()
        ql = Q_Learning(maze)
        ql.set_q_table(qt)
        path_iter = ql.simulate()

def load_maze():
    maze_rows = open(maze_filename).read().split("\n")
    maze_arr = []
    for row in maze_rows:
        if row:
            row = list(map(int, row.split(",")))
            maze_arr.append(row)
    return np.array(maze_arr)

def store_q_table(qt):
    file_str = ""
    for row in qt:
        for item in row:
            file_str += str(item) + ","
        file_str += ("\n")
    with open(q_table_filename, "w") as file:
        file.write(file_str)

def load_q_table():
    maze_rows = open(q_table_filename).read().split("\n")
    maze_arr = []
    for row in maze_rows:
        row_arr = []
        if not row:
            continue
        row = row.split(",")
        for item in row:
            if not item:
                continue
            row_arr.append(np.float64(item))
        maze_arr.append(row_arr)
    return np.array(maze_arr)

def choose_option():
    option = input("1.(T)rain\n2.(L)oad\n")
    return option.lower()

if __name__ == "__main__":
    main()
