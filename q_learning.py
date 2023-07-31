import numpy as np
import plot
import random
import time

class Q_Learning:
    def __init__(self, maze, draw_plot=True):
        # maze
        self.__maze = maze
        self.__maze_height = len(maze)
        self.__maze_width = len(maze[0])
        self.__maze_wall = 1

        # N, E, S, W
        self.__actions_lst = np.array([
            np.array([0, -1]),
            np.array([1, 0]),
            np.array([0, 1]),
            np.array([-1, 0])
        ])

        # q-table
        q_table_shape = (self.__maze_width * self.__maze_height, len(self.__actions_lst))
        self.__q_table = np.zeros(q_table_shape, dtype=np.float64)

        # q-learning settings
        self.__num_episodes = 300000
        self.__max_iteration = 100
        self.__max_iteration_growth_rate = 10000
        self.__max_iteration_growth_factor = 1.3
        self.__exploration_decay_rate = 0.00001
        self.__min_exploration_rate = 0.01
        self.__discount_factor = 0.8
        self.__learning_rate = 0.1

        # q-learning rewards
        self.__target_reward = 1
        self.__move_reward = -1 / len(self.__q_table) ** 2
        self.__wall_reward = 10 * self.__move_reward
        self.__off_grid_reward = 10 * self.__move_reward

        # states
        self.__start_state = np.array([0, 0])
        self.__target_state = np.array([self.__maze_width - 1, self.__maze_height - 1])
        self.__current_state = None

        # plot
        self.__plot_rate = 5000
        self.__plot_img_filename = "plot.png"
        self.__plot_img_format = "png"
        self.__plot_img_resolution = 300

        if draw_plot:
            plot.init()

    def train(self):
        for i in range(self.__num_episodes):
            self.__update_plot(i)
            self.__update_max_iteration(i)
            self.__current_state = self.__start_state
            for _ in range(self.__max_iteration):
                action_index = self.__select_action_index(i)
                action = self.__actions_lst[action_index]
                next_state, reward, complete = self.__process(action)
                self.__update_q_table(next_state, action_index, reward, complete)
                if complete:
                    break
                self.__current_state = next_state

    def save_plot(self):
        plot.save_plot(self.__plot_img_filename, self.__plot_img_format, self.__plot_img_resolution)
    
    def simulate(self):
        complete = False
        curr_state = self.__start_state
        states_explored = set()
        while not complete:
            if self.__valid_state(curr_state) and self.__maze_tile(curr_state) != self.__maze_wall and tuple(curr_state) not in states_explored:
                yield curr_state
                states_explored.add(tuple(curr_state))
                if np.array_equal(curr_state, self.__target_state):
                    complete = True
                action_index = np.argmax(self.__q_table[self.__q_table_index(curr_state)])
                action = self.__actions_lst[action_index]
                curr_state = np.add(curr_state, action)
            else:
                complete = True
    
    def get_q_table(self):
        return self.__q_table

    def set_q_table(self, q_table):
        self.__q_table = q_table

    def __update_max_iteration(self, episode):
        if episode % self.__max_iteration_growth_rate != 0 and len(list(self.simulate())) > 0.9 * self.__max_iteration:
            self.__max_iteration *= self.__max_iteration_growth_factor
            self.__max_iteration = int(self.__max_iteration)

    def __update_plot(self, curr_episode):
        if curr_episode % self.__plot_rate != 0:
            return
        y_coord = len(list(self.simulate()))
        plot.x_coords.append(curr_episode)
        plot.y_coords.append(y_coord)
        plot.update_plot()

    def __q_table_index(self, state):
        return state[0] * self.__maze_width + state[1]
    
    def __select_action_index(self, curr_episode):
        if random.random() < self.__get_exploration_rate(curr_episode):
            return random.randint(0, len(self.__actions_lst) - 1)
        q_table_index = self.__q_table_index(self.__current_state)
        return np.argmax(self.__q_table[q_table_index])
    
    def __process(self, action):
        next_state = np.add(self.__current_state, action)
        if not self.__valid_state(next_state):
            reward = self.__off_grid_reward
            complete = True
        elif self.__maze_tile(next_state) == self.__maze_wall:
            reward = self.__wall_reward
            complete = True
        elif np.array_equal(self.__current_state, self.__target_state):
            reward = self.__target_reward
            complete = True
        else:
            reward = self.__move_reward
            complete = False
        return next_state, reward, complete
    
    def __valid_state(self, state):
        return 0 <= state[0] < self.__maze_width and 0 <= state[1] < self.__maze_height
    
    def __maze_tile(self, state):
        return self.__maze[state[1]][state[0]]
    
    def __update_q_table(self, next_state, action_index, reward, complete):
        q_index = self.__q_table_index(self.__current_state)
        self.__q_table[q_index][action_index] = (1 - self.__learning_rate) * self.__q_table[q_index][action_index]
        self.__q_table[q_index][action_index] += self.__learning_rate * reward
        if not complete:
            next_state_max_q_value = max(self.__q_table[self.__q_table_index(next_state)])
            self.__q_table[q_index][action_index] += self.__learning_rate * self.__discount_factor * next_state_max_q_value
    
    def __get_exploration_rate(self, curr_episode):
        return max(self.__min_exploration_rate, np.exp(-self.__exploration_decay_rate * curr_episode))
