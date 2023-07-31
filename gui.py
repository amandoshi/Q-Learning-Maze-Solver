import numpy as np
import pygame
import time

class GUI:
    # colors
    BLACK = (0, 0, 0)
    GREEN = (87, 245, 66)
    RED = (245, 66, 66)
    YELLOW = (245, 230, 66)
    WHITE = (255, 255, 255)

    # maze
    TILE_SIZE = 20
    TILE_BORDER_SIZE = 1
    MAZE_PATH = 0
    MAZE_WALL = 1

    # animation
    TIME_DELAY = 0.2
    SOLVED_FILENAME = "solve.png"

    def __init__(self, maze, path_iter):
        # maze
        self.__maze = maze
        self.__path_iter = path_iter
        self.__start_state = np.array((0, 0))
        self.__target_state = np.array((self.__maze_width() - 1, self.__maze_height() - 1))

        # pygame
        screen_width = self.__maze_width() * GUI.TILE_SIZE + 1 * GUI.TILE_BORDER_SIZE
        screen_height = self.__maze_height() * GUI.TILE_SIZE + 1 * GUI.TILE_BORDER_SIZE
        self.__screen = pygame.display.set_mode((screen_width, screen_height))
        self.__exit = False
    
    def animate(self):
        self.__screen.fill(GUI.BLACK)
        self.__render_maze()
        time.sleep(GUI.TIME_DELAY)
        for state in self.__path_iter:
            if self.__exit:
                break
            if not (np.array_equal(state, self.__start_state) or np.array_equal(state, self.__target_state)):
                self.__render_tile(state, GUI.YELLOW)
            self.__process_event()
            pygame.display.flip()
            time.sleep(GUI.TIME_DELAY)
        pygame.image.save(self.__screen, GUI.SOLVED_FILENAME)
        pygame.quit()

    def __maze_width(self):
        return len(self.__maze[0])
    
    def __maze_height(self):
        return len(self.__maze)
    
    def __process_event(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.__exit = True

    def __render_maze(self):
        for y in range(self.__maze_height()):
            for x in range(self.__maze_width()):
                rect_color = self.__tile_colors(self.__maze[y][x])
                state = np.array([x, y])
                self.__render_tile(state, rect_color)
        self.__render_tile(self.__start_state, GUI.GREEN)
        self.__render_tile(self.__target_state, GUI.RED)
    
    def __render_tile(self, state, rect_color):
        x_coord, y_coord = (state[0] * GUI.TILE_SIZE, state[1] * GUI.TILE_SIZE)
        rect_shape = (x_coord + GUI.TILE_BORDER_SIZE, y_coord + GUI.TILE_BORDER_SIZE, 
                GUI.TILE_SIZE - GUI.TILE_BORDER_SIZE * 2, GUI.TILE_SIZE - GUI.TILE_BORDER_SIZE * 2)
        pygame.draw.rect(self.__screen, rect_color, rect_shape)

    def __tile_colors(self, value):
        match value:
            case GUI.MAZE_PATH: # path
                return GUI.WHITE
            case GUI.MAZE_WALL: # wall
                return GUI.BLACK
