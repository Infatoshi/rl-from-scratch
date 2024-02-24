import pygame
import numpy as np
import random

# Initialize Pygame
pygame.init()

# Setting up the display
screen_size = 400
tile_size = screen_size // 4
screen = pygame.display.set_mode((screen_size, screen_size))

# Colors
background_color = (0, 0, 0)
player_color = (255, 0, 0)  # Player color changed to red
goal_color = (0, 255, 0)  # Goal color changed to green
line_color = (0, 0, 0)

class GridGame:
    def __init__(self):
        self.board = np.zeros((4, 4))
        self.player_pos = self.random_position()
        self.goal_pos = self.random_position()
        while self.goal_pos == self.player_pos:
            self.goal_pos = self.random_position()
        self.done = False

    def random_position(self):
        return random.randint(0, 3), random.randint(0, 3)

    def move_player(self, action):
        if action == 'up' and self.player_pos[0] > 0:
            self.player_pos = (self.player_pos[0] - 1, self.player_pos[1])
        elif action == 'down' and self.player_pos[0] < 3:
            self.player_pos = (self.player_pos[0] + 1, self.player_pos[1])
        elif action == 'left' and self.player_pos[1] > 0:
            self.player_pos = (self.player_pos[0], self.player_pos[1] - 1)
        elif action == 'right' and self.player_pos[1] < 3:
            self.player_pos = (self.player_pos[0], self.player_pos[1] + 1)

        if self.player_pos == self.goal_pos:
            self.done = True

    def render(self):
        screen.fill(background_color)
        for row in range(4):
            for col in range(4):
                if (row, col) == self.player_pos:
                    pygame.draw.rect(screen, player_color, (col * tile_size, row * tile_size, tile_size, tile_size))
                elif (row, col) == self.goal_pos:
                    pygame.draw.rect(screen, goal_color, (col * tile_size, row * tile_size, tile_size, tile_size))
                pygame.draw.rect(screen, line_color, (col * tile_size, row * tile_size, tile_size, tile_size), 1)
        pygame.display.flip()

    def is_done(self):
        return self.done

# Example usage in a Jupyter Notebook
def run_game():
    while True:
        game = GridGame()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        game.move_player('up')
                    elif event.key == pygame.K_DOWN:
                        game.move_player('down')
                    elif event.key == pygame.K_LEFT:
                        game.move_player('left')
                    elif event.key == pygame.K_RIGHT:
                        game.move_player('right')
                    elif event.key == pygame.K_q:  # Quit game when 'Q' is pressed
                        running = False

                    if game.is_done():
                        print("Goal Reached!")
                        running = False

            game.render()
            pygame.time.wait(1) # Slow down the game loop
if __name__ == "__main__":
    run_game()

# Note: To run this in a Jupyter Notebook, place the run_game() call in a cell.
# However, keep in mind the limitations of Pygame in Jupyter as mentioned earlier.
