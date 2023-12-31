from collections import deque
import numpy as np


class Connect4Env:
    def __init__(self):
        self.current_player = Cell.Red
        self.winner = None
        self.grid = np.full((6, 7), Cell.Empty)
        self.height = [5 for y in range(7)]
        self.observation_space = self.state().shape
        self.action_space = 7  # Define the action space
        self.previous_potentials = {Cell.Red: 0, Cell.Yellow: 0}

    def reset(self):
        # Reset the game to start a new episode
        self.grid = np.full((6, 7), Cell.Empty)
        self.current_player = Cell.Red
        self.winner = None
        self.height = [5 for y in range(7)]
        self.previous_potentials = {Cell.Red: 0, Cell.Yellow: 0}
        return self.state()

    def step(self, action):
        # Execute the action in the game
        self.place(action, self.current_player)
        done = self.is_game_over()
        # Get the new state, reward, and done status
        new_state = self.state()
        reward = self.get_reward(action)
        self.current_player = Cell.Yellow if self.current_player == Cell.Red else Cell.Red
        return new_state, reward, done, {}

    def get_reward(self, action) -> float:
        if self.winner is not None:
            if self.winner == self.current_player:
                return 30
            elif self.winner != self.current_player:
                return -100
        res = 0
        cur = self.get_potential_wins(self.current_player)
        # If we make a new winning condition.
        # TODO: REIMPL, removed as to increase learning.
        # if cur > self.previous_potentials[self.current_player]:
        #    res += 5
        other_player = Cell.Yellow if self.current_player == Cell.Red else Cell.Red
        other = self.get_potential_wins(other_player)
        # THE OPPONENT IS GOING TO WIN
        if other > 0:
            res -= 100
        # IF WE BLOCKED EM.
        if other < self.previous_potentials[other_player]:
            res += 200
        self.previous_potentials[other_player] = other
        self.previous_potentials[self.current_player] = cur
        return res
        # ideas:
        # Reward blocking of wins.
        # Punish for allowing wins.

    def state(self):
        player_ch = np.full(self.grid.shape, self.current_player)
        return np.stack((self.grid, player_ch), axis=-1)

    def close(self):
        # Close and clean up the environment
        return

    def __str__(self):
        s = "-" * (self.observation_space[1] * 2 + 1) + "\n"
        for row in self.grid:
            s += "|"
            for cell in row:
                s += str(cell) + '|'
            s += "\n"
        s += "-" * (self.observation_space[1] * 2 + 1)
        return s

    def place(self, col, cell):
        if self.height[col] >= 0:
            self.grid[self.height[col]][col] = cell
            self.height[col] -= 1

    def is_game_over(self):
        if self.has_won(Cell.Red):
            self.winner = Cell.Red
            return True
        if self.has_won(Cell.Yellow):
            self.winner = Cell.Yellow
            return True
        return False

    def is_valid_position(self, row, col):
        return 0 <= row < len(self.grid) and 0 <= col < len(self.grid[0])

    def get_potential_wins(self, player):
        count = 0
        rows, cols = len(self.grid), len(self.grid[0])
        for row in range(rows):
            for col in range(cols - 3):
                for i in range(4):
                    # Check sequence with one empty space
                    if all(self.grid[row][col + j] == player if j != i else self.grid[row][col + j] == 0 for j in
                           range(4)):
                        if self.is_valid_position(row, col + i) and (
                                row == rows - 1 or self.grid[row + 1][col + i] != 0):
                            # Check if the empty space is playable (i.e., at the bottom or on top of another disc)
                            count += 1

            # Check for three in a column with an open space
        for col in range(cols):
            for row in range(rows - 3):
                for i in range(4):
                    if all(self.grid[row + j][col] == player if j != i else self.grid[row + j][col] == 0 for j in
                           range(4)):
                        if self.is_valid_position(row + i + 1, col) and (
                                row + i == rows - 1 or self.grid[row + i + 1][col] != 0):
                            count += 1

            # Check diagonally (two directions)
        for row in range(rows - 3):
            for col in range(cols - 3):
                for i in range(4):
                    # Diagonal from top-left to bottom-right
                    if all(self.grid[row + j][col + j] == player if j != i else self.grid[row + j][col + j] == 0 for j
                           in
                           range(4)):
                        if self.is_valid_position(row + i + 1, col + i) and (
                                row + i == rows - 1 or self.grid[row + i + 1][col + i] != 0):
                            count += 1
                    # Diagonal from bottom-left to top-right
                    if all(self.grid[row + 3 - j][col + j] == player if j != i else self.grid[row + 3 - j][col + j] == 0
                           for j
                           in range(4)):
                        if self.is_valid_position(row + 3 - i, col + i) and (
                                row + 3 - i == rows - 1 or self.grid[row + 3 - i + 1][col + i] != 0):
                            count += 1
        return count

    def has_won(self, player):
        rows, cols = len(self.grid), len(self.grid[0])

        # Check horizontally
        for row in range(rows):
            for col in range(cols - 3):  # Adjust for the self.grid size
                if all(self.grid[row][col + i] == player for i in range(4)):
                    return True

        # Check vertically
        for col in range(cols):
            for row in range(rows - 3):  # Adjust for the self.grid size
                if all(self.grid[row + i][col] == player for i in range(4)):
                    return True

        # Check diagonally (two directions)
        for row in range(rows - 3):
            for col in range(cols - 3):
                if all(self.grid[row + i][col + i] == player for i in range(4)):
                    return True
                if all(self.grid[row + 3 - i][col + i] == player for i in range(4)):
                    return True

        return False

    def bfs(self, row, col):
        q = deque()
        count = 0
        vis = [[False for x in range(7)] for y in range(6)]
        q.append((row, col, -1))
        team = self.grid[row][col]
        paths = {}
        while len(q) > 0:
            x, y, previous_dir = q.popleft()
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    r, c = x + dx, y + dy
                    if 0 <= r < 6 and 0 <= c < 7 and not vis[r][c] and self.grid[r][c] == team:
                        if previous_dir == -1 or (dx == previous_dir[0] and dy == previous_dir[1]):
                            if previous_dir == -1:
                                paths[(dx, dy)] = [(row, col)]
                            else:
                                paths[(dx, dy)].append((r, c))
                            previous_dir = -1 if dx == 0 and dy == 0 else (dx, dy)
                            q.append((r, c, previous_dir))
                            vis[r][c] = True
        return paths.values()


class Cell:
    Empty: int = 0
    Red: int = 1
    Yellow: int = 2
