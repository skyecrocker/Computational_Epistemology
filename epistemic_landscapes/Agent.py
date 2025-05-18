import numpy as np
import random
from Landscapes import Landscape
from Social_Network import *

class Agent():

    def __init__(self, coordinates: tuple, range: int = 5, grid_size: int = 41, strategy: str = "default", epsilon: float = 0.0):
        """
        Args:
            coordinates (tuple): The starting coordinates of the agent.
            range (int): How many tiles the agent can move in a timestep (*in either direction)
            grid_size (int): The size of the grid.
        """
        self.coordinates = coordinates
        self.range = range
        self.grid_knowledge = np.full((grid_size, grid_size), -1, dtype=float)
        self.visited_tiles = {coordinates}
        self.tiles_in_range = self.get_tiles_in_range()

        self.strategy = strategy

        if strategy == "epsilonGreedy":
            if not (0.0 <= epsilon <= 1.0):
                raise ValueError("Epsilon should be between 0 and 1.")
            self.epsilon = epsilon

        if strategy != "epsilonGreedy" and epsilon != 0.0:
            raise ValueError("Epsilon should only be set for epsilonGreedy strategy.")

    def update_grid_knowledge(self, coordinates: tuple, value: float):
        """Update the agent’s knowledge grid."""
        if self.grid_knowledge[coordinates] == -1:
            self.grid_knowledge[coordinates] = value
            self.tiles_in_range = self.get_tiles_in_range()

    def _get_tiles_in_range_nowrap(self):
        size = self.grid_knowledge.shape[0]
        
        if self.range == "unlimited" or self.range * 2 + 1 >= size:
            return self.grid_knowledge
        
        x, y = self.coordinates
        x_min, x_max = max(0, x - self.range), min(size, x + self.range + 1)
        y_min, y_max = max(0, y - self.range), min(size, y + self.range + 1)
        
        tiles_in_range = np.full_like(self.grid_knowledge, -2, dtype=float)
        tiles_in_range[x_min:x_max, y_min:y_max] = self.grid_knowledge[x_min:x_max, y_min:y_max]
        
        return tiles_in_range

    def get_tiles_in_range(self):
        """Return a sub-grid of known tiles within range, with efficient wrap-around behavior."""
        size = self.grid_knowledge.shape[0]

        if self.range == "unlimited" or self.range * 2 + 1 >= size:
            return self.grid_knowledge

        tiles_in_range = np.full_like(self.grid_knowledge, -2, dtype=float)
        x, y = self.coordinates

        # Generate indices for the range with wrap-around
        x_indices = [(x + i) % size for i in range(-self.range, self.range + 1)]
        y_indices = [(y + j) % size for j in range(-self.range, self.range + 1)]

        # Use np.ix_ to efficiently extract a submatrix with the wrap-around indices
        tiles_in_range[np.ix_(x_indices, y_indices)] = self.grid_knowledge[np.ix_(x_indices, y_indices)]

        return tiles_in_range

    def get_best_tile_in_range(self):
        """Find the highest-valued tile within the range."""
        return np.unravel_index(np.argmax(self.tiles_in_range, axis=None), self.tiles_in_range.shape)

    def get_action(self):
        if self.strategy == "default":
            return self._get_action_defaultStrategy()
        elif self.strategy == "epsilonGreedy":
            return self._get_action_epsilonGreedyStrategy()
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _get_action_defaultStrategy(self):
        """Choose the next move based on unexplored or highest-valued tiles."""
        best_tile = self.get_best_tile_in_range()
        
        search_range = range(-1, 2) if best_tile in self.visited_tiles else range(-2, 3)
        size = self.grid_knowledge.shape[0]

        # Find unexplored adjacent tiles
        possible_actions = [
            ((best_tile[0] + i) % size, (best_tile[1] + j) % size)
            for i in search_range for j in search_range
            if self.grid_knowledge[(best_tile[0] + i) % size, (best_tile[1] + j) % size] == -1
        ]

        # If no unexplored adjacent tiles, find unexplored tiles in range
        if not possible_actions:
            possible_actions = list(zip(*np.where(self.tiles_in_range == -1)))

        # If everything is explored, move to any tile in range
        if not possible_actions:
            possible_actions = list(zip(*np.where(self.tiles_in_range != -2)))

        #print(f"possible actions new: {possible_actions}", f"loc={self.coordinates}")
        #random.seed(50)
        return random.choice(possible_actions)

    def _get_action_epsilonGreedyStrategy(self):
        """Choose the next move based on epsilon-greedy strategy."""
        if random.random() < self.epsilon:
            possible_actions = []

           # Find unexplored tiles in range
            if not possible_actions:
                possible_actions = list(zip(*np.where(self.tiles_in_range == -1)))

            # If everything is explored, move to any tile in range
            if not possible_actions:
                possible_actions = list(zip(*np.where(self.tiles_in_range != -2)))

            #print(f"possible actions new: {possible_actions}", f"loc={self.coordinates}")
            #random.seed(50)
            return random.choice(possible_actions)
        else:
            return self._get_action_defaultStrategy()


    def move(self, coordinates: tuple):
        """Move the agent to new coordinates and update knowledge."""
        self.coordinates = coordinates
        self.tiles_in_range = self.get_tiles_in_range()
        self.visited_tiles.add(coordinates)

class Agent3D(Agent):

    def __init__(self, coordinates: tuple, range: int = 5, grid_size: int = 21, strategy: str = "default", epsilon: float = 0.0):
        """
        Args:
            coordinates (tuple): The starting coordinates of the agent.
            range (int): How many tiles the agent can move in a timestep (*in either direction)
            grid_size (int): The size of the grid.
        """
        self.coordinates = coordinates
        self.range = range
        self.grid_knowledge = np.full((grid_size, grid_size, grid_size), -1, dtype=float)
        self.visited_tiles = {coordinates}
        self.tiles_in_range = self.get_tiles_in_range()

        self.strategy = strategy

        if strategy == "epsilonGreedy":
            if not (0.0 <= epsilon <= 1.0):
                raise ValueError("Epsilon should be between 0 and 1.")
            self.epsilon = epsilon

        if strategy != "epsilonGreedy" and epsilon != 0.0:
            raise ValueError("Epsilon should only be set for epsilonGreedy strategy.")

    def update_grid_knowledge(self, coordinates: tuple, value: float):
        """Update the agent’s knowledge grid."""
        if self.grid_knowledge[coordinates] == -1:
            self.grid_knowledge[coordinates] = value
            self.tiles_in_range = self.get_tiles_in_range()

    def _get_tiles_in_range_nowrap(self):
        raise NotImplementedError("Need to update this")

        size = self.grid_knowledge.shape[0]
        
        if self.range == "unlimited" or self.range * 2 + 1 >= size:
            return self.grid_knowledge
        
        x, y = self.coordinates
        x_min, x_max = max(0, x - self.range), min(size, x + self.range + 1)
        y_min, y_max = max(0, y - self.range), min(size, y + self.range + 1)
        
        tiles_in_range = np.full_like(self.grid_knowledge, -2, dtype=float)
        tiles_in_range[x_min:x_max, y_min:y_max] = self.grid_knowledge[x_min:x_max, y_min:y_max]
        
        return tiles_in_range

    def get_tiles_in_range(self):
        """Return a sub-grid of known tiles within range, with efficient wrap-around behavior."""
        size = self.grid_knowledge.shape[0]

        if self.range == "unlimited" or self.range * 2 + 1 >= size:
            return self.grid_knowledge

        tiles_in_range = np.full_like(self.grid_knowledge, -2, dtype=float)
        x, y, z = self.coordinates

        # Generate indices for the range with wrap-around
        x_indices = [(x + i) % size for i in range(-self.range, self.range + 1)]
        y_indices = [(y + j) % size for j in range(-self.range, self.range + 1)]
        z_indices = [(z + k) % size for k in range(-self.range, self.range + 1)]

        # Use np.ix_ to efficiently extract a submatrix with the wrap-around indices
        tiles_in_range[np.ix_(x_indices, y_indices, z_indices)] = self.grid_knowledge[np.ix_(x_indices, y_indices, z_indices)]

        return tiles_in_range

    def get_best_tile_in_range(self):
        """Find the highest-valued tile within the range."""
        return np.unravel_index(np.argmax(self.tiles_in_range, axis=None), self.tiles_in_range.shape)

    def get_action(self):
        if self.strategy == "default":
            return self._get_action_defaultStrategy()
        elif self.strategy == "epsilonGreedy":
            return self._get_action_epsilonGreedyStrategy()
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _get_action_defaultStrategy(self):
        """Choose the next move based on unexplored or highest-valued tiles."""
        best_tile = self.get_best_tile_in_range()
        
        search_range = range(-1, 2)
        # search_range = range(-1, 2) if best_tile in self.visited_tiles else range(-2, 3)
        size = self.grid_knowledge.shape[0]

        # Find unexplored adjacent tiles
        possible_actions = [
            ((best_tile[0] + i) % size, (best_tile[1] + j) % size, (best_tile[2] + k) % size)
            for i in search_range for j in search_range for k in search_range
            if self.grid_knowledge[
            (best_tile[0] + i) % size, (best_tile[1] + j) % size, (best_tile[2] + k) % size
            ] == -1
        ]

        # If no unexplored adjacent tiles, find unexplored tiles in range
        if not possible_actions:
            possible_actions = list(zip(*np.where(self.tiles_in_range == -1)))

        # If everything is explored, move to any tile in range
        if not possible_actions:
            possible_actions = list(zip(*np.where(self.tiles_in_range != -2)))

        #print(f"possible actions new: {possible_actions}", f"loc={self.coordinates}")
        #random.seed(50)
        return random.choice(possible_actions)

    def _get_action_epsilonGreedyStrategy(self):
        """Choose the next move based on epsilon-greedy strategy."""
        if random.random() < self.epsilon:
            possible_actions = []

           # Find unexplored tiles in range
            if not possible_actions:
                possible_actions = list(zip(*np.where(self.tiles_in_range == -1)))

            # If everything is explored, move to any tile in range
            if not possible_actions:
                possible_actions = list(zip(*np.where(self.tiles_in_range != -2)))

            #print(f"possible actions new: {possible_actions}", f"loc={self.coordinates}")
            #random.seed(50)
            return random.choice(possible_actions)
        else:
            return self._get_action_defaultStrategy()


    def move(self, coordinates: tuple):
        """Move the agent to new coordinates and update knowledge."""
        self.coordinates = coordinates
        self.tiles_in_range = self.get_tiles_in_range()
        self.visited_tiles.add(coordinates)


@DeprecationWarning
class Retired_Agent():

    def __init__(self, coordinates: tuple, range: int = 5, grid_size: int = 41):
        """
        Args:
            coordinates (tuple): The starting coordinates of the agent.
            range (int): How many tiles the agent is able to move in a timestep (*in either direction)
            grid_size (int): The size of the grid.
        """
        self.coordinates = coordinates
        self.range = range
        self.grid_knowledge = np.full((grid_size, grid_size), -1, dtype=float)
        self.visited_tiles = set()
        self.visited_tiles.add(coordinates)
        self.tiles_in_range = self.get_tiles_in_range()

    def update_grid_knowledge(self, coordinates: tuple, value: float):
        """
        Update tile on agent's personal knowledge grid to reflect the actual value of the tile
        in the landscape.

        Args:
            coordinates (tuple): The coordinates of the tile.
            value (float): The actual value of the tile.
        """

        if self.grid_knowledge[coordinates] == -1:
            self.grid_knowledge[coordinates] = value
            self.tiles_in_range = self.get_tiles_in_range()

        return

    def get_tiles_in_range(self):

        if self.range == "unlimited" or self.range * 2 + 1 >= self.grid_knowledge.shape[0]:
            return self.grid_knowledge

        tiles_in_range = np.full((self.grid_knowledge.shape[0], self.grid_knowledge.shape[1]), -2, dtype=float)
        for i in range(-self.range, self.range + 1):
            x = i + self.coordinates[0]
            x = x - self.grid_knowledge.shape[0] if x >= self.grid_knowledge.shape[0] else x
            for j in range(-self.range, self.range + 1):
                y = j + self.coordinates[1]
                y = y - self.grid_knowledge.shape[1] if y >= self.grid_knowledge.shape[1] else y
                tiles_in_range[(x, y)] = self.grid_knowledge[(x, y)]

        return tiles_in_range

    def get_best_tile_in_range(self):
        best_tile = np.unravel_index(np.argmax(self.tiles_in_range, axis=None), self.tiles_in_range.shape)
        return best_tile

    def get_action(self):
        best_tile = self.get_best_tile_in_range()

        search_range = range(-1, 2) if best_tile in self.visited_tiles else range(-2, 3)

        # get unexplored adjecent tiles
        possible_actions = []
        for i in search_range:
            for j in search_range:
                x_coord = best_tile[0] + i
                if x_coord >= self.grid_knowledge.shape[0]:
                    x_coord -= self.grid_knowledge.shape[0]
                elif x_coord < 0:
                    x_coord += self.grid_knowledge.shape[0]
                y_coord = best_tile[1] + j
                if y_coord >= self.grid_knowledge.shape[1]:
                    y_coord -= self.grid_knowledge.shape[1]
                elif y_coord < 0:
                    y_coord += self.grid_knowledge.shape[1]
                if self.grid_knowledge[(x_coord, y_coord)] == -1:
                    possible_actions.append((x_coord, y_coord))

        # if all adjecent tiles are explored, move to random unvisited tile in range
        if len(possible_actions) == 0:

            # move to random unvisited location in range
            for i in range(len(self.tiles_in_range)):
                for j in range(len(self.tiles_in_range)):
                    if self.tiles_in_range[i, j] == -1:
                        possible_actions.append((i, j))

        # if all tiles in range are explored, move to any random tile in range
        if len(possible_actions) == 0:
            for i in range(len(self.tiles_in_range)):
                for j in range(len(self.tiles_in_range)):
                    if self.tiles_in_range[i, j] != -2:
                        possible_actions.append((i, j))
        
        #print(f"possible actions og: {possible_actions}", f"loc={self.coordinates}")
        #random.seed(50)
        return random.choice(possible_actions)


    def move(self, coordinates: tuple):
        self.coordinates = coordinates
        self.tiles_in_range = self.get_tiles_in_range()
        self.visited_tiles.add(coordinates)



if __name__ == "__main__":

    random.seed(42)
    np.random.seed(42)

    agent1 = Retired_Agent((1, 1), range=5, grid_size=10)
    agent2 = Agent((1, 1), range=5, grid_size=10)

    landscape = Landscape(size=41, sigma=3, seed=42, method="boostGlobal")


    network = Network([agent1, agent2],  makeIsolatedGraph(2), landscape.values)

    for i in range(6):
        network.step()
        print(f"Step {i+1}:")

