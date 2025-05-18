import numpy as np

class Network:
    def __init__(self, agents, adjacencyMatrix, landscape):
        """
        
        """
        self.agents = agents
        self.adjacencyMatrix = adjacencyMatrix
        self.landscape = landscape
        self.visits = np.zeros(landscape.shape)
        self._update_visits()
        self._update_agents_knowledge()

    def step(self):
        # Agents take actions and TODO receive rewards
        #print("actions")
        for i, agent in enumerate(self.agents):
            destination_tile = agent.get_action()
            #print(f"agent {i}: {agent.coordinates} => {destination_tile}")
            agent.move(destination_tile)
            #print("success")

        # Update agents' knowledge grids
        self._update_agents_knowledge()
        
        # Update visits
        self._update_visits()

    def has_found_global_peak(self):
        # Check if any agent has found the global peak
        for agent in self.agents:
            if np.max(agent.grid_knowledge) == 1:
                return True
        return False
    
    def num_agents_found_global_peak(self):
        # Count the number of agents that have found the global peak
        count = 0
        for agent in self.agents:
            if np.max(agent.grid_knowledge) == 1:
                count += 1
        return count

    def _update_agents_knowledge(self):
        for i in range(len(self.adjacencyMatrix)):
            for j in range(len(self.adjacencyMatrix)):
                if self.adjacencyMatrix[i][j] == 1:
                    coords = self.agents[j].coordinates
                    self.agents[i].update_grid_knowledge(coords, self.landscape[coords])

    def _update_visits(self):
        for agent in self.agents:
            self.visits[agent.coordinates] += 1



def makeTwoCliquesGraph(numAgents):
    """
    Generate a two cliques graph.

    Args:
        numAgents (int): Number of agents.

    Returns:
        list of lists: Adjacency matrix representing the two cliques graph.
    """
    m = []

    for i in range(numAgents):
        m.append([])
        for j in range(numAgents):
            if (i < numAgents / 2 and j < numAgents / 2) or (i >= numAgents / 2 and j >= numAgents / 2):
                m[i].append(1)
            else:
                m[i].append(0)

    m[0][numAgents - 1] = 1
    m[numAgents - 1][0] = 1

    return m

def makeStarGraph(numAgents):
    """
    Generate a star graph.

    Args:
        numAgents (int): Number of agents.

    Returns:
        list of lists: Adjacency matrix representing the star graph.
    """
    m = []
    for i in range(numAgents):
        m.append([])
        for j in range(numAgents):
            if i == 0 or j == 0 or i == j:
                m[i].append(1)
            else:
                m[i].append(0)
    return m

def makeCompleteGraph(numAgents):
    """
    Generate a complete graph.

    Args:
        numAgents (int): Number of agents.

    Returns:
        list of lists: Adjacency matrix representing the complete graph.
    """
    m = []
    for i in range(numAgents):
        m.append([])
        for j in range(numAgents):
            m[i].append(1)
    return m

def makeWheelGraph(numAgents):
    """
    Generate a wheel graph.

    Args:
        numAgents (int): Number of agents.

    Returns:
        list of lists: Adjacency matrix representing the wheel graph.
    """
    # m = []
    # for i in range(numAgents):
    #     m.append([])
    #     for j in range(numAgents):
    #         if i == 0 or j == 0 or i == j:
    #             m[i].append(1)
    #         elif j == (i + 1) % numAgents or j == (i - 1) % numAgents:
    #             m[i].append(1)
    #         else:
    #             m[i].append(0)
    # return m
    if numAgents < 4:
        raise ValueError("Number of agents must be at least 4 to form a wheel graph.")

    # Initialize an empty adjacency matrix
    m = [[0] * numAgents for _ in range(numAgents)]

    # Central node is at index 0, connects to all other nodes
    for i in range(1, numAgents):
        m[0][i] = 1
        m[i][0] = 1

    # Connect outer nodes in a ring (1 <-> 2 <-> 3 <-> ... <-> n-1 <-> 1)
    for i in range(1, numAgents - 1):
        m[i][i + 1] = 1
        m[i + 1][i] = 1

    # Connect the last node to the first outer node to complete the ring
    m[1][numAgents - 1] = 1
    m[numAgents - 1][1] = 1

    # make each agent connect to themselves
    for i in range(numAgents):
        m[i][i] = 1

    return m

def makeCycleGraph(numAgents):
    """
    Generate a cycle graph.

    Args:
        numAgents (int): Number of agents.

    Returns:
        list of lists: Adjacency matrix representing the cycle graph.
    """
    m = []
    for i in range(numAgents):
        m.append([])
        for j in range(numAgents):
            if i == j:
                m[i].append(1)
            elif j == (i + 1) % numAgents or j == (i - 1) % numAgents:
                m[i].append(1)
            else:
                m[i].append(0)
    return m

def makeLineGraph(numAgents):
    """
    Generate a line graph.

    Args:
        numAgents (int): Number of agents.

    Returns:
        list of lists: Adjacency matrix representing the line graph.
    """
    m = []
    for i in range(numAgents):
        m.append([])
        for j in range(numAgents):
            if i == j:
                m[i].append(1)
            elif j == (i + 1) or j == (i - 1):
                m[i].append(1)
            else:
                m[i].append(0)
    return m

def makeIsolatedGraph(numAgents):
    m = []
    for i in range(numAgents):
        m.append([])
        for j in range(numAgents):
            if i == j:
                m[i].append(1)
            else:
                m[i].append(0)
    return m