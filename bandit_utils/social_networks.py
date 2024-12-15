import numpy as np

class Network:
    def __init__(self, agents, machines, adjacencyMatrix, pulls_per_trial=1):
        """
        Initialize a network with agents, machines, and an adjacency matrix.

        Args:
            agents (list): List of Agent objects.
            machines (list): List of Machine objects.
            adjacencyMatrix (list of lists): Adjacency matrix representing connections between agents.
        """
        self.agents = agents
        self.machines = machines
        self.adjacencyMatrix = adjacencyMatrix
        self.pulls_per_trial = pulls_per_trial

    def getActs(self):
        """
        Get actions chosen by each agent.

        Returns:
            list: List of actions chosen by each agent.
        """
        acts = []
        for i in range(len(self.agents)):
            acts.append(self.agents[i].getMachineToPlay())
        return acts


    def getPayouts(self, acts, add_p_bonus=False, num_trials=1):
        """
        Get payouts obtained by each agent based on their actions.

        Args:
            acts (list): List of actions chosen by each agent.

        Returns:
            list: List of payouts obtained by each agent.
        """
        payouts = []
        for i in range(len(acts)):
            if add_p_bonus:
                payouts.append(self.machines[acts[i]].pull(self.agents[i].alphas[acts[i]]))
            else:
                #payouts.append(self.machines[acts[i]].pull(1))

                # For zollman setup:
                # arg: num_trials = 1000
                payouts.append(np.random.binomial(self.pulls_per_trial, self.machines[acts[i]].p))

        return payouts

    def step(self, num_trials=1):
        """
        Perform one step of interaction in the network.
        """
        acts = self.getActs()
        payouts = self.getPayouts(acts)

        for i, agent in enumerate(self.agents):
            agent.addReward(payouts[i])

        for i in range(len(self.adjacencyMatrix)):
            for j in range(len(self.adjacencyMatrix)):
                if self.adjacencyMatrix[i][j] == 1:
                    self.agents[i].update(acts[j], payouts[j], num_trials=self.pulls_per_trial)

    def hasConvergedTo(self, target_index):
        """
        Check if all agents have converged to a target machine.

        Args:
            target_index (int): Index of the target machine.

        Returns:
            bool: True if all agents have converged to the target machine, False otherwise.
        """
        for i in range(len(self.agents)):
            if self.agents[i].getBestMachine()[0] != target_index:
                return False
        return True

    def numConvergedTo(self, target_index):
        """
        Get the number of agents that have converged to a target machine.

        Args:
            target_index (int): Index of the target machine.

        Returns:
            int: Number of agents that have converged to the target machine.
        """
        count = 0
        for i in range(len(self.agents)):
            if self.agents[i].getBestMachine()[0] == target_index:
                count += 1
        return count

    def hasReachedConsensus(self):
        """
        Check if all agents have reached consensus.

        Returns:
            bool: True if all agents have chosen the same machine, False otherwise.
        """
        m = self.agents[0].getMachineToPlay(belief_only=True)
        return all(a.getMachineToPlay(belief_only=True) == m for a in self.agents)

    def getProbabilities(self):
        machine_list = []
        for machine_index in range(len(self.machines)):
            machine = self.machines[machine_index]
            agent_list = []
            for agent in self.agents:
                bonus = (agent.alphas[machine_index] - 1) * machine.p_bonus_per_success
                # Make sure the bonus doesn't exceed the maximum p value
                if machine.p + bonus > machine.max_p:
                    bonus = machine.max_p - machine.p

                agent_list.append(machine.p + bonus)
            machine_list.append(agent_list)
        return machine_list

                

class DummyNetwork(Network):
    def __init__(self, agents, machines, adjacencyMatrix):
        """
        Initialize a dummy network.

        Args:
            agents (list): List of Agent objects.
            machines (list): List of Machine objects.
            adjacencyMatrix (list of lists): Adjacency matrix representing connections between agents.
        """
        super().__init__(agents, machines, adjacencyMatrix)

    def hasDummyLearned(self, target_index):
        """
        Check if the dummy agent has learned the target machine.

        Args:
            target_index (int): Index of the target machine.

        Returns:
            bool: True if the dummy agent has learned the target machine, False otherwise.
        """
        return any(x == target_index for x in self.agents[0].getBestMachine())

    def step(self, add_p_bonus=False):
        """
        Perform one step of interaction in the dummy network.
        """
        acts = self.getActs()
        payouts = self.getPayouts(acts, add_p_bonus=add_p_bonus)


        for i in range(len(self.adjacencyMatrix)):
            for j in range(1, len(self.adjacencyMatrix)):
                if self.adjacencyMatrix[i][j] == 1:
                    self.agents[i].update(acts[j], payouts[j])


    def getDummyChoice(self):
        """
        Get the choice of the dummy agent.

        Returns:
            int: Index of the machine chosen by the dummy agent.
        """
        return self.agents[0].getMachineToPlay(belief_only=True)


class DisseminationDummyNetwork(DummyNetwork):
    def __init__(self, agents, machines, adjacency_matrices):
        """
        Initialize a dissemination dummy network.

        Args:
            agents (list): List of Agent objects.
            machines (list): List of Machine objects.
            adjacency_matrices (list of lists): List of adjacency matrices representing connections between agents for different machines.
        """
        super().__init__(agents, machines, adjacency_matrices[0])
        self.adjMatrices = adjacency_matrices

    def step(self, add_p_bonus=False):
        """
        Perform one step of interaction in the dissemination dummy network.
        """
        acts = self.getActs()
        payouts = self.getPayouts(acts, add_p_bonus=add_p_bonus)

        for i in range(len(acts)):
            for j in range(1, len(acts)):
                if self.adjMatrices[acts[j]][i][j] == 1:
                    self.agents[i].update(acts[j], payouts[j])



class ConductDummyNetwork(DummyNetwork):
    def __init__(self, agents, machines, adjacency_matrix, num_restricted):
        """
        Initialize a conduct dummy network.

        Args:
            agents (list): List of Agent objects.
            machines (list): List of Machine objects.
            adjacency_matrix (list of lists): Adjacency matrix representing connections between agents.
            num_restricted (int): Number of agents with restricted actions.
        """
        super().__init__(agents, machines, adjacency_matrix)
        self.num_restricted = num_restricted

    def step(self):
        """
        Perform one step of interaction in the conduct dummy network.
        """
        acts = self.getActs()

        for i in range(1, self.num_restricted + 1):
            acts[i] = 0

        payouts = self.getPayouts(acts)

        for i in range(len(acts)):
            for j in range(1, len(acts)):
                if self.adjacencyMatrix[i][j] == 1:
                    self.agents[i].update(acts[j], payouts[j])


class HybridDummyNetwork(DummyNetwork):
    def __init__(self, agents, machines, adjacency_matrices, num_restricted):
        """
        Initialize a hybrid dummy network.

        Args:
            agents (list): List of Agent objects.
            machines (list): List of Machine objects.
            adjacency_matrices (list of lists): List of adjacency matrices representing connections between agents for different machines.
            num_restricted (int): Number of agents with restricted actions.
        """
        super().__init__(agents, machines, adjacency_matrices[0])
        self.num_restricted = num_restricted
        self.adjMatrices = adjacency_matrices

    def step(self):
        """
        Perform one step of interaction in the hybrid dummy network.
        """
        acts = self.getActs()

        for i in range(1, self.num_restricted + 1):
            acts[i] = 0

        payouts = self.getPayouts(acts)

        for i in range(len(acts)):
            for j in range(1, len(acts)):
                if self.adjMatrices[acts[j]][i][j] == 1:
                    self.agents[i].update(acts[j], payouts[j])


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

if __name__ == "__main__":

    n = 4

    cyc = makeCycleGraph(n)
    w = makeWheelGraph(n)
    c = makeCompleteGraph(n)

    for i in w:
        print(i)

    print()

    for i in c:
        print(i)


    print()
    for i in cyc:
        print(i)