import random
import math

class Agent:
    def __init__(self, numMachines):
        """
        Initialize an Agent object.

        Args:
            numMachines (int): Number of machines the agent interacts with.
        """
        self.numMachines = numMachines

class BetaAgent(Agent):
    def __init__(self, numMachines, resiliance=0, epsilon=-1):
        """
        Initialize a BetaAgent object.

        Args:
            numMachines (int): Number of machines the agent interacts with.
        """
        super().__init__(numMachines)
        self.alphas = [0] * numMachines
        self.betas = [0] * numMachines
        self.resiliance = resiliance
        self.epsilon = epsilon

        self.totalReward = 0

        try:
            assert(resiliance == 0 or epsilon == -1)
        except:
            raise ValueError("Only one of resiliance or epsilon can be non-zero")

        self.reset()
        self.machineToPlay = random.choice(self.getBestMachine())
        self.stepsLeftUntilSwitch = resiliance

    def resetRandomInterval(self, aInterval, bInterval):
        """
        Reset the agent's parameters with random intervals.

        Args:
            aInterval (list): Interval for alpha parameters.
            bInterval (list): Interval for beta parameters.
        """
        for i in range(len(self.alphas)):
            self.alphas[i] = aInterval[0] + (aInterval[1] - aInterval[0]) * random.random()
            self.betas[i] = bInterval[0] + (bInterval[1] - bInterval[0]) * random.random()
        self.machineToPlay = random.choice(self.getBestMachine())
        self.stepsLeftUntilSwitch = self.resiliance
        self.totalReward = 0

    def resetJeffreyPriors(self):
        """
        Reset the agent's parameters with Jeffrey priors.
        """
        for i in range(len(self.alphas)):
            self.alphas[i] = 0.5
            self.betas[i] = 0.5
        self.machineToPlay = random.choice(self.getBestMachine())
        self.stepsLeftUntilSwitch = self.resiliance
        self.totalReward = 0

    def resetUniformPriors(self):
        """
        Reset the agent's parameters with uniform priors.
        """
        for i in range(len(self.alphas)):
            self.alphas[i] = 1
            self.betas[i] = 1
        self.machineToPlay = random.choice(self.getBestMachine())
        self.stepsLeftUntilSwitch = self.resiliance
        self.totalReward = 0

    def reset(self):
        """
        Reset the agent's parameters with uniform priors.
        """
        self.resetUniformPriors()

    def update(self, machineIndex, payout, num_trials=1):
        """
        Update the agent's parameters based on the outcome of an interaction.

        Args:
            machineIndex (int): Index of the machine played.
            payout (float): Payout received from the machine.
        """
        self.alphas[machineIndex] += payout
        self.betas[machineIndex] += num_trials - payout

    def getMachineToPlay(self, belief_only=False):
        """
        Choose a machine to play based on the agent's strategy.

        Returns:
            int: Index of the machine chosen.
        """
        if belief_only:
            best_list = self.getBestMachine()
            return random.choice(best_list)
        elif self.epsilon != -1:
            return self.getMachineToPlay_EpsilonGreedySetup()
        elif self.resiliance != 0:
           return self.getMachineToPlay_InertiaSetup()
        else:
            best_list = self.getBestMachine()
            return random.choice(best_list)

    def getMachineToPlay_InertiaSetup(self):
        choice = random.choice(self.getBestMachine())
        if self.machineToPlay == choice:
            self.stepsLeftUntilSwitch = self.resiliance
            return self.machineToPlay
        else:
            if self.stepsLeftUntilSwitch == 0:
                self.machineToPlay = choice
                self.stepsLeftUntilSwitch = self.resiliance
                return self.machineToPlay
            else:
                self.stepsLeftUntilSwitch -= 1
                return self.machineToPlay
            
    def getMachineToPlay_EpsilonGreedySetup(self):
        best_list = self.getBestMachine()
        best_choice =  random.choice(best_list)
        if random.random() < self.epsilon:
            return 1 if best_choice == 0 else 0
        else:
            return best_choice

    def getBestMachine(self):
        """
        Determine the best machine(s) based on the agent's strategy.

        Returns:
            list: Indices of the best machine(s).
        """
        exps = [self.alphas[i] / (self.alphas[i] + self.betas[i]) for i in range(len(self.alphas))]
        m = max(exps)
        bests = [i for i in range(len(exps)) if exps[i] >= m]
        return bests

    def addReward(self, reward):
        """
        Add reward to the agent's total reward.

        Args:
            reward (float): Reward to add.
        """
        self.totalReward += reward

# Subclasses of BetaAgent for different prior settings:

class BetaAgentUniformPriors(BetaAgent):
    def __init__(self, numMachines, resiliance=0, epsilon=-1):
        super().__init__(numMachines, resiliance, epsilon)

    def reset(self):
        self.resetUniformPriors()

class BetaAgentJeffreyPriors(BetaAgent):
    def __init__(self, numMachines, resiliance=0, epsilon=-1):
        super().__init__(numMachines, resiliance, epsilon)

    def reset(self):
        self.resetJeffreyPriors()

class BetaAgentRandomPriors(BetaAgent):
    def __init__(self, numMachines, resiliance=0, epsilon=-1):
        super().__init__(numMachines, resiliance, epsilon)

    def reset(self):
        self.resetRandomInterval([0, 4], [0, 4])

# Classes for agents with normal distribution:

class NormalAgentKnownVariance(Agent):
    def __init__(self, numMachines, knownVariances):
        """
        Initialize a NormalAgentKnownVariance object.

        Args:
            numMachines (int): Number of machines the agent interacts with.
            knownVariances (list): List of known variances for each machine.
        """
        super().__init__(numMachines)
        self.knownVariances = knownVariances
        self.means = [0] * numMachines
        self.variances = [0] * numMachines
        self.reset()

    def resetWith(self, meanInterval, varianceInterval):
        """
        Reset the agent's parameters with given intervals for means and variances.

        Args:
            meanInterval (list): Interval for means.
            varianceInterval (list): Interval for variances.
        """
        for i in range(len(self.means)):
            self.means[i] = meanInterval[0] + (meanInterval[1] - meanInterval[0]) * random.random()
            self.variances[i] = varianceInterval[0] + (varianceInterval[1] - varianceInterval[0]) * random.random()

    def resetImproper(self):
        """
        Reset the agent's parameters with improper settings.
        """
        for i in range(len(self.means)):
            self.means[i] = 0
            self.variances[i] = float('inf')

    def reset(self):
        """
        Reset the agent's parameters with improper settings.
        """
        self.resetImproper()

    def update(self, machine, payout):
        """
        Update the agent's parameters based on the outcome of an interaction.

        Args:
            machine (int): Index of the machine played.
            payout (float): Payout received from the machine.
        """
        n = 1

        if self.variances[machine] == float('inf'):
            self.means[machine] = (0 + payout / self.knownVariances[machine]) / (
                        0 + 1 / self.knownVariances[machine])
            self.variances[machine] = 1 / math.sqrt(1 / self.knownVariances[machine])
            return

        self.means[machine] = (n * self.variances[machine]) / (
                    n * self.variances[machine] + self.knownVariances[machine]) * payout + (
                                      self.knownVariances[machine]) / (
                                      n * self.variances[machine] + self.knownVariances[machine]) * self.means[machine]

        self.variances[machine] = (self.knownVariances[machine] * self.variances[machine]) / (
                    n * self.variances[machine] + self.knownVariances[machine])

    def getMachineToPlay(self):
        """
        Choose a machine to play based on the agent's strategy.

        Returns:
            int: Index of the machine chosen.
        """
        return self.getBestMachine()

    def getBestMachine(self):
        """
        Determine the best machine(s) based on the agent's strategy.

        Returns:
            list: Indices of the best machine(s).
        """
        m = max(self.means)
        bests = [i for i in range(len(self.means)) if self.means[i] >= m]
        return random.choice(bests)

# Using the method from the Clark paper online

class NormalAgentUnknownMeanAndVariance(Agent):
    def __init__(self, numMachines):
        """
        Initialize a NormalAgentUnknownMeanAndVariance object.

        Args:
            numMachines (int): Number of machines the agent interacts with.
        """
        super().__init__(numMachines)
        self.alphas = [0] * numMachines
        self.betas = [0] * numMachines
        self.gammas = [0] * numMachines
        self.reset()

    def reset(self):
        """
        Reset the agent's parameters.
        """
        self.alphas = [0] * self.numMachines
        self.betas = [0] * self.numMachines
        self.gammas = [0] * self.numMachines

    def update(self, machine, payout):
        """
        Update the agent's parameters based on the outcome of an interaction.

        Args:
            machine (int): Index of the machine played.
            payout (float): Payout received from the machine.
        """
        self.alphas[machine] += 1
        self.betas[machine] += payout
        self.gammas[machine] += payout * payout

    def getMachineToPlay(self):
        """
        Choose a machine to play based on the agent's strategy.

        Returns:
            int: Index of the machine chosen.
        """
        return self.getBestMachine()

    def getBestMachine(self):
        """
        Determine the best machine(s) based on the agent's strategy.

        Returns:
            list: Indices of the best machine(s).
        """
        expMeans = [self.betas[i] / self.alphas[i] if self.alphas[i] != 0 else 0 for i in range(len(self.alphas))]
        m = max(expMeans)
        bests = [i for i in range(len(expMeans)) if expMeans[i] >= m]
        return random.choice(bests)

    def sampleMeansList(self):
        """
        Get a list of sample means.

        Returns:
            list: List of sample means for each machine.
        """
        l = [self.betas[i] / self.alphas[i] if self.alphas[i] > 0 else 0 for i in range(len(self.alphas))]
        return l

    def sSquaredList(self):
        """
        Get a list of sample variances.

        Returns:
            list: List of sample variances for each machine.
        """
        l = [(self.alphas[i] * self.gammas[i] - self.betas[i] * self.betas[i]) / (
                    self.alphas[i] * (self.alphas[i] - 1)) if self.alphas[i] > 2 else 1 for i in
             range(len(self.alphas))]
        return l

    def varOfMuList(self):
        """
        Get a list of variances of sample means.

        Returns:
            list: List of variances of sample means for each machine.
        """
        l = self.sSquaredList()
        for i in range(len(l)):
            l[i] = l[i] / self.alphas[i]
        return l
