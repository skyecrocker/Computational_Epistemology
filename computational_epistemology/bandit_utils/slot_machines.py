import math
import random

class NormalMachine:
    """
    Represents a normal distribution slot machine.
    """

    def __init__(self, mean, variance):
        """
        Initializes the NormalMachine with the given mean and variance.
        """
        self.mean = mean
        self.variance = variance

    def erf(self, x):
        """
        Polynomial approximation for the error function (erf).
        """
        a = [0.278393, 0.230389, 0.000972, 0.078108]
        temp = 1 + a[0]*x + a[1]*x*x + a[2]*x*x*x + a[3]*x*x*x*x
        return 1 - 1 / math.pow(temp, 4)

    def erf_inverse(self, x):
        """
        Calculates the inverse of the error function (erf).
        """
        a = 0.140012
        t1 = 2 / (math.pi * a) + math.log(1 - x*x) / 2
        t2 = math.log(1 - x*x) / a
        return math.copysign(1, x) * math.sqrt(math.sqrt(t1*t1 - t2) - t1)

    def quantile_function(self, p):
        """
        Calculates the quantile function for the given probability (p).
        """
        return self.mean + math.sqrt(self.variance * 2) * self.erf_inverse(2*p - 1)

    def pull(self):
        """
        Pulls the slot machine and returns a random value based on the normal distribution.
        """
        r = random.random()
        return self.quantile_function(r)


class BernoulliMachine:
    """
    Represents a Bernoulli distribution slot machine.
    """

    def __init__(self, p, max_p=1.0, p_bonus_per_success=0.01):
        """
        Initializes the BernoulliMachine with the given probability (p).
        """
        self.p = p
        self.max_p = max_p
        self.p_bonus_per_success = p_bonus_per_success


    def pull(self, alpha=0):
        """
        Pulls the slot machine and returns either 1 or 0 based on the Bernoulli distribution.

        Args:
            alpha (int): The agents alpha value. Increase the p values based on the number of successesful pulls known by the agent.
        """
        bonus = (alpha - 1) * self.p_bonus_per_success

        # Make sure the bonus doesn't exceed the maximum p value
        if self.p + bonus > self.max_p:
            bonus = self.max_p - self.p

        r = random.random()
        if r < self.p + bonus:
            #print("alpha: " + str(alpha), "p_bonus: " + str(self.p + bonus), "max_p: " + str(self.max_p), "r: 1")
            return 1
        #print("alpha: " + str(alpha), "p_bonus: " + str(self.p + bonus), "max_p: " + str(self.max_p), "r: 0")
        return 0
    
