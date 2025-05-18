#!/usr/bin/env python2.7
from numpy.random import binomial, random
from numpy import mean, std, divide, prod, power
import csv

class Player:
    def __init__(self, belief):
        self.belief = belief
        #degree of belief in World 2 (in which expected return for informative action is .5 + epsilon)
        self.result = None
        self.friends = []
        #people whose results player can update on
    def update(self, n, eps):
        #updates player belief using Bayes rule based on binomial(n, 0.5 + eps) trial result
        self.n = n
        self.eps = eps
        for x in self.friends:
            if not x.result == None:
                top = prod([1-self.belief, power(divide(0.5 - self.eps, 0.5 + self.eps) , 2*x.result - self.n) ])
                bayes = divide(1, 1 + divide(top, self.belief))
                self.belief = bayes
           

class Game:
    def __init__(self, popsize, network, binom_n, epsilon):
        self.popsize = popsize
        self.network = network
        # options: cycle, complete, or rando. For rando: each round, players look at 3 random results. 
        ##Idea is to show that rando is equivalent to cycle, so latter has nothing to do with network structure
        self.binom_n = binom_n
        #binom_n = number of trials each round
        self.epsilon = epsilon
        #epsilon = difference between two actions- in world 2, option 2 gives .5 + epsilon, in world 1, gives .5 - epsilon. 
        # action 1 is uninformative, gives .5 in both worlds
        self.players = []
        self.converged = 0
        self.round_converged = []
        self.avg_round = None
        self.portion_converged = None
        self.st_dev = None

    def round(self):
        for x in self.players:
            if x.belief >= 0.5:
                x.result = binomial(self.binom_n, 0.5 + self.epsilon)
            #generates random outcome for players who believe they are in world 2 based on binomial distribution with p = epsilon + .5, n & epsilon given in input
            else:
                x.result = None
        for x in self.players:
            x.update(self.binom_n, self.epsilon)

    def trial(self):
        self.players = []
        for x in range(self.popsize):
            self.players.append(Player(random()))
            #creates players with uniform random initial degrees of belief in world 2
        for x in range(self.popsize):
            if self.network == "cycle":
                self.players[x].friends = [self.players[x], self.players[(x - 1) % self.popsize], self.players[(x + 1) % self.popsize]]
            if self.network == "star":
                if x == 0:
                    self.players[x].friends = self.players
                else:
                    self.players[x].friends = [self.players[x]]
            if self.network == "complete":
                self.players[x].friends = self.players
            #creates network neighbor list for players based on network type
        for i in range(10000):
            self.round()
            if all(x.belief > 0.99 for x in self.players):
                self.converged += 1
                self.round_converged.append(i+1)
                break
            if all(x.belief < 0.5 for x in self.players):
                break
            #runs trials 10000 times or until degrees of belief converged

    def run(self, times):
        self.times = times
        for i in range(times):
            self.trial()
        self.portion_converged = float(self.converged)/float(self.times)
        if self.converged > 0:
            self.avg_round = mean(self.round_converged)
            self.st_dev = std(self.round_converged)
        else:
            self.avg_round = "na"
            self.st_dev = "na"
        #runs "times" number of trials with different initial random degrees of belief


class Record:
    def __init__(self, population_sizes, epsilon_sizes, network_types, binomial_n):
        self.population_sizes = population_sizes
        self.epsilon_sizes = epsilon_sizes
        self.network_types = network_types
        self.binomial_n = binomial_n

    def record_run(self, rounds, file_name):
        self.rounds = rounds
        self.file_name = file_name
        with open(self.file_name, 'a') as f:
            writer = csv.writer(f)
            writer.writerows([["population size", "network type", "epsilon value", "binomial n", "portion converged", "average round converged", "standard deviation"]])
        for x in self.population_sizes:
            for y in self.epsilon_sizes:
                for z in self.network_types:
                    for n in self.binomial_n:
                        game = Game(x, z, n, y)
                        game.run(self.rounds)
                        with open(self.file_name, 'a') as f:
                            writer = csv.writer(f)
                            writer.writerows([[x, z, y, n, game.portion_converged, game.avg_round, game.st_dev]])
            #runs for various initial conditions, prints outcome and creates csv file
            



###TEST 


pop_sizes_test = list(range(4, 11))
ep_sizes_test = [0.001]
network_types_test = ["complete", "cycle", "star"]
binomial_n_test = [1000]

record_test = Record(pop_sizes_test, ep_sizes_test, network_types_test, binomial_n_test)

record_test.record_run(100, "./recreate_1000.csv")
### runs 100 times for each combination of paramters in test lists above. 
### we ran 10000+ times for many more parameter values, but this takes a long time
### the first entry here is the number of rounds run for each tuple (pop_sizes, ep_sizes, etc...), the second is file path
### output will be a csv file with entries for: "population size", "network type", "epsilon value", "binomial n", "portion converged", "average round converged", "standard deviation"


