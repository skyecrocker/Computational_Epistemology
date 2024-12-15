filename = 'computational_epistemology/bernoulli_simulations/output/proto_results1000.csv'

# Importing the necessary modules
import sys
sys.path.append('computational_epistemology/bandit_utils')
import argparse
import random
import slot_machines
import agents as beta_agents
import social_networks
import csv
import numpy as np
from time_utils import time_to_lock

'''
# Parsing command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--priors', default='uniform', help='Priors for agents')
parser.add_argument('-r', '--runs', type=int, default=100, help='Number of runs')
parser.add_argument('-s', '--steps', type=int, default=100, help='Number of steps')
parser.add_argument('-n', '--num_agents', type=int, default=9, help='Number of agents')
parser.add_argument('-c', '--which_arm_restricted', default='randomize', help='Which arm is restricted')
parser.add_argument('-q', type=float, default=0.6, help='Value of q')
args = parser.parse_args()

# Extracting command line arguments
priors = args.priors
runs = args.runs
steps = args.steps
num_agents = args.num_agents
which_arm_restricted = args.which_arm_restricted
q = args.q
'''

# classic params
priors = 'uniform'
runs = 1000
steps = 1000
num_agents = 9
which_arm_restricted = 'random'
#q = 0.6


# Creating the social networks
complete_graph = social_networks.makeCompleteGraph(num_agents)
star_graph = social_networks.makeStarGraph(num_agents)

machine_list = [0, 0] # dummy initialization, will be replaced later

# Creating the agent lists
agent_list = []
for i in range(num_agents):
    if priors == "uniform" or priors == "u":
        agent_list.append(beta_agents.BetaAgentUniformPriors(len(machine_list)))
    elif priors == "jeffrey" or priors == "j":
        agent_list.append(beta_agents.BetaAgentJeffreyPriors(len(machine_list)))
    elif priors == "random" or priors == "r":
        agent_list.append(beta_agents.BetaAgentRandomPriors(len(machine_list)))
    else:
        agent_list.append(beta_agents.BetaAgentUniformPriors(len(machine_list)))

# Set range for variable to test
p_values = np.arange(.51, 1, .01)

# Writing results to a CSV file
with open(filename, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['scenario', 'network_structure', 'arm_restricted', 'num_agents_restricted', 'p', 'num_agents', 'n_steps', 'n_success', 'n_runs', 'avg_time_to_converge'])

    params_tested = 0
    for q in p_values:

        # Setup the machines with new p value
        ps = [0.5, q]
        machine_list = [slot_machines.BernoulliMachine(p) for p in ps]
        machine_list_flipped = [slot_machines.BernoulliMachine(p) for p in reversed(ps)]

        ################################################################################
        ############## Baseline, complete graph, no restrictions #######################
        ################################################################################

        n_success = 0
        avg_time_to_converge = 0

        for r in range(runs):
            target = 0 if ps[0] > ps[1] else 1
            
            network = social_networks.DummyNetwork(agent_list, machine_list, complete_graph)

            for agent in agent_list:
                agent.reset()

            dummy_choices = []
            for t in range(steps):
                network.step()
                dummy_choices.insert(0, network.getDummyChoice())

            last_choice = dummy_choices.pop(0)
            if last_choice == target:
                n_success += 1

            avg_time_to_converge += time_to_lock(dummy_choices, steps)

        avg_time_to_converge /= runs
        writer.writerow(['baseline', 'complete', 'N/A', 'N/A', q, num_agents, steps, n_success, runs, avg_time_to_converge])

        ###############################################################################
        ######################### Restricting Dissemination ###########################
        ###############################################################################
        n_success = 0
        avg_time_to_converge = 0

        for r in range(runs):
            target = 0 if ps[0] > ps[1] else 1
            network = social_networks.DisseminationDummyNetwork(agent_list, machine_list, [complete_graph, star_graph])

            if which_arm_restricted == "randomize":
                if random.random() < 0.5:
                    network = social_networks.DisseminationDummyNetwork(agent_list, machine_list_flipped, [complete_graph, star_graph])
                    target = 0 if target == 1 else 1

            for agent in agent_list:
                agent.reset()

            dummy_choices = []
            for t in range(steps):
                network.step()
                dummy_choices.insert(0, network.getDummyChoice())

            last_choice = dummy_choices.pop(0)
            if last_choice == target:
                n_success += 1
            flipped = 1 if last_choice == 0 else 0
            avg_time_to_converge += dummy_choices.index(flipped) if flipped in dummy_choices else -1

        avg_time_to_converge /= runs
        writer.writerow(['restrict_dissemination', 'star', 'random', 'N/A', q, num_agents, steps, n_success, runs, avg_time_to_converge])


'''
        ###############################################################################
        ############################ Restricting Conduct ##############################
        ###############################################################################
        n_success = 0
        avg_time_to_converge = 0
        num_restricted = 2

        for r in range(runs):
            target = 0 if ps[0] > ps[1] else 1
            network = social_networks.ConductDummyNetwork(agent_list, machine_list, complete_graph, num_restricted)

            if which_arm_restricted == "randomize":
                if random.random() < 0.5:
                    network = social_networks.ConductDummyNetwork(agent_list, machine_list_flipped, complete_graph, num_restricted)
                    target = 0 if target == 1 else 1

            for agent in agent_list:
                agent.reset()

            dummy_choices = []
            for t in range(steps):
                network.step()
                dummy_choices.insert(0, network.getDummyChoice())

            last_choice = dummy_choices.pop(0)
            if last_choice == target:
                n_success += 1
            flipped = 1 if last_choice == 0 else 0
            avg_time_to_converge += dummy_choices.index(flipped) if flipped in dummy_choices else -1

        avg_time_to_converge /= runs
        writer.writerow(['restrict_conduct', 'complete', 'random', num_restricted, q, num_agents, steps, n_success, runs, avg_time_to_converge])

        ###############################################################################
        ############################# Restricting Both ################################
        ###############################################################################
        n_success = 0
        avg_time_to_converge = 0

        for r in range(runs):
            target = 0 if ps[0] > ps[1] else 1
            network = social_networks.HybridDummyNetwork(agent_list, machine_list, [complete_graph, star_graph], num_restricted)

            if which_arm_restricted == "randomize":
                if random.random() < 0.5:
                    network = social_networks.HybridDummyNetwork(agent_list, machine_list_flipped, [complete_graph, star_graph], num_restricted)
                    target = 0 if target == 1 else 1

            for agent in agent_list:
                agent.reset()

            dummy_choices = []
            for t in range(steps):
                network.step()
                dummy_choices.insert(0, network.getDummyChoice())

            last_choice = dummy_choices.pop(0)
            if last_choice == target:
                n_success += 1
            flipped = 1 if last_choice == 0 else 0
            avg_time_to_converge += dummy_choices.index(flipped) if flipped in dummy_choices else -1

        avg_time_to_converge /= runs
        writer.writerow(['restrict_both', 'star', 'random', num_restricted, q, num_agents, steps, n_success, runs, avg_time_to_converge])

        # Show progress
        params_tested += 1
        if params_tested % 10 == 0:
            print(f'{params_tested} parameters tested out of {len(p_values)}')
'''