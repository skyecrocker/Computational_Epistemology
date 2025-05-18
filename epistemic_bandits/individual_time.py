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

# Printing command line arguments
print("# Priors: " + priors + "; runs: " + str(runs) + "; steps: " + str(steps))
print("# Which arm is restricted: " + which_arm_restricted)
print("# q: " + str(q))
'''

output_file = 'c:/Users/skyec/Desktop/Projects/Research/computational_epistemology/bernoulli_simulations/output/time_nRange_p51.csv'

# Default params
num_machines = 2
priors = 'uniform'
runs = 1000
steps = 100
num_agents = 9
p = 0.51
which_arm_restricted = 'random'

p_values = np.arange(0.5, 0.6, 0.01)
num_agents_values = np.arange(3, 17, 1)


with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    metadata_header = ['scenario', 'which_arm_restricted', 'num_agents', 'p', 'sim_id', 'agent_id', 'was_correct', 'time_to_lock']
    writer.writerow(metadata_header)

    #for p in p_range:
    for num_agents in num_agents_values:

        ps = [0.5, p]

        # Creating the social networks
        complete_graph = social_networks.makeCompleteGraph(num_agents)
        star_graph = social_networks.makeStarGraph(num_agents)

        # Creating the machine lists
        machine_list = [slot_machines.BernoulliMachine(p) for p in ps]
        machine_list_flipped = [slot_machines.BernoulliMachine(p) for p in reversed(ps)]

        # Creating the agent lists
        agent_list = []
        for i in range(num_agents):
            if priors == "uniform" or priors == "u":
                agent_list.append(beta_agents.BetaAgentUniformPriors(num_machines))
            elif priors == "jeffrey" or priors == "j":
                agent_list.append(beta_agents.BetaAgentJeffreyPriors(num_machines))
            elif priors == "random" or priors == "r":
                agent_list.append(beta_agents.BetaAgentRandomPriors(num_machines))
            else:
                agent_list.append(beta_agents.BetaAgentUniformPriors(num_machines))

        n_success = 0

        # Baseline, complete graph, no restrictions
        for r in range(runs):
            target = 0 if ps[0] > ps[1] else 1
            network = social_networks.DummyNetwork(agent_list, machine_list, complete_graph)

            for agent in agent_list:
                agent.reset()

            agents_choices = [[] for _ in agent_list]
            for t in range(steps):
                network.step()
                for i, agent in enumerate(agent_list):
                    agents_choices[i].insert(0, agent.getMachineToPlay())

            last_choices = [choices.pop(0) for choices in agents_choices]
            flipped = [1 if last_choice == 0 else 0 for last_choice in last_choices]
            time_to_locks = [choices.index(f) if f in choices else -1 for choices, f in zip(agents_choices, flipped)]

            for i, agent in enumerate(agent_list):
                if time_to_locks[i] == -1:
                    ttl = 1
                else:
                    ttl = steps - time_to_locks[i] + 1

                rowdata = ['baseline', 'N/A', num_agents, p, r, i, last_choices[i] == target, ttl]
                writer.writerow(rowdata)

            if last_choices[0] == target:
                n_success += 1

        print("Baseline success rate: ", n_success / runs)

        n_success = 0
        
        # Restricting Dissemination
        for r in range(runs):
            target = 0 if ps[0] > ps[1] else 1
            network = social_networks.DisseminationDummyNetwork(agent_list, machine_list, [complete_graph, star_graph])

            if which_arm_restricted == "random":
                if random.random() < 0.5:
                    network = social_networks.DisseminationDummyNetwork(agent_list, machine_list_flipped, [complete_graph, star_graph])
                    target = 0 if target == 1 else 1

            for agent in agent_list:
                agent.reset()

            agents_choices = [[] for _ in agent_list]
            for t in range(steps):
                network.step()
                for i, agent in enumerate(agent_list):
                    agents_choices[i].insert(0, agent.getMachineToPlay())

            last_choices = [choices.pop(0) for choices in agents_choices]
            flipped = [1 if last_choice == 0 else 0 for last_choice in last_choices]
            time_to_locks = [choices.index(f) if f in choices else -1 for choices, f in zip(agents_choices, flipped)]

            for i, agent in enumerate(agent_list):
                if time_to_locks[i] == -1:
                    ttl = 1
                else:
                    ttl = steps - time_to_locks[i] + 1

                what_arm = 'target' if target == 1 else 'other'
                rowdata = ['restrict_dissemination', what_arm, num_agents, p, r, i, last_choices[i] == target, ttl]

                writer.writerow(rowdata)

            if last_choices[0] == target:
                n_success += 1

        print("Restricting dissemination success rate: ", n_success / runs)

