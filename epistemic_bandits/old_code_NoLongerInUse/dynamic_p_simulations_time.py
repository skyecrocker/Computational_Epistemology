filename = 'computational_epistemology/bernoulli_simulations/output/dynamic_steps_priors33_other6_target8_steps1000.csv'

# Importing the necessary modules
import sys
sys.path.append('computational_epistemology/bandit_utils')
import argparse
import random
import slot_machines
import agents as beta_agents
import social_networks
import csv

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

# Default params
num_machines = 2
priors = 'uniform'
runs = 10
steps = 1000
num_agents = 9
which_arm_restricted_params = ['random'] #['random', 'other', 'target']

# Modification Params
prior_ps = [.33, .33]
max_ps = [.6, .8]
p_bonus_per_success = 0.01

# Creating the social networks
complete_graph = social_networks.makeCompleteGraph(num_agents)
star_graph = social_networks.makeStarGraph(num_agents)

# Creating the machine lists
machine_list = [slot_machines.BernoulliMachine(prior_p, max_p, p_bonus_per_success) for prior_p, max_p in zip(prior_ps, max_ps)]
machine_list_flipped = [slot_machines.BernoulliMachine(prior_p, max_p, p_bonus_per_success) for prior_p, max_p in zip(reversed(prior_ps), reversed(max_ps))]

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


# Creating the machine lists
# Setup the machines with new p value
        machine_list = [slot_machines.BernoulliMachine(prior_p, max_p, p_bonus_per_success) for prior_p, max_p in zip(prior_ps, max_ps)]
        machine_list_flipped = [slot_machines.BernoulliMachine(prior_p, max_p, p_bonus_per_success) for prior_p, max_p in zip(reversed(prior_ps), reversed(max_ps))]

with open(filename, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["scenario", "which_arm_restricted", 'num_agents', 'sim_id', 'step_id', 'agent_id', 'other_p_with_bonus', 'target_p_with_bonus', 'choice'])

    # Baseline, complete graph, no restrictions
    for r in range(runs):
        target = 0 if max_ps[0] > max_ps[1] else 1
        network = social_networks.DummyNetwork(agent_list, machine_list, complete_graph)

        for agent in agent_list:
            agent.reset()

        dummy_choices = []
        for t in range(steps):
            current_ps = network.getProbabilities()
            network.step(add_p_bonus=True)
            dummy_choices.insert(0, network.getDummyChoice())
            for i, agent in enumerate(agent_list):
                writer.writerow(['baseline', 'N/A', num_agents, r, t, i, current_ps[0][i], current_ps[1][i], dummy_choices[0]])

        last_choice = dummy_choices.pop(0)
        flipped = 1 if last_choice == 0 else 0
        ttl = dummy_choices.index(flipped) if flipped in dummy_choices else -1
        print(ttl)
        if ttl == -1:
            ttl = 1
        else:
            ttl = steps - ttl + 1
        print(ttl)
        print("--------------")



    # Restricting Dissemination
    for which_arm_restricted in which_arm_restricted_params:
        for r in range(runs):
            target = 0 if max_ps[0] > max_ps[1] else 1
            network = social_networks.DisseminationDummyNetwork(agent_list, machine_list, [complete_graph, star_graph])

            flip_machines = False
            if which_arm_restricted == "random":
                if random.random() < 0.5:
                    flip_machines = True
                    network = social_networks.DisseminationDummyNetwork(agent_list, machine_list_flipped, [complete_graph, star_graph])
                    target = 0 if target == 1 else 1
            elif which_arm_restricted == "other":
                    flip_machines = True
                    network = social_networks.DisseminationDummyNetwork(agent_list, machine_list_flipped, [complete_graph, star_graph])
                    target = 0 if target == 1 else 1

            for agent in agent_list:
                agent.reset()

            dummy_choices = []
            for t in range(steps):
                current_ps = network.getProbabilities()
                network.step(add_p_bonus=True)
                dummy_choices.insert(0, network.getDummyChoice())
                dummy_choice = dummy_choices[0]
                if flip_machines:
                    current_ps = list(reversed(current_ps))
                    dummy_choice = 1 - dummy_choice
                for i, agent in enumerate(agent_list):
                    writer.writerow(['restrict_dissemination', 'other' if flip_machines else 'target', num_agents, r, t, i, current_ps[0][i], current_ps[1][i], dummy_choice])

            last_choice = dummy_choices.pop(0)
            flipped = 1 if last_choice == 0 else 0
            ttl = dummy_choices.index(flipped) if flipped in dummy_choices else -1
            print(ttl)
            if ttl == -1:
                ttl = 1
            else:
                ttl = steps - ttl + 1
            print(ttl)
            print("--------------")


    '''
    # Restricting Conduct
    for r in range(runs):
        target = 0 if ps[0] > ps[1] else 1
        network = social_networks.ConductDummyNetwork(agent_list, machine_list, complete_graph, 1)

        if which_arm_restricted == "randomize":
            if random.random() < 0.5:
                network = social_networks.ConductDummyNetwork(agent_list, machine_list_flipped, complete_graph, 1)
                target = 0 if target == 1 else 1

        for agent in agent_list:
            agent.reset()

        dummy_choices = []
        for t in range(steps):
            network.step()
            dummy_choices.insert(0, network.getDummyChoice())

        last_choice = dummy_choices.pop(0)
        flipped = 1 if last_choice == 0 else 0
        time_to_lock = dummy_choices.index(flipped) if flipped in dummy_choices else -1

        if time_to_lock == -1:
            time_to_lock = 1
        else:
            time_to_lock = steps - time_to_lock + 1

        if last_choice == target:
            lock_times["cond_succ"].append(time_to_lock)
        else:
            lock_times["cond_inc"].append(time_to_lock)

    # Restricting Both
    for r in range(runs):
        target = 0 if ps[0] > ps[1] else 1
        network = social_networks.HybridDummyNetwork(agent_list, machine_list, [complete_graph, star_graph], 1)

        if which_arm_restricted == "randomize":
            if random.random() < 0.5:
                network = social_networks.HybridDummyNetwork(agent_list, machine_list_flipped, [complete_graph, star_graph], 1)
                target = 0 if target == 1 else 1

        for agent in agent_list:
            agent.reset()

        dummy_choices = []
        for t in range(steps):
            network.step()
            dummy_choices.insert(0, network.getDummyChoice())

        last_choice = dummy_choices.pop(0)
        flipped = 1 if last_choice == 0 else 0
        time_to_lock = dummy_choices.index(flipped) if flipped in dummy_choices else -1

        if time_to_lock == -1:
            time_to_lock = 1
        else:
            time_to_lock = steps - time_to_lock + 1

        if last_choice == target:
            lock_times["hybrid_succ"].append(time_to_lock)
        else:
            lock_times["hybrid_inc"].append(time_to_lock)
    '''

