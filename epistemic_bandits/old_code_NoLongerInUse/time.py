# Importing the necessary modules
import sys
sys.path.append('computational_epistemology/bandit_utils')
import argparse
import random
import slot_machines
import agents as beta_agents
import social_networks
import csv

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

# Defining the probabilities for the slot machines
ps = [0.5, q]

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
        agent_list.append(beta_agents.BetaAgentUniformPriors(len(machine_list)))
    elif priors == "jeffrey" or priors == "j":
        agent_list.append(beta_agents.BetaAgentJeffreyPriors(len(machine_list)))
    elif priors == "random" or priors == "r":
        agent_list.append(beta_agents.BetaAgentRandomPriors(len(machine_list)))
    else:
        agent_list.append(beta_agents.BetaAgentUniformPriors(len(machine_list)))

# Initializing lock times dictionary
lock_times = {
    "comp_succ": [],
    "comp_inc": [],
    "diss_succ": [],
    "diss_inc": [],
    "cond_succ": [],
    "cond_inc": [],
    "hybrid_succ": [],
    "hybrid_inc": []
}

# Baseline, complete graph, no restrictions
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
    flipped = 1 if last_choice == 0 else 0
    time_to_lock = dummy_choices.index(flipped) if flipped in dummy_choices else -1

    if time_to_lock == -1:
        time_to_lock = 1
    else:
        time_to_lock = steps - time_to_lock + 1

    if last_choice == target:
        lock_times["comp_succ"].append(time_to_lock)
    else:
        lock_times["comp_inc"].append(time_to_lock)

# Restricting Dissemination
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
    flipped = 1 if last_choice == 0 else 0
    time_to_lock = dummy_choices.index(flipped) if flipped in dummy_choices else -1

    if time_to_lock == -1:
        time_to_lock = 1
    else:
        time_to_lock = steps - time_to_lock + 1

    if last_choice == target:
        lock_times["diss_succ"].append(time_to_lock)
    else:
        lock_times["diss_inc"].append(time_to_lock)

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

# Printing the results into csv format
# Specify the output file path
output_file = 'c:/Users/skyec/Desktop/Projects/Research/computational_epistemology/bernoulli_simulations/output/results2.csv'

# Open the output file in write mode
with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    # Write the header row
    writer.writerow(['index', 'comp_succ', 'comp_inc', 'diss_succ', 'diss_inc', 'cond_succ', 'cond_inc', 'hybrid_succ', 'hybrid_inc'])

    m = max(len(lock_times[key]) for key in lock_times)

    for i in range(m):
        row = [i]

        if i < len(lock_times["comp_succ"]):
            row.append(lock_times["comp_succ"][i])
        else:
            row.append('')

        if i < len(lock_times["comp_inc"]):
            row.append(lock_times["comp_inc"][i])
        else:
            row.append('')

        if i < len(lock_times["diss_succ"]):
            row.append(lock_times["diss_succ"][i])
        else:
            row.append('')

        if i < len(lock_times["diss_inc"]):
            row.append(lock_times["diss_inc"][i])
        else:
            row.append('')

        if i < len(lock_times["cond_succ"]):
            row.append(lock_times["cond_succ"][i])
        else:
            row.append('')

        if i < len(lock_times["cond_inc"]):
            row.append(lock_times["cond_inc"][i])
        else:
            row.append('')

        if i < len(lock_times["hybrid_succ"]):
            row.append(lock_times["hybrid_succ"][i])
        else:
            row.append('')

        if i < len(lock_times["hybrid_inc"]):
            row.append(lock_times["hybrid_inc"][i])
        else:
            row.append('')

        writer.writerow(row)

# Print the totals
print("# Totals:", [len(lock_times[key]) for key in lock_times])

