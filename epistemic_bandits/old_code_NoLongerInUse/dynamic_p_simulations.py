filename = 'computational_epistemology/bernoulli_simulations/output/dynamic_bonus01_priors33_other6_targetRange_steps1000.csv'

import sys
sys.path.append('computational_epistemology/bandit_utils')
import argparse
import random
import slot_machines
import agents as beta_agents
import social_networks
import csv
import numpy as np
import time

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

# Default params
num_machines = 2
priors = 'uniform'
runs = 1000
steps = 1000
num_agents = 9
which_arm_restricted_params = ['random'] #['random', 'other', 'target']

# Setup down below
prior_ps = [.33, .33]

# Set range for variable to test
prior_p_values = np.arange(.1, .4, .02)
max_p_values = np.arange(.6, 1, .02)

# Modification Params
# prior_ps = [.5, .5]
# max_ps = [.5, .99]
#p_bonus_per_success = 0.001

# Creating the social networks
complete_graph = social_networks.makeCompleteGraph(num_agents)
star_graph = social_networks.makeStarGraph(num_agents)

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

# Writing results to a CSV file
with open(filename, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['scenario', 'network_structure', 'arm_restricted', 'num_agents_restricted', 'prior_ps[0]', 'prior_ps[1]', 'p_increment[0]', 'p_increment[1]', 'max_ps[0]', 'max_ps[1]', 'avg_final_ps[0]', 'avg_final_ps[1]', 'num_agents', 'n_steps', 'n_success', 'n_runs', 'avg_time_to_converge', 'min_time_to_converge', 'median_time_to_converge', 'max_time_to_converge', 'stnd_dev_time_to_converge'])

    params_tested = 0
    for max_p in max_p_values:
    #for prior_p in prior_p_values:

        # Setup Modification Params with new p value
        prior_ps = [0.33, 0.33]
        max_ps = [.6, max_p]
        p_bonus_per_success = .01

        # Setup the machines with new p value
        machine_list = [slot_machines.BernoulliMachine(prior_p, max_p, p_bonus_per_success) for prior_p, max_p in zip(prior_ps, max_ps)]
        machine_list_flipped = [slot_machines.BernoulliMachine(prior_p, max_p, p_bonus_per_success) for prior_p, max_p in zip(reversed(prior_ps), reversed(max_ps))]

        ################################################################################
        ############## Baseline, complete graph, no restrictions #######################
        ################################################################################

        n_success = 0
        times_to_convergence = []
        avg_final_ps_0 = 0
        avg_final_ps_1 = 0

        for r in range(runs):
            target = 0 if max_ps[0] > max_ps[1] else 1
            
            network = social_networks.DummyNetwork(agent_list, machine_list, complete_graph)

            for agent in agent_list:
                agent.reset()

            dummy_choices = []
            for t in range(steps):
                #print(f"Step {t} of {steps} (complete)")
                network.step(add_p_bonus=True)
                dummy_choices.insert(0, network.getDummyChoice())

            # Keep track of successful runs
            last_choice = dummy_choices[0]
            if last_choice == target:
                n_success += 1

            # Keep track of time to converge
            flipped = 1 if last_choice == 0 else 0
            if flipped in dummy_choices:
                time_to_lock = steps - dummy_choices.index(flipped)
            else:
                time_to_lock = 0
            times_to_convergence.append(time_to_lock)

            # Keep track of final p values
            final_ps = network.getProbabilities()
            avg_final_ps_0 += np.mean(final_ps[0])
            avg_final_ps_1 += np.mean(final_ps[1])

        avg_final_ps_0 /= runs
        avg_final_ps_1 /= runs
        
        avg_time_to_converge = np.mean(times_to_convergence)
        min_time_to_converge = np.min(times_to_convergence)
        median_time_to_converge = np.median(times_to_convergence)
        max_time_to_converge = np.max(times_to_convergence)
        stnd_dev_time_to_converge = np.std(times_to_convergence)

        #writer.writerow(['scenario', 'network_structure', 'arm_restricted', 'num_agents_restricted', 'prior_ps[0]', 'prior_ps[1]', 'p_increment[0]', 'p_increment[1]', 'max_ps[0]', 'max_ps[1]', 'avg_final_ps[0]', 'avg_final_ps[1]', 'num_agents', 'n_steps', 'n_success', 'n_runs', 'avg_time_to_converge'])
        writer.writerow(['baseline', 'complete', 'N/A', 'N/A', prior_ps[0], prior_ps[1], p_bonus_per_success, p_bonus_per_success, max_ps[0], max_ps[1], avg_final_ps_0, avg_final_ps_1, num_agents, steps, n_success, runs, avg_time_to_converge, min_time_to_converge, median_time_to_converge, max_time_to_converge, stnd_dev_time_to_converge])

        ###############################################################################
        ######################### Restricting Dissemination ###########################
        ###############################################################################
        for which_arm_restricted in which_arm_restricted_params:

            n_success = 0
            times_to_convergence = []
            avg_final_ps_0 = 0
            avg_final_ps_1 = 0

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
                    network.step(add_p_bonus=True)
                    dummy_choices.insert(0, network.getDummyChoice())

                # Keep track of successful runs
                last_choice = dummy_choices[0]
                if last_choice == target:
                    n_success += 1

                # Keep track of time to converge
                flipped = 1 if last_choice == 0 else 0
                if flipped in dummy_choices:
                    time_to_lock = steps - dummy_choices.index(flipped)
                else:
                    time_to_lock = 0
                times_to_convergence.append(time_to_lock)

                # Keep track of final p values
                final_ps = network.getProbabilities()
                if flip_machines:
                    final_ps = list(reversed(final_ps))
                avg_final_ps_0 += np.mean(final_ps[0])
                avg_final_ps_1 += np.mean(final_ps[1])

            
            avg_time_to_converge = np.mean(times_to_convergence)
            min_time_to_converge = np.min(times_to_convergence)
            median_time_to_converge = np.median(times_to_convergence)
            max_time_to_converge = np.max(times_to_convergence)
            stnd_dev_time_to_converge = np.std(times_to_convergence)

            avg_final_ps_0 /= runs
            avg_final_ps_1 /= runs

            #writer.writerow(['scenario', 'network_structure', 'arm_restricted', 'num_agents_restricted', 'prior_ps[0]', 'prior_ps[1]', 'p_increment[0]', 'p_increment[1]', 'max_ps[0]', 'max_ps[1]', 'avg_final_ps[0]', 'avg_final_ps[1]', 'num_agents', 'n_steps', 'n_success', 'n_runs', 'avg_time_to_converge'])
            writer.writerow(['restrict_dissemination', 'star', which_arm_restricted, 'N/A', prior_ps[0], prior_ps[1], p_bonus_per_success, p_bonus_per_success, max_ps[0], max_ps[1], avg_final_ps_0, avg_final_ps_1, num_agents, steps, n_success, runs, avg_time_to_converge, min_time_to_converge, median_time_to_converge, max_time_to_converge, stnd_dev_time_to_converge])


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

        '''

        # Show progress
        params_tested += 1
        print(f'{params_tested} parameters tested out of {len(max_p_values)}')