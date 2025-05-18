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

###########################################
########### Simulation Settings ###########
###########################################

filename = 'computational_epistemology/bernoulli_simulations/output_fall2024/zollman_cycle_test.csv'

# Default params
num_machines = 2
priors = 'uniform'
runs = 100
steps = 100
which_arm_restricted_params = ['random'] # ['random', 'other', 'target']
pulls_per_trial = 1000

# Experiment-unique params
experiment_mode = 'default' # one of 'default', 'epsilon', 'inertia'

# Set range for variable to test
p_values = [.501] #np.arange(.501, .9, .005) # decimal values between 0 and 1
num_agents_values = [4, 5, 6, 7, 8, 9, 10, 11] # positive whole numbers

epsilon_values = [.1] # np.arange(.01, .2, .02) # decimal values between 0 and 1
inertia_values = [5] # np.arange(1, 10, 1) # positive whole numbers

###########################################
############## End Settings ###############
###########################################

# Experiment type safety
if experiment_mode == 'default':
    epsilon_values = [-1]
    inertia_values = [0]
elif experiment_mode == 'epsilon':
    inertia_values = [0]
elif experiment_mode == 'inertia':
    epsilon_values = [-1]
else:
    raise ValueError('Invalid experiment_mode')

# Writing results to a CSV file
with open(filename, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['scenario', 'network_structure', 'which_arm_restricted', 'ps[0]', 'ps[1]', 'num_agents', experiment_mode, 'n_steps', 'n_success', 'n_runs', 'avg_total_succ_pulls', 'min_total_succ_pulls', 'median_total_succ_pulls', 'max_total_succ_pulls', 'stnd_dev_total_succ_pulls', 'avg_ca_time_to_converge', 'min_ca_time_to_converge', 'median_ca_time_to_converge', 'max_ca_time_to_converge', 'stnd_dev_ca_time_to_converge', 'avg_group_time_to_converge', 'min_group_time_to_converge', 'median_group_time_to_converge', 'max_group_time_to_converge', 'stnd_dev_group_time_to_converge', 'times_group_converged', 'avg_group_time_to_converge_succ', 'min_group_time_to_converge_succ', 'median_group_time_to_converge_succ', 'max_group_time_to_converge_succ', 'stnd_dev_group_time_to_converge_succ', 'times_group_converged_succ'])

    ##############################
    ## Initializaing Parameters ##
    ##############################

    params_tested = 0
    for p in p_values:
        for num_agents in num_agents_values:
            for epsilon in epsilon_values:
                for inertia in inertia_values:
                
                    # Creating the social networks
                    complete_graph = social_networks.makeCompleteGraph(num_agents)
                    star_graph = social_networks.makeStarGraph(num_agents)
                    cycle_graph = social_networks.makeCycleGraph(num_agents)
                    wheel_graph = social_networks.makeWheelGraph(num_agents)

                    # Creating the agent lists
                    agent_list = []
                    for i in range(num_agents):
                        if priors == "uniform" or priors == "u":
                            agent_list.append(beta_agents.BetaAgentUniformPriors(num_machines, epsilon=epsilon, resiliance=inertia))
                        elif priors == "jeffrey" or priors == "j":
                            agent_list.append(beta_agents.BetaAgentJeffreyPriors(num_machines, epsilon=epsilon, resiliance=inertia))
                        elif priors == "random" or priors == "r":
                            agent_list.append(beta_agents.BetaAgentRandomPriors(num_machines, epsilon=epsilon, resiliance=inertia))
                        else:
                            agent_list.append(beta_agents.BetaAgentUniformPriors(num_machines, epsilon=epsilon, resiliance=inertia))

                    # Setup Modification Params with new p value
                    ps = [0.5, p]

                    # Setup the machines with new p value
                    machine_list = [slot_machines.BernoulliMachine(p) for p in ps]
                    machine_list_flipped = [slot_machines.BernoulliMachine(p) for p in reversed(ps)]

                    ################################################################################
                    ############## Baseline, complete graph, no restrictions #######################
                    ################################################################################

                    n_success = 0
                    total_successful_pulls = []
                    group_times_to_converge = []
                    group_times_to_converge_succ = []
                    central_authority_times_to_converge = []

                    for r in range(runs):
                        target = 0 if ps[0] > ps[1] else 1
                        
                        network = social_networks.DummyNetwork(agent_list, machine_list, complete_graph)

                        for agent in agent_list:
                            agent.reset()

                        agents_choices = [[] for agent in agent_list]
                        for t in range(steps):
                            network.step(pulls_per_trial)
                            for i, agent in enumerate(agent_list):
                                agents_choices[i].insert(0, agent.getMachineToPlay())

                        # total number of correct pulls over course of entire run
                        total_successful_pulls_in_this_trial = sum([1 if choice == target else 0 for choices in agents_choices for choice in choices])
                        total_successful_pulls.append(total_successful_pulls_in_this_trial)

                        # Keep track of successful runs
                        last_choices = [choices[0] for choices in agents_choices]

                        # see if central agent succeeded in finding the target
                        dummy_last_choice = last_choices[0]
                        if dummy_last_choice == target:
                            n_success += 1

                        # see if all agents converged to the same choice
                        all_agents_converged = all([last_choice == dummy_last_choice for last_choice in last_choices])

                        if all_agents_converged:

                            # individual time to converge
                            last_choices = [choices[0] for choices in agents_choices]
                            flipped = [1 if last_choice == 0 else 0 for last_choice in last_choices]
                            times_to_converge = [steps - choices.index(f) + 1 if f in choices else 1 for choices, f in zip(agents_choices, flipped)]

                            group_time_to_converge = max(times_to_converge)

                            group_times_to_converge.append(group_time_to_converge)

                            if dummy_last_choice == target:
                                group_times_to_converge_succ.append(group_time_to_converge)

                        central_authority_time_to_converge = times_to_converge[0]
                        central_authority_times_to_converge.append(central_authority_time_to_converge)

                    avg_total_successful_pulls = np.mean(total_successful_pulls)
                    min_total_successful_pulls = np.min(total_successful_pulls)
                    median_total_successful_pulls = np.median(total_successful_pulls)
                    max_total_successful_pulls = np.max(total_successful_pulls)
                    stnd_dev_total_successful_pulls = np.std(total_successful_pulls)

                    avg_central_authority_time_to_converge = np.mean(central_authority_times_to_converge)
                    min_central_authority_time_to_converge = np.min(central_authority_times_to_converge)
                    median_central_authority_time_to_converge = np.median(central_authority_times_to_converge)
                    max_central_authority_time_to_converge = np.max(central_authority_times_to_converge)
                    stnd_dev_central_authority_time_to_converge = np.std(central_authority_times_to_converge)

                    avg_group_time_to_converge = np.mean(group_times_to_converge)
                    min_group_time_to_converge = np.min(group_times_to_converge)
                    median_group_time_to_converge = np.median(group_times_to_converge)
                    max_group_time_to_converge = np.max(group_times_to_converge)
                    stnd_dev_group_time_to_converge = np.std(group_times_to_converge)

                    avg_group_time_to_converge_succ = np.mean(group_times_to_converge_succ)
                    min_group_time_to_converge_succ = np.min(group_times_to_converge_succ)
                    median_group_time_to_converge_succ = np.median(group_times_to_converge_succ)
                    max_group_time_to_converge_succ = np.max(group_times_to_converge_succ)
                    stnd_dev_group_time_to_converge_succ = np.std(group_times_to_converge_succ)

                    times_group_converged = len(group_times_to_converge)
                    times_group_converged_succ = len(group_times_to_converge_succ)

                    test_param = epsilon if experiment_mode == 'epsilon' else inertia if experiment_mode == 'inertia' else 'N/A'
                    # writer.writerow(['scenario', 'network_structure', 'which_arm_restricted', 'ps[0]', 'ps[1]', 'num_agents', 'steps_until_change', 'n_steps', 'n_success', 'n_runs', 'avg_ca_time_to_converge', 'min_ca_time_to_converge', 'median_ca_time_to_converge', 'max_ca_time_to_converge', 'stnd_dev_ca_time_to_converge', 'avg_group_time_to_converge', 'min_group_time_to_converge', 'median_group_time_to_converge', 'max_group_time_to_converge', 'stnd_dev_group_time_to_converge', 'times_group_converged'])
                    writer.writerow(['baseline', 'complete', 'N/A', ps[0], ps[1], num_agents, test_param, steps, n_success, runs, avg_total_successful_pulls, min_total_successful_pulls, median_total_successful_pulls, max_total_successful_pulls, stnd_dev_total_successful_pulls, avg_central_authority_time_to_converge, min_central_authority_time_to_converge, median_central_authority_time_to_converge, max_central_authority_time_to_converge, stnd_dev_central_authority_time_to_converge, avg_group_time_to_converge, min_group_time_to_converge, median_group_time_to_converge, max_group_time_to_converge, stnd_dev_group_time_to_converge, times_group_converged, avg_group_time_to_converge_succ, min_group_time_to_converge_succ, median_group_time_to_converge_succ, max_group_time_to_converge_succ, stnd_dev_group_time_to_converge_succ, times_group_converged_succ])

                    ###############################################################################
                    ######################### Restricting Dissemination ###########################
                    ###############################################################################
                    for which_arm_restricted in which_arm_restricted_params:

                        n_success = 0
                        total_successful_pulls = []
                        group_times_to_converge = []
                        group_times_to_converge_succ = []
                        central_authority_times_to_converge = []

                        for r in range(runs):
                            target = 0 if ps[0] > ps[1] else 1
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

                            agents_choices = [[] for agent in agent_list]
                            for t in range(steps):
                                network.step(pulls_per_trial)
                                for i, agent in enumerate(agent_list):
                                    agents_choices[i].insert(0, agent.getMachineToPlay())

                            # total number of correct pulls over course of entire run
                            total_successful_pulls_in_this_trial = sum([1 if choice == target else 0 for choices in agents_choices for choice in choices])
                            total_successful_pulls.append(total_successful_pulls_in_this_trial)

                            # Keep track of successful runs
                            last_choices = [choices[0] for choices in agents_choices]

                            # see if central agent succeeded in finding the target
                            dummy_last_choice = last_choices[0]
                            if dummy_last_choice == target:
                                n_success += 1

                            # see if all agents converged to the same choice
                            all_agents_converged = all([last_choice == dummy_last_choice for last_choice in last_choices])

                            if all_agents_converged:

                                # individual time to converge
                                last_choices = [choices[0] for choices in agents_choices]
                                flipped = [1 if last_choice == 0 else 0 for last_choice in last_choices]
                                times_to_converge = [steps - choices.index(f) + 1 if f in choices else 1 for choices, f in zip(agents_choices, flipped)]

                                group_time_to_converge = max(times_to_converge)

                                group_times_to_converge.append(group_time_to_converge)

                                if dummy_last_choice == target:
                                    group_times_to_converge_succ.append(group_time_to_converge)


                            central_authority_time_to_converge = times_to_converge[0]
                            central_authority_times_to_converge.append(central_authority_time_to_converge)

                        avg_total_successful_pulls = np.mean(total_successful_pulls)
                        min_total_successful_pulls = np.min(total_successful_pulls)
                        median_total_successful_pulls = np.median(total_successful_pulls)
                        max_total_successful_pulls = np.max(total_successful_pulls)
                        stnd_dev_total_successful_pulls = np.std(total_successful_pulls)

                        avg_central_authority_time_to_converge = np.mean(central_authority_times_to_converge)
                        min_central_authority_time_to_converge = np.min(central_authority_times_to_converge)
                        median_central_authority_time_to_converge = np.median(central_authority_times_to_converge)
                        max_central_authority_time_to_converge = np.max(central_authority_times_to_converge)
                        stnd_dev_central_authority_time_to_converge = np.std(central_authority_times_to_converge)

                        avg_group_time_to_converge = np.mean(group_times_to_converge)
                        min_group_time_to_converge = np.min(group_times_to_converge)
                        median_group_time_to_converge = np.median(group_times_to_converge)
                        max_group_time_to_converge = np.max(group_times_to_converge)
                        stnd_dev_group_time_to_converge = np.std(group_times_to_converge)

                        avg_group_time_to_converge_succ = np.mean(group_times_to_converge_succ)
                        min_group_time_to_converge_succ = np.min(group_times_to_converge_succ)
                        median_group_time_to_converge_succ = np.median(group_times_to_converge_succ)
                        max_group_time_to_converge_succ = np.max(group_times_to_converge_succ)
                        stnd_dev_group_time_to_converge_succ = np.std(group_times_to_converge_succ)

                        times_group_converged = len(group_times_to_converge)
                        times_group_converged_succ = len(group_times_to_converge_succ)

                        # writer.writerow(['scenario', 'network_structure', 'which_arm_restricted', 'ps[0]', 'ps[1]', 'num_agents', 'steps_until_change', 'n_steps', 'n_success', 'n_runs', 'avg_time_to_converge', 'min_time_to_converge', 'median_time_to_converge', 'max_time_to_converge', 'stnd_dev_time_to_converge'])
                        writer.writerow(['restrict_dissemination', 'star', which_arm_restricted, ps[0], ps[1], num_agents, test_param, steps, n_success, runs, avg_total_successful_pulls, min_total_successful_pulls, median_total_successful_pulls, max_total_successful_pulls, stnd_dev_total_successful_pulls, avg_central_authority_time_to_converge, min_central_authority_time_to_converge, median_central_authority_time_to_converge, max_central_authority_time_to_converge, stnd_dev_central_authority_time_to_converge, avg_group_time_to_converge, min_group_time_to_converge, median_group_time_to_converge, max_group_time_to_converge, stnd_dev_group_time_to_converge, times_group_converged, avg_group_time_to_converge_succ, min_group_time_to_converge_succ, median_group_time_to_converge_succ, max_group_time_to_converge_succ, stnd_dev_group_time_to_converge_succ, times_group_converged_succ])

                # Show progress
                params_tested += 1
                print(f'{params_tested} parameters tested out of {len(p_values) * len(num_agents_values) * len(epsilon_values)}')