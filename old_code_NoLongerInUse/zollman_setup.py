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

filename = 'computational_epistemology/bernoulli_simulations/output_fall2024/zollman_eps_Pulls.csv'#zollman_epsilon.csv'

# Default params
num_machines = 2
priors = 'uniform'
runs = 1000
steps = 10000
pulls_per_trial = 1000

# Experiment-unique params
experiment_mode = 'epsilon' # one of 'default', 'epsilon', 'inertia'

# Set range for variable to test
p_values = [0.5001, .501, .51, .6] # decimal values between 0 and 1
num_agents_values = [10] # positive whole numbers
network_strctures = ['complete', 'cycle', 'wheel']

epsilon_values = [.1] # np.arange(.01, .2, .02) # decimal values between 0 and 1
inertia_values = [5] # np.arange(1, 10, 1) # positive whole numbers

###########################################
############## End Settings ###############
###########################################
             
def run_simulation(runs, steps, network_structure, pulls_per_trial, num_agents, p, epsilon=-1, inertia=0, num_machines=2, priors='uniform'):

    # Creating the social network
    match network_structure:
        case "complete":
            network_graph = social_networks.makeCompleteGraph(num_agents)
        case "star":
            network_graph = social_networks.makeStarGraph(num_agents)
        case "cycle":
            network_graph = social_networks.makeCycleGraph(num_agents)
        case "wheel":
            network_graph = social_networks.makeWheelGraph(num_agents)

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

    ################################################################################
    ############## Baseline, complete graph, no restrictions #######################
    ################################################################################

    #n_success = 0
    group_times_to_converge = []
    group_times_to_converge_succ = []

    total_successful_pulls_at_step = {}
    n_converged_at_step = {}
    n_successes_at_step = {}

    for r in range(runs):
        if r % 100 == 0:
            print(f'Run {r}')

        target = 0 if ps[0] > ps[1] else 1
        
        network = social_networks.Network(agent_list, machine_list, network_graph, pulls_per_trial=pulls_per_trial)

        for agent in agent_list:
            agent.reset()

        agents_choices = [[] for agent in agent_list]
        for t in range(steps):
            network.step()
            for i, agent in enumerate(agent_list):
                agents_choices[i].insert(0, agent.getBestMachine())

            if t % 10 == 9:
                #print(f'Run {r}, Step {t}')

                # initialize dict values if they don't exist
                if t + 1 not in n_successes_at_step:
                    total_successful_pulls_at_step[t + 1] = 0
                    n_successes_at_step[t + 1] = 0
                    n_converged_at_step[t + 1] = 0

                successful_pulls_in_run = sum([agent.totalReward for agent in agent_list])
                num_agents_converged = network.numConvergedTo(target)
                successfully_converged = network.hasConvergedTo(target)

                total_successful_pulls_at_step[t + 1] += successful_pulls_in_run
                n_converged_at_step[t + 1] += num_agents_converged
                n_successes_at_step[t + 1] += 1 if successfully_converged else 0


    avg_total_successful_pulls_dict = {k: v / runs for k, v in total_successful_pulls_at_step.items()}
    individual_portion_converged_dict = {k: v / runs for k, v in n_converged_at_step.items()}  
    proportion_converged_dict = {k: v / runs for k, v in n_successes_at_step.items()}

    test_param = epsilon if experiment_mode == 'epsilon' else inertia if experiment_mode == 'inertia' else 'N/A'

    # writer.writerow(['network_structure', 'ps[0]', 'ps[1]', 'num_agents', 'steps_until_change', 'n_steps', 'n_success', 'n_runs'])

    for steps in proportion_converged_dict.keys():
        successful_pulls = avg_total_successful_pulls_dict[steps]
        individuals_converged = individual_portion_converged_dict[steps]
        success_rate = proportion_converged_dict[steps]
        writer.writerow([network_structure, ps[0], ps[1], num_agents, test_param, steps, runs, success_rate, individuals_converged, successful_pulls])

    # writer.writerow(['scenario', 'network_structure', 'which_arm_restricted', 'ps[0]', 'ps[1]', 'num_agents', 'steps_until_change', 'n_steps', 'n_success', 'n_runs', 'avg_ca_time_to_converge', 'min_ca_time_to_converge', 'median_ca_time_to_converge', 'max_ca_time_to_converge', 'stnd_dev_ca_time_to_converge', 'avg_group_time_to_converge', 'min_group_time_to_converge', 'median_group_time_to_converge', 'max_group_time_to_converge', 'stnd_dev_group_time_to_converge', 'times_group_converged'])
    #writer.writerow(['baseline', 'complete', 'N/A', ps[0], ps[1], num_agents, test_param, steps, n_success, runs, avg_total_successful_pulls, min_total_successful_pulls, median_total_successful_pulls, max_total_successful_pulls, stnd_dev_total_successful_pulls, avg_central_authority_time_to_converge, min_central_authority_time_to_converge, median_central_authority_time_to_converge, max_central_authority_time_to_converge, stnd_dev_central_authority_time_to_converge, avg_group_time_to_converge, min_group_time_to_converge, median_group_time_to_converge, max_group_time_to_converge, stnd_dev_group_time_to_converge, times_group_converged, avg_group_time_to_converge_succ, min_group_time_to_converge_succ, median_group_time_to_converge_succ, max_group_time_to_converge_succ, stnd_dev_group_time_to_converge_succ, times_group_converged_succ])


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
    writer.writerow(['network_structure', 'ps[0]', 'ps[1]', 'num_agents', experiment_mode, 'n_steps', 'n_runs', 'success_convergence_rate', 'individuals_converged', 'total_successful_pulls'])

    ##############################
    ## Initializaing Parameters ##
    ##############################
   
    params_tested = 0
    for p in p_values:
        for num_agents in num_agents_values:
            for network_structure in network_strctures:
                for epsilon in epsilon_values:

                    run_simulation(runs, steps, network_structure, pulls_per_trial, num_agents, p, epsilon=epsilon)

                    # Show progress
                    params_tested += 1
                    print(f'{params_tested} parameters tested out of {len(p_values) * len(num_agents_values) * len(network_strctures) * len(epsilon_values) * len(inertia_values)}')