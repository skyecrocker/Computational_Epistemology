import numpy as np
from Landscapes import Landscape, Landscape3D
from Agent import Agent, Agent3D
from Social_Network import *
import csv
import os
import matplotlib.pyplot as plt
import seaborn as sns
import time
from matplotlib.colors import *

print(os.chdir("epistemic_landscapes/"))
print(os.getcwd())

output_file = "landscape_results/3d_landscape_1step_allSigma_41grid.csv"

# Simulation Parameters
landscape_dim = 3
network_structure_values = ["complete", "isolated"] #, "star", "wheel", "line", "cycle", "two_cliques"]
grid_size_values = [41]
landscape_sigma_values = [0.5,1,2,3] # Higher values create a smoother landscape and vice versa NOTE recommend 1-5

num_agents_values = [10]
agent_range_values = [2, 4, 6, 'unlimited']

agent_strategy_values = ["default"] # default, epsilonGreedy
epsilon_values = [0, 0.5, .1, .15, .2, .3, .4, .5, .6, .7, .8, .9, 1.]

num_timesteps_values = [200]
num_simulations = 300

landscape_generation_method = "boostGlobal" # boostGlobal, dampenLocal, onePeak

viz = False # NOTE: recommend num_simulations = 1 or few
viz_every = 1 # int or 'end'



# Simulation

with open(output_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['network_structure', 'grid_size', 'sigma', 'num_agents', 'agent_range', 'agent_strategy', 'epsilon', 'num_timesteps', 'num_simulations', 'success_proportion', 'avg_time_to_peak', 'avg_max_visits', 'avg_visits', 'redundancy'])

    total_time = 0
    params_tested = 0
    for num_timesteps in num_timesteps_values:
        for grid_size in grid_size_values:
            for landscape_sigma in landscape_sigma_values:
                for num_agents in num_agents_values:
                    for agent_strategy in agent_strategy_values:

                        # Epsilon shenanigans
                        ran_default = False
                        for epsilon in epsilon_values:
                            if agent_strategy == "default" and not ran_default:
                                epsilon = 0
                                ran_default = True
                            elif agent_strategy == "default" and ran_default:
                                break

                            for agent_range in agent_range_values:
                                for network_structure in network_structure_values:
                                    if network_structure == "complete":
                                        network_graph = makeCompleteGraph(num_agents)
                                    elif network_structure == "isolated":
                                        network_graph = makeIsolatedGraph(num_agents)
                                    elif network_structure == "two_cliques":
                                        network_graph = makeTwoCliquesGraph(num_agents)
                                    elif network_structure == "star":
                                        network_graph = makeStarGraph(num_agents)
                                    elif network_structure == "wheel":
                                        network_graph = makeWheelGraph(num_agents)
                                    elif network_structure == "line":
                                        network_graph = makeLineGraph(num_agents)
                                    elif network_structure == "cycle":
                                        network_graph = makeCycleGraph(num_agents)
                                    else:
                                        print("Invalid network structure: ", network_structure)
                                        continue

                                    time_start = time.time()

                                    # Globals for parameter setup
                                    num_success = 0
                                    time_to_find_peak = []
                                    patch_visit_counts = {f'{i}+': [] for i in range(1, 11)}
                                    patch_visit_counts['avg'] = []
                                    patch_visit_counts['max'] = []
                                    patch_visit_counts['revisits'] = []

                                    for sim_id in range(num_simulations):
                                        
                                        if landscape_dim == 2:
                                            landscape = Landscape(size=grid_size, sigma=landscape_sigma, seed=None, method=landscape_generation_method)
                                        elif landscape_dim == 3:
                                            landscape = Landscape3D(size=grid_size, sigma=landscape_sigma, seed=None, method=landscape_generation_method)
                                        else:
                                            raise ValueError("Invalid landscape dimension. Must be 2 or 3.")

                                        # Create agents
                                        agents = []
                                        for i in range(num_agents):
                                            if landscape_dim == 2:
                                                agent = Agent(coordinates=(np.random.randint(0, grid_size), np.random.randint(0, grid_size)), range=agent_range, grid_size=grid_size, strategy=agent_strategy, epsilon=epsilon)
                                            elif landscape_dim == 3:
                                                agent = Agent3D(coordinates=(np.random.randint(0, grid_size), np.random.randint(0, grid_size), np.random.randint(0, grid_size)), range=agent_range, grid_size=grid_size, strategy=agent_strategy, epsilon=epsilon)

                                            agent.update_grid_knowledge(agent.coordinates, landscape.values[agent.coordinates])
                                            agents.append(agent)

                                        # Create Network
                                        network = Network(agents, network_graph, landscape.values)

                                        # Run simulation
                                        global_peak_coords = np.unravel_index(np.argmax(landscape.values, axis=None), landscape.values.shape)

                                        for t in range(num_timesteps):

                                            if network.has_found_global_peak():
                                                num_success += 1
                                                time_to_find_peak.append(t)
                                                break

                                            if viz and viz_every != 'end':
                                                if t % viz_every == 0:
                                                    # plot heatmap of visits
                                                    colors = ["white", "lightblue", "blue", "green", "yellow", "orange", "red", "darkred"]
                                                    boundaries = [0, 1, 2, 3, 4, 5, 10, 20, 100]  # Ranges: 1, 2, 3, 4, 5-10, 10-20, 20+

                                                    # Create colormap and normalization
                                                    cmap = ListedColormap(colors)
                                                    norm = BoundaryNorm(boundaries, cmap.N)

                                                    sns.heatmap(np.clip(network.visits, 0, 10), cmap=cmap, norm=norm, cbar=True, annot=False)
                                                    plt.title(f"{network_structure}, sim: {sim_id}, step: {t}, peak={global_peak_coords}")
                                                    plt.show()
                                            
                                            network.step()

                                        if viz and viz_every == 'end':
                                            # plot heatmap of visits
                                            colors = ["white", "lightblue", "blue", "green", "yellow", "orange", "red", "darkred"]
                                            boundaries = [0, 1, 2, 3, 4, 5, 10, 20, 100]

                                            # Create colormap and normalization
                                            cmap = ListedColormap(colors)
                                            norm = BoundaryNorm(boundaries, cmap.N)

                                            sns.heatmap(np.clip(network.visits, 0, 10), cmap=cmap, norm=norm, cbar=True, annot=False)
                                            plt.title(f"{network_structure}, sigma:{landscape_sigma}, range:{agent_range}, sim:{sim_id}, step:{t}, peak={global_peak_coords}")
                                            plt.show()

                                        # Record patch visit counts
                                        patch_visit_counts['avg'].append(np.mean(network.visits))
                                        patch_visit_counts['max'].append(np.max(network.visits))
                                        patch_visit_counts['revisits'].append(np.sum(network.visits) - np.count_nonzero(network.visits))

                                    # Calculate success proportion and average time to peak
                                    success_proportion = num_success / num_simulations
                                    avg_time_to_peak = np.mean(time_to_find_peak) if time_to_find_peak else 0

                                    avg_visits = np.mean(patch_visit_counts['avg'])
                                    max_visits = np.mean(patch_visit_counts['max']) 
                                    redundancy = np.mean(patch_visit_counts['revisits'])

                                    time_taken = time.time() - time_start
                                    total_time += time_taken

                                    # Write results to CSV
                                    writer.writerow([network_structure, grid_size, landscape_sigma, num_agents, agent_range, agent_strategy, epsilon, num_timesteps, num_simulations, success_proportion, avg_time_to_peak, max_visits, avg_visits, redundancy])
                                    params_tested += 1
                                    ep = int( (len(epsilon_values) * ("epsilonGreedy" in agent_strategy_values)) + ("default" in agent_strategy_values) )
                                    print(f"{params_tested} params tested out of {len(num_timesteps_values) * len(network_structure_values) * len(grid_size_values) * len(landscape_sigma_values) * len(num_agents_values) * len(agent_range_values) * ep} || time taken: {time_taken} || total time: {total_time}")

                        
