import sys
sys.path.append('computational_epistemology')
from bandit_utils.slot_machines import BernoulliMachine
from bandit_utils.agents import BetaAgentUniformPriors, BetaAgentJeffreyPriors, BetaAgentRandomPriors
from bandit_utils.social_networks import DummyNetwork
import json

def simulate(parameters):
    machine_list = []
    for p in parameters['p']:
        machine_list.append(BernoulliMachine(p))

    agent_list = []
    for i in range(len(parameters['graphs'][0])):
        if parameters['priors'] in ["uniform", "u"]:
            agent_list.append(BetaAgentUniformPriors(len(machine_list)))
        elif parameters['priors'] in ["jeffrey", "j"]:
            agent_list.append(BetaAgentJeffreyPriors(len(machine_list)))
        elif parameters['priors'] in ["random", "r"]:
            agent_list.append(BetaAgentRandomPriors(len(machine_list)))
        else:
            agent_list.append(BetaAgentUniformPriors(len(machine_list)))

    networks = []
    for graph in parameters['graphs']:
        networks.append(DummyNetwork(agent_list, machine_list, graph))

    success_counts = []
    consensus_counts = []
    total_times_to_lock = []
    total_times_to_successful_lock = []
    total_times_to_incorrect_lock = []

    for net_index, network in enumerate(networks):
        success_counts.append(0)
        consensus_counts.append(0)
        total_times_to_lock.append(0)
        total_times_to_successful_lock.append(0)
        total_times_to_incorrect_lock.append(0)

        for r in range(parameters['runs']):
            for a in agent_list:
                a.reset()

            dummy_choices = []
            for t in range(parameters['steps']):
                network.step()
                dummy_choices.insert(0, network.getDummyChoice())

            last_choice = dummy_choices.pop(0)
            flipped = 1 if last_choice == 0 else 0
            time_to_lock = dummy_choices.index(flipped) if flipped in dummy_choices else -1

            if time_to_lock == -1:
                time_to_lock = 0
            else:
                time_to_lock = parameters['steps'] - time_to_lock
            total_times_to_lock[net_index] += time_to_lock

            if last_choice == parameters['target']:
                total_times_to_successful_lock[net_index] += time_to_lock
            else:
                total_times_to_incorrect_lock[net_index] += time_to_lock

            if network.hasDummyLearned(parameters['target']):
                success_counts[net_index] += 1

            if network.hasReachedConsensus():
                consensus_counts[net_index] += 1

    return {
        'parameters': parameters,
        'success_counts': success_counts,
        'consensus_counts': consensus_counts,
        'total_times_to_lock': total_times_to_lock,
        'total_times_to_successful_lock': total_times_to_successful_lock,
        'total_times_to_incorrect_lock': total_times_to_incorrect_lock
    }

if __name__ == '__main__':

    # Read the parameters from command line arguments
    parameters = json.loads(sys.argv[1])

    # Call the simulate function with the parameters
    sim_results = simulate(parameters)

    # Print the simulation results
    print(json.dumps(sim_results))
