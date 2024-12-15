import subprocess
import sys
sys.path.append('computational_epistemology/bandit_utils')
import argparse
import multiprocessing
import time
from slot_machines import *
from agents import *
from social_networks import *

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--priors', default='uniform', help='Priors')
parser.add_argument('-r', '--runs', type=int, default=10, help='Number of runs')
parser.add_argument('-s', '--steps', type=int, default=1000, help='Number of steps')
parser.add_argument('-q', type=float, default=0.55, help='q value')
args = parser.parse_args()

# Get the number of CPU cores
num_cpus = multiprocessing.cpu_count()
print("# Number of cores:", num_cpus)

# Set the parameters
priors = args.priors
runs = args.runs
steps = args.steps
q = args.q

print("# Priors:", priors, "; runs:", runs, "; steps:", steps)
print("num_agents,p0,p1,success_complete,success_star,consensus_complete,consensus_star,time_complete,time_star,time_succ_complete,time_succ_star,time_incorrect_complete,time_incorrect_star")

num_agent_list = list(range(3, 41))
results_as_strings = []

# Global variables required for traversing the independent variable over multiple child processes
# And then determining that every independent variable has completed
proc_index = 0
completed_processes = 0

start_time = time.time()

def launch_next_child():
    global proc_index, completed_processes

    if proc_index >= len(num_agent_list):
        return

    complete_graph = makeCompleteGraph(num_agent_list[proc_index])
    star_graph = makeStarGraph(num_agent_list[proc_index])

    parameters = {
        'priors': priors,
        'p': [0.5, q],
        'target': 1,
        'runs': runs,
        'steps': steps,
        'graphs': [complete_graph, star_graph]
    }

    proc_index += 1

    child = subprocess.Popen(['python', './computational_epistemology/bernoulli_simulations/baseline_child.py'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    # Send parameters to the child process
    child.stdin.write(str(parameters).encode())
    child.stdin.close()

    # Wait for the child process to finish and get the output
    output = child.stdout.read().decode()

    print(convert_results_to_string(eval(output)))

    launch_next_child()

    completed_processes += 1
    if completed_processes >= len(num_agent_list):
        end_time = time.time()
        print("#", (end_time - start_time) / 60, "minutes elapsed")

def convert_results_to_string(res):
    s = str(len(res['parameters']['graphs'][0])) + ","
    s += str(res['parameters']['p'][0]) + "," + str(res['parameters']['p'][1]) + ","
    s += ",".join(map(str, res['success_counts'])) + ","
    s += ",".join(map(str, res['consensus_counts'])) + ","
    s += ",".join(map(str, res['total_times_to_lock'])) + ","
    s += ",".join(map(str, res['total_times_to_successful_lock'])) + ","
    s += ",".join(map(str, res['total_times_to_incorrect_lock']))

    return s

# Launch child processes
for i in range(num_cpus):
    launch_next_child()
