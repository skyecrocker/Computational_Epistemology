# Computational_Epistemology

Simulations, output, and analysis (January 2024 - May 2025) done by Skye Crocker for Dr Jens Kipper's research in Network Epistemology at the University of Rochester. In a nutshell, we investigate the effect of restricting dissemination on the epistemic/scientific progress of a community of agents/researchers, following up on previous research done by Wagner and Herington as well as Zollman. During Spring 2024 and Fall 2024 we run experiments in the form of basically a multi agent bandit problem, and in Spring 2025 we explore the idea of modelling it basically as a multi agent hill climbing problem. See reference_material folder for more info on the project background.

Here is a summary of folders and files:

## epsitemic_bandits

Code, output, and analysis for bandit problem setup. Experiments from Spring 2024 - Fall 2024.

* bandit_utils - Contains the core logic of the network, agents, etc... used in bandit simulations
* old_code_NoLongerInUse - Used to generate some of the data in the past, so keeping it just in case. But everything has now been streamlined into a few generalized scripts.
* Output_results - csv files containing data generated from simulations
* data_analysis.ipynb - Code for generating plots and viewing results
* simulation_master.py - Script for running most simulations, including both Wagner and Herington setup and Zollman setup, as well as adding inertia or epsilon parameters.
* individual_time.py - Simulations that keep track of and record individuals throughout the simulation. Output files significantly larger.

## epistemic_landscapes

Code, output, and analysis for bandit problem setup. Experiments during Spring 2025.

* Agent.py and Social_Network.py - Code for agent and network logic, readapted for 2d and 3d landscapes.

* Landscapes.py - Generation of 2d and 3d landscapes (separate classes). Also use this file to vizualize 2d landscapes directly or export 3d landscapes for viewing using LandscapeVisualization react js code.

* EpsitemicLandscapeSim.py - Use to run all simulations. At the top of the file contain all the parameter specifications.

* landscape_results - csv files containing data generated from simulations

* Landscape_analysis.ipynb - Generating plots and graphs for analysis

## LandscapeVisualization

React app that helps vizualize 3d landscape. Simply cd into folder and run "npm start" to use. Several landscape options will be there for viewing, but it is also possible/easy to create your using landscape.py and then placing the output json into the public folder of the react app and appending the link in app.js.

## Reference_material

Contains papers and code from Wagner and Herinton as well as Zollman. Also contains a summary of results and findings from our own research.
