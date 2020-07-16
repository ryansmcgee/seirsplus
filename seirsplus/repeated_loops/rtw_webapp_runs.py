### Model loops for web models
## Goal:

#### Need to install python3 tkinter
#### Need to add seirsplus to pythonpath

import sys
import networkx
import numpy as np
import pandas as pd
import itertools
import os

from extended_models import *
from networks import *
from sim_loops import *
from helper_functions import *
from repeated_loops.rtw_runs200624 import *

def write_parallel_inputs(pfilename):
    # Writes a file with arguments for the parallel files
    introduction_rate = [0, 7, 28] # parameterized in mean days to next introduction
    tats = [1,3,5] # test turnaround times
    testing_cadence = ['none', 'everyday', 'semiweekly', 'weekly', 'biweekly', 'monthly']
    pfile = open(pfilename, "w")
    for y in itertools.product(testing_cadence, introduction_rate, tats):
        c, ir, t = y
        pfile.write(f'{c},{ir},{t}\n')
    pfile.close()

def repeat_runs_webapp(n_repeats, simulation_fxn, save_escalation_time = False):
    """
    A wrapper for repeating the runs, that takes a simulation function defined above.

    NOTE - most of these parameters are defined outside the function.
    """
    output_frames = []
    model_overview = []
    for i in np.arange(0, n_repeats):
        G_baseline, G_quarantine, cohorts, teams = build_farz_graph(num_cohorts = num_cohorts, num_nodes_per_cohort = num_nodes_per_cohort, num_teams_per_cohort = number_teams_per_cohort, pct_contacts_intercohort = pct_contacts_intercohort)

        SIGMA, LAMDA, GAMMA, BETA, BETA_Q = basic_distributions(N, R0_mean = R0_MEAN, R0_coeffvar = np.random.uniform(R0_COEFFVAR_LOW, R0_COEFFVAR_HIGH))
        model = ExtSEIRSNetworkModel(G=G_baseline, p=P_GLOBALINTXN,
                                        beta=BETA, sigma=SIGMA, lamda=LAMDA, gamma=GAMMA,
                                        gamma_asym=GAMMA,
                                        G_Q=G_quarantine, q=q, beta_Q=BETA_Q, isolation_time=isolation_time,
                                        initE=INIT_EXPOSED, seed = i)
        intervention_time, escalation_time, escalation_from_screen, total_tests = simulation_fxn(model, MAX_TIME)

        thisout = get_regular_series_output(model, MAX_TIME)
        thisout['total_tests'] = total_tests
        if save_escalation_time:
            print(escalation_time)
            thisout['escalation_time'] = escalation_time
            thisout['escalation_from_screen'] = escalation_from_screen
        output_frames.append(thisout)
    return(pd.concat(output_frames))
### Write out the lists of parameters to loop over

r0_lists = [2.5, 2.0, 1.5, 1.0]
population_sizes = [50, 100, 500, 1000]

testing_cadence, introduction_rate, tats = sys.argv[1].split(',')
# introduction_rate = sys.argv[2] # parameterized in mean days to next introduction
# tats = sys.argv[3] # test turnaround times
# Dummy model fxn that takes in parameters
def dummy_testing_simulation(model, time):
    """
    Basic model with weekly testing on Mondays and
    with some level of symptomatic self isolation
    """
    avg_intros = 0
    if intro_rate != 0:
        avg_intros = 1/intro_rate
    return run_rtw_testing_sim(model=model, T=time, symptomatic_selfiso_compliance_rate=0.3,
                testing_cadence=cadence,
                isolation_lag = tat,
                average_introductions_per_day = avg_intros)

def assign_new_cols(df, newdict):
    for x in newdict.keys():
        df[x] = newdict[x]

# Define the network structure
# We decided on no subdivision, with 10 contacts on average
num_cohorts = 1 # number of different groups
number_teams_per_cohort = 1 # number of teams
# num_nodes_per_cohort = 100 # total number of people per group
# N = num_cohorts*num_nodes_per_cohort


pct_contacts_intercohort = 0.0
isolation_time=14
q = 0

R0_COEFFVAR_HIGH = 2.2
R0_COEFFVAR_LOW = 0.15
P_GLOBALINTXN = 0.3
MAX_TIME = 365

nrepeats = 2


# Thanks itertools!
for x in itertools.product(r0_lists, population_sizes ):

    R0_MEAN, N  = x

    intro_rate, tat, cadence = int(introduction_rate), int(tats), testing_cadence
    param_hash = hash((R0_MEAN, N, intro_rate, tat, cadence))

    results_name = f'{cadence}_{R0_MEAN}_{N}_{intro_rate}_{tat}_results.csv.gz'
    if os.path.exists(results_name):
        continue
    num_nodes_per_cohort = N

    INIT_EXPOSED = 1
    these_results = repeat_runs_webapp(nrepeats, dummy_testing_simulation)
    param_dict = {'r0':R0_MEAN, 'pop_size':N, 'cadence':cadence, 'tat':tat, 'intro_rate':intro_rate}
    assign_new_cols(these_results, param_dict)
    these_results.to_csv(results_name, index=False, compression='gzip')

    # param_outfile_name = f'rtw_params{param_hash}.csv'
    # param_outfile = open(param_outfile_name, 'w')
    #
    # param_outfile.write(f'{param_hash},{R0_MEAN},{N},{intro_rate},{tat},{cadence}\n')
    # param_outfile.close()
