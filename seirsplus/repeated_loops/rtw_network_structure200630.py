import sys
import networkx
import numpy as np
import pandas as pd

from extended_models import *
from networks import *
from sim_loops import *
from helper_functions import *
from rtw_runs200624 import *

INIT_EXPOSED = 1
R0_MEAN = 2.0
R0_COEFFVAR_HIGH = 2.2
R0_COEFFVAR_LOW = 0.15
P_GLOBALINTXN = 0.4
MAX_TIME = 200


repeats = 1000

num_cohorts = 10 # number of different groups
num_nodes_per_cohort = 100 # total number of people per group

N = num_cohorts*num_nodes_per_cohort
pct_contacts_intercohort = 0.2
isolation_time=14
q = 0


number_teams_per_cohort = 1 # number on a specific team is increased
num_cohorts = 1 # number of different groups
num_nodes_per_cohort = 1000 # total number of people per group
pct_contacts_intercohort = 0.0

no_teams = repeat_runs(repeats, semiweekly_testing_simulation)
no_teams.to_csv('/Users/julianhomburger/Data/covid/seirsplus/200624/seir_semiweekly_noteams_200629.csv', index=False)

number_teams_per_cohort = 2 # number on a specific team is increased
num_cohorts = 10 # number of different groups
num_nodes_per_cohort = 100 # total number of people per group
pct_contacts_intercohort = 0.2

big_teams = repeat_runs(repeats, semiweekly_testing_simulation)
big_teams.to_csv('/Users/julianhomburger/Data/covid/seirsplus/200624/seir_semiweekly_bigteams_200629.csv', index=False)
