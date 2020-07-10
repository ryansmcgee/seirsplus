import sys
import networkx
import numpy as np
import pandas as pd

from extended_models import *
from networks import *
from sim_loops import *
from helper_functions import *
from repeated_loops.rtw_runs200624 import *

def semiweekly_testing_simulation(model, time):
    """
    Basic model with weekly testing on Mondays and
    with some level of symptomatic self isolation
    """
    run_rtw_testing_sim(model=model, T=time, symptomatic_selfiso_compliance_rate=0.3, testing_cadence='semiweekly')

def semiweekly_testing_simulation_3dayTAT(model, time):
    """
    Basic model with weekly testing on Mondays and
    with some level of symptomatic self isolation
    """
    run_rtw_testing_sim(model=model, T=time, symptomatic_selfiso_compliance_rate=0.3, testing_cadence='semiweekly', isolation_lag=3)

def semiweekly_testing_simulation_5dayTAT(model, time):
    """
    Basic model with weekly testing on Mondays and
    with some level of symptomatic self isolation
    """
    run_rtw_testing_sim(model=model, T=time, symptomatic_selfiso_compliance_rate=0.3, testing_cadence='semiweekly', isolation_lag=5)

def semiweekly_testing_simulation_0dayTAT(model, time):
    """
    Basic model with weekly testing on Mondays and
    with some level of symptomatic self isolation
    """
    run_rtw_testing_sim(model=model, T=time, symptomatic_selfiso_compliance_rate=0.3, testing_cadence='semiweekly', isolation_lag=0)

def semiweekly_testing_simulation_0dayTAT_pocSens(model, time):
    """
    Basic model with weekly testing on Mondays and
    with some level of symptomatic self isolation
    """
    run_rtw_testing_sim(model=model, T=time, symptomatic_selfiso_compliance_rate=0.3, testing_cadence='semiweekly', isolation_lag=0, sensitivity_offset = 0.2)

def semiweekly_testing_simulation_0dayTAT_pocSens_half(model, time):
    """
    Basic model with weekly testing on Mondays and
    with some level of symptomatic self isolation
    """
    run_rtw_testing_sim(model=model, T=time, symptomatic_selfiso_compliance_rate=0.3, testing_cadence='semiweekly', isolation_lag=0, sensitivity_offset = 0.1)

def weekly_testing_simulation_TAT5(model, time):
    run_rtw_testing_sim(model=model, T=time, symptomatic_selfiso_compliance_rate=0.3, testing_cadence='weekly', isolation_lag=5)

def weekly_continuous_testing_simulation_TAT5(model, time):
    run_rtw_testing_sim(model=model, T=time, symptomatic_selfiso_compliance_rate=0.3, testing_cadence='everyday', pct_tested_per_day=1/7, isolation_lag=5)


num_cohorts = 10 # number of different groups
number_teams_per_cohort = 5 # number of teams
num_nodes_per_cohort = 100 # total number of people per group


N = num_cohorts*num_nodes_per_cohort
pct_contacts_intercohort = 0.2
isolation_time=14
q = 0

INIT_EXPOSED = 1
R0_MEAN = 2.0
R0_COEFFVAR_HIGH = 2.2
R0_COEFFVAR_LOW = 0.15
P_GLOBALINTXN = 0.4
MAX_TIME = 200


repeats = 1000
def main():
    tat3 = repeat_runs(repeats, semiweekly_testing_simulation_3dayTAT)
    tat5 = repeat_runs(repeats, semiweekly_testing_simulation_5dayTAT)

    tat3.to_csv('/Users/julianhomburger/Data/covid/seirsplus/200624/seir_semiweekly_tat3_200629.csv', index=False)
    tat5.to_csv('/Users/julianhomburger/Data/covid/seirsplus/200624/seir_semiweekly_tat5_200629.csv', index=False)

    tat0 = repeat_runs(repeats, semiweekly_testing_simulation_0dayTAT)
    tat0.to_csv('/Users/julianhomburger/Data/covid/seirsplus/200624/seir_semiweekly_tat0_200629.csv', index=False)

    ### Let's model faster TAT, lower sensitivity
    poc_tat0 = repeat_runs(repeats, semiweekly_testing_simulation_0dayTAT_pocSens)
    poc_tat0.to_csv('/Users/julianhomburger/Data/covid/seirsplus/200624/seir_semiweekly_tat0_poc_200629.csv', index=False)

    poc5_tat0 = repeat_runs(repeats, semiweekly_testing_simulation_0dayTAT_pocSens_half)
    poc5_tat0.to_csv('/Users/julianhomburger/Data/covid/seirsplus/200624/seir_semiweekly_tat0_poc5_200629.csv', index=False)

    weekly_tat5 = repeat_runs(repeats, weekly_testing_simulation_TAT5)
    weekly_tat5.to_csv('/Users/julianhomburger/Data/covid/seirsplus/200624/seir_weekly_tat5_200706.csv', index=False)

    weekly_continuous_tat5 = repeat_runs(repeats, weekly_continuous_testing_simulation_TAT5)
    weekly_continuous_tat5.to_csv('/Users/julianhomburger/Data/covid/seirsplus/200624/seir_weekly_continuous_tat5_200706.csv', index=False)
