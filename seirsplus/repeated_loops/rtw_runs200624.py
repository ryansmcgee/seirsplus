import sys
import networkx
import numpy as np
import pandas as pd

from extended_models import *
from networks import *
from sim_loops import *
from helper_functions import *

DEFAULT_FARZ_PARAMS = {'alpha':    5.0,       # clustering param
             'gamma':    5.0,       # assortativity param
             'beta':     0.5,       # prob within community edges
             'r':        1,         # max num communities node can be part of
             'q':        0.0,       # probability of multi-community membership
             'phi':      10,        # community size similarity (1 gives power law community sizes)
             'b':        0,
             'epsilon':  1e-6,
             'directed': False,
             'weighted': False}

def build_farz_graph(farz_params= DEFAULT_FARZ_PARAMS, num_cohorts=1, num_nodes_per_cohort=500, num_teams_per_cohort=5, pct_contacts_intercohort=0.0):
    """
    Builds the FARZ graph and returns the baseline graph, quarantine graph, cohorts, and teams
    """

    G_baseline, cohorts, teams = generate_workplace_contact_network(num_cohorts=num_cohorts, num_nodes_per_cohort=num_nodes_per_cohort, num_teams_per_cohort=num_teams_per_cohort,
                                       mean_intracohort_degree=10, pct_contacts_intercohort=pct_contacts_intercohort,
                                       farz_params=farz_params)
    G_quarantine = networkx.classes.function.create_empty_copy(G_baseline)
    return((G_baseline, G_quarantine, cohorts, teams))

## Simulation functions wrap detailed implementations of models and other functions
## Note that the parameters should be set outside these functions
def baseline_simulation(model, time):
    """
    A simulation with no interventions at all, but some syptomatic self isolation
    """
    return run_rtw_testing_sim(model=model, T=time, symptomatic_selfiso_compliance_rate=0.3)

def weekly_testing_simulation(model, time):
    """
    Basic model with weekly testing on Mondays and
    with some level of symptomatic self isolation
    """
    return run_rtw_testing_sim(model=model, T=time, symptomatic_selfiso_compliance_rate=0.3, testing_cadence='weekly')

def semiweekly_testing_simulation(model, time):
    """
    Basic model with weekly testing on Mondays and
    with some level of symptomatic self isolation
    """
    return run_rtw_testing_sim(model=model, T=time, symptomatic_selfiso_compliance_rate=0.3, testing_cadence='semiweekly')

def biweekly_testing_simulation(model, time):
    """
    Basic model with weekly testing on Mondays and
    with some level of symptomatic self isolation
    """
    return run_rtw_testing_sim(model=model, T=time, symptomatic_selfiso_compliance_rate=0.3, testing_cadence='biweekly')


def daily_testing_simulation(model, time):
    """
    Basic model with weekly testing on Mondays and
    with some level of symptomatic self isolation
    """
    return run_rtw_testing_sim(model=model, T=time, symptomatic_selfiso_compliance_rate=0.3, testing_cadence='everyday')

def weekly_continuous_testing_simulation(model, time):
    """
    Basic model with weekly testing on Mondays and
    with some level of symptomatic self isolation
    """
    return run_rtw_testing_sim(model=model, T=time, symptomatic_selfiso_compliance_rate=0.3, test_logistics='continuous', continuous_days_between_tests=7)

def monthly_continuous_testing_simulation(model, time):
    """
    Basic model with weekly testing on Mondays and
    with some level of symptomatic self isolation
    """
    return run_rtw_testing_sim(model=model, T=time, symptomatic_selfiso_compliance_rate=0.3, test_logistics='continuous', continuous_days_between_tests=28)

def weekly_escalation_testing_simulation(model, time):
    """
    Basic model with weekly testing on Mondays and
    with some level of symptomatic self isolation
    """
    return run_rtw_testing_sim(model=model, T=time, symptomatic_selfiso_compliance_rate=0.3, test_logistics='continuous', continuous_days_between_tests=28, escalate_on_positive=True, escalate_days_between_tests=7 )

def semiweekly_escalation_testing_simulation(model, time):
    """
    Basic model with weekly testing on Mondays and
    with some level of symptomatic self isolation
    """
    return run_rtw_testing_sim(model=model, T=time, symptomatic_selfiso_compliance_rate=0.3, test_logistics='continuous', continuous_days_between_tests=28, escalate_on_positive=True, escalate_days_between_tests=4 )

def no_symptomatic_baseline(model, time):
    return run_rtw_testing_sim(model=model, T=time, symptomatic_selfiso_compliance_rate=0.0)

def full_symptomatic_baseline(model, time):
    return run_rtw_testing_sim(model=model, T=time, symptomatic_selfiso_compliance_rate=1.0)

def high_symptomatic_baseline(model, time):
    return run_rtw_testing_sim(model=model, T=time, symptomatic_selfiso_compliance_rate=0.7)


def repeat_runs(n_repeats, simulation_fxn, save_escalation_time = False):
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


# Wrapper functions

### Define some parameters


### Graph params for N=1000
def make_params():
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
    baseline = repeat_runs(repeats, baseline_simulation)
    weekly_tests = repeat_runs(repeats, weekly_testing_simulation)
    semi_weekly_tests = repeat_runs(repeats, semiweekly_testing_simulation)

    baseline.to_csv('/Users/julianhomburger/Data/covid/seirsplus/200624/seir_baseline200629.csv', index=False)
    weekly_tests.to_csv('/Users/julianhomburger/Data/covid/seirsplus/200624/seir_weekly_tests200629.csv', index=False)
    semi_weekly_tests.to_csv('/Users/julianhomburger/Data/covid/seirsplus/200624/seir_semiweekly_tests200629.csv', index=False)

    # daily_tests = repeat_runs(repeats, daily_testing_simulation)
    # semiweekly_tests = repeat_runs(repeats, semiweekly_testing_simulation)
    #
    # # weekly_continuous = repeat_runs(repeats, weekly_continuous_testing_simulation)
    # semiweekly_tests.to_csv('/Users/julianhomburger/Data/covid/seirsplus/200624/seir_semiweekly_tests200626', index=False)
    # daily_tests.to_csv('/Users/julianhomburger/Data/covid/seirsplus/200624/seir_daily_tests200626', index=False)
    # weekly_continuous.to_csv('/Users/julianhomburger/Data/covid/seirsplus/200624/seir_weekly_continuous_tests200626.csv'. index=False)

    weekly_continuous = repeat_runs(repeats, weekly_continuous_testing_simulation)
    weekly_continuous.to_csv('/Users/julianhomburger/Data/covid/seirsplus/200624/seir_weekly_continuous_tests200706.csv', index=False)

    escalation_month_week = repeat_runs(repeats, weekly_escalation_testing_simulation, save_escalation_time=True)
    escalation_month_week.to_csv('/Users/julianhomburger/Data/covid/seirsplus/200624/seir_escalation_month_week_tests200706.csv', index=False)

    escalation_month_semiweek = repeat_runs(repeats, semiweekly_escalation_testing_simulation, save_escalation_time=True)
    escalation_month_semiweek.to_csv('/Users/julianhomburger/Data/covid/seirsplus/200624/seir_escalation_month_semiweek_tests200706.csv', index=False)


    monthly_continuous = repeat_runs(repeats, monthly_continuous_testing_simulation)
    monthly_continuous.to_csv('/Users/julianhomburger/Data/covid/seirsplus/200624/seir_monthly_continuous_tests200706.csv', index=False)
    ### Graph params for N=100
    num_cohorts = 2 # number of different groups
    number_teams_per_cohort = 5 # number of teams
    num_nodes_per_cohort = 50 # total number of people per group
    N = num_cohorts*num_nodes_per_cohort


    baseline_n100 = repeat_runs(repeats, baseline_simulation)
    weekly_tests_n100 = repeat_runs(repeats, weekly_testing_simulation)
    semi_weekly_tests_n100 = repeat_runs(repeats, semiweekly_testing_simulation)

    baseline_n100.to_csv('/Users/julianhomburger/Data/covid/seirsplus/200624/seir_baseline200629_n100.csv', index=False)
    weekly_tests_n100.to_csv('/Users/julianhomburger/Data/covid/seirsplus/200624/seir_weekly_tests200629_n100.csv', index=False)
    semi_weekly_tests_n100.to_csv('/Users/julianhomburger/Data/covid/seirsplus/200624/seir_semiweekly_tests200629_n100.csv', index=False)


    escalation_month_week_n100 = repeat_runs(repeats, weekly_escalation_testing_simulation, save_escalation_time=True)
    escalation_month_week_n100.to_csv('/Users/julianhomburger/Data/covid/seirsplus/200624/seir_escalation_month_week_tests200706_n100.csv', index=False)

    no_symptomatic = repeat_runs(repeats, no_symptomatic_baseline)
    no_symptomatic.to_csv('/Users/julianhomburger/Data/covid/seirsplus/200624/no_symptomatic_baseline200713.csv')
    full_symptomatic = repeat_runs(repeats, full_symptomatic_baseline)
    full_symptomatic.to_csv('/Users/julianhomburger/Data/covid/seirsplus/200624/all_symptomatic_baseline200713.csv')

    high_symptomatic = repeat_runs(repeats, high_symptomatic_baseline)
    high_symptomatic.to_csv('/Users/julianhomburger/Data/covid/seirsplus/200624/high_symptomatic_baseline200713.csv')
