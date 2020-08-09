import numpy

###

import sys
import networkx
import numpy as np
import pandas as pd

from extended_models import *
from networks import *
from sim_loops import *
from helper_functions import *

def get_regular_series_output(model, tmax):
    """
    Given a seirsplus model, returns a pandas data frame of a regular series (ie per day)
    of the number of infecteds, infecteds + exposed, and i+e+q.
    Fills in zero for days past the end of the models
    """
    total_i= model.numI_pre + model.numI_sym + model.numI_asym
    total_e = model.numE
    total_q = model.numQ_pre + model.numQ_sym + model.numQ_asym + model.numQ_R + model.numQ_S + model.numQ_E
    total_r = model.numR + model.numQ_R
    # Build the
    reg_e = []
    reg_i = []
    reg_q = []
    reg_r = []
    times = []
    for i in np.arange(0,tmax,1):
        last_timepoint = np.searchsorted(model.tseries, i)
        times.append(i)
        if last_timepoint < len(model.tseries):
            reg_e.append(total_e[last_timepoint])
            reg_i.append(total_i[last_timepoint])
            reg_q.append(total_q[last_timepoint])
            reg_r.append(total_r[last_timepoint])
        else:
            reg_e.append(total_e[-1])
            reg_i.append(total_i[-1])
            reg_q.append(total_q[-1])
            reg_r.append(total_r[-1])
    total_num_infected = model.total_num_infected()[-1]+model.total_num_recovered()[-1] # Overall infections in the model
    return(pd.DataFrame({'total_i':reg_i,
                        'total_q':reg_q,
                        'total_e': reg_e,
                        'overall_infections': total_num_infected,
                        'time':times,
                        'seed':model.seed}
    ))


def get_gamma_dist(mean, coeffvar, N):
    scale = mean*coeffvar**2
    shape = mean/scale
    return(numpy.random.gamma(scale=scale, shape=shape, size=N))

def basic_distributions(N, quarantine_effectiveness=1.0,
    latentPeriods_mean = 3.0,
    latentPeriods_coeffvar = 0.6,
    prodromalPeriod_mean = 2.2,
    prodromalPeriod_coeffvar = 0.5,
    symptomaticPeriod_mean = 4.0,
    symptomaticPeriod_coeffvar = 0.4,
    R0_mean = 2.0,
    R0_coeffvar = 1.0): # try drawing this from unif(0.2,2), 0.2 comes out with top 20th percentile at 80% of distribution value
    # 2.0 is close to imperial college distribution, could go a bit higher to tighten the distribution.
    # Look at if there is a 90/10 value to bound the low end on.
    """
    Builds and returns the basic distributions of the parameters seen in the ipython NB, including
    gamma, lambda, sigma, and betas
    """
    # Latend period development
    SIGMA  = 1 / get_gamma_dist(latentPeriods_mean, latentPeriods_coeffvar, N)

    # pre-symptomatic time
    LAMDA = 1 / get_gamma_dist(prodromalPeriod_mean, prodromalPeriod_coeffvar, N)

    # symptomatic period
    GAMMA = 1 / get_gamma_dist(symptomaticPeriod_mean, symptomaticPeriod_coeffvar, N)

    infectiousPeriods = 1/LAMDA + 1/GAMMA # Invert to calculate infectious periods
    R0 = get_gamma_dist(R0_mean, R0_coeffvar, N)

    BETA     = 1/infectiousPeriods * R0
    BETA_Q = BETA * ((1.0-quarantine_effectiveness)/R0_mean)

    return SIGMA, LAMDA, GAMMA, BETA, BETA_Q

def generate_hospitalization_params(N):
    # time to hospitalization
    onsetToHospitalizationPeriod_mean, onsetToHospitalizationPeriod_coeffvar = 11.0, 0.45
    ETA = 1 / get_gamma_dist(onsetToHospitalizationPeriod_mean, onsetToHospitalizationPeriod_coeffvar, N)

    # hospitalization to discharge
    hospitalizationToDischargePeriod_mean, hospitalizationToDischargePeriod_coeffvar = 11.0, 0.45
    GAMMA_H = 1 / get_gamma_dist(hospitalizationToDischargePeriod_mean, hospitalizationToDischargePeriod_coeffvar, N)

    # hospitalization to death
    hospitalizationToDeathPeriod_mean, hospitalizationToDeathPeriod_coeffvar = 7.0, 0.45
    MU_H = 1 / get_gamma_dist(hospitalizationToDeathPeriod_mean, hospitalizationToDeathPeriod_coeffvar)
    return ETA, GAMMA_H, MU_H
