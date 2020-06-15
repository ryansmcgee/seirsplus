import numpy

###

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
    R0_mean = 2.5,
    R0_coeffvar = 2.0):
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
