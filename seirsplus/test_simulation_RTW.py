from __future__ import division
# from models import *
from extended_models import *
from networks import *
from sim_loops import *
import networkx
import pickle
import gzip
import pandas
from scipy.stats import linregress

import matplotlib.pyplot as pyplot
pyplot.switch_backend('agg')
# import seaborn
# seaborn.set_style('ticks')


#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

import argparse

parser  = argparse.ArgumentParser()
parser.add_argument("-runID",                         "--runID",                          default="script",     type=str)
parser.add_argument("-T",                             "--T",                              default=300,          type=float)
parser.add_argument("-N",                             "--N",                              default=1000,        type=int)
parser.add_argument("-INIT_PCT_E",                    "--INIT_PCT_E",                     default=(1/10000),    type=float)
parser.add_argument("-INTERVENTION_START_PCT_INFECTED", "--INTERVENTION_START_PCT_INFECTED", default=(1/1000),   type=float)
parser.add_argument("-R0_mean",                       "--R0_mean",                        default=2.5,          type=float)
parser.add_argument("-R0_coeffvar",                   "--R0_coeffvar",                    default=0.2,          type=float)
parser.add_argument("-PCT_ASYMPTOMATIC",              "--PCT_ASYMPTOMATIC",               default=0.3,          type=float)
parser.add_argument("-P_GLOBALINTXN",                 "--P_GLOBALINTXN",                  default=0.1,         type=float)
parser.add_argument("-Q_GLOBALINTXN",                 "--Q_GLOBALINTXN",                  default=0.0,          type=float)
# parser.add_argument("-TESTING_INTERVAL",              "--TESTING_INTERVAL",               default=1,            type=float)
# parser.add_argument("-FRAC_TO_TEST_PER_INTERVAL",     "--FRAC_TO_TEST_PER_INTERVAL",      default=(1/10),       type=float)
# parser.add_argument("-TEST_FALSENEG_RATE",            "--TEST_FALSENEG_RATE",             default='temporal',   type=str)
# parser.add_argument("-TESTING_RANDOM_COMPLIANCE_RATE",          "--TESTING_RANDOM_COMPLIANCE_RATE",          default=0.8, type=float)
# parser.add_argument("-TESTING_TRACED_COMPLIANCE_RATE",          "--TESTING_TRACED_COMPLIANCE_RATE",          default=0.8, type=float)
# parser.add_argument("-TESTING_SELFSEEK_COMPLIANCE_RATE",        "--TESTING_SELFSEEK_COMPLIANCE_RATE",        default=1.0, type=float)
# parser.add_argument("-TESTING_DEGREE_BIAS",           "--TESTING_DEGREE_BIAS",            default=0,            type=float)
# parser.add_argument("-MAX_PCT_TESTS_FOR_TRACING",     "--MAX_PCT_TESTS_FOR_TRACING",      default=1.0,          type=float)
# parser.add_argument("-TRACING_COMPLIANCE_RATE",       "--TRACING_COMPLIANCE_RATE",        default=0.8,          type=float)
# parser.add_argument("-NUM_CONTACTS_TO_TRACE",         "--NUM_CONTACTS_TO_TRACE",          default=None,         type=float)
# parser.add_argument("-FRAC_CONTACTS_TO_TRACE",        "--FRAC_CONTACTS_TO_TRACE",         default=1.0,          type=float)
# parser.add_argument("-TRACING_INTERVAL_LAG",          "--TRACING_INTERVAL_LAG",           default=2,            type=int)
# parser.add_argument("-MAX_PCT_TESTS_FOR_SEEKING",     "--MAX_PCT_TESTS_FOR_SEEKING",      default=1.0,          type=float)
# parser.add_argument("-SYMPTOMATIC_SEEKTEST_COMPLIANCE_RATE",    "--SYMPTOMATIC_SEEKTEST_COMPLIANCE_RATE",     default=0.8, type=float)
# parser.add_argument("-SYMPTOMATIC_SELFISOLATE_COMPLIANCE_RATE", "--SYMPTOMATIC_SELFISOLATE_COMPLIANCE_RATE",  default=0.8, type=float)
# parser.add_argument("-TRACING_SELFISOLATE_COMPLIANCE_RATE",     "--TRACING_SELFISOLATE_COMPLIANCE_RATE",      default=0.8, type=float)
# parser.add_argument("-HOUSEHOLD_ISOLATION_COMPLIANCE_RATE",     "--HOUSEHOLD_ISOLATION_COMPLIANCE_RATE",      default=0.8, type=float)
# parser.add_argument("-DO_RANDOM_TESTING",             "--DO_RANDOM_TESTING",              default=True,         type=int)
# parser.add_argument("-DO_TRACING_TESTING",            "--DO_TRACING_TESTING",             default=True,         type=int)
# parser.add_argument("-DO_SEEKING_TESTING",            "--DO_SEEKING_TESTING",             default=True,         type=int)
# parser.add_argument("-DO_SYMPTOM_SELFISO",            "--DO_SYMPTOM_SELFISO",             default=True,         type=int)
# parser.add_argument("-DO_TRACING_SELFISO",            "--DO_TRACING_SELFISO",             default=False,        type=int)
# parser.add_argument("-DO_ISOLATE_POSITIVE_HOUSEHOLDS",          "--DO_ISOLATE_POSITIVE_HOUSEHOLDS",           default=False, type=int)
# parser.add_argument("-DO_ISOLATE_SYMPTOMATIC_HOUSEHOLDS",       "--DO_ISOLATE_SYMPTOMATIC_HOUSEHOLDS",        default=False, type=int)
# parser.add_argument("-DO_ISOLATE_TRACING_SELFISO_HOUSEHOLDS",   "--DO_ISOLATE_TRACING_SELFISO_HOUSEHOLDS",    default=False, type=int)
parser.add_argument("-BETA_PAIRWISE_MODE",            "--BETA_PAIRWISE_MODE",             default='mean',       type=str)
parser.add_argument("-ALPHA_PAIRWISE_MODE",           "--ALPHA_PAIRWISE_MODE",            default='mean',       type=str)
parser.add_argument("-TRANSITION_MODE",               "--TRANSITION_MODE",                default='exponential_rates',       type=str)

parser.add_argument("-outdir",                        "--outdir",                         default="./", type=str)
parser.add_argument("-figdir",                        "--figdir",                         default="./",    type=str)

args    = parser.parse_args()



#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

runID           = str(args.runID)

outdir          = str(args.outdir)
figdir          = str(args.figdir)


T = args.T

N = int(args.N)

INIT_PCT_E   = args.INIT_PCT_E
INIT_EXPOSED = 1 # max(int(N*INIT_PCT_E), 10)

INTERVENTION_START_PCT_INFECTED = args.INTERVENTION_START_PCT_INFECTED

R0_mean      = args.R0_mean
R0_coeffvar  = args.R0_coeffvar

PCT_ASYMPTOMATIC = args.PCT_ASYMPTOMATIC

P_GLOBALINTXN = args.P_GLOBALINTXN
Q_GLOBALINTXN = args.Q_GLOBALINTXN

# TESTING_INTERVAL          = args.TESTING_INTERVAL
# FRAC_TO_TEST_PER_INTERVAL = args.FRAC_TO_TEST_PER_INTERVAL
# TESTS_PER_INTERVAL        = int(N*FRAC_TO_TEST_PER_INTERVAL)
# TESTING_DEGREE_BIAS       = args.TESTING_DEGREE_BIAS

# TEST_FALSENEG_RATE        = args.TEST_FALSENEG_RATE if args.TEST_FALSENEG_RATE == "temporal" else float(args.TEST_FALSENEG_RATE)

# TESTING_RANDOM_COMPLIANCE_RATE   = args.TESTING_RANDOM_COMPLIANCE_RATE
# TESTING_TRACED_COMPLIANCE_RATE   = args.TESTING_TRACED_COMPLIANCE_RATE
# TESTING_SELFSEEK_COMPLIANCE_RATE = args.TESTING_SELFSEEK_COMPLIANCE_RATE

# MAX_PCT_TESTS_FOR_TRACING = args.MAX_PCT_TESTS_FOR_TRACING
# TRACING_COMPLIANCE_RATE   = args.TRACING_COMPLIANCE_RATE
# NUM_CONTACTS_TO_TRACE     = args.NUM_CONTACTS_TO_TRACE # If None, trace based on fraction of contacts given below
# FRAC_CONTACTS_TO_TRACE    = args.FRAC_CONTACTS_TO_TRACE
# TRACING_INTERVAL_LAG      = args.TRACING_INTERVAL_LAG

# MAX_PCT_TESTS_FOR_SEEKING = args.MAX_PCT_TESTS_FOR_SEEKING

# SYMPTOMATIC_SEEKTEST_COMPLIANCE_RATE    = args.SYMPTOMATIC_SEEKTEST_COMPLIANCE_RATE
# SYMPTOMATIC_SELFISOLATE_COMPLIANCE_RATE = args.SYMPTOMATIC_SELFISOLATE_COMPLIANCE_RATE
# TRACING_SELFISOLATE_COMPLIANCE_RATE     = args.TRACING_SELFISOLATE_COMPLIANCE_RATE
# HOUSEHOLD_ISOLATION_COMPLIANCE_RATE     = args.HOUSEHOLD_ISOLATION_COMPLIANCE_RATE

# DO_RANDOM_TESTING   = args.DO_RANDOM_TESTING
# DO_TRACING_TESTING  = args.DO_TRACING_TESTING
# DO_SEEKING_TESTING  = args.DO_SEEKING_TESTING
# DO_SYMPTOM_SELFISO  = args.DO_SYMPTOM_SELFISO
# DO_TRACING_SELFISO  = args.DO_TRACING_SELFISO

# DO_ISOLATE_POSITIVE_HOUSEHOLDS        = args.DO_ISOLATE_POSITIVE_HOUSEHOLDS 
# DO_ISOLATE_SYMPTOMATIC_HOUSEHOLDS     = args.DO_ISOLATE_SYMPTOMATIC_HOUSEHOLDS 
# DO_ISOLATE_TRACING_SELFISO_HOUSEHOLDS = args.DO_ISOLATE_TRACING_SELFISO_HOUSEHOLDS

BETA_PAIRWISE_MODE  = args.BETA_PAIRWISE_MODE
ALPHA_PAIRWISE_MODE = args.ALPHA_PAIRWISE_MODE
TRANSITION_MODE     = args.TRANSITION_MODE

runStr = ("_color_rtw"+"_"+runID)
print runStr


#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# # Initialize Model Parameters and Contact Networks

# ## Generate Population Contact Networks

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

farz_params={'alpha':    5.0,       # clustering param
             'gamma':    5.0,       # assortativity param
             'beta':     0.5,       # prob within community edges
             'r':        1,         # max num communities node can be part of
             'q':        0.0,       # probability of multi-community membership
             'phi':      10,        # community size similarity (1 gives power law community sizes)
             'b':        0, 
             'epsilon':  1e-6, 
             'directed': False, 
             'weighted': False}

G_baseline, cohorts, teams = generate_workplace_contact_network(num_cohorts=1, num_nodes_per_cohort=N, num_teams_per_cohort=int(N/50),
                                       mean_intracohort_degree=10, pct_contacts_intercohort=0.0,
                                       farz_params=farz_params)

G_quarantine = networkx.classes.function.create_empty_copy(G_baseline)


#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

degree      = [d[1] for d in G_baseline.degree()]
mean_degree = numpy.mean(degree)
median_degree = numpy.median(degree)
max_degree  = numpy.max(degree)
CV_degree   = numpy.std(degree)/mean_degree
print "baseline graph mean degree       = " + str(mean_degree)
print "baseline graph median degree     = " + str(median_degree)
print "baseline graph CV degree         = " + str(CV_degree)
print "baseline graph CV^2 degree       = " + str(CV_degree**2)

r = networkx.degree_assortativity_coefficient(G_baseline)
print "baseline graph assortativity     = " + str(r)

c = networkx.average_clustering(G_baseline)
print "baseline graph clustering coeff  = " + str(c)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# ## Define Model Parameters

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

latentPeriods_mean, latentPeriods_coeffvar = 3.0, 0.6
latentPeriods_scale     = latentPeriods_mean*latentPeriods_coeffvar**2    
latentPeriods_shape     = latentPeriods_mean/latentPeriods_scale          
latentPeriods           = numpy.random.gamma(scale=latentPeriods_scale, shape=latentPeriods_shape, size=N)
SIGMA  = 1 / latentPeriods
print("Latent period: mean       = " + str(numpy.mean(numpy.sort(latentPeriods))) )
print("Latent period: median     = "+str(numpy.percentile(numpy.sort(latentPeriods), q=50)) )
print("Latent period: 5  pctile  = "+str(numpy.percentile(numpy.sort(latentPeriods), q=5)) )
print("Latent period: 25 pctile  = "+str(numpy.percentile(numpy.sort(latentPeriods), q=25)) )
print("Latent period: 75 pctile  = "+str(numpy.percentile(numpy.sort(latentPeriods), q=75)) )
print("Latent period: 95 pctile  = "+str(numpy.percentile(numpy.sort(latentPeriods), q=95)) )
print("SIGMA mean   = " +str(numpy.mean(SIGMA)) )
print("SIGMA median = " +str(numpy.median(SIGMA)) )

prodromalPeriod_mean, prodromalPeriod_coeffvar = 2.2, 0.5
prodromalPeriod_scale   = prodromalPeriod_mean*prodromalPeriod_coeffvar**2    # gamma distn theta
prodromalPeriod_shape   = prodromalPeriod_mean/prodromalPeriod_scale          # gamma distn k
prodromalPeriods        = numpy.random.gamma(scale=prodromalPeriod_scale, shape=prodromalPeriod_shape, size=N)
LAMDA = 1 / prodromalPeriods
print("Prodromal period: mean       = "+ str(numpy.mean(numpy.sort(prodromalPeriods))) )
print("Prodromal period: median     = "+str(numpy.percentile(numpy.sort(prodromalPeriods), q=50)) )
print("Prodromal period: 5  pctile  = "+str(numpy.percentile(numpy.sort(prodromalPeriods), q=5)) )
print("Prodromal period: 25 pctile  = "+str(numpy.percentile(numpy.sort(prodromalPeriods), q=25)) )
print("Prodromal period: 75 pctile  = "+str(numpy.percentile(numpy.sort(prodromalPeriods), q=75)) )
print("Prodromal period: 95 pctile  = "+str(numpy.percentile(numpy.sort(prodromalPeriods), q=95)) )
print("LAMDA mean   = " +str(numpy.mean(LAMDA)) )
print("LAMDA median = " +str(numpy.median(LAMDA)) )

incubationPeriods = latentPeriods + prodromalPeriods
print("Incubation period: mean       = " + str(numpy.mean(numpy.sort(incubationPeriods))) )
print("Incubation period: median     = "+str(numpy.percentile(numpy.sort(incubationPeriods), q=50)) )
print("Incubation period: 5  pctile  = "+str(numpy.percentile(numpy.sort(incubationPeriods), q=5)) )
print("Incubation period: 25 pctile  = "+str(numpy.percentile(numpy.sort(incubationPeriods), q=25)) )
print("Incubation period: 75 pctile  = "+str(numpy.percentile(numpy.sort(incubationPeriods), q=75)) )
print("Incubation period: 95 pctile  = "+str(numpy.percentile(numpy.sort(incubationPeriods), q=95)) )
print("Incubation period: 97.5 pctile  = "+str(numpy.percentile(numpy.sort(incubationPeriods), q=97.5)) )

# pyplot.hist(prodromalPeriods, bins=range(int(max(prodromalPeriods))), alpha=0.75, color='darkorange', label='$I_{pre}$')
# pyplot.hist(latentPeriods, bins=range(int(max(latentPeriods))), alpha=0.75, color='gold', label='$E$')
# pyplot.hist(incubationPeriods, bins=range(int(max(incubationPeriods))), alpha=0.5, color='black', label='Pre-symptomatic')
# pyplot.xlim(0,30)
# pyplot.xlabel('period (days)')
# pyplot.ylabel('num nodes')
# pyplot.legend(loc='upper right')
# seaborn.despine()
# pyplot.show()
# # pyplot.savefig(figdir+"/incubationPeriods"+runStr+".png")
# pyplot.close()

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

symptomaticPeriod_mean, symptomaticPeriod_coeffvar = 4.0, 0.4
symptomaticPeriod_scale     = symptomaticPeriod_mean*symptomaticPeriod_coeffvar**2    # gamma distn theta
symptomaticPeriod_shape     = symptomaticPeriod_mean/symptomaticPeriod_scale          # gamma distn k
symptomaticPeriods          = numpy.random.gamma(scale=symptomaticPeriod_scale, shape=symptomaticPeriod_shape, size=N)
GAMMA = 1 / symptomaticPeriods
print("Symptomatic period: mean       = " + str(numpy.mean(numpy.sort(symptomaticPeriods))) )
print("Symptomatic period: median     = "+str(numpy.percentile(numpy.sort(symptomaticPeriods), q=50)) )
print("Symptomatic period: 5  pctile  = "+str(numpy.percentile(numpy.sort(symptomaticPeriods), q=5)) )
print("Symptomatic period: 25 pctile  = "+str(numpy.percentile(numpy.sort(symptomaticPeriods), q=25)) )
print("Symptomatic period: 75 pctile  = "+str(numpy.percentile(numpy.sort(symptomaticPeriods), q=75)) )
print("Symptomatic period: 95 pctile  = "+str(numpy.percentile(numpy.sort(symptomaticPeriods), q=95)) )
print("Symptomatic period: 99.9pctile = "+str(numpy.percentile(numpy.sort(symptomaticPeriods), q=99)) )
print("Symptomatic period: % in 7-10  = "+str(len(symptomaticPeriods[ numpy.where( (symptomaticPeriods >= 7)&(symptomaticPeriods <= 10) ) ])/len(symptomaticPeriods)) )
print("GAMMA mean   = " +str(numpy.mean(GAMMA)) )
print("GAMMA median = " +str(numpy.median(GAMMA)) )

infectiousPeriods = prodromalPeriods + symptomaticPeriods
print("Infectious period: mean       = " + str(numpy.mean(numpy.sort(infectiousPeriods))) )
print("Infectious period: median     = "+str(numpy.percentile(numpy.sort(infectiousPeriods), q=50)) )
print("Infectious period: 5  pctile  = "+str(numpy.percentile(numpy.sort(infectiousPeriods), q=5)) )
print("Infectious period: 25 pctile  = "+str(numpy.percentile(numpy.sort(infectiousPeriods), q=25)) )
print("Infectious period: 75 pctile  = "+str(numpy.percentile(numpy.sort(infectiousPeriods), q=75)) )
print("Infectious period: 95 pctile  = "+str(numpy.percentile(numpy.sort(infectiousPeriods), q=95)) )

# pyplot.hist(prodromalPeriods, bins=range(int(max(prodromalPeriods))), alpha=0.75, color='darkorange', label='$I_{pre}$ period')
# pyplot.hist(symptomaticPeriods, bins=range(int(max(symptomaticPeriods))), alpha=0.75, color='crimson', label='$I_{S/A}$')
# pyplot.hist(infectiousPeriods, bins=range(int(max(infectiousPeriods))), alpha=0.5, color='black', label='Infectious')
# pyplot.xlim(0,30)
# pyplot.xlabel('period')
# pyplot.ylabel('num nodes')
# pyplot.legend(loc='upper right')
# seaborn.despine()
# pyplot.show()
# # pyplot.savefig(figdir+"/infectiousPeriods"+runStr+".png")
# pyplot.close()

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

onsetToHospitalizationPeriod_mean, onsetToHospitalizationPeriod_coeffvar = 11.0, 0.45
onsetToHospitalizationPeriod_scale  = onsetToHospitalizationPeriod_mean*onsetToHospitalizationPeriod_coeffvar**2    # gamma distn theta
onsetToHospitalizationPeriod_shape  = onsetToHospitalizationPeriod_mean/onsetToHospitalizationPeriod_scale          # gamma distn k
onsetToHospitalizationPeriods           = numpy.random.gamma(scale=onsetToHospitalizationPeriod_scale, shape=onsetToHospitalizationPeriod_shape, size=N)
ETA = 1 / onsetToHospitalizationPeriods
print("Onset-to-Hospitalization period: mean       = " + str(numpy.mean(numpy.sort(onsetToHospitalizationPeriods))) )
print("Onset-to-Hospitalization period: median     = "+str(numpy.percentile(numpy.sort(onsetToHospitalizationPeriods), q=50)) )
print("Onset-to-Hospitalization period: 5  pctile  = "+str(numpy.percentile(numpy.sort(onsetToHospitalizationPeriods), q=5)) )
print("Onset-to-Hospitalization period: 25 pctile  = "+str(numpy.percentile(numpy.sort(onsetToHospitalizationPeriods), q=25)) )
print("Onset-to-Hospitalization period: 75 pctile  = "+str(numpy.percentile(numpy.sort(onsetToHospitalizationPeriods), q=75)) )
print("Onset-to-Hospitalization period: 95 pctile  = "+str(numpy.percentile(numpy.sort(onsetToHospitalizationPeriods), q=95)) )
print("ETA mean   = " +str(numpy.mean(ETA)) )
print("ETA median = " +str(numpy.median(ETA)) )

hospitalizationToDischargePeriod_mean, hospitalizationToDischargePeriod_coeffvar = 11.0, 0.45
hospitalizationToDischargePeriod_scale  = hospitalizationToDischargePeriod_mean*hospitalizationToDischargePeriod_coeffvar**2    # gamma distn theta
hospitalizationToDischargePeriod_shape  = hospitalizationToDischargePeriod_mean/hospitalizationToDischargePeriod_scale          # gamma distn k
hospitalizationToDischargePeriods       = numpy.random.gamma(scale=hospitalizationToDischargePeriod_scale, shape=hospitalizationToDischargePeriod_shape, size=N)
GAMMA_H = 1 / hospitalizationToDischargePeriods
print("Hospitalization-to-Discharge period: mean       = " + str(numpy.mean(numpy.sort(hospitalizationToDischargePeriods))) )
print("Hospitalization-to-Discharge period: median     = "+str(numpy.percentile(numpy.sort(hospitalizationToDischargePeriods), q=50)) )
print("Hospitalization-to-Discharge period: 5  pctile  = "+str(numpy.percentile(numpy.sort(hospitalizationToDischargePeriods), q=5)) )
print("Hospitalization-to-Discharge period: 25 pctile  = "+str(numpy.percentile(numpy.sort(hospitalizationToDischargePeriods), q=25)) )
print("Hospitalization-to-Discharge period: 75 pctile  = "+str(numpy.percentile(numpy.sort(hospitalizationToDischargePeriods), q=75)) )
print("Hospitalization-to-Discharge period: 95 pctile  = "+str(numpy.percentile(numpy.sort(hospitalizationToDischargePeriods), q=95)) )
print("GAMMA_H mean   = " +str(numpy.mean(GAMMA_H)) )
print("GAMMA_H median = " +str(numpy.median(GAMMA_H)) )

onsetToDischargePeriods = onsetToHospitalizationPeriods + hospitalizationToDischargePeriods
print("Onset-to-Discharge period: mean       = " + str(numpy.mean(numpy.sort(onsetToDischargePeriods))) )
print("Onset-to-Discharge period: median     = "+str(numpy.percentile(numpy.sort(onsetToDischargePeriods), q=50)) )
print("Onset-to-Discharge period: 5  pctile  = "+str(numpy.percentile(numpy.sort(onsetToDischargePeriods), q=5)) )
print("Onset-to-Discharge period: 25 pctile  = "+str(numpy.percentile(numpy.sort(onsetToDischargePeriods), q=25)) )
print("Onset-to-Discharge period: 75 pctile  = "+str(numpy.percentile(numpy.sort(onsetToDischargePeriods), q=75)) )
print("Onset-to-Discharge period: 95 pctile  = "+str(numpy.percentile(numpy.sort(onsetToDischargePeriods), q=95)) )

# pyplot.hist(onsetToHospitalizationPeriods, bins=range(int(max(onsetToHospitalizationPeriods))), alpha=0.5, color='crimson', label='Onset-to-hospitalization ($I_S$)')
# pyplot.hist(hospitalizationToDischargePeriods, bins=range(int(max(hospitalizationToDischargePeriods))), alpha=0.5, color='violet', label='Hospitalization-to-discharge ($H$)')
# pyplot.hist(onsetToDischargePeriods, bins=range(int(max(onsetToDischargePeriods))), alpha=0.5, color='black', label='Onset-to-discharge')
# pyplot.xlim(0,40)
# pyplot.xlabel('period')
# pyplot.ylabel('num nodes')
# pyplot.legend(loc='upper right')
# seaborn.despine()
# pyplot.show()
# # pyplot.savefig(figdir+"/onsetToDischargePeriods"+runStr+".png")
# pyplot.close()

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

hospitalizationToDeathPeriod_mean, hospitalizationToDeathPeriod_coeffvar = 7.0, 0.45
hospitalizationToDeathPeriod_scale  = hospitalizationToDeathPeriod_mean*hospitalizationToDeathPeriod_coeffvar**2    # gamma distn theta
hospitalizationToDeathPeriod_shape  = hospitalizationToDeathPeriod_mean/hospitalizationToDeathPeriod_scale          # gamma distn k
hospitalizationToDeathPeriods       = numpy.random.gamma(scale=hospitalizationToDeathPeriod_scale, shape=hospitalizationToDeathPeriod_shape, size=N)
MU_H = 1 / hospitalizationToDeathPeriods
print("Hospitalization-to-Death period: mean       = " + str(numpy.mean(numpy.sort(hospitalizationToDeathPeriods))) )
print("Hospitalization-to-Death period: median     = "+str(numpy.percentile(numpy.sort(hospitalizationToDeathPeriods), q=50)) )
print("Hospitalization-to-Death period: 5  pctile  = "+str(numpy.percentile(numpy.sort(hospitalizationToDeathPeriods), q=5)) )
print("Hospitalization-to-Death period: 25 pctile  = "+str(numpy.percentile(numpy.sort(hospitalizationToDeathPeriods), q=25)) )
print("Hospitalization-to-Death period: 75 pctile  = "+str(numpy.percentile(numpy.sort(hospitalizationToDeathPeriods), q=75)) )
print("Hospitalization-to-Death period: 95 pctile  = "+str(numpy.percentile(numpy.sort(hospitalizationToDeathPeriods), q=95)) )
print(len(hospitalizationToDeathPeriods[ numpy.where( (hospitalizationToDeathPeriods >= 7)&(hospitalizationToDeathPeriods <= 10) ) ])/len(hospitalizationToDeathPeriods) )
print("MU_H mean   = " +str(numpy.mean(MU_H)) )
print("MU_H median = " +str(numpy.median(MU_H)) )

onsetToDeathPeriods = onsetToHospitalizationPeriods + hospitalizationToDeathPeriods
print("Onset-to-Death period: mean       = " + str(numpy.mean(numpy.sort(onsetToDeathPeriods))) )
print("Onset-to-Death period: median     = "+str(numpy.percentile(numpy.sort(onsetToDeathPeriods), q=50)) )
print("Onset-to-Death period: 5  pctile  = "+str(numpy.percentile(numpy.sort(onsetToDeathPeriods), q=5)) )
print("Onset-to-Death period: 25 pctile  = "+str(numpy.percentile(numpy.sort(onsetToDeathPeriods), q=25)) )
print("Onset-to-Death period: 75 pctile  = "+str(numpy.percentile(numpy.sort(onsetToDeathPeriods), q=75)) )
print("Onset-to-Death period: 95 pctile  = "+str(numpy.percentile(numpy.sort(onsetToDeathPeriods), q=95)) )

# pyplot.hist(onsetToHospitalizationPeriods, bins=range(int(max(onsetToHospitalizationPeriods))), alpha=0.5, color='crimson', label='Onset-to-hospitalization ($I_S$)')
# pyplot.hist(hospitalizationToDeathPeriods, bins=range(int(max(hospitalizationToDeathPeriods))), alpha=0.5, color='violet', label='Hospitalization-to-death')
# pyplot.hist(onsetToDeathPeriods, bins=range(int(max(onsetToDeathPeriods))), alpha=0.5, color='black', label='Onset-to-death')
# pyplot.xlim(0,40)
# pyplot.xlabel('period')
# pyplot.ylabel('num nodes')
# pyplot.legend(loc='upper right')
# seaborn.despine()
# pyplot.show()
# # pyplot.savefig(figdir+"/onsetToDeathPeriods"+runStr+".png")
# pyplot.close()

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

R0_mean, R0_coeffvar = R0_mean, R0_coeffvar
R0_scale      = R0_mean*R0_coeffvar**2    # gamma distn theta
R0_shape      = R0_mean/R0_scale          # gamma distn k
R0 = numpy.random.gamma(scale=R0_scale, shape=R0_shape, size=N)

print("R0: mean       = " + str(numpy.mean(numpy.sort(R0))) )
print("R0: median     = " + str(numpy.median(numpy.sort(R0))) )
print("R0: 5  pctile  = "+str(numpy.percentile(numpy.sort(R0), q=5)) )
print("R0: 25 pctile  = "+str(numpy.percentile(numpy.sort(R0), q=25)) )
print("R0: 50 pctile  = "+str(numpy.percentile(numpy.sort(R0), q=50)) )
print("R0: 75 pctile  = "+str(numpy.percentile(numpy.sort(R0), q=75)) )
print("R0: 95 pctile  = "+str(numpy.percentile(numpy.sort(R0), q=95)) )
print "R0: < 1:    "+str(numpy.count_nonzero(R0[R0 < 1])/N*100)+"%"
print "R0: < 2:    "+str(numpy.count_nonzero(R0[R0 < 2])/N*100)+"%"
print "R0: > 3:    "+str(numpy.count_nonzero(R0[R0 > 3])/N*100)+"%"
print "R0: > 4:    "+str(numpy.count_nonzero(R0[R0 > 4])/N*100)+"%"
print "R0: 80 pctile:   "+str(numpy.percentile(numpy.sort(R0), q=80))
print "R0: Top 20 does: "+str( numpy.sum(R0[R0 > numpy.percentile(numpy.sort(R0), q=80)])/numpy.sum(R0) )

# pyplot.hist(R0, bins=numpy.arange(0, int(max(R0+1)), step=0.1), alpha=0.5, color='crimson', label='$R_0$')
# pyplot.xlim(0,5)
# pyplot.xlabel('$R_0$')
# pyplot.ylabel('num nodes')
# pyplot.legend(loc='upper right')
# pyplot.show()
# # pyplot.savefig(figdir+"/R0distn"+runStr+".png")
# pyplot.close()

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

BETA     = 1/infectiousPeriods * R0
print("mean beta = "+str(numpy.mean(BETA)) )
print("beta: mean       = " + str(numpy.mean(numpy.sort(BETA))) )
print("beta: 5  pctile  = "+str(numpy.percentile(numpy.sort(BETA), q=5)) )
print("beta: 25 pctile  = "+str(numpy.percentile(numpy.sort(BETA), q=25)) )
print("beta: 50 pctile  = "+str(numpy.percentile(numpy.sort(BETA), q=50)) )
print("beta: 75 pctile  = "+str(numpy.percentile(numpy.sort(BETA), q=75)) )
print("beta: 95 pctile  = "+str(numpy.percentile(numpy.sort(BETA), q=95)) )

BETA_Q = BETA * (0.3/R0_mean) 
print("mean beta_D = "+str(numpy.mean(BETA_Q)) )
print("beta_Q: mean       = " + str(numpy.mean(numpy.sort(BETA_Q))) )
print("beta_Q: 5  pctile  = "+str(numpy.percentile(numpy.sort(BETA_Q), q=5)) )
print("beta_Q: 25 pctile  = "+str(numpy.percentile(numpy.sort(BETA_Q), q=25)) )
print("beta_Q: 50 pctile  = "+str(numpy.percentile(numpy.sort(BETA_Q), q=50)) )
print("beta_Q: 75 pctile  = "+str(numpy.percentile(numpy.sort(BETA_Q), q=75)) )
print("beta_Q: 95 pctile  = "+str(numpy.percentile(numpy.sort(BETA_Q), q=95)) )


#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


# Source: Verity et al.
ageGroup_pctHospitalized = {'0-9':      0.0000,
                            '10-19':    0.0004,
                            '20-29':    0.0104,
                            '30-39':    0.0343,
                            '40-49':    0.0425,
                            '50-59':    0.0816,
                            '60-69':    0.118,
                            '70-79':    0.166,
                            '80+':      0.184 }
# PCT_HOSPITALIZED = [ageGroup_pctHospitalized[ageGroup] for ageGroup in individual_ageGroups]
PCT_HOSPITALIZED = numpy.mean([ageGroup_pctHospitalized[age] for age in ['20-29', '30-39', '40-49', '50-59']])
print PCT_HOSPITALIZED


#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


# Source: Verity et al.
ageGroup_hospitalFatalityRate = {'0-9':     0.0000,
                                 '10-19':   0.3627,
                                 '20-29':   0.0577,
                                 '30-39':   0.0426,
                                 '40-49':   0.0694,
                                 '50-59':   0.1532,
                                 '60-69':   0.3381,
                                 '70-79':   0.5187,
                                 '80+':     0.7283 }
# PCT_FATALITY = [ageGroup_hospitalFatalityRate[ageGroup] for ageGroup in individual_ageGroups]
PCT_FATALITY = numpy.mean([ageGroup_hospitalFatalityRate[age] for age in ['20-29', '30-39', '40-49', '50-59']])
print PCT_FATALITY


#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^



#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

###################################################
# # SCENARIO
###################################################

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

model = ExtSEIRSNetworkModel(G=G_baseline, p=P_GLOBALINTXN,
                                        beta=BETA, sigma=SIGMA, lamda=LAMDA, gamma=GAMMA, 
                                        gamma_asym=GAMMA, eta=ETA, gamma_H=GAMMA_H, mu_H=MU_H, 
                                        a=PCT_ASYMPTOMATIC, h=PCT_HOSPITALIZED, f=PCT_FATALITY,              
                                        beta_pairwise_mode=BETA_PAIRWISE_MODE, alpha_pairwise_mode=ALPHA_PAIRWISE_MODE,
                                        G_Q=G_quarantine, q=0, beta_Q=BETA_Q, isolation_time=14,
                                        initE=INIT_EXPOSED, transition_mode=TRANSITION_MODE)


#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

interventionInterval = run_rtw_testing_sim(model=model, T=T, 
                        intervention_start_pct_infected=0,
                        test_falseneg_rate='temporal', testing_cadence='workday', pct_tested_per_day=1.0,
                        testing_compliance_rate=1.0, symptomatic_seektest_compliance_rate=0.3, 
                        isolation_lag=1, positive_isolation_unit='individual', # 'team', 'workplace'
                        positive_isolation_compliance_rate=0.8, symptomatic_selfiso_compliance_rate=0.3, 
                        teams=teams, average_introductions_per_day=1/7)


#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

print("total percent infected: %0.2f%%" % ((model.total_num_infected()[-1]+model.total_num_recovered()[-1])/N * 100) )
print("total percent fatality: %0.2f%%" % (model.numF[-1]/N * 100) )
print("peak  pct hospitalized: %0.2f%%" % (numpy.max(model.numH)/N * 100) )

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

fig, ax = model.figure_infections(combine_Q_infected=False, show=False, use_seaborn=False,
                                    plot_Q_R='stacked', #plot_R='stacked', 
                                    plot_Q_S='stacked', #plot_S='stacked', 
                                    #dashed_reference_results=model_det,
                                    vlines=[interventionInterval[0]]
                                 )

fig.savefig(figdir+"/prevalence"+runStr+".png")

