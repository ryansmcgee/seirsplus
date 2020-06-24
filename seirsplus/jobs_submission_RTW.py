#!/usr/bin/env python

#~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~

from __future__ import division
import os

#~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~

import argparse

parser  = argparse.ArgumentParser()
parser.add_argument("-c",  "--cluster", default="")
parser.add_argument("-ckpt", "--ckpt", action='store_true')
parser.add_argument("-go", "--launch", action='store_true')
args    = parser.parse_args()

#~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~

working_path            = os.getcwd()
working_dir_name        = os.path.basename(working_path)

jobs_dir = "./jobs/"
logs_dir = "./logs/"

print(working_path)
print(working_dir_name)

data_outdir     = os.path.join(working_path, 'results/')
if not os.path.exists(data_outdir):
    os.mkdir(data_outdir)

figs_outdir    = os.path.join(working_path, 'figs/')
if not os.path.exists(figs_outdir):
    os.mkdir(figs_outdir)

jobs_dir    = os.path.join(working_path, 'jobs/')
if not os.path.exists(jobs_dir):
    os.mkdir(jobs_dir)

logs_dir    = os.path.join(working_path, 'logs/')
if not os.path.exists(logs_dir):
    os.mkdir(logs_dir)

#~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~

if(args.cluster == 'hyak'):
    max_cores = 11
elif(args.cluster == 'abisko'):
    max_cores = 6
else:
    max_cores = 1

if(args.cluster == 'hyak'):
    max_mem = 120
elif(args.cluster == 'abisko'):
    max_mem = 100
else:
    max_mem = 100

#~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~

REP_VALS                                    = [1, 2, 3, 4, 5]

N_VALS                                      = [100000] 

FRAC_TO_TEST_PER_INTERVAL_VALS              = [0.20, 0.10, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.0000]
                                              #[1/2**3, 1/2**4, 1/2**5, 1/2**6, 1/2**7, 1/2**8, 1/2**9, 1/2**10, 1/2**11, 1/2**12, 1/2**13, 0.0]
FRAC_CONTACTS_TO_TRACE_VALS                 = [1.00, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.00]

R0_MEAN_VALS                                = [1.5, 2.0, 2.5]
R0_COEFFVAR_VALS                            = [0.2, 2.0]

P_GLOBALINTXN_VALS                          = [0.25]#, 0.0, 1.0]

DO_ISOLATE_POSITIVE_HOUSEHOLDS_VALS         = [0, 1]    
DO_ISOLATE_TRACING_SELFISO_HOUSEHOLDS_VALS  = [0, 1]

TRANSITION_MODE_VALS                        = ['exponential_rates']#, 'time_in_state']








#~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~


for N in N_VALS:
    for TRANSITION_MODE in TRANSITION_MODE_VALS:
        for P in P_GLOBALINTXN_VALS:
            for R0 in R0_MEAN_VALS:
                for DO_ISOLATE_TRACING_SELFISO_HOUSEHOLDS in DO_ISOLATE_TRACING_SELFISO_HOUSEHOLDS_VALS:
                    for DO_ISOLATE_POSITIVE_HOUSEHOLDS in DO_ISOLATE_POSITIVE_HOUSEHOLDS_VALS:
                        for FRAC_TEST in FRAC_TO_TEST_PER_INTERVAL_VALS:
                            for FRAC_TRACE in FRAC_CONTACTS_TO_TRACE_VALS:

                                # Assume that self-isolation of traced households only in effect
                                # if self-isolation of positive households also in effect.
                                if(DO_ISOLATE_TRACING_SELFISO_HOUSEHOLDS and not DO_ISOLATE_POSITIVE_HOUSEHOLDS):
                                    continue

                                job_name    = ("sim"
                                                # +"_N"+str(N)
                                                +"_R0"+str(R0)[0]+"pt"+str(R0)[2:4]#+"cv"+str(R0_COEFFVAR)[0]+"pt"+str(R0_COEFFVAR)[2:4]
                                                +"_p"+str(P)[0]+"pt"+str(P)[2:4]
                                                +"_fracTest"+str(FRAC_TEST)[0]+"pt"+str(FRAC_TEST)[2:6]
                                                +"_fracTrace"+str(FRAC_TRACE)[0]+"pt"+str(FRAC_TRACE)[2:4]
                                                +("_isoPosHH" if DO_ISOLATE_POSITIVE_HOUSEHOLDS else "")
                                                +("_isoTraceHH" if DO_ISOLATE_TRACING_SELFISO_HOUSEHOLDS else "")
                                                +"_trans"+("Rates" if TRANSITION_MODE=='exponential_rates' else "Timer" if TRANSITION_MODE=='time_in_state' else "")
                                                # +"_"+runID
                                             )

                                job_file    = jobs_dir+'/'+job_name+'.job'

                                print("Creating job: " + job_name)

                                with open(job_file, 'w') as jf:

                                    jf.write("#!/bin/bash\n")
                                    jf.write("#SBATCH --job-name="+job_name+".job\n")
                                    if(args.cluster == "abisko"):
                                        jf.write("#SBATCH --account=snic2020-9-24\n") # snic2020-9-24 is Martin's group name
                                    elif(args.cluster == "hyak"):
                                        if(args.ckpt):
                                            jf.write("#SBATCH --account=stf-ckpt\n") 
                                            jf.write("#SBATCH --partition=ckpt\n") 
                                        else:
                                            jf.write("#SBATCH --account=stf\n") 
                                            jf.write("#SBATCH --partition=stf\n") 
                                    jf.write("#SBATCH --nodes=1\n")
                                    jf.write("#SBATCH --cores="+str(max_cores)+"\n")
                                    jf.write("#SBATCH --mem="+str(max_mem)+"G\n")
                                    jf.write("#SBATCH --time=24:00:00\n")
                                    jf.write("#SBATCH --output="+logs_dir+"/%x-%j.out\n")
                                    jf.write("#SBATCH --error="+logs_dir+"/%x-%j.err\n")
                                    jf.write("#SBATCH --export=all\n")
                                    jf.write("##SBATCH --qos=short\n")
                                    jf.write("\n")
                                    if(args.cluster == "abisko"):
                                        jf.write("ml GCC/6.4.0-2.28 OpenMPI/2.1.2\n")
                                        jf.write("ml Python/2.7.14\n")
                                        jf.write("ml matplotlib/2.1.2-Python-2.7.14\n")
                                        jf.write("\n")

                                    for R0_COEFFVAR in R0_COEFFVAR_VALS:
                                        for REP in REP_VALS:

                                            runID       = "rep"+str(REP)

                                            jf.write("python "+working_path+"/simulation_whitepaper.py"
                                                        +" --runID "                          +str(runID)
                                                        +" --N "                              +str(N)
                                                        +" --R0_mean "                        +str(R0)
                                                        +" --R0_coeffvar "                    +str(R0_COEFFVAR)
                                                        +" --P_GLOBALINTXN "                  +str(P)
                                                        +" --FRAC_TO_TEST_PER_INTERVAL "      +str(FRAC_TEST)
                                                        +" --FRAC_CONTACTS_TO_TRACE "         +str(FRAC_TRACE)
                                                        +" --DO_ISOLATE_POSITIVE_HOUSEHOLDS " +str(DO_ISOLATE_POSITIVE_HOUSEHOLDS)
                                                        +" --DO_ISOLATE_TRACING_SELFISO_HOUSEHOLDS " +str(DO_ISOLATE_TRACING_SELFISO_HOUSEHOLDS)
                                                        +" --DO_TRACING_SELFISO "             +str(DO_ISOLATE_TRACING_SELFISO_HOUSEHOLDS) # DO_ISOLATE_TRACING_SELFISO_HOUSEHOLDS should match DO_TRACING_SELFISO
                                                        +" --TRANSITION_MODE "                +str(TRANSITION_MODE)
                                                        +" --outdir "                         +str(data_outdir)
                                                        +" --figdir "                         +str(figs_outdir)
                                                    )
                                            if(len(P_GLOBALINTXN_VALS) > 1 or len(R0_COEFFVAR_VALS)>1):
                                                jf.write(" &\n")

                                    if(len(REP_VALS) > 1 or len(R0_COEFFVAR_VALS)>1):
                                            jf.write("wait")

                                if(args.launch):
                                    if(args.cluster == "abisko"):
                                        os.system("sbatch -A snic2020-9-24 %s" % job_file)
                                    elif(args.cluster == "hyak"):
                                        if(args.ckpt):
                                            os.system("sbatch -p ckpt -A stf-ckpt %s" % job_file)
                                        else:
                                            os.system("sbatch -p stf -A stf %s" % job_file)



                       