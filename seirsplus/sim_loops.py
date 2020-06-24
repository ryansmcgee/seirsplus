from __future__ import division
import pickle
import numpy

import time



def run_manual_random_testing_sim(model, T,
                                    testing_interval, tests_per_interval, test_falseneg_rate, intervention_start_pct_infected=0,
                                    do_random_testing=False, testing_random_compliance=None, testing_degree_bias=0,
                                    do_tracing_testing=False, tracing_compliance=None, testing_traced_compliance=None, max_pct_tests_for_tracing=1.0,
                                    num_contacts_to_trace=None, frac_contacts_to_trace=1.0, tracing_interval_lag=1,
                                    do_seeking_testing=False, symptomatic_seektest_compliance=None, testing_selfseek_compliance=None, max_pct_tests_for_seeking=1.0,
                                    do_symptom_selfiso=False, symptomatic_selfiso_compliance=None,
                                    do_tracing_selfiso=False, tracing_selfiso_compliance=None,
                                    isolate_positive_households=False, isolate_symptomatic_households=False, isolate_tracing_selfiso_households=False,
                                    household_isolation_compliance=None, households=None,
                                    print_interval=10, timeOfLastPrint=-1, verbose='t'):

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    temporal_falseneg_rates = {
                                model.E:        {0: 1.00, 1: 1.00, 2: 1.00, 3: 0.97},
                                model.I_pre:    {0: 0.97, 1: 0.67, 2: 0.38},
                                model.I_sym:    {0: 0.24, 1: 0.18, 2: 0.17, 3: 0.18, 4: 0.20, 5: 0.23, 6: 0.26, 7: 0.30, 8: 0.34, 9: 0.38, 10: 0.43, 11: 0.48, 12: 0.52, 13: 0.57, 14: 0.61, 15: 0.65, 16: 0.69, 17: 0.76, 18: 0.79, 19: 0.82, 20: 0.85, 21: 0.88, 22: 0.90, 23: 0.92, 24: 0.93, 25: 0.95, 26: 0.96, 27: 0.97, 28: 0.97, 29: 0.98, 30: 0.98, 31: 0.99},
                                model.I_asym:   {0: 0.24, 1: 0.18, 2: 0.17, 3: 0.18, 4: 0.20, 5: 0.23, 6: 0.26, 7: 0.30, 8: 0.34, 9: 0.38, 10: 0.43, 11: 0.48, 12: 0.52, 13: 0.57, 14: 0.61, 15: 0.65, 16: 0.69, 17: 0.76, 18: 0.79, 19: 0.82, 20: 0.85, 21: 0.88, 22: 0.90, 23: 0.92, 24: 0.93, 25: 0.95, 26: 0.96, 27: 0.97, 28: 0.97, 29: 0.98, 30: 0.98, 31: 0.99},
                                model.Q_E:      {0: 1.00, 1: 1.00, 2: 1.00, 3: 0.97},
                                model.Q_pre:    {0: 0.97, 1: 0.67, 2: 0.38},
                                model.Q_sym:    {0: 0.24, 1: 0.18, 2: 0.17, 3: 0.18, 4: 0.20, 5: 0.23, 6: 0.26, 7: 0.30, 8: 0.34, 9: 0.38, 10: 0.43, 11: 0.48, 12: 0.52, 13: 0.57, 14: 0.61, 15: 0.65, 16: 0.69, 17: 0.76, 18: 0.79, 19: 0.82, 20: 0.85, 21: 0.88, 22: 0.90, 23: 0.92, 24: 0.93, 25: 0.95, 26: 0.96, 27: 0.97, 28: 0.97, 29: 0.98, 30: 0.98, 31: 0.99},
                                model.Q_asym:   {0: 0.24, 1: 0.18, 2: 0.17, 3: 0.18, 4: 0.20, 5: 0.23, 6: 0.26, 7: 0.30, 8: 0.34, 9: 0.38, 10: 0.43, 11: 0.48, 12: 0.52, 13: 0.57, 14: 0.61, 15: 0.65, 16: 0.69, 17: 0.76, 18: 0.79, 19: 0.82, 20: 0.85, 21: 0.88, 22: 0.90, 23: 0.92, 24: 0.93, 25: 0.95, 26: 0.96, 27: 0.97, 28: 0.97, 29: 0.98, 30: 0.98, 31: 0.99},
                              }

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Custom simulation loop:
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    interventionOn        = False
    interventionStartTime = None

    max_tracing_tests_per_interval = int(tests_per_interval * max_pct_tests_for_tracing)
    max_seeking_tests_per_interval = int(tests_per_interval * max_pct_tests_for_seeking)

    timeOfLastTestingInterval = -1

    tracingPoolQueue = [[] for i in range(tracing_interval_lag)]

    model.tmax  = T
    running     = True
    while running:

        # time_start_iter = time.time()
        running = model.run_iteration()
        # time_end_iter = time.time()
        # print("time iter = "+str(time_end_iter - time_start_iter))

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Execute testing policy at designated intervals:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if(int(model.t)%testing_interval == 0 and int(model.t)!=int(timeOfLastTestingInterval)):

            timeOfLastTestingInterval = model.t

            currentPctInfected = model.total_num_infected()[model.tidx]/model.numNodes

            if(currentPctInfected >= intervention_start_pct_infected and not interventionOn):
                interventionOn        = True
                interventionStartTime = model.t

            if(interventionOn):

                print("[TESTING @ t = %.2f (%.2f%% infected)]" % (model.t, currentPctInfected*100))

                nodeStates                       = model.X.flatten()
                nodeTestedStatuses               = model.tested.flatten()
                nodeTestedInCurrentStateStatuses = model.testedInCurrentState.flatten()
                nodePositiveStatuses             = model.positive.flatten()

                tracingPool = tracingPoolQueue.pop(0)

                print("\t"+str(numpy.count_nonzero((nodePositiveStatuses==False)
                                                    &(nodeStates!=model.S)&(nodeStates!=model.Q_S)
                                                    &(nodeStates!=model.R)&(nodeStates!=model.Q_R)
                                                    &(nodeStates!=model.H)&(nodeStates!=model.F)))+" current undetected infections")

                #----------------------------------------
                # Manually enforce that some percentage of symptomatic cases self-isolate without a test
                #----------------------------------------
                if(do_symptom_selfiso):
                    symptomaticNodes = numpy.argwhere((nodeStates==model.I_sym)).flatten()
                    numSelfIsolated_symptoms = 0
                    numSelfIsolated_symptomaticHousemate = 0
                    for symptomaticNode in symptomaticNodes:
                        if(symptomatic_selfiso_compliance[symptomaticNode]):
                            if(model.X[symptomaticNode] == model.I_sym):
                                model.set_isolation(symptomaticNode, True)
                                numSelfIsolated_symptoms += 1

                                # Isolate the housemates of this symptomatic node (if applicable):
                                if(isolate_symptomatic_households):
                                    isolationHousemates = next((household['indices'] for household in households if symptomaticNode in household['indices']), None)
                                    for isolationHousemate in isolationHousemates:
                                        if(isolationHousemate != symptomaticNode):
                                            if(household_isolation_compliance[isolationHousemate]):
                                                model.set_isolation(isolationHousemate, True)
                                                numSelfIsolated_symptomaticHousemate += 1

                    print("\t"+str(numSelfIsolated_symptoms)+" self-isolated due to symptoms ("+str(numSelfIsolated_symptomaticHousemate)+" as housemates of symptomatic)")

                #----------------------------------------
                # Manually enforce that some percentage of the contacts of
                # detected positive cases self-isolate without a test:
                #----------------------------------------
                if(do_tracing_selfiso):
                    numSelfIsolated_trace = 0
                    numSelfIsolated_traceHousemate = 0
                    for traceNode in tracingPool:
                        if(tracing_selfiso_compliance[traceNode]):
                            model.set_isolation(traceNode, True)
                            numSelfIsolated_trace += 1

                            # Isolate the housemates of this self-isolating node (if applicable):
                            if(isolate_tracing_selfiso_households):
                                isolationHousemates = next((household['indices'] for household in households if traceNode in household['indices']), None)
                                for isolationHousemate in isolationHousemates:
                                    if(isolationHousemate != traceNode):
                                        if(household_isolation_compliance[isolationHousemate]):
                                            model.set_isolation(isolationHousemate, True)
                                            numSelfIsolated_traceHousemate += 1

                    print("\t"+str(numSelfIsolated_trace)+" self-isolated due to positive contact ("+str(numSelfIsolated_traceHousemate)+" as housemates of self-isolating contact)")

                #----------------------------------------
                # Update the nodeStates list after self-isolation updates to model.X:
                #----------------------------------------
                nodeStates = model.X.flatten()

                #----------------------------------------
                # Manually apply a designated portion of this interval's tests
                # to individuals identified by contact tracing:
                #----------------------------------------
                tracingSelection = []
                if(do_tracing_testing):
                    numTracingTests = min(len(tracingPool), max_tracing_tests_per_interval)
                    for trace in range(numTracingTests):
                        traceNode = tracingPool.pop()
                        if((nodePositiveStatuses[traceNode]==False)
                            and (testing_traced_compliance[traceNode]==True)
                            and (model.X[traceNode] != model.R)
                            and (model.X[traceNode] != model.Q_R)
                            and (model.X[traceNode] != model.H)
                            and (model.X[traceNode] != model.F)):
                            tracingSelection.append(traceNode)

                #----------------------------------------
                # Manually apply a designated portion of this interval's tests
                # to individuals self-seeking tests:
                #----------------------------------------
                seekingSelection = []
                if(do_seeking_testing):

                    seekingPool = numpy.argwhere((testing_selfseek_compliance==True)
                                                 & (nodeTestedInCurrentStateStatuses==False)
                                                 & (nodePositiveStatuses==False)
                                                 & ((nodeStates==model.I_sym)|(nodeStates==model.Q_sym))
                                                ).flatten()

                    numSeekingTests  = min(tests_per_interval-len(tracingSelection), max_seeking_tests_per_interval)

                    if(len(seekingPool) > 0):
                        seekingSelection = seekingPool[numpy.random.choice(len(seekingPool), min(numSeekingTests, len(seekingPool)), replace=False)]

                #----------------------------------------
                # Manually apply the remainder of this interval's tests to random testing policy:
                #----------------------------------------
                randomSelection = []
                if(do_random_testing):

                    testingPool = numpy.argwhere((testing_random_compliance==True)
                                                 & (nodePositiveStatuses==False)
                                                 & (nodeStates != model.R)
                                                 & (nodeStates != model.Q_R)
                                                 & (nodeStates != model.H)
                                                 & (nodeStates != model.F)
                                                ).flatten()

                    numRandomTests = max(tests_per_interval-len(tracingSelection)-len(seekingSelection), 0)

                    testingPool_degrees       = model.degree.flatten()[testingPool]
                    testingPool_degreeWeights = numpy.power(testingPool_degrees,testing_degree_bias)/numpy.sum(numpy.power(testingPool_degrees,testing_degree_bias))

                    if(len(testingPool) > 0):
                        randomSelection = testingPool[numpy.random.choice(len(testingPool), numRandomTests, p=testingPool_degreeWeights, replace=False)]

                #----------------------------------------
                # Perform the tests on the selected individuals:
                #----------------------------------------
                selectedToTest = numpy.concatenate((tracingSelection, seekingSelection, randomSelection)).astype(int)

                # numSelected            = len(selectedToTest)
                # numSelected_random     = len(randomSelection)
                # numSelected_tracing    = len(tracingSelection)
                # numSelected_selfseek   = len(seekingSelection)
                numTested              = 0
                numTested_random       = 0
                numTested_tracing      = 0
                numTested_selfseek     = 0
                numPositive            = 0
                numPositive_random     = 0
                numPositive_tracing    = 0
                numPositive_selfseek   = 0
                numIsolated_positiveHousemate = 0
                newTracingPool = []
                for i, testNode in enumerate(selectedToTest):

                    model.set_tested(testNode, True)

                    numTested += 1
                    if(i < len(tracingSelection)):
                        numTested_tracing  += 1
                    elif(i < len(tracingSelection)+len(seekingSelection)):
                        numTested_selfseek += 1
                    else:
                        numTested_random   += 1

                    # If the node to be tested is not infected, then the test is guaranteed negative,
                    # so don't bother going through with doing the test:
                    if(model.X[testNode] == model.S or model.X[testNode] == model.Q_S):
                        pass
                    # Also assume that latent infections are not picked up by tests:
                    elif(model.X[testNode] == model.E or model.X[testNode] == model.Q_E):
                        pass
                    elif(model.X[testNode] == model.I_pre or model.X[testNode] == model.Q_pre
                         or model.X[testNode] == model.I_sym or model.X[testNode] == model.Q_sym
                         or model.X[testNode] == model.I_asym or model.X[testNode] == model.Q_asym):

                        if(test_falseneg_rate == 'temporal'):
                            testNodeState       = model.X[testNode][0]
                            testNodeTimeInState = model.timer_state[testNode][0]
                            if(testNodeState in temporal_falseneg_rates.keys()):
                                falseneg_prob = temporal_falseneg_rates[testNodeState][ int(min(testNodeTimeInState, max(temporal_falseneg_rates[testNodeState].keys()))) ]
                            else:
                                falseneg_prob = 1.00
                        else:
                            falseneg_prob = test_falseneg_rate

                        if(numpy.random.rand() < (1-falseneg_prob)):
                            # The tested node has returned a positive test
                            numPositive += 1
                            if(i < len(tracingSelection)):
                                numPositive_tracing  += 1
                            elif(i < len(tracingSelection)+len(seekingSelection)):
                                numPositive_selfseek += 1
                            else:
                                numPositive_random   += 1

                            # Update the node's state to the appropriate detcted case state:
                            model.set_positive(testNode, True)
                            model.set_isolation(testNode, True)

                            # Add this node's neighbors to the contact tracing pool:
                            if(do_tracing_testing or do_tracing_selfiso):
                                if(tracing_compliance[testNode]):
                                    testNodeContacts = list(model.G[testNode].keys())
                                    numpy.random.shuffle(testNodeContacts)
                                    if(num_contacts_to_trace is None):
                                        numContactsToTrace = int(frac_contacts_to_trace*len(testNodeContacts))
                                    else:
                                        numContactsToTrace = num_contacts_to_trace
                                    newTracingPool.extend(testNodeContacts[0:numContactsToTrace])

                            # Isolate the housemates of this positive node (if applicable):
                            if(isolate_positive_households):
                                isolationHousemates = next((household['indices'] for household in households if testNode in household['indices']), None)
                                for isolationHousemate in isolationHousemates:
                                    if(isolationHousemate != testNode):
                                        if(household_isolation_compliance[isolationHousemate]):
                                            numIsolated_positiveHousemate += 1
                                            model.set_isolation(isolationHousemate, True)

                # Add the nodes to be traced to the tracing queue:
                tracingPoolQueue.append(newTracingPool)

                print("\t"+str(numPositive)+" isolated due to positive test ("+str(numIsolated_positiveHousemate)+" as housemates of positive)")

                # print("\t"+str(len(selectedToTest))+" selected ("+str(len(tracingSelection))+" as traces, "+str(len(seekingSelection))+" self-seeking), "+str(numTested)+" tested, [+ "+str(numPositive)+" positive +]")

                print("\t"+str(numTested_selfseek)+"\ttested self-seek\t[+ "+str(numPositive_selfseek)+" positive (%.2f %%) +]" % (numPositive_selfseek/numTested_selfseek*100 if numTested_selfseek>0 else 0))
                print("\t"+str(numTested_tracing)+"\ttested as traces\t[+ "+str(numPositive_tracing)+" positive (%.2f %%) +]" % (numPositive_tracing/numTested_tracing*100 if numTested_tracing>0 else 0))
                print("\t"+str(numTested_random)+"\ttested randomly\t\t[+ "+str(numPositive_random)+" positive (%.2f %%) +]" % (numPositive_random/numTested_random*100 if numTested_random>0 else 0))
                print("\t"+str(numTested)+"\ttested TOTAL\t[+ "+str(numPositive)+" positive (%.2f %%) +]" % (numPositive/numTested*100 if numTested>0 else 0))

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            else:
                print("[no intervention @ t = %.2f (%.2f%% infected)]" % (model.t, currentPctInfected*100))

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # if(print_interval):
        #     if(int(model.t)%print_interval == 0 and int(model.t)!=int(timeOfLastPrint)):
        #         timeOfLastPrint = model.t
        #         if(verbose=="t"):
        #             print("t = %.2f" % model.t)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    interventionInterval = (interventionStartTime, model.t)

    return interventionInterval


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def run_rtw_testing_sim(model, T,
                        intervention_start_pct_infected=0,
                        test_falseneg_rate='temporal', testing_cadence='everyday', pct_tested_per_day=1.0,
                        testing_compliance_rate=1.0, symptomatic_seektest_compliance_rate=0.0,
                        isolation_lag=1, positive_isolation_unit='individual', # 'team', 'workplace'
                        positive_isolation_compliance_rate=0.8, symptomatic_selfiso_compliance_rate=0.3,
                        teams=None, average_introductions_per_day=1/7,
                        print_interval=10, timeOfLastPrint=-1, verbose='t'):

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Testing cadences involve a repeating 28 day cycle starting on a Monday
    # (0:Mon, 1:Tue, 2:Wed, 3:Thu, 4:Fri, 5:Sat, 6:Sun, 7:Mon, 8:Tues, ...)
    # For each cadence, testing is done on the day numbers included in the associated list.

    cadence_testing_days    = {
                                'everyday':     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27],
                                'workday':      [0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25],
                                'semiweekly':   [0, 3, 7, 10, 14, 17, 21, 24],
                                'weekly':       [0, 7, 14, 21],
                                'biweekly':     [0, 14],
                                'monthly':      [0]
                            }

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    temporal_falseneg_rates = {
                                model.E:        {0: 1.00, 1: 1.00, 2: 1.00, 3: 1.00},
                                model.I_pre:    {0: 0.24, 1: 0.24, 2: 0.24},
                                model.I_sym:    {0: 0.24, 1: 0.18, 2: 0.17, 3: 0.18, 4: 0.20, 5: 0.23, 6: 0.26, 7: 0.30, 8: 0.34, 9: 0.38, 10: 0.43, 11: 0.48, 12: 0.52, 13: 0.57, 14: 0.61, 15: 0.65, 16: 0.69, 17: 0.76, 18: 0.79, 19: 0.82, 20: 0.85, 21: 0.88, 22: 0.90, 23: 0.92, 24: 0.93, 25: 0.95, 26: 0.96, 27: 0.97, 28: 0.97, 29: 0.98, 30: 0.98, 31: 0.99},
                                model.I_asym:   {0: 0.24, 1: 0.18, 2: 0.17, 3: 0.18, 4: 0.20, 5: 0.23, 6: 0.26, 7: 0.30, 8: 0.34, 9: 0.38, 10: 0.43, 11: 0.48, 12: 0.52, 13: 0.57, 14: 0.61, 15: 0.65, 16: 0.69, 17: 0.76, 18: 0.79, 19: 0.82, 20: 0.85, 21: 0.88, 22: 0.90, 23: 0.92, 24: 0.93, 25: 0.95, 26: 0.96, 27: 0.97, 28: 0.97, 29: 0.98, 30: 0.98, 31: 0.99},
                                model.Q_E:      {0: 1.00, 1: 1.00, 2: 1.00, 3: 1.00},
                                model.Q_pre:    {0: 0.24, 1: 0.24, 2: 0.24},
                                model.Q_sym:    {0: 0.24, 1: 0.18, 2: 0.17, 3: 0.18, 4: 0.20, 5: 0.23, 6: 0.26, 7: 0.30, 8: 0.34, 9: 0.38, 10: 0.43, 11: 0.48, 12: 0.52, 13: 0.57, 14: 0.61, 15: 0.65, 16: 0.69, 17: 0.76, 18: 0.79, 19: 0.82, 20: 0.85, 21: 0.88, 22: 0.90, 23: 0.92, 24: 0.93, 25: 0.95, 26: 0.96, 27: 0.97, 28: 0.97, 29: 0.98, 30: 0.98, 31: 0.99},
                                model.Q_asym:   {0: 0.24, 1: 0.18, 2: 0.17, 3: 0.18, 4: 0.20, 5: 0.23, 6: 0.26, 7: 0.30, 8: 0.34, 9: 0.38, 10: 0.43, 11: 0.48, 12: 0.52, 13: 0.57, 14: 0.61, 15: 0.65, 16: 0.69, 17: 0.76, 18: 0.79, 19: 0.82, 20: 0.85, 21: 0.88, 22: 0.90, 23: 0.92, 24: 0.93, 25: 0.95, 26: 0.96, 27: 0.97, 28: 0.97, 29: 0.98, 30: 0.98, 31: 0.99},
                              }

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Custom simulation loop:
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    interventionOn        = False
    interventionStartTime = None

    timeOfLastIntervention = -1
    timeOfLastIntroduction = -1

    testingDays      = cadence_testing_days[testing_cadence]
    cadenceDayNumber = 0

    testing_compliance              = (numpy.random.rand(model.numNodes) < testing_compliance_rate)
    symptomatic_seektest_compliance = (numpy.random.rand(model.numNodes) < symptomatic_seektest_compliance_rate)
    symptomatic_selfiso_compliance  = (numpy.random.rand(model.numNodes) < symptomatic_selfiso_compliance_rate)
    positive_isolation_compliance   = (numpy.random.rand(model.numNodes) < positive_isolation_compliance_rate)

    isolationQueue = [[] for i in range(isolation_lag)]

    model.tmax  = T
    running     = True
    while running:

        running = model.run_iteration()

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Introduce exogenous exposures randomly at designated intervals:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if(int(model.t)!=int(timeOfLastIntroduction)):

            timeOfLastIntroduction = model.t

            numNewExposures=numpy.random.poisson(lam=average_introductions_per_day)

            model.introduce_exposures(num_new_exposures=numNewExposures)

            if(numNewExposures > 0):
                print("[NEW EXPOSURE @ t = %.2f (%d exposed)]" % (model.t, numNewExposures))

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Execute testing policy at designated intervals:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        if(int(model.t)!=int(timeOfLastIntervention)):

            cadenceDayNumber = int(model.t % 28)

            timeOfLastIntervention = model.t

            currentNumInfected = model.total_num_infected()[model.tidx]
            currentPctInfected = model.total_num_infected()[model.tidx]/model.numNodes

            if(currentPctInfected >= intervention_start_pct_infected and not interventionOn):
                interventionOn        = True
                interventionStartTime = model.t

            if(interventionOn):

                print("[INTERVENTIONS @ t = %.2f (%d (%.2f%%) infected)]" % (model.t, currentNumInfected, currentPctInfected*100))

                nodeStates                       = model.X.flatten()
                nodeTestedStatuses               = model.tested.flatten()
                nodeTestedInCurrentStateStatuses = model.testedInCurrentState.flatten()
                nodePositiveStatuses             = model.positive.flatten()

                #----------------------------------------
                # Manually enforce that some percentage of symptomatic cases self-isolate without a test
                #----------------------------------------
                symptomaticNodes = numpy.argwhere((nodeStates==model.I_sym)).flatten()
                numSelfIsolated_symptoms = 0
                for symptomaticNode in symptomaticNodes:
                    if(symptomatic_selfiso_compliance[symptomaticNode]):
                        if(model.X[symptomaticNode] == model.I_sym):
                            model.set_isolation(symptomaticNode, True)
                            numSelfIsolated_symptoms += 1

                print("\t"+str(numSelfIsolated_symptoms)+" self-isolated due to symptoms")

                #----------------------------------------
                # Update the nodeStates list after self-isolation updates to model.X:
                #----------------------------------------
                nodeStates = model.X.flatten()

                #----------------------------------------
                # Test the designated percentage of the workplace population
                # on cadence testing days
                #----------------------------------------

                randomSelection = []

                if(cadenceDayNumber in testingDays):

                    testingPool = numpy.argwhere((testing_compliance==True)
                                                 & (nodePositiveStatuses==False)
                                                 & (nodeStates != model.R)
                                                 & (nodeStates != model.Q_R)
                                                 & (nodeStates != model.H)
                                                 & (nodeStates != model.F)
                                                ).flatten()

                    numRandomTests = int(model.numNodes * pct_tested_per_day)

                    if(len(testingPool) > 0):
                        randomSelection = testingPool[numpy.random.choice(len(testingPool), min(len(testingPool), numRandomTests), replace=False)]

                #----------------------------------------
                # Allow symptomatic individuals to self-seek tests
                # regardless of cadence testing days
                #----------------------------------------

                seekingSelection = []

                seekingPool = numpy.argwhere((symptomatic_seektest_compliance==True)
                                             & (nodeTestedInCurrentStateStatuses==False)
                                             & (nodePositiveStatuses==False)
                                             & ((nodeStates==model.I_sym)|(nodeStates==model.Q_sym))
                                            ).flatten()

                seekingSelection = [seekNode for seekNode in seekingPool if seekNode not in randomSelection]


                #----------------------------------------
                # Perform the tests on the selected individuals:
                #----------------------------------------
                selectedToTest = numpy.concatenate((seekingSelection, randomSelection)).astype(int)

                numTested              = 0
                numTested_random       = 0
                numTested_selfseek     = 0
                numPositive            = 0
                numPositive_random     = 0
                numPositive_selfseek   = 0
                numIsolated_positiveTeammate = 0
                numIsolated_positiveCoworker = 0

                newIsolationGroup = []

                for i, testNode in enumerate(selectedToTest):

                    model.set_tested(testNode, True)

                    numTested += 1
                    if(i < len(seekingSelection)):
                        numTested_selfseek  += 1
                    else:
                        numTested_random   += 1

                    # If the node to be tested is not infected, then the test is guaranteed negative,
                    # so don't bother going through with doing the test:
                    if(model.X[testNode] == model.S or model.X[testNode] == model.Q_S):
                        pass
                    elif(model.X[testNode] == model.E or model.X[testNode] == model.Q_E
                         or model.X[testNode] == model.I_pre or model.X[testNode] == model.Q_pre
                         or model.X[testNode] == model.I_sym or model.X[testNode] == model.Q_sym
                         or model.X[testNode] == model.I_asym or model.X[testNode] == model.Q_asym):

                        if(test_falseneg_rate == 'temporal'):
                            testNodeState       = model.X[testNode][0]
                            testNodeTimeInState = model.timer_state[testNode][0]
                            if(testNodeState in temporal_falseneg_rates.keys()):
                                falseneg_prob = temporal_falseneg_rates[testNodeState][ int(min(testNodeTimeInState, max(temporal_falseneg_rates[testNodeState].keys()))) ]
                            else:
                                falseneg_prob = 1.00
                        else:
                            falseneg_prob = test_falseneg_rate

                        if(numpy.random.rand() < (1-falseneg_prob)):
                            # +++++++++++++++++++++++++++++++++++++++++++++
                            # The tested node has returned a positive test
                            # +++++++++++++++++++++++++++++++++++++++++++++
                            numPositive += 1
                            if(i < len(seekingSelection)):
                                numPositive_selfseek  += 1
                            else:
                                numPositive_random   += 1

                            # Update the node's state to the appropriate detected case state:
                            model.set_positive(testNode, True)
                            # model.set_isolation(testNode, True)

                            # Add this positive node to the isolation group (if applicable):
                            if(positive_isolation_compliance[testNode]):
                                newIsolationGroup.append(testNode)

                            # Add the teammates of this positive node to the isolation group (if applicable):
                            if(positive_isolation_unit=='team'):
                                isolationTeams = [team for team, nodes in teams.items() if testNode in nodes]
                                for isolationTeam in isolationTeams:
                                    for isolationTeammate in teams[isolationTeam]:
                                        if(isolationTeammate != testNode):
                                            if(positive_isolation_compliance[isolationTeammate]):
                                                numIsolated_positiveTeammate += 1
                                                newIsolationGroup.append(testNode)

                            # Add the entire workplace to the isolation group (if applicable):
                            elif(positive_isolation_unit=='workplace'):
                                for node in range(model.numNodes):
                                    if(node != testNode):
                                        if(positive_isolation_compliance[node]):
                                            numIsolated_workplace += 1
                                            newIsolationGroup.append(testNode)

                # Add the nodes to be traced to the tracing queue:
                isolationQueue.append(newIsolationGroup)

                print("\t"+str(numTested_selfseek)+" tested self-seek [+ "+str(numPositive_selfseek)+" positive (%.2f %%) +]" % (numPositive_selfseek/numTested_selfseek*100 if numTested_selfseek>0 else 0))
                print("\t"+str(numTested_random)  +" tested randomly  [+ "+str(numPositive_random)+" positive (%.2f %%) +]" % (numPositive_random/numTested_random*100 if numTested_random>0 else 0))
                print("\t"+str(numTested)         +" tested TOTAL     [+ "+str(numPositive)+" positive (%.2f %%) +]" % (numPositive/numTested*100 if numTested>0 else 0))

                print("\t"+str(numPositive)+" flagged for isolation due to positive test ("+str(numIsolated_positiveTeammate)+" as teammates of positive, "+str(numIsolated_positiveCoworker)+" as coworkers of positive)")

                #----------------------------------------
                # Update the status of nodes who are to be isolated:
                #----------------------------------------

                isolationGroup = isolationQueue.pop(0)

                numIsolated = 0
                for isolationNode in isolationGroup:
                    model.set_isolation(isolationNode, True)
                    numIsolated += 1

                print("\t"+str(numIsolated)+" entered isolation")

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    interventionInterval = (interventionStartTime, model.t)

    return interventionInterval


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
