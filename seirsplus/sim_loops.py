from __future__ import division
import pickle
import numpy

import time
import random


def run_tti_sim(model, T, 
                intervention_start_pct_infected=0, average_introductions_per_day=0,
                testing_cadence='everyday', pct_tested_per_day=1.0, test_falseneg_rate='temporal', 
                testing_compliance_symptomatic=[None], max_pct_tests_for_symptomatics=1.0,
                testing_compliance_traced=[None], max_pct_tests_for_traces=1.0,
                testing_compliance_random=[None], random_testing_degree_bias=0,
                tracing_compliance=[None], num_contacts_to_trace=None, pct_contacts_to_trace=1.0, tracing_lag=1,
                isolation_compliance_symptomatic_individual=[None], isolation_compliance_symptomatic_groupmate=[None], 
                isolation_compliance_positive_individual=[None], isolation_compliance_positive_groupmate=[None],
                isolation_compliance_positive_contact=[None], isolation_compliance_positive_contactgroupmate=[None],
                isolation_lag_symptomatic=1, isolation_lag_positive=1, isolation_lag_contact=0, isolation_groups=None,
                cadence_testing_days=None, cadence_cycle_length=28, temporal_falseneg_rates=None,
                introduction_days = [], # introduce a single exposure in these days
                runTillEnd = False, # True: don't stop simulation if number of infected & isolated is zero, since more external infections may be introduced later
                budget_policy = None,
                # policy to adjust number of daily tests based on initial values and current  circumstances
                test_priority = 'random',
                # test_priority: how to to choose which nodes to test:
                # 'random' - use test budget for random fraction of eligible population, 'last_tested' - sort according to the time passed since testing (breaking ties randomly)
                # if test_priority is callable then use as a key to sort nodes (lower value is higher priority)
                history = None,
                # history is a  dictionary that, if provided, will be updated with history and summary information for logging
                # it preferably should be OrderedDict if we want to preserve ordering of logs
                stopping_policy=None,
                # stopping_policy: function that takes as input the model  and current history and decides whether to stop execution
                #                  returns True to stop, False to continue running
                verbose = True, # suppress printing if verbose is false - useful for running many simulations in parallel
                ):

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def sample(param):
        """
        if param is a single number between 0 and 1 then convert it to an arrau of NumNodes True/False randomly chosen with this probability:
        if param is a dictionary of form { group_name: prob }  then use the above separately for each group
        This allows parameters to be more compactly described (useful when running many executions in parallel)
        """
        if isinstance(param,(float,int)):
            return (numpy.random.rand(model.numNodes) < param)
        if isinstance(param,dict):
            arr = numpy.full(model.numNodes,False, dtype=bool)
            for group, p in param.items():
                mask = model.nodeGroupData[group]['mask']
                arr[mask] = (numpy.random.rand(model.numNodes) < p)[mask]
            return arr
        return param

    testing_compliance_random = sample(testing_compliance_random)
    testing_compliance_traced = sample(testing_compliance_traced)
    testing_compliance_symptomatic = sample(testing_compliance_symptomatic)
    tracing_compliance = sample(testing_compliance_symptomatic)
    isolation_compliance_symptomatic_individual = sample(isolation_compliance_symptomatic_individual)
    isolation_compliance_symptomatic_groupmate = sample(isolation_compliance_symptomatic_groupmate)
    isolation_compliance_positive_individual = sample(isolation_compliance_positive_individual)
    isolation_compliance_positive_groupmate = sample(isolation_compliance_positive_groupmate)
    isolation_compliance_positive_contact = sample(isolation_compliance_positive_contact)
    isolation_compliance_positive_contactgroupmate = sample(isolation_compliance_positive_contactgroupmate)






    # Testing cadences involve a repeating 28 day cycle starting on a Monday
    # (0:Mon, 1:Tue, 2:Wed, 3:Thu, 4:Fri, 5:Sat, 6:Sun, 7:Mon, 8:Tues, ...)
    # For each cadence, testing is done on the day numbers included in the associated list.

    if(cadence_testing_days is None):
        cadence_testing_days    = {
                                    'everyday':     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27],
                                    'workday':      [0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25],
                                    'semiweekly':   [0, 3, 7, 10, 14, 17, 21, 24],
                                    'weekly':       [0, 7, 14, 21],
                                    'biweekly':     [0, 14],
                                    'monthly':      [0],
                                    'cycle_start':  [0]
                                }

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    if(temporal_falseneg_rates is None):
        temporal_falseneg_rates = { 
                                    model.E:        {0: 1.00, 1: 1.00, 2: 1.00, 3: 1.00},
                                    model.I_pre:    {0: 0.25, 1: 0.25, 2: 0.22},
                                    model.I_sym:    {0: 0.19, 1: 0.16, 2: 0.16, 3: 0.17, 4: 0.19, 5: 0.22, 6: 0.26, 7: 0.29, 8: 0.34, 9: 0.38, 10: 0.43, 11: 0.48, 12: 0.52, 13: 0.57, 14: 0.62, 15: 0.66, 16: 0.70, 17: 0.76, 18: 0.79, 19: 0.82, 20: 0.85, 21: 0.88, 22: 0.90, 23: 0.92, 24: 0.93, 25: 0.95, 26: 0.96, 27: 0.97, 28: 0.97, 29: 0.98, 30: 0.98, 31: 0.99},
                                    model.I_asym:   {0: 0.19, 1: 0.16, 2: 0.16, 3: 0.17, 4: 0.19, 5: 0.22, 6: 0.26, 7: 0.29, 8: 0.34, 9: 0.38, 10: 0.43, 11: 0.48, 12: 0.52, 13: 0.57, 14: 0.62, 15: 0.66, 16: 0.70, 17: 0.76, 18: 0.79, 19: 0.82, 20: 0.85, 21: 0.88, 22: 0.90, 23: 0.92, 24: 0.93, 25: 0.95, 26: 0.96, 27: 0.97, 28: 0.97, 29: 0.98, 30: 0.98, 31: 0.99},
                                    model.Q_E:      {0: 1.00, 1: 1.00, 2: 1.00, 3: 1.00},
                                    model.Q_pre:    {0: 0.25, 1: 0.25, 2: 0.22},
                                    model.Q_sym:    {0: 0.19, 1: 0.16, 2: 0.16, 3: 0.17, 4: 0.19, 5: 0.22, 6: 0.26, 7: 0.29, 8: 0.34, 9: 0.38, 10: 0.43, 11: 0.48, 12: 0.52, 13: 0.57, 14: 0.62, 15: 0.66, 16: 0.70, 17: 0.76, 18: 0.79, 19: 0.82, 20: 0.85, 21: 0.88, 22: 0.90, 23: 0.92, 24: 0.93, 25: 0.95, 26: 0.96, 27: 0.97, 28: 0.97, 29: 0.98, 30: 0.98, 31: 0.99},
                                    model.Q_asym:   {0: 0.19, 1: 0.16, 2: 0.16, 3: 0.17, 4: 0.19, 5: 0.22, 6: 0.26, 7: 0.29, 8: 0.34, 9: 0.38, 10: 0.43, 11: 0.48, 12: 0.52, 13: 0.57, 14: 0.62, 15: 0.66, 16: 0.70, 17: 0.76, 18: 0.79, 19: 0.82, 20: 0.85, 21: 0.88, 22: 0.90, 23: 0.92, 24: 0.93, 25: 0.95, 26: 0.96, 27: 0.97, 28: 0.97, 29: 0.98, 30: 0.98, 31: 0.99},
                                  }

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Custom simulation loop:
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    interventionOn         = False
    interventionStartTime  = None

    timeOfLastIntervention = -1
    timeOfLastIntroduction = -1

    testingDays            = cadence_testing_days[testing_cadence]
    cadenceDayNumber       = 0

    tests_per_day                 = int(model.numNodes * pct_tested_per_day)
    max_tracing_tests_per_day     = int(tests_per_day * max_pct_tests_for_traces)
    max_symptomatic_tests_per_day = int(tests_per_day * max_pct_tests_for_symptomatics)

    tracingPoolQueue              = [[] for i in range(tracing_lag)]
    isolationQueue_symptomatic    = [[] for i in range(isolation_lag_symptomatic)]
    isolationQueue_positive       = [[] for i in range(isolation_lag_positive)]
    isolationQueue_contact        = [[] for i in range(isolation_lag_contact)]

    model.tmax  = T
    model.runTillEnd = runTillEnd
    if not hasattr(model,"lastPositive"):
        model.lastPositive = -1
    running     = True


    def log(d):
        # log values in dictionary d into history dict
        #nonlocal history # uncomment for Python 3.x
        #nonlocal model   # uncomment for Python 3.x
        if history is None: return #o/w assume it's a dictionary
        if model.t in history:
            history[model.t].update(d)
        else:
            history[model.t] = dict(d)

    def vprint(s):
        # print s if verbose is true
        if verbose: print(s)


    while running:

        running = model.run_iteration()
        if running and stopping_policy:
            running = not stopping_policy(model, history)
            if not running:
                model.finalize_data_series()

        if not (history is None): # log current state of the model
            d = {}
            statistics = ["N","numS","numE","numI_pre","numI_sym","numI_asym","numH","numR","numF","numQ_S","numQ_E","numQ_pre","numQ_sym","numQ_asym","numQ_R"]
            for att in statistics:
                    d[att] = getattr(model,att)[model.tidx]
                    if (model.nodeGroupData):
                        for groupName  in model.nodeGroupData:
                            groupData = model.nodeGroupData[groupName]
                            d[groupName+"/"+att] = groupData[att][model.tidx]
            d["numInfectious"] = sum( getattr(model,att)[model.tidx] for att in ["numI_pre","numI_sym","numI_asym"]) # number of infectionus non quaranteened people
            d["overallInfected"] = model.numNodes - model.numS[model.tidx] # total number of people infect (initial - susceptible)
            d["numNodes"] = model.numNodes
            if (model.nodeGroupData):
                for groupName in model.nodeGroupData:
                    groupData = model.nodeGroupData[groupName]
                    d[groupName+"/numInfectious"] = sum(groupData[att][model.tidx] for att in ["numI_pre","numI_sym","numI_asym"])
                    d[groupName + "/overallInfected"] = len(groupData['nodes']) - groupData['numS'][model.tidx]
            log(d)






        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Introduce exogenous exposures randomly:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if(int(model.t)!=int(timeOfLastIntroduction)):

            timeOfLastIntroduction = model.t
            baseline_exposure = 0
            if introduction_days and (int(model.t) in introduction_days):
                vprint("introduced new exposure at time "+ str(model.t))
                baseline_exposure = 1 # introduce a single exposure in that day
            if isinstance(average_introductions_per_day,dict):
                numNewExposures = {}
                for group,num in average_introductions_per_day.items():
                    numNewExposures[group] = baseline_exposure+numpy.random.poisson(lam=num)
            else:
                numNewExposures = baseline_exposure+numpy.random.poisson(lam=average_introductions_per_day)
            
            model.introduce_exposures(num_new_exposures=numNewExposures)
            log({"numNewExposures": numNewExposures})

            if(numNewExposures > 0):
                vprint("[NEW EXPOSURE @ t = %.2f (%d exposed)]" % (model.t, numNewExposures))

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Execute testing policy at designated intervals:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        if(int(model.t)!=int(timeOfLastIntervention)):
        
            cadenceDayNumber = int(model.t % cadence_cycle_length)

            timeOfLastIntervention = model.t

            currentNumInfected = model.total_num_infected()[model.tidx]
            currentPctInfected = model.total_num_infected()[model.tidx]/model.numNodes
            log({"currentNumInfected": currentNumInfected , "cadenceDayNumber": cadenceDayNumber })

            if(currentPctInfected >= intervention_start_pct_infected) and (not interventionOn):
                interventionOn        = True
                interventionStartTime = model.t

            log({"interventionOn": interventionOn})
            if(interventionOn):

                vprint("[INTERVENTIONS @ t = %.2f (%d (%.2f%%) infected)]" % (model.t, currentNumInfected, currentPctInfected*100))
                
                nodeStates                       = model.X.flatten()
                nodeTestedStatuses               = model.tested.flatten()
                nodeTestedInCurrentStateStatuses = model.testedInCurrentState.flatten()
                nodePositiveStatuses             = model.positive.flatten()

                log({"PositveStatuses" : sum(nodePositiveStatuses),
                     "TestedStatuses" : sum(nodeTestedStatuses),
                     "TestedInCurrentStateStatuses" : sum(nodeTestedInCurrentStateStatuses)
                     })

                # number of people that can be tested
                poolSize =  numpy.sum((nodePositiveStatuses==False)
                                    & (nodeStates != model.R)
                                    & (nodeStates != model.Q_R)
                                    & (nodeStates != model.H)
                                    & (nodeStates != model.F))
                log({"poolSize": poolSize})
                if budget_policy:
                    tests_per_day, max_tracing_tests_per_day, max_symptomatic_tests_per_day = budget_policy(
                        model,
                        history,
                        poolSize=poolSize,
                        pct_tested_per_day = pct_tested_per_day,
                        max_pct_tests_for_traces = max_pct_tests_for_traces,
                        max_pct_tests_for_symptomatics = max_pct_tests_for_symptomatics)
                log({"tests_per_day": tests_per_day,
                     "max_tracing_tests_per_day" : max_tracing_tests_per_day,
                     "max_symptomatic_tests_per_day" : max_symptomatic_tests_per_day
                })

                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                # tracingPoolQueue[0] = tracingPoolQueue[0]Queue.pop(0)

                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                
                newIsolationGroup_symptomatic = []
                newIsolationGroup_contact     = []

                #----------------------------------------
                # Isolate SYMPTOMATIC cases without a test:
                #----------------------------------------
                numSelfIsolated_symptoms = 0
                numSelfIsolated_symptomaticGroupmate = 0

                if(any(isolation_compliance_symptomatic_individual)):
                    symptomaticNodes = numpy.argwhere((nodeStates==model.I_sym)).flatten()
                    for symptomaticNode in symptomaticNodes:
                        if(isolation_compliance_symptomatic_individual[symptomaticNode]):
                            if(model.X[symptomaticNode] == model.I_sym):
                                numSelfIsolated_symptoms += 1   
                                newIsolationGroup_symptomatic.append(symptomaticNode)

                            #----------------------------------------
                            # Isolate the GROUPMATES of this SYMPTOMATIC node without a test:
                            #----------------------------------------
                            if(isolation_groups is not None) and any(isolation_compliance_symptomatic_groupmate):
                                for group in isolation_groups: # allow non disjoint groups
                                    if not symptomaticNode in group:
                                        continue
                                    for isolationGroupmate in group:
                                        if(isolationGroupmate != symptomaticNode):
                                            if (isolation_compliance_symptomatic_groupmate[isolationGroupmate]):
                                                numSelfIsolated_symptomaticGroupmate += 1
                                                newIsolationGroup_symptomatic.append(isolationGroupmate)

                #----------------------------------------
                # Isolate the CONTACTS of detected POSITIVE cases without a test:
                #----------------------------------------
                numSelfIsolated_positiveContact = 0
                numSelfIsolated_positiveContactGroupmate = 0

                if(any(isolation_compliance_positive_contact) or any(isolation_compliance_positive_contactgroupmate)):
                    for contactNode in tracingPoolQueue[0]:
                        if(isolation_compliance_positive_contact[contactNode]):
                            newIsolationGroup_contact.append(contactNode)
                            numSelfIsolated_positiveContact += 1 

                        #----------------------------------------
                        # Isolate the GROUPMATES of this self-isolating CONTACT without a test:
                        #----------------------------------------
                        if (isolation_groups is not None) and any(isolation_compliance_positive_contactgroupmate):
                            for group in isolation_groups:  # allow non disjoint groups
                                if not contactNode in group:
                                    continue
                                for isolationGroupmate in group:
                                    if (isolationGroupmate != contactNode): # do not include node itself
                                        if (isolation_compliance_positive_contactgroupmate[isolationGroupmate]):
                                            newIsolationGroup_contact.append(isolationGroupmate)
                                            numSelfIsolated_positiveContactGroupmate += 1


                #----------------------------------------
                # Update the nodeStates list after self-isolation updates to model.X:
                #----------------------------------------
                nodeStates = model.X.flatten()


                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


                #----------------------------------------
                # Allow SYMPTOMATIC individuals to self-seek tests
                # regardless of cadence testing days
                #----------------------------------------
                symptomaticSelection = []

                if(any(testing_compliance_symptomatic)):
                    
                    symptomaticPool = numpy.argwhere((testing_compliance_symptomatic==True)
                                                     & (nodeTestedInCurrentStateStatuses==False)
                                                     & (nodePositiveStatuses==False)
                                                     & ((nodeStates==model.I_sym)|(nodeStates==model.Q_sym))
                                                    ).flatten()

                    numSymptomaticTests  = min(len(symptomaticPool), max_symptomatic_tests_per_day)

                    log({"symptomaticPool": len(symptomaticPool), "numSymptomaticTests": numSymptomaticTests })
                    
                    if(len(symptomaticPool) > 0):
                        symptomaticSelection = symptomaticPool[numpy.random.choice(len(symptomaticPool), min(numSymptomaticTests, len(symptomaticPool)), replace=False)]


                #----------------------------------------
                # Test individuals randomly and via contact tracing
                # on cadence testing days:
                #----------------------------------------

                tracingSelection = []
                randomSelection = []

                if(cadenceDayNumber in testingDays):

                    #----------------------------------------
                    # Apply a designated portion of this day's tests 
                    # to individuals identified by CONTACT TRACING:
                    #----------------------------------------

                    tracingPool = tracingPoolQueue.pop(0)
                    log({"currentTracingPool" : len(tracingPool)})

                    if(any(testing_compliance_traced)):

                        numTracingTests = min(len(tracingPool), min(tests_per_day-len(symptomaticSelection), max_tracing_tests_per_day))
                        log({"numTracingTests" : numTracingTests})

                        for trace in range(numTracingTests):
                            traceNode = tracingPool.pop()
                            if((nodePositiveStatuses[traceNode]==False)
                                and (testing_compliance_traced[traceNode]==True)
                                and (model.X[traceNode] != model.R)
                                and (model.X[traceNode] != model.Q_R) 
                                and (model.X[traceNode] != model.H)
                                and (model.X[traceNode] != model.F)):
                                tracingSelection.append(traceNode)

                    #----------------------------------------
                    # Apply the remainder of this day's tests to random testing:
                    #----------------------------------------

                    if(any(testing_compliance_random)):
                        
                        testingPool = numpy.argwhere((testing_compliance_random==True)
                                                     & (nodePositiveStatuses==False)
                                                     & (nodeStates != model.R)
                                                     & (nodeStates != model.Q_R) 
                                                     & (nodeStates != model.H)
                                                     & (nodeStates != model.F)
                                                    ).flatten()
                        log({"testingPool" : len(testingPool)})
                        numRandomTests = max(min(tests_per_day-len(tracingSelection)-len(symptomaticSelection), len(testingPool)), 0)
                        log({"numRandomTests": numRandomTests})
                        testingPool_degrees       = model.degree.flatten()[testingPool]
                        testingPool_degreeWeights = numpy.power(testingPool_degrees,random_testing_degree_bias)/numpy.sum(numpy.power(testingPool_degrees,random_testing_degree_bias))

                        poolSize = len(testingPool)
                        if(poolSize > 0):
                            if callable(test_priority):
                                randomSelection = sorted(testingPool, key=test_priority)[:numRandomTests]
                            elif test_priority == 'last_tested':
                                # sort the pool according to the time they were last tested, breaking ties randomly
                                randomSelection = sorted(testingPool,key = lambda i: (model.testedTime[i], random.randint(0,poolSize*poolSize)))[:numRandomTests]
                            else:
                                randomSelection = testingPool[numpy.random.choice(len(testingPool), numRandomTests, p=testingPool_degreeWeights, replace=False)]

                
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


                #----------------------------------------
                # Perform the tests on the selected individuals:
                #----------------------------------------

                selectedToTest = numpy.concatenate((symptomaticSelection, tracingSelection, randomSelection)).astype(int)

                numTested                     = 0
                numTested_random              = 0
                numTested_tracing             = 0
                numTested_symptomatic         = 0
                numPositive                   = 0
                numPositive_random            = 0
                numPositive_tracing           = 0
                numPositive_symptomatic       = 0 
                numIsolated_positiveGroupmate = 0
                
                newTracingPool = []

                newIsolationGroup_positive = []

                for i, testNode in enumerate(selectedToTest):

                    model.set_tested(testNode, True)

                    numTested += 1
                    if(i < len(symptomaticSelection)):
                        numTested_symptomatic  += 1
                    elif(i < len(symptomaticSelection)+len(tracingSelection)):
                        numTested_tracing += 1
                    else:
                        numTested_random += 1                  

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
                            if(testNodeState in list(temporal_falseneg_rates.keys())):
                                falseneg_prob = temporal_falseneg_rates[testNodeState][ int(min(testNodeTimeInState, max(list(temporal_falseneg_rates[testNodeState].keys())))) ]
                            else:
                                falseneg_prob = 1.00
                        else:
                            falseneg_prob = test_falseneg_rate

                        if(numpy.random.rand() < (1-falseneg_prob)):
                            # +++++++++++++++++++++++++++++++++++++++++++++
                            # The tested node has returned a positive test
                            # +++++++++++++++++++++++++++++++++++++++++++++
                            numPositive += 1
                            if(i < len(symptomaticSelection)):
                                numPositive_symptomatic  += 1
                            elif(i < len(symptomaticSelection)+len(tracingSelection)):
                                numPositive_tracing += 1
                            else:
                                numPositive_random += 1 
                            
                            # Update the node's state to the appropriate detected case state:
                            model.set_positive(testNode, True)

                            #----------------------------------------
                            # Add this positive node to the isolation group:
                            #----------------------------------------
                            if(isolation_compliance_positive_individual[testNode]):
                                newIsolationGroup_positive.append(testNode)

                            #----------------------------------------
                            # Add the groupmates of this positive node to the isolation group:
                            #----------------------------------------
                            if (isolation_groups is not None) and any(isolation_compliance_symptomatic_groupmate):
                                for group in isolation_groups:  # allow non disjoint groups
                                    if not testNode in group:
                                        continue
                                    for isolationGroupmate in group:
                                        if (isolationGroupmate != testNode):
                                            if (isolation_compliance_symptomatic_groupmate[isolationGroupmate]):
                                                numSelfIsolated_symptomaticGroupmate += 1
                                                newIsolationGroup_symptomatic.append(isolationGroupmate)


                            #----------------------------------------  
                            # Add this node's neighbors to the contact tracing pool:
                            #----------------------------------------  
                            if(any(tracing_compliance) or any(isolation_compliance_positive_contact) or any(isolation_compliance_positive_contactgroupmate)):
                                if(tracing_compliance[testNode]):
                                    testNodeContacts = list(model.G[testNode].keys())
                                    numpy.random.shuffle(testNodeContacts)
                                    if(num_contacts_to_trace is None):
                                        numContactsToTrace = int(pct_contacts_to_trace*len(testNodeContacts))
                                    else:
                                        numContactsToTrace = num_contacts_to_trace
                                    newTracingPool.extend(testNodeContacts[0:numContactsToTrace])

        
                # Add the nodes to be isolated to the isolation queue:
                isolationQueue_positive.append(newIsolationGroup_positive)
                isolationQueue_symptomatic.append(newIsolationGroup_symptomatic)
                isolationQueue_contact.append(newIsolationGroup_contact)

                # Add the nodes to be traced to the tracing queue:
                log({"newTracingPool" : len(newTracingPool)})
                tracingPoolQueue.append(newTracingPool)

                if numPositive:
                    model.lastPositive = model.t

                vprint("\t"+str(numTested_symptomatic) +"\ttested due to symptoms  [+ "+str(numPositive_symptomatic)+" positive (%.2f %%) +]" % (numPositive_symptomatic/numTested_symptomatic*100 if numTested_symptomatic>0 else 0))
                vprint("\t"+str(numTested_tracing)     +"\ttested as traces        [+ "+str(numPositive_tracing)+" positive (%.2f %%) +]" % (numPositive_tracing/numTested_tracing*100 if numTested_tracing>0 else 0))
                vprint("\t"+str(numTested_random)      +"\ttested randomly         [+ "+str(numPositive_random)+" positive (%.2f %%) +]" % (numPositive_random/numTested_random*100 if numTested_random>0 else 0))
                vprint("\t"+str(numTested)             +"\ttested TOTAL            [+ "+str(numPositive)+" positive (%.2f %%) +]" % (numPositive/numTested*100 if numTested>0 else 0))

                vprint("\t"+str(numSelfIsolated_symptoms)        +" will isolate due to symptoms         ("+str(numSelfIsolated_symptomaticGroupmate)+" as groupmates of symptomatic)")
                vprint("\t"+str(numPositive)                     +" will isolate due to positive test    ("+str(numIsolated_positiveGroupmate)+" as groupmates of positive)")
                vprint("\t"+str(numSelfIsolated_positiveContact) +" will isolate due to positive contact ("+str(numSelfIsolated_positiveContactGroupmate)+" as groupmates of contact)")

                #----------------------------------------
                # Update the status of nodes who are to be isolated:
                #----------------------------------------

                numIsolated = 0

                isolationGroup_symptomatic = isolationQueue_symptomatic.pop(0)
                for isolationNode in isolationGroup_symptomatic:
                    model.set_isolation(isolationNode, True)
                    numIsolated += 1

                isolationGroup_contact = isolationQueue_contact.pop(0)
                for isolationNode in isolationGroup_contact:
                    model.set_isolation(isolationNode, True)
                    numIsolated += 1

                isolationGroup_positive = isolationQueue_positive.pop(0)
                enterIsolationPositive = len(isolationGroup_positive)
                for isolationNode in isolationGroup_positive:
                    model.set_isolation(isolationNode, True)
                    numIsolated += 1

                vprint("\t"+str(numIsolated)+" entered isolation")
                log({"numTested_symptomatic": numTested_symptomatic,
                     "numPositive_symptomatic" : numPositive_symptomatic,
                     "numTested_tracing" : numTested_tracing,
                     "numPositive_tracing" : numPositive_tracing,
                     "numTested" : numTested,
                     "numTested_random" : numTested_random,
                     "numSelfIsolated_symptoms":  numSelfIsolated_symptoms,
                    "numSelfIsolated_symptomaticGroupmate": numSelfIsolated_symptomaticGroupmate,
                    "numPositive" : numPositive,
                     "numIsolated_positiveGroupmate" : numIsolated_positiveGroupmate,
                     "numSelfIsolated_positiveContact" : numSelfIsolated_positiveContact,
                     "numIsolated" : numIsolated,
                     "numEnterIsolationPositiveNow": enterIsolationPositive
                    })

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    interventionInterval = (interventionStartTime, model.t)

    return interventionInterval



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Policy generation functions
def scale_to_pool(model, hist, poolSize, pct_tested_per_day,  max_pct_tests_for_symptomatics, max_pct_tests_for_traces):
    N = poolSize
    tests_per_day = int(N * pct_tested_per_day)
    max_tracing_tests_per_day = int(tests_per_day * max_pct_tests_for_traces)
    max_symptomatic_tests_per_day = int(tests_per_day * max_pct_tests_for_symptomatics)
    return tests_per_day, max_tracing_tests_per_day, max_symptomatic_tests_per_day

def hammer_and_dance(lag=1,hammer_wait = 7, test_schedule = [0], frac_of_pool=True):
    """Returns a budget policy function that will test everyone if there is a positive test result.
    Otherwise go by the other budget parameters - if frac_of_pool=True then testing budget depends on eligible pool and
    not on original number of nodes"""
    def test_policy(model, hist, poolSize, pct_tested_per_day,  max_pct_tests_for_symptomatics, max_pct_tests_for_traces):
        if not hasattr(model,"lastHammer"):
            model.lastHammer = -hammer_wait
        if (model.lastPositive>=0):
            detectionTime = model.lastPositive + lag
            if model.lastHammer < detectionTime - hammer_wait:
                model.lastHammer = detectionTime
                model.lastSchedule = [detectionTime + offset for offset in test_schedule]
            for i,t in enumerate(model.lastSchedule):
                if int(model.t) == int(t):
                    model.lastSchedule[i] = -1
                    # test everyone
                    return model.numNodes,model.numNodes,model.numNodes

        N = poolSize if frac_of_pool else model.numNodes
        tests_per_day = int(N * pct_tested_per_day)
        max_tracing_tests_per_day = int(tests_per_day * max_pct_tests_for_traces)
        max_symptomatic_tests_per_day = int(tests_per_day * max_pct_tests_for_symptomatics)
        return tests_per_day, max_tracing_tests_per_day, max_symptomatic_tests_per_day
    return test_policy

def stop_at_detection(lag=1):
    """Returns stopping policy function that stops after the first positive result"""
    def policy(model, hist):
        # stop if there was a positive result after lag time
        return (model.lastPositive>=0) and (model.lastPositive+lag <= model.t)
    return policy


def single_introduction(end):
    """Single return a single introduction day for the introduction_days parameter"""
    return [random.randint(0,end)]

def test_frequency(frequency):
    MAX_TIME = 365
    testing_cadence = f"every {frequency}"
    offset = random.randint(0,frequency-1)
    cadence_testing_days = { testing_cadence : [offset+i for i in range(MAX_TIME) if (i % frequency ==0)]  }
    cadence_cycle_length = MAX_TIME
    return testing_cadence, cadence_testing_days, cadence_cycle_length