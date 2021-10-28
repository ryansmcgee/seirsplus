from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import networkx as networkx
import numpy as np
import scipy as scipy
import scipy.integrate
import copy
import json

from utilities import *


########################################################
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#
#@                                                    @#
#@  CUSTOM COMPARTMENT MODELS WITH CONTACT NETWORKS   @#
#@                                                    @#
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#
########################################################


class CompartmentNetworkModel():

    def __init__(self, compartments, networks,
                    mixedness=0.0, openness=0.0,
                    isolation_period=None,
                    transition_mode='exponential_rates', 
                    local_trans_denom_mode='all_contacts',
                    log_infection_info=False,
                    store_Xseries=False,
                    node_groups=None,
                    seed=None):

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update model execution options:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if(seed is not None):
            np.random.seed(seed)
            self.seed = seed

        self.transition_mode        = transition_mode
        self.transition_timer_wt    = 1e5
        self.local_trans_denom_mode = local_trans_denom_mode
        self.log_infection_info     = log_infection_info

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update the contact networks specifications:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.pop_size = None # will be updated in update_networks()
        self.networks = {}
        self.update_networks(networks)

        self.mixedness = mixedness
        self.openness  = openness

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update the compartment model configuration and parameterizations:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.compartments    = {}
        self.infectivity_mat = {} 
        self.update_compartments(compartments)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize timekeeping:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.t       = 0 # current sim time
        self.tmax    = 0 # max sim time (will be set when run() is called)
        self.tidx    = 0 # current index in list of timesteps
        self.tseries = np.zeros(self.pop_size*min(len(self.compartments), 10))

        # Vectors holding the time that each node has their current state:
        self.state_timer        = np.zeros((self.pop_size,1))

        # Vectors holding the isolation status and isolation time for each node:
        self.isolation          = np.zeros(self.pop_size)
        self.isolation_period   = isolation_period
        self.isolation_timer    = np.zeros(self.pop_size)
        self.totalIsolationTime = np.zeros(self.pop_size)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize compartment IDs/metadata:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.stateID               = {}
        self.default_state         = list(self.compartments.keys())[0] # default to first compartment specified
        self.excludeFromEffPopSize = []
        self.node_flags            = [[]]*self.pop_size
        self.allNodeFlags          = set() 
        self.allCompartmentFlags   = set()
        for c, compartment in enumerate(self.compartments):
            comp_params = self.compartments[compartment]
            #----------------------------------------
            # Assign state ID number to each compartment (for internal state comparisons):
            self.stateID[compartment] = c
            #----------------------------------------
            # Update the default compartment for this model:
            if('default_state' in comp_params and comp_params['default_state']==True):
                self.default_state = compartment
            #----------------------------------------
            # Update which compartments are excluded when calculating effective population size (N):
            if('exclude_from_eff_pop' in comp_params and comp_params['exclude_from_eff_pop']==True):
                self.excludeFromEffPopSize.append(compartment)
            #----------------------------------------
            # Update which compartment flags are in use:
            if('flags' not in self.compartments[compartment]):
                self.compartments[compartment]['flags'] = []
            for flag in self.compartments[compartment]['flags']:
                self.allCompartmentFlags.add(flag)
            
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize other metadata:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.infectionLogs = []

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize data series for tracking node subgroups:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.nodeGroupData = None
        if(node_groups):
            self.nodeGroupData = {}
            for groupName, nodeList in node_groups.items():
                self.nodeGroupData[groupName] = {'nodes':   np.array(nodeList),
                                                 'mask':    np.in1d(range(self.pop_size), nodeList).reshape((self.pop_size,1))}
                for compartment in self.compartments:
                    self.nodeGroupData[groupName][compartment]    = np.zeros(self.pop_size*min(len(self.compartments), 10))
                    self.nodeGroupData[groupName][compartment][0] = np.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.S)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize counts/prevalences and the states of individuals:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.counts            = {}
        self.flag_counts       = {} 
        self.track_flag_counts = True
        self.store_Xseries     = store_Xseries
        self.process_initial_states()

        #----------------------------------------
        # Initialize exogenous prevalence for each compartment:
        self.exogenous_prevalence  = {}
        for c, compartment in enumerate(self.compartments):
            comp_params = self.compartments[compartment]
            self.exogenous_prevalence[compartment] = comp_params['exogenous_prevalence']
            
    
    ########################################################
    ########################################################


    def update_networks(self, new_networks):
        if(not isinstance(new_networks, dict)):
            raise BaseException("Specify networks with a dictionary of adjacency matrices or networkx objects.")
        else:
            # Store both a networkx object and a np adjacency matrix representation of each network:
            for netID, network in new_networks.items():
                if type(network)==np.ndarray:
                    new_networks[netID] = {"networkx":  networkx.from_numpy_matrix(network), 
                                         "adj_matrix":  scipy.sparse.csr_matrix(network)}
                elif type(network)==networkx.classes.graph.Graph:
                    new_networks[netID] = {"networkx":  network, 
                                         "adj_matrix":  networkx.adj_matrix(network)}
                else:
                    raise BaseException("Network", netID, "should be specified by an adjacency matrix or networkx object.")
                # Store the number of nodes and node degrees for each network:
                new_networks[netID]["num_nodes"] = int(new_networks[netID]["adj_matrix"].shape[1])
                new_networks[netID]["degree"]    = new_networks[netID]["adj_matrix"].sum(axis=0).reshape(new_networks[netID]["num_nodes"],1)
                # Set all individuals to be active participants in this network by default:
                new_networks[netID]['active']            = np.ones(new_networks[netID]["num_nodes"])
                # Set all individuals to be inactive in this network when in isolation by default:
                new_networks[netID]["active_isolation"]  = np.zeros(new_networks[netID]["num_nodes"])
            
            self.networks.update(new_networks)

            # Ensure all networks have the same number of nodes:
            for key, network in self.networks.items():
                if(self.pop_size is None):
                    self.pop_size = network["num_nodes"]
                if(network["num_nodes"] !=  self.pop_size):
                    raise BaseException("All networks must have the same number of nodes.")


    ########################################################


    def update_compartments(self, new_compartments):
        if(isinstance(new_compartments, str) and '.json' in new_compartments):
            with open(new_compartments) as compartments_file:
                new_compartments = json.load(compartments_file)
        elif(isinstance(new_compartments, dict)):
            pass
        elif(isinstance(new_compartments, CompartmentModelBuilider)):
            new_compartments = new_compartments.compartments
        else:
            raise BaseException("Specify compartments with a dictionary or JSON file.")

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Recursively pre-process and reshape parameter values for all compartments:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        def reshape_param_vals(nested_dict):
            for key, value in nested_dict.items():
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Do not recurse or reshape these params
                if(key in ['transmissibilities', 'initial_prevalence', 'exogenous_prevalence', 'default_state', 'exclude_from_eff_pop', 'flags']):
                    pass
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Recurse through sub dictionaries
                elif(isinstance(value, dict)):
                    reshape_param_vals(value)
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Convert all other parameter values to arrays corresponding to the population size:
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                else:
                    nested_dict[key] = np.array(value).reshape((self.pop_size, 1)) if isinstance(value, (list, np.ndarray)) else np.full(fill_value=value, shape=(self.pop_size,1))

        reshape_param_vals(new_compartments)

        #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Recursively process transition probabilities:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        def process_transition_params(nested_dict):
            for key, value in nested_dict.items():
                if(key == 'transitions' and len(value) > 0):
                    transn_dict = value
                    poststates  = list(transn_dict.keys())
                    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Ensure all transitions have a specified rate or time, as applicable:
                    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    self.process_transition_times(transn_dict)
                    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Decide the transition each individual will take according to given probabilities:
                    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    self.process_transition_probs(transn_dict)
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Recurse through sub dictionaries
                elif(isinstance(value, dict) and key != 'transitions'):
                    process_transition_params(value)
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Else do nothing
                else:
                    pass
        #----------------------------------------
        process_transition_params(new_compartments)

        #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Transmissibility parameters are preprocessed and shaped into pairwise matrices
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        for compartment, comp_dict in new_compartments.items():
            transm_dict = comp_dict['transmissibilities']
            if(len(transm_dict) == 0):
                pass
            #----------------------------------------        
            if('pairwise_mode' not in transm_dict):
                transm_dict['pairwise_mode'] = 'infected' # default when not provided
            #----------------------------------------
            if('local_transm_offset_mode' not in transm_dict):
                transm_dict['local_transm_offset_mode'] = 'none' # default when not provided
            #----------------------------------------    
            self.infectivity_mat[compartment] = {}
            for G in self.networks:
                #----------------------------------------
                # Process local transmissibility parameters for each network:
                #----------------------------------------
                self.process_local_transmissibility(transm_dict, G)
                #----------------------------------------
                # Process frequency-dependent transmission offset factors for each network:
                #----------------------------------------
                self.process_local_transm_offsets(transm_dict, G)
                #----------------------------------------
                # Pre-calculate Infectivity Matrices for each network,
                # which pre-combine transmissibility, adjacency, and freq-dep offset terms.
                #----------------------------------------
                # M_G = (AB)_G * D_G
                self.infectivity_mat[compartment][G] = scipy.sparse.csr_matrix.multiply(transm_dict[G], transm_dict['offsets'][G])
            #----------------------------------------
            if('exogenous' not in transm_dict or not isinstance(transm_dict['exogenous'], (int, float))):
                transm_dict['exogenous'] = 0.0
            #----------------------------------------
            if('global' not in transm_dict or not isinstance(transm_dict['global'], (int, float))):
                transm_dict['global'] = np.sum([np.sum(transm_dict[G][transm_dict[G]!=0]) for G in  self.networks]) / max(np.sum([transm_dict[G].count_nonzero() for G in  self.networks]), 1)

        #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check the initial and exogenous_prevalence params for each compartment, defaulting to 0 when missing or invalid:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        for compartment, comp_dict in new_compartments.items():
            if('initial_prevalence' not in comp_dict or not isinstance(comp_dict['initial_prevalence'], (int, float))):
                comp_dict['initial_prevalence'] = 0.0
            if('exogenous_prevalence' not in comp_dict or not isinstance(comp_dict['exogenous_prevalence'], (int, float))):
                comp_dict['exogenous_prevalence'] = 0.0

        #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update the model object with the new processed compartments
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.compartments.update(new_compartments)


    ########################################################
    ########################################################


    def calc_propensities(self):

        propensities     = []
        transitions      = []

        for compartment, comp_params in self.compartments.items():

            # Skip calculations for this compartment if no nodes are in this state:
            if(not np.any(self.X==self.stateID[compartment])):
                continue

            #----------------------------------------
            # Dict to store calcualted propensities of local infection for each infectious state
            # so that these local propensity terms do not have to be calculated more than once 
            # if needed for multiple susceptible states
            propensity_infection_local = {}

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Calc propensities of temporal transitions:
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            for destState, transition_params in comp_params['transitions'].items():
                if(destState not in self.compartments):
                    print("Destination state", destState, "is not a defined compartment.")
                    continue

                if(self.transition_mode == 'time_in_state'):
                    propensity_temporal_transition = (self.transition_timer_wt * (np.greater(self.state_timer, transition_params['time']) & (self.X==self.stateID[compartment])) * transition_params['active_path']) if any(transition_params['time']) else np.zeros_like(self.X)

                else: # exponential_rates
                    propensity_temporal_transition = transition_params['rate'] * (self.X==self.stateID[compartment]) * transition_params['active_path']

                propensities.append(propensity_temporal_transition)
                transitions.append({'from':compartment, 'to':destState, 'type':'temporal'})

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Calc propensities of transmission-induced transitions:
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            for infectiousState, susc_params in comp_params['susceptibilities'].items():
                if(infectiousState not in self.compartments):
                    print("Infectious state", infectiousState, "is not a defined compartment.")
                    continue

                # Skip calculations for this infectious compartment if no nodes are in this state:
                if(not np.any(self.X==self.stateID[infectiousState])):
                    continue

                #----------------------------------------
                # Compute the local transmission propensity terms for individuals in each contact network
                #----------------------------------------
                if(infectiousState not in propensity_infection_local):

                    propensity_infection_local[infectiousState] = np.zeros((self.pop_size, 1))

                    denom_numContacts = np.zeros((self.pop_size, 1))

                    #----------------------------------------
                    # Get the number of contacts relevant for the local transmission denominator for each individual:
                    for netID, G in self.networks.items():
                        bool_isGactive    = (((G['active']!=0)&(self.isolation==0)) | ((G['active_isolation']!=0)&(self.isolation!=0))).flatten()
                        denom_numContacts += G['adj_matrix'][:,np.argwhere(bool_isGactive).flatten()].sum(axis=1) if self.local_trans_denom_mode=='active_contacts' else G['degree']

                    #----------------------------------------
                    # Compute the local transmission propensity terms:
                    #----------------------------------------
                    for netID, G in self.networks.items():

                        M = self.infectivity_mat[infectiousState][netID]

                        #----------------------------------------
                        # Determine which individuals need local transmission propensity calculated (active in G and infectible, non-zero propensity)
                        # and which individuals are relevant in these calculations (active in G and infectious):
                        #----------------------------------------
                        bool_isGactive    = (((G['active']!=0)&(self.isolation==0)) | ((G['active_isolation']!=0)&(self.isolation!=0))).flatten()
                        bin_isGactive     = [1 if i else 0 for i in bool_isGactive]

                        bool_isInfectious = (self.X==self.stateID[infectiousState]).flatten()
                        j_isInfectious    = np.argwhere(bool_isInfectious).flatten()

                        bool_hasGactiveInfectiousContacts = np.asarray(scipy.sparse.csr_matrix.dot(M, scipy.sparse.diags(bin_isGactive))[:,j_isInfectious].sum(axis=1).astype(bool)).flatten()

                        bool_isInfectible = (bool_isGactive & bool_hasGactiveInfectiousContacts)
                        i_isInfectible    = np.argwhere(bool_isInfectible).flatten()

                        #----------------------------------------
                        # Compute the local transmission propensity terms for individuals in the current contact network G
                        #----------------------------------------
                        propensity_infection_local[infectiousState][i_isInfectible] += np.divide( scipy.sparse.csr_matrix.dot(M[i_isInfectible,:][:,j_isInfectious], (self.X==self.stateID[infectiousState])[j_isInfectious]), denom_numContacts[i_isInfectible], out=np.zeros_like(propensity_infection_local[infectiousState][i_isInfectible]), where=denom_numContacts[i_isInfectible]!=0 )

                #----------------------------------------
                # Compute the propensities of infection for individuals across all transmission modes (exogenous, global, local over all networks)
                #----------------------------------------
                transm_params = self.compartments[infectiousState]['transmissibilities']

                propensity_infection = ((self.X==self.stateID[compartment]) * 
                                        (
                                            susc_params['susceptibility'] *
                                            (    
                                                 (self.openness) * (transm_params['exogenous']*self.exogenous_prevalence[compartment])
                                             + (1-self.openness) * (
                                                                        (self.mixedness) * ((transm_params['global']*self.counts[infectiousState][self.tidx])/self.N[self.tidx])
                                                                    + (1-self.mixedness) * (propensity_infection_local[infectiousState])
                                                                   )
                                            )
                                        ))

                #----------------------------------------
                # Compute the propensities of each possible infection-induced transition according to the disease progression paths of each individual:
                #----------------------------------------
                for destState, transition_params in susc_params['transitions'].items():
                    if(destState not in self.compartments):
                        print("Destination state", destState, "is not a defined compartment.")
                        continue

                    propensity_infection_transition = propensity_infection * transition_params['active_path']

                    propensities.append(propensity_infection_transition)
                    transitions.append({'from':compartment, 'to':destState, 'type':'infection'})

            #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        
        #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        propensities = np.hstack(propensities) if len(propensities)>0 else np.array([[]])

        return propensities, transitions
        

    ########################################################
    ########################################################


    def run_iteration(self, max_dt=None, default_dt=0.1):

        max_dt = self.tmax if max_dt is None else max_dt

        if(self.tidx >= len(self.tseries)-1):
            # Room has run out in the timeseries storage arrays; double the size of these arrays:
            self.increase_data_series_length()

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Generate 2 random numbers uniformly distributed in (0,1)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        r1 = np.random.rand()
        r2 = np.random.rand()

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Calculate propensities
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        propensities, transitions = self.calc_propensities()

        if(propensities.sum() > 0):

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Calculate alpha
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            propensities_flat   = propensities.ravel(order='F')
            cumsum              = propensities_flat.cumsum()
            alpha               = propensities_flat.sum()

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute the time until the next event takes place
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            tau = (1/alpha)*np.log(float(1/r1))

            #----------------------------------------
            # If the time to next event exceeds the max allowed interval,
            # advance the system time by the max allowed interval,
            # but do not execute any events (recalculate Gillespie interval/event next iteration)
            if(tau > max_dt):

                # Advance time by max_dt step
                self.t           += max_dt
                self.state_timer += max_dt
                self.tidx        += 1

                # Update isolation timers/statuses
                i_isolated = np.argwhere(self.isolation==1).flatten()
                self.isolation_timer[i_isolated]    += max_dt
                self.totalIsolationTime[i_isolated] += max_dt
                if(self.isolation_period is not None):
                    i_exitingIsolation = np.argwhere(self.isolation_timer >= self.isolation_period).flatten()
                    for i in i_exitingIsolation:
                        self.set_isolation(node=i, isolation=False)

                # return without any further event execution
                self.update_data_series()
                return True

                #----------------------------------------

            else:
                # Advance time by tau
                self.t           += tau
                self.state_timer += tau
                self.tidx        += 1

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute which event takes place
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            transitionIdx  = np.searchsorted(cumsum,r2*alpha)
            transitionNode = transitionIdx % self.pop_size
            transition     = transitions[ int(transitionIdx/self.pop_size) ]

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Perform updates triggered by event:
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            assert(self.X[transitionNode]==self.stateID[transition['from']]), "Assertion error: Node "+str(transitionNode)+" has unexpected current state "+str(self.X[transitionNode])+" given the intended transition of "+transition['from']+"->"+transition['to']+"."
            self.set_state(transitionNode, transition['to']) # self.X[transitionNode] = self.stateID[transition['to']]
            self.state_timer[transitionNode] = 0.0
            # TODO: some version of this?   self.testedInCurrentState[transitionNode] = False

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Save information about infection events when they occur:
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if(self.log_infection_info and transition['type']=='infection'):
                infectionLog = {'t':                  self.t,
                                'infected_node':      transitionNode,
                                'preInfectionState':  transition['from'],
                                'postInfectionState': transition['to'],
                                'contact_info':       {} 
                                }
                for netID, G in self.networks.items():
                    infectionLog['contact_info'].update({'num_contacts':G['degree'][transitionNode,0],
                                                         'contacts_states': self.X[ np.argwhere(G['adj_matrix'][transitionNode,:]>0) ].flatten().tolist(),
                                                         'contacts_isolations': self.isolation[ np.argwhere(G['adj_matrix'][transitionNode,:]>0) ].flatten().tolist()
                                                        })
                self.infectionLogs.append(infectionLog)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        else: # propensities.sum() == 0
            # No tau calculated, advance time by default step
            tau               = default_dt
            self.t           += tau
            self.state_timer += tau
            self.tidx        += 1

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        self.update_data_series()

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # Update isolation timers/statuses
        i_isolated = np.argwhere(self.isolation==1).flatten()
        self.isolation_timer[i_isolated]    += tau
        self.totalIsolationTime[i_isolated] += tau
        if(self.isolation_period is not None):
            i_exitingIsolation = np.argwhere(self.isolation_timer >= self.isolation_period).flatten()
            for i in i_exitingIsolation:
                self.set_isolation(node=i, isolation=False)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Terminate if tmax reached:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if(self.t >= self.tmax):
            self.finalize_data_series()
            return False

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        return True


    ########################################################


    def run(self, T, checkpoints=None, max_dt=None, default_dt=0.1):
        if(T>0):
            self.tmax += T
        else:
            return False

        if(max_dt is None and self.transition_mode=='time_in_state'):
            max_dt = 1

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Pre-process checkpoint values:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # TODO 

        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Run the simulation loop:
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        running     = True
        while running:

            print(self.t)

            running = self.run_iteration(max_dt, default_dt)

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Handle checkpoints if applicable:
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # TODO

        #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        return True
    

    ########################################################
    ########################################################


    def update_data_series(self):
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update the time series:
        self.tseries[self.tidx]     = self.t
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update the data series of counts of nodes in each compartment:
        for compartment in self.compartments:
            self.counts[compartment][self.tidx] = np.count_nonzero(self.X==self.stateID[compartment])
            #------------------------------------
            if(compartment not in self.excludeFromEffPopSize):
                self.N[self.tidx] += self.counts[compartment][self.tidx]
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update the data series of counts of nodes with each flag:
        if(self.track_flag_counts):
            for flag in self.allCompartmentFlags.union(self.allNodeFlags):
                flag_count = len(self.get_individuals_by_flag(flag))
                self.flag_counts[flag][self.tidx] = flag_count
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Store system states
        if(self.store_Xseries):
            self.Xseries[self.tidx,:] = self.X.T

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Store system states for specified subgroups
        if(self.nodeGroupData):
            for groupName in self.nodeGroupData:
                for compartment in self.compartments:
                    self.nodeGroupData[groupName][compartment][self.tidx] = np.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.stateID[compartment])
                    #------------------------------------
                    if(compartment not in self.excludeFromEffPopSize):
                        self.nodeGroupData[groupName]['N'][self.tidx] += self.counts[compartment][self.tidx]


    ########################################################


    def increase_data_series_length(self):
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Allocate more entries for the time series:
        self.tseries = np.pad(self.tseries, [(0, self.pop_size*min(len(self.compartments), 10))], mode='constant', constant_values=0)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Allocate more entries for the data series of counts of nodes in each compartment:
        for compartment in self.compartments:
            self.counts[compartment] = np.pad(self.counts[compartment], [(0, self.pop_size*min(len(self.compartments), 10))], mode='constant', constant_values=0)
        #------------------------------------
        self.N = np.pad(self.N, [(0, self.pop_size*min(len(self.compartments), 10))], mode='constant', constant_values=0)
        #----------------------------------------
        # TODO: Allocate more entries for the data series of counts of nodes that have certain conditions?
        #        - infected, tested, vaccinated, positive, etc?
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Store system states
        if(self.store_Xseries):
            self.Xseries = self.Xseries[:self.tidx+1, :]

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Store system states for specified subgroups
        if(self.nodeGroupData):
            for groupName in self.nodeGroupData:
                for compartment in self.compartments:
                    self.nodeGroupData[groupName][compartment] = np.pad(self.nodeGroupData[groupName][compartment], [(0, self.pop_size*min(len(self.compartments), 10))], mode='constant', constant_values=0)
                #------------------------------------
                self.nodeGroupData[groupName]['N'] = np.pad(self.nodeGroupData[groupName]['N'], [(0, self.pop_size*min(len(self.compartments), 10))], mode='constant', constant_values=0)
                #------------------------------------
                # TODO: Allocate more entries for the data series of counts of nodes that have certain conditions?
                #        - infected, tested, vaccinated, positive, etc?


    ########################################################
    

    def finalize_data_series(self):
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Finalize the time series:
        self.tseries = np.array(self.tseries, dtype=float)[:self.tidx+1]
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Finalize the data series of counts of nodes in each compartment:
        for compartment in self.compartments:
            self.counts[compartment] = np.array(self.counts[compartment], dtype=float)[:self.tidx+1]
        #------------------------------------
        self.N = np.array(self.N, dtype=float)[:self.tidx+1]
        #----------------------------------------
        # TODO: Finalize the data series of counts of nodes that have certain conditions?
        #        - infected, tested, vaccinated, positive, etc?
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Store system states
        if(self.store_Xseries):
            self.Xseries = self.Xseries[:self.tidx+1, :]

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Store system states for specified subgroups
        if(self.nodeGroupData):
            for groupName in self.nodeGroupData:
                for compartment in self.compartments:
                    self.nodeGroupData[groupName][compartment] = np.array(self.nodeGroupData[groupName][compartment], dtype=float)[:self.tidx+1]
                #------------------------------------
                self.nodeGroupData[groupName]['N'] = np.array(self.nodeGroupData[groupName]['N'], dtype=float)[:self.tidx+1]
                #------------------------------------
                # TODO: Finalize the data series of counts of nodes that have certain conditions?
                #        - infected, tested, vaccinated, positive, etc?

    
    ########################################################
    ########################################################


    def process_transition_times(self, transn_dict):
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Ensure all transitions have a specified rate or time, as applicable:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        poststates  = list(transn_dict.keys())
        for poststate in poststates:
            if(self.transition_mode=='exponential_rates'):
                if('rate' in transn_dict[poststate]):
                    # Rate(s) provided, simply ensure correct shape:
                    transn_dict[poststate]['rate'] = np.array(transn_dict[poststate]['rate']).reshape((self.pop_size,1))
                elif('time' in transn_dict[poststate]):
                    # Time(s) provided, compute rates as inverse of times:
                    transn_dict[poststate]['rate'] = np.array(1/transn_dict[poststate]['time']).reshape((self.pop_size,1))
            elif(self.transition_mode=='time_in_state'):
                if('time' in transn_dict[poststate]):
                    # Rate(s) provided, simply ensure correct shape:
                    transn_dict[poststate]['time'] = np.array(transn_dict[poststate]['time']).reshape((self.pop_size,1))
                elif('rate' in transn_dict[poststate]):
                    # Rate(s) provided, compute rates as inverse of rates:
                    transn_dict[poststate]['time'] = np.array(1/transn_dict[poststate]['rate']).reshape((self.pop_size,1))
            else:
                raise BaseException("Unrecognized transmission_mode, "+self.transmission_mode+", provided.")


    ########################################################


    def process_transition_probs(self, transn_dict):
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Decide the transition each individual will take according to given probabilities:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        poststates  = list(transn_dict.keys())
        probs = []
        for poststate in poststates:
            try:             
                prob = transn_dict[poststate]['prob']
                prob = np.array(prob).reshape((self.pop_size, 1)) if isinstance(prob, (list, np.ndarray)) else np.full(fill_value=prob, shape=(self.pop_size,1))
                transn_dict[poststate]['prob'] = prob
                probs.append(prob)
            except KeyError: 
                if(len(poststates) == 1):
                    transn_dict[poststate]['prob'] = np.ones(shape=(self.pop_size,1))
                    probs.append(transn_dict[poststate]['prob']) 
                else:
                    print("Multiple transitions specified, but not all probabilities provided: Assuming equiprobable.")
                    transn_dict[poststate]['prob'] = np.full(1/len(poststates), shape=(self.pop_size,1))
                    probs.append(transn_dict[poststate]['prob']) 
        probs = np.array(probs).reshape((len(poststates), self.pop_size))
        #----------------------------------------
        rands = [poststates[np.random.choice(len(poststates), p=probs[:,i])] for i in range(self.pop_size)]
        #----------------------------------------
        for poststate in transn_dict:
            transn_dict[poststate]["active_path"] = np.array([1 if rands[i]==poststate else 0 for i in range(self.pop_size)]).reshape((self.pop_size,1))


    ########################################################


    def process_local_transmissibility(self, transm_dict, network):

        #----------------------------------------
        # Process local transmissibility parameters for each network:
        #----------------------------------------
        try:
            local_transm_vals = transm_dict[network]
            #----------------------------------------
            # Convert transmissibility value(s) to np array:
            local_transm_vals = np.array(local_transm_vals) if isinstance(local_transm_vals, (list, np.ndarray)) else np.full(fill_value=local_transm_vals, shape=(self.pop_size,1))
            #----------------------------------------
            # Generate matrix of pairwise transmissibility values:
            if(local_transm_vals.ndim == 2 and local_transm_vals.shape[0] == self.pop_size and local_transm_vals.shape[1] == self.pop_size):
                net_transm_mat = local_transm_vals
            elif((local_transm_vals.ndim == 1 and local_transm_vals.shape[0] == self.pop_size) or (local_transm_vals.ndim == 2 and (local_transm_vals.shape[0] == self.pop_size or local_transm_vals.shape[1] == self.pop_size))):
                local_transm_vals = local_transm_vals.reshape((self.pop_size,1))
                # Pre-multiply beta values by the adjacency matrix ("transmission weight connections")
                A_beta_pairwise_byInfected = scipy.sparse.csr_matrix.multiply(self.networks[network]["adj_matrix"], local_transm_vals.T).tocsr()
                A_beta_pairwise_byInfectee = scipy.sparse.csr_matrix.multiply(self.networks[network]["adj_matrix"], local_transm_vals).tocsr()    
                #------------------------------
                # Compute the effective pairwise beta values as a function of the infected/infectee pair:
                if(transm_dict['pairwise_mode'].lower() == 'infected'):
                    net_transm_mat = A_beta_pairwise_byInfected
                elif(transm_dict['pairwise_mode'].lower() == 'infectee'):
                    net_transm_mat = A_beta_pairwise_byInfectee
                elif(transm_dict['pairwise_mode'].lower() == 'min'):
                    net_transm_mat = scipy.sparse.csr_matrix.minimum(A_beta_pairwise_byInfected, A_beta_pairwise_byInfectee)
                elif(transm_dict['pairwise_mode'].lower() == 'max'):
                    net_transm_mat = scipy.sparse.csr_matrix.maximum(A_beta_pairwise_byInfected, A_beta_pairwise_byInfectee)
                elif(transm_dict['pairwise_mode'].lower() == 'mean' or transm_dict['pairwise_mode'] is None):
                    net_transm_mat = (A_beta_pairwise_byInfected + A_beta_pairwise_byInfectee)/2
                else:
                    raise BaseException("Unrecognized pairwise_mode value (support for 'infected', 'infectee', 'min', 'max', and 'mean').")
            else:
                raise BaseException("Invalid data type/shape for transmissibility values.")
            #----------------------------------------
            # Store the pairwise transmissibility matrix in the compartments dict
            transm_dict[network] = net_transm_mat
        except KeyError:
            # print("Transmissibility values not given for \""+str(G)+"\" network -- defaulting to 0.")
            transm_dict[network] = scipy.sparse.csr_matrix(np.zeros(shape=(self.pop_size, self.pop_size)))


    ########################################################


    def process_local_transm_offsets(self, transm_dict, network):
        #----------------------------------------
        # Process frequency-dependent transmission offset factors for each network:
        #----------------------------------------
        if('offsets' not in transm_dict):
            transm_dict['offsets'] = {}
        #----------------------------------------
        try:
            omega_vals = transm_dict['offsets'][network]
            #----------------------------------------
            # Convert omega value(s) to np array:
            omega_vals = np.array(omega_vals) if isinstance(omega_vals, (list, np.ndarray)) else np.full(fill_value=omega_vals, shape=(self.pop_size,1))
            #----------------------------------------
            # Store 2d np matrix of pairwise omega values:
            if(omega_vals.ndim == 2 and omega_vals.shape[0] == self.pop_size and omega_vals.shape[1] == self.pop_size):
                nested_dic[G+"_omega"] = omega_vals
            else:
                raise BaseException("Explicit omega values should be specified as an NxN 2d array. Else leave unspecified and omega values will be automatically calculated according to local_transm_offset_mode.")
        except KeyError:
            #----------------------------------------
            # Automatically generate omega matrix according to local_transm_offset_mode:
            if(transm_dict['local_transm_offset_mode'].lower() == 'pairwise_log'):
                with np.errstate(divide='ignore'): # ignore log(0) warning, then convert log(0) = -inf -> 0.0
                    omega = np.log(np.maximum(self.networks[network]["degree"],2))/np.log(np.mean(self.networks[network]["degree"])) 
                    omega[np.isneginf(omega)] = 0.0
            elif(transm_dict['local_transm_offset_mode'].lower() == 'pairwise_linear'):
                omega = np.maximum(self.networks[network]["degree"],2)/np.mean(self.networks[network]["degree"])
            elif(transm_dict['local_transm_offset_mode'].lower() == 'none'):
                omega = np.ones(shape=(self.pop_size, self.pop_size))
            else:
                raise BaseException("Unrecognized local_transm_offset_mode value (support for 'pairwise_log', 'pairwise_linear', and 'none').")
            omega_pairwise_byInfected = scipy.sparse.csr_matrix.multiply(self.networks[network]["adj_matrix"], omega.T).tocsr()
            omega_pairwise_byInfectee = scipy.sparse.csr_matrix.multiply(self.networks[network]["adj_matrix"], omega).tocsr()
            omega_mat = (omega_pairwise_byInfected + omega_pairwise_byInfectee)/2
            #----------------------------------------
            # Store the pairwise omega matrix in the compartments dict
            transm_dict['offsets'][network] = omega_mat


    ########################################################


    def process_initial_states(self):
        
        #----------------------------------------
        # Determine the iniital counts for each state given their specified initial prevalences
        initCountTotal = 0
        for c, compartment in enumerate(self.compartments):
            comp_params = self.compartments[compartment]
            #----------------------------------------
            # Instantiate data series for counts of nodes in each compartment:
            self.counts[compartment] = np.zeros(self.pop_size*min(len(self.compartments), 10))
            #----------------------------------------
            # Set initial counts for each compartment:
            if(comp_params['initial_prevalence'] > 0):
                self.counts[compartment][0] = min( max(int(self.pop_size*comp_params['initial_prevalence']),1), self.pop_size-initCountTotal )
                initCountTotal += self.counts[compartment][0]
            
        #----------------------------------------
        # Initialize remaining counts to the designated default compartment:
        if(initCountTotal < self.pop_size):
            if(self.default_state is not None):
                self.counts[self.default_state][0] = self.pop_size - initCountTotal
            else:
                raise BaseException("A default compartment must be designated ('default_state':True in config) when the total initial count is less than the population size.")

        #----------------------------------------
        # Initialize data series for effective population size (N):
        self.N = np.zeros(self.pop_size*min(len(self.compartments), 10))
        for c, compartment in enumerate(self.compartments):
            if(compartment not in self.excludeFromEffPopSize):
                self.N[0] += self.counts[compartment][0]

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize the states of individuals:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.X = np.concatenate([[self.stateID[comp]]*int(self.counts[comp][0]) for comp in self.compartments]).reshape((self.pop_size,1))
        np.random.shuffle(self.X)

        if(self.store_Xseries):
            self.Xseries        = np.zeros(shape=(6*self.pop_size, self.pop_size), dtype='uint8')
            self.Xseries[0,:]   = self.X.T

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        #----------------------------------------
        # Determine the iniital counts for each flag
        if(self.track_flag_counts):
            for flag in self.allCompartmentFlags.union(self.allNodeFlags):
                #----------------------------------------
                # Instantiate data series for counts of nodes with each flag:
                self.flag_counts[flag] = np.zeros(self.pop_size*min(len(self.compartments), 10))
                #----------------------------------------
                # Set initial counts for each flag:
                flag_count = len(self.get_individuals_by_flag(flag))
                self.flag_counts[flag][0] = flag_count

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        self.update_data_series()


    ########################################################
    ########################################################


    def set_state(self, node, state):
        # Using this function instead of setting self.X directly ensures that the data series are updated whenever a state changes.
        nodes = [node] if not isinstance(node, (list, np.ndarray)) else node
        for i in nodes:
            if(state in self.compartments):
                self.X[i] = self.stateID[state]
            elif(state in self.stateID):
                self.X[i] = state
            else:
                print("Unrecognized state, "+str(state)+". No state update performed.")
                return
        self.update_data_series()


    ########################################################


    def set_transition_rate(self, compartment, to, rate):
        # Note that it only makes sense to set a rate for temporal transitions.
        compartments = [compartment] if not isinstance(compartment, (list, np.ndarray)) else compartment
        destStates   = [to] if not isinstance(to, (list, np.ndarray)) else to
        for compartment in compartments:
            transn_dict = self.compartments[compartment]['transitions']
            for destState in destStates:
                try:
                    transn_dict[destState]['rate'] = rate
                    transn_dict[destState]['time'] = 1/rate
                except KeyError:
                    transn_dict[destState] = {'rate': rate}
                    transn_dict[destState] = {'time': 1/rate}
            self.process_transition_times(transn_dict)
            # process probs in case a new transition was added above
            self.process_transition_probs(transn_dict) 


    ########################################################


    def set_transition_time(self, compartment, to, time):
        # Note that it only makes sense to set a time for temporal transitions.
        compartments = [compartment] if not isinstance(compartment, (list, np.ndarray)) else compartment
        destStates   = [to] if not isinstance(to, (list, np.ndarray)) else to
        for compartment in compartments:
            transn_dict = self.compartments[compartment]['transitions']
            for destState in destStates:
                try:
                    transn_dict[destState]['time'] = time
                    transn_dict[destState]['rate'] = 1/time
                except KeyError:
                    transn_dict[destState] = {'time': time}
                    transn_dict[destState] = {'rate': 1/time}
            self.process_transition_times(transn_dict)
            # process probs in case a new transition was added above
            self.process_transition_probs(transn_dict) 


    ########################################################


    def set_transition_probability(self, compartment, probs_dict, upon_exposure_to=None):
        compartments     = [compartment] if not isinstance(compartment, (list, np.ndarray)) else compartment
        infectiousStates = [upon_exposure_to] if (not isinstance(upon_exposure_to, (list, np.ndarray)) and upon_exposure_to is not None) else upon_exposure_to
        for compartment in compartments:
            if(upon_exposure_to is None):
                transn_dict = self.compartments[compartment]['transitions']
                for destState in probs_dict:    
                    try:
                        transn_dict[destState]['prob'] = probs_dict[destState]
                    except KeyError:
                        # print("Compartment", compartment, "has no specified transition to", destState, "for which to set a probability.")
                        transn_dict[destState] = {'prob': probs_dict[destState]}
            else:
                for infectiousState in infectiousStates:
                    transn_dict = self.compartments[compartment]['susceptibilities'][infectiousState]['transitions']
                    for destState in probs_dict:    
                        try:
                            transn_dict[destState]['prob'] = probs_dict[destState]
                        except KeyError:
                            # print("Compartment", compartment, "has no specified transition to", destState, "for which to set a probability.")
                            transn_dict[destState] = {'prob': probs_dict[destState]}
            self.process_transition_probs(transn_dict)


    ########################################################


    def set_susceptibility(self, compartment, to, susceptibility):
        compartments     = [compartment] if not isinstance(compartment, (list, np.ndarray)) else compartment
        infectiousStates = [to] if not isinstance(to, (list, np.ndarray)) else to
        susceptibility   = np.array(susceptibility).reshape((self.pop_size,1))
        for compartment in compartments:
            for infectiousState in infectiousStates:
                susc_dict = self.compartments[compartment]['susceptibilities']
                try:
                    susc_dict[infectiousState]['susceptibility'] = susceptibility
                except KeyError:
                    susc_dict[infectiousState] = {'susceptibility': susceptibility}


    ########################################################


    def set_transmissibility(self, compartment, transm_mode, transmissibility):
        compartments = [compartment] if not isinstance(compartment, (list, np.ndarray)) else compartment
        transmModes  = [transm_mode] if not isinstance(transm_mode, (list, np.ndarray)) else transm_mode
        for compartment in compartments:
            transm_dict = self.compartments[compartment]['transmissibilities']
            for transmMode in transmModes:
                #----------------------------------------
                # Handle update to local transmissibility over the specified network:
                if(transmMode in self.networks):
                    transm_dict[transmMode] = transmissibility
                    self.process_local_transmissibility(transm_dict, transmMode)
                    #----------------------------------------
                    # Re-calculate Infectivity Matrices for updated compartments/networks,
                    # which pre-combine transmissibility, adjacency, and freq-dep offset terms.
                    #----------------------------------------
                    # M_G = (AB)_G * D_G
                    self.infectivity_mat[compartment][transmMode] = scipy.sparse.csr_matrix.multiply(transm_dict[transmMode], transm_dict['offsets'][transmMode])
                #----------------------------------------
                # Handle update to exogenous transmissibility:
                elif(transmMode=='exogenous'):
                    transm_dict['exogenous'] = np.array(transmissibility).reshape((self.pop_size,1))
                    self.exogenous_prevalence[compartment] = transm_dict['exogenous']
                #----------------------------------------
                else:
                    print("Transmission mode,", transmMode, "not recognized (expected 'exogenous', or network name in "+str(list(self.networks.keys()))+"); no update.")
            #----------------------------------------
            # Re-calculate global transmissibility as the mean of local transmissibilities.
            #----------------------------------------
            transm_dict['global'] = np.sum([np.sum(transm_dict[G][transm_dict[G]!=0]) for G in self.networks]) / max(np.sum([transm_dict[G].count_nonzero() for G in  self.networks]), 1)


    ########################################################


    def set_initial_prevalence(self, compartment, prevalence):
        compartments = [compartment] if not isinstance(compartment, (list, np.ndarray)) else compartment
        for compartment in compartments:
            self.compartments[compartment]['initial_prevalence'] = prevalence
        self.process_initial_states()

    ########################################################

    
    def set_exogenous_prevalence(self, compartment, prevalence):
        compartments = [compartment] if not isinstance(compartment, (list, np.ndarray)) else compartment
        for compartment in compartments:
            # Update the compartment model definition dictionary
            self.compartments[compartment]['exogenous_prevalence'] = prevalence
            # Update the exogenous prevalence variable in the model object
            self.exogenous_prevalence[compartment] = prevalence


    ########################################################

    
    def set_default_state(self, compartment):
        # Must be a single compartment given
        for c in self.compartments:
            self.compartments[c]['default_state'] = (c == compartment)
            if(c == compartment):
                self.default_state = self.stateID[c]    
       

    ########################################################

    
    def set_exclude_from_eff_pop(self, compartment, exclude=True):
        compartments = [compartment] if not isinstance(compartment, (list, np.ndarray)) else compartment
        for compartment in compartments:
            self.compartments[compartment]['exclude_from_eff_pop'] = exclude
            if(exclude and not compartment in self.excludeFromEffPopSize):
                self.excludeFromEffPopSize.append(compartment)
            elif(not exclude and compartment in self.excludeFromEffPopSize):
                self.excludeFromEffPopSize = [c for c in self.excludeFromEffPopSize if c!=compartment] # remove all occurrences of compartment


    ########################################################


    def set_isolation(self, node, isolation):
        nodes = [node] if not isinstance(node, (list, np.ndarray)) else node
        for node in nodes:
            if(isolation == True):
                # self.calc_infectious_time(node) <- TODO handle this?
                self.isolation[node] = 1
            elif(isolation == False):
                self.isolation[node] = 0
            # Reset the isolation timer:
            self.isolation_timer[node] = 0


    ########################################################


    def set_network_activity(self, network, node='all', active=None, active_isolation=None):
        nodes      = list(range(self.pop_size)) if node=='all' else [node] if not isinstance(node, (list, np.ndarray)) else node
        networks = [network] if not isinstance(network, (list, np.ndarray)) else network
        for i in nodes:
            for G in networks:
                if(active is not None):
                    self.networks[G]['active'][i] = 1 if active else 0
                if(active_isolation is not None):
                    self.networks[G]['active_isolation'][i] = 1 if active_isolation else 0


    ########################################################
    ########################################################


    def add_compartment_flag(self, compartment, flag):
        compartments = list(range(self.pop_size)) if compartment=='all' else [compartment] if not isinstance(compartment, (list, np.ndarray)) else compartment
        flags        = [flag] if not isinstance(flag, (list, np.ndarray)) else flag
        for compartment in compartments:
            for flag in flags:
                self.compartments[compartment]['flags'].append(flag)
                self.allCompartmentFlags.add(flag)
                if(flag not in self.flag_counts):
                    self.flag_counts[flag] = np.zeros_like(self.counts[compartment])
                    self.update_data_series()


    def remove_compartment_flag(self, compartment, flag):
        compartments = list(range(self.pop_size)) if compartment=='all' else [compartment] if not isinstance(compartment, (list, np.ndarray)) else compartment
        flags        = [flag] if not isinstance(flag, (list, np.ndarray)) else flag
        for compartment in compartments:
            for flag in flags:
                self.compartments[compartment]['flags'] = [f for f in self.compartments[compartment]['flags'] if f!=flag] # remove all occurrences of flag


    ########################################################


    def add_node_flag(self, node, flag):
        nodes = list(range(self.pop_size)) if node=='all' else [node] if not isinstance(node, (list, np.ndarray)) else node
        flags = [flag] if not isinstance(flag, (list, np.ndarray)) else flag
        for node in nodes:
            for flag in flags:
                self.node_flags.append(flag)
                self.allNodeFlags.add(flag)
                if(flag not in self.flag_counts):
                    self.flag_counts[flag] = np.zeros_like(self.counts[list(self.counts.keys())[0]])
                    self.update_data_series()


    def remove_node_flag(self, node, flag):
        nodes = list(range(self.pop_size)) if node=='all' else [node] if not isinstance(node, (list, np.ndarray)) else node
        flags = [flag] if not isinstance(flag, (list, np.ndarray)) else flag
        for node in nodes:
            for flag in flags:
                self.node_flags = [f for f in self.node_flags if f!=flag] # remove all occurrences of flag


    ########################################################


    def get_compartments_by_flag(self, flag):
        flags = [flag] if not isinstance(flag, (list, np.ndarray)) else flag
        flagged_compartments = set()
        for compartment, comp_dict in self.compartments.items():
            if(any([flag in comp_dict['flags'] for flag in flags])):
                flagged_compartments.add(compartment)
        return list(flagged_compartments)


    ########################################################


    def get_individuals_by_flag(self, flag):
        flags = [flag] if not isinstance(flag, (list, np.ndarray)) else flag
        flagged_individuals = set()
        for i in range(self.pop_size):
            # Check if individual i has this node flag:
            if(any([flag in self.node_flags[i] for flag in flags])):
                flagged_individuals.add(i)
            # Check if individual i is in a compartment with this flag:
            flagged_compartments  = self.get_compartments_by_flag(flag)
            if(any([ self.X[i]==self.stateID[c] for c in flagged_compartments ])):
                flagged_individuals.add(i)
        return list(flagged_individuals)


    ########################################################


    def get_count_by_flag(self, flag):
        flags = [flag] if not isinstance(flag, (list, np.ndarray)) else flag
        flag_counts_ = {}
        for flag in flags:
            flag_counts_[flag] = len(self.get_individuals_by_flag(flag))
        return flag_counts_


    ########################################################
    ########################################################


    def introduce_random_exposures(self, num, compartment='all', exposed_to='any'):
        compartments      = list(self.compartments.keys()) if compartment=='all' else [compartment] if not isinstance(compartment, (list, np.ndarray)) else compartment
        infectiousStates  = list(self.compartments.keys()) if exposed_to=='any'  else [exposed_to] if (not isinstance(exposed_to, (list, np.ndarray)) and exposed_to is not None) else exposed_to
        exposure_susceptibilities = []
        exposedNodes = []
        for exposure in range(num):
            for compartment in compartments:
                for infectiousState in infectiousStates:
                    if(infectiousState in self.compartments[compartment]['susceptibilities']):
                        exposure_susceptibilities.append({'susc_state': compartment, 'inf_state': infectiousState, 
                                                          'susceptibilities': self.compartments[compartment]['susceptibilities'][infectiousState]['susceptibility'].flatten(),
                                                          'mean_susceptibility': np.mean(self.compartments[compartment]['susceptibilities'][infectiousState]['susceptibility']),
                                                          })
            exposureType   = np.random.choice(exposure_susceptibilities, p=[d['mean_susceptibility'] for d in exposure_susceptibilities]/np.sum([d['mean_susceptibility'] for d in exposure_susceptibilities]))
            exposableNodes = [i for i in range(self.pop_size) if self.X[i]==self.stateID[exposureType['susc_state']]]
            if(len(exposableNodes) > 0):
                exposedNode    = np.random.choice(exposableNodes, p=exposureType['susceptibilities'][exposableNodes]/np.sum(exposureType['susceptibilities'][exposableNodes]))
                exposedNodes.append(exposedNode)
                #--------------------
                exposureTransitions = self.compartments[exposureType['susc_state']]['susceptibilities'][exposureType['inf_state']]['transitions']
                exposureTransitionsActiveStatuses = [exposureTransitions[dest]['active_path'].flatten()[exposedNode] for dest in exposureTransitions]
                destState = np.random.choice(list(exposureTransitions.keys()), p=exposureTransitionsActiveStatuses/np.sum(exposureTransitionsActiveStatuses))
                #--------------------
                self.set_state(exposedNode, destState)
        return exposedNodes




########################################################
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#
#@                                                    @#
#@  Programmatically building up compartment models   @#
#@                                                    @#
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#
########################################################

class CompartmentModelBuilider():

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def __init__(self):

       self.compartments = {}

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def add_compartment(self, name, transitions=None, transmissibilities=None, susceptibilities=None, 
                                initial_prevalence=0.0, exogenous_prevalence=0.0, flags=None, 
                                default_state=None, exclude_from_eff_pop=False):
        self.compartments[name] = { 'transitions':          transitions if transitions is not None else {}, 
                                    'transmissibilities':   transmissibilities if transmissibilities is not None else {}, 
                                    'susceptibilities':     susceptibilities if susceptibilities is not None else {},
                                    'initial_prevalence':   initial_prevalence, 
                                    'exogenous_prevalence': exogenous_prevalence, 
                                    'flags':                flags if flags is not None else [],
                                    'default_state':        default_state if default_state is not None else False,
                                    'exclude_from_eff_pop': exclude_from_eff_pop }
        if(default_state is None and not any([self.compartments[c]['default_state'] for c in self.compartments])):
            # There is no default state set so far, make this new compartment the default
            self.compartments[name]['default_state'] = True

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def add_compartments(self, names):
        for name in names:
            self.add_compartment(name)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def add_transition(self, compartment, to, upon_exposure_to=None, rate=None, time=None, prob=None):
        # Add transition for one compartment and destination state at a time
        infectiousStates = [upon_exposure_to] if (not isinstance(upon_exposure_to, (list, np.ndarray)) and upon_exposure_to is not None) else upon_exposure_to
        if(upon_exposure_to is None): # temporal transition
            transn_config = {}
            if(time is not None):
                transn_config.update({'time': time, 'rate': 1/time})
            if(rate is not None):
                transn_config.update({'rate': rate, 'time': 1/rate})
            if(prob is not None):
                transn_config.update({'prob': prob})
            self.compartments[compartment]['transitions'].update({to: transn_config})
        else: # transmission-induced transition
            for infectiousState in infectiousStates:
                transn_config = {}
                if(prob is not None):
                    # transmission-induced transition do not have rates/times
                    transn_config.update({'prob': prob})
                if(infectiousState in self.compartments[compartment]['susceptibilities']):
                    self.compartments[compartment]['susceptibilities'][infectiousState]['transitions'].update({to: transn_config})
                else:
                    self.compartments[compartment]['susceptibilities'].update({infectiousState: {'transitions': {to: transn_config}}})

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def set_transition_rate(self, compartment, to, rate):
        # Note that it only makes sense to set a rate for temporal transitions.
        compartments = [compartment] if not isinstance(compartment, (list, np.ndarray)) else compartment
        destStates   = [to] if not isinstance(to, (list, np.ndarray)) else to
        for compartment in compartments:
            transn_dict = self.compartments[compartment]['transitions']
            for destState in destStates:
                try:
                    transn_dict[destState]['rate'] = rate
                    transn_dict[destState]['time'] = 1/rate
                except KeyError:
                    transn_dict[destState] = {'rate': rate}
                    transn_dict[destState] = {'time': 1/rate}

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def set_transition_time(self, compartment, to, time):
        # Note that it only makes sense to set a time for temporal transitions.
        compartments = [compartment] if not isinstance(compartment, (list, np.ndarray)) else compartment
        destStates   = [to] if not isinstance(to, (list, np.ndarray)) else to
        for compartment in compartments:
            transn_dict = self.compartments[compartment]['transitions']
            for destState in destStates:
                try:
                    transn_dict[destState]['time'] = time
                    transn_dict[destState]['rate'] = 1/time
                except KeyError:
                    transn_dict[destState] = {'time': time}
                    transn_dict[destState] = {'rate': 1/time}

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def set_transition_probability(self, compartment, probs_dict, upon_exposure_to=None):
        compartments     = [compartment] if not isinstance(compartment, (list, np.ndarray)) else compartment
        infectiousStates = [upon_exposure_to] if (not isinstance(upon_exposure_to, (list, np.ndarray)) and upon_exposure_to is not None) else upon_exposure_to
        for compartment in compartments:
            if(upon_exposure_to is None):
                transn_dict = self.compartments[compartment]['transitions']
                for destState in probs_dict:    
                    try:
                        transn_dict[destState]['prob'] = probs_dict[destState]
                    except KeyError:
                        transn_dict[destState] = {'prob': probs_dict[destState]}
            else:
                for infectiousState in infectiousStates:
                    transn_dict = self.compartments[compartment]['susceptibilities'][infectiousState]['transitions']
                    for destState in probs_dict:    
                        try:
                            transn_dict[destState]['prob'] = probs_dict[destState]
                        except KeyError:
                            transn_dict[destState] = {'prob': probs_dict[destState]}
            
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def set_susceptibility(self, compartment, to, susceptibility=1.0, transitions={}):
        compartments     = [compartment] if not isinstance(compartment, (list, np.ndarray)) else compartment
        infectiousStates = [to] if not isinstance(to, (list, np.ndarray)) else to
        for compartment in compartments:
            for infectiousState in infectiousStates:
                self.compartments[compartment]['susceptibilities'].update({infectiousState: {'susceptibility': susceptibility, 'transitions': transitions}})

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def set_transmissibility(self, compartment, transm_mode, transmissibility=0.0):
        compartments = [compartment] if not isinstance(compartment, (list, np.ndarray)) else compartment
        transmModes  = [transm_mode] if not isinstance(transm_mode, (list, np.ndarray)) else transm_mode
        for compartment in compartments:
            transm_dict = self.compartments[compartment]['transmissibilities']
            for transmMode in transmModes:
                transm_dict = self.compartments[compartment]['transmissibilities'].update({transmMode: transmissibility})

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def set_initial_prevalence(self, compartment, prevalence=0.0):
        compartments = [compartment] if not isinstance(compartment, (list, np.ndarray)) else compartment
        for compartment in compartments:
            self.compartments[compartment]['initial_prevalence'] = prevalence
        
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def set_exogenous_prevalence(self, compartment, prevalence=0.0):
        compartments = [compartment] if not isinstance(compartment, (list, np.ndarray)) else compartment
        for compartment in compartments:
            self.compartments[compartment]['exogenous_prevalence'] = prevalence

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def set_default_state(self, compartment):
        for c in self.compartments:
            self.compartments[c]['default_state'] = (c == compartment)   
       
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def set_exclude_from_eff_pop(self, compartment, exclude=True):
        compartments = [compartment] if not isinstance(compartment, (list, np.ndarray)) else compartment
        for compartment in compartments:
            self.compartments[compartment]['exclude_from_eff_pop'] = exclude
            
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def add_compartment_flag(self, compartment, flag):
        compartments = list(range(self.pop_size)) if compartment=='all' else [compartment] if not isinstance(compartment, (list, np.ndarray)) else compartment
        flags        = [flag] if not isinstance(flag, (list, np.ndarray)) else flag
        for compartment in compartments:
            for flag in flags:
                self.compartments[compartment]['flags'].append(flag)
                
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def remove_compartment_flag(self, compartment, flag):
        compartments = list(range(self.pop_size)) if compartment=='all' else [compartment] if not isinstance(compartment, (list, np.ndarray)) else compartment
        flags        = [flag] if not isinstance(flag, (list, np.ndarray)) else flag
        for compartment in compartments:
            for flag in flags:
                self.compartments[compartment]['flags'] = [f for f in self.compartments[compartment]['flags'] if f!=flag] # remove all occurrences of flag

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def save_json(self, filename):
        with open(filename, "w") as outfile:
            json.dump(self.compartments, outfile, indent=6)




########################################################
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#
#@                                                    @#
#@  Pre-configured disease models                     @#
#@                                                    @#
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#
########################################################


class SARSCoV2NetworkModel(CompartmentNetworkModel):
    
    def __init__(self, networks,
                    R0_mean='default', R0_cv='default',
                    transmissibility='default', # overrides R0 params if given
                    susceptibility='default',
                    latent_period='default',
                    presymptomatic_period='default',
                    symptomatic_period='default',
                    pct_asymptomatic='default',
                    isolation_period='default',
                    # Other parent class args:
                    mixedness=0.0, openness=0.0,
                    transition_mode='time_in_state', 
                    local_trans_denom_mode='all_contacts',
                    log_infection_info=False,
                    store_Xseries=False,
                    node_groups=None,
                    seed=None):

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set the compartments specification for this model:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        compartments="compartments_SARSCoV2.json"

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Instantiate the base model, passing parent class args:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        super().__init__(compartments=compartments, networks=networks,
                             mixedness=mixedness, openness=openness,
                             transition_mode=transition_mode, 
                             local_trans_denom_mode=local_trans_denom_mode,
                             log_infection_info=log_infection_info,
                             store_Xseries=store_Xseries,
                             node_groups=node_groups,
                             seed=seed)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize disease-specific parameter distriutions:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        #----------------------------------------
        # Disease progression parameter distributions:
        latent_period         = gamma_dist(mean=3.0, coeffvar=0.6, N=self.pop_size) if latent_period=='default'         else np.array(latent_period).reshape((1,self.pop_size)) if isinstance(latent_period, (list, np.ndarray)) else np.full(fill_value=latent_period, shape=(1,self.pop_size))
        presymptomatic_period = gamma_dist(mean=2.2, coeffvar=0.5, N=self.pop_size) if presymptomatic_period=='default' else np.array(presymptomatic_period).reshape((1,self.pop_size)) if isinstance(presymptomatic_period, (list, np.ndarray)) else np.full(fill_value=presymptomatic_period, shape=(1,self.pop_size))
        symptomatic_period    = gamma_dist(mean=4.0, coeffvar=0.4, N=self.pop_size) if symptomatic_period=='default'    else np.array(symptomatic_period).reshape((1,self.pop_size)) if isinstance(symptomatic_period, (list, np.ndarray)) else np.full(fill_value=symptomatic_period, shape=(1,self.pop_size))
        infectious_period     = presymptomatic_period + symptomatic_period
        if(self.transition_mode=='time_in_state'):
            self.set_transition_time('E',        to='P',        time=latent_period) 
            self.set_transition_time('P',        to=['I', 'A'], time=presymptomatic_period) 
            self.set_transition_time(['I', 'A'], to='R',        time=symptomatic_period) 
        elif(self.transition_mode=='exponential_rates'):
            self.set_transition_rate('E',        to='P',        rate=1/latent_period) 
            self.set_transition_rate('P',        to=['I', 'A'], rate=1/presymptomatic_period) 
            self.set_transition_rate(['I', 'A'], to='R',        rate=1/symptomatic_period) 
        #----------------------------------------
        pct_asymptomatic = 0.3 if pct_asymptomatic=='default' else pct_asymptomatic
        self.set_transition_probability('P', {'I':1-pct_asymptomatic, 'A':pct_asymptomatic})
        
        #----------------------------------------
        # Susceptibility parameter distributions:
        if(susceptibility != 'default'):
            susceptibility  = np.array(susceptibility).reshape((1,self.pop_size)) if isinstance(susceptibility, (list, np.ndarray)) else np.full(fill_value=susceptibility, shape=(1,self.pop_size))
        else:
            susceptibility  = np.ones(shape=(1,self.pop_size))
        self.set_susceptibility('S', to=['P', 'I', 'A'], susceptibility=susceptibility)
        
        #----------------------------------------
        #Transmissibility parameter distributions:
        if(transmissibility != 'default'):
            transmissibility  = np.array(transmissibility).reshape((1,self.pop_size)) if isinstance(transmissibility, (list, np.ndarray)) else np.full(fill_value=transmissibility, shape=(1,self.pop_size))
        else:
            R0_mean           = 3.0 if R0_mean=='default' else R0_mean
            R0_cv             = 2.0 if R0_cv=='default' else R0_cv
            R0                = gamma_dist(R0_mean, R0_cv, self.pop_size)
            transmissibility  = 1/infectious_period * R0
        self.set_transmissibility(['P', 'I', 'A'], ['network'], transmissibility=transmissibility)
        #----------------------------------------
        self.mixedness = 0.2
        self.openness  = 0.0

        


    










