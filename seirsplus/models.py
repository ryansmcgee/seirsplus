from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import networkx as networkx
import numpy as numpy
import scipy as scipy


class SEIRSNetworkModel():
    """
    A class to simulate the SEIRS Model
    ===================================================
    Input:  G       Network adjacency matrix (numpy array) or Networkx graph object.
            beta    Rate of transmission (exposure) 
            sigma   Rate of infection (upon exposure) 
            gamma   Rate of recovery (upon infection) 
            xi      Rate of re-susceptibility (upon recovery)  
            mu_I    Rate of infection-related death  
            mu_0    Rate of baseline death   
            nu      Rate of baseline birth
            p       Rate of interaction outside adjacent nodes
            
            Q       Quarantine adjacency matrix (numpy array) or Networkx graph object.
            beta_D  Rate of transmission (exposure) for individuals with detected infections
            sigma_D Rate of infection (upon exposure) for individuals with detected infections
            gamma_D Rate of recovery (upon infection) for individuals with detected infections
            mu_D    Rate of infection-related death for individuals with detected infections
            theta_E Rate of baseline testing for exposed individuals
            theta_I Rate of baseline testing for infectious individuals
            phi_E   Rate of contact tracing testing for exposed individuals
            phi_I   Rate of contact tracing testing for infectious individuals
            psi_E   Rate of positive test results for exposed individuals
            psi_I   Rate of positive test results for exposed individuals
            q       Rate of quarantined individuals interaction outside adjacent nodes
            
            initE   Init number of exposed individuals       
            initI   Init number of infectious individuals      
            initD_E Init number of detected infectious individuals
            initD_I Init number of detected infectious individuals   
            initR   Init number of recovered individuals     
            initF   Init number of infection-related fatalities
                    (all remaining nodes initialized susceptible)   
    """

    def __init__(self, G, beta, sigma, gamma, xi=0, mu_I=0, mu_0=0, nu=0, p=1,
                    Q=None, beta_D=None, sigma_D=None, gamma_D=None, mu_D=None, 
                    theta_E=0, theta_I=0, phi_E=0, phi_I=0, psi_E=0, psi_I=0, q=0,
                    initE=0, initI=0, initD_E=0, initD_I=0, initR=0, initF=0):

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Setup Adjacency matrix:
        self.update_G(G)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Setup Quarantine Adjacency matrix:
        if(Q is None):
            Q = G # If no Q graph is provided, use G in its place
        self.update_Q(G)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Model Parameters:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.beta   = beta  if isinstance(beta, (list, numpy.ndarray)) else numpy.full(fill_value=beta, shape=(self.numNodes,1))
        self.sigma  = sigma if isinstance(sigma, (list, numpy.ndarray)) else numpy.full(fill_value=sigma, shape=(self.numNodes,1))
        self.gamma  = gamma if isinstance(gamma, (list, numpy.ndarray)) else numpy.full(fill_value=gamma, shape=(self.numNodes,1))
        self.xi     = xi    if isinstance(xi, (list, numpy.ndarray)) else numpy.full(fill_value=xi, shape=(self.numNodes,1))
        self.mu_I   = mu_I  if isinstance(mu_I, (list, numpy.ndarray)) else numpy.full(fill_value=mu_I, shape=(self.numNodes,1))
        self.mu_0   = mu_0  if isinstance(mu_0, (list, numpy.ndarray)) else numpy.full(fill_value=mu_0, shape=(self.numNodes,1))
        self.nu     = nu    if isinstance(nu, (list, numpy.ndarray)) else numpy.full(fill_value=nu, shape=(self.numNodes,1))
        self.p      = p     if isinstance(p, (list, numpy.ndarray)) else numpy.full(fill_value=p, shape=(self.numNodes,1))

        # Testing-related parameters:
        self.beta_D   = (beta_D if isinstance(beta_D, (list, numpy.ndarray)) else numpy.full(fill_value=beta_D, shape=(self.numNodes,1))) if beta_D is not None else self.beta
        self.sigma_D  = (sigma_D if isinstance(sigma_D, (list, numpy.ndarray)) else numpy.full(fill_value=sigma_D, shape=(self.numNodes,1))) if sigma_D is not None else self.sigma
        self.gamma_D  = (gamma_D if isinstance(gamma_D, (list, numpy.ndarray)) else numpy.full(fill_value=gamma_D, shape=(self.numNodes,1))) if gamma_D is not None else self.gamma
        self.mu_D     = (mu_D if isinstance(mu_D, (list, numpy.ndarray)) else numpy.full(fill_value=mu_D, shape=(self.numNodes,1))) if mu_D is not None else self.mu_I
        self.theta_E    = theta_E if isinstance(theta_E, (list, numpy.ndarray)) else numpy.full(fill_value=theta_E, shape=(self.numNodes,1))
        self.theta_I    = theta_I if isinstance(theta_I, (list, numpy.ndarray)) else numpy.full(fill_value=theta_I, shape=(self.numNodes,1))
        self.phi_E      = phi_E    if isinstance(phi_E, (list, numpy.ndarray)) else numpy.full(fill_value=phi_E, shape=(self.numNodes,1))
        self.phi_I      = phi_I  if isinstance(phi_I, (list, numpy.ndarray)) else numpy.full(fill_value=phi_I, shape=(self.numNodes,1))
        self.psi_E      = psi_E  if isinstance(psi_E, (list, numpy.ndarray)) else numpy.full(fill_value=psi_E, shape=(self.numNodes,1))
        self.psi_I      = psi_I    if isinstance(psi_I, (list, numpy.ndarray)) else numpy.full(fill_value=psi_I, shape=(self.numNodes,1))
        self.q   = q  if isinstance(q, (list, numpy.ndarray)) else numpy.full(fill_value=q, shape=(self.numNodes,1))

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Each node can undergo up to 4 transitions (sans vitality/re-susceptibility returns to S state),
        # so there are ~numNodes*4 events/timesteps expected; initialize numNodes*5 timestep slots to start 
        # (will be expanded during run if needed)
        self.tseries = numpy.zeros(5*self.numNodes)
        self.numE   = numpy.zeros(5*self.numNodes)
        self.numI   = numpy.zeros(5*self.numNodes)
        self.numD_E = numpy.zeros(5*self.numNodes)
        self.numD_I = numpy.zeros(5*self.numNodes)
        self.numR   = numpy.zeros(5*self.numNodes)
        self.numF   = numpy.zeros(5*self.numNodes)
        self.numS   = numpy.zeros(5*self.numNodes)
        self.N      = numpy.zeros(5*self.numNodes)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize Timekeeping:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.t      = 0
        self.tmax   = 0 # will be set when run() is called
        self.tidx   = 0
        self.tseries[0] = 0
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize Counts of inidividuals with each state:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.numE[0] = int(initE)
        self.numI[0] = int(initI)
        self.numD_E[0] = int(initD_E)
        self.numD_I[0] = int(initD_I)
        self.numR[0] = int(initR)
        self.numF[0] = int(initF)
        self.numS[0] = self.numNodes - self.numE[0] - self.numI[0] - self.numD_E[0] - self.numD_I[0] - self.numR[0] - self.numF[0]
        self.N[0]    = self.numS[0] + self.numE[0] + self.numI[0] + self.numD_E[0] + self.numD_I[0] + self.numR[0]
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Node states:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.S      = 1
        self.E      = 2
        self.I      = 3
        self.D_E    = 4
        self.D_I    = 5
        self.R      = 6
        self.F      = 7

        self.X = numpy.array([self.S]*int(self.numS[0]) + [self.E]*int(self.numE[0]) + [self.I]*int(self.numI[0]) + [self.D_E]*int(self.numD_E[0]) + [self.D_I]*int(self.numD_I[0]) + [self.R]*int(self.numR[0]) + [self.F]*int(self.numF[0])).reshape((self.numNodes,1))
        numpy.random.shuffle(self.X)

        self.transitions =  { 
                                'StoE': {'currentState':self.S, 'newState':self.E},
                                'EtoI': {'currentState':self.E, 'newState':self.I},
                                'ItoR': {'currentState':self.I, 'newState':self.R},
                                'ItoF': {'currentState':self.I, 'newState':self.F},
                                'RtoS': {'currentState':self.R, 'newState':self.S},
                                'EtoDE': {'currentState':self.E, 'newState':self.D_E},
                                'ItoDI': {'currentState':self.I, 'newState':self.D_I},
                                'DEtoDI': {'currentState':self.D_E, 'newState':self.D_I},
                                'DItoR': {'currentState':self.D_I, 'newState':self.R},
                                'DItoF': {'currentState':self.D_I, 'newState':self.F},
                                '_toS': {'currentState':True, 'newState':self.S},
                            }

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Helper variables:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.testing_scenario   = ( (numpy.any(self.psi_I) and (numpy.any(self.theta_I) or numpy.any(self.phi_I)))  
                                    or (numpy.any(self.psi_E) and (numpy.any(self.theta_E) or numpy.any(self.phi_E))) )
        self.tracing_scenario   = ( (numpy.any(self.psi_E) and numpy.any(self.phi_E)) 
                                    or (numpy.any(self.psi_I) and numpy.any(self.phi_I)) )
        self.vitality_scenario  = (numpy.any(self.mu_0) and numpy.any(self.nu))
        self.resusceptibility_scenario  = (numpy.any(self.xi))
         

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def node_degrees(self, Amat):
        return Amat.sum(axis=0).reshape(self.numNodes,1)   # sums of adj matrix cols

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def update_G(self, new_G):
        self.G = new_G
        # Adjacency matrix:
        if type(new_G)==numpy.ndarray:
            self.A = scipy.sparse.csr_matrix(new_G)
        elif type(new_G)==networkx.classes.graph.Graph:
            self.A = networkx.adj_matrix(new_G) # adj_matrix gives scipy.sparse csr_matrix
        else:
            raise BaseException("Input an adjacency matrix or networkx object only.")

        self.numNodes   = int(self.A.shape[1])
        self.degree     = numpy.asarray(self.node_degrees(self.A)).astype(float)

        return

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def update_Q(self, new_Q):
        self.Q = new_Q
        # Quarantine Adjacency matrix:
        if type(new_Q)==numpy.ndarray:
            self.A_Q = scipy.sparse.csr_matrix(new_Q)
        elif type(new_Q)==networkx.classes.graph.Graph:
            self.A_Q = networkx.adj_matrix(new_Q) # adj_matrix gives scipy.sparse csr_matrix
        else:
            raise BaseException("Input an adjacency matrix or networkx object only.")

        self.numNodes_Q   = int(self.A_Q.shape[1])
        self.degree_Q     = numpy.asarray(self.node_degrees(self.A_Q)).astype(float)

        assert(self.numNodes == self.numNodes_Q), "The normal and quarantine adjacency graphs must be of the same size."

        return


#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        
    def calc_propensities(self):

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Pre-calculate matrix multiplication terms that may be used in multiple propensity calculations,
        # and check to see if their computation is necessary before doing the multiplication
        numContacts_I = numpy.zeros(shape=(self.numNodes,1))
        if(numpy.any(self.numI[self.tidx]) 
            and numpy.any(self.beta!=0)):
            numContacts_I = numpy.asarray( scipy.sparse.csr_matrix.dot(self.A, self.X==self.I) )

        numQuarantineContacts_DI = numpy.zeros(shape=(self.numNodes,1))
        if(self.testing_scenario 
            and numpy.any(self.numD_I[self.tidx])
            and numpy.any(self.beta_D)):
            numQuarantineContacts_DI = numpy.asarray( scipy.sparse.csr_matrix.dot(self.A_Q, self.X==self.D_I) )

        numContacts_D = numpy.zeros(shape=(self.numNodes,1))
        if(self.tracing_scenario 
            and (numpy.any(self.numD_E[self.tidx]) or numpy.any(self.numD_I[self.tidx]))):
            numContacts_D = numpy.asarray( scipy.sparse.csr_matrix.dot(self.A, self.X==self.D_E)
                                            + scipy.sparse.csr_matrix.dot(self.A, self.X==self.D_I) )

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        propensities_StoE   = ( self.p*((self.beta*self.numI[self.tidx] + self.q*self.beta_D*self.numD_I[self.tidx])/self.N[self.tidx])
                                + (1-self.p)*numpy.divide((self.beta*numContacts_I + self.beta_D*numQuarantineContacts_DI), self.degree, out=numpy.zeros_like(self.degree), where=self.degree!=0)
                              )*(self.X==self.S)

        propensities_EtoI   = self.sigma*(self.X==self.E)

        propensities_ItoR   = self.gamma*(self.X==self.I)

        propensities_ItoF   = self.mu_I*(self.X==self.I)

        # propensities_EtoDE  = ( self.theta_E + numpy.divide((self.phi_E*numContacts_D), self.degree, out=numpy.zeros_like(self.degree), where=self.degree!=0) )*self.psi_E*(self.X==self.E)
        propensities_EtoDE  = (self.theta_E + self.phi_E*numContacts_D)*self.psi_E*(self.X==self.E)

        # propensities_ItoDI  = ( self.theta_I + numpy.divide((self.phi_I*numContacts_D), self.degree, out=numpy.zeros_like(self.degree), where=self.degree!=0) )*self.psi_I*(self.X==self.I)
        propensities_ItoDI  = (self.theta_I + self.phi_I*numContacts_D)*self.psi_I*(self.X==self.I)

        propensities_DEtoDI = self.sigma_D*(self.X==self.D_E)

        propensities_DItoR  = self.gamma_D*(self.X==self.D_I)

        propensities_DItoF  = self.mu_D*(self.X==self.D_I)

        propensities_RtoS   = self.xi*(self.X==self.R)

        propensities__toS   = self.nu*(self.X!=self.F)

        propensities = numpy.hstack([propensities_StoE, propensities_EtoI, 
                                     propensities_ItoR, propensities_ItoF, 
                                     propensities_EtoDE, propensities_ItoDI, propensities_DEtoDI, 
                                     propensities_DItoR, propensities_DItoF,
                                     propensities_RtoS, propensities__toS])

        columns = ['StoE', 'EtoI', 'ItoR', 'ItoF', 'EtoDE', 'ItoDI', 'DEtoDI', 'DItoR', 'DItoF', 'RtoS', '_toS']

        return propensities, columns


#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^    

    def increase_data_series_length(self):
        self.tseries = numpy.pad(self.tseries, [(0, 5*self.numNodes)], mode='constant', constant_values=0)
        self.numS = numpy.pad(self.numS, [(0, 5*self.numNodes)], mode='constant', constant_values=0)
        self.numE = numpy.pad(self.numE, [(0, 5*self.numNodes)], mode='constant', constant_values=0)
        self.numI = numpy.pad(self.numI, [(0, 5*self.numNodes)], mode='constant', constant_values=0)
        self.numD_E = numpy.pad(self.numD_E, [(0, 5*self.numNodes)], mode='constant', constant_values=0)
        self.numD_I = numpy.pad(self.numD_I, [(0, 5*self.numNodes)], mode='constant', constant_values=0)
        self.numR = numpy.pad(self.numR, [(0, 5*self.numNodes)], mode='constant', constant_values=0)
        self.numF = numpy.pad(self.numF, [(0, 5*self.numNodes)], mode='constant', constant_values=0)
        self.N = numpy.pad(self.N, [(0, 5*self.numNodes)], mode='constant', constant_values=0)
        return None

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ 

    def finalize_data_series(self):
        self.tseries = numpy.array(self.tseries, dtype=float)[:self.tidx+1]
        self.numS = numpy.array(self.numS, dtype=float)[:self.tidx+1]
        self.numE = numpy.array(self.numE, dtype=float)[:self.tidx+1]
        self.numI = numpy.array(self.numI, dtype=float)[:self.tidx+1]
        self.numD_E = numpy.array(self.numD_E, dtype=float)[:self.tidx+1]
        self.numD_I = numpy.array(self.numD_I, dtype=float)[:self.tidx+1]
        self.numR = numpy.array(self.numR, dtype=float)[:self.tidx+1]
        self.numF = numpy.array(self.numF, dtype=float)[:self.tidx+1]
        self.N = numpy.array(self.N, dtype=float)[:self.tidx+1]
        return None

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^     

    def run_iteration(self):

        if(self.tidx >= len(self.tseries)-1):
            # Room has run out in the timeseries storage arrays; double the size of these arrays:
            self.increase_data_series_length()

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 1. Generate 2 random numbers uniformly distributed in (0,1)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        r1 = numpy.random.rand()
        r2 = numpy.random.rand()

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 2. Calculate propensities
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        propensities, transitionTypes = self.calc_propensities()

        # Terminate when probability of all events is 0:
        if(propensities.sum() <= 0.0):            
            self.finalize_data_series()
            return False

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 3. Calculate alpha
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        propensities_flat   = propensities.ravel(order='F')
        cumsum              = propensities_flat.cumsum()
        alpha               = propensities_flat.sum()

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 4. Compute the time until the next event takes place
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        tau = (1/alpha)*numpy.log(float(1/r1))
        self.t += tau

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 5. Compute which event takes place
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        transitionIdx   = numpy.searchsorted(cumsum,r2*alpha)
        transitionNode  = transitionIdx % self.numNodes
        transitionType  = transitionTypes[ int(transitionIdx/self.numNodes) ]

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 6. Update node states and data series
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        assert(self.X[transitionNode] == self.transitions[transitionType]['currentState'] and self.X[transitionNode]!=self.F), "Assertion error: Node "+str(transitionNode)+" has unexpected current state "+str(self.X[transitionNode])+" given the intended transition of "+str(transitionType)+"."
        self.X[transitionNode] = self.transitions[transitionType]['newState']

        self.tidx += 1
        
        self.tseries[self.tidx]  = self.t
        self.numS[self.tidx]     = numpy.clip(numpy.count_nonzero(self.X==self.S), a_min=0, a_max=self.numNodes)
        self.numE[self.tidx]     = numpy.clip(numpy.count_nonzero(self.X==self.E), a_min=0, a_max=self.numNodes)
        self.numI[self.tidx]     = numpy.clip(numpy.count_nonzero(self.X==self.I), a_min=0, a_max=self.numNodes)
        self.numD_E[self.tidx]   = numpy.clip(numpy.count_nonzero(self.X==self.D_E), a_min=0, a_max=self.numNodes)
        self.numD_I[self.tidx]   = numpy.clip(numpy.count_nonzero(self.X==self.D_I), a_min=0, a_max=self.numNodes)
        self.numR[self.tidx]     = numpy.clip(numpy.count_nonzero(self.X==self.R), a_min=0, a_max=self.numNodes)
        self.numF[self.tidx]     = numpy.clip(numpy.count_nonzero(self.X==self.F), a_min=0, a_max=self.numNodes)
        self.N[self.tidx]        = numpy.clip((self.numS[0] + self.numE[0] + self.numI[0] + self.numD_E[0] + self.numD_I[0] + self.numR[0]), a_min=0, a_max=self.numNodes)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Terminate if tmax reached or num infectious and num exposed is 0:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if(self.t >= self.tmax or (self.numI[self.tidx]<1 and self.numE[self.tidx]<1 and self.numD_E[self.tidx]<1 and self.numD_I[self.tidx]<1)):
            self.finalize_data_series()
            return False

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        return True


#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def run(self, T, checkpoints=None, print_interval=10, verbose=False):
        if(T>0):
            self.tmax += T
        else:
            return False

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Pre-process checkpoint values:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if(checkpoints):
            numCheckpoints = len(checkpoints['t'])
            paramNames = ['G', 'beta', 'sigma', 'gamma', 'xi', 'mu_I', 'mu_0', 'nu', 'p',
                          'Q', 'beta_D', 'sigma_D', 'gamma_D', 'mu_D', 'q',
                          'theta_E', 'theta_I', 'phi_E', 'phi_I', 'psi_E', 'psi_I']
            for param in paramNames:
                # For params that don't have given checkpoint values (or bad value given), 
                # set their checkpoint values to the value they have now for all checkpoints.
                if(param not in checkpoints.keys() 
                    or not isinstance(checkpoints[param], (list, numpy.ndarray)) 
                    or len(checkpoints[param])!=numCheckpoints):
                    checkpoints[param] = [getattr(self, param)]*numCheckpoints
            checkpointIdx  = numpy.searchsorted(checkpoints['t'], self.t) # Finds 1st index in list greater than given val
            if(checkpointIdx >= numCheckpoints):
                # We are out of checkpoints, stop checking them:
                checkpoints = None 
            else:
                checkpointTime = checkpoints['t'][checkpointIdx]

        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Run the simulation loop:
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        print_reset = True
        running     = True
        while running:
            running = self.run_iteration()

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Handle checkpoints if applicable:
            if(checkpoints):
                if(self.t >= checkpointTime):
                    print("[Checkpoint: Updating parameters]")
                    # A checkpoint has been reached, update param values:
                    for param in paramNames:
                        if(param=='G'):
                            self.update_G(checkpoints[param][checkpointIdx])
                        elif(param=='Q'):
                            self.update_Q(checkpoints[param][checkpointIdx])
                        else:
                            setattr(self, param, checkpoints[param][checkpointIdx] if isinstance(checkpoints[param][checkpointIdx], (list, numpy.ndarray)) else numpy.full(fill_value=checkpoints[param][checkpointIdx], shape=(self.numNodes,1)))
                    # Update the next checkpoint time:
                    checkpointIdx  = numpy.searchsorted(checkpoints['t'], self.t) # Finds 1st index in list greater than given val
                    if(checkpointIdx >= numCheckpoints):
                        # We are out of checkpoints, stop checking them:
                        checkpoints = None 
                    else:
                        checkpointTime = checkpoints['t'][checkpointIdx]
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            if(print_interval):
                if(print_reset and (int(self.t) % print_interval == 0)):
                    print("t = %.2f" % self.t)
                    if(verbose):
                        print("\t S   = " + str(self.numS[self.tidx]))
                        print("\t E   = " + str(self.numE[self.tidx]))
                        print("\t I   = " + str(self.numI[self.tidx]))
                        print("\t D_E = " + str(self.numD_E[self.tidx]))
                        print("\t D_I = " + str(self.numD_I[self.tidx]))
                        print("\t R   = " + str(self.numR[self.tidx]))
                        print("\t F   = " + str(self.numF[self.tidx]))
                    print_reset = False
                elif(not print_reset and (int(self.t) % 10 != 0)):
                    print_reset = True

        return True


#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def plot(self, ax=None,  plot_S='line', plot_E='line', plot_I='line',plot_R='line', plot_F='line',
                            plot_D_E='line', plot_D_I='line', combine_D=True,
                            color_S='tab:green', color_E='orange', color_I='crimson', color_R='tab:blue', color_F='black',
                            color_D_E='mediumorchid', color_D_I='mediumorchid', color_reference='#E0E0E0',
                            dashed_reference_results=None, dashed_reference_label='reference', 
                            shaded_reference_results=None, shaded_reference_label='reference', 
                            vlines=[], vline_colors=[], vline_styles=[], vline_labels=[],
                            ylim=None, xlim=None, legend=True, title=None, side_title=None, plot_percentages=True):

        import matplotlib.pyplot as pyplot

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Create an Axes object if None provided:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if(not ax):
            fig, ax = pyplot.subplots()

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Prepare data series to be plotted:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Fseries     = self.numF/self.numNodes if plot_percentages else self.numF
        Eseries     = self.numE/self.numNodes if plot_percentages else self.numE
        Dseries     = (self.numD_E+self.numD_I)/self.numNodes if plot_percentages else (self.numD_E+self.numD_I)
        D_Eseries   = self.numD_E/self.numNodes if plot_percentages else self.numD_E
        D_Iseries   = self.numD_I/self.numNodes if plot_percentages else self.numD_I
        Iseries     = self.numI/self.numNodes if plot_percentages else self.numI
        Rseries     = self.numR/self.numNodes if plot_percentages else self.numR
        Sseries     = self.numS/self.numNodes if plot_percentages else self.numS 

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Draw the reference data:      
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if(dashed_reference_results):
            dashedReference_tseries  = dashed_reference_results.tseries[::int(self.numNodes/100)]
            dashedReference_IDEstack = (dashed_reference_results.numI + dashed_reference_results.numD_I + dashed_reference_results.numD_E + dashed_reference_results.numE)[::int(self.numNodes/100)] / (self.numNodes if plot_percentages else 1)
            ax.plot(dashedReference_tseries, dashedReference_IDEstack, color='#E0E0E0', linestyle='--', label='$I+D+E$ ('+dashed_reference_label+')', zorder=0)
        if(shaded_reference_results):
            shadedReference_tseries  = shaded_reference_results.tseries
            shadedReference_IDEstack = (shaded_reference_results.numI + shaded_reference_results.numD_I + shaded_reference_results.numD_E + shaded_reference_results.numE) / (self.numNodes if plot_percentages else 1)
            ax.fill_between(shaded_reference_results.tseries, shadedReference_IDEstack, 0, color='#EFEFEF', label='$I+D+E$ ('+shaded_reference_label+')', zorder=0)
            ax.plot(shaded_reference_results.tseries, shadedReference_IDEstack, color='#E0E0E0', zorder=1)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Draw the stacked variables:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        topstack = numpy.zeros_like(self.tseries)
        if(any(Fseries) and plot_F=='stacked'):
            ax.fill_between(numpy.ma.masked_where(Fseries<=0, self.tseries), numpy.ma.masked_where(Fseries<=0, topstack+Fseries), topstack, color=color_F, alpha=0.5, label='$F$', zorder=2)
            ax.plot(        numpy.ma.masked_where(Fseries<=0, self.tseries), numpy.ma.masked_where(Fseries<=0, topstack+Fseries),           color=color_F, zorder=3)
            topstack = topstack+Fseries
        if(any(Eseries) and plot_E=='stacked'):
            ax.fill_between(numpy.ma.masked_where(Eseries<=0, self.tseries), numpy.ma.masked_where(Eseries<=0, topstack+Eseries), topstack, color=color_E, alpha=0.5, label='$E$', zorder=2)
            ax.plot(        numpy.ma.masked_where(Eseries<=0, self.tseries), numpy.ma.masked_where(Eseries<=0, topstack+Eseries),           color=color_E, zorder=3)
            topstack = topstack+Eseries
        if(combine_D and plot_D_E=='stacked' and plot_D_I=='stacked'):
            ax.fill_between(numpy.ma.masked_where(Dseries<=0, self.tseries), numpy.ma.masked_where(Dseries<=0, topstack+Dseries), topstack, color=color_D_E, alpha=0.5, label='$D_{all}$', zorder=2)
            ax.plot(        numpy.ma.masked_where(Dseries<=0, self.tseries), numpy.ma.masked_where(Dseries<=0, topstack+Dseries),           color=color_D_E, zorder=3)
            topstack = topstack+Dseries
        else:
            if(any(D_Eseries) and plot_D_E=='stacked'):
                ax.fill_between(numpy.ma.masked_where(D_Eseries<=0, self.tseries), numpy.ma.masked_where(D_Eseries<=0, topstack+D_Eseries), topstack, color=color_D_E, alpha=0.5, label='$D_E$', zorder=2)
                ax.plot(        numpy.ma.masked_where(D_Eseries<=0, self.tseries), numpy.ma.masked_where(D_Eseries<=0, topstack+D_Eseries),           color=color_D_E, zorder=3)
                topstack = topstack+D_Eseries
            if(any(D_Iseries) and plot_D_I=='stacked'):
                ax.fill_between(numpy.ma.masked_where(D_Iseries<=0, self.tseries), numpy.ma.masked_where(D_Iseries<=0, topstack+D_Iseries), topstack, color=color_D_I, alpha=0.5, label='$D_I$', zorder=2)
                ax.plot(        numpy.ma.masked_where(D_Iseries<=0, self.tseries), numpy.ma.masked_where(D_Iseries<=0, topstack+D_Iseries),           color=color_D_I, zorder=3)
                topstack = topstack+D_Iseries
        if(any(Iseries) and plot_I=='stacked'):
            ax.fill_between(numpy.ma.masked_where(Iseries<=0, self.tseries), numpy.ma.masked_where(Iseries<=0, topstack+Iseries), topstack, color=color_I, alpha=0.5, label='$I$', zorder=2)
            ax.plot(        numpy.ma.masked_where(Iseries<=0, self.tseries), numpy.ma.masked_where(Iseries<=0, topstack+Iseries),           color=color_I, zorder=3)
            topstack = topstack+Iseries
        if(any(Rseries) and plot_R=='stacked'):
            ax.fill_between(numpy.ma.masked_where(Rseries<=0, self.tseries), numpy.ma.masked_where(Rseries<=0, topstack+Rseries), topstack, color=color_R, alpha=0.5, label='$R$', zorder=2)
            ax.plot(        numpy.ma.masked_where(Rseries<=0, self.tseries), numpy.ma.masked_where(Rseries<=0, topstack+Rseries),           color=color_R, zorder=3)
            topstack = topstack+Rseries
        if(any(Sseries) and plot_S=='stacked'):
            ax.fill_between(numpy.ma.masked_where(Sseries<=0, self.tseries), numpy.ma.masked_where(Sseries<=0, topstack+Sseries), topstack, color=color_S, alpha=0.5, label='$S$', zorder=2)
            ax.plot(        numpy.ma.masked_where(Sseries<=0, self.tseries), numpy.ma.masked_where(Sseries<=0, topstack+Sseries),           color=color_S, zorder=3)
            topstack = topstack+Sseries
        

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Draw the shaded variables:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if(any(Fseries) and plot_F=='shaded'):
            ax.fill_between(numpy.ma.masked_where(Fseries<=0, self.tseries), numpy.ma.masked_where(Fseries<=0, Fseries), 0, color=color_F, alpha=0.5, label='$F$', zorder=4)
            ax.plot(        numpy.ma.masked_where(Fseries<=0, self.tseries), numpy.ma.masked_where(Fseries<=0, Fseries),    color=color_F, zorder=5)
        if(any(Eseries) and plot_E=='shaded'):
            ax.fill_between(numpy.ma.masked_where(Eseries<=0, self.tseries), numpy.ma.masked_where(Eseries<=0, Eseries), 0, color=color_E, alpha=0.5, label='$E$', zorder=4)
            ax.plot(        numpy.ma.masked_where(Eseries<=0, self.tseries), numpy.ma.masked_where(Eseries<=0, Eseries),    color=color_E, zorder=5)
        if(combine_D and (any(Dseries) and plot_D_E=='shaded' and plot_D_E=='shaded')):
            ax.fill_between(numpy.ma.masked_where(Dseries<=0, self.tseries), numpy.ma.masked_where(Dseries<=0, Dseries), 0, color=color_D_E, alpha=0.5, label='$D_{all}$', zorder=4)
            ax.plot(        numpy.ma.masked_where(Dseries<=0, self.tseries), numpy.ma.masked_where(Dseries<=0, Dseries),    color=color_D_E, zorder=5)
        else:
            if(any(D_Eseries) and plot_D_E=='shaded'):
                ax.fill_between(numpy.ma.masked_where(D_Eseries<=0, self.tseries), numpy.ma.masked_where(D_Eseries<=0, D_Eseries), 0, color=color_D_E, alpha=0.5, label='$D_E$', zorder=4)
                ax.plot(        numpy.ma.masked_where(D_Eseries<=0, self.tseries), numpy.ma.masked_where(D_Eseries<=0, D_Eseries),    color=color_D_E, zorder=5)
            if(any(D_Iseries) and plot_D_I=='shaded'):
                ax.fill_between(numpy.ma.masked_where(D_Iseries<=0, self.tseries), numpy.ma.masked_where(D_Iseries<=0, D_Iseries), 0, color=color_D_I, alpha=0.5, label='$D_I$', zorder=4)
                ax.plot(        numpy.ma.masked_where(D_Iseries<=0, self.tseries), numpy.ma.masked_where(D_Iseries<=0, D_Iseries),    color=color_D_I, zorder=5)
        if(any(Iseries) and plot_I=='shaded'):
            ax.fill_between(numpy.ma.masked_where(Iseries<=0, self.tseries), numpy.ma.masked_where(Iseries<=0, Iseries), 0, color=color_I, alpha=0.5, label='$I$', zorder=4)
            ax.plot(        numpy.ma.masked_where(Iseries<=0, self.tseries), numpy.ma.masked_where(Iseries<=0, Iseries),    color=color_I, zorder=5)
        if(any(Sseries) and plot_S=='shaded'):
            ax.fill_between(numpy.ma.masked_where(Sseries<=0, self.tseries), numpy.ma.masked_where(Sseries<=0, Sseries), 0, color=color_S, alpha=0.5, label='$S$', zorder=4)
            ax.plot(        numpy.ma.masked_where(Sseries<=0, self.tseries), numpy.ma.masked_where(Sseries<=0, Sseries),    color=color_S, zorder=5)
        if(any(Rseries) and plot_R=='shaded'):
            ax.fill_between(numpy.ma.masked_where(Rseries<=0, self.tseries), numpy.ma.masked_where(Rseries<=0, Rseries), 0, color=color_R, alpha=0.5, label='$R$', zorder=4)
            ax.plot(        numpy.ma.masked_where(Rseries<=0, self.tseries), numpy.ma.masked_where(Rseries<=0, Rseries),    color=color_R, zorder=5)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Draw the line variables:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~        
        if(any(Fseries) and plot_F=='line'):
            ax.plot(numpy.ma.masked_where(Fseries<=0, self.tseries), numpy.ma.masked_where(Fseries<=0, Fseries), color=color_F, label='$F$', zorder=6)
        if(any(Eseries) and plot_E=='line'):
            ax.plot(numpy.ma.masked_where(Eseries<=0, self.tseries), numpy.ma.masked_where(Eseries<=0, Eseries), color=color_E, label='$E$', zorder=6)
        if(combine_D and (any(Dseries) and plot_D_E=='line' and plot_D_E=='line')):
            ax.plot(numpy.ma.masked_where(Dseries<=0, self.tseries), numpy.ma.masked_where(Dseries<=0, Dseries), color=color_D_E, label='$D_{all}$', zorder=6)
        else:
            if(any(D_Eseries) and plot_D_E=='line'):
                ax.plot(numpy.ma.masked_where(D_Eseries<=0, self.tseries), numpy.ma.masked_where(D_Eseries<=0, D_Eseries), color=color_D_E, label='$D_E$', zorder=6)
            if(any(D_Iseries) and plot_D_I=='line'):
                ax.plot(numpy.ma.masked_where(D_Iseries<=0, self.tseries), numpy.ma.masked_where(D_Iseries<=0, D_Iseries), color=color_D_I, label='$D_I$', zorder=6)
        if(any(Iseries) and plot_I=='line'):
            ax.plot(numpy.ma.masked_where(Iseries<=0, self.tseries), numpy.ma.masked_where(Iseries<=0, Iseries), color=color_I, label='$I$', zorder=6)
        if(any(Sseries) and plot_S=='line'):
            ax.plot(numpy.ma.masked_where(Sseries<=0, self.tseries), numpy.ma.masked_where(Sseries<=0, Sseries), color=color_S, label='$S$', zorder=6)
        if(any(Rseries) and plot_R=='line'):
            ax.plot(numpy.ma.masked_where(Rseries<=0, self.tseries), numpy.ma.masked_where(Rseries<=0, Rseries), color=color_R, label='$R$', zorder=6)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Draw the vertical line annotations:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if(len(vlines)>0 and len(vline_colors)==0):
            vline_colors = ['gray']*len(vlines)
        if(len(vlines)>0 and len(vline_labels)==0):
            vline_labels = [None]*len(vlines)
        if(len(vlines)>0 and len(vline_styles)==0):
            vline_styles = [':']*len(vlines)
        for vline_x, vline_color, vline_style, vline_label in zip(vlines, vline_colors, vline_styles, vline_labels):
            if(vline_x is not None):
                ax.axvline(x=vline_x, color=vline_color, linestyle=vline_style, alpha=1, label=vline_label)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Draw the plot labels:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ax.set_xlabel('days')
        ax.set_ylabel('percent of population' if plot_percentages else 'number of individuals')
        ax.set_xlim(0, (max(self.tseries) if not xlim else xlim))
        ax.set_ylim(0, ylim)
        if(plot_percentages):
            ax.set_yticklabels(['{:,.0%}'.format(y) for y in ax.get_yticks()])
        if(legend):
            legend_handles, legend_labels = ax.get_legend_handles_labels()
            ax.legend(legend_handles[::-1], legend_labels[::-1], loc='upper right', facecolor='white', edgecolor='none', framealpha=0.9, prop={'size': 8})
        if(title):
            ax.set_title(title, size=12)
        if(side_title):
            ax.annotate(side_title, (0, 0.5), xytext=(-45, 0), ha='right', va='center',
                size=12, rotation=90, xycoords='axes fraction', textcoords='offset points')
       
        return ax


#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def figure_basic(self, plot_S='line', plot_E='line', plot_I='line',plot_R='line', plot_F='line',
                        plot_D_E='line', plot_D_I='line', combine_D=True,
                        color_S='tab:green', color_E='orange', color_I='crimson', color_R='tab:blue', color_F='black',
                        color_D_E='mediumorchid', color_D_I='mediumorchid', color_reference='#E0E0E0',
                        dashed_reference_results=None, dashed_reference_label='reference', 
                        shaded_reference_results=None, shaded_reference_label='reference', 
                        vlines=[], vline_colors=[], vline_styles=[], vline_labels=[],
                        ylim=None, xlim=None, legend=True, title=None, side_title=None, plot_percentages=True,
                        figsize=(12,8), use_seaborn=True):

        import matplotlib.pyplot as pyplot

        fig, ax = pyplot.subplots(figsize=figsize)

        if(use_seaborn):
            import seaborn
            seaborn.set_style('ticks')
            seaborn.despine()

        self.plot(ax=ax, plot_S=plot_S, plot_E=plot_E, plot_I=plot_I,plot_R=plot_R, plot_F=plot_F,
                        plot_D_E=plot_D_E, plot_D_I=plot_D_I, combine_D=combine_D,
                        color_S=color_S, color_E=color_E, color_I=color_I, color_R=color_R, color_F=color_F,
                        color_D_E=color_D_E, color_D_I=color_D_I, color_reference=color_reference,
                        dashed_reference_results=dashed_reference_results, dashed_reference_label=dashed_reference_label, 
                        shaded_reference_results=shaded_reference_results, shaded_reference_label=shaded_reference_label, 
                        vlines=vlines, vline_colors=vline_colors, vline_styles=vline_styles, vline_labels=vline_labels,
                        ylim=ylim, xlim=xlim, legend=legend, title=title, side_title=side_title, plot_percentages=plot_percentages)

        pyplot.show()


#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def figure_infections(self, plot_S=False, plot_E='stacked', plot_I='stacked',plot_R=False, plot_F=False,
                            plot_D_E='stacked', plot_D_I='stacked', combine_D=True,
                            color_S='tab:green', color_E='orange', color_I='crimson', color_R='tab:blue', color_F='black',
                            color_D_E='mediumorchid', color_D_I='mediumorchid', color_reference='#E0E0E0',
                            dashed_reference_results=None, dashed_reference_label='reference', 
                            shaded_reference_results=None, shaded_reference_label='reference', 
                            vlines=[], vline_colors=[], vline_styles=[], vline_labels=[],
                            ylim=None, xlim=None, legend=True, title=None, side_title=None, plot_percentages=True,
                            figsize=(12,8), use_seaborn=True):

        import matplotlib.pyplot as pyplot

        fig, ax = pyplot.subplots(figsize=figsize)

        if(use_seaborn):
            import seaborn
            seaborn.set_style('ticks')
            seaborn.despine()

        self.plot(ax=ax, plot_S=plot_S, plot_E=plot_E, plot_I=plot_I,plot_R=plot_R, plot_F=plot_F,
                        plot_D_E=plot_D_E, plot_D_I=plot_D_I, combine_D=combine_D,
                        color_S=color_S, color_E=color_E, color_I=color_I, color_R=color_R, color_F=color_F,
                        color_D_E=color_D_E, color_D_I=color_D_I, color_reference=color_reference,
                        dashed_reference_results=dashed_reference_results, dashed_reference_label=dashed_reference_label, 
                        shaded_reference_results=shaded_reference_results, shaded_reference_label=shaded_reference_label, 
                        vlines=vlines, vline_colors=vline_colors, vline_styles=vline_styles, vline_labels=vline_labels, 
                        ylim=ylim, xlim=xlim, legend=legend, title=title, side_title=side_title, plot_percentages=plot_percentages)

        pyplot.show()


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Define a custom method for generating 
# power-law-like graphs with exponential tails 
# both above and below the degree mean and  
# where the mean degree be easily down-shifted
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def custom_exponential_graph(base_graph=None, scale=100, min_num_edges=0, m=9, n=None):
    # Generate a random preferential attachment power law graph as a starting point.
    # By the way this graph is constructed, it is expected to have 1 connected component.
    # Every node is added along with m=8 edges, so the min degree is m=8.
    if(base_graph):
        graph = base_graph.copy()
    else:
        assert(n is not None), "Argument n (number of nodes) must be provided when no base graph is given."
        graph = networkx.barabasi_albert_graph(n=n, m=m)

    # To get a graph with power-law-esque properties but without the fixed minimum degree,
    # We modify the graph by probabilistically dropping some edges from each node. 
    for node in graph:
        neighbors = graph[node].keys()
        quarantineEdgeNum = int( max(min(numpy.random.exponential(scale=scale, size=1), len(neighbors)), min_num_edges) )
        quarantineKeepNeighbors = numpy.random.choice(neighbors, size=quarantineEdgeNum, replace=False)
        for neighbor in neighbors:
            if(neighbor not in quarantineKeepNeighbors):
                graph.remove_edge(node, neighbor)
    
    return graph

