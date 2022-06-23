from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import networkx
import numpy
import scipy

from .base_plotable_model import BasePlotableModel


class SEIRSNetworkModel(BasePlotableModel):
    """
    A class to simulate the SEIRS Stochastic Network Model
    ======================================================
    Params:
            G               Network adjacency matrix (numpy array) or Networkx graph object.
            beta            Rate of transmission (global interactions)
            beta_local      Rate(s) of transmission between adjacent individuals (optional)
            sigma           Rate of progression to infectious state (inverse of latent period)
            gamma           Rate of recovery (inverse of symptomatic infectious period)
            mu_I            Rate of infection-related death
            xi              Rate of re-susceptibility (upon recovery)
            mu_0            Rate of baseline death
            nu              Rate of baseline birth
            p               Probability of individuals interacting with global population

            G_Q             Quarantine adjacency matrix (numpy array) or Networkx graph object.
            beta_Q          Rate of transmission for isolated individuals (global interactions)
            beta_Q_local    Rate(s) of transmission (exposure) for adjacent isolated individuals (optional)
            sigma_Q         Rate of progression to infectious state for isolated individuals
            gamma_Q         Rate of recovery for isolated individuals
            mu_Q            Rate of infection-related death for isolated individuals
            q               Probability of isolated individuals interacting with global population
            isolation_time  Time to remain in isolation upon positive test, self-isolation, etc.

            theta_E         Rate of random testing for exposed individuals
            theta_I         Rate of random testing for infectious individuals
            phi_E           Rate of testing when a close contact has tested positive for exposed individuals
            phi_I           Rate of testing when a close contact has tested positive for infectious  individuals
            psi_E           Probability of positive test for exposed individuals
            psi_I           Probability of positive test for infectious  individuals

            initE           Initial number of exposed individuals
            initI           Initial number of infectious individuals
            initR           Initial number of recovered individuals
            initF           Initial number of infection-related fatalities
            initQ_S         Initial number of isolated susceptible individuals
            initQ_E         Initial number of isolated exposed individuals
            initQ_I         Initial number of isolated infectious individuals
            initQ_R         Initial number of isolated recovered individuals
                            (all remaining nodes initialized susceptible)
    """

    plotting_number_property = 'numNodes'
    """Property to access the number to base plotting on."""

    def __init__(self, G, beta, sigma, gamma,
                    mu_I=0, alpha=1.0, xi=0, mu_0=0, nu=0, f=0, p=0,
                    beta_local=None, beta_pairwise_mode='infected', delta=None, delta_pairwise_mode=None,
                    G_Q=None, beta_Q=None, beta_Q_local=None, sigma_Q=None, gamma_Q=None, mu_Q=None, alpha_Q=None, delta_Q=None,
                    theta_E=0, theta_I=0, phi_E=0, phi_I=0, psi_E=1, psi_I=1, q=0, isolation_time=14,
                    initE=0, initI=0, initR=0, initF=0, initQ_E=0, initQ_I=0,
                    transition_mode='exponential_rates', node_groups=None, store_Xseries=False, seed=None):

        if seed is not None:
            numpy.random.seed(seed)
            self.seed = seed

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Model Parameters:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.parameters = { 'G':G, 'G_Q':G_Q,
                            'beta':beta, 'sigma':sigma, 'gamma':gamma, 'mu_I':mu_I,
                            'xi':xi, 'mu_0':mu_0, 'nu':nu, 'f':f, 'p':p,
                            'beta_local':beta_local, 'beta_pairwise_mode':beta_pairwise_mode,
                            'alpha':alpha, 'delta':delta, 'delta_pairwise_mode':delta_pairwise_mode,
                            'beta_Q':beta_Q, 'beta_Q_local':beta_Q_local, 'sigma_Q':sigma_Q, 'gamma_Q':gamma_Q, 'mu_Q':mu_Q,
                            'alpha_Q':alpha_Q, 'delta_Q':delta_Q,
                            'theta_E':theta_E, 'theta_I':theta_I, 'phi_E':phi_E, 'phi_I':phi_I, 'psi_E':psi_E, 'psi_I':psi_I,
                            'q':q, 'isolation_time':isolation_time,
                            'initE':initE, 'initI':initI, 'initR':initR, 'initF':initF,
                            'initQ_E':initQ_E, 'initQ_I':initQ_I }
        self.update_parameters()

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Each node can undergo 4-6 transitions (sans vitality/re-susceptibility returns to S state),
        # so there are ~numNodes*6 events/timesteps expected; initialize numNodes*6 timestep slots to start
        # (will be expanded during run if needed for some reason)
        self.tseries    = numpy.zeros(6*self.numNodes)
        self.numS       = numpy.zeros(6*self.numNodes)
        self.numE       = numpy.zeros(6*self.numNodes)
        self.numI       = numpy.zeros(6*self.numNodes)
        self.numR       = numpy.zeros(6*self.numNodes)
        self.numF       = numpy.zeros(6*self.numNodes)
        self.numQ_E     = numpy.zeros(6*self.numNodes)
        self.numQ_I     = numpy.zeros(6*self.numNodes)
        self.N          = numpy.zeros(6*self.numNodes)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize Timekeeping:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.t          = 0
        self.tmax       = 0 # will be set when run() is called
        self.tidx       = 0
        self.tseries[0] = 0

        # Vectors holding the time that each node has been in a given state or in isolation:
        self.timer_state     = numpy.zeros((self.numNodes,1))
        self.timer_isolation = numpy.zeros(self.numNodes)
        self.isolationTime   = isolation_time

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize Counts of individuals with each state:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.numE[0]        = int(initE)
        self.numI[0]        = int(initI)
        self.numR[0]        = int(initR)
        self.numF[0]        = int(initF)
        self.numQ_E[0]      = int(initQ_E)
        self.numQ_I[0]      = int(initQ_I)
        self.numS[0]        = (self.numNodes - self.numE[0] - self.numI[0] - self.numR[0] - self.numQ_E[0] - self.numQ_I[0] - self.numF[0])
        self.N[0]           = self.numNodes - self.numF[0]

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Node states:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.S          = 1
        self.E          = 2
        self.I          = 3
        self.R          = 4
        self.F          = 5
        self.Q_E        = 6
        self.Q_I        = 7

        self.X = numpy.array( [self.S]*int(self.numS[0]) + [self.E]*int(self.numE[0]) + [self.I]*int(self.numI[0])
                               + [self.R]*int(self.numR[0]) + [self.F]*int(self.numF[0])
                               + [self.Q_E]*int(self.numQ_E[0]) + [self.Q_I]*int(self.numQ_I[0])
                            ).reshape((self.numNodes,1))
        numpy.random.shuffle(self.X)

        self.store_Xseries = store_Xseries
        if store_Xseries:
            self.Xseries        = numpy.zeros(shape=(6*self.numNodes, self.numNodes), dtype='uint8')
            self.Xseries[0,:]   = self.X.T

        self.transitions =  {
                                'StoE': {'currentState':self.S, 'newState':self.E},
                                'EtoI': {'currentState':self.E, 'newState':self.I},
                                'ItoR': {'currentState':self.I, 'newState':self.R},
                                'ItoF': {'currentState':self.I, 'newState':self.F},
                                'RtoS': {'currentState':self.R, 'newState':self.S},
                                'EtoQE': {'currentState':self.E, 'newState':self.Q_E},
                                'ItoQI': {'currentState':self.I, 'newState':self.Q_I},
                                'QEtoQI': {'currentState':self.Q_E, 'newState':self.Q_I},
                                'QItoR': {'currentState':self.Q_I, 'newState':self.R},
                                'QItoF': {'currentState':self.Q_I, 'newState':self.F},
                                '_toS': {'currentState':True, 'newState':self.S},
                            }

        self.transition_mode = transition_mode

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize other node metadata:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.tested      = numpy.array([False]*self.numNodes).reshape((self.numNodes,1))
        self.positive    = numpy.array([False]*self.numNodes).reshape((self.numNodes,1))
        self.numTested   = numpy.zeros(6*self.numNodes)
        self.numPositive = numpy.zeros(6*self.numNodes)

        self.testedInCurrentState = numpy.array([False]*self.numNodes).reshape((self.numNodes,1))

        self.infectionsLog = []

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize node subgroup data series:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.nodeGroupData = None
        if node_groups:
            self.nodeGroupData = {}
            for groupName, nodeList in node_groups.items():
                self.nodeGroupData[groupName] = {'nodes':   numpy.array(nodeList),
                                                 'mask':    numpy.isin(range(self.numNodes), nodeList).reshape((self.numNodes,1))}
                self.nodeGroupData[groupName]['numS']           = numpy.zeros(6*self.numNodes)
                self.nodeGroupData[groupName]['numE']           = numpy.zeros(6*self.numNodes)
                self.nodeGroupData[groupName]['numI']           = numpy.zeros(6*self.numNodes)
                self.nodeGroupData[groupName]['numR']           = numpy.zeros(6*self.numNodes)
                self.nodeGroupData[groupName]['numF']           = numpy.zeros(6*self.numNodes)
                self.nodeGroupData[groupName]['numQ_E']         = numpy.zeros(6*self.numNodes)
                self.nodeGroupData[groupName]['numQ_I']         = numpy.zeros(6*self.numNodes)
                self.nodeGroupData[groupName]['N']              = numpy.zeros(6*self.numNodes)
                self.nodeGroupData[groupName]['numPositive']    = numpy.zeros(6*self.numNodes)
                self.nodeGroupData[groupName]['numTested']      = numpy.zeros(6*self.numNodes)
                self.nodeGroupData[groupName]['numS'][0]        = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.S)
                self.nodeGroupData[groupName]['numE'][0]        = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.E)
                self.nodeGroupData[groupName]['numI'][0]        = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.I)
                self.nodeGroupData[groupName]['numR'][0]        = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.R)
                self.nodeGroupData[groupName]['numF'][0]        = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.F)
                self.nodeGroupData[groupName]['numQ_E'][0]      = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.Q_E)
                self.nodeGroupData[groupName]['numQ_I'][0]      = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.Q_I)
                self.nodeGroupData[groupName]['N'][0]           = self.numNodes - self.numF[0]


#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def update_parameters(self):

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Model graphs:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.G = self.parameters['G']
        # Adjacency matrix:
        if type(self.G)==numpy.ndarray:
            self.A = scipy.sparse.csr_matrix(self.G)
        elif type(self.G)==networkx.classes.graph.Graph:
            self.A = networkx.adj_matrix(self.G) # adj_matrix gives scipy.sparse csr_matrix
        else:
            raise BaseException("Input an adjacency matrix or networkx object only.")
        self.numNodes   = int(self.A.shape[1])
        self.degree     = numpy.asarray(self.node_degrees(self.A)).astype(float)
        #----------------------------------------
        if self.parameters['G_Q'] is None:
            self.G_Q = self.G # If no Q graph is provided, use G in its place
        else:
            self.G_Q = self.parameters['G_Q']
        # Quarantine Adjacency matrix:
        if type(self.G_Q)==numpy.ndarray:
            self.A_Q = scipy.sparse.csr_matrix(self.G_Q)
        elif type(self.G_Q)==networkx.classes.graph.Graph:
            self.A_Q = networkx.adj_matrix(self.G_Q) # adj_matrix gives scipy.sparse csr_matrix
        else:
            raise BaseException("Input an adjacency matrix or networkx object only.")
        self.numNodes_Q   = int(self.A_Q.shape[1])
        self.degree_Q     = numpy.asarray(self.node_degrees(self.A_Q)).astype(float)
        #----------------------------------------
        assert(self.numNodes == self.numNodes_Q), "The normal and quarantine adjacency graphs must be of the same size."

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Model parameters:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.beta           = numpy.array(self.parameters['beta']).reshape((self.numNodes, 1))          if isinstance(self.parameters['beta'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['beta'], shape=(self.numNodes,1))
        self.sigma          = numpy.array(self.parameters['sigma']).reshape((self.numNodes, 1))         if isinstance(self.parameters['sigma'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['sigma'], shape=(self.numNodes,1))
        self.gamma          = numpy.array(self.parameters['gamma']).reshape((self.numNodes, 1))         if isinstance(self.parameters['gamma'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['gamma'], shape=(self.numNodes,1))
        self.mu_I           = numpy.array(self.parameters['mu_I']).reshape((self.numNodes, 1))          if isinstance(self.parameters['mu_I'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['mu_I'], shape=(self.numNodes,1))
        self.alpha          = numpy.array(self.parameters['alpha']).reshape((self.numNodes, 1))         if isinstance(self.parameters['alpha'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['alpha'], shape=(self.numNodes,1))
        self.xi             = numpy.array(self.parameters['xi']).reshape((self.numNodes, 1))            if isinstance(self.parameters['xi'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['xi'], shape=(self.numNodes,1))
        self.mu_0           = numpy.array(self.parameters['mu_0']).reshape((self.numNodes, 1))          if isinstance(self.parameters['mu_0'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['mu_0'], shape=(self.numNodes,1))
        self.nu             = numpy.array(self.parameters['nu']).reshape((self.numNodes, 1))            if isinstance(self.parameters['nu'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['nu'], shape=(self.numNodes,1))
        self.f              = numpy.array(self.parameters['f']).reshape((self.numNodes, 1))             if isinstance(self.parameters['f'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['f'], shape=(self.numNodes,1))
        self.p              = numpy.array(self.parameters['p']).reshape((self.numNodes, 1))             if isinstance(self.parameters['p'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['p'], shape=(self.numNodes,1))

        self.rand_f = numpy.random.rand(self.f.shape[0], self.f.shape[1])

        #----------------------------------------
        # Testing-related parameters:
        #----------------------------------------
        self.beta_Q         = (numpy.array(self.parameters['beta_Q']).reshape((self.numNodes, 1))       if isinstance(self.parameters['beta_Q'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['beta_Q'], shape=(self.numNodes,1))) if self.parameters['beta_Q'] is not None else self.beta
        self.sigma_Q        = (numpy.array(self.parameters['sigma_Q']).reshape((self.numNodes, 1))      if isinstance(self.parameters['sigma_Q'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['sigma_Q'], shape=(self.numNodes,1))) if self.parameters['sigma_Q'] is not None else self.sigma
        self.gamma_Q        = (numpy.array(self.parameters['gamma_Q']).reshape((self.numNodes, 1))  if isinstance(self.parameters['gamma_Q'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['gamma_Q'], shape=(self.numNodes,1))) if self.parameters['gamma_Q'] is not None else self.gamma
        self.mu_Q           = (numpy.array(self.parameters['mu_Q']).reshape((self.numNodes, 1))  if isinstance(self.parameters['mu_Q'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['mu_Q'], shape=(self.numNodes,1))) if self.parameters['mu_Q'] is not None else self.mu_I
        self.alpha_Q        = (numpy.array(self.parameters['alpha_Q']).reshape((self.numNodes, 1))      if isinstance(self.parameters['alpha_Q'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['alpha_Q'], shape=(self.numNodes,1))) if self.parameters['alpha_Q'] is not None else self.alpha
        self.theta_E        = numpy.array(self.parameters['theta_E']).reshape((self.numNodes, 1))       if isinstance(self.parameters['theta_E'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['theta_E'], shape=(self.numNodes,1))
        self.theta_I        = numpy.array(self.parameters['theta_I']).reshape((self.numNodes, 1))     if isinstance(self.parameters['theta_I'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['theta_I'], shape=(self.numNodes,1))
        self.phi_E          = numpy.array(self.parameters['phi_E']).reshape((self.numNodes, 1))         if isinstance(self.parameters['phi_E'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['phi_E'], shape=(self.numNodes,1))
        self.phi_I          = numpy.array(self.parameters['phi_I']).reshape((self.numNodes, 1))       if isinstance(self.parameters['phi_I'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['phi_I'], shape=(self.numNodes,1))
        self.psi_E          = numpy.array(self.parameters['psi_E']).reshape((self.numNodes, 1))           if isinstance(self.parameters['psi_E'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['psi_E'], shape=(self.numNodes,1))
        self.psi_I          = numpy.array(self.parameters['psi_I']).reshape((self.numNodes, 1))         if isinstance(self.parameters['psi_I'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['psi_I'], shape=(self.numNodes,1))
        self.q              = numpy.array(self.parameters['q']).reshape((self.numNodes, 1))             if isinstance(self.parameters['q'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['q'], shape=(self.numNodes,1))

        #----------------------------------------

        self.beta_pairwise_mode = self.parameters['beta_pairwise_mode']

        #----------------------------------------
        # Global transmission parameters:
        #----------------------------------------
        if (self.beta_pairwise_mode == 'infected' or self.beta_pairwise_mode is None):
            self.beta_global         = numpy.full_like(self.beta, fill_value=numpy.mean(self.beta))
            self.beta_Q_global       = numpy.full_like(self.beta_Q, fill_value=numpy.mean(self.beta_Q))
        elif self.beta_pairwise_mode == 'infectee':
            self.beta_global         = self.beta
            self.beta_Q_global       = self.beta_Q
        elif self.beta_pairwise_mode == 'min':
            self.beta_global         = numpy.minimum(self.beta, numpy.mean(beta))
            self.beta_Q_global       = numpy.minimum(self.beta_Q, numpy.mean(beta_Q))
        elif self.beta_pairwise_mode == 'max':
            self.beta_global         = numpy.maximum(self.beta, numpy.mean(beta))
            self.beta_Q_global       = numpy.maximum(self.beta_Q, numpy.mean(beta_Q))
        elif self.beta_pairwise_mode == 'mean':
            self.beta_global         = (self.beta + numpy.full_like(self.beta, fill_value=numpy.mean(self.beta)))/2
            self.beta_Q_global       = (self.beta_Q + numpy.full_like(self.beta_Q, fill_value=numpy.mean(self.beta_Q)))/2

        #----------------------------------------
        # Local transmission parameters:
        #----------------------------------------
        self.beta_local         = self.beta      if self.parameters['beta_local'] is None      else numpy.array(self.parameters['beta_local'])      if isinstance(self.parameters['beta_local'], (list, numpy.ndarray))      else numpy.full(fill_value=self.parameters['beta_local'], shape=(self.numNodes,1))
        self.beta_Q_local       = self.beta_Q    if self.parameters['beta_Q_local'] is None    else numpy.array(self.parameters['beta_Q_local'])    if isinstance(self.parameters['beta_Q_local'], (list, numpy.ndarray))    else numpy.full(fill_value=self.parameters['beta_Q_local'], shape=(self.numNodes,1))

        #----------------------------------------
        if (self.beta_local.ndim == 2 and self.beta_local.shape[0] == self.numNodes and self.beta_local.shape[1] == self.numNodes):
            self.A_beta_pairwise = self.beta_local
        elif ((self.beta_local.ndim == 1 and self.beta_local.shape[0] == self.numNodes) or (self.beta_local.ndim == 2 and (self.beta_local.shape[0] == self.numNodes or self.beta_local.shape[1] == self.numNodes))):
            self.beta_local = self.beta_local.reshape((self.numNodes,1))
            # Pre-multiply beta values by the adjacency matrix ("transmission weight connections")
            A_beta_pairwise_byInfected = scipy.sparse.csr_matrix.multiply(self.A, self.beta_local.T).tocsr()
            A_beta_pairwise_byInfectee = scipy.sparse.csr_matrix.multiply(self.A, self.beta_local).tocsr()
            #------------------------------
            # Compute the effective pairwise beta values as a function of the infected/infectee pair:
            if self.beta_pairwise_mode == 'infected':
                self.A_beta_pairwise = A_beta_pairwise_byInfected
            elif self.beta_pairwise_mode == 'infectee':
                self.A_beta_pairwise = A_beta_pairwise_byInfectee
            elif self.beta_pairwise_mode == 'min':
                self.A_beta_pairwise = scipy.sparse.csr_matrix.minimum(A_beta_pairwise_byInfected, A_beta_pairwise_byInfectee)
            elif self.beta_pairwise_mode == 'max':
                self.A_beta_pairwise = scipy.sparse.csr_matrix.maximum(A_beta_pairwise_byInfected, A_beta_pairwise_byInfectee)
            elif self.beta_pairwise_mode == 'mean' or self.beta_pairwise_mode is None:
                self.A_beta_pairwise = (A_beta_pairwise_byInfected + A_beta_pairwise_byInfectee)/2
            else:
                print("Unrecognized beta_pairwise_mode value (support for 'infected', 'infectee', 'min', 'max', and 'mean').")
        else:
            print("Invalid values given for beta_local (expected 1xN list/array or NxN 2d array)")
        #----------------------------------------
        if (self.beta_Q_local.ndim == 2 and self.beta_Q_local.shape[0] == self.numNodes and self.beta_Q_local.shape[1] == self.numNodes):
            self.A_Q_beta_Q_pairwise = self.beta_Q_local
        elif ((self.beta_Q_local.ndim == 1 and self.beta_Q_local.shape[0] == self.numNodes) or (self.beta_Q_local.ndim == 2 and (self.beta_Q_local.shape[0] == self.numNodes or self.beta_Q_local.shape[1] == self.numNodes))):
            self.beta_Q_local = self.beta_Q_local.reshape((self.numNodes,1))
            # Pre-multiply beta_Q values by the isolation adjacency matrix ("transmission weight connections")
            A_Q_beta_Q_pairwise_byInfected      = scipy.sparse.csr_matrix.multiply(self.A_Q, self.beta_Q_local.T).tocsr()
            A_Q_beta_Q_pairwise_byInfectee      = scipy.sparse.csr_matrix.multiply(self.A_Q, self.beta_Q_local).tocsr()
            #------------------------------
            # Compute the effective pairwise beta values as a function of the infected/infectee pair:
            if self.beta_pairwise_mode == 'infected':
                self.A_Q_beta_Q_pairwise = A_Q_beta_Q_pairwise_byInfected
            elif self.beta_pairwise_mode == 'infectee':
                self.A_Q_beta_Q_pairwise = A_Q_beta_Q_pairwise_byInfectee
            elif self.beta_pairwise_mode == 'min':
                self.A_Q_beta_Q_pairwise = scipy.sparse.csr_matrix.minimum(A_Q_beta_Q_pairwise_byInfected, A_Q_beta_Q_pairwise_byInfectee)
            elif self.beta_pairwise_mode == 'max':
                self.A_Q_beta_Q_pairwise = scipy.sparse.csr_matrix.maximum(A_Q_beta_Q_pairwise_byInfected, A_Q_beta_Q_pairwise_byInfectee)
            elif (self.beta_pairwise_mode == 'mean' or self.beta_pairwise_mode is None):
                self.A_Q_beta_Q_pairwise = (A_Q_beta_Q_pairwise_byInfected + A_Q_beta_Q_pairwise_byInfectee)/2
            else:
                print("Unrecognized beta_pairwise_mode value (support for 'infected', 'infectee', 'min', 'max', and 'mean').")
        else:
            print("Invalid values given for beta_Q_local (expected 1xN list/array or NxN 2d array)")
        #----------------------------------------

        #----------------------------------------
        # Degree-based transmission scaling parameters:
        #----------------------------------------
        self.delta_pairwise_mode = self.parameters['delta_pairwise_mode']
        with numpy.errstate(divide='ignore'): # ignore log(0) warning, then convert log(0) = -inf -> 0.0
            self.delta               = numpy.log(self.degree)/numpy.log(numpy.mean(self.degree))     if self.parameters['delta'] is None   else numpy.array(self.parameters['delta'])   if isinstance(self.parameters['delta'], (list, numpy.ndarray))   else numpy.full(fill_value=self.parameters['delta'], shape=(self.numNodes,1))
            self.delta_Q             = numpy.log(self.degree_Q)/numpy.log(numpy.mean(self.degree_Q)) if self.parameters['delta_Q'] is None else numpy.array(self.parameters['delta_Q']) if isinstance(self.parameters['delta_Q'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['delta_Q'], shape=(self.numNodes,1))
        self.delta[numpy.isneginf(self.delta)] = 0.0
        self.delta_Q[numpy.isneginf(self.delta_Q)] = 0.0
        #----------------------------------------
        if (self.delta.ndim == 2 and self.delta.shape[0] == self.numNodes and self.delta.shape[1] == self.numNodes):
            self.A_delta_pairwise = self.delta
        elif ((self.delta.ndim == 1 and self.delta.shape[0] == self.numNodes) or (self.delta.ndim == 2 and (self.delta.shape[0] == self.numNodes or self.delta.shape[1] == self.numNodes))):
            self.delta = self.delta.reshape((self.numNodes,1))
            # Pre-multiply delta values by the adjacency matrix ("transmission weight connections")
            A_delta_pairwise_byInfected = scipy.sparse.csr_matrix.multiply(self.A, self.delta.T).tocsr()
            A_delta_pairwise_byInfectee = scipy.sparse.csr_matrix.multiply(self.A, self.delta).tocsr()
            #------------------------------
            # Compute the effective pairwise delta values as a function of the infected/infectee pair:
            if self.delta_pairwise_mode == 'infected':
                self.A_delta_pairwise = A_delta_pairwise_byInfected
            elif self.delta_pairwise_mode == 'infectee':
                self.A_delta_pairwise = A_delta_pairwise_byInfectee
            elif self.delta_pairwise_mode == 'min':
                self.A_delta_pairwise = scipy.sparse.csr_matrix.minimum(A_delta_pairwise_byInfected, A_delta_pairwise_byInfectee)
            elif self.delta_pairwise_mode == 'max':
                self.A_delta_pairwise = scipy.sparse.csr_matrix.maximum(A_delta_pairwise_byInfected, A_delta_pairwise_byInfectee)
            elif self.delta_pairwise_mode == 'mean':
                self.A_delta_pairwise = (A_delta_pairwise_byInfected + A_delta_pairwise_byInfectee)/2
            elif self.delta_pairwise_mode is None:
                self.A_delta_pairwise = self.A
            else:
                print("Unrecognized delta_pairwise_mode value (support for 'infected', 'infectee', 'min', 'max', and 'mean').")
        else:
            print("Invalid values given for delta (expected 1xN list/array or NxN 2d array)")
        #----------------------------------------
        if (self.delta_Q.ndim == 2 and self.delta_Q.shape[0] == self.numNodes and self.delta_Q.shape[1] == self.numNodes):
            self.A_Q_delta_Q_pairwise = self.delta_Q
        elif ((self.delta_Q.ndim == 1 and self.delta_Q.shape[0] == self.numNodes) or (self.delta_Q.ndim == 2 and (self.delta_Q.shape[0] == self.numNodes or self.delta_Q.shape[1] == self.numNodes))):
            self.delta_Q = self.delta_Q.reshape((self.numNodes,1))
            # Pre-multiply delta_Q values by the isolation adjacency matrix ("transmission weight connections")
            A_Q_delta_Q_pairwise_byInfected      = scipy.sparse.csr_matrix.multiply(self.A_Q, self.delta_Q).tocsr()
            A_Q_delta_Q_pairwise_byInfectee      = scipy.sparse.csr_matrix.multiply(self.A_Q, self.delta_Q.T).tocsr()
            #------------------------------
            # Compute the effective pairwise delta values as a function of the infected/infectee pair:
            if self.delta_pairwise_mode == 'infected':
                self.A_Q_delta_Q_pairwise = A_Q_delta_Q_pairwise_byInfected
            elif self.delta_pairwise_mode == 'infectee':
                self.A_Q_delta_Q_pairwise = A_Q_delta_Q_pairwise_byInfectee
            elif self.delta_pairwise_mode == 'min':
                self.A_Q_delta_Q_pairwise = scipy.sparse.csr_matrix.minimum(A_Q_delta_Q_pairwise_byInfected, A_Q_delta_Q_pairwise_byInfectee)
            elif self.delta_pairwise_mode == 'max':
                self.A_Q_delta_Q_pairwise = scipy.sparse.csr_matrix.maximum(A_Q_delta_Q_pairwise_byInfected, A_Q_delta_Q_pairwise_byInfectee)
            elif self.delta_pairwise_mode == 'mean':
                self.A_Q_delta_Q_pairwise = (A_Q_delta_Q_pairwise_byInfected + A_Q_delta_Q_pairwise_byInfectee)/2
            elif (self.delta_pairwise_mode is None):
                self.A_Q_delta_Q_pairwise = self.A
            else:
                print("Unrecognized delta_pairwise_mode value (support for 'infected', 'infectee', 'min', 'max', and 'mean').")
        else:
            print("Invalid values given for delta_Q (expected 1xN list/array or NxN 2d array)")

        #----------------------------------------
        # Pre-calculate the pairwise delta*beta values:
        #----------------------------------------
        self.A_deltabeta          = scipy.sparse.csr_matrix.multiply(self.A_delta_pairwise, self.A_beta_pairwise)
        self.A_Q_deltabeta_Q      = scipy.sparse.csr_matrix.multiply(self.A_Q_delta_Q_pairwise, self.A_Q_beta_Q_pairwise)


#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def node_degrees(self, Amat):
        return Amat.sum(axis=0).reshape(self.numNodes,1)   # sums of adj matrix cols


#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def total_num_susceptible(self, t_idx=None):
        if t_idx is None:
            return self.numS[:]
        else:
            return self.numS[t_idx]

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def total_num_infected(self, t_idx=None):
        if t_idx is None:
            return self.numE[:] + self.numI[:] + self.numQ_E[:] + self.numQ_I[:]
        else:
            return self.numE[t_idx] + self.numI[t_idx] + self.numQ_E[t_idx] + self.numQ_I[t_idx]

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def total_num_isolated(self, t_idx=None):
        if t_idx is None:
            return self.numQ_E[:] + self.numQ_I[:]
        else:
            return self.numQ_E[t_idx] + self.numQ_I[t_idx]

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def total_num_tested(self, t_idx=None):
        if t_idx is None:
            return self.numTested[:]
        else:
            return self.numTested[t_idx]

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def total_num_positive(self, t_idx=None):
        if t_idx is None:
            return self.numPositive[:]
        else:
            return self.numPositive[t_idx]

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def total_num_recovered(self, t_idx=None):
        if t_idx is None:
            return self.numR[:]
        else:
            return self.numR[t_idx]


#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def calc_propensities(self):

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Pre-calculate matrix multiplication terms that may be used in multiple propensity calculations,
        # and check to see if their computation is necessary before doing the multiplication
        #------------------------------------

        self.transmissionTerms_I = numpy.zeros(shape=(self.numNodes,1))
        if numpy.any(self.numI[self.tidx]):
            self.transmissionTerms_I = numpy.asarray(scipy.sparse.csr_matrix.dot(self.A_deltabeta, self.X==self.I))

        #------------------------------------

        self.transmissionTerms_Q = numpy.zeros(shape=(self.numNodes,1))
        if numpy.any(self.numQ_I[self.tidx]):
            self.transmissionTerms_Q = numpy.asarray(scipy.sparse.csr_matrix.dot(self.A_Q_deltabeta_Q, self.X==self.Q_I))

        #------------------------------------

        numContacts_Q = numpy.zeros(shape=(self.numNodes,1))
        if numpy.any(self.positive) and (numpy.any(self.phi_E) or numpy.any(self.phi_I)):
            numContacts_Q = numpy.asarray(scipy.sparse.csr_matrix.dot(self.A, ((self.positive)&(self.X!=self.R)&(self.X!=self.F))))

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        propensities_StoE       = (self.alpha *
                                    (self.p*((self.beta_global*self.numI[self.tidx] + self.q*self.beta_Q_global*self.numQ_I[self.tidx])/self.N[self.tidx])
                                     + (1-self.p)*(numpy.divide(self.transmissionTerms_I, self.degree, out=numpy.zeros_like(self.degree), where=self.degree!=0)
                                                  +numpy.divide(self.transmissionTerms_Q, self.degree_Q, out=numpy.zeros_like(self.degree_Q), where=self.degree_Q!=0)))
                                  )*(self.X==self.S)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        if self.transition_mode == 'time_in_state':

            propensities_EtoI     = 1e5 * ((self.X==self.E) & numpy.greater(self.timer_state, 1/self.sigma))

            propensities_ItoR     = 1e5 * ((self.X==self.I) & numpy.greater(self.timer_state, 1/self.gamma) & numpy.greater_equal(self.rand_f, self.f))

            propensities_ItoF     = 1e5 * ((self.X==self.I) & numpy.greater(self.timer_state, 1/self.mu_I) & numpy.less(self.rand_f, self.f))

            propensities_EtoQE    = numpy.zeros_like(propensities_StoE)

            propensities_ItoQI    = numpy.zeros_like(propensities_StoE)

            propensities_QEtoQI   = 1e5 * ((self.X==self.Q_E) & numpy.greater(self.timer_state, 1/self.sigma_Q))

            propensities_QItoR    = 1e5 * ((self.X==self.Q_I) & numpy.greater(self.timer_state, 1/self.gamma_Q) & numpy.greater_equal(self.rand_f, self.f))

            propensities_QItoF    = 1e5 * ((self.X==self.Q_I) & numpy.greater(self.timer_state, 1/self.mu_Q) & numpy.less(self.rand_f, self.f))

            propensities_RtoS     = 1e5 * ((self.X==self.R) & numpy.greater(self.timer_state, 1/self.xi))

            propensities__toS     = 1e5 * ((self.X!=self.F) & numpy.greater(self.timer_state, 1/self.nu))

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        else: # exponential_rates

            propensities_EtoI        = self.sigma * (self.X==self.E)

            propensities_ItoR        = self.gamma * ((self.X==self.I) & (numpy.greater_equal(self.rand_f, self.f)))

            propensities_ItoF        = self.mu_I * ((self.X==self.I) & (numpy.less(self.rand_f, self.f)))

            propensities_EtoQE       = (self.theta_E + self.phi_E*numContacts_Q)*self.psi_E * (self.X==self.E)

            propensities_ItoQI       = (self.theta_I + self.phi_I*numContacts_Q)*self.psi_I * (self.X==self.I)

            propensities_QEtoQI      = self.sigma_Q * (self.X==self.Q_E)

            propensities_QItoR       = self.gamma_Q * ((self.X==self.Q_I) & (numpy.greater_equal(self.rand_f, self.f)))

            propensities_QItoF       = self.mu_Q * ((self.X==self.Q_I) & (numpy.less(self.rand_f, self.f)))

            propensities_RtoS        = self.xi * (self.X==self.R)

            propensities__toS        = self.nu * (self.X!=self.F)










        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        propensities = numpy.hstack([propensities_StoE, propensities_EtoI,
                                     propensities_ItoR, propensities_ItoF,
                                     propensities_EtoQE, propensities_ItoQI, propensities_QEtoQI,
                                     propensities_QItoR, propensities_QItoF,
                                     propensities_RtoS, propensities__toS])

        columns = ['StoE', 'EtoI', 'ItoR', 'ItoF', 'EtoQE', 'ItoQI', 'QEtoQI', 'QItoR', 'QItoF', 'RtoS', '_toS']

        return propensities, columns


#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def set_isolation(self, node, isolate):
        # Move this node in/out of the appropriate isolation state:
        if isolate == True:
            if self.X[node] == self.E:
                self.X[node] = self.Q_E
                self.timer_state = 0
            elif self.X[node] == self.I:
                self.X[node] = self.Q_I
                self.timer_state = 0
        elif isolate == False:
            if self.X[node] == self.Q_E:
                self.X[node] = self.E
                self.timer_state = 0
            elif self.X[node] == self.Q_I:
                self.X[node] = self.I
                self.timer_state = 0
        # Reset the isolation timer:
        self.timer_isolation[node] = 0

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def set_tested(self, node, tested):
        self.tested[node] = tested
        self.testedInCurrentState[node] = tested

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def set_positive(self, node, positive):
        self.positive[node] = positive

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def introduce_exposures(self, num_new_exposures):
        exposedNodes = numpy.random.choice(range(self.numNodes), size=num_new_exposures, replace=False)
        for exposedNode in exposedNodes:
            if self.X[exposedNode] == self.S:
                self.X[exposedNode] = self.E


#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def increase_data_series_length(self):
        self.tseries     = numpy.pad(self.tseries, [(0, 6*self.numNodes)], mode='constant', constant_values=0)
        self.numS        = numpy.pad(self.numS, [(0, 6*self.numNodes)], mode='constant', constant_values=0)
        self.numE        = numpy.pad(self.numE, [(0, 6*self.numNodes)], mode='constant', constant_values=0)
        self.numI        = numpy.pad(self.numI, [(0, 6*self.numNodes)], mode='constant', constant_values=0)
        self.numR        = numpy.pad(self.numR, [(0, 6*self.numNodes)], mode='constant', constant_values=0)
        self.numF        = numpy.pad(self.numF, [(0, 6*self.numNodes)], mode='constant', constant_values=0)
        self.numQ_E      = numpy.pad(self.numQ_E, [(0, 6*self.numNodes)], mode='constant', constant_values=0)
        self.numQ_I      = numpy.pad(self.numQ_I, [(0, 6*self.numNodes)], mode='constant', constant_values=0)
        self.N           = numpy.pad(self.N, [(0, 6*self.numNodes)], mode='constant', constant_values=0)
        self.numTested   = numpy.pad(self.numTested, [(0, 6*self.numNodes)], mode='constant', constant_values=0)
        self.numPositive = numpy.pad(self.numPositive, [(0, 6*self.numNodes)], mode='constant', constant_values=0)

        if self.store_Xseries:
            self.Xseries = numpy.pad(self.Xseries, [(0, 6*self.numNodes), (0,0)], mode='constant', constant_values=0)

        if self.nodeGroupData:
            for groupName in self.nodeGroupData:
                self.nodeGroupData[groupName]['numS']        = numpy.pad(self.nodeGroupData[groupName]['numS'], [(0, 6*self.numNodes)], mode='constant', constant_values=0)
                self.nodeGroupData[groupName]['numE']        = numpy.pad(self.nodeGroupData[groupName]['numE'], [(0, 6*self.numNodes)], mode='constant', constant_values=0)
                self.nodeGroupData[groupName]['numI']        = numpy.pad(self.nodeGroupData[groupName]['numI'], [(0, 6*self.numNodes)], mode='constant', constant_values=0)
                self.nodeGroupData[groupName]['numR']        = numpy.pad(self.nodeGroupData[groupName]['numR'], [(0, 6*self.numNodes)], mode='constant', constant_values=0)
                self.nodeGroupData[groupName]['numF']        = numpy.pad(self.nodeGroupData[groupName]['numF'], [(0, 6*self.numNodes)], mode='constant', constant_values=0)
                self.nodeGroupData[groupName]['numQ_E']      = numpy.pad(self.nodeGroupData[groupName]['numQ_E'], [(0, 6*self.numNodes)], mode='constant', constant_values=0)
                self.nodeGroupData[groupName]['numQ_I']      = numpy.pad(self.nodeGroupData[groupName]['numQ_I'], [(0, 6*self.numNodes)], mode='constant', constant_values=0)
                self.nodeGroupData[groupName]['N']           = numpy.pad(self.nodeGroupData[groupName]['N'], [(0, 6*self.numNodes)], mode='constant', constant_values=0)
                self.nodeGroupData[groupName]['numTested']   = numpy.pad(self.nodeGroupData[groupName]['numTested'], [(0, 6*self.numNodes)], mode='constant', constant_values=0)
                self.nodeGroupData[groupName]['numPositive'] = numpy.pad(self.nodeGroupData[groupName]['numPositive'], [(0, 6*self.numNodes)], mode='constant', constant_values=0)

        return None

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def finalize_data_series(self):
        self.tseries     = numpy.array(self.tseries, dtype=float)[:self.tidx+1]
        self.numS        = numpy.array(self.numS, dtype=float)[:self.tidx+1]
        self.numE        = numpy.array(self.numE, dtype=float)[:self.tidx+1]
        self.numI        = numpy.array(self.numI, dtype=float)[:self.tidx+1]
        self.numR        = numpy.array(self.numR, dtype=float)[:self.tidx+1]
        self.numF        = numpy.array(self.numF, dtype=float)[:self.tidx+1]
        self.numQ_E      = numpy.array(self.numQ_E, dtype=float)[:self.tidx+1]
        self.numQ_I      = numpy.array(self.numQ_I, dtype=float)[:self.tidx+1]
        self.N           = numpy.array(self.N, dtype=float)[:self.tidx+1]
        self.numTested   = numpy.array(self.numTested, dtype=float)[:self.tidx+1]
        self.numPositive = numpy.array(self.numPositive, dtype=float)[:self.tidx+1]

        if self.store_Xseries:
            self.Xseries = self.Xseries[:self.tidx+1, :]

        if self.nodeGroupData:
            for groupName in self.nodeGroupData:
                self.nodeGroupData[groupName]['numS']        = numpy.array(self.nodeGroupData[groupName]['numS'], dtype=float)[:self.tidx+1]
                self.nodeGroupData[groupName]['numE']        = numpy.array(self.nodeGroupData[groupName]['numE'], dtype=float)[:self.tidx+1]
                self.nodeGroupData[groupName]['numI']        = numpy.array(self.nodeGroupData[groupName]['numI'], dtype=float)[:self.tidx+1]
                self.nodeGroupData[groupName]['numR']        = numpy.array(self.nodeGroupData[groupName]['numR'], dtype=float)[:self.tidx+1]
                self.nodeGroupData[groupName]['numF']        = numpy.array(self.nodeGroupData[groupName]['numF'], dtype=float)[:self.tidx+1]
                self.nodeGroupData[groupName]['numQ_E']      = numpy.array(self.nodeGroupData[groupName]['numQ_E'], dtype=float)[:self.tidx+1]
                self.nodeGroupData[groupName]['numQ_I']      = numpy.array(self.nodeGroupData[groupName]['numQ_I'], dtype=float)[:self.tidx+1]
                self.nodeGroupData[groupName]['N']           = numpy.array(self.nodeGroupData[groupName]['N'], dtype=float)[:self.tidx+1]
                self.nodeGroupData[groupName]['numTested']   = numpy.array(self.nodeGroupData[groupName]['numTested'], dtype=float)[:self.tidx+1]
                self.nodeGroupData[groupName]['numPositive'] = numpy.array(self.nodeGroupData[groupName]['numPositive'], dtype=float)[:self.tidx+1]

        return None

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def run_iteration(self):

        if self.tidx >= len(self.tseries) - 1:
            # Room has run out in the timeseries storage arrays; double the size of these arrays:
            self.increase_data_series_length()

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Generate 2 random numbers uniformly distributed in (0,1)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        r1 = numpy.random.rand()
        r2 = numpy.random.rand()

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Calculate propensities
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        propensities, transitionTypes = self.calc_propensities()

        if propensities.sum() > 0:

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Calculate alpha
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            propensities_flat   = propensities.ravel(order='F')
            cumsum              = propensities_flat.cumsum()
            alpha               = propensities_flat.sum()

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute the time until the next event takes place
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            tau = (1/alpha)*numpy.log(float(1/r1))
            self.t += tau
            self.timer_state += tau

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute which event takes place
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            transitionIdx   = numpy.searchsorted(cumsum,r2*alpha)
            transitionNode  = transitionIdx % self.numNodes
            transitionType  = transitionTypes[ int(transitionIdx/self.numNodes) ]

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Perform updates triggered by rate propensities:
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            assert(self.X[transitionNode] == self.transitions[transitionType]['currentState'] and self.X[transitionNode]!=self.F), "Assertion error: Node "+str(transitionNode)+" has unexpected current state "+str(self.X[transitionNode])+" given the intended transition of "+str(transitionType)+"."
            self.X[transitionNode] = self.transitions[transitionType]['newState']

            self.testedInCurrentState[transitionNode] = False

            self.timer_state = 0.0

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            # Save information about infection events when they occur:
            if transitionType == 'StoE':
                transitionNode_GNbrs  = list(self.G[transitionNode].keys())
                transitionNode_GQNbrs = list(self.G_Q[transitionNode].keys())
                self.infectionsLog.append({ 't':                            self.t,
                                            'infected_node':                transitionNode,
                                            'infection_type':               transitionType,
                                            'infected_node_degree':         self.degree[transitionNode],
                                            'local_contact_nodes':          transitionNode_GNbrs,
                                            'local_contact_node_states':    self.X[transitionNode_GNbrs].flatten(),
                                            'isolation_contact_nodes':      transitionNode_GQNbrs,
                                            'isolation_contact_node_states':self.X[transitionNode_GQNbrs].flatten() })

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            if (transitionType in ['EtoQE', 'ItoQI']):
                self.set_positive(node=transitionNode, positive=True)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        else:

            tau = 0.01
            self.t += tau
            self.timer_state += tau

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        self.tidx += 1

        self.tseries[self.tidx]     = self.t
        self.numS[self.tidx]        = numpy.clip(numpy.count_nonzero(self.X==self.S), a_min=0, a_max=self.numNodes)
        self.numE[self.tidx]        = numpy.clip(numpy.count_nonzero(self.X==self.E), a_min=0, a_max=self.numNodes)
        self.numI[self.tidx]        = numpy.clip(numpy.count_nonzero(self.X==self.I), a_min=0, a_max=self.numNodes)
        self.numF[self.tidx]        = numpy.clip(numpy.count_nonzero(self.X==self.F), a_min=0, a_max=self.numNodes)
        self.numQ_E[self.tidx]      = numpy.clip(numpy.count_nonzero(self.X==self.Q_E), a_min=0, a_max=self.numNodes)
        self.numQ_I[self.tidx]      = numpy.clip(numpy.count_nonzero(self.X==self.Q_I), a_min=0, a_max=self.numNodes)
        self.numTested[self.tidx]   = numpy.clip(numpy.count_nonzero(self.tested), a_min=0, a_max=self.numNodes)
        self.numPositive[self.tidx] = numpy.clip(numpy.count_nonzero(self.positive), a_min=0, a_max=self.numNodes)

        self.N[self.tidx]           = numpy.clip((self.numNodes - self.numF[self.tidx]), a_min=0, a_max=self.numNodes)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update testing and isolation statuses
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        isolatedNodes = numpy.argwhere((self.X==self.Q_E)|(self.X==self.Q_I))[:,0].flatten()
        self.timer_isolation[isolatedNodes] = self.timer_isolation[isolatedNodes] + tau

        nodesExitingIsolation = numpy.argwhere(self.timer_isolation >= self.isolationTime)
        for isoNode in nodesExitingIsolation:
            self.set_isolation(node=isoNode, isolate=False)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Store system states
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if self.store_Xseries:
            self.Xseries[self.tidx,:] = self.X.T

        if self.nodeGroupData:
            for groupName in self.nodeGroupData:
                self.nodeGroupData[groupName]['numS'][self.tidx]        = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.S)
                self.nodeGroupData[groupName]['numE'][self.tidx]        = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.E)
                self.nodeGroupData[groupName]['numI'][self.tidx]        = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.I)
                self.nodeGroupData[groupName]['numR'][self.tidx]        = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.R)
                self.nodeGroupData[groupName]['numF'][self.tidx]        = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.F)
                self.nodeGroupData[groupName]['numQ_E'][self.tidx]      = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.Q_E)
                self.nodeGroupData[groupName]['numQ_I'][self.tidx]      = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.Q_I)
                self.nodeGroupData[groupName]['N'][self.tidx]           = numpy.clip((self.nodeGroupData[groupName]['numS'][0] + self.nodeGroupData[groupName]['numE'][0] + self.nodeGroupData[groupName]['numI'][0] + self.nodeGroupData[groupName]['numQ_E'][0] + self.nodeGroupData[groupName]['numQ_I'][0] + self.nodeGroupData[groupName]['numR'][0]), a_min=0, a_max=self.numNodes)
                self.nodeGroupData[groupName]['numTested'][self.tidx]   = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.tested)
                self.nodeGroupData[groupName]['numPositive'][self.tidx] = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.positive)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Terminate if tmax reached or num infections is 0:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if (self.t >= self.tmax or (self.total_num_infected(self.tidx) < 1 and self.total_num_isolated(self.tidx) < 1)):
            self.finalize_data_series()
            return False

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        return True


#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def run(self, T, checkpoints=None, print_interval=10, verbose='t'):
        if T > 0:
            self.tmax += T
        else:
            return False

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Pre-process checkpoint values:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if checkpoints:
            numCheckpoints = len(checkpoints['t'])
            for chkpt_param, chkpt_values in checkpoints.items():
                assert(isinstance(chkpt_values, (list, numpy.ndarray)) and len(chkpt_values)==numCheckpoints), "Expecting a list of values with length equal to number of checkpoint times ("+str(numCheckpoints)+") for each checkpoint parameter."
            checkpointIdx  = numpy.searchsorted(checkpoints['t'], self.t) # Finds 1st index in list greater than given val
            if checkpointIdx >= numCheckpoints:
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
            if checkpoints:
                if self.t >= checkpointTime:
                    if verbose is not False:
                        print("[Checkpoint: Updating parameters]")
                    # A checkpoint has been reached, update param values:
                    for param in list(self.parameters.keys()):
                        if param in list(checkpoints.keys()):
                            self.parameters.update({param: checkpoints[param][checkpointIdx]})
                    # Update parameter data structures and scenario flags:
                    self.update_parameters()
                    # Update the next checkpoint time:
                    checkpointIdx  = numpy.searchsorted(checkpoints['t'], self.t) # Finds 1st index in list greater than given val
                    if checkpointIdx >= numCheckpoints:
                        # We are out of checkpoints, stop checking them:
                        checkpoints = None
                    else:
                        checkpointTime = checkpoints['t'][checkpointIdx]
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            if print_interval:
                if (print_reset and (int(self.t) % print_interval == 0)):
                    if verbose == "t":
                        print("t = %.2f" % self.t)
                    if verbose == True:
                        print("t = %.2f" % self.t)
                        print("\t S      = " + str(self.numS[self.tidx]))
                        print("\t E      = " + str(self.numE[self.tidx]))
                        print("\t I  = " + str(self.numI[self.tidx]))
                        print("\t R      = " + str(self.numR[self.tidx]))
                        print("\t F      = " + str(self.numF[self.tidx]))
                        print("\t Q_E    = " + str(self.numQ_E[self.tidx]))
                        print("\t Q_I  = " + str(self.numQ_I[self.tidx]))
                    print_reset = False
                elif (not print_reset and (int(self.t) % 10 != 0)):
                    print_reset = True

        return True
