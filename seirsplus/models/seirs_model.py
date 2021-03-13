from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy
import scipy.integrate

from .base_plotable_model import BasePlotableModel


class SEIRSModel(BasePlotableModel):
    """
    A class to simulate the Deterministic SEIRS Model
    ===================================================
    Params: beta    Rate of transmission (exposure)
            sigma   Rate of infection (upon exposure)
            gamma   Rate of recovery (upon infection)
            xi      Rate of re-susceptibility (upon recovery)
            mu_I    Rate of infection-related death
            mu_0    Rate of baseline death
            nu      Rate of baseline birth

            beta_Q  Rate of transmission (exposure) for individuals with detected infections
            sigma_Q Rate of infection (upon exposure) for individuals with detected infections
            gamma_Q Rate of recovery (upon infection) for individuals with detected infections
            mu_Q    Rate of infection-related death for individuals with detected infections
            theta_E Rate of baseline testing for exposed individuals
            theta_I Rate of baseline testing for infectious individuals
            psi_E   Probability of positive test results for exposed individuals
            psi_I   Probability of positive test results for exposed individuals
            q       Probability of quarantined individuals interacting with others

            initE   Init number of exposed individuals
            initI   Init number of infectious individuals
            initQ_E Init number of detected infectious individuals
            initQ_I Init number of detected infectious individuals
            initR   Init number of recovered individuals
            initF   Init number of infection-related fatalities
                    (all remaining nodes initialized susceptible)
    """

    plotting_number_property = 'N'
    """Property to access the number to base plotting on."""


    def __init__(self, initN, beta, sigma, gamma, xi=0, mu_I=0, mu_0=0, nu=0, p=0,
                        beta_Q=None, sigma_Q=None, gamma_Q=None, mu_Q=None,
                        theta_E=0, theta_I=0, psi_E=0, psi_I=0, q=0,
                        initE=0, initI=10, initQ_E=0, initQ_I=0, initR=0, initF=0):

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Model Parameters:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.beta   = beta
        self.sigma  = sigma
        self.gamma  = gamma
        self.xi     = xi
        self.mu_I   = mu_I
        self.mu_0   = mu_0
        self.nu     = nu
        self.p      = p

        # Testing-related parameters:
        self.beta_Q   = beta_Q  if beta_Q is not None else self.beta
        self.sigma_Q  = sigma_Q if sigma_Q is not None else self.sigma
        self.gamma_Q  = gamma_Q if gamma_Q is not None else self.gamma
        self.mu_Q     = mu_Q    if mu_Q is not None else self.mu_I
        self.theta_E  = theta_E if theta_E is not None else self.theta_E
        self.theta_I  = theta_I if theta_I is not None else self.theta_I
        self.psi_E    = psi_E   if psi_E is not None else self.psi_E
        self.psi_I    = psi_I   if psi_I is not None else self.psi_I
        self.q        = q       if q is not None else self.q

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize Timekeeping:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.t       = 0
        self.tmax    = 0 # will be set when run() is called
        self.tseries = numpy.array([0])

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize Counts of inidividuals with each state:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.N          = numpy.array([int(initN)])
        self.numE       = numpy.array([int(initE)])
        self.numI       = numpy.array([int(initI)])
        self.numQ_E     = numpy.array([int(initQ_E)])
        self.numQ_I     = numpy.array([int(initQ_I)])
        self.numR       = numpy.array([int(initR)])
        self.numF       = numpy.array([int(initF)])
        self.numS       = numpy.array([self.N[-1] - self.numE[-1] - self.numI[-1] - self.numQ_E[-1] - self.numQ_I[-1] - self.numR[-1] - self.numF[-1]])
        assert(self.numS[0] >= 0), "The specified initial population size N must be greater than or equal to the initial compartment counts."


#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    @staticmethod
    def system_dfes(t, variables, beta, sigma, gamma, xi, mu_I, mu_0, nu,
                                  beta_Q, sigma_Q, gamma_Q, mu_Q, theta_E, theta_I, psi_E, psi_I, q):

        S, E, I, Q_E, Q_I, R, F = variables    # variables is a list with compartment counts as elements

        N   = S + E + I + Q_E + Q_I + R

        dS  = - (beta*S*I)/N - q*(beta_Q*S*Q_I)/N + xi*R + nu*N - mu_0*S

        dE  = (beta*S*I)/N + q*(beta_Q*S*Q_I)/N - sigma*E - theta_E*psi_E*E - mu_0*E

        dI  = sigma*E - gamma*I - mu_I*I - theta_I*psi_I*I - mu_0*I

        dDE = theta_E*psi_E*E - sigma_Q*Q_E - mu_0*Q_E

        dDI = theta_I*psi_I*I + sigma_Q*Q_E - gamma_Q*Q_I - mu_Q*Q_I - mu_0*Q_I

        dR  = gamma*I + gamma_Q*Q_I - xi*R - mu_0*R

        dF  = mu_I*I + mu_Q*Q_I

        return [dS, dE, dI, dDE, dDI, dR, dF]


#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def run_epoch(self, runtime, dt=0.1):

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Create a list of times at which the ODE solver should output system values.
        # Append this list of times as the model's time series
        t_eval    = numpy.arange(start=self.t, stop=self.t+runtime, step=dt)

        # Define the range of time values for the integration:
        t_span          = [self.t, self.t+runtime]

        # Define the initial conditions as the system's current state:
        # (which will be the t=0 condition if this is the first run of this model,
        # else where the last sim left off)

        init_cond       = [self.numS[-1], self.numE[-1], self.numI[-1], self.numQ_E[-1], self.numQ_I[-1], self.numR[-1], self.numF[-1]]

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Solve the system of differential eqns:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        solution        = scipy.integrate.solve_ivp(lambda t, X: SEIRSModel.system_dfes(t, X, self.beta, self.sigma, self.gamma, self.xi, self.mu_I, self.mu_0, self.nu,
                                                                                            self.beta_Q, self.sigma_Q, self.gamma_Q, self.mu_Q, self.theta_E, self.theta_I, self.psi_E, self.psi_I, self.q
                                                                                        ),
                                                        t_span=t_span, y0=init_cond, t_eval=t_eval
                                                   )

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Store the solution output as the model's time series and data series:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.tseries    = numpy.append(self.tseries, solution['t'])
        self.numS       = numpy.append(self.numS, solution['y'][0])
        self.numE       = numpy.append(self.numE, solution['y'][1])
        self.numI       = numpy.append(self.numI, solution['y'][2])
        self.numQ_E       = numpy.append(self.numQ_E, solution['y'][3])
        self.numQ_I       = numpy.append(self.numQ_I, solution['y'][4])
        self.numR       = numpy.append(self.numR, solution['y'][5])
        self.numF       = numpy.append(self.numF, solution['y'][6])

        self.t = self.tseries[-1]


#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def run(self, T, dt=0.1, checkpoints=None, verbose=False):

        if T > 0:
            self.tmax += T
        else:
            return False

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Pre-process checkpoint values:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if checkpoints:
            numCheckpoints = len(checkpoints['t'])
            paramNames = ['beta', 'sigma', 'gamma', 'xi', 'mu_I', 'mu_0', 'nu',
                          'beta_Q', 'sigma_Q', 'gamma_Q', 'mu_Q',
                          'theta_E', 'theta_I', 'psi_E', 'psi_I', 'q']
            for param in paramNames:
                # For params that don't have given checkpoint values (or bad value given),
                # set their checkpoint values to the value they have now for all checkpoints.
                if (param not in list(checkpoints.keys())
                    or not isinstance(checkpoints[param], (list, numpy.ndarray))
                    or len(checkpoints[param])!=numCheckpoints):
                    checkpoints[param] = [getattr(self, param)]*numCheckpoints

        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Run the simulation loop:
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if not checkpoints:
            self.run_epoch(runtime=self.tmax, dt=dt)

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            print("t = %.2f" % self.t)
            if verbose:
                print("\t S   = " + str(self.numS[-1]))
                print("\t E   = " + str(self.numE[-1]))
                print("\t I   = " + str(self.numI[-1]))
                print("\t Q_E = " + str(self.numQ_E[-1]))
                print("\t Q_I = " + str(self.numQ_I[-1]))
                print("\t R   = " + str(self.numR[-1]))
                print("\t F   = " + str(self.numF[-1]))


        else: # checkpoints provided
            for checkpointIdx, checkpointTime in enumerate(checkpoints['t']):
                # Run the sim until the next checkpoint time:
                self.run_epoch(runtime=checkpointTime-self.t, dt=dt)
                # Having reached the checkpoint, update applicable parameters:
                print("[Checkpoint: Updating parameters]")
                for param in paramNames:
                    setattr(self, param, checkpoints[param][checkpointIdx])

                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                print("t = %.2f" % self.t)
                if verbose:
                    print("\t S   = " + str(self.numS[-1]))
                    print("\t E   = " + str(self.numE[-1]))
                    print("\t I   = " + str(self.numI[-1]))
                    print("\t Q_E = " + str(self.numQ_E[-1]))
                    print("\t Q_I = " + str(self.numQ_I[-1]))
                    print("\t R   = " + str(self.numR[-1]))
                    print("\t F   = " + str(self.numF[-1]))

            if self.t < self.tmax:
                self.run_epoch(runtime=self.tmax-self.t, dt=dt)

        return True

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

    def total_num_recovered(self, t_idx=None):
        if t_idx is None:
            return self.numR[:]
        else:
            return self.numR[t_idx]
