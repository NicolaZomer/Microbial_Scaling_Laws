import numpy as np
from scipy import stats
from pynverse import inversefunc
import emcee
from .plot_funcs import plot_func_sim



"""
Starting model
"""

#Cell size evolution x(t) for the starting model
def x_function_start(t, pars):
    (omega1, _, mu, _, xb) = pars

    x = (xb+mu)*np.exp(omega1*t)-mu
    return x
    

#Hazard rate function h(t) for the starting model
def h_start(t, pars):
    (_, omega2, _, nu, _) = pars

    h = omega2*(1+x_function_start(t, pars)/nu)
    return h


#Logarithm of survival function s(t) for the starting model
def log_CDF_start(t, pars):
    # t, omega1, xb are usually arrays of the same size, but floats values are also compatible
    (omega1, omega2, mu, nu, xb) = pars

    ln_s_ = omega2*t*(mu/nu - 1) + (omega2/omega1)*((mu + xb)/nu)*(1-np.exp(omega1*t))
    return ln_s_


#Survival function s(t) for the starting model
def CDF_start(t, pars):
    return np.exp(log_CDF_start(t, pars))




"""
Model 1
"""

#Cell size evolution x(t) for model 1
def x_function_mod1(t, pars):
    (omega1, _, _, _, xb) = pars

    x_ = (xb)*np.exp(omega1*t)
    return x_


#Hazard rate function h(t) for model 1
def h_mod1(t, pars):
    (_, omega2, mu, nu, _) = pars

    h_ = omega2*((x_function_mod1(t, pars) + nu)/(mu+nu)) # if x(t) ≥ mu
    h_ = np.reshape(h_, -1)
    h_[x_function_mod1(t, pars) < mu] = 0                 # if x(t) < mu
    
    # h_ is a vector in all cases (with length ≥ 1)
    return h_




#Logarithm of survival function s(t)
def log_CDF_mod1(t, pars):

    # t, omega1, xb are usually arrays of the same size, but floats values are also compatible
    (omega1, omega2, mu, nu, xb) = pars
    t_arr = np.reshape(t, -1)

    # threshold time (array)
    t0 = (1.0/omega1) * np.log(mu/xb)
    t0 = np.reshape(t0, -1)
    t0[t0 < 0] = 0
    

    ln_s_ = - ( (xb/(mu+nu)) * (omega2/omega1) * (np.exp(omega1*t_arr)-np.exp(omega1*t0)) +\
                (nu/(mu+nu)) * omega2 * (t_arr-t0) )
    ln_s_[t_arr < t0] = 0                 # if x(t) < mu

    # ln_s_ is a vector in all cases (with length ≥ 1)
    return ln_s_


#Survival function s(t)
def CDF_mod1(t, pars):
    return np.exp(log_CDF_mod1(t, pars))




"""
Model 2
"""

#Cell size evolution m(t)
def m_function(t, pars):
    (omega1, _, _, _, mb) = pars

    m_ = (mb)*np.exp(omega1*t)
    return m_


#Protein content evolution p(t)
def p_function(t, pars):
    (omega1, _, _, _, mb) = pars

    p_ = (mb)*(np.exp(omega1*t) - 1)
    return p_


#Hazard rate function h(t)
def h_mod2(t, pars):
    (_, omega2, mu, nu, _) = pars

    h_ = omega2*((p_function(t, pars) + nu)/(mu+nu)) # if p(t) ≥ mu
    h_ = np.reshape(h_, -1)
    h_[p_function(t, pars) < mu] = 0                 # if p(t) < mu
    
    # h_ is a vector in all cases (with length ≥ 1)
    return h_


#Logarithm of survival function s(t) for a float t
# (MIGHT BE REPLACED BY THE MORE COMPACT FUNCTION BELOW)
"""
def log_CDF_float_mod2(t, pars):
    # omega1, xb are arrays of the same length as t
    (omega1, omega2, mu, nu, mb) = pars

    # threshold time
    t0 = (1.0/omega1) * np.log(1 + (mu/mb))

    if t>=t0:
        ln_s_ = - ( (mb/(mu+nu)) * (omega2/omega1) * (np.exp(omega1*t)-np.exp(omega1*t0)) +\
                    ((nu-mb)/(mu+nu)) * omega2 * (t-t0) )
    else:
        ln_s_ = 0                 # if p(t) < mu
        

    return ln_s_
"""


#Logarithm of survival function s(t) for an array t
# (MIGHT BE REPLACED BY THE MORE COMPACT FUNCTION BELOW)
"""
def log_CDF_arr_mod2(t, pars):
    # omega1, xb are arrays of the same length as t
    (omega1, omega2, mu, nu, mb) = pars

    # threshold time
    t0 = (1.0/omega1) * np.log(1 + (mu/mb))

    ln_s_ = - ( (mb/(mu+nu)) * (omega2/omega1) * (np.exp(omega1*t)-np.exp(omega1*t0)) +\
                ((nu-mb)/(mu+nu)) * omega2 * (t-t0) )
    
    ln_s_[t < t0] = 0                 # if p(t) < mu


    return ln_s_
"""


#Logarithm of survival function s(t)
def log_CDF_mod2(t, pars):
    """
    if type(t) == np.ndarray: # array
        ln_s_ = log_CDF_arr_mod2(t, pars)
    else: # float
        ln_s_ = log_CDF_float_mod2(t, pars)
    """

    # t, omega1, xb are usually arrays of the same size, but floats values are also compatible
    (omega1, omega2, mu, nu, xb) = pars
    t_arr = np.reshape(t, -1)

    # threshold time (array) always ≥ 0
    t0 = (1.0/omega1) * np.log(1 + (mu/xb))
    t0 = np.reshape(t0, -1)
    

    ln_s_ = - ( (xb/(mu+nu)) * (omega2/omega1) * (np.exp(omega1*t_arr)-np.exp(omega1*t0)) +\
                ((nu-xb)/(mu+nu)) * omega2 * (t_arr-t0) )
    ln_s_[t_arr < t0] = 0                 # if p(t) < mu

    # ln_s_ is a vector in all cases (with length ≥ 1)
    return ln_s_


#Survival function s(t)
def CDF_mod2(t, pars):
    return np.exp(log_CDF_mod2(t, pars))
    



"""
Functions common to all models
"""

#PDF function, common for all the models
def PDF(t, pars, h_func, cdf_func):
   
    unnormalized = h_func(t, pars)*cdf_func(t, pars)
    unnormalized = np.array(unnormalized)
    t = np.reshape(t, -1) # t = np.array(t)
    idx = np.argsort(t) # idx = np.argsort(np.array(t))
    normalization = np.trapz(x=t[idx], y=unnormalized[idx]) 
    # normalization = np.trapz(x=t[np.array(idx)], y=unnormalized[np.array(idx)])

    return(unnormalized/normalization)


#Log unnormalized posterior, for emcee
def j_log_unnorm_posterior_emcee(thetas, y_times, omega_1, frac, PDF, xb, h_func, cdf_func, \
                                priors, info = False):
    if info:
        print('''
        The priors arguments that this function takes in input has to be a dictionary
        with the various priors, the keys must be named:
        - mu_nu
        - omega2
        - a
        - b
        - c
        - d
        ''')
    try:

        # parameters sampled by EMCEE following the log_posterior
        # (to not be confused with 'pars', which are used in the PDFs and CDFs)
        omega_2, mu, nu, a, b, c, d = np.array(thetas)

        # WARNING: here we are assuming that in the computation of the tau likelihood the omega_1 are fixed to
        # the maximum of Gamma(a=c, scale=d) => omg = d*(c-1)
        # this means that the cell has the same omega_1 during its entire lifetime
        # more rigorously, we should consider the real-data omega_1 in the computation of the tau likelihood
        omg = d*(c-1)
        
        
        ret = (
            np.sum(np.log(PDF(t = np.array(y_times), pars=(omg, omega_2, mu, nu, xb),
                              h_func=h_func, cdf_func=cdf_func))) +     # likelihood( tau    | omega1,omega2,...)
            np.sum(np.log(stats.beta.pdf(frac, a=a, b=b))) +            # likelihood( frac   | a,b              )
            np.sum(np.log(stats.gamma.pdf(omega_1, a=c, scale=d))) +    # likelihood( omega1 | c,d              )

            priors['mu_nu'](mu, nu) +                                   # prior(mu,nu)
            np.log(priors['omega2'](omega_2)) +                         # prior(omega2)
            np.log(priors['a'](a)) +                                    # prior(a)
            np.log(priors['b'](b)) +                                    # prior(b)
            np.log(priors['c'](c)) +                                    # prior(c)
            np.log(priors['d'](d))                                      # prior(d)
        )
        
        if np.isfinite(ret):
            return(ret)
        else:
            return(-np.inf)

    except:
        
        return(-np.inf)


#Log unnormalized posterior 2, for emcee
def j_log_unnorm_posterior_emcee_2(thetas, y_times, omega_1, frac, PDF, xb, h_func, cdf_func, \
                                priors, info = False):
    if info:
        print('''
        The priors arguments that this function takes in input has to be a dictionary
        with the various priors, the keys must be named:
        - mu_nu
        - omega2
        - a
        - b
        - c
        - d
        ''')
    try:

        # parameters sampled by EMCEE following the log_posterior
        # (to not be confused with 'pars', which are used in the PDFs and CDFs)
        omega_2, mu, nu, a, b, c, d = np.array(thetas)
        
         
        ret = (
            np.sum(np.log(PDF(t = np.array(y_times), pars=(omega_1, omega_2, mu, nu, xb),
                              h_func=h_func, cdf_func=cdf_func))) +     # likelihood( tau    | omega1,omega2,...)
            np.sum(np.log(stats.beta.pdf(frac, a=a, b=b))) +            # likelihood( frac   | a,b              )
            np.sum(np.log(stats.gamma.pdf(omega_1, a=c, scale=d))) +    # likelihood( omega1 | c,d              )

            priors['mu_nu'](mu, nu) +                                   # prior(mu,nu)
            np.log(priors['omega2'](omega_2)) +                         # prior(omega2)
            np.log(priors['a'](a)) +                                    # prior(a)
            np.log(priors['b'](b)) +                                    # prior(b)
            np.log(priors['c'](c)) +                                    # prior(c)
            np.log(priors['d'](d))                                      # prior(d)
        )
        
        if np.isfinite(ret):
            return(ret)
        else:
            return(-np.inf)

    except:
        
        return(-np.inf)



# Function to draw the times from a model, so to simulate the time series 
# The division rates are sampled from the inferred beta distribution and the growth rates from the gamma. 

def sim_t_draw(log_CDF, x_function, size, points_per_evolution, xb, model, pars_new):
    (omega2, mu, nu, a, b, c, d) = pars_new

    
    #Find tau numerically
    
    def draw_tau_numerical(log_K, parameters):
        (omega1, _, mu, _, xb) = parameters
        if model =='s':
            t0 = 0
        elif model == '1':
            t0 = max([0, (1.0/omega1) * np.log(mu/xb)])
        elif model == '2':
            t0 = max([0, (1.0/omega1) * np.log(1 + (mu/xb))])
        else:
            raise Exception("Model has to be either 's', '1', or '2' ")

        t_max = 100/omega1 # 100 tau

        # invert function
        tau = inversefunc(log_CDF, args=(parameters,), y_values=log_K,  domain=[t0, t_max], open_domain=True)
        return tau

    t = 0

    all_times = np.zeros(points_per_evolution*size)
    cell_sizes = np.zeros(points_per_evolution*size)
    sim_t_starting = []

    # sample omega1 and frac
    np.random.seed(29071981)
    frac = np.random.beta(a, b, size=size)
    omg1 = np.random.gamma(shape=c, scale=d, size=size)
    s_drawn = np.random.uniform(low=0, high = 1, size = size)
    log_s_drawn = np.log(s_drawn)

    for i in range(size): 
        parameters = (omg1[i], omega2, mu, nu, xb) # omega1, omega2, mu, nu, xb
        tau = draw_tau_numerical(log_s_drawn[i], parameters=parameters)
        sim_t_starting.append(tau)

        # evolution
        times = np.linspace(0, tau, points_per_evolution)
        xt = x_function(times, parameters)
        
        # store times and sizes
        all_times[i*points_per_evolution : (i+1)*points_per_evolution] = np.linspace(t, t+tau, points_per_evolution)
        cell_sizes[i*points_per_evolution : (i+1)*points_per_evolution] = xt

        # update the initial time and the starting size
        xb = xt[-1]*frac[i]
        t = t+tau

    sim_t_starting = np.asarray(sim_t_starting)

    return sim_t_starting, all_times, cell_sizes, frac, omg1


# Simulation using the real data in the right way
def sim_t_draw_real(log_CDF, x_function, size, points_per_evolution, xb, frac, omega_1, model, pars_new):
    (omega2, mu, nu, a, b, c, d) = pars_new

    
    #Find tau numerically
    
    def draw_tau_numerical(log_K, parameters):
        (omega1, _, mu, _, xb) = parameters
        if model =='s':
            t0 = 0
        elif model == '1':
            t0 = max([0, (1.0/omega1) * np.log(mu/xb)])
        elif model == '2':
            t0 = max([0, (1.0/omega1) * np.log(1 + (mu/xb))])
        else:
            raise Exception("Model has to be either 's', '1', or '2' ")

        t_max = 100/omega1 # 100 tau

        # invert function
        tau = inversefunc(log_CDF, args=(parameters,), y_values=log_K,  domain=[t0, t_max], open_domain=True)
        return tau

    t = 0

    all_times = np.zeros(points_per_evolution*size)
    cell_sizes = np.zeros(points_per_evolution*size)
    sim_t_starting = []

    np.random.seed(29071981)
    s_drawn = np.random.uniform(low=0, high = 1, size = size)
    log_s_drawn = np.log(s_drawn)

    for i in range(size): 
        parameters = (omega_1[i], omega2, mu, nu, xb) # omega1, omega2, mu, nu, xb
        tau = draw_tau_numerical(log_s_drawn[i], parameters=parameters)
        sim_t_starting.append(tau)

        # evolution
        times = np.linspace(0, tau, points_per_evolution)
        xt = x_function(times, parameters)
        
        # store times and sizes
        all_times[i*points_per_evolution : (i+1)*points_per_evolution] = np.linspace(t, t+tau, points_per_evolution)
        cell_sizes[i*points_per_evolution : (i+1)*points_per_evolution] = xt

        # update the initial time and the starting size
        xb = xt[-1]*frac[i]
        t = t+tau

    sim_t_starting = np.asarray(sim_t_starting)

    return sim_t_starting, all_times, cell_sizes


    

# Predictive density estimate
def predictive_density(df_, size, p0, h_func, cdf_func, priors, N_perm=10, burn_in=1700, n_steps=5000, n_walkers=20):

    np.random.seed(24082003)

    log_predD = []
    size_train = int(0.5*size)

    for i in range(N_perm):

        # split data into train and test sets
        perm = np.random.permutation(size)
        df_train = df_.iloc[perm[:size_train],:]
        df_test = df_.iloc[perm[size_train:],:]

        # emcee, using real data
        sampler_train = emcee.EnsembleSampler(
            n_walkers, 7, j_log_unnorm_posterior_emcee_2, 
            kwargs={'y_times': np.array(df_train['generationtime']), 
                    'frac': np.array(df_train['division_ratio']), 
                    'omega_1': np.array(df_train['growth_rate']), 
                    'PDF': PDF, 
                    'h_func' : h_func,
                    'cdf_func' : cdf_func,
                    'xb': np.array(df_train['length_birth']),
                    'priors': priors}, 
            a=2
        )

        # burn-in 
        pos, prob, state = sampler_train.run_mcmc(p0, burn_in)
        sampler_train.reset()

        # run mcmc
        sampler_train.run_mcmc(pos, n_steps, rstate0=state)

        # get chains
        chain_train = sampler_train.get_chain(flat=True)


        # MAX PARAMETERS

        # omega2 max
        _, _, max_omega2_train = plot_func_sim(chain = chain_train, parameter = 'omega_2', plot=False)

        # mu max
        _, _, max_mu_train = plot_func_sim(chain = chain_train, parameter = 'mu', plot=False)

        # nu max
        _, _, max_nu_train = plot_func_sim(chain = chain_train, parameter = 'nu', plot=False)

        # a max
        _, _, max_a_train = plot_func_sim(chain = chain_train, parameter = 'a', plot=False)

        # b max
        _, _, max_b_train = plot_func_sim(chain = chain_train, parameter = 'b', plot=False)

        # c max
        _, _, max_c_train = plot_func_sim(chain = chain_train, parameter = 'c', plot=False)

        # d max
        _, _, max_d_train = plot_func_sim(chain = chain_train, parameter = 'd', plot=False)


        # logarithm of predictive density with current train+test set
        log_predD_i =   np.sum(np.log(PDF(t = np.array(df_test['generationtime']),
                                            pars = (np.array(df_test['growth_rate']),
                                                    max_omega2_train,    
                                                    max_mu_train,
                                                    max_nu_train,
                                                    np.array(df_test['length_birth'])),
                                            h_func=h_func,
                                            cdf_func=cdf_func))) +\
                        np.sum(np.log(stats.beta.pdf(np.array(df_test['division_ratio']),
                                                a=max_a_train,
                                                b=max_b_train))) +\
                        np.sum(np.log(stats.gamma.pdf(np.array(df_test['growth_rate']),
                                                a=max_c_train,
                                                scale=max_d_train)))

        log_predD.append(log_predD_i)

    return(np.asarray(log_predD))


'''
Priors of omega1, omega2, mu, nu
'''

def prior_omega2(omega2):
    return(stats.lognorm.pdf(omega2, s=np.sqrt(1/3 - np.log(0.9)), loc=0, scale=np.exp(1/3 )))
    #return(stats.lognorm.pdf(x, s=1/2, loc=0, scale=np.exp(np.log(0.9)+1/4)))

def prior_mu(mu):
    return(stats.beta.pdf(mu, a=2, b=5))

def prior_nu(nu):
    return(stats.lognorm.pdf(nu, s=1/3, loc=0.1, scale=np.exp(1/9)))
    
def prior_omega1(omega1):
    return(stats.lognorm.pdf(omega1, s=1/3, loc=0, scale=np.exp(1/9)))