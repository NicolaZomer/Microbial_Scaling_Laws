import numpy as np
from scipy import stats
from pynverse import inversefunc


"""
Starting model
"""

#Cell size evolution x(t) (for the )starting model


def x_function_start(t, pars):
    (omega1, _, mu, _, xb) = pars

    x = (xb+mu/omega1)*np.exp(omega1*t)-mu/omega1
    return x
    

#Hazard rate function h(t) for the starting model

def h_start(t, pars):
    (_, omega2, _, nu, _) = pars

    h = omega2*(1+x_function_start(t, pars)/nu)
    return h


#Logarithm of survival function s(t) for the starting model

def log_CDF_start(t, pars):
    # omega1, xb are arrays of the same length as t
    (omega1, omega2, mu, nu, xb) = pars

    ln_s_ = omega2*t*(mu/nu - 1) + (omega1/omega2)*((mu + xb)/nu)*(1-np.exp(omega1*t))
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
    h_[x_function_mod1(t, pars) < mu] = 0                 # if x(t) < mu
    
    return h_



#Logarithm of survival function s(t) for a float t for model 1

def log_CDF_float_mod1(t, pars):
    # omega1, xb are arrays of the same length as t
    (omega1, omega2, mu, nu, xb) = pars

    if (type(omega1) == np.ndarray) or (type(xb)==np.ndarray):
        t0 = (1.0/omega1) * np.log(mu/xb)
        t0[t0 < 0] = 0
    
    else:

        # threshold time
        t0 = max([0, (1.0/omega1) * np.log(mu/xb)])

  
    if t>=t0:
        ln_s_ = - ( (xb/(mu+nu)) * (omega2/omega1) * (np.exp(omega1*t)-np.exp(omega1*t0)) +\
                    ((nu-xb)/(mu+nu)) * omega2 * (t-t0) )
    else:
        ln_s_ = 0                 # if p(t) < mu
        

    return ln_s_



#Logarithm of survival function s(t) for an array t

def log_CDF_arr_mod1(t, pars):
    # omega1, xb are arrays of the same length as t
    (omega1, omega2, mu, nu, xb) = pars

    if type(omega1) == np.ndarray:
        t0 = (1.0/omega1) * np.log(mu/xb)
        t0[t0 < 0] = 0
    
    else:

        # threshold time
        t0 = max([0, (1.0/omega1) * np.log(mu/xb)])
    
    ln_s_ = - ( (xb/(mu+nu)) * (omega2/omega1) * (np.exp(omega1*t)-np.exp(omega1*t0)) +\
                (nu/(mu+nu)) * omega2 * (t-t0) )
    ln_s_[t < t0] = 0                 # if x(t) < mu

    return ln_s_



    

#Logarithm of survival function s(t)

def log_CDF_mod1(t, pars):
    
    if type(t) == np.ndarray: # array
        ln_s_ = log_CDF_arr_mod1(t, pars)
    else: # float
        ln_s_ = log_CDF_float_mod1(t, pars)

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
    h_[p_function(t, pars) < mu] = 0                 # if p(t) < mu
    
    return h_



#Logarithm of survival function s(t) for a float t

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



#Logarithm of survival function s(t) for an array t

def log_CDF_arr_mod2(t, pars):
    # omega1, xb are arrays of the same length as t
    (omega1, omega2, mu, nu, mb) = pars

    # threshold time
    t0 = (1.0/omega1) * np.log(1 + (mu/mb))

    ln_s_ = - ( (mb/(mu+nu)) * (omega2/omega1) * (np.exp(omega1*t)-np.exp(omega1*t0)) +\
                ((nu-mb)/(mu+nu)) * omega2 * (t-t0) )
    
    ln_s_[t < t0] = 0                 # if p(t) < mu


    return ln_s_



#Logarithm of survival function s(t)

def log_CDF_mod2(t, pars):

    if type(t) == np.ndarray: # array
        ln_s_ = log_CDF_arr_mod2(t, pars)
    else: # float
        ln_s_ = log_CDF_float_mod2(t, pars)

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
    t = np.array(t)
    idx = np.argsort(np.array(t))
    normalization = np.trapz(x=t[np.array(idx)], y=unnormalized[np.array(idx)])

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


#Log unnormalized posterior, for emcee

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


# Function to draw the times from a model
# Simulation of the time series 
#The division rates are sampled from the inferred beta distribution 

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

        # find maximum monotonically increasing sequence
        #tau_seq = np.linspace(t0, 10/omega1, num=1000)
        #s_seq = CDF(tau_seq, parameters)
        #t_max = tau_seq[np.argmax((s_seq[1:]-s_seq[:-1])>=0)]
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
        #tau = draw_tau_numerical(s_drawn[i], parameters=parameters)
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

        # find maximum monotonically increasing sequence
        #tau_seq = np.linspace(t0, 10/omega1, num=1000)
        #s_seq = CDF(tau_seq, parameters)
        #t_max = tau_seq[np.argmax((s_seq[1:]-s_seq[:-1])>=0)]
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
        #tau = draw_tau_numerical(s_drawn[i], parameters=parameters)
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


    


