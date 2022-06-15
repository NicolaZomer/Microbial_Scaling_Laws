# Explaining microbial scaling laws using Bayesian inference
In this project, we want to combine methods from _Statistical Physics_ and _Bayesian Data Analysis_ to elucidate the principles behind cellular growth and division. We will study various classes of individual-based growth-division models and infer individal-level processes (model structures and likely ranges of associated parameters) from sigle-cell observations. In the _Bayesian framework_, we formalize our process understanding the form of different rate functions, expressing the dependence of growth and division rates on variables characterizing a cell’s state (such as size and protein content), and calculate the Bayesian posteriors for the parameters of these functions.

## Group 
- [Tommaso Amico](https://github.com/tommasoamico)
- [Andrea Lazzari](https://github.com/AndreaLazzari)
- [Paolo Zinesi](https://github.com/PaoloZinesi)
- [Nicola Zomer](https://github.com/NicolaZomer)

## Organization of the repository
The notebook [Microbial_Scaling_Laws.ipynb](./Microbial_Scaling_Laws.ipynb) is the root file of the repository and includes:
- the theoretical results, both general and specific of the single models;
- the explanation of the methods used and the description of the workflow followed;
- the import and the 3-dimensional plot of the data used in the analysis;
- the hyperlinks to the notebooks of the individual models;
- the general results of all models. 

The repository is then divided into 5 folders:
1. [analysis_real_data](./analysis_real_data)
2. [analysis_sim_data](./analysis_sim_data)
3. [data](./data)
4. [images](./images)
5. [real_data_alternative_way](./real_data_alternative_way), where we use the means and stds as parameters for the Gamma and Beta distributions instead of a,b,c,d

In the folder [analysis_real_data](./analysis_real_data) it is possible to find a Python package, containing the functions used in the analysis of real data: [Fernando_package](./analysis_real_data/Fernando_package).

## Overview
### Growth and division processes: general model 
In our models we consider the evolution of a single non-interacting cell, which undergoes 2 processes:
- **growth:** the cell size $x(t)$ evolves according to the following equation
    $$ \dot{x}=g(x(t)) \quad, \quad x(0)=x_b $$
    In some cases this relation can be expressed in vectorial form, where $\underline{x}$ is the vector of the traits characterizing the cell's state (see model 2). 
- **division:** it is ruled by the _hazard rate function_ $h(x(t))$, which represents the istantaneous probability of the cell to divide. This function is related to the so called _survival function_ $s(t)$, by the relation
    $$ \frac{\dot{s}(t)}{s(t)}=-h(t) \quad , \quad s(0)=1 $$
    where $s(t)$ gives the probability that the cell will survive (meaning not divide in this case) past a certain time $t$.

While the growth is a deterministic process, division is a stochastic event. Since division does not always divide the cell into two equal parts, we introduce a parameter $frac$, which is treated as a stpchastic variable, such that after the division
$$ \underline{x}_{div} = \left(frac\cdot x, (1-frac)x\right)) $$

Finally, we assume that the division ratios $frac$ are distributed according to a Beta function and that the growth rates $\omega_1$ follow a Gamma function, hence denoting by $f$ the probability density distribution we obtain

$$
\begin{align}
f(frac|a, b) &= Beta(a, b) \\
f(\omega_1|c, d) &= Gamma(c, d)
\end{align}
$$

### Model 0 ("Starting model")
**Notebook: [Model 0](./analysis_real_data/REAL_Model_0.ipynb)**

We start with a very simple stochastic model, biologically not very realistic, but useful to start familiarizing with the problem. In this first model we define $g(x)$ and $h(x)$ as 2 linear functions

$$
\begin{align}
g(x) &\equiv \omega_1(\mu+x) \\
h(x) &\equiv \omega_2(1+x/\nu)
\end{align}
$$

where $\omega_1$ and $\omega_2$ are frequencies, while $\mu$ and $\nu$ are sizes (tipycally measured in $\mu m$). The ratio between $\omega_1$ and $\omega_2$ is the order parameter that triggers the phase transition. The parameters $\mu$ and $\nu$ are necessary to cut off the probability distribution (in zero and for large values of $x$), which is important both for physical reasons and for making the distribution normalizable. Introducing these parameters is a mathematical trick, useful for example to prevent the cell from having a too small size, which however is difficult to justify from a biological point of view. We will then see better models, biologically speaking.

### Model 1
**Notebook: [Model 1](./analysis_real_data/REAL_Model_1.ipynb)**

As in the previous model, even in this case the cell growth is governed by a single trait, which is the size. However, this model is biologically more realistic, mainly because a lower bound is placed on the size of the cell such that it can divide. 

Also in this case the processes considered are growth and division, governed by $g(x)$ and $h(x)$ respectively. In this model we define $g(x)$ and $h(x)$ as follows

$$
\begin{align}
g(x)&= \omega_1 x \\
h(x)&=
\begin{cases}
    0  & , x<\mu \\
    \omega_2 \cdot \frac{x+v}{u+v} & , x\geq \mu
\end{cases}
\end{align}
$$

where $g(x)$ again corresponds to an exponential growth, while $h(x)$ is lower bounded by $u$.   

### Model 2
**Notebook: [Model 2](./analysis_real_data/REAL_Model_2.ipynb)**

The main difference between this model and the previous ones is that here we consider 2 traits: the cell size $m(t)$ and its protein content $p(t)$. We call $\underline{x}$ the vector
$$ \underline{x} = (m, p) $$

As before, the traits evolution and the cell division are governed by $g(\underline{x})$ and $h(p)$ respectively, which are defined as 

$$
\begin{align}
g(\underline{x})&=\omega_1m\cdot (1,c) \\
h(p)&=
    \begin{cases}
    0   & , p<\mu \\
    \omega_2 \cdot \frac{p+v}{u+v} & , p\geq \mu
    \end{cases}
\end{align}
$$

From $g(\underline{x})$ we can notice that the cell size still grows exponentially and the protein content also follows this evolution, scaled by the factor $c$. As $c$ doesn't have a real meaning, we set it to $1$. 

Moreover, in this model the condition under which the cell can divide is that it contains a minimum amount of a specific type of protein, which we call $u$. If $p\geq u$ the cell can divide, otherwise it cannot. Unlike model 1, we do not have any condition on the size of the cell for the division to take place and $h$ depens only on $p$.

The initial conditions for $m(t)$ and $p(t)$ are

$$
\begin{align}
    p(t=0) &= 0 \\
    m(t=0) &= m_b
\end{align}
$$

and the division process occurs in the following way
$$ (m,  p) \rightarrow (frac\cdot m, 0) + ((1-frac)\cdot m, 0)$$
where $frac$ is the division ratio.

### Bayesian Data Analysis
For all models, the set of parameters to be inferred is 
$$ \underline{\theta} = \{\mu, \nu, \omega_2, a, b, c, d\} $$

Applying the Bayes theorem, we can write
$$ f(\underline{\theta}|\tau, \omega_1, frac, M) \propto f(\tau, \omega_1, frac|\underline{\theta}, M)\cdot f(\underline{\theta}, M) $$
where $M$ is the background information given by the selected model and $\tau$, $\omega_1$ and $frac$ are provided by the data.

Regarding the likelihood, $f(\tau, \omega_1, frac|\underline{\theta})$, applying the chain rule and exploiting the fact that $frac$ and $\omega_1$ are independent, it can be written as the product of the conditional probability density function of each random variable of interest

$$
\begin{align}
    f(\tau, \omega_1, frac|\underline{\theta}) &= f(\tau|\omega_1, frac, \underline{\theta}) \cdot f(\omega_1, frac|\underline{\theta}) \\
    &=f(\tau|\omega_1, frac, \underline{\theta}) \cdot f(\omega_1|\underline{\theta}) \cdot f(frac|\underline{\theta})
\end{align}
$$ 

where the last 2 are respectively the $Gamma(c, d)$ and $Beta(a, b)$ distributions, while the former is the probability density function of division times, which depends on the selected model and it is the derivative of the survival function $s(t)$.

**Workflow**
- _Calibration_: <br> 
  Performing Markov Chain Monte Carlo (MCMC), via the Python implementation [emcee](https://emcee.readthedocs.io/en/stable/), we find the posterior distribution of $\theta$ and the marginalized posterior of each parameter, of which we calculate the maximum, the median and the 95% credibility interval. Then, we use this results to generate a simulated time series, that can be compared with the real data, to find which model is statistically better.  
- _Validation_: <br>
  Model validation and comparison is achieved by 
  - making a boxplot of the simulated and real interdivision times
  - computing the overlap of the histograms of the interdivision times
  - calculating the predictive density

## References
[1] Held J, Lorimer T, Pomati F, Stoop R, Albert C. Second-order phase transition in phytoplankton trait dynamics. _Chaos_. 2020; 30(5):053109. https://doi.org/10.1063/1.5141755 

[2] Zheng, H., Bai, Y., Jiang, M. et al. General quantitative relations linking cell growth and the cell cycle in Escherichia coli. _Nature Microbiology_. 2020;  5(8):995–1001. https://doi.org/10.1038/s41564-020-0717-x 

[3] emcee documentation: https://emcee.readthedocs.io/en/stable/
