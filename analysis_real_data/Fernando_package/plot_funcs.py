# Module for the functions of the various plots

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go


#Plot cumulative distribution (for the lognormal)
def cumulative(array, title='Cumulative', xlabel='x axis', ylabel='y axis'):
    fig, ax = plt.subplots(1,1,figsize=(15,10))
    array = np.array(array)
    array = np.sort(array)
    array = array[~np.isnan(array)]
    cumul = 1 - np.arange(0, len(array))/(len(array))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.minorticks_on()
    ax.scatter(array, cumul, s=150, edgecolor='black', alpha=0.7)
    return fig, ax, cumul, array



#Plot of the posterior distribution of simulated data
def plot_func_sim(chain, parameter, x_median=-0.1, y_median=2.26, x_max=0.02, \
            y_max=2.2, info = False, plot=True):

    if info:
        print('''
        This function as takes as a parameter argument that should be either:
        - omega_2
        - mu
        - nu
        - a
        - b
        - c
        - d
        
        
        
     
        ''')
    if parameter == 'omega_2':
       parameter = '$\\omega_2$'
       index = 0
    elif parameter == 'mu':
        parameter = '$\\mu$'
        index = 1
    elif parameter == 'nu':
        parameter = '$\\nu$'
        index = 2
    elif parameter == 'a':
        index = 3
    elif parameter == 'b':
        index = 4
    elif parameter == 'c':
        index = 5
    elif parameter == 'd':
        index = 6
    else:
        return 'The parameter should be chose between [omega_2, mu, nu, a, b, c, d]'
    if plot:
        fig, ax = plt.subplots(1,1 , figsize=(15, 10))
        res_param = ax.hist(chain[:,index], bins='fd', edgecolor='black', alpha=0.5, density=True)

        counts_param = res_param[0]
        edges_param = res_param[1]
        patches_param = res_param[2]
        centers_param = (edges_param[:-1] + edges_param[1:]) / 2

        tmp = np.cumsum(np.diff(edges_param)*counts_param)

        max_index = np.argmax(counts_param)
        max_param = (edges_param[max_index] + edges_param[max_index + 1])/2
        median_param = (edges_param[len(tmp[tmp<0.5])+1] + edges_param[len(tmp[tmp<0.5])+2])/2
        print('Median value of ', parameter,':', round(median_param, 4))
        for i in range(len(tmp[tmp<0.025])):
            patches_param[i].set_facecolor('green')
        for i in range(len(tmp[tmp<0.975]), len(tmp)):
            patches_param[i].set_facecolor('green')

        cred_int_low = patches_param[len(tmp[tmp<0.025])].get_x()
        cred_int_high = patches_param[len(tmp[tmp<0.975])].get_x()

        ax.minorticks_on()
        ax.set_ylabel('Counts', fontsize=15)
        ax.set_xlabel(parameter, fontsize=15)
        ax.set_title(parameter + ' posterior', fontsize=20)
        ax.axvline(median_param, color='crimson', linestyle='dashed',  linewidth=3, label='median '+parameter + ' = ' + str(round(median_param, 4)))
        ax.axvline(max_param, color='darkgreen', linestyle='dashed',  linewidth=3, label='max '+ parameter + ' = ' + str(round(max_param, 4)))

        ax.text(median_param+x_median, y_median, 'median '+parameter , color='crimson', fontsize=17)
        ax.text(max_param+x_max, y_max,'max '+parameter, color='darkgreen', fontsize=17)

        print('Max value of ', parameter,' :', round(max_param, 4))
        print('Credibility interval of ', parameter,' : [', round(cred_int_low, 4), ',', round(cred_int_high, 4), ']')

        ax.legend(fontsize=17, facecolor='aliceblue', shadow = True, edgecolor='black')

    else:
        res_param = np.histogram(chain[:,index], bins='fd', density=True)
        counts_param = res_param[0]
        edges_param = res_param[1]
        centers_param = (edges_param[:-1] + edges_param[1:]) / 2
        tmp = np.cumsum(np.diff(edges_param)*counts_param)
        max_index = np.argmax(counts_param)
        max_param = (edges_param[max_index] + edges_param[max_index + 1])/2
        median_param = (edges_param[len(tmp[tmp<0.5])+1] + edges_param[len(tmp[tmp<0.5])+2])/2

    if plot:
        return fig, ax, centers_param, counts_param, max_param, cred_int_low, cred_int_high
    else:
        return centers_param, counts_param, max_param



# Function to plot the chains of the sampler
def chains_plot(chain, title_list = ['$\\omega_2$', '$\\mu$', '$\\nu$', 'a', 'b', 'c', 'd'], info = True):

    if info:
        print('''
        The chain to be put in input has to be the flattened one
        
        ''')

    fig , ax = plt.subplots(chain.shape[1],1, figsize=(45, 100))

    for i, i_title in zip(range(len(title_list)), title_list):
        ax[i].scatter(range(0,chain.shape[0]) , chain[:,i], s=1)
        ax[i].set_ylabel('Value of the chain', fontsize=30)
        ax[i].set_xlabel('Iteration Step', fontsize=30)
        ax[i].set_title('Chain of '+i_title, fontsize=40)
        ax[i].minorticks_on()
        ax[i].tick_params(axis='both', which='major', labelsize=25, width=3, length=25)
        ax[i].tick_params(axis='both', which='minor', width=3, length=10)

    return fig, ax



#From our beloved Amos package we copy the boxplot function
def boxplot(y, colors, figsize=(15,10), linewidth=2, color_median='black', linewidth_median = 3, size_props = 12, title = 'boxplot', font_title = 20, xlabel='x_axis', x_font=15, ylabel='y_axis', y_font=15, labels=False, list_labels=[] ):
    
    fig, ax = plt.subplots(1,1,figsize=figsize)
    bp = ax.boxplot(y,boxprops=dict(linewidth=linewidth), patch_artist=True, medianprops=dict(color=color_median, linewidth=linewidth_median), flierprops=dict(markersize=size_props))
    
    ax.set_title(title, fontsize=font_title)
    ax.set_xlabel(xlabel, fontsize=x_font)
    ax.set_ylabel(ylabel, fontsize=y_font)

    #Still to insert the case were y is a matrix
    
    for array, x_point, color in zip(y, range(1, len(y)+1), colors) :
        tmp = np.ones([len(array),])*x_point + 0.35
        ax.scatter(tmp, array, s=77, alpha=0.5, edgecolor='black', color=color)

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
   
    ax.minorticks_on()
    ax.xaxis.set_tick_params(which='minor', bottom=False , size=10)
    ax.yaxis.set_tick_params(labelsize=14, size=10)

    if labels:
        ax.set_xticklabels(list_labels)
           
    return fig, ax, bp



# Get informations about the boxplot function
def info_boxplot():
    print('''
    The boxplot function takes only one mandatory argument: y that is either a matrix-like with the series to plot as columns or an array-like of array-likes.
    
    Then it accepts the following key-word arguments:
    1) y: a list of the sequences to plot
    2) colors: a list with the colors of the boxes,
    

    3) title, xlabel and ylabel: self explanatory, we have also as far as the fontsize font_title, 
        x_font, y_font.

    4) color_median: the color of the median
    5) linewidth_median: self explanatory
    6) size_props: size of the outliers
    
    ''')




#Time series plot
def plot_evol(all_times, cell_sizes):
    fig, ax = plt.subplots(1,1 , figsize=(25, 10))
    ax.plot(all_times, cell_sizes, linewidth=3, color='C0')
    ax.set_title('Cell size evolution', fontsize=25)
    ax.set_xlabel('t', fontsize = 20)
    ax.set_ylabel('Cell size', fontsize=20)
    ax.minorticks_on()
    ax.grid(alpha=0.5)
    return fig, ax



# Histograms (with equal bins) + estimation of overlap
def overlap_hist(real_data, sim_data):

    # define bins to be used for both the histograms
    real_data_unique = np.unique(np.round(real_data, decimals=3))
    delta_t = np.round(np.min(real_data_unique[1:]-real_data_unique[:-1]), decimals=3)
    t_bins = np.arange(0.5*delta_t, np.max(pd.concat([real_data,sim_data])) + 1.5*delta_t, delta_t)

    # plot histograms with equal bins
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    density1, _, _ = ax.hist(real_data, bins=t_bins, edgecolor='black', alpha=0.5, density=True, label='Real data')
    density2, _, _ = ax.hist(sim_data, bins=t_bins, edgecolor='black', alpha=0.5, density=True, label='Simulated data')
    ax.set_title('Interdivision times, $\\tau$', fontsize=20)
    ax.legend()

    # estimate 0 ≤ overlap ≤ 1
    overlap = np.sum(np.sqrt(density1*density2)) * delta_t
    x_text = ax.get_xlim()[0] + 0.65*(ax.get_xlim()[1]-ax.get_xlim()[0])
    y_text = ax.get_ylim()[0] + 0.75*(ax.get_ylim()[1]-ax.get_ylim()[0])
    ax.text(x_text, y_text, f'overlap = {np.round(overlap, decimals=4)}', fontsize=24)

    return fig, ax



# 3D scatterplot of data (generation time, growth rate, division ratio)
def plot_3d_interactive(df_, real_data_title=True):
    if real_data_title: 
        title= 'Real Data'
    else: 
        title= 'Simulation Results'

    fig = px.scatter_3d(
        df_, x='generationtime', y='growth_rate', z='division_ratio', 
        opacity=0.7, title=title)
        
    fig.update_traces(marker_size = 3)
    fig.update_layout(
        scene = dict(
            aspectmode='cube'
        )
    )

    return fig



def comparison_3d(df1, df2, df3, df4):
    fig = make_subplots(rows=2, cols=2,
                        specs=[[{'type': 'scene'}, {'type': 'scene'}],
                            [{'type': 'scene'}, {'type': 'scene'}]],
                        subplot_titles=("Real data", "Linear model", "Cut-off model", "Protein model"),
                        )

    fig.add_trace(go.Scatter3d( x=df1['generationtime'], y=df1['growth_rate'], z=df1['division_ratio'], 
            mode='markers', marker=dict(
            size=3,
            color='blue',
            opacity=0.3, 
        )), row=1, col=1)



    fig.update_layout(
        title='3d scatter plots comparison', 
        
        scene=dict(
                xaxis_title='generationtime',
                yaxis_title='growth_rate',
                zaxis_title='division_ratio'),
                scene2=dict(
                xaxis_title='generationtime',
                yaxis_title='growth_rate',
                zaxis_title='division_ratio'),
                scene3=dict(
                xaxis_title='generationtime',
                yaxis_title='growth_rate',
                zaxis_title='division_ratio'),
                scene4=dict(
                xaxis_title='generationtime',
                yaxis_title='growth_rate',
                zaxis_title='division_ratio')
    )

    fig.add_trace(go.Scatter3d( x=df2['generationtime'], y=df2['growth_rate'], z=df2['division_ratio'], 
            mode='markers', marker=dict(
            size=3,
            color='blue',
            opacity=0.3
        )), row=1, col=2)

    fig.add_trace(go.Scatter3d( x=df3['generationtime'], y=df3['growth_rate'], z=df3['division_ratio'], 
            mode='markers', marker=dict(
            size=3,
            color='blue',
            opacity=0.3
        )), row=2, col=1)

    fig.add_trace(go.Scatter3d( x=df4['generationtime'], y=df4['growth_rate'], z=df4['division_ratio'], 
            mode='markers', marker=dict(
            size=3,
            color='blue',
            opacity=0.3
        )), row=2, col=2)
    fig.update_layout(showlegend=False, width=1000, height=1000)
    return fig