import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from itertools import cycle
from sklearn.metrics import mean_squared_error
import os
import scipy.optimize as optimization

def twin_plot(df, x_label, y1_label, y2_label):
    fig, ax1 = plt.subplots(figsize = (12,6))
    x = df[x_label]
    y = df[y1_label]

    _ = ax1.plot(x,y,'-', label = y1_label)
    _ = ax1.set_xlabel(x_label)
    _ = ax1.set_ylabel(y1_label)

    ax2 = ax1.twinx()
    y = df[y2_label]
    _ = ax2.plot(x,y,'r-.', label = y2_label)
    _ = ax2.set_ylabel(y2_label)

    _ = fig.legend()
    fig.tight_layout()
    plt.show()

def curve_fit_plot(func, df, x_name, y_name, return_df=False):
    '''Plots fitted curve and real data points, and return rmse'''
    df = df.dropna()

    par, cov = optimization.curve_fit(func, df[x_name], df[y_name])


    y = func(df[x_name], par[0], par[1])
    plt.figure(figsize=(8,5))
    plt.plot(df[x_name], y, label = 'fitted')
    plt.plot(df[x_name], df[y_name], '-', label = 'real')
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.legend()
    plt.tight_layout()
    plt.show()
    print('RMSE: %f'%(mean_squared_error(y, df[y_name])))
    if return_df: return par, df
    return par

def plot_graph(df, x_name, y_name):
    '''generates graph with axis labels'''
    plt.figure(figsize=(8,4))
    plt.plot(df[x_name], df[y_name])
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.tight_layout()

def pic_path(title):
    return os.path.abspath(os.path.join('..', os.pardir, './Media/')) + '/' + title

def plotgraphsx(df, plotarr, step, filename = ""):
    """ f#funkcija grafikų atvaizdavimui su regionaisunkcija grafikų atvaizdavimui """
    f, axarr = plt.subplots(len(plotarr), sharex=True, figsize=(20, len(plotarr) * 4))
    idx = 0
    for fig in axarr:
        fig.plot(df.loc[:,[plotarr[idx][0]]])
        fig.set_title(plotarr[idx][1])
        fig.tick_params(direction='in')
        fig.xaxis.grid()
        idx= idx + 1
    start, end = axarr[0].get_xlim()
    major_ticks = np.arange(int(start), end, step)
    axarr[0].set_xticks(major_ticks)
    if filename != "":
        plt.savefig(filename)

    return f, axarr

def plotgraphsx_label(df, plotarr, profile, step = 5, startx = 0.):
    """funkcija grafikų atvaizdavimui su regionais"""
    f, axarr = plt.subplots(len(plotarr), sharex=True, figsize=(20, len(plotarr) * 4))
    idx = 0
    for fig in axarr:
        fig.plot(df.loc[:,[plotarr[idx][0]]])
        fig.set_title(plotarr[idx][1])
        fig.tick_params(direction='in')
        fig.xaxis.grid()
        ymin, ymax = fig.get_ylim()
        for index, row in profile.iterrows():
            fig.annotate(row["Veiksmas"],horizontalalignment='right',
                         xy=(row["Viso"], ymin),fontsize=16)
        idx= idx + 1
    start, end = axarr[0].get_xlim()
    major_ticks = np.arange(int(startx), end, step)
    axarr[0].set_xticks(major_ticks)
    axarr[0].set_xlim(startx)
    return f, axarr

def scatter3d(x,y,z, cs, colorsMap='jet', figsize={15,10}):
    cm = plt.get_cmap(colorsMap)
    cNorm = matplotlib.colors.Normalize(vmin=min(cs), vmax=max(cs))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    fig = plt.figure(figsize = figsize)
    ax = Axes3D(fig)
    ax.scatter(x, y, z, c=scalarMap.to_rgba(cs))
    scalarMap.set_array(cs)
    fig.colorbar(scalarMap)
    return ax

def plot_subplots(params):
    paramtree = {
        'xcol': 'index',
        'ycol' : 'Y',
        'stdcol': 'STD',
        'xaxislabel': 'X',
        'yaxislabel': 'Y',
        'fig_cols': 2,
        'fig_rows': 3,
        'labeltemplate': '%s',
        'titletemplate': '%s',
        'sharex': True,
        'sharey': True,
        'max_x_ticks': 5,
        'max_y_ticks':4,
        'showlegend':True,
        'pdffilename': 'Image.pdf',
        "pdfpath": os.path.abspath(os.path.join(os.getcwd())),
        'savetopdf':False,
        'takeuniquey':False,
        'lines': ['-', '--', '-.', ':', '-'],
    }
    paramtree.update(params)
    p = paramtree
    df = p['df']
    if p['fig_cols'] == 3:
        figy = 22
    else:
        figy = 10 * p['fig_cols']
    f, axs = plt.subplots(p['fig_rows'], p['fig_cols'], figsize=(figy, p['fig_rows'] * 7.5), sharex=p['sharex'], sharey=p['sharey'])
    axs = np.array(axs)
    len_param1arr = len(p['param1arr'])
    for  index, ax in enumerate(axs.reshape(-1)):
        if index >= len_param1arr:
            ax.set_visible(False)
            continue
        linecycler = cycle(p['lines'])
        param1 = p['param1arr'][index]
        if p['takeuniquey']:
            p['param2arr'] = df[df[p['param1name']] == param1][p['param2name']].unique()
        for param2 in p['param2arr']:
            mask = ((df[p['param2name']] == param2) & (df[p['param1name']] == param1))
            if p['xcol'] == 'index':
                x = df[mask].index.values
            else:
                x = df[mask][p['xcol']].values
            y = df[mask][p['ycol']].values
            if len(p['stdcol'])>0:
                err = df[mask][p['stdcol']].values
            label = p['labeltemplate'] % param2
            if len(p['stdcol'])>0:
                ax.errorbar(x,y, yerr=err,fmt=next(linecycler),label=label)
            else:
                ax.plot(x,y, next(linecycler),label=label)
            ax.set_xlabel(p['xaxislabel'])
            ax.set_ylabel(p['yaxislabel'])
            ax.xaxis.set_major_locator(plt.MaxNLocator(p['max_x_ticks']))
            ax.yaxis.set_major_locator(plt.MaxNLocator(p['max_y_ticks']))
            ax.set_title(p['titletemplate'] % param1)
            if p['showlegend']:
                ax.legend(loc='best')
            ax.xaxis.set_tick_params(labelbottom=True)
            ax.yaxis.set_tick_params(labelleft=True)
    f.tight_layout()
    if p['savetopdf']:
        path  = os.path.join(p['pdfpath'], p['pdffilename'])
        f.savefig(path, bbox_inches='tight')


# ----------------------------------------------------------------------------
#                       NEW FUNCTIONS USED IN DATA_CLASS                      
# ----------------------------------------------------------------------------

def cycle_cell_plot(data, xcol, ycol, cycles, status = None, 
                    title = '', cells = True, fmt = ''):
    """Plots cycling data of selected columns for selected cycles 
    with optionally chosen run status. 
    
    Args:
        data (pandas.DataFrame): cycling data  
        xcol (str): column name for x axis
        ycol (str): column name for y axis
        cycles (list): which cycles to plot (int, 'mean', 'max')
            'mean' - find mean cycle for all cells together
            'max' - find last cycle for all cells together 
        status (str, optional): channel run status ('char', 'dis', None)
        title (str, optional): title of the plot; 
            if cells = None, text to add to legend.
        cells (bool, optional): plot cells in different colors, 
            if None, plots cells in same color and adds title to legend labels
        fmt (str, optional): matplotlib basic plot formatting ('.' - dots) 
    """
    # Select data by run status
    if status == 'char':
        data = data[data['Channel Run Status'] == 7].copy()
    elif status == 'dis':
        data = data[data['Channel Run Status'] == 13].copy()    
    # Choose cycles to plot
    cycles = np.array(cycles)
    if 'mean' in cycles:
        cycles[cycles == 'mean'] = data['Cycles Count'].unique().mean().round()
    if 'max' in cycles:
        cycles[cycles == 'max'] = data['Cycles Count'].max()
    data = data[data['Cycles Count'].isin(cycles.astype(float))]
    # Plotting
    if cells is None:
        for idx, X in data.groupby('Cycles Count'):
            plt.plot(X[xcol], X[ycol], fmt, label = '{}: {}'.format(title, idx))
    elif cells == True:
        # data.groupby(['Cycles Count', 'Cell']).apply(
            # lambda X: plt.plot(X[xcol], X[ycol], fmt, label = X.name)) # will work with pandas .25
        for idx, X in data.groupby(['Cycles Count', 'Cell']):
            plt.plot(X[xcol], X[ycol], fmt, label = idx)
    else:
        # data.groupby('Cycles Count').apply(
            # lambda X: plt.plot(X[xcol], X[ycol], fmt, label = X.name)) # will work with pandas .25
        for idx, X in data.groupby('Cycles Count'):
            plt.plot(X[xcol], X[ycol], fmt, label = idx)
    plt.legend()
    plt.xlabel(xcol)
    plt.ylabel(ycol)
    if cells is not None:
        plt.title(title)
        plt.show()