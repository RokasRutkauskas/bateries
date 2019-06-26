import bpti_functions as bpti
import display_functions as displ
import git
import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class battery(object):

    """Class with cycling data, processing methods and plotting

    Methods:
        get_data: returns cycling dataframe with merged chosen cells.
        add_SOH: add SOH to dataframe
        add_SOC: add SOC to dataframe
        add_dis_time: add Discharge time to dataframe
        decode_status: decodes Channel Run and Control statuses
        roll: apply rolling function to datafarme by groups
        cycle_plot: plots chosen cycling data
    """

    def __init__(self, btype, cell=None, temp=None, prog=None, curr = None, path=None):
        """Creates batery object, that will be used in further operations.
        For selecting cells with different types btype = 'PF-C21|MJ1-C31'.
        
        Args:
            btype (string): batery type ('PF','MJ1',...)
            cell (list, optional): strings with cell numbers ('C21', ...)
            temp (list, optional): integer with cycling temperatures
            prog (list, optional): strings with cycling programs
            curr (list, optional): integers with cycling currents
            path (string, optional): path to pkl file directory
                (if None, pkl files should be in directory: 'Data/new_data/')
        """
        if path is None:
            # Find git repository path
            git_repo = git.Repo(os.getcwd(), search_parent_directories=True)
            git_root = git_repo.git.rev_parse("--show-toplevel")
            self.path = git_root + '/Data/new_data/'
        else:
            self.path = path
        info = pd.read_excel(self.path + 'data_map.xlsx')
        # select cycling data by parameters
        select = info.copy()
        if temp is not None and len(temp) > 0:
            select = select[select.Temperature.isin(temp)]
        if prog is not None and len(prog) > 0:
            select = select[select.Program.isin(prog)]
        if curr is not None and len(curr) > 0:
            select = select[select.Current.isin(curr)]
        if cell is not None and len(cell) > 0:
            cells = select.copy().Cell.values
            select = select[select.Cell.str.contains(
                '|'.join(btype + '-' + pd.Series(cell) + '$')
                )]
            if select.shape[0] == 0:
                print('\033[1;34mNo selected cells have chosen parameters:')
                print("Avalable cells: ",cells)
                return None
        else:
            select = select[select.Cell.str.contains(btype)]
        # read data
        # if select.shape[0] == 1:
        #     self.data = pd.DataFrame(bpti.pkl_zip_load(self.path + select.Filename.iloc[0]))
        # else:
        self.data = pd.concat([pd.DataFrame(bpti.pkl_zip_load(i))
            for i in self.path + select.Filename], keys = select.Cell)
        self.data.reset_index(0, inplace = True)
        # delete duplicate charger timestamps
        self.data = self.data.drop_duplicates(['Cell', 'Charger Timestamp [s]'])

    def get_data(self, cols=None, drop=None):
        """Returns cycling dataframe with merged chosen cells.

        Args:
            cols (list, optional): names of columns to return in dataframe
            drop (list, optional):  names of columns to drop from dataframe

        Returns:
            pandas.DataFrame: chosen cycling data
        """
        if cols is not None:
            return self.data[cols]
        elif drop is not None:
            return self.data.drop(drop, axis=1)
        else:
            return self.data

    def add_SOH(self, output=False):
        """Add SOH to dataframe

        Args:
            output (bool, optional): should data be returned (Default: False)

        Returns:
            pandas.DataFrame: chosen cycling data or None
        """
        df = self.data[self.data['Cycles Count'] > 1]
        # nominal_soh = df['Output Capacity [mAh]'].max()
        # df_soh = df.groupby(['Cell', 'Cycles Count']) \
        #     [['Output Capacity [mAh]']].max() / nominal_soh * 100
        df_soh = df.groupby('Cell').apply(lambda X: 
                X.groupby('Cycles Count')[['Output Capacity [mAh]']].max() / 
                X['Output Capacity [mAh]'].max()* 100)
        df_soh.columns = ['SOH']
        df_soh.reset_index(inplace = True)
        self.data = pd.merge(self.data, df_soh, on = ['Cell', 'Cycles Count'])
        if output:
            return self.data

    def add_SOC(self, output=False):
        """Add SOC to dataframe

        Args:
            output (bool, optional): should data be returned (Default: False)

        Returns:
            pandas.DataFrame: chosen cycling data or None
        """
        df = self.data[self.data["Cycles Count"] != 1].copy()
        df['SOC'] = df.groupby(['Cell', 'Cycles Count'], sort = False).apply(
                lambda x: x[["Output Capacity [mAh]"]]
                / x[["Output Capacity [mAh]"]].max() * 100)
        df.loc[df['SOC'] < 0, 'SOC'] = 0 # change negative values to 0
        self.data = df
        if output:
            return self.data

    def add_dis_time(self, mode="full", output=False):
        """Add Discharge time to dataframe

        Args:
            mode (string, optional): selected cycle part
                'char' - charge
                'dis'  - discharge
                'idle  - resting time
                'full  - all parts
            output (bool, optional): should data be returned (Default: False)

        Returns:
            pandas.DataFrame: chosen cycling data or None
        """
        switch = {"dis": [13], "char": [7], "idle": [36], "full": [7, 13, 36]}
        df = self.data[self.data["Channel Run Status"].isin(switch[mode])].copy()
        df = df[df["Cycles Count"] != 1]

        df["Discharge time"] = df.groupby(["Cell", "Cycles Count"], sort=False).apply(
            lambda x: x[["Charger Timestamp [s]"]] - x[["Charger Timestamp [s]"]].min()
        )
        self.data = df
        if output:
            return self.data

    def decode_status(self, output=False):
        """Decodes Channel Run and Control statuses from numbers into 
        Delay, Charge, Discharge and Delay, CC, CV
        
        Args:
            output (bool, optional): should data be returned (Default: False)
        
        Returns:
            pandas.DataFrame: cycling data with decoded statuses or None
        """
        df = self.data
        # Decode channel control status
        df['Control_status'] = df['Channel Control Status']\
            .agg(lambda x: int('{0:08b}'.format(int(x))[-2:],2))\
            .replace([0,1,2,3], ['Delay', 'CC', 'CV', 'CC'])
        # Decode channel run status
        df['Run_status'] = df['Channel Run Status']\
            .agg(lambda x: int('{0:08b}'.format(int(x))[-2:],2))\
            .replace([0,1,2,3], ['Delay', 'Discharge', 'Delay', 'Charge'])
        # When current = 0, change status to Delay 
        df.loc[df['Output Current [A]'].round(1) == 0, 
              ['Control_status', 'Run_status']] = 'Delay'
        # Adjust wrong Delay statuses
        char = ((df.Run_status != 'Delay') & 
                (df.Control_status == 'Delay') & 
                (df['Output Current [A]'] > 0))
        df.loc[char, 'Run_status'] = 'Delay' # Delay -> Charge
        df.loc[char, 'Control_status'] = 'Delay' # Delay -> CC
        dis = ((df.Run_status != 'Delay') & 
                (df.Control_status == 'Delay') & 
                (df['Output Current [A]'] < 0))
        df.loc[dis, 'Run_status'] = 'Discharge'
        df.loc[dis, 'Control_status'] = 'CC'
        # Change CV to CC when discharge
        cvdis = ((df.Run_status == 'Discharge') & (df.Control_status == 'CV'))
        df.loc[cvdis, 'Control_status'] = 'CC'
        self.data = df.drop(['Channel Control Status', 'Channel Run Status'], axis=1)
        if output:
            return self.data

    def roll(self, window, gr, label_col, roll_col=None, label_method="mean", remove=True):
        """Apply rolling function to datafarme by groups

        Args:
            window (int): rolling window size
            gr (list): names of group columns
            label_col (TYPE): name of column to aggregate
            roll_col (None, optional): name of columns for rolling
            label_method (string, optional): method to aggregate label_col.
                ("mean", "last", "first")
            remove (bool, optional): If grouped dataframe doesn't have enough
                instances, should they be removed

        Returns:
            multi index pandas.DataFrame: chosen cycling data  with
                rolled columns and aggregated label column
        """
        df = self.data
        grsize = df.groupby(gr).apply(lambda D: D.shape[0] > window)
        if grsize.all():
            return df.groupby(gr).apply(
                lambda D: bpti.rolling(D, window, label_col, roll_col, label_method)
            )
        elif remove:
            df1 = df[~df[gr].isin(grsize[~grsize].index)]
            print("\033[34;1m Cycles {} were removed \033[0m".
                format(list(grsize[~grsize].index)))
            return df1.groupby(gr).apply(
                lambda D: bpti.rolling(D, window, label_col, roll_col, label_method))
        else:
            print("\033[31;1m Error: There is group where window size is \
                greater then number of instances \033[0m")

# PLOTTING FUNCTIONS
    def cycle_plot(self, xcol, ycol, cycles, status = None, df = None, pltype = 'best'):
        """Plots cycling data of selected columns for selected cycles
        with optionally chosen run status and chosen plot type.

        Args:
            xcol (str): column name for x axis
            ycol (str): column name for y axis
            cycles (list): which cycles to plot (int, 'mean', 'max')
                'mean' - find mean cycle for all cells together
                'max' - find last cycle for all cells together
            status (str, optional): channel run status ('char', 'dis', None)
            df (pandas.DataFrame, optional): cycling data;
                if None, data is selected from class
            pltype (str, optional): plot type
                (all types are the same if there is one cell)
                ('best' and 'one' are the same if cycles are specified in numbers)
                'best'    - several cells in one graph, 'mean'('max') cycles calculated separately
                'one'     - several cells in one graph, 'mean'('max') cycles calculated together
                'several' - plots separate graph for each cell
                'other'   - plots data not considering that there may be several cells (bad choice)
        """
        if df is None:
            df = self.data
        if pltype == 'best':
            plt.figure(figsize = (8,6))
            # Plots several cells in one graph when 'mean' and 'max' cycles calculated separately
            for idx, data in df.groupby('Cell'):
                displ.cycle_cell_plot(data, xcol, ycol, cycles, status, idx, None)
            plt.show()
        elif pltype == 'one':
            plt.figure(figsize = (8,6))
            # Plots several cells in one graph
            displ.cycle_cell_plot(df, xcol, ycol, cycles, status)
        elif pltype == 'several':
            # Plots cells separately
            # df.groupby('Cell').apply(lambda data:
                # displ.cycle_cell_plot(data, xcol, ycol, cycles, status, data.name)) # will work with pandas .25
            for idx, data in df.groupby('Cell'):
                plt.figure(figsize = (8,6))
                displ.cycle_cell_plot(data, xcol, ycol, cycles, status, idx, False)
        else:
            plt.figure(figsize = (8,6))
            # Plot selected cycling data (not considering that there may be several cells)
            displ.cycle_cell_plot(df, xcol, ycol, cycles, status, cells = False)


class battery_summary(object):

    """Class with summary cycling data and its plotting

    Methods:
        get_data: returns chosen summary cycling dataframe
        summary_plot: plot selected summary data for specified columns and groups
    """

    def __init__(self, stype='mean', path=None):

        """Creates batery summary object, that will be used in further operations.

        Args:
            stype (string): summary type ('mean' or 'separate')
            path (string, optional): path to none default .pkl file directory
                (if None, .pkl files should be in directory: 'Data/new_data/')

        """

        if path is None:
            # Find git repository path
            git_repo = git.Repo(os.getcwd(), search_parent_directories=True)
            git_root = git_repo.git.rev_parse("--show-toplevel")
            self.path = git_root + '/Data/new_data/'
        else:
            self.path = path
        # Select mean or separate cell summary
        if stype == 'mean':
            self.data = pd.DataFrame(bpti.pkl_zip_load(
                        self.path + 'Summary_cycling_mean_map.pkl'))
        elif stype == 'separate':
            self.data = pd.DataFrame(bpti.pkl_zip_load(
                        self.path + 'Summary_cycling_separate_cells_map.pkl'))
        else:
            print('bad summary type string')

    def get_data(self, btype=None, temp=None, curr=None, prog = None, cell = None,
                 cols=None, drop=None):
        """Returns summary dataframe.
        
        Args:
            btype (list, optional): Battery types to filter against
            temp (list, optional): temperatures to filter against
            curr (list, optional): currents to filter against
            prog (list, optional): cycling program to filter separate cells data against
            cell (list, optional): cell numbers to filter separate cells data against
            cols (list, optional): names of columns to return in dataframe
            drop (list, optional): names of columns to drop from dataframe
        
        Returns:
            pandas.DataFrame: chosen summary data
        """
        # Filter by type or temperature, ...
        df = self.data.copy()
        if btype is not None and len(btype) > 0:
            df = df[df['Type'].isin(btype)]
        if temp is not None and len(temp) > 0:
            df = df[df['Temperature'].isin(temp)]
        if curr is not None and len(curr) > 0:
            df = df[df['Current'].isin(curr)]
        # Additional filtering for separate cells
        if prog is not None and len(prog) > 0:
            df = df[df.Program.isin(prog)]
        if cell is not None and len(cell) > 0:
            cells = df.copy().Cell.values
            df = df[df.Cell.str.contains(
                '|'.join(pd.Series(cell) + '$')
                )]
            if df.shape[0] == 0:
                print('\033[1;34mNo selected cells have chosen parameters:')
                print("Avalable cells: ",cells)
                return None
        # Filter by columns
        if cols is not None:
            return df[cols]
        elif drop is not None:
            return df.drop(drop, axis=1)
        else:
            return df

# PLOTTING FUNCTIONS
    def summary_plot(self, xcol, ycol, gr=None, fmt = '', df=None, show = True, **kwargs):
        """Plot selected summary data for specified columns and groups
        
        Args:
            xcol (string): column name for x axis
            ycol (string): column name for y axis
            gr (list, optional): names of columns to group data
            fmt (str, optional): matplotlib basic plot formatting ('.' - dots)
            df (pandas.DataFrame, optional): summary cycling data
            **kwargs: "get_data" function arguments for data filtering
        """
        if df is None:
            df = self.get_data(**kwargs).reset_index()
        if show:
            plt.figure(figsize = (8,6))
        if gr is None or len(gr) < 1:
            plt.plot(df.sort_values(xcol)[xcol], df.sort_values(xcol)[ycol],
                     fmt)
        else:
            for idx, X in df.groupby(gr):
                plt.plot(X.sort_values(xcol)[xcol], X.sort_values(xcol)[ycol],
                         fmt, label = idx)
                plt.legend(title = ',\n'.join(gr))
        plt.xlabel(xcol)
        plt.ylabel(ycol.replace('_', ' '))
        if show:
            plt.show()