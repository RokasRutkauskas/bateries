import os
import numpy as np
import math
import pandas as pd
import pickle as cPickle
import gzip
import ntpath
from scipy import signal
import scipy.integrate as integrate
from uncertainties import unumpy
from scipy import stats
import re
import glob
import psutil

def fromnegative(row, maxcaparr):
    """Corrects negative capacity.

    Args:
        row (type): Description of parameter `row`.
        maxcaparr (type): Description of parameter `maxcaparr`.

    Returns:
        type: Corrected row.

    """
    out = row
    out['Capacity'] = (row["Output Capacity [mAh]"] + maxcaparr[row["Cycles Count"].iloc[0]])
    return out

def pkl_zip_write(object, filename, protocol = -1):
    """Creates .pkl object and saves it to file.

    Args:
        object (type): Object to be saved.
        filename (str): name of the new .pkl file.
        protocol (int): saving protocol.

    """
    file = gzip.GzipFile(filename, 'wb')
    cPickle.dump(object, file, protocol)
    file.close()

def pkl_zip_append(object, filename, protocol = -1):
    """Appends object to already existing .pkl file.

    Args:
        object (type): object to be appended.
        filename (str): name of the .pkl file to which object will be appended.
        protocol (int): saving protocol.

    """
    file = gzip.GzipFile(filename, 'ab')
    cPickle.dump(object, file, protocol)
    file.close()

def correct_cycling_data(df, by_process = "discgarge"):
    """Corrects cycling data.

    Args:
        df (pandas DataFrame): Pandas DataFrame.
        by_process (str): Description of parameter `by_process`.

    Returns:
        Pandas DataFrame: Pandas df with corrected data.

    """
    df.set_index('Charger Timestamp [s]', inplace=True, drop=False)
    df['Capacity'] = df['Output Capacity [mAh]']

    # gets max capacity of charge in pre last cycle
    last_cycle = df["Cycles Count"].max()
    if last_cycle > 1:
        condition = (df["Cycles Count"] == (
            last_cycle - 1)) & (df['Channel Run Status'] == 7)
    else:
        condition = (df["Cycles Count"] == (
            last_cycle)) & (df['Channel Run Status'] == 7)
    last_charge_c_max = df.loc[condition,
                               'Output Capacity [mAh]'].max()

    # condition for discharge (to be more exactly - condition for not charge)
    condition = (df['Output Capacity [mAh]'] <= 0) & (
        df['Channel Run Status'] != 7)

    # Gets max Capacity of each discarge cycle
    maxcapcharge = abs(df.loc[condition].groupby(
        'Cycles Count')['Output Capacity [mAh]'].min())
    # Corrects max capacity for last cycle - as discarche is not full
    maxcapcharge[last_cycle] = last_charge_c_max

    grouped = df.loc[condition].groupby('Cycles Count')
    df.loc[condition] = grouped.apply(
        lambda x: fromnegative(x, maxcapcharge)).values
    mask = (df['Channel Run Status'] == 7) & (df['Capacity'] < 0)
    df.loc[mask, 'Capacity'] = 0

    df['Old Capacity'] = df['Output Capacity [mAh]']
    df['Output Capacity [mAh]'] = df['Capacity']

    # Delete temporary columns of dataframe
    df = df.drop(['Capacity'], axis=1)
    return df

def correct_time(df, time_step = 5):
    """Corrects time.

    Args:
        df (Pandas DataFrame): Pandas DataFrame.
        time_step (int): Description of parameter `time_step`.

    Returns:
        Pandas DataFrame: Pandas DataFrame with correct time.

    """
    df['Timeshift'] = df['Charger Timestamp [s]'].shift(1)
    df['Time, s'] = df['Charger Timestamp [s]']
    df['addtime'] = 0
    addtime = 0
    logcount = df.loc[df['Charger Timestamp [s]'] < df['Timeshift'], 'Log Count'].values
    lenlc = len(logcount)

    for idx, lc in enumerate(logcount):
        if idx + 1 == lenlc:
            lcend = int(df.tail(1)['Log Count'].values) + 1
        else:
            lcend = logcount[idx+1]
        addtime = addtime + int(df.loc[df['Log Count'] == lc, 'Timeshift']) + time_step
        df.loc[(df['Log Count'] >= lc) & (df['Log Count'] < lcend), 'addtime'] = addtime
    df['Time, s'] = df['Time, s'] + df['addtime']
    df['Charger Timestamp [s]'] = df['Time, s']
    #Delete temporary columns of dataframe
    df.drop(['Timeshift','Time, s', 'addtime'], axis=1, inplace = True)
    return df

def correct_cycle_count(df):
    """Corrects cycle count.

    Args:
        df (Pandas DataFrame): Pandas DataFrame to be corrected.
    Returns:
        Pandas DataFrame: Corrected pandas DataFrame.

    """
    df = df[np.isfinite(df['Cycles Count'])]
    df['Cycleshift'] = df['Cycles Count'].shift(1)
    df['Cycle'] = df['Cycles Count']
    df['addcycle'] = 0
    addcycle = 0
    logcount = df.loc[df['Cycles Count'] < df['Cycleshift'], 'Log Count'].values
    lenlc = len(logcount)
    for idx, lc in enumerate(logcount):
        if idx + 1 == lenlc:
            lcend = int(df.tail(1)['Log Count'].values) + 1
        else:
            lcend = logcount[idx+1]
        addcycle = addcycle + int(df.loc[df['Log Count'] == lc, 'Cycleshift'])
        df.loc[(df['Log Count'] >= lc) & (df['Log Count'] < lcend), 'addcycle'] = addcycle
    df['Cycle'] = df['Cycle'] + df['addcycle']
    df['Cycles Count'] = df['Cycle']
    df.drop(['Cycleshift','Cycle', 'addcycle'], axis=1, inplace = True)
    return df

def memory_usage(print_it = True):
    """Prints how much memory is used.

    Args:
        print_it (bool): To print it, or not to print it: that is the question.

    Returns:
        str: mean usage(mb).

    """
    process = psutil.Process(os.getpid())
    mean_usage_mb = process.memory_info().rss / 1024 ** 2
    if print_it:
        print("Memory usage: {:03.2f} MB".format(mean_usage_mb))
    return mean_usage_mb

def load_cycling_data(path):
    """Loads DataFrame from .csv file.

    Args:
        path (str): path of .csv file.

    Returns:
        Pandas DataFrame: Pandas DataFrame.

    """
    columns = [
        "Log Count",
        "Charger Program Startup Count",
        "Charger Timestamp [s]",
        "Input Voltage [V]",
        "Output Voltage [V]",
        "Output Current [A]",
        "Output Power [W]",
        "Cell Voltage [V]",
        "Cell Int Resistance [mΩ]",
        "Cells Int Resistance [mΩ]",
        "Line Int Resistance [mΩ]",
        "Output Capacity [mAh]",
        "Internal Temperature [°C]",
        "External Temperature [°C]",
        "Cycles Count",
        "Channel Control Status",
        'Channel Run Status'
    ]
    df = pd.read_csv(path, sep=",", index_col=0, skiprows=23, names=columns)
    return df

def get_data_file_paths_list(rel_path, ext='txt', recursive=True):
    """Gets file paths in directory.

    Args:
        rel_path (str): relative path of directory to check.
        ext (str): extension of the files.
        recursive (bool): search recursively or not.

    Returns:
        str: list of filepaths.

    """
    # gets file paths list
    rel_path = os.path.join(
        '..',
        'Duomenys',
        rel_path
    )
    data_path = os.path.abspath(os.path.join(os.getcwd(), rel_path))
    if recursive:
        file_path_list = [y for x in os.walk(
            data_path) for y in glob.glob(os.path.join(x[0],  "*." + ext))]
    else:
        file_path_list = glob.glob(os.path.join(data_path, "*." + ext))
    return file_path_list

def sub_cycle_max_cap(row, maxcaparr):
    out = row
    out['Sub_Cycle_max_capacity'] = maxcaparr[row["Cycles Count"].iloc[0]]
    return out

def Cmax(row, maxcaparr):
    out = row
    """out['SOC'] = (row["Output Capacity [mAh]"] + maxcaparr[row["Counter"].iloc[0]])/maxcap
       we use row["Counter"].iloc[0] as row is DataFrame and not only row of DataFrame
       all row["Counter"] rows has the same value"""
    out['Cmax of cycle'] = (row["Output Capacity [mAh]"] + maxcaparr[row["Cycles Count"].iloc[0]])
    return out

def calculate_SOC(df):
    # for discharge
    df['Sub_Cycle_max_capacity'] = 0  # You need this row to avoid errors in cycle_max_cap function
    condition = (df['Output Current [A]'] < 0)
    grouped = df.loc[condition].groupby('Cycles Count')
    maxcapcharge = grouped['Output Capacity [mAh]'].max()
    df.loc[condition] = grouped.apply(lambda x: sub_cycle_max_cap(x, maxcapcharge)).values
    # for charge
    condition = (df["Output Current [A]"] > 0)
    grouped = df.loc[condition].groupby('Cycles Count')
    maxcapcharge = grouped['Output Capacity [mAh]'].max()
    df.loc[condition] = grouped.apply(lambda x: sub_cycle_max_cap(x, maxcapcharge)).values

    condition = (df['Sub_Cycle_max_capacity'] != 0)
    masked = df[condition]
    df.loc[condition, 'SOC'] = masked['Output Capacity [mAh]'] / masked['Sub_Cycle_max_capacity']
    return df

def calculate_SOH(df):
    capacity_maksimum = df['Output Capacity [mAh]'].max()
    condition = (df["Output Current [A]"] != 0)
    df['Sub_Cycle_max_capacity'] = 0  # You need this row to avoid errors in cycle_max_cap function
    grouped = df.loc[condition].groupby('Cycles Count')
    maxcapcharge = grouped['Output Capacity [mAh]'].max()

    df.loc[condition] = grouped.apply(lambda x: sub_cycle_max_cap(x, maxcapcharge)).values
    condition = (df['Sub_Cycle_max_capacity'] != 0)
    masked = df[condition]
    df.loc[condition, 'SOH'] = masked['Sub_Cycle_max_capacity'] / capacity_maksimum
    return df

def save(object, filename, protocol = -1):
    """Save an object to a compressed disk file.
       Works well with huge objects.
    """
    file = gzip.GzipFile(filename, 'wb')
    cPickle.dump(object, file, protocol)
    file.close()

def load(filename):
    """Loads a compressed object from disk
    """
    file = gzip.GzipFile(filename, 'rb')
    object = cPickle.load(file, encoding='latin1')
    file.close()
    return object

def pkl_zip_load(filename):
    """Loads a compressed object from disk
    """
    file = gzip.GzipFile(filename, 'rb')
    objs = {}
    while 1:
        try:
            objs.update(pd.read_pickle(file))
        except EOFError:
            break
    file.close()
    return objs

def filter_file_paths_list(file_path_list, regex):
    """filters file paths list"""
    regex = re.compile(regex)
    new_file_path_list = filter(regex.match, file_path_list)
    file_list = [path_leaf(w) for w in new_file_path_list]
    return new_file_path_list, file_list

def random_error(data, alpha):
    """Function to add std values from intial cycling before storage
       Calculates mean error as in http://www.fbml.ff.vu.lt/sites/default/files/I_knyga_-_2_skyrius_-_Matavimai_ir_matavimu_paklaidos_0.pdf"""
    t_bounds = stats.t.interval(alpha, len(data) - 1)
    err = stats.sem(data)*abs(t_bounds[0])
    return err

def calcdays(x):
    out = x
    startdate = out.date.min()
    out['days'] = (out.date - startdate).dt.days
    #print out['days'].head()
    return out

def storage_add_std(x, std_a, std_b, days_diff):
    cell_type = x['type'].iloc[0]
    soc_init = x['soc_init'].iloc[0]

    c_max_std_a = std_a['c_max', 'std'].loc[(cell_type, soc_init)]
    c_now_std_a = std_a['c_now', 'std'].loc[(cell_type, soc_init)]
    c_max_mean_a = std_a['c_max', 'mean'].loc[(cell_type, soc_init)]
    if days_diff != 0:
        c_max_std_b = std_b['c_max', 'std'].loc[(
            cell_type, soc_init)]/days_diff.loc[(cell_type, soc_init)]
        c_now_std_b = std_b['c_now', 'std'].loc[(
            cell_type, soc_init)]/days_diff.loc[(cell_type, soc_init)]
        x['c_max_std'] = c_max_std_a + c_max_std_b * x['days']
        x['c_now_std'] = c_now_std_a + c_now_std_b * x['days']

    else:
        x['c_max_std'] = c_max_std_a
        x['c_now_std'] = c_now_std_a

    c_max_init = unumpy.uarray(c_max_mean_a, c_max_std_a)
    c_now = unumpy.uarray(x['c_now'], x['c_now_std'])
    c_max = unumpy.uarray(x['c_max'], x['c_max_std'])

    soh = c_max/c_max_init*100.
    soc = c_now/c_max*100.
    x['soh'] = unumpy.nominal_values(soh)
    x['soc'] = unumpy.nominal_values(soc)
    x['soh_std'] = unumpy.std_devs(soh)
    x['soc_std'] = unumpy.std_devs(soc)
    return x

def storage_replace_by_means(df, stage):
    df2 = df[df['stage'] == stage]
    if len(df2) > 1:
        data = {
            'type': df2.type.iloc[0],
            'cell_nr': df2.type.iloc[0],
            'soc_init': df2.soc_init.astype(float).mean(),
            'date': df2.date.min(),
            'days': df2.days.min(),
            'soc': df2.soc.astype(float).mean(),
            'c_max': df2.c_max.mean(),
            'c_now': df2.c_now.mean(),
            'c_max_init': df2.c_max_init.mean(),
            'soh': df2.soh.mean(),
            'soh_std': df2.soh_std.mean(),
            'soc_std': df2.soc_std.mean(),
            'stage': df2.stage.mean(),
            'c_now_std': df2.c_now_std.mean(),
            'c_max_std': df2.c_max_std.mean(),

        }
        out = pd.DataFrame(data=data, index=[0])
    else:
        out = pd.DataFrame()
    return out

def combine_storage_data(temp, include_after = True):
    """Writes initial value of SoC and Capacity from data
    before storage to data after storage"""

    data_folder_name = 'Before_storage'
    path = os.path.abspath(os.path.join(os.getcwd(), '..', 'Duomenys_preprocesinti', 'summary_' +
                                        data_folder_name.lower() + '_' + str(temp) + 'C.pkl'))
    df_before = load(path)
    if include_after:
        data_folder_name = 'After_storage'
        path = os.path.abspath(os.path.join(os.getcwd(), '..', 'Duomenys_preprocesinti', 'summary_' +
                                            data_folder_name.lower() + '_' + str(temp) + 'C.pkl'))
        df_after = load(path)
        if df_after.empty:
            print('After excluded')
            include_after = False
    data_folder_name = 'Storage'
    path = os.path.abspath(os.path.join(os.getcwd(), '..', 'Duomenys_preprocesinti', 'summary_' +
                                        data_folder_name.lower() + '_' + str(temp) + 'C.pkl'))
    df_storage = load(path)

    df_before['date'] = pd.to_datetime(df_before['date'], format='%Y-%m-%d')
    df_storage['date'] = pd.to_datetime(df_storage['date'], format='%Y-%m-%d')

    df_storage = df_storage.drop(['soc_init'], axis=1)
    df_storage = df_storage.drop(['c_max_init'], axis=1)
    df_storage = pd.merge(df_storage,
                          df_before[['cell_nr', 'soc_init', 'c_max_init']],
                          on='cell_nr',
                          how='left')

    df_before['stage'] = 0
    df_storage['stage'] = 1

    if include_after:
        df_after['date'] = pd.to_datetime(df_after['date'], format='%Y-%m-%d')
        df_after = df_after.drop(['soc_init'], axis=1)
        df_after = df_after.drop(['c_max_init'], axis=1)
        df_after = pd.merge(df_after,
                        df_before[['cell_nr', 'soc_init', 'c_max_init']],
                        on='cell_nr',
                        how='left')
        df_after['stage'] = 2

    df_summary = df_before.append([df_storage, df_after], ignore_index=True)
    df_summary['c_max_std'] = 0
    df_summary['c_now_std'] = 0
    df_summary['soh_std'] = 0
    df_summary['soc_std'] = 0
    df_summary['soh'] = 0

    grouped_summary = df_summary.groupby(['type', 'soc_init'])
    df_summary = grouped_summary.apply(lambda x: calcdays(x))
    grouped_summary = df_summary.groupby(['type', 'soc_init'])

    stat_before = df_before.groupby(['type', 'soc_init']).describe()

    date_stat_before = df_before.groupby(['type', 'soc_init'])['date'].min()

    if include_after:
        stat_after = df_after.groupby(['type', 'soc_init']).describe()
        date_stat_after = df_after.groupby(['type', 'soc_init'])['date'].max()
        stat_diff = stat_after - stat_before
        days_diff = (date_stat_after - date_stat_before).dt.days
    else:
        stat_diff = 0
        days_diff = 0
    std_b = stat_diff
    std_a = stat_before
    df_summary = grouped_summary.apply(
        lambda x: storage_add_std(x, std_a, std_b, days_diff))
    grouped_summary = df_summary.groupby(['type', 'soc_init'])
    # print list(df_summary)
    df_before_one_row = grouped_summary.apply(
        lambda x: storage_replace_by_means(x, 0))
    if include_after:
        df_after_one_row = grouped_summary.apply(
            lambda x: storage_replace_by_means(x, 2))


    df_summary = df_summary.loc[df_summary.stage == 1]
    if include_after:
        df_summary = df_summary.append([df_before_one_row.reset_index(
            drop=True), df_after_one_row.reset_index(drop=True)])
    else:
           df_summary = df_summary.append([df_before_one_row.reset_index(
            drop=True)])
    df_summary.reset_index(drop=True).head(10000)
    df_summary['delta_soc'] = df_summary['soc'] -         df_summary['soc_init'].astype(float)
#     print df_summary['delta_soc']
    soh = unumpy.uarray(df_summary['soh'],df_summary['soh_std'])
    soh_100 = unumpy.uarray(100., df_summary['soh_std'] )
    delta_soh =  soh_100 - soh

    soc = unumpy.uarray(df_summary['soc'],df_summary['soc_std'])
    soc_init = unumpy.uarray(df_summary['soc_init'], df_summary['soc_std'] )
    delta_soc =  soc_init - soc

#     c_max = unumpy.uarray(df_summary['soc'],df_summary['soc_std'])
#     soc_init = unumpy.uarray(df_summary['soc_init'], df_summary['soc_std'] )
#     delta_soc =  soc_init - soc

    df_summary['delta_soh_std'] = unumpy.std_devs(delta_soh)
    df_summary['delta_soh'] = unumpy.nominal_values(delta_soh)

    df_summary['delta_soc_std'] = unumpy.std_devs(delta_soc)
    df_summary['delta_soc'] = unumpy.nominal_values(delta_soc)

    #df_summary['delta_soh_std'] = df_summary['soh_std']*math.sqrt(2)
    df_summary['delta_c'] = df_summary['c_max'] - df_summary['c_max_init']
    df_summary['delta_c_std'] = df_summary['c_max_std']*math.sqrt(2)
    path = os.path.abspath(os.path.join(os.getcwd(),
                                        '..', 'Duomenys_preprocesinti', 'summary_storage_' + str(temp) + 'C.pkl'))

    save(df_summary.reset_index(drop=True), path)
    print('Done')
    return df_summary

def mfilter (df, std, sigmaf = 1):
    for index, row in df.iterrows():
        if row['ay'] < std['ay'] * sigmaf and  row['ay'] > -std['ay'] * sigmaf:
            df.set_value(index, 'ay', 0.)
        if row['ax'] < std['ax'] * sigmaf and  row['ax'] > -std['ax'] * sigmaf:
            df.set_value(index, 'ax', 0.)
        if row['az'] < std['az'] * sigmaf and  row['az'] > -std['az'] * sigmaf:
            df.set_value(index, 'az', 0.)
        if row['gy'] < std['gy'] * sigmaf and  row['gy'] > -std['gy'] * sigmaf:
            df.set_value(index, 'gy', 0.)
        if row['gx'] < std['gx'] * sigmaf and  row['gx'] > -std['gx'] * sigmaf:
            df.set_value(index, 'gx', 0.)
        if row['gz'] < std['gz'] * sigmaf and  row['gz'] > -std['gz'] * sigmaf:
            df.set_value(index, 'gz', 0.)
    return df

def vp (df,inittime = 0.02):
    """ Funkcija greičiui ir pozicijai apskaičiuoti """
    df.set_value(0, 'vx', 0)
    df.set_value(0, 'vy', 0)
    df.set_value(0, 'vz', 0)
    df.set_value(0, 'px', 0)
    df.set_value(0, 'py', 0)
    df.set_value(0, 'pz', 0)
    df.loc[inittime:,['vx']] = integrate.cumtrapz(df['ax'], df.index) * 9.81
    df.loc[inittime:,['vy']] = integrate.cumtrapz(df['ay'], df.index) * 9.81
    df.loc[inittime:,['vz']] = integrate.cumtrapz(df['az'], df.index) * 9.81
    df.loc[inittime:,['px']] = integrate.cumtrapz(df['vx'], df.index)
    df.loc[inittime:,['py']] = integrate.cumtrapz(df['vy'], df.index)
    df.loc[inittime:,['pz']] = integrate.cumtrapz(df['vz'], df.index)
    return df

def angles (df, vid):
    """ Funkcija kampams apskaičiuoti """
    pitch = 0.
    roll = 0.
    yaw = 0.
    pitch = 0.
    apitch_init = math.atan2(vid["ax"]*9.81, vid["az"]*9.81)
    aroll_init =  math.atan2(vid["ay"]*9.81, vid["az"]*9.81)
    for index, row in df.iterrows():
        apitch = math.atan2(row['ax']*9.81, row['az']*9.81)
        aroll = math.atan2(row['ay']*9.81, row['az']*9.81)
        gpitch = pitch + row['gx']*0.02
        groll = roll + row['gy']*0.02
        gyaw = row['gz']*0.02
        # complementary filter coeficient calculation a = T/(T+dt), T = time_constant, dt = update_rate
        pitch = gpitch*0.97 + apitch*0.03
        roll = groll*0.97 + aroll*0.03
        yaw = yaw + gyaw
        df.set_value(index, 'apitch', apitch)
        df.set_value(index, 'aroll', aroll)
        df.set_value(index, 'gpitch', gpitch)
        df.set_value(index, 'groll', groll)
        df.set_value(index, 'gyaw', gyaw)
        df.set_value(index, 'pitch', ((apitch_init+pitch-math.pi)*180/math.pi))
        df.set_value(index, 'roll', (aroll_init+roll+math.pi)*180/math.pi)
        df.set_value(index, 'yaw', yaw*180/math.pi)
    return df


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    return signal.butter(order, normal_cutoff, btype='high', analog=False)

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    return signal.butter(order, normal_cutoff, btype='low', analog=False)

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    return signal.lfilter(b, a, data)

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def rolling(df, w, label_col, roll_col = None, label_method = "mean"):
    """Windows roll on DataFrame and creates new DataFrame with more columns

    Args:
        df -- data (pandas.DataFrame)
        w -- window size (int)
        label_col -- name of column to aggregate while doing rolling on other columns (string)
        roll_col -- name of columns to do rolling (list of strings). If None takes all columns in df except label_col.
        label_method -- method to aggragate label_col. ("mean", "last", "first")
    Return:
        pandas.DataFrame with rolled columns and aggregated label column in index
    """
    if roll_col is None:
        roll_col = df.columns[df.columns != label_col]
    if df.shape[0] > w:
        dfr = pd.concat([df[roll_col].shift(periods = i) for i in range(w)[::-1]], axis = 1).iloc[w-1:]
        if label_method == "last":
            dfr.index = df[label_col].iloc[w-1:]
        elif label_method == "first":
            dfr.index = df[label_col].iloc[:-(w-1)]
        else:
            dfr.index = df[label_col].rolling(w).mean()[w-1:]
        return dfr
    else:
        df.index = df[label_col]
        return df[roll_col]
def group_rolling(df, window, gr, label_col, roll_col=None, label_method = "mean", remove = True):
    """Apply rolling function to datafarme by groups

    Args:
        df -- data (pandas.DataFrame)
        window -- window size (int)
        gr -- name of group column (string)
        label_col -- name of column to aggregate while doing rolling on other columns (string)
        roll_col -- name of columns to do rolling (list of strings). If None takes all columns in df except label_col.
        label_method -- method to aggregate label_col. ("mean", "last", "first")
        remove -- If grouped dataframe doesn't have enough instances, should they be removed (bool, Default - True)
    Return:
        multi index pandas.DataFrame with rolled columns and aggregated label column in second index
            first index is group column values
    """
    grsize = df.groupby(gr).apply(lambda D: D.shape[0] > window)
    if grsize.all():
        return df.groupby(gr).apply(lambda D: rolling(D, window, label_col, roll_col, label_method))
    elif remove:
        df1 = df[~df[gr].isin(grsize[~grsize].index)]
        print("\033[34;1m Cycles {} were removed \033[0m".format(list(grsize[~grsize].index)))
        return df1.groupby(gr).apply(lambda D: rolling(D, window, label_col, roll_col, label_method))
    else:
        print("\033[31;1m Error: There is group where window size is greater then number of instances \033[0m")
# def rolling_windows(
#     df,
#     window_size,
#     column_for_y,
#     columns_for_x,
#     column_for_group = 'None'
# ):
#     columns_all = columns_for_x + [column_for_y]
#     if column_for_group != 'None':
#         columns_all += [column_for_group]
#         df = df[columns_all].groupby(column_for_group)
#     #duplicates rows so we have all windows for rolling
#     # by this method all possible rolling windows are created and let have equal number of point in each window
#     df_rolled_windows = pd.concat([df
#              .apply(lambda x: pd
#                  .concat([x.iloc[i: i + window_size] for i in range(len(x.index) - window_size + 1)]))])
#     if column_for_group != 'None':
#         del df_rolled_windows[column_for_group]
#     # make rolling over dataframe that has all windows and calculates mean
#     df_rolled_windows_mean = df_rolled_windows.rolling(
#         window_size, min_periods=window_size).mean()
#     # takes each window_size row from rolled mean of dataframe that has all windows
#     # so we have mean of each rolled window
#     Y = df_rolled_windows_mean.iloc[window_size-1::window_size, :]
#     # makes Y_fit as numpy array of labels
#     Y = Y[column_for_y].values.flatten()
#     del df_rolled_windows[column_for_y]
#     # makes numpy array with data splitted into windows with window_size rows
#     X = np.array(np.vsplit(df_rolled_windows.iloc[0:len(
#         df_rolled_windows)].values, len(df_rolled_windows)/window_size))
#     # reshape numpy array so that all data of window is in one row
#     X = X.reshape(X.shape[0], X.shape[1]*X.shape[2])
#     return X, Y
