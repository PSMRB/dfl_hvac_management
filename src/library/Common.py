# Description: This file contains all the functions and classes used in SEVERAL other files.
import errno  # for FileNotFoundError

from copy import deepcopy
import calendar
import random
import tempfile
from datetime import datetime, timedelta
# to be able to get timezone from string like "Europe/Paris"
from zoneinfo import ZoneInfo
import numpy as np
import os
import pandas as pd
import torch

from src.library.Building import BuildingModel
from src.library.Classes import Database
from src.library.Base import getcolumnname

import scienceplots
import shlex
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import time


def flattenlist(listoflists):
    """
    Flatten a list of lists. Specifically, there will not be any nested list left but all the other elements
    (np.array, pd.DataFrame, strings, etc.) will be unchanged.
    :param listoflists: the list to be flattened
    :return: the flattened list
    """
    flatlist = []
    for sublist in listoflists:
        if not isinstance(sublist, list):
            flatlist.append(sublist)
        else:  # sublist is a list
            for item in sublist:
                flatlist.append(item)

    return flatlist


def splitdate(dates):
    """
    Split an iterable of string into meaningful elements
    :param dates: an iterable of string formatted as "DD/MM hh:mm" or an iterable of pd.Timestamp objects
    :return:
    """
    days = []
    months = []
    hours = []
    minutes = []
    if type(dates[0]) == str:
        for date in dates:
            months.append(int(date[:2]))
            days.append(int(date[3:5]))
            hours.append(int(date[6:8]))
            minutes.append(int(date[9:]))
    elif type(dates[0]) == pd.Timestamp:
        for date in dates:
            months.append(date.month)
            days.append(date.day)
            hours.append(date.hour)
            minutes.append(date.minute)
    else:
        raise TypeError("The type of the dates is not recognized. It should be a list of string or pd.Timestamp"
                        " objects.")

    return months, days, hours, minutes


def getcumulativehours(dates):
    """
    Return the number of hours given a date
    :param date: a string formated as "DD/MM hh:mm"
    :return:
    """
    nhourpermonth = [0, 744, 1416, 2160, 2880, 3624, 4344, 5088, 5832, 6552, 7296, 8016]
    X = splitdate(dates)
    M, D, h, m = map(np.array, X)
    hours = (D - 1) * 24 + np.array([nhourpermonth[M_i - 1] for M_i in M]) + h + m * 100 / 60

    return hours


# Create lag into data
def lag_data(data: pd.DataFrame, lags):
    """

    :param data: is the df containing the variables of interest. The first index is assumed to be 0.
    :param lags: is the list with the lags to apply
    :return: the data with every single variable lagged by each lag belonging to lags (and the original variables)
    """
    if data.index.values[0] != 0:
        raise KeyError(f"The index of the dataframe passed as parameter do not start at 0.")

    data_lag = data
    # lag the data and add (t-lag) to the column name
    for t in lags:
        data_tmp = data.iloc[:-t, :]
        data_tmp.index = data_tmp.index + t
        data_tmp.columns = [n[:-5] + f"(t-{t})" for n in list(data_tmp.columns)]
        data_lag = pd.concat([data_lag, data_tmp], axis=1)

    return data_lag.dropna()  # remove the last rows which contain NaN


def renamecolumns(data: pd.DataFrame):
    """
    Rename the columns of the dataframe data.
    :param data:
    :return: No need to return data since the modifications of dataframe (and list) affect the original directly.
    """
    new_labels = []
    for col in data.columns:
        # if column name does not contain time step information (t-0, t-1, etc.)
        if '(t-' not in col:
            # Remove the "(Hourly)" or "(TimeStep)" part of the column name
            if "(Hourly)" in col:
                tmp = col.split("(Hourly)")
            else:
                tmp = col.split("(TimeStep)")

            # Simplify the name (if not in list of variables to keep) and add the time step information
            if col not in ['Cooling:Electricity[Wh]', 'Heating:Electricity[Wh]', "Electricity:Facility [Wh](TimeStep)"]:
                tmp = tmp[0].split(":")
                new_labels.append(tmp[-1] + "(t-0)")
            else:
                new_labels.append(tmp[0] + "(t-0)")
        else:
            new_labels.append(col)
    data.columns = new_labels


def modeltrainandtest(db: Database, model):
    """
    Function which train and test a model. The MSE, MAE and R2 are computed and printed.
    :param db:
    :param model:
    :return: the model
    """
    # Fit the model
    start_time = time.time()
    model.fit(db.X_train, db.y_train_true)
    training_time = time.time() - start_time
    metrics = {"training_time": training_time}

    # print statistical scores
    metrics.update(computemetrics(db, model))
    printmetrics(metrics)

    return model, metrics


def createdir(dir_path, up=0):
    """
    Create a directory if it does not exist. If up is specified, it creates the parent directory (up=1), the grandparent
    directory (up=2), etc.
    :param dir_path:
    :param up:
    :return:
    """
    for u in range(up, -1, -1):
        base_path = path_up(dir_path, u)
        if not os.path.exists(base_path):
            try:
                os.mkdir(base_path)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
                time.sleep(1)  # wait for the directory to be created
    return base_path


def readvarseso(folder_path: str):
    """
    Read the eplusout.eso file in the folder_path and convert it to a csv file.
    ATTENTION: For some obscure reason, the number of variables in the csv file is limited to 256.
    :param folder_path:
    :return:
    """
    ReadVarsESO_filepath = "/Applications/EnergyPlus-22-2-0/PostProcess/ReadVarsESO"
    eso_filepath = os.path.join(folder_path, "eplusout.eso")
    # rename eso file to eplusout.eso because it is the only file name understood by ReadVarsESO
    fileindir = os.listdir(folder_path)
    current_eso_filepaths = [f for f in fileindir if ".eso" in f]
    # If there are several eso file, remove a former eplusout.eso file
    if len(current_eso_filepaths) > 1:
        # remove a potential old version of eplusout.eso
        if os.path.exists(eso_filepath):
            os.remove(eso_filepath)
            fileindir = os.listdir(folder_path)
            current_eso_filepaths = [f for f in fileindir if ".eso" in f]
    if len(current_eso_filepaths) == 1:
        current_eso_filepath = os.path.join(folder_path, current_eso_filepaths[0])
        os.rename(current_eso_filepath, eso_filepath)
        os.system(f"""osascript -e 'tell application "Terminal" to do script "cd {shlex.quote(folder_path)};
        {ReadVarsESO_filepath}"'""")
    elif len(current_eso_filepaths) == 0:
        raise FileNotFoundError("No eso file in this directory.")
    else:  # len(eso_filename) > 1:
        raise FileNotFoundError("Several eso file in this directory. There should be only one.")


def create_clean_eplusoutcsv(buildingmodel: BuildingModel, skipcsv=False):
    """
    This file is to be used as a subfile for reading eplusout.eso files, convert them to .csv, clean them and save them
    as an .xlsx file for comfort.
    This script is a part of SensibleLoadActuator_vx.py and of GenerateDatasets.py
    :param folder_path: ABSOLUTE folder_path to the directory containing the eplusout.eso file. The file eplusout.csv
                        will be stored there too.
                        The path must be absolute because python and the terminal do not consider the same working
                        directory.
    :param frequency:
    :param zone_names: a list of the zone_names names
    :param skipcsv: if True, skip the generation of the csv file. The file must be named eplusout.csv and be in the
                    folder_path.
    :return:
    """

    ########## 1. Generate the eplusout.csv ##########
    folder_path = buildingmodel.output_folderpath
    if not skipcsv:
        readvarseso(folder_path)

        ########## 2. Gather the csv files into an excel ##########
        #### 2.1 Read the csv files

        # Read eplusout.csv
        eplusout_filepath = os.path.join(folder_path + "/eplusout.csv")
        eplusout = pd.read_csv(eplusout_filepath, header=0)

        # As long as we do not get the full optimization (one year and sizing days), we read again
        while eplusout.shape[0] < 8760 * buildingmodel.simulationfrequency + buildingmodel.nb_warmupts:
            time.sleep(2)  # wait for the ReadVarsESO to finish
            eplusout = pd.read_csv(eplusout_filepath, header=0)
    else:
        eplusout = buildingmodel.fullsimulation

    ### Add new columns
    # Add the cumulative solar radiation
    col = list(eplusout.columns)
    diff_rad = getcolumnname("Site Diffuse Solar Radiation Rate per Area", col)
    dir_rad = getcolumnname("Site Direct Solar Radiation Rate per Area", col)
    index = max(eplusout.columns.get_loc(diff_rad),
                eplusout.columns.get_loc(dir_rad))
    eplusout.insert(index + 1, f"Site Direct+Diffuse Solar Radiation Rate per Area, ENVIRONMENT",
                    eplusout[diff_rad] + eplusout[dir_rad])

    # Compute the average air temperature in the building
    zmat_col = getcolumnname(f"Zone Mean Air Temperature", col)
    zmat_col = [zmat_col] if type(zmat_col) == str else zmat_col
    eplusout["Building Mean Air Temperature"] = eplusout.loc[:, zmat_col].mean(axis=1)

    # Convert to Wh rather than J
    col = list(eplusout.columns)
    col_to_convert = [c for c in col if (("energy" in c.lower() or "electricity" in c.lower() or "Heating:NaturalGas"
                                          in c) and not ("rate" in c.lower()))]
    column_name_to_convert = flattenlist([getcolumnname(ctc, col) for ctc in col_to_convert])
    eplusout = convertJtoWh(eplusout, column_name_to_convert)

    # remove all monthly measurements (there shoudn't be any)
    column_names = list(eplusout.columns)
    for cn in column_names:
        if "(Monthly)" in cn:
            eplusout.drop(columns=[cn], inplace=True)

    # Round numbers
    eplusout = pd.concat([eplusout.iloc[:, :2], round(eplusout.iloc[:, 2:], 2)], axis=1)

    # Shift all the zone data which is not the temperature or the relative humidity
    for col in eplusout.columns:
        if (sum([ping in col.lower() for ping in ["energy", "rate", "electricity", "naturalgas"]]) >= 1 and
                "Solar" not in col):
            eplusout[col] = eplusout[col].shift(-1)

    # Remove the last row which is NaN
    check = eplusout.iloc[-1, :]
    eplusout.drop(eplusout.tail(1).index, inplace=True)

    return eplusout


def computemetrics(db, reg, y_hat=None):
    """
    Compute metrics for the regression
    :param db:
    :param reg:
    :param y_hat:
    :return: a dictionary with the metrics
    """
    if y_hat is None:
        y_hat = reg.predict(db.X_test)
    r_squared = reg.score(db.X_test, db.y_test_true)
    mae = mean_absolute_error(db.y_test_true, y_hat)
    mse = mean_squared_error(db.y_test_true, y_hat)
    metrics = {"R2": r_squared, "MAE": mae, "MSE": mse}
    return metrics


def compute_metrics(y_hat, y_true):
    """
    Given a vector of predictions (hat) and true targets, compute a bunch of error metrics.
    :return: a dictionary with the metrics
    """
    y_hat = np.array(y_hat)
    y_true = np.array(y_true)
    err = y_true - y_hat
    mu_err = np.mean(err)
    std_err = np.std(err)
    mae = mean_absolute_error(y_true, y_hat)
    mse = mean_squared_error(y_true, y_hat)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs(err / y_true)) * 100
    r2 = 1 - np.sum(err ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    metrics = {"MAE": mae, "MSE": mse, "RMSE": rmse, "MAPE": mape, "R2": r2, "mu_err": mu_err, "std_err": std_err}
    return metrics


def printmetrics(metrics: dict):
    # print("\u0332".join("\nMetrics for 1 time step ahead forecast:"))
    print(f"mean MAE = {np.round(metrics['MAE'], decimals=3)}")
    print(f"mean MSE = {np.round(metrics['MSE'], decimals=3)}")
    print(f"mean R\N{SUPERSCRIPT TWO} = {np.round(metrics['R2'], decimals=3)}")


def path_up(path, n):
    """
    Go up n times in the path
    :param path:
    :param n: number of times to go up
    :return:
    """
    for _ in range(n):
        path = os.path.dirname(path)
    return path


def normalizedata(data: pd.DataFrame, resp_name: list):
    """
    Normalize the data
    :param data:
    :param resp_name: is a list of strings
    :param rnd:
    :return:
    """
    # Normalize input_batch
    scaler_input, scaler_output = StandardScaler(), StandardScaler()  # Standardize the data to have mean = 0 and variance 1
    data_wo_output = data.drop(resp_name, axis=1, inplace=False)
    input_norm = scaler_input.fit_transform(data_wo_output)  # input_norm is a numpy array
    output_norm = scaler_output.fit_transform(data[resp_name])  # output_norm is a numpy array
    scalers = {"input": scaler_input, "output": scaler_output}
    data_norm = pd.DataFrame(np.concatenate((input_norm, output_norm), axis=1),
                             columns=data_wo_output.columns.tolist() + resp_name)
    return data_norm, scalers


def convertJtoWh(df: pd.DataFrame, column_names: list):
    """
    Convert the column in Joules to Wh
    :param df: pd.DataFrame which is modified
    :param column_names: the name of the columns to convert
    :return:
    """
    # For performance issues:
    #   1) get the list of the columns to convert
    #   2) create a new dataframe with the columns already existing
    #   3) Write the new columns and drop the old columns
    #   4) Concatenate the new df with the old one
    new_column_names = []
    for cn in column_names:
        column_name_new = cn.split("[J]")[0] + "[Wh]"
        if "[J]" in cn:
            column_name_new += cn.split("[J]")[1]
        new_column_names.append(column_name_new)

    new_df = pd.DataFrame(index=df.index, columns=new_column_names)
    for cn, cn_new in zip(column_names, new_column_names):
        new_df[cn_new] = df[cn] / 3600
        df = df.drop(columns=[cn], inplace=False)
    df = pd.concat([df, new_df], axis=1)

    return df


def from_dic_to_numpy(dic: dict) -> np.ndarray:
    """
    Convert a dictionary whose keys are integers similar to 1D-array indices into a numpy array
    :param dic:
    :return:
    """
    # Determine the shape of the array based on the maximum indices in the keys
    max_row, max_col = max(dic.keys())
    # Create the array filled with zeros (or any default value)
    array_shape = (max_row + 1, max_col + 1)
    array = np.zeros(array_shape)
    # Fill the array with values from the dictionary using the tuple keys
    for key, value in dic.items():
        array[key] = value
    return array


def floor(v):
    return int(np.floor(v))


def floor_to_half_integer(value):
    """
    floor to the closest half integer. e.g. 1.2 -> 1.0, 1.7 -> 1.5, 1.8 -> 2.0
    :param value:
    :return:
    """
    floor_part = np.floor(value)
    fractional_part = abs(value - floor_part)
    if fractional_part >= 0.5:
        return floor_part + 0.5
    else:
        return floor_part


def ceil_to_half_integer(value):
    """
    ceil to the closest half integer. e.g. 1.2 -> 1.5, 1.7 -> 2.0, 1.8 -> 2.0
    :param value:
    :return:
    """
    ceil_part = np.ceil(value)
    fractional_part = abs(ceil_part - value)
    if fractional_part >= 0.5:
        return ceil_part - 0.5
    else:
        return ceil_part


def getzoneidsuffix(zone_names: list):
    """
    Get the suffix of the zone id. For example, if the zone name is "THERMAL ZONE LIVING SPACE", the suffix is "LIVING
    SPACE".
    :param zone_names: a list containing the names of the zone_names as strings
    :return:
    """
    return [str.lower(z.replace(" ", "")) for z in zone_names]


def getbestmodel(summarypath, constraints, metric='R2_24h', sense='max'):
    """
    Function to read excel file which summarizes model performances and return the datetime of the best model.
    :param path: path to the summary file containing the metrics of the models
    :param constraints: a dict containing the constraints on the model (e.g. its architecture)
    :param metric: string : the metric to use to select the best model. The name of the df column
    :param sense: string : 'max' or 'min' of the metric to select the best model
    :return: datetime of the best model as a string
    """
    sparse = constraints.get("sparse", None)
    if sparse is not None:
        sheet_name = "NotSparse" if sparse is False else "Sparse"
        df = pd.read_excel(summarypath, sheet_name=sheet_name, header=0)
    else:
        df = pd.read_excel(summarypath, header=0)
    df.set_index(df.columns[0], inplace=True, drop=True)
    # filter the df
    for k, v in constraints.items():
        df = df[df[k] == v]

    if len(df) == 0:
        raise ValueError("No model found with the given constraints")
    elif len(df) == 1:
        datetime = df.index[0]
    else:
        datetime = df[metric].idxmax() if sense == 'max' else df[metric].idxmin()

    return datetime


def get_time(api, state, year=2017, tzinfo="UTC"):
    """
    function to get the time from E+ (00:01 till 24:00) and convert it to standard format (from 00:00 to 23:59)
    :param api: an E+ api object
    :param state: A state as returned by the E+ API
    :param year: the year to use to convert E+ Timestep to datetime
    :return:
    """

    minutes = api.exchange.minutes(state) % 60
    hour = api.exchange.hour(state) + api.exchange.minutes(state) // 60
    day = api.exchange.day_of_month(state)
    month = api.exchange.month(state)

    # Convert 24:00 to 00:00
    if hour == 24:
        hour = 23
        # record the date and time in a datetime object (year arbitrarily set to 2000)
        my_datetime = datetime(year, month, day,
                               hour, minutes)
        my_datetime = my_datetime + timedelta(hours=1)
    else:
        # record the date and time in a datetime object (year arbitrarily set to 2000)
        my_datetime = datetime(year, month, day,
                               hour, minutes)
    my_datetime_str = my_datetime.strftime("%m/%d %H:%M")

    return my_datetime.replace(tzinfo=ZoneInfo(tzinfo)), my_datetime_str


def closest_datetime_before(datetime_vector, reference_datetime):
    """
    Given a vector of datetime, return the closest datetime before the reference datetime.
    :param datetime_vector:
    :param reference_datetime:
    :return:
    """
    # compute the time difference between the reference datetime and the datetime vector
    delta = datetime_vector - reference_datetime
    # keep only the negative time differences (i.e. the datetime before the reference datetime)
    delta = delta[delta <= timedelta(0)]
    # return the datetime which is the closest to the reference datetime (since timedelta are negative, the closest
    # datetime is the one with the maximum time difference = the closest to 0)
    return reference_datetime + max(delta)


def create_datetime_from_Eplus_timestep(timestep, year=2017, tzinfo="UTC"):
    """
    Create a datetime object from an E+ timestep
    :param timestep: an integer
    :param year: an integer
    :param tzinfo: a string
    :return:
    """
    try:
        parsed_datetime = datetime.strptime(timestep, "%m/%d %H:%M").replace(year=year)
    except ValueError:
        # if the timestep is 02/29, convert it to 03/01
        if timestep.month == 2 and timestep.day == 29 and not calendar.isleap(year):
            timestep[:5] = "03/01"
            year += 1
            parsed_datetime = datetime.strptime(timestep, "%m/%d %H:%M").replace(year=year)
        print(timestep)
        debug = 1
    my_datetime = parsed_datetime.replace(tzinfo=ZoneInfo(tzinfo))
    return my_datetime


def map_dict(d, func):
    """
    Apply a function to all the values of a dictionary
    :param d: a dictionary
    :param func: a function
    :return:
    """
    return {k: func(v) for k, v in d.items()}


def iter_dic(d, f, *args, inplace=True, **kwargs):
    """
    Recursive function to iterate over a dictionary and apply a function to each leaf value.
    Args:
        d:
        f:
        *args, **kwargs: extra arguments of the function f
    Returns:

    """
    if not inplace:
        d = deepcopy(d)
    for k, v in d.items():
        if isinstance(v, dict):
            iter_dic(v, f, *args, inplace=True, **kwargs)
        else:
            d[k] = f(v, *args, **kwargs)
    return d


def agg_itr_of_dict(dict_itr, fn=lambda x: x, *args, **kwargs):
    """
    Return an aggregated list of values from a list of dictionaries
    :param dict_itr: an iterable of dictionaries (e.g. a list of dictionaries)
    :param fn: the function to apply to the list which contains all the values that are associated with a given key.
        E.g. mean return the means of all the values that were with the kay 'A' through all the dictionaries.
        Per default, the function is the identity function so nothing is done.
    :param args: additional arguments to pass to the function fn
    :param kwargs: additional keyword arguments to pass to the function fn
    :return: a dictionary with the same keys as the dictionaries in dict_itr and the values are the result of the
        function fn applied to the list of values associated with the key.
    """
    # agg_dict = {}  # aggregated dictionary
    # # transform the list of dictionaries into a dictionary of lists
    # for d in dict_itr:
    #     for k, v in d.items():
    #         if k not in agg_dict:
    #             agg_dict[k] = [v]
    #         else:
    #             agg_dict[k].append(v)
    df = pd.DataFrame(dict_itr)
    agg_dict = df.to_dict(orient="list")
    # apply the function to the dictionary of lists
    for k, v in agg_dict.items():
        agg_dict[k] = fn(v, *args, **kwargs)
    return agg_dict


def make_reproducible(seed: int = None):
    """
    Function to make the code reproducible.
    :param seed: an integer for the seed. If not specified, a random seed is chosen.
    :return: the seed, the worker seed function and the generator. The last two are to be used in the dataloader.
    """
    seed = random.randint(0, 10000) if seed is None else seed
    random.seed(seed)  # to make python code reproducible
    np.random.seed(seed)  # to make numpy code reproducible
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():  # for people with NVIDIA GPU
        torch.cuda.manual_seed_all(seed)

    ### For the dataloader
    # 1) the worker_init_fn must be set to the following function
    def worker_init_fn(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    # 2) the dataloader must be created with the following argument: generator=seed_worker
    generator = torch.Generator()
    generator.manual_seed(seed)

    return seed, worker_init_fn, generator


def get_available_device():
    """
    Function to get the available device (GPU or CPU)
    :return: the device
    """
    device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    return device


def save_newline_excel(filepath: str, newline: dict, index_col: int = 0, header: int = 0):
    """
    Save a new line in an excel file. If the file does not exist, create it.
    :param filepath: the path to the excel file
    :param newline: the dictionary containing the new line
    :return:
    """
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    # load the excel file if it exists, otherwise create a new one
    try:
        df = pd.read_excel(filepath, index_col=index_col, header=header)
    except FileNotFoundError:
        df = pd.DataFrame(columns=newline.keys())
    # add the new line to the dataframe
    df = pd.concat([df, pd.DataFrame(newline, index=[df.shape[0]])], axis=0)
    # save the dataframe to the excel file
    index_bool = True if index_col is not None else False
    header_bool = True if header is not None else False
    df.to_excel(filepath, index=index_bool, header=header_bool, na_rep="NaN")

def detach_clone(x):
    """
    Detach and clone a tensor if it is a tensor
    :param x:
    :return:
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().clone()
    return x

def df_linear_interp(df, col_name, query_time):
    """
    Evaluate via linear interpolation the value of a column at a given time.
    :param df: pd.DataFrame, the dataframe containing the temperature setpoints at given time steps
    :param col_name: str, the name of the column containing the temperature setpoints
    :param query_time: datetime, the datetime at which the temperature setpoints are to be evaluated
    :return: the temperature setpoints at datetime
    """
    # compute the number of seconds since start
    seconds = (df.index - df.index[0]).total_seconds()
    # linear interpolation
    return np.interp((query_time - df.index[0]).total_seconds(), seconds, df[col_name])

def pad_nested_lists(nl, len_nl=None, how=None, value=np.nan):
    """
    Given a list of lists (or nested list, nl), makes sure that each list has the same length.
    Per default, each is padded by its average to have the length of the longest list.
    nl: the list of lists
    len_nl: the final length of all the nested lists once padded. Per default, it is the length of the longest list.
    how: the function describing how to fill.
    value: the value to fill the list with. If specified, how is ignored.
    """
    max_len_nl = max(map(len, nl))
    # if no len_nl specified, assign the value of the longest list
    if len_nl == None:
        len_nl = max_len_nl
    # if the desired length is shorter than the longest list, error
    if max_len_nl > len_nl:
        raise ValueError("The required lenght is smaller than the longest list.")
    # pad each list with how or the value
    for l in nl:
        l.extend([value if value is not None else how(l)] * (len_nl - len(l)))
    return nl


def append_and_return(x, l):
    """
    Append element x to list l gathering the tensors and return x
    Args:
        x: element to append and return
        l: list

    Returns: x

    """
    l.append(x)
    return x



