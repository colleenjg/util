"""
gen_util.py

This module contains general purpose functions.

Authors: Colleen Gillon

Date: October, 2018

Note: this code uses python 3.7.

"""

import copy
import datetime
import logging
import multiprocessing
import os
import random
import re
import sys
import time
import warnings
from pathlib import Path

from joblib import Parallel, delayed
import numexpr
import numpy as np

from util import logger_util

logger = logging.getLogger(__name__)


#############################################
class TempWarningFilter():
    """
    Context manager for temporarily filtering specific warnings.

    Optional init args:
        - msgs (list)  : Beginning of message in the warning to filter. 
                         Must be the same length as categs.
                         default: []
        - categs (list): Categories of the warning to filter. Must be 
                         the same length as msgs.
                         default: []    
    """

    def __init__(self, msgs=[], categs=[]):
        self.orig_warnings = warnings.filters

        self.msgs = list_if_not(msgs)
        self.categs = list_if_not(categs)

        if len(self.msgs) != len(self.categs):
            raise ValueError("Must provide as many 'msgs' as 'categs'.")


    def __enter__(self):
        for msg, categ in zip(self.msgs, self.categs):
            warnings.filterwarnings("ignore", message=msg, category=categ)


    def __exit__(self, exc_type, exc_value, exc_traceback):
        warnings.filters = self.orig_warnings


#############################################
class TimeIt():
    """
    Context manager for timing a function, and logging duration to the logger
    (in HHh MMmin SSsec).
    """

    def __init__(self):
        return

    def __enter__(self):
        self.start = time.time()


    def __exit__(self, exc_type, exc_value, exc_traceback):

        end = time.time()
        duration = end - self.start # in seconds

        rem_duration = duration
        hours, mins = 0, 0
        duration_str = ""
        fail_str = " (failed)" if exc_type else ""

        secs_per_hour = 60 * 60
        if duration > secs_per_hour:
            hours = int(rem_duration // secs_per_hour)
            rem_duration = rem_duration - hours * secs_per_hour
            duration_str = f"{duration_str}{hours}h "
        
        secs_per_min = 60
        if duration > secs_per_min:
            mins = int(rem_duration // secs_per_min)
            rem_duration = rem_duration - mins * secs_per_min
            duration_str = f"{duration_str}{mins}m "
        
        secs = rem_duration
        duration_str = f"{duration_str}{secs:.2f}s"

        logger.info(f"Duration: {duration_str}{fail_str}")


#############################################
#############################################
def accepted_values_error(varname, wrong_val, accept_vals):
    """
    accepted_values_error(varname, wrong_value, accept_values)

    Raises a value error with a message indicating the variable name,
    accepted values for that variable and wrong value stored in the variable.

    Required args:
        - varname (str)     : name of the variable
        - wrong_val (item)  : value stored in the variable
        - accept_vals (list): list of accepted values for the variable
    """

    val_str = ", ".join([f"'{x}'" for x in accept_vals])
    error_message = (f"'{varname}' value '{wrong_val}' unsupported. Must be in "
        f"{val_str}.")
    raise ValueError(error_message)


#############################################
def CC_config_cache():
    """
    CC_config_cache()

    Checks whether python appears to be running on a Compute Canada cluster. 
    Specifically, looks for a scratch directory under the "SCRATCH" key in the 
    os environment variables. 
    
    If scratch is found, sets the following keys, if they don't already exist, 
    to a writable location on scratch:
        - "MPLCONFIGDIR": the matplotlib config directory
        - "XDG_CONFIG_HOME": the default user config directory
        - "XDG_CACHE_HOME" : the default user config directory

    Doing this addresses a warning from packages like matplotlib, astropy about 
    the default config/cache directories not being writable.

    Tested on niagara Compute Canada cluster (2020).
    """

    existing_keys = os.environ.keys()

    if "SCRATCH" in existing_keys:
        gen_config_dir = Path(os.environ["SCRATCH"], ".config")
        gen_cache_dir = Path(os.environ["SCRATCH"], ".cache")

        if "MPLCONFIGDIR" not in existing_keys:
            os.environ["MPLCONFIGDIR"] = str(Path(gen_config_dir, "matplotlib"))

        # for astropy... create writable cache and config directory
        for key, gen_dir in zip(
            ["CONFIG", "CACHE"], [gen_config_dir, gen_cache_dir]):
            if f"XDG_{key}_HOME" not in existing_keys:
                os.environ[f"XDG_{key}_HOME"] = str(gen_dir)
                astropy_dir = Path(gen_dir, "astropy")
                astropy_dir.mkdir(parents=True, exist_ok=True)


#############################################
def extend_sys_path(file_path, parents=1):
    """
    extend_sys_path(file_path)

    Extends system path by adding the parents of the file path.

    If __file__ is passed as the file_path, this ensures that the parent 
    directory paths are correctly added, relative to the current working 
    directory.

    Required args:
        - file_path (Path): local file path relative to which to add parents
    
    Optional args:
        - parents (int): number of parent directories to add
                         parents=1
    """
    
    local_path = Path(file_path).parent
    
    add_paths = [local_path]
    
    n_parts = len(local_path.parts)
    for i in range(parents):
        if i < n_parts:
            add_path = Path(*local_path.parts[:-i])
        elif i == n_parts:
            add_path = Path(".")
        elif i > n_parts:
            add_path = Path(*[[".."] * (i - n_parts)])
        add_paths.append(add_path)

    add_paths = [str(add_path) for add_path in add_paths]

    sys.path.extend(add_paths)


#############################################
def create_time_str():
    """
    create_time_str()

    Returns a string in a format appropriate for a directory or filename
    containing date and time information based on time at which the function is
    called.

    Return:
        dirname (str): string containing date and time formatted as 
                       YYMMDD_HHMMSS
    """

    now = datetime.datetime.now()
    dirname = (f"{now.year:02d}{now.month:02d}{now.day:02d}_"
        f"{now.hour:02d}{now.minute:02d}{now.second:02d}")
    return dirname
    
    
#############################################
def remove_if(vals, rem):
    """
    remove_if(vals, rem)

    Returns input with items removed from it, if they were are the input.

    Required args:
        - vals (item or list): item or list from which to remove elements
        - rem (item or list) : item or list of items to remove from vals

    Returns:
        - vals (list): list with items removed.
    """

    if not isinstance(rem, list):
        rem = [rem]
    if not isinstance(vals, list):
        vals = [vals]
    for i in rem:
        if i in vals:
            vals.remove(i)
    return vals


#############################################
def is_iterable(item, excl_strings=False):
    """
    is_iterable(item)

    Returns whether item is an iterable, based on whether the iter() function 
    can be used with it.

    Required args:
        - item (obj): item

    Optional args:
        - excl_strings (bool): if True, a False value is returned for a string, 
                              even though it is an iterable.
                              default: False
    
    Returns:
        - (bool): whether the input is an iterable
    """

    if excl_strings and isinstance(item, str):
        return False

    try:
        _ = iter(item)
        return True

    except TypeError:
        return False


#############################################
def list_if_not(items, any_iterable=False, excl_strings=False):
    """
    list_if_not(items)

    Returns input in a list, if it is not a list, or, optionally, if it is not 
    an iterable.

    Required args:
        - items (obj or list): item or list

    Optional args:
        - any_iterable (bool): if True, if items is an iterable, it is not 
                               placed in a list.
                               default: False
        - excl_strings (bool): if True, strings are excluded from allowed 
                               iterables, and placed in a list.
                               default: False

    Returns:
        - items (list): list version of input.
    """
    
    make_list = False

    if any_iterable and not is_iterable(item, excl_strings=excl_strings):
        make_list = True    
    
    elif not isinstance(items, list):
        make_list = True        
        
    if make_list:
        items = [items]

    return items


#############################################
def delist_if_not(items):
    """
    delist_if_not(items)

    If a list or iterable contains only one element, returns the element. 
    Otherwise, returns the original list.

    Required args:
        - items (iterable): list or iterable

    Returns:
        - items (item or iterable): iterable or only item in the iterable.
    """
    
    if len(items) == 1:
        items = items[0]

    return items


#############################################
def remove_lett(lett_str, rem):
    """
    remove_lett(lett_str, rem)

    Returns input string with letters remove, as well as a list of the letters
    that were actually present, and removed.

    Required args:
        - lett_str (str): string of letters
        - rem (str)     : string of letters to remove

    Returns:
        - lett_str (str): string of letters where the letters to remove have 
                          been removed
        - removed (str) : string of letters that were actually present and 
                          removed
    """

    if not isinstance(lett_str, str):
        raise TypeError("lett_str must be a string.")
    
    if not isinstance(rem, str):
        raise TypeError("rem must be a string.")

    removed = ""
    for lett in rem:
        if lett in lett_str:
            lett_str = lett_str.replace(lett, "")
            removed += lett

    return lett_str, removed


#############################################
def slice_idx(axis, pos):
    """
    slice_idx(axis, pos)

    Returns a tuple to index an array based on an axis and position on that
    axis.

    Required args:
        - axis (int)            : axis number (non negative)
        - pos (int, list, slice): position(s) on axis

    Returns:
        - sl_idx (slice): slice corresponding to axis and position passed.
    """

    if axis is None and pos is None:
        sl_idx = tuple([slice(None)])

    elif axis < 0:
        raise NotImplementedError("Negative axis values not accepted, as "
            "they are not correctly differentiated from 0.")

    else:
        sl_idx = tuple([slice(None)] * axis + [pos])

    return sl_idx


#############################################
def remove_idx(items, rem, axis=0):
    """
    remove_idx(items, rem)

    Returns input with items at specific indices in a specified axis removed.

    Required args:
        - items (item or array-like): array or list from which to remove 
                                      elements
        - rem (item or array-like)  : list of idx to remove from items

    Optional args:
        - axis (int): axis along which to remove indices if items is an array
                      default: 0

    Returns:
        - items (array-like): list or array with specified items removed.
    """

    rem = list_if_not(rem)

    if isinstance(items, list):
        make_list = True
        items     = np.asarray(items, dtype=object)
    else:
        make_list = False

    all_idx = items.shape[axis]
    keep = sorted(set(range(all_idx)) - set(rem))
    keep_slice = slice_idx(axis, keep)

    items = items[keep_slice]

    if make_list:
        items = items.tolist()
    
    return items


#############################################
def pos_idx(idx, leng):
    """
    pos_idx(idx, leng)

    Returns a list of indices with any negative indices replaced with
    positive indices (e.g. -1 -> 4 for an axis of length 5).

    Required args:
        - idx (int or list): index or list of indices
        - leng (int)       : number of axes/dimensions

    Returns:
        - idx (int or list): modified index or list of indices (all positive)
    """

    if isinstance(idx, int):
        if idx < 0:
            idx = leng + idx
    
    else:
        for i in range(len(idx)):
            if idx[i] < 0:
                idx[i] = leng + idx[i]
        
    return idx


#############################################
def consec(idx, smallest=False):
    """
    consec(idx)

    Returns the first of each consecutive series in the input, as well as the
    corresponding number of consecutive values.
    
    Required args:
        - idx (list)  : list of values, e.g. indices
    
    Optional args:
        - smallest (bool): if True, the smallest interval present is considered 
                           consecutive
                           default: False
    
    Returns:
        - firsts (list)  : list of values with consecutive values removed
        - n_consec (list): list of number of consecutive values corresponding
                             to (and including) the values in firsts
    """


    if len(idx) == 0:
        return [], []

    interv = 1
    if smallest:
        interv = min(np.diff(idx))

    consec_bool = np.diff(idx) == interv
    firsts = []
    n_consec = []

    firsts.append(idx[0])
    count = 1
    for i in range(len(consec_bool)):
        if consec_bool[i] == 0:
            n_consec.append(count)
            firsts.append(idx[i+1])
            count = 1
        else:
            count += 1
    n_consec.append(count)
    
    return firsts, n_consec
    

#############################################
def deepcopy_items(item_list):
    """
    deepcopy_items(item_list)

    Returns a deep copy of each item in the input.

    Required args:
        - item_list (list): list of items to deep copy

    Returns:
        - new_item_list (list): list of deep copies of items
    """
    
    item_list = list_if_not(item_list)

    new_item_list = []
    for item in item_list:
        new_item_list.append(copy.deepcopy(item))

    return new_item_list


#############################################
def intlist_to_str(intlist):
    """
    intlist_to_str(intlist)

    Returns a string corresponding to the list of values, e.g. 1-4 or 1-3-6.

    Required args:
        - intlist (list): list of int values

    Returns:
        - intstr (str): corresponding string. If range, end is included
    """

    if isinstance(intlist, list):
        extr = [min(intlist), max(intlist) + 1]
        if set(intlist) == set(range(*extr)):
            intstr = f"{extr[0]}-{extr[1]-1}"
        else:
            intstr = "-".join([str(i) for i in sorted(intlist)])
    else:
        raise TypeError("'intlist' must be a list.")

    return intstr


#############################################
def str_to_list(item_str, only_int=False):
    """
    str_to_list(item_str)

    Returns a list of items taken from the input string, in which different 
    items are separated by spaces. 

    Required args:
        - item_str (str): items separated by spaces

    Optional args:
        - only_int (bool): if True, items are converted to ints
                           default: False

    Returns:
        - item_list (list): list of values.
    """

    if len(item_str) == 0:
        item_list = []
    else:
        item_list = item_str.split()
        if only_int:
            item_list = [int(re.findall("\d+", it)[0]) for it in item_list]
        
    return item_list


#############################################
def conv_type(item, dtype=int):
    """
    conv_type(item)

    Returns input item converted to a specific type (int, float or str). 

    Required args:
        - item (item): value to convert

    Optional args:
        - dtype (dtype): target datatype (int, float or str)
                         default: int

    Returns:
        - item (item): converted value
    """

    if dtype in [int, "int"]:
        item = int(item)
    elif dtype in [float, "float"]:
        item = float(item)
    elif dtype in [str, "str"]:
        item = str(item)
    else:
        accepted_values_error("dtype", dtype, ["int", "float", "str"])

    return item


#############################################
def conv_types(items, dtype=int):
    """
    conv_types(items)

    Returns input list with items converted to a specific type (int, float or 
    str). 

    Required args:
        - items (list): values to convert

    Optional args:
        - dtype (dtype): target datatype (int, float or str)
                         default: int

    Returns:
        - items (list): converted values
    """

    items = list_if_not(items)

    for i in range(len(items)):
        items[i] = conv_type(items[i], dtype)

    return items


#############################################
def get_closest_idx(targ_vals, src_vals, allow_out=False):
    """
    get_closest_idx(targ_vals, src_vals)

    Returns index of closest value in targ_vals for each value in src_vals. 

    Required args:
        - targ_vals (1D array): target array of values
        - src_vals (1D array): values for which to find closest value in 
                               targ_vals

    Returns:
        - idxs (1D): array with same length as src_vals, identifying closest 
                     values in targ_vals
    """

    targ_vals = np.asarray(targ_vals)
    src_vals = np.asarray(src_vals)

    if (np.argsort(targ_vals) != np.arange(len(targ_vals))).any():
        raise RuntimeError("Expected all targ_vals to be sorted.")

    if (np.argsort(src_vals) != np.arange(len(src_vals))).any():
        raise RuntimeError("Expected all src_vals to be sorted.")

    idxs = np.searchsorted(targ_vals, src_vals)
    idxs_incr = idxs - 1

    min_val = np.where(idxs_incr == -1)[0]
    if len(min_val):
        idxs_incr[min_val] = 0

    max_val = np.where(idxs == len(targ_vals))[0]
    if len(max_val):
        idxs[max_val] = len(targ_vals) - 1

    all_vals = np.stack([targ_vals[idxs], targ_vals[idxs_incr]])
    val_diff = np.absolute(all_vals - src_vals.reshape(1, -1))

    all_idxs = np.stack([idxs, idxs_incr]).T
    idxs = all_idxs[np.arange(len(all_idxs)), np.argmin(val_diff, axis=0)]

    return idxs


#############################################
def get_df_label_vals(df, label, vals=None):
    """
    get_df_label_vals(df, label)

    Returns values for a specific label in a dataframe. If the vals is "any", 
    "all" or None, returns all different values for that label.
    Otherwise, vals are returned as a list.

    Required args:
        - df (pandas df): dataframe
        - label (str)   : label of the dataframe column of interest

    Optional args:
        - val (str or list): values to return. If val is None, "any" or "all", 
                             all values are returned.
                             default=None
    Return:
        - vals (list): values
    """

    if vals in [None, "any", "all"]:
        vals = df[label].unique().tolist()
    else:
        vals = list_if_not(vals)
    return vals


#############################################
def get_df_vals(df, cols=[], criteria=[], label=None, unique=True, dtype=None, 
                single=False):
    """
    get_df_vals(df, cols, criteria)

    Returns dataframe lines or values that correspond to the specified 
    criteria. 

    Required args:
        - df (pd.DataFrame): dataframe

    Optional args:
        - cols (list)    : ordered list of columns for which criteria are 
                           provided
                           default: []
        - criteria (list): ordered list of single criteria for each column
                           default: []
        - label (str)    : column for which to return values
                           if None, the dataframe lines are returned instead
                           default: None
        - unique (bool)  : if True, only unique values are returned for the 
                           column of interest
                           default: True
        - dtype (dtype)  : if not None, values are converted to the specified 
                           datatype (int, float or str)
                           dtype: None
        - single (bool)  : if True, checks whether only one value or row is 
                           found and if so, returns it
                           dtype: False 

    Returns:
        if label is None:
            - lines (pd.Dataframe): dataframe containing lines corresponding to 
                                    the specified criteria.
        else:
            if single:
            - vals (item)         : value from a specific column corresponding 
                                    to the specified criteria. 
            else:
            - vals (list)         : list of values from a specific column 
                                    corresponding to the specified criteria. 
    """

    if not isinstance(cols, list):
        cols = [cols]
        criteria = [criteria]

    if len(cols) != len(criteria):
        raise ValueError("Must pass the same number of columns and criteria.")

    for att, cri in zip(cols, criteria):
        df = df.loc[(df[att] == cri)]
        
    if label is not None:
        vals = df[label].tolist()
        if unique:
            vals = sorted(list(set(vals)))
        if dtype is not None:
            vals = conv_types(vals, dtype)
        if single:
            if len(vals) != 1:
                raise RuntimeError("Expected to find 1 value, but "
                    f"found {len(vals)}.")
            else:
                vals = vals[0]
        return vals
    else: 
        if single and len(df) != 1:
            raise RuntimeError("Expected to find 1 dataframe line, but "
                f"found {len(df)}.")
        return df


#############################################
def set_df_vals(df, idx, cols, vals, in_place=False):
    """
    set_df_vals(df, attributes, criteria)

    Returns dataframe with certain values changed. These are specified by one
    index and a list of columns and corresponding new values.

    Required args:
        - df (pd.DataFrame): dataframe
        - idx (int)        : dataframe line index (for use with .loc)
        - cols (list)      : ordered list of columns for which vals are 
                             provided
        - vals (list)   : ordered list of values for each column

    Optional args:
        - in_place (bool): if True, changes are made in place. Otherwise, a 
                           deep copy of the dataframe is made first.
                           default: False

    Returns:
        - df (pd.Dataframe): dataframe containing modified lines. 
    """

    if not in_place:
        df = df.copy(deep=True)

    cols = list_if_not(cols)
    vals = list_if_not(vals)

    if len(cols) != len(vals):
        raise ValueError("Must pass the same number of columns and values.")

    for col, val in zip(cols, vals):
        df.loc[idx, col] = val
    
    return df


#############################################
def drop_unique(df, in_place=False):
    """
    drop_unique(df)

    Returns dataframe with columns containing only a unique value dropped.

    Required args:
        - df (pd.DataFrame): dataframe

    Optional args:
        - in_place (bool): if True, changes are made in place. Otherwise, a 
                           deep copy of the dataframe is made first.
                           default: False

    Returns:
        - df (pd.DataFrame): dataframe with columns containing only a unique 
                             value dropped
    """

    if not in_place:
        df = df.copy(deep=True)

    for col in df.columns:
        uniq_vals = df[col].unique().tolist()
        if len(uniq_vals) == 1:
            df = df.drop(columns=col)

    return df


#############################################
def set_object_columns(df, cols, in_place=False):
    """
    set_object_columns(df, cols)

    Returns dataframe with columns converted to object columns. If a column 
    does not exist, it is created first.

    Required args:
        - df (pandas df): dataframe
        - cols (list)   : list of columns to convert to or create as 
                          object columns 

    Optional args:
        - in_place (bool): if True, changes are made in place. Otherwise, a 
                           deep copy of the dataframe is made first.
                           default: False
    """

    if not in_place:
        df = df.copy(deep=True)

    cols = list_if_not(cols)

    for col in cols:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = df[col].astype(object)

    return df


#############################################
def num_ranges(ns, pre=0, leng=10):
    """
    num_ranges(ns)

    Returns all indices within the specified range of the provided reference 
    indices. 

    Required args:
        - ns (list): list of reference numbers

    Optional args:
        - pre (num) : indices to include before reference to include
                      default: 0
        - leng (num): length of range
                      default: 10
    Returns:
        - num_ran (2D array): array of indices where each row is the range
                              around one of the input numbers (ns x ranges)
    """

    post = float(leng) - pre

    pre, post = [int(np.around(p)) for p in [pre, post]]

    num_ran = np.asarray([list(range(n-pre, n+post)) for n in ns])

    return num_ran


#############################################
def get_device(cuda=False, device=None):
    """
    get_device()

    Returns name of device to use based on cuda availability and whether cuda  
    is requested, either via the "cuda" or "device" variable, with "device" 
    taking precedence.

    Optional args:
        - cuda (bool) : if True, cuda is used (if available), but will be 
                        overridden by device.
                        default: False
        - device (str): indicates device to use, either "cpu" or "cuda", and 
                        will override cuda variable if not None
                        default: None 
        
    Returns:
        - device (str): device to use
    """

    if device is None:
        if cuda:
            device = "cuda"
        else:
            device = "cpu"
    if device == "cuda":
        import torch
        if not(torch.cuda.is_available()):
            device = "cpu"

    return device


#############################################
def hierarch_argsort(data, sorter="fwd", axis=0, dtypes=None):
    """
    hierarch_argsort(data)

    Returns the sorting argument and sorted data. Data is sorted hierarchically
    based on the sorter (top -> bottom hierarchy) along the specified axis.

    Required args:
        - data (nd array): array of data to use for sorting

    Optional args:
        - sorter (str or list): order to use for the sorting hierarchy, from
                                top to bottom (list of indices or "fwd" or 
                                "rev")
                                default: "fwd"
        - axis (int)          : axis number
                                default: 0
        - dtypes (list)       : datatypes to which to convert each data sorting
                                sub array (one per sorting position)
                                default: None
    
    Returns:
        - overall_sort (list): sorting index
        - data (nd array)    : sorted data array
    """

    if len(data.shape) != 2:
        raise NotImplementedError("Only implemented for 2D arrays.")

    axis, rem_axis = pos_idx([axis, 1-axis], len(data.shape))
    axis_len = data.shape[axis]

    data = copy.deepcopy(data)

    if sorter in ["fwd", "rev"]:
        sorter = range(axis_len)
        if sorter == "rev":
            sorter = reversed(sorter)
    else:
        sorter = list_if_not(sorter)
        sorter = pos_idx(sorter, data.shape[axis])

    if dtypes is None:
        dtypes = [None] * len(sorter)
    elif len(dtypes) != len(sorter):
        raise ValueError("If 'dtypes' are provided, must pass one per "
            "sorting position.")

    overall_sort = np.asarray(range(data.shape[rem_axis]))

    for i, dt in zip(reversed(sorter), dtypes):
        sc_idx = slice_idx(axis, i)
        sort_data = data[sc_idx]
        if dt is not None:
            sort_data = sort_data.astype(dt)
        sort_arr = np.argsort(sort_data)
        overall_sort = overall_sort[sort_arr]
        sort_slice   = slice_idx(rem_axis, sort_arr)
        data         = data[sort_slice]

    return overall_sort, data


#############################################
def compile_dict_list(dict_list):
    """
    compile_dict_list(dict_list)

    Returns a dictionary of lists created from a list of dictionaries with 
    shared keys.

    Required args:
        - dict_list (list): list of dictionaries with shared keys

    Returns:
        - full_dict (dict): dictionary with lists for each key
    """

    full_dict = dict()

    all_keys = []
    for sing_dict in dict_list:
        all_keys.extend(sing_dict.keys())
    all_keys = list(set(all_keys))

    for key in all_keys:
        vals = [sub_dict[key] for sub_dict in dict_list 
            if key in sub_dict.keys()]
        full_dict[key] = vals

    return full_dict


#############################################
def num_to_str(num, n_dec=2, dec_sep="-"):
    """
    num_to_str(num)

    Returns number converted to a string with the specified number of decimals 
    and decimal separator

    Required args:
        - num (num): number
    
    Optional args:
        - n_dec (int)  : number of decimals to retain
                         default: 2
        - dec_sep (str): string to use as a separator
                         default: "-"
    
    Returns:
        - num_str (str): number as a string
    """

    num_str = str(int(num))

    num_res = np.round(num % 1, n_dec)
    if num_res != 0:
        num_str = f"{num_str}{dec_sep}{str(num_res)[2:]}"

    return num_str


#############################################
def keep_dict_keys(in_dict, keep_if):
    """
    keep_dict_keys(in_dict, keep_if)

    Returns dictionary with only specified keys retained, if they are present.

    Required args:
        - in_dict (dict): input dictionary
        - keep_if (list): list of keys to keep if they are in the input 
                          dictionary
    
    Returns:
        - out_dict (dict): dictionary with keys retained
    """

    out_dict = dict()
    for key in keep_if:
        if key in in_dict.keys():
            out_dict[key] = in_dict[key]

    return out_dict


#############################################
def get_n_cores(n_tasks, parallel=True, max_cores="all"):
    """
    get_n_cores(n_tasks)

    Returns number of cores available.

    Required args:
        - n_tasks (int): number of tasks to run
    
    Optional args:
        - parallel (bool)       : if False, n_jobs of None is returned
                                  default: True
        - max_cores (str or num): max number or proportion of cores to use 
                                  ("all", proportion or int)
                                  default: "all"

    Returns:
        - n_cores (int): number of cores that are usable (None if not 
                         parallel)
    """

    if not parallel:
        n_cores = None

    else:
        n_cores = multiprocessing.cpu_count()
        if max_cores != "all":
            max_cores = float(max_cores)
            if max_cores >= 0.0 and max_cores <= 1.0:
                n_cores = int(n_cores * max_cores)
            else:
                n_cores = np.min(n_cores, max_cores)
        n_cores = int(n_cores)

    return n_cores


#############################################
def get_n_jobs(n_tasks, parallel=True, max_cores="all"):
    """
    get_n_jobs(n_tasks)

    Returns number of jobs corresponding to the criteria passed.

    Required args:
        - n_tasks (int): number of tasks to run
    
    Optional args:
        - parallel (bool)       : if False, n_jobs of None is returned
                                  default: True
        - max_cores (str or num): max number or proportion of cores to use 
                                  ("all", proportion or int)
                                  default: "all"

    Returns:
        - n_jobs (int): number of jobs to use (None if not parallel or fewer 
                        than 2 jobs calculated)
    """

    if not parallel:
        n_jobs = None

    else:
        n_cores = get_n_cores(n_tasks, parallel, max_cores)
        n_jobs = min(int(n_tasks), n_cores)
        if n_jobs < 2:
            n_jobs = None

    return n_jobs


#############################################
def n_cores_numba(n_tasks, parallel=True, max_cores="all", allow="around", 
                  set_now=False):
    """
    n_cores_numba(n_tasks)

    If parallel, returns the number of cores available for numba for each thread 
    within a parallel script. Optionally also sets NUMEXPR_MAX_THREADS environment 
    variable.

    Required args:
        - n_tasks (int): number of tasks to run
    
    Optional args:
        - parallel (bool)       : if False, n_jobs of None is returned
                                  default: True
        - max_cores (str or num): max number or proportion of cores to use 
                                  ("all", proportion or int)
                                  default: "all"
        - allow (str)           : how to round the estimated number of cores 
                                  to allocated to each thread, 
                                  e.g. "around", "ceil", "floor"
                                  default: "around"
        - set_now (bool)        : if True, NUMEXPR_MAX_THREADS is set
                                  default: False

    Returns:
        - split_cores (int): number of cores to allocate to each thread (None, 
                             if not parallel)
    """

    n_cores = get_n_cores(n_tasks, parallel, max_cores)
    
    if n_cores is None:
        return

    remaining_cores = n_cores - n_tasks

    # calculate split for the cores (minimum 1)
    split_cores = np.max([0, remaining_cores / n_tasks]) + 1 # include current core
    
    if allow == "around":
        split_cores = int(np.around(split_cores))
    elif allow == "ceil":
        split_cores = int(np.ceil(split_cores))
    elif allow == "floor":
        split_cores = int(np.floor(split_cores))
    else:
        accepted_values_error(
            "split_cores", split_cores, ["around", "ceil", "floor"])

    if set_now:
        numexpr.set_num_threads(split_cores)

    return split_cores


#############################################
class ProgressParallel(Parallel):
    """
    Class allowing joblib Parallel to work with tqdm.
    
    Taken from https://stackoverflow.com/questions/37804279/how-can-we-use-tqdm-in-a-parallel-execution-with-joblib.
    """

    def __init__(self, use_tqdm=True, total=None, *args, **kwargs):
        """
        Initializes a joblib Parallel object that works with tqdm.

        Optional args:
            - use_tqdm (bool): if True, tqdm is used
                               default: True
            - total (int)    : number of items in the progress bar
                               default: None
        """

        from tqdm.auto import tqdm
        self._use_tqdm = use_tqdm
        self._total = total
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        from tqdm.auto import tqdm
        with tqdm(disable=not self._use_tqdm, total=self._total) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


#############################################
def parallel_wrap(fct, loop_arg, args_list=None, args_dict=None, parallel=True, 
                  max_cores="all", zip_output=False, mult_loop=False, 
                  pass_parallel=False, use_tqdm=False):
    """
    parallel_wrap(fct, loop_arg)

    Wraps functions to run them in parallel if parallel is True (not 
    implemented as a python wrapper, to enable additional flexibility).

    Required args:
        - fct (function) : python function
        - loop_arg (list): argument(s) over which to loop (must be first 
                           arguments of fct)
                           if multiple arguments, they must already be zipped 
                           (where the length is the number of items to loop 
                           over), and mult_loop must be set to True
    
    Optional args:
        - args_list (list)      : function input argument list    
                                  default: None
        - args_dict (dict)      : function input argument dictionary
                                  default: None
        - parallel (bool)       : if False, n_jobs of None is returned
                                  default: True
        - max_cores (str or num): max number or proportion of cores to use 
                                  ("all", proportion or int)
                                  default: "all"
        - zip_output (bool)     : if True, outputs are zipped, and tuples are
                                  converted to lists
                                  default: False
        - mult_loop (bool)      : if True, the loop argument contains multiple 
                                  consecutive first arguments
        - pass_parallel (bool)  : if True, 'parallel' argument is passed to the 
                                  function to ensure that 
                                  (1) if this function does run in parallel, 
                                  subfunctions will not sprout parallel joblib 
                                  processes.
                                  (2) is this function does not run in 
                                  parallel, the value of 'parallel' is still 
                                  passed on.
                                  default: False
        - use_tqdm (bool)       : if True, tqdm is used for progress bars.
                                  default: False

    Returns:
        - outputs (list of tuples): outputs, structured as 
                                        (loop_arg length) x 
                                        (number of output values), 
                                    or if zip_output, structured as 
                                        (number of output values) x 
                                        (loop_arg length)
    """

    loop_arg = list(loop_arg)
    n_jobs = get_n_jobs(len(loop_arg), parallel, max_cores)
    
    if args_list is None: args_list = []
    args_list = list_if_not(args_list)

    # to allow multiple arguments to be looped over (mimicks zipping)
    if not mult_loop:
        loop_arg = [(arg, ) for arg in loop_arg]

    # enable information to be passed to the function as to whether it can 
    # sprout parallel processes
    if pass_parallel and args_dict is None:
        args_dict = dict()

    if n_jobs is not None and n_jobs > 1:
        if use_tqdm:
            ParallelUse = ProgressParallel(
                use_tqdm=True, total=len(loop_arg), n_jobs=n_jobs
                )
        else:
            ParallelUse = Parallel(n_jobs=n_jobs)

        if pass_parallel: 
            # prevent subfunctions from also sprouting parallel processes
            args_dict["parallel"] = False 
        if args_dict is None:
            outputs = ParallelUse(
                delayed(fct)(*arg, *args_list) for arg in loop_arg
                )
        else:
            outputs = ParallelUse(
                delayed(fct)(*arg, *args_list, **args_dict) for arg in loop_arg
                )
    else:
        if pass_parallel: # pass parallel on
            args_dict["parallel"] = parallel
        if use_tqdm:
            from tqdm import tqdm
            loop_arg = tqdm(loop_arg)

        outputs = []
        if args_dict is None:
            for arg in loop_arg:
                outputs.append(fct(*arg, *args_list))
        else:
            for arg in loop_arg:
                outputs.append(fct(*arg, *args_list, **args_dict))

    if zip_output:
        outputs = [*zip(*outputs)]

    return outputs


#############################################
def get_df_unique_vals(df, axis="index", info="length"):
    """
    get_df_unique_vals(df)

    Returns a list of unique values for each level of the requested axis, in 
    hierarchical order.

    Required args:
        - df (pd.DataFrame): hierarchical dataframe
    
    Optional args:
        - axis (str): Axis for which to return unique values ("index" or 
                      "columns")
                      default: "index"

    Returns:
        - unique_vals (list): unique values for each index or column level, in 
                              hierarchical order
    """

    if axis in ["ind", "idx", "index"]:
        unique_vals = [df.index.unique(row) for row in df.index.names]
    elif axis in ["col", "cols", "columns"]:
        unique_vals = [df.columns.unique(col) for col in df.columns.names]
    else:
        accepted_values_error("axis", axis, ["index", "columns"])


    return unique_vals


#############################################
def reshape_df_data(df, squeeze_rows=False, squeeze_cols=False):
    """
    reshape_df_data(df)

    Returns data array extracted from dataframe and reshaped into as many
    axes as index/column levels, if possible, in hierarchical order.

    Required args:
        - df (pd.DataFrame): hierarchical dataframe
    
    Optional args:
        - squeeze_rows (bool): if True, rows of length 1 are squeezed out
                               default: False
        - squeeze_cols (bool): if True, columns of length 1 are squeezed out
]                              default: False

    Returns:
        - df_data (nd array): dataframe data reshaped into an array
    """

    row_dims = [len(df.index.unique(row)) for row in df.index.names]
    col_dims = [len(df.columns.unique(col)) for col in df.columns.names]

    if squeeze_rows:
        row_dims = filter(lambda dim: dim != 1, row_dims)
    if squeeze_cols:
        col_dims = filter(lambda dim: dim != 1, col_dims)

    new_dims = [*row_dims, *col_dims]

    if np.prod(new_dims) != df.size:
        raise RuntimeError("Unable to automatically reshape dataframe data, as "
            "levels are not shared across all labels.")

    df_data = df.to_numpy().reshape(new_dims)

    return df_data


#############################################
def get_alternating_consec(vals, first=True):
    """
    get_alternating_consec(vals)

    Returns values with only alternating consecutive values retained.

    Required args:
        - vals (list): list of values
    
    Optional args:
        - first (bool): if True, alternating starts by including the first 
                        value of each consecutive portion. If False, it is the 
                        second value.
                        default: True

    Returns:
        - vals (list): list of values with only alternating consecutive values
                       retained
    """

    if first:
        incl_order = [True, False]
    else:
        incl_order = [False, True]

    incl_arr = np.tile(incl_order, len(vals)//2 + 1)
    vals = np.asarray(vals)
    diffs = np.insert(np.diff(vals), -1, 1000)
    ret_vals = []
    prev_idx = 0
    for lim in np.where(diffs > 1)[0]:
        ret_vals.extend(vals[prev_idx:lim + 1][
            incl_arr[: lim + 1 - prev_idx]].tolist())
        prev_idx = lim + 1

    return ret_vals


#############################################
def jit_function(function):
    """
    jit_function(function)

    Returns a function wrapped by numba to run faster. Code source: 
    https://ilovesymposia.com/2017/03/15/prettier-lowlevelcallables-with-numba-jit-and-decorators/

    Required args:
        - function (function): Function to wrap

    Returns:
        - (LowLevelCallable): Wrapped function
    """


    import numba
    from numba import cfunc, carray
    from numba.types import intc, CPointer, float64, intp, voidptr
    from scipy import LowLevelCallable

    jitted_function = numba.jit(function, nopython=True)
    @cfunc(intc(CPointer(float64), intp, CPointer(float64), voidptr))
    def wrapped(values_ptr, len_values, result, data):
        values = carray(values_ptr, (len_values,), dtype=float64)
        result[0] = jitted_function(values)
        return 1
    
    return LowLevelCallable(wrapped.ctypes)

