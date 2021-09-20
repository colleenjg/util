"""
math_util.py

This module contains basic math functions.

Authors: Colleen Gillon

Date: October, 2018

Note: this code uses python 3.7.

"""

import copy
import logging
import os
import re
import warnings

import numpy as np
import scipy.ndimage as scn
import scipy.stats as scist

from util import gen_util, logger_util

logger = logging.getLogger(__name__)

TAB = "    "

# Set default max array size for permutation tests 
LIM_E6_SIZE = 350
if "LIM_E6_SIZE" in os.environ.keys():
    LIM_E6_SIZE = int(os.environ["LIM_E6_SIZE"])

# Minimum number of examples outside for confidence interval edge calculation
MIN_N = 2


#############################################
def mean_med(data, stats="mean", axis=None, nanpol=None):
    """
    mean_med(data)

    Returns the mean or median of the data along a specified axis, depending on
    which statistic is requested.

    Required args:
        - data (nd array): data on which to calculate statistic

    Optional args:
        - stats (str) : "mean" or "median"
                        default: "mean"
        - axis (int)  : axis along which to take the statistic
                        default: None
        - nanpol (str): policy for NaNs, "omit" or None
                        default: None
    
    Returns:
        - me (nd array or num): mean or median of data along specified axis
    """

    if stats == "mean":
        if nanpol is None:
            me = np.mean(data, axis=axis)
        elif nanpol == "omit":
            me = np.nanmean(data, axis=axis)
    elif stats == "median":
        if nanpol is None:
            me = np.median(data, axis=axis)
        elif nanpol == "omit":
            me = np.nanmedian(data, axis=axis)
    else:
        gen_util.accepted_values_error("stats", stats, ["mean", "median"])
    if nanpol is not None and nanpol != "omit":
        gen_util.accepted_values_error("nanpol", nanpol, ["None", "omit"])

    return me


#############################################
def error_stat_name(stats="mean", error="sem", qu=[25, 75]):
    """
    error_stat_name()

    Returns the name(s) of the error statistic(s).

    Optional args:
        - stats (str) : "mean" or "median"
                        default: "mean"
        - error (str) : "std" (for std or quintiles) or "sem" (for SEM or MAD)
                        default: "sem"
        - qu (list)   : quintiles to take, if median and std along which 
                        to take the statistic
                        default: [25, 75]
    
    Returns:
        - error_names (str or list): name(s) of error statistic(s)
    """

    if stats == "mean":
        if error == "std":
            error_name = error
        elif error == "sem":
            error_name = "SEM"
    elif stats == "median":
        if error == "std":
            error_name = [f"q{qu[0]}", f"q{qu[1]}"]        
        elif error == "sem":
            error_name = "MAD"
    else:
        gen_util.accepted_values_error("stats", stats, ["mean", "median"])
    if error not in ["std", "sem"]:
        gen_util.accepted_values_error("error", error, ["std", "sem"])

    return error_name


#############################################
def error_stat(data, stats="mean", error="sem", axis=None, nanpol=None, 
               qu=[25, 75]):
    """
    error_stat(data)

    Returns the std, SEM, quartiles or median absolute deviation (MAD) of data 
    along a specified axis, depending on which statistic is requested.

    Required args:
        - data (nd array): data on which to calculate statistic

    Optional args:
        - stats (str) : "mean" or "median"
                        default: "mean"
        - error (str) : "std" (for std or quintiles) or "sem" (for SEM or MAD)
                        default: "sem"
        - axis (int)  : axis along which to take the statistic
                        default: None
        - nanpol (str): policy for NaNs, "omit" or None
                        default: None
        - qu (list)   : quintiles to take, if median and std along which 
                        to take the statistic
                        default: [25, 75]
    
    Returns:
        - error (nd array or num): std, SEM, quintiles or MAD of data along 
                                   specified axis
    """

    if stats == "mean" and error == "std":
        if nanpol is None:
            error = np.std(data, axis=axis)
        elif nanpol == "omit":
            error = np.nanstd(data, axis=axis)
    elif stats == "mean" and error == "sem":
        if nanpol is None:
            error = scist.sem(data, axis=axis)
        elif nanpol == "omit":
            error = scist.sem(data, axis=axis, nan_policy="omit")
    elif stats == "median" and error == "std":
        if nanpol is None:
            error = [np.percentile(data, qu[0], axis=axis), 
                np.percentile(data, qu[1], axis=axis)]
        elif nanpol == "omit":
            error = [np.nanpercentile(data, qu[0], axis=axis), 
                np.nanpercentile(data, qu[1], axis=axis)]
        
    elif stats == "median" and error == "sem":
        # MAD: median(abs(x - median(x)))
        if axis is not None:
            me_shape       = list(data.shape)
            me_shape[axis] = 1
        else:
            me_shape = -1
        if nanpol is None:
            me    = np.asarray(np.median(data, axis=axis)).reshape(me_shape)
            error = np.median(np.absolute(data - me), axis=axis)
        elif nanpol == "omit":
            me    = np.asarray(np.nanmedian(data, axis=axis)).reshape(me_shape)
            error = np.nanmedian(np.absolute(data - me), axis=axis)
    elif stats != "median" and stats != "mean":
        gen_util.accepted_values_error("stats", stats, ["mean", "median"])
    else:
        gen_util.accepted_values_error("error", error, ["std", "sem"])
    if nanpol is not None and nanpol != "omit":
        gen_util.accepted_values_error("nanpol", nanpol, ["[None]", "omit"])

    error = np.asarray(error)
    if len(error.shape) == 0:
        error = error.item()

    return error


#############################################
def outlier_bounds(data, fences="outer", axis=None, nanpol=None):
    """
    outlier_bounds(data)

    Returns outlier fence bounds for the data.

    Required args:
        - data (nd array): data on which to calculate statistic

    Optional args:
        - fences (str): "inner": inner fences [Q1 - 1.5 * IQR, Q3 + 1.5 * IQR]
                        "outer": outer fences [Q1 - 3.0 * IQR, Q3 + 3.0 * IQR]
                        default: "outer"
        - axis (int)  : axis along which to take the statistic
                        default: None
        - nanpol (str): policy for NaNs, "omit" or None
                        default: None
    
    Returns:
        - fences (list): [lower fence, upper fence] limits
    """

    if nanpol is None:
        q1 = np.percentile(data, q=25, axis=axis)
        q3 = np.percentile(data, q=75, axis=axis)
    elif nanpol == "omit":
        q1 = np.nanpercentile(data, q=25, axis=axis)
        q3 = np.nanpercentile(data, q=75, axis=axis)
    else:
        gen_util.accepted_values_error("nanpol", nanpol, ["[None]", "omit"])

    iqr = q3 - q1

    if fences == "inner":
        low_fence = q1 - 1.5 * iqr
        upp_fence = q3 + 1.5 * iqr
    elif fences == "outer":
        low_fence = q1 - 3.0 * iqr
        upp_fence = q3 + 3.0 * iqr
    else:
        gen_util.accepted_values_error("fences", fences, ["inner", "outer"])

    fences = [low_fence, upp_fence]

    return fences


#############################################
def get_stats(data, stats="mean", error="sem", axes=None, nanpol=None,
              qu=[25, 75]):
    """
    get_stats(data)

    Returns statistics calculated as follows: means/medians are calculated 
    along each axis successively, then the full statistics are calculated along 
    the last axis in the list. 
    
    Returns statistics (me, error x values) statistics as a single array.
    Note that is stats="median" and error="std", the error will be in two 
    rows/cols.
    
    Required args:
        - data (nd array): data array (at least 2D)

    Optional args:
        - stats (str)       : stats to take, i.e., "mean" or "median"
                              default: "mean"
        - error (str)       : error to take, i.e., "std" (for std or quintiles) 
                              or "sem" (for SEM or MAD)
                              default: "std"
        - axes (int or list): axes along which to  take statistics. If a list  
                              is passed.
                              If None, axes are ordered reverse sequentially 
                              (-1 to 0).
                              default: None
        - nanpol (str)      : policy for NaNs, "omit" or None
                              default: None
        - qu (list)         : quintiles to take, if median and std along which 
                              to take the statistic
                              default: [25, 75]

    Returns:
        - data_stats (nd array): stats array, structured as: 
                                 stat type (me, error x values) x 
                                     remaining_dims
    """

    data = np.asarray(data)

    if data.shape == 1:
        raise ValueError("Data array must comprise at least 2 dimensions.")

    if axes is None:
        # reversed list of axes, omitting last one
        axes = list(range(0, len(data.shape)))[::-1]
    axes = gen_util.list_if_not(axes)

    # make axis numbers positive
    axes = gen_util.pos_idx(axes, len(data.shape))

    if len(axes) > len(data.shape):
        raise ValueError("Must provide no more axes value than the number of "
            "data axes.")
    
    if len(axes) > 1:
        # take the mean/median successively across axes
        prev = []
        for ax in axes[:-1]:
            # update axis number based on previously removed axes
            sub = sum(p < ax for p in prev)
            prev.append(ax)
            ax = ax-sub
            data = mean_med(data, stats=stats, axis=ax, nanpol=nanpol)
        axis = axes[-1]
        if axis != -1:
            sub = sum(p < axis for p in prev)
            axis = axis-sub
    else:
        axis = axes[0]
        
    # mean/med along units axis (last)
    me  = mean_med(data, stats=stats, axis=axis, nanpol=nanpol) 
    err = error_stat(
        data, stats=stats, error=error, axis=axis, nanpol=nanpol, qu=qu)
    
    # ensures that these are arrays
    me = np.asarray(me)
    err = np.asarray(err)

    if stats=="median" and error=="std":
        me = np.expand_dims(me, 0)
        data_stats = np.concatenate([me, err], axis=0)
    else:
        data_stats = np.stack([me, err])

    return data_stats


#############################################
def log_stats(stats, stat_str=None, ret_str_only=False):
    """
    log_stats(stats)

    Logs the statistics.

    Required args:
        - stats (array-like): stats, structured as [me, err]

    Optional args:
        - stat_str (str)     : string associated with statistics
                               default: None
        - ret_str_only (bool): if True, string is returned instead of being logged
                               default: False
    
    Returns:
        if ret_str_only:
            full_stat_str: full string associated with statistics
    """

    me = stats[0]
    err = stats[1:]
    
    err_str = "/".join([f"{e:.3f}" for e in err])

    plusmin = u"\u00B1"

    if stat_str is None:
        stat_str = ""
    else:
        stat_str = f"{stat_str}: "

    full_stat_str = u"{}{:.5f} {} {}".format(stat_str, me, plusmin, err_str)
    if ret_str_only:
        return full_stat_str
    else:
        logger.info(full_stat_str)
        

#############################################
def integ(data, dx, axis=None, nanpol=None):
    """
    integ(data, dx)

    Returns integral of data along specified axis.

    Required args:
        - data (nd array): data on which to calculate integral
        - dx (num)       : interval between data points

    Optional args:
        - axis (int)  : axis along which to take the statistic
                        default: None
        - nanpol (str): policy for NaNs, "omit" or None
                        default: None
    
    Returns:
        - integ_data (nd array): integral of data along specified axis
    """

    # sum * freq
    if nanpol == "omit":
        integ_data = np.nansum(data, axis) * dx
    elif nanpol is None:
        integ_data = np.sum(data, axis) * dx
    else:
        gen_util.accepted_values_error("nanpol", nanpol, ["None", "omit"])

    return integ_data


#############################################
def get_near_square_divisors(val):
    """
    get_near_square_divisors(val)

    Returns near-square divisors of a number.

    Required args:
        - val (int): value for which to get divisors
    
    Returns:
        - divs (list): list of divisor values in order [high, low]
    """

    if int(val) != float(val):
        raise TypeError("'val' must be an int.")

    i = int(np.max([np.floor(np.sqrt(val)), 1]))
    j = int(np.ceil(val / i))

    divs = [i, j]
    if j > i:
        divs = divs[::-1]

    return divs


#############################################
def get_divisors(val, min_val=None, max_val=None):
    """
    get_divisors(val)

    Returns divisors of a number, optionally within a certain range.

    Required args:
        - val (int): value for which to get divisors

    Optional args:
        - min_val (int): minimum divisor value
                         default: None
        - max_val (int): maximum divisor value
                         default: None
    
    Returns:
        - divs (list): list of divisor values
    """

    if int(val) != float(val):
        raise TypeError("'val' must be an int.")

    if min_val is None:
        min_val = 1

    if max_val is None:
        max_val = val

    divs = []
    for i in range(min_val, max_val + 1):
        if val % i == 0:
            divs.append(i)

    return divs


#############################################
def rolling_mean(vals, win=3):
    """
    rolling_mean(vals)

    Returns rolling mean over the last dimension of the input data.

    Required args:
        - vals (nd array): data array, for which rolling mean will be taken 
                           along last dimension

    Optional args:
        - win (int): length of the rolling mean window
                     default: 3

    Returns:
        - vals_out (nd array): rolling mean data array 
    """

    targ_dims = tuple([1] * (len(vals.shape) - 1) + [win])
    weights = (np.repeat(1.0, win)/win).reshape(targ_dims)
    vals_out = scn.convolve(vals, weights, mode="mirror")

    return vals_out


#############################################
def calc_op(data, op="diff", dim=0, rev=False, nanpol=None, axis=-1):
    """
    calc_op(data)

    Returns result of specified operation performed on a data array defined
    by the specified dimension.

    Required args:
        - data (nd array): data on which to run operation, with length 2 along 
                           dim (or list of arrays if dim = 0).

    Optional args:
        - op (str)    : "diff": index 1 - 0
                        "ratio": index 1/0, or 
                        "rel_diff": (index 1 - 0)/(index 1 + 0)
                        "d-prime": (mean(index 1) - mean(index 0)) / 
                                   (sqrt(
                                     1/2 * (std(index 1)**2 + std(index 0)**2))
                        default: "diff"
        - dim (int)   : dimension along which to do operation
                        default: 0
        - rev (bool)  : if True, indices 1 and 0 are reversed
                        default: False
        - nanpol (str): policy for NaNs, "omit" or None
                        default: None
        - axis (int)  : axis along which to take stats, e.g. std for "d-prime"
                        default: -1
    
    Returns:
        - data (nd array): data on which operation has been applied
    """
    
    if dim == 0: # allows for list
        len_dim = len(data)
    else:
        len_dim = data.shape[dim]
    if len_dim != 2:
        raise ValueError(f"Data should have length 2 along dim: {dim}")

    if isinstance(op, int):
        data_idx = gen_util.slice_idx(dim, op)
        data = data[data_idx]
    else:
        if rev:
            fir, sec = [0, 1]
        else:
            fir, sec = [1, 0]
        if dim == 0: # allows for list
            fir_idx, sec_idx = fir, sec
        else:
            fir_idx = gen_util.slice_idx(dim, fir)
            sec_idx = gen_util.slice_idx(dim, sec)
        if op == "diff":
            data = (data[fir_idx] - data[sec_idx])
        elif op == "ratio":
            data = (data[fir_idx]/data[sec_idx])
        elif op == "rel_diff":
            data = (data[fir_idx] - data[sec_idx])/ \
                (data[fir_idx] + data[sec_idx])
        elif op == "d-prime":
            mean_diff = (np.mean(data[fir_idx], axis=axis) 
                        - np.mean(data[sec_idx], axis=axis))
            stds = (np.std(data[fir_idx], axis=axis),
                   np.std(data[sec_idx], axis=axis))
            div = np.sqrt(0.5 * np.sum(np.power(stds, 2), axis=0))
            data = mean_diff/div
        else:
            gen_util.accepted_values_error(
                "op", op, ["diff", "ratio", "rel_diff", "d-prime"])
    
    return data


#############################################
def scale_fact_names(sc_type="min_max", extrem="reg"):
    """
    scale_fact_names()

    Returns names of factors returned by scale_facts() depending on the 
    scaling type and extrema used.

    Optional args:
        - sc_type (str) : type of scaling to use
                          "min_max"  : (data - min)/(max - min)
                          "scale"    : (data - 0.0)/std
                          "stand"    : (data - mean)/std
                          "stand_rob": (data - median)/IQR (75-25)
                          "center"   : (data - mean)/1.0
                          "unit"     : (data - 0.0)/abs(mean)
                          default: "min_max"
        - extrem (str)  : only needed if min_max  or stand_rob scaling is used. 
                          "reg": the minimum and maximum (min_max) or 25-75 IQR 
                                 of the data are used 
                          "perc": the 5th and 95th percentiles are used as min
                                  and max respectively (robust to outliers)
    
    Returns:
        - sub (str): type of statistic subtracted from the data
        - div (str): type of statistic by which the data is divided
    """

    if sc_type == "stand":
        sub = "mean"
        div = "std"
    elif sc_type == "stand_rob":
        sub = "median"
        if extrem == "reg":
            div = "IQR"
        elif extrem == "perc":
            div = "IQR_5_95"
        else:
            gen_util.accepted_values_error("extrem", extrem, ["reg", "perc"])
    elif sc_type == "center":
        sub = "mean"
        div = "unit"
    elif sc_type == "scale":
        sub = "null"
        div = "std"
    elif sc_type == "unit":
        sub = "null"
        div = "abs_mean"
    elif sc_type == "min_max":
        if extrem == "reg":
            sub = "minim"
            div = "range"
        elif extrem == "perc":
            sub = "p5"
            div = "IQR_5_95"
    else:
        gen_util.accepted_values_error("sc_type", sc_type, 
                 ["stand", "stand_rob", "center", "scale", "min_max"])

    return sub, div


#############################################
def scale_facts(data, axis=None, pos=None, sc_type="min_max", extrem="reg", 
                mult=1.0, shift=0.0, nanpol=None, allow_0=False):
    """
    scale_facts(data)

    Returns scaling factors.

    Required args:
        - data (nd array): data to scale

    Optional args:
        - axis (int)    : axis along which to calculate scaling values (if None, 
                          entire data array is used)     
        - pos (int)     : position along axis along which to calculate scaling 
                          values (if None, each position is scaled separately)
        - sc_type (str) : type of scaling to use
                          "min_max"  : (data - min)/(max - min)
                          "scale"    : (data - 0.0)/std
                          "stand"    : (data - mean)/std
                          "stand_rob": (data - median)/IQR (75-25)
                          "center"   : (data - mean)/1.0
                          "unit"     : (data - 0.0)/abs(mean)
                          default: "min_max"
        - extrem (str)  : only needed if min_max  or stand_rob scaling is used. 
                          "reg": the minimum and maximum (min_max) or 25-75 IQR 
                                 of the data are used 
                          "perc": the 5th and 95th percentiles are used as min
                                  and max respectively (robust to outliers)
        - mult (num)    : value by which to multiply scaled data
                          default: 1.0
        - shift (num)   : value by which to shift scaled data (applied after
                          mult)
                          default: 0.0
        - nanpol (str)  : policy for NaNs, "omit" or None
                          default: None
        - allow_0 (bool): if True, div == 0 is allowed (likely resulting from 
                          np.nans)

    Returns:
        - sub (float or list): value(s) to subtract from scaled data
        - div (float or list): value(s) by which to divide scaled data
        - mult (num)         : value by which to multiply scaled data
        - shift (num)        : value by which to shift scaled data (applied 
                               after mult)
    """  

    if pos is not None and axis is None:
        raise ValueError("Must pass an axis if passing a position.")
    
    if pos is not None:
        sc_idx = gen_util.slice_idx(axis, pos) # for a slice
        axis = None
    else:
        sc_idx = gen_util.slice_idx(None, None) # for entire data

    if sc_type == "stand":
        sub = mean_med(data[sc_idx], stats="mean", axis=axis, nanpol=nanpol)
        div = error_stat(data[sc_idx], stats="mean", error="std", axis=axis, 
            nanpol=nanpol)
    elif sc_type == "stand_rob":
        sub = mean_med(data[sc_idx], stats="median", axis=axis, nanpol=nanpol)
        if extrem == "reg":
            qus = [25, 75]
        elif extrem == "perc":
            qus = [5, 95]
        else:
            gen_util.accepted_values_error("extrem", extrem, ["reg", "perc"])
        qs  = error_stat(
            data[sc_idx], stats="median", error="std", axis=axis, qu=qus, 
            nanpol=nanpol)
        div = qs[1] - qs[0]
    elif sc_type == "center":
        sub = mean_med(data[sc_idx], stats="mean", axis=axis, nanpol=nanpol)
        div = 1.0
    elif sc_type == "scale":
        sub = 0.0
        div = error_stat(
            data[sc_idx], stats="mean", error="std", axis=axis, nanpol=nanpol)
    elif sc_type == "unit":
        sub = 0.0
        div = np.absolute(mean_med(data[sc_idx], stats="mean", axis=axis, 
                                   nanpol=nanpol))
    elif sc_type == "min_max":
        if nanpol is not None and nanpol != "omit":
            gen_util.accepted_values_error("nanpol", nanpol, ["[None]", "omit"])
        if extrem == "reg":
            if nanpol is None:
                minim = np.min(data[sc_idx], axis=axis)
                maxim = np.max(data[sc_idx], axis=axis)
            elif nanpol == "omit":
                minim = np.nanmin(data[sc_idx], axis=axis)
                maxim = np.nanmax(data[sc_idx], axis=axis)
        elif extrem == "perc":
            if nanpol is None:
                minim = np.percentile(data[sc_idx], 5, axis=axis)
                maxim = np.percentile(data[sc_idx], 95, axis=axis)
            elif nanpol == "omit":
                minim = np.nanpercentile(data[sc_idx], 5, axis=axis)
                maxim = np.nanpercentile(data[sc_idx], 95, axis=axis)
        else:
            gen_util.accepted_values_error("extrem", extrem, ["reg", "perc"])
        sub = minim
        div = maxim - minim
    else:
        gen_util.accepted_values_error(
            "sc_type", sc_type, 
            ["stand", "stand_rob", "center", "scale", "min_max"])
    
    if not allow_0 and (np.asarray(div) == 0).any():
        raise RuntimeError("Scaling cannot proceed due to division by 0.")

    if isinstance(sub, np.ndarray):
        sub = sub.tolist()
    if isinstance(div, np.ndarray):
        div = div.tolist()

    return sub, div, mult, shift


#############################################
def shift_extrem(data, ext_p=[5, 95]):
    """
    Returns data array with values above and below the threshold percentiles 
    replaced with the nearest threshold, for each channel.

    Required args:
        - data (2D array): data array, structured as vals x channels

    Optional args:
        - ext_p (list): percentile values to use [low, high]
                        default: [5, 95]
    
    Returns:
        - data (2D array): data array with extreme values replaced by nearest 
                           threshold, structured as vals x channels
    """

    data = copy.deepcopy(data)

    p_lo, p_hi = ext_p

    if p_hi < p_lo:
        raise ValueError("p_lo must be smaller than p_hi.")
    lo, hi = [np.nanpercentile(data, p, axis=0).reshape([1, -1]) 
        for p in ext_p]

    above = np.where(data > hi)
    data[above] = hi[np.arange(len(hi)), above[1]]

    below = np.where(data < lo)
    data[below] = lo[np.arange(len(lo)), below[1]]

    return data


#############################################
def scale_data(data, axis=None, pos=None, sc_type="min_max", extrem="reg", 
               mult=1.0, shift=0.0, facts=None, nanpol=None):
    """
    scale_data(data)

    Returns scaled data, and factors if None are passed.

    Required args:
        - data (nd array): data to scale

    Optional args:
        - axis (int)   : axis to collapse when scaling values (if None, 
                         entire data array is collapsed)   
        - pos (int)    : position along axis to retain when calculating scaling 
                         values (if None, each position is scaled separately)
        - sc_type (str) : type of scaling to use
                          "min_max"  : (data - min)/(max - min)
                          "scale"    : (data - 0.0)/std
                          "stand"    : (data - mean)/std
                          "stand_rob": (data - median)/IQR (75-25)
                          "center"   : (data - mean)/1.0
                          "unit"     : (data - 0.0)/abs(mean)
                          default: "min_max"
        - extrem (str)  : only needed if min_max  or stand_rob scaling is used. 
                          "reg": the minimum and maximum (min_max) or 25-75 IQR 
                                 of the data are used 
                          "perc": the 5th and 95th percentiles are used as min
                                  and max respectively (robust to outliers)
        - mult (num)   : value by which to multiply scaled data
                         default: 1.0
        - shift (num)  : value by which to shift scaled data (applied after
                         mult)
                         default: 0.0
        - facts (list) : list of sub, div, mult and shift values to use on data
                         (overrides sc_type, extrem, mult and shift), where
                         sub is the value subtracted and div is the value
                         used as divisor (before applying mult and shift)
                         default: None
        - nanpol (str) : policy for NaNs, "omit" or None
                         default: None

    Returns:
        - sc_data (nd array): scaled data
        if facts value passed is None:
        - facts (list)      : list of sub, div, mult and shift values used on
                              data, where sub is the value(s) subtracted and 
                              div is the value(s) used as divisor(s) (before 
                              applying mult and shift)
    """  
    
    ret_facts = False
    if facts is None:
        facts = scale_facts(data, axis, pos, sc_type, extrem=extrem, mult=mult, 
            shift=shift, nanpol=nanpol)
        ret_facts = True
    elif len(facts) != 4:
        raise ValueError("If passing factors, must pass 4 items: "
            "sub, div, mult and shift.")

    sub, div, mult, shift = [np.asarray(fact).astype(float) for fact in facts]

    if axis is not None:
        sub = np.expand_dims(sub, axis)
        div = np.expand_dims(div, axis)
    
    data = (data - sub)/div * mult + shift

    if ret_facts:
        return data, facts

    else:
        return data


#############################################
def calc_mag_change(data, change_dim, item_dim, order=1, op="diff", 
                    stats="mean", error="sem", scale=False, axis=0, pos=0, 
                    sc_type="unit"):
    """
    calc_mag_change(data, change_dim, item_dim)

    Returns the magnitude diff/ratio or statistics of diff/ratio between 
    dimensions.

    Required args:
        - data (nd array) : data, with at least 2 dimensions or 3 if scaling
        - change_dim (int): dimension along which to calculate change
        - item_dim (int)  : dimension along which to scale or take statistics
    
    Optional args:
        - order (int)    : order of the norm (or "stats" to take change stats)
                           default: 1
        - op (str)       : "diff": index 1 - 0, or "ratio": index 1/0, or 
                           "rel_diff": (index 1 - 0)/(index 1 + 0)
                           default: "diff"
        - stats (str)    : stats to take, i.e., "mean" or "median"
                           default: "mean"
        - error (str)    : error to take, i.e., "std" (for std or quintiles) 
                           or "sem" (for SEM or MAD)
                           default: "std"
        - scale (bool)   : if True, data is scaled using axis, pos and sc_type
                           default: False
        - axis (int)     : axis along which to calculate scaling values (if  
                           None, entire data array is used)     
        - pos (int)      : position along axis along which to calculate scaling 
                           values (if None, each position is scaled separately)
        - sc_type (str) : type of scaling to use
                          "min_max"  : (data - min)/(max - min)
                          "scale"    : (data - 0.0)/std
                          "stand"    : (data - mean)/std
                          "stand_rob": (data - median)/IQR (75-25)
                          "center"   : (data - mean)/1.0
                          "unit"     : (data - 0.0)/abs(mean)
                          default: "min_max"

    Returns:
        if order == "stats:
            - data_ch_stats (nd array): array of magnitude change statistics,
                                        where first dimension are the stats
                                        (me, de)
        elif order is an int:
            - data_ch_norm (nd array) : array of norm values
    """

    if op not in ["diff", "ratio"]:
        raise ValueError("op can only take values 'diff' or 'ratio'.")
    
    data_change = np.absolute(calc_op(data, op, dim=change_dim))

    if item_dim > change_dim: # adjust dimension if needed
        item_dim -= 1

    if scale and axis is not None and axis > change_dim: # adjust dim if needed
        axis += -1

    if order == "stats":
        if scale:
            data_change, _ = scale_data(data_change, axis, pos, sc_type)
        data_ch_stats = get_stats(data_change, stats, error, axes=item_dim)
        return data_ch_stats
    else:
        data_ch_norm = np.linalg.norm(
            data_change, ord=int(order), axis=item_dim)
        if scale:
            data_ch_norm, _ = scale_data(data_ch_norm, axis, pos, sc_type)
        return data_ch_norm


#############################################
def check_n_rand(n_rand=1000, p_val=0.05, min_n=MIN_N, raise_err=True):
    """
    check_n_rand()

    Checks whether number of random values is sufficient to evaluate the 
    specified p-value, and raises an error if not.
    """

    out_vals = int(n_rand * p_val)
    if out_vals < min_n:
        error_msg = (f"Insufficient number of values ({out_vals}) outside "
            f"the CI (min. {min_n}) if using {n_rand} random values.")
        if raise_err:
            raise RuntimeError(error_msg)
        else:
            warnings.warn(error_msg, category=RuntimeWarning, stacklevel=1)


#############################################
def adj_n_perms(n_comp, p_val=0.05, n_perms=10000, min_n=MIN_N):
    """
    adj_n_perms(n_comp)

    Returns new p-value, based on the original p-value and a new number of 
    permutations using a Bonferroni correction.
    
    Specifically, the p-value is divided by the number of comparisons,
    and the number of permutations is increased if necessary to ensure a 
    sufficient number of permuted datapoints will be outside the CI 
    to properly measure the significance threshold.

    Required args:
        - n_comp (int): number of comparisons
    
    Optional args:
        - n_perms (int): original number of permutations
                         default: 10,000
        - p_val (num)  : original p_value
                         default: 0.05
        - min_n (int)  : minimum number of values required outside of the CI
                         default: 20
    
    Return:
        - new_p_val (num)  : new p-value
        - new_n_perms (num): new number of permutations
    """
    
    new_p_val   = float(p_val) / n_comp
    new_n_perms = int(np.ceil(np.max([n_perms, float(min_n) / new_p_val])))

    return new_p_val, new_n_perms


#############################################
def run_permute(all_data, n_perms=10000, lim_e6=LIM_E6_SIZE):
    """
    run_permute(all_data)

    Returns array containing data permuted the number of times requested. Will
    throw an AssertionError if permuted data array is projected to exceed 
    limit. 

    Required args:
        - all_data (2D array)  : full data on which to run permutation
                                 (items x datapoints to permute (all groups))

    Optional args:
        - n_perms  (int): nbr of permutations to run
                          default: 10000
        - lim_e6 (num)  : limit (when multiplied by 1e6) to permuted data array 
                          size at which an AssertionError is thrown. "none" for
                          no limit.
                          Default value be set externally by setting 
                          LIM_E6_SIZE as an environment variable.
                          default: LIM_E6_SIZE
    Returns:
        - permed_data (3D array): array of multiple permutation of the data, 
                                  structured as: 
                                  items x datapoints x permutations
    """

    if len(all_data.shape) > 2:
        raise NotImplementedError("Permutation analysis only implemented for "
            "2D data.")

    # checks final size of permutation array and throws an error if
    # it is bigger than accepted limit.
    perm_size = np.product(all_data.shape) * n_perms
    if lim_e6 != "none":
        lim = int(lim_e6*1e6)
        fold = int(np.ceil(float(perm_size)/lim))
        permute_cri = (f"Permutation array is up to {fold}x allowed size "
            f"({lim_e6} * 10^6).")
        if perm_size > lim:
            raise RuntimeError(permute_cri)

    # (item x datapoints (all groups))
    # (sample with n_perms in first dimension, so results are consistent 
    # within permutations, for different values of n_perms)
    perms_idxs = np.argsort(
        np.random.rand(n_perms, all_data.shape[1]).T, axis=0
        )[np.newaxis, :, :]

    dim_data   = np.arange(all_data.shape[0])[:, np.newaxis, np.newaxis]

    # generate permutation array
    permed_data = np.stack(all_data[dim_data, perms_idxs])

    return permed_data


#############################################
def permute_diff_ratio(all_data, div="half", n_perms=10000, stats="mean", 
                       nanpol=None, op="diff"):       
    """
    permute_diff_ratio(all_data)

    Returns all group mean/medians or differences/ratios between two groups 
    resulting from the permutation analysis on input data.

    Required args:
        - all_data (2D array): full data on which to run permutation
                               (items x datapoints to permute (all groups))

    Optional args:
        - div (str or int)  : nbr of datapoints in first group
                              default: "half"
        - n_perms (int)     : nbr of permutations to run
                              default: 10000
        - stats (str)       : statistic parameter, i.e. "mean" or "median" to 
                              use for groups
                              default: "mean"
        - nanpol (str)      : policy for NaNs, "omit" or None when taking 
                              statistics
                              default: None
        - op (str)          : operation to use to compare groups, 
                              "diff": grp2-grp1
                              "ratio": grp2/grp1
                              "rel_diff": (grp2-grp1)/(grp2+grp1)
                              "d-prime": (mean(index 1) - mean(index 0)) / 
                                         (sqrt(1/2 * 
                                             (std(index 1)**2 + std(index 0)**2)
                                         )
                              "none"
                              default: "diff"

    Returns:
        - all_rand_vals (2 or 3D array): permutation results, structured as:
                                             (grps if op is "none" x) 
                                             items x perms
    """

    if len(all_data.shape) > 2:
        raise NotImplementedError("Significant difference/ratio analysis only "
            "implemented for 2D data.")
    
    all_rand_res = []
    perm = True
    n_perms_tot = n_perms
    perms_done = 0

    if div == "half":
        div = int(all_data.shape[1]//2)

    while perm:
        try:
            perms_rem = n_perms_tot - perms_done
            if perms_rem < n_perms:
                n_perms = perms_rem
            permed_data = run_permute(all_data, n_perms=int(n_perms))

            if op != "d-prime":
                axis = None
                rand = np.stack([
                    mean_med(
                        permed_data[:, 0:div], stats, axis=1, nanpol=nanpol
                        ), 
                    mean_med(
                        permed_data[:, div:], stats, axis=1, nanpol=nanpol
                        )
                    ])
            else:
                # don't take mean yet
                axis = 1
                rand = [permed_data[:, 0:div], permed_data[:, div:]]

            if op == "none":
                rand_res = rand
            else:
                # calculate grp2-grp1 or grp2/grp1... -> elem x perms
                rand_res = calc_op(rand, op, dim=0, nanpol=nanpol, axis=axis)

            del permed_data
            del rand
            all_rand_res.append(rand_res)
            perms_done += n_perms

            if perms_done >= n_perms_tot:
                perm = False

        except RuntimeError as err:
            # retrieve ?x from error message.
            err_n_str = str(err)[: str(err).find("x allowed")]
            n = int(re.findall("\d+", err_n_str)[0])
            n_perms = int(n_perms // n)
            logger.warning(f"{err} Running in {n} batches.")

    all_rand_res = np.concatenate(all_rand_res, axis=-1)

    return all_rand_res


#############################################
def lin_interp_nan(data_arr):
    """
    lin_interp_nan(data_arr)

    Linearly interpolate NaNs in data array.

    Required args:
        - data_arr (1D array): data array

    Returns:
        - data_arr_interp (1D array): linearly interpolated data array
    """

    arr_len = len(data_arr)

    # get indices of non NaN values
    nan_idx = np.where(1 - np.isnan(data_arr))[0]

    arr_no_nans = data_arr[nan_idx]
    data_arr_interp = np.interp(range(arr_len), nan_idx, arr_no_nans)

    return data_arr_interp


#############################################
def get_percentiles(CI=0.95, tails=2):
    """
    get_percentiles()

    Returns percentiles and names corresponding to the confidence interval
    (centered on the median).

    Optional args:
        - CI (num)          : confidence interval
                              default: 0.95
        - tails (str or int): which tail(s) to test: "hi", "lo", "2"
                              default: 2

    Returns:
        - ps (list)     : list of percentile values, e.g., [2.5, 97.5]
        - p_names (list): list of percentile names, e.g., ["p2-5", "p97-5"]
    """

    if CI < 0 or CI > 1:
        raise ValueError("CI must be between 0 and 1.")

    CI *= 100

    if tails == "hi":
        ps = [0.0, CI]
    elif tails == "lo":
        ps = [100 - CI, 100]
    elif tails in ["2", 2]:
        ps = [0.5 * (100 + v) for v in [-CI, CI]]
    else:
        gen_util.accepted_values_error("tails", tails, ["hi", "lo", 2])

    p_names = []
    for p in ps:
        p_names.append(f"p{gen_util.num_to_str(p)}")

    return ps, p_names
    

#############################################
def log_elem_list(elems, tail="hi", act_vals=None):
    """
    log_elem_list(rand_vals, act_vals)

    Logs numbers of elements showing significant difference in a specific tail,
    and optionally their actual values.

    Required args:
        - elems (1D array): array of elements showing significant differences

    Optional args:
        - tails (str)        : which tail the elements are in: "hi", "lo"
                               default: "hi"
        - act_vals (1D array): array of actual values corresponding to elems 
                               (same length). If None, actual values are not 
                               logged.
    """

    if len(elems) == 0:
        logger.info(f"Signif {tail}: None", extra={"spacing": f"{TAB}{TAB}"})
    else:
        elems_pr = ", ".join(f"{x}" for x in elems)
        logger.info(
            f"Signif {tail}: {elems_pr}", extra={"spacing": f"{TAB}{TAB}"})
        if act_vals is not None:
            if len(act_vals) != len(elems):
                raise ValueError("'elems' and 'act_vals' should be the "
                    f"same length, but are of length {len(elems)} and "
                    f"{len(act_vals)} respectively.")
            vals_pr = ", ".join([f"{x:.2f}" for x in act_vals])
            logger.info(f"Vals: {vals_pr}", extra={"spacing": f"{TAB}{TAB}"})


#############################################    
def id_elem(rand_vals, act_vals, tails=2, p_val=0.05, min_n=20, 
            log_elems=False, ret_th=False, nanpol="omit", ret_pval=False):
    """
    id_elem(rand_vals, act_vals)

    Returns elements whose actual values are beyond the threshold(s) obtained 
    with null distributions of randomly generated values. 
    Optionally also returns the threshold(s) for each element.
    Optionally also logs significant element indices and their values.

    Required args:
        - rand_vals (2D array): random values for each element: elem x val
                                (or 1D, but will be treated as if it were 2D
                                 with 1 element)
        - act_vals (1D array) : actual values for each element

    Optional args:
        - tails (str or int): which tail(s) to test: "hi", "lo", "2", 2
                              default: 2
        - p_val (num)       : p-value to use for significance thresholding 
                              (0 to 1)
                              default: 0.05
        - min_n (int)       : minimum number of values required outside of the 
                              CI
                              default: 100
        - log_elems (bool)  : if True, the indices of significant elements and
                              their actual values are logged
                              default: False
        - ret_th (bool)     : if True, thresholds are returned for each element
                              default: False
        - nanpol (str)      : if "omit", NaNs in act_vals are allowed, and
                              prevented from leading to positive significance
                              evaluation
                              default: False
        - ret_pval (bool)   : if True, p-values are returned for each element
                              default: False

    Returns:
        - elems (list): list of elements showing significant differences, or 
                        list of lists if 2-tailed analysis [lo, hi].
        if ret_th, also:
        - threshs (list): list of threshold(s) for each element, either one 
                          value per element if 1-tailed analysis, or list of 2 
                          thresholds if 2-tailed [lo, hi].
        if ret_pval, also:
        - act_pvals (list): list of p-values for each element.
    """

    act_vals  = np.asarray(act_vals)
    rand_vals = np.asarray(rand_vals)

    single = False
    if len(rand_vals.shape) == 1:
        single = True

    if nanpol == "omit":
        act_vals = copy.deepcopy(act_vals)
        rand_vals = copy.deepcopy(rand_vals)
        nan_idx = np.where(~np.isfinite(act_vals))[0]
        if len(nan_idx) != 0:
            if single:
                act_vals  = np.asarray(np.nan)
                rand_vals = np.full(len(rand_vals), np.nan)
            else:
                act_vals[nan_idx]  = 1 # to prevent positive signif evaluation
                rand_vals[nan_idx] = 1

    nan_act_vals  = np.isnan(act_vals).any()
    nan_rand_vals = np.isnan(rand_vals).any()

    if nan_act_vals > 0:
        raise RuntimeError("NaNs encountered in actual values.")
    if nan_rand_vals > 0:
        raise RuntimeError("NaNs encountered in random values.")

    check_n_rand(rand_vals.shape[-1], p_val)

    if tails == "lo":
        threshs = np.percentile(rand_vals, p_val*100, axis=-1)
        elems = np.where(act_vals < threshs)[0]
        if log_elems:
            log_elem_list(elems, "lo", act_vals[elems])
        elems = elems.tolist()
    elif tails == "hi":
        threshs = np.percentile(rand_vals, 100-p_val*100, axis=-1)
        elems = np.where(act_vals > threshs)[0]
        if log_elems:
            log_elem_list(elems, "hi", act_vals[elems])
        elems = elems.tolist()
    elif str(tails) == "2":
        lo_threshs = np.percentile(rand_vals, p_val*100/2., axis=-1)
        lo_elems = np.where(act_vals < lo_threshs)[0]
        hi_threshs = np.percentile(rand_vals, 100-p_val*100/2., axis=-1)
        hi_elems = np.where(act_vals > hi_threshs)[0]
        if log_elems:
            log_elem_list(lo_elems, "lo", act_vals[lo_elems])
            log_elem_list(hi_elems, "hi", act_vals[hi_elems])
        elems = [lo_elems.tolist(), hi_elems.tolist()]
    else:
        gen_util.accepted_values_error("tails", tails, ["hi", "lo", "2"])
    
    if ret_pval:
        act_percs = [scist.percentileofscore(sub_rand_vals, val) 
            for (sub_rand_vals, val) in zip(rand_vals, act_vals)]
        act_pvals = []
        for perc in act_percs:
            if str(tails) in ["hi", "2"] and perc > 50:
                perc = 100 - perc
            act_pvals.append(perc / 100)

    returns = [elems]
    if ret_th:
        if tails in ["lo", "hi"]:
            if single:
                threshs = [threshs]
            else:
                threshs = threshs.tolist()
        else:
            if single:
                threshs = [[lo_threshs, hi_threshs]]
            else:
                threshs = [[lo, hi] for lo, hi in zip(lo_threshs, hi_threshs)]
        returns = returns + [threshs]
    if ret_pval:
        returns = returns + [act_pvals]
    
    if not ret_pval and not ret_th:
        returns = elems

    return returns


#############################################
def get_p_val_from_rand(act_data, rand_data, return_CIs=False, p_thresh=0.05, 
                        tails=2, multcomp=None):
    """
    get_p_val_from_rand(act_data, rand_data)

    Returns p-value obtained from a random null distribution of the data.

    Required args:
        - act_data (num)      : actual data
        - rand_data (1D array): random values
    
    Optional args:
        - return_CIs (bool) : if True, confidence intervals (CI) are returned 
                              as well
                              default: False
        - p_thresh (float)  : p-value to use to build CI
                              default: 0.05
        - tails (str or int): which tail(s) to use in building CI, and 
                              inverting p-values above 0.5
                              default: 2
        - multcomp (int)    : number of comparisons to correct CI for
                              default: None

    Returns:
        - p_val (float): p-value calculated from a randomly generated 
                         null distribution (not corrected)
        if return_CIs:
        - null_CI (list): null confidence interval (low, median, high), 
                          adjusted for multiple comparisons
    """

    sorted_rand_data = np.sort(rand_data)
    if len(rand_data.shape) != 1:
        raise ValueError("Expected rand_data to be 1-dimensional.")

    perc = scist.percentileofscore(sorted_rand_data, act_data, kind='mean')
    if str(tails) in ["hi", "2"] and perc > 50:
        perc = 100 - perc
    p_val = perc / 100

    if return_CIs:
        multcomp = 1 if not multcomp else multcomp
        corr_p_thresh = p_thresh / multcomp
        check_n_rand(len(sorted_rand_data), p_val=corr_p_thresh)

        percs = get_percentiles(CI=(1 - corr_p_thresh), tails=tails)[0]
        percs = [percs[0], 50, percs[1]]
        null_CI = [np.percentile(sorted_rand_data, p, axis=-1) for p in percs]
        
        return p_val, null_CI
    
    else:
        return p_val


#############################################
def get_diff_p_val(act_data, n_perms=10000, stats="mean", op="diff", 
                   return_CIs=False, p_thresh=0.05, tails=2, multcomp=None, 
                   paired=False):
    """
    get_diff_p_val(act_data)

    Returns p-value obtained from a random null distribution constructed from 
    the actual data.

    Required args:
        - act_data (array-like): full data on which to run permutation
                                 (groups x datapoints to permute)
    
    Optional args:
        - n_perms (int)     : number of permutations
                              default: 10000
        - stats (str)       : stats to use for each group
                              default: "mean"
        - op (str)          : operation to use to compare the group stats
                              (see calc_op())
                              default: "diff"
        - return_CIs (bool) : if True, confidence intervals (CI) are returned 
                              as well
                              default: False
        - p_thresh (float)  : p-value to use to build CI
                              default: 0.05
        - tails (str or int): which tail(s) to use in building CI
                              default: 2
        - multcomp (int)    : number of comparisons to correct CI for
                              default: None
        - paired (bool)     : if True, paired comparisons are done.  
                              default: False

    Returns:
        - p_val (float) : p-value calculated from a randomly generated 
                          null distribution (not corrected)
        if return_CIs:
        - null_CI (list): null confidence interval (low, median, high), 
                          adjusted for multiple comparisons
    """

    if len(act_data) != 2:
        raise ValueError("Expected 'act_data' to comprise 2 groups.")

    grp1, grp2 = [np.asarray(grp_data) for grp_data in act_data]

    if len(grp1.shape) != 1 or len(grp2.shape) != 1:
        raise ValueError("Expected act_data groups to be 1-dimensional.")

    real_diff = mean_med(grp2, stats=stats) - mean_med(grp1, stats=stats)

    if not paired:
        concat = np.concatenate([grp1, grp2], axis=0).reshape(1, -1) # add items dim
        div = len(grp1)
    else:
        if len(grp1) != len(grp2):
            raise ValueError(
                "If data is paired, groups must have the same length."
                )
        concat = np.vstack([grp1, grp2]).T # datapoints x groups
        div = 1 # permuting 2 values for each datapoint, since data is paired

    rand_diffs = permute_diff_ratio(
        concat, div=div, n_perms=n_perms, stats=stats, op=op
        )

    if not paired:
        rand_diffs = np.squeeze(rand_diffs)
        rand_diffs = \
            rand_diffs.reshape(1) if len(rand_diffs.shape) == 0 else rand_diffs
    else:
        # permute within rows, as permute_diff_ratio permutes columns together! 
        rand_perm = np.argsort(
            np.random.permutation(rand_diffs.size).reshape(rand_diffs.shape),
            axis=1)
        data_idx = np.arange(len(rand_diffs)).reshape(-1, 1)
        rand_diffs = rand_diffs[data_idx, rand_perm]
        rand_diffs = mean_med(rand_diffs, stats=stats, axis=0)

    returns = get_p_val_from_rand(
        real_diff, rand_diffs, return_CIs=return_CIs, p_thresh=p_thresh, 
        tails=tails, multcomp=multcomp
        )
    
    if return_CIs:
        p_val, null_CI = returns
        return p_val, null_CI

    else:
        p_val = returns
        return p_val


#############################################
def comp_vals_acr_groups(act_data, n_perms=None, normal=True, stats="mean", 
                         paired=False):
    """
    comp_vals_acr_groups(act_data)

    Returns p values for comparisons across groups (unpaired).

    Required args:
        - act_data (array-like): full data on which to run permutation
                                 (groups x datapoints to permute)

    Optional args:
        - n_perms (int): number of permutations to do if doing a permutation 
                         test. If None, a different test is used
                         default: None
        - stats (str)  : stats to use for each group
                         default: "mean"
        - normal (bool): whether data is expected to be normal or not 
                         (determines whether a t-test or Mann Whitney test 
                         will be done. Ignored if n_perms is not None.)
                         default: True
        - paired (bool): if True, paired comparisons are done.  
                         default: False

    Returns:
        - p_vals (1D array): p values for each comparison, organized by 
                             group pairs (where the second group is cycled 
                             in the inner loop, e.g., 0-1, 0-2, 1-2, including 
                             None groups)
    """

    n_comp = sum(range(len(act_data)))
    p_vals = np.full(n_comp, np.nan)
    i = 0
    for g, g_data in enumerate(act_data):
        for g_data_2 in act_data[g + 1:]:
            if g_data is not None and g_data_2 is not None and \
                len(g_data) != 0 and len(g_data_2) != 0:
                
                if n_perms is not None:
                    p_vals[i] = get_diff_p_val(
                        [g_data, g_data_2], n_perms, stats=stats, op="diff", 
                        paired=paired)
                elif normal:
                    fct = scist.ttest_rel if paired else scist.ttest_ind
                    p_vals[i] = fct(g_data, g_data_2, axis=None)[1]
                else:
                    fct = scist.wilcoxon if paired else scist.mannwhitneyu 
                    p_vals[i] = fct(g_data, g_data_2)[1]
            i += 1
    
    return p_vals


#############################################
def autocorr(data, lag):
    """
    Calculates autocorrelation on data series.

    Required args:
        - data (1D array): 1D dataseries
        - lag (int)      : lag in steps
    
    Returns:
        - autoc_snip (1D array): 1D array of autocorrelations at specified lag
    """

    autoc = np.correlate(data, data, "full")
    mid = int((autoc.shape[0] - 1)//2)
    autoc_snip = autoc[mid - lag:mid + lag + 1]
    autoc_snip /= np.max(autoc_snip)
    return autoc_snip


#############################################
def autocorr_stats(data, lag, spu=None, byitem=True, stats="mean", error="std", 
                   nanpol=None):
    """
    autocorr_stats(data, lag)
    
    Returns average autocorrelation across data series.

    Required args:
        - data (list or 2-3D array): list of series or single series 
                                     (2D array), where autocorrelation is 
                                     calculated along the last axis. 
                                     Structured as: 
                                         (blocks x ) item x frame
        - lag (num)                : lag for which to calculate 
                                     autocorrelation (in steps ir in units 
                                     if steps per units (spu) is provided).

    Optional args:
        - spu (num)    : spu (steps per unit) value to calculate lag in steps
                         default: None
        - byitem (bool): if True, autocorrelation statistics are taken by 
                         item, else across items
                         default: True
        - stats (str)  : statistic parameter, i.e. "mean" or "median"
                         default: "mean"
        - error (str)  : error statistic parameter, i.e. "std" or "sem"
                         default: "std
        - nanpol (str) : policy for NaNs, "omit" or None when taking statistics
                         default: None

    Returns:
        - xran (array-like)             : range of lag values in frames or in
                                          units if fpu is not None.
                                          (length is equal to last  
                                          dimension of autocorr_stats) 
        - autocorr_stats (2 or 3D array): autocorr statistics, structured as 
                                          follows:
                                          stats (me, de) x (item if item x) lag
    """
    
    if spu is None:
        lag_fr = int(lag)
    else:
        lag_fr = int(lag * spu)

    snip_len = 2 * lag_fr + 1

    data = gen_util.list_if_not(data)
    n_series = len(data)
    n_items  = len(data[0])

    autocorr_snips = np.empty((n_series, n_items, snip_len))

    for s, series in enumerate(data):
        sc_vals = series - np.mean(series, axis=1)[:, np.newaxis]
        for i, item in enumerate(sc_vals):
            autocorr_snips[s, i] = autocorr(item, lag_fr)

    xran = np.linspace(-lag, lag, snip_len)

    # take autocorrelations statistics for each lag across blocks
    if byitem:
        axes = 0
    else:
        axes = [0, 1]

    autocorr_stats = get_stats(
        autocorr_snips, stats, error, axes=axes, nanpol=nanpol)

    return xran, autocorr_stats


#############################################
def calculate_snr(data, return_stats=False):
    """
    calculate_snr(data)
    
    Returns SNR for data (std of estimated noise / mean of signal).

    Required args:
        - data (1D array): data for which to calculate SNR (flattened if not 1D)

    Optional args:
        - return_stats (bool): if True, additional stats are returned
                               default: False

    Returns:
        - snr (float): SNR of data
        if return_stats:
        - data_median (float)  : median of full data
        - noise_data (1D array): noisy data
        - noise_mean (float)   : mean of the noise 
        - noise_std (float)    : standard deviation of the noise
        - noise_thr (float)    : noise threshold
        - signal_mean (float)  : mean of the signal
    """
    
    data = np.asarray(data).reshape(-1)
    data_median = np.median(data)

    lower_vals = np.where(data <= data_median)[0]
    noise_data = np.concatenate(
        [data[lower_vals], 2 * data_median - data[lower_vals]]
        )

    noise_mean = np.mean(noise_data)
    noise_std = np.std(noise_data)
    noise_thr = scist.norm.ppf(0.95, noise_mean, noise_std)
    signal_mean = np.mean(data[np.where(data > noise_thr)])

    snr = signal_mean / noise_std
    
    if return_stats:
        return [snr, data_median, noise_data, noise_mean, noise_std, 
            noise_thr, signal_mean]
    else:
        return snr


#############################################
def get_order_of_mag(val):
    """
    get_order_of_mag(val)
    
    Returns order of magnitude for a value.

    Required args:
        - val (float): value to round
    
    Returns:
        - order (int): order of magnitude for rounding value
    """

    order = int(np.floor(np.log10(val)))

    return order


#############################################
def round_by_order_of_mag(val, n_sig=1, direc="any", decimal_only=False):
    """
    round_by_order_of_mag(val)
    
    Returns value, rounded by the order of magnitude.

    Required args:
        - val (float): value to round
    
    Optional args:
        - n_sig (int)        : number of significant digits
                               default: 1
        - direc (str)        : direction in which to round value
                               default: "any"
        - decimal_only (bool): if True, only decimals are rounded
                               default: False

    Returns:
        - rounded_val (float): rounded value
    """
    
    if n_sig < 1:
        raise ValueError("'n_sig' must be at least 1.")

    o = int(-get_order_of_mag(val) + n_sig - 1)

    if decimal_only and o < 0:
        o = 0

    if direc == "any":
        rounded_val = np.around(val, o)
    elif direc == "up":
        rounded_val = np.ceil(val * 10**o) / 10**o
    elif direc == "down":
        rounded_val = np.floor(val * 10**o) / 10**o
    else:
        gen_util.accepted_values_error("direc", direc, ["any", "up", "down"])

    return rounded_val


#############################################
def bootstrapped_std(data, n=None, n_samples=1000, proportion=False, 
                     randst=None):
    """
    bootstrapped_std(data)
    
    Returns bootstrapped standard deviation of the mean or proportion.

    Required args:
        - data (float or 1D array): proportion or full data for mean
    
    Optional args:
        - n (int)          : number of datapoints in dataset. Required if 
                             proportion is True.
                             default: None
        - n_samples (int)  : number of samplings to take for bootstrapping
                             default: 1000
        - proportion (bool): if True, data is a proportion (0-1)
                             default: False
        - randst (int)     : seed or random state to use when generating random 
                             values.
                             default: None

    Returns:
        - bootstrapped_std (float): bootstrapped standard deviation of mean or 
                                    percentage
    """

    if randst is None:
        randst = np.random
    elif isinstance(randst, int):
        randst = np.random.RandomState(randst) 

    # random values
    if proportion:
        if data < 0 or data > 1:
            raise ValueError("'data' must lie between 0 and 1 if it is a "
                "proportion.")
        if n is None:
            raise ValueError("Must provide n if data is a proportion.")
        
        # random proportions
        rand_data = np.mean(
            randst.rand(n * n_samples).reshape(n, n_samples) < data, 
            axis=0)
        
    else:
        if n is not None and n != len(data):
            raise ValueError("If n is provided, it must be the length of data.")
        n = len(data)

        choices = np.arange(n)
   
        # random means
        rand_data = np.mean(
            data[randst.choice(choices, (n, n_samples), replace=True)], 
            axis=0)

    bootstrapped_std = np.std(rand_data)
    
    return bootstrapped_std


#############################################
def binom_CI(p_thresh, perc, n_items, null_perc, tails=2, multcomp=None):
    """
    binom_CI(p_thresh, perc, n_items, null_perc)

    Returns theoretical confidence intervals over a percentage using binomial 
    distribution.

    NOTE: p_values and CIs may not correspond exactly, due to rounding effects,
    given that the binomial distribution is a discrete probability distribution.

    Specifically, scist.binom.ppf always returns an integer number of items, 
    regardless of the thresholds (p_thresh) requested. Thus, lower values of 
    n_items yield lower resolution CIs, with a larger range of p-value 
    thresholds outputting the same CI values.

    As a result, at the CI edges, p_val may be a better measure of 
    significance than the CIs.

    Required args:
        - p_thresh (float) : desired p-value, two-tailed, 
                             e.g. 0.05 for a CI ranging from 2.5 to 97.5
        - perc (float)     : percentage of significant items (0-100)
        - n_items (int)    : number of items
        - null_perc (float): null percentage expected (0-100)

    Optional args:
        - tails (str or int): which tail(s) to use in evaluating p-value
                              default: 2
        - multcomp (int)    : number of comparisons to correct CIs for
                              default: None

    Returns:
        - CI (1D array)     : low, median and high values for the confidence 
                              interval over the true percentage (perc)
        - null_CI (1D array): low, median and high values for the confidence 
                              interval over the null percentage expected 
                              (null_perc)
        - p_val (float)     : uncorrected p-value of the true percentage given 
                              the theoretical null distribution
    """

    multcomp = 1 if not multcomp else multcomp
    
    # Calculate the confidence interval for "frac_pos/sig".
    threshs = [p_thresh / (multcomp * 2), 0.5, 1 - p_thresh / (multcomp * 2)]
    CI_low, CI_med, CI_high = [
        scist.binom.ppf(thresh, n_items, perc / 100) / n_items 
        for thresh in threshs
        ]
    
    # Calculate the p-val of the true value for the null distro
    p_val = scist.binom.cdf(perc / 100 * n_items, n_items, null_perc / 100)

    if str(tails) in ["hi", "2"] and p_val > 0.50:
        p_val = 1 - p_val

    # Calculate the confidence interval for the null distro
    null_CI_low, null_CI_med, null_CI_high = [
        scist.binom.ppf(thresh, n_items, null_perc / 100) / n_items 
        for thresh in threshs
        ]
    
    CI = np.asarray([CI_low, CI_med, CI_high]) * 100
    null_CI = np.asarray([null_CI_low, null_CI_med, null_CI_high]) * 100
        
    return CI, null_CI, p_val

