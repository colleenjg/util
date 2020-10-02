"""
debug_util.py

This module contains basic debugging tools.

Authors: Colleen Gillon

Date: October, 2018

Note: this code uses python 3.7.

"""

import logging

from matplotlib import pyplot as plt
import numpy as np

from util import gen_util, logger_util, plot_util

logger = logging.getLogger(__name__)


#############################################
def set_debug_mode(function):
    """
    Wrapper for temporarily setting logging level to debug 
    while running a function.
    """

    def wrapper(*args, **kwargs):
        prev_level = logger.level
        logger.setLevel(logging.DEBUG)

        returns = function(*args, **kwargs)

        logger.setLevel(prev_level)

        return returns

    return wrapper


#############################################
#############################################
@set_debug_mode
def check_match(data1, data2, ret_diff=False):
    """
    check_match(data1, data2)

    Returns whether two data arrays match, including NaN patterns. Logs a 
    warning if NaN patterns do not match. Optionally return indices of 
    different values.

    Required args:
        - data1 (nd array): First data array
        - data2 (nd array): Second data array

    Optional args:
        - ret_diff (bool): if True, indices of different values also returned
                           default: False

    Returns
        - match_all (bool): Whether the two data arrays match.
        if ret_diff:
        - diff_idx (list): indices of different values, in each axis

    """

    if data1.shape != data2.shape:
        raise ValueError('Both data arrays must have same shape.')

    match = ((data1 == data2) | (np.isnan(data1) & np.isnan(data2)))
    if ~(np.isnan(data1) == np.isnan(data2)).all():
        logger.debug('Different NaN patterns.')
    match_all = match.all()
    if ret_diff:
        diff_idx = np.where(~match)
        return match_all, diff_idx
    else:
        return match_all
    

#############################################
@set_debug_mode
def calculate_data_diffs(data1, data2):
    """
    calculate_data_diffs(data1, data2)

    Returns differences between two data arrays.

    Required args:
        - data1 (nd array): First data array
        - data2 (nd array): Second data array

    Returns
        - non_zero_diffs (1D array): data differences
        - perc_diff (float)        : percentage of data values that are 
                                     different
        - nan_mask (nd array)      : array of indices where values are not 
                                     NaNs, in each axis

    """

    if data1.shape != data2.shape:
        raise ValueError('Both data arrays must have same shape.')
    # calculate some information
    nan_mask = np.where(np.isfinite(np.multiply(data1, data2))) # remove NaNs
    diff_data = data1[nan_mask] - data2[nan_mask]
    non_zero_diffs = diff_data[np.where(diff_data)]
    perc_diff = non_zero_diffs.size/data1.size * 100.

    return non_zero_diffs, perc_diff, nan_mask
    

#############################################
@set_debug_mode
def match_log(match, names=None):
    """
    match_log(match)

    Logs whether 2 named data arrays based on boolean input.

    Required args:
        - match (bool): Whether data arrays match

    Optional args:
        - names (list): list of data array names
                        default: None
    """

    if names is None:
        names = ['Data array 2', 'Data array 1']
    elif len(names) != 2:
        raise ValueError('If passing `names`, must pass 2 names.')

    if match:
        logger.debug(f'{names[1].capitalize()} matches {names[0].lower()}.')
    else:
        logger.debug(f'{names[1].capitalize()} does not exactly match '
            f'{names[0].lower()}.')


#############################################
@set_debug_mode
def log_max_diff(data1, data2, axis=0, axis_label='IDs'):
    """
    log_max_diff(data1, data2)

    Logs indices of items that are different between 2 data arrays, along a
    specified axis, and well as the maximum absolute difference between data
    arrays.

    Required args:
        - data1 (nd array): First data array
        - data2 (nd array): Second data array

    Optional args:
        - axis (int)       : data axis along which to identify indices with 
                             differences between data arrays
                             default: 0
        - axis_labels (str): label of axis
                             default: 'IDs'
    """
    
    _, diff_idx = check_match(data1, data2, ret_diff=True)
    diff_items = [str(roi_n) for roi_n in sorted(set(diff_idx[axis]))]
    diff_items_str = ', '.join(diff_items)
    logger.debug(f'Diff {axis_label}: {diff_items_str}')
    act_diff = (data2[diff_idx] - data1[diff_idx])
    logger.debug(f'Max absolute diff: {np.nanmax(np.absolute(act_diff))}')


#############################################
@set_debug_mode
def plot_diff_data(data1, data2, max=10, title=None, labels=None, 
                   xlabel=None, datatype=None):
    """
    plot_diff_data(data1, data2)

    Plots overlayed values in 2 data arrays in for items with the greatest
    absolute difference.

    Required args:
        - data1 (nd array): First data array
        - data2 (nd array): Second data array
    
    Optional args:
        - max (int)      : maximum number of items to plot
                           default: 10
        - title (str)    : subplot title
                           default: None
        - labels (list)  : data array labels ('None' for no legend)
                           default: None
        - xlabel (str)   : x axis label
                           default: None
        - datatype (str) : data type
                           default: None
    """


    if data1.shape != data2.shape:
        raise ValueError('Both data arrays must have same shape.')

    data = [data1, data2]
    cols = ['blue', 'red']
    use_legend = True
    if labels is None:
        labels = ['1', '2']
    elif labels in ['None', 'none']:
        labels = [None, None]
        use_legend = False
    elif len(labels) != 2:
        raise ValueError('If passing `labels`, must pass 2 labels.')
    
    one_dim = False
    if len(data[0].shape) == 1:
        data = [sub_data.reshape(1, -1) for sub_data in data]
        xlabel = datatype
        one_dim = True

    items_max_diffs = np.nansum(np.absolute(data[1] - data[0]), axis=1)
    n_not_zero = len(np.where((items_max_diffs != 0 * \
        ~np.isnan(items_max_diffs)))[0])
    
    if n_not_zero == 0:
        n_nan = np.isnan(items_max_diffs).sum()
        if n_nan:
            logger.debug(f'Differences in all {n_nan} {datatype} due to NaNs.')
        else:
            logger.debug(f'No {datatype} show differences.')
        return

    n_plot = np.min([max, n_not_zero])
    item_max_diff_idx = np.argsort(items_max_diffs)[-n_plot:][::-1]
    
    # set plotting parameters
    figsize=(14, 6)
    _, ax = plt.subplots(
        len(item_max_diff_idx), figsize=figsize, squeeze=False, sharex=True)

    if not one_dim:
        logger.debug(f'{n_plot}/{n_not_zero} {datatype} with greatest cumulative '
            'absolute differences shown')

    subtitle = ''
    if title is not None:
        subtitle = f'{title}: '
    title = (f'{subtitle}{len(item_max_diff_idx)}/{n_not_zero} {datatype} '
        f'with greatest cumul. abs. diffs ({len(data[0])} ROIs total)')
    
    for s, item_idx in enumerate(item_max_diff_idx):
        subax = ax.reshape(-1)[s]
        for sub_data, col, label in zip(data, cols, labels):
            subax.plot(sub_data[item_idx], c=col, alpha=0.3, label=label)
        subax.set_ylabel(item_idx)
        subax.yaxis.set_label_position('right')
        if s == 0:
            if use_legend:
                subax.legend()
            subax.set_title(title)
        if s == len(item_max_diff_idx) - 1:
            subax.set_xlabel(xlabel)


#############################################
@set_debug_mode
def plot_diff_distrib(data1, data2, bins=100, title=None, labels=None, 
                      datatype=None):
    """
    plot_diff_distrib(data1, data2)

    Plots overlayed distribution of values in 2 data arrays in one subplot 
    and distribution of differences in another.

    Required args:
        - data1 (nd array): First data array
        - data2 (nd array): Second data array
    
    Optional args:
        - bins (int)     : number of bins to split data into. If too big with 
                           respect to the number of data point (> half), will 
                           be cut in half until threshold is met.
                           default: 100
        - title (str)    : subplot title
                           default: None
        - labels (list)  : data array labels
                           default: None
        - datatype (str) : data type (used in x-axis label)
                           default: None
    """


    cols = ['blue', 'red']
    if labels is None:
        labels = ['1', '2']
    elif len(labels) != 2:
        raise ValueError('If passing `labels`, must pass 2 labels.')

    non_zero_diffs, perc_diff, nan_mask = calculate_data_diffs(data1, data2)

    if len(non_zero_diffs) == 0:
        n_nan = np.isnan(non_zero_diffs).sum()
        if n_nan:
            logger.debug(f'Differences in all {n_nan} {datatype} due to NaNs.')
        else:
            logger.debug(f'No {datatype} show differences.')
        return

    # set plotting parameters
    figsize=(14, 4)
    _, ax = plt.subplots(ncols=2, figsize=figsize)
    while bins > data1.size//2:
        bins = bins//2
    bins = np.max([2, bins])
    label_diffs = f'diff {perc_diff:.2f}%\nof values'
    
    # ROI trace histogram
    for sub_data, col, label in zip([data1, data2], cols, labels):
        ax[0].hist(
            sub_data[nan_mask], bins, color=col, alpha=0.3, label=label, 
            density=True)
    if title is not None:
        ax[0].set_title(title)
    if datatype is not None:
        ax[0].set_xlabel(datatype)
    ax[0].set_ylabel(f'Density ({bins} bins)')
    ax[0].legend()

    # Difference histogram
    ax[1].hist(
        non_zero_diffs, bins, color='green', alpha=0.6, label=label_diffs, 
        density=True)
    for fct in [np.min, np.max]:
        ax[1].axvline(fct(non_zero_diffs), c='red', ls='dashed')
    if title is not None:
        ax[1].set_title(title)
    ax[1].legend(loc='upper right')
    if datatype is not None:
        ax[1].set_xlabel(f'Non zero diffs in {datatype} values')

