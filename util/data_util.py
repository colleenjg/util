"""
data_util.py

This module contains basic dataset construction tools.

Authors: Colleen Gillon

Date: October, 2018

Note: this code uses python 3.7.

"""

import numpy as np

from util import gen_util, rand_util



#############################################
def bal_classes(data, targets, randst=None):
    """
    bal_classes(data, targets)

    Returns resampled data arrays where classes are balanced.

    Required args:
        - data (nd array)   : array of dataset datapoints, where the first
                              dimension is the samples.
        - targets (nd array): array of targets, where the first dimension
                              is the samples. Must be of the same length as 
                              data.

    Optional args:
        - randst (int): seed or random state to use when generating random 
                        values.
                        default: None

    Returns:
        - data (nd array)   : array of sampled dataset datapoints, where the
                              first dimension is the samples.
        - targets (nd array): array of sampled targets, where the first 
                              dimension is the samples.
    """

    randst = rand_util.get_np_rand_state(randst)    

    if len(data) != len(targets):
        raise ValueError("data and targets must be of the same length.")

    cl_n   = np.unique(targets).tolist()
    counts = np.unique(targets, return_counts=True)[1]
    
    count_min = np.min(counts)
    
    sample_idx = []
    for cl in cl_n:
        idx = randst.choice(np.where(targets==cl)[0], count_min, replace=False)
        sample_idx.extend(idx.tolist())
    
    sample_idx = sorted(sample_idx)

    data = data[sorted(sample_idx)]
    targets = targets[sample_idx]

    return data, targets


#############################################
def data_indices(n, train_n, val_n, test_n=None, targets=None, thresh_cl=2, 
                 strat_cl=True, randst=None):
    """
    data_indices(n, train_n, val_n)

    Returns dataset indices assigned randomly to training, validation and 
    testing sets.
    Allows for a set to be empty, and also allows for only a subset of all 
    indices to be assigned if test_n is provided.

    Will keep shuffling until each non empty set contains the minimum number of 
    occurrences per class, if targets is provided.

    Required args:
        - n (int)      : length of dataset
        - train_n (int): nbr of indices to assign to training set
        - val_n (int)  : nbr of indices to assign to validation set

    Optional args:
        - test_n (int)      : nbr of indices to assign to test set. If test_n 
                              is None, test_n is inferred from n, train_n and 
                              val_n so that all indices are assigned.
                              default: None
        - targets (nd array): array of targets, where the first dimension
                              is the samples. Must be of the same length 
                              as data.
                              default: None
        - thresh_cl (int)   : size threshold for classes in each non empty set 
                              beneath which the indices are reselected (only if
                              targets are passed). Raises an error if it is
                              impossible. 
                              default: 2
        - strat_cl (bool)   : if True, sets are stratified by class. 
                              default: True
        - randst (int)      : seed or random state to use when generating 
                              random values.
                              default: None

    Returns:
        - train_idx (list): unsorted list of indices assigned to training set.
        - val_idx (list)  : unsorted list of indices assigned to validation set.
        - test_idx (list) : unsorted list of indices assigned to test set. 
    """

    if test_n is None:
        test_n = n - train_n - val_n
    
    if (train_n + val_n + test_n) != n:
        raise ValueError("train_n, val_n and test_n must sum to n.")

    if targets is not None and len(targets) != n:
        raise ValueError("If targets are provided, must be as many as n.")

    mixed_idx = list(range(n))

    randst = rand_util.get_np_rand_state(randst)

    if targets is not None and strat_cl:
        cl_vals, cl_ns = np.unique(targets, return_counts=True)
        props = [float(cl_n)/n for cl_n in cl_ns.tolist()]
        train_idx, val_idx, test_idx = [], [], []
        for val, prop in zip(cl_vals, props):
            cl_mixed_idx = np.asarray(mixed_idx)[np.where(targets == val)[0]]
            randst.shuffle(cl_mixed_idx)
            set_ns    = [int(np.ceil(set_n * prop)) 
                for set_n in [0, val_n, test_n]]
            set_ns[0] = len(cl_mixed_idx) - sum(set_ns)
            for s, set_n in enumerate(set_ns):
                if [train_idx, val_idx, test_idx][s] != 0 and thresh_cl != 0:
                    if set_n < thresh_cl:
                        raise RuntimeError(
                            "Sets cannot meet the threshold requirement."
                            )
            train_idx.extend(cl_mixed_idx[0 : set_ns[0]])
            val_idx.extend(cl_mixed_idx[set_ns[0] : set_ns[0] + set_ns[1]])
            test_idx.extend(cl_mixed_idx[set_ns[0] + set_ns[1] : 
                set_ns[0] + set_ns[1] + set_ns[2]])
        
    else:
        cont_shuff = True
        while cont_shuff:
            randst.shuffle(mixed_idx)
            cont_shuff = False
            # count occurrences of each class in each non empty set and ensure 
            # above threshold, otherwise reshuff
            train_idx = mixed_idx[: train_n]
            val_idx = mixed_idx[train_idx : train_idx + val_n]
            test_idx = mixed_idx[train_idx + val_n :]

            # check whether reshuffling is needed
            if targets is not None and thresh_cl != 0: 
                # count number of classes
                n_cl = len(np.unique(targets).tolist())
                for s in [train_idx, val_idx, test_idx]:
                    if len(s) != 0: 
                        counts = np.unique(targets[s], return_counts=True)[1]
                        count_min = np.min(counts)
                        set_n_cl = len(counts)
                        # check all classes are in the set and above threshold
                        if count_min < thresh_cl or set_n_cl < n_cl:
                            cont_shuff = True

    return train_idx, val_idx, test_idx


#############################################
def check_prop(train_p, val_p=0, test_p=0):
    """
    check_prop(train_p)

    Checks that the proportions assigned to the sets are acceptable. Throws an
    error if proportions sum to greater than 1 or if a proportion is < 0. 
    Logs a warning (no error) if the sum to less than 1.
    
    Required args:
        - train_p (num): proportion of dataset assigned to training set

    Optional args:
        - val_p (num) : proportion of dataset assigned to validation set.
                        default: 0
        - test_p (num): proportion of dataset assigned to test set.
                        default: 0
    """

    set_p = [[x, y] for x, y in zip([train_p, val_p, test_p], 
        ["train_p", "val_p", "test_p"])]
    
    sum_p = sum(list(zip(*set_p))[0])
    min_p = min(list(zip(*set_p))[0])

    # raise error if proportions sum to > 1 or if a proportion is < 0.
    if sum_p != 1.0 or min_p < 0.0:
        props = [f"\n{y}: {x}" for x, y in set_p]
        prop_str = f"{''.join(props)}\nsum_p: {sum_p}"
        
        if min_p < 0.0:
            raise ValueError(f"Proportions must not be < 0. {prop_str}")

        elif sum_p > 1.0:
            raise ValueError(f"Proportions must not sum to > 1. {prop_str}")
    
        elif len(set_p) == 3:
        # if all values are given and sum != 1.0
            raise ValueError(f"Proportions given do not sum to 1. {prop_str}")


#############################################
def split_idx(n, train_p=0.75, val_p=None, test_p=None, thresh_set=10, 
              targets=None, thresh_cl=2, strat_cl=True):
    """
    split_idx(n)

    Returns dataset indices split into training, validation and test sets. If 
    val_p and test_p are None, the non training proportion is split between 
    them. If targets are passed, the number of targets from each class in the 
    sets are checked.

    Required args:
        - n (int): length of dataset

    Optional args:
        - train_p (num)     : proportion of dataset assigned to training set
                              default: 0.75
        - val_p (num)       : proportion of dataset assigned to validation set. 
                              If None, proportion is calculated based on 
                              train_p and test_p.
                              default: None
        - test_p (num)      : proportion of dataset assigned to test set. If 
                              None, proportion is calculated based on train_p 
                              and val_p.
                              default: None
        - thresh_set (int)  : size threshold for sets beneath which an error is
                              thrown if the set's proportion is not 0.
                              default: 10
        - targets (nd array): array of targets, where the first dimension
                              is the samples. Must be of the same length 
                              as data.
                              default: None
        - thresh_cl (int)   : size threshold for classes in each non empty set 
                              beneath which the indices are reselected (only if
                              targets are passed). Not checked if thresh_cl is 
                              0.
                              default: 2
        - strat_cl (bool)   : if True, sets are stratified by class. 
                              default: True

    Returns:
        - train_idx (list): unsorted list of indices assigned to training set.
        - val_idx (list)  : unsorted list of indices assigned to validation set.
        - test_idx (list) : unsorted list of indices assigned to test set. 
    """
    
    if val_p is None and test_p is None:
        # split half half
        val_p = (1.0 - train_p)/2
        test_p = val_p
    elif val_p is None:
        val_p = 1.0 - train_p - test_p
    elif test_p is None:
        test_p = 1.0 - train_p - val_p

    check_prop(train_p, val_p, test_p)
    
    val_n = int(np.ceil(val_p*n))
    test_n = int(np.ceil(test_p*n))
    train_n = n - val_n - test_n

    # raise error if val or test n is below threshold (unless prop is 0)
    for set_n, set_p, name in zip(
        [val_n, test_n], [val_p, test_p], ["val n", "test n"]):
        if set_n < thresh_set:
            if set_p != 0:
                raise RuntimeError(f"{name} is {set_n} (below threshold "
                    f"of {thresh_set})")

    train_idx, val_idx, test_idx = data_indices(
        n, train_n, val_n, test_n, targets, thresh_cl, strat_cl)

    return train_idx, val_idx, test_idx


#############################################
def split_data(data, set_idxs):
    """
    split_data(data, set_idxs)

    Returns data (or targets), split into training, validation and test sets.

    Required args:
        - data (nd array)       : array, where the first dimension is the 
                                  samples.
        - set_idxs (nested list): nested list of indices structured as:
                                  set (train, val, test) x indx

    Returns:
        - sets (list of nd arrays): list of numpy arrays containing the data 
                                    for the train, val and test sets 
                                    respectively. If a group is empty, None is 
                                    used instead of an empty array.
    """

    sets = []
    for set_idx in set_idxs:
        if len(set_idx) > 0:
            sets.append(data[set_idx])
        else:
            sets.append(None)
    
    return sets


#############################################
def get_n_wins(leng, win_leng, step_size=1):
    """
    get_n_wins(leng, win_leng)

    Returns the number of windows is the data dimension, based on the 
    specified window length and step_size.

    Required args:
        - leng (int)    : length of the data along the dimension of interest
        - win_leng (int): length of the windows to use
    
    Optional args:
        - step_size (int): step size between each window


    Returns:
        - n_wins (int): number of windows along the dimension of interest
    """

    if leng < win_leng:
        n_wins = 0
    else:
        n_wins = int((leng - win_leng) // step_size) + 1

    return n_wins


##########################################
def get_win_xrans(xran, win_leng, idx, step_size=1):
    """
    get_win_xrans(xran, win_leng, idx)

    Returns x ranges for the specified windows, based on the full x range, 
    window length and specified indices.

    Required args:
        - xran (array-like): Full range of x values
        - win_leng (int)   : length of the windows used
        - idx (list)       : list of indices for which to return x ranges

    Optional args:
        - step_size (int): step size between each window

    Returns:
        - xrans (list): nested list of x values, structured as index x x_vals    
    """

    idx = gen_util.list_if_not(idx)
    n_wins = get_n_wins(len(xran), win_leng, step_size=1)
    xrans = []
    for i in idx:
        win_i = i%n_wins
        xrans.append(xran[win_i : win_i + win_leng])

    return xrans


#############################################
def window_1d(data, win_leng, step_size=1, writeable=False):
    """
    window_1d(data, win_leng)

    Returns original data array with updated stride view to be interpreted
    as a 2D array with the original data split into windows.

    Note: Uses "numpy.lib.stride_tricks.as_strided" function to allow
    windowing without copying the data. May lead to unexpected behaviours
    when using functions on the new array. 
    See: https://docs.scipy.org/doc/numpy/reference/generated/
    numpy.lib.stride_tricks.as_strided.html 

    Required args:
        - data (1D array): array of samples
        - win_leng (int) : length of the windows to extract

    Optional args:
        - step_size (int) : number of samples between window starts 
                            default: 1
        - writeable (bool): if False, the array is unwriteable, to avoid bugs
                            that may occur with strided arrays
                            default: False

    Returns:
        - strided_data (2D array): original data array, with updated stride
                                   view, structured as:
                                       n_win x win_leng
    """

    # bytes to step in each dimension when traversing array
    strides = data.strides[0] 
    # resulting number of windows
    n_wins = get_n_wins(data.shape[0], win_leng, step_size)

    strided_data = np.lib.stride_tricks.as_strided(
        data, shape=[n_wins, int(win_leng)], 
        strides=[strides * step_size, strides], writeable=writeable)

    return strided_data


#############################################
def window_2d(data, win_leng, step_size=1, writeable=False):
    """
    window_2d(data, win_leng)

    Returns original data array with updated stride view to be interpreted
    as a 3D array with the original data split into windows along the first
    dimension.

    Note: Uses "numpy.lib.stride_tricks.as_strided" function to allow
    windowing without copying the data. May lead to unexpected behaviours
    when using functions on the new array. 
    See: https://docs.scipy.org/doc/numpy/reference/generated/
    numpy.lib.stride_tricks.as_strided.html 

    Required args:
        - data (2D array): n_samples x n_items 
                           (windows extracted along sample dimension)
        - win_leng (int) : length of the windows to extract

    Optional args:
        - step_size (int) : number of samples between window starts 
                            default: 1
        - writeable (bool): if False, the array is unwriteable, to avoid bugs
                            that may occur with strided arrays
                            default: False

    Returns:
        - strided_data (3D array): original data array, with updated stride
                                   view, structured as:
                                       n_wins x n_items x win_leng
    """
    
    # bytes to step in each dimension when traversing array
    strides = data.strides
    n_wins  = get_n_wins(data.shape[0], win_leng, step_size)
    n_items = data.shape[1]

    strided_data = np.lib.stride_tricks.as_strided(
        data, shape=[n_wins, n_items, int(win_leng)],
        strides=[strides[0] * step_size, strides[1], strides[0]], 
        writeable=writeable)
    
    return strided_data

