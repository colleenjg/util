"""
rand_util.py

This module contains basic random process functions.

Authors: Colleen Gillon

Date: October, 2018

Note: this code uses python 3.7.

"""

import copy
import os
import re
import warnings

import numpy as np
import scipy.stats

from util import gen_util, logger_util, math_util

TAB = "    "

# Set default max array size for permutation tests 
LIM_E6_SIZE = 200
if "LIM_E6_SIZE" in os.environ.keys():
    LIM_E6_SIZE = int(os.environ["LIM_E6_SIZE"])

# Minimum number of examples outside for confidence interval edge calculation
MIN_N = 2


logger = logger_util.get_module_logger(name=__name__)


#############################################
def get_np_rand_state(seed, set_none=False):
    """
    get_np_rand_state(seed)

    Returns a np.RandomState object initialized with the seed provided, or the 
    numpy random module, if seed is None or -1.

    Required args:
        - seed (int): random state seed

    Optional args:
        - set_none (bool): if True, if seed is None or -1, a randomly obtained 
                           random state is returned, instead of the numpy 
                           random module.
                           default: False

    Returns:
        - randst (np.random.RandomState or np.random): random state or module
    """

    if seed in [None, -1]:
        if set_none:
            randst = np.random.RandomState(None)
        else:
            randst = np.random
    else:
        if isinstance(seed, np.random.RandomState):
            randst = seed
        else:
            randst = np.random.RandomState(seed)

    return randst


#############################################
def seed_all(seed=None, device="cpu", log_seed=True, seed_now=True, 
             seed_torch=False):
    """
    seed_all()

    Seeds different random number generators using the seed provided or a
    randomly generated seed if no seed is given.

    Optional args:
        - seed (int or None): seed value to use. (-1 treated as None)
                              default: None
        - device (str)      : if "cuda", torch.cuda, else if "cpu", cuda is not
                              seeded
                              default: "cpu"
        - log_seed (bool)   : if True, seed value is logged
                              default: True
        - seed_now (bool)   : if True, random number generators are seeded now
                              default: True
        - seed_torch (bool) : if True, torch is seeded
                              default: False
    Returns:
        - seed (int): seed value
    """
 
    if seed in [None, -1]:
        MAX_INT32 = 2**32
        seed = np.random.randint(1, MAX_INT32)
        if log_seed:
            logger.info(f"Random seed: {seed}")
    else:
        if log_seed:
            logger.info(f"Preset seed: {seed}")
    
    if seed_now:
        np.random.seed(seed)
        if seed_torch:
            import torch
            torch.manual_seed(seed)
            if device == "cuda":
                torch.cuda.manual_seed_all(seed)
    
    return seed


#############################################
def split_random_state(randst, n=10):
    """
    split_random_state(randst)

    Returns as many new random states as requested, generated from the input 
    random state.

    Required args:
        - randst (np.random.RandomState): random state
    
    Optional args:
        - n (int): number of new random state to generate

    Returns
        - randsts (list): new random states
    """

    MAX_INT32 = 2**32

    randst = get_np_rand_state(randst, set_none=True)

    randsts = []
    for _ in range(n):
        new_seed = randst.randint(MAX_INT32)
        randsts.append(np.random.RandomState(new_seed))

    return randsts


#############################################
def bootstrapped_std(data, n=None, n_samples=1000, proportion=False, 
                     stats="mean", randst=None, choices=None, 
                     return_rand=False, return_choices=False, nanpol=None):
    """
    bootstrapped_std(data)
    
    Returns bootstrapped standard deviation of the statistic across the 
    randomly generated values.

    Required args:
        - data (float or 1D array): proportion or full data for mean
    
    Optional args:
        - n (int)              : number of datapoints in dataset. Required if 
                                 proportion is True.
                                 default: None
        - n_samples (int)      : number of samplings to take for bootstrapping
                                 default: 1000
        - proportion (bool)    : if True, data is a proportion (0-1)
                                 default: False
        - stats (str)          : statistic parameter, i.e. "mean", "median", 
                                 "std" to use for groups
                                 default: "mean"
        - randst (int)         : seed or random state to use when generating 
                                 random values.
                                 default: None
        - choices (1D array)   : int array to index data, if proportion is 
                                 False. If None, random values as selected.
                                 default: None 
        - return_rand (bool)   : if True, random data is returned
                                 default: False
        - return_choices (bool): if True, choices are returned
                                 default: False
        - nanpol (str)         : policy for NaNs, "omit" or None
                                 default: None

    Returns:
        - bootstrapped_std (float): bootstrapped standard deviation of mean or 
                                    percentage
        if return_rand:
        - rand_data (1D array)    : randomly generated means
        if return_choices:
        - choices (2D array)      : randomly generated choices (n x n_samples)
    """

    if randst is None:
        randst = np.random
    elif isinstance(randst, int):
        randst = np.random.RandomState(randst) 

    n_samples = int(n_samples)

    if stats == "median":
        raise NotImplementedError(
            "'median' value for stats is not implemented, as it is a robust "
            "statistic, but standard deviation is not."
            )

    # random values
    if proportion:
        if return_choices or choices is not None:
            raise ValueError(
                "return_choices and choices only apply if proportion is False."
                )
        if data < 0 or data > 1:
            raise ValueError("'data' must lie between 0 and 1 if it is a "
                "proportion.")
        if n is None:
            raise ValueError("Must provide n if data is a proportion.")
        
        # random proportions
        rand_data = math_util.calc_stat(
            randst.rand(n * n_samples).reshape(n, n_samples) < data, 
            stats=stats, axis=0, nanpol=nanpol
            )
        
    else:
        if n is not None and n != len(data):
            raise ValueError("If n is provided, it must be the length of data.")
        n = len(data)

        if choices is not None:
            choices = np.asarray(choices)
            if len(choices) != n:
                raise ValueError("choices should have as many values as data.")
            if choices.min() < 0 or choices.max() >= n:
                raise ValueError(
                    "choices must cannot contain values below 0 or above "
                    "data length - 1."
                    )
            if (choices.astype(int) != choices).any():
                raise ValueError("choices is contain indices for data.")
            choices = choices.astype(int)

        else:
            choices = np.arange(n)
            choices = randst.choice(choices, (n, n_samples), replace=True)
   
        # random statistics
        rand_data = math_util.calc_stat(
            data[choices], stats=stats, nanpol=nanpol, axis=0
            )

    bootstrapped_std = math_util.error_stat(
        rand_data, stats="mean", error="std", nanpol=nanpol
        )
    
    returns = [bootstrapped_std]
    if return_rand:
        returns = returns + [rand_data]
    if return_choices:
        returns = returns + [choices]
    if not (return_rand or return_choices):
        returns = bootstrapped_std

    return returns


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
def check_corr_pairing(corr_type="corr", paired=True):
    """
    check_corr_pairing()

    Checks that correlation type and permutation pairing are appropriately 
    matched.

    Optional args:
        - corr_type (str): type of correlation analysed
                           default: "corr"
        - paired (bool)  : type of permutation pairing
                           default: True
    """
    
    # a few checks for correlations
    corr_types = ["corr", "diff_corr", "R_sqr", "diff_R_sqr"]
    if corr_type not in corr_types:
        gen_util.accepted_values_error("corr_type", corr_type, corr_types)
    if paired == "within":
        if "diff" in corr_type:
            warnings.warn(
                "Regular correlation is recommended if permutation is only "
                "within groups being correlated.", 
                category=UserWarning, stacklevel=1
                )
    else:
        if "diff" not in corr_type:
            raise ValueError(
                "Difference correlation is required if permutation is "
                "between groups being correlated."
                )
                    

#############################################
def run_permute(all_data, n_perms=10000, lim_e6=LIM_E6_SIZE, paired=False, 
                randst=None):
    """
    run_permute(all_data)

    Returns array containing data permuted the number of times requested. Will
    throw an RuntimeError if permuted data array is projected to exceed 
    limit. 

    NOTE: if data is not paired, to save memory, datapoints are permuted 
    together across items, for each permutation.

    Required args:
        - all_data (2D array): full data on which to run permutation
                               if paired: groups x datapoints to permute (2)
                               else: items x datapoints to permute (all groups)

    Optional args:
        - n_perms  (int): nbr of permutations to run
                          default: 10000
        - lim_e6 (num)  : limit (when multiplied by 1e6) to permuted data array 
                          size at which an AssertionError is thrown. "none" for
                          no limit.
                          Default value be set externally by setting 
                          LIM_E6_SIZE as an environment variable.
                          default: LIM_E6_SIZE
        - paired (bool) : if True, all_data is paired
                          if "within", pairs are shuffled, instead of 
                          datapoints
                          default: False
        - randst (int)  : seed or random state for random processes
                          default: None

    Returns:
        - permed_data (3D array): array of multiple permutation of the data, 
                                  structured as: 
                                  items x datapoints x permutations
    """

    randst = get_np_rand_state(randst)

    if len(all_data.shape) > 2:
        raise NotImplementedError("Permutation analysis only implemented for "
            "2D data.")

    if paired and all_data.shape[1] != 2:
        raise ValueError(
            "If paired is True, second dimension of all_data should be of "
            "length 2."
            )

    # checks final size of permutation array and throws an error if
    # it is bigger than accepted limit.
    perm_size = np.product(all_data.shape) * n_perms
    if lim_e6 != "none":
        lim = int(lim_e6 * 1e6)
        fold = int(np.ceil(float(perm_size)/lim))
        permute_cri = (f"Permutation array is up to {fold}x allowed size "
            f"({lim_e6} * 10^6).")
        if perm_size > lim:
            raise RuntimeError(permute_cri)

    # (sample with n_perms in first dimension, so results are consistent 
    # within permutations, for different values of n_perms)
    if paired:
        if paired == "within":
            all_data = all_data.T # shuffle pairings instead of datapoints
        rand_shape = (n_perms, ) + all_data.shape    
        transpose = (1, 2, 0)
        sort_axis = 1
    else:
        rand_shape = (n_perms, all_data.shape[1]) # permute columns together
        transpose = (1, 0)
        sort_axis = 0

    # (item x datapoints (all groups))
    perm_idxs = np.argsort(
        np.transpose(randst.rand(*rand_shape), transpose), axis=sort_axis
        )
    if not paired:
        perm_idxs = perm_idxs[np.newaxis, :, :] # add row dimension
        sort_axis = 1

    dim_data = np.arange(all_data.shape[0])[:, np.newaxis, np.newaxis]

    # generate permutated array
    permed_data = np.stack(all_data[dim_data, perm_idxs])

    if paired == "within": # transpose back
        permed_data = np.transpose(permed_data, (1, 0, 2))

    return permed_data


#############################################
def permute_diff_ratio(all_data, div="half", n_perms=10000, stats="mean", 
                       nanpol=None, op="diff", paired=False, randst=None):       
    """
    permute_diff_ratio(all_data)

    Returns all group mean/medians or differences/ratios between two groups 
    resulting from the permutation analysis on input data.

    Required args:
        - all_data (2D array): full data on which to run permutation
                               if paired: groups x datapoints to permute (2)
                               else: items x datapoints to permute (all groups)

    Optional args:
        - div (str or int)  : nbr of datapoints in first group 
                              (ignored if data is paired)
                              default: "half"
        - n_perms (int)     : nbr of permutations to run
                              default: 10000
        - stats (str)       : statistic parameter, i.e. "mean", "median", "std" 
                              to use for groups
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
                              "corr": pearson correlation 
                                      (only possible if paired is True)
                              "diff_corr": pearson correlation between 
                                           grp1 and grp2 - grp1 
                                           (only possible if paired is True)
                              "R_sqr": R squared or percent explained variance
                              "diff_R_sqr": R squared or percent explained 
                                            variance from diff_corr
                              "none"
                              default: "diff"
        - paired (bool)     : if True, all_data is paired
                              if "within", pairs are shuffled, instead of 
                              datapoints
                              default: False
        - randst (int)      : seed or random state for random processes
                              default: None

    Returns:
        - all_rand_vals (2 or 3D array): permutation results, structured as:
                                             (grps if op is "none" x) 
                                             items x perms
    """

    if len(all_data.shape) != 2:
        raise NotImplementedError("Significant difference/ratio analysis only "
            "implemented for 2D data.")

    corr_types = ["corr", "diff_corr", "R_sqr", "diff_R_sqr"]
    if op in corr_types:
        check_corr_pairing(corr_type=op, paired=paired)

    all_rand_res = []
    perm = True
    n_perms_tot = n_perms
    perms_done = 0

    randst = get_np_rand_state(randst)

    if div == "half":
        div = int(all_data.shape[1] // 2)
    
    if op in corr_types and not paired:
        if div != all_data.shape[1] / 2:
            raise ValueError(
                "For correlation operations, even if permutation analysis is "
                "not paired, the data must be split exactly in 2."
                )

    while perm:
        try:
            perms_rem = n_perms_tot - perms_done
            if perms_rem < n_perms:
                n_perms = perms_rem
            permed_data = run_permute(
                all_data, n_perms=int(n_perms), paired=paired, randst=randst
                )
            
            if op not in ["d-prime"] + corr_types:
                axis = None
                if paired:
                    rand = math_util.calc_stat(
                        permed_data, stats=stats, axis=0, nanpol=nanpol
                        )
                    
                else:
                    rand = np.stack([
                        math_util.calc_stat(
                            permed_data[:, 0:div], stats, axis=1, nanpol=nanpol
                        ), 
                        math_util.calc_stat(
                            permed_data[:, div:], stats, axis=1, nanpol=nanpol
                        )
                        ])
            else:
                axis = 1
                if paired:                    
                    # dummy item dimension, then 
                    # transpose to (2, 1, datapoints, perms)
                    rand = np.transpose(permed_data[np.newaxis], (2, 0, 1, 3))
                else:
                    rand = [permed_data[:, :div], permed_data[:, div:]]

            if op.lower() == "none":
                rand_res = rand
            else:
                # calculate grp2-grp1 or grp2/grp1... -> elem x perms
                rand_res = math_util.calc_op(
                    rand, op, dim=0, nanpol=nanpol, axis=axis
                    )

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
def get_p_val_from_rand(act_data, rand_data, return_CIs=False, p_thresh=0.05, 
                        tails=2, multcomp=None, nanpol=None):
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
        - nanpol (str)      : policy for NaNs, "omit" or None when taking 
                              statistics
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
    
    if np.isnan(act_data):
        p_val = np.NaN
        if return_CIs:
            null_CI = [np.nan, np.nan, np.nan]
            return p_val, null_CI
        else:
            return p_val
    
    nans = np.isnan(sorted_rand_data)
    if nans.sum():
        if nanpol == "omit": # remove NaNs
            sorted_rand_data = sorted_rand_data[~nans]
        else:
            raise ValueError(
                "If 'nanpol' is None, sorted_rand_data should not include "
                "NaN values, unless act_data is NaN.")

    perc = scipy.stats.percentileofscore(
        sorted_rand_data, act_data, kind="mean"
        )
    if str(tails) in ["hi", "2"] and perc > 50:
        perc = 100 - perc
    p_val = perc / 100

    if return_CIs:
        multcomp = 1 if not multcomp else multcomp
        corr_p_thresh = p_thresh / multcomp
        check_n_rand(len(sorted_rand_data), p_val=corr_p_thresh)

        percs = math_util.get_percentiles(
            CI=(1 - corr_p_thresh), tails=tails
            )[0]
        percs = [percs[0], 50, percs[1]]
        null_CI = [np.percentile(sorted_rand_data, p, axis=-1) for p in percs]
        
        return p_val, null_CI
    
    else:
        return p_val


#############################################
def get_op_p_val(act_data, n_perms=10000, stats="mean", op="diff", 
                 return_CIs=False, p_thresh=0.05, tails=2, multcomp=None, 
                 paired=False, return_rand=False, nanpol=None, randst=None):
    """
    get_op_p_val(act_data)

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
                              (see math_util.calc_op())
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
                              if "within", pairs are shuffled, instead of 
                              datapoints
                              default: False
        - return_rand (bool): if True, random data is returned
                              default: False
        - nanpol (str)      : policy for NaNs, "omit" or None when taking 
                              statistics
                              default: None
        - randst (int)      : seed or random state for random processes
                              default: None

    Returns:
        - p_val (float) : p-value calculated from a randomly generated 
                          null distribution (not corrected)
        if return_CIs:
        - null_CI (list): null confidence interval (low, median, high), 
                          adjusted for multiple comparisons
        if return_rand:
        - rand_vals (1D array): randomly generated data
    """

    if len(act_data) != 2:
        raise ValueError("Expected 'act_data' to comprise 2 groups.")

    grp1, grp2 = [np.asarray(grp_data) for grp_data in act_data]

    if len(grp1.shape) != 1 or len(grp2.shape) != 1:
        raise ValueError("Expected act_data groups to be 1-dimensional.")

    data = [grp1, grp2]
    corr_types = ["corr", "diff_corr", "R_sqr", "diff_R_sqr"]
    if op not in ["d-prime"] + corr_types:
        data = [
            math_util.calc_stat(grp, stats=stats, nanpol=nanpol)
            for grp in data
            ]
    real_val = math_util.calc_op(data, op=op, nanpol=nanpol)

    if paired:
        if len(grp1) != len(grp2):
            raise ValueError(
                "If data is paired, groups must have the same length."
                )
        concat = np.vstack([grp1, grp2]).T # groups x datapoints (2)
        div = None
    else:
        if op in corr_types and (len(grp1) != len(grp2)):
            raise ValueError(
                "If operation involves correlation, groups must have the "
                "same length."
                )
        concat = np.concatenate([grp1, grp2], axis=0).reshape(1, -1) # add items dim
        div = len(grp1)

    rand_vals = permute_diff_ratio(
        concat, div=div, n_perms=n_perms, stats=stats, op=op, paired=paired, 
        nanpol=nanpol, randst=randst
        ).squeeze()

    if len(rand_vals.shape) == 0:
        rand_vals = rand_vals.reshape(1)

    returns = get_p_val_from_rand(
        real_val, rand_vals, return_CIs=return_CIs, p_thresh=p_thresh, 
        tails=tails, multcomp=multcomp
        )
    
    if return_rand:
        if return_CIs:
            p_val, null_CI = returns
            returns = [p_val, null_CI, rand_vals]
        else:
            p_val = returns
            returns = [p_val, rand_vals]
    
    return returns


#############################################
def comp_vals_acr_groups(act_data, n_perms=None, normal=True, stats="mean", 
                         paired=False, nanpol=None, randst=None):
    """
    comp_vals_acr_groups(act_data)

    Returns p-values for comparisons across groups.

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
                         if "within", pairs are shuffled, instead of 
                         datapoints
                         default: False
        - nanpol (str) : policy for NaNs, "omit" or None when taking statistics
                         default: None
        - randst (int) : seed or random state for random processes
                         default: None

    Returns:
        - p_vals (1D array): p values (not adjusted for tails) for each 
                             comparison, organized by group pairs (where the 
                             second group is cycled in the inner loop, 
                             e.g., 0-1, 0-2, 1-2, including None groups)
    """

    n_comp = sum(range(len(act_data)))
    p_vals = np.full(n_comp, np.nan)
    i = 0
    for g, g_data in enumerate(act_data):
        for g_data_2 in act_data[g + 1:]:
            if g_data is not None and g_data_2 is not None and \
                len(g_data) != 0 and len(g_data_2) != 0:
                
                if n_perms is not None:
                    p_vals[i] = get_op_p_val(
                        [g_data, g_data_2], n_perms, stats=stats, op="diff", 
                        paired=paired, nanpol=nanpol, randst=randst)
                elif normal:
                    fct = scipy.stats.ttest_rel if paired else scipy.stats.ttest_ind
                    # reverse 2-tail adjustment
                    p_vals[i] = fct(g_data, g_data_2, axis=None)[1] / 2
                else:
                    fct = scipy.stats.wilcoxon if paired else scipy.stats.mannwhitneyu 
                    # reverse 2-tail adjustment
                    p_vals[i] = fct(g_data, g_data_2)[1] / 2
            i += 1
    
    return p_vals


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
def id_elem(rand_vals, act_vals, tails=2, p_val=0.05, min_n=MIN_N, 
            log_elems=False, ret_th=False, ret_pval=False, nanpol=None):
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
                              default: MIN_N
        - log_elems (bool)  : if True, the indices of significant elements and
                              their actual values are logged
                              default: False
        - ret_th (bool)     : if True, thresholds are returned for each element
                              default: False
        - ret_pval (bool)   : if True, p-values are returned for each element
                              default: False
        - nanpol (str)      : policy for NaNs, "omit" or None when taking 
                              statistics
                              default: None

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
        raise NotImplementedError("NaNs encountered in actual values.")
    if nan_rand_vals > 0:
        raise NotImplementedError("NaNs encountered in random values.")

    check_n_rand(rand_vals.shape[-1], p_val, min_n=min_n)

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
        act_percs = [scipy.stats.percentileofscore(sub_rand_vals, val) 
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

