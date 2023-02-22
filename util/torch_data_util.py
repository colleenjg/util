"""
data_util.py

This module contains basic pytorch dataset tools.

Authors: Colleen Gillon

Date: October, 2018

Note: this code uses python 3.7.

"""

import numpy as np
import torch
import torch.utils.data

from util import data_util, gen_util, math_util, rand_util



#############################################
class CustomDs(torch.utils.data.TensorDataset):
    """
    The CustomDs object is a TensorDataset object. It takes data and 
    optionally corresponding targets and initializes a custom TensorDataset.
    """

    def __init__(self, data, targets=None):
        """
        self.__init__(data)

        Returns a CustomDs object using the specified data array, and
        optionally corresponding targets.

        Initializes data, targets and n_samples attributes.

        Required args:
            - data (nd array): array of dataset datapoints, where the first
                               dimension is the samples.

        Optional args:
            - targets (nd array): array of targets, where the first dimension
                                  is the samples. Must be of the same length 
                                  as data.
                                  default: None
        """
        self.data = torch.Tensor(data)
        if targets is not None and len(targets.shape) == 1:
            targets = targets.reshape(-1, 1)
        self.targets = torch.Tensor(targets)
        self.n_samples = self.data.shape[0]

        if self.targets is not None and (len(self.data) != len(self.targets)):
            raise ValueError("data and targets must be of the same length.")
    
    def __len__(self):
        """
        self.__len__()

        Returns length of dataset, i.e. number of samples.

        Returns:
            - n_samples (int): length of dataset, i.e., nbr of samples.
        """

        return self.n_samples
    
    def __getitem__(self, index):
        """
        self.__getitem__()

        Returns data point and targets, if not None, corresponding to index
        provided.

        Required args:
            - index (int): index

        Returns:
            - (torch Tensor): data at specified index
            
            if self.targets is not None:
            - (torch Tensor): targets at specified index
        """

        if self.targets is not None:
            return [self.data[index], self.targets[index]]
        else:
            return torch.Tensor(self.data[index])


#############################################
def init_dl(data, targets=None, batchsize=200, shuffle=False):
    """
    init_dl(data)

    Returns a torch DataLoader.

    Required args:
        - data (nd array): array of dataset datapoints, where the first
                           dimension is the samples.

    Optional args:
        - targets (nd array): array of targets, where the first dimension
                              is the samples. Must be of the same length 
                              as data.
                              default: None
        - batchsize (int )  : nbr of samples dataloader will load per batch
                              default: 200
        - shuffle (bool)    : if True, data is reshuffled at each epoch
                              default: False

    Returns:
        - dl (torch DataLoader): torch DataLoader. If data is None, dl is None. 
    """

    if data is None:
        dl = None
    else:
        dl = torch.utils.data.DataLoader(
            CustomDs(data, targets), batch_size=batchsize, shuffle=shuffle)
    return dl


#############################################
def scale_datasets(set_data, sc_dim="all", sc_type="min_max", extrem="reg", 
                   mult=1.0, shift=0.0, sc_facts=None):
    """
    scale_datasets(set_data)

    Returns scaled set_data (sets scaled based on either the factors
    passed or the factors calculated on the first set.) to between 

    Required args:
        - set_data (list): list of datasets (torch Tensors) to scale
    
    Optional args:
        - sc_dim (int)    : data array dimension along which to scale 
                            data ("last", "all")
                            default: "all"
        - sc_type (str)   : type of scaling to use
                            "min_max"  : (data - min)/(max - min)
                            "scale"    : (data - 0.0)/std
                            "stand"    : (data - mean)/std
                            "stand_rob": (data - median)/IQR (75-25)
                            "center"   : (data - mean)/1.0
                            "unit"     : (data - 0.0)/abs(mean)
                            default: "min_max"
        - extrem (str)    : only needed if min_max  or stand_rob scaling is 
                            used. 
                            "reg": the minimum and maximum (min_max) or 
                                   25-75 IQR of the data are used 
                            "perc": the 5th and 95th percentiles are used as 
                                    min and max respectively (robust to 
                                    outliers)
        - mult (num)      : value by which to multiply scaled data
                            default: 1.0
        - shift (num)     : value by which to shift scaled data (applied after
                            mult)
                            default: 0.0
        - sc_facts (list) : list of sub, div, mult and shift values to use on 
                            data (overrides all other optional arguments), 
                            where sub is the value subtracted and div is the 
                            value used as divisor (before applying mult and 
                            shift)
                            default: None


    Returns:
        - set_data (list)            : list of datasets (torch Tensors) to 
                                       scale
        if sc_facts is None, also:
        - sc_facts_list (nested list): list of scaling factors structured as 
                                       stat (mean, std or perc 0.05, perc 0.95) 
                                       (x vals)
                                       default: None
    """

    set_data = gen_util.list_if_not(set_data)

    new = False
    if sc_facts is None:
        new = True
        if sc_dim == "all":
            data_flat = set_data[0].reshape([-1]).numpy()
        elif sc_dim == "last":
            data_flat = set_data[0].reshape([-1, set_data[0].shape[-1]]).numpy()
        else:
            gen_util.accepted_values_error("sc_dim", sc_dim, ["all", "last"])
        sc_facts = math_util.scale_facts(
            data_flat, 0, sc_type=sc_type, extrem=extrem, mult=mult, shift=shift)

    for i in range(len(set_data)):
        sc_data = math_util.scale_data(set_data[i].numpy(), 0, facts=sc_facts)
        set_data[i] = torch.Tensor(sc_data)

    if new: 
        sc_facts_list = []
        for fact in sc_facts:
            if isinstance(fact, np.ndarray):
                fact = fact.tolist()
            sc_facts_list.append(fact)
        return set_data, sc_facts_list

    return set_data


#############################################
def split_data(data, set_idxs):
    """
    split_data(data, set_idxs)

    Returns data (or targets), split into torch Tensor training, validation and 
    test sets.

    Required args:
        - data (nd array)       : array, where the first dimension is the 
                                  samples.
        - set_idxs (nested list): nested list of indices structured as:
                                  set (train, val, test) x indx

    Returns:
        - sets (list of torch Tensors): list of torch Tensors containing the 
                                        data for the train, val 
                                        and test sets respectively.
                                        If a group is empty, None is used
                                        instead of an empty tensor.
    """

    sets = data_util.split_data(data, set_idxs)

    for s in range(len(sets)):
        if sets[s] is not None:
            sets[s] = torch.Tensor(sets[s])
        sets[s]
    
    return sets
    

#############################################
def create_dls(data, targets=None, train_p=0.75, val_p=None, test_p=None, 
               sc_dim="none", sc_type=None, extrem="reg", mult=1.0, shift=0.0, 
               shuffle=False, batchsize=200, thresh_set=5, thresh_cl=2, 
               strat_cl=True, train_shuff=True, randst=None):
    """
    create_dls(data)

    Returns torch DataLoaders for each set (training, validation, test).
    
    If a scaling dimension is passed, each set is scaled based on scaling 
    factors calculated on the training set and the scaling factors are also 
    returned.

    If shuffle is True, targets are shuffled for each dataset and the shuffled 
    indices are also returned.

    Required args:
        - data (nd array): array of dataset datapoints, where the first
                           dimension is the samples.

    Optional args:
        - targets (nd array): array of targets, where the first dimension
                              is the samples. Must be of the same length 
                              as data.
                              default: None
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
        - sc_dim (int)      : data array dimension along which to scale 
                              data ("last", "all")
                              default: "all"
        - sc_type (str)     : type of scaling to use
                              "min_max"  : (data - min)/(max - min)
                              "scale"    : (data - 0.0)/std
                              "stand"    : (data - mean)/std
                              "stand_rob": (data - median)/IQR (75-25)
                              "center"   : (data - mean)/1.0
                              "unit"     : (data - 0.0)/abs(mean)
                              default: "min_max"
        - extrem (str)      : only needed if min_max  or stand_rob scaling is 
                              used. 
                              "reg": the minimum and maximum (min_max) or 
                                     25-75 IQR of the data are used 
                              "perc": the 5th and 95th percentiles are used as 
                                      min and max respectively (robust to 
                                      outliers)
        - mult (num)        : value by which to multiply scaled data
                              default: 1.0
        - shift (num)       : value by which to shift scaled data (applied 
                              after mult)
                              default: 0.0
        - shuffle (bool)    : if True, targets are shuffled in all sets to 
                              create randomized datasets.
                              default: False
        - batchsize (int)   : nbr of samples dataloader will load per batch
                              default: 200
        - thresh_set (int)  : size threshold for sets beneath which an error is
                              thrown if the set's proportion is not 0.
                              default: 5
        - thresh_cl (int)   : size threshold for classes in each non empty set 
                              beneath which the indices are reselected (only if
                              targets are passed). Not checked if thresh_cl is 
                              0.
                              default: 2
        - strat_cl (bool)   : if True, sets are stratified by class. 
                              default: True
        - train_shuff (bool): if True, training data is set to be reshuffled at 
                              each epoch
                              default: True
        - randst (int)      : seed or random state to use when generating 
                              random values.
                              default: None
    Returns:
        - returns (list): 
            - dls (list of torch DataLoaders): list of torch DataLoaders for 
                                               each set. If a set is empty, the 
                                               corresponding dls value is None.
            Optional:
            if shuffle:
            - shuff_reidx (list): list of indices with which targets were
                                  shuffled
            if sc_dim is not None:
            - sc_facts (nested list): list of scaling factors structured as 
                                        stat (mean, std or perc 0.05, perc 0.95) 
                                        (x vals)
    """

    returns = []

    # shuffle targets first
    if targets is not None:
        if shuffle:
            randst = rand_util.get_np_rand_state(randst)
            shuff_reidx = list(range(len(targets)))
            randst.shuffle(shuff_reidx)
            returns.append(shuff_reidx)
            targets = targets[shuff_reidx]
    else:
        set_targets = [None] * 3

    # data: samples x []
    set_idxs = data_util.split_idx(
        n=len(data), train_p=train_p, val_p=val_p, test_p=test_p, 
        thresh_set=thresh_set, targets=targets, thresh_cl=thresh_cl, 
        strat_cl=strat_cl)

    set_data = split_data(data, set_idxs)
    if targets is not None:
        set_targets = split_data(targets, set_idxs)

    if sc_dim not in ["None", "none"]:
        set_data, sc_facts = scale_datasets(
            set_data, sc_dim, sc_type, extrem, mult, shift)
        returns.append(sc_facts)
    
    dls = []
    # if training set, shuffle targets
    for i, (data, targ) in enumerate(zip(set_data, set_targets)):
        if train_shuff and i == 0:
            shuff = True
        else:
            shuff = False
        dls.append(init_dl(data, targ, batchsize, shuff))

    returns = [dls] + returns

    return returns