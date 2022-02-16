"""
nn_util.py

This module contains basic pytorch neural network tools.

Authors: Colleen Gillon

Date: March, 2021

Note: this code uses python 3.7.

"""


#############################################
def calculate_conv_output(d_in, ks, ss=1, ps=0, ds=1):
    """
    calculate_conv_output(d_in, ks)

    Returns output dimension for a series of convolutions.

    Required args:
        - d_in (int)      : input dimension 
        - ks (int or list): kernel for each convolution, in order 

    Optional args:
        - ss (int or list): stride for each convolution, in order 
                            default: 1
        - ps (int or list): padding for each convolution, in order 
                            default: 0
        - ds (int or list): dilation for each convolution, in order 
                            default: 1

    Returns:
        - dim (int): output dimension
    """

    vals = [ks, ss, ps, ds]    
    n_vals = [len(val) for val in vals if isinstance(val, list)]
    
    if len(n_vals):
        if len(set(n_vals)) != 1:
            raise ValueError(
                "Must provide same number of values for ks, ss, ps and ds."
                )
        else:
            n_vals = n_vals[0]
    else:
        n_vals = 1

    for v, val in enumerate(vals):
        if not isinstance(val, list):
            vals[v] = [val] * n_vals

    dim = d_in
    for k, s, p, d in zip(*vals):
        dim = int(1 + (dim + 2 * p - d * (k - 1) - 1) / s)
    
    return dim

