# Basic utilities

## 1. Description
This package contains basic utility code used for to perform basic computations needed across projects.

## 2. Installation
The code itself can be obtained by cloning the [github repo](https://github.com/colleenjg/util), and installed by navigating to the main directory and running `pip install .`

All requirements will be automatically installed, except:
* `numexpr`: needed only for `gen_util.n_cores_numba()`.
* `torch`: needed only for the `torch_data_util` module, and certain `logreg_util` functions.
* `joblib`: needed for parallelization functions in the `gen_util` module.
* `dandi`: needed for the `dandi_download_util` module.

The code is written in Python 3.

## 3. Author
This code was written by: Colleen Gillon.
