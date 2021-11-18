"""
file_util.py

This module contains functions for dealing with reading and writing files.

Authors: Colleen Gillon

Date: October, 2018

Note: this code uses python 3.7.

"""

import glob
import json
import logging
import pickle
import warnings
from pathlib import Path

import pandas as pd

from util import gen_util, logger_util

logger = logging.getLogger(__name__)


#############################################
def checkexists(pathname):
    """
    checkexists(pathname)

    Checks whether a path exists and raises an error if not. 
    If path is an empty string, does not raise a warning.

    Required args:
        - pathname (Path): path
    """

    if len(str(pathname)) == 0:
        warnings.warn("'pathname' is an empty string, so path cannot be "
            "evaluated as existing or not.", RuntimeWarning, stacklevel=1)
    elif not Path(pathname).exists():
        raise OSError(f"{pathname} does not exist.")


#############################################
def checkfile(filename):
    """
    checkfile(filename)

    Checks whether a file exists and raises an error if not.

    Required args:
        - filename (Path): name of file
    """

    if not Path(filename).is_file():
        raise OSError(f"{filename} does not exist.")


#############################################
def split_path(full_path):
    """
    split_path(full_path)

    Returns a path fully split.
 
    Required args:
        - full_path (Path): full path
    
    Returns:
        - all_parts (list): list of path parts
    """

    all_parts = list(Path(full_path).parts)
    
    return all_parts


#############################################
def get_up_path(full_path, n_levels=1):
    """
    get_up_path(full_path)

    Returns a path up from the full path (a certain number of levels up).
 
    Required args:
        - full_path (Path): full path
    
    Optional args:
        - n_levels (int): number of path levels to go up
    
    Returns:
        - up_path (Path): path up from full_path
    """

    for path_type in ["relative", "absolute"]:
        use_path = Path(full_path)
        if path_type == "absolute":
            use_path = use_path.resolve()
        path_parts = use_path.parts
        if len(path_parts) > n_levels:
            up_path = Path(*path_parts[: -n_levels])
        elif use_path.is_absolute():
            raise RuntimeError(f"It is not possible to go {n_levels} levels up.")

    return up_path


#############################################
def add_ext(filename, filetype="pickle"):
    """
    add_ext(filename)

    Returns a file name with extension added if there wasn't already an
    extension. Only adds pickle, json or csv extensions.
 
    Required args:
        - filename (Path): name of file, can include the whole directory name
                           and extension
    
    Optional args:
        - filetype (str): type of file (pickle, pkl, json, png, csv, svg, jpg).
                          Overridden if extension already in filename.
                          Can include ""
                          default: "pickle"

    Returns:
        - filename (Path): file name, including extension
        - ext (str)      : extension, including ""
    """

    filename = Path(filename)
    ext = filename.suffix

    filetype = filetype.replace(".", "")

    if ext == "":
        filetypes = ["pkl", "pickle", "json", "csv", "png", "svg", "jpg"]
        file_exts  = [".pkl", ".pkl", ".json", ".csv", ".png", ".svg", ".jpg"]
        if filetype not in filetypes:
            gen_util.accepted_values_error("filetype", filetype, filetypes)
        ext = file_exts[filetypes.index(filetype)]
        filename = filename.with_suffix(ext)

    return filename, ext


#############################################
def loadfile(filename, fulldir=".", filetype="pickle", dtype=None):
    """
    loadfile(filename)

    Returns safely opened and loaded pickle, json or csv. If the file 
    name includes the extension, it will override the filetype argument. 
 
    Required args:
        - filename (Path): name of file, can include the whole directory name
                           and extension
    
    Optional args:
        - fulldir (Path): directory in which file is saed
                          default: "."
        - filetype (str): type of file (pickle, pkl, json, csv)
                          default: "pickle"
        - dtype (str)   : datatype for csv
                          default: None

    Returns:
        - datafile (dict or pd df): loaded file
    """

    filename, ext = add_ext(filename, filetype)
    fullname = Path(fulldir, filename)
    
    if fullname.is_file():
        if ext == ".pkl":
            try:
                with open(fullname, "rb") as f:
                    datafile = pickle.load(f)
            except:
                # load a python 2 pkl in python 3 (works in python 3.6)
                with open(fullname, "rb") as f: 
                    datafile = pickle.load(f, encoding="latin1")
        elif ext == ".json":
            with open(fullname, "rb") as f:
                datafile = json.load(f)
        elif ext == ".csv":
            datafile = pd.read_csv(fullname, dtype=dtype)
        else:
            raise ValueError("'ext' must be in '.pkl', '.json', '.csv'.")
    else:
        raise OSError(f"{fullname} is not an existing file.")

    return datafile


#############################################
def glob_depth(direc, pattern, depth=0):
    """
    glob_depth(direc, pattern)

    Returns all files in a directory containing the specified pattern at the 
    specified depth. 
 
    Required args:
        - direc (Path) : path of the directory in which to search
        - pattern (str): pattern to search for
    
    Optional args:
        - depth (int): depth at which to search for pattern
                       default: 0

    Returns:
        - match_paths (list): list of paths that match the pattern, at the 
                              specified depth
    """

    direc_path = Path(direc, *(["*"] * depth))
    match_paths = glob.glob(f"{direc_path}*{pattern}*")
    match_paths = [Path(match_path) for match_path in match_paths]

    return match_paths


#############################################
def rename_files(direc, pattern, replace="", depth=0, log=True, 
                 dry_run=False):
    """
    rename_files(direc, pattern)

    Renames all files in a directory containing the specified pattern by 
    replacing the pattern. 
 
    Required args:
        - direc (Path) : path of the directory in which to search
        - pattern (str): pattern to replace
    
    Optional args:
        - replace (str) : string with which to replace pattern
                          default: ""
        - depth (int)   : depth at which to search for pattern
                          default: 0
        - log (bool)    : if True, logs old and new names of each renamed file 
                          default: True
        - dry_run (bool): if True, runs a dry run logging old and new names
                          default: False
    """

    change_paths = glob_depth(direc, pattern, depth=depth)

    if len(change_paths) == 0:
        logger.info("No pattern matches found.")
        return

    if dry_run:
        logger.info("DRY RUN ONLY")

    for change_path in change_paths:
        new_path_name = Path(str(change_path).replace(pattern, replace))
        if log or dry_run:
            logger.info(f"{change_path} -> {new_path_name}", 
                extra={"spacing": "\n"})
        if not dry_run:
            change_path.rename(new_path_name)

    return


#############################################
def get_unique_path(savename, fulldir=".", ext=None):
    """
    get_unique_path(savename)

    Returns a unique version of savename by adding numbers if a file by the 
    same name already exists. 

    Required args:
        - savename (Path): name under which to save info, can include the 
                           whole directory name and extension
   
    Optional args:
        - fulldir (Path): directory to append savename to
                          default: "."
        - ext (str)     : extension to use which, if provided, overrides any
                          extension in savename
                          default: None
    
    Returns:
        - fullname (Path): savename with full directory and extension, modified 
                           with a number if needed
    """

    savename = Path(savename)
    if ext is None:
        ext = savename.suffix
        savename = Path(savename.parent, savename.stem)
    elif "." not in ext:
        ext = f".{ext}"

    fullname = Path(fulldir, savename).with_suffix(ext)
    count = 1
    while fullname.exists():
        fullname = Path(fulldir, f"{savename}_{count}").with_suffix(ext)
        count += 1 

    return fullname

#############################################
def saveinfo(saveobj, savename="info", fulldir=".", save_as="pickle", 
             sort=True, overwrite=False):
    """
    saveinfo(saveobj)

    Saves dictionary or csv as a pickle, json or csv, under a specific 
    directory and optional name. If savename includes the extension, it will 
    override the save_as argument.

    Required args:
        - saveobj (dict): object to save
    
    Optional args:
        - fulldir (Path)  : directory in which to save file
                            default: "."
        - savename (str)  : name under which to save info, can include the 
                            whole directory name and extension
                            default: "info"
        - save_as (str)   : type of file to save as (pickle, pkl, json, csv).
                            Overridden if extension included in savename.
                            default: "pickle"
        - sort (bool)     : whether to sort keys alphabetically, if saving a 
                            dictionary as .json
                            default: True
        - overwrite (bool): if False, file name is modified if needed to prevent 
                            overwriting
    """


    # create directory if it doesn't exist
    createdir(fulldir, log_dir=False)
    
    # get extension and savename
    savename, ext = add_ext(savename, save_as) 
    fullname      = Path(fulldir, savename)

    # check if file aready exists, and if so, add number at end
    if not overwrite:
        fullname = get_unique_path(fullname)


    if ext == ".pkl":
        with open(fullname, "wb") as f:
            pickle.dump(saveobj, f)
    elif ext == ".json":
        with open(fullname, "w") as f:
            json.dump(saveobj, f, sort_keys=sort)
    elif ext == ".csv":
        saveobj.to_csv(fullname)
    
    return fullname


#############################################
def checkdir(dirname):
    """
    checkdir(dirname)

    Checks whether the specified directory exists and throws an error if it
    does not.
 
    Required args:
        - dirname (Path): directory path
    """

    # check that the directory exists
    if len(str(dirname)) == 0: # i.e., ""
        return

    dirname = Path(dirname)
    if not dirname.is_dir():
        raise OSError(f"{dirname} either does not exist or is not a "
            "directory.")


#############################################
def createdir(dirname, unique=False, log_dir=True):
    """
    createdir(dirname)

    Creates specified directory if it does not exist, and returns final
    directory name.
 
    Required args:
        - dirname (Path or list): path or hierarchical list of directories, 
                                  e.g. ["dir", "subdir", "subsubdir"]

    Optional args:
        - unique (bool) : if True, ensures that a new directory is created by  
                          adding a suffix, e.g. "_1" if necessary
                          default: False
        - log_dir (bool): if True, the name of the created directory is 
                          logged
                          default: True

    Returns:
        - dirname (Path): name of new directory
    """

    # convert directory list to full path
    dirname = Path(*gen_util.list_if_not(dirname))

    if len(str(dirname)) == 0:
        exists = True
    else:
        exists = dirname.exists()

    if unique and exists:
        i=1
        while Path(f"{dirname}_{i}").exists():
            i += 1
        dirname = Path(f"{dirname}_{i}")
        dirname.mkdir(parents=True)
    elif not exists:
        try:
            dirname.mkdir(parents=True)
        # included due to problems when running parallel scripts
        except FileExistsError:
            pass

    if log_dir:
        logger.info(f"Directory: {dirname}")

    return dirname


#############################################
def getfiles(dirname=".", filetype="all", criteria=None):
    """
    getfiles()

    Returns a list of all files in given directory.

    Optional args:
        - dirname (Path)        : directory
                                  default: "."
        - filetype (str)        : type of file to return: "all", "subdirs" or 
                                  "files"
                                  default: "all"
        - criteria (list or str): criteria for including files, i.e., contains 
                                  specified strings
                                  default: None

    Returns:
        - files (list): list of files in directory
    """

    if len(str(dirname)) == 0:
        dirname = "."

    dirname = Path(dirname)

    allfiles = dirname.iterdir()

    if criteria is not None:
        criteria = gen_util.list_if_not(criteria)
        for cri in criteria:
            allfiles = [x for x in allfiles if cri in str(x)]
    
    allfiles = [Path(dirname, x) for x in allfiles]

    if filetype == "subdirs":
        allfiles = [x for x in allfiles if x.is_dir()]

    elif filetype == "files":
        allfiles = [x for x in allfiles if not x.is_dir()]

    elif filetype != "all":
        gen_util.accepted_values_error(
            "filetype", filetype, ["all", "subdirs", "files"])
    
    return allfiles

    
