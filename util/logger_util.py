"""
logger_util.py

This module contains logging functions.

Authors: Colleen Gillon

Date: October, 2018

Note: this code uses python 3.7.

"""

import logging
import sys
from pathlib import Path
import warnings

from util import gen_util


#############################################
class TempChangeLogLevel():
    """
    Context manager for temporarily changing logging level.

    Optional init args:
        - logger (logger) : logging Logger object. If None, root logger is used.
                            default: None
        - level (int, str): logging level to temporarily set logger to.
                            If None,log level is not changed.
                            default: "info"
    """

    def __init__(self, logger=None, level="info"):

        if logger is None or logger.level == logging.NOTSET:
            logger = logging.getLogger()
        
        self.logger = logger
        self.level = level


    def __enter__(self):

        if self.level is not None:
            self.prev_level = self.logger.level
            set_level(level=self.level, logger=self.logger)


    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.level is not None:
            set_level(level=self.prev_level, logger=self.logger)


#############################################
#############################################
class BasicLogFormatter(logging.Formatter):
    """
    BasicLogFormatter()

    Basic formatting class that formats different level logs differently. 
    Allows a spacing extra argument to add space at the beginning of the log.
    """

    dbg_fmt  = "%(spacing)s%(levelname)s: %(module)s: %(lineno)d: %(msg)s"
    info_fmt = "%(spacing)s%(msg)s"
    wrn_fmt  = "%(spacing)s%(levelname)s: %(msg)s"
    err_fmt  = "%(spacing)s%(levelname)s: %(module)s: %(lineno)d: %(msg)s"
    crt_fmt  = "%(spacing)s%(levelname)s: %(module)s: %(lineno)d: %(msg)s"

    def __init__(self, fmt="%(spacing)s%(levelname)s: %(msg)s"):
        """
        Optional args:
            - fmt (str): default format style.
        """
        super().__init__(fmt=fmt, datefmt=None, style="%") 

    def format(self, record):

        if not hasattr(record, "spacing"):
            record.spacing = ""

        # Original format as default
        format_orig = self._style._fmt

        # Replace default as needed
        if record.levelno == logging.DEBUG:
            self._style._fmt = BasicLogFormatter.dbg_fmt
        elif record.levelno == logging.INFO:
            self._style._fmt = BasicLogFormatter.info_fmt
        elif record.levelno == logging.WARNING:
            self._style._fmt = BasicLogFormatter.wrn_fmt
        elif record.levelno == logging.ERROR:
            self._style._fmt = BasicLogFormatter.err_fmt
        elif record.levelno == logging.CRITICAL:
            self._style._fmt = BasicLogFormatter.crt_fmt

        # Call the original formatter class to do the grunt work
        formatted_log = logging.Formatter.format(self, record)

        # Restore default format
        self._style._fmt = format_orig

        return formatted_log


#############################################
def set_level(level="info", logger=None, return_only=False):
    """
    set_level()

    Sets level of the named logger.

    Optional args:
        - level (int or str): level of the logger ("info", "error", "warning", 
                              "debug", "critical", 10, 50)
                              default: "info"
        - logger (Logger)   : logging Logger. If None, the root logger is set.
                              default: None
        - return_only (bool): if True, level is not set, but only returned
                              default: False

    Returns:
        - level (int): logging level requested
    """
    
    if logger is None:
        logger = logging.getLogger()

    if isinstance(level, int):
        level = level
    elif level.lower() == "debug":
        level = logging.DEBUG
    elif level.lower() == "info":
        level = logging.INFO
    elif level.lower() == "warning":
        level = logging.WARNING
    elif level.lower() == "error":
        level = logging.ERROR
    elif level.lower() == "critical":
        level = logging.CRITICAL
    else:
        gen_util.accepted_values_error(
            "level", level, 
            ["debug", "info", "warning", "error", "critical"])

    if not return_only:
        logger.setLevel(level)

    return level

#############################################
def level_at_least(level="info", logger=None):
    """
    level_at_least()

    Returns whether level is equal or above specified level.

    Optional args:
        - level (int or str): level of the logger ("info", "error", "warning", 
                              "debug", "critical", 10, 50)
                              default: "info"
        - logger (Logger)   : logging Logger. If None, the root logger is set.
                              default: None

    Returns:
        - at_least (bool): whether current logging level is above specified 
                           level
    """
    

    if logger is None:
        logger = logging.getLogger()

    curr_level = logger.level

    level = set_level(level, logger=logger, return_only=True)

    at_least = (curr_level >= level)

    return at_least


#############################################
def get_logger(logtype="stream", name=None, filename="logs.log", 
               fulldir=".", level="info", fmt=None, skip_exists=True):
    """
    get_logger()

    Returns logger. 

    Optional args:
        - logtype (str)     : type or types of handlers to add to logger 
                              ("stream", "file", "both", "none")
                              default: "stream"
        - name (str)        : logger name. If None, the root logger is returned.
                              default: None
        - filename (str)    : name under which to save file handler, if it is 
                              included
                              default: "logs.log"
        - fulldir (str)     : path under which to save file handler, if it is
                              included
                              default: "."
        - level (str)       : level of the logger ("info", "error", "warning", 
                               "debug", "critical")
                              default: "info"
        - fmt (Formatter)   : logging Formatter to use for the handlers
                              default: None
        - skip_exists (bool): if a logger with the name already has handlers, 
                              does nothing and returns existing logger
                              default: True

    Returns:
        - logger (Logger): logger object
    """

    # create one instance
    logger = logging.getLogger(name)

    # skip if logger already has handlers
    if skip_exists and len(logger.handlers) != 0:
        return logger

    logger.handlers = []
    
    # create handlers
    sh, fh = None, None
    if logtype in ["stream", "both"]:
        sh = logging.StreamHandler(sys.stdout)
        if fmt is not None:
            sh.setFormatter(fmt)
        logger.addHandler(sh)
    if logtype in ["file", "both"]:
        fh = logging.FileHandler(Path(fulldir, filename))
        if fmt is not None:
            fh.setFormatter(fmt)
        logger.addHandler(fh)
    all_types = ["file", "stream", "both", "none"]
    if logtype not in all_types:
        gen_util.accepted_values_error("logtype", logtype, all_types)
    
    set_level(level, name)

    return logger


#############################################
def get_logger_with_basic_format(**logger_kw):
    """
    get_logger_with_basic_format()

    Returns logger with basic formatting, defined by BasicLogFormatter class.

    Keyword args:
        - logger_kw (dict): keyword arguments for get_logger()
        
    Returns:
        - logger (Logger): logger object
    """


    basic_formatter = BasicLogFormatter()

    logger = get_logger(fmt=basic_formatter, **logger_kw)

    return logger


#############################################
def warnings_simple(message, category, filename, lineno, file=None, line=None):
    """
    warnings_simple(message, category, filename, lineno()

    Warning format that doesn't cite the line of code.
    Adapted from: https://pymotw.com/2/warnings/

    Required args: warnings module arguments
        
    Returns:
        - (str): formatting string
    """

    return '%s:%s: %s:\n%s\n' % (filename, lineno, category.__name__, message)



# set logger and warnings format
logger = get_logger_with_basic_format()
warnings.formatwarning = warnings_simple

