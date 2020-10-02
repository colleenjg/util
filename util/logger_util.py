"""
logger_util.py

This module contains logging functions.

Authors: Colleen Gillon

Date: October, 2018

Note: this code uses python 3.7.

"""

import logging
import os
import sys

from util import gen_util


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
        super().__init__(fmt=fmt, datefmt=None, style='%') 

    def format(self, record):

        if not hasattr(record, 'spacing'):
            record.spacing = ''

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
def set_level(level='info', name=None):
    """
    set_level()

    Sets level of the named logger.

    Optional args:
        - level (str)       : level of the logger ('info', 'error', 'warning', 
                               'debug', 'critical')
                              default: 'info'
        - name (str)        : logger name. If None, the root logger is set.
                              default: None

    Returns:
        - logger (Logger): logger object
    """
    
    logger = logging.getLogger(name)

    if level.lower() == 'debug':
        level = logging.DEBUG
    elif level.lower() == 'info':
        level = logging.INFO
    elif level.lower() == 'warning':
        level = logging.WARNING
    elif level.lower() == 'error':
        level = logging.ERROR
    elif level.lower() == 'critical':
        level = logging.CRITICAL
    else:
        gen_util.accepted_values_error(
            'level', level, 
            ['debug', 'info', 'warning', 'error', 'critical'])

    logger.setLevel(level)


#############################################
def get_logger(logtype='stream', name=None, filename='logs.log', 
               fulldir='', level='info', fmt=None, set_root=True, 
               skip_exists=True):
    """
    get_logger()

    Returns logger. 

    Optional args:
        - logtype (str)     : type or types of handlers to add to logger 
                              ('stream', 'file', 'both', 'none')
                              default: 'stream'
        - name (str)        : logger name. If None, the root logger is returned.
                              default: None
        - filename (str)    : name under which to save file handler, if it is 
                              included
                              default: 'logs.log'
        - fulldir (str)     : path under which to save file handler, if it is
                              included
                              default: ''
        - level (str)       : level of the logger ('info', 'error', 'warning', 
                               'debug', 'critical')
                              default: 'info'
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
    if logtype in ['stream', 'both']:
        sh = logging.StreamHandler(sys.stdout)
        if fmt is not None:
            sh.setFormatter(fmt)
        logger.addHandler(sh)
    if logtype in ['file', 'both']:
        fh = logging.FileHandler(os.path.join(fulldir, filename))
        if fmt is not None:
            fh.setFormatter(fmt)
        logger.addHandler(fh)
    all_types = ['file', 'stream', 'both', 'none']
    if logtype not in all_types:
        gen_util.accepted_values_error('logtype', logtype, all_types)
    
    set_level(level, name)

    return logger


#############################################
def get_logger_with_basic_format(**logger_kw):
    """
    get_logger_with_basic_format()

    Returns logger with basic formatting, defined by BasicLogFormatter class.

    Kewyord args:
        - logger_kw (dict): keyword arguments for get_logger()
        
    Returns:
        - logger (Logger): logger object
    """


    basic_formatter = BasicLogFormatter()

    logger = get_logger(fmt=basic_formatter, **logger_kw)

    return logger


logger = get_logger_with_basic_format()

