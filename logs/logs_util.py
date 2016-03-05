import os
import time


def time_string(precision='minute'):
    """
    Author: Isaac
    returns a string representing the date in the form '12-Jul-2013' etc.
    intended use: handy naming of files.
    """
    t = time.asctime()
    precision_bound = 10  # precision == 'day'
    yrbd = 19
    if precision == 'minute':
        precision_bound = 16
    elif precision == 'second':
        precision_bound = 19
    elif precision == 'year':
        precision_bound = 0
        yrbd = 20
    t = t[4:precision_bound] + t[yrbd:24]
    t = t.replace(' ', '-')
    t = t.replace(':', '_')
    return t
