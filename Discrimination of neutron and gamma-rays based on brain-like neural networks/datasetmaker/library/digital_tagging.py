#!/usr/bin/env python3
#http://patorjk.com/software/taag/#p=display&f=ANSI%20Shadow
# --------------------------------------------------------------
# ██████╗ ██╗ ██████╗ ██╗████████╗ █████╗ ██╗              
# ██╔══██╗██║██╔════╝ ██║╚══██╔══╝██╔══██╗██║              
# ██║  ██║██║██║  ███╗██║   ██║   ███████║██║              
# ██║  ██║██║██║   ██║██║   ██║   ██╔══██║██║              
# ██████╔╝██║╚██████╔╝██║   ██║   ██║  ██║███████╗         
# ╚═════╝ ╚═╝ ╚═════╝ ╚═╝   ╚═╝   ╚═╝  ╚═╝╚══════╝         
                                                         
# ████████╗ █████╗  ██████╗  ██████╗ ██╗███╗   ██╗ ██████╗ 
# ╚══██╔══╝██╔══██╗██╔════╝ ██╔════╝ ██║████╗  ██║██╔════╝ 
#    ██║   ███████║██║  ███╗██║  ███╗██║██╔██╗ ██║██║  ███╗
#    ██║   ██╔══██║██║   ██║██║   ██║██║██║╚██╗██║██║   ██║
#    ██║   ██║  ██║╚██████╔╝╚██████╔╝██║██║ ╚████║╚██████╔╝
#    ╚═╝   ╚═╝  ╚═╝ ╚═════╝  ╚═════╝ ╚═╝╚═╝  ╚═══╝ ╚═════╝ 
# --------------------------------------------------------------                                                         
#  A library of methods for digitiser based neutron tagging data
# --------------------------------------------------------------                                                         
                                                                                                             
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import nicholai_utility as nu
from math import exp
# from uncertainties import ufloat
# from uncertainties.umath import * #Get all methods for library
from math import isnan
from tqdm import tqdm #Library for progress bars

"""
OUTDATED LIBRARY!

Library containing methods for neutron tagging analysis based on digitizer data from JADAQ.
"""

def PSchargeFOM_optimization(min=0, max=250):
    """
    Method itterates through several combinations of the a and b parameters and finds the maximum FoM value.
    Returns container with FOM values and related a and b values. 
    Return format: FOM = [[FOM],[a],[b]]

    Arguments:
    - min.........Minum value for a and b to start from.
    - max.........Maximum value for a and b to stop at.

    -----------------------------------------------
    Nicholai Mauritzson
    2018-06-18
    """
    FOM = [[],[],[]] #Container for FOM, a and b values.
    for i in tqdm(np.arange(min, max, 5)):
        for j in np.arange(min, max, 5):
            FOM[0].append(PSDchargeFOM(df1.query('ch==1'), LG='QDC_LG', SG='QDC_SG', a=i, b=j))
            FOM[1].append(i)
            FOM[2].append(j)
    print(f'Maximum FOM = {max(FOM[0])}')
    print(f'Constants: a = {FOM[1][np.argmax(FOM[0])]}, b = {FOM[2][np.argmax(FOM[0])]}')
    return FOM

def baselineCalc(df, col_samples='samples', sampleLength=20):
    """
    Baseline calculator using the first 20 sample points as default.
    Returns event array containing one sample array with baseline subtracted.

    Arguments: 
    - df..............The data frame containing the data.
    - col_samples.....The column name for the sampling points. Default: 'samples'.
    - sampleLength....The maximum number of samples points to include in calculation, starting from zero. Default: 20.

    ---------------------------------------
    Nicholai Mauritzson
    2019-06-20
    """
    print('baselineCalc(): Calculating coarse baseline values...')
    return df[f'{col_samples}'].apply(lambda x: np.int16(x) - np.int16(np.round(np.average(x[0:20]))))

def baselineFineCalc(df, col_samples='samples', sampleLength=20):
    """
    NOTE: use this method AFTER baselineCalc(), to get fine resolution value for the remaining baseline to subtract.
    Returns a column with fine baseline settings.
    
    Arguments:
    - df..............The data frame containing the data.
    - col_samples.....The column name for the sampling points. Default: 'samples'.
    - sampleLength....The maximum number of samples points to include in calculation, starting from zero. Default: 20.

    ----------------------------------------
    Nicholai Mauritzson
    2019-06-20
    """
    print('baselineFineCalc(): Calculating fine baseline values...')
    return df[f'{col_samples}'].apply(lambda x: np.int16(0.5 + 1000 * np.average(x[0:sampleLength])))

def pulse_integration(df, col_samples='samples', col_baselineFine='baselineFine', gateStart=0, gateWindow=100):
    """
    Method uses a trapezoidal integration technique for samples pulses. Returns baseline subtracted integration value
    of a given gate size.
    
    Arguments:
    - col_samples.........Name of the column containg the samples.
    - col_baselineFine....Name of the column containing the baselinesubtracted samples.
    - gateStart...........Starting point of the integration window, given in sample number.
    - gateWindow..........The lenght of the integration window. Number of samples to integrate over, starting from 'gateStart'.

    ------------------------------------------
    Nicholai Mauritzson
    2019-07-23
    """
    print('QDC(): Calculating QDC values...')
    gateEnd = gateStart+gateWindow
    return np.abs(df[f'{col_samples}'].apply(lambda x: 1000*np.trapz(x[gateStart:gateEnd])) - (gateWindow*df[f'{col_baselineFine}']))/1000

def PSDcharge(df, col_LG='qdc_det0', col_SG='qdc_sg_det0', a=0, b=0):
    """
    Charged-based PSD calculator.
    Returns columns of calulated PSD values. Each row is an individual event.

    - df.........The data frame containing the data.
    - LG.........Name of column which contains the long gate QDC values.
    - SG.........Name of column which contains the short gate QDC values.
    - a..........Linearisation parameter effecting the short gate value.
    - b..........Linearisation parameter effecting the long gate value.

    Rasmus derived a=287 and b=120 as the best separations values for his thesis.
    ------------------------------------------------
    Nicholai Mauritzson
    2019-06-18
    """
    return 1 - ((df[f'{col_SG}'] + a) / (df[f'{col_LG}'] + b))

def PSDchargeFOM(df, LG='qdc_det0', SG='qdc_lg_det0', a=0, b=0):
    """
    Method calculates FOM for charged-based PSD.
    
    Arguments:
    - df.........The data frame containing the data.
    - LG.........Name of column which contains the long gate QDC values.
    - SG.........Name of column which contains the short gate QDC values.
    - a..........Linearisation parameter effecting the short gate value.
    - b..........Linearisation parameter effecting the long gate value.
    
    ----------------------------------------
    Nicholai Mauritzson
    2019-07-23
    """
    from scipy.optimize import curve_fit

    #Calculate the PS for the neutrons and make histogram.
    y_n, x_n = np.histogram(PSDcharge(df, LG, SG, a, b), range=(0.04, 0.23), bins=int(round((0.23-0.04)*100))) 
    #Calculate the PS for the gammas and make histogram.
    y_g, x_g = np.histogram(PSDcharge(df, LG, SG, a, b), range=(0.25, 0.40), bins=int(round((0.40-0.25)*100)))
    x_n = nu.getBinCenters(x_n)
    x_g = nu.getBinCenters(x_g)
    mean_n = sum(x_n * y_n) / sum(y_n)
    mean_g = sum(x_g * y_g) / sum(y_g)
    std_n = np.sqrt(sum(y_n * (x_n - mean_n)**2) / sum(y_n))
    std_g = np.sqrt(sum(y_g * (x_g - mean_g)**2) / sum(y_g))

    y, x = np.histogram(PSDcharge(df, LG, SG, a, b), bins=((1-(-1))*100)) #Calculate the PS and make histogram.

    #Fitting the neutron PS peak.
    popt_n, pcov_n = curve_fit(nm.gaussFunc, x_n, y_n, p0 = [max(y_n), mean_n, std_n])
    C_n = popt_n[1]
    W_n = popt_n[2]*2.355
    #Fitting the gamma PS peak.
    popt_g, pcov_g = curve_fit(nm.gaussFunc, x_g, y_g, p0 = [max(y_g), mean_g, std_g])
    C_g = popt_g[1]
    W_g = popt_g[2]*2.355

    FOM = (C_n-C_g)/(W_n+W_g)

    return abs(FOM)

def gateChecker(df, gateStart=0, gateWindow=200):
    """
    Simple method for checking where the gate falls relative the sample pulses.
    Plots 500 random sample pulses and overlayes the given gate lenght.

    Arguments:
    - df............The data frame containing the data.
    - gateStart.....The starting point of the gate given in ns.
    - gateWindow....The total lenght of the gate window given in ns. Default: 200.
    ------------------------------
    Nicholai Mauritzson
    2019-06-18

    """
    from random import randint
    for i in range(500):
        plt.plot(df.samples[randint(0, len(df))])
    plt.plot(np.ones(2)*gateStart, range(1024,524,-250), linewidth=3, color='black', label=f'Gate start/stop: {gateStart}/{gateStart+gateWindow} ns')
    plt.plot(np.ones(2)*(gateStart+gateWindow), range(1024,524,-250), linewidth=3, color='black')
    # plt.xlim(gateStart, gateStart+gateWindow)
    plt.ylim((0,1150))
    plt.title(f'500 random pulses. Gate width={gateWindow} ns')
    plt.xlabel('Sample point [ns]')
    plt.ylabel('ADC value')
    plt.show()