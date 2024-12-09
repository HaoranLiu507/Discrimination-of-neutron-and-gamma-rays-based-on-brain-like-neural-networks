#!/usr/bin/env python3
# #http://patorjk.com/software/taag/#p=display&f=ANSI%20Shadow
# ------------------------------------------------------------------------------
#                 ███████╗██╗ ██████╗ ███╗   ██╗ █████╗ ██╗                    
#                 ██╔════╝██║██╔════╝ ████╗  ██║██╔══██╗██║                    
#                 ███████╗██║██║  ███╗██╔██╗ ██║███████║██║                    
#                 ╚════██║██║██║   ██║██║╚██╗██║██╔══██║██║                    
#                 ███████║██║╚██████╔╝██║ ╚████║██║  ██║███████╗               
#                 ╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝               
#  ██████╗ ███████╗███╗   ██╗███████╗██████╗  █████╗ ████████╗ ██████╗ ██████╗ 
# ██╔════╝ ██╔════╝████╗  ██║██╔════╝██╔══██╗██╔══██╗╚══██╔══╝██╔═══██╗██╔══██╗
# ██║  ███╗█████╗  ██╔██╗ ██║█████╗  ██████╔╝███████║   ██║   ██║   ██║██████╔╝
# ██║   ██║██╔══╝  ██║╚██╗██║██╔══╝  ██╔══██╗██╔══██║   ██║   ██║   ██║██╔══██╗
# ╚██████╔╝███████╗██║ ╚████║███████╗██║  ██║██║  ██║   ██║   ╚██████╔╝██║  ██║
#  ╚═════╝ ╚══════╝╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝
#         A library of methods for signal generation and testing
#               
#           Author: Nicholai Mauritzson 2019
#                   nicholai.mauritzson@nuclear.lu.se
# ------------------------------------------------------------------------------
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dask.dataframe as dd
import dask.array as da
from math import cos, pi, radians, log
from scipy import signal
from scipy.stats import chisquare
from tqdm import tqdm
from scipy.optimize import curve_fit
from matplotlib.colors import LogNorm
from dask.diagnostics import ProgressBar
# import helpers as hlp # helper routines

# IMPORT FROM LOCAL LIBRARY
sys.path.insert(0, "../library/") #Import my own libraries
import processing_math as promath
import processing_utility as prout
import processing_pd as propd
import processing_data as prodata
import nicholai_exclusive as ne

# IMPORT FROM MAIN DIRECTORY
sys.path.insert(0, "../")
import processing

def SignalGenerator(time, ph, mu, std, baseline, polarity='-', noise=False, noise_multi=1, cfd_true=False, cfd_frac=0.35, time_offset=0):
    """
    Function generates a data frame to be used for testing analysis code.
    titime_offset..an array or list with the time values.
    ph.............the maximum pulseheight of the pulse.
    mu.............the position of the maximum pulseheight of the peak.
    baseline.......the baseline value. This is a constant which is added on to the amplitude values.
    polarity.......the polarity of the pulse to generate.
    noise..........boolean to determine if noise should be included in the pulse. Default: False
    noise_multi....a multiplicaiton factor for increasing or suppresing the generated noise. Default: 1
    cfd_true.......boolean to determine if an analytical calculation of the CFD position should be included i the data frame. Default: False
    cfd_frac.......the fraction of peak amplitude at which the CFD crossing point is determined.
    offset.........an offset in time which to give the "time" varaible. Useful for studing "walk effect" with CFD. Default: 0
    """

    #TODO: Add uncertainty in start of signal as seen in the real pulses. Ca +-7 ns should be enough.

    #Offset the time with a constant
    if time_offset != 0:
        time = time + time_offset

    #Generate a data frame
    df = pd.DataFrame()

    #Generate a pulse
    df['samples'] = [pulse_gen(time, ph, mu, std, baseline, polarity)]
    
    #Calculate the pulse baseline
    df[f'baseline'] = np.mean(df[f'samples'][0][:20])

    #Find peak index of pulse
    if polarity == '-':
        df['peakidx'] = np.argmin(df['samples'][0])
    else:
        df['peakidx'] = np.argmax(df['samples'][0])

    #Get pulse amplitude and correct for baseline
    if polarity == '-':
        df[f'amplitude'] = np.min(df[f'samples'][0])
    else:
        df[f'amplitude'] = np.max(df[f'samples'][0])
    
    df[f'amplitude'] = np.abs(df[f'amplitude'] - df[f'baseline'])

    #Calculate analytical CFD position (round to nearest 1/1000 of x-unit) (no noise)
    if cfd_true:
        df['cfd_true'] = int(round((mu-np.sqrt(-(log(cfd_frac)*2*std*std)))*1000))-time_offset*1000
    
    #:::::::::::::::::::::::::::::
    #       GENERATE NOISE
    #:::::::::::::::::::::::::::::
    
    if noise:
        #Generate noise to pulse
        df['samples_noise'] = [noise_gen(df['samples'][0].copy(), noise_multi)]

        #Calculate the pulse baseline with noise
        df[f'baseline_noise'] = np.mean(df[f'samples_noise'][0][:20])

        #Find peak index of pulse with noise
        if polarity == '-':
            df['peakidx_noise'] = np.argmin(df['samples_noise'][0])
        else:
            df['peakidx_noise'] = np.argmax(df['samples_noise'][0])

        #Get pulse amplitude and correct for baseline with noise
        if polarity == '-':
            df[f'amplitude_noise'] = np.min(df[f'samples_noise'][0])
        else:
            df[f'amplitude_noise'] = np.max(df[f'samples_noise'][0])
        
        df[f'amplitude_noise'] = np.abs(df[f'amplitude_noise'] - df[f'baseline_noise'])

    return df

def pulse_gen(x, ph, mu, std, baseline, polarity='-'):
    """
    Function to generate a pulse with some tail. 
    Uses two Gaussian and varies the standard deviation to create a "tail".

    x...........an array or list with the time values.
    ph..........the maximum pulseheight of the pulse.
    mu..........the position of the maximum pulseheight of the peak.
    baseline....the baseline value. This is a constant which is added on to the amplitude values.
    """
    
    std_offset = 4
    samples = np.arange(0)
    for xi in x:
        if polarity=='-':
            if xi > mu: #After the mean value is reached create a tail.
                samples = np.append(samples, -round(promath.gaussFunc(xi, ph, mu, std*std_offset)+baseline))
            else:# Before the mean value is reached use normal std to create rise.
                samples = np.append(samples, -round(promath.gaussFunc(xi, ph, mu, std)+baseline)) 
        else: 
            if xi > mu: #After the mean value is reached create a tail.
                samples = np.append(samples, -round(promath.gaussFunc(xi, ph, mu, std*std_offset)-baseline))
            else:# Before the mean value is reached use normal std to create rise.
                samples = np.append(samples, -round(promath.gaussFunc(xi, ph, mu, std)-baseline)) 
    return samples

def noise_gen(signal, amplitude_multi=1):
    """
    Function to generate and add noise to signal. Based on statistical diviation.

    signal............an array or list of samples values constituting the signal.
    amplitude_multi...a multiplication factor for the noise amplitude. Use to increase or suppress the generated noise. Default: 1

    Function will return signal with a random noise component added to each samples point. 
    This is based on the statistical diviation, sqrt(N)
    """
    import random

    for i in range(len(signal)):
        error = np.sqrt(np.abs(signal[i]))*amplitude_multi
        signal[i] += random.uniform(-error,error)
    return signal

def cfd_testing(time, ph, mu, std, baseline, polarity, offset_unit, noise=False, noise_multi=1):
    """
    Function for testing the resolution of the CFD algorithm in "processing.py"
    Itterates through several offset values and returns the CFD value for each.
    with a zero multiplicity. 

    time.........an array or list with the time values.
    ph...........the maximum pulseheight of the pulse.
    mu...........the position of the maximum pulseheight of the peak.
    baseline.....the baseline value. This is a constant which is added on to the amplitude values.
    polarity.....the polarity of the pulse to generate.    
    offset_unit..the stepping size of the offset. Effectively the resolution.
    noise........boolean to determine if noise should be included in the pulse. Default: False
    noise_multi....a multiplicaiton factor for increasing or suppresing the generated noise. Default: 1
    """
    cfd_recon = np.arange(0)
    cfd_true = np.arange(0)
    offset = np.arange(0)
    for i in np.arange(0, 1, offset_unit):
        df = SignalGenerator(time, ph, mu, std, baseline, '-', noise, noise_multi, cfd_true=True, cfd_frac=0.35, time_offset=i)
        cfd_recon = np.append(cfd_recon, processing.cfd(df['samples'][0], df['baseline'][0], df['peakidx'][0], 0.35, polarity='-', backwards=True))
        offset = np.append(offset, i)
        cfd_true = np.append(cfd_true, df['cfd_true'][0])
        # plt.plot(df['samples'][0])
    # plt.xlabel('Time [ns]')
    # plt.ylabel('Amplitude [mV]')
    # plt.grid(which='both')
    # plt.title('Generated signal samples, offset res.: 22 ps')
    # plt.show()
    return cfd_recon, cfd_true, offset

def cfd_testing_variable(time, ph, mu, std, baseline, polarity, noise=False):
    """
    Function to itterate through several combinations of signal amplitude and offset 
    to determine the poinst at which multiplicity is no longer 0.

    This point defined the limit of the CFD resolution.
    """
    min_offset = []
    amp = []
    for i in np.arange(50,500,10): #Amplitude variation
        for j in np.arange(0.2, 0.001, -0.001): #Time offset variation
            cfd, offset = cfd_testing(np.arange(0, 1000, 1), i, 200, 5, 19, '-', j, noise)
            if len(cfd)/len(np.unique(cfd)) != 1:
                print(f'Smallest offset found for amplitude={i} mV at: offset={round(j,4)}, ')
                min_offset.append(j)
                amp.append(i)
                break
    return min_offset, amp
