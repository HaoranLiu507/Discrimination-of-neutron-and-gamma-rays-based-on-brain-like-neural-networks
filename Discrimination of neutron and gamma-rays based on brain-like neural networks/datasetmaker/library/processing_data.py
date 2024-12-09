#!/usr/bin/env python3
import sys
import os

import pandas as pd
import numpy as np
import dask.dataframe as dd
from library import  processing_math as promath
from library import  processing_simulation as prosim
from library import  processing_pd as propd
from library import  processing_utility as prout
import logging
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import interpolate
from lmfit import Model

sys.path.insert(0, "../")
# import processing as pro
from tqdm import tqdm

pd.options.mode.chained_assignment = None

"""
A library of methods for data processing of neutron tagging data.
"""

def tof_calibration(df, flashrange, distance, phcut=0, qdccut=0, colsuffix='', numBins=np.arange(-250,750,1), calUnit=1e-9, energyCal=False, QDC='lg'):
    """
    Method for calibrating time-of-flight values to SI units or energy
    Parameters:
    df..................DataFrame containing the relevant time-of-flight values.
    flashrange..........A dicionary with each entry named the same as the tof columns containing an array with min and max values for fitting the gamma flash. 
                        Example: flashrange = {'tof_ch2':[min, max], 'tof_ch3': [min, max], ...}. Units: uncalibrated tof units.
    distance............The distance between the detector and source in meters.
    phcut...............Cut to apply on the pulse height (in mV)
    qdccut..............Cut to apply on the QDC value.
    colsuffix...........String to append to the column storing the result. If empty, the results will overwrite the previous values.
    numBins.............List of bins to use for binning ToF values.
    calUnit.............Multiplication constant for converting tof bin-values to seconds. Default 1e-9, i.e. 1 ns per tof bin.
    energyCal...........If True, will convert time-of-flight to neutron energy and add as additional columns named: tofE_ch#.
    QDC.................either lg or sg for long gate or short gate. Used for 'qdccut'. Default 'lg'
    """

    print('tof_calibration(): running...') 
    
    #speed-of-light [m/s]
    c = 2.99792e8

    # loop through all tof channels based on flashrange indices.
    for tof_col in flashrange:
        ch = tof_col.strip("tof_ch")
        start = flashrange[f'{tof_col}'][0] #start of fit range
        stop = flashrange[f'{tof_col}'][1] #end of fit range
        if phcut>0:
            # create binned data for fitting.
            y_val, x_val = np.histogram(df.query(f'amplitude_ch{ch}>{phcut}')[f'{tof_col}'], bins=numBins)

        elif qdccut>0:
            # create binned data for fitting.
            y_val, x_val = np.histogram(df.query(f'qdc_{QDC}_ch{ch}>{phcut}')[f'{tof_col}'], bins=numBins)
        try:
            # try and fit the data.
            fit_val = promath.gaussFit(getBinCenters(x_val), y_val, start, stop, error=False) #Fitting range with Gaussian func. Error not used.
            print (f'- YAP_ch{ch} - RAW FIT: A={round(fit_val[0], 2)}, mu={round(fit_val[1], 2)} ns, std={round(fit_val[2], 2)} ns')

            # get T0 position by subtracting time based on distance (between source and detector) in and speed-of-light in seconds in SI.
            T0 = fit_val[1]*calUnit - (distance / c) 
            # convert TOF values from digitizer units to seconds to match T0 value.
            df[f'{tof_col}'] = df[f'{tof_col}'] * calUnit
            # calibrate T0 position by subtract T0 position from all ToF values, setting T0 at time zero in data.
            df[f'{tof_col}'] = df[f'{tof_col}'] - T0

            # check if energy conversion should be made.
            if energyCal: # if True...
                # copy tof_ch# columns into new columns called tofE_ch# to be used for energy conversion.
                df[f'tofE_ch{ch}'] = df[f'{tof_col}']
                # replace negative TOF values with np.nan to produce cleaner tof energy spectrum.
                df.loc[df[f'tofE_ch{ch}'] < 0,f'tofE_ch{ch}'] = np.nan
                # convert TOF values to energy.
                df[f'tofE_ch{ch}'] = TOFtoEnergy(df[f'tofE_ch{ch}'], distance)

        except RuntimeError:
            print(f'Gflash fit, optimal parameters not found for {tof_col}. No calibration done!')
            return 0
    return df

def tofEnergySlicer(df, minE=1.05, maxE=6.105, dE=0.1, colKeep='qdc_lg_ch1', n_thr=0, y_thr=[6450, 6985, 7500, 7850], QDC='lg'):
    """
    Routine for slicing energy values and returning a specified column value.
    Will use YAPs connected to channel 2, 3, 4 and 5
    Returns a separate DataFrame for each YAP.

    Parameters:
    df........the DataSet containing all data
    minE......the minimum energy or starting energy of slice [MeV]
    maxE......the maximum energy or stopping energy (exlusive) of slice [MeV]
    dE........the energy step-size or bin-width [MeV]
    colKeep...the column name of where to get the values which will be saved.
    n_thr.....the threshold to use for neutron detector qdc value. Default = 0
    y_thr.....list of threshold to use for yap detectors in qdc value. Default around 3.0 MeVee for each.
    QDC.......either 'lg' or 'sg' for long gate or short gate QDC. Used for 'n_thr'. Default is 'lg'

    NOTE: Default parameter values will slice the energy spectra in 0.1 MeV centered at 1.0, 1.1, 1.2 ... 5.8, 5.9 and 6.0 MeV
    and return the corresponding value in column 'colKeep'.

    -------------------
    Nicholai Mauritzson
    2021-12-20
    """
    print('----------------------------------------------')
    print(f'tofEnergySlicer() - Processing for column: {colKeep}')

    #YAP channel numbers
    y2_ch = 2
    y3_ch = 3
    y4_ch = 4
    y5_ch = 5

    dfCopy = df.copy()

    #calculate log10 for all energy values. Used to linearize energy cut.
    dfCopy[f'tofE_ch{y2_ch}'] = np.log(df[f'tofE_ch{y2_ch}']) 
    dfCopy[f'tofE_ch{y3_ch}'] = np.log(df[f'tofE_ch{y3_ch}']) 
    dfCopy[f'tofE_ch{y4_ch}'] = np.log(df[f'tofE_ch{y4_ch}'])
    dfCopy[f'tofE_ch{y5_ch}'] = np.log(df[f'tofE_ch{y5_ch}'])

    #make DataFrame to hold and sliced energie values for each YAP
    df_y2 = pd.DataFrame()
    df_y3 = pd.DataFrame()
    df_y4 = pd.DataFrame()
    df_y5 = pd.DataFrame()

    #loop through the energy range and query the TOF energy for all YAPs
    for i in np.arange(minE, maxE, dE):
        minE = np.log(i-dE) #calculate log of min energy
        maxE = np.log(i) #calculate log of max energy
        print(f'<< Slicing from {np.round(i-dE, 3)} to {np.round(i, 3)} MeV, centered on {np.round(i-dE/2, 3)} >>')
        
        #Perform first slice on df_copy for each YAP detector
        y2_temp = dfCopy.query(f'(tofE_ch{y2_ch}>{minE} and tofE_ch{y2_ch}<={maxE}) and qdc_{QDC}_ch1>{n_thr} and qdc_lg_ch{y2_ch}>{y_thr[0]}')[colKeep]
        y3_temp = dfCopy.query(f'(tofE_ch{y3_ch}>{minE} and tofE_ch{y3_ch}<={maxE}) and qdc_{QDC}_ch1>{n_thr} and qdc_lg_ch{y3_ch}>{y_thr[1]}')[colKeep]
        y4_temp = dfCopy.query(f'(tofE_ch{y4_ch}>{minE} and tofE_ch{y4_ch}<={maxE}) and qdc_{QDC}_ch1>{n_thr} and qdc_lg_ch{y4_ch}>{y_thr[2]}')[colKeep]
        y5_temp = dfCopy.query(f'(tofE_ch{y5_ch}>{minE} and tofE_ch{y5_ch}<={maxE}) and qdc_{QDC}_ch1>{n_thr} and qdc_lg_ch{y5_ch}>{y_thr[3]}')[colKeep]

        #Append list of indices for sliced events with multiplicity > 1
        mpIdx = uniqueEventHelper(y2_temp, y3_temp, y4_temp, y5_temp)

        #concatenate into final DataFrame and remove all multiplicity > 1 events 
        # (ignoring index errors since 'mpIdx' is common across the entire data-set)
        df_y2 = pd.concat([df_y2, pd.DataFrame({f'keV{int(round(i-dE/2, 2)*1000)}': y2_temp.drop(mpIdx, errors='ignore')})]) #concatenate df_y2_temp into yap 2 DataFrame
        df_y3 = pd.concat([df_y3, pd.DataFrame({f'keV{int(round(i-dE/2, 2)*1000)}': y3_temp.drop(mpIdx, errors='ignore')})]) #concatenate df_y3_temp into yap 3 DataFrame
        df_y4 = pd.concat([df_y4, pd.DataFrame({f'keV{int(round(i-dE/2, 2)*1000)}': y4_temp.drop(mpIdx, errors='ignore')})]) #concatenate df_y4_temp into yap 4 DataFrame
        df_y5 = pd.concat([df_y5, pd.DataFrame({f'keV{int(round(i-dE/2, 2)*1000)}': y5_temp.drop(mpIdx, errors='ignore')})]) #concatenate df_y5_temp into yap 5 DataFrame

    return df_y2, df_y3, df_y4, df_y5

def tofEnergySlicerRebin(df, reBinFactor = 2, skipCols = 0):
    """
    Takes data in the output-format of tofEnergySlicer() and concatenated the desired columns to make more
    course energy-bins, increasing statistics.
    Default: reBinFactor = 2, which adds every-other column (energy-bin) together.
    This effectively reduces the energy bin-width by factor two and increasese the statistics
    by the same amount.

    --------------------------
    Nicholai Mauritzson
    2021-10-18
    """

    dfReBin = pd.DataFrame()
    dfReBinTMP = pd.Series()
    
    # if skipCols != 0:
    columns = df.columns[skipCols:] #get list of current column names
        
    count = 0
    energyValue = 0
    dfReBin = pd.DataFrame()
    dfReBin['gamma'] = df.gamma
    dfReBin['random'] = df.random

    for col in columns:

        if count == reBinFactor:
            dfReBin = pd.concat([dfReBin, pd.DataFrame({f'MeV{int(round(energyValue/reBinFactor, 2))}': dfReBinTMP})])
            print(f'merged! -> average = {round(energyValue/reBinFactor, 2)}')
            count = 0 #reset counter
            energyValue = 0 #reset energy value calculator
        
        if not(col == 'gamma' or col == 'random'):#Keep the gamma and random columns
            dfReBinTMP = pd.concat([dfReBinTMP, df[f'{col}'].dropna()])
            print(f'{col} << stored!')
            count += 1
            energyValue += int(col.split("MeV")[1])
        
    return dfReBin

def uniqueEventHelper(Y2Data, Y3Data, Y4Data, Y5Data, verbose=False):
    """
    Routine for identifying unique events between 4 YAPs based on DataFrame event-index.
    Will return an index list of all intersecting events, which to REMOVE. 
    These intersecting events are either multiplicity 2 or higher events. 

    -------------
    Nicholai Mauritzson
    2022-04-14
    """
    if verbose:
        print('----------------------------------------')
        print('>> Running uniqueEventsHelper()....')
    
    #Find the intersections/multiplicities between indexes
    set23 = np.intersect1d(Y2Data.index, Y3Data.index)
    set24 = np.intersect1d(Y2Data.index, Y4Data.index)
    set25 = np.intersect1d(Y2Data.index, Y5Data.index)
    set34 = np.intersect1d(Y3Data.index, Y4Data.index)
    set35 = np.intersect1d(Y3Data.index, Y5Data.index)
    set45 = np.intersect1d(Y4Data.index, Y5Data.index)
    
    NumEvts = len(Y2Data) + len(Y3Data) + len(Y4Data) + len(Y5Data) #Sum all total events (incl. multiplicities)
    totalCopies = len(set23) + len(set24) + len(set25) + len(set34) + len(set35) + len(set45) #Sum the total multiplicities
    if verbose:
        print(f'>> Number of events = {NumEvts}')
        print(f'>> Total multiplicity events = {totalCopies}')
        try:
            print(f'>> Ratio = {round(totalCopies/NumEvts*100, 2)}%')
        except ZeroDivisionError:
            print(f'>> Ratio = 0%')    
        print('----------------------------------------')
    
    intersectingIdx = np.concatenate((set23, set24, set25, set34, set35, set45)) #Concatenate all interesection into one list
    intersectingIdx = np.unique(intersectingIdx) #Removed reoccuring intersecting events (multiplicity >= 2)

    return intersectingIdx


def concatenate(df, cols):
    """
    Method for concatenating columns from the same pandas DataFrame. Returns new data frame with all data concatenated into one column.
    
    df..........Data Frame containing the data.
    cols........List or array of column names (strings) to be concatenated together.

    """
    concat_temp = []
    for col in cols: #itterate through all file names.
            concat_temp.append(df[f'{col}'])
    return pd.concat(concat_temp)

def TOFtoEnergy(data, distance):
    """
    Method for converting neutron time-of-flight to non-relativistic kinetic energy in MeV
    
    'data'........the time-of-flight value of the neutron in seconds.
    'distance'....the distance between source and detector, in meters.

    Returns kinetic energy of the neutron in MeV.
    -------------------------------------
    Nicholai Mauritzson
    2019-06-03
    """
    m_n = 938.27231 #mass of neutron MeV/c^2
    c = 2.99792458e8 #speed of light (m/s)
    return (m_n * distance**2)/(2 * c**2 * data**2)

def EnergytoTOF(E, distance):
    """
    Method for converting neutron kinetic energy to time-of-flight time.
    
    'E'...........the kinetic energy of the neutron in MeV.
    'distance'....the distance between source and detector, in meters.

    Returns kinetic energy of the neutron in MeV.
    -------------------------------------
    Nicholai Mauritzson
    2019-09-02
    """
    m_n = 938.27231 #mass of neutron MeV/c^2
    c = 2.99792458e8 #speed of light (m/s)
    return np.sqrt((m_n * distance**2)/(2 * c**2 * E))

def PSDcharge(df, LG='qdc_det0', SG='qdc_sg_det0'):
    """
    Method takes a pandas data frame as input and calculateds the pulse shape based on charge integration.
    Data frame must contain columns 'qdc_det0' and 'qdc_stg_det0' for the long gate and short gate, respectively.
    ------------------------------------------------
    Nicholai Mauritzson
    2019-06-18
    """
    return (df[f'{LG}']-df[f'{SG}'])/df[f'{LG}']

def QDCenergyCalibration(df, col='qdc_ld_ch1', a=0, b=0, c=0):
    """
    A method which takes a data frame and calibrated the the QDC values in one column, converting them using the three parameters a, b and c.
    NOTE: calibration function is in the form: y = a+bx+bx^2, a polynomial.

    'df'........This is the data frame containg the column to be calibrated.
    'col'.......This is is the name of the column containing QDC values to be calibrated. Default is 'qdc_det0'.
    'a'.........This is the zero:th order parameter. Default is 0.
    'b'.........This is the first order parameter. Default is 0.
    'c'.........This is the second order parameter. Default is 0.
    """
    df[f'{col}'] = a + df[f'{col}'] + b*df[f'{col}'] + c*df[f'{col}']**2
    return df

def calculatorNeutronVelocity(Ekin=None, v=None):
    """
    Method takes either the velocity or kinetic energy of a neutron (non relativistic) and calculates the other.

    'Ekin'.....The neutron kinteic enegy, given in MeV.
    'v'........The velocity, given in m/s.
    """
    m_n = 938.27231 #mass of neutron MeV/c^2
    c = 2.99792458e8 #speed of light (m/s)

    if Ekin==None:
        #Calculate Ekin based on velocity 'v'.
        if v==None:
            print('calculatorNeutronVelocity() - Error: Either kinteic energy or velocity needs to be given.')
            return 0
        else:
            print(f'Ekin = {round(m_n*v**2/(2*c**2),2)} MeV')
            print(f'Velocity = {v} m/s')
            return m_n*v**2/(2*c**2)

    if v==None:
        #Calculate velocity 'v' based on Ekin.
        if Ekin==None:
            print('calculatorNeutronVelocity() - Error: Either kinteic energy or velocity needs to be given.')
            return 0
        else:
            print(f'Ekin = {Ekin} MeV')
            print(f'Velocity = {round(np.sqrt(2*Ekin*c**2/(m_n)),2)} m/s')
            return np.sqrt(2*Ekin*c**2/(m_n))

def TOFerrorCalculator(distance, distance_error):
    """
    !!! WORK IN PROGRESS !!!

    TODO: Use gaussian fit errors of TDC calibration data (with different lenght delay cables) to weight linear fit.
    TODO: Make weighted linear fit of TDC position vs nano seconds.
    TODO: Use linear fit error to estimate the error in nano seconds.
    TODO: Include the systematic error in the measured distance.
    """

def getBinCenters(bins):
    """ 
    Calculate center values for given bins. 
    Author: Hanno Perrey 
    """
    return np.array([np.mean([bins[i],bins[i+1]]) for i in range(0, len(bins)-1)])

def peakFinder_df(df, threshold=2, distance=20, prominence=20, polarity='-'):
    """
    Function to add peak-index array column to DataFrame. 
    
    Useful values for parameters:
    distance = 220 
    prominence = 6 
    width = 5
    """
    channels = np.arange(0)
    for i in range(len(df.columns)): # get list of columns names for each channels sample-set
        if 'sample' in df.columns[i]:
            channels = np.append(channels, df.columns[i])

    for ch in channels: #loop through all sample channels
        ch_num = ch.split('ch')[-1] #get channel number
        df[f'peak_finder_ch{ch_num}'] = np.nan #create column for peak index for current channel
        df[f'peak_finder_ch{ch_num}'] = df[f'peak_finder_ch{ch_num}'].astype(object) #cast type as object to enable it to hold arrays
        print(f'going through column: {ch}')
        for idx in tqdm(df.index): #loop through all events for current samples column
            df[f'peak_finder_ch{ch_num}'][idx] = [prout.peakFinder(df[f'{ch}'][idx], prominence=prominence, polarity='-')]

def cfdX(samples, baseline, peakidx, delay, time_frac=0.2, polarity='-'):
    """
    CFD algorithm using crossing point method.

    TODO: implement polarity parameter for negativ and positive pulses

    1. Signal is copied
    2. Signal copy is delayed by delay
    3. Signal copy is inverted and attenuated by time_frac
    4. Signal and signal copy is added together. 
    5. Zero crossing points extracted as the CFD value.
    """

    # ========== SIGNAL PROCESSING ==========
    #Subtract baseline from signal
    signal = samples-baseline
    #Copy original signal and offset by the delay
    signal_copy = signal[delay:-1]
    #Invert copy of signal and apply time_fraction
    signal_copy = signal_copy*(-time_frac)
    #Calculate the sum of both signals
    sum = signal[0:len(signal_copy)]+signal_copy

    # ========== CFD COARSE ==========
    #Find first index after zero-crossing
    rise_index = np.nan #set default value for rise_index
    if (peakidx-20)>=0 and (peakidx+15)<=len(signal_copy): #Filter out events with index out-of-range.
        for i in range(15): #Try and find the zero-crossing points within 15 samples.
            max_idx = np.argmax(sum[peakidx-20:peakidx+15])+peakidx-20 #find index of maximum value (for negative polarity)
            if sum[max_idx+i]<0: #when using negative polarity
                rise_index = max_idx+i
                break
            
    # ========== CFD FINE ==========
    if rise_index is np.nan: #check if rise_index was not found.
        return np.nan
    else:
        # Calculting the slope between the two samples points.
        slope_rise = (sum[rise_index] - sum[rise_index-1]) / (rise_index-(rise_index-1)) #calculate the slope of straight line between points
        # Getting the stright line constant
        if slope_rise!=0: #remove bad events with no slope
            const = sum[rise_index]-(slope_rise*(rise_index))
            # Calculating the zero crossing point between the two index.
            return -const/slope_rise
        else:
            return np.nan

        
def CFD_threshold(sample, baseline, thr):
    """
    Time pick-off method using a constant threshold.

    Methods find the two samples points on either side of the threshold and then
    interpolates a straight line across them to get a higher float value for the time
    """
    #make sample values positive
    sample = abs(sample-baseline)
    #get the location of the first sample point above threshold
    rise_index = np.argmax(sample>thr) 
    
    #calculate the slope between the two points across the threshold.
    slope_rise = (sample[rise_index]-sample[rise_index-1]) / (rise_index-(rise_index-1)) #calculate the slope of straight line between points
    if slope_rise!=0: #remove bad events with no slope
        const = sample[rise_index]-(slope_rise*(rise_index))
        # Calculating the zero crossing point between the two index.
        return thr-const/slope_rise
    else:
        return np.nan

def linearEnergyFit(E_data, QDC_data, QDC_err, ped = 0, plot = False):
    """
    TODO: implement plot = False trigger
    TODO: return fitted linear fit values incl errors as dict: See promat.comptonEdgeFit() for reference.
    --------------------------------------
    Nicholai Mauritzson
    2020-08-19
    """
    #FITTING "QDC" VS ENERGY
    print(' -------------------------------')
    print('Fitting QDC vs. ENERGY')
    k_guess = (QDC_data[2]-QDC_data[1])/(E_data[2]-E_data[1])
    m_guess = 0

    print(f'k guess = {k_guess}')
    print(f'm guess = {m_guess}')

    popt1, pcov1 = curve_fit(promath.linearFunc, E_data, QDC_data, p0=[k_guess, m_guess] , sigma=QDC_err, absolute_sigma=True) #Make a linear error weighted fit.
    xvalues1 = np.linspace(0, 5) #Energy, MeVee
    yvalues1 = promath.linearFunc(xvalues1, popt1[0], popt1[1]) #QDC values
    fitError1 = np.sqrt(np.diag(pcov1))
    print(f'fitError = {fitError1}')

    error1 = []
    for i in range(len(xvalues1)): #Calculation of error propagation, linear fit
        constError1 = promath.errorPropConst(xvalues1[i], fitError1[0])
        error1.append(promath.errorPropAdd([constError1, fitError1[1]]))

    print('Missing pedestal by: %.4f'%(popt1[1]-ped)) #Print the difference from pedestal position.


    # FITTING ENERGY VS QDC
    #::::::::::::::::::::::::::::::::::::::
    print(' -------------------------------')
    print('Fitting ENERGY vs. QDC')
    popt2, pcov2 = curve_fit(promath.linearFunc, QDC_data, E_data) #Make a linear fit.
    xvalues2 = np.linspace(0, np.max(QDC_data)*1.1) #QDC values
    yvalues2 = promath.linearFunc(xvalues2, popt2[0], popt2[1]) #Energy, MeVee
    fitError2 = np.sqrt(np.diag(pcov2))
    print(f'fitError = {fitError2}')

    error2 = []
    for i in range(len(xvalues2)): #Calculation of error propagation, linear fit
        constError2 = promath.errorPropConst(xvalues2[i], fitError2[0])
        error2.append(promath.errorPropAdd([constError2, fitError2[1]]))

    print('Missing pedestal by: %.4f'%(popt2[1]-ped)) #Print the difference from pedestal position.

    if plot:    
        plt.figure(0)
        plt.title('QDC vs. Energy')
        plt.scatter(0, ped, label='Zero position (not fitted)', marker='x', color='black')
        plt.scatter(E_data, QDC_data, color='purple', lw=3, label='Data')
        plt.plot(xvalues1, yvalues1, 'k--',label=f'Wighted fit: y={round(popt1[0],2)}x+{round(popt1[1],2)}')
        plt.fill_between(xvalues1, promath.linearFunc(xvalues1, popt1[0], popt1[1])+error1, promath.linearFunc(xvalues1, popt1[0], popt1[1])-error1, color='green', alpha=0.45, label='error')
        plt.xlabel('Energy [MeV$_{ee}$]')
        plt.ylabel('QDC [arb. units]')
        plt.grid(which='both')
        plt.legend()

        plt.figure(1)
        plt.title('Energy vs. QDC')
        plt.scatter(0, ped, label='Zero position (not fitted)', marker='x', color='black')
        plt.scatter(QDC_data, E_data, color='purple', lw=3, label='Data')
        plt.plot(xvalues2, yvalues2, 'k--', label=f'Wighted fit: y={round(popt2[0],8)}x+{round(popt2[1], 8)}')
        plt.fill_between(xvalues2, promath.linearFunc(xvalues2, popt2[0], popt2[1])+error2, promath.linearFunc(xvalues2, popt2[0], popt2[1])-error2, color='green', alpha=0.45, label='error')
        plt.xlabel('QDC [arb. units]')
        plt.ylabel('Energy [MeV$_{ee}$]')
        plt.grid(which='both')
        plt.legend()
        plt.show()

    dic_res = {
    '1_xval':xvalues1,#QDC vs Energy
    '1_yval':yvalues1,#QDC vs Energy
    '1_param_k':popt1[0], #QDC vs Energy
    '1_param_m':popt1[1], #QDC vs Energy
    '1_param_k_error':fitError1[0], #QDC vs Energy
    '1_param_m_error':fitError1[1], #QDC vs Energy
    '2_xval':xvalues2,#Energy vs QDC
    '2_yval':yvalues2,#Energy vs QDC
    '2_param_k':popt2[0], #Energy vs QDC
    '2_param_m':popt2[1], #Energy vs QDC
    '2_param_k_error':fitError2[0], #Energy vs QDC
    '2_param_m_error':fitError2[1]} #Energy vs QDC

    return  dic_res

def reBinning(bins, counts, binCoeff=1):
    """
    Method used to rebin already binned data down to a smaller bin width

    Parameters:
    bins.............The bin values of the current histogram.
    counts...........The count value of the current histogram.
    binCoeff.........Rebinning coefficient, must be integer. Binwidth will be a factor 1/binCoeff smaller.

    ----------------------
    Nicholai Mauritzson
    2021-03-16
    """
    if type(binCoeff) != int:
        print('ValueError: Parameter binCoeff must be an integer!')
        return 0
    elif binCoeff == 0:
        print('ValueError: binCoeff must be greater than 0')
        return 0
    else:
        # currentNumBins = len(bins) #get current number of bins
        # newNumBins = int(round(currentNumBins/binCoeff)) #calculate the new number of bins to use based on parameter binCoeff, rounding to closes integer.
        bins_new = np.array([np.mean([bins[i], bins[i+binCoeff]]) for i in np.arange(0, len(bins)-binCoeff, binCoeff)])
        # bins_new = np.array([np.mean([bins[i], bins[i+binCoeff]]) for i in np.arange(0, len(bins), binCoeff)])
        counts_new = np.array([np.sum(counts[i:i+binCoeff]) for i in np.arange(0, len(bins)-binCoeff, binCoeff)])

        return bins_new, counts_new

def getLiveTime(path, runNum, col='evtno'):
    return 0

def liveTimeCalc(df, silent=True):
    """
    Method for deriving the live time. 
    Takes dataframe of containg a column named evtno which has the event numbers.
    Calculates and returns the fraction which is live_time.

    The deadtime is then simply: 1 - live_time

    -------------------------------------
    Written by Hanno Perrey,
    Modified by Nicholai Mauritzson
    2020-12-14
    """
    evtdiff = df.evtno.diff()
    live_time = len(evtdiff)/evtdiff[evtdiff>0].sum() #Do not include negative evtdiff values in the live_time calculations.

    # the following (optional) output allows to check that there are no
    # "oddities" such as overflows or jumps in the event numbers
    if not silent:
        print(f"  -  number of events: {len(evtdiff)}")
        print(f"  -  number of triggers counted: {evtdiff.sum()}")
        print(f"  -  maximum diff between events: {evtdiff.max()}")
        print(f"  -  average diff between events: {evtdiff.mean()}")
        print(f"  -  number of diff >1: {evtdiff[evtdiff > 1].count()}")
        print(f"  -  number of values below 0 (overflows?): {evtdiff[evtdiff < 0].count()}")
        print(f"  -  minimum diff between events: {evtdiff.min()}")
        print(f"Determined live time to be: {100*live_time}%")
    return live_time

def getRandDist(bins, data):
    """ calculates a random distribution for each bin based on bin content, thus allowing rebinning of histogram """
    return np.array([
        # get random number in the range lowedge < x < highedge for the current bin
        (bins[binno + 1] - bins[binno])*np.random.random_sample() + bins[binno]
        # loop over all bins
        for binno in range(0, len(bins)-1)
        # for each entry in bin; convert float to int; does not handle negative entries!
        for count in range (0, max(0,int(round(data[binno]))))
    ])

def getEvenDist(bins, data):
    """ calculates a random distribution for each bin based on bin content, thus allowing rebinning of histogram """
    return np.array([
        # get fraction number in the range lowedge < x < highedge for the current bin
        (bins[binno + 1] - bins[binno])*(count/data[binno]) + bins[binno]
        # loop over all bins
        for binno in range(0, len(bins)-1)
        # for each entry in bin; convert float to int; does not handle negative entries!
        for count in range (0, max(0,int(round(data[binno]))))
    ]) 

def PSFOM(mean_n, mean_g, FWHM_n, FWHM_g):
    return np.abs(mean_n-mean_g)/(FWHM_n+FWHM_g)

def PSFOM_error(FOM, mean_n, mean_g, std_n, std_g, err_mean_n, err_mean_g, err_std_n, err_std_g):
        sum1_err = promath.errorPropAdd((err_mean_n, err_mean_g))
        sum2_err = promath.errorPropAdd((err_std_n, err_std_g))
        # final_err = promath.errorPropMulti(FOM, (mean_n-mean_g, std_n+std_g), (sum1_err, sum2_err))
        final_err = promath.errorPropMulti(FOM, (mean_n, mean_g, std_n, std_g), (err_mean_n, err_mean_g, err_std_n, err_std_g))
        return final_err


def PSFOMnumeric(binCentersNeutron, binCentersGamma, countsNeutron, countsGamma, fitLim_n=[0,1], fitLim_g=[0,1], stdFactor=3, error=False, plot=False, return_param=False, verbose=False):
    """
    'Numeric' method!

    method takes binned data in form of bincenters and counts for a gamma and a neutron PS data-set respectively.
    The 'FitLim' variables are used to define the initial fit (Gaussian) around the data. 
    After, parameters from this fit is used to constrain the numeric intergration and FOM caluclations. 
    after ters from this the distribution between a set start and stop number. 
    Returns the FOM and error.

    parameters:
    binCentersNeutron.....x-values for neutron PS data
    binCentersGamma.......x-values for gamma PS data
    countsNeutron.........y-values for neutron PS data
    countsGamma...........y-values for gamma PS data
    fitLim_n..............initial guess-values for exploratory gaussian fit, neutron x-values.
    fitLim_g..............initial guess-values for exploratory gaussian fit, gamma x-values.
    stdFactor.............number of sigma (std) around centroid over which to perform the numeric intergraction and FOM calculations.
    plot..................when True, will show a plots of data and fits overlain. Default: False
    return_param..........When True, will return a DataFrame with neutron and gamma mean valuues, and error, and neutron and gamma standard diviation and error. Default: False

    ----------------------------
    Nicholai Mauritzson
    2022-02-11
    """
    #perform exploratory gaussian fit for both data sets
    popt_n, pcov_n = promath.gaussFit(binCentersNeutron,    countsNeutron, fitLim_n[0], fitLim_n[1], error=True)
    popt_g, pcov_g = promath.gaussFit(binCentersGamma,      countsGamma,   fitLim_g[0], fitLim_g[1], error=True)

    #rename fitted paramters for convenience
    amp_n = popt_n[0]
    err_amp_n = pcov_n[0]
    centroidFit_n = popt_n[1]
    stdFit_n = popt_n[2]

    amp_g = popt_g[0]
    err_amp_g = pcov_g[0]
    centroidFit_g = popt_g[1]
    stdFit_g = popt_g[2]
    
    #slice data based on gaussian fit paramters
    x_n, y_n = binDataSlicer(binCentersNeutron, countsNeutron,  centroidFit_n - (stdFactor*stdFit_n), centroidFit_n + (stdFactor*stdFit_n))
    x_g, y_g = binDataSlicer(binCentersGamma,   countsGamma,    centroidFit_g - (stdFactor*stdFit_g), centroidFit_g + (stdFactor*stdFit_g))
    
    #Calculate numeric values: neutrons
    try:
        mean_n = np.average(x_n, weights=y_n)
        std_n = np.sqrt(np.abs(np.average((x_n-mean_n)**2, weights=y_n)))
    except ZeroDivisionError:
        print('PSFOMnumeric()... ZeroDivisionError neutrons')
        mean_n = 0
        std_n = 1

    #Calculate numeric values: gammas
    try:
        mean_g = np.average(x_g, weights=y_g)
        std_g = np.sqrt(np.abs(np.average((x_g-mean_g)**2, weights=y_g)))
    except ZeroDivisionError:
        print('PSFOMnumeric()... ZeroDivisionError gammas')
        mean_g = 0
        std_g = 1
        
    A = 2*np.sqrt(2*np.log(2)) #conversion factor from std to FWHM for a standard distribution
    FOM = PSFOM(mean_n, mean_g, std_n*A, std_g*A)

    if verbose:
        print('-------------PSFOMnumeric())----------------------')
        print(f'amp_n = {round(amp_n,3)}')
        print(f'mean_n = {round(mean_n,3)}')
        print(f'std_n = {round(std_n,3)}')
        print(f'amp_g = {round(amp_g,3)}')
        print(f'mean_g = {round(mean_g,3)}')
        print(f'std_g = {round(std_g,3)}')
        print(f'FOM = {round(FOM,3)}')

    if plot:
        plt.title(round(FOM, 3))
        plt.step(x_n, y_n, color='blue', lw=2)
        plt.step(x_g, y_g, color='red', lw=2)
        plt.plot(np.arange(0, 1, 0.001),promath.gaussFunc(np.arange(0, 1, 0.001), popt_n[0], popt_n[1], popt_n[2]), color='black')
        plt.plot(np.arange(0, 1, 0.001),promath.gaussFunc(np.arange(0, 1, 0.001), popt_g[0], popt_g[1], popt_g[2]), color='black')
        plt.show()

    if error:
        err_mean_n = std_n/np.sqrt(len(x_n))
        err_mean_g = std_g/np.sqrt(len(x_g))
        err_std_n = std_n*0.05 #set error of sigma manually neutron
        err_std_g = std_n*0.05 #set error of sigma manually neutron
        if verbose:
            print(f'mean_n_error = {round(err_mean_n,4)}')
            print(f'std_n_error = {round(err_std_n,4)}')
            print(f'mean_g_error = {round(err_mean_g,4)}')
            print(f'std_g_error = {round(err_std_g,4)}')
        #Get error of FOM
        FOM_err = PSFOM_error(FOM, mean_n, mean_g, std_n, std_g, err_mean_n, err_mean_g, err_std_n, err_std_g)
        
        if return_param:
            paramters = pd.DataFrame({  'amp_n': [amp_n],
                                        'err_amp_n': [err_amp_n],
                                        'mean_n': [mean_n],
                                        'err_mean_n': [err_mean_n],
                                        'std_n': [std_n],
                                        'err_std_n': [err_std_n],
                                        'amp_g': [amp_g],
                                        'err_amp_g': [err_amp_g],
                                        'mean_g': [mean_g],
                                        'err_mean_g': [err_mean_g],
                                        'std_g': [std_g],
                                        'err_std_g': [err_std_g]})
            return FOM, FOM_err, paramters
            
        else:
            return FOM, FOM_err
    print('--------------------------------------------------')
   
    return FOM


def PSFOMdoubleGauss(binCentersNeutron, binCentersGamma, countsNeutron, countsGamma, fitLim_n=[0,1], fitLim_g=[0,1], param=[1,1,1,1,1,1], error=False, plot=False, return_param=False, verbose=False):
    """
    Double gaussin fit and FOM calculation method.

    method takes binned data in form of bincenters and counts for a gamma and a neutron PS data-set, respectively.
    The 'FitLim' variables are used to define the initial fit (Gaussian for gamma and double Gaussian for neutron data) around the data. 
    After, parameters from this fit is used to calculate the FOM.
    
    Returns the FOM and error.

    parameters:
    binCentersNeutron.....x-values for neutron PS data
    binCentersGamma.......x-values for gamma PS data
    countsNeutron.........y-values for neutron PS data
    countsGamma...........y-values for gamma PS data
    fitLim_n..............initial guess-values for exploratory gaussian fit, neutron x-values.
    fitLim_g..............initial guess-values for exploratory gaussian fit, gamma x-values.
    plot..................when True, will show a plots of data and fits overlain. Default: False
    param.................List of first guess for fitted paramters (amp1, mean1, std1, amp2, mean2, std2)
    return_param..........When True, will return a DataFrame with neutron and gamma mean valuues, and error, and neutron and gamma standard diviation and error. Default: False

    ----------------------------
    Nicholai Mauritzson
    2022-04-26
    """
    #perform exploratory gaussian fit for both data sets
    popt_n, pcov_n = promath.doubleGaussFit(binCentersNeutron,  countsNeutron, start = fitLim_n[0], stop = fitLim_n[1], param = param, error=True)
    popt_g, pcov_g = promath.gaussFit(binCentersGamma,          countsGamma,   start = fitLim_g[0], stop = fitLim_g[1], error=True)

    #rename fitted paramters for convenience
    amp_n1 = popt_n[0]
    err_amp_n1 = pcov_n[0]
    mean_n1 = popt_n[1]
    std_n1 = popt_n[2]
        
    amp_n2 = popt_n[3]
    err_amp_n2 = pcov_n[3]
    mean_n2 = popt_n[4]
    std_n2 = popt_n[5]
    
    amp_g = popt_g[0]
    err_amp_g = pcov_g[0]
    mean_g = popt_g[1]
    std_g = popt_g[2]
            
    A = 2*np.sqrt(2*np.log(2)) #conversion factor from std to FWHM for a standard distribution

    FOM = PSFOM(mean_n2, mean_g, std_n2*A, std_g*A)
    if verbose:
        print('-------------PSFOMdoubleGaussian())----------------------')
        print(f'amp_n2 = {round(amp_n2,3)}')
        print(f'mean_n2 = {round(mean_n2,3)}')
        print(f'std_n2 = {round(std_n2,3)}')
        print(f'amp_g = {round(amp_g,3)}')
        print(f'mean_g = {round(mean_g,3)}')
        print(f'std_g = {round(std_g,3)}')
        print(f'FOM = {round(FOM,3)}')

    if plot:
        plt.title(np.round(FOM, 3))
        plt.step(binCentersNeutron, countsNeutron, color='blue', lw=2)
        plt.step(binCentersGamma, countsGamma, color='red', lw=2)
        plt.plot(np.arange(0, 1, 0.001), promath.doubleGaussFunc(np.arange(0, 1, 0.001), popt_n[0], popt_n[1], popt_n[2], popt_n[3], popt_n[4], popt_n[5]), color='black')
        plt.plot(np.arange(0, 1, 0.001), promath.gaussFunc(np.arange(0, 1, 0.001), popt_g[0], popt_g[1], popt_g[2]), color='black')
        plt.show()

    if error:
        err_mean_n1 = pcov_n[1]
        err_mean_n2 = pcov_n[4]
        err_mean_g = pcov_g[1]
        err_std_n1 = pcov_n[2]
        err_std_n2 = pcov_n[5]
        err_std_g = pcov_g[2]
        if verbose:
            print(f'mean_n_error = {np.round(err_mean_n2,4)}')
            print(f'std_n_error = {np.round(err_std_n2,4)}')
            print(f'mean_g_error = {np.round(err_mean_g,4)}')
            print(f'std_g_error = {np.round(err_std_g,4)}')
        
        #Get error of FOM
        FOM_err = PSFOM_error(FOM, mean_n2, mean_g, std_n2, std_g, err_mean_n2, err_mean_g, err_std_n2, err_std_g)
        
        if return_param:
            paramters = pd.DataFrame({  'amp_n1': [amp_n1],
                                        'err_amp_n1': [err_amp_n1],
                                        'mean_n1': [mean_n1],
                                        'err_mean_n1': [err_mean_n1],
                                        'std_n1': [std_n1],
                                        'err_std_n1': [err_std_n1],
                                        'amp_n2': [amp_n2],
                                        'err_amp_n2': [err_amp_n2],
                                        'mean_n2': [mean_n2],
                                        'err_mean_n2': [err_mean_n2],
                                        'std_n2': [std_n2],
                                        'err_std_n2': [err_std_n2],
                                        'amp_g': [amp_g],
                                        'err_amp_g': [err_amp_g],
                                        'mean_g': [mean_g],
                                        'err_mean_g': [err_mean_g],
                                        'std_g': [std_g],
                                        'err_std_g': [err_std_g]})
            return FOM, FOM_err, paramters
            
        else:
            return FOM, FOM_err
    print('--------------------------------------------------')
   
    return FOM

def PSFOMneutron(data, binRange=np.arange(0, 0.75 , 0.008), gLength=None, randLength=None, nLength=None, plot=False, error=False):
    """
    Method for calculating FOM as a function of neutron kinetic energy
    ---------------------------
    Nicholai Mauritzson
    2021-11-08
    """
    cols = data.columns[3:]#remove column names "gamma" and "random" and "MeV10"
    #derive random PS
    yRand, xRand = np.histogram(data.random, bins=binRange)
    #derive gamma PS
    yGamma, xGamma = np.histogram(data.gamma, bins=binRange)
    xGamma = getBinCenters(xGamma)
    Rg = gLength/randLength
    #subtract random background from gamma data
    yGamma = yGamma-yRand*Rg

    #empty container for FoM results
    FoM = np.empty(0)
    FoM_err = np.empty(0)

    for col in cols:
    
        yNeutron, xNeutron = np.histogram(data[f'{col}'], bins=binRange)
        xNeutron = getBinCenters(xNeutron)
        #calculate Rn for current neutron energy
        Rn = nLength[f'{col}'].item()/randLength
        #subtract random background from gamma data
        yNeutron = yNeutron-yRand*Rn
        # meanNeutron = np.average(xNeutron, weights=yNeutron)
        #Fit gaussian to gamma PS data
        # pNeutron, errNeutron = curve_fit(promath.gaussFunc, xNeutron, yNeutron, p0 = [np.max(yNeutron), meanNeutron, 0.1])
        FoM_tmp, FoM_err_tmp = PSFOMnumeric(xNeutron, xGamma, yNeutron, yGamma, lim_n=[0, 1], lim_g=[0, 1], error=True)
        FoM = np.append(FoM, FoM_tmp)
        FoM_err = np.append(FoM_err, FoM_err_tmp)

        if plot:
            plt.step(xNeutron, yNeutron, label='neutrons')
            plt.step(xGamma, yGamma, label='gammas')
            plt.legend()
            plt.xlim([0, 1])
            plt.show()
    if error:
        return FoM, FoM_err
    return FoM

def PScalc(SG_data, LG_data, a=0, b=0):
    """
    Charged-based pulse-shape calculator.
    Returns columns of calulated PSD values. Each row is an individual event.
    
    - SG_data....Short gate QDC data.
    - LG_data....Long gate QDC data.
    - a..........Linearisation parameter effecting the short gate value.
    - b..........Linearisation parameter effecting the long gate value.

    Rasmus derived a=287 and b=120 as the best separations values for his thesis.
    ------------------------------------------------
    Nicholai Mauritzson
    2021-09-24
    """
    return 1 - ((SG_data + a) / (LG_data + b))

def PScalc2(SG_data, LG_data, a=0, b=0):
    """
    Charged-based pulse-shape calculator.
    Returns columns of calulated PSD values. Each row is an individual event.
    
    - SG_data....Short gate QDC data.
    - LG_data....Long gate QDC data.
    - a..........Linearisation parameter effecting the short gate value.
    - b..........Linearisation parameter effecting the long gate value.

    Rasmus derived a=287 and b=120 as the best separations values for his thesis.
    ------------------------------------------------
    Nicholai Mauritzson
    2021-10-12
    """
    return 1 - ((a*SG_data) / (b*LG_data))

def sliceLoad(path, numFiles=1, type='PS', merge=True):
    """
    Function with loads and merged sliced data.
    Type.....string, either 'QDC_lg', 'QDC_sg', 'PS' or 'raw_QDC_lg', 'raw_QDC_sg' or 'raw_PS'
    merge....if True, will return one data set with all four YAPs merged.
    ------------------------------------
    Nicholai Mauritzson
    2021-10-26
    """
    df_YAP2 = pd.read_parquet(f"{path}yap2_{type}_part0.pkl") #load initial data set
    df_YAP3 = pd.read_parquet(f"{path}yap3_{type}_part0.pkl") #load initial data set
    df_YAP4 = pd.read_parquet(f"{path}yap4_{type}_part0.pkl") #load initial data set
    df_YAP5 = pd.read_parquet(f"{path}yap5_{type}_part0.pkl") #load initial data set

    for i in range(numFiles):
        df_YAP2 = pd.concat([df_YAP2, pd.read_parquet(f"{path}yap2_{type}_part{i+1}.pkl")]) #merge with initial data-set
        df_YAP3 = pd.concat([df_YAP3, pd.read_parquet(f"{path}yap3_{type}_part{i+1}.pkl")]) #merge with initial data-set
        df_YAP4 = pd.concat([df_YAP4, pd.read_parquet(f"{path}yap4_{type}_part{i+1}.pkl")]) #merge with initial data-set
        df_YAP5 = pd.concat([df_YAP5, pd.read_parquet(f"{path}yap5_{type}_part{i+1}.pkl")]) #merge with initial data-set
    if merge:
        df_total = pd.concat([df_YAP2,df_YAP3,df_YAP4,df_YAP5])
        return df_total
    else:
        return df_YAP2, df_YAP3, df_YAP4, df_YAP5

def sliceRandomSubtractionNeutron(df_YAP2, df_YAP3, df_YAP4, df_YAP5, numBins=None, randLength=0, nLength=0):
    """
    Routine will take data one data-set for each YAP, bin it and then subtract the time-normalized random background selection from it.
    Finally it will merge all randomsubtracted YAP data-sets into one data-set and return the the counts and bin centers of that new data-set.
    
    -------------------------
    Nicholai Mauritzson
    2021-10-18
    """
    countsFin = pd.DataFrame() #np.empty(numBins) #create empty array to hold final binned data-set
    countsRandom_YAP2, binEdgeRandom = np.histogram(df_YAP2.random, bins=numBins) #Bin the random data set for YAP2
    countsRandom_YAP3, binEdgeRandom = np.histogram(df_YAP3.random, bins=numBins) #Bin the random data set for YAP3
    countsRandom_YAP4, binEdgeRandom = np.histogram(df_YAP4.random, bins=numBins) #Bin the random data set for YAP4
    countsRandom_YAP5, binEdgeRandom = np.histogram(df_YAP5.random, bins=numBins) #Bin the random data set for YAP5
    # for YAP in [df_YAP2, df_YAP3, df_YAP4, df_YAP5]:

    for idx, Tn in enumerate(df_YAP2.columns.drop(['gamma', 'random'])): #loop through all the neutron kinetic energies
        countsNeutron_YAP2, binEdgeNeutron = np.histogram(df_YAP2[f'{Tn}'], bins=numBins) #Bin the current neutron data-set for YAP2
        countsNeutron_YAP3, binEdgeNeutron = np.histogram(df_YAP3[f'{Tn}'], bins=numBins) #Bin the current neutron data-set for YAP3
        countsNeutron_YAP4, binEdgeNeutron = np.histogram(df_YAP4[f'{Tn}'], bins=numBins) #Bin the current neutron data-set for YAP4
        countsNeutron_YAP5, binEdgeNeutron = np.histogram(df_YAP5[f'{Tn}'], bins=numBins) #Bin the current neutron data-set for YAP5
        binCenterNeutron = getBinCenters(binEdgeNeutron) #get the center of the bins

        #Calculate time normalization factor based on selection areas.        
        Rn = nLength[idx]/randLength 
        
        #Calculate and append final counts data-set for specific neutron kinetic energy
        countsFin[f'{Tn}'] = (countsNeutron_YAP2-(Rn*countsRandom_YAP2)) + (countsNeutron_YAP3-(Rn*countsRandom_YAP3)) + (countsNeutron_YAP4-(Rn*countsRandom_YAP4)) + (countsNeutron_YAP5-(Rn*countsRandom_YAP5))

        # Tn = int(Tn.split("MeV")[1])/10 #Get the neutron kinetic energy
        print(f'> > > processing... Tn={Tn} MeV')

    return binCenterNeutron, countsFin

def sliceRandomSubtraction(data1, data2, data3, data4, rand1, rand2, rand3, rand4, numBins=None, dataLength=0, randLength=0):
    """takes individual YAP data and random data to individually subtract, merge and then return"""
    y1rand, x1rand = np.histogram(rand1, bins=numBins)
    y2rand, x2rand = np.histogram(rand2, bins=numBins)
    y3rand, x3rand = np.histogram(rand3, bins=numBins)
    y4rand, x4rand = np.histogram(rand4, bins=numBins)

    y1data, x1data = np.histogram(data1, bins=numBins)
    y2data, x2data = np.histogram(data2, bins=numBins)
    y3data, x3data = np.histogram(data3, bins=numBins)
    y4data, x4data = np.histogram(data4, bins=numBins)
    
    y1data = y1data - y1rand*(dataLength/randLength)
    y2data = y2data - y2rand*(dataLength/randLength)
    y3data = y3data - y3rand*(dataLength/randLength)
    y4data = y4data - y4rand*(dataLength/randLength)

    return getBinCenters(x1data), y1data+y2data+y3data+y4data

def YAPmergerThreshold(path, type, threshold, QDC=False, NPG=False):
    """
    Method will take path to folder containing PS x,y values for threshold tests and threshold.
    Returns x, y values of all four YAPs with y values summed with gamma and neutron distibutions separate.

    path..........path to YAP2, YAP3, YAP4 and YAP5 folders.
    type..........either 'N' or 'YAP' for neutron thresholds or YAP thresholds
    threshold....threshold to sum, in units of QDC.
    QDC..........if True, then QDC values will be loaded instead of PS for the same threshold
    ----------------------------------
    Nicholai Mauritzson
    2021-12-01
    """
    if QDC and NPG:
        print('Error: either QDC=True or NPG=True, not both...')
        return 0
    
    if NPG:
        ygYAP2 =    np.load(f'{path}/YAP2/{type}_{threshold}/yGammaNPGQDC.npy')
        ynYAP2 =    np.load(f'{path}/YAP2/{type}_{threshold}/yNeutronNPGQDC.npy')
        ygYAP3 =    np.load(f'{path}/YAP3/{type}_{threshold}/yGammaNPGQDC.npy')
        ynYAP3 =    np.load(f'{path}/YAP3/{type}_{threshold}/yNeutronNPGQDC.npy')
        ygYAP4 =    np.load(f'{path}/YAP4/{type}_{threshold}/yGammaNPGQDC.npy')
        ynYAP4 =    np.load(f'{path}/YAP4/{type}_{threshold}/yNeutronNPGQDC.npy')
        ygYAP5 =    np.load(f'{path}/YAP5/{type}_{threshold}/yGammaNPGQDC.npy')
        ynYAP5 =    np.load(f'{path}/YAP5/{type}_{threshold}/yNeutronNPGQDC.npy')
        x =         np.load(f'{path}/YAP2/{type}_{threshold}/xNPGQDC.npy')
    elif QDC:
        ygYAP2 =    np.load(f'{path}/YAP2/{type}_{threshold}/yGammaQDC.npy')
        ynYAP2 =    np.load(f'{path}/YAP2/{type}_{threshold}/yNeutronQDC.npy')
        ygYAP3 =    np.load(f'{path}/YAP3/{type}_{threshold}/yGammaQDC.npy')
        ynYAP3 =    np.load(f'{path}/YAP3/{type}_{threshold}/yNeutronQDC.npy')
        ygYAP4 =    np.load(f'{path}/YAP4/{type}_{threshold}/yGammaQDC.npy')
        ynYAP4 =    np.load(f'{path}/YAP4/{type}_{threshold}/yNeutronQDC.npy')
        ygYAP5 =    np.load(f'{path}/YAP5/{type}_{threshold}/yGammaQDC.npy')
        ynYAP5 =    np.load(f'{path}/YAP5/{type}_{threshold}/yNeutronQDC.npy')
        x =         np.load(f'{path}/YAP2/{type}_{threshold}/xQDC.npy')
    else:
        ygYAP2 =    np.load(f'{path}/YAP2/{type}_{threshold}/yGamma.npy')
        ynYAP2 =    np.load(f'{path}/YAP2/{type}_{threshold}/yNeutron.npy')
        ygYAP3 =    np.load(f'{path}/YAP3/{type}_{threshold}/yGamma.npy')
        ynYAP3 =    np.load(f'{path}/YAP3/{type}_{threshold}/yNeutron.npy')
        ygYAP4 =    np.load(f'{path}/YAP4/{type}_{threshold}/yGamma.npy')
        ynYAP4 =    np.load(f'{path}/YAP4/{type}_{threshold}/yNeutron.npy')
        ygYAP5 =    np.load(f'{path}/YAP5/{type}_{threshold}/yGamma.npy')
        ynYAP5 =    np.load(f'{path}/YAP5/{type}_{threshold}/yNeutron.npy')
        x =         np.load(f'{path}/YAP2/{type}_{threshold}/x.npy')
    
    yg_sum = ygYAP2 + ygYAP3 + ygYAP4 + ygYAP5
    yn_sum = ynYAP2 + ynYAP3 + ynYAP4 + ynYAP5
    
    return x, yg_sum, yn_sum

def statisticsFilterPS(data1, data2, dataRand, data1Length, data2Length, randLenght, bins, plot=False):
    """
    Statistics filter for PS data. Used to align PS peaks between two data sets.

    1. Bins 'data1' and 'data2' and 'dataRand' using 'bins'.
    2. Subtracts random events from both dat sets
    3. Selects the data set with the highest peak.
    4. Get the ratio diff beteween the peaks.
    5. Derives number of events needed to scale both peaks to the same height.
    6. Randomly selects events from the data set with the highest peaks to match both data sets.
    7. Rebin data and subtracts random events again
    8. Returns binCenters, counts1 and counts2

    -----------------------------
    Nicholai Mauritzson
    2021-11-18
    """
    counts1, x = np.histogram(data1, bins)
    counts2, x = np.histogram(data2, bins)
    countsRand, x = np.histogram(dataRand, bins)
    counts1 = counts1 - (data1Length/randLenght)*countsRand
    counts2 = counts2 - (data2Length/randLenght)*countsRand

    max1 = np.max(counts1)
    max2 = np.max(counts2)
    runNum1 = len(data1)
    runNum2 = len(data2)
    runNumRand = len(dataRand)

    x = getBinCenters(x)

    if plot:
        plt.figure()
        plt.subplot(2,1,1)
        plt.step(x, counts1, color='red')
        plt.step(x, counts2, color='blue')
        plt.ylabel('before')

    if max2 > max1: #if data2 has the highest peak
        ratio = max1/max2
        counts2, xTMP = np.histogram(data2.sample(np.int(runNum2*ratio)), bins)
        countsRand, xTMP = np.histogram(dataRand.sample(np.int(runNumRand*ratio)), bins)
        counts2 = counts2 - (data2Length/randLenght)*countsRand
        if plot:
            plt.subplot(2,1,2)
            plt.step(x, counts1, color='red')
            plt.step(x, counts2, color='blue')
            plt.ylabel('after')
            plt.show()

    else: #if data1 has the highest peak
        ratio = max2/max1
        counts1, xTMP = np.histogram(data1.sample(np.int(runNum1*ratio)), bins)
        countsRand, xTMP = np.histogram(dataRand.sample(np.int(runNumRand*ratio)), bins)
        counts1 = counts1 - (data1Length/randLenght)*countsRand
        if plot:
            plt.subplot(2,1,2)
            plt.step(x, counts1, color='red')
            plt.step(x, counts2, color='blue')
            plt.ylabel('after')
            plt.show()

    return x, counts1, counts2

def calibrateMyDetector(path = None, data = None, inverse = True, verbose=False):
    """
    Method for calibrating QDC to MeVee based on calibration data.
    If inverse=False, a linear function is used.
    If inverse=True, an inverse linear function is used.

    Parameters:
    path......path to calibration files
    data......data to calibrate
    inverse...is the calibration paramters inverse linear (True) or linear (False)
    verbose...if True print current calibration pramters to screen. Default: False

    ---------------------------
    Nicholai Mauritzson
    2022-08-24
    """
    popt = np.load(f'{path}')
    if inverse:
        dataCal = promath.linearFuncInv(data, popt[0], popt[1])
    if not inverse:
        dataCal = promath.linearFunc(data, popt[0], popt[1])
        
    if verbose:
        print(f'calibrateMyDetector() -> popt = {popt}')
   
    return dataCal

def ToT_binned(xInitial, yInitial, leftMin, leftMax, rightMin, rightMax, ratio=True):
    """
    Method calculates the tail-to-total of binned data,
    or the integral of the left peak if ratio=False
    
    --------------------------
    Nicholai Mauritzson
    2021-11-29
    """
    
    yLeft = []
    yRight = []
    for i in range(len(xInitial)):
        if xInitial[i] > leftMin and xInitial[i] <= leftMax:
            yLeft.append(yInitial[i]) #Get y-values in range to fit
        if xInitial[i] > rightMin and xInitial[i] <= rightMax:
            yRight.append(yInitial[i]) #Get y-values in range to fit
    
    leftSum = np.sum(yLeft)
    rightSum = np.sum(yRight)
    if ratio:
        return leftSum/rightSum
    else:
        return leftSum

def npgBimodalFit(x_input, y_input, start=0, stop=0, parameters=[0, 0, 0, 0, 0, 0], return_par = False):
    """
    Method for processing neutron PS data.
    Method will:
    - Fit a bimodal gaussian to data.
    - Integrate non-prompt-gamma component of distribution
    - Return integral value
    """
    if start+stop != 0:
        x, y = promath.binDataSlicer(x_input, y_input, start, stop)
    else:
        x = x_input
        y = y_input
    
    if np.sum(parameters) != 0:
        popt, pcov = curve_fit(promath.gaussFuncBimodal, x, y, p0=parameters)
    else:
        popt, pcov = curve_fit(promath.gaussFuncBimodal, x, y)
    
    if popt[1] > popt[4]: #sort paramters from smallest to largest mean gaussians
        popt = [popt[3], popt[4], popt[5], popt[0], popt[1], popt[2]]

    NPG_integral = np.sum(promath.gaussFunc(x, popt[0], popt[1], popt[2]))

    if return_par:
        return NPG_integral, np.abs(popt), np.abs(pcov)
    else:
        return NPG_integral

def randomTOFSubtraction(dataGamma, dataNeutron, dataRand, gLength, nLength, randLength, numBins, plot=False):
        """
        Methods for random subtracting TOF data.

        -------------------------------------
        Nicholai Mauritzson
        2021-12-02
        """
        Rg = gLength/randLength
        Rn = nLength/randLength

        yRand, x = np.histogram(dataRand, bins=numBins)

        yGamma, x = np.histogram(dataGamma, bins=numBins)

        yNeutron, x = np.histogram(dataNeutron, bins=numBins)
        
        if plot:
            plt.step(getBinCenters(x), yRand, color='black')
            plt.step(getBinCenters(x), yNeutron, color='blue', alpha=0.5)
            plt.step(getBinCenters(x), yGamma, color='red', alpha=0.5)
            plt.step(getBinCenters(x), yNeutron-yRand*Rn, color='blue')
            plt.step(getBinCenters(x), yGamma-yRand*Rg, color='red')
            plt.yscale('log')
            plt.show()
        
        return getBinCenters(x), yGamma-yRand*Rg, yNeutron-yRand*Rn

def randomTOFSubtractionQDC(dataQDC, dataRand, QDCLength, randLength, numBins, error=False, plot=False):
        """
        Methods for random subtracting TOF sliced QDC data.
        TODO: Add error propagation boolean
        -------------------------------------
        Nicholai Mauritzson
        2022-02-18
        """
        R_QDC = QDCLength/randLength

        yRand, x = np.histogram(dataRand, bins=numBins)

        yQDC, x = np.histogram(dataQDC, bins=numBins)
        
        if plot:
            plt.step(getBinCenters(x), yRand, color='black')
            plt.step(getBinCenters(x), yQDC, color='blue', alpha=0.5)
            plt.step(getBinCenters(x), yQDC-yRand*R_QDC, color='blue')
            plt.yscale('log')
            plt.show()
        
        if error:#calculate and propagate statistical error
            y_error = promath.errorPropAdd([np.sqrt(yQDC), np.sqrt(yRand*R_QDC)])

            return getBinCenters(x), (yQDC-yRand*R_QDC), y_error
        else: 
            return getBinCenters(x), (yQDC-yRand*R_QDC)

def PSscaler(x, y1, y2, plot=False):
    """
    Method which takes binned PS data y1 and y2 and scales down the heighest peak to match t he lowest peak
    """
    max1 = np.max(y1)
    max2 = np.max(y2)
    print('-------------------------------------------')
    print(f'G_max > N_max: {max1/max2}')

    if max1 > max2:
        y1New = y1*max2/max1
        y2New = y2
        
    else:
        y1New = y1
        y2New = y2*max1/max2

    print('-------------------------------------------')

    if plot:
        plt.step(x, y1New, label='data 1', color='blue')
        plt.step(x, y2New, label='data 2', color='red')
        plt.step(x, y1New, alpha=0.5, color='blue')
        plt.step(x, y2New, alpha=0.5, color='red')
        plt.legend()
        plt.show()

    return y1New, y2New

def binDataSlicer(x_input, y_input, start, stop):
    """
    Helper method for slicing binned x,y data-set between 'start' and 'stop' c-xalues.
    """
    y = np.empty(0)
    x = np.empty(0)
    for i in range(len(x_input)):
        if x_input[i] >= start and x_input[i] <= stop:
            x = np.append(x, x_input[i]) #Get x-values in range to fit
            y = np.append(y, y_input[i]) #Get y-values in range to fit
    return x, y


def numericAverage(binCenters, counts, fitLim=[0,1], stdFactor=3):
    """
    Numeric average and std calculator.
    
    Derived the weighted mean and stadard diviation of a binned data set (x, y)-values.
    An gaussian fit is fitted across a range (fitLim) and the average and the data is the cut at +/- standard diviation of the fit times 'stdFactor'.

    parameters:
    binCenters.....x-values for data.
    counts.........y-values for data.
    fitLim.........initial guess-values for exploratory gaussian fit
    fitLim_g..............initial guess-values for exploratory gaussian fit, gamma x-values.
    stdFactor.............number of sigma (std) around centroid over which to perform the numeric intergraction and FOM calculations.
    
    ----------------------------
    Nicholai Mauritzson
    2022-02-25
    """

    #perform exploratory gaussian fit for data set
    popt, pcov = promath.gaussFit(binCenters, counts, fitLim[0], fitLim[1], error=True)
    
    #rename fitted paramters for convenience
    centroidFit = popt[1]
    stdFit = popt[2]
    
    #slice data based on gaussian fit paramters
    x, y = binDataSlicer(binCenters, counts,  centroidFit - (stdFactor*stdFit), centroidFit + (stdFactor*stdFit))
    
    #Calculate numeric values for data
    mean = np.average(x, weights=y)
    std = np.sqrt(np.abs(np.average((x-mean)**2, weights=y)))
    
    #Calculate the errors fo numeric values for data
    err_mean = std/np.sqrt(len(x))
    err_std = 0


    print('-------------numericAverage())----------------------')
    print(f'mean = {round(mean,3)}+/-{round(err_mean,4)}')
    print(f'std = {round(std,3)}+/-{round(err_std,4)}')
    
    #save to DataFrame
    paramters = dict({  'meanVal': mean,
                        'err_meanVal': err_mean,
                        'stdVal': std,
                        'err_stdVal': err_std})
    return paramters
    


def gainAlignCalculator(dataPath, runList, QDCThrLow=15000, QDCThrHigh=20000, numBins=200, colName='qdc_lg_ch1', verbose=False):
    """
    Script to check gain variation across data set using chi2 minimization.
    Works by getting relative offset of QDC data with respect to 1st run in 'runList'.

    2022-04-20
    /Nicholai Mauritzson
    """

    gainOffsets = np.ones(len(runList)) #define offset list

    binRange = np.arange(QDCThrLow, QDCThrHigh, numBins) #define bin range array for histograms

    for i, run in enumerate(runList):
        print(f'Reading run: {run} ({i+1}/{len(runList)}) ...')
        dfCurrent = propd.load_parquet_merge(dataPath, [run], keep_col=[f'{colName}'], full=False)
        chi2 = 99999 #reset chi2 value
        chi2Fine = 999999
        gainFinal = 1 #reset gain value

        if i == 0: #save first run of runList as the master data
            # masterLenght = int(dfCurrent.index[-1]*0.5) #save 50% of number of events in first data set.
            # idxList = np.arange(0, masterLenght, 1) #make list of index to use for all other data, this will ensure similar count statistics
            yMaster, xMaster = np.histogram(dfCurrent, bins=binRange)
            gainOffsets[i] = 1
        else:
            for gainCorse in np.arange(0.82, 1.18, 0.01):
                yCurrent, xCurrent = np.histogram(dfCurrent*gainCorse, bins=binRange)
                if round(chi2,2) > round(promath.chi2(yMaster, yCurrent, np.sqrt(yMaster)),2): #if smaller chi2 is found
                    chi2 = promath.chi2(yMaster, yCurrent, np.sqrt(yMaster))
                    if verbose:
                        print(f'gainCorse = {round(gainCorse,4)}, chi2={round(chi2,2)}')
                elif round(chi2,2) < round(promath.chi2(yMaster, yCurrent, np.sqrt(yMaster)),2): #if larger chi2 is found: end course corse gain loop
                    if verbose:
                        print(f'gainCorse = {round(gainCorse,4)}, chi2={round(promath.chi2(yMaster, yCurrent, np.sqrt(yMaster)),2)} *SKIPPED*')
                    break
                
            for gainFine in np.arange(gainCorse*0.99, gainCorse*1.01, 0.0002):
                yCurrent, xCurrent = np.histogram(dfCurrent*gainFine, bins=binRange)
                if round(chi2Fine,2) > round(promath.chi2(yMaster, yCurrent, np.sqrt(yMaster)),2): #if smaller chi2 is found
                    chi2Fine = promath.chi2(yMaster, yCurrent, np.sqrt(yMaster))
                    gainFinal = gainFine
                    if verbose:
                        print(f'gainFine = {round(gainFine,4)}, chi2={round(chi2Fine,2)}')

                elif round(chi2Fine*1.01,2) < round(promath.chi2(yMaster, yCurrent, np.sqrt(yMaster)),2): #if larger chi2 is found
                    if verbose:
                        print(f'gainFine = {round(gainFine,4)}, chi2={round(promath.chi2(yMaster, yCurrent, np.sqrt(yMaster)),2)} *SKIPPED*')
                    break
        if i>0:#save current gain offset
            print(f'Final value -> gain={round(gainFinal*100,2)}%, chi2={round(chi2Fine,2)}')
            if verbose:
                print('------------------------------------------------------------')

            gainOffsets[i] = gainFinal

    return gainOffsets



def dataAveraging(y_data, window):
    """
    Function takes binned data and averages the values across "window" steps, effectively smoothing the data.

    -----------------------------
    Nicholai Mauritzson
    2022-12-14
    """
    y_data_new = np.zeros(len(y_data))

    for i in range(len(y_data)):
        try:     
            y_data_new[i] = np.average(y_data[i-window:i+window+1])
        except Exception as err: 
            err

    return y_data_new

def HH_TP_FD(pathData, detector, gate, numBinsData, fitRangeData, relErrSystematic=0, fitError=False, plot=True):
    """
    Method to calculate the HH TP and FD location of the maximum transfer edges.

    Parameters.
    pathData............path to TOF sliced neutron QDC data.
    detector............name of detector, string
    gate................either 'SG' or 'LG' for short gate or long gate. Default: 'LG'
    numBinsData.........array of binrages for each energy: 1.50 - 6.25 MeV
    fitRangeData........array of start and stop values for fitting maximum transfer edge data: 1.50 - 6.25 MeV. Format: [[start, stop], [start, stop], ...]
    relErrSystematic....systematic uncertainity to add to statistic error (relative value)
    fitError............determine if statistical error should included in Gaussian fits for HH and TP methods. Default: False
    plot................determine if a plot of the final results for all three methods should be shown. Default: True

    ------------------------------
    Nicholai Mauritzson
    22-08-2022
    """
    print('-------------------------------------')
    print('HH_TP_FD() running ...')
    print(f'- detector: {detector}')
    print(f'- gate: {gate}')
    print(f'- fitError = {fitError}')
    print('-------------------------------------')
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #   Load, random subtract and determine HH, TP and FD
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    HH_data = pd.DataFrame()
    TP_data = pd.DataFrame()
    FD_data = pd.DataFrame()

    i=0 #define counter for data

    for i, E in enumerate(np.arange(2000, 6250, 250)):
        print(f'Energy = {E}')

        nQDC       = np.load(f'{pathData}/{detector}/TOF_slice/QDC/keV{E}/nQDC_{gate}.npy')
        randQDC    = np.load(f'{pathData}/{detector}/TOF_slice/QDC/keV{E}/randQDC_{gate}.npy')
        nLength    = np.load(f'{pathData}/{detector}/TOF_slice/QDC/keV{E}/nLength.npy')
        randLength = np.load(f'{pathData}/{detector}/TOF_slice/QDC/keV{E}/randLength.npy')

        nQDC = calibrateMyDetector(f'{pathData}/{detector}/Ecal/popt_{gate}.npy', nQDC)
        randQDC = calibrateMyDetector(f'{pathData}/{detector}/Ecal/popt_{gate}.npy', randQDC)

        x_neutron_QDC, y_neutron_QDC = randomTOFSubtractionQDC( nQDC,
                                                                randQDC,
                                                                nLength, 
                                                                randLength,
                                                                numBinsData[i], 
                                                                plot=False)

        #calculate HH values for data
        HHres_data = halfHeight(x_neutron_QDC, 
                                y_neutron_QDC, 
                                fitRangeData[i][0], 
                                fitRangeData[i][1], 
                                relErrSystematic=relErrSystematic['HH_sysErr'][i], 
                                y_error=fitError, 
                                plot=False)

        HHres_data['energy'] = E/1000 #add current energy in MeV

        #calculate TP values
        TPres_data = turningPoint(  x_neutron_QDC, 
                                    y_neutron_QDC, 
                                    fitRangeData[i][0], 
                                    fitRangeData[i][1], 
                                    relErrSystematic=relErrSystematic['TP_sysErr'][i], 
                                    y_error=fitError, 
                                    plot=False)
        TPres_data['energy'] = E/1000 #add current energy in MeV

        #calculate FD values
        FDres_data = firstDerivative(   x_neutron_QDC, 
                                        y_neutron_QDC, 
                                        fitRangeData[i][0], 
                                        window=5, 
                                        relErrSystematic=relErrSystematic['FD_sysErr'][i], 
                                        plot=False)

        FDres_data['energy'] = E/1000 #add current energy in MeV
        
        HH_data = pd.concat([HH_data, HHres_data])
        TP_data = pd.concat([TP_data, TPres_data])
        FD_data = pd.concat([FD_data, FDres_data])
        
    if plot:
        #plot HH, TP and FD
        plt.figure()
        plt.suptitle(detector)
        plt.subplot(3,1,1)
        plt.scatter(HH_data['energy'], HH_data['HH_loc'], color='royalblue',label='HH data', s=60, marker='o', )
        plt.legend()
        plt.ylabel('Light Yield [MeV$_{ee}$]')
        plt.ylim([0, max(HH_data['HH_loc'])*1.3])

        plt.subplot(3,1,2)
        plt.scatter(TP_data['energy'], TP_data['TP_loc'], color='royalblue', label='TP data', s=60, marker='o')
        plt.legend()
        plt.xlabel('Neutron Energy [MeV]')
        plt.ylabel('Light Yield [MeV$_{ee}$]')
        plt.ylim([0, max(TP_data['TP_loc'])*1.3])

        plt.subplot(3,1,3)
        plt.scatter(FD_data['energy'], FD_data['FD_loc'], color='royalblue', label='FD data', s=60, marker='o')
        plt.legend()
        plt.ylabel('Light Yield [MeV$_{ee}$]')
        plt.ylim([0, max(FD_data['FD_loc'])*1.3])

        plt.show()
    
    #Print errors to screen (relative %)
    print(HH_data.HH_loc_err/HH_data.HH_loc*100)
    print(TP_data.TP_loc_err/TP_data.TP_loc*100)
    print(FD_data.FD_loc_err/FD_data.FD_loc*100)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #   Save to disk
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    HH_data.to_pickle(f'/media/gheed/Seagate_Expansion_Drive1/data/lightYield_paper/{detector}/TOF_slice/HH_TP_FD/HH_data_{gate}.pkl')
    FD_data.to_pickle(f'/media/gheed/Seagate_Expansion_Drive1/data/lightYield_paper/{detector}/TOF_slice/HH_TP_FD/FD_data_{gate}.pkl')
    TP_data.to_pickle(f'/media/gheed/Seagate_Expansion_Drive1/data/lightYield_paper/{detector}/TOF_slice/HH_TP_FD/TP_data_{gate}.pkl')

def halfHeight(xData, yData, fitStart=0, fitStop=0, relErrSystematic=0, y_error=False, plot=False):
    """
    Function will fit a Gaussian to (xData, yData) from 'fitStart' to 'fitStop'. 
    Will return the x-value at 50% of Gaussian amplitude (half-height value)
    
    xData..............the x-values of the data set
    yData..............the y-values of the data set
    fitStart...........the starting value of Gaussian fit
    fitStop............the end value of Gaussian fit
    relErrSystematic...systematic uncertainity to add to statistic error (relative value)
    y_error............boolean to determine if y error should be used for fit.
    plot...............boolean to show a plot of results. Default: False

    ----------------------------
    Nicholai Mauritzson
    2022-06-10
    """
    if fitStop==0:
        print('ERROR: halfHeight() needs fitStop to be > 0')
        return 0
    
    if y_error:
        popt, pcov = promath.gaussFit(xData, yData, fitStart, fitStop, y_error=True, error=True)
    else:
        popt, pcov = promath.gaussFit(xData, yData, fitStart, fitStop, y_error=False, error=True)

    mean = np.abs(popt[1])
    std = np.abs(popt[2])
    
    mean_err = np.abs(pcov[1])
    std_err = np.abs(pcov[2])

    #Calculate full width at half maximum (FWHM)
    FWHM = 2*np.sqrt(2*np.log(2))*std

    #calculate the location of the half maximum
    HH_loc = mean + FWHM/2 

    #calculate the error the the location of the half maximum
    # HH_loc_err = np.abs(xData[0]-xData[1])*1/4 #error calculated based on binwidth
    HH_loc_err = np.sqrt(mean_err**2 + (2*np.sqrt(2*np.log(2))/2)**2 * std_err**2) * HH_loc 

    

    # print(f'mean = {mean} +/- {mean_err}')
    # print(f'std = {std} +/- {std_err}')
    # print(f'FWHM = {FWHM}')
    # print(f'HH_loc = {HH_loc}')
    # print(f'mean_err/mean = {(mean_err/mean)}')
    # print(f'std_err/std = {(std_err/std)}')
    # # print(f'Old error = {HH_loc_err1}')
    # print(f'Squares err = {HH_loc_err}')
    # print('-------------------------')



    #add systematic error to total error.
    if relErrSystematic>0:
        relErrStatistic = HH_loc_err/HH_loc
        relErrTotal = np.sqrt(relErrStatistic**2 + relErrSystematic**2)
        HH_loc_err = HH_loc*relErrTotal

        print(f'HH_loc_err = {np.round(relErrStatistic*100, 3)}%')
        print(f'sys_err = {np.round(relErrSystematic*100, 3)}%')
        print(f'total_err = {np.round(relErrTotal*100, 3)}%')
        print('--------------------------------')
        
        # print(f'HH error: stat={np.round(HH_loc_err/HH_loc*100,3)}%, syst={np.round(relErrSystematic*100,3)}%, total={np.round(relErrTotal*100,3)}%')

    #Position in y-scale of HH
    amp_loc = np.abs(popt[0]/2)

    #Error in position in y-scale of HH
    amp_loc_err = pcov[0] 

    if plot:
        plt.step(xData, yData, label='Data')
        plt.plot(np.arange(xData[0], xData[-1], 0.001), promath.gaussFunc(np.arange(xData[0], xData[-1], 0.001), popt[0], popt[1], popt[2]), label='Fit', color='black')
        plt.scatter(HH_loc, amp_loc, color='purple', s=100, marker='x')
        plt.legend()
        plt.title('Half-Height method')
        plt.show()

    res = pd.DataFrame({'HH_loc':           [HH_loc],
                        'HH_loc_err':       [HH_loc_err],
                        'HH_amp_loc':       [amp_loc],
                        'HH_amp_loc_err':   [amp_loc_err],
                        'fit_mean':         [mean],
                        'fit_mean_err':     [mean_err],
                        'fit_std':          [std],
                        'fit_std_err':      [std_err]})
    return res


def turningPoint(xData, yData, fitStart=0, fitStop=0, relErrSystematic=0, y_error=False, plot=False):
    """
    Function will fit a Gaussian to (xData, yData) from 'fitStart' to 'fitStop'. 
    Will return the x-value at the tuning point of the first derivative of the fitted Gaussian function.
    
    xData..............the x-values of the data set
    yData..............the y-values of the data set
    fitStart...........the starting value of Gaussian fit
    fitStop............the end value of Gaussian fit
    relErrSystematic...systematic uncertainity to add to statistic error (relative value)
    plot...............boolean to show a plot of results. Default: False

    ----------------------------
    Nicholai Mauritzson
    2022-08-12
    """

    if fitStop==0:
        print('ERROR: turningPoint() needs fitStop to be > 0')
        return 0

    #fitting a Gaussian to data
    if y_error:
        popt, pcov = promath.gaussFit(xData, yData, fitStart, fitStop, y_error=True, error=True)
    else:
        popt, pcov = promath.gaussFit(xData, yData, fitStart, fitStop, y_error=False, error=True)

    #calculating the gradient of Gaussian fit
    xVal = np.arange(xData[0], xData[-1], 0.001)
    dx = xVal[1]-xVal[0]
    yGradient = np.gradient(promath.gaussFunc(np.arange(xData[0], xData[-1], 0.001), popt[0], popt[1], popt[2]), dx)

    #slice out relevant data
    x, y = binDataSlicer(xVal, yGradient, fitStart, fitStop)

    #Getting TP position by scanning the gradient data backward and looking for the turningpoint
    for i in np.flip(range(len(x))): 
        if (y[i] - y[i-1]) < 0:
            break
    
    TP_loc = x[i]
    TP_loc_err = np.abs(xData[0]-xData[1])*1/4 #error calculated based on binwidth

    TP_amp_loc = y[i]
    TP_amp_loc_err = np.sqrt(np.abs(y[i]))

    #add systematic error to total error.
    if relErrSystematic>0:
        relErrStatistic = TP_loc_err/TP_loc
        relErrTotal = np.sqrt(relErrStatistic**2 + relErrSystematic**2)
        TP_loc_err = TP_loc*relErrTotal

    #save mean and std
    mean = np.abs(popt[1])
    std = np.abs(popt[2])
    
    mean_err = np.abs(pcov[1])
    std_err = np.abs(pcov[2])

    #making plot
    if plot:
        plt.step(xData, yData, label='Data')
        plt.plot(np.arange(xData[0], xData[-1], 0.001), promath.gaussFunc(np.arange(xData[0], xData[-1], 0.001), popt[0], popt[1], popt[2]), label='Fit', color='black')
        plt.plot(xVal, yGradient, label='Fit gradient', color='black', ls='dashed')
        plt.scatter(TP_loc, TP_amp_loc, label='TP', marker='x', s=80)
        plt.vlines(TP_loc, -3000,2000, lw=1)
        plt.hlines(TP_amp_loc, 0, 4, lw=1)
        plt.legend()
        plt.title('Turning-point method')
        plt.show()

    res = pd.DataFrame({'TP_loc':           [TP_loc],
                        'TP_loc_err':       [TP_loc_err],
                        'TP_amp_loc':       [TP_amp_loc],
                        'TP_amp_loc_err':   [TP_amp_loc_err],
                        'fit_mean':         [mean],
                        'fit_mean_err':     [mean_err],
                        'fit_std':          [std],
                        'fit_std_err':      [std_err]})
    return res


def firstDerivative(xData, yData, start=0, window=5, relErrSystematic=0, plot=False):
    """
    Funtion will calculate the gradrient of the data and find its minimum and location.
    This minimum will be the first deriviative point
    
    NOTE: The FD point is the *maximum* of the gradient around the proton energy-transfer edge

    xData..............the x-values of the data set
    yData..............the y-values of the data set
    start..............the starting for determining the minimum point
    window.............the number of bins to interpolate the slope across on each side. i.e. +-'window'.
    relErrSystematic...systematic uncertainity to add to statistic error (relative value)
    plot...............boolean to show a plot of results. Default: False

    ----------------------------
    Nicholai Mauritzson
    2022-11-26
    """
    
    #Calculate the first derivative (gradient) of the data
    gradientData = firstDerivativeHelper(yData, window) 
    
    #iterpolate around the minumum
    # interpolFunc = interpolate.interp1d(xData, gradientData, kind='cubic')
    interpolFunc = interpolate.splrep(xData, gradientData, s=100)

    #make finer step-size xvalues
    xFine = np.arange(start, max(xData)*0.99, 0.001)

    #find minimum position in y-data
    FDy_min = np.min(interpolate.splev(xFine, interpolFunc, der=0))

    #find index location of FDy_min
    minIdx = np.where(interpolate.splev(xFine, interpolFunc, der=0) == FDy_min)[-1][-1]

    #find x-value of minmum position
    FDx_min = xFine[minIdx]

    #calculate error in FDy_min based on statistical counts
    FDy_min_err = np.sqrt(np.abs(FDy_min)*window)

    #Calculate error for FDx_min based on bin width.
    FDx_min_err = np.abs(xData[0]-xData[1])*0.5#0.4 #error calculated based on binwidth
    
    #rename variables for saving
    FD_loc = FDx_min #xval
    amp_loc = FDy_min #yval
    FD_loc_err = FDx_min_err #xval err
    amp_loc_err = FDy_min_err #yval err

    #add systematic error to total error.
    if relErrSystematic>0:
        relErrStatistic = FD_loc_err/FD_loc
        relErrTotal = np.sqrt(relErrStatistic**2 + relErrSystematic**2)
        FD_loc_err = FD_loc*relErrTotal 

    if plot:
        plt.step(xData, yData, label='Data')
        plt.step(xData, gradientData, label='Gradient of data')
        plt.plot(xFine, interpolate.splev(xFine, interpolFunc, der=0), color='black', label='interpolation')
        plt.scatter(FD_loc, amp_loc, color='purple', label='TP', s=100, marker='x')
        plt.vlines(FD_loc, -500,2000, lw=1)
        plt.hlines(amp_loc, 0, 4, lw=1)
        plt.legend()
        plt.title('First-derivative method')
        plt.show()


    res = pd.DataFrame({'FD_loc':           [FD_loc],
                        'FD_loc_err':       [FD_loc_err],
                        'FD_amp_loc':       [amp_loc],
                        'FD_amp_loc_err':   [amp_loc_err]})
    return res


def firstDerivativeHelper(y_data, window):
    """
    Function takes binned data as calculate the first derivative.
    Linear interpolation between +-'window' around each data point is fitted and the sloped saved.

    -----------------------------
    Nicholai Mauritzson
    2022-10-25
    """
    data_slope = np.zeros(len(y_data))
    for i in range(len(y_data)):
        try:     
            popt, pcov = curve_fit( promath.linearFunc, 
                                    np.arange(2*window + 1), 
                                    y_data[i-window:i+window+1])
            data_slope[i] = popt[0]
        except Exception as err: 
            err

    return data_slope


def HH_TP_FD_fitting(pathData, detector, gate, fitError=False, plot=True):
    """
    Method to fit the HH, TP and FD location of the maximum transfer edges with two different function, Kornilov and Cecil.

    Parameters.
    pathData............path to HH, TP and FD values for data
    detector............name of detector, string
    gate................either 'SG' or 'LG' for short gate or long gate. Default: 'LG'
    fitError............determine if errors should included in curve_fit for Kornilov and Cecil functions.
    plot................determine if a plot of the final results for data and simulation should be shown. Default: True

    ------------------------------
    Nicholai Mauritzson
    22-08-2022
    """
    
    print('-------------------------------------')
    print('HH_TP_FD_fitting() running ...')
    print(f'- detector: {detector}')
    print(f'- gate: {gate}')
    print(f'- fitError = {fitError}')

    ###########################################
    ### LOEAD DATA AND SIMUILATION ############
    ###########################################
    HH_data = pd.read_pickle(f'{pathData}/{detector}/TOF_slice/HH_TP_FD/HH_data_{gate}.pkl')
    TP_data = pd.read_pickle(f'{pathData}/{detector}/TOF_slice/HH_TP_FD/TP_data_{gate}.pkl')
    FD_data = pd.read_pickle(f'{pathData}/{detector}/TOF_slice/HH_TP_FD/FD_data_{gate}.pkl')

    energy = HH_data.energy

    HH_loc = HH_data.HH_loc
    TP_loc = TP_data.TP_loc
    FD_loc = FD_data.FD_loc
    
    HH_loc_err = HH_data.HH_loc_err
    TP_loc_err = TP_data.TP_loc_err
    FD_loc_err = FD_data.FD_loc_err

    ###########################################
    ### FITTING with or without errors ########
    ###########################################   
    if detector == 'NE213A':
        if fitError:
            #Data: Kornilov equation
            
            #lmfit test
            # model = Model(promath.kornilovEq_NE213)
            # params_guess = model.make_params(L0=0.7)
            # result_HH_k = model.fit(HH_loc, params_guess, weights=1/HH_loc_err, Ep=energy)
            # result_TP_k = model.fit(TP_loc, params_guess, weights=1/TP_loc_err, Ep=energy)
            # result_FD_k = model.fit(FD_loc, params_guess, weights=1/FD_loc_err, Ep=energy)

            # popt_HH_data_k = [result_HH_k.best_values['L0']]
            # pcov_HH_data_k = np.diag(np.sqrt(result_HH_k.covar))
            # redChi2_HH_k = result_HH_k.redchi

            # popt_TP_data_k = [result_TP_k.best_values['L0']]
            # pcov_TP_data_k = np.diag(np.sqrt(result_TP_k.covar))
            # redChi2_TP_k = result_TP_k.redchi

            # popt_FD_data_k = [result_FD_k.best_values['L0']]
            # pcov_FD_data_k = np.diag(np.sqrt(result_FD_k.covar))
            # redChi2_FD_k = result_FD_k.redchi
            #end lmfit test

            
            popt_HH_data_k, pcov_HH_data_k = curve_fit(promath.kornilovEq_NE213, energy, HH_loc, sigma=HH_loc_err, absolute_sigma=False)
            pcov_HH_data_k = np.diag(np.sqrt(pcov_HH_data_k))

            popt_TP_data_k, pcov_TP_data_k = curve_fit(promath.kornilovEq_NE213, energy, TP_loc, sigma=TP_loc_err, absolute_sigma=False)
            pcov_TP_data_k = np.diag(np.sqrt(pcov_TP_data_k))

            popt_FD_data_k, pcov_FD_data_k = curve_fit(promath.kornilovEq_NE213, energy, FD_loc, sigma=FD_loc_err, absolute_sigma=False)
            pcov_FD_data_k = np.diag(np.sqrt(pcov_FD_data_k))

            redChi2_HH_k = promath.chi2red(HH_loc, promath.kornilovEq_NE213(energy, popt_HH_data_k[0]), HH_loc_err, 1)
            redChi2_TP_k = promath.chi2red(TP_loc, promath.kornilovEq_NE213(energy, popt_TP_data_k[0]), TP_loc_err, 1)
            redChi2_FD_k = promath.chi2red(FD_loc, promath.kornilovEq_NE213(energy, popt_FD_data_k[0]), FD_loc_err, 1)    

            ###########################################################################
            #Data: Cecil equation

            #lmfit test
            # model = Model(promath.cecilEq_NE213)
            # params_guess = model.make_params(C=1.0)
            # result_HH_c = model.fit(HH_loc, params_guess, weights=1/HH_loc_err, Ep=energy)
            # result_TP_c = model.fit(TP_loc, params_guess, weights=1/TP_loc_err, Ep=energy)
            # result_FD_c = model.fit(FD_loc, params_guess, weights=1/FD_loc_err, Ep=energy)

            # popt_HH_data_c = [result_HH_c.best_values['C']]
            # pcov_HH_data_c = np.diag(np.sqrt(result_HH_c.covar))
            # redChi2_HH_c = result_HH_c.redchi

            # popt_TP_data_c = [result_TP_c.best_values['C']]
            # pcov_TP_data_c = np.diag(np.sqrt(result_TP_c.covar))
            # redChi2_TP_c = result_TP_c.redchi

            # popt_FD_data_c = [result_FD_c.best_values['C']]
            # pcov_FD_data_c = np.diag(np.sqrt(result_FD_c.covar))
            # redChi2_FD_c = result_FD_c.redchi
            #end lmfit test

            popt_HH_data_c, pcov_HH_data_c = curve_fit(promath.cecilEq_NE213, energy, HH_loc, sigma=HH_loc_err, absolute_sigma=False)
            pcov_HH_data_c = np.diag(np.sqrt(pcov_HH_data_c))

            popt_TP_data_c, pcov_TP_data_c = curve_fit(promath.cecilEq_NE213, energy, TP_loc, sigma=TP_loc_err, absolute_sigma=False)
            pcov_TP_data_c = np.diag(np.sqrt(pcov_TP_data_c))

            popt_FD_data_c, pcov_FD_data_c = curve_fit(promath.cecilEq_NE213, energy, FD_loc, sigma=FD_loc_err, absolute_sigma=False)
            pcov_FD_data_c = np.diag(np.sqrt(pcov_FD_data_c))
            
            redChi2_HH_c = promath.chi2red(HH_loc, promath.cecilEq_NE213(energy, popt_HH_data_c[0]), HH_loc_err, 1)
            redChi2_TP_c = promath.chi2red(TP_loc, promath.cecilEq_NE213(energy, popt_TP_data_c[0]), TP_loc_err, 1)
            redChi2_FD_c = promath.chi2red(FD_loc, promath.cecilEq_NE213(energy, popt_FD_data_c[0]), FD_loc_err, 1)   

           ###########################################################################
            #Data: Poly2... not used
            popt_HH_data_n, pcov_HH_data_n =  [0],[0]
        
            popt_TP_data_n, pcov_TP_data_n =  [0],[0]
            
            popt_FD_data_n, pcov_FD_data_n =  [0],[0]
                        
            redChi2_HH_n = 0
            redChi2_TP_n = 0
            redChi2_FD_n = 0
            
            L1 = 3.67 #2.47#NE213
            C = [0.83, 2.82, 0.25, 0.93]

        else:
            #Data: Kornilov equation
            # model = Model(promath.kornilovEq_NE213)
            # params_guess = model.make_params(L0=0.7)
            # result_HH_k = model.fit(HH_loc, params_guess, Ep=energy)
            # result_TP_k = model.fit(TP_loc, params_guess, Ep=energy)
            # result_FD_k = model.fit(FD_loc, params_guess, Ep=energy)

            # popt_HH_data_k = [result_HH_k.best_values['L0']]
            # pcov_HH_data_k = np.diag(np.sqrt(result_HH_k.covar))
            # redChi2_HH_k = result_HH_k.redchi
            # popt_TP_data_k = [result_TP_k.best_values['L0']]
            # pcov_TP_data_k = np.diag(np.sqrt(result_TP_k.covar))
            # redChi2_TP_k = result_TP_k.redchi
            # popt_FD_data_k = [result_FD_k.best_values['L0']]
            # pcov_FD_data_k = np.diag(np.sqrt(result_FD_k.covar))
            # redChi2_FD_k = result_FD_k.redchi

            popt_HH_data_k, pcov_HH_data_k = curve_fit(promath.kornilovEq_NE213, energy, HH_loc)
            pcov_HH_data_k = np.diag(np.sqrt(pcov_HH_data_k))

            popt_TP_data_k, pcov_TP_data_k = curve_fit(promath.kornilovEq_NE213, energy, TP_loc)
            pcov_TP_data_k = np.diag(np.sqrt(pcov_TP_data_k))

            popt_FD_data_k, pcov_FD_data_k = curve_fit(promath.kornilovEq_NE213, energy, FD_loc)
            pcov_FD_data_k = np.diag(np.sqrt(pcov_FD_data_k))

            redChi2_HH_k = promath.chi2red(HH_loc, promath.kornilovEq_NE213(energy, popt_HH_data_k[0]), HH_loc_err, 1)
            redChi2_TP_k = promath.chi2red(TP_loc, promath.kornilovEq_NE213(energy, popt_TP_data_k[0]), TP_loc_err, 1)
            redChi2_FD_k = promath.chi2red(FD_loc, promath.kornilovEq_NE213(energy, popt_FD_data_k[0]), FD_loc_err, 1)    

            ###########################################################################
            #Data: Cecil equation
            # model = Model(promath.cecilEq_NE213)
            # params_guess = model.make_params(C=1.0)
            # result_HH_c = model.fit(HH_loc, params_guess, Ep=energy)
            # result_TP_c = model.fit(TP_loc, params_guess, Ep=energy)
            # result_FD_c = model.fit(FD_loc, params_guess, Ep=energy)

            # popt_HH_data_c = [result_HH_c.best_values['C']]
            # pcov_HH_data_c = np.diag(np.sqrt(result_HH_c.covar))
            # redChi2_HH_c = result_HH_c.redchi
            # popt_TP_data_c = [result_TP_c.best_values['C']]
            # pcov_TP_data_c = np.diag(np.sqrt(result_TP_c.covar))
            # redChi2_TP_c = result_TP_c.redchi
            # popt_FD_data_c = [result_FD_c.best_values['C']]
            # pcov_FD_data_c = np.diag(np.sqrt(result_FD_c.covar))
            # redChi2_FD_c = result_FD_c.redchi

            popt_HH_data_c, pcov_HH_data_c = curve_fit(promath.cecilEq_NE213, energy, HH_loc)
            pcov_HH_data_c = np.diag(np.sqrt(pcov_HH_data_c))

            popt_TP_data_c, pcov_TP_data_c = curve_fit(promath.cecilEq_NE213, energy, TP_loc)
            pcov_TP_data_c = np.diag(np.sqrt(pcov_TP_data_c))

            popt_FD_data_c, pcov_FD_data_c = curve_fit(promath.cecilEq_NE213, energy, FD_loc)
            pcov_FD_data_c = np.diag(np.sqrt(pcov_FD_data_c))

            redChi2_HH_c = promath.chi2red(HH_loc, promath.cecilEq_NE213(energy, popt_HH_data_c[0]), HH_loc_err, 1)
            redChi2_TP_c = promath.chi2red(TP_loc, promath.cecilEq_NE213(energy, popt_TP_data_c[0]), TP_loc_err, 1)
            redChi2_FD_c = promath.chi2red(FD_loc, promath.cecilEq_NE213(energy, popt_FD_data_c[0]), FD_loc_err, 1)   

           ###########################################################################
            #Data: Poly2... not used
            popt_HH_data_n, pcov_HH_data_n =  [0],[0]
        
            popt_TP_data_n, pcov_TP_data_n =  [0],[0]
            
            popt_FD_data_n, pcov_FD_data_n =  [0],[0]
                        
            redChi2_HH_n = 0
            redChi2_TP_n = 0
            redChi2_FD_n = 0

            L1 = 3.67 #2.47 #NE213
            C = [0.83, 2.82, 0.25, 0.93]

    if detector == 'EJ305':
        if fitError:
            #Data: Kornilov equation
            popt_HH_data_k, pcov_HH_data_k = curve_fit(promath.kornilovEq_EJ305, energy, HH_loc, sigma=HH_loc_err, absolute_sigma=False)
            pcov_HH_data_k = np.diag(np.sqrt(pcov_HH_data_k))

            popt_TP_data_k, pcov_TP_data_k = curve_fit(promath.kornilovEq_EJ305, energy, TP_loc, sigma=TP_loc_err, absolute_sigma=False)
            pcov_TP_data_k = np.diag(np.sqrt(pcov_TP_data_k))

            popt_FD_data_k, pcov_FD_data_k = curve_fit(promath.kornilovEq_EJ305, energy, FD_loc, sigma=FD_loc_err, absolute_sigma=False)
            pcov_FD_data_k = np.diag(np.sqrt(pcov_FD_data_k))

            redChi2_HH_k = promath.chi2red(HH_loc, promath.kornilovEq_EJ305(energy, popt_HH_data_k[0]), HH_loc_err, 1)
            redChi2_TP_k = promath.chi2red(TP_loc, promath.kornilovEq_EJ305(energy, popt_TP_data_k[0]), TP_loc_err, 1)
            redChi2_FD_k = promath.chi2red(FD_loc, promath.kornilovEq_EJ305(energy, popt_FD_data_k[0]), FD_loc_err, 1)    

            ###########################################################################
            #Data: Cecil equation
            popt_HH_data_c, pcov_HH_data_c = curve_fit(promath.cecilEq_EJ309, energy, HH_loc, sigma=HH_loc_err, absolute_sigma=False)
            pcov_HH_data_c = np.diag(np.sqrt(pcov_HH_data_c))

            popt_TP_data_c, pcov_TP_data_c = curve_fit(promath.cecilEq_EJ309, energy, TP_loc, sigma=TP_loc_err, absolute_sigma=False)
            pcov_TP_data_c = np.diag(np.sqrt(pcov_TP_data_c))

            popt_FD_data_c, pcov_FD_data_c = curve_fit(promath.cecilEq_EJ309, energy, FD_loc, sigma=FD_loc_err, absolute_sigma=False)
            pcov_FD_data_c = np.diag(np.sqrt(pcov_FD_data_c))

            redChi2_HH_c = promath.chi2red(HH_loc, promath.cecilEq_EJ309(energy, popt_HH_data_c[0]), HH_loc_err, 1)
            redChi2_TP_c = promath.chi2red(TP_loc, promath.cecilEq_EJ309(energy, popt_TP_data_c[0]), TP_loc_err, 1)
            redChi2_FD_c = promath.chi2red(FD_loc, promath.cecilEq_EJ309(energy, popt_FD_data_c[0]), FD_loc_err, 1)   

           ###########################################################################
            #Data: Poly2... not used
            popt_HH_data_n, pcov_HH_data_n =  [0],[0]
        
            popt_TP_data_n, pcov_TP_data_n =  [0],[0]
            
            popt_FD_data_n, pcov_FD_data_n =  [0],[0]
                        
            redChi2_HH_n = 0
            redChi2_TP_n = 0
            redChi2_FD_n = 0
            

            L1 = 6.55#8.5700387 #EJ305
            # C = [1.0, 8.2, 0.1, 0.88] #NE224
            C = [0.817, 2.63, 0.297, 1] #EJ309

        else:
            #Data: Kornilov equation
            popt_HH_data_k, pcov_HH_data_k = curve_fit(promath.kornilovEq_EJ305, energy, HH_loc)
            pcov_HH_data_k = np.diag(np.sqrt(pcov_HH_data_k))

            popt_TP_data_k, pcov_TP_data_k = curve_fit(promath.kornilovEq_EJ305, energy, TP_loc)
            pcov_TP_data_k = np.diag(np.sqrt(pcov_TP_data_k))

            popt_FD_data_k, pcov_FD_data_k = curve_fit(promath.kornilovEq_EJ305, energy, FD_loc)
            pcov_FD_data_k = np.diag(np.sqrt(pcov_FD_data_k))

            redChi2_HH_k = promath.chi2red(HH_loc, promath.kornilovEq_EJ305(energy, popt_HH_data_k[0]), HH_loc_err, 1)
            redChi2_TP_k = promath.chi2red(TP_loc, promath.kornilovEq_EJ305(energy, popt_TP_data_k[0]), TP_loc_err, 1)
            redChi2_FD_k = promath.chi2red(FD_loc, promath.kornilovEq_EJ305(energy, popt_FD_data_k[0]), FD_loc_err, 1)    

            ###########################################################################
            #Data: Cecil equation
            popt_HH_data_c, pcov_HH_data_c = curve_fit(promath.cecilEq_EJ309, energy, HH_loc)
            pcov_HH_data_c = np.diag(np.sqrt(pcov_HH_data_c))

            popt_TP_data_c, pcov_TP_data_c = curve_fit(promath.cecilEq_EJ309, energy, TP_loc)
            pcov_TP_data_c = np.diag(np.sqrt(pcov_TP_data_c))

            popt_FD_data_c, pcov_FD_data_c = curve_fit(promath.cecilEq_EJ309, energy, FD_loc)
            pcov_FD_data_c = np.diag(np.sqrt(pcov_FD_data_c))
            
            redChi2_HH_c = promath.chi2red(HH_loc, promath.cecilEq_EJ309(energy, popt_HH_data_c[0]), HH_loc_err, 1)
            redChi2_TP_c = promath.chi2red(TP_loc, promath.cecilEq_EJ309(energy, popt_TP_data_c[0]), TP_loc_err, 1)
            redChi2_FD_c = promath.chi2red(FD_loc, promath.cecilEq_EJ309(energy, popt_FD_data_c[0]), FD_loc_err, 1)   

           ###########################################################################
            #Data: Poly2... not used
            popt_HH_data_n, pcov_HH_data_n =  [0],[0]
        
            popt_TP_data_n, pcov_TP_data_n =  [0],[0]
            
            popt_FD_data_n, pcov_FD_data_n =  [0],[0]
                        
            redChi2_HH_n = 0
            redChi2_TP_n = 0
            redChi2_FD_n = 0
            
                        
            L1 = 6.55 #8.5700387 #EJ305
            # C = [1.0, 8.2, 0.1, 0.88] #NE224
            C = [0.817, 2.63, 0.297, 1] #EJ309

    if detector == 'EJ321P':
        if fitError:
            #Data: Kornilov equation
            popt_HH_data_k, pcov_HH_data_k = curve_fit(promath.kornilovEq_EJ321P, energy, HH_loc, sigma=HH_loc_err, absolute_sigma=False)
            pcov_HH_data_k = np.diag(np.sqrt(pcov_HH_data_k))

            popt_TP_data_k, pcov_TP_data_k = curve_fit(promath.kornilovEq_EJ321P, energy, TP_loc, sigma=TP_loc_err, absolute_sigma=False)
            pcov_TP_data_k = np.diag(np.sqrt(pcov_TP_data_k))

            popt_FD_data_k, pcov_FD_data_k = curve_fit(promath.kornilovEq_EJ321P, energy, FD_loc, sigma=FD_loc_err, absolute_sigma=False)
            pcov_FD_data_k = np.diag(np.sqrt(pcov_FD_data_k))

            redChi2_HH_k = promath.chi2red(HH_loc, promath.kornilovEq_EJ321P(energy, popt_HH_data_k[0]), HH_loc_err, 1)
            redChi2_TP_k = promath.chi2red(TP_loc, promath.kornilovEq_EJ321P(energy, popt_TP_data_k[0]), TP_loc_err, 1)
            redChi2_FD_k = promath.chi2red(FD_loc, promath.kornilovEq_EJ321P(energy, popt_FD_data_k[0]), FD_loc_err, 1)    

            ###########################################################################
            #Data: Cecil equation
            popt_HH_data_c, pcov_HH_data_c = curve_fit(promath.cecilEq_EJ321P, energy, HH_loc, sigma=HH_loc_err, absolute_sigma=False)
            pcov_HH_data_c = np.diag(np.sqrt(pcov_HH_data_c))

            popt_TP_data_c, pcov_TP_data_c = curve_fit(promath.cecilEq_EJ321P, energy, TP_loc, sigma=TP_loc_err, absolute_sigma=False)
            pcov_TP_data_c = np.diag(np.sqrt(pcov_TP_data_c))

            popt_FD_data_c, pcov_FD_data_c = curve_fit(promath.cecilEq_EJ321P, energy, FD_loc, sigma=FD_loc_err, absolute_sigma=False)
            pcov_FD_data_c = np.diag(np.sqrt(pcov_FD_data_c))
            
            redChi2_HH_c = promath.chi2red(HH_loc, promath.cecilEq_EJ321P(energy, popt_HH_data_c[0]), HH_loc_err, 1)
            redChi2_TP_c = promath.chi2red(TP_loc, promath.cecilEq_EJ321P(energy, popt_TP_data_c[0]), TP_loc_err, 1)
            redChi2_FD_c = promath.chi2red(FD_loc, promath.cecilEq_EJ321P(energy, popt_FD_data_c[0]), FD_loc_err, 1)  
            
            ###########################################################################
            #Poly2 fit
            popt_HH_data_n, pcov_HH_data_n = curve_fit(promath.poly2Eq_EJ321P, energy, HH_loc, sigma=HH_loc_err, absolute_sigma=False)
            pcov_HH_data_n = np.diag(np.sqrt(pcov_HH_data_n))

            popt_TP_data_n, pcov_TP_data_n = curve_fit(promath.poly2Eq_EJ321P, energy, TP_loc, sigma=HH_loc_err, absolute_sigma=False)
            pcov_TP_data_n = np.diag(np.sqrt(pcov_TP_data_n))

            popt_FD_data_n, pcov_FD_data_n = curve_fit(promath.poly2Eq_EJ321P, energy, FD_loc, sigma=HH_loc_err, absolute_sigma=False)
            pcov_FD_data_n = np.diag(np.sqrt(pcov_FD_data_n))
            
            redChi2_HH_n = promath.chi2red(HH_loc, promath.poly2Eq_EJ321P(energy, popt_HH_data_n[0]), HH_loc_err, 1)
            redChi2_TP_n = promath.chi2red(TP_loc, promath.poly2Eq_EJ321P(energy, popt_TP_data_n[0]), TP_loc_err, 1)
            redChi2_FD_n = promath.chi2red(FD_loc, promath.poly2Eq_EJ321P(energy, popt_FD_data_n[0]), FD_loc_err, 1)   

            L1 = 6.68 #EJ321P
            C = [0.43, 0.77, 0.26, 2.13] #EJ321P fitted
            poly = [0.03463543958969215, 0.10759911522298186]

        else:
            #Data: Kornilov equation
            popt_HH_data_k, pcov_HH_data_k = curve_fit(promath.kornilovEq_EJ321P, energy, HH_loc)
            pcov_HH_data_k = np.diag(np.sqrt(pcov_HH_data_k))

            popt_TP_data_k, pcov_TP_data_k = curve_fit(promath.kornilovEq_EJ321P, energy, TP_loc)
            pcov_TP_data_k = np.diag(np.sqrt(pcov_TP_data_k))

            popt_FD_data_k, pcov_FD_data_k = curve_fit(promath.kornilovEq_EJ321P, energy, FD_loc)
            pcov_FD_data_k = np.diag(np.sqrt(pcov_FD_data_k))

            redChi2_HH_k = promath.chi2red(HH_loc, promath.kornilovEq_EJ321P(energy, popt_HH_data_k[0]), HH_loc_err, 1)
            redChi2_TP_k = promath.chi2red(TP_loc, promath.kornilovEq_EJ321P(energy, popt_TP_data_k[0]), TP_loc_err, 1)
            redChi2_FD_k = promath.chi2red(FD_loc, promath.kornilovEq_EJ321P(energy, popt_FD_data_k[0]), FD_loc_err, 1)    

            ###########################################################################
            popt_HH_data_c, pcov_HH_data_c = curve_fit(promath.cecilEq_EJ321P, energy, HH_loc)
            pcov_HH_data_c = np.diag(np.sqrt(pcov_HH_data_c))

            popt_TP_data_c, pcov_TP_data_c = curve_fit(promath.cecilEq_EJ321P, energy, TP_loc)
            pcov_TP_data_c = np.diag(np.sqrt(pcov_TP_data_c))

            popt_FD_data_c, pcov_FD_data_c = curve_fit(promath.cecilEq_EJ321P, energy, FD_loc)
            pcov_FD_data_c = np.diag(np.sqrt(pcov_FD_data_c))
            
            redChi2_HH_c = promath.chi2red(HH_loc, promath.cecilEq_EJ321P(energy, popt_HH_data_c[0]), HH_loc_err, 1)
            redChi2_TP_c = promath.chi2red(TP_loc, promath.cecilEq_EJ321P(energy, popt_TP_data_c[0]), TP_loc_err, 1)
            redChi2_FD_c = promath.chi2red(FD_loc, promath.cecilEq_EJ321P(energy, popt_FD_data_c[0]), FD_loc_err, 1)   
            
            ###########################################################################
            #Poly2 fit
            popt_HH_data_n, pcov_HH_data_n = curve_fit(promath.poly2Eq_EJ321P, energy, HH_loc)
            pcov_HH_data_n = np.diag(np.sqrt(pcov_HH_data_n))

            popt_TP_data_n, pcov_TP_data_n = curve_fit(promath.poly2Eq_EJ321P, energy, TP_loc)
            pcov_TP_data_n = np.diag(np.sqrt(pcov_TP_data_n))

            popt_FD_data_n, pcov_FD_data_n = curve_fit(promath.poly2Eq_EJ321P, energy, FD_loc)
            pcov_FD_data_n = np.diag(np.sqrt(pcov_FD_data_n))
            
            redChi2_HH_n = promath.chi2red(HH_loc, promath.poly2Eq_EJ321P(energy, popt_HH_data_n[0]), HH_loc_err, 1)
            redChi2_TP_n = promath.chi2red(TP_loc, promath.poly2Eq_EJ321P(energy, popt_TP_data_n[0]), TP_loc_err, 1)
            redChi2_FD_n = promath.chi2red(FD_loc, promath.poly2Eq_EJ321P(energy, popt_FD_data_n[0]), FD_loc_err, 1)   

            L1 = 6.68 #EJ321P
            C = [0.43, 0.77, 0.26, 2.13] #EJ321P fitted
            poly = [0.03463543958969215, 0.10759911522298186]

    if detector == 'EJ331':
        if fitError:
            #Data: Kornilov equation
            popt_HH_data_k, pcov_HH_data_k = curve_fit(promath.kornilovEq_EJ331, energy, HH_loc, sigma=HH_loc_err, absolute_sigma=False)
            pcov_HH_data_k = np.diag(np.sqrt(pcov_HH_data_k))

            popt_TP_data_k, pcov_TP_data_k = curve_fit(promath.kornilovEq_EJ331, energy, TP_loc, sigma=TP_loc_err, absolute_sigma=False)
            pcov_TP_data_k = np.diag(np.sqrt(pcov_TP_data_k))

            popt_FD_data_k, pcov_FD_data_k = curve_fit(promath.kornilovEq_EJ331, energy, FD_loc, sigma=FD_loc_err, absolute_sigma=False)
            pcov_FD_data_k = np.diag(np.sqrt(pcov_FD_data_k))
            
            redChi2_HH_k = promath.chi2red(HH_loc, promath.kornilovEq_EJ331(energy, popt_HH_data_k[0]), HH_loc_err, 1)
            redChi2_TP_k = promath.chi2red(TP_loc, promath.kornilovEq_EJ331(energy, popt_TP_data_k[0]), TP_loc_err, 1)
            redChi2_FD_k = promath.chi2red(FD_loc, promath.kornilovEq_EJ331(energy, popt_FD_data_k[0]), FD_loc_err, 1)    

            ###########################################################################
            #Data: Cecil equation... not used
            popt_HH_data_c, pcov_HH_data_c = curve_fit(promath.cecilEq_EJ309, energy, HH_loc, sigma=HH_loc_err, absolute_sigma=False)
            pcov_HH_data_c = np.diag(np.sqrt(pcov_HH_data_c))

            popt_TP_data_c, pcov_TP_data_c = curve_fit(promath.cecilEq_EJ309, energy, TP_loc, sigma=TP_loc_err, absolute_sigma=False)
            pcov_TP_data_c = np.diag(np.sqrt(pcov_TP_data_c))

            popt_FD_data_c, pcov_FD_data_c = curve_fit(promath.cecilEq_EJ309, energy, FD_loc, sigma=FD_loc_err, absolute_sigma=False)
            pcov_FD_data_c = np.diag(np.sqrt(pcov_FD_data_c))
            
            redChi2_HH_c = promath.chi2red(HH_loc, promath.cecilEq_EJ309(energy, popt_HH_data_c[0]), HH_loc_err, 1)
            redChi2_TP_c = promath.chi2red(TP_loc, promath.cecilEq_EJ309(energy, popt_TP_data_c[0]), TP_loc_err, 1)
            redChi2_FD_c = promath.chi2red(FD_loc, promath.cecilEq_EJ309(energy, popt_FD_data_c[0]), FD_loc_err, 1)

           ###########################################################################
            #Data: Poly2... not used
            popt_HH_data_n, pcov_HH_data_n =  [0],[0]
        
            popt_TP_data_n, pcov_TP_data_n =  [0],[0]
            
            popt_FD_data_n, pcov_FD_data_n =  [0],[0]
                        
            redChi2_HH_n = 0
            redChi2_TP_n = 0
            redChi2_FD_n = 0

            L1 = 5.34#7.6669692 #EJ331
            # C = [0,0,0,0]
            C = [0.817, 2.63, 0.297, 1] #EJ309

        else:
            #Data: Kornilov equation
            popt_HH_data_k, pcov_HH_data_k = curve_fit(promath.kornilovEq_EJ331, energy, HH_loc)
            pcov_HH_data_k = np.diag(np.sqrt(pcov_HH_data_k))

            popt_TP_data_k, pcov_TP_data_k = curve_fit(promath.kornilovEq_EJ331, energy, TP_loc)
            pcov_TP_data_k = np.diag(np.sqrt(pcov_TP_data_k))

            popt_FD_data_k, pcov_FD_data_k = curve_fit(promath.kornilovEq_EJ331, energy, FD_loc)
            pcov_FD_data_k = np.diag(np.sqrt(pcov_FD_data_k))

            redChi2_HH_k = promath.chi2red(HH_loc, promath.kornilovEq_EJ331(energy, popt_HH_data_k[0]), HH_loc_err, 1)
            redChi2_TP_k = promath.chi2red(TP_loc, promath.kornilovEq_EJ331(energy, popt_TP_data_k[0]), TP_loc_err, 1)
            redChi2_FD_k = promath.chi2red(FD_loc, promath.kornilovEq_EJ331(energy, popt_FD_data_k[0]), FD_loc_err, 1)    

            ###########################################################################
            #Data: Cecil equation
            popt_HH_data_c, pcov_HH_data_c = curve_fit(promath.cecilEq_EJ309, energy, HH_loc)
            pcov_HH_data_c = np.diag(np.sqrt(pcov_HH_data_c))

            popt_TP_data_c, pcov_TP_data_c = curve_fit(promath.cecilEq_EJ309, energy, TP_loc)
            pcov_TP_data_c = np.diag(np.sqrt(pcov_TP_data_c))

            popt_FD_data_c, pcov_FD_data_c = curve_fit(promath.cecilEq_EJ309, energy, FD_loc)
            pcov_FD_data_c = np.diag(np.sqrt(pcov_FD_data_c))
            
            redChi2_HH_c = promath.chi2red(HH_loc, promath.cecilEq_EJ309(energy, popt_HH_data_c[0]), HH_loc_err, 1)
            redChi2_TP_c = promath.chi2red(TP_loc, promath.cecilEq_EJ309(energy, popt_TP_data_c[0]), TP_loc_err, 1)
            redChi2_FD_c = promath.chi2red(FD_loc, promath.cecilEq_EJ309(energy, popt_FD_data_c[0]), FD_loc_err, 1) 

           ###########################################################################
            #Data: Poly2... not used
            popt_HH_data_n, pcov_HH_data_n =  [0],[0]
        
            popt_TP_data_n, pcov_TP_data_n =  [0],[0]
            
            popt_FD_data_n, pcov_FD_data_n =  [0],[0]
                        
            redChi2_HH_n = 0
            redChi2_TP_n = 0
            redChi2_FD_n = 0

            L1 = 5.34 #7.6669692 #EJ331
            # C = [0,0,0,0]
            C = [0.817, 2.63, 0.297, 1] #EJ309

    if plot:
        ###########################################
        ### PLOTTING ##############################
        ###########################################
        xVal = np.arange(1, 6, 0.001)

        # Kornilov equation
        plt.figure()
        plt.suptitle(f'Kornilov Equation, {detector}')

        plt.subplot(3,1,1)
        plt.scatter(energy, HH_loc, label='HH data', color='royalblue', marker='o', s=5)
        plt.errorbar(energy, HH_loc, yerr=HH_loc_err, ls='none', color='royalblue')
        plt.plot(xVal, promath.kornilovEq(xVal, popt_HH_data_k[0], L1), color='royalblue', label=f'L$_0$={round(popt_HH_data_k[0],3)}')

        plt.legend()
        plt.ylabel('Light Yield [MeV$_{ee}$]')

        plt.subplot(3,1,2)
        plt.scatter(energy, TP_loc, label='TP data', color='royalblue', marker='o', s=5)
        plt.errorbar(energy, TP_loc, yerr=TP_loc_err, ls='none', color='royalblue')
        plt.plot(xVal, promath.kornilovEq(xVal, popt_TP_data_k[0], L1), color='royalblue', label=f'L$_0$={round(popt_TP_data_k[0],3)}')

        plt.legend()
        plt.ylabel('Light Yield [MeV$_{ee}$]')

        plt.subplot(3,1,3)
        plt.scatter(energy, FD_loc, label='FD data', color='royalblue', marker='o', s=5)
        plt.errorbar(energy, FD_loc, yerr=FD_loc_err, ls='none', color='royalblue')
        plt.plot(xVal, promath.kornilovEq(xVal, popt_FD_data_k[0], L1), color='royalblue', label=f'L$_0$={round(popt_FD_data_k[0],3)}')

        plt.legend()
        plt.xlabel('Neutron Energy [MeV]')
        plt.ylabel('Light Yield [MeV$_{ee}$]')

        # Cecil equation
        plt.figure()
        plt.suptitle(f'Cecil Equation, {detector}')
        plt.subplot(3,1,1)
        plt.scatter(energy, HH_loc, label='HH data', color='royalblue', marker='o', s=5)
        plt.errorbar(energy, HH_loc, yerr=HH_loc_err, ls='none', color='royalblue')
        plt.plot(xVal, promath.cecilEq(xVal, popt_HH_data_c[0], C[0], C[1], C[2], C[3]), color='royalblue', label=f'C={round(popt_HH_data_c[0],3)}')

        plt.legend()
        plt.ylabel('Light Yield [MeV$_{ee}$]')

        plt.subplot(3,1,2)
        plt.scatter(energy, TP_loc, label='TP data', color='royalblue', marker='o', s=5)
        plt.errorbar(energy, TP_loc, yerr=TP_loc_err, ls='none', color='royalblue')
        plt.plot(xVal, promath.cecilEq(xVal, popt_TP_data_c[0], C[0], C[1], C[2], C[3]), color='royalblue', label=f'C={round(popt_TP_data_c[0],3)}')

        plt.legend()
        plt.ylabel('Light Yield [MeV$_{ee}$]')

        plt.subplot(3,1,3)
        plt.scatter(energy, FD_loc, label='data', color='royalblue', marker='o', s=5)
        plt.errorbar(energy, FD_loc, yerr=FD_loc_err, ls='none', color='royalblue')
        plt.plot(xVal, promath.cecilEq(xVal, popt_FD_data_c[0], C[0], C[1], C[2], C[3]), color='royalblue', label=f'C={round(popt_FD_data_c[0],3)}')

        plt.legend()
        plt.xlabel('Neutron Energy [MeV]')
        plt.ylabel('Light Yield [MeV$_{ee}$]')
        plt.show()
        
    print('--------------------------------------------')
    print(f'- HH: L0 = {np.round(popt_HH_data_k[0], 3)}+/-{np.round(pcov_HH_data_k[0], 3)}, X2/dof = {np.round(redChi2_HH_k, 2)}')
    print(f'- TP: L0 = {np.round(popt_TP_data_k[0], 3)}+/-{np.round(pcov_TP_data_k[0], 3)}, X2/dof = {np.round(redChi2_TP_k, 2)}')
    print(f'- FD: L0 = {np.round(popt_FD_data_k[0], 3)}+/-{np.round(pcov_FD_data_k[0], 3)}, X2/dof = {np.round(redChi2_FD_k, 2)}')
    print(f'- HH: K = {np.round(popt_HH_data_c[0], 3)}+/-{np.round(pcov_HH_data_c[0], 3)}, X2/dof = {np.round(redChi2_HH_c, 2)}')
    print(f'- TP: K = {np.round(popt_TP_data_c[0], 3)}+/-{np.round(pcov_TP_data_c[0], 3)}, X2/dof = {np.round(redChi2_TP_c, 2)}')
    print(f'- FD: K = {np.round(popt_FD_data_c[0], 3)}+/-{np.round(pcov_FD_data_c[0], 3)}, X2/dof = {np.round(redChi2_FD_c, 2)}')
    print(f'- HH: N = {np.round(popt_HH_data_n[0], 3)}+/-{np.round(pcov_HH_data_n[0], 3)}, X2/dof = {np.round(redChi2_HH_n, 2)}')
    print(f'- TP: N = {np.round(popt_TP_data_n[0], 3)}+/-{np.round(pcov_TP_data_n[0], 3)}, X2/dof = {np.round(redChi2_TP_n, 2)}')
    print(f'- FD: N = {np.round(popt_FD_data_n[0], 3)}+/-{np.round(pcov_FD_data_n[0], 3)}, X2/dof = {np.round(redChi2_FD_n, 2)}')
    
    print('--------------------------------------------')


    ###########################################
    ### SAVE DATA TO DISK #####################
    ###########################################

    kornilovDict = {'method':       ['HH_data',         'TP_data',          'FD_data',          ],
                    'L0':           [popt_HH_data_k[0], popt_TP_data_k[0],  popt_FD_data_k[0]  ],  
                    'L0_err':       [pcov_HH_data_k[0], pcov_TP_data_k[0],  pcov_FD_data_k[0]  ],
                    'redchi2':      [redChi2_HH_k, redChi2_TP_k, redChi2_FD_k]} 
                    
    cecilDict = {'method':          ['HH_data',         'TP_data',          'FD_data',        ],
                    'C':            [popt_HH_data_c[0],  popt_TP_data_c[0],  popt_FD_data_c[0]],  
                    'C_err':        [pcov_HH_data_c[0],  pcov_TP_data_c[0],  pcov_FD_data_c[0]],                
                    'redchi2':      [redChi2_HH_c, redChi2_TP_c, redChi2_FD_c]}
    
    poly2Dict = {'method':          ['HH_data',         'TP_data',          'FD_data',        ],
                    'N':            [popt_HH_data_n[0],  popt_TP_data_n[0],  popt_FD_data_n[0]],  
                    'N_err':        [pcov_HH_data_n[0],  pcov_TP_data_n[0],  pcov_FD_data_n[0]],                
                    'redchi2':      [redChi2_HH_n, redChi2_TP_n, redChi2_FD_n]}
    
    df_kornilov = pd.DataFrame(kornilovDict)
    df_cecil = pd.DataFrame(cecilDict)
    df_poly2 = pd.DataFrame(poly2Dict)

    #Save to disk
    df_kornilov.to_pickle(f'{pathData}/{detector}/TOF_slice/HH_TP_FD/kornilov_{gate}_result.pkl')
    df_cecil.to_pickle(f'{pathData}/{detector}/TOF_slice/HH_TP_FD/cecil_{gate}_result.pkl')
    df_poly2.to_pickle(f'{pathData}/{detector}/TOF_slice/HH_TP_FD/poly2_{gate}_result.pkl')


def integralNorm(data, forcePositive=False):
    """
    Return the integral = 1 normalized version of 'data'.
    
    -----------------------
    Nicholai Mauritzson
    2022-08-30
    """
    #change all negative values into 0
    if forcePositive:
        data = np.where(data<0, 0, data)

    return data/np.sum(data)

def chi2Minimize(data, model, verbose=False):
    """
    Function takes a input 'model' and scales it linearly from 0.1 to 10. 
    Stops when minimal Pearson's chi2 to 'data' is found and return scaling value.
    
    ----------------------------
    Nicholai Mauritzson
    29-08-2022
    """

    #predefine chi2 value
    chi2Min = 1e15
    
    #predefine scaling value
    scaleBest = 1
   
    #loop through scaling values
    for i, scale in enumerate(np.arange(0.001, 10, 0.001)):
        chi2Current = promath.chi2(data, model*scale, np.sqrt(data))
        if chi2Current < chi2Min:
            chi2Min = chi2Current
            scaleBest = scale
            saveIt = i
    if verbose:
        print(f'best scale = {round(scaleBest,3)}, itteration: {saveIt}')

    return scaleBest
    
def scintPhotonVsEnergy(pathData, pathSim, detector, gate, numBinsList, smearFactorList, offsetsGainList, start, stop, lightYieldFactor=1, plot=True):
    """
    Method to plot and show simulated sinctillation phonton yield vs data for 2-6 MeV in 0.5 MeV bins.

    Parameters.
    pathData............path to TOF sliced neutron QDC data.
    pathSim.............path to simulated neutron data.
    detector............name of detector, string
    gate................either 'SG' or 'LG' for short gate or long gate. Default: 'LG'
    numBinsList.........list of bin arrays for each energy in order: 2, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5 and 6.0 MeV.
    smearFactorList.....list of simulation smearing values to use for each energy in order: 2, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5 and 6.0 MeV.
    offsetsGainList.....list of simulation offset valuues to use for each energy in order: 2, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5 and 6.0 MeV.
    lightYieldFactor....light yield factor to apply to simulations. Default: 1 = no change
    start...............list of starting values for range over which to calculate Chi2 in order: 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0 MeV
    stop................list of ending values for range over which to calculate Chi2 in order: 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0 MeV
    plot................determine if a plot of the final results for all three methods should be shown. Default: True
    chi2Total...........if True, will return the total Chi2 sum between simulation and data across all energies.

    ------------------------------
    Nicholai Mauritzson
    07-09-2022
    """

    print('-------------------------------------')
    print('scintPhotonVsEnergy() running ...')
    print(f'- detector: {detector}')
    print(f'- gate: {gate}')
    print('-------------------------------------')

    names = [ 'evtNum', 
            'optPhotonSum', 
            'optPhotonSumQE', 
            'optPhotonSumCompton', 
            'optPhotonSumComptonQE', 
            'optPhotonSumNPQE', 
            'xLoc', 
            'yLoc', 
            'zLoc', 
            'CsMin', 
            'optPhotonParentID', 
            'optPhotonParentCreatorProcess', 
            'optPhotonParentStartingEnergy',
            'total_edep',
            'gamma_edep',
            'proton_edep',
            'nScatters',
            'nScattersProton']

    #define list for processing
    # energies = np.empty(0, float)
    # scales = np.empty(0, float)
    # pearsonChi2 = np.empty(0, float)
    # popt = np.empty((0, 3), float)

    chi2Total = 0

    for i, energy in enumerate(np.arange(2000, 6500, 500)):

        #import simulations
        simCurrent = pd.read_csv(f"{pathSim}/{detector}/neutrons/neutron_range/{int(energy)}keV/isotropic/CSV_optphoton_data_sum.csv", names=names)
        
        #randomize binning
        y, x = np.histogram(simCurrent.optPhotonSumQE, bins=4096, range=[0, 4096]) 
        simCurrent.optPhotonSumQE = getRandDist(x, y)
        
        #calibrate simulation
        simCurrent.optPhotonSumQE = calibrateMyDetector(f'{pathData}/{detector}/Simcal/popt.npy', simCurrent.optPhotonSumQE*lightYieldFactor, inverse=False)
        
        #smear simulation
        simCurrent.optPhotonSumQE = simCurrent.optPhotonSumQE.apply(lambda x: prosim.gaussSmear(x, smearFactorList[i])) 
        
        #binning simulations and applying gain offset
        ySimCurrent, xSimCurrent = np.histogram(simCurrent.optPhotonSumQE*offsetsGainList[i], bins=numBinsList[i])
        xSimCurrent = getBinCenters(xSimCurrent)

        #import data
        nQDC       = np.load(f'{pathData}/{detector}/TOF_slice/QDC/keV{energy}/nQDC_{gate}.npy')
        randQDC    = np.load(f'{pathData}/{detector}/TOF_slice/QDC/keV{energy}/randQDC_{gate}.npy')
        nLength    = np.load(f'{pathData}/{detector}/TOF_slice/QDC/keV{energy}/nLength.npy')
        randLength = np.load(f'{pathData}/{detector}/TOF_slice/QDC/keV{energy}/randLength.npy')
        nQDC = calibrateMyDetector(f'{pathData}/{detector}/Ecal/popt_LG.npy', nQDC)
        randQDC = calibrateMyDetector(f'{pathData}/{detector}/Ecal/popt_LG.npy', randQDC)

        #random subtract data
        xDataCurrent, yDataCurrent = randomTOFSubtractionQDC(   nQDC,
                                                                randQDC,
                                                                nLength, 
                                                                randLength,
                                                                numBinsList[i], 
                                                                plot=False)
        #normalize data and simulation to 1
        yDataCurrent = integralNorm(yDataCurrent, forcePositive=True)
        ySimCurrent = integralNorm(ySimCurrent, forcePositive=True)

        #fitting data to get range for Chi2
        poptCurrent = promath.gaussFit(xDataCurrent, yDataCurrent, start[i], stop[i], y_error=False, error=False)
        
        #########################################################################
        # THIS PART WILL SLICE DATA SETS TO BE USED FOR FINAL Chi2 CALCULATIONS #
        #########################################################################
        #slicing data and simulations for Chi2 calculations from mean value of fit to mean value +3 sigma of fit.
        xDataCurrentChi2, yDataCurrentChi2 = binDataSlicer(xDataCurrent, yDataCurrent, poptCurrent[1]*1, poptCurrent[1]+3*poptCurrent[2])
        xSimCurrentChi2, ySimCurrentChi2 = binDataSlicer(xSimCurrent, ySimCurrent, poptCurrent[1]*1, poptCurrent[1]+3*poptCurrent[2])
        ###############################################################################################

        ###############################################################################################
        # THIS PART WILL SLICE DATA SETS TO BE USED FOR CALCULATING SCALING PRIOR TO Chi2 CALCULATION #
        ###############################################################################################
        #slicing data and simulation for determining optimal scaling
        xDataCurrentChi2, yDataCurrentScale = binDataSlicer(xDataCurrent, yDataCurrent, poptCurrent[1]*1.1, poptCurrent[1]*1.3)
        xSimCurrentChi2, ySimCurrentScale = binDataSlicer(xSimCurrent, ySimCurrent, poptCurrent[1]*1.1, poptCurrent[1]*1.3)
        ###############################################################################################

        #Determine optimal scaling for minimal Chi2 value
        scaleCurrent = chi2Minimize(yDataCurrentScale, ySimCurrentScale, verbose=True)

        #Calculate current Chi2 value after applying optimal scaling
        # chi2Current = difference(yDataCurrentChi2, ySimCurrentChi2 * scaleCurrent)
        # chi2Current = promath.pearsonChi2(yDataCurrentChi2, ySimCurrentChi2 * scaleCurrent)
        chi2Current = promath.chi2(yDataCurrentChi2, ySimCurrentChi2 * scaleCurrent, np.sqrt(yDataCurrentChi2))

        #calculate total chi2
        chi2Total += chi2Current

        #print to screen
        print(f'Chi2 @ {int(energy)} keV = {round(chi2Current, 3)}')
        
        #plot 
        if plot:
            if i == 0:
                plt.figure()
                plt.suptitle(f'{detector}')

            plt.subplot(9, 1, i+1)
            E = energy
            plt.step(xDataCurrent, yDataCurrent, label='Data')
            plt.step(xSimCurrent, ySimCurrent * scaleCurrent, label='Sim', lw=2)
            plt.step(np.arange(0, 4, 0.01), promath.gaussFunc(np.arange(0, 4, 0.01), poptCurrent[0], poptCurrent[1], poptCurrent[2]), color='black', label='Gauss fit', alpha=0.5)
            plt.vlines([poptCurrent[1]*1.1, poptCurrent[1]+3*poptCurrent[2]], 0, 10000, color='black', alpha=0.4,)
            plt.vlines([start[i], stop[i]], 0, 10000, color='red', alpha=0.4,)
            plt.legend()
            plt.ylim([0, np.max(yDataCurrent)*1.25])
            plt.ylabel('counts')
    if plot:
        plt.show()

    print(f'Chi2 total = {round(chi2Total, 3)}')
    return chi2Total


def simLightYield(pathData, pathSim, detector, numBinsRange, fitRange, smearFactor, edepLow=1, edepHigh=1.001, edepCol='total_edep', fitError=False, plot=False):
    """
    Function will load pencilbeam monoenergetic neutron simulations from 1-7 MeV in 0.25 MeV bins 
    and determine the location in MeVee for each energy.
    This will be done by fitting a Gaussian between 'start' and 'stop' values and taking the mean of the data set around +-3 sigma from the centroid.

    Parameters.
    
    pathData............path to calibration data
    pathSim.............path to simulated neutron data.
    detector............name of detector, string
    numBinsRange........array of binrages for each energy: 1.0 - 7.0 MeV
    fitRange............array of start and stop values for fitting maximum transfer edge data: 1.0 - 7.0 MeV. Format: [[start, stop], [start, stop], ...]
    smearFactor.........array of smearing values in percentage for each energy 1.0 - 7.0 MeV. 0 = no smearing
    fitError............determine if statistical error should included in Gaussian fits for HH and TP methods. Default: False
    plot................determine if a plot of the final results for all three methods should be shown. Default: True
    simSave.............determine if simulation results should also be save to disk. Default: True

    ------------------------------
    Nicholai Mauritzson
    23-09-2022
    """
    print('-------------------------------------')
    print('simLightYield() running ...')
    print(f'- detector: {detector}')
    print(f'- fitError = {fitError}')

    sigma = 3
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #    SIMULATION COLUMN NAMES
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    names = [ 'evtNum', 
            'optPhotonSum', 
            'optPhotonSumQE', 
            'optPhotonSumCompton', 
            'optPhotonSumComptonQE', 
            'optPhotonSumNPQE', 
            'xLoc', 
            'yLoc', 
            'zLoc', 
            'CsMin', 
            'optPhotonParentID', 
            'optPhotonParentCreatorProcess', 
            'optPhotonParentStartingEnergy',
            'total_edep',
            'gamma_edep',
            'proton_edep',
            'nScatters',
            'nScattersProton']

    #predefine columns for result DataFrame    
    result = {'energy':[], 'means':[], 'means_err':[], 'redchi2':[]}

    #Determining maximum transfer edge locations
    for i, E in enumerate(np.arange(2000, 6250, 250)):
        # print(f'Tn = {E/1000} MeV')
        neutron_sim_pen_raw = pd.read_csv(f"{pathSim}/{detector}/neutrons/neutron_range/{int(E)}keV/pencilbeam/CSV_optphoton_data_sum.csv", names=names)
        
        #find maximum edep location
        y, x = np.histogram(neutron_sim_pen_raw[edepCol], bins=np.arange(E-10, E+10, 0.3))
        x = getBinCenters(x)
        maxEdepLoc = x[np.argmax(y)]
        if edepCol == 'proton_edep':
            maxEdepLoc = E

        #apply edep cuts
        neutron_sim_pen_cut = neutron_sim_pen_raw.query(f'{maxEdepLoc*edepLow}<{edepCol}<{maxEdepLoc*edepHigh}')
        
        #randomize binning
        y, x = np.histogram(neutron_sim_pen_cut['optPhotonSumQE'], bins=4096, range=[0, 4096]) 
        neutron_sim_pen_cut['optPhotonSumQE'] = getRandDist(x, y)

        #apply smearing
        neutron_sim_pen_cut['optPhotonSumQEsmear'] = neutron_sim_pen_cut['optPhotonSumQE'].apply(lambda x: prosim.gaussSmear(x, smearFactor[i]))

        #calibrate simulations
        neutron_sim_pen_cut['optPhotonSumQEsmearcal'] = calibrateMyDetector(f'{pathData}/{detector}/Simcal/popt.npy', neutron_sim_pen_cut['optPhotonSumQEsmear'], inverse=False)
        
        #bin simulations
        y_sim, x_sim = np.histogram(neutron_sim_pen_cut['optPhotonSumQEsmearcal'], bins=numBinsRange[i])
        x_sim = getBinCenters(x_sim)
        
        #Fit gaussian
        popt, pcov = promath.gaussFit(x_sim, y_sim, fitRange[i][0], fitRange[i][1], y_error=True, error=True)
        mean = popt[1]
        std = popt[2]

        #Determine mean position
        meanPos = np.mean(neutron_sim_pen_cut.query(f'optPhotonSumQEsmearcal>{mean-sigma*std} and optPhotonSumQEsmearcal<{mean+sigma*std}')['optPhotonSumQEsmearcal'])
        
        print(f'Fitted mean/averaged mean: {np.round(mean/meanPos,2)}')
        # print(f'Averaged mean: {meanPos}')

        #Determine mean position error (statistical error)
        meanPosErr = np.std(neutron_sim_pen_cut.query(f'optPhotonSumQEsmearcal>{mean-sigma*std} and optPhotonSumQEsmearcal<{mean+sigma*std}')['optPhotonSumQEsmearcal']) / np.sqrt(len(neutron_sim_pen_cut.query(f'optPhotonSumQEsmearcal>{mean-sigma*std} and optPhotonSumQEsmearcal<{mean+sigma*std}')))

        #Adding systematic errors
        sim_birks_error = np.load(f'{pathData}/{detector}/birks/total_error.npy') #load relative error from kB optimization and smearing
     
        #Merging all errors
        totalRelError = np.sqrt((meanPosErr/meanPos)**2 + sim_birks_error[i]**2)
        meanPosErr = meanPos*totalRelError

        # print(f'Total error = {totalRelError}%')

        #Append energy, mean and mean error to dictionary
        result['energy'].append(E/1000)
        result['means'].append(meanPos)
        result['means_err'].append(meanPosErr)
        result['redchi2'].append(promath.chi2red(y_sim, promath.gaussFunc(x_sim, popt[0], popt[1], popt[2]), np.sqrt(y_sim), 3))

        if plot:
            plt.title(f'Tn = {E/1000} MeV')
            plt.step(x_sim, y_sim, label=f'mean = {meanPos}', color='black')
            plt.vlines(meanPos, ymin=0, ymax=np.max(y_sim)*1.2, lw=2, color='black', ls='dotted')
            plt.vlines([mean-sigma*std, mean+sigma*std], ymin=0, ymax=np.max(y_sim)*1.2, lw=1, color='black', ls='solid')
            plt.vlines(mean, ymin=0, ymax=np.max(y_sim)*1.2, lw=1, color='red', ls='solid')
            plt.plot(np.arange(0,E/1000, 0.001), promath.gaussFunc(np.arange(0,E/1000, 0.001), popt[0], popt[1], popt[2]), color='tomato', label='GaussFit')
            plt.vlines([fitRange[i][0], fitRange[i][1]], ymin=0, ymax=np.max(y_sim*1.1), color='black', alpha=0.25)
            plt.xlabel('Energy [MeVee]')
            plt.ylabel('Cunts')
            plt.legend()
            plt.show()
    
    #merge all results to 'finalResult' DataFrame
    finalResult = pd.DataFrame(result)

    #Save final dictionary to disk.
    finalResult.to_pickle(f'/media/gheed/Seagate_Expansion_Drive1/data/lightYield_paper/{detector}/TOF_slice/simulation/sim_LightYield.pkl')

    #Fitting with Kornilov and Cecil equations
    #renaming total absolute errors
    simErrorAbsolute = finalResult['means_err']
    
    if detector == 'NE213A':
        if fitError:
            #Kornilov equation
            popt_sim_k, pcov_sim_k = curve_fit(promath.kornilovEq_NE213, result['energy'], result['means'], sigma=simErrorAbsolute, absolute_sigma=False)
            pcov_sim_k = np.diag(np.sqrt(pcov_sim_k))
            #Cecile equation
            popt_sim_c, pcov_sim_c = curve_fit(promath.cecilEq_NE213, result['energy'], result['means'], sigma=simErrorAbsolute, absolute_sigma=False)
            pcov_sim_c = np.diag(np.sqrt(pcov_sim_c))
            #Poly2... no in use
            popt_sim_n, pcov_sim_n = [0],[0]
                        
            redChi2_k = promath.chi2red(np.array(result['means']), promath.kornilovEq_NE213(np.array(result['energy']), popt_sim_k[0]), simErrorAbsolute, 1)
            redChi2_c = promath.chi2red(np.array(result['means']), promath.cecilEq_NE213(np.array(result['energy']), popt_sim_c[0]), simErrorAbsolute, 1)
            redChi2_n = 0
            
            L1 = 3.67#2.47 #NE213
            C = [0.83, 2.82, 0.25, 0.93]

        else:
            #Kornilov equation
            popt_sim_k, pcov_sim_k = curve_fit(promath.kornilovEq_NE213, result['energy'], result['means'])
            pcov_sim_k = np.diag(np.sqrt(pcov_sim_k))
            #Cecile equation
            popt_sim_c, pcov_sim_c = curve_fit(promath.cecilEq_NE213, result['energy'], result['means'])
            pcov_sim_c = np.diag(np.sqrt(pcov_sim_c))
            #Poly2... no in use
            popt_sim_n, pcov_sim_n = [0],[0]

            # print(np.array(result['means']))
            # print(promath.kornilovEq_NE213(np.array(result['energy']), popt_sim_k[0]))
            # print(simErrorAbsolute)

            redChi2_k = promath.chi2red(np.array(result['means']), promath.kornilovEq_NE213(np.array(result['energy']), popt_sim_k[0]), simErrorAbsolute, 1)
            redChi2_c = promath.chi2red(np.array(result['means']), promath.cecilEq_NE213(np.array(result['energy']), popt_sim_c[0]), simErrorAbsolute, 1)
            redChi2_n = 0

            L1 = 3.67#2.47 #NE213
            C = [0.83, 2.82, 0.25, 0.93]

    if detector == 'EJ305':
        if fitError:
            #Kornilov equation
            popt_sim_k, pcov_sim_k = curve_fit(promath.kornilovEq_EJ305, result['energy'], result['means'], sigma=simErrorAbsolute, absolute_sigma=False)
            pcov_sim_k = np.diag(np.sqrt(pcov_sim_k))
            #Cecile equation
            popt_sim_c, pcov_sim_c = curve_fit(promath.cecilEq_EJ309, result['energy'], result['means'], sigma=simErrorAbsolute, absolute_sigma=False)
            pcov_sim_c = np.diag(np.sqrt(pcov_sim_c))
            #Poly2... no in use
            popt_sim_n, pcov_sim_n = [0],[0]

            redChi2_k = promath.chi2red(np.array(result['means']), promath.kornilovEq_EJ305(np.array(result['energy']), popt_sim_k[0]), simErrorAbsolute, 1)
            redChi2_c = promath.chi2red(np.array(result['means']), promath.cecilEq_EJ309(np.array(result['energy']), popt_sim_c[0]), simErrorAbsolute, 1)
            redChi2_n = 0
            
            L1 = 6.55#8.5700387 #EJ305
            # C = [1.0, 8.2, 0.1, 0.88] #NE224
            C = [0.817, 2.63, 0.297, 1] #EJ309
            
        else:
            #Kornilov equation
            popt_sim_k, pcov_sim_k = curve_fit(promath.kornilovEq_EJ305, result['energy'], result['means'])
            pcov_sim_k = np.diag(np.sqrt(pcov_sim_k))
            #Cecile equation
            popt_sim_c, pcov_sim_c = curve_fit(promath.cecilEq_EJ309, result['energy'], result['means'])
            pcov_sim_c = np.diag(np.sqrt(pcov_sim_c))
            #Poly2... no in use
            popt_sim_n, pcov_sim_n = [0],[0]

            redChi2_k = promath.chi2red(np.array(result['means']), promath.kornilovEq_EJ305(np.array(result['energy']), popt_sim_k[0]), simErrorAbsolute, 1)
            redChi2_c = promath.chi2red(np.array(result['means']), promath.cecilEq_EJ309(np.array(result['energy']), popt_sim_c[0]), simErrorAbsolute, 1)
            redChi2_n = 0
            
            L1 = 6.55#8.5700387 #EJ305
            # C = [1.0, 8.2, 0.1, 0.88] #NE224
            C = [0.817, 2.63, 0.297, 1] #EJ309

    if detector == 'EJ321P':
        if fitError:
            #Kornilov equation
            popt_sim_k, pcov_sim_k = curve_fit(promath.kornilovEq_EJ321P, result['energy'], result['means'], sigma=simErrorAbsolute, absolute_sigma=False)
            pcov_sim_k = np.diag(np.sqrt(pcov_sim_k))
            #Cecile equation
            popt_sim_c, pcov_sim_c = curve_fit(promath.cecilEq_EJ321P, result['energy'], result['means'], sigma=simErrorAbsolute, absolute_sigma=False)
            pcov_sim_c = np.diag(np.sqrt(pcov_sim_c))
            #Poly2... no in use
            popt_sim_n, pcov_sim_n = curve_fit(promath.poly2Eq_EJ321P, result['energy'], result['means'], sigma=simErrorAbsolute, absolute_sigma=False)
            pcov_sim_n = np.diag(np.sqrt(pcov_sim_n))

            redChi2_k = promath.chi2red(np.array(result['means']), promath.kornilovEq_EJ321P(np.array(result['energy']), popt_sim_k[0]), simErrorAbsolute, 1)
            redChi2_c = promath.chi2red(np.array(result['means']), promath.cecilEq_EJ321P(np.array(result['energy']), popt_sim_c[0]), simErrorAbsolute, 1)
            redChi2_n = promath.chi2red(np.array(result['means']), promath.poly2Eq_EJ321P(np.array(result['energy']), popt_sim_n[0]), simErrorAbsolute, 1)
            
            L1 = 6.68#6.6176005 #EJ321P
            C = [0.43, 0.77, 0.26, 2.13]
            poly = [0.03463543958969215, 0.10759911522298186]

        else:
            #Kornilov equation
            popt_sim_k, pcov_sim_k = curve_fit(promath.kornilovEq_EJ321P, result['energy'], result['means'])
            pcov_sim_k = np.diag(np.sqrt(pcov_sim_k))
            #Cecile equation
            popt_sim_c, pcov_sim_c = curve_fit(promath.cecilEq_EJ321P, result['energy'], result['means'])
            pcov_sim_c = np.diag(np.sqrt(pcov_sim_c))
            #Poly2... no in use
            popt_sim_n, pcov_sim_n = curve_fit(promath.poly2Eq_EJ321P, result['energy'], result['means'])
            pcov_sim_n = np.diag(np.sqrt(pcov_sim_n))

            redChi2_k = promath.chi2red(np.array(result['means']), promath.kornilovEq_EJ321P(np.array(result['energy']), popt_sim_k[0]), simErrorAbsolute, 1)
            redChi2_c = promath.chi2red(np.array(result['means']), promath.cecilEq_EJ321P(np.array(result['energy']), popt_sim_c[0]), simErrorAbsolute, 1)
            redChi2_n = promath.chi2red(np.array(result['means']), promath.poly2Eq_EJ321P(np.array(result['energy']), popt_sim_n[0]), simErrorAbsolute, 1)
            
            L1 = 6.68#6.6176005 #EJ321P
            C = [0.43, 0.77, 0.26, 2.13]
            poly = [0.03463543958969215, 0.10759911522298186]

    if detector == 'EJ331':
        if fitError:
            #Kornilov equation
            popt_sim_k, pcov_sim_k = curve_fit(promath.kornilovEq_EJ331, result['energy'], result['means'], sigma=simErrorAbsolute, absolute_sigma=False)
            pcov_sim_k = np.diag(np.sqrt(pcov_sim_k))
            #Cecile equation
            popt_sim_c, pcov_sim_c = curve_fit(promath.cecilEq_EJ309, result['energy'], result['means'], sigma=simErrorAbsolute, absolute_sigma=False)
            pcov_sim_c = np.diag(np.sqrt(pcov_sim_c))
            #Poly2... no in use
            popt_sim_n, pcov_sim_n = [0],[0]
            
            redChi2_k = promath.chi2red(np.array(result['means']), promath.kornilovEq_EJ331(np.array(result['energy']), popt_sim_k[0]), simErrorAbsolute, 1)
            redChi2_c = promath.chi2red(np.array(result['means']), promath.cecilEq_EJ309(np.array(result['energy']), popt_sim_c[0]), simErrorAbsolute, 1)
            redChi2_n = 0

            L1 = 5.34#7.6669692 #EJ331
            # C = [0,0,0,0]
            C = [0.817, 2.63, 0.297, 1] #EJ309

        else:
            #Kornilov equation
            popt_sim_k, pcov_sim_k = curve_fit(promath.kornilovEq_EJ331, result['energy'], result['means'])
            pcov_sim_k = np.diag(np.sqrt(pcov_sim_k))
            #Cecile equation
            popt_sim_c, pcov_sim_c = curve_fit(promath.cecilEq_EJ309, result['energy'], result['means'])
            pcov_sim_c = np.diag(np.sqrt(pcov_sim_c))
            #Poly2... no in use
            popt_sim_n, pcov_sim_n = [0],[0]
            
            
            redChi2_k = promath.chi2red(np.array(result['means']), promath.kornilovEq_EJ331(np.array(result['energy']), popt_sim_k[0]), simErrorAbsolute, 1)
            redChi2_c = promath.chi2red(np.array(result['means']), promath.cecilEq_EJ309(np.array(result['energy']), popt_sim_c[0]), simErrorAbsolute, 1)
            redChi2_n = 0

            L1 = 5.34#7.6669692 #EJ331
            # C = [0,0,0,0]
            C = [0.817, 2.63, 0.297, 1] #EJ309
    # ###############################################

    #make and save DataFrame
    kornilovDict = {'L0':           np.array([popt_sim_k[0]]),  
                    'L0_err':       np.array([pcov_sim_k[0]]),
                    'redchi2':      redChi2_k}
                    
    cecilDict = {   'C':            np.array([popt_sim_c[0]]),  
                    'C_err':        np.array([pcov_sim_c[0]]),                
                    'redchi2':      redChi2_c} 

    poly2Dict = {   'N':            np.array([popt_sim_n[0]]),  
                    'N_err':        np.array([pcov_sim_n[0]]),                
                    'redchi2':      redChi2_n} 
   
    df_kornilov = pd.DataFrame(kornilovDict)
    df_cecil = pd.DataFrame(cecilDict)
    df_poly2 = pd.DataFrame(poly2Dict)
    
    print('--------------------------------------------')
    print(f'- SMD: L0 = {np.round(popt_sim_k[0], 3)}+/-{np.round(pcov_sim_k[0], 3)}, X2/dof = {np.round(redChi2_k, 2)}')
    print(f'- SMD: K = {np.round(popt_sim_c[0], 3)}+/-{np.round(pcov_sim_c[0], 3)}, X2/dof = {np.round(redChi2_c, 2)}')
    print(f'- SMD: N = {np.round(popt_sim_n[0], 3)}+/-{np.round(pcov_sim_n[0], 3)}, X2/dof = {np.round(redChi2_n, 2)}')
    print('--------------------------------------------')


    if plot:
        plt.scatter(result['energy'], result['means'], label=f'Simulations points', color='black')
        plt.errorbar(result['energy'], result['means'], yerr=result['means_err'], color='black', linestyle='')
        plt.plot(np.arange(1, 7, 0.01), promath.kornilovEq(np.arange(1, 7, 0.01), popt_sim_k[0], L1), color='royalblue', label=f'Kornilov, chi2={np.round(kornilovDict["redchi2"],3)}')
        plt.plot(np.arange(1, 7, 0.01), promath.cecilEq(np.arange(1, 7, 0.01), popt_sim_c[0], C[0], C[1], C[2], C[3]), color='tomato', label=f'Cecil, chi2={np.round(cecilDict["redchi2"],3)}')
        plt.xlabel('Tn [MeV]')
        plt.ylabel('LY [MeVee]')
        plt.legend()
        plt.show()   
        
    #Save to disk
    df_kornilov.to_pickle(f'{pathData}/{detector}/TOF_slice/simulation/kornilov_result.pkl')
    df_cecil.to_pickle(f'{pathData}/{detector}/TOF_slice/simulation/cecil_result.pkl')
    df_poly2.to_pickle(f'{pathData}/{detector}/TOF_slice/simulation/poly2_result.pkl')
