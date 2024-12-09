#!/usr/bin/env python3
#http://patorjk.com/software/taag/#p=display&f=ANSI%20Shadow
# -------------------------------------------------------------------------------------
# ██████╗ ██╗      ██████╗ ████████╗    ██╗  ██╗███████╗██╗     ██████╗ ███████╗██████╗ 
# ██╔══██╗██║     ██╔═══██╗╚══██╔══╝    ██║  ██║██╔════╝██║     ██╔══██╗██╔════╝██╔══██╗
# ██████╔╝██║     ██║   ██║   ██║       ███████║█████╗  ██║     ██████╔╝█████╗  ██████╔╝
# ██╔═══╝ ██║     ██║   ██║   ██║       ██╔══██║██╔══╝  ██║     ██╔═══╝ ██╔══╝  ██╔══██╗
# ██║     ███████╗╚██████╔╝   ██║       ██║  ██║███████╗███████╗██║     ███████╗██║  ██║
# ╚═╝     ╚══════╝ ╚═════╝    ╚═╝       ╚═╝  ╚═╝╚══════╝╚══════╝╚═╝     ╚══════╝╚═╝  ╚═╝
#         A library of plotting methods used for neutron tagging analysis
#                  Some will only work on Nicholai's computer.
#       
#   Author: Nicholai Mauritzson 2019
#           nicholai.mauritzson@nuclear.lu.se                                                                                      
# -------------------------------------------------------------------------------------

"""
Script with several plotting methods using pandas dataframes.
NOTE: Methods will only work with analog data from the AQUADAQ.

..............................
NOTE: 
1) The format of ALL plotting scripts should be: 
- f(df1, df2, df3,..., runNum, col, binRange, numBins, 
- runNum can be used to tag the titles of the plots and/or to import metadata
2) Methods should take pd.df or list as input. NOT file paths. 
3) The metods should only produce figures based on one data file. 
- If multiple figures are required from a list of runs then the user 
  needs to make their own loop, maybe using the prout.loadData() method.
4) All arguments are not nesseescary.
..............................
"""
import sys
from library import processing_math as promath
from library import processing_utility as prout
from library import nicholai_exclusive as ne
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import cos, pi, radians
from scipy import signal
from scipy.stats import chisquare
from tqdm import tqdm
from scipy.optimize import curve_fit
from matplotlib.colors import LogNorm

##### QDC BASED PLOTTING METHODS #####
def QDCselftimed(df, runNum=None, binRange=[0,4095], numBins=4096, x_label=None, main_title=None, y_scale='log'):
    """
    Method takes pandas data frame 'df' as input and produces a stacked QDC spectrum
    were on part is selftrigggered and the other the remaining events (pedestal trigger)
    """
    if runNum==None: #if runNum argument is not given.
        print('MISSING ARGUMENT - QDCselftimed(): runNum needs to be an integer of the run number of \'df\'')
        return 0
    #Import log book and extract relevant meta data.
    lgbk = ne.metaData(runNum)
    #Making histogram
    plt.hist([df.query(f'tdc_st_det0>={lgbk.st_min} and tdc_st_det0<={lgbk.st_max}').qdc_det0, df.query(f'not (tdc_st_det0>={lgbk.st_min} and tdc_st_det0<={lgbk.st_max})').qdc_det0], range=binRange, bins=numBins, stacked=True, label=['Detector triggered','Pedestal triggered'])
    if main_title==None:
        plt.title(f'Detector vs. Pedestal triggered QDC, run {runNum}')
    else:
        plt.title(f'{main_title}')
    if x_label==None:
        plt.xlabel('QDC [arb. units]')
    else:
        plt.xlabel[f'{x_label}']
    plt.legend()
    plt.yscale(f'{y_scale}')
    plt.show()

def QDCYAPplot(df, runNum=None, binRange=[0,4095], numBins=4096, x_label=None, main_title=None, y_scale='log'):
    """
    Methods takes pandas data frame as input and produces a 2x2 subplot figure based on column 'qdc_yap#' in each histogram.
    ------------------------------------------
    Nicholai Mauritzson
    2019-05-10
    """
    #YAP0
    plt.subplot(2,2,1)
    plt.hist(df.qdc_yap0, range=binRange, bins=numBins, label='YAP0')
    plt.yscale(f'{y_scale}')
    if x_label==None:
        plt.xlabel('QDC [arb. units]')
    else:
        plt.xlabel(f'{x_label}')
    if main_title==None:
        plt.suptitle('QDC spectra from each YAP')
    else:
        plt.suptitle(f'{main_title}')
    plt.ylabel('counts')
    plt.legend()
    #YAP1
    plt.subplot(2,2,2)
    plt.hist(df.qdc_yap1, range=binRange, bins=numBins, label='YAP1')
    plt.yscale(f'{y_scale}')
    if x_label==None:
        plt.xlabel('QDC [arb. units]')
    else:
        plt.xlabel(f'{x_label}')
    if main_title==None:
        plt.suptitle('QDC spectra from each YAP')
    else:
        plt.suptitle(f'{main_title}')
    plt.ylabel('counts')
    plt.legend()
    #YAP2
    plt.subplot(2,2,3)
    plt.hist(df.qdc_yap2, range=binRange, bins=numBins, label='YAP2')
    plt.yscale(f'{y_scale}')
    if x_label==None:
        plt.xlabel('QDC [arb. units]')
    else:
        plt.xlabel(f'{x_label}')
    if main_title==None:
        plt.suptitle('QDC spectra from each YAP')
    else:
        plt.suptitle(f'{main_title}')
    plt.ylabel('counts')
    plt.legend()
    #YAP3
    plt.subplot(2,2,4)
    plt.hist(df.qdc_yap3, range=binRange, bins=numBins, label='YAP3')
    plt.yscale(f'{y_scale}')
    if x_label==None:
        plt.xlabel('QDC [arb. units]')
    else:
        plt.xlabel(f'{x_label}')
    if main_title==None:
        plt.suptitle('QDC spectra from each YAP')
    else:
        plt.suptitle(f'{main_title}')
    plt.ylabel('counts')
    plt.legend()

    plt.show()

def QDCmultiPlot(df, runNum=None, col='qdc_det0', binRange=[0,4095], numBins=[4096], x_label='QDC [arb. units]', main_title='QDCYAPplot()', y_scale='log'):
    """
    Quick and dirty method for importing and making histograms of a specific column of several data files (panda dataframes). 

    -> 'df'..........This is the pandas data frame containg the data. Can be list of data frames.
    -> 'runNum'......This is the run number of the df. If 'df' is list then runNum needs to be a list, containing all the run numbers.
    -> 'col'.........This is the name of the column two make histogram of in the 'df'. Needs to be string.
    -> 'binRange'....Set the bin range of the histogram. Needs to be tuple, like binRange = (min, max)
    -> 'numBins'.....Set the binning for the histogram. Needs to be integer.
                
    ---------------------------------------------------------------------
    Nicholai Mauritzson
    Edit: 2019-05-21
    """
    if type(runNum)==int: #Check if the runNumber only has one entry.
        plt.hist(df[f"{col}"], range=binRange, bins=numBins, histtype='step', label=f'run: {RunNum}') #Plot the one histogram.
    elif type(runNum)==list: 
        for i in range(len(df)): #If runNum has more than one entry, loop through them all.
            plt.hist(df[i][f"{col}"], range=binRange, bins=numBins, histtype='step', label=f"run: {runNum[i]}") #Plot the data in a histogram.
    else:
        print('VALUE ERROR - QDCmultiPlot(): \'runNum\' needs to be an integer or list of integers')
        return None

    plt.ylabel("counts")
    plt.xlabel(f"{x_label}")
    plt.yscale(f'{y_scale}')
    plt.title(f'{main_title}')
    plt.legend()
    plt.show()

def QDCsubPlot(load_path, runNum=None, col='qdc_det0', binRange=[0,4095], numBins=[4096]):
    """
    !!! FIXME: make method conform to plotting standards of other methods !!!

    Quick and dirty method for importing and making a subplot of histograms of several data files (panda dataframes). 

    -> load_path...This is the path to were the data files are located. Needs to be string, and must end with "/"
    -> runNum......This is a list of run numbers (integers) used to find the data. Format: "Data{runNum}.hdf5".
    -> col.........This is the name of the column in the pandas dataframe. Needs to be string.
    -> binRange....Set the bin range of the histograms. Needs to be tuple, like binRange = (min, max)
    -> numBins.....Set the binning for the histograms. Needs to be integer.
                
    ---------------------------------------------------------------------
    Nicholai Mauritzson
    Edit: 2019-04-29
    """
    numDataFiles = len(runNum)
    numCols = 3
    numRows = int(np.ceil(numDataFiles/numCols)) #Calculate number of rows based on number of columns.
    lgbk = pd.read_csv("/home/gheed/Documents/projects/TNT/analysis/analog/code_repo/analysis/NE321P_study/logs/QDC_study.csv") #import log book for metadata
    for i in range(len(runNum)): #If runNum has more than one entry, loop through them all.
            df = pd.read_hdf(f"{load_path}Data{runNum[i]}.hdf5") #Import the data.
            plt.subplot(numRows, numCols, i+1)
            ph = lgbk.query(f"run=={runNum[i]}").pulsegen_ph.item()
            plt.hist(df[f"{col}"], range=binRange, bins=numBins, histtype='step', label=f"Data{runNum[i]}, {ph}mV") #Plot the data in a histogram.
            plt.ylabel("counts")
            plt.xlabel(f"{col}")
            plt.yscale('log')
            plt.legend()
    plt.show()

def QDCquickPlot(df, runNum=None, col='qdc_det0', binRange=[0,4095], numBins=4096, x_label='QDC [arb. units]', y_scale='log'):
    """
    A quick method for making QDC histograms of data.
    - 'df'.........the panda data frame containing the data to be plotted. Can be a single entry or list of dataframes.
                   If list of data frames are given then the spectra from each will be put in the same histogram.
    - 'col'........this is the name of the column in the pandas data frame 'df' which will be plotted (default is col='qdc_det0'). 
    - 'runNum'.....if supplied this will be written in the title, else it will say 'N/A' 
                   If multiple dataframes were given as 'df' then runNum can also be a list with respective run numbers.
    - 'binRange'...the range of binning for the histogram (optional).
    - 'numBins'....the number of bins to divide the data up into (optional).
    - 'x_label'....the label of the x-axis (optional)
    - 'y_scale'....set the scale of the y-axis, eg. linear, log etc. Default is logaritmic.
    -----------------------------------------
    Nicholai Mauritzsonplt.hist(df[f'{col}'], range=binRange, bins=numBins)
    2019-05-10
    """
    if runNum==None:
        runNum = 'N/A'
    if type(df)==list and (type(runNum)==None or type(runNum)==int): #If df is a list but runNum is not. 
        runNum = []
        for i in range(len(df)):
            runNum.append('N/A')
    if type(df)==list: #if multiple data frames were given.
        for i in range(len(df)):
            plt.hist(df[i][f'{col}'], range=binRange, bins=numBins, label=f'Run: {runNum[i]}: {col}')
    else: #if only one data frame was given.
        plt.hist(df[f'{col}'], range=binRange, bins=numBins, label=f'Run: {runNum}: {col}')
    plt.title('QDC spectrum')
    plt.xlabel(f'{x_label}')
    plt.ylabel('counts')
    plt.yscale(f'{y_scale}')
    plt.legend()
    plt.show()

##### TDC BASED PLOTTING #####
def TDCYAPplot(df, runNum=None, binRange=[0,4095], numBins=4096, x_label='TDC [arb. units]', main_title='TDC spectra from each YAP', y_scale='log', st=False):
    """
    Methods takes pandas data frame as input and produces a 2x2 subplot figure based on column 'tdc_det0_yap#' in each histogram.
    
    'df'...........This is the pandas data frame containing the data.    
    'runNum'.......This is the run number to be studies (integer)
    'binRange'.....This is the range of bins which should be included.
    'numBins'......This is the number of bins to plot.
    'x_label'......This is the label for the x-axis of each figure. Default is 'TDC [arb. units]'.
    'main_title'...This is the main title above all four figures. Default is no title.
    'y_scale'......This is the scale to use for the y axis of each subplot. Default is logarithmic.
    'st'...........True means only events which were selftrigered by the detector is shown. Default: False
                   NOTE: if no logbook is available for run 'st' should be set to False.
    ------------------------------------------
    Nicholai Mauritzson
    2019-05-14
    """
    if st!=True and st!=False:
        print('VALUE ERROR - TDCYAPplot(): argument st must be either True or False.')
        return 0
    if runNum==None:
        print('VALUE ERROR - TDCYAPplot(): must supplie a valid run number.')
        return 0

    #YAP0
    plt.subplot(2,2,1)
    if st==True: #if self triggered bit should be used.
        lgbk = ne.metaData(runNum) #Get log book for perticular run number.
        plt.hist(df.query(f'tdc_st_det0>={lgbk.st_min} and tdc_st_det0<={lgbk.st_max}').tdc_det0_yap0, range=binRange, bins=numBins, label='YAP0')
    elif st==False: #If self triggered bit should NOT be used.
        plt.hist(df.tdc_det0_yap0, range=binRange, bins=numBins, label='YAP0')
    plt.yscale(f'{y_scale}')
    plt.xlabel(f'{x_label}')
    plt.ylabel('counts')
    plt.legend()
    #YAP1
    plt.subplot(2,2,2)
    if st==True: #if self triggered bit should be used.
        plt.hist(df.query(f'tdc_st_det0>={lgbk.st_min} and tdc_st_det0<={lgbk.st_max}').tdc_det0_yap1, range=binRange, bins=numBins, label='YAP1')
    elif st==False: #If self triggered bit should NOT be used.
        plt.hist(df.tdc_det0_yap1, range=binRange, bins=numBins, label='YAP1')
    plt.yscale(f'{y_scale}')
    plt.xlabel(f'{x_label}')
    plt.ylabel('counts')
    plt.legend()
    #YAP2
    plt.subplot(2,2,3)
    if st==True: #if self triggered bit should be used.
        plt.hist(df.query(f'tdc_st_det0>={lgbk.st_min} and tdc_st_det0<={lgbk.st_max}').tdc_det0_yap2, range=binRange, bins=numBins, label='YAP2')
    elif st==False: #If self triggered bit should NOT be used.
        plt.hist(df.tdc_det0_yap2, range=binRange, bins=numBins, label='YAP2')
    plt.yscale(f'{y_scale}')
    plt.xlabel(f'{x_label}')
    plt.ylabel('counts')
    plt.legend()
    #YAP3
    plt.subplot(2,2,4)
    if st==True: #if self triggered bit should be used.
        plt.hist(df.query(f'tdc_st_det0>={lgbk.st_min} and tdc_st_det0<={lgbk.st_max}').tdc_det0_yap3, range=binRange, bins=numBins, label='YAP3')
    elif st==False: #If self triggered bit should NOT be used.
        plt.hist(df.tdc_det0_yap3, range=binRange, bins=numBins, label='YAP3')
    plt.yscale(f'{y_scale}')
    plt.xlabel(f'{x_label}')
    plt.ylabel('counts')
    plt.legend()

    plt.suptitle(f'Run: {runNum}, '+f'{main_title}')
    plt.show()


##### OTHER PLOTTING ######
def TOFvsQDC(df, runNum, timeCal=True):
    """
    Quick method for looking at TOF vs QDC as a 2D histogram.
    """
    if timeCal==True:
        plt.hist2d(df.tdc_det0_yap0*1e9, df.qdc_det0, range=[[.1,100],[80,4000]], bins=[50,1000], norm=LogNorm())
    else:
        plt.hist2d(df.tdc_det0_yap0, df.qdc_det0, range=[[80,4000],[80,4000]], bins=[1000,1000], norm=LogNorm())
    plt.ylabel('QDC')
    plt.xlabel('TOF [ns]')
    plt.title(f'run: {runNum}')
    plt.colorbar()
    plt.show()

def TOFEnergyCuts(df, runName, distance):
    """
    WIP: QDC subplots of TOF energy cuts, 0-7 MeV
    """
    plt.subplot(4,2,1)
    plt.hist(prout.tofEnergyCut(df, [0,1], 1.306), range=(0,4000), bins=4000)
    plt.yscale('log')
    plt.subplot(4,2,2)
    plt.hist(prout.tofEnergyCut(df, [1,2], 1.306), range=(0,4000), bins=4000)
    plt.yscale('log')
    plt.subplot(4,2,3)
    plt.hist(prout.tofEnergyCut(df, [2,3], 1.306), range=(0,4000), bins=4000)
    plt.yscale('log')
    plt.subplot(4,2,4)
    plt.hist(prout.tofEnergyCut(df, [3,4], 1.306), range=(0,4000), bins=4000)
    plt.yscale('log')
    plt.subplot(4,2,5)
    plt.hist(prout.tofEnergyCut(df, [4,5], 1.306), range=(0,4000), bins=4000)
    plt.yscale('log')
    plt.subplot(4,2,6)
    plt.hist(prout.tofEnergyCut(df, [5,6], 1.306), range=(0,4000), bins=4000)
    plt.yscale('log')
    plt.subplot(4,2,7)
    plt.hist(prout.tofEnergyCut(df, [7,8], 1.306), range=(0,4000), bins=4000)
    plt.yscale('log')

    plt.show()

def ratioPlot(df1, df2, col, binRange=[0,4095], numBins=4096, x_label=None, main_title=None, y_scale='log', legend_labels=['Data1', 'Data2']):
    """
    Method for comparing two pandas data frame columns of data and calculating their relative ratio.
    Makes a subfigure plot with ratio below and the two histograms above in the same frame.

    TODO: Add error bars to histogram from sqrt(N) for all bins.
    """
    # import seaborn as sns
    # sns.set(color_codes = True) #make 'em more pretty...
    fig = plt.figure()
    ax1 = fig.add_axes([0.1, 0.35, 0.8, 0.6], xticklabels=[], xlim=binRange, yscale=f'{y_scale}') #create upper figure (histogram)
    ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.25], xlim=binRange) #Create lower figure (ratio plot)
    data1 = ax1.hist(df1.qdc_det0, bins=numBins, histtype='step', linewidth=2, range=binRange, label=legend_labels[0]) #Get binned data for ratio calculations
    data2 = ax1.hist(df2.qdc_det0, bins=numBins, histtype='step', linewidth=2, range=binRange, label=legend_labels[1]) #Get binned data for ratio calculations
    ax2.plot(promath.ratioCalculator(data1[0], data2[0]), label=f'Ratio: {legend_labels[0]}/{legend_labels[1]}') #Calulate and plot ratio between the two data sets.
    ax1.legend() #Show legend
    ax2.legend() #Show legend
    ax1.set_ylabel('counts') 
    ax2.set_ylabel('counts')
    if x_label == None:
        ax2.set_xlabel(f'{col}')
    else:
        ax2.set_xlabel(f'{x_label}')
    if main_title == None:
        plt.suptitle(f'Ratio plot: {col}')
    else:
        plt.suptitle(f'{main_title}')
    plt.show()

def PSDquickPlot(df, runNum=None, a=0, b=0):
    """
    a and b are optional parameters be used for linearisation of the gamma and neutron shapes shapes.
    -----------------------------------------
    Nicholai Mauritzson
    2019-05-10
    """
    plt.hexbin(df.qdc_det0, (1-(df.qdc_sg_det0 + a)/(df.qdc_det0 + b)), extent=(0, 4095, -1, 1), norm=LogNorm())
    plt.colorbar()
    plt.xlabel('LG QDC [arb. units]')
    plt.ylabel('PS')
    plt.title(f'PSD run {runNum}')
    plt.show()

def QDCcalibrationPlot(E_data, QDC_data, QDC_err, ped=0):
    """
    Method for plotting calibration points and errors for QDC energy calibration.
    --------------------------------------
    Nicholai Mauritzson
    2019-08-28
    """
    #FITTING QDC VS ENERGY
    print(' -------------------------------')
    print('Fitting QDC vs. ENERGY')
    popt, pcov = curve_fit(promath.linearFunc, E_data, QDC_data, sigma=1/QDC_err**2, absolute_sigma=False) #Make a linear error weighted fit.
    xvalues = np.linspace(0,5) #Energy, MeVee
    yvalues = promath.linearFunc(xvalues, popt[0], popt[1]) #QDC values
    fitError = np.sqrt(np.diag(pcov))
    print(f'fitError = {fitError}')

    error = []
    for i in range(len(xvalues)): #Calculation of error propagation, linear fit
        constError = promath.errorPropConst(xvalues[i], fitError[0])
        error.append(promath.errorPropAdd([constError, fitError[1]]))

    print('Missing pedestal by: %.4f'%(popt[1]-ped)) #Print the difference from pedestal position.

    plt.figure(0)
    plt.title('QDC vs. Energy')
    plt.scatter(0, ped, label='Zero position (not fitted)', marker='x', color='black')
    plt.scatter(E_data, QDC_data, color='purple', lw=3, label='Data')
    plt.plot(xvalues, yvalues, 'k--',label=f'Wighted fit: y={round(popt[0],2)}x+{round(popt[1],2)}')
    plt.fill_between(xvalues, promath.linearFunc(xvalues, popt[0], popt[1])+error, promath.linearFunc(xvalues, popt[0], popt[1])-error, color='green', alpha=0.45, label='error')
    plt.xlabel('Energy [MeV$_{ee}$]')
    plt.ylabel('QDC [arb. units]')
    plt.grid(which='both')
    plt.legend()

    # FITTING ENERGY VS QDC
    #::::::::::::::::::::::::::::::::::::::
    print(' -------------------------------')
    print('Fitting ENERGY vs. QDC')
    popt, pcov = curve_fit(promath.linearFunc, QDC_data, E_data) #Make a linear error weighted fit.
    xvalues = np.linspace(0, np.max(QDC_data)*1.1) #QDC values
    yvalues = promath.linearFunc(xvalues, popt[0], popt[1]) #Energy, MeVee
    fitError = np.sqrt(np.diag(pcov))
    print(f'fitError = {fitError}')

    error = []
    for i in range(len(xvalues)): #Calculation of error propagation, linear fit
        constError = promath.errorPropConst(xvalues[i], fitError[0])
        error.append(promath.errorPropAdd([constError, fitError[1]]))

    print('Missing pedestal by: %.4f'%(popt[1]-ped)) #Print the difference from pedestal position.

    plt.figure(1)
    plt.title('Energy vs. QDC')
    plt.scatter(0, ped, label='Zero position (not fitted)', marker='x', color='black')
    plt.scatter(QDC_data, E_data, color='purple', lw=3, label='Data')
    plt.plot(xvalues, yvalues, 'k--', label=f'Wighted fit: y={round(popt[0],8)}x+{round(popt[1], 8)}')
    plt.fill_between(xvalues, promath.linearFunc(xvalues, popt[0], popt[1])+error, promath.linearFunc(xvalues, popt[0], popt[1])-error, color='green', alpha=0.45, label='error')
    plt.ylabel('Energy [MeV$_{ee}$]')
    plt.xlabel('QDC [arb. units]')
    plt.grid(which='both')
    plt.legend()

    plt.show()


def TOFenergyPlot(hist, bins, runNum=None):
    """
    Method for plotting TOF as energy histogram. Inputs should be from nicholai_utility.tofEnergyCal().
    """
    plt.plot(bins, hist, ls='steps', linewidth=2)
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('Energy [MeV]')
    plt.ylabel('counts')
    plt.title('Time-of-flight energy spectrum in MeV (non-relativistic)')
    plt.show()

def tof_slicer_plot(df_sliced, scale='QDC', numBins=200, binRange=(0,15000)):
    """
    Routine to plot output from processing_data.tof_slicer().

    scale......parameter (string) to set what scale is used on the x-axis. Either QDC or MeVee. Default: MeVee
    """
    #get name of sliced energy columns
    cols = df_sliced.columns 
    
    #check what x-scale is used
    if scale=='MeVee':
        binRange=(0,7)

    #get number of energy slices in df_sliced
    num = len(cols) 
    
    subidx = 1 #index for subplots
    for col in cols:
        plt.subplot(num,1,subidx)
        N = len(df_sliced[f'{col}'].dropna()) #get number of events for current energy slice.
        plt.hist(df_sliced[f'{col}'], bins=numBins, range=binRange, histtype='step', lw=2, label=f'E={col}, N={N}')
        plt.legend()
        # plt.ylabel('counts')
        # plt.yscale('log')
        if col==cols[-1]:
            plt.xlabel('QDC [arb. units]')

        subidx += 1 #increase subplot index by 1.
    plt.show()