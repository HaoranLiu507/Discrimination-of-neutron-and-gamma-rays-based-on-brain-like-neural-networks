#!/usr/bin/env python3
#http://patorjk.com/software/taag/#p=display&f=ANSI%20Shadow
# ----------------------------------------------------------
# ██╗   ██╗████████╗██╗██╗     ██╗████████╗██╗   ██╗
# ██║   ██║╚══██╔══╝██║██║     ██║╚══██╔══╝╚██╗ ██╔╝
# ██║   ██║   ██║   ██║██║     ██║   ██║    ╚████╔╝ 
# ██║   ██║   ██║   ██║██║     ██║   ██║     ╚██╔╝  
# ╚██████╔╝   ██║   ██║███████╗██║   ██║      ██║   
#  ╚═════╝    ╚═╝   ╚═╝╚══════╝╚═╝   ╚═╝      ╚═╝   
#         A library of utility methods for
#         processing and handling of data
#       
#   Author: Nicholai Mauritzson 2019
#           nicholai.mauritzson@nuclear.lu.se
# ----------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from library import processing_math as promath
from library import nicholai_plot_helper as nplt
from tqdm import tqdm #Library for progress bars

def printFormatting(title, descriptions, values, errors=None, unit=('Units missing!')):
    """
    !!!CURRENTLY A WORK IN PROGRESS!!!
    
    Method which prints information to console in a nice way.
    - 'title'..........String containing desired title of the print-out.
    - 'descritpions'...List of strings containing the descriptions of each line to be printed. description=('variable1, varible2, ...).
    - 'values'.........List of variables for each description. value=(val1, val2, ...).
    - 'errors'.........List of errors for each variable (optional). errors=(err1, err2, ...).
    - 'units'..........List of strings containing the unit of each variable. units=(unit1, unit2, ...).
    """
    numEnt = len(descriptions)
    str_len = []
    dots = []

    for i in range(numEnt):
        str_len.append(len(descriptions[i]))

    for i in range(numEnt):
        dots.append(str_len[i]*'.')
    max_dots = len(max(dots, key=len))

    print_dots=[]
    for i in range(numEnt):
        print_dots.append((max_dots-str_len[i]+5)*'.')
        
    print()#Create vertical empty space in terminal
    print('______________________________________________________________________') 
    print('<<<<< %s >>>>>'% title) #Print title
    if errors is not None:
        for i in range(numEnt):
            print('%s%s%.4f (+/-%.4f %s)'%(descriptions[i], print_dots[i], values[i], errors[i], units[i]))
            
    print('______________________________________________________________________')
    print()#Create vertical empty space in terminal
    return 0

def randomGauss(mean, sigma, numEnt):
    """
    Methods takes mean and sigma as well as number of entries and returns an array of normally distributed values around 'mean'
    """
    return np.random.normal(mean, sigma, numEnt)

def logFileStripper(file_path, runNum):
    """
    Method for opening and retriving meta data from aquadaq .log file. 
    Will return number of events, starting time, stopping time, real time, cpu time and calculate the rate (based on cpu time)   

    TODO: Get number of events by loading data (event number only!) and check shape.
    TODO: Figure out how to get live time/real time of measurement.
    ---------------------------------------------------------------------
    Nicholai Mauritzson
    Edit: 2019-05-10
    """
    # if type(runNum)==int:
    f = open(f"{file_path}Data{runNum}.log", "r") #Open aquadaq log file
    for line in f: #Start itterating through file.
        if "starting run" in line: #Find the starting time of run.
            start_time = line[1:9]
        if "processed" in line: #Find the number of events of run.
            num_events = int(line.split(" ")[6]) #Split line and get number of events as an integer.
        if "RealTime" in line: #Get real time and cpu time as well as stop time of run.
            stop_time = line[1:9] #Saveing stopping time.
            split_1 = line.split("=") #Splitting on "=" sign
            real_time = float(split_1[1].split(" ")[0]) #Splittin on space and saving variable as float.
            cpu_time = float(split_1[-1].split(" ")[0]) #Splittin on space and saving variable as float.
            f.close()
            break

    rate = round(num_events/cpu_time,2) #Calculate the rate for this run number.
    print("##### OUTPUT FROM logFileStripper() #####")
    print(f"RUN: {runNum}")
    print(f"NUMBER OF EVENTS = {num_events}")
    print(f"START = {start_time}")
    print(f"STOP = {stop_time}")
    print(f"REAL TIME = {real_time}")
    print(f"CPU TIME = {cpu_time}")
    print(f"RATE = {rate}")
    print("#########################################")
    return num_events, start_time, stop_time, real_time, cpu_time, rate 

def get_bad_pixel_color(cmap_name='magma'):
    """
    Used with plt.hist2d().
    Will create a color map ('my_cmap') to be used with:
    plt.hist(x,y,cmap=my_cmap, norm=logNorm()).
    Takes care of "bad" pixels (usually shown in white) when using logNorm() scale.

    Parameters:

    cmap_name.....name of color map from plt.cm library. string.
    """
    cmap = plt.cm.get_cmap(f'{cmap_name}')
    low_color = cmap(1) #Get color for lowest pixel value in RGB
    my_cmap = plt.cm.get_cmap(f'{cmap_name}') #create my_cmap and set the color map 
    my_cmap.set_bad(low_color[:-1]) #extract the color value and set to "bad" pixels
    return my_cmap 

def peakFinder(bins, threshold=2, distance=20, prominence=5, polarity='-', plot=False):
    """
    TODO: Include this function with the analysis script:-
            - either when setting the interpolation rage around the correct maximum (the first peak)
            - or for when looking for np.argmin() peak index value. NOTE: this value is given directly from peakFinder[0] = location first peak.
    TODO: test this function with idx=582673 run 215. Modify parameters to work with this pulse.

    Parameters:

    - bins..........A signal with peaks (array or list).
    - threshold.....Required threshold of peaks, the vertical distance to its neighbouring samples. 
                    Either a number, None, an array matching x or a 2-element sequence of the former. 
                    The first element is always interpreted as the minimal and the second, if supplied, 
                    as the maximal required threshold.
    - distance......Required minimal horizontal distance (>= 1) in samples between neighbouring peaks. ̈́
                    Smaller peaks are removed first until the condition is fulfilled for all remaining peaks.
    - prominence....Required prominence of peaks. Either a number, None, an array matching x or a 2-element sequence of the former. 
                    The first element is always interpreted as the minimal and the second, 
                    if supplied, as the maximal required prominence.
    - polarity......The polarity of the pulses. Either negative ('-') or positive ('+'). Default: '-'
    - plot..........Boolean value indicating if signal and peak positions should be plotted or not. Default: False
    """
    from scipy.signal import find_peaks  

    
    if polarity=='-': #if negative signals with negative polaririty is used.
        res = find_peaks(bins*-1, distance=distance, prominence=prominence)[0]
        if plot:
            plt.title(f"processing_utility.peakFinder(bins, thr={threshold}, dist={distance},prom={prominence},pol='{polarity}')")
            plt.plot(bins, label='input data')
            for i in res:
                plt.plot(i, bins[i], 'o', markersize=10, alpha=.8, label=f'peak finder (x={i})')
            plt.legend()
            plt.show()
        return res

    else: #if signals with positive polarity is used.
        res = find_peaks(bins, distance=distance, prominence=prominence)[0]
        if plot:
            plt.title(f'processing_utility.peakFinder(bins, threshold={threshold}, distance={distance},prominance={prominance},polarity={polarity})')
            plt.plot(bins, label='input data')
            for i in res:
                plt.plot(i, bins[i], label=f'peak finder (x={i})')
            plt.show()
        return res
