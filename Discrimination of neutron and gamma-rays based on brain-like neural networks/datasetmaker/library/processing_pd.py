#!/usr/bin/env python3
import sys
from os import listdir
import pandas as pd
import numpy as np
import dask.dataframe as dd
from library import processing_math as promath
from library import processing_utility as prout
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm


"""
A library of methods for data processing of pandas data frames.
"""

def load_parquet(path, exclude_col='', keep_col='', num_events=0, use_engine='pyarrow', full=False):
    """
    Method for loading parquet data to memory as pandas DataFrame. 
    Possibility to add which columns to exclude. Useful for larger data sets.

    path...........path folder containing the parque data.
    exlude_col.....string or array of string of column names to exlude from load data.
    keep_col......string or array of string of column names to include from load data. Default is empty.
    num_events.....number of events (rows) to include in from the data set (integer).
    use_engine.....specify which engine to use when loading data. Default: 'pyarrow'.
    full..........set to true if entire data set should be loaded into memory.
    """
    #get list of columns using Dask
    cols = dd.read_parquet(path, engine=use_engine).columns #get columns from ddf
    


    #set to exclude_all samples as standard.
    if exclude_col == '' and full==False:
        exclude_col = (
            'samples_ch0',
            'samples_ch1',
            'samples_ch2',
            'samples_ch3',
            'samples_ch4',
            'samples_ch5',
            'samples_ch6',
            'samples_ch7'
            )

    if (keep_col == ''):
        keep_col = []
        for col in cols:
            if not(col in exclude_col):
                keep_col.append(col)

    #load data using Dask
    ddf = dd.read_parquet(path, engine=use_engine, columns=keep_col) #load ddf with designated column to keep
    # ddf = ddf.reset_index()

    if num_events==0:
        return ddf.compute() #return dataset with all events (rows)
    elif num_events<=len(ddf):
        return ddf.loc[0:num_events-1].compute() #return num_events from index 0 to index 'num_event'
    else:
        print("ERROR: 'num_events' if greater than number of avilable events in DataFrame")        
        print(f"-> path = {path}")



def load_parquet_merge(path, runNum, exclude_col='', keep_col='', use_engine='pyarrow', full=False):
    """
    Method for loading multiple data-sets (parquet format) to memory as pandas DataFrame. 
    Will load each run data-set and concatenated into single pandas DataFrame.
    Possibility to add which columns to exclude. Useful for larger data sets.

    NOTE: Method looks for folders in cooked data folder called run#####, where ##### is the run number with leading zeroes.

    path..........path to cooked folder containing the cooked runs.
    runNum........run numbers, exluding leading zeroes, array of integers.
    exclude_col...string or array of string of column names to exlude from load data. Default all instances of samples_ch0-7
    keep_col......string or array of string of column names to include from load data. Default is empty.
    use_engine....specify which engine to use when loading data. Default: 'pyarrow'
    full..........set to true if entire data set should be loaded into memory.
    """

    #set to exclude_all samples as standard.
    if exclude_col == '' and full==False:
        exclude_col = (
            'samples_ch0',
            'samples_ch1',
            'samples_ch2',
            'samples_ch3',
            'samples_ch4',
            'samples_ch5',
            'samples_ch6',
            'samples_ch7'
            )
    #Status print out
    print(f'Loading {len(runNum)} data files.')

    #get array of column names using dask.
    try:
        cols = dd.read_parquet(f'{path}run{runNum[0]:05}', engine=use_engine).columns        
    except TypeError:
        print('TypeError: runNum must be a list or array.')
        return 0
    
    #check if columns to keep are given, if not, then use exclude
    if (keep_col == ''):
        keep_col = []
        for col in cols:
            if not(col in exclude_col):
                keep_col.append(col)
    
    dataFrames = []
    if full: #return data frame with all columns included
        for run in tqdm(runNum): #itterate through all file names.
            dataFrames.append(pd.read_parquet(f'{path}run{run:05}', engine=use_engine))
        return pd.concat(dataFrames)
    else: #return data frame without specified excluded columns
        for run in tqdm(runNum): # itterate through all file names.
            # print(f'Loading run: {path}run{run:05}')
            dataFrames.append(pd.read_parquet(f'{path}run{run:05}', engine=use_engine, columns=keep_col))
        return pd.concat(dataFrames)


def load_parquet_merge_gainadjust(path, runNum, exclude_col='', keep_col='', use_engine='pyarrow', full=False, gainOffsets=[], gainOffsetCol = 'qdc_lg_ch1'):
    """
    Method for loading multiple data-sets (parquet format) to memory as pandas DataFrame. 
    Will load each run data-set and concatenated into single pandas DataFrame.
    Possibility to add which columns to exclude. Useful for larger data sets.

    NOTE: Method looks for folders in cooked data folder called run#####, where ##### is the run number with leading zeroes.

    path............path to cooked folder containing the cooked runs.
    runNum..........run numbers, exluding leading zeroes, array of integers.
    exclude_col.....string or array of string of column names to exlude from load data. Default all instances of samples_ch0-7
    keep_col........string or array of string of column names to include from load data. Default is empty.
    use_engine......specify which engine to use when loading data. Default: 'pyarrow'
    full............set to true if entire data set should be loaded into memory.
    gainOffsets.....list of offsets to apply to QDC values
    gainOffsetCol...name of column to apply gain offsets to, Default = 'qdc_lg_ch1'
    """

    #set to exclude_all samples as standard.
    if exclude_col == '' and full==False:
        exclude_col = (
            'samples_ch0',
            'samples_ch1',
            'samples_ch2',
            'samples_ch3',
            'samples_ch4',
            'samples_ch5',
            'samples_ch6',
            'samples_ch7'
            )
    #Status print out
    print(f'Loading {len(runNum)} data files.')

    #get array of column names using dask.
    try:
        cols = dd.read_parquet(f'{path}run{runNum[0]:05}', engine=use_engine).columns        
    except TypeError:
        print('TypeError: runNum must be a list or array.')
        return 0
    
    #check if columns to keep are given, if not, then use exclude
    if (keep_col == ''):
        keep_col = []
        for col in cols:
            if not(col in exclude_col):
                keep_col.append(col)
    
    dataFrames = []
    if full: #return data frame with all columns included
        for run in tqdm(runNum): #itterate through all file names.
            dataFrames.append(pd.read_parquet(f'{path}run{run:05}', engine=use_engine))
        return pd.concat(dataFrames)
    else: #return data frame without specified excluded columns
        for i, run in enumerate(tqdm(runNum)): # itterate through all file names.
            dataFrame = pd.read_parquet(f'{path}run{run:05}', engine=use_engine, columns=keep_col)
            #gainalign QDC data
            dataFrame[gainOffsetCol] = dataFrame[gainOffsetCol]*gainOffsets[i]
            #concatenate all data to one dataframe
            dataFrames.append(dataFrame)
        return pd.concat(dataFrames)