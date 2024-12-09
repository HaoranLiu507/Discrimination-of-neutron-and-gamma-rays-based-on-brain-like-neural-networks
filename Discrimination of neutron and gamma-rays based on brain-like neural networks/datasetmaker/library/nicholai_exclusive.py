#!/usr/bin/env python3
#http://patorjk.com/software/taag/#p=display&f=ANSI%20Shadow
# --------------------------------------------------------------
# ███████╗██╗  ██╗██╗     ██╗   ██╗███████╗██╗██╗   ██╗███████╗
# ██╔════╝╚██╗██╔╝██║     ██║   ██║██╔════╝██║██║   ██║██╔════╝
# █████╗   ╚███╔╝ ██║     ██║   ██║███████╗██║██║   ██║█████╗  
# ██╔══╝   ██╔██╗ ██║     ██║   ██║╚════██║██║╚██╗ ██╔╝██╔══╝  
# ███████╗██╔╝ ██╗███████╗╚██████╔╝███████║██║ ╚████╔╝ ███████╗
# ╚══════╝╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝  ╚══════╝
#         A library of methods which works exlusively on 
#         Nicholai's laptop with external drive attached.
#       
#   Author: Nicholai Mauritzson 2019
#           nicholai.mauritzson@nuclear.lu.se
# --------------------------------------------------------------                                                         

import pandas as pd

def loadData(runNum, detector, col=None):
    """
    Method for loading a pandas data frame, based on run number and detector.
    NOTE: will only work on Nicholai computer with attached external drive.
    
    - 'runNum'....Run number to load data from.
    - 'col'.......If set, will only load certain columns. Can be a list.
    -----------------------------------------
    Nicholai Mauritzson
    2019-06-11
    """
    
    if col is None:
        return pd.read_hdf(f"/media/gheed/Seagate_Expansion_Drive1/Data/TNT/{detector}_cup/jadaq_{runNum:05}.hdf5")
    else:
        return pd.read_hdf(f"/media/gheed/Seagate_Expansion_Drive1/Data/TNT/{detector}_cup/jadaq_{runNum:05}.hdf5", usecols=col)[col]

def loadlgbk(lgbk_name):
    """
    Method for loading log file based on a 'key'

    Log books:
    - 'QDC_study'
    - 'calibration_cup'
    -----------------------------------------
    Nicholai Mauritzson
    2019-05-10
    """

    if lgbk_name == 'QDC_study':
        return pd.read_csv(f'/home/gheed/Documents/projects/TNT/analysis/analog/code_repo/analysis/NE321P_study/logs/{lgbk_name}.csv')
    elif lgbk_name == 'lgbk_main':
        return pd.read_csv(f'/home/gheed/Documents/projects/TNT/analysis/analog/code_repo/analysis/NE321P_study/logs/{lgbk_name}.csv')
    else:
        print('VALUE ERROR - loadlgbk(): lgbk_name does not match any known log book! Please try again...')
        return 0    
class metaData:
    """
    Class which creates and object containing metadata for a specfic run. 
    NOTE: - Will only work on Nicholais computer. 
          - Using hardcoded file path to log-book via loadlgbk() method in nicholai_exclusive.py.
    -------------------------------------------
    Nicholai Mauritzson
    2019-05-14
    """
    def __init__(self, runNum):
        """
        Class constructor method which reads in the logbook and extracts all meta data for 'runNum', 
        making them instances of 'self'.
        """
        lgbk = loadlgbk('lgbk_main')
        if runNum in lgbk.run.unique():
            lgbk = lgbk.query(f'run=={runNum}') #Select all meta data for run 'runNum'
        else:
            print(f'VALUE ERROR - metaData__init__(): run number {runNum} does not exist!')
            return None

        self.run            = runNum
        self.pedestal       = lgbk.pedestal.item()
        self.date           = lgbk.date.item()
        self.start_time     = lgbk.start_time.item()
        self.stop_time      = lgbk.stop_time.item()
        self.real_time      = lgbk.real_time.item()
        self.cpu_time       = lgbk.cpu_time.item()
        self.events         = lgbk.events.item()
        self.rate           = lgbk.rate.item()
        self.source         = lgbk.source.item()
        self.distance       = lgbk.distance.item()
        self.detector       = lgbk.detector.item()
        self.voltage        = lgbk.voltage.item()
        self.cfd_threshold  = lgbk.cfd_threshold.item()
        self.iped           = lgbk.iped.item()
        self.LG_ch          = lgbk.LG_ch.item()
        self.SG_ch          = lgbk.SG_ch.item()
        self.LG_time        = lgbk.LG_time.item()
        self.SG_time        = lgbk.SG_time.item()
        self.prescaler      = lgbk.prescaler.item()
        self.yap0_tdc_ch    = lgbk.yap0_tdc_ch.item()
        self.yap1_tdc_ch    = lgbk.yap1_tdc_ch.item()
        self.yap2_tdc_ch    = lgbk.yap2_tdc_ch.item()
        self.yap3_tdc_ch    = lgbk.yap3_tdc_ch.item()
        self.yap0_gflash_min= lgbk.yap0_gflash_min.item()
        self.yap0_gflash_max= lgbk.yap0_gflash_max.item()
        self.yap1_gflash_min= lgbk.yap1_gflash_min.item()
        self.yap1_gflash_max= lgbk.yap1_gflash_max.item()
        self.yap1_gflash_min= lgbk.yap2_gflash_min.item()
        self.yap2_gflash_max= lgbk.yap2_gflash_max.item()
        self.yap3_gflash_min= lgbk.yap3_gflash_min.item()
        self.yap3_gflash_max= lgbk.yap3_gflash_max.item()
        self.st_tdc_ch      = lgbk.st_tdc_ch.item()
        self.st_min         = lgbk.st_min.item()
        self.st_max         = lgbk.st_max.item()
        self.sheilding      = lgbk.sheilding.item()
        self.collimated     = lgbk.collimated.item()
        self.beamport       = lgbk.beamport.item()
        self.comment        = lgbk.comment.item()

    def getSTmin(runNum):
        lgbk = metaData(runNum)
        if lgbk.st_min=='NA':
            print(f'WARNING - getSTmin(): no st minimum found in metadata for run {runNum}.')
            min = 0
            return min
        else: 
            return lgbk.st_min

    def getSTmax(runNum):
        lgbk = metaData(runNum)
        if lgbk.st_max=='NA':
            print(f'WARNING - getSTmax(): no st maximum found in metadata for run {runNum}.')
            min = 100
            return min
        else: 
            return lgbk.st_max

def printTNTlogo():
    """
    Prints large version of TNT logo to console...
    """
    print('(((############(((############(((############(((###########')
    print('(((############(((############(((############(((###########')
    print('(((########%%%#(((########%%%#(((########%%%#(((########%%%')
    print('(((########%%%#(((########%%%#(((########%%%#(((########%%%')
    print('(((########%%%#(((########%%%#(((########%%%#(((########%%%')
    print('(((########%%%#(((########%%%#(((########%%%#(((########%%%')
    print('(((########%%%#(((########%%%#(((########%%%#(((########%%%')
    print('(((########%%%#(((########%%%#(((########%%%#(((########%%%')
    print('(((########%%%#(((########%%%#(((########%%%#(((########%%%')
    print('...........................................................')
    print('...........................................................')
    print('.......@@@@@@@@@@@.,,.@@@@.,,....@@@@.,..@@@@@@@@@@@.......')
    print('.......@@@@@@@@@@@....@@@@.......@@@@....@@@@@@@@@@@.......')
    print('...........@@@(.......@@@@@@@(...@@@@.......(@@@....,......')
    print('...........@@@(.......@@@@@@@(...@@@@.......(@@@...........')
    print('...........@@@(.......@@@@...(@@@@@@@.......(@@@...........')
    print('...........@@@(.......@@@@...(@@@@@@@.......(@@@...........')
    print('...*******,@@@(...***,@@@@.......@@@@.......(@@@,*******...')
    print('...*******,@@@(...****@@@@.......@@@@.......(@@@,*******...')
    print('*******....*******....***********....*******,...***********')
    print('*******....,,*****....***,,,,****....,*,,**,,...****,**,,,*')
    print('#######%%%%%%%%#######%%%%%%%%#######%%%%%%%%#######%%%%%%%')
    print('#######%%%#%%%########%%%%%%%########%%%#%%%%#######%%%%%%%')
    print('(((########%%%#(((########%%%#(((########%%%#(((########%%%')
    print('(((########%%%#(((########%%%#(((########%%%#(((########%%%')
    print('(((########%%%#(((########%%%#(((########%%%#(((########%%%')
    print('(((########%%%#(((########%%%#(((########%%%#(((########%%%')
    print('(((########%%%#(((########%%%#(((########%%%#(((########%%%')
    print('###########%%%############%%%############%%%############%%%')

def loadMergeDataSet(runNum, detector, col=None):
    """
    Method to read in a list of runs, and returned one merged collumn.

    'runNum'.....The run numbers for the data files to be merged (list).
    'detector'...The relevant detector used to get the data. This will determine folder name (hard coded). Must be string.
    'col'........This is the column name which should be merged across all run numbers. Default is None which will load all columns.
    ---------------------------------------------------------------------
    Nicholai Mauritzson
    Edit: 2019-05-21
    """

    print(f'##### loadMergeDataSet(): #####')
    print(f'> Loading {len(runNum)} files...')
    # runNum_lenght = len(str(runNum[0]))
    dataFrames = []#pd.DataFrame(columns=[f'{col}']) #Create list to hold all dataframes prior to merging
    for run in runNum:
        print(f'> Working on run {run}...')
        # file_path = getFilePath(run, f'{detector}', 'hdf5')
        if col is None:
            df_temp = loadData(run, detector, merged=False)
        else:
            df_temp = loadData(run, detector, merged=False, col=col)
        dataFrames.append(df_temp) #Append selected columns to one file
    return pd.concat(dataFrames) #Merge all data frames.
    # merged_dataFrames.to_hdf(f'{file_path[:(-5-len(str(runNum[0])))]}'+f'{runNum[0]}_merged.hdf5', key="w") #Save merged files as on hdf5 file. Named after first run in data set.