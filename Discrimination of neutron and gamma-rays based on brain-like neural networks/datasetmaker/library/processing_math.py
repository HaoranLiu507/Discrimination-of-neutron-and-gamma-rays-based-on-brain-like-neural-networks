#!/usr/bin/env python3
#http://patorjk.com/software/taag/#p=display&f=ANSI%20Shadow
# ----------------------------------------------------------
#          ███╗   ███╗ █████╗ ████████╗██╗  ██╗
#          ████╗ ████║██╔══██╗╚══██╔══╝██║  ██║
#          ██╔████╔██║███████║   ██║   ███████║
#          ██║╚██╔╝██║██╔══██║   ██║   ██╔══██║
#          ██║ ╚═╝ ██║██║  ██║   ██║   ██║  ██║
#          ╚═╝     ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝
#  A library of methods for regular mathematical operations.
#       Some methods are based on pandas DataFrames
#       
#   Author: Nicholai Mauritzson 2018-2019
#           nicholai.mauritzson@nuclear.lu.se
# ----------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from library import processing_utility as prout
from library import processing_data as prodata
from math import exp
from scipy.stats import moyal
# from uncertainties import ufloat
# from uncertainties.umath import * #Get all methods for library
from math import isnan
from tqdm import tqdm #Library for progress bars

def simpleIntegrate(df, col=None, start=0, stop=None): #WORK IN PROGRESS
    """
        'col' is the column name containing the data to be integrated.
        Option: if no index value is given to 'stop'. Integration will go until 
                last index value of column.
        ---------------------------------------------------------------------
        Nicholai Mauritzson
        Edit: 2018-12-12
    """
    if stop==None:
        stop = len(df[col])

    return df.loc[start:stop][col].sum()

def resCalc(mean, sigma):
    """
    Takes a mean 'mean' and standard diviation 'sigma' of a distribution as input.
    Returns the resolution (decimal form) at 'mean' value.
    NOTE: method assumes a Gaussian distribution.
    """
    return (sigma*2*np.sqrt(2*np.log(2)))/mean

def gaussFit(x_input=None, y_input=None, start=0, stop=None, y_error=True, error=True):
    """
        1) Takes 'x_input' and 'y_input' values as lists, 'start' and 'stop' define the range of the fit in terms of 'x'.
        2) Fits the selected intervall with a Gaussian function 'gaussFunc()'.
        3) Returns the constant ('const'), mean value ('mean') and the standard diviation ('sigma') of the fit.
        4) Optional: If no values for start and stop are given then the default is to try and fit the entire spectrum.

        Parameters:
        - x_input.......x-values for fit
        - y_input.......y-values for fit
        - start.........An optional start value for fit using x-values. Default: 0
        - stop..........An optional stop valyes for fit using x-values. Default: Use entire x-range.
        - y_error.......If true the error for y-values it taken as sqrt(y-values). Else no error is used.
        - error.........A boolean value. When set to True, will return error of fitted parameters. When False, will only return fitted values.

        Returns:
        popt - Containing the fitted variables (np.array). Format: popt = [max, mean, std]
        np.sqrt(np.diag(pcov)) - Containing the errors for the fitted variabled (np.array) (If 'error' is True)
        ---------------------------------------------------------------------
        Nicholai Mauritzson
        Edit: 2022-04-26
    """
    if stop == None:
        stop = np.max(x_input)
    
    x, y = prodata.binDataSlicer(x_input, y_input, start, stop)

    meanTEMP = np.mean(x) #Calculate the mean
    sigmaTEMP = np.std(x) #Calculate the standard deviation.
    try:
        if y_error:
            popt, pcov = curve_fit(gaussFunc, x, y, p0 = [max(y), meanTEMP, sigmaTEMP], sigma=np.sqrt(y), absolute_sigma=True) #Fit the data
        else:
            popt, pcov = curve_fit(gaussFunc, x, y, p0 = [max(y), meanTEMP, sigmaTEMP]) #Fit the data
    except RuntimeError:
        print('gaussFit(): RuntimeError')
        popt = [1,1,1]
        pcov = [1,1,1]
    
    # Check if error should be included in return.
    if error:
        return popt, np.sqrt(np.diag(pcov)) #popt = [const, mean, sigma]
    else:
        return popt #popt = [const, mean, sigma]

def gaussFunc(x, a, x0, sigma):
    """
        A generic Gaussian function.
        - 'x'......This is the independent variable data (list format).
        - 'a'......This is a constant.
        - 'x0'.....This is the mean value of distribution position.
        - 'sigma'..This is the standard diviation of the distribution.
        ---------------------------------------------------------------------
        Nicholai Mauritzson
        Edit: 2019-04-03
    """
    a = np.abs(a)
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

def gaussFuncBimodal(x, a1, mu1, sig1, a2, mu2, sig2):
    """
        A bimodal normal distribution function.
        - 'x'.......This is the independent variable data (list format).
        - 'a1'......This is a constant for function 1.
        - 'mu1'.....This is the mean value of distribution position for function 1.
        - 'sig1'....This is the standard diviation of the distribution for function 1.
        - 'a2'......This is a constant for function 2.
        - 'mu2'.....This is the mean value of distribution position for function 2.
        - 'sig2'....This is the standard diviation of the distribution for function 2.
        ---------------------------------------------------------------------
        Nicholai Mauritzson
        Edit: 2021-10-01
    """
    return gaussFunc(x, a1, mu1, sig1)+gaussFunc(x, a2, mu2, sig2)

def gaussFuncInv(f, a, x0, sigma):
    """
        A Gaussian function solving for the independet variable 'x'.
        - 'f'......This is the gaussian value at 'x'
        - 'a'......This is a constant.
        - 'x0'.....This is the mean value of distribution position.
        - 'sigma'..This is the standard diviation of the distribution.
        ---------------------------------------------------------------------
        Nicholai Mauritzson
        Edit: 2020-01-06
    """
    return np.sqrt(-2*(np.log(f)-np.log(a)))*sigma+x0

def gaussFuncDer(x, a, x0, sigma):
    """
        The first derivative of a generic Gaussian function.
        - 'x'......This is the independent variable data (list format).
        - 'a'......This is a constant.
        - 'x0'.....This is the mean value of distribution position.
        - 'sigma'..This is the standard diviation of the distribution.
        ---------------------------------------------------------------------
        Nicholai Mauritzson
        Edit: 2019-12-19
    """
    return -a * np.exp(-(x - x0)**2 / (2 * sigma**2))*(x - x0)**2 / (2 * sigma**2)

def linearFunc(x,k,m):
    """
    A generic linear equation.
    - 'x'......This is the independent variable data (list format).
    - 'k'......This is the coefficient to 'x' (slope).
    - 'm'......This is a constant.
    ---------------------------------------------------------------------
    Nicholai Mauritzson
    Edit: 2019-04-03
    """
    return k*x+m

def quadraticFunc(x,A,B,C):
    """
    A generic quadratic equation.
    - 'x'......This is the independent variable data (list format).
    - 'A'......This is a constant
    - 'B'......This is a constant
    - 'C'......This is a constant
    
    y = Ax^2 + Bx + C
    ---------------------------------------------------------------------
    Nicholai Mauritzson
    Edit: 2020-12-16
    """
    return A*x**2+B*x+C

def linearFuncInv(y,k,m):
    """
    The inverse of a generic linear equation, i.e x = (y-m)/k
    - 'y'......This is the independent variable data (list format).
    - 'k'......This is the coefficient to 'x' (slope).
    - 'm'......This is a constant.
    ---------------------------------------------------------------------
    Nicholai Mauritzson
    Edit: 2019-05-22
    """
    return (y-m)/k

def reciprocalFunc(x):
    """
    A basic reciprocal function: f(x) = 1/x
    """
    return 1/x

def reciprocalConstantFunc(x, A, B):
    """
    A vairation of the reciprocal function including constants: f(x) = A/x + B
    """
    return A*reciprocalFunc(x) + B

def reciprocalSqrtFunc(x):
    """
    A basic reciprocal of square root function: f(x) = 1/sqrt(x)
    """
    return 1/np.sqrt(x)

def reciprocalConstantSqrtFunc(x, A, B):
    """
    A vairation of the reciprocal of square root function including constants: f(x) = A/sqrt(x) + B
    """
    return A*reciprocalSqrtFunc(x) + B

def expFunc(x, A, B, C):
    """
    A generic exponetial function (base e).
    - 'x'......This is the independent variable data (list format).
    - 'A'......This is a constant.
    - 'B'......This is a constant.
    - 'C'......This is a constant.
    ---------------------------------------------------------------------
    Nicholai Mauritzson
    Edit: 2021-06-08
    """
    return A + B*np.exp(C * x)

def pearsonChi2(obs, exp, verbose=False):
    """
    Chisquare calculator for numerical data.
    obs.......observed values (list)
    exp.......expected value (fit-values or model-values) (list)
    verbose...print out current chi2. Default: False    
    """
    #prevent division with zero
    exp = np.where(exp==0, 1, exp)

    chi2Val = np.sum( (obs-exp)**2/exp )
    
    if verbose:
        print(f'-chi2(): chi2 = {round(chi2Val,3)}')
    
    return chi2Val

def chi2(obs, exp, obs_var, verbose=False):
    """
    Chisquare calculator for numerical data.
    obs.......observed values (list)
    exp.......expected value (fit-values or model-values) (list)
    obs_var...deviation (error) in observed values 'obs' (list)
    verbose...print out current chi2. Default: False    
    """
    #replace all negative obs values with 0
    obs = np.where(obs<0, 0, obs)

    #replace all negative exp values with 0
    exp = np.where(exp<0, 0, exp)
    
    #replace all negative obs_var values with 1
    obs_var = np.where(obs_var<=0, 1, obs_var)
    
    #calulate chi2
    chi2Val = np.sum( ((obs-exp)/obs_var)**2 )
    
    if verbose:
        print(f'-chi2(): chi2 = {round(chi2Val,3)}')
    
    return chi2Val

def chi2red(obs, exp, obs_var, npar, verbose=False):
    """
    The reduced chi-squared function Chi2/N.D.F.
    obs.......observed values (list)
    exp.......expected value (fit-values or model-values) (list)
    obs_var...deviation (error) in observed values 'obs' (list)
    npar......number of fitted parameters (int)
    verbose...print out current chi2. Default: False

    More info:  https://en.wikipedia.org/wiki/Reduced_chi-squared_statistic
                https://arxiv.org/pdf/1012.3754.pdf
    ------------------------
    Nicholai Mauritzson
    2022-08-19
    """

    #replace all 0 in obs_var with 1
    obs_var[obs_var == 0] = 1

    #calulate chi2
    chi2Val = chi2(obs, exp, obs_var)
    
    #calculate number of degrees of freedom
    ndeg = len(obs)-npar

    if verbose:
        print(f'-chi2red(): ndeg = {ndeg}, chi2red = {round(chi2Val/ndeg, 3)}')

    return chi2Val/ndeg


def chi2redNorm(obs, exp, obs_var, npar):
    """
    The reduced chi-squared function Chi2/N.D.F.
    obs.......observed values (list)
    exp.......expected value (fit-values or model-values) (list)
    obs_var...deviation (error) in observed values 'obs' (list)
    npar......number of fitted parameters (int)

    ------------------------
    Nicholai Mauritzson
    2022-11-08
    """

    #normalize obs and exp values to integral=1 before calculating Chi2 values
    obs = prodata.integralNorm(obs, forcePositive=False)
    exp = prodata.integralNorm(exp, forcePositive=False)

    #replace all 0 in obs_var with 1
    obs_var[obs_var == 0] = 1

    #calulate chi2
    chi2Val = chi2(obs, exp, obs_var)
    
    #calculate number of degrees of freedom
    ndeg = len(obs)-npar
    
    return chi2Val/ndeg
    
def ratioCalculator(data1, data2):
    """
    Method for calulating the ratio between two idential lenght arrays of data. These can also be lists.
        1) Takes two arrays 'data1' and 'data2' as input.
        2) Calculated the ratio between each entry as data1[i]/data2[i] and saves this to a new list, 'ratio'.
           NOTE: if data1[i] or data2[i] = 0 then ratio[i] = 0. Values can be negative.
           NOTE: calculated ratio is limited to 4 significant digits.
        3) Returns list 'ratio'
        ---------------------------------------------------------------------
        Nicholai Mauritzson
        Edit: 2019-05-08
    """
    ratio = []
    length_limit = min([len(data1), len(data2)]) #The shortest lenght array will set the limit on number of iterations.
    if len(data1) == len(data2):
        print('WARNING - ratioCalculator(): Arrays are not of identical length. Shortest array will set range starting from first entry.')
    for i in range(length_limit):
        if data2[i] == 0: #Set ratio = 0 when denominator = 0 (data2[i]).
            ratio.append(0)
        else:
            ratio.append(round(data1[i]/data2[i],4)) #append 'ratio' list with calucalted ratio. Limited to 3 significant digits.
    return ratio

def comptonMax(Ef):
    """
    Takes photon energy input and returns the maximum electron recoil energy
    -------------------------------------------------------------------------
    Nicholai Mauritzson
    Edit: 2019-03-26
    """
    return 2*Ef**2/(0.5109989+2*Ef)

def comptonEdgeFit(data, min, max, Ef, bg_sub=False, bg_step=10, fit_lim=None, binFactor=1, plot=False):
    """
    Method is designed to fit and return the edge positions and maximum electron recoil energy of a Compton distribution.
    NOTE: takes unbinned data. For binned data use: "comptonEdgeFitBinned()"
    ---------------------------------------------------------------------
    Inputs: 
    - 'data'........A pandas data frame or series with one column containing the relevant data to be fitted
    - 'min'.........The minimum x-value (ADC) for fit
    - 'max'.........The maximum x-value (ADC) for fit
    - 'bg_sub'......Set to true of background should be subtracted. Will take steps to the right of max acording to bg_step.
    - 'bg_step'.....Set the number of steps to take to the right of max, used for fitting and subtracting a linear background.
    - 'Ef'..........The photon energy in MeV
    - 'fit_lim'.....Boundary parameters for scipy.optimize.curve_fit method (Guassian):
                    Format: fit_lim = [[const_min, mean_min, sigma_min],[const_max, mean_max, sigma_max]] 
    - 'binFactor'...Minimization factor for number of bins. Number of bins will be multiplied with 1/binFactor and rounded to closest integer.
    - 'plot'........Boolean value indicating if a plot of the fit should be shown. Default: False

    1) Creates a histogram, saves x,y data.
    2) Fits the data and prints to console optimal fit parameters.
    3) Tries to find the Compton edge @ 89% and 50% of Compton edge maximum (Knox and Flynn methods).
    4) Calculates the maximum electron recoil energy based on 'Ef'.
    5) Calculated the error of Compton edge for @ 89% and 50% using error propagation.
    6) (Optional) Plots histogram of original data, Gaussian fit and markers for 89% and 50% of maximum.

    Method returns four lists containg the x-value (ADC) for CE at 89% and 50%, the y-values for CE at 89% and 50%, 
    the optimal fitting paramters, const, mean and sigma and their errors.
    - (p,p2) 
    - (y,y2)
    - (const, mean, sigma),
    - (fitError[0], fitError[1], fitError[2])
    
    ---------------------------------------------------------------------
    Nicholai Mauritzson
    Edit: 2020-11-12
    """
    # print(min)
    # print(max)
    if bg_sub:
        #1)____________MAKE HISTOGRAM AND EXTRACT X,Y VALUES___________
        #Create binned data of 'data' and save the x,y values.
        N = np.histogram(data, range=[min, max], bins=int(round((max-min)/binFactor))) 
        bins = N[1]   #Save bin positions.
        val = N[0]   #Save bin heights.
        bin_centers = prodata.getBinCenters(bins)

        popt, pcov = curve_fit(linearFunc, bin_centers[-bg_step:], val[-bg_step:])
        pcov = np.sqrt(np.diag(pcov))
        # chi2reduced = chi2red(val[-bg_step:], linearFunc(bin_centers[-bg_step:], popt[0], popt[1])), 2)

        #Subtracting the background
        bg = linearFunc(bin_centers, popt[0], popt[1])
        val = val - bg

    else:
        #Create binned data of 'data' and save the x,y values.
        N = np.histogram(data, range=[min, max], bins=int(round((max-min)/binFactor))) 
        bins = N[1]   #Save bin positions.
        val = N[0]   #Save bin heights.
    
    #2)____________FITS HISTOGRAM DATA WITH GAUSSIAN_______________
    #Calcaulate the mean value of distibution. Used as first quess for Gaussian fit.
    meanTEMP = bin_centers[0]
    
    #Calculate stdev of distribution. Used as first guess for Gaussian fit.
    sigmaTEMP = (max-min)/3#(max-min)/3#np.sqrt(sum(val * (bin_centers - meanTEMP)**2) / sum(val)) 

    
    if fit_lim != None: #Check if boundary limits are set for Gaussian fit
        popt, pcov = curve_fit(gaussFunc, bin_centers, val, p0 = [np.max(val), meanTEMP, sigmaTEMP], bounds=fit_lim)
    else: #If not...
        popt, pcov = curve_fit(gaussFunc, bin_centers, val, p0 = [np.max(val), meanTEMP, sigmaTEMP])

    const, mean, sigma = popt[0], popt[1], np.abs(popt[2]) #saving result of fit as individual variables.
    
    fitError = np.sqrt(np.diag(pcov)) #Calculate and store the errors of each variable from Gaussian fit errors = [const_err, mean_err, sigma_err]
    
    #calculate the Chi2/NFD of the Gaussian fit
    chi2reduced = round(chi2red(val, gaussFunc(bin_centers, const, mean, sigma), np.sqrt(val), 3), 3)

    #Print to console: optimal fitting parameters
    print('______________________________________________________________________')
    print('>>>> Optimal fitting parameters (Gaussian) <<<')
    print('Method: scipy.curve_fit(gaussianFunc, const, mean, sigma))')
    print(f'-> Maximum height.........const = {round(const,5)} +/- {round(fitError[0],5)}')
    print(f'-> Mean position...........mean = {round(mean,5)} +/- {round(fitError[1],5)}')
    print(f'-> Standard deviation.....sigma = {round(sigma,5)} +/- {round(fitError[2],5)}')
    print(f'-> Reduced chisquare...Chi2/ndf = {chi2reduced}')
    print('______________________________________________________________________')
    print() #Create vertical space on terminal

    #3a)__________FINDING COMPTON EDGE @ 89% OF MAX________________Knox Method 
    print('______________________________________________________________________')
    print('>>>> Finding compton edges... <<<<')  
    for i in tqdm(np.arange(bin_centers[0], bin_centers[-1], 0.01), desc='Finding CE @ 89% of max'): #Loop for finding 89% of maximum with 4 decimal points
        if gaussFunc(i, const, mean, sigma)<=0.89*const:
            p = i #Saving compton edge value
            y = gaussFunc(i, const, mean, sigma) #Save y value at x. Used to derive error in y and x.
            break
    else:
        p = np.nan
        y = np.nan
        p_err = np.nan
        y_err = np.nan
        print('!!! FAILURE TO LOCATE CE @ 89%%... !!!')

    #3b)__________FINDING COMPTON EDGE @ 50% OF MAX_______________Flynn Method
    for i in tqdm(np.arange(bin_centers[0], bin_centers[-1], 0.01), desc='Finding CE @ 50% of max'): #Loop for finding 50% of maximum with 4 decimal points
        if gaussFunc(i, const, mean, sigma)<=0.5*const:
            p2 = i #Saving compton edge value
            y2 = gaussFunc(i, const, mean, sigma) #Save y value at x. Used to derive error in y and x
            break
    else:
        p2 = np.nan
        y2 = np.nan
        p2_err = np.nan
        y2_err = np.nan
        print('!!! FAILURE TO LOCATE CE @ 50%... !!!')
    print('______________________________________________________________________')
    print()#Create vertical empty space in terminal

    #4)____________MAXIMUM ELECTRON RECOIL ENERGY__________________
    E_recoil_max = comptonMax(Ef) #Calculate maximum electron recoil energy

    #5)____________ERROR PROPAGATION CALCULATION___________________
    if isnan(y)==False:
        y_err = errorPropGauss(p, const, fitError[0], mean, fitError[1], sigma, fitError[2])
        p_err = errorPropComptonEdgeFit(fitError[1], fitError[2], .89) #Find the error in Compton edge position at 89%
     
    if isnan(y2)==False:
        y2_err = errorPropGauss(p2, const, fitError[0], mean, fitError[1], sigma, fitError[2])
        p2_err = errorPropComptonEdgeFit(fitError[1], fitError[2], .50) #Find the error in Compton edge position at 50%
    
    print('______________________________________________________________________')
    print('>>>> Results <<<<')
    if isnan(y)==False: #If 89% Compton edge was found, print the error in G(x) (y-value)
        print('-> 89%%, G(x) = %.4f +/- %.4f'%(y, y_err))
        print('-> 89%% Compton edge found at ADC value: %.4f (+/-%.4f)'  % (p, p_err)) #Printing compton edge value (ADC) to console
    if isnan(y2)==False: #If 50% Compton edge was found, print the error in G(x) (y-value)
        print('-> 50%%, G(x) = %.4f +/- %.4f'%(y2, y2_err))
        print('-> 50%% Compton edge found at ADC value: %.4f (+/-%.4f)'  % (p2, p2_err)) #Printing compton edge value (ADC) to console
    print('-> Photon energy: %.4f MeV' % Ef)
    print('-> Maximum electron recoil energy: %.4f MeV' % E_recoil_max)
    print('______________________________________________________________________')


    #6) (optional)_____PLOTTING AND PRINTING RESULTS TO TERMINAL________________ 
    if plot:
        #Plot orginal data for visual purpose.
        plt.step(bin_centers, val, lw=3, label='data') 

        #Plot the Gaussian fit.
        plt.plot(bin_centers, gaussFunc(bin_centers, const, mean, sigma), color='r', linewidth=3, label=f'Gaussian fit, $\chi^2/$ndf = {chi2reduced}')

        #Plot compton edges for 89% and 50% point.
        if isnan(y)==False: #If 89% Compton edge was found
            plt.plot(p, y, color='black', marker='o', markersize=10, label='Compton edge (89%)') #Mark 89% of maximum point
        if isnan(y2)==False: #If 50% Compton edge was found
            plt.plot(p2, y2, color='green', marker='o', markersize=10, label='Compton edge (50%)') #Mark 50% of maximum point

        print()#Create vertical empty space in terminal
        plt.grid(which='both')
        plt.xlabel('QDC [arb.units]')
        plt.ylabel('counts')
        plt.title(f'processing_math.comptonEdgeFit(), Ef={Ef} MeV')
        # plt.yscale('log')
        plt.legend()
        plt.show() #Show plots
    
    #make dictionary to return
    dic_res = {
    'hist':val,
    'bins':bin_centers,
    'CE_89_x':p,
    'CE_89_x_error':p_err,
    'CE_89_y':y,
    'CE_89_y_error':y_err,
    'CE_50_x':p2,
    'CE_50_x_error':p2_err,
    'CE_50_y':y2,
    'CE_50_y_error':y2_err,
    'gauss_const':const,
    'gauss_const_error':fitError[0],
    'gauss_mean':mean,
    'gauss_mean_error':fitError[1],
    'gauss_std':sigma,
    'gauss_std_error':fitError[2],
    'Ef':Ef,
    'recoil_max':E_recoil_max}

    return dic_res 

def comptonEdgeFitBinned(bin_centers, counts, Ef, bgSub=False, bgStep=10, fit_lim=None, plot=False):
    """
    Method is designed to fit and return the edge positions and maximum electron recoil energy of a Compton distribution.
    NOTE: function takes binned data (x,y) and assume the . For unbinned data use: "comptonEdgeFit()"
    ---------------------------------------------------------------------
    Inputs: 
    - 'bin_centers'...An array containting the x_values to be fitted
    - 'counts'........An array containting the relevant y_values to be fitted
    - 'Ef'............The photon energy in MeV
    - 'bgSub'.........Set to true of background should be subtracted. Will take steps to the right of max acording to bg_step.
    - 'bgStep'........Set the number of steps to take to the right of max, used for fitting and subtracting a linear background.
    - 'fit_lim'.......Boundary parameters for scipy.optimize.curve_fit method (Guassian):
                      Format: fit_lim = [[const_min, mean_min, sigma_min],[const_max, mean_max, sigma_max]] 
    - 'plot'..........Boolean value indicating if a plot of the fit should be shown. Default: False

    Method returns four lists containg the x-value (ADC) for CE at 89% and 50%, the y-values for CE at 89% and 50%, 
    the optimal fitting paramters, const, mean and sigma and their errors.
    - (p,p2) 
    - (y,y2)
    - (const, mean, sigma),
    - (fitError[0], fitError[1], fitError[2])
    
    ---------------------------------------------------------------------
    Nicholai Mauritzson
    Edit: 2020-11-23
    """
    #____________OPTIONAL BACKGROUND SUBTRACTION_______________
    if bgSub:
        #linear fitting of background
        popt, pcov = curve_fit(linearFunc, bin_centers[-bgStep:], counts[-bgStep:])
        pcov = np.sqrt(np.diag(pcov))
        #Subtracting the background
        bgCounts = linearFunc(bin_centers, popt[0], popt[1])
        counts = counts - bgCounts


    #____________FITS HISTOGRAM DATA WITH GAUSSIAN_______________
    #Calcaulate the mean value of distibution. Used as first quess for Gaussian fit.
    meanTEMP = np.max(counts)
    #Calculate stdev of distribution. Used as first guess for Gaussian fit.
    sigmaTEMP = np.sqrt(sum(counts * (bin_centers - meanTEMP)**2) / sum(counts)) 
    if fit_lim != None: #Check if boundary limits are set for Gaussian fit
        popt, pcov = curve_fit(gaussFunc, bin_centers, counts, p0 = [np.max(counts), meanTEMP, sigmaTEMP], bounds=fit_lim)
    else: #If not...
        popt, pcov = curve_fit(gaussFunc, bin_centers, counts, p0 = [np.max(counts), meanTEMP, sigmaTEMP])
    const, mean, sigma = popt[0], popt[1], np.abs(popt[2]) #saving result of fit as individual variables.
    fitError = np.sqrt(np.diag(pcov)) #Calculate and store the errors of each variable from Gaussian fit errors = [const_err, mean_err, sigma_err]

    #calculate the Chi2/NFD of the Gaussian fit
    chi2reduced = round(chi2red(counts, gaussFunc(bin_centers, const, mean, sigma), np.sqrt(counts), 3), 3)

    #Print to console: optimal fitting parameters
    print('______________________________________________________________________')
    print('>>>> Optimal fitting parameters (Gaussian) <<<')
    print('Method: scipy.curve_fit(gaussianFunc, const, mean, sigma))')
    print(f'-> Maximum height.........const = {round(const,5)} +/- {round(fitError[0],5)}')
    print(f'-> Mean position...........mean = {round(mean,5)} +/- {round(fitError[1],5)}')
    print(f'-> Standard deviation.....sigma = {round(sigma,5)} +/- {round(fitError[2],5)}')
    print(f'-> Reduced chisquare...Chi2/ndf = {chi2reduced}')
    print('______________________________________________________________________')
    print() #Create vertical space on terminal

    #__________FINDING COMPTON EDGE @ 89% OF MAX________________Knox Method 
    print('______________________________________________________________________')
    print('>>>> Finding compton edges... <<<<')  
    for i in tqdm(np.arange(bin_centers[0], bin_centers[-1], 0.01), desc='Finding CE @ 89% of max'): #Loop for finding 89% of maximum with 4 decimal points
        if gaussFunc(i, const, mean, sigma)<=0.89*const:
            p = i #Saving compton edge value
            y = gaussFunc(i, const, mean, sigma) #Save y value at x. Used to derive error in y and x.
            break
    else:
        p = np.nan
        y = np.nan
        p_err = np.nan
        print('!!! FAILURE TO LOCATE CE @ 89%%... !!!')

    #__________FINDING COMPTON EDGE @ 50% OF MAX_______________Flynn Method
    for i in tqdm(np.arange(bin_centers[0], bin_centers[-1], 0.01), desc='Finding CE @ 50% of max'): #Loop for finding 50% of maximum with 4 decimal points
        if gaussFunc(i, const, mean, sigma)<=0.5*const:
            p2 = i #Saving compton edge value
            y2 = gaussFunc(i, const, mean, sigma) #Save y value at x. Used to derive error in y and x
            break
    else:
        p2 = np.nan
        y2 = np.nan
        p2_err = np.nan
        print('!!! FAILURE TO LOCATE CE @ 50%... !!!')
    print('______________________________________________________________________')
    print()#Create vertical empty space in terminal

    #____________MAXIMUM ELECTRON RECOIL ENERGY__________________
    E_recoil_max = comptonMax(Ef) #Calculate maximum electron recoil energy

    #____________ERROR PROPAGATION CALCULATION___________________
    if isnan(y)==False:
        y_err = errorPropGauss(p, const, fitError[0], mean, fitError[1], sigma, fitError[2])
        p_err = errorPropComptonEdgeFit(fitError[1], fitError[2], .89) #Find the error in Compton edge position at 89%
     
    if isnan(y2)==False:
        y2_err = errorPropGauss(p2, const, fitError[0], mean, fitError[1], sigma, fitError[2])
        p2_err = errorPropComptonEdgeFit(fitError[1], fitError[2], .50) #Find the error in Compton edge position at 50%
    
    print('______________________________________________________________________')
    print('>>>> Results <<<<')
    if isnan(y)==False: #If 89% Compton edge was found, print the error in G(x) (y-value)
        print('-> 89%%, G(x) = %.4f +/- %.4f'%(y, y_err))
        print('-> 89%% Compton edge found at ADC value: %.4f (+/-%.4f)'  % (p, p_err)) #Printing compton edge value (ADC) to console
    if isnan(y2)==False: #If 50% Compton edge was found, print the error in G(x) (y-value)
        print('-> 50%%, G(x) = %.4f +/- %.4f'%(y2, y2_err))
        print('-> 50%% Compton edge found at ADC value: %.4f (+/-%.4f)'  % (p2, p2_err)) #Printing compton edge value (ADC) to console
    print('-> Photon energy: %.4f MeV' % Ef)
    print('-> Maximum electron recoil energy: %.4f MeV' % E_recoil_max)
    print('______________________________________________________________________')


    #(optional)_____PLOTTING AND PRINTING RESULTS TO TERMINAL________________ 
    #Increse plotting points for Gaussian plot by x100
    x_long = np.arange(bin_centers[0], bin_centers[-1], 0.01) 
   
    if plot:
        #Plot orginal data for visual purpose.
        plt.scatter(bin_centers, counts, label='data') 

        #Plot the Gaussian fit.
        plt.plot(x_long, gaussFunc(x_long, const, mean, sigma), color='r', linewidth=3, label=f'Gaussian fit, $\chi^2/$ndf = {chi2reduced}')

        #Plot compton edges for 89% and 50% point.
        if isnan(y)==False: #If 89% Compton edge was found
            plt.plot(p, y, color='black', marker='o', markersize=10, label='Compton edge (89%)') #Mark 89% of maximum point
        if isnan(y2)==False: #If 50% Compton edge was found
            plt.plot(p2, y2, color='green', marker='o', markersize=10, label='Compton edge (50%)') #Mark 50% of maximum point

        print()#Create vertical empty space in terminal
        plt.grid(which='both')
        plt.xlabel('QDC [arb.units]')
        plt.ylabel('counts')
        plt.title(f'processing_math.comptonEdgeFit(), Ef={Ef} MeV')
        plt.legend()
        plt.show() #Show plots
    
    #make dictionary to return
    dic_res = {
    'hist':counts,
    'bins':bin_centers,
    'CE_89_x':p,
    'CE_89_x_error':p_err,
    'CE_89_y':y,
    'CE_89_y_error':y_err,
    'CE_50_x':p2,
    'CE_50_x_error':p2_err,
    'CE_50_y':y2,
    'CE_50_y_error':y2_err,
    'gauss_const':const,
    'gauss_const_error':fitError[0],
    'gauss_mean':mean,
    'gauss_mean_error':fitError[1],
    'gauss_std':sigma,
    'gauss_std_error':fitError[2],
    'Ef':Ef,
    'recoil_max':E_recoil_max}

    return dic_res 

def errorPropMulti(R, variables, errors):
    """
    Method for calculating error of R through error propagation with mutiplication and/or division.
    Ex: R(x)=a*b/c, were a, b and c are the variables.
    Input:
    - 'R'..........This is the product.
    - 'variables'..This are the variables which has an uncertainty as list = (a, b, c).
    - 'errors'.....This is the uncertainties of the variables as list = (err_a, err_b, err_c).
    Return:
    - Error of R.
    ---------------------------------------------------------------------
    Nicholai Mauritzson
    Edit: 2019-03-31
    """
    sum = 0
    for i in range(len(variables)):
        sum += (errors[i]/variables[i])**2
    return R*np.sqrt(sum)

def errorPropAdd(errors):
    """
    Method for calculating error propagation with addition and/or subtraction.
    Ex: R(x)=a+b-c, were a,b and c are the variables.
    Input:
    - 'errors'.....This is the uncertainties of the variables as list = (err_a, err_b, err_c).
    Return:
    - Error of sum.
    ---------------------------------------------------------------------
    Nicholai Mauritzson
    Edit: 2019-03-31
    """
    sum = 0
    for i in range(len(errors)):
        sum += errors[i]**2
    return np.sqrt(sum)

def errorPropConst(constant, var_err):
    """
    Method for calculating error propagation with multiplication with a constant.
    Ex: R(x)=a*x, were a is a constant and x is a variable with error.
    Input:
    - 'var_err'.....This is the uncertainty of the variable x.
    Return:
    - Error of product.
    ---------------------------------------------------------------------
    Nicholai Mauritzson
    Edit: 2019-04-04
    """
    return var_err*constant

def errorPropPower(R, x, x_error, n):
    """
    Method for calculating error of R through error propagation with an exponent.
    Ex: R(x)=x^n, were x is the variable and n is a fixed number.
    Input:
    - 'R'.........This is calculated quantity.
    - 'x'.........This is the variable which has an uncertainty.
    - 'x_error'...This is the uncertainty of the variable
    - 'n'.........This is the exponent (constant)
    Return:
    - Error of R.
    ---------------------------------------------------------------------
    Nicholai Mauritzson
    Edit: 2019-03-28
    """
    return np.abs(R)*np.abs(n)*x_error/np.abs(x)

def errorPropExp(f, exp_error):
    """
    Method calculates and returns the error of f(x), were f(x)=e^x
    - f...........This is the value of f(x)
    - exp_error...This is the error in the exponent of the function.
    ---------------------------------------------------------------------
    Nicholai Mauritzson
    Edit: 2020-01-06 
    """
    return f*exp_error

def errorPropLn(x, x_err):
    """
    Method calulates and treturn the error of f(x), were f(x)=ln(x) (natural logarithm)
    - x..........This is the value of f(x)
    - x_err......This is the error of x
    ---------------------------------------------------------------------
    Nicholai Mauritzson
    Edit: 2020-01-06 
    """
    return x_err/x

def errorPropLog(x, x_err):
    """
    Method calulates and treturn the error of f(x), were f(x)=log(x) (base 10 logarithm)
    - x..........This is the value of f(x)
    - x_err......This is the error of x
    ---------------------------------------------------------------------
    Nicholai Mauritzson
    Edit: 2020-01-06 
    """
    return 0.434*x_err/x

def errorPropGauss(x, const, const_err, mean, mean_err, sigma, sigma_err):
    """
    Method calculates and returns the error in G(x), were G(x) is a Gaussian function. This is done through error propagation.
    - R...........This is the answer to G(x).
    - x...........This is the x-value.
    - const.......This is the constant of the function.
    - const_err...This is the error for the contant.
    - mean........This is the mean value of the distrubution.
    - mean_err....This is the error in the mean value.
    - sigma.......This is the standard deviation of the distrubution.
    - sigma_err...This is the error in the standard diviation.

    ---------------------------------------------------------------------
    Nicholai Mauritzson
    Edit: 2020-01-06    
    """
    #addition
    zero = np.abs(x-mean)
    zero_err = mean_err
    #multiplication
    alpha = np.abs((zero)/(sigma))
    alpha_err = errorPropMulti(alpha, (zero, sigma), (zero_err, sigma_err)) 
    #power of 2
    beta = np.abs(alpha**2) 
    beta_err = errorPropPower(beta, alpha, alpha_err, 2)
    #multiplication with constant
    gamma = np.abs(0.5*beta)
    gamma_err = errorPropConst(0.5, beta_err)
    #exponential
    delta = np.abs(np.exp(-gamma))
    delta_err = errorPropExp(delta, gamma_err)
    #multiplication
    echo = np.abs(const*gamma)
    echo_err = errorPropMulti(echo, (const, delta), (const_err, delta_err))
    
    return echo_err

def errorPropGaussInv(f, f_error, a, a_error, mean, mean_error, sigma, sigma_error):
    """
        A function which calculates the error 'x' for a Gaussian function.
        - 'f'...........This is the values of a Gaussian at 'x'.
        - 'f_error'.....This is the error in the Gaussian value at 'x'.
        - 'a'...........This is the constant of the Gaussian.
        - 'a_error'.....This is the error in the constant.
        - 'mean'........This is the mean value of the gaussian.
        - 'mean_error'..This is the error in the mean value.
        - 'sigma'.......This is the standard diviation of the Gaussian.
        - 'sigma_error'.This is the error in the standard diviation of the Gaussian.
        ---------------------------------------------------------------------
        Nicholai Mauritzson
        Edit: 2020-01-06
    """
    #Function x = np.sqrt(-2*(np.log(f)-np.log(a)))*sigma+mean

    #Natural logarithm
    zero = np.abs(np.log(f))
    zero_err = errorPropLn(np.abs(f), np.abs(f_error))
    #Natural logarithm
    alpha = np.abs(np.log(a))
    alpha_err = errorPropLn(np.abs(a), np.abs(a_error))
    #Addition
    beta = np.abs(zero-alpha)
    beta_err = errorPropAdd((zero_err, alpha_err))
    #Multiplication with constant
    gamma = np.abs(2*beta)
    gamma_err = errorPropConst(2, beta_err)
    #Power of 0.5
    delta = np.sqrt(gamma)
    delta_err = errorPropPower(delta, gamma, gamma_err, 0.5)
    #multiplication
    echo = np.abs(delta*sigma)
    echo_err = errorPropMulti(echo, (delta, np.abs(sigma)), (delta_err, np.abs(sigma_error)))
    #Addition
    epsilon = np.abs(echo+mean)
    epsilon_err = errorPropAdd((echo_err, np.abs(mean_error)))

    return epsilon_err

def errorPropComptonEdgeFit(mean_err, sigma_err, g):
    """
    Find the error of x when G(x)= g*A = A*e^(1/2*(x-mean)^2/sigma^2), were 'g' is the fraction of A at which point the Compton edge is located. 
    Example:    g = .89 (Knox Method)
                g = .50 (Flynn Method)
    Originally derived by Rasmus Höjers, 2018.
    """
    return mean_err + sigma_err*np.sqrt(-2*np.log(g))

def errorPropLinearFuncInv(y, k, kerr, m, merr):
    """
    Calculate error for the inverse of a linear fit, i.e. the error in x as a function of y.
    Func: x=(y-(m))/k
    """
    return np.sqrt((y*kerr/k**2)**2 + (m/k*np.sqrt((merr/m)**2+(kerr/k)**2))**2)

def lightResponseFunction(Tp, a1, a2, a3, a4):
    """
    Converts Neutron energy deposition (proton recoil in MeV) to MeV electron equivalent.

    Returns: MeVee
    """
    return a1*Tp-a2(1- exp(-a3*pow(Tp, a4)))

def kornilovEq(Ep, L0, L1):
    """
    Energy calibration equation.
    Equation 2 from Julius paper: Light-yield response of NE213 scintillator...
    Used to convert proton recoil energy (Pe) to electron recoil energy (Ee).

    Returns electron recoil energy Ee
    """
    
    return L0*Ep**2/(Ep+L1)

def kornilovEq_NE213(Ep, L0):
    """
    Energy calibration equation. Equivalen with NE213A scintillator.
    from https://doi.org/10.1016/j.nima.2008.10.032
    Used to convert proton recoil energy (Pe) to electron recoil energy (Ee).

    Returns electron recoil energy Ee
    """
    L1 = 3.67# +/- 0.19 Dervied from average of all L1 from fitting kornilovEq(Ep, L0, L1)
    #L1 = 2.47 Mean value of (2.41, 2.49, 2.41) taken from Table 1 in https://doi.org/10.1016/j.nima.2008.10.032
    
    return L0*Ep**2/(Ep+L1)

def kornilovEq_EJ305(Ep, L0):
    """
    Energy calibration equation for EJ305
    Fitting parameter L1 determined for each detector using average if all fitted methods
    """
    L1 = 6.55 #+/- 0.38 Dervied from average of all L1 from fitting kornilovEq(Ep, L0, L1)

    return L0*Ep**2/(Ep+L1)

def kornilovEq_EJ331(Ep, L0):
    """
    Energy calibration equation for EJ331
    Fitting parameter L1 determined for each detector using average if all fitted methods
    """
    L1 = 5.34 #+/- 0.48 Dervied from average of all L1 from fitting kornilovEq(Ep, L0, L1)
 
    return L0*Ep**2/(Ep+L1)

def kornilovEq_EJ321P(Ep, L0):
    """
    Energy calibration equation for EJ321P
    Fitting parameter L1 determined for each detector using average if all fitted methods
    """
    L1 = 6.68 #+/- 0.57 Dervied from average of all L1 from fitting kornilovEq(Ep, L0, L1)
 
    return L0*Ep**2/(Ep+L1)


def cecilEq(Ep, K, p1, p2, p3, p4):
    """
    Used to convert proton recoil energy (Ep) to electron recoil energy (Ee).
    Returns electron recoil energy Ee
    """
    return K*(p1*Ep-p2*(1-np.exp(-p3*np.power(Ep,p4))))

def cecilEq_NE213(Ep, C):
    """
    Energy calibration equation. Equivalen with NE213A scintillator.
    from https://doi.org/10.1016/0029-554X(79)90417-8
    Used to convert proton recoil energy (Ep) to electron recoil energy (Ee).
    
    Returns electron recoil energy Ee
    """
    p1 = 0.83
    p2 = 2.82
    p3 = 0.25
    p4 = 0.93

    return C*(p1*Ep-p2*(1-np.exp(-p3*np.power(Ep,p4))))

def cecilEq_NE224(Ep, C):
    """
    Energy calibration equation. Equivalen with EJ305 scintillator.
    from https://doi.org/10.1016/0029-554X(79)90417-8
    Used to convert proton recoil energy (Ep) to electron recoil energy (Ee).
    
    Returns electron recoil energy Ee
    """
    p1 = 1.0
    p2 = 8.2
    p3 = 0.1 
    p4 = 0.88 

    return C*(p1*Ep-p2*(1-np.exp(-p3*np.power(Ep,p4))))

def cecilEq_EJ309(Ep, C):
    """    
    Energy calibration equation. Equivalen with EJ305 scintillator.
    from https://doi.org/10.1016/j.nima.2013.03.032
    Used to convert proton recoil energy (Ep) to electron recoil energy (Ee).
    
    Returns electron recoil energy Ee
    """
    p1 = 0.817
    p2 = 2.63
    p3 = 0.297
    p4 = 1

    return C*(p1*Ep-p2*(1-np.exp(-p3*np.power(Ep,p4))))

def cecilEq_EJ321P(Ep, K):
    """
    Used to convert proton recoil energy (Ep) to electron recoil energy (Ee).
    Returns electron recoil energy Ee
    """
    #values derive from average of HH, TP and FD methods
    p1 = 0.43 #+/- 0.01
    p2 = 0.77 #+/- 0.04
    p3 = 0.26 #+/- 0.07
    p4 = 2.13 #+/- 0.43

    return K*(p1*Ep-p2*(1-np.exp(-p3*np.power(Ep,p4))))

def poly2Eq_EJ321P(Ep, N):
    """
    Used to convert proton recoil energy (Ep) to electron recoil energy (Ee).
    Returns electron recoil energy Ee
    """
    #values derive from average of HH, TP and FD methods
    A = 0.03463543958969215 #+/- 0.001824665639541493
    B = 0.10759911522298186 #+/- 0.008767418704202378
    
    return N*(A*Ep**2 + B*Ep)


def poly2(Ep, N, A, B):
    """
    Used to convert proton recoil energy (Ep) to electron recoil energy (Ee).
    Returns electron recoil energy Ee
    """

    return N*(A*Ep**2 + B*Ep)


def doubleGaussFunc(x, amp1, mean1, std1, amp2, mean2, std2):
    """
    A double Gaussian function. 
    f(x) = G1(x)+G2(x)
    """
    amp1 = np.abs(amp1)
    amp2 = np.abs(amp2)

    return  gaussFunc(x, amp1, mean1, std1) + gaussFunc(x, amp2, mean2, std2)


def doubleGaussFit(x_input, y_input, start=0, stop=1, param=[1,1,1,1,1,1], error=False):
    """
    Fitting routine for doubleGaussFunc() using the scipy.optimize library and curve_fit() routine.

    Parameters:
    x.......x-values for fit.
    y.......y-values for fit.
    param...list of starting parameters to give to fit. Example: param = (amp1, mean1, std1, amp2, mean2, std2)
    start...start values to fit from (x-value)
    stop....stop values to fit to (x-value)
    error...a boolean, indicating if the fitting errors should be returned. Default: False
    """
    #check if all starting paramters are given
    if len(param) != 6:
        print('doubleGaussFit(): All 6 starting parameters needs to be given...')
        return 0

    #check if range is given, else fit entire data-set
    x, y = prodata.binDataSlicer(x_input, y_input, start, stop)

    #fit the data for the specified range
    try:
        popt, pcov = curve_fit(doubleGaussFunc, x, y, p0 = param) 
    except RuntimeError:
        print('gaussFit(): RuntimeError')
        popt = [1,1,1]
        pcov = [1,1,1]
    
    if error:
        return popt, np.sqrt(np.diag(pcov))
    else: 
        return popt

def YAP_distance_calc(g_flash, D):
    """
    Method to calcualte the distance between the YAP and source based on
    g-flash=D/c-d/c, 
    were D is the distance between the neutorn detector and source and d is the distance between the YAP and source.
    """
    c = 299792458.0 #speed of light
    return -(g_flash-D/c)*c

def leastSquare(f1, f2):
    """
    Least square sun calculator between two different functions.

    Paramters: 
    f1.........an array of values
    f2.........an array of values

    Carl Fredrich Gauss
    1801
    """
    sum = np.arange(0)
    for i in range(len(f1)):
        sum = np.append(sum, (f1[i]-f2[i])**2)
        return sum

def plotAlignTest(x1, y1, x2, y2):
    """
    Method takes two arrays (x1,y1) and (x2, y2) and return the calulated value of:
    sum(|x1-x2|) + sum(|y1-y2|)

    This can be used to minize the difference between two different arrays in X and Y when determening an X and Y offset to the data.
    """
    return np.sum(np.abs(x1-x2)) + np.sum(np.abs(y1-y2))


def quadratic(x,a,b,c):
    """
    A simple quadratic polynomial
    """
    return a+b*x*x + c*x




def landauFunc(x):
    return 1/np.sqrt(2*np.pi)*np.exp(-(x+np.exp(-x))/2)

def logNormalFunc(x, mean, std):
    #plt.plot(np.arange(0,1000,.1), logNormalFunc(np.arange(0,1000,.1), 5, 1)) 
    return 1/(x*std*np.sqrt(2*np.pi))*np.exp(-(np.log(x)-mean)**2/(2*std**2))

def expGaussFunc(x, h, mu, std, tau):
    """
    Used the EMG form.
    See: https://en.wikipedia.org/wiki/Exponentially_modified_Gaussian_distribution

    h is amplitude, 
    mu is mean value
    std is standard diviation
    tau is 1/lambda the exponent relaxation time.

    Position of mean value is given by: mu + tau
    """
    from scipy.special import erfc
    return h*std/tau * np.sqrt(np.pi/2)*np.exp(1/2*(std/tau)**2 - (x-mu)/tau) * erfc(1/np.sqrt(2)*(std/tau - (x-mu)/std))

def landauFunc(x, A, mu, std):
    """
    A Landau function based on a Moyal pdf. Note due to being a PDF the amplitude A will need to be larger than expected.
    """
    return moyal.pdf(x, loc=mu, scale=std)*A

def poissonFunc(x, A, mean):
    ans = []

    for i in range(len(x)):
        print(f'A = {A}')

        xint = int(x[i])
        print(f'xint = {xint}')

        ans.append( A * (np.exp(-mean) * mean**xint) / np.math.factorial(xint))
        print(f'ans = {ans[i]}')
    
    return ans

#                               `-.
#               -._ `. `-.`-. `-.
#              _._ `-._`.   .--.  `.
#           .-'   '-.  `-|\/    \|   `-.
#         .'         '-._\   (o)O) `-.
#        /         /         _.--.\ '. `-. `-.
#       /|    (    |  /  -. ( -._( -._ '. '.
#      /  \    \-.__\ \_.-'`.`.__'.   `-, '. .'
#      |  /\    |  / \ \     `--')/  .-'.'.'
#  .._/  /  /  /  / / \ \          .' . .' .'
# /  ___/  |  /   \ \  \ \__       '.'. . .
# \  \___  \ (     \ \  `._ `.     .' . ' .'
#  \ `-._\ (  `-.__ | \    )//   .'  .' .-'
#   \_-._\  \  `-._\)//    ""_.-' .-' .' .'
#     `-'    \ -._\ ""_..--''  .-' .'
#             \/    .' .-'.-'  .-' .-'
#                 .-'.' .'  .' .-'
# "PRECIOUSSSS!! What has the nasty Bagginsess
#            got in it's pocketssss?"