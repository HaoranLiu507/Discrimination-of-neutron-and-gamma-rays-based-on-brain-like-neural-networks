import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from lmfit import Parameters, fit_report, minimize
import sys
import os
from library import processing_data as prodata
from library import processing_math as promath
# import tqdm



def energyCalibration(QDC_data):
    """
    Takes QDC values as input and, using chargeCalibration() function, will return corresponding energy values [MeVee].
    Calibration is for data analysed with QDC integration lenghts of 500ns.

    Nicholai Mauritzson
    2020-10-05
    """
    energy_cal = np.array([1,2,3,4]) #Energy values used for calibration
    QDC_cal = np.array([1,2,3,4]) #Corresponding QDC values to energy

    popt, pcov = curve_fit(promath.linearFunc, QDC_cal, energy_cal, p0=[1e14, -200]) #fitting the charge and QDC_cal values to get linear interpolation coefficients.

    return QDC_data*popt[0]+popt[1]

def chargeCalibration(Q_data, plot=False):
    """
    Take charge values (Coulombs) from simulation and converts it to QDC channels.
    Calibration is for QDC integration lenghts of 500ns.
    Charge injector used to make this calibration:
    - Capacitance: 111e-12 Farads
    - Resistance: 983.6 Ohm

    If plot=True, program will produce calibration plot of charge to QDC and exit.

    Nicholai Mauritzson 
    2020-10-05
    """
    Q = np.array([1.11e-11, 2.22e-11, 3.33e-11, 4.44e-11, 5.55e-11, 6.66e-11, 7.77e-11, 8.88e-11, 9.99e-11, 1.11e-10]) #generated charge [C] from charge injector circuit.
    QDC_cal = np.array([908.51120595, 2502.11725786, 4211.8347989, 5916.39842889, 7676.38471528, 9433.02979986, 11190.85143695, 12988.14107532, 14775.87644921, 16559.08759556]) #Integration values [arb. units] from digitizer analysis.
    
    popt, pcov = curve_fit(promath.linearFunc, Q, QDC_cal, p0=[1e14, -200.]) #fitting the charge and QDC_cal values to get linear interpolation coefficients.

    if plot:
        plt.scatter(Q, QDC_cal, label='data')
        plt.plot(Q, Q*popt[0]+popt[1], label=f'k={popt[0]}, m={popt[1]}') 
        plt.xlim([.8e-11, 1.3e-10])
        plt.xlabel('Charge [C]')
        plt.ylabel('QDC values [arb units]')
        plt.legend()
        plt.show()
        return


    return Q_data*popt[0] + popt[1]


def loadData(pathToFile):   
    """
    load npz data. Return andas data frame with two columns: "bin_centers" and "counts"
    NOTE: Only works for binned data not dictcount data!
    NOTE: make it work for dicount data as well!

    """
    df = pd.DataFrame()
    npzData = np.load(pathToFile) #import data
    df['bin_edges'] = npzData['bin_edges'][:-1]
    df['bin_centers'] = prodata.getBinCenters(npzData['bin_edges'])
    df['counts'] = npzData['counts']

    return df

def dataMerger(path, filename, binName='bin_edges', countsName='counts'):
    """
    Function is used to merge together unbinned histograms from all 1-6 MeV ranges of energies into one pandas data frame.
    Each column represent one neutron energy run.

    Parameters:
    path........path to main directory. Function will atomatically get name of subfolders for each energy run and label the columns in the data frame accordingly.
    filename....filename of the npz files you which to add to data frame. Name needs to be common across all runs in 'path'.

    /Nicholai Mauritzson 2020-08-13
    """
    paths = [i[0] for i in os.walk(path)][1:] #save all paths to subdirectories within 'path' in an array. remove fist entry which is simply 'path'

    energies = np.arange(0)
    for path_str in paths: #extra energy information from subfolder paths to use as column headers in data frame.
        energies = np.append(energies, str(path_str[-6:]))

    pdMerged = pd.DataFrame()  #create empty data frame

    for i in range(len(paths)):
        pdMerged[str(energies[i])] = pd.Series(unbinning_npz(str(paths[i]+'/'+filename), binName, countsName)) #unbin data and add to new column in data frame.

    return pdMerged

def unbinning(bins, counts):
    """
    Function for unbinning already binned data

    /Nicholai Mauritzson 2020-11-26
    """

    unbinned_data = np.arange(0) #make empty array to hold unbinned data
    for i in range(len(counts)): #loop for number of bins in orignal data
        unbinned_data = np.append(unbinned_data, np.repeat(bins[i], counts[i]))
    return unbinned_data


def unbinning_npz(path, binName='bin_edges', countsName='counts'):
    """
    Function for unbinning data. Takes path to npz file containg bins and counts of a histogram. 
    Will return array suitable for histogramming with different bin values. 
    NOTE: you are not able to use finer binning than the original data.

    Parameter:
    bins.....name of bins value array.
    counts...name of related counts value array.

    /Nicholai Mauritzson 2020-08-13
    """
    f = np.load(path) #opening npz file
    bins = f[binName] #extracting bins
    counts = f[countsName] #extracting counts

    unbinned_data = np.arange(0) #make empty array to hold unbinned data
    for i in range(len(counts)): #loop for number of bins in orignal data
        unbinned_data = np.append(unbinned_data, np.repeat(bins[i], counts[i]))
    return unbinned_data

def getQE(waveLenght):
    """
    Function to calculate the quantum efficiency of the PMT (9821B p1). Will return QE based on wavelenght input.
    """
    from scipy.interpolate import interp1d

    wl = [280.49968,284.16044,287.82715,289.66495,292.42462,296.08836,298.851,300.6992,302.54888,304.39856,305.34863,309.0094,309.95354,310.8947,313.65438,316.4096,319.17373,319.20787,322.88348,325.63425,328.39689,330.24657,333.90734,338.4781,343.0444,348.50586,356.68245,366.65825,376.63109,384.7869,392.02975,400.17962,410.13613,419.18264,429.12578,437.2623,443.57882,452.61345,459.83849,467.05313,476.08183,483.29647,491.41962,499.54278,510.37513,517.59274,526.61699,534.73865,543.75547,550.97606,557.28664,567.22385,575.35888,585.31241,600.70662,618.82487,626.07663]
    QE = [0.00000,0.01058,0.02334,0.03119,0.04395,0.05573,0.06948,0.08077,0.09255,0.10433,0.11857,0.12937,0.14164,0.15293,0.1657,0.17699,0.19122,0.20251,0.21822,0.22804,0.24178,0.25356,0.26436,0.27614,0.28645,0.29282,0.29723,0.29673,0.29524,0.29278,0.28835,0.28392,0.27703,0.26917,0.25786,0.24901,0.2382,0.22641,0.21609,0.20233,0.18857,0.17482,0.16155,0.14828,0.13108,0.11831,0.10308,0.08932,0.07163,0.05984,0.04707,0.0338,0.02446,0.01659,0.00822,0.00084,0.00000]
    interpol = interp1d(wl, QE)       
    
    return interpol(waveLenght)

def NE213A_emission_spectra(wavelenght = 380, plot=False):
    """
    Will return the relative intensity at a specific wavelenght [nm] based on the spectrum of the light emission for NE213.
    If plot is set to True, a figure of the entire spectrum is produced.

    Reference data taken from:
    Article: Photocathode non-uniformity contribution to the energy resolution of scintillators
    Authors: M Mottaghian, R Koohi-Fayegh, Nima Ghal-Eh, Nima Ghal-Eh, Gholam reza Etaati
    DOI: 10.1093/rpd/ncq041 

    -------------------------------
    Nicholai Mauritzson
    2020-11-05
    """
    
    #wavelenght from data
    wl = [380, 380.246599278521, 381.5769201863821, 382.7106192695025, 383.8535247256877, 384.0488313542763, 385.7612167443259, 387.279610701939, 388.4251465504284, 390.8865361491057, 392.78370659852703, 395.24180820682403, 397.51315196152115, 400.1612994137983, 403.5591086727792, 406.3894107921239, 409.4084435592966, 411.48250789117697, 412.9923530738013, 414.68829851194954, 416.5723169998497, 418.2630016533895, 419.5762250112731, 421.0742334285285, 422.5689538554036, 424.6239478430783, 426.6736810461446, 429.09232676987824, 431.1414023748685, 432.6328348113633, 434.5004133473621, 436.55803772734106, 438.42890425372013, 440.6759168796032, 442.1732676987825, 444.22891928453333, 445.9103975650083, 447.2163873440553, 449.0793626935217, 450.5668495415602, 452.79939500977, 455.0319404779799, 457.26514354426575, 459.5042649932361, 462.301687208778, 466.2282053209079, 469.59181947993386, 474.4481812716068, 478.18399594168045, 482.1045956711258, 486.7768300015031, 490.51264467157677, 494.43718998947844, 498.36305050353224, 502.8498421764617, 506.5882872388396, 509.57772809258984, 514.0671501578236, 519.8645347963325, 528]  
    #intensity from data
    int = [0, 0.05600621899894809, 0.10511235532842345, 0.1296595520817676, 0.17876803697580046, 0.19981822110326175, 0.2682178152713063, 0.3190759807605591, 0.37520197655193155, 0.4418377987374117, 0.5032175334435594, 0.5610814670073652, 0.6207021268600631, 0.6855812227566511, 0.7504509243950097, 0.8012926499323614, 0.8556407823538252, 0.8889481249060573, 0.9169993799789569, 0.9415395310386291, 0.967831711258079, 0.9783368405230721, 0.9818291560198407, 0.9783016120547121, 0.9660021794679091, 0.9484325680144295, 0.9168279347662708, 0.8694292048699834, 0.8360701938974897, 0.8149988726890125, 0.7974316098000901, 0.78687950924395, 0.7780841349767023, 0.7727928190290094, 0.7675108973395461, 0.7516956636104013, 0.7376395047347062, 0.7218336652637907, 0.6919857583045241, 0.6603881707500374, 0.6165005448669773, 0.572612918983917, 0.5304796708251917, 0.5041358221854801, 0.4672586615060874, 0.4426480535096948, 0.41629011348263933, 0.37236960769577643, 0.33898945964226657, 0.29858945212686006, 0.26344318352622875, 0.2300630354727191, 0.20018929430332189, 0.17382430858259434, 0.14394352171952507, 0.1175808845633548, 0.09298201938974904, 0.07011874342401936, 0.03671276116037858, 0 ]
    #interpolate the data with a spline fitting function
    interpfunc = interp1d(wl, int)
    
    if plot:
        plt.figure(0)
        plt.plot(wl, interpfunc(wl), lw=3)
        plt.xlabel('wavelength [nm]')
        plt.ylabel('relative intensity')
        plt.title('NE213 light-emission spectrum')
        plt.show()
    return interpfunc(wavelenght)

def calibrationFit(data, numBins, binRange, Ef=0, plot=False, compton=True):
    """
    Method will take data in the form of a histogram and fit a gaussian around it.
    It will return the fitted parameters, their errors and the Chi2/ndf value of the fit.
    
    Parameters:
    data............array of data to be binned and fitted
    numBins.........the number of bins to make histogram with
    binRange........the range of the bins of the historgram
    Ef..............The energy of the original gamma-ray in MeV (optional)
    plot............A flag to determine if a plot should shown of the fit. (optional)
    compton.........A flag to determine if the maximum energy deposition should be determine using the maximum compton energy function. If false maximum energy = Ef

    -----------------------------------
    Nicholai Mauritzson
    2021-05-19
    """

    #make binned data
    counts, bin_edges = np.histogram(data, numBins, binRange)
    bin_centers = prodata.getBinCenters(bin_edges) #calculate center of bins

    #Fit the data with the given parameters
    par, par_err = promath.gaussFit(bin_centers, counts, error=True)

    #calculate expected values based on fit
    exp = promath.gaussFunc(bin_centers, par[0], par[1], par[2])
    
    #Calculate the reduced Chi2 value
    chi2reduced = promath.chi2red(counts, exp, np.sqrt(counts), 3)

    #calculate maximum recoil energy of gamma ray
    if Ef>0 and compton:
        E_recoil_max = promath.comptonMax(Ef)
    else:
        E_recoil_max = Ef

    #Print to console: optimal fitting parameters
    print('______________________________________________________________________')
    print('>>>> Optimal fitting parameters (Gaussian) <<<')
    print('Method: scipy.curve_fit(gaussianFunc, const, mean, sigma))')
    print(f'-> Maximum height.........const = {np.round(par[0],5)} +/- {np.round(par_err[0],5)}')
    print(f'-> Mean position...........mean = {np.round(par[1],5)} +/- {np.round(par_err[1],5)}')
    print(f'-> Standard deviation.....sigma = {np.round(par[2],5)} +/- {np.round(par_err[2],5)}')
    print(f'-> Reduced chisquare...Chi2/ndf = {np.round(chi2reduced,2)}')
    print(f'-> Resolution.........FWHM/mean = {np.round( (par[2] / par[1])*100, 2)} %')
    print(f'-> Photon energy.............Ef = {Ef} MeV')
    if Ef>0:
        print(f'-> Maximum electron recoil energy: {E_recoil_max} MeV')
    print('______________________________________________________________________')
    

    #plot if flag is True
    if plot:
        x_fit = np.arange(binRange[0]*0.65, binRange[-1]*1.2, 0.1)
        y_fit = promath.gaussFunc(x_fit, par[0], par[1], par[2])
        plt.title(f'Fitted gaussian to data (Ef = {Ef} MeV')
        plt.step(bin_centers, counts, label='Data mid', where='mid')
        plt.plot(x_fit, y_fit, lw=2, color='red', label=f'Gaussian fit $\chi^2/ndf = {round(chi2reduced, 2)}$, R={round( (par[2] / par[1])*100, 2)}%')
        plt.xlim([binRange[0]*0.65, binRange[-1]*1.2])
        plt.legend()
        plt.xlabel('bin_centers')
        plt.ylabel('counts')
        plt.show()


    #put all results in one dictionary for return
    dic_res = {
        'gauss_const':      par[0],
        'gauss_const_error':par_err[0],
        'gauss_mean':       par[1],
        'gauss_mean_error': par_err[1],
        'gauss_std':        par[2],
        'gauss_std_error':  par_err[2],
        'chi2reduced':      chi2reduced,
        'recoil_max':       E_recoil_max}
    
    return dic_res


def calibrationFitBinned(bin_centers, counts, start=0, stop=0, Ef=0, plot=False, compton=True):
    """
    Same as "calibrationFit" but takes binned data as input.
    Method will take data fit a gaussian around it.
    It will return the fitted parameters, their errors and the Chi2/ndf value of the fit.
    
    Parameters:
    counts..........the counts of the binned data (y-values)
    bin_centers.....the centers of the bins (x-values)
    min.............starting value to perform fit
    max.............end value to perform fit to
    Ef..............The energy of the original gamma-ray in MeV (optional)
    plot............A flag to determine if a plot should shown of the fit. (optional)
    compton.........A flag to determine if the maximum energy deposition should be determine using the maximum compton energy function. If false maximum energy = Ef

    -----------------------------------
    Nicholai Mauritzson
    2021-06-04
    """
    if stop == 0:
        stop = np.max(bin_centers)
    
    # Get range to fit from input parameters.
    x, y = prodata.binDataSlicer(bin_centers, counts, start, stop)
    
    #Fit the data with the given parameters
    par, par_err = promath.gaussFit(x, y, error=True)

    #calculate expected values based on fit
    exp = promath.gaussFunc(x, par[0], par[1], par[2])
    
    #Calculate the reduced Chi2 value
    chi2reduced = promath.chi2red(y, exp, np.sqrt(y), 3)

    #calculate maximum recoil energy of gamma ray
    if Ef>0 and compton:
        E_recoil_max = promath.comptonMax(Ef)
    else:
        E_recoil_max = Ef

    #Print to console: optimal fitting parameters
    print('______________________________________________________________________')
    print('>>>> Optimal fitting parameters (Gaussian) <<<')
    print('Method: scipy.curve_fit(gaussianFunc, const, mean, sigma))')
    print(f'-> Maximum height.........const = {np.round(par[0],5)} +/- {np.round(par_err[0],5)}')
    print(f'-> Mean position...........mean = {np.round(par[1],5)} +/- {np.round(par_err[1],5)}')
    print(f'-> Standard deviation.....sigma = {np.round(par[2],5)} +/- {np.round(par_err[2],5)}')
    print(f'-> Reduced chisquare...Chi2/ndf = {np.round(chi2reduced,2)}')
    print(f'-> Resolution.........FWHM/mean = {np.round( (par[2] / par[1])*100, 2)} %')
    print(f'-> Photon energy.............Ef = {Ef} MeV')
    if Ef>0:
        print(f'-> Maximum electron recoil energy: {E_recoil_max} MeV')
    print('______________________________________________________________________')
    

    #plot if flag is True
    if plot:
        x_fit = np.arange(x[0]*0.65, x[-1]*1.2, 0.1)
        y_fit = promath.gaussFunc(x_fit, par[0], par[1], par[2])
        plt.title(f'Fitted gaussian to data (Ef = {Ef} MeV')
        plt.step(bin_centers, counts, label='Data mid', where='mid')
        plt.plot(x_fit, y_fit, lw=2, color='red', label=f'Gaussian fit $\chi^2/ndf = {round(chi2reduced, 2)}$, R={round( (par[2] / par[1])*100, 2)}%')
        plt.vlines(x[0], ymin=0, ymax=np.max(y), color='black')
        plt.vlines(x[-1], ymin=0, ymax=np.max(y), color='black')
        plt.xlim([x[0]*0.65, x[-1]*1.2])
        
        plt.legend()
        plt.xlabel('bin_centers')
        plt.ylabel('counts')
        plt.show()


    #put all results in one dictionary for return
    dic_res = {
        'gauss_const':      par[0],
        'gauss_const_error':par_err[0],
        'gauss_mean':       par[1],
        'gauss_mean_error': par_err[1],
        'gauss_std':        par[2],
        'gauss_std_error':  par_err[2],
        'chi2reduced':      chi2reduced,
        'recoil_max':       E_recoil_max}
    
    return dic_res

def muonStopPoly(E=10, plot=False):
    """
    Method returns the stopping power of a muon in Polyethylene in units of MeV/cm
    Data taken from:
    MUON STOPPING POWER AND RANGE TABLES 10 MeV–100 TeV, Donald E. GROOM etal. (TABLE III-4. Muons in Polyethylene [C2 H4 ])

    Paramters:
    E..........The energy in units of MeV to calulate the stopping power for.
    plot.......If set to True a plot of the stopping power [MeV/cm] will be shown.

    --------------------------
    Nicholai Mauritzson
    2020-11-11
    """
    if E>100000000 or E<10:
        print(f'Input error: Variable E needs to be within 10-100e6 MeV)')
        return 0
    #reference: Triumf Kinematic Handbook page VII - 78.
    # densityNE102 = 1.03#g/cm3
    # energyNE102 = np.array([1.004270673722338, 1.059112669108065, 1.177905551093826, 1.5133697713418575, 1.8437529081465056, 2.212343282017394, 2.842412486915657, 4.348853553024801, 5.298073601495028, 6.309153322012243, 7.863815472505145, 9.508428243226987, 12.78737426748008, 16.809293123124558, 22.098403655629745, 26.722660811641013, 35.40345863110316, 50.23231891392443, 64.08015294160876, 75.18441407583207, 88.21877399028817, 104.30506038336175, 120.54789463581109, 139.32013315285556, 171.1423267251517, 205.4927074129172, 266.2851951947411, 327.1511185262829, 383.95720575556487, 475.3301617717235, 583.9978124924776, 680.2192685502523, 835.7276621536125, 995.946419327242])
    # stoppingNE102 = np.array([50.81093664008732, 48.79277338492152, 44.6931522649013, 36.498167695362426, 31.247405115334704, 26.934352250188915, 21.99564039176928, 15.587973775139146, 13.256272964491572, 11.503882010933713, 9.716441262059298, 8.43124388841297, 6.791783063823427, 5.545695807261242, 4.620209344349366, 4.09052874250846, 3.476955654897245, 2.9148962143602435, 2.632323796216024, 2.4759070464698554, 2.360215237287091, 2.2498294885385315, 2.1884609848418712, 2.1287664272220823, 2.0699649148392028, 2.040224888877341, 2.0100196747565526, 2.0076119697641546, 2.00574130442957, 2.0032497951298867, 2.0143071711122063, 2.0260550597078373, 2.0372383143426496, 2.0488470703639172])
    
    #reference: MUON STOPPING POWER AND RANGE TABLES 10 MeV–100 TeV, Donald E. GROOM etal. (TABLE III-4. Muons in Polyethylene [C2 H4 ])
    energy = np.array([     10,     14,     20,     30,     40,     80,     100,    140,    200,    300,    400,    800,    1000,   1400,   2000,   3000,   4000,   8000,   10000,  14000,   20000,  30000,  40000,  80000,  100000, 140000, 200000, 300000, 400000, 800000, 1000000, 1400000, 2000000,    3000000,    4000000,    8000000,    10000000,   14000000,   20000000,   30000000,   40000000,   80000000,   100000000])
    stopping = np.array([   8.467,  6.596,  5.145,  3.987,  3.401,  2.547,  2.384,  2.217,  2.120,  2.081,  2.084,  2.157,  2.191,  2.247,  2.309,  2.380,  2.430,  2.547,  2.584,  2.640,  2.700,  2.772,  2.828,  2.995,  3.065,  3.193,  3.373,  3.657,  3.935,  5.033,  5.584,  6.678,  8.338,      11.102,     13.897,     25.193,     30.902,     42.339,     59.656,     88.689,     117.962,    236.299,    295.995]) #MeV cm2/g
    range = np.array([6.596e-1, 1.201, 2.242, 4.481, 7.215, 2.124e1, 2.937e1, 4.685e1, 7.464e1, 1.224e2, 1.705e2, 3.592e2, 4.512e2, 6.313e2, 8.946e2, 1.321e3, 1.736e3, 3.340e3, 4.119e3, 5.650e3, 7.869e3, 1.155e4, 1.512e4, 2.884e4, 3.544e4, 4.822e4, 6.650e4, 9.496e4, 1.213e5, 2.110e5, 2.487e5, 3.141e5, 3.944e5, 4.980e5, 5.783e5, 7.891e5, 8.607e5, 9.708e5, 1.090e6, 1.226e6, 1.324e6, 1.559e6, 1.634e6]) #g/cm2
    MIP = 328 #threshold for minium ionzing particles
    mu_crit = 1280000 #threshold for muon critical energy: which is the energy at which electronic and radiative losses are equal
    density = .890 #g/cm3

    func = interp1d(energy, stopping)
    if plot:
        plt.title('Stopping power Polyethylene [MeV/cm]')
        plt.plot(np.arange(10,100000000,10), func(np.arange(10,100000000,10))*density, lw=2, label='$\mu$ data')
        # plt.plot(energyNE102, stoppingNE102*densityNE102, label='NE-102 data')
        plt.yscale('log')
        plt.xscale('log')
        plt.vlines(mu_crit, 3, 10, color='red', lw=2, label='$\mu$ critical energy limit')
        plt.vlines(MIP, 0, 5, lw=2, label='Minimum ionizing limit')    
        plt.xlabel('Kinetic Energy [MeV]')
        plt.ylabel('MeV/cm')
        plt.legend()
        plt.show()
    return func(E)*density

def PMTGainAlignSmear(model, data, scale, gain, smear, bg = np.zeros(10), data_weight=1, bg_weight=1, scaleBound=[-np.inf, np.inf], gainBound=[-np.inf, np.inf], smearBound=[-np.inf, np.inf], stepSize=1, binRange=list(range(0,500,2)), plot=False):
    """
    Function to align histogram of a model to a data set by offseting the bins and scaling the counts. 
    Function will return best paramters for the scale and gain based on a minimizing (scipy) algorithm.

    model..........the model to be varied. Needs to be unbinned.
    data...........the data set agaist which the model will be compared. Needs to be unbinned.
    bg.............background data set, will be subtracted from "data". optional
    scale..........initial guess for scale paramter
    gain...........initial guess for gain paramter
    smear..........initial guess for smearing contant (standard diviaiton of a gaussian with mean=1)
    data_weight....weight for histgram (see numpy.histogram())
    bg_weight......weight for histgram (see numpy.histogram())
    scaleBound.....boundary conditions for scale parameter, default -inf to +inf
    gainBound......boundary conditions for gain parameter, default -inf to +inf
    smearBound.....boundary conditions for smear parameter, default -inf to +inf
    stepSize.......gives the step size for the minimizing algorithm to use
    binRange.......list defining the binning for the histograms
    smear..........apply a smearing function to the data be
    plot...........set to True to produce plots of aligned histograms. Default: False

    --------------------------
    Nicholai Mauritzson
    2021-05-03
    """

    def residual(pars, x, data=None):
        """residual fuction used by the minimizing algorithm"""

        vals = pars.valuesdict()
        scale = vals['scale']
        gain = vals['gain']
        smear = vals['smear']


        x_smear = np.zeros(len(x))
        for i in range(len(x)):
            x_smear = gaussSmear(x[i], smear) #apply smearing function

        w_data = np.empty(len(data)) #Make weighting array same length as data-set
        w_data.fill(1/data_weight) #Fill array with 1/data_weight

        w_bg = np.empty(len(bg)) #Make weighting array same length as background data-set
        w_bg.fill(1/bg_weight) #Fill array with 1/bg_weight

        hist, binEdges = np.histogram(x_smear * gain, binRange) #Binning and offsetting the model.
        model = hist * scale #Apply scaling factor

        dataHist,   dataBinEdges    = np.histogram(data, binRange, weights=w_data) #Binning the data set.
        bgHist,     bgBinEdges      = np.histogram(bg, binRange, weights=w_bg) #Binning the background data set.

        ans = ((dataHist-bgHist) - model).astype(np.float)  #subtracting the background and calculating the differences

        return ans

    fit_params = Parameters() #Container for parameters
    fit_params.add('scale', value=scale,    min=scaleBound[0],  max=scaleBound[1]) #Initial guess for parameter
    fit_params.add('gain',  value=gain,     min=gainBound[0],   max=gainBound[1]) #Initial guess for parameter
    fit_params.add('smear', value=smear,    min=smearBound[0],  max=smearBound[1]) #Initial guess for parameter

    out = minimize(residual, fit_params, args=(model,), kws={'data': data}, epsfcn=stepSize)
    print(fit_report(out)) #Print results to screen

    if plot: #if plot==True, produce figure of align histograms
        # newBinRange = list(range(0, np.max(binRange)*2, np.int(binRange[1]-binRange[0]))) #changing 'binRange' to show more of the spectra for the plots.

        w_data = np.empty(len(data)) #Make weighting array same length as data-set
        w_data.fill(1/data_weight) #Fill array with 1/data_weight

        w_bg = np.empty(len(bg)) #Make weighting array same length as background data-set
        w_bg.fill(1/bg_weight) #Fill array with 1/bg_weight

        DataCounts,         DataBin_edges =         np.histogram(data, binRange, weights = w_data)
        ModelCounts,        ModelBin_edges =        np.histogram(gaussSmear(model, out.params['smear'].value)*out.params['gain'].value, binRange)
        StartModelCounts,   StartModelBin_edges =   np.histogram(model, binRange)
        bgHist,             bgBinEdges =            np.histogram(bg, binRange, weights = w_bg)

        plt.figure()
        plt.step(prodata.getBinCenters(DataBin_edges), DataCounts-bgHist, label='Data')
        plt.step(prodata.getBinCenters(StartModelBin_edges), StartModelCounts, label= 'Model start' )
        plt.step(prodata.getBinCenters(ModelBin_edges), ModelCounts*out.params['scale'].value, label= f'Model best values, scale={np.round(out.params["scale"].value, 3)}, gain={np.round(out.params["gain"].value, 3)}, smear={np.round(out.params["smear"].value, 3)}' )
        plt.vlines([binRange[0], binRange[-1]], ymin = 0, ymax = np.max(ModelCounts*out.params['scale'].value)*1.3, linestyle='dashed', color='black', label='Limits')
        plt.ylim([0, np.max(ModelCounts*out.params['scale'].value)*2])
        plt.ylabel('counts')
        plt.xlabel('arb. units')
        plt.legend()
        plt.show()

    results = {
        'scale':out.params['scale'].value,
        'gain':out.params['gain'].value,
        'smear':out.params['smear'].value,
        'scale_err':np.diag(np.sqrt(out.covar))[0],
        'gain_err':np.diag(np.sqrt(out.covar))[1],
        'smear_err':np.diag(np.sqrt(out.covar))[2],
        'redchi':out.redchi,
        'chisqr':out.chisqr
    }

    return results

def PMTGainAlign(model, data, scale, gain, bg = np.zeros(10), data_weight=1, bg_weight=1, scaleBound=[-np.inf, np.inf], gainBound=[-np.inf, np.inf], stepSize=1, binRange=list(range(0,500,2)), plot=False):
    """
    Function to align histogram of a model to a data set by offseting the bins and scaling the counts. 
    Function will return best paramters for the scale and gain based on a minimizing (scipy) algorithm.

    model..........the model to be varied. Needs to be unbinned.
    data...........the data set agaist which the model will be compared. Needs to be unbinned.
    bg.............background data set, will be subtracted from "data". optional
    scale..........initial guess for scale parameter
    gain...........initial guess for gain parameter
    data_weight....weight for histgram (see numpy.histogram())
    bg_weight......weight for histgram (see numpy.histogram())
    scaleBound.....boundary conditions for scale parameter, default -inf to +inf
    gainBound......boundary conditions for gain parameter, default -inf to +inf
    stepSize.......gives the step size for the minimizing algorithm to use
    binRange.......list defining the binning for the histograms
    plot...........set to True to produce plots of aligned histograms. Default: False

    --------------------------
    Nicholai Mauritzson
    2021-04-28
    """

    def residual(pars, x, data=None):
        """residual fuction used by the minimizing algorithm"""

        vals = pars.valuesdict()
        scale = vals['scale']
        gain = vals['gain']

        w_data = np.empty(len(data)) #Make weighting array same length as data-set
        w_data.fill(1/data_weight) #Fill array with 1/data_weight

        w_bg = np.empty(len(bg)) #Make weighting array same length as background data-set
        w_bg.fill(1/bg_weight) #Fill array with 1/bg_weight

        hist, binEdges = np.histogram(x * gain, binRange) #Binning and offsetting the model.
        model = hist * scale #Apply scaling factor

        dataHist,   dataBinEdges    = np.histogram(data, binRange, weights=w_data) #Binning the data set.
        bgHist,     bgBinEdges      = np.histogram(bg, binRange, weights=w_bg) #Binning the background data set.

        ans = ((dataHist-bgHist) - model).astype(np.float)  #subtracting the background and calculating the differences

        return ans

    fit_params = Parameters() #Container for parameters
    fit_params.add('scale', value=scale,    min=scaleBound[0],  max=scaleBound[1]) #Initial guess for parameter
    fit_params.add('gain',  value=gain,     min=gainBound[0],   max=gainBound[1]) #Initial guess for parameter

    out = minimize(residual, fit_params, args=(model,), kws={'data': data}, epsfcn=stepSize)
    print(fit_report(out)) #Print results to screen

    if plot: #if plot==True, produce figure of align histograms
        # newBinRange = list(range(0, np.max(binRange)*2, np.int(binRange[1]-binRange[0]))) #changing 'binRange' to show more of the spectra for the plots.

        w_data = np.empty(len(data)) #Make weighting array same length as data-set
        w_data.fill(1/data_weight) #Fill array with 1/data_weight

        w_bg = np.empty(len(bg)) #Make weighting array same length as background data-set
        w_bg.fill(1/bg_weight) #Fill array with 1/bg_weight

        DataCounts,         DataBin_edges =         np.histogram(data, binRange, weights = w_data)
        ModelCounts,        ModelBin_edges =        np.histogram(model*out.params['gain'].value, binRange)
        StartModelCounts,   StartModelBin_edges =   np.histogram(model, binRange)
        bgHist,             bgBinEdges =            np.histogram(bg, binRange, weights = w_bg)

        plt.figure()
        plt.step(prodata.getBinCenters(DataBin_edges), DataCounts-bgHist, label='Data')
        plt.step(prodata.getBinCenters(StartModelBin_edges), StartModelCounts, label= 'Model start' )
        plt.step(prodata.getBinCenters(ModelBin_edges), ModelCounts*out.params['scale'].value, label= f'Model best values, scale={np.round(out.params["scale"].value, 3)}, gain={round(out.params["gain"].value, 3)}' )
        plt.vlines([binRange[0], binRange[-1]], ymin = 0, ymax = np.max(ModelCounts*out.params['scale'].value)*1.3, linestyle='dashed', color='black', label='Limits')
        plt.ylim([0, np.max(ModelCounts*out.params['scale'].value)*2])
        plt.ylabel('counts')
        plt.xlabel('arb. units')
        plt.legend()
        plt.show()

    #TODO: Make dictionary with important outputs from minimize function
    results = {
        'scale':out.params['scale'].value,
        'gain':out.params['gain'].value,
        'scale_err':np.diag(np.sqrt(out.covar))[0],
        'gain_err':np.diag(np.sqrt(out.covar))[1],
        'redchi':out.redchi,
        'chisqr':out.chisqr
    }

    return results

def PMTGainAlignBirks(model, data, scale, gain, bg = np.zeros(10), data_weight=1, bg_weight=1, scaleBound=[-np.inf, np.inf], gainBound=[-np.inf, np.inf], stepSize=1, binRange=list(range(0,500,2)), plot=False):
    """
    Function to align histogram of a model to a data set by offseting the bins and scaling the counts. 
    Function will return best paramters for the scale and gain based on a minimizing (scipy) algorithm.

    model..........the model to be varied. Needs to be unbinned.
    data...........the data set agaist which the model will be compared. Needs to be unbinned.
    bg.............background data set, will be subtracted from "data". optional
    scale..........initial guess for scale parameter
    gain...........initial guess for gain parameter
    data_weight....weight for histogram
    bg_weight......weight for histogram
    scaleBound.....boundary conditions for scale parameter, default -inf to +inf
    gainBound......boundary conditions for gain parameter, default -inf to +inf
    stepSize.......gives the step size for the minimizing algorithm to use
    binRange.......list defining the binning for the histograms
    plot...........set to True to produce plots of aligned histograms. Default: False

    --------------------------
    Nicholai Mauritzson
    2022-07-08
    """

    def residual(pars, x, data=None):
        """residual fuction used by the minimizing algorithm"""

        vals = pars.valuesdict()
        scale = vals['scale']
        gain = vals['gain']

        R = data_weight/bg_weight #scaling value for background subtraction
        
        hist, binEdges = np.histogram(x * gain, binRange) #Binning and offsetting the model.
        model = hist * scale #Apply scaling factor

        dataHist,   dataBinEdges    = np.histogram(data, binRange) #Binning the data set.
        bgHist,     bgBinEdges      = np.histogram(bg, binRange) #Binning the background data set.

        ans = ((dataHist-bgHist*R) - model).astype(np.float)  #subtracting the background and calculating the differences

        return ans

    fit_params = Parameters() #Container for parameters
    fit_params.add('scale', value=scale,    min=scaleBound[0],  max=scaleBound[1]) #Initial guess for parameter
    fit_params.add('gain',  value=gain,     min=gainBound[0],   max=gainBound[1]) #Initial guess for parameter

    out = minimize(residual, fit_params, args=(model,), kws={'data': data}, epsfcn=stepSize)
    print(fit_report(out)) #Print results to screen

    if plot: #if plot==True, produce figure of align histograms
        R = data_weight/bg_weight

        DataCounts,         DataBin_edges =         np.histogram(data, binRange)
        ModelCounts,        ModelBin_edges =        np.histogram(model*out.params['gain'].value, binRange)
        StartModelCounts,   StartModelBin_edges =   np.histogram(model, binRange)
        bgHist,             bgBinEdges =            np.histogram(bg, binRange)

        plt.figure()
        plt.step(prodata.getBinCenters(DataBin_edges), DataCounts-bgHist*R, label='Data')
        plt.step(prodata.getBinCenters(StartModelBin_edges), StartModelCounts, label= 'Model start' )
        plt.step(prodata.getBinCenters(ModelBin_edges), ModelCounts*out.params['scale'].value, label= f'Model best values, scale={np.round(out.params["scale"].value, 3)}, gain={round(out.params["gain"].value, 3)}' )
        plt.vlines([binRange[0], binRange[-1]], ymin = 0, ymax = np.max(ModelCounts*out.params['scale'].value)*1.3, linestyle='dashed', color='black', label='Limits')
        plt.ylim([0, np.max(ModelCounts*out.params['scale'].value)*2])
        plt.ylabel('counts')
        plt.xlabel('arb. units')
        plt.legend()
        plt.show()

    results = {
        'scale':out.params['scale'].value,
        'gain':out.params['gain'].value,
        'scale_err':np.diag(np.sqrt(out.covar))[0],
        'gain_err':np.diag(np.sqrt(out.covar))[1],
        'redchi':out.redchi,
        'chisqr':out.chisqr
    }

    return results

def smearMinimization(model, data, bestScale, bestOffset, smear, binRange, smearBound, stepSize):
    """
    TODO: Finish function...

    """
    def residual(pars, x, data=None):
        """residual fuction used by the minimizing algorithm"""

        vals = pars.valuesdict()
        smear = vals['smear']
        
        x = x.apply(lambda j: gaussSmear(j, smear))

        hist, binEdges = np.histogram(x * bestOffset, binRange) #Binning and offsetting the model.
        model = hist * bestScale #Apply scaling factor

        # dataHist,   dataBinEdges    = np.histogram(data, binRange) #Binning the data set.
        # bgHist,     bgBinEdges      = np.histogram(bg, binRange) #Binning the background data set.

        ans = (data - model).astype(np.float)  #subtracting the background and calculating the differences

        return ans

    fit_params = Parameters() #Container for parameters
    fit_params.add('smear', value=smear, min=smearBound[0],  max=smearBound[1]) #Initial guess for parameter


    out = minimize(residual, fit_params, 'leastsq', args=(model,), kws={'data': data}, epsfcn=stepSize)
    print(fit_report(out)) #Print results to screen

    # model_y, model_x = np.histogram(model.apply(lambda x: gaussSmear(x, out.params['smear'].value))*bestOffset, binRange)
    # model_x = prodata.getBinCenters(model_x)
    # plt.step(model_x, model_y*bestScale)
    # plt.step(model_x, data)
    # plt.show()


    results = {
        'smear':out.params['smear'].value,
        'smear_err':np.diag(np.sqrt(out.covar)),
        'redchi':out.redchi,
        'chisqr':out.chisqr
    }
    return results


def PMTGainAlignNeutrons(model, data, scale, gain, scaleBound=[-np.inf, np.inf], gainBound=[-np.inf, np.inf], stepSize=1, binRange=np.arange(0,30000,100), plot=False):
    """
    Function to align model to already binned data by offsetting the bins and scaling the counts. 
    Function will return best paramters for the scale and gain based on a minimizing (scipy) algorithm.

    model..........unbinned data of the model to be varied.
    data...........unbinned data for the data to be compared with.
    scale..........initial guess for scale parameter
    gain...........initial guess for gain parameter
    scaleBound.....boundary conditions for scale parameter, default -inf to +inf
    gainBound......boundary conditions for gain parameter, default -inf to +inf
    stepSize.......gives the step size for the minimizing algorithm to use
    binRange.......numpy array of bins.
    plot...........set to True to produce plots of aligned histograms. Default: False

    --------------------------
    Nicholai Mauritzson
    2022-02-24
    """

    def residual(pars, x, data=None):
        """residual fuction used by the minimizing algorithm"""

        vals = pars.valuesdict()
        scale = vals['scale']
        gain = vals['gain']

        #Modifying the model with gain and scale
        hist, binEdges = np.histogram(x * gain, binRange) #Binning and offsetting the model.
        model = hist * scale #Apply scaling factor
        #Slicing the model with alignRange
        dataHist,   dataBinEdges    = np.histogram(data, binRange) #Binning the data set.

        ans = (dataHist - model).astype(np.float)  #subtracting the background and calculating the differences

        return ans

    fit_params = Parameters() #Container for parameters
    fit_params.add('scale', value=scale,    min=scaleBound[0],  max=scaleBound[1]) #Initial guess for parameter
    fit_params.add('gain',  value=gain,     min=gainBound[0],   max=gainBound[1]) #Initial guess for parameter

    out = minimize(residual, fit_params, args=(model,), kws={'data': data}, epsfcn=stepSize)
    print(fit_report(out)) #Print results to screen

    if plot: #if plot==True, produce figure of align histograms

        # DataCounts,         DataBin_edges =         np.histogram(data, binRange, weights = w_data)
        ModelCounts,        ModelBin_edges =        np.histogram(model*out.params['gain'].value, binRange)
        StartModelCounts,   StartModelBin_edges =   np.histogram(model, binRange)

        plt.figure()
        y_data, x_data = np.histogram(data, binRange)
        plt.step(prodata.getBinCenters(x_data),                 y_data,                                 label='Data')
        plt.step(prodata.getBinCenters(StartModelBin_edges),    StartModelCounts,                       label= 'Model start' )
        plt.step(prodata.getBinCenters(ModelBin_edges),         ModelCounts*out.params['scale'].value,  label= f'Model best values, scale={np.round(out.params["scale"].value, 3)}, gain={round(out.params["gain"].value, 3)}' )
        # plt.ylim([0, np.max(ModelCounts*out.params['scale'].value)*2])
        plt.ylabel('counts')
        plt.xlabel('arb. units')
        plt.legend()
        plt.show()

    #TODO: Make dictionary with important outputs from minimize function
    results = {
        'scale':out.params['scale'].value,
        'gain':out.params['gain'].value,
        'scale_err':np.diag(np.sqrt(out.covar))[0],
        'gain_err':np.diag(np.sqrt(out.covar))[1],
        'redchi':out.redchi,
        'chisqr':out.chisqr
    }

    return results

def gaussSmear(val, sig):
    """
    A gaussian smearing function. Takes data as input, applies a Gaussian randomization with standard diviation "std".

    val......input data on which to apply smearing, expect a single number.
    sig.......standard divation of Gaussian function for smearing.

    --------------------------
    Nicholai Mauritzson
    2021-04-13
    """
    return np.random.normal(1, sig) * val
    
def birksOptimization(pathData, pathSim, energy, detector, numBins, birksFolder, birksVal, ampCoeff, gainCoeff, smearCoeff, optRangeBins, lightYieldFactor=1, plotTest=False, plotFinal=False, optimizeOff=False):
        """
        Method to determine optimal Birks parameter for a given detector.
        This function will fid the optimal offset in gain for each Birks parameter given and calculates the optmial birks paramter where gainoffset = 1.
        Returns DataFrame with all offsets, errors, and Chi2 for each Birks parameter given.

        Parameters:
        pathData............path to TOF sliced neutron QDC data.
        pathSim.............path to simulated neutron data.
        energy..............the energy to use, in keV
        detector............name of detector, string
        numBins.............bins and range to use for data and sim
        birksFolder.........a list of names of folder for each Birks parameters to be used. Must match folder name 'birks####'
        birksVal............a list of values for each Birks parameters to be used. Must be same lenght as 'birksFolder'
        ampCoeff............a list of first guess amplitude offsets to use for each Birks parameter. Must be same lenght as 'birksFolder'
        gainCoeff...........a list of first guess gain offsets use for each Birks parameter. Must be same lenght as 'birksFolder'
        smearCoeff..........a list of resolution smearing values to use for each Birks parameter. Must be same lenght as 'birksFolder'
        optRangeBins........bins and range over which simulation should be otpimized to data.
        lightYieldFactor....light yield factor to apply to simulations. Default: 1 = no change
        plotTest............if True, will show a plot with initial amplitude and gain offsets applied before optimization.
        plotFinal...........if True, will show a plot with optimal amplitude and gain offsets for each Birks parameter.
        optimizeOff.........if True, optimization will not be made. 
        
        ------------------------------
        Nicholai Mauritzson
        01-09-2022
        """

        print('-------------------------------------')
        print('birksOptimization() running ...')
        print(f'- detector: {detector}')
        print(f'- energy: {energy} keV')
        print(f'- birksVal: {birksVal}')
        print(f'- lightYieldFactor = {lightYieldFactor}')
        print('-------------------------------------')

        #predefine variables
        kBs = np.empty(0)
        offset = np.empty(0)
        offset_err = np.empty(0)
        scale = np.empty(0)
        scale_err = np.empty(0)
        redChi = np.empty(0)
        smearCoeffOpt = np.empty(0)
        smearCoeffOpt_err = np.empty(0)

        #list of column names for simulation
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

        #Loop through all Birks parameters                
        for i in range(len(birksFolder)):
                print(f'working on kB = {birksVal[i]} mm/MeV')
                sim = pd.read_csv(f'{pathSim}/{detector}/neutrons/neutron_range/{int(energy)}keV/isotropic/birks{birksFolder[i]}/CSV_optphoton_data_sum.csv', names=names)

                #randomize binning
                y, x = np.histogram(sim['optPhotonSumQE'], bins=4096, range=[0, 4096]) 
                sim['optPhotonSumQE'] = prodata.getRandDist(x, y)

                #calibrate simulations
                sim['optPhotonSumQEcal'] = prodata.calibrateMyDetector(f'{pathData}/{detector}/Simcal/popt.npy', sim['optPhotonSumQE'], inverse=False)
        
                #apply smearing
                sim['optPhotonSumQEsmearcal'] = sim['optPhotonSumQEcal'].apply(lambda x: gaussSmear(x, smearCoeff[i]))
        
                #Applying light yield factor to simulation.
                sim['optPhotonSumQEsmearcal'] = sim['optPhotonSumQEsmearcal'] * lightYieldFactor
                
                
                #Load data
                nQDC       = np.load(f'{pathData}/{detector}/TOF_slice/QDC/keV{energy}/nQDC_LG.npy')
                randQDC    = np.load(f'{pathData}/{detector}/TOF_slice/QDC/keV{energy}/randQDC_LG.npy')
                nLength    = np.load(f'{pathData}/{detector}/TOF_slice/QDC/keV{energy}/nLength.npy')
                randLength = np.load(f'{pathData}/{detector}/TOF_slice/QDC/keV{energy}/randLength.npy')
                
                #calibrate data to MeVee
                nQDC = prodata.calibrateMyDetector(f'{pathData}/{detector}/Ecal/popt_LG.npy', nQDC)
                randQDC = prodata.calibrateMyDetector(f'{pathData}/{detector}/Ecal/popt_LG.npy', randQDC)


                #tets plot
                if plotTest:
                        countsSim, binsSim = np.histogram(sim['optPhotonSumQEsmearcal']*gainCoeff[i], bins=numBins)
                        binsSim = prodata.getBinCenters(binsSim)
                        R = nLength/randLength
                        countsBG, bins = np.histogram(randQDC, bins=numBins)
                        counts, bins = np.histogram(nQDC, bins=numBins)
                        bins = prodata.getBinCenters(bins)

                        plt.step(bins, counts-countsBG*R, label='data-BG')
                        plt.step(binsSim, countsSim*ampCoeff[i], label='sim')
                        plt.title(f'E = {energy} keV, kB = {birksVal[i]} mm/MeV')
                        plt.legend()
                        plt.show()
                if not(optimizeOff):
                        #optimize gain offset
                        resultCurrent = PMTGainAlignBirks(      model           = sim['optPhotonSumQEsmearcal'], 
                                                                data            = nQDC, 
                                                                bg              = randQDC,
                                                                scale           = ampCoeff[i],
                                                                gain            = gainCoeff[i],
                                                                data_weight     = nLength,
                                                                bg_weight       = randLength,
                                                                scaleBound      = [ampCoeff[i]*0.2, ampCoeff[i]*4],
                                                                gainBound       = [gainCoeff[i]*0.2, gainCoeff[i]*4],
                                                                stepSize        = 1e-4,
                                                                binRange        = optRangeBins,
                                                                plot            = plotFinal)

                        #Calculate optimal smearing
                        
                        #Bin data (data)
                        xData, yData = prodata.randomTOFSubtractionQDC( nQDC,
                                                                        randQDC,
                                                                        nLength, 
                                                                        randLength,
                                                                        optRangeBins, 
                                                                        plot=False)
                        #Energy calibrate simulation (model)
                        model = prodata.calibrateMyDetector(f'{pathData}/{detector}/Simcal/popt.npy', sim['optPhotonSumQE'], inverse=False)

                        resultSmear = smearMinimization(    model       = model, 
                                                            data        = yData, 
                                                            bestScale   = resultCurrent['scale'], 
                                                            bestOffset  = resultCurrent['gain'], 
                                                            smear       = smearCoeff[i], 
                                                            binRange    = optRangeBins, 
                                                            smearBound  = [smearCoeff[i]*0.75, smearCoeff[i]*1.25], 
                                                            stepSize    = 0.1)
                        

                        #append results from current Birks parameter
                        kBs = np.append(kBs, birksVal[i])
                        offset = np.append(offset, resultCurrent['gain'])
                        offset_err = np.append(offset_err, resultCurrent['gain_err'])                        
                        scale = np.append(scale, resultCurrent['scale'])
                        scale_err = np.append(scale_err, resultCurrent['scale_err'])
                        redChi = np.append(redChi, resultCurrent['redchi'])
                        smearCoeffOpt = np.append(smearCoeffOpt, resultSmear['smear'])
                        smearCoeffOpt_err = np.append(smearCoeffOpt_err, resultSmear['smear_err'])


        if not(optimizeOff):
            #Fit and plot to determine optimal Birks parameter
            popt, pcov = curve_fit(promath.linearFunc, offset, kBs)
            best_kB = round(promath.linearFunc(1, popt[0], popt[1]), 5) #calculate best Birks constant value

            popt_smear, pcov_smear = curve_fit(promath.linearFunc, kBs, smearCoeffOpt)
            best_smear = round(promath.linearFunc(best_kB, popt_smear[0], popt_smear[1]), 5) #calculate best Birks constant value

            # Make data frame and save to disk
            save = {'kB':         kBs,  
                    'offset':     offset,
                    'offset_err': offset_err,
                    'scale':      scale,
                    'scale_err':  scale_err,
                    'smear':      smearCoeffOpt,
                    'smear_err':  smearCoeffOpt_err,
                    'redchi':     redChi,
                    'optimal_kB': best_kB,
                    'optimal_smear':best_smear
                    }

            df = pd.DataFrame(data=save)
            df.to_csv(f'{pathData}/{detector}/birks/{energy}keV.csv', index=False)
            
            if plotFinal:
                plt.scatter(offset, kBs)
                plt.plot(np.arange(0.4, 1.7, 0.01), promath.linearFunc(np.arange(0.4, 1.7, 0.01), popt[0], popt[1]))
                plt.title(f'{detector}, energy={energy} keV: best k$_B$={best_kB} mm/MeV')
                plt.xlabel('offset values')
                plt.ylabel('k$_B$ values [mm/MeV]')
                plt.show()

            return df
           
def simProcessing(sim, pathData, detector, smearVal, numBins):
        """
        Helper routine for simualtions processing includes:
        - randomizing binning
        - calibrating simulations
        - smearing simulations
        - binning simulations
        
        sim..........pd.DataFrame containing the simulation
        pathData.....path to location of calibration
        detector.....string of which scintillator to calibrate, 'NE213A', 'EJ305', 'EJ321P' or 'EJ331'
        numBins......np.array with bins to use for binning the simulations
        """
        
        #randomize binning
        y, x = np.histogram(sim.optPhotonSumQE, bins=4096, range=[0, 4096]) 
        sim.optPhotonSumQE = prodata.getRandDist(x, y)

        #calibrate simulation
        sim.optPhotonSumQE = prodata.calibrateMyDetector(f'{pathData}/{detector}/Simcal/popt.npy', sim.optPhotonSumQE, inverse=False)

        #smear simulation
        sim.optPhotonSumQE = sim.optPhotonSumQE.apply(lambda x: gaussSmear(x, smearVal)) 

        #binning simulations and applying gain offset
        ySimCurrent, xSimCurrent = np.histogram(sim.optPhotonSumQE, bins=numBins)
        xSimCurrent = prodata.getBinCenters(xSimCurrent)
        
        return xSimCurrent, ySimCurrent
