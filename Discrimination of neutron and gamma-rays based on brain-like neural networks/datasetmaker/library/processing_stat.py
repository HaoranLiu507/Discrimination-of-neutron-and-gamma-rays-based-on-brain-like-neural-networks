#Statistical methods

def chi2(obs, exp, obs_var):
    """
    Chisquare calculator for numerical data.
    obs.......observed values (list)
    exp.......expected value (fit-values or model-values) (list)
    obs_var...variance (error) in observed values 'obs' (list)
    """
    #calulate chi2
    chi2 = np.sum( ((obs-exp)/obs_var)**2 )
    print(f'...promath.chi2()...')
    print(f'Chi2 = {chi2}')
    return chi2

def chi2red(obs, exp, obs_var, npar):
    """
    The reduced chi-squared function Chi2/N.D.F.
    obs.......observed values (list)
    exp.......expected value (fit-values or model-values) (list)
    obs_var...variance (error) in observed values 'obs' (list)
    npar......number of fitted parameters (int)

    More info:  https://en.wikipedia.org/wiki/Reduced_chi-squared_statistic
                https://arxiv.org/pdf/1012.3754.pdf
    """
    #calulate chi2
    chi2 = np.sum( ((obs-exp)/obs_var)**2 )
        
    #calculate number of degrees of freedom
    ndeg = len(obs)-npar
    
    print(f'...promath.chi2red()...')
    print(f'Chi2 = {chi2}')
    print(f'ndeg = {ndeg}')
    print(f'Chi2/ndeg = {chi2/ndeg}')

    return chi2/ndeg