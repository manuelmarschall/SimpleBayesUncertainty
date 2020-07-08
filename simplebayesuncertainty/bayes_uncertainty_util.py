'''
 License
 =======
 
 copyright Gerd Wuebbeler, Manuel Marschall (PTB) 2020
 
 This software is licensed under the BSD-like license:

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:
 
 1. Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.
 2. Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in
    the documentation and/or other materials provided with the distribution.

 DISCLAIMER
 ==========
 This software was developed at Physikalisch-Technische Bundesanstalt
 (PTB). The software is made available "as is" free of cost. PTB assumes
 no responsibility whatsoever for its use by other parties, and makes no
 guarantees, expressed or implied, about its quality, reliability, safety,
 suitability or any other characteristic. In no event will PTB be liable
 for any direct, indirect or consequential damage arising in connection

Using this software in publications requires citing the following
 Paper: https://doi.org/10.1088/1681-7575/aba3b8
'''
from __future__ import (division, print_function, absolute_import)
import numpy as np
from itertools import product                               # cartesian product of sets
from scipy.integrate import cumtrapz                        # trapezoidal rule for integration
from scipy.stats import t as student_t                      # student-t distribution
from scipy.stats import gaussian_kde                        # kernel density estimation
from scipy.stats import norm, gamma                         # normal and gamma distribution  
from matplotlib import rc                                   # plot parameter
rc('font', family='serif')
rc('font', size=12)
# rc('text', usetex=True)        
import matplotlib.pyplot as plt                             # plot environment

def nig_prior(U_y0, sig0):
    """
    Returns parameter of normal inverse gamma prior according to 
    the choice in the paper. See Section 2.3.

    Arguments:
        U_y0 {float} -- uncertainty of parameter
        sig0 {float} -- standard deviation of measurement device

    Returns:
        tuple -- (\lambda, a, b)
    """    
    a = 1
    llambda = (0.28*U_y0/sig0)**2
    b = (sig0**2)/1.44
    return llambda, a, b

def inverse_transform_sampling(x, pd, cnt):
    """
    Implementation of inverse transform sampling using 
    interpolation of the inverse CDF

    Arguments:
        x {list} -- density nodes
        pd {list} -- density values
        cnt {int} -- number of interpolation nodes

    Returns:
        list -- random samples
    """    
    cdf = cumtrapz(pd, x=x)
    cdf = cdf/cdf[-1] 
    # numpy unique returns unique values in sorted order
    # therefore consider only the indices
    _ ,ia = np.unique(cdf, return_index=True)
    # then, sort the indices to obtain the original order again
    cdf_unique = cdf[np.sort(ia)]
    x_unique=x[np.sort(ia)]
    return np.interp(np.random.rand(cnt), cdf_unique, x_unique)


def bayes_uncertainty(X, y0, U_y0, sig0, alpha, B_S1_samples, n_samples, bootstrap=1):
    """
    Implementation of the simple Bayesian approach according to paper.
    Returns a tuple with three lists
        - Y_samples - posterior samples of unknown
        - B_samples - posterior samples of type B quantity B
        - phi_samples - samples of variance of measurement device

    Arguments:
        X {list} -- measurement data
        y0 {float} -- prior mean of unknown
        U_y0 {float} -- prior uncertainty of unknown
        sig0 {float} -- std. deviation of measurement device
        alpha {float} -- influence of measurement device (alpha=1)
        B_S1_samples {list} -- samples of type B quantity B
        n_samples {int} -- number of returned samples

    Keyword Arguments:
        bootstrap {int} -- number of sub-sets to estimate model error (default: 1)

    Returns:
        tuple -- Y_samples, B_samples, phi_samples
    """    

    assert isinstance(bootstrap, int)
    assert bootstrap >= 1

    # Evaluate Type A data
    n = len(X)
    xm = np.mean(X)
    s2 = np.var(X, ddof=1)

    # Calculate NIG prior parameters
    a = 1
    llambda=(0.28*U_y0/sig0)**2
    b=(sig0**2)/1.44
    
    # Create prior PDF pi(B) from B_S1_samples
    ## 
    # histogram returns the values of the pdf at the bin, normalised such that the integral over the range is 1
    # and the edge positions of the bins (n+1)
    ##
    p_B, x_B = np.histogram(B_S1_samples, bins="fd", density=True)
    x_B = 0.5*(x_B[:-1]+x_B[1:])

    # interpolate the pdf and extend left and right with 0
    prior_B_pdf = lambda B: np.interp(B, x_B, p_B, left=0, right=0) 
    mB_S1_samples=np.mean(B_S1_samples)

    # Define functions
    al2 = alpha**2 
    lmn = llambda*n
    Yhat = lambda B: (al2/(al2+lmn))*(y0+lmn*(alpha*xm+B)/al2)
    psi = lambda B: (llambda*al2/((al2+lmn)*(n+2*a)))*((n-1)*s2+2*b+(n/(al2+lmn))*(y0-(alpha*xm+B))**2)

    posterior_B = lambda B: (prior_B_pdf(B)*(psi(B)**(-(n+2*a)/2)))

    # Find suitable grid for B
    ngrid = 10000
    B_hat= y0-alpha*xm
    B_scale = (al2+lmn)*((n-1)*s2+2*b)/(n*(n-1+2*a))
    Bgrid_mean = 0.5*(mB_S1_samples+B_hat)
    Bgrid_u = np.sqrt(np.var(B_S1_samples, ddof=1)+B_scale+(mB_S1_samples-B_hat)**2)
    Bgrid1 = np.linspace(Bgrid_mean-5*Bgrid_u, Bgrid_mean+5*Bgrid_u, ngrid)
    hlp = posterior_B(Bgrid1)
    ind = np.argwhere(hlp>1e-10*max(hlp))[:, 0]
    Bgrid = np.linspace(Bgrid1[ind[0]], Bgrid1[ind[-1]], ngrid)

    # Monte-Carlo sampling
    # (i)  : sample from marginal posterior of summarized Type B effect B
    B_samples = inverse_transform_sampling(Bgrid, posterior_B(Bgrid), n_samples)
    # (ii) : sample from marginal posterior of the measurand Y conditional on B
    Y_samples = Yhat(B_samples)+np.sqrt(psi(B_samples))*np.random.standard_t(n+2*a, n_samples)
    # (iii): sample from marginal posterior of the variance parameter phi conditional on B(optional)
    a_cond = a+n/2
    b_cond = b+((n-1)*s2+(n/(al2+lmn))*(y0-(alpha*xm+B_samples))**2)/2
    phi_samples = b_cond / np.random.gamma(a_cond, 1, n_samples)

    if bootstrap > 1:
        print("  Start bootstrapping with {} x {:.2e} sub-samples".format(bootstrap, len(B_S1_samples)))
        res = {
            "B": [],
            "Y": [],
            "phi": []
        }
        for _ in range(bootstrap):
            # print("  run bootstrap {}/{}".format(lia+1, bootstrap))
            sub_B_samples = np.random.choice(B_S1_samples, size=len(B_S1_samples), replace=True)
            curr_Y_samples, curr_B_samples, curr_phi_samples = bayes_uncertainty(X, y0, U_y0, sig0, alpha, sub_B_samples, n_samples, bootstrap=1)
            res["B"].append(curr_B_samples)
            res["Y"].append(curr_Y_samples)
            res["phi"].append(curr_phi_samples)
        return Y_samples, B_samples, phi_samples, res
    return Y_samples, B_samples, phi_samples

def tlocscale(x, mu, scale2, nu):
    """
    shifted and scaled student-t pdf

    Arguments:
        x {list} -- nodes to evaluate density at
        mu {float} -- shift
        scale2 {float} -- scale
        nu {int} -- degrees of freedom

    Returns:
        list -- evaluations of pdf
    """    
    scale=np.sqrt(scale2)
    return student_t.pdf((x-mu)/scale,nu)/scale

def plot_result_phi(phi_samples, unc_0, sig0,
                    xlim=None,
                    n_bins=200,
                    output="figure2.pdf",
                    interactive=False,
                    use_kde=False):
    """
    Helper function to plot the posterior results of phi

    Arguments:
        phi_samples {list or array} -- posterior samples of phi
        unc_0 {float} -- uncertainty of measurand
        sig0 {float} -- uncertainty of measurement device

    Keyword Arguments:
        xlim {tuple} -- bounds to plot in (default: {None})
        n_bins {int} -- number of bins for histogram (default: {200})
        output {str} -- path and name of output file (default: {"figure2.pdf"})
        interactive {bool} -- flag to hold the image (default: {False})
        use_kde {bool} -- flag to use kernel density estimation (default: {False})
    """

    _, a, b = nig_prior(unc_0, sig0)   # Note that lambda is a Python specific keyword
    # define the inverse gamma pdf
    invgampdf = lambda _x, _a, _b: (gamma.pdf(1/_x, _a, scale=1/_b)/(_x**2))
    # reconstruct the pdf from samples of phi
    m_phi = np.mean(phi_samples)
    u_phi = np.std(phi_samples, ddof=1)
    x_grid = np.linspace(np.max([0, m_phi-6*u_phi]), m_phi+6*u_phi, n_bins)
    x_phi, p_phi = get_pdf_from_samples(phi_samples, method="kde" if use_kde else "hist", bins=x_grid)

    fig = plt.figure()
    plt.plot(np.sqrt(x_phi), 2*np.sqrt(x_phi)*invgampdf(x_phi, a, b), '--b', label="Prior")
    plt.plot(np.sqrt(x_phi), 2*np.sqrt(x_phi)*p_phi, '-b', label="Posterior")
    plt.xlabel("sigma=sqrt(phi)", fontsize=14)
    plt.ylabel("Probability density", fontsize=14)
    if xlim is not None:
        plt.xlim(xlim)

    plt.legend(fontsize=12)
    fig.tight_layout()
    # plt.show(block=False if not interactive else True)
    fig.savefig(output, dpi=300, format="pdf")

def plot_result(bayes_samples, 
                mean_0,
                unc_0,
                sig0,
                s1_samples=None,
                mean_gum=None,
                u_gum=None,
                title="Example",
                xlabel="Y",
                xlim=None,
                n_bins=200,
                output="figure.pdf",
                hold=False,
                interactive=False,
                use_kde=False):
    """
    plots the resulting posterior to a file

    Arguments:
        bayes_samples {list or array} -- posterior samples
        mean_0 {float} -- mean of measurand
        unc_0 {float} -- uncertainty of measurand
        sig0 {float} -- uncertainty of measurement device

    Keyword Arguments:
        s1_samples {list or array} -- GUM S1 samples (default: {None})
        mean_gum {float} -- mean by GUM (default: {None})
        u_gum {float} -- uncertainty by GUM (default: {None})
        title {str} -- title of figure (default: {"Example"})
        xlabel {str} -- x label string (default: {"Y"})
        xlim {tuple} -- bounds to plot in (default: {None})
        n_bins {int} -- number of bins in histogram (default: {200})
        output {str} -- path and name of figure (default: {"figure.pdf"})
        hold {bool} -- flag to hold the image (experimental) (default: {False})
        interactive {bool} -- flag to hold the image (default: {False})
        use_kde {bool} -- flag to use kernel density estimation (default: {False})
    """

    llambda, a, b = nig_prior(unc_0, sig0)   # Note that lambda is a Python specific keyword
    fig = plt.figure()

    # determine plotting range
    mean = np.mean(bayes_samples)
    unc = np.std(bayes_samples, ddof=1)
    x_grid = np.linspace(mean-6*unc, mean+6*unc, n_bins)
    x_bayes, p_bayes = get_pdf_from_samples(bayes_samples, method="kde" if use_kde else "hist", bins=x_grid)
    if s1_samples is not None:
        p_s1, _ = np.histogram(s1_samples, np.linspace(mean-6*unc, mean+6*unc, n_bins), density=True)
        plt.plot(x_bayes, p_s1, '-g', label="GUM-S1")

    # prior of Y is a scaled and shifted student-t distribution
    plt.plot(x_bayes, tlocscale(x_bayes, mean_0, llambda*b/a, 2*a), '--b', label="Prior")
    plt.plot(x_bayes, p_bayes, '-b', label="Posterior")
    if mean_gum is not None and u_gum is not None:
        plt.plot(x_bayes, norm.pdf(x_bayes, loc=mean_gum, scale=u_gum), '-r', label="GUM")

    plt.legend(fontsize=12)
    if xlim is not None:
        plt.xlim(xlim)
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel("Probability density", fontsize=14)

    fig.tight_layout()
    # if hold:
    #     plt.show(block=False if not interactive else True)
    fig.savefig(output, dpi=300, format="pdf")

def plot_sensitivity(bayes_samples,
                     x, mean_0, unc_0, sig0, 
                     alpha, B_S1_samples, n_samples,
                     xlim=None,
                     xlabel="", output="figure3.pdf",
                     interactive=False, 
                     use_kde=False):
    """
    Helper function to plot the results of the sensitivity analysis.

    Arguments:
        bayes_samples {list or array} -- posterior samples 
        x {list or array} -- measurements
        mean_0 {float} -- mean of measurand
        unc_0 {float} -- uncertainty of measurand
        sig0 {float} -- measurement device uncertainty
        alpha {float} -- model parameter Y = alpha X + B
        B_S1_samples {list or array} -- samples of B
        n_samples {int} -- number of samples to create for every bootstrap

    Keyword Arguments:
        xlim {tuple} -- bounds to plot in (default: {None})
        xlabel {str} -- x label string (default: {""})
        output {str} -- path and name of output file (default: {"figure3.pdf"})
        interactive {bool} -- flag to hold image (default: {False})
        use_kde {bool} -- flag to use kernel density estimation (default: {False})
    """
    # Sensitivity analysis
    dlt = 0.1
    delta_U_y0 = np.array([1, -1])*dlt + 1
    delta_sig0 = np.array([1, -1])*dlt + 1

    mean = np.mean(bayes_samples)
    unc = np.std(bayes_samples, ddof=1)
    x_grid = np.linspace(mean-6*unc, mean+6*unc, 200)
    x_bayes, p_bayes = get_pdf_from_samples(bayes_samples, method="kde" if use_kde else "hist", bins=x_grid)
    fig = plt.figure()
    plt.plot(x_bayes, p_bayes, '-b', linewidth=1.5, label="orig. Posterior")
    for d_Uy0, d_sig0 in product(delta_U_y0, delta_sig0):
        Y_samples_sens, _, _ = bayes_uncertainty(x, mean_0, d_Uy0*unc_0, d_sig0*sig0, alpha, B_S1_samples, n_samples)
        _, p_Y_sens = get_pdf_from_samples(Y_samples_sens, method="kde" if use_kde else "hist", bins=x_grid)
        plt.plot(x_bayes, p_Y_sens, alpha=0.5, label="Uy0*{}, sig0*{}".format(d_Uy0, d_sig0))
        
    if xlim is not None:
        plt.xlim(xlim)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel("Probability density", fontsize=14)
    plt.legend(fontsize=12)
    
    fig.tight_layout()
    # plt.show(block=False if not interactive else True)
    fig.savefig(output, dpi=300, format="pdf")

def import_file(file_path):
    """
    Utility function to import samples from file.
    Expected format: newline separated floats.

    Example: 
        12.3342
        11.3123
        1.34e+1

    Arguments:
        file_path {str} -- name and path to file

    Returns:
        list -- samples

    TODO: appropriate error handling
    """    
    import os 
    assert os.path.exists(file_path)
    retval = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            retval.append(float(line))
    retval = np.array(retval)
    return retval

def export_samples(samples, file_path):
    """
    Utility function to export samples to file.

    Arguments:
        samples {list} -- samples to export
        file_path {str} -- name and path to file
    
    Returns:
        None
    TODO: appropriate error handling
    """    
    with open(file_path, 'w') as f:
        for sample in samples:
            f.write(str(sample) + "\n")


def get_pdf_from_samples(samples, method="kde", *args, **kwargs):
    """
    Method to construct a pdf from given samples.
    The employed method can be chosen, default kernel density estimation using Gaussian kernels
    with Scott's  bandwith selection.
    TODO: Consider Silverman bandwith selection. 
    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html for details
    Alternatively, histograms can be chosen.
    Return type depends on input values.

    Arguments:
        samples {list} -- list of samples 
       
    Keyword Arguments:
        method {str} -- methods string {"kde", "hist"} (default: {"kde"})

    Returns:
        callable or (list, list) -- PDF as function or (x, y) values
    """
    used_method = "kde"
    bins = kwargs.pop("bins", None)
    if method == "hist":
        assert bins is not None
        used_method = "hist"

    if used_method == "kde":
        kde = gaussian_kde(samples, **kwargs)
        if bins is not None and not isinstance(bins, str):
            return bins, kde.evaluate(bins)
        retval = lambda _x: kde.evaluate(_x)
        return retval
    elif used_method == "hist":
        p, x = np.histogram(samples, bins=bins, density=True)
        x = 0.5*(x[:-1] + x[1:])
        return x, p
    else:
        raise ValueError("unknown density estimation method: {}".format(method))

def analyse_bootstrap_res(res):
    """
    Processes the result of the bootstrap algorithm by estimating the uncertainty
    for the given quantity bootstraps.

    Arguments:
        res {dict} -- dictionary containing bootstrap results

    Returns:
        dict -- estimated uncertainty over bootstrap ensembles
    """
    assert len(res["Y"]) == len(res["B"])
    assert len(res["Y"]) == len(res["phi"])
    lb = 2.5
    ub = 97.5
    mean_y = []
    std_y = []
    mean_b = []
    std_b = []
    mean_phi = []
    std_phi = []
    lb_y = []
    lb_b = []
    lb_phi = []
    ub_y = []
    ub_b = []
    ub_phi = []
    for lia in range(len(res["Y"])):
        mean_y.append(np.mean(res["Y"][lia]))
        mean_b.append(np.mean(res["B"][lia]))
        mean_phi.append(np.mean(res["phi"][lia]))

        std_y.append(np.std(res["Y"][lia], ddof=1))
        std_b.append(np.std(res["B"][lia], ddof=1))
        std_phi.append(np.std(res["phi"][lia], ddof=1))
        
        lb_y.append(np.percentile(res["Y"][lia], lb))
        lb_b.append(np.percentile(res["B"][lia], lb))
        lb_phi.append(np.percentile(res["phi"][lia], lb))
        
        ub_y.append(np.percentile(res["Y"][lia], ub))
        ub_b.append(np.percentile(res["B"][lia], ub))
        ub_phi.append(np.percentile(res["phi"][lia], ub))
    retval = {
        "u_m_y": np.std(mean_y, ddof=1),
        "u_m_b": np.std(mean_b, ddof=1),
        "u_m_phi": np.std(mean_phi, ddof=1),
        "u_u_y": np.std(std_y, ddof=1),
        "u_u_b": np.std(std_b, ddof=1),
        "u_u_phi": np.std(std_phi, ddof=1),
        "u_lb_y": np.std(lb_y, ddof=1),
        "u_lb_b": np.std(lb_b, ddof=1),
        "u_lb_phi": np.std(lb_phi, ddof=1),
        "u_ub_y": np.std(ub_y, ddof=1),
        "u_ub_b": np.std(ub_b, ddof=1),
        "u_ub_phi": np.std(ub_phi, ddof=1)
    }
    
    # print("  Y: \n    uncertainty of mean: {:.5f} \n    uncertainty of std.: {:.5f}".format(u_m_y, u_u_y))
    # print("  B: \n    uncertainty of mean: {:.5f} \n    uncertainty of std.: {:.5f}".format(u_m_b, u_u_b))
    # print("  Phi: \n    uncertainty of mean: {:.5f} \n    uncertainty of std.: {:.5f}".format(u_m_phi, u_u_phi))
    return retval

def print_results(res, bootstrap_res=None):
    """
    Simple summary method that creates an output table containing the important
    statistical quantities of the given sample sets

    Arguments:
        res {dict} -- Result dictionary containing the keys {y, phi, b} 
                      with the posterior samples of bayes_uncertainty

    Keyword Arguments:
        boot_res {dict} -- bootstrap results (default: {None})

    Returns 
        str -- summary of results as printable string
    """
    dash = '-'*(4*15+15)
    mean = "mean"
    std = "uncertainty"
    lb = "2.5% perc."
    ub = "97.5% perc."
    b_str = "B (input)"
    y_str = "Y (measurand)"
    phi_str = "phi (device)"

    retval_str = dash + "\n"
    retval_str += '{:<15s}{:>15s}{:>15s}{:>15s}{:>15s}\n'.format("Posterior", mean, 
                                                                 std, lb, ub)
    retval_str += dash + "\n"
    retval_str += '{:<15s}{:>15.2e}{:>15.2e}{:>15.2e}{:>15.2e}\n'.format(y_str, np.mean(res["y"]), 
                                                                         np.std(res["y"], ddof=1), 
                                                                         np.percentile(res["y"], 2.5), 
                                                                         np.percentile(res["y"], 97.5))
    retval_str += '{:<15s}{:>15.2e}{:>15.2e}{:>15.2e}{:>15.2e}\n'.format(b_str, np.mean(res["b"]), 
                                                                         np.std(res["b"], ddof=1), 
                                                                         np.percentile(res["b"], 2.5), 
                                                                         np.percentile(res["b"], 97.5))
    retval_str += '{:<15s}{:>15.2e}{:>15s}{:>15.2e}{:>15.2e}\n'.format(phi_str, np.mean(res["phi"]), 
                                                                       "N/A", np.percentile(res["phi"], 2.5), 
                                                                       np.percentile(res["phi"], 97.5))

    retval_str += dash + "\n"
    if bootstrap_res is not None:
        retval_str += dash + "\n"
        retval_str += '{:>15s}\n'.format("Standard deviation")        
        retval_str += '{:<15s}{:>15s}{:>15s}{:>15s}{:>15s}\n'.format("of Bootstrapping", mean, 
                                                                     std, lb, ub)
        retval_str += dash + "\n"
        retval_str += '{:<15s}{:>15.2e}{:>15.2e}{:>15.2e}{:>15.2e}\n'.format(y_str, bootstrap_res["u_m_y"], 
                                                                             bootstrap_res["u_u_y"], 
                                                                             bootstrap_res["u_lb_y"], 
                                                                             bootstrap_res["u_ub_y"])
        retval_str += '{:<15s}{:>15.2e}{:>15.2e}{:>15.2e}{:>15.2e}\n'.format(b_str, bootstrap_res["u_m_b"], 
                                                                             bootstrap_res["u_u_b"], 
                                                                             bootstrap_res["u_lb_b"], 
                                                                             bootstrap_res["u_ub_b"])
        retval_str += '{:<15s}{:>15.2e}{:>15s}{:>15.2e}{:>15.2e}\n'.format(phi_str, bootstrap_res["u_m_phi"], 
                                                                           "N/A", bootstrap_res["u_lb_phi"], 
                                                                           bootstrap_res["u_ub_phi"])
        retval_str += dash + "\n"
    return retval_str

def generate_mass_example_samples(n_samples, path):
    """
    internal function to create Monte-Carlo samples as described in 
    the mass example
    This function was used to generate mass/B_samples_init.dat

    Arguments:
        n_samples {int} -- number of samples
        path {str} -- path to store the resulting samples

    Returns 
        None
    """
    # samples for B (input)     
    n_B_samples = n_samples        # number of samples
    B_samples=5+22.5*np.random.randn(n_B_samples) + \
                0+np.sqrt(12)*15/np.sqrt(3)*(np.random.rand(n_B_samples)-0.5) + \
                0+np.sqrt(12)*10/np.sqrt(3)*(np.random.rand(n_B_samples)-0.5) + \
                0+ np.sqrt(12)*10/np.sqrt(3)*(np.random.rand(n_B_samples)-0.5)
    export_samples(B_samples, path)

def generate_mass_example_measurements(path):
    """
    internal function to create the measurements as described in 
    the mass example
    This function was used to generate mass/measurements.dat

    Arguments:
        path {str} -- path to store the resulting measurements

    Returns 
        None
    """
    mraw=np.array([10,20,25,15,25,50,55,20,25,45,40,20]) #difference reading
    seq= np.array([1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1], dtype=np.bool)    #1=standard, 0=unknown mass
    # observed differences (n=3)
    # x = E[20-10, 25-15], E[50-25, 55-20], E[45-25, 40-20] = [10, 30, 20]
    x = np.mean(np.reshape(mraw[np.logical_not(seq)]-mraw[seq],[2, 3], order="F"), axis=0)
    export_samples(x, path)
