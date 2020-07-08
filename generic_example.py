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

Using this software in publications requires citing the following paper

G. Wübbeler, M. Marschall and C. Elster (2020). A simple method for Bayesian uncertainty evaluation in linear models. 
Metrologia   https://doi.org/10.1088/1681-7575/aba3b8
'''
from __future__ import (division, print_function, absolute_import)
from simplebayesuncertainty.bayes_uncertainty_util import (import_file, 
               bayes_uncertainty, plot_result, plot_result_phi, plot_sensitivity, 
               export_samples, analyse_bootstrap_res, print_results)
import numpy as np
import time
import os

# #### Definition of the measurement model
alpha = 1               # model parameter
# measurement data (provided inline or by file)
x = [1, 2]
# samples for B (input). Here, N(2, 2), alternatively provide a file  
B_samples = np.random.normal(loc=2, scale=2, size=int(1e7))
# prior for Y (measurand)
y0 = 0                  # mean
U_y0 = 10               # half width of 95 percent interval
# prior for repeatability
sig0 = 1                # measurement device uncertainty
# #### Additional parameter 
n_samples = int(1e6)    # number of output samples
bootstrap_iteration = 10 # apply bootstrapping to B-samples 
#                       # bootstrap_iteration = 1 -> no sub-sampling
#                       # bootstrap_iteration > 1 -> apply subsampling
use_kde = False         # more advanced density estimators 
                        # for smooth plots
# WARNING: kernel density estimates can result in longer runtime 
# when `n_samples` is large.
# The following paths must exists. So please create them in advance.
plot_path = "generic/"  # path to save the plots ( "" = current directory)
export_path = "generic/" # path to store the resulting i.i.d. samples
# ####
# #### START Bayesian uncertainty procedure
# WARNING: Do not edit the code below unless you know what you are doing.
# ####
if not os.path.exists(plot_path):
    raise ValueError("Directory {} does not exist. Create it first!".format(plot_path))
if not os.path.exists(export_path):
    raise ValueError("Directory {} does not exist. Create it first!".format(export_path))

if bootstrap_iteration > 50:
   raise ValueError("Taking more than 50 bootstrap iteration is not recommended")

res_str = "Run Bayes uncertainty estimation.\n"
print("Run Bayes uncertainty estimation.")
start = time.time()
if bootstrap_iteration > 1:
   Y_samples, new_B_samples, phi_samples, res = bayes_uncertainty(x, y0, U_y0, sig0, alpha, B_samples, n_samples, bootstrap=bootstrap_iteration)
   bootstrap_res = analyse_bootstrap_res(res)
   res_str += "  Start bootstrapping with {} x {:.2e} sub-samples\n".format(bootstrap_iteration, len(B_samples))
else:
   Y_samples, new_B_samples, phi_samples = bayes_uncertainty(x, y0, U_y0, sig0, alpha, B_samples, n_samples, bootstrap=1)
   bootstrap_res = None
duration = time.time() - start
res_str += "Bayes uncertainty estimation done.\n"
print("Bayes uncertainty estimation done.")
   
sampling_res = {
   "y": Y_samples,
   "phi": phi_samples,
   "b": new_B_samples
}
p_res_str = print_results(sampling_res, bootstrap_res=bootstrap_res)
res_str += p_res_str
print(p_res_str)

res_str += "{} i.i.d. samples were created in {:.2f} seconds\n".format(n_samples, duration)
print("{} i.i.d. samples were created in {:.2f} seconds".format(n_samples, duration))
# Write plots
plot_result(Y_samples, y0, U_y0, sig0, output=plot_path + "y.pdf", use_kde=use_kde)
plot_result_phi(phi_samples, U_y0, sig0, output=plot_path + "phi.pdf", use_kde=use_kde)
plot_sensitivity(Y_samples, x, y0, U_y0, sig0, alpha, B_samples, n_samples, output=plot_path + "sensitivity.pdf", use_kde=use_kde)
res_str += "Plots written to:          {:s}\n".format(plot_path)
print("Plots written to:          {:s}".format(plot_path))
# Write samples
export_samples(phi_samples, export_path + 'phi_samples.dat')
export_samples(Y_samples, export_path + 'Y_samples.dat')
export_samples(new_B_samples, export_path + 'B_samples.dat')
res_str += "Samples written to:        {:s}\n".format(export_path)
print("Samples written to:        {:s}".format(export_path))

res_str += "Posterior details written: {:s}\n".format(export_path + "posterior.txt")
with open(export_path + "posterior.txt", "w") as f:
   f.write(res_str)
print("Posterior details written: {:s}".format(export_path + "posterior.txt"))
