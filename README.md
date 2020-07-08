# License
 
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



This repository contains the python code provided in the paper "A simple method for Bayesian uncertainty evaluation in linear models".

## Motivation
The paper provides a simple and easy method to employ the Bayesian paradigm for typical applications in metrology. 
The suggested choice for the prior, the sampling methods and the analysis of the resulting posterior is covered in this repository.

### Installation and running the code 

To run the script one needs a $\geq$ python 3.6 installation with the default packages
* numpy
* scipy
* matplotlib

Installation guides for Linux, Windows and Mac can be found here: https://realpython.com/installing-python/

## Quick guide for Windows:

1. Download Python https://www.python.org/downloads/release/python-382/ (bottom of the page: "Windows x86-64 executable installer") 
2. Install Python using the installer and check "Add Python x.x to Path"
3. Run a terminal, e.g. CMD
4. Check the installation by typing

	```
	python
	```
   a command prompt should appear such as 

	```
	C:\Users\Marschall\Projects\simple_bayes>python
	Python 3.8.2 (tags/v3.8.2:7b3ab59, Feb 25 2020, 22:45:29) [MSC v.1916 32 bit (Intel)] on win32
	Type "help", "copyright", "credits" or "license" for more information.
	>>>
	```


5. Close the Python prompt using
	```
	exit()
	```
6. Install dependencies
	```
	python -m pip install numpy scipy matplotlib
	```
7. Navigate to downloaded or cloned files of the repo using `cd`
8. Run the example program
	```
	python generic_example.py
	```
   or
   ```
	python mass_example.py
	```

### Implementation details

The files `mass_example.py` and `generic_example.py` contain the examples from the paper and the main functionality is provided in the `bayes_uncertainty_util.py` package. 
Here, most of the routines are collected and called from the script files. 
In the directories `mass/` and `generic/` we provide the corresponding measurements and samples of B to repeat the experiments from the paper. 

### Contact

In case of problems or questions please contact `manuel.marschall@ptb.de` or `gerd.wuebbeler@ptb.de`.
