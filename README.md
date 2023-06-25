# min-formulas
## General Description
This python module contains functions to recalculate mineral atom formulas from chemical analyses given in oxide weight percent. 

Input data are provided as a ndarray of mineral analyses in wt% oxides, with the following order of columns: SiO2 TiO2 Al2O3 Cr2O3 FeO MnO MgO CaO Na2O K2O P2O5.

Functions for most of the common rock-forming minerals are available, including olivine, orthopyroxene, clinopyroxene, amphibole, garnet, feldspar, biotite and staurolite. More details about cation site occupancy models for each minerals are provided in function description.


## Setup
To use these functions, just copy-paste the file min-formula.py in your project folder and import it in your main project. A short working example using data stored in the folder test_compo is provided 
in the Notebook min_recalc.ipynb. 

## Technologies
This package is written with Python 3.6
