#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 13:10:53 2021
@author: jacobje

This module contains fuctions to recalculate mineral atom formulas from analyses in weight percent oxides.
Input data must be a nx11 array of mineral analyses in wt% oxides, with the following order of columns: SiO2 TiO2 Al2O3 Cr2O3 FeO MnO MgO CaO Na2O K2O P2O5
"""
import numpy as np


# a few mineral compositions for testing
minerals = np.array(['Grt', 'Opx', 'Cpx', 'Bt', 'Pl', 'Kfs', 'Phg', 'Amp-hbl', 'Amp-act']) # 'Chl', 'Ep', 'St', 'Crd', 'Ol', 'Spl', 'Mt', 'Ttn', 'Srp'])
compo_min = np.array([[37.44, 0.03, 21.25, 0.00, 32.85, 1.36, 4.66, 3.02, 0.00, 0.00, 0.00],   # Grt
                      [51.51, 0.10, 0.57, 0.00, 30.00, 0.56, 17.16, 0.72, 0.04, 0.00, 0.00],   # Opx
                      [51.07, 0.38, 2.73, 0.00, 11.48, 0.22, 12.27, 21.75, 0.37, 0.00, 0.00],  # Cpx   
                      [35.77, 3.65, 15.29, 0.18, 18.55, 0.00, 12.27, 0.11, 0.12, 9.04, 0.00],  # Bt
                      [57.29, 0.00, 27.00, 0.00, 0.19, 0.00, 0.00, 8.95, 6.72, 0.11, 0.00],    # Pl
                      [64.46, 0.00, 18.61, 0.00, 0.05, 0.00, 0.00, 0.00, 0.56, 15.42, 0.00],   # Kfs
                      [50.72, 0.49, 27.65, 0.00, 3.70, 0.03, 2.37, 0.00, 0.17, 11.53, 0.00],   # Phg
                      [43.97, 1.51, 10.95, 0.10, 14.16, 0.21, 11.93, 11.81, 1.96, 0.59, 0.00], # Amp (Hbl)
                      [53.66, 0.17, 2.67, 0.08, 12.41, 0.20, 15.42, 12.70, 0.19, 0.00, 0.00]]) # Amp (Act)



# TO DO : White Mica (muscovite-phengite), Spinel-Magnetite

# # Generic Structural formula: recalculate the structural formulas of a set of mineral using the Oxy-number defined by user
#===============================================================================
#  Format: Si, Ti, Al, Cr, Fe, Mn, Mg, Ca, Na, K, P, SumCat = f(SiO2 TiO2 Al2O3 Cr2O3 FeO MnO MgO CaO Na2O K2O P2O5)
# Checked 12-04-22

def StrGeneral(data,N_Oxygen):

    # Atom mass of oxides
    #SiO2 / TIO2 / Al2O3 / Cr2O3 / FeO / MnO / MgO / CaO / Na2O / K2O / P2O5
    Nb_Cations = np.array([1,1,2,2,1,1,1,1,2,2,2]) # nb cations
    Nb_Oxygens = np.array([2,2,3,3,1,1,1,1,1,1,5]) # nb oxygens
    MolarMass = np.array([60.0848,79.8988,101.96,151.99,71.8464,70.9374,40.3044,56.0794,61.9789,94.195,141.94]) # atomic mass of oxides

    data[data<0.001] = 0 # replace values below threshold by zeros 
    # recalculate atom formulas normalized to appropriate nb of O (NbOx)
    MolesOxides = data / MolarMass
    MolesCations = MolesOxides * Nb_Cations
    MolesOxygen = MolesOxides * Nb_Oxygens
    Ox_Factor = N_Oxygen / np.sum(MolesOxygen, axis = 1)
    
    Cations = MolesCations * Ox_Factor[:,np.newaxis]
    [Si, Ti, Al, Cr, Fe, Mn, Mg, Ca, Na, K, P] = Cations.T
   
    SumCat = np.sum(Cations, axis=1)
    return(Si, Ti, Al, Cr, Fe, Mn, Mg, Ca, Na, K, P, SumCat)

#===============================================================================
#===============================================================================
   
# Generic Structural formul, with Fe3+: recalculate the structural formulas of a set of mineral using the Oxy-number defined by user.
# This function includes estimation of Fe2+/Fe3+ based on charge balance calculation
#===============================================================================
#  Format: Si, Ti, Al, Cr, Fe2, Fe3, Mn, Mg, Ca, Na, K, P, SumCat = f(SiO2 TiO2 Al2O3 Cr2O3 FeO MnO MgO CaO Na2O K2O P2O5)
# Checked 12-04-22

def StrGeneralFe3(data,N_Oxygen,N_cation):
    
    #SiO2 / TIO2 / Al2O3 / Cr2O3 / FeO / Fe2O3 / MnO / MgO / CaO / Na2O / K2O / P2O5
    Cationcharge = np.array([4,4,3,3,2,3,2,2,2,1,1,5]) # cation charge
    Nb_Cations = np.array([1,1,2,2,1,2,1,1,1,2,2,2]) # nb cations
    Nb_Oxygens = np.array([2,2,3,3,1,3,1,1,1,1,1,5]) # nb oxygens
    MolarMass = np.array([60.0848,79.8988,101.96,151.99,71.8464,159.69,70.9374,40.3044,56.0794,61.9789,94.195,141.94]) # atomic mass of oxides
    
    data[data<0.001] = 0 # replace values below threshold by zeros
    data = np.concatenate([data[:,0:5], np.zeros([data.shape[0],1]), data[:,5:]], axis=1)  # add column for Fe2O3. 
    
    # recalculate atom formulas normalized to Oxygen and cation basis
    MolesOxides = data / MolarMass
    MolesCations = MolesOxides * Nb_Cations
    MolesOxygen = MolesOxides * Nb_Oxygens
    
    Ox_Factor = N_Oxygen / np.sum(MolesOxygen, axis = 1)
    Cat_Factor = N_cation / np.sum(MolesCations, axis = 1)
    
    Norm_Obasis = MolesCations * Ox_Factor[:, np.newaxis]   # np.newaxis needed to multiply line by line
    Norm_Cbasis = MolesCations * Cat_Factor[:, np.newaxis]   # np.newaxis needed to multiply line by line
    
    # Fe3+ estimations
    SumCharges = np.sum(Norm_Cbasis * Cationcharge[np.newaxis,:], axis=1)
    FeO_t = data[:,4]
    Fe2O3_c = np.maximum(0,(2 * N_Oxygen - SumCharges) / Cat_Factor / 2 * MolarMass[5])   # np.maximum used to avoid negative values for Fe2O3 If 12-sum_charge < 0
    FeO_c = np.maximum(0,FeO_t - 0.8998 * Fe2O3_c) # np.maximum used to avoid negative values for FeO
        
        
    data[:,[4,5]] = np.array([FeO_c,Fe2O3_c]).T  # reassign FeO and Fe2O3 in data array
    
    # Reclaculate formula with Fe3
    MolesOxides = data / MolarMass
    MolesCations = MolesOxides * Nb_Cations
    MolesOxygen = MolesOxides * Nb_Oxygens
    
    Cat_Factor = N_cation / np.sum(MolesCations, axis = 1)
    Norm_Cbasis = MolesCations * Cat_Factor[:, np.newaxis]
    
    [Si, Ti, Al, Cr, Fe2, Fe3, Mn, Mg, Ca, Na, K, P] = Norm_Cbasis.T
    SumCat = np.sum(Norm_Cbasis, axis=1)
    return(Si, Ti, Al, Cr, Fe2, Fe3, Mn, Mg, Ca, Na, K, P, SumCat)
    

#===============================================================================
#===============================================================================

    
# Structural formula of olivine (Mg,Fe,Mn) 
# Checked 13-04-22
#===============================================================================
#  Format: Si Ti Al Cr Fe Mn Mg SumCat XFo XFa XTeph = f(SiO2 TiO2 Al2O3 Cr2O3 FeO MnO MgO CaO Na2O K2O P2O5)
#   -----------------------------------
#   End-members      T(1)      X(2)
#   -----------------------------------
#   Forsterite        Si       Mg,Mg  
#   Fayalite          Si       Fe,Fe 
#   Tephroite         Si       Mn,Mn 
#   ----------------------------------- 
# 4 Oxygens

def StrOlivine(data):

    [Si, Ti, Al, Cr, Fe, Mn, Mg, Ca, Na, K, P, SumCat] = StrGeneral(data,4)
    
    # solid solution end-members
    Xfo = Mg / (Fe+Mg+Mn)
    Xfa = Fe / (Fe+Mg+Mn)
    Xteph = Mn / (Fe+Mg+Mn)
        
    return(Si,Ti,Al,Cr,Fe,Mn,Mg,SumCat,Xfo,Xfa,Xteph)

#===============================================================================
#===============================================================================
    
    
# Structural formula of Garnet (without Fe3, only Al end-members) 
# Checked 12-04-21
#===============================================================================
#  Format: Si Ti Al Cr Fe Mn Mg Ca SumCat Si_T Al_T Al_M2 Fe_M1 Mg_M1 Ca_M1 Mn_M1 SumM2 SUmM1 Xalm Xpyr Xgrs Xsps = f(SiO2 TiO2 Al2O3 Cr2O3 FeO MnO MgO CaO Na2O K2O P2O5)
#   ---------------------------------------------
#   End-members      T(3)      M1(3)      M2(2)
#   ---------------------------------------------
#   Almandine         Si        Fe         Al 
#   Pyrope            Si        Mg         Al 
#   Grossular         Si        Ca         Al 
#   Spessartine       Si        Mn         Al 
#   ---------------------------------------------
# 12 Oxygens

    
def StrGarnet(data):

    [Si, Ti, Al, Cr, Fe, Mn, Mg, Ca, Na, K, P, SumCat] = StrGeneral(data,12)
    
    # structural formulas
    zeros = np.zeros(np.shape(data)[0])  # array of zeros. Used for cation site balance calculation

    Si_T = Si
    Al_T = np.maximum(zeros,3-Si_T)
    Al_M2 = Al-Al_T
    Ti_M2 = Ti
    Cr_M2 = Cr
    Fe_M1 = Fe
    Mg_M1 = Mg
    Mn_M1 = Mn
    Ca_M1 = Ca
    SumM2 = Al_M2 + Ti_M2 + Cr_M2
    SumM1 = Fe_M1 + Mg_M1 + Ca_M1 + Mn_M1
    
    # solid solution end-members (only Al end-members)
    Xalm = Fe / SumM1
    Xpyr = Mg / SumM1
    Xgrs = Ca / SumM1
    Xsps = Mn / SumM1
        
    return(Si,Ti,Al,Cr,Fe,Mn,Mg,Ca,SumCat,Si_T,Al_T,Al_M2,Cr_M2,Ti_M2,Fe_M1,Mg_M1,Ca_M1,Mn_M1,SumM2,SumM1,Xalm,Xpyr,Xgrs,Xsps)
        
  
#===============================================================================
#===============================================================================
    
    
# Structural formula of Garnet (without Fe3, Cr end-members) 
# Checked 12-04-21
#===============================================================================
#  Format: Si Ti Al Cr Fe Mn Mg Ca SumCat Si_T Al_T Al_M2 Cr_M2 Ti_M2 Fe_M1 Mg_M1 Ca_M1 Mn_M1 SumM2 SUmM1 Xalm Xpyr Xgrs Xsps Xuv = f(SiO2 TiO2 Al2O3 Cr2O3 FeO MnO MgO CaO Na2O K2O P2O5)
#   ---------------------------------------------
#   End-members      T(3)      M1(3)      M2(2)
#   ---------------------------------------------
#   Almandine         Si        Fe         Al 
#   Pyrope            Si        Mg         Al 
#   Grossular         Si        Ca         Al 
#   Spessartine       Si        Mn         Al 
#   Uvarovite         Si        Ca         Cr
#   ---------------------------------------------
# 12 Oxygens

    
def StrGarnetCr(data):

    [Si, Ti, Al, Cr, Fe, Mn, Mg, Ca, Na, K, P, SumCat] = StrGeneral(data,12)
    
    # structural formulas
    zeros = np.zeros(np.shape(data)[0])  # array of zeros. Used for cation site balance recalculation

    Si_T = Si
    Al_T = np.maximum(zeros,3-Si_T)
    Al_M2 = Al-Al_T
    Ti_M2 = Ti
    Cr_M2 = Cr
    Fe_M1 = Fe
    Mg_M1 = Mg
    Mn_M1 = Mn
    Ca_M1 = Ca
    SumM2 = Al_M2 + Ti_M2 + Cr_M2
    SumM1 = Fe_M1 + Mg_M1 + Ca_M1 + Mn_M1
    
    # solid solution end-members 
    Xapgs = Al_M2 / SumM2
    Xuv = Cr_M2 / SumM2
    Xalm = Fe / SumM1 * Xapgs
    Xpyr = Mg / SumM1 * Xapgs
    Xgrs = Ca / SumM1 * Xapgs
    Xsps = Mn / SumM1 * Xapgs
        
    return(Si,Ti,Al,Cr,Fe,Mn,Mg,Ca,SumCat,Si_T,Al_T,Al_M2,Cr_M2,Ti_M2,Fe_M1,Mg_M1,Ca_M1,Mn_M1,SumM2,SumM1,Xalm,Xpyr,Xgrs,Xsps,Xuv)
        
  
#===============================================================================
#===============================================================================  
 
    
# Structural formula of Garnet (with Fe3)
# Fe3 approximated either on the basis of sum(charges) = 24 (method = 1), or on the basis of sum(M2_site) = 2 (method = 2)
# Checked 12-04-22 - both methods yield slightly different results. This is expected because method 2 only takes into account cation balance on M2 site, while method 1 takes into account all  cations
#===============================================================================
#===============================================================================
#  Format: Si Ti Al Cr Fe2 Fe3 Mn Mg Ca SumCat Si_T Al_T Al_M2 Fe3_M2 Cr_M2 Ti_M2 Fe_M1 Mg_M1 Ca_M1 Mn_M1 SumM2 SUmM1 Xalm Xpyr Xgrs Xsps Xuv Xandr = f(SiO2 TiO2 Al2O3 Cr2O3 FeO MnO MgO CaO Na2O K2O P2O5)
#   ---------------------------------------------
#   End-members      T(3)      M1(3)      M2(2)
#   ---------------------------------------------
#   Almandine         Si        Fe2        Al 
#   Pyrope            Si        Mg         Al 
#   Grossular         Si        Ca         Al 
#   Spessartine       Si        Mn         Al 
#   Uvarovite         Si        Ca         Cr
#   Andradite         Si        Ca         Fe3
#   ---------------------------------------------
# 12 Oxygens

def StrGarnetFe3(data,method):
    
    #SiO2 / TIO2 / Al2O3 / Cr2O3 / FeO / Fe2O3 / MnO / MgO / CaO / Na2O / K2O / P2O5
    Cationcharge = np.array([4,4,3,3,2,3,2,2,2,1,1,5]) # cation charge
    Nb_Cations = np.array([1,1,2,2,1,2,1,1,1,2,2,2]) # nb cations
    Nb_Oxygens = np.array([2,2,3,3,1,3,1,1,1,1,1,5]) # nb oxygens
    MolarMass = np.array([60.0848,79.8988,101.96,151.99,71.8464,159.69,70.9374,40.3044,56.0794,61.9789,94.195,141.94]) # atomic mass of oxides
    
    data[data<0.001] = 0 # replace values below threshold by zeros
    data = np.concatenate([data[:,0:5], np.zeros([data.shape[0],1]), data[:,5:]], axis=1)  # add column for Fe2O3. 
    
    # recalculate atom formulas normalized to Oxygen and cation basis
    MolesOxides = data / MolarMass
    MolesCations = MolesOxides * Nb_Cations
    MolesOxygen = MolesOxides * Nb_Oxygens
    
    Ox_Factor = 12 / np.sum(MolesOxygen, axis = 1)
    Cat_Factor = 8 / np.sum(MolesCations, axis = 1)
    
    Norm_Obasis = MolesCations * Ox_Factor[:, np.newaxis]   # np.newaxis needed to multiply line by line
    Norm_Cbasis = MolesCations * Cat_Factor[:, np.newaxis]   # np.newaxis needed to multiply line by line
    
    # Fe3+ estimations
    if method == 1:
        SumCharges = np.sum(Norm_Cbasis * Cationcharge[np.newaxis,:], axis=1)
        FeO_t = data[:,4]
        Fe2O3_c = np.maximum(0,(24 - SumCharges) / Cat_Factor / 2 * MolarMass[5])   # np.maximum used to avoid negative values for Fe2O3 If 24-sum_charge < 0
        FeO_c = np.maximum(0,FeO_t - 0.8998 * Fe2O3_c) # np.maximum used to avoid negative values for FeO
        
    if method == 2:
        FeO_t = data[:,4]
        Fe3_c = 2 - np.sum(Norm_Cbasis[:,[1,2,3]], axis = 1)   # balance on M2 site (Al,Ti,Cr). Assumes no Al on T site
        Fe2O3_c = Fe3_c / Cat_Factor/2 * MolarMass[5]
        FeO_c = FeO_t - 0.8998 * Fe2O3_c
        
    data[:,[4,5]] = np.array([FeO_c,Fe2O3_c]).T  # reassign FeO and Fe2O3 in data array
    
    # Reclaculate formula with Fe3
    MolesOxides = data / MolarMass
    MolesCations = MolesOxides * Nb_Cations
    MolesOxygen = MolesOxides * Nb_Oxygens
    
    Cat_Factor = 8 / np.sum(MolesCations, axis = 1)
    Norm_Cbasis = MolesCations * Cat_Factor[:, np.newaxis]
    
    [Si, Ti, Al, Cr, Fe2, Fe3, Mn, Mg, Ca, Na, K, P] = Norm_Cbasis.T
    SumCat = np.sum(Norm_Cbasis, axis=1)

    
    # structural formulas
    zeros = np.zeros(np.shape(data)[0])  # array of zeros. Used for testing in cation site balance recalculation

    Si_T = Si
    Al_T = np.maximum(zeros,3-Si_T)
    Al_M2 = Al-Al_T
    Ti_M2 = Ti
    Cr_M2 = Cr
    Fe3_M2 = Fe3
    Fe2_M1 = Fe2
    Mg_M1 = Mg
    Mn_M1 = Mn
    Ca_M1 = Ca
    SumM2 = Al_M2 + Ti_M2 + Cr_M2 + Fe3_M2
    SumM1 = Fe2_M1 + Mg_M1 + Ca_M1 + Mn_M1
    
    # solid solution end-members
    Xapgs = Al_M2 / SumM2
    Xuv = Cr_M2 / SumM2
    Xandr = Fe3_M2 / SumM2
    Xalm = Fe2/ SumM1 * Xapgs
    Xpyr = Mg / SumM1 * Xapgs
    Xgrs = Ca / SumM1 * Xapgs
    Xsps = Mn / SumM1 * Xapgs
    
    return(Si,Ti,Al,Cr,Fe2,Fe3,Mn,Mg,Ca,SumCat,Si_T,Al_T,Al_M2,Cr_M2,Ti_M2,Fe2_M1,Mg_M1,Ca_M1,Mn_M1,SumM2,SumM1,Xalm,Xpyr,Xgrs,Xsps,Xandr,Xuv)


#===============================================================================
#===============================================================================

# Structural formula of Orthopyroxene (without Fe3) 
# Checked 13-04-22
#===============================================================================
#  Format: Si Al Fe Mn Mg Ca SumCat Si_T2 Al_T2 Al_M1 Mg_M1 Fe_M1 Fe_M2 Mg_M2 Xen Xfs Xmgts = f(SiO2 TiO2 Al2O3 Cr2O3 FeO MnO MgO CaO Na2O K2O P2O5)
#   End-members    T1(1)   T2(1)    M1(1)    M2(1)
#   ----------------------------------------------------
#   Enstatite       Si      Si       Mg       Mg 
#   Ferrosilite     Si      Si       Fe       Fe 
#   Mg-Tschermak    Si      Al       Al       Mg 
#   Diopside        Si      Si       Mg       Ca 
#   ---------------------------------------------
# 6 Oxygens
    
def StrOpx(data):

    [Si, Ti, Al, Cr, Fe, Mn, Mg, Ca, Na, K, P, SumCat] = StrGeneral(data,6)
    
    # structural formulas
    zeros = np.zeros(np.shape(data)[0])  # array of zeros. Used for testing in cation site balance recalculation

    Si_T2 = Si / 2
    Al_T2 = np.maximum(zeros,1-Si_T2)
    
    Al_M1 = Al - 2*Al_T2
    Cr_M1 = Cr
    Ti_M1 = Ti
    Mn_M1 = Mn
    MgFe_M1 = 1 - Al_M1 - Ti_M1 - Cr_M1
    XMg = Mg/(Fe+Mg)
    Mg_M1 = MgFe_M1 * XMg
    Fe_M1 = MgFe_M1 * (1-XMg)
    Mg_M2 = Mg - Mg_M1
    Fe_M2 = Fe - Fe_M1
    Ca_M2 = Ca
    
    SumM2 = Mg_M2 + Fe_M2 + Ca_M2
    SumM1 = Fe_M1 + Mg_M1 + Al_M1 + Ti_M1 + Cr_M1 + Mn_M1
    SumT2 = Si_T2 + Al_T2
    
    # solid solution end-members
    Xen = Mg_M1
    Xfs = Fe_M1
    Xmgts = Al_M1
    Xdi = Ca_M2
        
    return(Si,Ti,Al,Cr,Fe,Mn,Mg,Ca,SumCat,Si_T2,Al_T2,Al_M1,Cr_M1,Ti_M1,Fe_M1,Mg_M1,Mn_M1,Fe_M2,Mg_M2,SumM1,SumM2,XMg,Xen,Xfs,Xmgts,Xdi)
  
#===============================================================================
#===============================================================================

# Structural formula of Orthopyroxene (with Fe3) 
# checked 18/08/22
#===============================================================================
#  Format: Si Al Fe2 Fe3 Mn Mg Ca SumCat Si_T1 Si_T2 Al_T2 Al_M1 Mg_M1 Fe2_M1 Fe3_M1 Fe2_M2 Mg_M2 XMg Xen Xfs Xmgts = f(SiO2 TiO2 Al2O3 Cr2O3 FeO MnO MgO CaO Na2O K2O P2O5)
#   End-members    T1(1)   T2(1)    M1(1)    M2(1)
#   ----------------------------------------------------
#   Enstatite       Si      Si       Mg       Mg 
#   Ferrosilite     Si      Si       Fe       Fe 
#   Diopside        Si      Si       Mg       Ca 
#   Mg-Tschermak    Si      Al       Al       Mg 
#   mots            Si      Al       Fe3      Mg
#   ---------------------------------------------
# 6 Oxygens

def StrOpxFe3(data):
        
    [Si,Ti,Al,Cr,Fe2,Fe3,Mn,Mg,Ca,Na,K,P,SumCat] = StrGeneralFe3(data,6,4)
   
    # structural formulas
    zeros = np.zeros(np.shape(data)[0])  # array of zeros. Used for testing in cation site balance recalculation

    Si_T1 = np.ones(np.shape(data)[0])
    Si_T2 = Si - Si_T1
    Al_T2 = np.maximum(zeros,1-Si_T2)
    
    Al_M1 = Al - Al_T2
    Fe3_M1 = Fe3
    Cr_M1 = Cr
    Ti_M1 = Ti
    Mn_M1 = Mn
    MgFe_M1 = 1 - Al_M1 - Fe3_M1 - Ti_M1 - Cr_M1 - Mn_M1
    XMg = Mg/(Fe2+Mg)
    Mg_M1 = MgFe_M1 * XMg
    Fe2_M1 = MgFe_M1 * (1-XMg)
    
    Mg_M2 = Mg - Mg_M1
    Fe2_M2 = Fe2 - Fe2_M1
    Ca_M2 = Ca
    
    SumM2 = Mg_M2 + Fe2_M2 + Ca_M2
    SumM1 = Fe2_M1 + Mg_M1 + Al_M1 + Fe3_M1 + Cr_M1 + Ti_M1 + Mn_M1
    
    # solid solution end-members

    # solid solution end-members
    Xen = Mg_M1
    Xfs = Fe2_M1
    Xmgts = Al_M1
    Xdi = Ca_M2
    Xmots = Fe3_M1 
        
    return(Si,Ti,Al,Cr,Fe2,Fe3,Mn,Mg,Ca,SumCat,Si_T2,Al_T2,Al_M1,Cr_M1,Ti_M1,Fe2_M1,Fe3_M1,Mg_M1,Mn_M1,Fe2_M2,Mg_M2,SumM1,SumM2,XMg,Xen,Xfs,Xmgts,Xdi,Xmots)
    
#===============================================================================
#===============================================================================
   
# Structural formula of Clinopyroxene (without Fe3) 
# Checked 13-04-22
#===============================================================================
#  Format: Si,Al,Fe,Mn,Mg,Ca,SumCat,Si_T2,Al_T2,Al_M1,Fe_M1,Mg_M1,Fe_M2,Mg_M2,Ca_M2,Na_M2,SumM1,SumM2,Xwo,Xen,Xfs,Xhed,Xdi,Xjd,Xcats = f(SiO2 TiO2 Al2O3 Cr2O3 FeO MnO MgO CaO Na2O K2O P2O5)
#   End-members    T1(1)   T2(1)    M1(1)    M2(1)
#   ----------------------------------------------------
#   Diopside        Si      Si       Mg       Ca 
#   Hedenbergite    Si      Si       Fe       Ca 
#   Ca-Tschermak    Si      Al       Al       Ca 
#   cEnstatite      Si      Si       Mg       Mg 
#   cFerrosilite    Si      Si       Fe       Fe 
#   Jadeite         Si      Al       Al       Na 
#   ---------------------------------------------
# 6 Oxygens
    
def StrCpx(data):

    [Si, Ti, Al, Cr, Fe, Mn, Mg, Ca, Na, K, P, SumCat] = StrGeneral(data,6)
    
    # structural formulas
    zeros = np.zeros(np.shape(data)[0])  # array of zeros. Used for testing in cation site balance recalculation

    Si_T2 = Si / 2
    Al_T2 = np.maximum(zeros,1-Si_T2)
    Al_M1 = Al - 2 * Al_T2
  
    Cr_M1 = Cr
    Ti_M1 = Ti
    Mn_M1 = Mn
    MgFe_M1 = 1 - Al_M1 - Ti_M1 - Cr_M1 - Mn_M1
    XMg = Mg/(Fe+Mg)
    Mg_M1 = MgFe_M1 * XMg
    Fe_M1 = MgFe_M1 * (1-XMg)
    Mg_M2 = Mg - Mg_M1
    Fe_M2 = Fe - Fe_M1
    Ca_M2 = Ca
    Na_M2 = Na
    
    SumM2 = Mg_M2 + Fe_M2 + Ca_M2 + Na_M2
    SumM1 = Fe_M1 + Mg_M1 + Al_M1 + Cr_M1 + Ti_M1 + Mn_M1
    
    # solid solution end-members
    Xhed = Fe_M1
    Xjd = Na_M2
    Xdi = Mg_M1
    Xcats = Al_T2
    
    # WEF ternary space
    Xwo = Ca / (Ca + Mg + Fe)
    Xen = Mg / (Ca + Mg + Fe)
    Xfs = Fe / (Ca + Mg + Fe)
        
    return(Si,Ti,Al,Cr,Fe,Mn,Mg,Ca,Na,SumCat,Si_T2,Al_T2,Al_M1,Cr_M1,Ti_M1,Fe_M1,Mg_M1,Mn_M1,Fe_M2,Mg_M2,Ca_M2,Na_M2,SumM1,SumM2,Xwo,Xen,Xfs,Xhed,Xdi,Xjd,Xcats)
  
 
#===============================================================================
#===============================================================================

# Structural formula of Clinopyroxene (with Fe3) 
# Checked 18/08/22
#===============================================================================
#  Format: Si,Al,Fe2,Fe3,Mn,Mg,Ca,SumCat,Si_T1,Al_T1,Al_M1,Fe3_M1,Fe2_M1,Mg_M1,Fe2_M2,Mg_M2,Ca_M2,Na_M2,SumM1,SumM2,Xwo,Xen,Xfs,Xhed,Xdi,Xjd,Xcats,Xaeg = f(SiO2 TiO2 Al2O3 Cr2O3 FeO MnO MgO CaO Na2O K2O P2O5)
#   End-members    T1(1)   T2(1)    M1(1)    M2(1)
#   ----------------------------------------------------
#   Diopside        Si      Si       Mg       Ca 
#   Hedenbergite    Si      Si       Fe       Ca 
#   Ca-Tschermak    Si      Al       Al       Ca 
#   cEnstatite      Si      Si       Mg       Mg 
#   cFerrosilite    Si      Si       Fe       Fe 
#   Jadeite         Si      Si       Al       Na 
#   Aegyrine        Si      Si       Fe3      Na     
#   ---------------------------------------------
# 6 Oxygens
    
def StrCpxFe3(data):
        
    [Si,Ti,Al,Cr,Fe2,Fe3,Mn,Mg,Ca,Na,K,P,SumCat] = StrGeneralFe3(data,6,4)
   
    # structural formulas
    zeros = np.zeros(np.shape(data)[0])  # array of zeros. Used for testing in cation site balance recalculation

    Si_T1 = np.ones(np.shape(data)[0])
    Si_T2 = Si - Si_T1
    Al_T2 = np.maximum(zeros,1-Si_T2)
    
    Al_M1 = Al - Al_T2
    Fe3_M1 = Fe3
    Cr_M1 = Cr
    Ti_M1 = Ti
    Mn_M1 = Mn
    MgFe_M1 = 1 - Al_M1 - Fe3_M1 - Cr_M1 - Ti_M1 - Mn_M1
    XMg = Mg/(Fe2+Mg)
    Mg_M1 = MgFe_M1 * XMg
    Fe2_M1 = MgFe_M1 * (1-XMg)
    
    Mg_M2 = Mg - Mg_M1
    Fe2_M2 = Fe2 - Fe2_M1
    Ca_M2 = Ca
    Na_M2 = Na
    
    SumM2 = Mg_M2 + Fe2_M2 + Ca_M2 + Na_M2
    SumM1 = Fe2_M1 + Mg_M1 + Al_M1 + Fe3_M1 + Cr_M1 + Ti_M1
    
    # solid solution end-members
    Xhed = Fe2_M1
    Xdi = Mg_M1
    Xjd = Na_M2 * Al_M1/(Al_M1+Fe3_M1+Cr)
    Xaeg = Na_M2 * Fe3_M1/(Al_M1+Fe3_M1+Cr)
    Xcats = Al_T2 / 2
    
    # WEF ternary space
    Xwo = Ca / (Ca + Mg + Fe2)
    Xen = Mg / (Ca + Mg + Fe2)
    Xfs = Fe2 / (Ca + Mg + Fe2)    
        
    return(Si,Ti,Al,Cr,Fe2,Fe3,Mn,Mg,Ca,Na,SumCat,Si_T2,Al_T2,Al_M1,Cr_M1,Ti_M1,Fe3_M1,Fe2_M1,Mg_M1,Mn_M1,Fe2_M2,Mg_M2,Ca_M2,Na_M2,SumM1,SumM2,Xwo,Xen,Xfs,Xhed,Xdi,Xjd,Xcats,Xaeg)
    
    
#===============================================================================
#===============================================================================
    

# Structural formula of Amphibole

#===============================================================================
# Format: Si,Ti,Al,Cr,Fe,Mn,Mg,Ca,Na,K,Si4,Aliv,Alvi,Al_T2,Al_M2,XMg,Ti_M2,Mg_M2,Fe_M2,Fe_M13,Mg_M13,Ca_M4,Na_M4,Na_A,V_A,XTr,XFtr,XTs,XPrg,XGln] = StructFctAmphiboles(SiO2 TiO2 Al2O3 Cr2O3 FeO MnO MgO CaO Na2O K2O P2O5)
#   End-member       A(1)   M1-3(3)   M2(2)   M4(2)   T1(2)   T2(6)   OH(2)
#   ---------------------------------------------------------------------------
#   Tschermackite    V      Mg        Al      Ca      SiAl    Si      OH
#   Pargasite        Na     Mg        MgAl    Ca      SiAl    Si      OH
#   K-pargasite      K      Mg        MgAl    Ca      SiAl    Si      OH
#   Glaucophane      V      Mg        Al      Na      Si      Si      OH
#   Ti-tschermack    V      Mg        Ti      Ca      Si      Si      O
#   Tremolite        V      Mg        Mg      Ca      Si      Si      OH
#   Cummingotnite    V      Mg        Mg      Mg      Si      Si      OH
#   Grunerite        V      Mg        Al      Ca      Si      Si      OH
#   ---------------------------------------------------------------------------
#   Only Mg end-members are given in this table for simplicity. Each one have an Fe equivalent, obtained by replacing Mg by Fe(2+) on M sites
#   ---------------------------------------------------------------------------
# 23 Oxygens 
def StrAmphibole(data):
    
    [Si, Ti, Al, Cr, Fe, Mn, Mg, Ca, Na, K, P, SumCat] = StrGeneral(data,23)
    
    #  structural formulas
    zeros = np.zeros(np.shape(data)[0])  # array of zeros. Used for cation site balance recalculation
    
    XMg = Mg / (Mg+Fe)
    Al_T2 = np.maximum(zeros, 8 - Si) # if Si > 8, no Al on tetrahedral site
    
    # M2
    Al_M2 = Al - Al_T2
    Ti_M2 = Ti
    Cr_M2 = Cr
    Mn_M2 = Mn
    FeMg_M2 = np.maximum(zeros, 2 - (Al_M2 + Ti_M2 + Cr_M2 + Mn_M2))  # FeMg allowed on M2 only if sum of Al+Ti+Cr+Mn on M2 is <2
    Mg_M2 = FeMg_M2 * XMg
    Fe_M2 = FeMg_M2 * (1-XMg)
    
    # M1-3
    FeMg_M13 = np.minimum(3, Fe + Mg - FeMg_M2)  #if available Fe + Mg > 3, 3 FeMg on M13 site and remaining FeMg on M4
    Mg_M13 = FeMg_M13 * XMg
    Fe_M13 = FeMg_M13 * (1-XMg)
    
    SumC = Al_M2 + Ti_M2 + Cr_M2 + Mn_M2 + FeMg_M2 + FeMg_M13  # sum M13+M2 site (C in Leake, 1997). Should equal 5
    
    # M4
    FeMg_M4 = Fe + Mg - FeMg_M2 - FeMg_M13  # remaining Fe and Mg go on M4 
    Ca_M4 = Ca
    Na_M4 = np.maximum(zeros, 2 - Ca_M4 - FeMg_M4)
    
    # A
    Na_A = Na - Na_M4
    K_A = K
    V_A = 1 - Na_A - K_A
    SumA = Na_A + K_A
    
    # classification
    # amphibole group: MgFeMnLi, calcic, sodi-calcic, sodic
    amp_group = []
    
    for i in np.arange(0,np.shape(data)[0]):
        if (Ca_M4[i] + Na_M4[i] < 1 and FeMg_M4[i] >= 1):
            amp_group.append('MgFeMnLi')
        else:
            if Na_M4[i] < 0.5:
                amp_group.append('Calcic')
            elif Na_M4[i] >= 1.5:
                amp_group.append('Sodic')
            else:
                amp_group.append('Sodi-calcic')
    
    # amphibole name
    amp_name = []
    
    for i in np.arange(0,np.shape(data)[0]):
        # MgFeMnLi group
        if all( [amp_group[i] == 'MgFeMnLi', Si[i] < 7, XMg[i] >= 0.5 ] ):
            amp_name.append('gedrite')
        if all( [amp_group[i] == 'MgFeMnLi', Si[i] < 7, XMg[i] < 0.5 ] ):
            amp_name.append('Fe-gedrite')
        if all( [amp_group[i] == 'MgFeMnLi', Si[i] >= 7, XMg[i] >= 0.5 ] ):
            amp_name.append('anthophyllite / cummingtonite')
        if all( [amp_group[i] == 'MgFeMnLi', Si[i] >= 7, XMg[i] < 0.5 ] ):
            amp_name.append('Fe-anthophyllite / grunerite')
            
        # Calcic group - Ca_M4 >= 1.5 - Na+K (A) >= 0.5
        if all( [amp_group[i] == 'Calcic', Ca_M4[i] >= 1.5, Na_A[i] + K_A[i] >= 0.5, Si[i] < 5.5, Ti[i] < 0.5, XMg[i] >= 0.5 ] ):
            amp_name.append('Mg-sadanagaite')
        if all( [amp_group[i] == 'Calcic', Ca_M4[i] >= 1.5, Na_A[i] + K_A[i] >= 0.5, Si[i] < 5.5, Ti[i] < 0.5, XMg[i] < 0.5 ] ):
            amp_name.append('sadanagaite') 
        if all( [amp_group[i] == 'Calcic', Ca_M4[i] >= 1.5, Na_A[i] + K_A[i] >= 0.5, 6.5 > Si[i] >= 5.5, Ti[i] < 0.5, XMg[i] >= 0.5 ] ):
            amp_name.append('pargasite')
        if all( [amp_group[i] == 'Calcic', Ca_M4[i] >= 1.5, Na_A[i] + K_A[i] >= 0.5, 6.5 > Si[i] >= 5.5, Ti[i] < 0.5, XMg[i] < 0.5 ] ):
            amp_name.append('Fe-pargasite')
        if all( [amp_group[i] == 'Calcic', Ca_M4[i] >= 1.5, Na_A[i] + K_A[i] >= 0.5, Si[i] >= 6.5, Ti[i] < 0.5, XMg[i] >= 0.5 ] ):
            amp_name.append('edenite')
        if all( [amp_group[i] == 'Calcic', Ca_M4[i] >= 1.5, Na_A[i] + K_A[i] >= 0.5, Si[i] >= 6.5, Ti[i] < 0.5, XMg[i] < 0.5 ] ):
            amp_name.append('Fe-edenite')
        if all( [amp_group[i] == 'Calcic', Ca_M4[i] >= 1.5, Na_A[i] + K_A[i] >= 0.5, 6.5 > Si[i] >= 5.5, Ti[i] >= 0.5, XMg[i] >= 0.5 ] ):
            amp_name.append('kaersutite')
        if all( [amp_group[i] == 'Calcic', Ca_M4[i] >= 1.5, Na_A[i] + K_A[i] >= 0.5, 6.5 > Si[i] >= 5.5, Ti[i] >= 0.5, XMg[i] < 0.5 ] ):
            amp_name.append('Fe-kaersutite')            
            
        # Calcic group Ca_M4 >= 1.5 - Na+K (A) < 0.5
        if all( [amp_group[i] == 'Calcic', Ca_M4[i] >= 1.5, Na_A[i] + K_A[i] < 0.5, Si[i] < 6.5, XMg[i] >= 0.5 ] ):
            amp_name.append('tschermakite')
        if all( [amp_group[i] == 'Calcic', Ca_M4[i] >= 1.5, Na_A[i] + K_A[i] < 0.5, Si[i] < 6.5, XMg[i] < 0.5 ] ):
            amp_name.append('Fe-tschermakite')
        if all( [amp_group[i] == 'Calcic', Ca_M4[i] >= 1.5, Na_A[i] + K_A[i] < 0.5, 7.5 > Si[i] >= 6.5, XMg[i] >= 0.5 ] ):
            amp_name.append('Mg-hornblende')
        if all( [amp_group[i] == 'Calcic', Ca_M4[i] >= 1.5, Na_A[i] + K_A[i] < 0.5, 7.5 > Si[i] >= 6.5, XMg[i] < 0.5 ] ):
            amp_name.append('Fe-hornblende')
        if all( [amp_group[i] == 'Calcic', Ca_M4[i] >= 1.5, Na_A[i] + K_A[i] < 0.5, Si[i] >= 7.5, XMg[i] >= 0.9 ] ):
            amp_name.append('tremolite')
        if all( [amp_group[i] == 'Calcic', Ca_M4[i] >= 1.5, Na_A[i] + K_A[i] < 0.5, Si[i] >= 7.5, 0.9 > XMg[i] >= 0.5 ] ):
            amp_name.append('actinolite')
        if all( [amp_group[i] == 'Calcic', Ca_M4[i] >= 1.5, Na_A[i] + K_A[i] < 0.5, Si[i] >= 7.5, XMg[i] < 0.5 ] ):
            amp_name.append('Fe-actinolite')
            
        # Sodi-calcic group
        if all( [amp_group[i] == 'Sodi-calcic', Na_A[i] + K_A[i] >= 0.5, Si[i] < 6.5, XMg[i] >= 0.5 ] ):
            amp_name.append('Mg-taramite')
        if all( [amp_group[i] == 'Sodi-calcic', Na_A[i] + K_A[i] >= 0.5, Si[i] < 6.5, XMg[i] < 0.5 ] ):
            amp_name.append('taramite')
        if all( [amp_group[i] == 'Sodi-calcic', Na_A[i] + K_A[i] >= 0.5, 7.5 > Si[i] >= 6.5, XMg[i] >= 0.5 ] ):
            amp_name.append('Mg-katophorite')
        if all( [amp_group[i] == 'Sodi-calcic', Na_A[i] + K_A[i] >= 0.5, 7.5 > Si[i] >= 6.5, XMg[i] < 0.5 ] ):
            amp_name.append('katophorite')
        if all( [amp_group[i] == 'Sodi-calcic', Na_A[i] + K_A[i] >= 0.5, Si[i] >= 7.5, XMg[i] >= 0.5 ] ):
            amp_name.append('richterite')
        if all( [amp_group[i] == 'Sodi-calcic', Na_A[i] + K_A[i] >= 0.5, Si[i] >= 7.5, XMg[i] < 0.5 ] ):
            amp_name.append('Fe-richterite')
        
        if all( [amp_group[i] == 'Sodi-calcic', Na_A[i] + K_A[i] < 0.5, Si[i] < 7.5, XMg[i] >= 0.5 ] ):
            amp_name.append('baroisite')
        if all( [amp_group[i] == 'Sodi-calcic', Na_A[i] + K_A[i] < 0.5, Si[i] < 7.5, XMg[i] < 0.5 ] ):
            amp_name.append('Fe-baroisite')
        if all( [amp_group[i] == 'Sodi-calcic', Na_A[i] + K_A[i] < 0.5, Si[i] >= 7.5, XMg[i] >= 0.5 ] ):
            amp_name.append('winchite')
        if all( [amp_group[i] == 'Sodi-calcic', Na_A[i] + K_A[i] < 0.5, Si[i] >= 7.5, XMg[i] < 0.5 ] ):
            amp_name.append('Fe-winchite')
            
        # Sodic group: Na-amphiboles are a zoo, some rare types only occur in very specific settings with exotic rock compositions
        # Thus, a simplified scheme is presented here, with only the (Fe)-glaucophane / riebeckite poles
        if all( [amp_group[i] == 'Sodic', Na_A[i] + K_A[i] < 0.5, Si[i] >= 7.0, XMg[i] >= 0.5 ] ):
            amp_name.append('glaucophane / Mg-riebeckite')
        if all( [amp_group[i] == 'Sodic', Na_A[i] + K_A[i] < 0.5, Si[i] >= 7.0, XMg[i] < 0.5 ] ):
            amp_name.append('Fe-glaucophane / riebeckite')
            
            
        if len(amp_name)<=i:
            amp_name.append('not classified')
    
    return(Si,Ti,Al,Cr,Fe,Mn,Mg,Ca,Na,K,SumCat,Al_T2,Al_M2,Ti_M2,Mg_M2,Fe_M2,Fe_M13,Mg_M13,Ca_M4,Na_M4,SumC,Na_A,K_A,SumA,XMg,amp_group,amp_name)

    
#===============================================================================
#===============================================================================
    

# Structural formula of Amphibole (with Fe3+)

#===============================================================================
# Format: Si,Ti,Al,Cr,Fe2,Fe3,Mn,Mg,Ca,Na,K,Si4,Aliv,Alvi,Al_T2,Al_M2,Fe3_M2,Ti_M2,Mg_M2,Fe_M2,Fe_M13,Mg_M13,Ca_M4,Na_M4,Na_A,V_A,XMg,XTr,XFtr,XTs,XPrg,XGln] = StructFctAmphiboles(SiO2 TiO2 Al2O3 Cr2O3 FeO MnO MgO CaO Na2O K2O P2O5)
#   ---------------------------------------------------------------------------
#   End-member       A(1)   M1-3(3)   M2(2)   M4(2)   T1(2)   T2(6)   OH(2)
#   ---------------------------------------------------------------------------
#   Tschermackite    V      Mg        Al      Ca      SiAl    Si      OH
#   Pargasite        Na     Mg        MgAl    Ca      SiAl    Si      OH
#   K-pargasite      K      Mg        MgAl    Ca      SiAl    Si      OH
#   Mg-riebeckite    V      Mg        Fe3     Na      Si      Si      OH
#   Glaucophane      V      Mg        Al      Na      Si      Si      OH
#   Ti-tschermack    V      Mg        Ti      Ca      Si      Si      O
#   Tremolite        V      Mg        Mg      Ca      Si      Si      OH
#   Cummingotnite    V      Mg        Mg      Mg      Si      Si      OH
#   Grunerite        V      Mg        Al      Ca      Si      Si      OH
#   ---------------------------------------------------------------------------
#   Only Mg end-members are given in this table for simplicity. Each one have an Fe equivalent, obtained by replacing Mg by Fe(2+) on M sites
#   ---------------------------------------------------------------------------
# 23 Oxygens 
def StrAmphiboleFe3(data):
    

    # Str formula without Fe3+
    [Si, Ti, Al, Cr, Fe, Mn, Mg, Ca, Na, K, P, SumCat] = StrGeneral(data,23)
 
    
    zeros = np.zeros(np.shape(data)[0])  # array of zeros. Used for Fe3+ and cation site balance recalculation
    
    Cat_subset = np.array([Si, Ti, Al, Cr, Fe, Mn, Mg])  # cations on T1 + M13 + M2 sites. should sum to 13
    CatSum_subset = np.sum(Cat_subset, axis=0)
    Cat_Factor = 13 / CatSum_subset
    [Si, Ti, Al, Cr, Fe, Mn, Mg] = Cat_subset * Cat_Factor[np.newaxis,:]  # re-normalization to 13 cations

    
    # Fe3+ estimation
    Fe3 = 2 * 21 * (np.maximum(zeros,1-Cat_Factor)) # droop method: Fe3 = 2*N_ox * (1-T/S). If SumCat_subset < 13 (sum too low), no Fe3+ is considered
    Fe2 = np.maximum(zeros, Fe-Fe3)  # avoid negative values for Fe2+
   

    #  structural formulas

    
    XMg = Mg / (Mg+Fe2)
    Al_T2 = np.maximum(zeros, 8 - Si) # if Si > 8, no Al on tetrahedral site
    
    # M2
    Al_M2 = Al - Al_T2
    Fe3_M2 = Fe3
    Ti_M2 = Ti
    Cr_M2 = Cr
    Mn_M2 = Mn
    FeMg_M2 = np.maximum(zeros, 2 - (Al_M2 + Ti_M2 + Cr_M2 + Mn_M2 + Fe3_M2))  # FeMg allowed on M2 only if sum of Al+Ti+Cr+Mn on M2 is <2
    Mg_M2 = FeMg_M2 * XMg
    Fe_M2 = FeMg_M2 * (1-XMg)
    
    # M1-3
    FeMg_M13 = np.minimum(3, Fe2 + Mg - FeMg_M2)  #if available Fe + Mg > 3, 3 FeMg on M13 site and remaining FeMg on M4
    Mg_M13 = FeMg_M13 * XMg
    Fe_M13 = FeMg_M13 * (1-XMg)
    
    SumC = Al_M2 + Fe3_M2 + Ti_M2 + Cr_M2 + Mn_M2 + FeMg_M2 + FeMg_M13  # sum M13+M2 site (C in Leake, 1997). Should equal 5
    
    # M4
    FeMg_M4 = Fe2 + Mg - FeMg_M2 - FeMg_M13  # remaining Fe and Mg go on M4 
    Ca_M4 = Ca
    Na_M4 = np.maximum(zeros, 2 - Ca_M4 - FeMg_M4)
    
    # A
    Na_A = Na - Na_M4
    K_A = K
    V_A = 1 - Na_A - K_A
    SumA = Na_A + K_A
    
    # classification
    # amphibole group: MgFeMnLi, calcic, sodi-calcic, sodic
    amp_group = []
    
    for i in np.arange(0,np.shape(data)[0]):
        if (Ca_M4[i] + Na_M4[i] < 1 and FeMg_M4[i] >= 1):
            amp_group.append('MgFeMnLi')
        else:
            if Na_M4[i] < 0.5:
                amp_group.append('Calcic')
            elif Na_M4[i] >= 1.5:
                amp_group.append('Sodic')
            else:
                amp_group.append('Sodi-calcic')
    
    # amphibole name
    amp_name = []
    
    for i in np.arange(0,np.shape(data)[0]):
        # MgFeMnLi group
        if all( [amp_group[i] == 'MgFeMnLi', Si[i] < 7, XMg[i] >= 0.5 ] ):
            amp_name.append('gedrite')
        if all( [amp_group[i] == 'MgFeMnLi', Si[i] < 7, XMg[i] < 0.5 ] ):
            amp_name.append('Fe-gedrite')
        if all( [amp_group[i] == 'MgFeMnLi', Si[i] >= 7, XMg[i] >= 0.5 ] ):
            amp_name.append('anthophyllite / cummingtonite')
        if all( [amp_group[i] == 'MgFeMnLi', Si[i] >= 7, XMg[i] < 0.5 ] ):
            amp_name.append('Fe-anthophyllite / grunerite')
            
        # Calcic group - Ca_M4 >= 1.5 - Na+K (A) >= 0.5
        if all( [amp_group[i] == 'Calcic', Ca_M4[i] >= 1.5, Na_A[i] + K_A[i] >= 0.5, Si[i] < 5.5, Ti[i] < 0.5, XMg[i] >= 0.5 ] ):
            amp_name.append('Mg-sadanagaite')
        if all( [amp_group[i] == 'Calcic', Ca_M4[i] >= 1.5, Na_A[i] + K_A[i] >= 0.5, Si[i] < 5.5, Ti[i] < 0.5, XMg[i] < 0.5 ] ):
            amp_name.append('sadanagaite') 
        if all( [amp_group[i] == 'Calcic', Ca_M4[i] >= 1.5, Na_A[i] + K_A[i] >= 0.5, 6.5 > Si[i] >= 5.5, Ti[i] < 0.5, XMg[i] >= 0.5, Al_M2[i] >= Fe3_M2[i] ] ):
            amp_name.append('pargasite')
        if all( [amp_group[i] == 'Calcic', Ca_M4[i] >= 1.5, Na_A[i] + K_A[i] >= 0.5, 6.5 > Si[i] >= 5.5, Ti[i] < 0.5, XMg[i] < 0.5, Al_M2[i] >= Fe3_M2[i] ] ):
            amp_name.append('Fe-pargasite')
        if all( [amp_group[i] == 'Calcic', Ca_M4[i] >= 1.5, Na_A[i] + K_A[i] >= 0.5, 6.5 > Si[i] >= 5.5, Ti[i] < 0.5, XMg[i] >= 0.5, Al_M2[i] < Fe3_M2[i] ] ):
            amp_name.append('Mg-hastingsite')
        if all( [amp_group[i] == 'Calcic', Ca_M4[i] >= 1.5, Na_A[i] + K_A[i] >= 0.5, 6.5 > Si[i] >= 5.5, Ti[i] < 0.5, XMg[i] < 0.5, Al_M2[i] < Fe3_M2[i] ] ):
            amp_name.append('hastingsite')            
        if all( [amp_group[i] == 'Calcic', Ca_M4[i] >= 1.5, Na_A[i] + K_A[i] >= 0.5, Si[i] >= 6.5, Ti[i] < 0.5, XMg[i] >= 0.5 ] ):
            amp_name.append('edenite')
        if all( [amp_group[i] == 'Calcic', Ca_M4[i] >= 1.5, Na_A[i] + K_A[i] >= 0.5, Si[i] >= 6.5, Ti[i] < 0.5, XMg[i] < 0.5 ] ):
            amp_name.append('Fe-edenite')
        if all( [amp_group[i] == 'Calcic', Ca_M4[i] >= 1.5, Na_A[i] + K_A[i] >= 0.5, 6.5 > Si[i] >= 5.5, Ti[i] >= 0.5, XMg[i] >= 0.5 ] ):
            amp_name.append('kaersutite')
        if all( [amp_group[i] == 'Calcic', Ca_M4[i] >= 1.5, Na_A[i] + K_A[i] >= 0.5, 6.5 > Si[i] >= 5.5, Ti[i] >= 0.5, XMg[i] < 0.5 ] ):
            amp_name.append('Fe-kaersutite')            
            
        # Calcic group Ca_M4 >= 1.5 - Na+K (A) < 0.5
        if all( [amp_group[i] == 'Calcic', Ca_M4[i] >= 1.5, Na_A[i] + K_A[i] < 0.5, Si[i] < 6.5, XMg[i] >= 0.5 ] ):
            amp_name.append('tschermakite')
        if all( [amp_group[i] == 'Calcic', Ca_M4[i] >= 1.5, Na_A[i] + K_A[i] < 0.5, Si[i] < 6.5, XMg[i] < 0.5 ] ):
            amp_name.append('Fe-tschermakite')
        if all( [amp_group[i] == 'Calcic', Ca_M4[i] >= 1.5, Na_A[i] + K_A[i] < 0.5, 7.5 > Si[i] >= 6.5, XMg[i] >= 0.5 ] ):
            amp_name.append('Mg-hornblende')
        if all( [amp_group[i] == 'Calcic', Ca_M4[i] >= 1.5, Na_A[i] + K_A[i] < 0.5, 7.5 > Si[i] >= 6.5, XMg[i] < 0.5 ] ):
            amp_name.append('Fe-hornblende')
        if all( [amp_group[i] == 'Calcic', Ca_M4[i] >= 1.5, Na_A[i] + K_A[i] < 0.5, Si[i] >= 7.5, XMg[i] >= 0.9 ] ):
            amp_name.append('tremolite')
        if all( [amp_group[i] == 'Calcic', Ca_M4[i] >= 1.5, Na_A[i] + K_A[i] < 0.5, Si[i] >= 7.5, 0.9 > XMg[i] >= 0.5 ] ):
            amp_name.append('actinolite')
        if all( [amp_group[i] == 'Calcic', Ca_M4[i] >= 1.5, Na_A[i] + K_A[i] < 0.5, Si[i] >= 7.5, XMg[i] < 0.5 ] ):
            amp_name.append('Fe-actinolite')
            
        # Sodi-calcic group
        if all( [amp_group[i] == 'Sodi-calcic', Na_A[i] + K_A[i] >= 0.5, Si[i] < 6.5, XMg[i] >= 0.5 ] ):
            amp_name.append('Mg-taramite')
        if all( [amp_group[i] == 'Sodi-calcic', Na_A[i] + K_A[i] >= 0.5, Si[i] < 6.5, XMg[i] < 0.5 ] ):
            amp_name.append('taramite')
        if all( [amp_group[i] == 'Sodi-calcic', Na_A[i] + K_A[i] >= 0.5, 7.5 > Si[i] >= 6.5, XMg[i] >= 0.5 ] ):
            amp_name.append('Mg-katophorite')
        if all( [amp_group[i] == 'Sodi-calcic', Na_A[i] + K_A[i] >= 0.5, 7.5 > Si[i] >= 6.5, XMg[i] < 0.5 ] ):
            amp_name.append('katophorite')
        if all( [amp_group[i] == 'Sodi-calcic', Na_A[i] + K_A[i] >= 0.5, Si[i] >= 7.5, XMg[i] >= 0.5 ] ):
            amp_name.append('richterite')
        if all( [amp_group[i] == 'Sodi-calcic', Na_A[i] + K_A[i] >= 0.5, Si[i] >= 7.5, XMg[i] < 0.5 ] ):
            amp_name.append('Fe-richterite')
        
        if all( [amp_group[i] == 'Sodi-calcic', Na_A[i] + K_A[i] < 0.5, Si[i] < 7.5, XMg[i] >= 0.5 ] ):
            amp_name.append('baroisite')
        if all( [amp_group[i] == 'Sodi-calcic', Na_A[i] + K_A[i] < 0.5, Si[i] < 7.5, XMg[i] < 0.5 ] ):
            amp_name.append('Fe-baroisite')
        if all( [amp_group[i] == 'Sodi-calcic', Na_A[i] + K_A[i] < 0.5, Si[i] >= 7.5, XMg[i] >= 0.5 ] ):
            amp_name.append('winchite')
        if all( [amp_group[i] == 'Sodi-calcic', Na_A[i] + K_A[i] < 0.5, Si[i] >= 7.5, XMg[i] < 0.5 ] ):
            amp_name.append('Fe-winchite')
            
        # Sodic group: Na-amphiboles are a zoo, some rare types only occur in very specific settings with exotic rock compositions
        # Thus, a simplified scheme is presented here, with only the (Fe)-glaucophane / riebeckite poles
        if all( [amp_group[i] == 'Sodic', Na_A[i] + K_A[i] < 0.5, Si[i] >= 7.0, XMg[i] >= 0.5, Al_M2[i] >= Fe3_M2[i] ] ):
            amp_name.append('glaucophane')
        if all( [amp_group[i] == 'Sodic', Na_A[i] + K_A[i] < 0.5, Si[i] >= 7.0, XMg[i] < 0.5, Al_M2[i] >= Fe3_M2[i] ] ):
            amp_name.append('Fe-glaucophane')
        if all( [amp_group[i] == 'Sodic', Na_A[i] + K_A[i] < 0.5, Si[i] >= 7.0, XMg[i] >= 0.5, Al_M2[i] < Fe3_M2[i] ] ):
            amp_name.append('Mg-riebeckite')
        if all( [amp_group[i] == 'Sodic', Na_A[i] + K_A[i] < 0.5, Si[i] >= 7.0, XMg[i] < 0.5, Al_M2[i] < Fe3_M2[i] ] ):
            amp_name.append('riebeckite')
            
            
        if len(amp_name)<=i:
            amp_name.append('not classified')
    
    return(Si,Ti,Al,Cr,Fe2,Fe3,Mn,Mg,Ca,Na,K,SumCat,Al_T2,Al_M2,Fe3_M2,Ti_M2,Mg_M2,Fe_M2,Fe_M13,Mg_M13,Ca_M4,Na_M4,SumC,Na_A,K_A,SumA,XMg,amp_group,amp_name)

    
#===============================================================================
#===============================================================================
# Structural formula of Biotite 
# checked 18/08/22: problem of Fe-Mg partition on M1-M2 sites. Does not affect end-member calculations, but Mg_M1, Mg_M2 etc may be wrong.
#===============================================================================
#  Format: Si Ti Al Cr Fe Mn Mg Ca Na SumCat Si_T1 Si_T2 Al_T2 Al_M1 Mg_M1 Fe_M1 Mn_M1 V_M1 Ti_M2 Fe_M2 Mg_M2 Mn_M2 K_A Ca_A Na_A Vac_A XMg Xann Xphl Xsid Xeast Xtibt Xmnbt = f(SiO2 TiO2 Al2O3 Cr2O3 FeO MnO MgO CaO Na2O K2O P2O5)
#   ----------------------------------------------------
#   End-member      T1(2)   T2(2)   M1(1)   M2(2)   A(1)
#   ----------------------------------------------------
#   Phlogopite      Si,Si   Si,Al   Mg      Mg,Mg   K
#   Annite          Si,Si   Si,Al   Fe      Fe,Fe   K
#   Eastonite       Si,Si   Al,Al   Al      Mg,Mg   K
#   Siderophyllite  Si,Si   Al,Al   Al      Fe,Fe   K
#   Ti-biotite      Si,Si   Si,Al   Mg      Ti,Mg   K
#   Mn-biotite      Si,Si   Si,Al   Mn      Mn,Mn   K
#   ----------------------------------------------------   
#   Not considered:
#   Muscovite       Si,Si   Si,Al   V       Mg,Mg   K
#   ----------------------------------------------------
#      - Ti assumed to be ordered onto M2
#      - Al assumed to be ordered onto M1
#      - Mn added (ordered onto M12)
# 11 Oxygens 
    
def StrBiotite(data):

    [Si, Ti, Al, Cr, Fe, Mn, Mg, Ca, Na, K, P, SumCat] = StrGeneral(data,11)
    
    # structural formulas
    zeros = np.zeros(np.shape(data)[0])  # array of zeros. Used for cation site balance recalculation
    
    XMg = Mg / (Mg + Fe)
    Si_T1 = 2 * np.ones(np.shape(data)[0])
    Si_T2 = np.maximum(zeros,Si - Si_T1)  # if Si < 2, no Si on T2
    
    Al_T2 = np.minimum(2 - Si_T2, Al)  # if Al < 2-Si_T2; Al_T2 = Al (all Al on T2); otherwise Al_T2 = 2-Si_T2 and remaining Al goes on M1
    Alvi = Al - Al_T2
    SumT2 = Si_T2 + Al_T2
    
    # M2
    Mn_M2 = 2./3 * Mn 
    Ti_M2 = Ti      # Ti on M2
    Mg_M2 = Ti_M2   # contribution from tibt
    FeMgEqui_M2 = 2 - (Mn_M2 + Ti_M2 + Mg_M2)
    Mg_M2 = Mg_M2 + FeMgEqui_M2 * XMg
    Fe_M2 = FeMgEqui_M2 * (1 - XMg)
    SumM2 = Mn_M2 + Ti_M2 + Mg_M2 + Fe_M2
    
    # M1
    Mn_M1 = Mn/3.
    Mg_M1 = Ti_M2 # contribution from tibt
    Al_M1 = np.minimum(Alvi, 1 - (Mn_M1 + Mg_M1)) # if Alvi + (Mn_M1+Mg_M1) < 1; Al_M1 = Alvi; else Al_M1 = 1 - (Mn_M1 + Mg_M1)
    FeMg_Equi_M1 = (Fe + Mg) - (Fe_M2 + Mg_M2) - Mg_M1
    V_M1 = np.maximum(zeros, 1 - (FeMg_Equi_M1 + Mn_M1 + Mg_M1 + Al_M1))  # if FeMg_M1+Al_M1+Mn_M1+Mg_M1 < 1; then there is an empty site V_M1

    AvailableFeMgM1 = 1 - (Mn_M1 + Mg_M1 + Al_M1 + V_M1)  # remaining FeMg on M1. If V = 0, it is 1-(Mn_M1+Mg_M1+Al_M1); else if V>0, it is FeMg_Equi_M1
    Mg_M1 = Mg - Mg_M2
    Fe_M1 = Fe - Fe_M2
    SumM1 = Mn_M1 + Mg_M1 + Al_M1 + Fe_M1
    
    # A
    K_A = K
    Ca_A = Ca
    Na_A = Na
    V_A = np.maximum(0, 1- (K_A + Ca_A + Na_A))
    SumA = K_A + Na_A + Ca_A
    
    # end-members
    Xtibt = Ti_M2
    Xmnbt = Mn_M2 / 2.
    Xord = 1 - (Xtibt + Xmnbt)
    Xsideast = Al_M1
    Xphlann = Xord - Xsideast
    
    Xann = Xphlann * (1 - XMg)
    Xphl = Xphlann * XMg
    Xsid = Xsideast * (1 - XMg)
    Xeast = Xsideast * XMg
    Xsum = Xtibt + Xmnbt + Xann +Xphl + Xsid + Xeast
        
    return(Si,Ti,Al,Cr,Fe,Mn,Mg,Ca,Na,K,SumCat,Si_T1,Si_T2,Al_T2,SumT2,Al_M1,Mg_M1,Fe_M1,Mn_M1,SumM1,Ti_M2,Mg_M2,Fe_M2,Mn_M2,SumM2,K_A,Ca_A,Na_A,SumA,XMg,Xann,Xphl,Xsid,Xeast,Xtibt,Xmnbt,Xsum)
    
    
#===============================================================================
#===============================================================================  

# Structural formula of feldspars (Ca,Na,K) 
# Checked 13-04-22
#===============================================================================
#  Format: Si Al Fe Mg Ca Na K SumCat SumT SumM Xab Xan Xmic= f(SiO2 TiO2 Al2O3 Cr2O3 FeO MnO MgO CaO Na2O K2O P2O5)
#   --------------------------------------
#   End-members    T1(2)   T2(2)    M(1)
#   --------------------------------------
#   Albite          Si      SiAl     Na 
#   Anorthite       Si      AlAl     Ca 
#   Microcline      Si      SiAl     K    
#   --------------------------------------
#  Normalization to 8 Oxygens

def StrFeldspar(data):

    [Si, Ti, Al, Cr, Fe, Mn, Mg, Ca, Na, K, P, SumCat] = StrGeneral(data,8)
    
    # structural formulas
    Si_T = Si
    Al_T = Al
    K_M1 = K
    Na_M1 = Na
    Ca_M1 = Ca
    SumT = Si_T + Al_T
    SumM = K_M1 + Na_M1 + Ca_M1
    
    # solid solution end-members
    Xab = Na/(Na+Ca+K)
    Xan = Ca/(Na+Ca+K)
    Xmic = K/(Ca+Na+K)
        
    return(Si,Al,Fe,Mg,Ca,Na,K,SumCat,SumT,SumM,Xab,Xan,Xmic)
    
#===============================================================================
#===============================================================================  

# Structural formula of Staurolite
#===============================================================================
#  Format: Si Al Fe Mn Mg SumCat= f(SiO2 TiO2 Al2O3 Cr2O3 FeO MnO MgO CaO Na2O K2O P2O5)
#  22 Oxygens

def StrStaurolite(data):    

    [Si, Ti, Al, Cr, Fe, Mn, Mg, Ca, Na, K, P, SumCat] = StrGeneral(data,22)
    
    st = Fe / (Fe + Mg + Mn)
    mgst = Mg / (Fe + Mg + Mn)
    mnst = Mn / (Fe + Mg + Mn)
    
    return(Si,Al,Fe,Mn,Mg,st,mgst,mnst)
    


 # Structural formula of Amphibole
 #===============================================================================
 #  Format: Si Al Fe Mn Mg SumCat= f(SiO2 TiO2 Al2O3 Cr2O3 FeO MnO MgO CaO Na2O K2O P2O5)
 #  23 Oxygens   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    

    
