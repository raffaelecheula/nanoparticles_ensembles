#!/usr/bin/env python3

################################################################################
# Raffaele Cheula*[a][b], Matteo Maestri**[a], Giannis Mpourmpakis***[b]
# [a] Politecnico di Milano, [b] University of Pittsburgh
# * raffaele.cheula@polimi.it
# ** matteo.maestri@polimi.it
# *** gmpourmp@pitt.edu
# Modeling Morphology and Catalytic Activity of Nanoparticle Ensembles 
# Under Reaction Conditions
# ACS Catalysis 2020
################################################################################

from __future__ import absolute_import, division, print_function
import numpy as np
import cantera as ct
from nanoparticles.nanoparticle_units import *

################################################################################
# GAS
################################################################################

T_low  =  200.00 # [K]
T_mid  = 1000.00 # [K]
T_high = 3500.00 # [K]
P_ref  =    1.00 * atm

################################################################################
# CO
################################################################################

CO = ct.Species(name = 'CO')

coeffs = np.zeros(15)

coeffs[0] = T_mid
coeffs[1:8]  = [ 3.57953347E+00, -6.10353680E-04,  1.01681433E-06,
                 9.07005884E-10, -9.04424499E-13,  0.00000000E+00,
                 3.50840928E+00]
coeffs[8:15] = [ 2.71518561E+00,  2.06252743E-03, -9.98825771E-07,
                 2.30053008E-10, -2.03647716E-14,  1.92213599E+02,
                 7.81868772E+00]

CO.thermo = ct.NasaPoly2(T_low  = T_low  ,
                         T_high = T_high ,
                         P_ref  = P_ref  ,
                         coeffs = coeffs )

################################################################################
# N2
################################################################################

N2 = ct.Species(name = 'N2')

coeffs = np.zeros(15)

coeffs[0] = T_mid
coeffs[1:8]  = [ 3.29867700E+00,  1.40823990E-03, -3.96322180E-06,
                 5.64151480E-09, -2.44485400E-12,  0.00000000E+00,
                 3.95037200E+00]
coeffs[8:15] = [ 2.92663788E+00,  1.48797700E-03, -5.68476030E-07,
                 1.00970400E-10, -6.75335090E-15, +9.81046160E+01,
                 5.98054018E+00]

N2.thermo = ct.NasaPoly2(T_low  = T_low  ,
                         T_high = T_high ,
                         P_ref  = P_ref  ,
                         coeffs = coeffs )

################################################################################
# GAS
################################################################################

gas = ct.Solution(thermo = 'IdealGas', species = [CO, N2])

################################################################################
# END
################################################################################
