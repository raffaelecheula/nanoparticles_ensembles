################################################################################
# Raffaele Cheula*[a][b], Matteo Maestri**[a], Giannis Mpourmpakis***[b]
# [a] Politecnico di Milano, [b] University of Pittsburgh
# * raffaele.cheula@polimi.it
# ** matteo.maestri@polimi.it
# *** gmpourmp@pitt.edu
# Modeling Morphology and Catalytic Activity of Nanoparticle Ensembles 
# Under Reaction Conditions
# ACS Catalysis 2020, 10, 11, 6149–6158
################################################################################

from __future__ import absolute_import, division, print_function
import numpy as np
import cantera as ct
from shape.units import *

################################################################################
# GAS
################################################################################

T_low  =  200.00 # [K]
T_mid  = 1000.00 # [K]
T_high = 3500.00 # [K]
P_ref  =    1.00 * atm

CO_ref  = -1.41518724E+04
CO2_ref = -4.87591660E+04
H2_ref  = -8.35033546E+02
N2_ref  = -9.22795384E+02

H_ref  = H2_ref/2.
O_ref  = CO2_ref-CO_ref
C_ref  = 2*CO_ref-CO2_ref
N_ref  = N2_ref/2.

################################################################################
# CO
################################################################################

CO = ct.Species(name = 'CO')

coeffs = np.zeros(15)

coeffs[0] = T_mid
coeffs[1:8]  = [ 3.57953347E+00, -6.10353680E-04,  1.01681433E-06,
                 9.07005884E-10, -9.04424499E-13, -1.43440860E+04-C_ref-O_ref,
                 3.50840928E+00]
coeffs[8:15] = [ 2.71518561E+00,  2.06252743E-03, -9.98825771E-07,
                 2.30053008E-10, -2.03647716E-14, -1.41518724E+04-C_ref-O_ref,
                 7.81868772E+00]

CO.thermo = ct.NasaPoly2(T_low  = T_low  ,
                         T_high = T_high ,
                         P_ref  = P_ref  ,
                         coeffs = coeffs )

################################################################################
# CO2
################################################################################

CO2 = ct.Species(name = 'CO2')

coeffs = np.zeros(15)

coeffs[0] = T_mid
coeffs[1:8]  = [ 2.35677352E+00,  8.98459677E-03, -7.12356269E-06,
                 2.45919022E-09, -1.43699548E-13, -4.83719697E+04-C_ref-2*O_ref,
                 9.90105222E+00]
coeffs[8:15] = [ 3.85746029E+00,  4.41437026E-03, -2.21481404E-06,
                 5.23490188E-10, -4.72084164E-14, -4.87591660E+04-C_ref-2*O_ref,
                 2.27163806E+00]

CO2.thermo = ct.NasaPoly2(T_low  = T_low  ,
                          T_high = T_high ,
                          P_ref  = P_ref  ,
                          coeffs = coeffs )

################################################################################
# H2
################################################################################

H2 = ct.Species(name = 'H2')

coeffs = np.zeros(15)

coeffs[0] = T_mid
coeffs[1:8]  = [ 3.29812400E+00,  8.24944120E-04, -8.14301470E-07,
                -9.47543430E-11,  4.13487200E-13, -1.01252100E+03-2*H_ref,
                -3.29409400E+00]
coeffs[8:15] = [ 2.99142220E+00,  7.00064410E-04, -5.63382800E-08,
                -9.23157820E-12,  1.58275200E-15, -8.35033546E+02-2*H_ref,
                -1.35510641E+00]

H2.thermo = ct.NasaPoly2(T_low  = T_low  ,
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
                 5.64151480E-09, -2.44485400E-12, -1.02090000E+03-2*N_ref,
                 3.95037200E+00]
coeffs[8:15] = [ 2.92663788E+00,  1.48797700E-03, -5.68476030E-07,
                 1.00970400E-10, -6.75335090E-15, -9.22795384E+02-2*N_ref,
                 5.98054018E+00]

N2.thermo = ct.NasaPoly2(T_low  = T_low  ,
                         T_high = T_high ,
                         P_ref  = P_ref  ,
                         coeffs = coeffs )

################################################################################
# GAS
################################################################################

gas = ct.Solution(thermo = 'IdealGas', species = [CO, CO2, H2, N2])

################################################################################
# END
################################################################################
