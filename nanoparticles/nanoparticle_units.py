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
import ase.units

################################################################################
# PREFIX
################################################################################

exa   = 1e+18
peta  = 1e+15
tera  = 1e+12
giga  = 1e+09
mega  = 1e+06
kilo  = 1e+03
hecto = 1e+02
deca  = 1e+01
deci  = 1e-01
centi =	1e-02
milli = 1e-03
micro = 1e-06
nano  = 1e-09
pico  = 1e-12
femto = 1e-15
atto  = 1e-18

################################################################################
# BASIC UNITS
################################################################################

meter    = mt   = 1. # [mt]
second   = sec  = 1. # [sec]
kilogram = kg   = 1. # [kg]
kilomole = kmol = 1. # [kmol]
Kelvin   = K    = 1. # [K]
Pascal   = Pa   = 1. # [kg/mt/sec^2]
Newton   = N    = 1. # [kg*mt/sec^2]
Joule    = J    = 1. # [kg*mt^2/sec^2]
Watt     = W    = 1. # [kg*mt^2/sec^3]
gram     = kilogram/kilo
mol      = mole = kilomole/kilo

################################################################################
# DERIVED UNITS
################################################################################

Angstrom   = Ang = 1e-10*meter
inch       = 0.0254*meter
litre      = 1e-03*meter**3
minute     = 60*second
hour       = 3600*second
Hertz      = Hz = 1/second
Dalton     = amu = 1.660539040e-27*kg
atmosphere = atm = 101325*Pascal
bar        = 1e+05*Pascal
calorie    = cal = 4.184*Joule
eV         = 1e+03/ase.units.kJ*Joule
molecule   = 1./ase.units.mol*mole

################################################################################
# CONSTANTS
################################################################################

k_Boltzmann = kB   = ase.units.kB*eV   # [J/K]
h_Planck    = hP   = 6.62607e-34*J*sec # [J*sec]
N_Avogadro  = Navo = 1./molecule       # [1/kmol]
R_ideal_gas = Rgas = kB*Navo           # [J/kmol/K]
c_light     = c_l  = ase.units._c      # [m/s]

################################################################################
# CONSTRUCTED UNITS
################################################################################

decimeter  = deci*meter
centimeter = centi*meter
millimeter = milli*meter
micrometer = micro*meter
nanometer  = nano*meter

hectogram = hecto*gram
decagram  = deca*gram
decigram  = deci*gram
centigram = centi*gram
milligram = milli*gram
microgram = micro*gram
nanogram  = nano*gram

millimole = milli*mole
micromole = micro*mole
nanomole  = nano*mole

kiloPascal = kilo*Pascal
megaPascal = mega*Pascal

kiloNewton = kilo*Newton
megaNewton = mega*Newton

kiloJoule  = kilo*Joule
megaJoule  = mega*Joule

################################################################################
# CELSIUS TO KELVIN
################################################################################

def Celsius_to_Kelvin(n):

    return n+273.15 # [K]

################################################################################
# KELVIN TO CELSIUS
################################################################################

def Kelvin_to_Celsius(n):

    return n-273.15 # [C]

################################################################################
# NORMAL LITER
################################################################################

def NormalLiter(temperature, pressure):

    return 1e-3 * (101325/pressure) * (temperature/273.15) # [m^3]

################################################################################
# NORMAL CUBIC METER
################################################################################

def NormalCubicMeter(temperature, pressure):

    return (101325/pressure) * (temperature/273.15) # [m^3]

################################################################################
# END
################################################################################
