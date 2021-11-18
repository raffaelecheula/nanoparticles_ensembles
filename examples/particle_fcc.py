#!/usr/bin/env python3

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
import os, sys, timeit, random
import pickle
import numpy as np
import copy as cp
import matplotlib.pyplot as plt
from collections import OrderedDict
from itertools import permutations, product
from ase import Atoms
from ase.build import bulk
from nanoparticle_units import *
from nanoparticle_cython import calculate_neighbors, FccParticleShape
from nanoparticle_utils import (e_relax_from_bond_ols ,
                                cluster_add_adsorbates)

################################################################################
# MEASURE TIME START
################################################################################

measure_time = True

if measure_time is True:
    start = timeit.default_timer()

################################################################################
# MATERIAL DATA
################################################################################

element          = 'Ni'
lattice_constant = +3.5240 # [Ang]

e_coh_bulk       = -4.8201 # [eV]
e_twin           = +0.0347 # [eV]
shear_modulus    = +95 * (giga*Pa)*(J/eV)*(Ang/mt)**3 # [eV/Ang^3]

k_strain_dec     = 3.78e-4 # [-]
k_strain_ico     = 4.31e-3 # [-]

a_model_relax    = 1.10e-4 # [eV]
b_model_relax    = 3.50    # [-]

e_relax_list = np.zeros(13)
for i in range(13):
    e_relax_list[i] = a_model_relax*(12-i)**b_model_relax

################################################################################
# OPERATIVE CONDITIONS
################################################################################

temperature = Celsius_to_Kelvin(400.) # [K]
pressure    =  1.00 * atm

x_CO  = 0.01
x_CO2 = 0.10
x_H2  = 0.40
x_N2  = 1.-x_CO-x_CO2-x_H2

print(f'temperature = {temperature:9.4f} K')
print(f'pressure    = {pressure/atm:9.4f} atm')

################################################################################
# ADSORBATES DATA
################################################################################

from phases import gas

gas.TPX = temperature, pressure, [x_CO, x_CO2, x_H2, x_N2]

dmu_CO  = gas['CO'].chemical_potentials[0] * (J/eV)/(kmol/molecule) # [eV]
dmu_CO2 = gas['CO2'].chemical_potentials[0] * (J/eV)/(kmol/molecule) # [eV]
dmu_H2  = gas['H2'].chemical_potentials[0] * (J/eV)/(kmol/molecule) # [eV]

print('deltamu CO  = {:9.4f} eV'.format(dmu_CO))
print('deltamu CO2 = {:9.4f} eV'.format(dmu_CO2))
print('deltamu H2  = {:9.4f} eV'.format(dmu_H2))

bond_length    = +1.20 # [Ang]
sites_equilib  = False
single_cov_fun = True

alpha_cov      = +3.0340E+02 # [eV/Ang^2]
beta_cov       = +3.3144E+00 # [-]

g_bind_dict = {}

def g_bind_fun_top(cn_ave): return (-2.4652-dmu_CO)+(+0.056727)*cn_ave
def g_bind_fun_brg(cn_ave): return (-2.8510-dmu_CO)+(+0.100590)*cn_ave
def g_bind_fun_hcp(cn_ave): return (-2.8899-dmu_CO)+(+0.098000)*cn_ave
def g_bind_fun_fcc(cn_ave): return (-2.8899-dmu_CO)+(+0.098000)*cn_ave
def g_bind_fun_hol(cn_ave): return (-1.8210-dmu_CO)+(-0.030000)*cn_ave
def g_bind_fun_hol(cn_ave): return (-1.8210-dmu_CO)+(-0.030000)*cn_ave

g_bind_dict['top'] = g_bind_fun_top
g_bind_dict['brg'] = g_bind_fun_brg
g_bind_dict['hcp'] = g_bind_fun_hcp
g_bind_dict['fcc'] = g_bind_fun_fcc
g_bind_dict['hol'] = g_bind_fun_hol
g_bind_dict['lbr'] = g_bind_fun_lbr

n_bonds_dict = {}

n_bonds_dict['top'] = 1
n_bonds_dict['brg'] = 2
n_bonds_dict['hcp'] = 3
n_bonds_dict['fcc'] = 3
n_bonds_dict['hol'] = 4
n_bonds_dict['lbr'] = 2

################################################################################
# BULK
################################################################################

layers_max_100 = 8 # 12

bulk_atoms = bulk(element, 'fcc', a = lattice_constant, cubic = True)

layers_vac = 2 if layers_max_100 % 2 == 0 else 1

size = [layers_max_100+layers_vac+2]*3

bulk_atoms *= size

positions    = bulk_atoms.get_positions()
cell         = np.array(bulk_atoms.cell)
interact_len = np.sqrt(2*lattice_constant)*1.2

neighbors = calculate_neighbors(positions    = positions   ,
                                cell         = cell        ,
                                interact_len = interact_len)

################################################################################
# CALCULATION PARAMETERS
################################################################################

step_dict = OrderedDict()

step_dict[(1,0,0)] = 1/1 # 1/1
step_dict[(1,1,0)] = 1/1 # 1/1
step_dict[(1,1,1)] = 1/1 # 1/1
step_dict[(2,1,0)] = 1/1 # 1/2
step_dict[(2,1,1)] = 1/1 # 1/2
step_dict[(3,1,0)] = 1/1 # 1/3
step_dict[(3,1,1)] = 1/1 # 1/3
step_dict[(3,2,1)] = 1/1 # 1/3

layers_dict = OrderedDict()

layers_dict[(1,0,0)] = 0
layers_dict[(1,1,0)] = 4
layers_dict[(1,1,1)] = 6 # 18
layers_dict[(2,1,0)] = 0
layers_dict[(2,1,1)] = 0
layers_dict[(3,1,0)] = 0
layers_dict[(3,1,1)] = 0
layers_dict[(3,2,1)] = 0

translation_type = 0

scale_one = 1.
scale_two = 1.

n_coord_min = 0

miller_symmetry = True

################################################################################
# CREATE PARTICLE
################################################################################

plane_dist_100 = lattice_constant/2

c_max_dict = {}
c_min_dict = {}
plane_dist = {}

d_max_dict = {}
d_min_dict = {}
i_max_dict = {}

d_dict_all = OrderedDict()

d_100 = plane_dist_100 * layers_max_100

for hkl in step_dict:

    hkl_scal = [i/max(hkl) for i in hkl]
    denom    = np.sqrt(sum([i**2 for i in hkl_scal]))

    c_max_dict[hkl] = sum(hkl_scal)/denom
    c_min_dict[hkl] = 1./denom

    plane_dist[hkl] = plane_dist_100/denom*step_dict[hkl]

for hkl in step_dict:

    d_max_dict[hkl] = d_100*c_max_dict[hkl]
    d_min_dict[hkl] = d_100*c_min_dict[hkl]

    i_max_dict[hkl] = int(np.around((d_max_dict[hkl]-d_min_dict[hkl]) / 
                          plane_dist[hkl]))

for hkl in layers_dict:

    d_dict_all[hkl] = d_max_dict[hkl]-layers_dict[hkl]*plane_dist[hkl]+1e-3

miller_indices = np.array([hkl for hkl in d_dict_all])

planes_distances = np.array([d_dict_all[hkl] for hkl in d_dict_all])

translation_vect = [np.array([0.]*3)                               ,
                    np.array([lattice_constant/2.]+[0.]*2)         ,
                    np.array([np.sqrt(lattice_constant)/2.]*2+[0.]),
                    np.array([np.sqrt(lattice_constant)/2.]*3      )]

translation = translation_vect[translation_type]

print('\n CLEAN PARTICLE \n')

particle = FccParticleShape(positions        = positions       ,
                            neighbors        = neighbors       ,
                            cell             = cell            ,
                            lattice_constant = lattice_constant,
                            translation      = translation     ,
                            miller_indices   = miller_indices  ,
                            planes_distances = planes_distances,
                            scale_one        = scale_one       ,
                            scale_two        = scale_two       ,
                            n_coord_min      = n_coord_min     ,
                            interact_len     = interact_len    ,
                            e_coh_bulk       = e_coh_bulk      ,
                            e_relax_list     = e_relax_list    ,
                            miller_symmetry  = miller_symmetry )

particle.get_shape()

particle.get_energy_clean(energy_model = 'SRB')

atoms = particle.to_ase_atoms(symbol = element)

atoms.write('pw.xsf')

atoms.edit()

################################################################################
# PRINT COORDINATION NUMBERS AND FORMATION ENERGY
################################################################################

print('Number of atoms = {:d}\n'.format(particle.n_atoms))

print('Coordination number distribution:')
for i in range(13):
    print('{:5d}'.format(i), end = ' ')
print('')
for i in range(13):
    print('{:5d}'.format(particle.n_coord_dist[i]), end = ' ')
print('\n')

n_coord_ave = particle.get_n_coord_ave()

print('Average coordination number = {:.2f}\n'.format(n_coord_ave))

print('Formation energy = {0:+10.4f} eV = {1:+8.4f} eV/atom'.format(
      particle.e_form_clean, particle.e_spec_clean))

print('Diameter = {0:+10.4f} A'.format(particle.get_diameter()))

################################################################################
# GET ENERGY WITH ADSORBATES
################################################################################

adsorption = True

if adsorption is True:

    print('\n CO ADSORPTION \n')
    
    particle.get_active_sites(specify_n_coord  = True ,
                              specify_supp_int = False,
                              specify_facets   = False,
                              check_duplicates = False,
                              multiple_facets  = False,
                              convex_sites     = True )
    
    particle.get_energy_with_ads(g_bind_dict    = g_bind_dict   ,
                                 n_bonds_dict   = n_bonds_dict  ,
                                 alpha_cov      = alpha_cov     ,
                                 beta_cov       = beta_cov      ,
                                 bond_length    = bond_length   ,
                                 temperature    = temperature   ,
                                 sites_equilib  = sites_equilib ,
                                 single_cov_fun = single_cov_fun)
    
    print('coverage = {:6.4f}\n'.format(particle.coverage))
    
    print('Formation energy = {0:+10.4f} eV = {1:+8.4f} eV/atom\n'.format(
        particle.e_form_ads, particle.e_spec_ads))
    
    print('Energy change    = {0:+10.4f} eV = {1:+8.4f} eV/atom\n'.format(
        particle.e_form_ads-particle.e_form_clean,
        particle.e_spec_ads-particle.e_spec_clean))

################################################################################
# ADD ADSORBATES
################################################################################

plot_energy = False

if plot_energy is True:

    fig = plt.figure(2)
    fig.set_size_inches(8, 6)
    
    x_vect = particle.coverage_vect
    y_vect = particle.e_form_ads_vect
    
    line, = plt.plot(x_vect, y_vect, marker = 'o', alpha = 0.5)
    
    point = plt.plot(particle.coverage, particle.e_form_ads, 
                        marker     = 'o'                      ,
                        color      = line.get_color()         ,
                        alpha      = 0.7                      ,
                        markersize = 10                       )
    
    plt.axis([0., 1., min(y_vect)-5., max(y_vect)+5.])
    
    tick_size  = 14
    label_size = 16
    
    plt.yticks(fontsize = tick_size)
    plt.xticks(fontsize = tick_size)
    
    plt.ylabel('formation energy [eV/atom]', fontsize = label_size)
    plt.xlabel('coverage [-]', fontsize = label_size)

    plt.show()

################################################################################
# ADD ADSORBATES
################################################################################

add_adsorbates = False

if add_adsorbates is True:

    random_adsorption = True
    
    CO = Atoms('CO', positions = [(0., 0., 0.), (0., 0., 1.2)])
    
    n_top = len([i for i in particle.n_coord if particle.n_coord[i] < 10])
    n_coord_zip = list(zip(range(len(particle.n_coord)), particle.n_coord))
    
    n_coord_dict = {}
    for i in range(len(particle.n_coord)):
        try: n_coord_dict[particle.n_coord[i]] += [i]
        except: n_coord_dict[particle.n_coord[i]] = [i]
    
    top_list = []
    for i in range(10):
        try:
            random.shuffle(n_coord_dict[i]) 
            top_list += n_coord_dict[i]
        except: pass
    
    if adsorption is True:
        coverage = particle.coverage
    else:
        coverage = 0.
    
    top_list_original = np.copy(top_list)
    
    if random_adsorption is True:
        random.shuffle(top_list)
    
    n_ads = int(coverage*len(top_list))
    
    sites_list = top_list[:n_ads]
    
    coverage = len(sites_list)/len(top_list)
    print('nanoparticle coverage = {:.2f} ML\n'.format(coverage))
    
    atoms = cluster_add_adsorbates(atoms      = atoms      ,
                                   adsorbate  = CO         ,
                                   sites_list = sites_list ,
                                   distance   = bond_length)
    
    atoms.center(vacuum = 10./2.)

    atoms.write('pw.xsf')

################################################################################
# MEASURE TIME END
################################################################################

if measure_time is True:
    stop = timeit.default_timer()-start
    print('\nExecution time = {0:6.3} s\n'.format(stop))

################################################################################
# END
################################################################################
