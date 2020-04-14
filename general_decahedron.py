#!/usr/bin/env python3

################################################################################
# Raffaele Cheula, LCCP, Politecnico di Milano, raffaele.cheula@polimi.it
################################################################################

from __future__ import absolute_import, division, print_function
import os, timeit, random
try: import cPickle as pickle
except: import _pickle as pickle
import numpy as np
import copy as cp
import matplotlib.pyplot as plt
from ase import Atoms
from ase.build import bulk
from nanoparticles.nanoparticle_units import *
from nanoparticles.nanoparticle_utils import (decahedron_grid       ,
                                              e_relax_from_bond_ols ,
                                              cluster_add_adsorbates)
from nanoparticles.active_sites_shells import (get_surface_shell   , 
                                               get_fcc_active_shell,
                                               count_active_sites  , 
                                               plot_kmc_grid       )
from nanoparticles.nanoparticle_cython import (calculate_neighbors, 
                                               DecahedronShape    )

################################################################################
# MEASURE TIME START
################################################################################

measure_time = False

if measure_time is True:
    start = timeit.default_timer()

################################################################################
# PARTICLE DATA
################################################################################

element          = 'Rh'
lattice_constant = +3.8305 # [Ang]

e_coh_bulk       = -6.1502 # [eV]
m_bond_ols       = +2.6800 # [-]
e_twin           = +0.0081 # [eV]
shear_modulus    = +145 * (giga*Pa)*(J/eV)*(Ang/mt)**3 # [eV/Ang^3]

k_strain         = 3.78e-4 # [-]

e_relax_list = e_relax_from_bond_ols(e_coh_bulk = e_coh_bulk,
                                     m_bond_ols = m_bond_ols)

################################################################################
# OPERATIVE CONDITIONS
################################################################################

temperature = Celsius_to_Kelvin(500.) # [K]
pressure    =  1.00 * atm

CO_molfrac  = 0.01

print('temperature = {:9.4f} K'.format(temperature))
print('pressure    = {:9.4f} atm'.format(pressure/atm))
print('CO mol frac = {:9.4f}'.format(CO_molfrac))

################################################################################
# ADSORBATES DATA
################################################################################

averag_e_bind = True
e_form_denom  = 'N met'
thermo_model  = 'HinderedThermo' # HarmonicThermo | HinderedThermo | IdealThermo

adsorbate     = 'CO'
bond_len      = +1.2000     # [Ang]

y_zero_e_bind = -1.8759E+00 # [eV]
m_ang_e_bind  = +4.1877E-02 # [-]

alpha_cov     = +3.0340E+02 # [eV/Ang^2]
beta_cov      = +3.3144E+00 # [-]

y_zero_deltaf_harmonic = -7.999E-01 # [eV/K]
m_ang_deltaf_harmonic  = +2.348E-02 # [-]

y_zero_deltaf_hindered = -8.690E-01 # [eV/K]
m_ang_deltaf_hindered  = +2.770E-02 # [-]

from phases import gas

gas.TPX = temperature, pressure, [CO_molfrac, 1.-CO_molfrac]

delta_mu_ads = gas['CO'].chemical_potentials[0] * (J/eV)/(kmol/molecule) # [eV]

print('\ndeltamu CO  = {:9.4f} eV'.format(delta_mu_ads))

if thermo_model == 'HarmonicThermo':

    entropy_model = '2D lattice gas'
    y_zero_e_bind += y_zero_deltaf_harmonic
    m_ang_e_bind  += m_ang_deltaf_harmonic

elif thermo_model == 'HinderedThermo':

    entropy_model = '2D ideal gas'
    y_zero_e_bind += y_zero_deltaf_hindered
    m_ang_e_bind  += m_ang_deltaf_hindered

elif thermo_model == 'IdealThermo':

    entropy_model = '2D ideal gas'
    y_zero_e_bind += 2./3.*(gas['CO'].standard_gibbs_RT[0]*Rgas*temperature * 
                            (J/eV)/(kmol/molecule)) # [eV]

################################################################################
# BINDING ENERGY CORRECTION FUNCTIONS
################################################################################

f_e_bind_corr = [lambda coverage: 0.]*13

################################################################################
# BULK
################################################################################

size = 9

bulk_atoms = decahedron_grid(element          = element         ,
                             lattice_constant = lattice_constant,
                             size             = size            ,
                             heigth           = 1               )

positions = bulk_atoms.get_positions()
cell = np.array(bulk_atoms.cell)

interact_len = np.sqrt(2*lattice_constant)*1.2

neighbors = calculate_neighbors(positions     = positions   ,
                                cell          = cell        ,
                                interact_len  = interact_len)

################################################################################
# CALCULATION PARAMETERS
################################################################################

layers_max = np.array([6]*5)
layers_min = np.array([4]*5)

planes_miller = np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])

n_coord_min = 6

layers_hole = 0

################################################################################
# CREATE PARTICLE
################################################################################

particle = DecahedronShape(positions        = positions       ,
                           neighbors        = neighbors       ,
                           layers_min       = layers_min      ,
                           layers_max       = layers_max      ,
                           planes_miller    = planes_miller   ,
                           layers_hole      = layers_hole     ,
                           lattice_constant = lattice_constant,
                           n_coord_min      = n_coord_min     ,
                           e_coh_bulk       = e_coh_bulk      ,
                           e_relax_list     = e_relax_list    ,
                           e_twin           = e_twin          ,
                           shear_modulus    = shear_modulus   ,
                           k_strain         = k_strain        )

particle.get_shape()
particle.get_energy()

atoms = Atoms(element+str(particle.n_atoms), positions = particle.positions)

atoms.set_pbc(True)
atoms.set_cell(cell)

atoms.center(vacuum = 10./2.)

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

print('Formation energy = {0:.4f} eV = {1:.4f} eV/atom\n'.format(
      particle.e_form_clean, particle.e_spec_clean))

print('number of atoms in twin boundaries = {}\n'.format(particle.n_twin))

################################################################################
# GET ENERGY WITH ADSORBATES
################################################################################

adsorption = True

if adsorption is True:

    print('\n CO ADSORPTION \n')
    
    particle.get_energy_with_ads(bond_len      = bond_len     ,
                                 y_zero_e_bind = y_zero_e_bind,
                                 m_ang_e_bind  = m_ang_e_bind ,
                                 alpha_cov     = alpha_cov    ,
                                 beta_cov      = beta_cov     ,
                                 temperature   = temperature  ,
                                 delta_mu_ads  = delta_mu_ads ,
                                 f_e_bind_corr = f_e_bind_corr,
                                 entropy_model = entropy_model,
                                 averag_e_bind = averag_e_bind,
                                 e_form_denom  = e_form_denom )
    
    print('coverage = {:6.4f}\n'.format(particle.coverage))
    
    print('Formation energy = {0:+10.4f} eV = {1:+8.4f} eV/atom\n'.format(
        particle.e_form_ads, particle.e_spec_ads))
    
    print('Energy change    = {0:+10.4f} eV = {1:+8.4f} eV/atom\n'.format(
        particle.e_form_ads-particle.e_form_clean,
        particle.e_spec_ads-particle.e_spec_clean))

################################################################################
# FORMATION ERERGY TREND PLOT
################################################################################

show_plot = False

store_plot_vect = False

filename = 'pickle_object.pkl'

if adsorption is True:

    tick_size = 14
    label_size = 16

    fig = plt.figure(1)
    fig.set_size_inches(16, 10)
    
    x_vect = [0.]+list(particle.coverage_list)
    y_vect = [particle.e_spec_clean]+list(particle.e_spec_ads_list)
    
    if store_plot_vect is True:
        
        fileobj = open(filename, 'wb')
        pickle.dump([x_vect, y_vect], file = fileobj)
        fileobj.close()
    
    line, = plt.plot(x_vect, y_vect, marker = 'o', alpha = 0.5)
    point = plt.plot(particle.coverage, particle.e_spec_ads, 
                     marker = 'o', color = line.get_color(),
                     alpha = 0.7, markersize = 10.)
    
    y_min = np.min(particle.e_spec_ads_list)-0.03
    y_max = np.max(particle.e_spec_ads_list)+0.03
    
    plt.axis([0., 1., y_min, y_max])
    
    plt.yticks(fontsize = tick_size)
    plt.xticks(fontsize = tick_size)
    
    plt.ylabel('formation energy [eV/atom]', fontsize = label_size)
    plt.xlabel('coverage', fontsize = label_size)

    if show_plot is True:

        plt.show()

################################################################################
# COUNT ACTIVE SITES
################################################################################

count_sites = True

plot_kmc_grid_3D = False

plot_top_distrib = False

specify_supp_int = False
specify_n_coord  = ('top',)
check_duplicates = False
specify_facets   = False
multiple_facets  = True

half_plot   = False
facet_color = False

n_coord_max = 12

if count_sites is True:

    supp_contact = [False for i in range(particle.n_atoms)]
    
    surface = get_surface_shell(element      = element           ,
                                positions    = particle.positions,
                                neighbors    = particle.neighbors, 
                                indices      = particle.indices  , 
                                n_coord      = particle.n_coord  ,
                                supp_contact = supp_contact      ,
                                n_coord_max  = n_coord_max       )
    
    active_sites = get_fcc_active_shell(surface          = surface         ,
                                        specify_supp_int = specify_supp_int,
                                        specify_n_coord  = specify_n_coord ,
                                        specify_facets   = specify_facets  ,
                                        check_duplicates = check_duplicates,
                                        multiple_facets  = multiple_facets )
    
    active_sites_dict = count_active_sites(active_sites  = active_sites, 
                                           print_distrib = True        )
    
    if plot_kmc_grid_3D is True:

        plot_kmc_grid(active_sites = active_sites, 
                      plot_type    = '3D'        , 
                      half_plot    = half_plot   ,
                      facet_color  = facet_color )

    if plot_top_distrib is True:

        fig = plt.figure(1)
        fig.set_size_inches(10, 10)

        if specify_facets is True:

            plot_top_distribution(active_sites_dict = active_sites_dict,
                                  n_atoms_tot       = particle.n_atoms ,
                                  percentual        = True             )

        else:
            
            plot_sites_distribution(active_sites_dict = active_sites_dict,
                                    n_atoms_tot       = particle.n_atoms ,
                                    percentual        = True             )

################################################################################
# ADD ADSORBATES
################################################################################

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

atoms = cluster_add_adsorbates(atoms      = atoms     ,
                               adsorbate  = CO        ,
                               sites_list = sites_list,
                               distance   = bond_len  )

atoms.center(vacuum = 10./2.)

atoms.write('dec.xsf')

################################################################################
# MEASURE TIME END
################################################################################

if measure_time is True:
    stop = timeit.default_timer()-start
    print('\nExecution time = {0:6.3} s\n'.format(stop))

################################################################################
# END
################################################################################
