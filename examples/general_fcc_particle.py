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
import os, timeit, random
try: import cPickle as pickle
except: import _pickle as pickle
import numpy as np
import copy as cp
import matplotlib.pyplot as plt
from collections import OrderedDict
from itertools import permutations, product
from ase import Atoms
from ase.build import bulk
from nanoparticles.nanoparticle_units import *
from nanoparticles.nanoparticle_utils import (e_relax_from_bond_ols ,
                                              cluster_add_adsorbates)
from nanoparticles.active_sites_shells import (get_surface_shell      , 
                                               get_fcc_active_shell   ,
                                               count_active_sites     , 
                                               plot_kmc_grid          ,
                                               plot_top_distribution  ,
                                               plot_sites_distribution)
from nanoparticles.nanoparticle_cython import (calculate_neighbors,
                                               FccParticleShape   )

################################################################################
# MEASURE TIME START
################################################################################

measure_time = False

if measure_time is True:
    start = timeit.default_timer()

################################################################################
# PARTICLE DATA
################################################################################

bulk_type        = 'fcc'
element          = 'Rh'
lattice_constant = +3.8305 # [Ang]

e_coh_bulk       = -6.1502 # [eV]
m_bond_ols       = +2.6800 # [-]
e_twin           = +0.0081 # [eV]
shear_modulus    = +155 * (giga*Pa)*(J/eV)*(Ang/mt)**3 # [eV/Ang^3]

k_strain_dec     = 3.78e-4 # [-]
k_strain_ico     = 4.31e-3 # [-]

e_relax_list = e_relax_from_bond_ols(e_coh_bulk = e_coh_bulk,
                                     m_bond_ols = m_bond_ols)

deltaf_surf_atom = +0.1270 # [eV]

################################################################################
# OPERATIVE CONDITIONS
################################################################################

temperature = Celsius_to_Kelvin(500.) # [K]
pressure    =  1.00 * atm

CO_molfrac  = 0.05

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

layers_max_100 = 7

bulk_atoms = bulk(element, bulk_type, a = lattice_constant, cubic = True)

layers_vac = 2 if layers_max_100 % 2 == 0 else 1

size = [layers_max_100+layers_vac+2]*3

bulk_atoms *= size

positions = bulk_atoms.get_positions()
cell = np.array(bulk_atoms.cell)

interact_len = np.sqrt(2*lattice_constant)*1.2

neighbors = calculate_neighbors(positions     = positions   ,
                                cell          = cell        ,
                                interact_len  = interact_len)

################################################################################
# CALCULATION PARAMETERS
################################################################################

step_dict = OrderedDict()

step_dict[(1,0,0)] = 1/1
step_dict[(1,1,0)] = 1/1
step_dict[(1,1,1)] = 1/1
step_dict[(2,1,0)] = 1/2
step_dict[(2,1,1)] = 1/2
step_dict[(3,1,0)] = 1/3
step_dict[(3,1,1)] = 1/3
step_dict[(3,2,1)] = 1/3
step_dict[(3,3,1)] = 1/3

layers_dict = OrderedDict()

dist_dict = OrderedDict()

dist = 10.

dist_dict[(1,0,0)] = dist
dist_dict[(1,1,0)] = dist
dist_dict[(1,1,1)] = dist
dist_dict[(2,1,0)] = dist
dist_dict[(2,1,1)] = dist
dist_dict[(3,1,0)] = dist
dist_dict[(3,1,1)] = dist
dist_dict[(3,2,1)] = dist
dist_dict[(3,3,1)] = dist

translation_type = 0

scale_one = 1.
scale_two = 1.

if translation_type == 0:
    translation = np.array([0.]*3)
elif translation_type == 1:
    translation = np.array([lattice_constant/2.]+[0.]*2)
elif translation_type == 2:
    translation = np.array([np.sqrt(lattice_constant)/2.]*2+[0.])
elif translation_type == 3:
    translation = np.array([np.sqrt(lattice_constant)/2.]*3)

n_coord_min = 0

miller_symmetry = True

sign_vect = list(product((+1, -1), repeat = 3))

miller_indices_all = {}

for hkl in step_dict:

    indices = []
    
    for sign in sign_vect:
        index_sign = tuple(np.multiply(hkl, sign))
        for index_perm in permutations(index_sign, 3):
            indices += [index_perm]
    
    indices = list(dict.fromkeys(indices))

    miller_indices_all[hkl] = indices

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

d_dict = OrderedDict()
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

    d_dict[hkl] = d_max_dict[hkl]-layers_dict[hkl]*plane_dist[hkl] + 1e-3

for hkl in [hkl for hkl in dist_dict if dist_dict[hkl] is not None]:

    d_dict[hkl] = dist_dict[hkl]

if miller_symmetry is False:

    for hkl in d_dict:
        for hkl_all in miller_indices_all[hkl]:
            d_dict_all[hkl_all] = d_dict[hkl]

else:
    d_dict_all = d_dict

miller_indices = np.array([hkl for hkl in d_dict_all])

planes_distances = np.array([d_dict_all[hkl] for hkl in d_dict_all])

print('\n CLEAN PARTICLE \n')

particle = FccParticleShape(positions        = positions       ,
                            neighbors        = neighbors       ,
                            cell             = cell            ,
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
particle.get_energy()

atoms = Atoms(element+str(particle.n_atoms), positions = particle.positions)

atoms.set_pbc(True)
atoms.set_cell(cell)

n_atoms_remove = 0

animation = []

e_form_list = []
n_atoms_list = []

for i in range(n_atoms_remove):

    particle.remove_atoms(n_iterations  = 1    ,
                          remove_groups = False)
    
    particle.get_energy()
    
    e_form_list += [particle.e_spec]
    n_atoms_list += [particle.n_atoms]
    
    atoms = Atoms(element+str(particle.n_atoms), positions = particle.positions)
    
    atoms.set_pbc(True)
    atoms.set_cell(cell)
    
    animation += [atoms]
    
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

print('Formation energy = {0:+10.4f} eV = {1:+8.4f} eV/atom'.format(
      particle.e_form_clean, particle.e_spec_clean))

################################################################################
# SHOW PLOT
################################################################################

show_plot = False

if show_plot is True:

    tick_size = 14
    label_size = 16
    
    fig = plt.figure(0)
    fig.set_size_inches(16, 10)
    
    plt.plot(n_atoms_list, e_form_list, marker = 'o', alpha = 0.5)
    
    plt.yticks(fontsize = tick_size)
    plt.xticks(fontsize = tick_size)
    
    plt.ylabel('formation energy [eV/atom]', fontsize = label_size)
    plt.xlabel('N atoms', fontsize = label_size)

    plt.show()

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

    x_vect = [0.]+list(particle.coverage_list)
    y_vect = [particle.e_spec_clean]+list(particle.e_spec_ads_list)
    
    if store_plot_vect is True:
        
        fileobj = open(filename, 'wb')
        pickle.dump([x_vect, y_vect], file = fileobj)
        fileobj.close()

    if show_plot is True:

        tick_size = 14
        label_size = 16
    
        fig = plt.figure(1)
        fig.set_size_inches(16, 10)
        
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

        plt.show()

################################################################################
# STORE PARTICLES
################################################################################

store_particle = False

dirname = 'fcc'

n_max = 1500 # [atom]
step  = 10

if not os.path.isdir(dirname):
    os.mkdir(dirname)

particles = {}

if store_particle is True:

    n_group = int(np.floor(particle.n_atoms/float(step))*step)

    group = '{0:04d}_{1:04d}'.format(n_group, n_group+step)
    filename = os.path.join(dirname, '{0}_{1}.pkl'.format(dirname, group))
    
    if not os.path.isfile(filename):
        fileobj = open(filename, 'wb')
        particles[n_group] = []
        pickle.dump(particles[n_group], file = fileobj)
        fileobj.close()
    
    fileobj = open(filename, 'rb')
    particles[n_group] = pickle.load(fileobj)
    fileobj.close()

    particles[n_group] += [particle]

    fileobj = open(filename, 'wb')
    pickle.dump(particles[n_group], file = fileobj)
    fileobj.close()

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

atoms.write('fcc.xsf')

################################################################################
# MEASURE TIME END
################################################################################

if measure_time is True:
    stop = timeit.default_timer()-start
    print('\nExecution time = {0:6.3} s\n'.format(stop))

################################################################################
# END
################################################################################
