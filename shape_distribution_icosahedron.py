#!/usr/bin/env python3

################################################################################
# Raffaele Cheula, LCCP, Politecnico di Milano, raffaele.cheula@polimi.it
################################################################################

from __future__ import absolute_import, division, print_function
import os, timeit
try: import cPickle as pickle
except: import _pickle as pickle
import numpy as np
from ase.build import bulk
from nanoparticle_units import *
from nanoparticle_utils import icosahedron_grid, e_relax_from_bond_ols
from nanoparticle_cython import calculate_neighbors, IcosahedronShape

################################################################################
# RUN
################################################################################

run = True

################################################################################
# MEASURE TIME START
################################################################################

measure_time = True

if measure_time is True:
    start = timeit.default_timer()

################################################################################
# READ PARTICLES
################################################################################

dirname = 'ico'

n_max = 1500 # [atom]
step  = 10   # [atom]

min_diff_n = 1     # [atom]
min_diff_e = 0.005 # [eV/atom]

if not os.path.isdir(dirname):
    os.mkdir(dirname)

particles_dict = {}

n_particles_old = 0

if run is True:

    for n_group in range(0, n_max, step):

        group = '{0:04d}_{1:04d}'.format(n_group, n_group+step)
        filename = os.path.join(dirname, '{0}_{1}.pkl'.format(dirname, group))

        if not os.path.isfile(filename):
            particles_dict[n_group] = []
        else:
            fileobj = open(filename, 'rb')
            particles_dict[n_group] = pickle.load(fileobj)
            fileobj.close()
            n_particles_old += len(particles_dict[n_group])

################################################################################
# PARTICLES DATA
################################################################################

element          = 'Rh'
lattice_constant = +3.8305 # [Ang]

e_coh_bulk       = -6.1502 # [eV]
m_bond_ols       = +2.6800 # [-]
e_twin           = +0.0081 # [eV]
shear_modulus    = +145 * giga*Pa * J/eV * (Ang/mt)**3 # [eV/Ang**3]

k_strain         = 4.31e-3 # [-]

e_relax_list = e_relax_from_bond_ols(e_coh_bulk = e_coh_bulk,
                                     m_bond_ols = m_bond_ols)

################################################################################
# CALCULATION PARAMETERS
################################################################################

size = 18

size_vector = range(2, 17)

size_vector = range(2, 4)

n_coord_min_vect = [0, 7]

n_coord_min_iter_vect = range(12)

disorder_vector = [False, True]

################################################################################
# CREATE POPULATION OF PARTICLES
################################################################################

interact_len = np.sqrt(2*lattice_constant)*1.2

atoms = icosahedron_grid(element          = element         ,
                         lattice_constant = lattice_constant,
                         size             = size            )

positions = atoms.get_positions()
cell = atoms.cell

neighbors = calculate_neighbors(positions     = positions   ,
                                cell          = cell        ,
                                interact_len  = interact_len)

count = 0

print('\n  N particles   N processes\n')

for size in size_vector:

    for n_cut in range(20):

        for disorder in disorder_vector:

            if disorder is False:
                layers = np.array([size]*(20-n_cut)+[size-1]*n_cut)
            else:
                if n_cut < 10:
                    layers = np.array([size]*(20-2*n_cut)+[size, size-1]*n_cut)
                else:
                    layers = np.array([size, size-1]*(20-n_cut) + 
                                      [size-1]*(2*n_cut-20))
    
            for n_coord_min in n_coord_min_vect:
    
                for n_coord_min_iter in n_coord_min_iter_vect:
    
                    if run is True:
    
                        particle =  IcosahedronShape(
                                        positions        = positions       ,
                                        neighbors        = neighbors       ,
                                        layers           = layers          ,
                                        lattice_constant = lattice_constant,
                                        n_coord_min      = n_coord_min     ,
                                        n_coord_min_iter = n_coord_min_iter,
                                        e_coh_bulk       = e_coh_bulk      ,
                                        e_relax_list     = e_relax_list    ,
                                        e_twin           = e_twin          ,
                                        shear_modulus    = shear_modulus   ,
                                        k_strain         = k_strain        )
    
                        try:
                            particle.get_shape()
                            particle.get_energy()
                            n_atoms = particle.n_atoms

                        except:
                            n_atoms = 0

                        if 0 < n_atoms < n_max:
    
                            duplicate = False
                            duplicates = []
    
                            n_group = int(np.floor(n_atoms/float(step))*step)
    
                            n_coord_dist = particle.n_coord_dist
                            e_spec_clean = particle.e_spec_clean
    
                            for index in range(len(particles_dict[n_group])):
    
                                old = particles_dict[n_group][index]
    
                                if np.array_equal(n_coord_dist,
                                                  old.n_coord_dist):
    
                                    duplicate = True
                                    break
    
                                elif abs(n_atoms-old.n_atoms) <= min_diff_n:
    
                                    diff_e = e_spec_clean-old.e_spec_clean
    
                                    if 0. < diff_e < min_diff_e:
    
                                        duplicate = True
                                        break
    
                                    elif -min_diff_e < diff_e <= 0.:
    
                                        duplicates += [index]

                            for i in sorted(duplicates, reverse = True):
                                del particles_dict[n_group][i]
    
                            if duplicate is False:
                                particles_dict[n_group] += [particle]
    
                    if count % 100 == 0.:
                        print('{0:13d} {1:13d}'.format(count, 1))
                    count += 1

################################################################################
# STORE PARTICLES
################################################################################

particles_dict = dict(particles_dict)

if run is True:

    for n_group in range(0, n_max, step):

        group = '{0:04d}_{1:04d}'.format(n_group, n_group+step)

        filename = os.path.join(dirname, '{0}_{1}.pkl'.format(dirname, group))

        fileobj = open(filename, 'wb')
        pickle.dump(particles_dict[n_group], file = fileobj)
        fileobj.close()

n_particles = sum([len(particles_dict[n]) for n in particles_dict])

print('\nNumber of particles tot: {:13d}'.format(n_particles))
print('\nNumber of particles new: {:13d}'.format(n_particles-n_particles_old))

################################################################################
# MEASURE TIME END
################################################################################

if measure_time is True:
    stop = timeit.default_timer()-start
    print('\nExecution time = {0:6.3} s\n'.format(stop))

################################################################################
# END
################################################################################
