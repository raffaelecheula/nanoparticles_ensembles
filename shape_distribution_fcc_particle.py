#!/usr/bin/env python3

################################################################################
# Raffaele Cheula, LCCP, Politecnico di Milano, raffaele.cheula@polimi.it
################################################################################

from __future__ import absolute_import, division, print_function
import os, timeit
try: import cPickle as pickle
except: import _pickle as pickle
import numpy as np
from collections import OrderedDict
from ase.build import bulk
from nanoparticle_units import *
from nanoparticle_utils import e_relax_from_bond_ols
from nanoparticle_cython import calculate_neighbors, FccParticleShape

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

dirname = 'fcc'

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

################################################################################
# BULK
################################################################################

layers_min_100 = 2
layers_max_100 = 4

atoms = bulk(element, bulk_type, a = lattice_constant, cubic = True)

layers_vac = 2 if layers_max_100 % 2 == 0 else 1

size = [layers_max_100+layers_vac]*3

atoms *= size

positions = atoms.get_positions()
cell = atoms.cell

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

plane_dist_100 = lattice_constant/2

max_spherical = False

if max_spherical is True:
    d_max_tot = plane_dist_100*layers_max_100
else:
    d_max_tot = 2.*plane_dist_100*layers_max_100

scale_vect = [(1.0, 1.0), (0.9, 1.0), (1.0, 0.9), (0.8, 1.0), (1.0, 0.8)]
#scale_vect = [(0.7, 1.0), (1.0, 0.7), (0.6, 1.0), (1.0, 0.6)]

n_coord_min = 0

translation_vect = [np.array([0.]*3)                               ,
                    np.array([lattice_constant/2.]+[0.]*2)         ,
                    np.array([np.sqrt(lattice_constant)/2.]*2+[0.]),
                    np.array([np.sqrt(lattice_constant)/2.]*3      )]

################################################################################
# CREATE POPULATION OF PARTICLES
################################################################################

c_max_dict = {}
c_min_dict = {}
plane_dist = {}

d_max_dict = {}
d_min_dict = {}
i_max_dict = {}

d_dict = OrderedDict()

layers_dict = OrderedDict()

for hkl in step_dict:

    hkl_scal = [i/max(hkl) for i in hkl]
    denom    = np.sqrt(sum([i**2 for i in hkl_scal]))

    c_max_dict[hkl] = sum(hkl_scal)/denom
    c_min_dict[hkl] = 1./denom

    plane_dist[hkl] = plane_dist_100/denom*step_dict[hkl]

count = 0

print('\n  N particles   N processes\n')

for j in range(layers_min_100, layers_max_100):

    d_100 = plane_dist_100*j

    for hkl in step_dict:

        d_max_dict[hkl] = d_100*c_max_dict[hkl]
        d_min_dict[hkl] = d_100*c_min_dict[hkl]

        i_max_dict[hkl] = int(np.around((d_max_dict[hkl]-d_min_dict[hkl]) / 
                              plane_dist[hkl]))

    for (a,b,c,d,e,f,g) in [(a,b,c,d,e,f,g) 
                             for a in range(i_max_dict[(1,1,0)])
                             for b in range(i_max_dict[(1,1,1)])
                             for c in range(i_max_dict[(2,1,0)])
                             for d in range(i_max_dict[(2,1,1)])
                             for e in range(i_max_dict[(3,1,0)])
                             for f in range(i_max_dict[(3,1,1)])
                             for g in range(i_max_dict[(3,2,1)])]:

        layers_dict[(1,0,0)] = 0
        layers_dict[(1,1,0)] = a
        layers_dict[(1,1,1)] = b
        layers_dict[(2,1,0)] = c
        layers_dict[(2,1,1)] = d
        layers_dict[(3,1,0)] = e
        layers_dict[(3,1,1)] = f
        layers_dict[(3,2,1)] = g

        for hkl in layers_dict:
            d_dict[hkl] = d_max_dict[hkl]-layers_dict[hkl]*plane_dist[hkl]+1e-3

        d_list = [d_dict[xkl] for xkl in d_dict][1:]

        if max(d_list) < min(d_list)*1.2 and max(d_list) < d_max_tot:

            miller_indices = np.array([hkl for hkl in d_dict])

            planes_distances = np.array([d_dict[hkl] for hkl in d_dict])

            for scale_one, scale_two in scale_vect:

                for translation in translation_vect:

                    if run is True:

                        particle = FccParticleShape(
                                       positions        = np.copy(positions),
                                       neighbors        = np.copy(neighbors),
                                       cell             = cell              ,
                                       translation      = translation       ,
                                       miller_indices   = miller_indices    ,
                                       planes_distances = planes_distances  ,
                                       scale_one        = scale_one         ,
                                       scale_two        = scale_two         ,
                                       n_coord_min      = n_coord_min       ,
                                       interact_len     = interact_len      ,
                                       e_coh_bulk       = e_coh_bulk        ,
                                       e_relax_list     = e_relax_list      )

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

                                if (np.array_equal(n_coord_dist, 
                                                   old.n_coord_dist)):

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
