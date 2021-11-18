#!/usr/bin/env python3

################################################################################
# Raffaele Cheula, LCCP, Politecnico di Milano, raffaele.cheula@polimi.it
################################################################################

from __future__ import absolute_import, division, print_function
from nanoparticle_units import *
import os, io, timeit
import numpy as np
import pickle
import matplotlib.pyplot as plt
from nanoparticle_utils import e_relax_from_bond_ols, cluster_add_adsorbates

################################################################################
# MEASURE TIME START
################################################################################

measure_time = True

if measure_time is True:
    time_start = timeit.default_timer()

################################################################################
# DATA
################################################################################

dirmother_old = 'Ni_01_hkl_2'
dirmother_new = 'Ni_02_hkl_2'

bulk_types = ['fcc'] # fcc | dec | ico

n_min_read =    20
n_max_read =  5000

count_active_sites  = True
clean_particle_data = True

################################################################################
# MATERIAL DATA
################################################################################

lattice_constant_old = +3.5240 # [Ang]
lattice_constant_new = +3.5240 # [Ang]

################################################################################
# CONVERT PARTICLES
################################################################################

step = 10

n_particles = {}

for bulk_type in bulk_types:

    dirname = os.path.join(dirmother_new, bulk_type)

    if not os.path.isdir(dirname):
        os.makedirs(dirname)

    n_particles[bulk_type] = 0

    n_groups = int((n_max_read-n_min_read)/step)

    print(f'\nAnalyzing {bulk_type} particles')

    for ng, group in enumerate(range(n_min_read, n_max_read, step)):

        print(f'\nGroup {ng+1}/{n_groups}: {group}-{group+step}')

        filename = f'{bulk_type}_{group:04d}_{group+step:04d}.pkl'
        
        filepath = os.path.join(dirmother_old, bulk_type, filename)

        with open(filepath, 'rb') as fileobj:
            particles = pickle.load(fileobj)

        for i_p, particle in enumerate(particles):

            if (i_p+1) % 10 == 0 or (i_p+1) == len(particles):
                print(f'Particle {i_p+1}/{len(particles)}', end = '\r')

            if count_active_sites is True:

                particle.get_active_sites(specify_n_coord  = True ,
                                          specify_supp_int = False,
                                          specify_facets   = False,
                                          check_duplicates = False,
                                          multiple_facets  = False,
                                          convex_sites     = True )
                
                particle.get_active_sites_dict(with_tags = True)

            if clean_particle_data is True:
                
                particle.set_lattice_constant(lc_new = lattice_constant_new,
                                              lc_old = lattice_constant_old)
                
                particle.reduce_data(only_active_sites_dict = True)

            n_particles[bulk_type] += 1

        filepath = os.path.join(dirmother_new, bulk_type, filename)

        with open(filepath, 'wb') as fileobj:
            pickle.dump(particles, file = fileobj)

        print('')

    print(f'\nNumber of {bulk_type} particles: {n_particles[bulk_type]}')

################################################################################
# MEASURE TIME END
################################################################################

if measure_time is True:
    print(f'\nExecution time = {timeit.default_timer()-time_start:6.3} s\n')

################################################################################
# END
################################################################################
