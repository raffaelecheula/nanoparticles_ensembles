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
import os
import sys
import time
import timeit
import pickle
import numpy as np
import multiprocessing as mp
from collections import OrderedDict
from ase.build import bulk
from nanoparticles.nanoparticle_units import *
from nanoparticles.nanoparticle_cython import (calculate_neighbors, 
                                               FccParticleShape)

################################################################################
# RUN
################################################################################

run = False

################################################################################
# PRINT N CPUS
################################################################################

sys.path.append(os.getcwd())

try: ncpus = os.environ['SLURM_JOB_CPUS_PER_NODE']
except KeyError: ncpus = mp.cpu_count()

print('Number of cpus: {}'.format(ncpus))

################################################################################
# MEASURE TIME START
################################################################################

measure_time = True

if measure_time is True:
    start = timeit.default_timer()

################################################################################
# READ PARTICLES
################################################################################

dirmother = 'Ni_01_hkl_2'

bulk_type = 'fcc'

n_max = 30000 # [atom]
step  =    10 # [atom]

dirname = os.path.join(dirmother, bulk_type)

if not os.path.isdir(dirname):
    os.makedirs(dirname)

manager = mp.Manager()

particles_dict = manager.dict() 
particles_lock = mp.Lock()

n_particles_old = 0

if run is True:

    for n_group in range(0, n_max, step):

        group = '{0:04d}_{1:04d}'.format(n_group, n_group+step)
        filepath = os.path.join(dirname, '{0}_{1}.pkl'.format(bulk_type, group))

        if not os.path.isfile(filepath):
            particles_dict[n_group] = []
        
        else:
            with open(filepath, 'rb') as fileobj:
                particles_dict[n_group] = pickle.load(fileobj)
            n_particles_old += len(particles_dict[n_group])

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
# BULK
################################################################################

layers_min_100 =  1 # modify this to produce more particles
layers_max_100 = 28 # modify this to produce more particles

atoms = bulk(element, 'fcc', a = lattice_constant, cubic = True)

layers_vac = 2 if layers_max_100 % 2 == 0 else 1

size = [layers_max_100+layers_vac]*3

atoms *= size

positions = atoms.get_positions()
cell = np.array(atoms.cell)

interact_len = np.sqrt(2*lattice_constant)*1.2

neighbors = calculate_neighbors(positions    = positions   ,
                                cell         = cell        ,
                                interact_len = interact_len)

################################################################################
# CREATE PARTICLE
################################################################################

def create_particle(positions       ,
                    neighbors       ,
                    cell            ,
                    lattice_constant,
                    translation     ,
                    miller_indices  ,
                    planes_distances,
                    scale_one       ,
                    scale_two       ,
                    n_coord_min     ,
                    interact_len    ,
                    particles_dict  ,
                    particles_lock  ):

    particle = FccParticleShape(positions        = np.copy(positions),
                                neighbors        = np.copy(neighbors),
                                cell             = cell              ,
                                lattice_constant = lattice_constant  ,
                                translation      = translation       ,
                                miller_indices   = miller_indices    ,
                                planes_distances = planes_distances  ,
                                scale_one        = scale_one         ,
                                scale_two        = scale_two         ,
                                n_coord_min      = n_coord_min       ,
                                interact_len     = interact_len      ,
                                miller_symmetry  = True              )

    try:
        particle.get_shape()
        n_atoms = particle.n_atoms
    
    except:
        n_atoms = 0
    
    particles_lock.acquire()
    
    if 0 < n_atoms < n_max:
    
        duplicate = False
    
        n_group = int(np.floor(n_atoms/float(step))*step)
    
        n_coord_dist = particle.n_coord_dist
    
        for index in range(len(particles_dict[n_group])):
    
            particle_old = particles_dict[n_group][index]
    
            if np.array_equal(n_coord_dist, particle_old.n_coord_dist):
    
                duplicate = True
                
                break
    
        if duplicate is False:
            particles_dict[n_group] += [particle]

    number.value -= 1

    particles_lock.release()

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

plane_dist_100 = lattice_constant/2

scale_vect = [(1.0, 1.0)]
#scale_vect = [(1.0, 1.0), (0.9, 1.0), (1.0, 0.9)]

translation_vect = [np.array([0.]*3)]
#translation_vect = [np.array([0.]*3)                               ,
#                    np.array([lattice_constant/2.]+[0.]*2)         ,
#                    np.array([np.sqrt(lattice_constant)/2.]*2+[0.]),
#                    np.array([np.sqrt(lattice_constant)/2.]*3      )]

n_coord_min = 0

################################################################################
# CREATE POPULATION OF PARTICLES
################################################################################

c_max_dict = {}
c_min_dict = {}
plane_dist = {}

d_max_dict = {}
d_min_dict = {}

i_max_dict = {}
j_max_dict = {}
j_min_dict = {}

d_dict = OrderedDict()

layers_dict = OrderedDict()

for hkl in step_dict:

    hkl_scal = [i/max(hkl) for i in hkl]
    denom    = np.sqrt(sum([i**2 for i in hkl_scal]))

    c_max_dict[hkl] = sum(hkl_scal)/denom
    c_min_dict[hkl] = 1./denom

    plane_dist[hkl] = plane_dist_100/denom*step_dict[hkl]

count = 0

n_max_process = 10

number = mp.Value('i', 0)
number.value = 0

processes = []

print('\n  N particles   N processes\n')

for j in range(layers_min_100, layers_max_100):

    d_100 = plane_dist_100*j

    for hkl in step_dict:

        d_max_dict[hkl] = d_100*c_max_dict[hkl]
        d_min_dict[hkl] = d_100*c_min_dict[hkl]

        i_max_dict[hkl] = int(np.around((d_max_dict[hkl]-d_min_dict[hkl]) / 
                              plane_dist[hkl]))

        j_min_dict[hkl] = int(np.around(i_max_dict[hkl]*3./8.))
        j_max_dict[hkl] = int(np.around(i_max_dict[hkl]*4./8.))

        #j_min_dict[hkl] = 0
        #j_max_dict[hkl] = i_max_dict[hkl]

    #j_tot_max = max([j_max_dict[j] for j in j_max_dict])
    #
    #for (a,b,c,d,e) in [(a,b,c,d,e) 
    #                    for a in range(j_max_dict[(1,1,0)])
    #                    for b in range(j_max_dict[(1,1,1)])
    #                    for c in range(j_max_dict[(2,1,0)])
    #                    for d in range(j_max_dict[(2,1,1)])
    #                    for e in range(j_max_dict[(3,1,1)])
    #                    if a+b+c+d+e < j_tot_max]:
    #
    #    layers_dict[(1,0,0)] = 0
    #    layers_dict[(1,1,0)] = j_min_dict[(1,1,0)]+a
    #    layers_dict[(1,1,1)] = j_min_dict[(1,1,1)]+b
    #    layers_dict[(2,1,0)] = j_min_dict[(2,1,0)]+c
    #    layers_dict[(2,1,1)] = j_min_dict[(2,1,1)]+d
    #    layers_dict[(3,1,1)] = j_min_dict[(3,1,1)]+e

    for (a,b,c) in [(a,b,c) 
                    for a in range(j_max_dict[(1,1,0)])
                    for b in range(j_max_dict[(1,1,1)])
                    for c in range(j_max_dict[(2,1,1)])]:
        
        layers_dict[(1,0,0)] = 0
        layers_dict[(1,1,0)] = j_min_dict[(1,1,0)]+a
        layers_dict[(1,1,1)] = j_min_dict[(1,1,1)]+b
        layers_dict[(2,1,1)] = j_min_dict[(2,1,1)]+c

        for hkl in layers_dict:
            d_dict[hkl] = d_max_dict[hkl]-layers_dict[hkl]*plane_dist[hkl]+1e-3

        d_list = [d_dict[xkl] for xkl in d_dict][1:]

        miller_indices = np.array([hkl for hkl in d_dict])
        
        planes_distances = np.array([d_dict[hkl] for hkl in d_dict])
        
        for scale_one, scale_two in scale_vect:
        
            for translation in translation_vect:
        
                if run is True:
        
                    while number.value >= n_max_process:
                        time.sleep(1e-7)
        
                    args = (positions       ,
                            neighbors       ,
                            cell            ,
                            lattice_constant,
                            translation     ,
                            miller_indices  ,
                            planes_distances,
                            scale_one       ,
                            scale_two       ,
                            n_coord_min     ,
                            interact_len    ,
                            particles_dict  ,
                            particles_lock  )
                    
                    process = mp.Process(target = create_particle,
                                         args   = args           )
        
                    process.start()
        
                    processes += [process]
        
                    with number.get_lock():
                        number.value += 1
        
                if count % 100 == 0.:
                    print('{0:13d} {1:13d}'.format(count, number.value))
                
                count += 1
        
                if count % 100 == 0. and count not in (0, 100):
                    
                    for process in processes[:100]:
                        process.join()
                        del process
                    
                    processes = processes[100:]

for process in processes:
    process.join()

################################################################################
# STORE PARTICLES
################################################################################

particles_dict = dict(particles_dict)

if run is True:

    for n_group in range(0, n_max, step):

        group = '{0:04d}_{1:04d}'.format(n_group, n_group+step)

        filepath = os.path.join(dirname, '{0}_{1}.pkl'.format(bulk_type, group))

        with open(filepath, 'wb') as fileobj:
            pickle.dump(particles_dict[n_group], file = fileobj)

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
