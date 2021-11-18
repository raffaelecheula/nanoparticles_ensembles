#!/usr/bin/env python3

################################################################################
# Raffaele Cheula, LCCP, Politecnico di Milano, raffaele.cheula@polimi.it
################################################################################

from __future__ import absolute_import, division, print_function
import os, sys, time, timeit
import pickle
import numpy as np
import multiprocessing as mp
from collections import OrderedDict
from ase.build import bulk
from nanoparticle_units import *
from nanoparticle_utils import icosahedron_grid, e_relax_from_bond_ols
from nanoparticle_cython import calculate_neighbors, IcosahedronShape

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

bulk_type = 'ico'

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
        filename = os.path.join(dirname, '{0}_{1}.pkl'.format(bulk_type, group))

        if not os.path.isfile(filename):
            particles_dict[n_group] = []
        
        else:
            fileobj = open(filename, 'rb')
            particles_dict[n_group] = pickle.load(fileobj)
            fileobj.close()
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
# CREATE PARTICLES
################################################################################

def create_particle(positions       ,
                    neighbors       ,
                    layers          ,
                    lattice_constant,
                    n_coord_min     ,
                    n_coord_min_iter,
                    e_coh_bulk      ,
                    e_relax_list    ,
                    e_twin          ,
                    shear_modulus   ,
                    k_strain        ,
                    particles_dict  ,
                    particles_lock  ):

    particle = IcosahedronShape(positions        = positions       ,
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
        n_atoms = particle.n_atoms
    
    except:
        n_atoms = 0
    
    particles_lock.acquire()
    
    if 0 < n_atoms < n_max:
    
        duplicate = False
    
        n_group = int(np.floor(n_atoms/float(step))*step)
    
        n_coord_dist = particle.n_coord_dist
    
        for index in range(len(particles_dict[n_group])):
    
            old = particles_dict[n_group][index]
    
            if np.array_equal(n_coord_dist, old.n_coord_dist):
    
                duplicate = True
                
                break
    
        if duplicate is False:
            particles_dict[n_group] += [particle]

    number.value -= 1

    particles_lock.release()

################################################################################
# CALCULATION PARAMETERS
################################################################################

size_bulk             = 22  # modify this to produce more particles
size_vector           = range(1, 20)  # modify this to produce more particles
n_coord_min_vect      = [0]
n_coord_min_iter_vect = [0]
disorder_vector       = [False, True]

################################################################################
# CREATE POPULATION OF PARTICLES
################################################################################

interact_len = np.sqrt(2*lattice_constant)*1.2

atoms = icosahedron_grid(element          = element         ,
                         lattice_constant = lattice_constant,
                         size             = size_bulk       )

positions = atoms.get_positions()
cell = np.array(atoms.cell)

neighbors = calculate_neighbors(positions    = positions   ,
                                cell         = cell        ,
                                interact_len = interact_len)

count = 0

n_max_process = 12

number = mp.Value('i', 0)
number.value = 0

processes = []

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

                        while number.value >= n_max_process:
                            time.sleep(1e-7)

                        args = (positions       ,
                                neighbors       ,
                                layers          ,
                                lattice_constant,
                                n_coord_min     ,
                                n_coord_min_iter,
                                e_coh_bulk      ,
                                e_relax_list    ,
                                e_twin          ,
                                shear_modulus   ,
                                k_strain        ,
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

        filename = os.path.join(dirname, '{0}_{1}.pkl'.format(bulk_type, group))

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
