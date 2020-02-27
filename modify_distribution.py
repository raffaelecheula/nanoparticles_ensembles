#!/usr/bin/env python3

################################################################################
# Raffaele Cheula, LCCP, Politecnico di Milano, raffaele.cheula@polimi.it
################################################################################

from __future__ import absolute_import, division, print_function
import os, ase, timeit, sys
import numpy as np
import copy as cp
try: import cPickle as pickle
except: import _pickle as pickle
from math import exp, sqrt
from collections import OrderedDict
from ase import Atoms
import matplotlib.pyplot as plt
from nanoparticle_units import *
from nanoparticle_utils import e_relax_from_bond_ols
from active_sites_shells import (get_surface_shell    , 
                                 get_fcc_active_shell ,
                                 count_active_sites   ,
                                 plot_top_distribution)

################################################################################
# DATA
################################################################################

dirmother = 'particles/'

element = 'Rh'

get_fcc_particles = True
get_dec_particles = True
get_ico_particles = True

n_min_read =   20
n_max_read = 1200

n_min_anal =   20
n_max_anal = 1200

store_particles = True

plot_e_form_n_atoms = False

plot_n_min =    0
plot_n_max = 1200
plot_e_min = 0.30
plot_e_max = 1.30

a_thr = 0.80
b_thr = 0.01

################################################################################
# PARTICLE DATA
################################################################################

e_coh_bulk       = -6.1502 # [eV]
m_bond_ols       = +2.6800 # [-]
e_twin           = +0.0081 # [eV]
shear_modulus    = +155 * (giga*Pa)*(J/eV)*(Ang/mt)**3 # [eV/Ang^3]

k_strain_dec     = 3.78e-4 # [-]
k_strain_ico     = 4.31e-3 # [-]

e_relax_list = e_relax_from_bond_ols(e_coh_bulk = e_coh_bulk,
                                     m_bond_ols = m_bond_ols)

################################################################################
# READ PARTICLES
################################################################################

fcc_particles_dict = {}
dec_particles_dict = {}
ico_particles_dict = {}

step = 10

n_max_per_group = 50

if get_fcc_particles is True:

    bulktyp = 'fcc'
    dirname = dirmother+bulktyp

    for n_group in range(n_min_read, n_max_read, step):

        group = '{0:04d}_{1:04d}'.format(n_group, n_group+step)

        filename = os.path.join(dirname, '{0}_{1}.pkl'.format(bulktyp, group))

        fileobj = open(filename, 'rb')

        particles = pickle.load(fileobj)

        fileobj.close()

        particles = sorted(particles, key = lambda x: x.e_spec)

        particles = particles[:n_max_per_group]

        fcc_particles_dict[n_group] = particles

if get_dec_particles is True:

    bulktyp = 'dec'
    dirname = dirmother+bulktyp

    for n_group in range(n_min_read, n_max_read, step):

        group = '{0:04d}_{1:04d}'.format(n_group, n_group+step)

        filename = os.path.join(dirname, '{0}_{1}.pkl'.format(bulktyp, group))

        fileobj = open(filename, 'rb')

        particles = pickle.load(fileobj)

        fileobj.close()

        particles = sorted(particles, key = lambda x: x.e_spec)

        particles = particles[:n_max_per_group]

        dec_particles_dict[n_group] = particles

if get_ico_particles is True:

    bulktyp = 'ico'
    dirname = dirmother+bulktyp

    for n_group in range(n_min_read, n_max_read, step):

        group = '{0:04d}_{1:04d}'.format(n_group, n_group+step)

        filename = os.path.join(dirname, '{0}_{1}.pkl'.format(bulktyp, group))

        fileobj = open(filename, 'rb')

        particles = pickle.load(fileobj)

        fileobj.close()

        particles = sorted(particles, key = lambda x: x.e_spec)

        particles = particles[:n_max_per_group]

        ico_particles_dict[n_group] = particles

################################################################################
# CREATE PARTICLES
################################################################################

n_iterations = 1

n_atoms_list_fcc = []
e_form_list_fcc  = []
e_spec_list_fcc  = []

e_spec_min_fcc = {}

if get_fcc_particles is True:

    for n_group_zero in range(n_max_anal-step, n_min_anal-step, -step):

        print('fcc {:6d}'.format(n_group_zero))

        for fcc_particle in fcc_particles_dict[n_group_zero]:

            stop = False

            while stop is False:
            
                particle = cp.deepcopy(fcc_particle)
           
                particle.remove_atoms(n_iterations  = n_iterations,
                                      remove_groups = False       )
            
                n_atoms = particle.n_atoms
            
                if n_atoms > n_min_read:
                
                    stop = False
                
                    n_group = int(np.floor(n_atoms/float(step))*step)
                
                    for index in range(len(fcc_particles_dict[n_group])):
                
                        part_old = fcc_particles_dict[n_group][index]
                
                        if np.array_equal(particle.n_coord_dist, 
                                          part_old.n_coord_dist):
                
                            stop = True
                            break
                
                    if stop is False:
                        
                        particle.e_coh_bulk    = e_coh_bulk
                        particle.e_relax_list  = e_relax_list

                        particle.get_energy()
                        
                        e_thr = -e_coh_bulk*a_thr*n_atoms**(-1./3.)-b_thr
                        
                        if particle.e_spec > e_thr:
                        
                            stop = True
                        
                        else:
                        
                            fcc_particles_dict[n_group] += [particle]
           
                else:

                    stop = True

n_atoms_list_dec = []
e_form_list_dec  = []
e_spec_list_dec  = []

e_spec_min_dec = {}

if get_dec_particles is True:

    for n_group_zero in range(n_max_anal-step, n_min_anal-step, -step):

        print('dec {:6d}'.format(n_group_zero))

        for dec_particle in dec_particles_dict[n_group_zero]:

            stop = False

            while stop is False:
            
                particle = cp.deepcopy(dec_particle)
           
                particle.remove_atoms(n_iterations  = n_iterations,
                                      remove_groups = False       )
            
                n_atoms = particle.n_atoms
            
                if n_atoms > n_min_read:
                
                    stop = False
                
                    n_group = int(np.floor(n_atoms/float(step))*step)
                
                    for index in range(len(dec_particles_dict[n_group])):
                
                        part_old = dec_particles_dict[n_group][index]
                
                        if np.array_equal(particle.n_coord_dist, 
                                          part_old.n_coord_dist):
                
                            stop = True
                            break
                
                    if stop is False:
                        
                        particle.e_coh_bulk    = e_coh_bulk
                        particle.e_relax_list  = e_relax_list
                        particle.e_twin        = e_twin
                        particle.shear_modulus = shear_modulus
                        particle.k_strain      = k_strain_dec

                        particle.get_energy()
                        
                        e_thr = -e_coh_bulk*a_thr*n_atoms**(-1./3.)-b_thr
                        
                        if particle.e_spec > e_thr:
                        
                            stop = True
                        
                        else:
                        
                            dec_particles_dict[n_group] += [particle]
           
                else:

                    stop = True


n_atoms_list_ico = []
e_form_list_ico  = []
e_spec_list_ico  = []

e_spec_min_ico = {}

if get_ico_particles is True:

    for n_group_zero in range(n_max_anal-step, n_min_anal-step, -step):

        print('ico {:6d}'.format(n_group_zero))

        for ico_particle in ico_particles_dict[n_group_zero]:

            stop = False

            while stop is False:
            
                particle = cp.deepcopy(ico_particle)
           
                particle.remove_atoms(n_iterations  = n_iterations,
                                      remove_groups = False       )
            
                n_atoms = particle.n_atoms
            
                if n_atoms > n_min_read:
                
                    stop = False
                
                    n_group = int(np.floor(n_atoms/float(step))*step)
                
                    for index in range(len(ico_particles_dict[n_group])):
                
                        part_old = ico_particles_dict[n_group][index]
                
                        if np.array_equal(particle.n_coord_dist, 
                                          part_old.n_coord_dist):
                
                            stop = True
                            break
                
                    if stop is False:
                        
                        particle.e_coh_bulk    = e_coh_bulk
                        particle.e_relax_list  = e_relax_list
                        particle.e_twin        = e_twin
                        particle.shear_modulus = shear_modulus
                        particle.k_strain      = k_strain_ico

                        particle.get_energy()
                        
                        e_thr = -e_coh_bulk*a_thr*n_atoms**(-1./3.)-b_thr
                        
                        if particle.e_spec > e_thr:
                        
                            stop = True
                        
                        else:
                        
                            ico_particles_dict[n_group] += [particle]
           
                else:

                    stop = True

################################################################################
# STORE PARTICLES
################################################################################

dirmother = 'particles/'

if store_particles is True:

    if get_fcc_particles is True:
    
        bulktyp = 'fcc'
        dirname = dirmother+bulktyp
    
        for n_group in range(n_min_read, n_max_read, step):
    
            group = '{0:04d}_{1:04d}'.format(n_group, n_group+step)
    
            filename = os.path.join(dirname, '{0}_{1}.pkl'.format(bulktyp, 
                                                                  group  ))
    
            fileobj = open(filename, 'wb')
            pickle.dump(fcc_particles_dict[n_group], file = fileobj)
            fileobj.close()
    
    if get_dec_particles is True:
    
        bulktyp = 'dec'
        dirname = dirmother+bulktyp
    
        for n_group in range(n_min_read, n_max_read, step):
    
            group = '{0:04d}_{1:04d}'.format(n_group, n_group+step)
    
            filename = os.path.join(dirname, '{0}_{1}.pkl'.format(bulktyp, 
                                                                  group  ))
    
            fileobj = open(filename, 'wb')
            pickle.dump(dec_particles_dict[n_group], file = fileobj)
            fileobj.close()
    
    if get_ico_particles is True:
    
        bulktyp = 'ico'
        dirname = dirmother+bulktyp
    
        for n_group in range(n_min_read, n_max_read, step):
    
            group = '{0:04d}_{1:04d}'.format(n_group, n_group+step)
    
            filename = os.path.join(dirname, '{0}_{1}.pkl'.format(bulktyp, 
                                                                  group  ))
    
            fileobj = open(filename, 'wb')
            pickle.dump(ico_particles_dict[n_group], file = fileobj)
            fileobj.close()

################################################################################
# FORMATION ENERGY vs N ATOMS PLOT
################################################################################

NC_12 = 0
markersize = 4
width = 0.8

tick_size = 14
label_size = 16

if plot_e_form_n_atoms is True:

    n_atoms_list_fcc = []
    e_form_list_fcc  = []
    e_spec_list_fcc  = []
    
    if get_fcc_particles is True:
    
        for n_group_zero in range(n_min_read, n_max_read, step):
    
            for particle in fcc_particles_dict[n_group_zero]:
    
                n_atoms_list_fcc += [particle.n_atoms]
                e_form_list_fcc  += [particle.e_form]
                e_spec_list_fcc  += [particle.e_spec]
    
    n_atoms_list_dec = []
    e_form_list_dec  = []
    e_spec_list_dec  = []
    
    if get_dec_particles is True:
    
        for n_group_zero in range(n_min_read, n_max_read, step):
    
            for particle in dec_particles_dict[n_group_zero]:
    
                n_atoms_list_dec += [particle.n_atoms]
                e_form_list_dec  += [particle.e_form]
                e_spec_list_dec  += [particle.e_spec]
    
    n_atoms_list_ico = []
    e_form_list_ico  = []
    e_spec_list_ico  = []
    
    if get_ico_particles is True:
    
        for n_group_zero in range(n_min_read, n_max_read, step):
    
            for particle in ico_particles_dict[n_group_zero]:
    
                n_atoms_list_ico += [particle.n_atoms]
                e_form_list_ico  += [particle.e_form]
                e_spec_list_ico  += [particle.e_spec]

    fig = plt.figure(1)
    fig.set_size_inches(16, 10)
    
    if get_fcc_particles is True:
    
        plt.plot(n_atoms_list_fcc, e_spec_list_fcc, marker = 'o', alpha = 0.6,
                 linestyle = ' ', markersize = markersize, 
                 color = 'darkorange')
    
        plt.plot(e_spec_min_fcc.keys(), e_spec_min_fcc.values(), marker = 'o',
                 alpha = 0.9, linestyle = ' ', markersize = markersize,
                 color = 'darkorange')
    
    if get_dec_particles is True:
    
        plt.plot(n_atoms_list_dec, e_spec_list_dec, marker = 'o', alpha = 0.2,
                 linestyle = ' ', markersize = markersize,
                 color = 'forestgreen')
    
        plt.plot(e_spec_min_dec.keys(), e_spec_min_dec.values(), marker = 'o',
                 alpha = 0.6, linestyle = ' ', markersize = markersize,
                 color = 'forestgreen')
    
    if get_ico_particles is True:
    
        plt.plot(n_atoms_list_ico, e_spec_list_ico, marker = 'o', alpha = 0.2,
                 linestyle = ' ', markersize = markersize,
                 color = 'darkviolet')
    
        plt.plot(e_spec_min_ico.keys(), e_spec_min_ico.values(), marker = 'o',
                 alpha = 0.6, linestyle = ' ', markersize = markersize,
                 color = 'darkviolet')
    
    plt.axis([plot_n_min, plot_n_max, plot_e_min, plot_e_max])
    
    plt.yticks(fontsize = tick_size)
    plt.xticks(fontsize = tick_size)
    
    plt.ylabel('formation energy [eV/atom]', fontsize = label_size)
    plt.xlabel('number of atoms', fontsize = label_size)

    plt.show()

################################################################################
# END
################################################################################
