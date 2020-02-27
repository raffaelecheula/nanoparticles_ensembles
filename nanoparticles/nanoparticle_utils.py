################################################################################
# Raffaele Cheula, LCCP, Politecnico di Milano, raffaele.cheula@polimi.it
################################################################################

from __future__ import absolute_import, division, print_function
import copy as cp
import numpy as np
from math import pi, atan, sqrt
from ase import Atoms
from ase.cluster.decahedron import Decahedron
from ase.cluster.icosahedron import Icosahedron

################################################################################
# GET INTERACT LEN
################################################################################

def get_interact_len(bulk, bulk_type):

    lattice_constant = bulk.cell[0][0]

    if bulk_type is 'fcc' or 'fcc_reduced':
        interact_len = sqrt(lattice_constant * 2)

    elif bulk_type is 'bcc':
        interact_len = ((lattice_constant / 2.)**2. * 3.)**0.5

    elif bulk_type is 'simple_cubic':
        interact_len = lattice_constant

    return interact_len

################################################################################
# BOUNDARY ATOMS
################################################################################

def boundary_atoms(atoms, base_boundary = False, outer_boundary = True, 
                   epsi = 1e-3):

    atoms_plus = cp.deepcopy(atoms)

    for a in atoms:
        if abs(a.position[0]) < epsi and abs(a.position[1]) < epsi:
            a_plus = cp.deepcopy(a)
            a_plus.position[:2] += sum(atoms.cell[:2])[:2]
            atoms_plus += a_plus

        if (abs(a.position[0] - a.position[1] * atoms.cell[1][0] /
            atoms.cell[1][1]) < epsi):
            a_plus = cp.deepcopy(a)
            a_plus.position[:2] += atoms.cell[0][:2]
            atoms_plus += a_plus

        if (abs(a.position[1] - a.position[0] * atoms.cell[0][1] /
            atoms.cell[0][0]) < epsi):
            a_plus = cp.deepcopy(a)
            a_plus.position[:2] += atoms.cell[1][:2]
            atoms_plus += a_plus

    if base_boundary is True:
        for a in atoms:
            if abs(a.position[2]) < epsi:
                a_plus = cp.deepcopy(a)
                a_plus.position[2] += atoms.cell[2][2]
                atoms_plus += a_plus

    if outer_boundary is True:
        for a in atoms:
            if (abs(a.position[0] - atoms.cell[0][0] - atoms.cell[1][0]) <
                epsi and abs(a.position[1] - atoms.cell[0][1] -
                atoms.cell[1][1]) < epsi):
                a_plus = cp.deepcopy(a)
                a_plus.position[:2] -= sum(atoms.cell[:2])[:2]
                atoms_plus += a_plus

            if (abs(a.position[0] - atoms.cell[0][0] - a.position[1] *
                atoms.cell[1][0] / atoms.cell[1][1]) < epsi):
                a_plus = cp.deepcopy(a)
                a_plus.position[:2] -= atoms.cell[0][:2]
                atoms_plus += a_plus

            if (abs(a.position[1] - atoms.cell[1][1] - a.position[0] *
                atoms.cell[0][1] / atoms.cell[0][0]) < epsi):
                a_plus = cp.deepcopy(a)
                a_plus.position[:2] -= atoms.cell[1][:2]
                atoms_plus += a_plus

    return atoms_plus

################################################################################
# NEIGHBOR ATOMS
################################################################################

def get_neighbor_atoms(bulk, interact_len, epsi = 1e-4):

    atoms = bulk[:]
    cell_vectors = atoms.cell

    atoms *= (3, 3, 3)
    atoms = boundary_atoms(atoms, base_boundary = True)
    atoms.translate(-sum(cell_vectors))

    del atoms [[ a.index for a in atoms \
        if np.linalg.norm(a.position) > interact_len + epsi ]]

    atoms.set_cell(cell_vectors)

    del atoms [[ a.index for a in atoms \
        if np.array_equal(a.position, [0., 0., 0.]) ]]

    return atoms

################################################################################
# ROTATE NANOPARTICLE
################################################################################

def rotate_nanoparticle(atoms, contact_index):
    
    vector_y = np.cross((0, 0, 1), contact_index)
    vector_z = np.cross(contact_index, vector_y)

    if np.linalg.norm(contact_index[:2]):
        rotation_angle = (atan(contact_index[2] /
                          np.linalg.norm(contact_index[:2])))
        atoms.rotate(90 + rotation_angle*180/pi, vector_y)

    if contact_index[0]:
        rotation_angle = -atan(contact_index[1]/contact_index[0])
        atoms.rotate(rotation_angle*180/pi, 'z')

    return atoms

################################################################################
# CLUSTER ADD ADSORBATES
################################################################################

def cluster_add_adsorbates(atoms, adsorbate, sites_list, distance,
                           first_element = 0):

    center = np.dot(atoms.cell/2., np.ones(3))

    for site_num in sites_list:

        site_pos = np.zeros(3)
        if isinstance(site_num, int):
            if first_element == 1:
                site_num -= 1
            site_pos += atoms[site_num].position
        else:
            for num in site_num:
                if first_element == 1:
                    num -= 1
                site_pos += atoms[num].position/len(site_num)
        direction = site_pos-center
        direction /= np.linalg.norm(direction)
        ads_center = site_pos + direction*distance
    
        ads = adsorbate.copy()
    
        vector_z = direction
        vector_x = np.cross((0, 0, 1), direction)
        vector_y = np.cross(direction, vector_x)
    
        matrix = np.array([vector_x, vector_y, vector_z]).T
        
        positions = []
        for a in ads:
            positions += [ads_center+np.dot(matrix, a.position)]
        ads.positions = positions
    
        atoms += ads

    return atoms

################################################################################
# DECAHEDRON GRID
################################################################################

def decahedron_grid(element, lattice_constant, size, heigth):

    atoms = Decahedron(symbol = element, p = size, q = heigth, r = 0,
                       latticeconstant = lattice_constant)

    atoms.rotate(360./10., 'z')

    atoms.set_pbc(True)

    atoms.center(vacuum = 5.)

    return atoms

################################################################################
# OCTAHEDRON GRID
################################################################################

def icosahedron_grid(element, lattice_constant, size):

    atoms = Icosahedron(symbol = element, noshells = size,
                        latticeconstant = lattice_constant)

    atoms.set_pbc(True)

    atoms.center(vacuum = 5.)

    return atoms

################################################################################
# E RELAX FROM BOND OLS
################################################################################

def e_relax_from_bond_ols(e_coh_bulk, m_bond_ols):

    Eb_bond_ols = -e_coh_bulk/12

    e_relax_list = np.zeros(13)

    for n_coord in range(3, 13):

        e_relax_list[n_coord] = (Eb_bond_ols*(1-(2/(1+np.exp((12-n_coord) / 
                                 (8*n_coord))))**-m_bond_ols))

    e_relax_list[2] = e_relax_list[3]+(e_relax_list[3]-e_relax_list[4])
    e_relax_list[1] = e_relax_list[3]+(e_relax_list[3]-e_relax_list[4])*2

    return e_relax_list

################################################################################
# END
################################################################################
