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

# PYTHON IMPORT

from __future__ import absolute_import, division, print_function
import numpy as np
import spglib as sp
from scipy import optimize
from scipy.spatial import ConvexHull

# CYTHON IMPORT

cimport cython
cimport numpy as np
from libc.math cimport exp, sqrt, pow

################################################################################
# CONSTANTS
################################################################################

cdef:
    double     kB_eV = 8.61733e-5 # [eV/K]

################################################################################
# MATRICES
################################################################################

cdef:
    np.ndarray matrix_zero = np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                       [[1, 0, 0], [0, 0, 1], [0, 1, 0]],
                                       [[0, 1, 0], [1, 0, 0], [0, 0, 1]],
                                       [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
                                       [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
                                       [[0, 0, 1], [0, 1, 0], [1, 0, 0]]])

    np.ndarray matrix_one  = np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                       [[1, 0, 0], [0, 0, 1], [0, 1, 0]],
                                       [[0, 1, 0], [1, 0, 0], [0, 0, 1]],
                                       [[0, 0, 1], [1, 0, 0], [0, 1, 0]]])

    np.ndarray matrix_two  = np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                       [[1, 0, 0], [0, 0, 1], [0, 1, 0]]])

cdef:
    float      angle        = -np.pi*2./5.
    np.ndarray deca_rot_mat = np.array([[+np.cos(angle), -np.sin(angle)],
                                       [+np.sin(angle), +np.cos(angle)]])
    np.ndarray deca_hkl     = np.array([[1.,       np.tan(np.pi/5.), 0.],
                                        [1., 3.*sqrt(5.-2*sqrt(5.)), 0.]])

cdef:
    int        i
    float      norm
    float      pt            = 0.5+sqrt(5.)/2.
    np.ndarray octa_vert     = np.array([[+pt, +0., +1.],
                                         [+pt, +0., -1.],
                                         [-pt, +0., +1.],
                                         [-pt, +0., -1.],
                                         [+1., +pt, +0.],
                                         [-1., +pt, +0.],
                                         [+1., -pt, +0.],
                                         [-1., -pt, +0.],
                                         [+0., +1., +pt],
                                         [+0., -1., +pt],
                                         [+0., +1., -pt],
                                         [+0., -1., -pt]])
    np.ndarray octa_triplets = np.array([[ 8, 9, 0], [ 8, 0, 4], [ 8, 4, 5],
                                         [ 8, 5, 2], [ 8, 2, 9], [ 2, 9, 7],
                                         [ 2, 7, 3], [ 2, 3, 5], [ 3, 5,10],
                                         [ 5,10, 4], [10, 4, 1], [10, 1,11],
                                         [10,11, 3], [11, 3, 7], [11, 7, 6],
                                         [ 7, 6, 9], [ 6, 9, 0], [ 6, 0, 1],
                                         [ 0, 1, 4], [6, 1, 11]])
    np.ndarray octa_norm     = np.zeros((len(octa_vert), 3), dtype = float)

for i in range(len(octa_vert)):
    norm = np.linalg.norm(octa_vert[i])
    octa_norm[i] = octa_vert[i]/norm

################################################################################
# PARTICLE SHAPE
################################################################################

class ParticleShape:

    # ----------------------------------------------------------------------
    #  TO ASE ATOMS
    # ----------------------------------------------------------------------

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def to_ase_atoms(self, 
                     str        symbol):

        from ase import Atoms
    
        cdef:
            str        symbols = symbol+str(self.n_atoms)
    
        atoms = Atoms(symbols   = symbols       ,
                      positions = self.positions)
    
        return atoms

    # ----------------------------------------------------------------------
    #  REMOVE ATOMS
    # ----------------------------------------------------------------------

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def remove_atoms(self, 
                     int        n_iterations  ,
                     int        remove_groups = False):

        positions = particle_remove_atoms(self,
                                          n_iterations  = n_iterations ,
                                          remove_groups = remove_groups)

        return positions

    # ----------------------------------------------------------------------
    #  GET MULTIPLICITY
    # ----------------------------------------------------------------------

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def get_multiplicity(self,
                         int        multip_bulk):
    
        multiplicity = particle_multiplicity(self, multip_bulk = multip_bulk)
    
        return multiplicity

    # ----------------------------------------------------------------------
    #  GET ENERGY
    # ----------------------------------------------------------------------

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def get_energy(self):

        cdef:
            int        n_atoms      = self.n_atoms
            np.ndarray n_coord_dist = self.n_coord_dist
            double     e_coh_bulk   = self.e_coh_bulk
            np.ndarray e_relax_list = self.e_relax_list

        cdef:
            int        i
            double     e_form_clean, v_atom, e_strain
            double     e_coh = 0.

        for i in range(13):
            e_coh += n_coord_dist[i]*e_coh_bulk*np.sqrt(i)/np.sqrt(12)
            e_coh += n_coord_dist[i]*e_relax_list[i]

        if self.particle_type in ('decahedron', 'icosahedron'):

            v_atom = 1./4.*self.lattice_constant**3
            
            e_strain = self.k_strain*self.shear_modulus*v_atom*self.n_atoms
            
            e_coh += e_strain
            
            e_coh += self.n_twin*self.e_twin

        e_form_clean = e_coh-e_coh_bulk*n_atoms

        self.e_form_clean = e_form_clean
        self.e_spec_clean = e_form_clean/n_atoms
        self.e_form       = e_form_clean
        self.e_spec       = e_form_clean/n_atoms

        return e_form_clean

    # ----------------------------------------------------------------------
    #  GET ENERGY WITH ADS
    # ----------------------------------------------------------------------

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def get_energy_with_ads(self,
                            float      bond_len     ,
                            float      y_zero_e_bind,
                            float      m_ang_e_bind ,
                            float      alpha_cov    ,
                            float      beta_cov     ,
                            float      temperature  ,
                            float      delta_mu_ads ,
                            list       f_e_bind_corr,
                            str        entropy_model,
                            int        averag_e_bind,
                            str        e_form_denom ):

        cdef:
            float      area_surf
            float      e_form_ads

        area_surf = get_surface_area(self,
                                     bond_len = bond_len)

        e_form_ads = get_e_form_with_ads(self,
                                         y_zero_e_bind = y_zero_e_bind,
                                         m_ang_e_bind  = m_ang_e_bind ,
                                         alpha_cov     = alpha_cov    ,
                                         beta_cov      = beta_cov     ,
                                         area_surf     = area_surf    ,
                                         temperature   = temperature  ,
                                         delta_mu_ads  = delta_mu_ads ,
                                         f_e_bind_corr = f_e_bind_corr,
                                         entropy_model = entropy_model,
                                         averag_e_bind = averag_e_bind,
                                         e_form_denom  = e_form_denom )

        return e_form_ads

    # ----------------------------------------------------------------------
    #  GET ACTIVE SITES
    # ----------------------------------------------------------------------

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def get_active_sites(self):

        from nanoparticle_active_sites import (get_surface_shell     ,
                                               get_active_sites_shell)

        surface = get_surface_shell(positions    = self.positions,
                                    neighbors    = self.neighbors, 
                                    indices      = self.indices  , 
                                    n_coord      = self.n_coord  ,
                                    supp_contact = None          ,
                                    n_coord_max  = 12            )
        
        active_sites = get_active_sites_shell(surface          = surface,
                                              specify_n_coord  = True   ,
                                              specify_supp_int = False  ,
                                              specify_facets   = False  ,
                                              check_duplicates = False  ,
                                              multiple_facets  = False  )

        self.active_sites = active_sites

        return active_sites

    # ----------------------------------------------------------------------
    #  GET ACTIVE SITES DICT
    # ----------------------------------------------------------------------

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def get_active_sites_dict(self):

        from nanoparticle_active_sites import count_active_sites

        active_sites_dict = count_active_sites(
                                            active_sites = self.active_sites,
                                            with_tags    = True             )

        self.active_sites_dict = active_sites_dict

        return active_sites_dict

################################################################################
# FCC PARTICLE SHAPE
################################################################################

class FccParticleShape(ParticleShape):

    def __init__(self,
                 np.ndarray positions       ,
                 np.ndarray neighbors       ,
                 np.ndarray cell            ,
                 np.ndarray translation     ,
                 np.ndarray miller_indices  ,
                 np.ndarray planes_distances,
                 np.ndarray e_relax_list    ,
                 float      scale_one       , 
                 float      scale_two       , 
                 int        n_coord_min     ,
                 float      interact_len    , 
                 float      e_coh_bulk      ,
                 int        miller_symmetry ):

        self.positions        = positions
        self.neighbors        = neighbors
        self.cell             = cell
        self.translation      = translation
        self.miller_indices   = miller_indices
        self.planes_distances = planes_distances
        self.e_relax_list     = e_relax_list
        self.scale_one        = scale_one
        self.scale_two        = scale_two
        self.n_coord_min      = n_coord_min
        self.interact_len     = interact_len
        self.e_coh_bulk       = e_coh_bulk
        self.particle_type    = 'fcc particle'
        self.miller_symmetry  = miller_symmetry

    # ----------------------------------------------------------------------
    #  GET SHAPE
    # ----------------------------------------------------------------------

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def get_shape(self):

        cdef:
            np.ndarray positions        = self.positions
            np.ndarray neighbors        = self.neighbors
            np.ndarray cell             = self.cell
            np.ndarray translation      = self.translation
            np.ndarray miller_indices   = self.miller_indices
            np.ndarray planes_distances = self.planes_distances
            float      scale_one        = self.scale_one
            float      scale_two        = self.scale_two
            int        n_coord_min      = self.n_coord_min
            float      interact_len     = self.interact_len
            int        miller_symmetry  = self.miller_symmetry

        cdef:
            int        i, j, p, q
            list       del_indices
            int        max_coord     = 12
            np.ndarray abs_pos       = np.empty((len(positions), 3))
            np.ndarray miller_vector = np.empty(3)
            np.ndarray index_vector  = np.empty(3)
            np.ndarray abs_pos_p     = np.empty(3)
            float      dot_pos_index
            float      planes_dist_i
            np.ndarray cell_diag     = sum(cell)
            np.ndarray cell_trans    = cell_diag/2.+translation
            np.ndarray indices       = np.arange(len(positions))

        for p in range(len(positions)):
            positions[p] -= cell_trans

        if miller_symmetry is False:

            for i in range(len(miller_indices)):
    
                miller_vector = (miller_indices[i] / 
                                 np.linalg.norm(miller_indices[i]))
    
                del_indices = []
    
                for p in range(len(positions)):
                
                    dot_pos_index = np.dot(positions[p], miller_vector)
                
                    if dot_pos_index > planes_distances[i]:
                        del_indices += [p]
                
                positions = np.delete(positions, del_indices, axis = 0)
                abs_pos   = np.delete(abs_pos, del_indices, axis = 0)
                neighbors = np.delete(neighbors, del_indices, axis = 0)
                indices   = np.delete(indices, del_indices, axis = 0)

        else:

            for p in range(len(positions)):
                abs_pos[p] = abs(positions[p])
    
            for i in range(len(miller_indices)):
    
                miller_vector = (miller_indices[i] / 
                                 np.linalg.norm(miller_indices[i]))
    
                del_indices = []
    
                planes_dist_i = planes_distances[i]
    
                for j in range(6):
    
                    index_vector = np.dot(matrix_zero[j], miller_vector)
    
                    del_indices = []
    
                    for p in range(len(positions)):
    
                        abs_pos_p = abs_pos[p]
                        dot_pos_index = np.dot(abs_pos_p, index_vector)
    
                        if dot_pos_index > planes_dist_i:
                            del_indices += [p]
    
                    positions = np.delete(positions, del_indices, axis = 0)
                    abs_pos   = np.delete(abs_pos, del_indices, axis = 0)
                    neighbors = np.delete(neighbors, del_indices, axis = 0)
                    indices   = np.delete(indices, del_indices, axis = 0)
    
                if scale_one < 1.:
                
                    for j in range(4):
                
                        index_vector = np.dot(matrix_one[j], miller_vector)
                
                        del_indices = []
                
                        for p in range(len(positions)):
                
                            abs_pos_p = abs_pos[p]
                            dot_pos_index = np.dot(abs_pos_p, index_vector)
                
                            if dot_pos_index > planes_dist_i*scale_one:
                                del_indices += [p]
                
                        positions = np.delete(positions, del_indices, axis = 0)
                        abs_pos   = np.delete(abs_pos, del_indices, axis = 0)
                        neighbors = np.delete(neighbors, del_indices, axis = 0)
                        indices   = np.delete(indices, del_indices, axis = 0)
                
                if scale_two < 1.:
                
                    for j in range(2):
                
                        index_vector = np.dot(matrix_two[j], miller_vector)
                    
                        del_indices = []
                    
                        for p in range(len(positions)):
                    
                            abs_pos_p = abs_pos[p]
                            dot_pos_index = np.dot(abs_pos_p, index_vector)
                    
                            if dot_pos_index > planes_dist_i*scale_two:
                                del_indices += [p]
                    
                        positions = np.delete(positions, del_indices, axis = 0)
                        abs_pos   = np.delete(abs_pos, del_indices, axis = 0)
                        neighbors = np.delete(neighbors, del_indices, axis = 0)
                        indices   = np.delete(indices, del_indices, axis = 0)

        for p in range(len(positions)):
            positions[p] += cell_trans

        cdef:
            float      dispersion
            int        n_atoms      = len(positions)
            np.ndarray n_coord      = np.zeros(n_atoms, dtype = int)
            np.ndarray n_coord_dist = np.zeros(13, dtype = int)
            int        deleted      = 1

        while deleted == 1:

            for p in range(n_atoms):
                for q in np.nditer(neighbors[p]):
                    if q in indices:
                        n_coord[p] += 1

            del_indices = []

            if n_coord_min > 0:

                for i in range(len(n_coord)):
                    if n_coord[i] < n_coord_min:
                        del_indices += [i]

                positions = np.delete(positions, del_indices, axis = 0)
                neighbors = np.delete(neighbors, del_indices, axis = 0)
                indices   = np.delete(indices, del_indices, axis = 0)

                n_atoms = len(positions)

            if len(del_indices) == 0:
                deleted = 0
            else:
                n_coord = np.zeros(n_atoms, dtype = int)

        bincount = np.bincount(n_coord)

        for i in range(len(bincount)):
            n_coord_dist[i] = bincount[i]

        dispersion = np.sum(n_coord_dist[:10])/n_atoms

        self.n_atoms      = n_atoms
        self.positions    = positions
        self.neighbors    = neighbors
        self.indices      = indices
        self.n_coord      = n_coord
        self.n_coord_dist = n_coord_dist
        self.dispersion   = dispersion

        return positions

################################################################################
# DECAHEDRON SHAPE
################################################################################

class DecahedronShape(ParticleShape):

    def __init__(self,
                 np.ndarray positions       ,
                 np.ndarray neighbors       ,
                 np.ndarray layers_min      ,
                 np.ndarray layers_max      ,
                 np.ndarray planes_miller   ,
                 int        layers_hole     ,
                 np.ndarray e_relax_list    ,
                 float      lattice_constant,
                 int        n_coord_min     ,
                 float      e_coh_bulk      ,
                 float      e_twin          ,
                 float      shear_modulus   ,
                 float      k_strain        ):

        self.positions        = positions
        self.neighbors        = neighbors
        self.layers_min       = layers_min
        self.layers_max       = layers_max
        self.planes_miller    = planes_miller
        self.layers_hole      = layers_hole
        self.lattice_constant = lattice_constant
        self.n_coord_min      = n_coord_min
        self.e_coh_bulk       = e_coh_bulk
        self.e_relax_list     = e_relax_list
        self.e_twin           = e_twin
        self.shear_modulus    = shear_modulus
        self.k_strain         = k_strain
        self.particle_type    = 'decahedron'

        for i in range(5):
            if self.layers_min[i] < 1:
                self.layers_min[i] = 1
            if self.layers_max[i] < 1:
                self.layers_max[i] = 1
            if self.layers_max[i] < self.layers_min[i]:
                self.layers_min[i] = self.layers_max[i]

        cdef:
            float      heigth_min = 1e9
            float      heigth_max = 0.

        for i in range(len(positions)):

            if positions[i][2] < heigth_min:
                heigth_min = positions[i][2]

            if positions[i][2] > heigth_max:
                heigth_max = positions[i][2]

        self.layers_z = int(np.around((heigth_max-heigth_min) / 
                            (lattice_constant*np.sqrt(2.)/2.))+1)

    # ----------------------------------------------------------------------
    #  GET SHAPE
    # ----------------------------------------------------------------------

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def get_shape(self):

        cdef:
            np.ndarray positions        = self.positions
            np.ndarray neighbors        = self.neighbors
            np.ndarray layers_min       = self.layers_min
            np.ndarray layers_max       = self.layers_max
            np.ndarray planes_miller    = self.planes_miller
            int        layers_hole      = self.layers_hole
            float      lattice_constant = self.lattice_constant
            int        n_coord_min      = self.n_coord_min
            int        layers_z         = self.layers_z

        cdef:
            int i, j, p
            float      pos_x, pos_y, pos_z, pos_diag_r, pos_diag_l
            list       del_indices
            float      plane_dist_t, plane_dist_r, plane_dist_l, max_z, max_z_p
            np.ndarray index_vector_r, index_vector_l
            np.ndarray center     = np.zeros(3, dtype = float)
            float      diag_r     = np.tan(np.pi*3./10.)-1e-3
            float      diag_l     = np.tan(np.pi*3./10.)+1e-3
            np.ndarray indices    = np.arange(len(positions))
            float      len_100    = 0.495422*lattice_constant
            np.ndarray len_hkl    = np.array([0.58241*lattice_constant,
                                              0.60050*lattice_constant])

        for p in range(len(positions)):
            center += positions[p]

        for i in range(len(center)):
            center[i] /= len(positions)

        for p in range(len(positions)):
            positions[p] -= center

        for i in range(len(deca_hkl)):
            deca_hkl[i] = deca_hkl[i]/np.linalg.norm(deca_hkl[i])

        max_z = (layers_z/2.-layers_hole)*lattice_constant*np.sqrt(2.)/2.+1e-3

        for i in range(5):

            j = i-1
            if j < 0:
                j = 4

            del_indices = []

            plane_dist_t = len_100*layers_max[i]

            index_vector_r = deca_hkl[planes_miller[i][0]]
            plane_dist_r = len_hkl[planes_miller[i][0]]*layers_min[i]

            index_vector_l = np.copy(deca_hkl[planes_miller[i][1]])
            index_vector_l[0] = -index_vector_l[0]
            plane_dist_l = len_hkl[planes_miller[i][1]]*layers_min[j]

            for p in range(len(positions)):

                pos_x = positions[p][0]
                pos_y = positions[p][1]
                pos_z = positions[p][2]

                if pos_y > plane_dist_t:
                    del_indices += [p]

                pos_diag_r = pos_x*diag_r
                dot_pos_ind = np.dot(positions[p], index_vector_r)

                if pos_y > pos_diag_r and dot_pos_ind > plane_dist_r:
                    del_indices += [p]

                pos_diag_l = -pos_x*diag_l
                dot_pos_ind = np.dot(positions[p], index_vector_l)

                if pos_y > pos_diag_l and dot_pos_ind > plane_dist_l:
                    del_indices += [p]

                max_z_p = max_z+pos_y*np.sqrt(2)/2.

                if (((pos_x >= 0. and pos_y > pos_diag_r)
                or (pos_x < 0. and pos_y > pos_diag_l))
                and (pos_z > max_z_p or pos_z < -max_z_p)):
                    del_indices += [p]

            positions = np.delete(positions, del_indices, axis = 0)
            neighbors = np.delete(neighbors, del_indices, axis = 0)
            indices   = np.delete(indices, del_indices, axis = 0)

            for p in range(len(positions)):
                positions[p][:2] = np.dot(positions[p][:2], deca_rot_mat)

        cdef:
            float      dispersion
            int        n_atoms      = len(positions)
            np.ndarray n_coord      = np.zeros(n_atoms, dtype = int)
            np.ndarray n_coord_dist = np.zeros(13, dtype = int)
            int        deleted      = 1

        while deleted == 1:

            for p in range(n_atoms):
                for q in np.nditer(neighbors[p]):
                    if q in indices:
                        n_coord[p] += 1

            del_indices = []

            if n_coord_min > 0:

                for i in range(len(n_coord)):
                    if n_coord[i] < n_coord_min:
                        del_indices += [i]

                positions = np.delete(positions, del_indices, axis = 0)
                neighbors = np.delete(neighbors, del_indices, axis = 0)
                indices   = np.delete(indices, del_indices, axis = 0)

                n_atoms = len(positions)

            if len(del_indices) == 0:
                deleted = 0
            else:
                n_coord = np.zeros(n_atoms, dtype = int)

        for p in range(len(positions)):
            positions[p] += center

        bincount = np.bincount(n_coord)

        for i in range(len(bincount)):
            n_coord_dist[i] = bincount[i]

        dispersion = np.sum(n_coord_dist[:10])/n_atoms

        self.n_atoms      = n_atoms
        self.positions    = positions
        self.neighbors    = neighbors
        self.indices      = indices
        self.n_coord      = n_coord
        self.n_coord_dist = n_coord_dist
        self.dispersion   = dispersion

        self.get_n_twin()

        return positions

    # ----------------------------------------------------------------------
    #  GET N TWIN
    # ----------------------------------------------------------------------

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def get_n_twin(self):

        cdef:
            np.ndarray layers_min  = self.layers_min
            int        layers_hole = self.layers_hole
            int        layers_z    = self.layers_z

        cdef:
            int        i, j
            int        n_twin = 0

        for i in range(5):
            for j in range(layers_z-layers_min[i], layers_z+1):
                n_twin += j
            for j in range(layers_hole+1):
                n_twin -= 2*j

        self.n_twin = n_twin

        return n_twin

################################################################################
# ICOSAHEDRON SHAPE
################################################################################

class IcosahedronShape(ParticleShape):

    def __init__(self,
                 np.ndarray positions       ,
                 np.ndarray neighbors       ,
                 np.ndarray layers          ,
                 np.ndarray e_relax_list    ,
                 float      lattice_constant,
                 int        n_coord_min     ,
                 int        n_coord_min_iter,
                 float      e_coh_bulk      ,
                 float      e_twin          ,
                 float      shear_modulus   ,
                 float      k_strain        ):

        self.positions        = positions
        self.neighbors        = neighbors
        self.layers           = layers
        self.lattice_constant = lattice_constant
        self.n_coord_min      = n_coord_min
        self.n_coord_min_iter = n_coord_min_iter
        self.e_coh_bulk       = e_coh_bulk
        self.e_relax_list     = e_relax_list
        self.e_twin           = e_twin
        self.shear_modulus    = shear_modulus
        self.k_strain         = k_strain
        self.particle_type    = 'icosahedron'

    # ----------------------------------------------------------------------
    #  GET SHAPE
    # ----------------------------------------------------------------------

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def get_shape(self):

        cdef:
            np.ndarray positions        = self.positions
            np.ndarray neighbors        = self.neighbors
            np.ndarray layers           = self.layers
            float      lattice_constant = self.lattice_constant
            int        n_coord_min      = self.n_coord_min
            int        n_coord_min_iter = self.n_coord_min_iter

        cdef:
            int        p1, p2, p3
            float      distance
            list       del_indices
            np.ndarray norm
            np.ndarray center  = np.zeros(3, dtype = float)
            np.ndarray indices = np.arange(len(positions))

        for p in range(len(positions)):
            center += positions[p]

        for i in range(len(center)):
            center[i] /= len(positions)

        for p in range(len(positions)):
            positions[p] -= center

        for i in range(len(octa_triplets)):
        
            distance = 0.5*lattice_constant*(layers[i]+0.5)
        
            p1, p2, p3 = octa_triplets[i]
            
            norm = octa_vert[p1]+octa_vert[p2]+octa_vert[p3]
            
            norm = norm/np.linalg.norm(norm)
            
            del_indices = []
            
            for p in range(len(positions)):
            
                if (norm[0]*(positions[p][0]-octa_vert[p3][0]) +
                    norm[1]*(positions[p][1]-octa_vert[p3][1]) +
                    norm[2]*(positions[p][2]-octa_vert[p3][2])) > distance:
                    del_indices += [p]
            
            positions = np.delete(positions, del_indices, axis = 0)
            neighbors = np.delete(neighbors, del_indices, axis = 0)
            indices   = np.delete(indices, del_indices, axis = 0)

        cdef:
            float      dispersion
            int        iteration
            float      distance_max, distance_min, dis
            int        n_atoms      = len(positions)
            np.ndarray n_coord      = np.zeros(n_atoms, dtype = int)
            np.ndarray n_coord_dist = np.zeros(13, dtype = int)

        for iteration in range(n_coord_min_iter+1):

            n_coord = np.zeros(n_atoms, dtype = int)

            for p in range(n_atoms):
                for q in np.nditer(neighbors[p]):
                    if q in indices:
                        n_coord[p] += 1

            if iteration != n_coord_min_iter:

                del_indices = []
    
                for i in range(len(n_coord)):
                    
                    vertex = False
                    
                    for vert in octa_norm:
                        pos_norm = positions[i]/np.linalg.norm(positions[i])
                        if np.dot(pos_norm, vert) >= 0.999:
                            vertex = True
                    
                    if n_coord[i] < n_coord_min:
                        if vertex is False or n_coord[i] < 5:
                            del_indices += [i]
    
                positions = np.delete(positions, del_indices, axis = 0)
                neighbors = np.delete(neighbors, del_indices, axis = 0)
                indices   = np.delete(indices, del_indices, axis = 0)

            n_atoms = len(positions)
            
            if n_atoms <= 13:
                break

        for p in range(len(positions)):
            positions[p] += center

        bincount = np.bincount(n_coord)

        for i in range(len(bincount)):
            n_coord_dist[i] = bincount[i]

        dispersion = np.sum(n_coord_dist[:10])/n_atoms

        self.n_atoms      = n_atoms
        self.positions    = positions
        self.neighbors    = neighbors
        self.indices      = indices
        self.n_coord      = n_coord
        self.n_coord_dist = n_coord_dist
        self.dispersion   = dispersion

        self.get_n_twin()

        return positions

    # ----------------------------------------------------------------------
    #  GET N TWIN
    # ----------------------------------------------------------------------

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def get_n_twin(self):

        cdef:
            np.ndarray layers = self.layers

        cdef:
            int        i, j
            float      n_twin = 0.

        for i in range(20):
            for j in range(layers[i]+1):
                n_twin += j*1.5

        self.n_twin = n_twin

        return n_twin

################################################################################
# CALCULATE NEIGHBORS
################################################################################

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def calculate_neighbors(np.ndarray positions   ,
                        np.ndarray cell        ,
                        float      interact_len):

    cdef:
        int        i, j, k, p, q, c
        list       box_num
        float      distance
        np.ndarray cell_diag = sum(cell)
        int        n_atoms   = len(positions)
        int        n_max     = 108
        int        max_coord = 12
        np.ndarray dist_vect = np.zeros(3, dtype = float)
        np.ndarray box_len   = np.zeros(3, dtype = float)
        np.ndarray neighbors = np.zeros((n_atoms, max_coord), dtype = int)
        np.ndarray n_coord   = np.zeros(n_atoms, dtype = int)

    box_num = [int(np.ceil(cell_diag[0]/interact_len)),
               int(np.ceil(cell_diag[1]/interact_len)),
               int(np.ceil(cell_diag[2]/interact_len))]
    
    box_len = np.divide(cell_diag, box_num)
    
    cdef:
        np.ndarray i_arrays = np.zeros((n_atoms, 3), dtype = int)
        np.ndarray i_matrix = np.zeros(box_num+[n_max], dtype = int)
        np.ndarray i_count  = np.zeros(box_num, dtype = int)
    
    for i in range(len(i_matrix)):
        i_matrix[i] = -1
    
    for p in range(n_atoms):
        i = int(np.floor(positions[p][0]/box_len[0]+1e-6))
        j = int(np.floor(positions[p][1]/box_len[1]+1e-6))
        k = int(np.floor(positions[p][2]/box_len[2]+1e-6))
        i_arrays[p] = np.array([i,j,k])
        for ii in (i-1, i, i+1):
            for jj in (j-1, j, j+1):
                for kk in (k-1, k, k+1):
                    if (0 <= ii < box_num[0] and
                        0 <= jj < box_num[1] and
                        0 <= kk < box_num[2]):
                        c = int(i_count[ii,jj,kk])
                        i_matrix[ii,jj,kk,c] = p
                        i_count[ii,jj,kk] += 1

    for i in range(n_atoms):
        for j in range(max_coord):
            neighbors[i,j] = -1

    for p in range(n_atoms):
        i = int(i_arrays[p][0])
        j = int(i_arrays[p][1])
        k = int(i_arrays[p][2])
        for q in i_matrix[i,j,k]:
            if q >= 0 and q != p:
                dist_vect = positions[p]-positions[q]
                distance = sqrt(pow(dist_vect[0],2)+
                                pow(dist_vect[1],2)+
                                pow(dist_vect[2],2))
                if distance < interact_len:
                    c = int(n_coord[p])
                    neighbors[p,c] = q
                    n_coord[p] += 1

    return neighbors

################################################################################
# GET SURFACE AREA
################################################################################

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def get_surface_area(object,
                     float      bond_len):

    cdef:
        list       shell_tmp  = []
        np.ndarray positions  = np.copy(object.positions)
        np.ndarray n_coord    = object.n_coord
        np.ndarray center     = np.zeros(3, dtype = float)
        np.ndarray direction  = np.zeros(3, dtype = float)

    for i in range(len(positions)):
        if n_coord[i] != 12:
            shell_tmp += [positions[i]]

    cdef:
        float      area_surf
        np.ndarray shell = np.array(shell_tmp)

    for i in range(len(positions)):
        center += positions[i]/len(positions)

    for i in range(len(shell)):
        direction = shell[i]-center
        if np.linalg.norm(direction) > 1e-3:
            direction /= np.linalg.norm(direction)
            shell[i] += direction*bond_len

    hull = ConvexHull(shell)

    area_surf = hull.area

    object.area_surf = area_surf

    return area_surf

################################################################################
# GET E FORMATION WITH ADS
################################################################################

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def get_e_form_with_ads(object,
                        float      y_zero_e_bind,
                        float      m_ang_e_bind ,
                        float      alpha_cov    ,
                        float      beta_cov     ,
                        float      area_surf    ,
                        float      temperature  ,
                        float      delta_mu_ads ,
                        list       f_e_bind_corr,
                        str        entropy_model,
                        int        averag_e_bind,
                        str        e_form_denom ,
                        int        n_coord_thr  = 10):

    cdef:
        int        i, j, cn
        int        n_atoms          = object.n_atoms
        float      e_form_clean     = object.e_form_clean
        np.ndarray n_coord          = object.n_coord
        np.ndarray n_coord_dist     = object.n_coord_dist
        np.ndarray n_coord_top      = np.array([], dtype = int)
        np.ndarray indices_uns      = np.array([], dtype = int)
        np.ndarray n_coord_dist_uns = np.array([], dtype = int)
        np.ndarray indices_sat      = np.array([], dtype = int)
        np.ndarray n_coord_dist_sat = np.array([], dtype = int)

    for i in range(n_atoms):
        if n_coord[i] < n_coord_thr:
            n_coord_top = np.append(n_coord_top, n_coord[i])

    n_coord_top = np.sort(n_coord_top)

    for cn in range(n_coord_thr):
        if n_coord_dist[cn] > 0:
            indices_uns = np.append(indices_uns, cn)
            n_coord_dist_uns = np.append(n_coord_dist_uns, n_coord_dist[cn])

    cdef:
        float      e_bind
        float      cov_tot
        float      coverage
        float      e_bind_corr
        int        n_ads_tot       = 0
        int        n_uns_tot       = 0
        float      e_bind_ave      = 0.
        float      g_bind_ave      = 0.
        float      e_form_zero     = 0.
        int        n_atoms_top     = len(n_coord_top)
        int        n_indices_uns   = len(indices_uns)
        int        n_indices_sat   = 0
        np.ndarray n_ads           = np.zeros(n_coord_thr, dtype = float)
        np.ndarray cov             = np.zeros(n_coord_thr, dtype = float)
        np.ndarray g_bind          = np.zeros(n_coord_thr, dtype = float)
        np.ndarray e_form_ads_list = np.zeros(n_atoms_top, dtype = float)
        np.ndarray e_spec_ads_list = np.zeros(n_atoms_top, dtype = float)
        np.ndarray coverage_list   = np.zeros(n_atoms_top, dtype = float)
        np.ndarray cov_uns         = np.zeros(n_indices_uns, dtype = float)
        np.ndarray n_ads_uns       = np.zeros(n_indices_uns, dtype = float)
        np.ndarray g_bind_uns      = np.array([], dtype = float)
        np.ndarray g_bind_sat      = np.array([], dtype = float)

    if e_form_denom not in ('N met', 'N met + N ads'):

        raise NameError("e_form_denom: 'N met' | 'N met + N ads'")

    if entropy_model not in (None, '2D ideal gas', '2D lattice gas'):

        raise NameError("entropy_model: '2D ideal gas' | '2D lattice gas'")

    if averag_e_bind is True:

        for cn in range(n_coord_thr):
            
            e_bind = y_zero_e_bind+m_ang_e_bind*cn

            e_bind_ave += e_bind*n_coord_dist[cn]/n_atoms_top

        g_bind_ave = e_bind_ave-delta_mu_ads

        for i in range(n_atoms_top):
    
            n_ads_tot += 1
            cov_tot = float(n_ads_tot)/n_atoms_top
            coverage_list[i] = cov_tot
    
            e_form_zero = e_form_clean+n_ads_tot*g_bind_ave
    
            delta_e_cov = n_ads_tot*alpha_cov*(area_surf/n_ads_tot)**-beta_cov
    
            e_bind_corr = 0.
            for n in range(13):
                e_bind_corr += cov_tot*n_coord_dist[n]*f_e_bind_corr[n](cov_tot)
    
            if cov_tot > 0.999:
                cov_tot = 0.999
    
            if entropy_model is None:
                
                s_conf = 0.
    
            elif entropy_model == '2D lattice gas':

                s_conf = n_ads_tot*kB_eV*(np.log((1-cov_tot)/cov_tot) - 
                                          np.log(1-cov_tot)/cov_tot)

            elif entropy_model == '2D ideal gas':

                s_conf = n_ads_tot*kB_eV*(-np.log(cov_tot))

            e_form_ads_list[i] = (e_form_zero + delta_e_cov + e_bind_corr - 
                                  temperature*s_conf)

            if e_form_denom == 'N met':

                e_spec_ads_list[i] = e_form_ads_list[i]/n_atoms
            
            elif e_form_denom == 'N met + N ads':
            
                e_spec_ads_list[i] = e_form_ads_list[i]/(n_atoms+n_ads_tot)

    elif averag_e_bind is False:

        for cn in range(n_coord_thr):

            g_bind[cn] = y_zero_e_bind+m_ang_e_bind*cn-delta_mu_ads

            if cn in indices_uns:
                g_bind_uns = np.append(g_bind_uns, g_bind[cn])

        i = -1
        repeat = False
        
        while i < n_atoms_top-1:
            
            if repeat is False:
                i += 1
                n_ads_tot += 1
                n_uns_tot += 1

            cov_tot = float(n_ads_tot)/n_atoms_top
            coverage_list[i] = cov_tot
    
            args = (n_coord_dist_uns, g_bind_uns, n_uns_tot, temperature)
    
            n_ads_uns = np.array([float(n_uns_tot)/len(n_ads_uns)] * 
                                 len(n_ads_uns))
    
            repeat = False
    
            sol = optimize.root(get_n_ads_uns, n_ads_uns, args = args)

            n_ads_uns = sol.x

            for j in range(n_indices_uns):

                if n_ads_uns[j] >= n_coord_dist_uns[j]:
                    
                    n_indices_sat += 1
                    indices_sat = np.append(indices_sat, indices_uns[j])
                    n_coord_dist_sat = np.append(n_coord_dist_sat,
                                                 n_coord_dist_uns[j])
                    g_bind_sat = np.append(g_bind_sat, g_bind_uns[j])
                    
                    n_uns_tot -= n_coord_dist_uns[j]
                    
                    n_indices_uns -= 1
                    indices_uns = np.delete(indices_uns, j)
                    n_coord_dist_uns = np.delete(n_coord_dist_uns, j)
                    g_bind_uns = np.delete(g_bind_uns, j)
                    n_ads_uns = np.delete(n_ads_uns, j)
                    
                    repeat = True
                    
                    break

            g_bind_ave = 0.
            e_bind_corr = 0.
            
            for j in range(n_indices_uns):
                
                cn = indices_uns[j]
                cov_uns[j] = n_ads_uns[j]/n_coord_dist_uns[j]
                e_bind_corr += n_ads_uns[j]*f_e_bind_corr[cn](cov_uns[j])

                g_bind_ave += n_ads_uns[j]*g_bind_uns[j]
            
            for j in range(n_indices_sat):
                
                cn = indices_sat[j]
                e_bind_corr += n_coord_dist_sat[j]*f_e_bind_corr[cn](1.)

                g_bind_ave += n_coord_dist_sat[j]*g_bind_sat[j]
            
            e_form_zero = e_form_clean+g_bind_ave
            
            if cov_tot > 0.999:
                cov_tot = 0.999
    
            if entropy_model is None:
                
                s_conf = 0.
    
            elif entropy_model == '2D lattice gas':

                s_conf = n_ads_tot*kB_eV*(np.log((1-cov_tot)/cov_tot) - 
                                          np.log(1-cov_tot)/cov_tot)

            elif entropy_model == '2D ideal gas':

                s_conf = n_ads_tot*kB_eV*(-np.log(cov_tot))

            delta_e_cov = n_ads_tot*alpha_cov*(area_surf/n_ads_tot)**-beta_cov
    
            e_form_ads_list[i] = (e_form_zero + delta_e_cov + e_bind_corr - 
                                  temperature*s_conf)
    
            if e_form_denom == 'N met':

                e_spec_ads_list[i] = e_form_ads_list[i]/n_atoms
            
            elif e_form_denom == 'N met + N ads':
            
                e_spec_ads_list[i] = e_form_ads_list[i]/(n_atoms+n_ads_tot)

    else:

        e_form_zero += e_form_clean

        for i in range(n_atoms_top):
    
            n_ads_tot += 1
            cov_tot = float(n_ads_tot)/n_atoms_top
            coverage_list[i] = cov_tot
    
            cn = n_coord_top[i]
            n_ads[cn] += 1
            cov[cn] = float(n_ads[cn])/n_coord_dist[cn]
            
            e_bind_corr = 0.
            for n in range(n_coord_thr):
                e_bind_corr += n_ads[n]*f_e_bind_corr[n](cov[n])
    
            e_bind = y_zero_e_bind+m_ang_e_bind*n_coord_top[i]
    
            e_form_zero += e_bind-delta_mu_ads
    
            if cov_tot > 0.999:
                cov_tot = 0.999
    
            if entropy_model is None:
    
                s_conf = 0.
    
            elif entropy_model == '2D lattice gas':

                s_conf = n_ads_tot*kB_eV*(np.log((1-cov_tot)/cov_tot) - 
                                          np.log(1-cov_tot)/cov_tot)

            elif entropy_model == '2D ideal gas':

                s_conf = n_ads_tot*kB_eV*(-np.log(cov_tot))

            delta_e_cov = n_ads_tot*alpha_cov*(area_surf/n_ads_tot)**-beta_cov
    
            e_form_ads_list[i] = (e_form_zero + delta_e_cov + e_bind_corr - 
                                  temperature*s_conf)
    
            if e_form_denom == 'N met':

                e_spec_ads_list[i] = e_form_ads_list[i]/n_atoms
            
            elif e_form_denom == 'N met + N ads':
            
                e_spec_ads_list[i] = e_form_ads_list[i]/(n_atoms+n_ads_tot)

    e_spec_ads = np.min(e_spec_ads_list)
    coverage   = coverage_list[np.argmin(e_spec_ads_list)]

    object.e_form_ads_list = e_form_ads_list
    object.e_spec_ads_list = e_spec_ads_list
    object.coverage_list   = coverage_list
    object.coverage        = coverage
    object.e_spec_ads      = e_spec_ads
    object.e_spec          = e_spec_ads

    if e_form_denom == 'N met':

        object.e_form_ads = e_spec_ads*n_atoms
        object.e_form     = e_spec_ads*n_atoms
    
    elif e_form_denom == 'N met + N ads':
    
        object.e_form_ads = e_spec_ads*(n_atoms+n_ads_tot)
        object.e_form     = e_spec_ads*(n_atoms+n_ads_tot)

    return object.e_form_ads

################################################################################
# GET N ADS UNS
################################################################################

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def get_n_ads_uns(np.ndarray n_ads_uns       ,
                  np.ndarray n_coord_dist_uns,
                  np.ndarray g_bind_uns      ,
                  int        n_ads_tot       ,
                  float      temperature     ,
                  float      epsi            = 1e+00,
                  float      scale_factor    = 1e+04):
    
    cdef:
        int        i
        np.ndarray err = np.zeros(len(n_ads_uns))
    
    for i in range(len(n_ads_uns)):
    
        if n_ads_uns[i] > n_coord_dist_uns[i]-epsi:
            err[i] += scale_factor*(n_ads_uns[i]-n_coord_dist_uns[i]+epsi)
            n_ads_uns[i] = n_ads_tot-epsi
        
        elif n_ads_uns[i] < 0.+epsi:
            err[i] += scale_factor*(n_ads_uns[i]-0.-epsi)
            n_ads_uns[i] = 0.+epsi
    
    err[0] = np.sum(n_ads_uns)-n_ads_tot

    for i in range(1, len(n_ads_uns)):

        err[i] = (n_ads_uns[0]*n_coord_dist_uns[i] - 
                  n_ads_uns[i]*n_coord_dist_uns[0] *
                  np.exp(-(g_bind_uns[0]-g_bind_uns[i])/(kB_eV*temperature)))

    return err

################################################################################
# REMOVE ATOMS
################################################################################

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def particle_remove_atoms(object,
                          int        n_iterations  ,
                          int        remove_groups = False):

    cdef:
        int        i, p, q
        int        n_coord_min
        list       del_indices
        int        n_removed     = 0
        np.ndarray positions     = object.positions
        np.ndarray neighbors     = object.neighbors
        np.ndarray indices       = object.indices
        np.ndarray n_coord       = object.n_coord
        np.ndarray n_coord_dist  = np.zeros(13, dtype = int)
        np.ndarray n_coord_neigh = np.zeros(len(positions), dtype = int)

    n_coord_min = np.min(n_coord)

    for p in range(len(n_coord)):
        if n_coord[p] == n_coord_min:
            for q in neighbors[p]:
                if q in indices:
                    i = np.where(indices == q)[0]
                    n_coord_neigh[p] += n_coord[i]
        else:
            n_coord_neigh[p] = 1000

    n_coord_neigh_min = np.min(n_coord_neigh)

    del_indices = []

    for i in range(len(n_coord_neigh)):
        if n_coord_neigh[i] == n_coord_neigh_min:
            del_indices += [i]
            n_removed += 1
            if n_removed >= n_iterations and remove_groups is False:
                break

    positions = np.delete(positions, del_indices, axis = 0)
    neighbors = np.delete(neighbors, del_indices, axis = 0)
    indices   = np.delete(indices, del_indices, axis = 0)

    n_atoms = len(positions)

    n_coord = np.zeros(n_atoms, dtype = int)

    for p in range(n_atoms):
        for q in np.nditer(neighbors[p]):
            if q in indices:
                n_coord[p] += 1

    n_coord_neigh = np.zeros(n_atoms, dtype = int)

    bincount = np.bincount(n_coord)

    for i in range(len(bincount)):
        n_coord_dist[i] = bincount[i]

    dispersion = np.sum(n_coord_dist[:10])/n_atoms

    object.n_atoms      = n_atoms
    object.positions    = positions
    object.neighbors    = neighbors
    object.indices      = indices
    object.n_coord      = n_coord
    object.n_coord_dist = n_coord_dist
    object.dispersion   = dispersion

    return positions

################################################################################
# GET MULTIPLICITY
################################################################################

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def particle_multiplicity(object,
                          float      multip_bulk):

    cdef:
        tuple      cell
        float      multiplicity
        float      n_symmetries
        np.ndarray lattice   = np.identity(3)*(np.max(object.positions)+10.)
        np.ndarray positions = object.positions
        np.ndarray numbers   = np.array([1]*object.n_atoms)
    
    cell = (lattice, positions, numbers)
    
    n_symmetries = len(sp.get_symmetry(cell, symprec = 1e-3)['rotations'])

    multiplicity = multip_bulk/n_symmetries

    object.n_symmetries = n_symmetries
    object.multiplicity = multiplicity

    return multiplicity

################################################################################
# END
################################################################################
