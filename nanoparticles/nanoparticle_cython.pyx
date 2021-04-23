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
    #  SET LATTICE CONSTANT
    # ----------------------------------------------------------------------

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def set_lattice_constant(self,
                             float      lc_new,
                             float      lc_old = 0.):

        if lc_old == 0.:
            lc_old = self.lattice_constant

        if lc_new != lc_old:
            for position in self.positions:
                position *= lc_new/lc_old

        self.lattice_constant = lc_new

        return self.positions

    # ----------------------------------------------------------------------
    #  CLEAN DATA
    # ----------------------------------------------------------------------

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def reduce_data(self):

        for site in self.active_sites:

            site.reduce_data()

        vars_all = [var for var in vars(self)]

        vars_reduced = ['particle_type'   ,
                        'n_atoms'         ,
                        'positions'       ,
                        'n_coord_dist'    ,
                        'lattice_constant',
                        'active_sites'    ]
    
        for var in [var for var in vars_all if var not in vars_reduced]:

            vars(self)[var] = None

            del vars(self)[var]
    
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

        positions = remove_atoms(self,
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
    
        multiplicity = get_multiplicity(self, multip_bulk = multip_bulk)
    
        return multiplicity

    # ----------------------------------------------------------------------
    #  GET ENERGY
    # ----------------------------------------------------------------------

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def get_energy_clean(self):

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
    #  GET SURFACE AREA
    # ----------------------------------------------------------------------

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def get_surface_area(self,
                         float      bond_length = 0.):

        cdef:
            float      area_surf

        area_surf = get_surface_area(self,
                                     bond_length = bond_length)

        return area_surf

    # ----------------------------------------------------------------------
    #  GET ENERGY WITH ADS
    # ----------------------------------------------------------------------

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def get_energy_with_ads(self,
                            dict       g_bind_dict   ,
                            float      bond_length   ,
                            float      temperature   ,
                            int        sites_equilib = True):

        cdef:
            float      e_form_ads

        e_form_ads = get_energy_with_ads(self,
                                         g_bind_dict   = g_bind_dict  ,
                                         bond_length   = bond_length  ,
                                         temperature   = temperature  ,
                                         sites_equilib = sites_equilib)

        return e_form_ads

    # ----------------------------------------------------------------------
    #  GET ACTIVE SITES
    # ----------------------------------------------------------------------

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def get_active_sites(self,
                         int        specify_n_coord  = True ,
                         int        specify_supp_int = False,
                         int        specify_facets   = False,
                         int        check_duplicates = False,
                         int        multiple_facets  = False,
                         int        convex_sites     = True ):

        from nanoparticle_active_sites import get_surface, get_active_sites

        surface = get_surface(positions    = self.positions,
                              neighbors    = self.neighbors, 
                              indices      = self.indices  , 
                              n_coord      = self.n_coord  ,
                              supp_contact = None          ,
                              n_coord_max  = 12            )
        
        active_sites = get_active_sites(surface          = surface         ,
                                        specify_n_coord  = specify_n_coord ,
                                        specify_supp_int = specify_supp_int,
                                        specify_facets   = specify_facets  ,
                                        check_duplicates = check_duplicates,
                                        multiple_facets  = multiple_facets ,
                                        convex_sites     = convex_sites    )

        self.active_sites = active_sites

        return active_sites

    # ----------------------------------------------------------------------
    #  GET ACTIVE SITES DICT
    # ----------------------------------------------------------------------

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def get_active_sites_dict(self,
                              int        with_tags  = True):

        from nanoparticle_active_sites import get_active_sites_dict

        active_sites_dict = get_active_sites_dict(
                                            active_sites = self.active_sites,
                                            with_tags    = with_tags        )

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
                 float      lattice_constant,
                 float      scale_one       , 
                 float      scale_two       , 
                 int        n_coord_min     ,
                 float      interact_len    , 
                 float      e_coh_bulk      ,
                 int        miller_symmetry ,
                 list       active_sites    = []):

        self.particle_type    = 'fcc particle'
        self.positions        = positions
        self.neighbors        = neighbors
        self.cell             = cell
        self.translation      = translation
        self.miller_indices   = miller_indices
        self.planes_distances = planes_distances
        self.e_relax_list     = e_relax_list
        self.lattice_constant = lattice_constant
        self.scale_one        = scale_one
        self.scale_two        = scale_two
        self.n_coord_min      = n_coord_min
        self.interact_len     = interact_len
        self.e_coh_bulk       = e_coh_bulk
        self.miller_symmetry  = miller_symmetry
        self.active_sites     = active_sites

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
                 float      k_strain        ,
                 list       active_sites    = []):

        self.particle_type    = 'decahedron'
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
        self.active_sites     = active_sites

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
                 float      k_strain        ,
                 list       active_sites    = []):

        self.particle_type    = 'icosahedron'
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
        self.active_sites     = active_sites

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
                     float      bond_length):

    from scipy.spatial import ConvexHull

    cdef:
        np.ndarray positions = np.copy(object.positions)
        np.ndarray center    = np.zeros(3, dtype = float)
        np.ndarray direction = np.zeros(3, dtype = float)

    cdef:
        float      area_surf

    for i in range(len(positions)):
        center += positions[i]/len(positions)

    for i in range(len(positions)):
    
        direction = positions[i]-center
    
        if np.linalg.norm(direction) > 1e-4:
            direction /= np.linalg.norm(direction)
            positions[i] += direction*bond_length

    hull = ConvexHull(positions)

    area_surf = hull.area

    object.area_surf = area_surf

    return area_surf

################################################################################
# GET E FORMATION WITH ADS
################################################################################

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def get_energy_with_ads(object,
                        dict       g_bind_dict   ,
                        float      bond_length   ,
                        float      temperature   ,
                        int        sites_equilib = True):

    cdef:
        float      area_surf
        list       active_sites_sel = []
        float      kB_T             = kB_eV*temperature
        float      e_form_clean     = object.e_form_clean
        float      e_spec_clean     = object.e_spec_clean
        int        n_atoms          = object.n_atoms
        int        n_atoms_surf     = sum(object.n_coord_dist[:10])

    area_surf = object.get_surface_area(bond_length = bond_length)

    for site in object.active_sites:
        if site.name in g_bind_dict:
            active_sites_sel += [site]
    
    for site in active_sites_sel:
        
        site.prob     = 0.
        site.prob_i   = 0.
        site.coverage = 0.
        
        site.n_coord_ave = sum(site.n_coord)/len(site.n_coord)
        site.g_bind = g_bind_dict[site.name](site.n_coord_ave, area_surf)
    
    active_sites_sel = sorted(active_sites_sel, key = lambda x: x.g_bind)
    
    cdef:
        int        i_ads
        float      area_per_ads, denom
        int        n_ads_i         = 0
        int        n_ads           = 0
        float      coverage        = 0.
        int        n_points        = int(n_atoms_surf)+1
        np.ndarray e_form_ads_vect = np.array([e_form_clean]*n_points)
        np.ndarray e_spec_ads_vect = np.array([e_spec_clean]*n_points)
        np.ndarray coverage_vect   = np.zeros(n_points, dtype = float)
    
    for i_ads in range(n_points-1):
        
        n_ads_i += 1
        
        area_per_ads = area_surf/n_ads_i
        
        denom = 0.
        
        for site in active_sites_sel:
            
            site.g_bind = g_bind_dict[site.name](
                                        cn_ave       = site.n_coord_ave,
                                        area_per_ads = area_per_ads    )
        
            if sites_equilib:
            
                site.prob_i = np.exp(-site.g_bind/kB_T)*(1.-site.prob)
        
                denom += site.prob_i
        
        for i_site, site in enumerate(active_sites_sel):
            
            if sites_equilib:
                site.prob_i /= denom
                site.prob += site.prob_i
                site.g_bind += kB_T*np.log(site.prob)
                
            else:
                site.prob = 1. if i_site < n_ads_i else 0.
            
            e_form_ads_vect[n_ads_i] += site.g_bind*site.prob
        
        e_spec_ads_vect[n_ads_i] = e_form_ads_vect[n_ads_i]/n_atoms
    
        coverage_vect[n_ads_i] = float(n_ads_i)/n_atoms_surf
    
        if e_form_ads_vect[n_ads_i] == np.min(e_form_ads_vect):
            
            n_ads      = n_ads_i
            e_form_ads = e_form_ads_vect[n_ads_i]
            coverage   = coverage_vect[n_ads_i]
        
            for site in active_sites_sel:
                site.coverage = site.prob
    
    object.e_form_ads      = e_form_ads
    object.e_spec_ads      = e_form_ads/n_atoms
    object.e_form          = e_form_ads
    object.e_spec          = e_form_ads/n_atoms
    object.e_form_ads_vect = e_form_ads_vect
    object.e_spec_ads_vect = e_spec_ads_vect
    object.coverage_vect   = coverage_vect
    object.n_ads           = n_ads
    object.coverage        = coverage

    return e_form_ads

################################################################################
# REMOVE ATOMS
################################################################################

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def remove_atoms(object,
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
def get_multiplicity(object,
                     float      multip_bulk):

    import spglib

    cdef:
        tuple      cell
        float      multiplicity
        float      n_symmetries
        np.ndarray lattice   = np.identity(3)*(np.max(object.positions)+10.)
        np.ndarray positions = object.positions
        np.ndarray numbers   = np.array([1]*object.n_atoms)
    
    cell = (lattice, positions, numbers)
    
    n_symmetries = len(spglib.get_symmetry(cell, symprec = 1e-3)['rotations'])

    multiplicity = multip_bulk/n_symmetries

    object.n_symmetries = n_symmetries
    object.multiplicity = multiplicity

    return multiplicity

################################################################################
# END
################################################################################
