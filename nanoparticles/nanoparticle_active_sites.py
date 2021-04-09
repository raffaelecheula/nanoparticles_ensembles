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
import numpy as np
import copy as cp
import matplotlib.pyplot as plt
from collections import OrderedDict
from operator import attrgetter
from ase import Atom, Atoms
from ase.build.tools import rotation_matrix
from nanoparticle_utils import get_interact_len, get_neighbor_atoms

################################################################################
# COLORS
################################################################################

all_sites_names = ['top', 'brg', 'lbr',
                   'hcp', 'fcc', 'hol',
                   'lho', 'bho', 'tho']

black  = np.array([  0/255,   0/255,   0/255])
white  = np.array([255/255, 255/255, 255/255])
red    = np.array([255/255,  51/255,  51/255])
blue   = np.array([ 51/255,  51/255, 255/255])
yellow = np.array([255/255, 255/255,  51/255])
orange = np.array([255/255, 165/255,  51/255])
green  = np.array([ 51/255, 234/255,  51/255])
violet = np.array([153/255,  51/255, 255/255])
brown  = np.array([102/255,   0/255,   0/255])
grey   = np.array([105/255, 105/255, 105/255])
purple = np.array([128/255,   0/255, 128/255])
forest = np.array([ 34/255, 139/255,  34/255])

colors_dict = {'top': red   ,
               'brg': blue  ,
               'lbr': green ,
               'hcp': orange,
               'fcc': yellow,
               'hol': violet,
               'lho': forest,
               'bho': brown ,
               'tho': purple}

facet_colors_dict = {'100': red   ,
                     '110': orange,
                     '111': blue  ,
                     '311': green ,
                     '210': violet,
                     'dec': forest,
                     'aat': brown }

sizes_dict = {'top': 1.0,
              'brg': 0.3,
              'lbr': 0.5,
              'hcp': 0.5,
              'fcc': 0.5,
              'hol': 0.5,
              'lho': 0.5,
              'bho': 0.5,
              'tho': 0.5}

dummy_species_dict = {'top': 'Rh',
                      'brg': 'H' ,
                      'lbr': 'N' ,
                      'hcp': 'F' ,
                      'fcc': 'O' ,
                      'hol': 'P' ,
                      'lho': 'S' ,
                      'bho': 'B' ,
                      'tho': 'Be'}

################################################################################
# SURFACE ATOM CLASS
################################################################################

class SurfaceAtom():

    def __init__(self,
                 position     = np.zeros(3),
                 neighbors    = []         ,
                 neigh_zero   = []         ,
                 n_coord      = None       ,
                 supp_contact = False      ,
                 index        = None       ,
                 index_zero   = None       ):

        self.position     = position
        self.neighbors    = neighbors
        self.neigh_zero   = neigh_zero
        self.n_coord      = n_coord
        self.supp_contact = supp_contact
        self.index        = index
        self.index_zero   = index_zero

################################################################################
# ACTIVE SITE CLASS
################################################################################

class ActiveSite():

    def __init__(self, 
                 name         = ''         ,
                 tag          = ''         ,
                 position     = np.zeros(3),
                 position_xy  = np.zeros(2),
                 neighbors    = []         ,
                 n_coord      = None       ,
                 index        = None       ,
                 supp_contact = 0          ,
                 facet_type   = []         ,
                 deleted      = False      ):

        self.name         = name
        self.tag          = tag
        self.position     = position
        self.position_xy  = position_xy
        self.neighbors    = neighbors
        self.n_coord      = n_coord
        self.index        = index
        self.supp_contact = supp_contact
        self.facet_type   = facet_type
        self.deleted      = deleted

    def delete(self):

        self.__init__()
        self.deleted = True

################################################################################
# GET SURFACE SHELL
################################################################################

def get_surface_shell(positions, neighbors, indices, n_coord,
                      supp_contact = None, n_coord_max = 12):

    if supp_contact is None:
        supp_contact = [False for j in range(len(n_coord))]

    indices_zero = np.copy(indices)

    del_indices = []

    for i in range(len(n_coord)):
        if n_coord[i] >= n_coord_max:
            del_indices += [i]

    indices = np.delete(indices, del_indices, axis = 0)

    indices_zero_dict = {}
    
    for i in range(len(indices_zero)):
        indices_zero_dict[indices_zero[i]] = i

    indices_dict = {}

    for i in range(len(indices)):
        indices_dict[indices[i]] = i

    index = 0

    surface = []

    for i in range(len(n_coord)):
        
        if n_coord[i] < n_coord_max:
        
            neigh_zero_new = [ indices_zero_dict[j] for j in neighbors[i]
                               if j in indices_zero_dict ]
        
            neighbors_new = [ indices_dict[j] for j in neighbors[i]
                              if j in indices_dict ]
        
            surface += [SurfaceAtom(neighbors    = neighbors_new  ,
                                    neigh_zero   = neigh_zero_new ,
                                    position     = positions[i]   ,
                                    n_coord      = n_coord[i]     ,
                                    supp_contact = supp_contact[i],
                                    index        = index          ,
                                    index_zero   = i              )]
            
            index += 1

    return surface

################################################################################
# GET ACTIVE SITES SHELL
################################################################################

def get_active_sites_shell(surface,
                           specify_n_coord  = True ,
                           specify_supp_int = True ,
                           specify_facets   = False,
                           check_duplicates = False,
                           multiple_facets  = False):

    active_sites = []
    index = 0

    """
    Identification of top (top), and long-hollow (lho) active sites. 
    """

    for a in surface:

        if a.n_coord == 11:

            bi_vect_zero = [ bi for bi in a.neigh_zero 
                             if bi in [c.index_zero for c in surface] ]

            bi_vect = [ bi for bi in [ b.index for b in surface
                                       if b.index_zero in bi_vect_zero ]
                        if len([ ci for ci in a.neigh_zero
                                 if ci in surface[bi].neigh_zero ]) == 3 ]

            position  = np.zeros(3)
            neighbors = []
            n_coord   = []
            
            for bi in bi_vect:

                b = surface[bi]

                position += b.position/(len(bi_vect))
                neighbors += [b.index]
                n_coord += [b.n_coord]

            if len(bi_vect) == 5:
                position = a.position.copy()
                facet_type = ['dec']
            else:
                facet_type = []

            active_sites += [ActiveSite(name       = 'lho'     ,
                                        position   = position  ,
                                        neighbors  = neighbors ,
                                        n_coord    = n_coord   ,
                                        index      = index     ,
                                        facet_type = facet_type)]

            index += 1

        else:

            position  = a.position.copy()
            neighbors = []
            n_coord   = [a.n_coord]

            supp_contact = 1 if a.supp_contact is True else 0

            active_sites += [ActiveSite(name         = 'top'       ,
                                        position     = position    ,
                                        neighbors    = neighbors   ,
                                        n_coord      = n_coord     ,
                                        index        = index       ,
                                        supp_contact = supp_contact)]

            index += 1

    """
    Identification of dechaedron-hollow (dec), long-bridge (lbr),
    hollow (hol), and bridge (brg) active sites. 
    """

    for a in surface:

        if a.n_coord == 11:

            a_site = active_sites[a.index]

            if 'dec' not in a_site.facet_type:

                for bi, ci in [ (bi, ci) for bi in a_site.neighbors[:4]
                                for ci in a_site.neighbors[:4]
                                if bi > ci
                                if bi not in surface[ci].neighbors
                                if len([ di for di in surface[bi].neighbors
                                   if di in surface[ci].neighbors ]) == 2 ]:
    
                    b = surface[bi]
                    c = surface[ci]
                    
                    di_vect = [ di for di in active_sites[bi].neighbors
                                if di in active_sites[ci].neighbors
                                if active_sites[di].name == 'lbr' ]
                    
                    for di in di_vect:
                    
                        d_site = active_sites[di]
                        
                        d_site.neighbors += [a_site.index]
                        a_site.neighbors += [d_site.index]
                        
                        for ii in (b.index, c.index):
                            active_sites[ii].neighbors += [d_site.index]
                            active_sites[ii].neighbors += [a_site.index]
                        
                    if len(di_vect) == 0:
                    
                        position  = (b.position+c.position)/2.
                        neighbors = [b.index, c.index, a.index]
                        n_coord   = [b.n_coord, c.n_coord]
                        
                        active_sites += [ActiveSite(name      = 'lbr'    ,
                                                    position  = position ,
                                                    neighbors = neighbors,
                                                    n_coord   = n_coord  ,
                                                    index     = index    )]
            
                        a_site.neighbors += [index]
                        
                        for ii in (b.index, c.index):
                            active_sites[ii].neighbors += [index]
                            active_sites[ii].neighbors += [a.index]
                        
                        index += 1

            else:
                
                for bi in a_site.neighbors:
                    active_sites[bi].neighbors += [a.index]

        else:

            for bi in [ bi for bi in a.neighbors
                        if bi > a.index 
                        if surface[bi].n_coord < 11 ]:

                b = surface[bi]

                for ci in [ ci for ci in a.neighbors
                            if ci > b.index
                            if ci not in b.neighbors 
                            if surface[ci].n_coord < 11 ]:

                    c = surface[ci]

                    for di in [ di for di in b.neighbors
                                if di in c.neighbors 
                                if di > a.index
                                if di not in a.neighbors 
                                if surface[di].n_coord < 11 ]:

                        d = surface[di]

                        if [ ei for ei in a.neighbors
                             if ei in b.neighbors 
                             if ei in c.neighbors 
                             if ei in d.neighbors ]:
                            continue

                        position  = (a.position+b.position + 
                                     c.position+d.position)/4.
                        neighbors = [a.index, b.index,
                                     c.index, d.index]
                        n_coord   = [a.n_coord, b.n_coord,
                                     c.n_coord, d.n_coord]

                        active_sites += [ActiveSite(name      = 'hol'    ,
                                                    position  = position ,
                                                    neighbors = neighbors, 
                                                    n_coord   = n_coord  ,
                                                    index     = index    )]

                        active_sites[index].facet_type = ['100']

                        for ii in (a.index, b.index, c.index, d.index):
                            active_sites[ii].neighbors += [index]
                        
                        index += 1

                #if len([ ci for ci in a.neigh_zero
                #         if ci in b.neigh_zero
                #         if not [ s for s in surface 
                #                  if s.index_zero == ci
                #                  if s.n_coord < 4 ] ]) > 3:
                
                if len([ ci for ci in a.neigh_zero
                         if ci in b.neigh_zero ]) > 3:
                    
                    for ci, di in [(ci, di) for ci in a.neighbors
                                    for di in a.neighbors
                                    if ci > di
                                    if ci in b.neighbors
                                    if di in b.neighbors
                                    if di not in surface[ci].neighbors ]:
                    
                        c = surface[ci]
                        d = surface[di]
                    
                        position  = (c.position+d.position)/2.
                        neighbors = [a.index, b.index, c.index, d.index]
                        n_coord   = [c.n_coord, d.n_coord]
                        
                        active_sites += [ActiveSite(name      = 'lbr'    ,
                                                    position  = position ,
                                                    neighbors = neighbors,
                                                    n_coord   = n_coord  ,
                                                    index     = index    )]
                    
                        for i in (a.index, b.index, c.index, d.index):
                            active_sites[i].neighbors += [index]
                        
                        index += 1
                    
                    continue

                position  = (a.position + b.position)/2.
                neighbors = [a.index, b.index]
                n_coord   = [a.n_coord, b.n_coord]

                active_sites += [ActiveSite(name      = 'brg'    ,
                                            position  = position ,
                                            neighbors = neighbors,
                                            n_coord   = n_coord  ,
                                            index     = index    )]

                for i in (a.index, b.index):
                    active_sites[i].neighbors += [index]

                index += 1

                """
                for ci in [ ci for ci in a.neighbors
                            if ci in b.neighbors 
                            if ci > b.index
                            if surface[ci].n_coord < 11 ]:

                    c = surface[ci]

                    if ( a.supp_contact is True
                         or b.supp_contact is True 
                         or c.supp_contact is True ):
                         continue
                    
                    if [ di for di in a.neighbors 
                         if di in b.neighbors
                         if di in c.neighbors 
                         if surface[di].n_coord == 3 ]:
                        continue

                    #if ( len([ di for di in a.neigh_zero
                    #           if di in c.neigh_zero ]) > 3 or
                    #     len([ di for di in b.neigh_zero
                    #           if di in c.neigh_zero ]) > 3 ):
                    #    continue

                    position  = (a.position+b.position+c.position)/3.
                    neighbors = [a.index, b.index, c.index]
                    n_coord   = [a.n_coord, b.n_coord, c.n_coord]

                    if [ i for i in a.neigh_zero
                        if i in b.neigh_zero
                        if i in c.neigh_zero ]:

                        active_sites += [ActiveSite(name      = 'fcc'    ,
                                                    position  = position ,
                                                    neighbors = neighbors,
                                                    n_coord   = n_coord  ,
                                                    index     = index    )]

                    else:

                        active_sites += [ActiveSite(name      = 'hcp'    ,
                                                    position  = position ,
                                                    neighbors = neighbors,
                                                    n_coord   = n_coord  ,
                                                    index     = index    )]

                    for i in (a.index, b.index, c.index):
                        active_sites[i].neighbors += [index]

                    index += 1
                """

    

    """
    Identification of fcc (fcc), and hcp (hcp) active sites. 
    """

    for a in surface:

        ai = a.index

        for bi, ci in [ (bi, ci) for bi in a.neighbors
                        for ci in a.neighbors
                        if ci in surface[bi].neighbors
                        if bi > a.index
                        if ci > bi ]:

                b = surface[bi]
                c = surface[ci]

                if [ di for di in a.neighbors 
                     if di in b.neighbors
                     if di in c.neighbors 
                     if surface[di].n_coord == 3 ]:
                    continue

                if len([ d for d in active_sites
                         if d.name == 'brg'
                         if d.neighbors[0] in [ai, bi, ci]
                         if d.neighbors[1] in [ai, bi, ci] ]) == 3:

                    position  = (a.position+b.position+c.position)/3.
                    neighbors = [a.index, b.index, c.index]
                    n_coord   = [a.n_coord, b.n_coord, c.n_coord]

                    if [ ei for ei in a.neigh_zero
                        if ei in b.neigh_zero
                        if ei in c.neigh_zero ]:

                        active_sites += [ActiveSite(name      = 'fcc'    ,
                                                    position  = position ,
                                                    neighbors = neighbors,
                                                    n_coord   = n_coord  ,
                                                    index     = index    )]

                    else:
                    
                        active_sites += [ActiveSite(name      = 'hcp'    ,
                                                    position  = position ,
                                                    neighbors = neighbors,
                                                    n_coord   = n_coord  ,
                                                    index     = index    )]

                    for i in (a.index, b.index, c.index):
                        active_sites[i].neighbors += [index]

                    index += 1
    """
    Identification of theta-hollow (tho), and beta-hollow (bho) active sites. 
    """

    for a_site in [ a_site for a_site in active_sites 
                    if a_site.name == 'lbr' ]:
    
        for b in [ b for b in surface 
                   if b.n_coord < 11
                   if a_site.neighbors[0] in b.neighbors
                   if a_site.neighbors[1] in b.neighbors ]:
    
            a_site.neighbors += [b.index]
            active_sites[b.index].neighbors += [a_site.index]

    for a_site in [ a_site for a_site in active_sites 
                    if a_site.name == 'brg' ]:
    
        for b_site in [ b_site for b_site in active_sites 
                        if b_site.index != a_site.index
                        if a_site.neighbors[0] in b_site.neighbors
                        if a_site.neighbors[1] in b_site.neighbors ]:
    
            a_site.neighbors += [b_site.index]
            b_site.neighbors += [a_site.index]

    for a in [ a for a in active_sites if a.name == 'brg' ]:

        for bi, ci in [ (bi, ci) for bi in a.neighbors for ci in a.neighbors
                        if active_sites[bi].name in ('fcc', 'hcp')
                        if active_sites[ci].name == 'hol' ]:
    
            b = active_sites[bi]
            c = active_sites[ci]

            d = surface[a.neighbors[0]]
            e = surface[a.neighbors[1]]
            
            fi = [ fi for fi in d.neighbors if fi in e.neighbors ][0]

            f = surface[fi]
            
            try: 
                gi = [ gi for gi in surface[c.neighbors[0]].neigh_zero
                           if gi in surface[c.neighbors[1]].neigh_zero
                           if gi in surface[c.neighbors[2]].neigh_zero
                           if gi in surface[c.neighbors[3]].neigh_zero ][0]
            except:
                continue
            
            mi, ni = [ i for i in c.neighbors[:4] if i not in a.neighbors[:2] ]
            
            m = surface[mi]
            n = surface[ni]
            
            dir_1 = (d.position+e.position)/2.-(m.position+n.position)/2.
            dir_2 = f.position-(d.position+e.position)/2.
            
            dot_prod = np.dot(dir_1/np.linalg.norm(dir_1),
                              dir_2/np.linalg.norm(dir_2))
            
            if gi not in f.neigh_zero and dot_prod < 0.8:
            
                #a.position = c.position
                a.position = (2.*c.position+active_sites[bi].position)/3.
                a.name = 'bho'
                a.facet_type = ['311']
                
                for hi in c.neighbors:
            
                    a.neighbors += [hi]

                    for j, li in enumerate(active_sites[hi].neighbors):
                        if li == c.index:
                            active_sites[hi].neighbors[j] = a.index

                c.delete()

                for hi in a.neighbors:
                    if '311' not in active_sites[hi].facet_type:
                        active_sites[hi].facet_type += ['311']
                for hi in b.neighbors:
                    if '311' not in active_sites[hi].facet_type:
                        active_sites[hi].facet_type += ['311']

    for a in [ a for a in active_sites if a.name == 'lbr' ]:

        for bi, ci in [ (bi, ci) for bi in a.neighbors for ci in a.neighbors
                      if active_sites[bi].name == 'lho'
                      if active_sites[ci].name == 'top'
                      if bi not in active_sites[ci].neighbors ]:
    
            b = active_sites[bi]
            c = active_sites[ci]

            d = surface[a.neighbors[0]]
            e = surface[a.neighbors[1]]
            
            if surface[ci].index_zero not in surface[bi].neigh_zero:
            
                a.name = 'tho'
                a.facet_type = ['210']
                
                a.position = (a.position+active_sites[bi].position)/2.
                
                for hi in b.neighbors:
            
                    a.neighbors += [hi]
            
                    for j, li in enumerate(active_sites[hi].neighbors):
                        if li == b.index:
                            active_sites[hi].neighbors[j] = a.index
            
                b.delete()

                for hi in a.neighbors:
                    if '210' not in active_sites[hi].facet_type:
                        active_sites[hi].facet_type += ['210']

    """
    Identification of the facets of active sites. 
    """

    for a in [ a for a in active_sites if a.n_coord in ([1], [2], [3], [4]) ]:
    
        a.facet_type = ['aat']

    for a in [ a for a in active_sites if a.name == 'lho' ]:

        a.facet_type = ['110']

        for bi in a.neighbors:
        
            b = active_sites[bi]
        
            if '110' not in b.facet_type:
                b.facet_type += ['110']

    for a in [ a for a in active_sites if a.name == 'tho' ]:
    
        if not a.facet_type:
            a.facet_type = ['110']
    
        for bi in a.neighbors:
        
            b = active_sites[bi]
        
            if b.n_coord == [7] and '110' not in b.facet_type:
                b.facet_type += ['110']

    for a in [ a for a in active_sites 
               if a.name in ('fcc', 'hcp') ]:
    
        if not a.facet_type:
            a.facet_type = ['111']
    
            for bi in a.neighbors:
    
                b = active_sites[bi]
    
                if '111' not in b.facet_type:
                    b.facet_type += ['111']

    for a in [ a for a in active_sites if a.name == 'hol' ]:
    
        for bi in a.neighbors:

            b = active_sites[bi]

            if '100' not in b.facet_type:
                b.facet_type += ['100']

    for a in [ a for a in active_sites if a.n_coord == [9] ]:
    
        if not a.facet_type:
            a.facet_type = ['111']
    
        for bi in a.neighbors:
            
            b = active_sites[bi]
            
            if not b.facet_type:
                b.facet_type = ['111']

    for a in [ a for a in active_sites if a.n_coord == [8] ]:
    
        if not a.facet_type:
            a.facet_type = ['100']
    
        for bi in a.neighbors:
    
            b = active_sites[bi]
    
            if not b.facet_type:
                b.facet_type = ['100']

    """
    Identification of sites contacting the support.
    """

    for a in [ a for a in active_sites 
               if a.name == 'top' and a.supp_contact == 1 ]:

        for bi in a.neighbors:
            active_sites[bi].supp_contact += 1

    """
    Creation of the active sites tags. 
    """

    for a in [ a for a in active_sites if a.deleted is False ]:
    
        if specify_n_coord is True:

            a.tag += ','.join(['{0:02d}'.format(n) for n in sorted(a.n_coord)])

        elif specify_facets is True:
    
            facets = a.facet_type[:]
    
            if multiple_facets is False:
                facets = [facets[0]]
    
            a.tag += ','.join(['{}'.format(n) for n in sorted(facets)])
    
        if specify_supp_int is True:

            if len(a.tag) > 0:
                a.tag += '_'

            a.tag += 'S{0:02d}'.format(a.supp_contact)
    
    """
    Checking of duplicates active sites. 
    """

    if check_duplicates is True:

        for a in [ a for a in active_sites if a.deleted is False ]:

            for b in [ b for b in active_sites if b.deleted is False 
                       if b.index > a.index ]:

                if np.allclose(a.position, b.position) is True:

                    print('Duplicate site! {0} {1}'.format(a.name, b.name))

    return active_sites

################################################################################
# COUNT ACTIVE SITES
################################################################################

def count_active_sites(active_sites, with_tags = False):

    active_sites_dict = {}

    for name in all_sites_names:
        if with_tags is False:
            active_sites_dict[name] = 0
        else:
            active_sites_dict[name] = {}

    for a in [ a for a in active_sites if a.deleted is False ]:

        if with_tags is False:
            active_sites_dict[a.name] += 1

        else:
            if a.tag not in active_sites_dict[a.name]:
                active_sites_dict[a.name][a.tag] = 1
            else:
                active_sites_dict[a.name][a.tag] += 1

    return active_sites_dict

################################################################################
# REDUCE ACTIVE SITES
################################################################################

def reduce_active_sites(active_sites_dict):

    active_sites_dict_reduced = {}

    for name in active_sites_dict:

        number = sum([ active_sites_dict[name][tag]
                        for tag in active_sites_dict[name]])

        active_sites_dict_reduced[name] = number

    return active_sites_dict_reduced

################################################################################
# PLOT SITES DISTRIBUTION
################################################################################

def plot_sites_distribution(active_sites_dict):

    alpha = 1.
    width = 0.8

    tick_size = 14
    label_size = 16

    ylabel = 'number of sites'

    plt.ylim([0, 300])

    for name in active_sites_dict:

        plt.bar(name, active_sites_dict[name], width = width, alpha = alpha, 
                edgecolor = 'k')

    plt.yticks(fontsize = tick_size)
    plt.xticks(fontsize = tick_size)
    
    plt.ylabel(ylabel, fontsize = label_size)
    plt.xlabel('active sites', fontsize = label_size)

    plt.show()

################################################################################
# ACTIVE SITES TO ASE
################################################################################

def active_sites_to_ase(active_sites, vacuum = 10.):

    atoms = Atoms()

    for a in [ a for a in active_sites if a.deleted is False ]:

        atoms += Atom(dummy_species_dict[a.name], a.position)

    atoms.center(vacuum = vacuum/2.)

    return atoms

################################################################################
# PLOT KMC GRID
################################################################################

def plot_active_sites_grid(active_sites,
                           plot_type        = '3D' ,
                           specify_n_coord  = True ,
                           specify_supp_int = False,
                           specify_facets   = False,
                           half_plot        = False):

    from mayavi import mlab

    active_sites_dict = count_active_sites(active_sites = active_sites,
                                           with_tags    = True        )

    heigths = [a.position[2] for a in active_sites if a.deleted is False]

    z_half = (max(heigths)+min(heigths))/2.

    color_dict_new = {}

    for name in all_sites_names:
    
        color_dict_new[name] = {}
    
        for tag in active_sites_dict[name]:
    
            color = colors_dict[name].copy()
    
            n_support = tag.split('S')[1:]
            
            tag_new = tag.split('_')[0]
    
            if specify_n_coord is True:
        
                n_coord = [ int(i) for i in tag_new.split(',') ]
                n_coord_ave = sum(n_coord)/len(n_coord)
        
                color += [ -0.1*(n_coord_ave-6) for j in range(3) ]
                    
                for k in range(3):
                    if color[k] > 1.:
                        color[k] = 1.
                    elif color[k] < 0.:
                        color[k] = 0.
        
            elif specify_facets is True:
        
                facets = [ str(f) for f in tag_new.split(',') ]
        
                color = black.copy()
        
                for facet in facets:
                    for i in range(3):
                        color[i] += facet_colors_dict[facet][i]/len(facets)
        
            if specify_supp_int is True and len(n_support) > 0:
                for i in range(int(n_support[0])):
                    color = (2*color+grey)/3.
    
            color_dict_new[name][tag] = color

    figure = mlab.figure(bgcolor = tuple(white), fgcolor = tuple(black))

    mlab.clf()

    z = 0.
    z_new = 0.

    for name in all_sites_names:

        for tag in active_sites_dict[name]:

            num_point = 0
    
            x_point = []
            y_point = []
            z_point = []
            x_link  = []
            y_link  = []
            z_link  = []
    
            connections = []
    
            for a in [ a for a in active_sites if a.deleted is False 
                    and a.name == name and a.tag == tag ]:
    
                color = tuple(color_dict_new[name][tag])
    
                if plot_type == '3D':
                    x, y, z = a.position
                else:
                    x, y = a.position_xy
    
                if half_plot is True and z < z_half-1e-3:
                    continue
    
                x_point.append(x)
                y_point.append(y)
                z_point.append(z)
                x_link.append(x)
                y_link.append(y)
                z_link.append(z)
                num_link = num_point
    
                for n in a.neighbors:
    
                    num_link += 1
    
                    if plot_type == '3D':
                        x_new, y_new, z_new = active_sites[n].position
                    else:
                        x_new, y_new = active_sites[n].position_xy
    
                    x_link.append((x+x_new)/2)
                    y_link.append((y+y_new)/2)
                    z_link.append((z+z_new)/2)
    
                    connections.append([num_point, num_link])
    
                num_point = num_link + 1
    
            size = sizes_dict[name]
    
            mlab.points3d(np.array(x_point),
                        np.array(y_point),
                        np.array(z_point), 
                        color            = color,
                        scale_factor     = size )
    
            pts = mlab.points3d(np.array(x_link),
                                np.array(y_link),
                                np.array(z_link), 
                                color           = color,
                                scale_factor    = 0.1  )
    
            pts.mlab_source.dataset.lines = np.array(connections)
            tube = mlab.pipeline.tube(pts, tube_radius = 0.08)
            mlab.pipeline.surface(tube, color = color)

    mlab.gcf().scene.parallel_projection = True

    mlab.show()

################################################################################
# END
################################################################################
