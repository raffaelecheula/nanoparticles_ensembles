################################################################################
# Raffaele Cheula, LCCP, Politecnico di Milano, raffaele.cheula@polimi.it
################################################################################

from __future__ import absolute_import, division, print_function
import numpy as np
import copy as cp
import matplotlib.pyplot as plt
from collections import OrderedDict
from operator import attrgetter
from ase import Atom
from ase.build.tools import rotation_matrix
from nanoparticle_utils import get_interact_len, get_neighbor_atoms

################################################################################
# COLORS
################################################################################

black  = np.array([  0/255,   0/255,   0/255])
white  = np.array([255/255, 255/255, 255/255])
red    = np.array([255/255,  51/255,  51/255])
blue   = np.array([ 51/255,  51/255, 255/255])
yellow = np.array([255/255, 255/255,  51/255])
orange = np.array([255/255, 165/255,  51/255])
green  = np.array([  0/255, 204/255,  51/255])
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
               'lho': purple,
               'bho': brown ,
               'tho': forest}

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

################################################################################
# SURFACE ATOM CLASS
################################################################################

class SurfaceAtom():

    def __init__(self,
                 symbol       = None ,
                 position     = None ,
                 neighbors    = None ,
                 neigh_zero   = None ,
                 n_coord      = None ,
                 supp_contact = False,
                 index        = None ,
                 index_zero   = None ):

        if neighbors is None:
            neighbors = []

        if neigh_zero is None:
            neigh_zero = []

        if position is None:
            position = np.zeros(3)

        self.position     = position
        self.neigh_zero   = neigh_zero
        self.neighbors    = neighbors
        self.n_coord      = n_coord
        self.supp_contact = supp_contact
        self.index        = index
        self.index_zero   = index_zero

################################################################################
# ACTIVE SITE CLASS
################################################################################

class ActiveSite():

    def __init__(self, 
                 name         = None ,
                 tag          = None ,
                 position     = None ,
                 position_xy  = None ,
                 neighbors    = None ,
                 n_coord      = None ,
                 index        = None ,
                 supp_contact = None ,
                 facet_type   = None ,
                 deleted      = False):

        if name is None:
            name = ''

        if tag is None:
            tag = ''

        if position is None:
            position = np.zeros(3)
        
        if position_xy is None:
            position_xy = np.zeros(2)

        if neighbors is None:
            neighbors = []

        if facet_type is None:
            facet_type = []

        if supp_contact is None:
            supp_contact = 0

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

def get_surface_shell_old(atoms, elements, interact_len, n_coord_max,
                         support_symbol = 'Es', epsi = 1e-4):

    heigths = [ a.position[2] for a in atoms if a.symbol == support_symbol ]
    
    if heigths:
        heigth = max(heigths)
    
    del atoms [[ a.index for a in atoms 
                 if a.symbol == support_symbol
                 if a.position[2] < heigth-epsi ]]

    surface = []
    internal_index = []
    index = 0

    for a in atoms:

        supp_contact = False

        neigh_zero = [ b.index for b in atoms 
                       if b.symbol in elements
                       if b.index != a.index 
                       if np.linalg.norm(a.position-b.position) < 
                          interact_len+epsi ]

        n_coord = len(neigh_zero)

        neigh_support = [ b.index for b in atoms 
                          if b.symbol == support_symbol 
                          if b.index != a.index 
                          if np.linalg.norm(a.position-b.position) < 
                             interact_len+epsi ]

        neigh_support = [ bi for bi in neigh_zero 
                          if atoms[bi] == support_symbol ] 

        n_supp = len(neigh_support)

        if n_supp > 0:
            supp_contact = True

        if n_coord >= n_coord_max:
            internal_index += [a.index]

        elif n_coord+n_supp < n_coord_max and a.symbol in elements:

            surface += [SurfaceAtom(neigh_zero   = neigh_zero  ,
                                    symbol       = a.symbol    ,
                                    position     = a.position  ,
                                    n_coord      = n_coord     ,
                                    supp_contact = supp_contact,
                                    index        = index       ,
                                    index_zero   = a.index     )]

            index += 1

    for a in surface:

        a.neighbors = [ b.index for b in surface 
                        if b.index != a.index
                        if np.linalg.norm(a.position-b.position) < 
                           interact_len + epsi ]

    return surface

################################################################################
# GET SURFACE SHELL
################################################################################

def get_surface_shell(element, positions, neighbors, indices, n_coord,
                      supp_contact, n_coord_max):

    indices_zero = np.copy(indices)

    del_indices = []

    for i in range(len(n_coord)):
        if n_coord[i] >= n_coord_max:
            del_indices += [i]

    indices   = np.delete(indices, del_indices, axis = 0)

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
        
            neigh_zero_new = []
            for j in [j for j in neighbors[i] if j in indices_zero_dict]:
                neigh_zero_new += [indices_zero_dict[j]]
        
            neighbors_new = []
            for j in [j for j in neighbors[i] if j in indices_dict]:
                neighbors_new += [indices_dict[j]]
        
            surface += [SurfaceAtom(neighbors    = neighbors_new  ,
                                    neigh_zero   = neigh_zero_new ,
                                    symbol       = element        ,
                                    position     = positions[i]   ,
                                    n_coord      = n_coord[i]     ,
                                    supp_contact = supp_contact[i],
                                    index        = index          ,
                                    index_zero   = i              )]
            
            index += 1

    return surface

################################################################################
# GET FCC ACTIVE SHELL
################################################################################

def get_fcc_active_shell(surface          ,
                         specify_n_coord  = []   ,
                         specify_supp_int = True ,
                         specify_facets   = True ,
                         check_duplicates = False,
                         multiple_facets  = False):

    active_sites = []
    index = 0

    """
    Identification of top (top) and long-hollow (lho) active sites. 
    """

    for a in surface:

        if a.n_coord == 11:

            indices_zero = [ bi for bi in a.neigh_zero 
                             if bi in [ c.index_zero for c in surface ] ]

            indices = [ bi for bi in [ b.index for b in surface
                                       if b.index_zero in indices_zero ]
                        if len([ ci for ci in a.neigh_zero
                                if ci in surface[bi].neigh_zero ]) == 3 ]

            position  = np.zeros(3)
            neighbors = []
            n_coord   = []
            
            for bi in indices:

                b = surface[bi]

                position += b.position/(len(indices))
                neighbors += [b.index]
                n_coord += [b.n_coord]

            facet_type = []
            if len(indices) == 5:
                facet_type += ['dec']
                position = a.position

            active_sites += [ActiveSite(name       = 'lho'     ,
                                        position   = position  ,
                                        neighbors  = neighbors ,
                                        n_coord    = n_coord   ,
                                        index      = index     ,
                                        facet_type = facet_type)]

            index += 1

        else:

            position  = a.position
            neighbors = []
            n_coord   = [a.n_coord]

            if specify_supp_int is True:
                supp_contact = 0 if a.supp_contact is True else 1
            else:
                supp_contact = 0

            active_sites += [ActiveSite(name         = 'top'       ,
                                        position     = position    ,
                                        neighbors    = neighbors   ,
                                        n_coord      = n_coord     ,
                                        index        = index       ,
                                        supp_contact = supp_contact)]

            index += 1

    """
    Identification of dechaedron-hollow (dec), long-bridge (lbr), hollow (hol)
    bridge (brg), fcc (fcc) and hcp (hcp) active sites. 
    """

    for a in surface:

        if a.n_coord == 11:

            a = active_sites[a.index]

            if 'dec' not in a.facet_type:

                for bi, ci in [ (bi, ci) for bi in a.neighbors[:4]
                                for ci in a.neighbors[:4]
                                if bi > ci
                                if bi not in surface[ci].neighbors
                                if len([ di for di in surface[bi].neighbors
                                      if di in surface[ci].neighbors ]) == 2 ]:
    
                    b = surface[bi]
                    c = surface[ci]
                    
                    di = [ di for di in active_sites[bi].neighbors
                           if di in active_sites[ci].neighbors
                           if active_sites[di].name == 'lbr' ]
                    
                    if di:
                        
                        d = active_sites[di[0]]
                        d.neighbors += [a.index]
                        a.neighbors += [d.index]
                        
                        for i in (b.index, c.index):
                            active_sites[i].neighbors += [d.index]
                            active_sites[i].neighbors += [a.index]
                        
                    else:
                    
                        position  = (b.position+c.position)/2.
                        neighbors = [b.index, c.index, a.index]
                        n_coord   = [b.n_coord, c.n_coord]
                        
                        active_sites += [ActiveSite(name      = 'lbr'    ,
                                                    position  = position ,
                                                    neighbors = neighbors,
                                                    n_coord   = n_coord  ,
                                                    index     = index    )]
            
                        a.neighbors += [index]
                        
                        for i in (b.index, c.index):
                            active_sites[i].neighbors += [index]
                            active_sites[i].neighbors += [a.index]
                        
                        index += 1

            else:
                
                for bi in a.neighbors:
                    active_sites[bi].neighbors += [a.index]

        else:

            for bi in [ bi for bi in a.neighbors
                        if bi > a.index 
                        if surface[bi].n_coord < 11 ]:

                b = surface[bi]

                for di in [ di for di in a.neighbors
                            if di > b.index 
                            if di not in b.neighbors 
                            if surface[di].n_coord < 11 ]:

                    d = surface[di]

                    for ei in [ ei for ei in b.neighbors
                               if ei in d.neighbors 
                               if ei > b.index 
                               if ei not in a.neighbors 
                               if surface[ei].n_coord < 11 ]:

                        e = surface[ei]

                        if not [ fi for fi in a.neighbors
                                 if fi in b.neighbors 
                                 if fi in d.neighbors 
                                 if fi in e.neighbors ]:

                            position  = (a.position+b.position + 
                                         d.position+e.position)/4.
                            neighbors = [a.index, b.index, d.index, e.index]
                            n_coord   = [a.n_coord, b.n_coord,
                                         d.n_coord, e.n_coord]

                            active_sites += [ActiveSite(name      = 'hol'    ,
                                                        position  = position ,
                                                        neighbors = neighbors, 
                                                        n_coord   = n_coord  ,
                                                        index     = index    )]

                            active_sites[index].facet_type = ['100']

                            for i in (a.index, b.index, d.index, e.index):
                            
                                active_sites[i].neighbors += [index]
                            
                            index += 1

                if not [ ci for ci in a.neighbors
                         if ci in b.neighbors 
                         if a.supp_contact == True
                         if b.supp_contact == True 
                         if surface[ci].supp_contact == True 
                         if surface[ci].n_coord == 3 ]:

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

                for ci in [ ci for ci in a.neighbors
                            if ci in b.neighbors 
                            if ci > b.index
                            if surface[ci].n_coord < 11 
                            if (a.supp_contact == False
                                or b.supp_contact == False 
                                or surface[ci].supp_contact == False)
                            if not [ di for di in a.neighbors 
                                     if di in b.neighbors
                                     if di in surface[ci].neighbors 
                                     if surface[di].n_coord == 3 ] ]:

                    c = surface[ci]

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
    Identification of theta-hollow (tho) and beta-hollow (bho) active sites. 
    """

    for a in [ a for a in active_sites if a.name == 'lbr' ]:
    
        for b in [ b for b in surface 
                   if b.n_coord < 11
                   if a.neighbors[0] in b.neighbors
                   if a.neighbors[1] in b.neighbors ]:
    
            a.neighbors += [b.index]
            active_sites[b.index].neighbors += [a.index]

    for a in [ a for a in active_sites if a.name == 'brg' ]:
    
        for b in [ b for b in active_sites 
                   if b.index != a.index
                   if a.neighbors[0] in b.neighbors
                   if a.neighbors[1] in b.neighbors ]:
    
            a.neighbors += [b.index]
            b.neighbors += [a.index]

    for a in [ a for a in active_sites if a.name == 'brg' ]:

        for bi, ci in [ (bi, ci) for bi in a.neighbors for ci in a.neighbors
                      if ((active_sites[bi].name, active_sites[ci].name)
                          in (('fcc', 'fcc'), ('hcp', 'hcp')))
                      if active_sites[bi].index != active_sites[ci].index ]:
    
            b = active_sites[bi]
            c = active_sites[ci]

            d = surface[a.neighbors[0]]
            e = surface[a.neighbors[1]]

            indices = [ gi for gi in d.neighbors
                        if gi in e.neighbors
                        if not [ li for li in surface[gi].neighbors
                                 if li in d.neighbors
                                 if li in e.neighbors
                                 if surface[li].n_coord == 3 ] ]

            gi = indices[0]
            hi = indices[1]

            g = surface[gi]
            h = surface[hi]
            
            dir_1 = b.position-a.position
            dir_2 = a.position-c.position
            
            dot_prod = np.dot(dir_1/np.linalg.norm(dir_1),
                              dir_2/np.linalg.norm(dir_2))
            
            if len([ li for li in g.neigh_zero
                     if li in h.neigh_zero]) == 2 and dot_prod < 0.8:
            
                a.name = 'tho'
                a.position = (g.position+h.position)/2.
                
                for fi in b.neighbors+c.neighbors:
                
                    a.neighbors += [fi]
                
                    for j, gi in enumerate(active_sites[fi].neighbors):
                        if gi in (b.index, c.index):
                            active_sites[fi].neighbors[j] = a.index
                
                b.delete()
                c.delete()

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
            
            gi = [ gi for gi in surface[c.neighbors[0]].neigh_zero
                   if gi in surface[c.neighbors[1]].neigh_zero
                   if gi in surface[c.neighbors[2]].neigh_zero
                   if gi in surface[c.neighbors[3]].neigh_zero ][0]
            
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
            
                #a.name = 'bho'
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

    for a in [ a for a in active_sites
               if a.n_coord == [9] ]:
    
        if not a.facet_type:
            a.facet_type = ['111']
    
        for bi in a.neighbors:
            
            b = active_sites[bi]
            
            if not b.facet_type:
                b.facet_type = ['111']

    for a in [ a for a in active_sites
               if a.n_coord == [8] ]:
    
        if not a.facet_type:
            a.facet_type = ['100']
    
        for bi in a.neighbors:
    
            b = active_sites[bi]
    
            if not b.facet_type:
                b.facet_type = ['100']

    """
    Identification of sites contacting the support. 
    """

    if specify_supp_int is True:
    
        for a in [ a for a in active_sites 
                   if a.name == 'top' 
                   if a.supp_contact == 1 ]:
    
            for bi in a.neighbors:
                active_sites[bi].supp_contact += 1

    """
    Creation of the active sites tags. 
    """

    for a in [ a for a in active_sites if a.deleted is False ]:
    
        if a.name in specify_n_coord:
    
            a.tag = '_'.join([a.name] + [ '{0:02d}'.format(n)
                                          for n in sorted(a.n_coord) ])

        else:

            a.tag = a.name
    
        if specify_facets is True:
    
            if multiple_facets is True:
        
                for facet in a.facet_type:
                    a.tag += '_('+str(facet)+')'
    
            else:
                
                try: a.tag += '_('+str(a.facet_type[0])+')'
                except: pass

    """
    Checking of duplicates active sites. 
    """

    if check_duplicates is True:

        for a in active_sites:

            for b in [ b for b in active_sites if b.index > a.index ]:

                if np.allclose(a.position, b.position) is True:

                    print('Duplicate site! {0} {1}'.format(a.name, b.name))

    return active_sites

################################################################################
# COUNT ACTIVE SITES
################################################################################

def count_active_sites(active_sites, print_distrib = False):

    active_sites_dict = OrderedDict()

    for a in [ a for a in active_sites if a.deleted is False ]:
        if a.tag not in active_sites_dict:
            active_sites_dict[a.tag] = 1
        else:
            active_sites_dict[a.tag] += 1

    active_sites_dict = OrderedDict(sorted(active_sites_dict.items()))

    if print_distrib is True:
        print('Active sites distribution:')
        for act in active_sites_dict:
            print(' {0:30s} = {1:5d}'.format(act, active_sites_dict[act]))
        print('')

    return active_sites_dict

################################################################################
# PLOT TOP DISTRIBUTION
################################################################################

def plot_top_distribution(active_sites_dict,
                          n_atoms_tot      ,
                          show_plot        = True,
                          percentual       = True):

    alpha = 1.
    width = 0.8

    tick_size = 14
    label_size = 16

    n_coords_sum = np.zeros(13)

    x_vect = np.arange(2, 12)
    
    if percentual is True:
        y_max  = 0.4*100
        ylabel = 'number of atoms [%]'
    else:
        y_max  = int(0.4*n_atoms_tot)
        ylabel = 'number of atoms'

    plt.axis([x_vect[0], x_vect[-1]+1, 0., y_max])

    for tag in [ tag for tag in active_sites_dict 
                 if tag.split('_')[0] in ('top', 'lho') ]:

        name = tag.split('_(')[0]

        if name.split('_')[0] == 'top':
            n_coord = int([ i for i in name.split('_')[1:] if i != 'S' ][0])
        else:
            n_coord = 11

        facet = [ i.split(')')[0] for i in tag.split('_(')[1:]][0]

        color = facet_colors_dict[facet]

        n_coords_scal = np.zeros(13)
        
        if percentual is True:
            n_coords_scal[n_coord] += active_sites_dict[tag]/n_atoms_tot*100
        else:
            n_coords_scal[n_coord] += active_sites_dict[tag]

        y_vect = n_coords_scal[2:12]
        b_vect = n_coords_sum[2:12]

        plt.bar(x_vect, y_vect, width, alpha = alpha, bottom = b_vect,
                color = color, edgecolor = 'k', linewidth = 0.3)

        n_coords_sum += n_coords_scal

    plt.yticks(fontsize = tick_size)
    plt.xticks(x_vect[1:], fontsize = tick_size)
    
    plt.ylabel(ylabel, fontsize = label_size)
    plt.xlabel('coordination number', fontsize = label_size)

    if show_plot is True:
        plt.show()

################################################################################
# PLOT SITES DISTRIBUTION
################################################################################

def plot_sites_distribution(active_sites_dict,
                            n_atoms_tot      ,
                            show_plot        = True,
                            percentual       = True):

    alpha = 1.
    width = 0.8

    tick_size = 14
    label_size = 16

    y_max  = int(n_atoms_tot)
    ylabel = 'number of sites'

    plt.ylim([0, 300])

    for tag in active_sites_dict:

        name = tag.split('_(')[0]

        plt.bar(tag, active_sites_dict[tag], width, alpha = alpha, 
                edgecolor = 'k')

    plt.yticks(fontsize = tick_size)
    plt.xticks(fontsize = tick_size)
    
    plt.ylabel(ylabel, fontsize = label_size)
    plt.xlabel('active sites', fontsize = label_size)

    if show_plot is True:
        plt.show()

################################################################################
# FROM ACTIVE SHELL TO ASE
################################################################################

def from_active_shell_to_ase(atoms, active_sites):

    atoms = atoms[:]

    del atoms [[ a.index for a in atoms ]]

    dummy_species_dict = {'top': 'Rh',
                          'brg': 'H' ,
                          'lbr': 'N' ,
                          'hcp': 'F' ,
                          'fcc': 'O' ,
                          'hol': 'P' ,
                          'lho': 'S' ,
                          'bho': 'B' ,
                          'tho': 'Te'}

    for a in [ a for a in active_sites if a.deleted is False ]:

        atoms += Atom(dummy_species_dict[a.name], a.position)

    return atoms

################################################################################
# PLOT KMC GRID
################################################################################

def plot_kmc_grid(active_sites, plot_type = '3D', half_plot = False,
                  facet_color = False, active_sites_dict = None):

    from mayavi import mlab
    from tvtk.api import tvtk

    import sys
    import warnings

    if not sys.warnoptions:
        warnings.simplefilter("ignore")

    if active_sites_dict is None:
        active_sites_dict = count_active_sites(active_sites  = active_sites,
                                               print_distrib = False       )

    heigths = [a.position[2] for a in active_sites if a.deleted is False]

    z_half = (max(heigths)+min(heigths))/2.

    for tag in active_sites_dict:

        if facet_color is False:

            name = tag.split('_(')[0]
    
            color = cp.deepcopy(colors_dict[name.split('_')[0]])

            support = [ i for i in name.split('_')[1:] if i == 'S' ]
            n_coord = [ i for i in name.split('_')[1:] if i != 'S' ]
    
            for i in range(len(support)):
                color = (color + grey)/2.
    
            for i in range(len(n_coord)):
                color += [ 0.1*(int(n_coord[i])-8) for j in range(3) ]
                for k in range(len(color)):
                    if color[k] > 1.:
                        color[k] = 1.
                    elif color[k] < 0.:
                        color[k] = 0.
    
            colors_dict[name] = color

        else:

            name = tag.split('_(')[0]

            facets = [i.split(')')[0] for i in tag.split('_(')[1:]]

            color = np.copy(black)

            for facet in facets:
                for i in range(3):
                    color[i] += facet_colors_dict[facet][i]/len(facets)

            colors_dict[tag] = color

    figure = mlab.figure(bgcolor = tuple(white), fgcolor = tuple(black))
    mlab.clf()

    z = 0.
    z_new = 0.

    for name in colors_dict:

        color = (0., 0., 0.)
        num_point = 0

        x_point = []
        y_point = []
        z_point = []
        x_link  = []
        y_link  = []
        z_link  = []

        connections = []

        for a in [ a for a in active_sites if a.deleted is False
                   if ((facet_color is False and a.tag.split('_(')[0] == name)
                        or (facet_color is True and a.tag == name)) ]:

            color = colors_dict[name]
            color = tuple(color)

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

        size = sizes_dict[name.split('_')[0]]
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
        tube = mlab.pipeline.tube(pts, tube_radius = 0.07 + 3e-2*size)
        mlab.pipeline.surface(tube, color = color)

    mlab.gcf().scene.parallel_projection = True

    mlab.show()

################################################################################
# GET SITES DISTRIBUTION
################################################################################

def get_sites_distribution(animation           ,
                           interact_len        ,
                           n_coord_max         ,
                           vacuum_symbol       = 'X'     , 
                           link_len            = None    ,
                           specify_site_coord  = ('top',),
                           specify_support_int = True    ,
                           epsi                = 1e-4    ):

    active_sites_vects = {'top': [],
                          'brg': [],
                          'lbr': [],
                          'hcp': [],
                          'fcc': [],
                          'hol': [],
                          'lho': []}

    elements = get_atom_list(animation[0])

    for atoms in animation:

        del atoms [[ a.index for a in atoms if a.symbol is vacuum_symbol ]]

        surface = get_surface_shell(atoms        = atoms       ,
                                    elements     = elements    ,
                                    interact_len = interact_len,
                                    n_coord_max  = n_coord_max )

        active_sites = get_fcc_active_shell(atoms       = atoms              , 
                                    surface             = surface            ,
                                    interact_len        = interact_len       ,
                                    link_len            = link_len           ,
                                    specify_site_coord  = specify_site_coord ,
                                    specify_support_int = specify_support_int,
                                    epsi                = epsi               )

        active_sites_dict = count_active_sites(active_sites)

        for name in active_sites_vects:
            active_sites_vects[name] += [active_sites_dict[name]]

    return active_sites_vects

################################################################################
# PLOT ACTIVE SITES DISTRIBUTION
################################################################################

def plot_active_sites_distribution(active_sites_vects, x_axis):

    import matplotlib.pyplot as plt

    for name in active_sites_vects:
        plt.plot(x_axis, active_sites_vects[name], 
                 label = str(name))

    plt.legend()
    plt.show()

################################################################################
# END
################################################################################
