#!/usr/bin/env python3

################################################################################
# Raffaele Cheula, LCCP, Politecnico di Milano, raffaele.cheula@polimi.it
################################################################################

from __future__ import absolute_import, division, print_function
import os, io, timeit
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import curve_fit
from nanoparticle_units import *
from nanoparticle_utils import e_relax_from_bond_ols, cluster_add_adsorbates
from nanoparticle_active_sites import all_sites_names, remove_tags_active_sites

################################################################################
# MEASURE TIME START
################################################################################

measure_time = True

if measure_time is True:
    time_start = timeit.default_timer()

################################################################################
# CALCULATION DATA
################################################################################

hkl = 1 # 1 | 3

if hkl == 3:
    dirmother = 'Ni_08_hkl3_sites'
elif hkl == 1:
    dirmother = 'Ni_10_hkl1_sites'

bulk_types = []
bulk_types += ['fcc']
bulk_types += ['dec']
bulk_types += ['ico']

if hkl == 3:

    n_min_read =    20
    n_max_read =  5000
    
    a_diff_n   =  1e-3
    step_anal  =    10
    smooth     =     1
    mu_shift   =  -0.5

elif hkl == 1:

    n_min_read =    20
    n_max_read = 30000
    
    a_diff_n   =  1e-3
    step_anal  =   100
    smooth     =     1
    mu_shift   =  -0.5

n_max_per_group      = None
calc_energy_clean    = True
count_active_sites   = False
calc_energy_with_ads = False # True
check_e_thr          = False
fit_e_form_min       = True
plot_e_form_n_atoms  = True
analyze_data         = True # False
show_plots           = True
save_plots           = False

select_particle      = False
selected_particle    = ('fcc', 0)

################################################################################
# FIGURES DATA
################################################################################

tick_size  =   14
label_size =   16

color_dict = {'fcc': 'darkorange' ,
              'dec': 'forestgreen',
              'ico': 'darkviolet' }

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

energy_model     = 'SRB'

e_relax_list = np.zeros(13)

for i in range(13):
    e_relax_list[i] = a_model_relax*(12-i)**b_model_relax

################################################################################
# OPERATIVE CONDITIONS
################################################################################

temperature = Celsius_to_Kelvin(400.) # [K]
pressure    =  1.00 * atm

x_CO  = 0.80
x_CO2 = 0.00
x_H2  = 0.00
x_N2  = 1.-x_CO-x_CO2-x_H2

print(f'temperature = {temperature:9.4f} K')
print(f'pressure    = {pressure/atm:9.4f} atm')

################################################################################
# ADSORBATES DATA
################################################################################

from phases import gas

gas.TPX = temperature, pressure, [x_CO, x_CO2, x_H2, x_N2]

dmu_CO  = gas['CO'].chemical_potentials[0] * (J/eV)/(kmol/molecule) # [eV]
dmu_CO2 = gas['CO2'].chemical_potentials[0] * (J/eV)/(kmol/molecule) # [eV]
dmu_H2  = gas['H2'].chemical_potentials[0] * (J/eV)/(kmol/molecule) # [eV]

print('deltamu CO  = {:9.4f} eV'.format(dmu_CO))
print('deltamu CO2 = {:9.4f} eV'.format(dmu_CO2))
print('deltamu H2  = {:9.4f} eV'.format(dmu_H2))

bond_length    = +1.20 # [Ang]
#bond_length    = +0.00 # [Ang]
sites_equilib  = False
single_cov_fun = True

alpha_cov      = +3.0340E+02 # [eV/Ang^2]
beta_cov       = +3.3144E+00 # [-]

g_bind_dict = {}

def g_bind_fun_top(cn_ave): return (-0.7895-1.3579-dmu_CO)+(0.00424)*cn_ave
def g_bind_fun_brg(cn_ave): return (-1.8262-1.3579-dmu_CO)+(0.13375)*cn_ave
def g_bind_fun_hcp(cn_ave): return (-1.9094-1.3579-dmu_CO)+(0.13900)*cn_ave
def g_bind_fun_fcc(cn_ave): return (-1.9094-1.3579-dmu_CO)+(0.13900)*cn_ave
def g_bind_fun_hol(cn_ave): return (-2.1783-1.3579-dmu_CO)+(0.17500)*cn_ave
def g_bind_fun_lbr(cn_ave): return (-0.1000-1.3579-dmu_CO)+(0.00000)*cn_ave

#def g_bind_fun_lbr(cn_ave): return (-2.9287-dmu_CO2-0.5*dmu_H2)+(0.298)*cn_ave

g_bind_dict['top'] = g_bind_fun_top
g_bind_dict['brg'] = g_bind_fun_brg
g_bind_dict['hcp'] = g_bind_fun_hcp
g_bind_dict['fcc'] = g_bind_fun_fcc
g_bind_dict['hol'] = g_bind_fun_hol
g_bind_dict['lbr'] = g_bind_fun_lbr

n_bonds_dict = {}

n_bonds_dict['top'] = 1
n_bonds_dict['brg'] = 2
n_bonds_dict['hcp'] = 3
n_bonds_dict['fcc'] = 3
n_bonds_dict['hol'] = 4
n_bonds_dict['lbr'] = 2

################################################################################
# READ PARTICLES
################################################################################

step = 10

n_particles_tot   = 0
n_particles_dict  = {}
n_atoms_list      = []
n_atoms_surf_list = []
n_coord_ave_list  = []
e_form_list       = []
e_spec_list       = []
site_dist_list    = []
bulk_type_list    = []
n_atoms_min_list  = []
e_form_min_list   = []
e_spec_min_list   = []
diameter_list     = []
coverage_list     = []
area_surf_list    = []

for bulk_type in bulk_types:

    n_particles_dict[bulk_type] = 0
    
    n_groups = int((n_max_read-n_min_read)/step)

    print(f'\nAnalyzing {bulk_type} particles')

    for ng, group in enumerate(range(n_min_read, n_max_read, step)):

        print(f'\nGroup {ng+1}/{n_groups}: {group}-{group+step}')

        filename = f'{bulk_type}_{group:04d}_{group+step:04d}.pkl'
        
        filepath = os.path.join(dirmother, bulk_type, filename)

        with open(filepath, 'rb') as fileobj:
            particles = pickle.load(fileobj)

        for i_p, particle in enumerate(particles):

            if (i_p+1) % 10 == 0 or (i_p+1) == len(particles):
                print(f'Particle {i_p+1}/{len(particles)}', end = '\r')

            particle.set_lattice_constant(lc_new = lattice_constant,
                                          lc_old = lattice_constant)

            if calc_energy_clean is True:

                particle.e_coh_bulk   = e_coh_bulk
                particle.e_relax_list = e_relax_list

                if bulk_type == 'dec':
                    particle.e_twin        = e_twin
                    particle.shear_modulus = shear_modulus
                    particle.k_strain      = k_strain_dec
                
                    particle.n_twin = 0
                
                elif bulk_type == 'ico':
                    particle.e_twin        = e_twin
                    particle.shear_modulus = shear_modulus
                    particle.k_strain      = k_strain_ico

                    particle.n_twin = 0

                particle.get_energy_clean(energy_model = energy_model)

            if count_active_sites is True:
                
                particle.get_active_sites(specify_n_coord  = True ,
                                          specify_supp_int = False,
                                          specify_facets   = False,
                                          check_duplicates = False,
                                          multiple_facets  = False,
                                          convex_sites     = True )
                
                particle.get_active_sites_dict(with_tags = True)

            if calc_energy_with_ads is True:
                
                if 'active_sites' not in dir(particle):
                    particle.get_active_sites_from_dict(with_tags = True)

                particle.get_energy_with_ads(g_bind_dict    = g_bind_dict   ,
                                             n_bonds_dict   = n_bonds_dict  ,
                                             alpha_cov      = alpha_cov     ,
                                             beta_cov       = beta_cov      ,
                                             bond_length    = bond_length   ,
                                             temperature    = temperature   ,
                                             sites_equilib  = sites_equilib ,
                                             single_cov_fun = single_cov_fun)

        if check_e_thr is True:

            for i_p in reversed(range(len(particles))):

                n_atoms = particles[i_p].n_atoms
                e_spec  = particles[i_p].e_spec

                e_thr = -e_coh_bulk*a_thr*n_atoms**(-1./3.)-b_thr

                if e_spec > e_thr:
                    del particles[i_p]

        particles = sorted(particles, key = lambda x: x.e_spec)

        if n_max_per_group is not None:

            particles = particles[:n_max_per_group]

        for particle in particles:
            
            if select_particle is True:
                if (bulk_type, n_particles_tot) == selected_particle:
                    particle_sel = particle
            
            n_particles_tot += 1
            n_particles_dict[bulk_type] += 1
            
            bulk_type_list    += [bulk_type]
            n_atoms_list      += [particle.n_atoms]
            n_atoms_surf_list += [particle.get_n_atoms_surf()]
            n_coord_ave_list  += [particle.get_n_coord_ave()]
            e_form_list       += [particle.e_form]
            e_spec_list       += [particle.e_spec]
            diameter_list     += [particle.get_diameter()]
            
            if calc_energy_with_ads is True:
                coverage_list     += [particle.coverage]
                area_surf_list    += [particle.area_surf]
            
            if 'active_sites_dict' in dir(particle):
                site_dist = remove_tags_active_sites(particle.active_sites_dict)
                site_dist_list += [site_dist]

        if len(particles) > 0:
            n_atoms_min_list += [particles[0].n_atoms]
            e_form_min_list  += [particles[0].e_form]
            e_spec_min_list  += [particles[0].e_spec]

        print('')

    print(f'\nNumber of {bulk_type} particles: {n_particles_dict[bulk_type]}')

print(f'\nNumber of tot particles: {n_particles_tot}')

################################################################################
# FIT FORMATION ENERGY MIN
################################################################################

if fit_e_form_min is True:

    n_atoms_red_list = []
    e_form_red_list  = []
    e_spec_red_list  = []

    step_group = 10

    index = -1

    for i_m in range(len(n_atoms_min_list)):
        
        index_new = int(np.floor(n_atoms_min_list[i_m]/step_group))
        
        if index_new > index:
   
            n_atoms_red_list += [n_atoms_min_list[i_m]]
            e_form_red_list  += [e_form_min_list[i_m]]
            e_spec_red_list  += [e_spec_min_list[i_m]]
   
            index = index_new
   
        if index_new == index and e_spec_min_list[i_m] < e_spec_red_list[-1]:
            
            n_atoms_red_list[-1] = n_atoms_min_list[i_m]
            e_form_red_list[-1]  = e_form_min_list[i_m]
            e_spec_red_list[-1]  = e_spec_min_list[i_m]
   
    def e_form_funct(n_atoms, a_model, b_model):
        return a_model+b_model*n_atoms**(2/3)
   
    popt, pcov = curve_fit(f     = e_form_funct    ,
                           xdata = n_atoms_red_list,
                           ydata = e_form_red_list ,
                           p0    = [0., 1.]        )
   
    a_model, b_model = popt
   
    for i_r in range(len(n_atoms_red_list)):
   
        e_form_red_list[i_r] = e_form_funct(n_atoms = n_atoms_red_list[i_r],
                                            a_model = a_model              ,
                                            b_model = b_model              )
   
        e_spec_red_list[i_r] = e_form_red_list[i_r]/n_atoms_red_list[i_r]

################################################################################
# BOLTZMANN DISTRIBUTION
################################################################################

class BoltzmannDistribution:
    
    def __init__(self,
                 n_particles_tot = [] ,
                 e_form_diff_max = 10.,
                 smooth          =  1.):
    
        self.n_particles_tot = n_particles_tot
        self.e_form_diff_max = e_form_diff_max
        self.smooth          = smooth

    ########################################################################
    # BOLTZMANN DISTRIBUTION
    ########################################################################

    def get_distribution(self, mu_res, a_diff_n):

        freq_vect      = np.zeros(self.n_particles_tot)
        e_form_mu_vect = np.zeros(self.n_particles_tot)
    
        denom = 0.
        
        for i_p in range(self.n_particles_tot):
        
            e_form_mu_vect[i_p] = e_form_list[i_p]-n_atoms_list[i_p]*mu_res
            
            # To obtain a Gaussian distribution centered in n_atoms_tar
            e_form_mu_vect[i_p] += (a_diff_n * 
                                    (n_atoms_list[i_p]-self.n_atoms_tar)**2)
        
        min_e_form_mu_vect = np.min(e_form_mu_vect)
        
        for i_p in range(self.n_particles_tot):
        
            e_form_tmp = (e_form_mu_vect[i_p]-min_e_form_mu_vect)/self.smooth
        
            if e_form_tmp < self.e_form_diff_max:
                freq_vect[i_p] = np.exp(-e_form_tmp/(kB/eV*temperature))
            else:
                freq_vect[i_p] = 0.
        
            denom += freq_vect[i_p]
        
        for i_p in range(self.n_particles_tot):
            
            freq_vect[i_p] /= denom
        
        n_atoms_ave = 0.
    
        for i_p in range(self.n_particles_tot):
            
            n_atoms_ave += n_atoms_list[i_p]*freq_vect[i_p]
        
        variance = 0.
        
        for i_p in range(self.n_particles_tot):
        
            variance += ((n_atoms_list[i_p]-n_atoms_ave)**2)*freq_vect[i_p]
        
        self.freq_vect      = freq_vect
        self.e_form_mu_vect = e_form_mu_vect
        self.n_atoms_ave    = n_atoms_ave
        self.variance       = variance
        
        return freq_vect, e_form_mu_vect, n_atoms_ave, variance

    ########################################################################
    # GET RESIDUALS
    ########################################################################

    def _get_residuals(self, x, n_atoms_ave, variance):

        self.get_distribution(mu_res = x[0], a_diff_n = x[1])

        res = (self.n_atoms_ave-n_atoms_ave)**2
        res += (self.variance-variance)**2

        return res

    ########################################################################
    # GET OPTIMIZED PARAMETERS
    ########################################################################

    def get_optimized_parameters(self, n_atoms_ave, variance):

        from scipy.optimize import minimize, root, Bounds, basinhopping
        
        x0   = [0.5, 1e-4]
        args = (n_atoms_ave, variance)
        
        self.n_atoms_tar = n_atoms_ave
        
        minimizer = 'minimize' # minimize | basinhopping
        
        if minimizer == 'minimize':
        
            method = 'COBYLA' # L-BFGS-B | COBYLA
            
            options = {'rhobeg'  : 1e+01,
                       'tol'     : 1e-06,
                       'disp'    : True ,
                       'maxiter' : 1e+02,
                       'catol'   : 1e-06}
            
            sol = minimize(fun     = self._get_residuals,
                           x0      = x0                 ,
                           args    = args               ,
                           method  = method             ,
                           options = options            )
    
        elif minimizer == 'basinhopping':
    
            minimizer_kwargs = {'args': args}
        
            sol = basinhopping(func             = self._get_residuals,
                               x0               = x0                 ,
                               minimizer_kwargs = minimizer_kwargs   ,
                               niter            = 200                )
    
        print(sol)
        print('')
        print(f'n_atoms_ave = {bd.n_atoms_ave:10.4f}')
        print(f'variance    = {bd.variance:10.4f}')
        print('')
    
        mu_res, a_diff_n = sol.x
        
        self.mu_res   = mu_res
        self.a_diff_n = a_diff_n
        
        return mu_res, a_diff_n

################################################################################
# FORMATION ENERGY PLOT
################################################################################

if plot_e_form_n_atoms is True:

    fig = plt.figure(1)
    fig.set_size_inches(16, 10)
    
    #n_atoms_ave = 4300
    #
    ##bd = BoltzmannDistribution(n_particles_tot = n_particles_tot,
    ##                           smooth          = smooth         )
    ##
    ##mu_res, a_diff_n = bd.get_optimized_parameters(n_atoms_ave = n_atoms_ave,
    ##                                               variance    = variance   )
    ##
    ##bd.get_distribution(mu_res = mu_res, a_diff_n = a_diff_n)
    #
    #bd = BoltzmannDistribution(n_particles_tot = n_particles_tot,
    #                           smooth          = smooth         )
    #
    #bd.n_atoms_tar = n_atoms_ave
    #
    #e_form_res = e_form_funct(n_atoms = n_atoms_ave,
    #                          a_model = a_model    ,
    #                          b_model = b_model    )
    #
    #mu_res = e_form_res/n_atoms_ave
    #
    #bd.get_distribution(mu_res = mu_res, a_diff_n = a_diff_n)
    #
    #e_form_mu_vect = bd.e_form_mu_vect
    #freq_vect      = bd.freq_vect
    
    """
    
    for i_p in range(n_particles_tot):

        color = color_dict[bulk_type]
    
        #if site_dist_list[i_p]['lho'] > 0.:
        #    color = 'green'
        #
        #if site_dist_list[i_p]['lho'] > 0.1*site_dist_list[i_p]['top']:
        #    color = 'blue'
    
        y = e_form_list[i_p] # e_form_mu_vect[i_p]
    
        alpha = 0.6
    
        plt.plot(n_atoms_list[i_p], y,
                 marker     = 'o'         ,
                 alpha      = alpha       ,
                 linestyle  = ' '         ,
                 markersize = 4           , 
                 color      = color       )

    plt.plot(n_atoms_red_list, e_form_red_list,
             linestyle  = '--'   ,
             color      = 'black')

    plot_n_min = n_min_read
    plot_n_max = n_max_read
    plot_e_min =     0.
    plot_e_max = +5000.
    
    plt.axis([plot_n_min, plot_n_max, plot_e_min, plot_e_max])
    
    plt.yticks(fontsize = tick_size)
    plt.xticks(fontsize = tick_size)
    
    plt.ylabel('formation energy [eV]', fontsize = label_size)
    plt.xlabel('number of atoms', fontsize = label_size)
    
    if save_plots is True:
        plt.savefig('e_form_vs_n_atoms.png')

    """

    for bulk_type in reversed(bulk_types):
    
        n_atoms_vect = np.zeros(n_particles_dict[bulk_type])
        e_spec_vect  = np.zeros(n_particles_dict[bulk_type])
    
        j_p = 0
    
        for i_p in range(n_particles_tot):
        
            if bulk_type == bulk_type_list[i_p]:
        
                n_atoms_vect[j_p] = n_atoms_list[i_p]
                e_spec_vect[j_p]  = e_spec_list[i_p]
    
                j_p += 1
    
        color = color_dict[bulk_type]
    
        plt.plot(n_atoms_vect, e_spec_vect,
                 marker     = 'o'         ,
                 alpha      = 0.6         ,
                 linestyle  = ' '         ,
                 markersize = 8           , 
                 color      = color       )

    plt.plot(n_atoms_red_list, e_spec_red_list,
             linestyle  = '--'   ,
             color      = 'black')

    plot_n_min = n_min_read
    plot_n_max = n_max_read
    plot_e_min =     0.
    plot_e_max =     2.
    
    plt.axis([plot_n_min, plot_n_max, plot_e_min, plot_e_max])
    
    plt.yticks(fontsize = tick_size)
    plt.xticks(fontsize = tick_size)
    
    plt.ylabel('formation energy [eV/atom]', fontsize = label_size)
    plt.xlabel('number of atoms', fontsize = label_size)
    
    if save_plots is True:
        plt.savefig('e_form_vs_n_atoms.png')

################################################################################
# ANALYZE DATA
################################################################################

if analyze_data is True:

    plot_sites_dist  = True
    plot_n_coord_ave = False
    plot_dispersion  = False
    
    x_axis_diameter  = True

    filter_data      = False

    print('\n ANALYZE DATA \n')

    n_points = int(np.floor((n_max_read-n_min_read)/step_anal))

    n_atoms_ave_ens_vect = np.zeros(n_points)
    n_coord_ave_ens_vect = np.zeros(n_points)
    dispersion_ens_vect  = np.zeros(n_points)
    diameter_ens_vect    = np.zeros(n_points)

    sites_dist_ens = {}
    
    for site in all_sites_names:
        sites_dist_ens[site] = np.zeros(n_points)

    for i_n in range(n_points):
    
        n_atoms_ave = n_min_read+(i_n+1)*step_anal
    
        bd = BoltzmannDistribution(n_particles_tot = n_particles_tot,
                                   smooth          = smooth         )
    
        bd.n_atoms_tar = n_atoms_ave
        
        e_form_res = e_form_funct(n_atoms = n_atoms_ave,
                                  a_model = a_model    ,
                                  b_model = b_model    )
    
        mu_res = e_form_res/n_atoms_ave
        
        mu_res += mu_shift
        
        bd.get_distribution(mu_res = mu_res, a_diff_n = a_diff_n)
        
        e_form_mu_vect = bd.e_form_mu_vect
        freq_vect      = bd.freq_vect
        
        j_p = np.argmax(freq_vect)
        
        n_atoms_ave_ens_vect[i_n] = bd.n_atoms_ave
        
        n_coord_ave_ens_vect[i_n] = 0.
        dispersion_ens_vect[i_n]  = 0.
        diameter_ens_vect[i_n]    = 0.
        
        for i_p in range(n_particles_tot):
        
            n_coord_ave_ens_vect[i_n] += n_coord_ave_list[i_p]*freq_vect[i_p]
        
            dispersion_ens_vect[i_n] += (n_atoms_surf_list[i_p] / 
                                         n_atoms_list[i_p])*freq_vect[i_p]
        
            diameter_ens_vect[i_n] += diameter_list[i_p]*freq_vect[i_p]
        
            for site in all_sites_names:
                
                sites_dist_ens[site][i_n] += ( site_dist_list[i_p][site]
                                                / n_atoms_surf_list[i_p]
                                                * freq_vect[i_p] )
        
        print(f'i gss       = {j_p:7d}')
        print(f'n atoms ave = {n_atoms_ave:7d}')
        print(f'n atoms gss = {n_atoms_list[j_p]:7d}')
        print(f'n coord     = {n_coord_ave_ens_vect[i_n]:7.4f}')
        print(f'dispersion  = {dispersion_ens_vect[i_n]:7.4f}')
        print(f'diameter    = {diameter_ens_vect[i_n]/10.:7.4f} nm')
        
        if calc_energy_with_ads is True:
            
            area_ads = area_surf_list[j_p]/(coverage_list[j_p] *
                                            n_atoms_surf_list[j_p])
            
            print(f'coverage    = {coverage_list[j_p]:7.4f}')
            print(f'area / ads  = {area_ads:7.4f}')
        
        print('')
    
    def dispersion_funct(diameter, a_model, b_model):
        return a_model*diameter**b_model
    
    p0 = [0.5, -1.]
    
    popt, pcov = curve_fit(f     = dispersion_funct    ,
                           xdata = diameter_ens_vect   ,
                           ydata = dispersion_ens_vect ,
                           p0    = p0                  )
    
    a_model, b_model = popt
    
    print(f'dispersion parameters = {a_model:.4f} {b_model:.4f}')
    
    dispersion_fit = np.zeros(len(diameter_ens_vect))
    
    for i in range(len(diameter_ens_vect)):
    
        dispersion_fit[i] = dispersion_funct(diameter = diameter_ens_vect[i],
                                             a_model  = a_model             ,
                                             b_model  = b_model             )
    
    if plot_sites_dist is True:

        fig = plt.figure(4)
        fig.set_size_inches(8, 6)
        
        if x_axis_diameter is True:
            x_vect = [i/10. for i in diameter_ens_vect]
            xlabel = 'diameter [nm]'
            
            plt.axis([0, 10, 0., 1.5])
        
        else:
            x_vect = n_atoms_ave_ens_vect
            xlabel = 'n atoms [-]'
        
            plt.axis([0, n_max_read, 0., 1.5])
        
        for site in sites_dist_ens:
            
            y_vect = sites_dist_ens[site]
            
            if filter_data is True:
            
                pass
                #sigma = 0.5
                #
                #y_new = np.zeros(n_points)
                #
                #for i_n in range(n_points):
                #    for j_m in range(n_points):
                #        y_new[i_n] += 1/(sigma*np.sqrt(2*np.pi)*(
                #            y_vect[j_m]*np.exp(-(x_vect[i_n]-x_vect[j_m])**2)
                #            /(2*sigma**2))
            
            plt.plot(x_vect, y_vect  ,
                     marker    = 'o' ,
                     linestyle = '-' ,
                     label     = site)
        
        plt.yticks(fontsize = tick_size)
        plt.xticks(fontsize = tick_size)
        
        plt.ylabel('n sites [-]', fontsize = label_size)
        plt.xlabel(xlabel, fontsize = label_size)
        
        plt.legend()
        
        if save_plots is True:
            plt.savefig('site_dist_vs_n_atoms.png')

    if plot_n_coord_ave is True:

        fig = plt.figure(5)
        fig.set_size_inches(8, 6)
        
        if x_axis_diameter is True:
            x_vect = [i/10. for i in diameter_ens_vect]
            xlabel = 'diameter [nm]'
        else:
            x_vect = n_atoms_ave_ens_vect
            xlabel = 'n atoms [-]'
        
        y_vect = n_coord_ave_ens_vect
        
        plt.plot(x_vect, y_vect,
                 marker    = 'o',
                 linestyle = '' )
        
        #plt.axis([0., max(y_vect)+0.5, n_min_read, n_max_read])
        
        plt.yticks(fontsize = tick_size)
        plt.xticks(fontsize = tick_size)
        
        plt.ylabel('n coord ave [-]', fontsize = label_size)
        plt.xlabel(xlabel, fontsize = label_size)
        
        if save_plots is True:
            plt.savefig('n_coord_ave_vs_n_atoms.png')

    if plot_dispersion is True:

        fig = plt.figure(6)
        fig.set_size_inches(8, 6)
        
        if x_axis_diameter is True:
            x_vect = [i/10. for i in diameter_ens_vect]
            xlabel = 'diameter [nm]'
        else:
            x_vect = n_atoms_ave_ens_vect
            xlabel = 'n atoms [-]'

        y_vect = dispersion_ens_vect
        
        plt.plot(x_vect, y_vect,
                 marker    = 'o',
                 linestyle = '' )
        
        y_vect = dispersion_fit
        
        plt.plot(x_vect, y_vect,
                 marker    = ' ',
                 linestyle = '-')
        
        #plt.axis([0., max(y_vect)+0.5, n_min_read, n_max_read])
        
        plt.yticks(fontsize = tick_size)
        plt.xticks(fontsize = tick_size)
        
        plt.ylabel('dispersion [-]', fontsize = label_size)
        plt.xlabel(xlabel, fontsize = label_size)
        
        if save_plots is True:
            plt.savefig('dispersion_vs_n_atoms.png')

################################################################################
# SELECT PARTICLE
################################################################################

if select_particle is True:

    print('\n SELECTED PARTICLE \n')

    count_active_sites   = False
    calc_energy_with_ads = False
    write_ase_atoms      = True
    plot_sites_grid      = False
    plot_energy          = False

    particle = particle_sel

    print(f'e form ads   {particle.e_form:.4f}')

    if count_active_sites is True:

        particle.get_active_sites(specify_n_coord  = True ,
                                  specify_supp_int = False,
                                  specify_facets   = False,
                                  check_duplicates = False,
                                  multiple_facets  = False,
                                  convex_sites     = True )
                    
        particle.get_active_sites_dict(with_tags = True)

    if calc_energy_with_ads is True:

        print(f'e form clean {particle.e_form_clean:.4f}')

        particle.get_active_sites_from_dict(with_tags = True)

        particle.get_energy_with_ads(g_bind_dict   = g_bind_dict  ,
                                     bond_length   = bond_length  ,
                                     temperature   = temperature  ,
                                     sites_equilib = sites_equilib)

        print(f'e form ads   {particle.e_form_ads:.4f}')

    if write_ase_atoms is True:

        atoms = particle.to_ase_atoms(symbol = element)
    
        atoms.write('pw.xsf')

    if plot_sites_grid is True:

        particle.plot_active_sites_grid(plot_type        = '3D' ,
                                        specify_n_coord  = True ,
                                        specify_supp_int = False,
                                        specify_facets   = False,
                                        half_plot        = False)

    if plot_energy is True:

        fig = plt.figure(2)
        fig.set_size_inches(8, 6)
        
        x_vect = particle.coverage_vect
        y_vect = particle.e_form_ads_vect
        
        line, = plt.plot(x_vect, y_vect, marker = 'o', alpha = 0.5)
        
        point = plt.plot(particle.coverage, particle.e_form_ads, 
                         marker     = 'o'                      ,
                         color      = line.get_color()         ,
                         alpha      = 0.7                      ,
                         markersize = 10                       )
        
        plt.axis([0., 1., min(y_vect)-5., max(y_vect)+5.])
        
        tick_size  = 14
        label_size = 16
        
        plt.yticks(fontsize = tick_size)
        plt.xticks(fontsize = tick_size)
        
        plt.ylabel('formation energy [eV/atom]', fontsize = label_size)
        plt.xlabel('coverage [-]', fontsize = label_size)
        
        if save_plots is True:
            plt.savefig('e_spec_vs_coverage.png')

################################################################################
# MEASURE TIME END
################################################################################

if measure_time is True:
    print(f'\nExecution time = {timeit.default_timer()-time_start:6.3} s\n')

################################################################################
# SHOW PLOTS
################################################################################

if show_plots is True:
    plt.show()

################################################################################
# END
################################################################################
