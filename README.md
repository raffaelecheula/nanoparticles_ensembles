# nanoparticles_ensembles

A Python/Cython library useful to:
* Produce ensembles of nanoparticles (fcc single crystals, decahedra and icosahedra shapes) with arbitrary size, low or high symmetry.
* Calculate the distribution of active sites of the nanoparticles.
* Calculate the probabilities of the nanoparticles with a Boltzmann distribution and analyze the effect of the presence of metastable shapes in thermal equilibrium with the ground state shape on the catalytic activity.

## **Requirements:**
* NumPy
* SciPy
* Matplotlib
* ASE (Atomic Simulation Environment)
* Cantera
* Cython

## **Installation:**
To compile the Cython code, navigate to the folder `nanoparticles` and type in a terminal: `bash cython_cmd`

## Authors:
* Raffaele Cheula (raffaele.cheula@polimi.it)

## Reference:
R. Cheula, M. Maestri, G. Mpourmpakis, "Modeling Morphology and Catalytic Activity of Nanoparticle Ensembles Under Reaction Conditions", ACS Catalysis, 2020, under revision
