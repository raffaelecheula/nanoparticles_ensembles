# nanoparticles_ensembles

A Python/Cython library useful to:
* Produce ensembles of nanoparticles (fcc single crystals, decahedra and icosahedra) with arbitrary size and shape, low or high level of symmetry.
* Obtain the 3D grid of active sites of the nanoparticles and calculate the corresponding active sites distribution.
* Calculate the probabilities of the nanoparticles with a Boltzmann distribution and analyze the effect of the presence of metastable shapes in the ensemble on the catalytic activity.

## **Requirements:**
* NumPy
* SciPy
* Matplotlib
* ASE (Atomic Simulation Environment)
* Cantera
* Cython

## **Installation:**
cd nanoparticles
pip install -e .

## Authors:
* Raffaele Cheula (raffaele.cheula@polimi.it)

## Reference:
R. Cheula, M. Maestri, G. Mpourmpakis, "Modeling Morphology and Catalytic Activity of Nanoparticle Ensembles Under Reaction Conditions", ACS Catalysis, 2020, DOI: 10.1021/acscatal.0c01005
