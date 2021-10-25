import setuptools

with open('requirements.txt', 'r') as f:
    requirements = f.readlines()

with open('README.md', 'r') as f:
    readme = f.read()

setuptools.setup(
    name="nanoparticles_ensembles",
    version="0.1.1",
    url="https://github.com/raffaelecheula/nanoparticles_ensembles.git",

    author="Raffaele Cheula",
    author_email="cheula.raffaele@gmail.com",

    description="Produce ensembles of particles, obtain 3D grid of active sites, calculate probabilities of metastable particles.",
    long_description=readme,
    license='GPL-3.0',

    packages=[
        'nanoparticles',
    ],
    package_dir={
        'nanoparticles': 'nanoparticles'
    },
    install_requires=requirements,
    python_requires='>=3.5, <4',

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Chemistry',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python',
        'Programming Language :: Cython',
    ],
)
