Culture and cooperation in a spatial public goods game
# Culture and cooperation in a spatial public goods game

## Software

Imported from https://sites.google.com/site/alexdstivala/home/culture_cooperation

This software is free under the terms of the GNU General Public License.
It is derived from code developed for an earlier publication
[Ultrametric distribution of culture vectors in an extended Axelrod model of cultural dissemination](http://munk.cis.unimelb.edu.au/~stivalaa/ultrametric_axelrod/).
It uses Python
and parallelization using MPI (with [mpi4py](http://mpi4py.scipy.org/)). It also requires the Python libraries [NumPy](http://www.numpy.org/) (part of the [SciPy package](http://www.scipy.org/)) and 
[igraph](http://igraph.sourceforge.net/).

The Python code was run with NumPy version 1.9.1, SciPy version 0.14.1, igraph version 0.6 and mpi4py version 1.3.1 under Python version 2.7.9 on a Lenovo x86 cluster (992 Intel Haswell compute cores running at 2.3GHz) running Linux (RHEL 6) with Open MPI version 1.10.0.
The C++ code was compiled with gcc version 4.9.2. 

### Running the models

The model can be run with a command line such as: `mpirun --mca mpi_warn_on_fork 0 python ./lattice-python-mpi/src/axelrod/geo/expphysicstimeline/multiruninitmain.py m:100 F:5 strategy_update_rule:fermi culture_update_rule:fermi ./lattice-jointactivity-simcoop-social-noise-constantmpcr-cpp-end/model 10000`


## Reference

If you use our software, data, or results in your research, please cite:

- A. Stivala, Y. Kashima and M. Kirley 2016 [Culture and cooperation in a spatial public goods game](http://link.aps.org/doi/10.1103/PhysRevE.94.032303 )Phys. Rev. E94:032303 [doi: 10.1103/PhysRevE.94.032303](http://dx.doi.org/10.1103/PhysRevE.94.032303)

