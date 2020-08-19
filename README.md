# Lanczos

The code currently uses mainly NumPy for almost all the interations on the code, it's well write as functions to speedup.

# lanczos.py file 

It's the most stable one, working fine with square grid, not so fast or optimized because it's the first version of the code...

# lanczos2.py

Less stable because I'm currently make the addition of hexagonal honeycomb lattices with numpy ...

# lanczos_sparse_matrix.py 

It's cuncurrently to the above one, but with the time my plan is add sparse matrix with SciPy framework to the program be able to work with bigger matrices.

# Comentaries

Currently the code works fine with PyPy (ignoring the matplolib and pandas part, off course), with that all the "for" relations become much more faster;
In the end of the code, the data uses Matplotlib and Pandas do show the matrices, eventualy it will become a new .py file, but by now, it works fine for our prouposes;
I'll also add numba and MPI in the future for most speed (PyPy will not work with this both);

Also, with SWIG I'm working in a C++->Python wrapper to make the Hamiltonian generation faster;
