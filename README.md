# Lanczos

The code currently uses mainly NumPy for almost all the interations on the code, it's well write as functions to speedup.

# lanczos.py file 

It's the most stable one, working fine with square grid, not so fast or optimized because it's the first version of the code, I think I'll delete it eventualy because the file "lanczos2.py" is also stable and with more features...

# lanczos2.py

Alsi stable, but with the newest implementations. The square lattice keep not to fast (usualy it dont represents any real physical problem, so I'll not invest my time here by now) but the honneycomb lattice is good optmized with NumPy and works really much better than the square one).

Currently the code is all write in functions forms, but enventually I'll add some OPP features for make it easier to read.

# lanczos_sparse_matrix.py 

It's cuncurrently to the above one, but with the time my plan is add sparse matrix with SciPy framework to the program be able to work with bigger matrices. Everything here works fine (like lanczos2) but for small cases is better use lanczos2 (it's considerable faster), for large systems use this one. It's also fast, but slower than the above one.

# Comentaries

By now the code works fine with PyPy (ignoring the matplolib and pandas part, off course), with that all the "for" relations become much more faster;
In the end of the code, the data uses Matplotlib and Pandas do show the matrices, eventualy it will become a new .py  file, but by now, it works fine for our prouposes;

I'll also add numba and MPI in the future for most speed (PyPy will not work with this both);

Also, with SWIG I'm working in a C++->Python wrapper to make the Hamiltonian generation faster (these are the cwpp.py cwpp.i and generate.sh files);

We are using single orbitals, so to speedup the code we make alpha_n=0 for all n. That improves the code execution in 30%.
