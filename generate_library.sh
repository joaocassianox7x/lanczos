#!/bin/bash


current_directory=`pwd`

echo "Estage 1"
python-config --cflags

echo "Estage 2"

library_name=$1
wrap='_wrap'
ul='_'

rm -rf $library_name.o $library_name.py $library_name.pyc $ul$library_namecwpp.so $library_name$wrap.o 


echo "Estage 3"
swig -c++ -python $library_name.i

echo "Estage 4"
g++ -c -fpic $library_name$wrap.cxx $library_name.cpp -I/usr/include/python2.7


echo "Estage 5"
g++ -shared $library_name.o $library_name$wrap.o -o $ul$library_name.so

export PYTHONPATH="$current_directory/$library_name.py"

echo "### DONE ###"
