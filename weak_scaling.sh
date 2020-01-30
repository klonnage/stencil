#!/bin/bash

module load compiler/gcc/9.1.0 compiler/intel/2019_update4 mpi/openmpi/3.1.4
if [ $# -ne 1 ]
then
    echo "Preciser une taille svp"
    exit 0
fi

size=$1
echo "size,time,nproc" > scala-weak-$size.csv
mpicc -DSTENCIL_SIZE_X=$size -DSTENCIL_SIZE_Y=$size -Wall -g -O4 -std=gnu99 stencil_mpi.c -o stencil_mpi -lm -lrt -lgomp
for np in `seq 1 1 7`
do
    n=`echo "($size - 2)*$np + 2" | bc`
    echo -n "$n," >> scala-weak-$size.csv
    salloc -N 3 -p mistral --exclusive mpirun --map-by core --bind-to core -np $((np*np)) ./stencil_mpi >> scala-weak-$size.csv
done
