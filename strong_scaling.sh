#!/bin/bash

module load compiler/gcc/9.1.0 compiler/intel/2019_update4 mpi/openmpi/3.1.4

if [ $# -ne 1 ]
then
    echo "Preciser une taille svp"
    exit 0
fi

size=$1
echo "size,time,nproc" > scala-strong-$size.csv
for np in `seq 1 1 6`
do
    n=`echo "($size - 2)/$np + 2" | bc`
    mpicc -DSTENCIL_SIZE_X=$n -DSTENCIL_SIZE_Y=$n -Wall -g -O4 -std=gnu99 stencil_mpi.c -o stencil_mpi -lm -lrt -lgomp
    echo -n "$size," >> scala-strong-$size.csv
    salloc -N 2 -p mistral --exclusive mpirun --map-by core --bind-to core -np $((np*np)) ./stencil_mpi >> scala-strong-$size.csv
done
