MPICC   = mpicc
CC      = gcc
CFLAGS += -Wall -g -O4 -std=gnu99 -fopenmp -DOPENMP
LDLIBS += -lm -lrt -lgomp

all: stencil

stencil_mpi:stencil_mpi.c
	${MPICC} ${CFLAGS} $< -o $@ ${LDLIBS}

clean:
	-rm stencil

