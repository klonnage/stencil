MPICC   = mpicc
CC      = gcc
CFLAGS += -Wall -g -O4 -std=gnu99 -fopenmp
LDLIBS += -lm -lrt -lgomp

all: stencil

stencil_all_in_one:stencil.c
	${CC} ${CFLAGS} -DALL_IN_ONE -o $@ $< ${LDLIBS} 

stencil_omp:stencil.c
	${CC} ${CFLAGS} -DOPENMP -o $@ $< ${LDLIBS} 

stencil_mpi:stencil_mpi.c
	${MPICC} ${CFLAGS} $< -o $@ ${LDLIBS}


stencil_mpi_omp:stencil_mpi.c
	${MPICC} -DOMP ${CFLAGS} $< -o $@ ${LDLIBS}

clean:
	-rm stencil

