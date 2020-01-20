CC      = gcc
CFLAGS += -Wall -g -O4 -std=gnu99 -fopenmp -DOPENMP
LDLIBS += -lm -lrt -lgomp

all: stencil

clean:
	-rm stencil

