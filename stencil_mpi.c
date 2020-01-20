#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <getopt.h>
#include <mpi.h>

static int STENCIL_SIZE_X;
static int STENCIL_SIZE_Y;

/** number of buffers for N-buffering; should be at least 2 */
#define STENCIL_NBUFFERS 2

/** conduction coeff used in computation */
static const double alpha = 0.02;

/** threshold for convergence */
static const double epsilon = 0.0001;

/** max number of steps */
static const int stencil_max_steps = 10000;

static double **values[STENCIL_NBUFFERS];

/** latest computed buffer */
static int current_buffer = 0;

/** Grid communicator */
static MPI_Comm grid;

/** Grid informations */
static int dimensions[2];
static const int periodic[] = {0, 0};
static const int reordering = 0;
static int coords[2];
static int grid_rank;
static int max_dim_1;
static int max_dim_2;
static int new_stencil_x;
static int new_stencil_y;

static void init_value(int x, int y, double *vx, double *vy) {

  *vx = (x == 0)*x*(coords[0] == 0) /* x only on the first row n the process grid */
    + (x == new_stencil_x - 1)*(STENCIL_SIZE_X - x);
}

/** init stencil values to 0, borders to non-zero */
static void stencil_init(void)
{
  /* malloc all values */
  for (int buff = 0; buff < STENCIL_NBUFFERS; ++ buff) {
    values[buff] = malloc(sizeof(double*) * new_stencil_x);
    for (int x = 0; x < new_stencil_x; ++ x) {
      values[buff][x] = malloc(sizeof(double) * new_stencil_y);
    }
  }

  int b, x, y;
  for(b = 0; b < STENCIL_NBUFFERS; b++) {
    for(x = 0; x < new_stencil_x; x++) {
      for(y = 0; y < new_stencil_y; y++) {
        values[b][x][y] = 0.0;
      }
    }
    for(x = 0; x < STENCIL_SIZE_X; x++) {
      values[b][x][0] = x;
      values[b][x][STENCIL_SIZE_Y - 1] = STENCIL_SIZE_X - x;
    }
    for(y = 0; y < STENCIL_SIZE_Y; y++) {
      values[b][0][y] = y;
      values[b][STENCIL_SIZE_X - 1][y] = STENCIL_SIZE_Y - y;
    }
  }
}

/** display a (part of) the stencil values */
static void stencil_display(int b, int x0, int x1, int y0, int y1)
{
  int x, y;
  for(y = y0; y <= y1; y++)
  {
    for(x = x0; x <= x1; x++)
    {
      printf("%8.5g ", values[b][x][y]);
    }
    printf("\n");
  }
}

static void stencil_free() {
  for (int buff = 0; buff < STENCIL_NBUFFERS; buff++) {
    for (int x = 0; x < new_stencil_x; x++)
    {
      free(values[buff][x]);
    }
    free(values[buff]);
  }
}

/** compute the next stencil step */
static void stencil_step(void)
{
  int prev_buffer = current_buffer;
  int next_buffer = (current_buffer + 1) % STENCIL_NBUFFERS;
  int x, y;
  for(x = 1; x < STENCIL_SIZE_X - 1; x++) {
    for(y = 1; y < STENCIL_SIZE_Y - 1; y++) {
      values[next_buffer][x][y] =
      alpha * values[prev_buffer][x - 1][y] +
      alpha * values[prev_buffer][x + 1][y] +
      alpha * values[prev_buffer][x][y - 1] +
      alpha * values[prev_buffer][x][y + 1] +
      (1.0 - 4.0 * alpha) * values[prev_buffer][x][y];
    }
  }
  current_buffer = next_buffer;
}

/** return 1 if computation has converged */
static int stencil_test_convergence(void)
{
  int prev_buffer = (current_buffer - 1 + STENCIL_NBUFFERS) % STENCIL_NBUFFERS;
  int x, y;
  for(x = 1; x < STENCIL_SIZE_X - 1; x++) {
    for(y = 1; y < STENCIL_SIZE_Y - 1; y++) {
      if(fabs(values[prev_buffer][x][y] - values[current_buffer][x][y]) > epsilon)
        return 0;
    }
  }
  return 1;
}

static int stencil_all_in_one(void) {
  int prev_buffer = current_buffer;
  int next_buffer = (current_buffer + 1) % STENCIL_NBUFFERS;
  int x, y;
  int has_converged = 1;
  for(x = 1; x < STENCIL_SIZE_X - 1; x++) {
    for(y = 1; y < STENCIL_SIZE_Y - 1; y++) {
      values[next_buffer][x][y] =
      alpha * values[prev_buffer][x - 1][y] +
      alpha * values[prev_buffer][x + 1][y] +
      alpha * values[prev_buffer][x][y - 1] +
      alpha * values[prev_buffer][x][y + 1] +
      (1.0 - 4.0 * alpha) * values[prev_buffer][x][y];
      has_converged = has_converged && (fabs(values[next_buffer][x][y] - values[current_buffer][x][y]) > 1);
    }
  }
  current_buffer = next_buffer;
  return has_converged;
}

/** Return a divisor, the closest inferior to sqrt(x) */
static int best_divisor(int x) {
  int d = 2, max_d = 1;
  while (d*d <= x) {
    if (x % d == 0) {
      max_d = d;
    }
    ++d;
  }
  return max_d;
}

int main(int argc, char**argv)
{
  MPI_Init(&argc, &argv);
  int MAX_STENCIL_SIZE = -1;
  char opt;
  while ((opt = getopt(argc, argv, "s:")) != -1) {
    switch (opt) {
    case 's':
      MAX_STENCIL_SIZE = atoi(optarg);
      break;
    default: /* '?' */
      fprintf(stderr, "Usage: %s [-t nsecs] [-n] name\n",
              argv[0]);
      exit(EXIT_FAILURE);
    }
  }

  if(MAX_STENCIL_SIZE < 0) {
    fprintf(stderr, "No sie specified\n");
    exit(EXIT_FAILURE);
  }

  int wsize, wrank;
  MPI_Comm_size(MPI_COMM_WORLD, &wsize);
  MPI_Comm_rank(MPI_COMM_WORLD, &wrank);

  int fdim  = best_divisor(wsize);
  max_dim_1 = dimensions[0] = fdim;
  max_dim_2 = dimensions[1] = wsize / fdim;
  MPI_Cart_create(MPI_COMM_WORLD, 2, dimensions, periodic, reordering, &grid);

  MPI_Comm_rank(grid, &grid_rank);
  MPI_Cart_coords(grid, grid_rank, 2, coords);

  printf("size,time,nstep\n");
  for (int size = 10; size < MAX_STENCIL_SIZE; size *= 1.25) {
    STENCIL_SIZE_Y = STENCIL_SIZE_X = size;
    /* TODO : Corriger le calcul de new_stencil_* */
    new_stencil_x = (STENCIL_SIZE_X / max_dim_1) + (STENCIL_SIZE_X % max_dim_1);
    new_stencil_y = (STENCIL_SIZE_Y / max_dim_2) + (STENCIL_SIZE_Y % max_dim_2);

    //for (STENCIL_SIZE_Y = 10; STENCIL_SIZE_Y < MAX_STENCIL_SIZE_Y; STENCIL_SIZE_Y *= 1.25) {
      stencil_init();
      //stencil_display(current_buffer, 0, STENCIL_SIZE_X - 1, 0, STENCIL_SIZE_Y - 1);

      struct timespec t1, t2;
      clock_gettime(CLOCK_MONOTONIC, &t1);
      int s;
      for(s = 0; s < stencil_max_steps; s++) {
#ifdef ALL_IN_ONE
        if(stencil_all_in_one()) {
          break;
        }
#else
        stencil_step();
        if(stencil_test_convergence())
        {
          break;
        }
#endif
      }
      clock_gettime(CLOCK_MONOTONIC, &t2);
      const double t_usec = (t2.tv_sec - t1.tv_sec) * 1000000.0 + (t2.tv_nsec - t1.tv_nsec) / 1000.0;
      printf("%d,%g,%d\n", size, t_usec, s);
      //stencil_display(current_buffer, 0, STENCIL_SIZE_X - 1, 0, STENCIL_SIZE_Y - 1);
      stencil_free();
    //}
  }
  MPI_Comm_free(&grid);
  MPI_Finalize();
  return 0;
}

