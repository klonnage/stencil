#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <getopt.h>

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

/** init stencil values to 0, borders to non-zero */
static void stencil_init(void)
{
  /* malloc all values */
  for (int buff = 0; buff < STENCIL_NBUFFERS; ++ buff) {
    values[buff] = malloc(sizeof(double*) * STENCIL_SIZE_X);
    for (int x = 0; x < STENCIL_SIZE_X; ++ x) {
      values[buff][x] = malloc(sizeof(double) * STENCIL_SIZE_Y);
    }
  }

  int b, x, y;
  for(b = 0; b < STENCIL_NBUFFERS; b++) {
    for(x = 0; x < STENCIL_SIZE_X; x++) {
      for(y = 0; y < STENCIL_SIZE_Y; y++) {
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
    for (int x = 0; x < STENCIL_SIZE_X; x++)
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

/* OpenMP function */
static void stencil_step_omp(void)
{
  int prev_buffer = current_buffer;
  int next_buffer = (current_buffer + 1) % STENCIL_NBUFFERS;
  int x, y;
  #pragma omp parallel for collapse(2) shared(values) firstprivate(prev_buffer, next_buffer)
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

/* OpenMP verison */
static int stencil_test_convergence_omp(void)
{
  int prev_buffer = (current_buffer - 1 + STENCIL_NBUFFERS) % STENCIL_NBUFFERS;
  int x, y, has_converged = 1;
  #pragma omp parallel
  {
  #pragma omp for collapse(2) firstprivate(prev_buffer, values) private(x, y) reduction(&& : has_converged)
  for(x = 1; x < STENCIL_SIZE_X - 1; x++) {
    for(y = 1; y < STENCIL_SIZE_Y - 1; y++) {
      if(fabs(values[prev_buffer][x][y] - values[current_buffer][x][y]) > epsilon) {
        has_converged = 0;
      }
    }
  }
  }
  return has_converged;
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
      has_converged = has_converged && (fabs(values[next_buffer][x][y] - values[current_buffer][x][y]) > epsilon);
    }
  }
  current_buffer = next_buffer;
  return has_converged;
}

int main(int argc, char**argv)
{
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

  printf("size,time,nstep\n");
  for (int size = 10; size < MAX_STENCIL_SIZE; size *= 1.25) {
    STENCIL_SIZE_Y = STENCIL_SIZE_X = size;
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
#elif OPENMP
        stencil_step_omp();
        if(stencil_test_convergence_omp())
        {
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
  return 0;
}

