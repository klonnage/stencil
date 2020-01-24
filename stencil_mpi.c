#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <getopt.h>
#include <mpi.h>

#define STENCIL_SIZE_X 10
#define STENCIL_SIZE_Y 10

/** number of buffers for N-buffering; should be at least 2 */
#define STENCIL_NBUFFERS 2

/** conduction coeff used in computation */
static const double alpha = 0.02;

/** threshold for convergence */
static const double epsilon = 0.0001;

/** max number of steps */
static const int stencil_max_steps = 10000;

static double values[STENCIL_NBUFFERS][STENCIL_SIZE_X][STENCIL_SIZE_Y];

/** latest computed buffer */
static int current_buffer = 0;

/** Grid communicator */
static MPI_Comm grid;

/** Grid information */
static int       dimensions[2];
static int const periodic[] = {0, 0};
static int const reordering = 0;
static int       coords[2];
static int       grid_rank;
static int       max_dim_1;
static int       max_dim_2;
static int       new_stencil_x;
static int       new_stencil_y;

/*************** Directions definitions *****************/

/* Enum for all directions */
enum Directions{
  UPLEFT, UP, UPRIGHT, LEFT, CENTER, RIGHT, DOWNLEFT, DOWN, DOWNRIGHT, LAST_DIRECTION
};
enum Directions opposite  [] = {DOWNRIGHT, DOWN, DOWNLEFT, RIGHT, CENTER, LEFT, UPRIGHT, UP, UPLEFT, LAST_DIRECTION};

/* Rank shift info */
int neighbors[LAST_DIRECTION];
int default_rank = MPI_PROC_NULL;
int procdim;

/** Assigns the good rank to all neighbors */
void init_ranks() {
  /* Convenient way to say there is no one arround me */
  for(int d = 0; d < LAST_DIRECTION; ++d) {
    neighbors[d] = MPI_PROC_NULL;
  }
  neighbors[CENTER] = grid_rank;
  /* Get direct neighbors */
  MPI_Cart_shift(grid, 0,  1, neighbors + LEFT, neighbors + RIGHT);
  MPI_Cart_shift(grid, 1,  1, neighbors + UP  , neighbors + DOWN);
  /* Set left sided directions */
  MPI_Sendrecv(neighbors + UP  , 1, MPI_INT, neighbors[RIGHT], UPRIGHT  , neighbors + UPLEFT  , 1, MPI_INT, neighbors[LEFT], UPRIGHT  , grid, MPI_STATUS_IGNORE);
  MPI_Sendrecv(neighbors + DOWN, 1, MPI_INT, neighbors[RIGHT], DOWNRIGHT, neighbors + DOWNLEFT, 1, MPI_INT, neighbors[LEFT], DOWNRIGHT, grid, MPI_STATUS_IGNORE);
  /* Set right sided directions */
  MPI_Sendrecv(neighbors + UP  , 1, MPI_INT, neighbors[LEFT], UPLEFT  , neighbors + UPRIGHT  , 1, MPI_INT, neighbors[RIGHT], UPLEFT  , grid, MPI_STATUS_IGNORE);
  MPI_Sendrecv(neighbors + DOWN, 1, MPI_INT, neighbors[LEFT], DOWNLEFT, neighbors + DOWNRIGHT, 1, MPI_INT, neighbors[RIGHT], DOWNLEFT, grid, MPI_STATUS_IGNORE);
}


/********************* Types ****************************/

/** Declarations */
MPI_Datatype row, col;
MPI_Datatype comm_types[LAST_DIRECTION];

/** Allocate types and set array of type to ease communications */
void create_types() {
  /** Contiguous columns and vector row */
  MPI_Type_contiguous(STENCIL_SIZE_Y-2, MPI_DOUBLE, &col);
  MPI_Type_vector(STENCIL_SIZE_X-2, 1, STENCIL_SIZE_Y, MPI_DOUBLE, &row);
  //MPI_Type_vector(1, STENCIL_SIZE_X-2, STENCIL_SIZE_Y, MPI_DOUBLE, &col);

  MPI_Type_commit(&row);
  MPI_Type_commit(&col);

  /* Corners */
  comm_types[DOWNLEFT]  =
  comm_types[DOWNRIGHT] =
  comm_types[UPRIGHT]   =
  comm_types[UPLEFT]    = MPI_DOUBLE;
  /* Verticals */
  comm_types[UP] = comm_types[DOWN] = row;
  /* Horizontal */
  comm_types[RIGHT] = comm_types[LEFT] = col;
}

/** Send data to dest and receive from source  */
void my_send_recv_directions(void* tosend, void* torecv, enum Directions dir) {
  MPI_Datatype type = comm_types[dir] ;
  if(dir != UP && dir != DOWN) MPI_Sendrecv(tosend, 1, type, neighbors[dir], dir, torecv, 1, type, neighbors[opposite[dir]], dir, grid, MPI_STATUS_IGNORE);
  else {
    double sbuff[STENCIL_SIZE_X - 2], rbuff[STENCIL_SIZE_X - 2];
    double *rtosend = (double*)tosend;
    double *rtorecv = (double*)torecv;

    for (int i = 0; i < STENCIL_SIZE_X - 2; ++i) {
      rbuff[i] = *(rtorecv + i*STENCIL_SIZE_Y);
      sbuff[i] = *(rtosend + i*STENCIL_SIZE_Y);
    }
    

    MPI_Sendrecv(sbuff, STENCIL_SIZE_X-2, MPI_DOUBLE, neighbors[dir], dir, rbuff, STENCIL_SIZE_X-2, MPI_DOUBLE, neighbors[opposite[dir]], dir, grid, MPI_STATUS_IGNORE);

    for (int i = 0; i < STENCIL_SIZE_X-2; ++i) {
      *(rtorecv + i*STENCIL_SIZE_Y) = rbuff[i];
    }
  }
}

/** Free types */
void free_types() {
    MPI_Type_free(&row);
    MPI_Type_free(&col);
}

/************* Indexe global / local computation ******/

void local2global(int xl, int yl, int *xg, int *yg) {
  *xg = coords[0]*(STENCIL_SIZE_X - 1) + xl;
  *yg = coords[1]*(STENCIL_SIZE_Y - 1) + yl;
}

void global2local(int *xl, int *yl, int xg, int yg) {
  *xl = (xg % (STENCIL_SIZE_X - 1));
  *yl = (yg % (STENCIL_SIZE_Y - 1));
}

static int index_send[LAST_DIRECTION][2];
static int index_recv[LAST_DIRECTION][2];

static void init_indexes_comm() {
  /* Send */
  index_send[UPLEFT][0] = index_send[LEFT][0] = index_send[UP][0] = 1;
  index_send[UPLEFT][1] = index_send[LEFT][1] = index_send[UP][1] = 1;

  index_send[UPRIGHT][0] = index_send[RIGHT][0] = STENCIL_SIZE_X - 2;
  index_send[UPRIGHT][1] = index_send[RIGHT][1] = 1;

  index_send[DOWNLEFT][0] = index_send[DOWN][0] = 1;
  index_send[DOWNLEFT][1] = index_send[DOWN][1] = STENCIL_SIZE_Y - 2;

  index_send[DOWNRIGHT][0] = STENCIL_SIZE_X - 2;
  index_send[DOWNRIGHT][1] = STENCIL_SIZE_Y - 2;

  /* Recv */
  index_recv[UPLEFT][0] = index_recv[LEFT][0] = index_recv[DOWNLEFT][0] = STENCIL_SIZE_X - 1;
  index_recv[UPRIGHT][0] = index_recv[RIGHT][0] = index_recv[DOWNRIGHT][0] = 0;
  index_recv[UP][0] = index_recv[DOWN][0] = 1;

  index_recv[UPLEFT][1] = index_recv[UP][1] = index_recv[UPRIGHT][1] = STENCIL_SIZE_Y - 1;
  index_recv[DOWNLEFT][1] = index_recv[DOWN][1] = index_recv[DOWNRIGHT][1] = 0;
  index_recv[RIGHT][1] = index_recv[LEFT][1] = 1;
}

/************* Stencil operations *********************/

/** init stencil values to 0, borders to non-zero */
static void stencil_init(void)
{
  /* malloc all values */
  int b, x, y, xg, yg;
  for(b = 0; b < STENCIL_NBUFFERS; b++) {
    for(x = 0; x < new_stencil_x; x++) {
      for(y = 0; y < new_stencil_y; y++) {
        values[b][x][y] = 0.0;
      }
    }
    for(x = 0; x < STENCIL_SIZE_X; x++) {
      local2global(x, 0, &xg, &yg);
      if (coords[1] == 0) values[b][x][0] = xg;
      if (coords[1] == max_dim_2 - 1) values[b][x][STENCIL_SIZE_Y - 1] = (STENCIL_SIZE_X-1)*max_dim_1 - xg;
    }
    for(y = 0; y < STENCIL_SIZE_Y; y++) {
      local2global(0, y, &xg, &yg);
      if (coords[0] == 0) values[b][0][y] = yg;
      if (coords[0] == max_dim_1 - 1) values[b][STENCIL_SIZE_X - 1][y] = (STENCIL_SIZE_Y-1)*max_dim_2 - yg;
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
      printf("%4.5g ", values[b][x][y]);
    }
    printf("\n");
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


/* Send and receive data to all neighbors */
void stencil_update() {
  // values[current_buffer][index_send[d][0] + 2][index_send[d][1]] = 3.;

  /*MPI_Datatype type = comm_types[d];
  MPI_Sendrecv(&values[current_buffer][index_send[d][0]][index_send[d][1]], 1, row, neighbors[d], d,
               buff, STENCIL_SIZE_X-2, MPI_DOUBLE, neighbors[opposite[d]], d, grid, MPI_STATUS_IGNORE);*/
  
  /*my_send_recv_directions(&values[current_buffer][index_send[d][0]][index_send[d][1]],
                          &values[current_buffer][index_recv[d][0]][index_recv[d][1]], d);
  for (int i = 0; i < STENCIL_SIZE_X - 2; i++)
  {
    printf("%4.2g ",values[current_buffer][index_send[d][0] + i][index_send[d][1]]);
  }
  puts("");
  
  for (int i = 0; i < STENCIL_SIZE_X - 2; i++)
  {
    printf("%4.2g ", buff[i]);//values[current_buffer][index_recv[d][0] + i][index_recv[d][1]]);
  }
  puts("\n");*/

  for (int i = 0; i < LAST_DIRECTION; ++i) {
    if (i != CENTER) {
      my_send_recv_directions(&values[current_buffer][index_send[i][0]][index_send[i][1]],
                              &values[current_buffer][index_recv[i][0]][index_recv[i][1]], i);
    }
  }
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

  int wsize, wrank;
  MPI_Comm_size(MPI_COMM_WORLD, &wsize);
  MPI_Comm_rank(MPI_COMM_WORLD, &wrank);

  /** Create the grid */
  int fdim  = best_divisor(wsize);
  max_dim_1 = dimensions[0] = fdim;
  max_dim_2 = dimensions[1] = wsize / fdim;
  MPI_Cart_create(MPI_COMM_WORLD, 2, dimensions, periodic, reordering, &grid);

  MPI_Comm_rank(grid, &grid_rank);
  MPI_Cart_coords(grid, grid_rank, 2, coords);

  init_ranks();
  init_indexes_comm();

  create_types();

  int size = STENCIL_SIZE_X;
  //for (int size = 10; size < MAX_STENCIL_SIZE; size *= 1.25) {
    /* TODO : Corriger le calcul de new_stencil_* */
    
    //for (STENCIL_SIZE_Y = 10; STENCIL_SIZE_Y < MAX_STENCIL_SIZE_Y; STENCIL_SIZE_Y *= 1.25) {
  printf("[%d] : %d, %d\n", grid_rank, coords[0], coords[1]);
  stencil_init();

  for (int step = 0; step < 4; ++step)
  {
    stencil_step();
    stencil_update();
    printf("\nstep %d : \n", step + 1);
    stencil_display((step + 1)%2, 0, STENCIL_SIZE_X-1, 0, STENCIL_SIZE_Y-1);
  }
  
#if 0
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
  //}
#endif//0
  MPI_Comm_free(&grid);

  free_types();

  MPI_Finalize();
  return 0;
}

