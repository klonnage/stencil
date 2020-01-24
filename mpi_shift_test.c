#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

/* Enum for all directions */
enum Directions{
  UPLEFT, UP, UPRIGHT, LEFT, CENTER, RIGHT, DOWNLEFT, DOWN, DOWNRIGHT, LAST_DIRECTION
};
enum Directions opposite[] = {DOWNRIGHT, DOWN, DOWNLEFT, RIGHT, CENTER, LEFT, UPRIGHT, UP, UPLEFT, LAST_DIRECTION};

/* Rank shift info */
int neighbors[LAST_DIRECTION];
int default_rank = MPI_PROC_NULL;
int procdim;

/* Grid info */
MPI_Comm grid;
int      ndims=2;
int      dims[] = {0, 0};
int      periods[] = {0, 0};
int      coords[2];
int      cart_rank;


int init_ranks() {
  for(int d = 0; d < LAST_DIRECTION; ++d)
    neighbors[d] = MPI_PROC_NULL;
  neighbors[CENTER] = cart_rank;
  /* Get direct neighbors */
  MPI_Cart_shift(grid, 0,  1, neighbors + LEFT, neighbors + RIGHT);
  MPI_Cart_shift(grid, 1,  0, neighbors + UP  , neighbors + DOWN);
  /* Set left sided directions */
  MPI_Sendrecv(neighbors + UP  , 1, MPI_INT, neighbors[RIGHT], UPRIGHT  , neighbors + UPLEFT  , 1, MPI_INT, neighbors[LEFT], UPRIGHT  , grid, MPI_STATUS_IGNORE);
  MPI_Sendrecv(neighbors + DOWN, 1, MPI_INT, neighbors[RIGHT], DOWNRIGHT, neighbors + DOWNLEFT, 1, MPI_INT, neighbors[LEFT], DOWNRIGHT, grid, MPI_STATUS_IGNORE);
  /* Set right sided directions */
  MPI_Sendrecv(neighbors + UP  , 1, MPI_INT, neighbors[LEFT], UPLEFT  , neighbors + UPRIGHT  , 1, MPI_INT, neighbors[RIGHT], UPLEFT  , grid, MPI_STATUS_IGNORE);
  MPI_Sendrecv(neighbors + DOWN, 1, MPI_INT, neighbors[LEFT], DOWNLEFT, neighbors + DOWNRIGHT, 1, MPI_INT, neighbors[RIGHT], DOWNLEFT, grid, MPI_STATUS_IGNORE);
}

void my_send_recv_directions(void* tosend, void* torecv, MPI_Datatype type, enum Directions dir) {
  MPI_Sendrecv(tosend, 1, type, neighbors[dir], 0, torecv, 1, type, neighbors[opposite[dir]], 0, grid, MPI_STATUS_IGNORE);
  //printf("[%d] : in : %d -> out : %d\n", cart_rank, *(int*)tosend, *(int*)torecv);
}

int sqrt_int(int n) {
  int d;
  for(d = 2; d*d < n; ++d);
  return (d*d == n) ? d : -1;
}

int print_map(int *buff) {
  int _coords[2];
  for (int i = 0; i < dims[1]; ++i)
  {
    _coords[1] = i; 
    for (int j = 0; j < dims[0]; ++j)
    {
      _coords[0] = j;
      int rnk;
      MPI_Cart_rank(grid, _coords, &rnk);
      if (buff == NULL && cart_rank == 0) {
        printf("%2d ", rnk);
      }
      else if (buff != NULL) {
        if (cart_rank == 0) {
          int toprint;
          if (rnk == 0) {
            toprint = *buff;
          } else {
            MPI_Recv(&toprint, 1, MPI_INT, rnk, 1, grid, MPI_STATUS_IGNORE);
            //printf("%d\n", rnk);
          }
          printf("%3d ", toprint);
        }
        else if (cart_rank == rnk) {
          MPI_Send(buff, 1, MPI_INT, 0, 1, grid);
        }
      }
    }
    if (cart_rank == 0) {
      printf("\n");
    }
  }
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  
  dims[0] = dims[1] = procdim = sqrt_int(size);
  if (procdim < 0) {
    if(!rank){fprintf(stderr, "Dimension is not a square\n");}
    MPI_Finalize();
    exit(EXIT_SUCCESS);
  }
 
  MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, 1, &grid);
  MPI_Comm_rank(grid, &cart_rank);

  MPI_Cart_coords(grid, cart_rank, ndims, coords);

  print_map(NULL);
  
  init_ranks();

  for (int rnk = 0; rnk < size; ++rnk) {
    if (rnk == cart_rank) {
      if (!rnk) {printf("[%d] NULL : %d\n", rnk, (int)MPI_PROC_NULL);}
      //printf("[%d] -> : %d / %d\n", rnk, sources[0], neighbors[0]);
      printf("[%d] %3d %3d %3d\n", rnk, neighbors[UPLEFT], neighbors[UP], neighbors[UPRIGHT]);      
      printf("[%d] %3d %3d %3d\n", rnk, neighbors[LEFT], neighbors[CENTER], neighbors[RIGHT]);
      printf("[%d] %3d %3d %3d\n\n", rnk, neighbors[DOWNLEFT], neighbors[DOWN], neighbors[DOWNRIGHT]);
    }
    MPI_Barrier(grid);
  }

  /* Communicate value through a dimension */
  int val = cart_rank, out = MPI_PROC_NULL;
  my_send_recv_directions(&val, &out, MPI_INT, UPRIGHT);

  print_map(&val);
  if(cart_rank == 0) {puts("");}
  print_map(&out);


  MPI_Comm_free(&grid);

  MPI_Finalize();
  return 0;
}
