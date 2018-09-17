#include<stdio.h>
#include<mpi.h>
#include<time.h>
#define ROOT 0
#define TEST 3

int main(int argc, char * argv[])
{
  MPI_Init(&argc, &argv); 
  int world_rank;
  int world_size;
  MPI_Request request, *request_array;
  MPI_Status  status, *status_array;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);


  int C = 3;
  int H = 1024;
  int W = 1024;
  int c, x, y;
  int i,j,k;
  struct timespec local_start, local_end;
  double local_start_dbl, local_end_dbl;
  double global_start_dbl, global_end_dbl;
  double checksum;
  double *ir, *buf, *o, *local_start_array,*local_end_array;


  // C2: using MPI_Allreduce()
  checksum = 0.0;
  ir = NULL;
  o  = NULL;


  ir =  (double*)malloc(sizeof(double)*C*H*W);
  o  =  (double*)malloc(sizeof(double)*C*H*W);
  for(k=0; k<C*H*W; ++k)
  {
    c = k/(H*W);
    x = (k - c*H*W)/W;
    y = (k - c*H*W)%W;
    // ir[k] = 0.0 + (world_rank + 1) + c*(x+y);
    ir[k] = 0.0 + world_rank + c*(x+y);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  clock_gettime(CLOCK_MONOTONIC, &local_start);
  MPI_Allreduce(ir, o, C*H*W, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  clock_gettime(CLOCK_MONOTONIC, &local_end);
  local_start_dbl = (double)local_start.tv_sec + (double)local_start.tv_nsec/1.0e9;
  local_end_dbl   = (double)local_end.tv_sec   + (double)local_end.tv_nsec/1.0e9;

  if(world_rank==ROOT)
  {
    for(k = 0;k<C*H*W;++k)
    {
      o[k] /= (1.0*world_size);
      checksum += o[k];
    }
    printf("%f, %4.3lf\n",checksum, 1000.0*(local_end_dbl-local_start_dbl));
  }

  free(ir);
  free(o);

  MPI_Finalize();
  return 0;
}
