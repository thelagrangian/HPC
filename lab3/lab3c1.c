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
  int num_worker = world_size - 1;
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

  // C1: using MPI_Isend() and MPI_Irecv()
  checksum = 0.0;
  request_array = NULL;
  status_array  = NULL;
  ir = NULL;
  buf= NULL;
  o  = NULL;

  if(world_rank > ROOT)
  {
    ir  = (double*)malloc(sizeof(double)*C*H*W);
    for(k=0; k<C*H*W; ++k)
    {
      c = k/(H*W);
      x = (k - c*H*W)/W;
      y = (k - c*H*W)%W;
      ir[k] = 0.0 + world_rank + c*(x+y);
    }
  }
  else if(world_rank==ROOT)
  {
    buf = (double*)malloc(sizeof(double)*C*H*W*num_worker);
    o   = (double*)malloc(sizeof(double)*C*H*W);
    request_array = (MPI_Request*)malloc(sizeof(MPI_Request)*num_worker);
    status_array  = (MPI_Status *)malloc(sizeof(MPI_Status) *num_worker);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  clock_gettime(CLOCK_MONOTONIC, &local_start);
  if(world_rank > ROOT)
  {
    MPI_Isend(ir, C*H*W, MPI_DOUBLE, ROOT, 0, MPI_COMM_WORLD, &request);
    MPI_Wait(&request, &status);
  }
  else if(world_rank==ROOT)
  {
    for(j=1;j<=num_worker;++j)
      MPI_Irecv(buf+(j-1)*C*H*W, C*H*W, MPI_DOUBLE, j, 0, MPI_COMM_WORLD, request_array+(j-1));
    for(j=1;j<=num_worker;++j)
      MPI_Wait(request_array+(j-1), status_array+(j-1));
    for(k=0; k<C*H*W; ++k)
    {
      o[k] = 0.0;
      for(j=0;j<num_worker;++j)
      {
        o[k] += *(buf+j*C*H*W+k);
      }
      o[k] /= (1.0*num_worker);
    }
  }
  clock_gettime(CLOCK_MONOTONIC, &local_end);
  local_start_dbl = (double)local_start.tv_sec + (double)local_start.tv_nsec/1.0e9;
  local_end_dbl   = (double)local_end.tv_sec   + (double)local_end.tv_nsec/1.0e9;
    
  if(world_rank==ROOT)
  {
    for(k = 0;k<C*H*W;++k)
      checksum += o[k];
    printf("%f, %4.3lf\n",checksum, 1000.0*(local_end_dbl-local_start_dbl));
  }

  free(ir);
  free(buf);
  free(o);
  free(request_array);
  free(status_array);


  MPI_Finalize();
  return 0;
}
