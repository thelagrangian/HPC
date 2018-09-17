#include<stdio.h>
#include<cublas_v2.h>
#include<time.h>

#define CUBLAS_CALL(x) do                          \
{                                                  \
  cublasStatus_t status = (x);                     \
  if (status != CUBLAS_STATUS_SUCCESS)             \
  {                                                \
    fprintf(stderr, "%s:%d ERROR: %s\n", __FILE__, \
    __LINE__, status);                             \
    exit(-1);                                      \
  }                                                \
} while (0)


__global__
void matrix_multi_intuitive(int ar, int ac, int bc, const float*a, const float*b, float*c)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;
  if(x<ar && y<bc)
  {
    float sum = 0.0;
    for(int j=0;j<ac;++j)
      sum += a[x*ac+j]*b[j*bc+y];
    c[x*bc+y]=sum;
  }
}


__global__
void matrix_multi(int ar, int ac, int bc, const float*a, const float*b, float*c)
{
  extern __shared__ float sm[];
  float* asub = &sm[0];
  float* bsub = &sm[blockDim.x*ac];

  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;
  int lx= threadIdx.x;
  int ly= threadIdx.y;

  //bring the tiling matrix block of a into shared memory
  for(int j = ly; j<ac; j+=blockDim.y)
    if(x<ar)
      asub[lx*ac+j] = a[x*ac+j];
  //bring the tiling matrix block of b into shared memory
  for(int i = lx; i<ac; i+=blockDim.x)
    if(y<bc)
      bsub[i*blockDim.y+ly] =b[i*bc+y];
  __syncthreads();


  if(x<ar && y<bc)
  {
    float sum = 0.0;
    for(int j = 0; j<ac; ++j)
      sum += (asub[lx*ac+j]*bsub[j*blockDim.y+ly]);
    c[x*bc+y] = sum;
  }
  __syncthreads();
}


void matrix_multi_cublas(int ar, int ac, int bc, const float*a, const float*b, float*c, cublasHandle_t& handle)
{
  float alpha = 1.0;
  float beta  = 0.0;
  const float *palpha = &alpha;
  const float *pbeta  = &beta;
  //cuBlas uses column-major form when representing a matrix as an array
  //c = a*b => c^t = (a*b)^t => c^t = b^t * a^t,
  //where b^t and a^t are with column-major form
  //therefore, the parameters used in the following cublasSgemm call are for b^t and a^t
  //NOTE: c^t, a^t, and b^t are all of column-major. The following function call preserves the proper order
  CUBLAS_CALL(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, bc, ar, ac, palpha, b, bc, a, ac, pbeta, c, bc));
}


int main(int argc, char*argv[])
{
  //define the sizes of matrices
  int ar = 17;
  int ac = 23;
  int bc = 29;


  //a(ar x ac)*b(ac x bc) = c(ar x bc)
  //memory on host
  float* a = (float*)malloc(sizeof(float)*ar*ac);
  float* b = (float*)malloc(sizeof(float)*ac*bc);
  float* c1= (float*)malloc(sizeof(float)*ar*bc);
  float* c2= (float*)malloc(sizeof(float)*ar*bc);
  float* c3= (float*)malloc(sizeof(float)*ar*bc);


  int blockx = (1<<5);
  int blocky = (1<<5);

  //memory on device
  float *dev_a, *dev_b, *dev_c1, *dev_c2, *dev_c3;
  cudaMalloc(&dev_a, sizeof(float)*ar*ac);
  cudaMalloc(&dev_b, sizeof(float)*ac*bc);
  cudaMalloc(&dev_c1,sizeof(float)*ar*bc);
  cudaMalloc(&dev_c2,sizeof(float)*ar*bc);
  cudaMalloc(&dev_c3,sizeof(float)*ar*bc);


  for(int i=0;i<ar;++i)
    for(int j=0;j<ac;++j)
      a[i*ac+j]=2*i+j+2;

  for(int i=0;i<ac;++i)
    for(int j=0;j<bc;++j)
      b[i*bc+j]=(i+1)*(3*j+1);

  cudaMemcpy(dev_a, a, sizeof(float)*ar*ac, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b, b, sizeof(float)*ac*bc, cudaMemcpyHostToDevice);

  //dynmem: the size of dynamically allocated shared memory
  int dynmem = sizeof(float)*(ac*blockx+blocky*ac);
  int gridx = (ar+blockx-1)/blockx;
  int gridy = (bc+blocky-1)/blocky;

  dim3 blockD(blockx, blocky);
  dim3 gridD(gridx, gridy);
  struct timespec ts1, te1, ts2, te2, ts3, te3;
  double t1, t2, t3;
  double checksum1 = 0., checksum2 = 0., checksum3 = 0.;


  //matrix multiplication with intuitive algorithm
  clock_gettime(CLOCK_MONOTONIC, &ts1);
  matrix_multi_intuitive<<<gridD, blockD>>>(ar, ac, bc, dev_a, dev_b, dev_c1);
  cudaDeviceSynchronize();
  clock_gettime(CLOCK_MONOTONIC, &te1);
  t1 = ((double)te1.tv_sec +(double)te1.tv_nsec/1.0e9)-((double)ts1.tv_sec+(double)ts1.tv_nsec/1.0e9);

  //improved matrix multiplication with shared memory and tiling
  clock_gettime(CLOCK_MONOTONIC, &ts2);
  matrix_multi<<<gridD,blockD,dynmem>>>(ar, ac, bc, dev_a, dev_b, dev_c2);
  cudaDeviceSynchronize();
  clock_gettime(CLOCK_MONOTONIC, &te2);
  t2 = ((double)te2.tv_sec +(double)te2.tv_nsec/1.0e9)-((double)ts2.tv_sec+(double)ts2.tv_nsec/1.0e9);

  //matrix multiplication with cuBlas
  cublasHandle_t handle;
  cublasCreate(&handle);
  clock_gettime(CLOCK_MONOTONIC, &ts3);
  matrix_multi_cublas(ar, ac, bc, dev_a, dev_b, dev_c3, handle);
  clock_gettime(CLOCK_MONOTONIC, &te3);
  t3 = ((double)te3.tv_sec +(double)te3.tv_nsec/1.0e9)-((double)ts3.tv_sec+(double)ts3.tv_nsec/1.0e9);
  cublasDestroy(handle);


  cudaMemcpy(c1, dev_c1, sizeof(float)*ar*bc, cudaMemcpyDeviceToHost);
  cudaMemcpy(c2, dev_c2, sizeof(float)*ar*bc, cudaMemcpyDeviceToHost);
  cudaMemcpy(c3, dev_c3, sizeof(float)*ar*bc, cudaMemcpyDeviceToHost);

  for(int i=0;i<ar*bc;++i)
  {
    checksum1 += c1[i];
    checksum2 += c2[i];
    checksum3 += c3[i];
  }

  //time check is in terms of second
  printf("%f,%f\n",checksum1,t1);
  printf("%f,%f\n",checksum2,t2);
  printf("%f,%f\n",checksum3,t3);

  //the array form of c3 matrix is already in the proper order
  /*
  for(int i=0;i<ar;++i)
  {
    for(int j=0;j<bc;++j)
      printf("%f ",c3[i*bc+j]);
    printf("\n");
  }
  */


  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c1);
  cudaFree(dev_c2);
  cudaFree(dev_c3);
  free(a);
  free(b);
  free(c1);
  free(c2);
  free(c3);

  return 0;
}
