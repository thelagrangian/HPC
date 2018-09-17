#include<iostream>
#include<stdio.h>
#include<time.h>
#include<iomanip>
#include<cudnn.h>
using namespace std;

#define BLOCK_Z 5
#define BLOCK_H (1<<3)
#define BLOCK_W (1<<3)

#define CUDNN_CALL(x) do                           \
{                                                  \
  cudnnStatus_t status = (x);                      \
  if (status != CUDNN_STATUS_SUCCESS)              \
  {                                                \
    fprintf(stderr, "%s:%d ERROR: %s\n", __FILE__, \
    __LINE__, cudnnGetErrorString(status));        \
    exit(-1);                                      \
  }                                                \
} while (0)


//device code, intuitive implementation of convolution
__global__
void conv(int C, int IH, int IW, int K, int FH, int FW, int OH, int OW, float *in, float*fil, float*out)
{
  int x = blockIdx.y * blockDim.y + threadIdx.y;
  int y = blockIdx.z * blockDim.z + threadIdx.z;
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  float sum=0.0;
  if(k<K && x<OH && y<OW)
  {
    for(int c=0;c<C;++c)
    {
      for(int i=0;i<FH;++i)
      {
        for(int j=0;j<FW;++j)
        {
          sum += fil[FW-1-j+FW*(FH-1-i+FH*(c+C*k))]*in[y+j+IW*(x+i+c*IH)];
        }
      }
    }
    out[y+OW*(x+OH*k)] = sum;
  }
  else{}
  
}

//device code, convolution with shared memory and tiling, using dynamic shared memory
__global__
void conv_tiling(int C, int IH, int IW, int K, int FH, int FW, int OH, int OW, float*in, float*fil, float*out)
{
  extern __shared__ float sm[];
  int x = blockIdx.y * blockDim.y + threadIdx.y;
  int y = blockIdx.z * blockDim.z + threadIdx.z;
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  int lx= threadIdx.y;
  int ly= threadIdx.z;
  int lk= threadIdx.x;

  float sum=0.0;
  //K*C*FH*FW + C*(BLOCK_H+FH-1)*(BLOCK_W+FW-1)
  float *fil_s = &sm[0];
  float *in_s  = &sm[K*C*FH*FW];

  //bringing in filter tensor to shared memory (entire tensor)
  if(lx<FH && ly<FW)
  {
    for(int l=0;l<(K*C+blockDim.x-1)/blockDim.x;++l)
    {
      int tk = lk+l*blockDim.x;
      if(tk < K*C)
        fil_s[ly+FW*(lx+FH*tk)]=fil[FW-1-ly+FW*(FH-1-lx+FH*tk)];
    }
  }
  __syncthreads();

  //bringing in input tensor to shared memory (only the portion related to the current block)
  int LH = BLOCK_H+FH-1;
  int LW = BLOCK_W+FW-1;
  if(lk<C)
  {
    for(int i=0;i<(LH+blockDim.y-1)/blockDim.y;++i)
    {
      for(int j=0;j<(LW+blockDim.z-1)/blockDim.z;++j)
      {
        int tx = lx + i*blockDim.y;
        int ty = ly + j*blockDim.z;
        if(tx<LH && ty<LW)
          in_s[ty+LW*(tx+LH*lk)]=in[y+BLOCK_W*j+IW*(x+BLOCK_H*i+IH*k)];
      }
    }
  }
  __syncthreads();

  if(k<K && x<OH && y<OW)
  {
    for(int c=0;c<C;++c)
    {
      for(int i=0;i<FH;++i)
      {
        for(int j=0;j<FW;++j)
          sum += fil_s[j+FW*(i+FH*(c+C*k))]*in_s[ly+j+LW*(lx+i+LH*c)];
      }
    }
    out[x+OW*(y+OH*k)]=sum;
  }
  __syncthreads();
 
}


//host code
int main()
{
  int H = 1024; //row# input
  int W = 1024; //col# input
  int C = 3;
  int K = 5;
  int FH= 3;    //row# filter
  int FW= 3;    //row# filter
  int padding = 1;//2;
  int OH = H + 2*padding - FH + 1; //row# output
  int OW = W + 2*padding - FW + 1; //col# output
  float *in = (float*)malloc(C*(H+2*padding)*(W+2*padding)*sizeof(float));
  float *fil= (float*)malloc(K*C*FH*FW*sizeof(float));
  float *out= (float*)malloc(K*OH*OW*sizeof(float));
  float *out2=(float*)malloc(K*OH*OW*sizeof(float));
  float *out3=(float*)malloc(K*OH*OW*sizeof(float));
  //input tensor assignment, including padding elements
  for(int c=0;c<C;++c)
  {
    for(int i=0;i<H+2*padding;++i)
    {
      for(int j=0;j<W+2*padding;++j)
      {
        if(padding<=i && i<H+padding && padding<=j && j<W+padding)
          in[(c*(H+2*padding)+i)*(W+2*padding)+j]=1.0*c*((i-padding)+(j-padding));
        else
          in[(c*(H+2*padding)+i)*(W+2*padding)+j]=0.0;
      }
    }
  }
  //filter tensor assignment
  for(int k=0;k<K;++k)
  {
    for(int c=0;c<C;++c)
    {
      for(int i=0;i<FH;++i)
      {
        for(int j=0;j<FW;++j)
          fil[j+FW*(i+FH*(c+C*k))]= 1.0*(c+k)*(i+j);
      }
    }
  }
  
  float*in_d;
  float*fil_d;
  float*out_d;
  float*out2_d;
  float*out3_d;
  cudaMalloc(&in_d,  C*(H+2*padding)*(W+2*padding)*sizeof(float));
  cudaMalloc(&fil_d, K*C*FH*FW*sizeof(float));
  cudaMalloc(&out_d, K*OH*OW*sizeof(float));
  cudaMalloc(&out2_d,K*OH*OW*sizeof(float));
  cudaMalloc(&out3_d,K*OH*OW*sizeof(float));

  cudaMemcpy(in_d, in,  C*(H+2*padding)*(W+2*padding)*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(fil_d,fil, K*C*FH*FW*sizeof(float),cudaMemcpyHostToDevice);

  dim3 blockD(BLOCK_Z,BLOCK_H,BLOCK_W);
  int gridx = (K   + blockD.x - 1)/blockD.x;
  int gridy = (OH  + blockD.y - 1)/blockD.y;
  int gridz = (OW  + blockD.z - 1)/blockD.z;
  dim3 gridD(gridx,gridy,gridz);

  /*
  cout<<"grid dimension:\n";
  cout<<gridx<<", "<<gridy<<", "<<gridz<<endl;
  cout<<"block dimension:\n";
  cout<<blockD.x<<", "<<blockD.y<<", "<<blockD.z<<endl;
  */

  struct timespec tstart, tend;
  struct timespec tstart2,tend2;
  struct timespec tstart3,tend3;
  double tspent,tspent2,tspent3;

  //without tiling:
  clock_gettime(CLOCK_MONOTONIC, &tstart);
  conv<<<gridD, blockD>>>(C, (H+2*padding), (W+2*padding), K, FH, FW, OH, OW, in_d, fil_d, out_d);
  cudaDeviceSynchronize();
  clock_gettime(CLOCK_MONOTONIC, &tend);
  tspent = ((double)tend.tv_sec +(double)tend.tv_nsec/1.0e9)-((double)tstart.tv_sec+(double)tstart.tv_nsec/1.0e9);


  //with tiling, dynamic shared memory:
  int dynmem = (K*C*FH*FW + C*(BLOCK_H+FH-1)*(BLOCK_W+FW-1))*sizeof(float);
  clock_gettime(CLOCK_MONOTONIC, &tstart2);
  conv_tiling<<<gridD, blockD, dynmem>>>(C, (H+2*padding), (W+2*padding), K, FH, FW, OH, OW, in_d, fil_d, out2_d);
  cudaDeviceSynchronize();
  clock_gettime(CLOCK_MONOTONIC, &tend2);
  tspent2 = ((double)tend2.tv_sec +(double)tend2.tv_nsec/1.0e9)-((double)tstart2.tv_sec+(double)tstart2.tv_nsec/1.0e9);

  //using cudnn
  cudnnHandle_t cudnn;
  CUDNN_CALL(cudnnCreate(&cudnn));

  cudnnTensorDescriptor_t input_dp;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&input_dp));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(input_dp,
                                        CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT,
                                        1,
                                        C,
                                        (H+2*padding),
                                        (W+2*padding)));

  cudnnTensorDescriptor_t output_dp;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&output_dp));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(output_dp,
                                        CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT,
                                        1,
                                        K,
                                        OH,
                                        OW));

  cudnnFilterDescriptor_t fil_dp;
  CUDNN_CALL(cudnnCreateFilterDescriptor(&fil_dp));
  CUDNN_CALL(cudnnSetFilter4dDescriptor(fil_dp,
                                        CUDNN_DATA_FLOAT,
                                        CUDNN_TENSOR_NCHW,
                                        K,
                                        C,
                                        FH,
                                        FW));


  cudnnConvolutionDescriptor_t conv_dp;
  CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_dp));
  CUDNN_CALL(cudnnSetConvolution2dDescriptor(conv_dp,
                                             0,
                                             0,
                                             1,
                                             1,
                                             1,
                                             1,
                                             CUDNN_CONVOLUTION,
                                             CUDNN_DATA_FLOAT));

  cudnnConvolutionFwdAlgo_t conv_algo= CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
  /*
  CUDNN_CALL(cudnnGetConvolutionForwardAlgorithm(cudnn,
                                                 input_dp,
                                                 fil_dp,
                                                 conv_dp,
                                                 output_dp,
                                                 CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                 0,
                                                 &conv_algo));
  */

  size_t wsp_size = 0;
  CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
                                                     input_dp,
                                                     fil_dp,
                                                     conv_dp,
                                                     output_dp,
                                                     conv_algo,
                                                     &wsp_size));

  void*wsp_d;
  cudaMalloc(&wsp_d, wsp_size);

  const float alpha = 1, beta=0;
  clock_gettime(CLOCK_MONOTONIC, &tstart3);
  CUDNN_CALL(cudnnConvolutionForward(cudnn,
                                     &alpha,
                                     input_dp,
                                     in_d,
                                     fil_dp,
                                     fil_d,
                                     conv_dp,
                                     conv_algo,
                                     wsp_d,
                                     wsp_size,
                                     &beta,
                                     output_dp,
                                     out3_d));
  cudaDeviceSynchronize();
  clock_gettime(CLOCK_MONOTONIC, &tend3);
  tspent3 = ((double)tend3.tv_sec +(double)tend3.tv_nsec/1.0e9)-((double)tstart3.tv_sec+(double)tstart3.tv_nsec/1.0e9);

  cudaMemcpy(out,  out_d,K*OH*OW*sizeof(float),cudaMemcpyDeviceToHost);
  cudaMemcpy(out2,out2_d,K*OH*OW*sizeof(float),cudaMemcpyDeviceToHost);
  cudaMemcpy(out3,out3_d,K*OH*OW*sizeof(float),cudaMemcpyDeviceToHost);

  //float checksum=0.0, checksum2=0.0, checksum3=0.0;
  double checksum=0.0, checksum2=0.0, checksum3=0.0;
  for(int k=0;k<K;++k)
  {
    for(int i=0;i<OH;++i)
    {
      for(int j=0;j<OW;++j)
      {
        checksum  += out[j+OW*(i+OH*k)];
        checksum2 +=out2[j+OW*(i+OH*k)];
        checksum3 +=out3[j+OW*(i+OH*k)];
      }
    }
  }

  //%4.3lf
  printf("%f,%4.3lf\n",checksum , tspent*1000);
  printf("%f,%4.3lf\n",checksum2, tspent2*1000);
  printf("%f,%4.3lf\n",checksum3, tspent3*1000);
  //printf("%f,%f\n",checksum , tspent);
  //printf("%f,%f\n",checksum2, tspent2);
  //printf("%f,%f\n",checksum3, tspent3);

  cudaFree(in_d);
  cudaFree(fil_d);
  cudaFree(out_d);
  cudaFree(out2_d);
  cudaFree(out3_d);
  cudaFree(wsp_d);
  free(in);
  free(fil);
  free(out);
  free(out2);
  free(out3);

  cudnnDestroyTensorDescriptor(input_dp);
  cudnnDestroyTensorDescriptor(output_dp);
  cudnnDestroyFilterDescriptor(fil_dp);
  cudnnDestroyConvolutionDescriptor(conv_dp);

  cudnnDestroy(cudnn);
  return 0;
}
