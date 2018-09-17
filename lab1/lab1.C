#include<stdio.h>
#include<time.h>
#include<stdlib.h>
#include"mkl.h"
#define len 1000000000

int main()
{
  const int timer = 16;
  double a[len];
  double dummy;
  int i;
  int t=0;
  struct timespec start, end;
  for(i=0;i<len;i++)
    a[i]=0.0;


  double spent, bw,maxbw=0;
  printf("C1:\n");
  printf("Intel(R) Xeon(R) CPU E5-2680 v4 specified max memory bandwidth: 76.8 GB/s\n");
  while(t<timer)
  {
    dummy = 0.;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for(i=0;i<len;i+=20)
    {
      dummy += (a[i]+ a[i+1]+ a[i+2]+ a[i+3]+ a[i+4]+ a[i+5]+ a[i+6]+ a[i+7]+ a[i+8]+ a[i+9]+
             a[i+10]+a[i+11]+a[i+12]+a[i+13]+a[i+14]+a[i+15]+a[i+16]+a[i+17]+a[i+18]+a[i+19]);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    spent = ((double)end.tv_sec + (double)end.tv_nsec/1.0e9) - ((double)start.tv_sec + (double)start.tv_nsec/1.0e9);
    //bw = double(sizeof(double))*len/spent;
    bw = sizeof(a)/spent;
    maxbw = maxbw>bw?maxbw:bw;
    printf("#%d run calculated: %f, measured bandwidth: %f Gigabytes/sec\n",t+1,dummy,bw/1.0e9);
    ++t;
  }
  printf("Best bandwidth over %d runs: %f Gigabytes/sec\n",timer,maxbw/1.0e9);


  printf("\nC4:\n");
  const int n_in = 50176;
  const int n0   = 4000;
  const int n1   = 1000;

  double w0[n_in][n0];
  double w1[n0][n1];
  double x0[n_in];
  double z0[n0];
  double z1[n1];

  int j;
  for(i=0; i<n_in; ++i)
    for(j=0;j<n0;++j)
      w0[i][j] = 0.5+ ((double)((i+j)%50)-30.)/50.;

  for(i=0;i<n0;++i)
    for(j=0;j<n1;++j)
      w1[i][j] = 0.5+ ((double)((i+j)%50)-30.)/50.;

  for(j=0;j<n_in;++j)
    x0[j] = 0.5+((double)(j%50)-30.)/50.;

  clock_gettime(CLOCK_MONOTONIC, &start);
  for(j=0;j<n0;++j)
  {
    for(i=0;i<n_in;++i)
      z0[j] += x0[i]*w0[i][j];
    z0[j] = z0[j]>0.?z0[j]:0.;
  }

  for(j=0;j<n1;++j)
  {
    for(i=0;i<n0;++i)
      z1[j] += z0[i]*w1[i][j];
    z1[j] = z1[j]>0.?z1[j]:0.;
  }

  clock_gettime(CLOCK_MONOTONIC,&end);
  spent = ((double)end.tv_sec + (double)end.tv_nsec/1.0e9) - ((double)start.tv_sec + (double)start.tv_nsec/1.0e9);
  printf("C inference used %f seconds\n",spent);
  double checksum =0.;
  for(j=0;j<n1;++j)
    checksum += z1[j];
  printf("C inference found checksum: %f\n", checksum);
 
  printf("\nC5:\n");
  double *w0_mkl = (double *)mkl_malloc(n_in*n0*sizeof(double),64);
  double *w1_mkl = (double *)mkl_malloc(n0*n1*sizeof(double),64);
  double *x0_mkl = (double *)mkl_malloc(n_in*sizeof(double),64);
  double *z0_mkl = (double *)mkl_malloc(n0*sizeof(double),64);
  double *z1_mkl = (double *)mkl_malloc(n1*sizeof(double),64);
  if(w0_mkl ==0 || w1_mkl==0 || x0_mkl ==0 || z0_mkl==0|| z1_mkl==0)
  {
    printf( "\n ERROR: Cannot allocate memory for matrices. Aborting... \n\n");
    mkl_free(w0_mkl);
    mkl_free(w1_mkl);
    mkl_free(x0_mkl);
    mkl_free(z0_mkl);
    mkl_free(z1_mkl);
    return 1;
  }

  int k;
  for(k=0;k<(n_in*n0);++k)
  {
    i = k/n_in;
    j = k%n_in;
    w0_mkl[k] = (0.5+ ((double)((i+j)%50)-30.)/50.);
  }

  for(k=0;k<(n0*n1);++k)
  {
    i = k/n0;
    j = k%n0;
    w1_mkl[k] =  (0.5+ ((double)((i+j)%50)-30.)/50.);
  }

  for(k=0;k<n_in;++k)
    x0_mkl[k] = (0.5+((double)(j%50)-30.)/50.);
  for(k=0;k<n0;++k)
    z0_mkl[k] = 0.;
  for(k=0;k<n1;++k)
    z1_mkl[k] = 0.;

  clock_gettime(CLOCK_MONOTONIC,&start);
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,1,n0,n_in,1.0, x0_mkl,n_in,w0_mkl,n0,0.0,z0_mkl,n0);
  for(k=0;k<n0;++k)
    z0_mkl[k] = z0_mkl[k]>0.0?z0_mkl[k]:0.0;
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,1,n1,n0,1.0,z0_mkl,n0,w1_mkl,n1,0.0,z1_mkl,n1);
  for(k=0;k<n1;++k)
    z1_mkl[k] = z1_mkl[k]>0.0?z1_mkl[k]:0.0;

  clock_gettime(CLOCK_MONOTONIC,&end);
  spent = ((double)end.tv_sec + (double)end.tv_nsec/1.0e9) - ((double)start.tv_sec + (double)start.tv_nsec/1.0e9);
  printf("C MKL inference used %f seconds\n",spent);
  double checksum_mkl =0.;
  for(k=0;k<n1;++k)
    checksum_mkl += z1[k];
  printf("C MKL inference found checksum: %f\n", checksum_mkl);
  
  mkl_free(w0_mkl);
  mkl_free(w1_mkl);
  mkl_free(x0_mkl);
  mkl_free(z0_mkl);
  mkl_free(z1_mkl);
 

  return 0;
}
