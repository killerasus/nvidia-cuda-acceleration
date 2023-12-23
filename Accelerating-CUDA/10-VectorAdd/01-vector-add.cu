#include <stdio.h>
#include <assert.h>

inline cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}


void initWith(float num, float *a, int N)
{
  for(int i = 0; i < N; ++i)
  {
    a[i] = num;
  }
}

__global__ void addVectorsInto(float *result, float *a, float *b, int N)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  
  for(int i = index; i < N; i += stride)
    result[i] = a[i] + b[i];
}

void checkElementsAre(float target, float *array, int N)
{
  for(int i = 0; i < N; i++)
  {
    if(array[i] != target)
    {
      printf("FAIL: array[%d] - %0.0f does not equal %0.0f\n", i, array[i], target);
      exit(1);
    }
  }
  printf("SUCCESS! All values added correctly.\n");
}

int main()
{
  const int N = 2<<20;
  size_t size = N * sizeof(float);

  float *a;
  float *b;
  float *c;

  if(checkCuda(cudaMallocManaged(&a, size)) != cudaSuccess)
    exit(1);
    
  if(checkCuda(cudaMallocManaged(&b, size)) != cudaSuccess)
  {
    cudaFree(a);
    exit(1);
  }
  
  if(checkCuda(cudaMallocManaged(&c, size)) != cudaSuccess)
  {
    cudaFree(a);
    cudaFree(b);
    exit(1);
  }

  initWith(3, a, N);
  initWith(4, b, N);
  initWith(0, c, N);

  size_t threads_per_block = 256;

  // Ensure there are at least `N` threads in the grid, but only 1 block's worth extra
  size_t number_of_blocks = (N + threads_per_block - 1) / threads_per_block;

  cudaError_t syncErr, asyncErr;
  addVectorsInto<<<number_of_blocks, threads_per_block>>>(c, a, b, N);

  syncErr = cudaGetLastError();
  asyncErr = cudaDeviceSynchronize();

  if(checkCuda(syncErr) != cudaSuccess || checkCuda(asyncErr) != cudaSuccess)
  {
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    exit(1);
  }

  checkElementsAre(7, c, N);

  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
}
