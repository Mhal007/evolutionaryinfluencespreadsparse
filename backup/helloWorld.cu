#include <stdio.h>
#include <iostream>

using namespace std;

const int N = 16;
const int blocksize = 16;

__global__
void hello(char *a, int *b)
{
	a[threadIdx.x] += b[threadIdx.x];
}

int main()
{
	char a[N] = "Hello \0\0\0\0\0\0";
	int b[N] = {15, 10, 6, 0, -11, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

	char *ad;
	int *bd;
	const int csize = N*sizeof(char);
	const int isize = N*sizeof(int);

	printf("%s", a);

    cudaError_t err1 = cudaMalloc( (void**)&ad, csize );
    cudaError_t err2 = cudaMalloc( (void**)&bd, isize );
	

    if (err1 != cudaSuccess) {
        cout << "Error allocating memory for ad." << endl;
        printf("CUDA error: %s\n", cudaGetErrorString(err1));
        return 0;
    }
    if (err2 != cudaSuccess) {
        cout << "Error allocating memory for bd." << endl;
        printf("CUDA error: %s\n", cudaGetErrorString(err2));
        return 0;
    }
    
	
	cudaMemcpy( ad, a, csize, cudaMemcpyHostToDevice );
	cudaMemcpy( bd, b, isize, cudaMemcpyHostToDevice );

	dim3 dimBlock( blocksize, 1 );
	dim3 dimGrid( 1, 1 );
	hello<<<dimGrid, dimBlock>>>(ad, bd);
	cudaMemcpy( a, ad, csize, cudaMemcpyDeviceToHost );
	cudaFree( ad );
	cudaFree( bd );

	printf("%s\n", a);
	return EXIT_SUCCESS;
}
