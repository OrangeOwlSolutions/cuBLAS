include <cublas_v2.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/random.h>
#include <thrust/sequence.h>

#include <stdio.h>
#include <iostream>

#include "Utilities.cuh"
#include "TimingGPU.cuh"

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

/***********************************************************/
/* SQUARED ABSOLUTE VALUE FUNCTOR - NEEDED FOR APPROACH #1 */
/***********************************************************/
struct abs2 {
	__host__ __device__ double operator()(const float &x) const { return x * x; }
};

// --- Required for approach #2
__device__ float *vals;

/******************************************/
/* ROW_REDUCTION - NEEDED FOR APPROACH #2 */
/******************************************/
struct row_reduction {

    const int Ncols;    // --- Number of columns

    row_reduction(int _Ncols) : Ncols(_Ncols) {}

    __device__ float operator()(float& x, int& y ) {
        float temp = 0.f;
        for (int i = 0; i<Ncols; i++)
            temp += vals[i + (y*Ncols)] * vals[i + (y*Ncols)];
        return temp;
    }
};

/************************************************/
/* KERNEL FUNCTION TO ASSEMBLE THE FINAL RESULT */
/************************************************/
__global__ void assemble_final_result(const float * __restrict__ d_norms_x_2, const float * __restrict__ d_norms_y_2, float * __restrict__ d_dots,
									  const int NX, const int NY) {

	const int i = threadIdx.x + blockIdx.x * gridDim.x;
	const int j = threadIdx.y + blockIdx.y * gridDim.y;

	if ((i < NY) && (j < NX)) d_dots[i * NX+ j] = d_norms_x_2[j] + d_norms_y_2[i] - 2 * d_dots[i * NX+ j];

}

/********/
/* MAIN */
/********/
int main()
{
    //const int Ndims = 128;		// --- Number of rows
    //const int NX	= 1000;		// --- Number of columns
    //const int NY	= 2000;		// --- Number of columns

    const int Ndims = 3;		// --- Number of rows
    const int NX	= 4;		// --- Number of columns
    const int NY	= 5;		// --- Number of columns

	// --- Random uniform integer distribution between 10 and 99
    thrust::default_random_engine rng;
    thrust::uniform_int_distribution<int> dist(10, 99);

    // --- Matrices allocation and initialization
    thrust::device_vector<float> d_X(Ndims * NX);
    thrust::device_vector<float> d_Y(Ndims * NY);
    for (size_t i = 0; i < d_X.size(); i++) d_X[i] = (float)dist(rng);
    for (size_t i = 0; i < d_Y.size(); i++) d_Y[i] = (float)dist(rng);

    TimingGPU timerGPU;

	// --- cuBLAS handle creation
	cublasHandle_t handle;
    cublasSafeCall(cublasCreate(&handle));

	/**********************************************/
    /* CALCULATING THE NORMS OF THE ELEMENTS OF X */
    /**********************************************/
    thrust::device_vector<float> d_norms_x_2(NX);

	// --- Approach nr. 1
	//timerGPU.StartCounter();
	thrust::device_vector<float> d_X_2(Ndims * NX);
	thrust::transform(d_X.begin(), d_X.end(), d_X_2.begin(), abs2());

	thrust::device_vector<float> d_ones(Ndims, 1.f);

    float alpha = 1.f;
    float beta  = 0.f;
    cublasSafeCall(cublasSgemv(handle, CUBLAS_OP_T, Ndims, NX, &alpha, thrust::raw_pointer_cast(d_X_2.data()), Ndims, 
                               thrust::raw_pointer_cast(d_ones.data()), 1, &beta, thrust::raw_pointer_cast(d_norms_x_2.data()), 1));
	
	//printf("Timing for approach #1 = %f\n", timerGPU.GetCounter());

    // --- Approach nr. 2
	//timerGPU.StartCounter();
 //   float *s_vals = thrust::raw_pointer_cast(&d_X[0]);
 //   gpuErrchk(cudaMemcpyToSymbol(vals, &s_vals, sizeof(float *)));
 //   thrust::transform(d_norms_x_2.begin(), d_norms_x_2.end(), thrust::counting_iterator<int>(0),  d_norms_x_2.begin(), row_reduction(Ndims));

	//printf("Timing for approach #2 = %f\n", timerGPU.GetCounter());

	/**********************************************/
    /* CALCULATING THE NORMS OF THE ELEMENTS OF Y */
    /**********************************************/
    thrust::device_vector<float> d_norms_y_2(NX);

	thrust::device_vector<float> d_Y_2(Ndims * NX);
	thrust::transform(d_Y.begin(), d_Y.end(), d_Y_2.begin(), abs2());

    cublasSafeCall(cublasSgemv(handle, CUBLAS_OP_T, Ndims, NY, &alpha, thrust::raw_pointer_cast(d_Y_2.data()), Ndims, 
                               thrust::raw_pointer_cast(d_ones.data()), 1, &beta, thrust::raw_pointer_cast(d_norms_y_2.data()), 1));


	/***********************************/
    /* CALCULATING THE SCALAR PRODUCTS */
    /***********************************/
    thrust::device_vector<float> d_dots(NX * NY);

	cublasSafeCall(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, NX, NY, Ndims, &alpha,
		                       thrust::raw_pointer_cast(d_X.data()), Ndims, thrust::raw_pointer_cast(d_Y.data()), Ndims, &beta,
							   thrust::raw_pointer_cast(d_dots.data()), NX));

	/*****************************/
	/* ASSEMBLE THE FINAL RESULT */
	/*****************************/
	
	dim3 dimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);
	dim3 dimGrid(iDivUp(NX, BLOCK_SIZE_X), iDivUp(NY, BLOCK_SIZE_Y));
	assemble_final_result<<<dimGrid, dimBlock>>>(thrust::raw_pointer_cast(d_norms_x_2.data()), thrust::raw_pointer_cast(d_norms_y_2.data()), 
		                                         thrust::raw_pointer_cast(d_dots.data()), NX, NY);
	
	for(int i = 0; i < NX * NY; i++) std::cout << d_dots[i] << "\n";

	return 0;
}
