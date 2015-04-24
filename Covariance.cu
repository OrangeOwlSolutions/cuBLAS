#include <cublas_v2.h>

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

/*************************************/
/* CONVERT LINEAR INDEX TO ROW INDEX */
/*************************************/
template <typename T>
struct linear_index_to_row_index : public thrust::unary_function<T,T> {
	
	T Ncols; // --- Number of columns
  
	__host__ __device__ linear_index_to_row_index(T Ncols) : Ncols(Ncols) {}

	__host__ __device__ T operator()(T i) { return i / Ncols; }
};

/********/
/* MAIN */
/********/
int main()
{
    const int Nsamples = 3;		// --- Number of realizations for each random variable (number of rows of the X matrix)
    const int NX	= 4;		// --- Number of random variables (number of columns of the X matrix)

	// --- Random uniform integer distribution between 10 and 99
    thrust::default_random_engine rng;
    thrust::uniform_int_distribution<int> dist(10, 99);

    // --- Matrix allocation and initialization
    thrust::device_vector<float> d_X(Nsamples * NX);
    for (size_t i = 0; i < d_X.size(); i++) d_X[i] = (float)dist(rng);

	// --- cuBLAS handle creation
	cublasHandle_t handle;
    cublasSafeCall(cublasCreate(&handle));

	/*************************************************/
    /* CALCULATING THE MEANS OF THE RANDOM VARIABLES */
	/*************************************************/
    // --- Array containing the means multiplied by Nsamples
	thrust::device_vector<float> d_means(NX);

	thrust::device_vector<float> d_ones(Nsamples, 1.f);

    float alpha = 1.f / (float)Nsamples;
    float beta  = 0.f;
    cublasSafeCall(cublasSgemv(handle, CUBLAS_OP_T, Nsamples, NX, &alpha, thrust::raw_pointer_cast(d_X.data()), Nsamples, 
                               thrust::raw_pointer_cast(d_ones.data()), 1, &beta, thrust::raw_pointer_cast(d_means.data()), 1));
	
	/**********************************************/
    /* SUBTRACTING THE MEANS FROM THE MATRIX ROWS */
	/**********************************************/
	thrust::transform(
                d_X.begin(), d_X.end(),
                thrust::make_permutation_iterator(
                        d_means.begin(),
                        thrust::make_transform_iterator(thrust::make_counting_iterator(0), linear_index_to_row_index<int>(Nsamples))),
                d_X.begin(),
				thrust::minus<float>());	
	
	/*************************************/
    /* CALCULATING THE COVARIANCE MATRIX */
	/*************************************/
    thrust::device_vector<float> d_cov(NX * NX);

    alpha = 1.f;
	cublasSafeCall(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, NX, NX, Nsamples, &alpha,
		                       thrust::raw_pointer_cast(d_X.data()), Nsamples, thrust::raw_pointer_cast(d_X.data()), Nsamples, &beta,
							   thrust::raw_pointer_cast(d_cov.data()), NX));

	// --- Final normalization by Nsamples - 1
	thrust::transform(
                d_cov.begin(), d_cov.end(),
                thrust::make_constant_iterator((float)(Nsamples-1)),
                d_cov.begin(),
				thrust::divides<float>());	

	for(int i = 0; i < NX * NX; i++) std::cout << d_cov[i] << "\n";

	return 0;
}
