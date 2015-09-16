#include <thrust/device_vector.h>
#include <thrust/random.h>

#include <stdio.h>
#include <iostream>

#include "Utilities.cuh"

/********/
/* MAIN */
/********/
int main()
{
    const int N_basis_functions = 5;     // --- Number of rows					-> Number of basis functions
    const int N_sampling_points = 8;     // --- Number of columns				-> Number of sampling points of the basis functions

    // --- Random uniform integer distribution between 10 and 99
    thrust::default_random_engine rng;
    thrust::uniform_int_distribution<int> dist(10, 99);

    // --- Matrix allocation and initialization
    thrust::device_vector<float> d_basis_functions_real(N_basis_functions * N_sampling_points);
    for (size_t i = 0; i < d_basis_functions_real.size(); i++) d_basis_functions_real[i] = (float)dist(rng);

    thrust::device_vector<double> d_basis_functions_double_real(N_basis_functions * N_sampling_points);
    for (size_t i = 0; i < d_basis_functions_double_real.size(); i++) d_basis_functions_double_real[i] = (double)dist(rng);

	/************************************/
    /* COMPUTING THE LINEAR COMBINATION */
    /************************************/
    cublasHandle_t handle;
    cublasSafeCall(cublasCreate(&handle));

    thrust::device_vector<float>  d_linear_combination_real(N_sampling_points);
    thrust::device_vector<double> d_linear_combination_double_real(N_sampling_points);
    thrust::device_vector<float>  d_coeff_real(N_basis_functions, 1.f);
    thrust::device_vector<double> d_coeff_double_real(N_basis_functions, 1.);

	linearCombination(thrust::raw_pointer_cast(d_coeff_real.data()), thrust::raw_pointer_cast(d_basis_functions_real.data()), thrust::raw_pointer_cast(d_linear_combination_real.data()),
	                  N_basis_functions, N_sampling_points, handle);
	linearCombination(thrust::raw_pointer_cast(d_coeff_double_real.data()), thrust::raw_pointer_cast(d_basis_functions_double_real.data()), thrust::raw_pointer_cast(d_linear_combination_double_real.data()),
	                  N_basis_functions, N_sampling_points, handle);
					   
	/*************************/
    /* DISPLAYING THE RESULT */
    /*************************/
    std::cout << "Real case \n\n";
	for(int j = 0; j < N_sampling_points; j++) {
        std::cout << "Column " << j << " - [ ";
        for(int i = 0; i < N_basis_functions; i++)
            std::cout << d_basis_functions_real[i * N_sampling_points + j] << " ";
        std::cout << "] = " << d_linear_combination_real[j] << "\n";
    }

    std::cout << "\n\nDouble real case \n\n";
	for(int j = 0; j < N_sampling_points; j++) {
        std::cout << "Column " << j << " - [ ";
        for(int i = 0; i < N_basis_functions; i++)
            std::cout << d_basis_functions_double_real[i * N_sampling_points + j] << " ";
        std::cout << "] = " << d_linear_combination_double_real[j] << "\n";
    }

	return 0;
}
