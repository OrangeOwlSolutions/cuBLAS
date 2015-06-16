#include <thrust/device_vector.h>
#include <thrust/random.h>

#include <cublas_v2.h>

#include "Utilities.cuh"

/********/
/* MAIN */
/********/
int main()
{
	/**************************/
	/* SETTING UP THE PROBLEM */
	/**************************/
  
	//const int Nrows1 = 10;			// --- Number of rows of matrix 1
	//const int Ncols1 = 10;			// --- Number of columns of matrix 1

	//const int Nrows2 = 15;			// --- Number of rows of matrix 2
	//const int Ncols2 = 15;			// --- Number of columns of matrix 2

	//const int Nrows3 = 12;			// --- Number of rows of matrix 3
	//const int Ncols3 = 12;			// --- Number of columns of matrix 3

	const int Nrows1 = 10;			// --- Number of rows of matrix 1
	const int Ncols1 = 9;			// --- Number of columns of matrix 1

	const int Nrows2 = 15;			// --- Number of rows of matrix 2
	const int Ncols2 = 13;			// --- Number of columns of matrix 2

	const int Nrows3 = 10;			// --- Number of rows of matrix 3
	const int Ncols3 = 12;			// --- Number of columns of matrix 3

	const int Nrows = 5;			// --- Number of rows of submatrix matrix 3 = Number of rows of submatrix 1
	const int Ncols = 3;			// --- Number of columns of submatrix matrix 3 = Number of columns of submatrix 2

	const int Nrowscols = 4;		// --- Number of columns of submatrix 1 and of rows of submatrix 2

	const int x1 = 3;				// --- Offset for submatrix multiplication along the rows
	const int y1 = 2;				// --- Offset for submatrix multiplication along the columns
	
	const int x2 = 6;				// --- Offset for submatrix multiplication along the rows
	const int y2 = 4;				// --- Offset for submatrix multiplication along the columns

	const int x3 = 3;				// --- Offset for submatrix multiplication along the rows
	const int y3 = 5;				// --- Offset for submatrix multiplication along the columns

	// --- Random uniform integer distribution between 0 and 100
	thrust::default_random_engine rng;
	thrust::uniform_int_distribution<int> dist(0, 20);

	// --- Matrix allocation and initialization
	thrust::device_vector<float> d_matrix1(Nrows1 * Ncols1);
	thrust::device_vector<float> d_matrix2(Nrows2 * Ncols2);
	for (size_t i = 0; i < d_matrix1.size(); i++) d_matrix1[i] = (float)dist(rng);
	for (size_t i = 0; i < d_matrix2.size(); i++) d_matrix2[i] = (float)dist(rng);

	printf("\n\nOriginal full size matrix A\n");
	for(int i = 0; i < Nrows1; i++) {
		std::cout << "[ ";
		for(int j = 0; j < Ncols1; j++) 
			std::cout << d_matrix1[j * Nrows1 + i] << " ";
		std::cout << "]\n";
	}

	printf("\n\nOriginal full size matrix B\n");
	for(int i = 0; i < Nrows2; i++) {
		std::cout << "[ ";
		for(int j = 0; j < Ncols2; j++) 
			std::cout << d_matrix2[j * Nrows2 + i] << " ";
		std::cout << "]\n";
	}

	/*************************/
	/* MATRIX MULTIPLICATION */
	/*************************/
	cublasHandle_t handle;

	cublasSafeCall(cublasCreate(&handle));

	thrust::device_vector<float> d_matrix3(Nrows3 * Ncols3, 10.f);

	float alpha = 1.f;
	float beta  = 0.f;
	cublasSafeCall(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, Nrows, Ncols, Nrowscols, &alpha,
				   thrust::raw_pointer_cast(d_matrix1.data())+x1+Nrows1*y1, Nrows1, thrust::raw_pointer_cast(d_matrix2.data())+x2+Nrows2*y2, Nrows2,
				   &beta, thrust::raw_pointer_cast(d_matrix3.data())+x3+Nrows3*y3, Nrows3));

	printf("\n\nResult full size matrix C\n");
	for(int i = 0; i < Nrows3; i++) {
		std::cout << "[ ";
		for(int j = 0; j < Ncols3; j++) 
			std::cout << d_matrix3[j * Nrows3 + i] << " ";
		std::cout << "]\n";
	}

	return 0; 
}
