
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>

#include <Windows.h>

using namespace std;

//Cria a matriz a ser utilizada
__host__ int * criaMatriz(int n);

//Imprime os valores da matriz utilizada
__host__ void imprimirMatriz(int * Mat, int n);

//Gera a matatriz transpota
__host__ int * criaTransposta(int * mat, int n);

//Multiplica Matriz
__host__ int * MultiplicaMatriz(int * mat1, int * mat2, int n);

//Cria matriz transposta em GPU
__global__ void criaTranspostaGPU(int *mat, int * matT, int n);

//Multiplica Matriz em GPU
__global__ void MultiplicaMatrizGPU(int *mat1, int * mat2, int * matR, int n);


int main()
{

	//Colhendo o numero de linhas e colunas da matriz
	cout << "Digite o número de linhas e colunas da matriz (1-9): ";
	char cn = getchar();
	int n = cn - '0';
	cout<<"Gerando matrizes "<<n<<"x"<<n<< endl;

	//Matrizes geradas
	int * h_mat1 = criaMatriz(n);
	int * h_mat2 = criaMatriz(n);

	
	LARGE_INTEGER freq;
	QueryPerformanceFrequency(&freq);
	

	LARGE_INTEGER time1, time2;
	QueryPerformanceCounter(&time1);

	int * h_mat2T = criaTransposta(h_mat2, n);
	int * h_matR = MultiplicaMatriz(h_mat1,h_mat2T, n);
		
	QueryPerformanceCounter(&time2);

	double interv = static_cast<double>(time2.QuadPart - time1.QuadPart);// / freq.QuadPart;
	cout << "TempoGasto em CPU: " << interv << endl;

	

	//Imprimindo as matrizes 
	cout << "Resultado em CPU " << endl;
	cout << "Matriz 1: " << endl;
	imprimirMatriz(h_mat1, n);
	cout << "Matriz 2: " << endl;
	imprimirMatriz(h_mat2, n);
	cout << "Matriz 2T: " << endl;
	imprimirMatriz(h_mat2T, n);
	cout << "Matriz Resultado: " << endl;
	imprimirMatriz(h_matR, n);


	//Realizando operação em GPU

	

	//Criando matrizes de GPU
	int* d_mat1;
	int* d_mat2;
	int* d_mat2T;
	int* d_matR;

	cudaMalloc((void **)& d_mat1, n*n*sizeof(int));
	cudaMalloc((void **)& d_mat2, n*n*sizeof(int));
	cudaMalloc((void **)& d_mat2T, n*n*sizeof(int));
	cudaMalloc((void **)& d_matR, n*n*sizeof(int));
	
	

	cudaMemcpy(d_mat1, h_mat1, n*n*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_mat2, h_mat2, n*n*sizeof(int), cudaMemcpyHostToDevice);

	LARGE_INTEGER freq2;
	QueryPerformanceFrequency(&freq2);

	LARGE_INTEGER time12, time22;
	QueryPerformanceCounter(&time12);

	//Calculando o tamanho do bloco, como o numero só captura até 9, camos usar blocos de 16

	dim3 blockSize(16, 16, 1);

	//Cria matriz transposta em GPU
	criaTranspostaGPU <<<1, blockSize >>>(d_mat2, d_mat2T, n);

	//Multiplica Matrizes
	MultiplicaMatrizGPU <<<1, blockSize >>>(d_mat1, d_mat2T, d_matR, n);







	QueryPerformanceCounter(&time22);


	int * h_mat2TG = (int *)malloc(n * n * sizeof(int));
	int * h_mat2RG = (int *)malloc(n * n * sizeof(int));
	cudaMemcpy(h_mat2TG, d_mat2T, n*n*sizeof(int), cudaMemcpyDeviceToHost);
	//cudaMemcpy(d_mat2, h_mat2, n*n*sizeof(int), cudaMemcpyHostToDevice);

	interv = static_cast<double>(time22.QuadPart - time12.QuadPart);// / freq.QuadPart;
	cout << "TempoGasto em GPU: " << interv << endl;
	//Imprimindo as matrizes 
	cout << "Resultado em GPU " << endl;
	cout << "Matriz 1: " << endl;
	imprimirMatriz(h_mat1, n);
	cout << "Matriz 2: " << endl;
	imprimirMatriz(h_mat2, n);
	cout << "Matriz 2T: " << endl;
	imprimirMatriz(h_mat2TG, n);
	cout << "Matriz Resultado: " << endl;
	imprimirMatriz(h_matR, n);


	getchar();
	getchar();

    return 0;
}

__global__ void MultiplicaMatrizGPU(int *mat1, int * mat2, int * matR, int n)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (y < n && x < n) {
		int sum = 0;
		for (int z = 0; z < n; z++)
		{
			sum += mat1[(x*n) + z] * mat2[(z*n) + y];
		}
		matR[(x*n) + y] = sum;
	}

}

__global__ void criaTranspostaGPU(int *mat, int * matT, int n)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (y < n && x < n) {
		matT[(y*n) + x] = mat[(x*n) + y];
	}

}


__host__ int * MultiplicaMatriz(int * mat1, int * mat2, int n)
{

	int * result = (int *)malloc(n * n * sizeof(int));

	for (int x = 0; x < n; x++)
	{
		
		for (int y = 0; y < n; y++)
		{

			int sum = 0;
			for (int z = 0; z < n; z++)
			{
				sum += mat1[(x*n) + z] * mat2[(z*n) + y];
			}
			result[(x*n) + y] = sum;

		}
	}


	return result;

}

__host__ int * criaTransposta(int *mat, int n)
{

	int * matT = (int *)malloc(n * n * sizeof(int));


	for (int x = 0; x < n; x++)
	{
		
		for (int y = 0; y < n; y++)
		{
			matT[(y*n) + x] = mat[(x*n) + y];

		}

	}

	return matT;

}


__host__ void imprimirMatriz(int * mat, int n)
{

	for (int x = 0; x < n; x++)
	{
		cout << "| ";
		for (int y = 0; y < n; y++)
		{
			cout << mat[(x*n) + y];
			
			if (y!= n-1) cout << ",";
		}

		cout << " | "<<endl;
	}
}

__host__ int * criaMatriz(int n)
{
	int *pMat = (int *)malloc(n * n * sizeof(int));
	if (!pMat) {
		cout << "Ferrou" << endl;
	}

	//srand( get_clock_sec() );
	
	for (int i = 0; i < n*n; i++) {

		//pMat[ i ] = my_rand( numeroDeElementos * 1000 );
		pMat[i] = rand() % 100;

	}

	return pMat;
}
