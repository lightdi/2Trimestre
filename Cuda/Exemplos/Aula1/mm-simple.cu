//-----------------------------------------
// Autor: Farias
// Data : Nov 2010
// Goal : Multiply two matrices in GPU
//-----------------------------------------

/***************************************************************************************************
	Includes
***************************************************************************************************/

#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>

#include "rf-time.h"


/***************************************************************************************************
	Defines
***************************************************************************************************/

#define ERROR_ALOCATING_MATRIX "Erro: na alocacao de memoria para matriz."
#define ERROR_INVALID_PARMS    "Erro: invalid parameters."
#define ERROR_NULL_VECTOR      "Erro: tentativa de imprimir vetor nulo."

#define ELEM(l,c,WID) ((c)+(l)*(WID))


/***************************************************************************************************
	Functions
***************************************************************************************************/

using namespace std;


/**************************************************************************************************/

__host__ void erro( const char tipoDeErro[] ) {

	fprintf( stderr, "%s\n", tipoDeErro );
	exit(0);

}


/**************************************************************************************************/

__host__ int *createMatrix( int width, int height ) {

	int *pMat = (int *)malloc( width * height * sizeof( int ) );
	if( !pMat ) {
		erro( ERROR_ALOCATING_MATRIX );
	}

	//srand( get_clock_sec() );

	for( int i = 0 ; i < width*height ; i++ ) {

		//pMat[ i ] = my_rand( numeroDeElementos * 1000 );
		pMat[ i ] = 1;

	}
	
	return pMat;

}

/**************************************************************************************************/

__host__ void printMatrix( int *mat, int width, int height ) {

	int l, c;
	
	if( !mat ) {
		erro( ERROR_NULL_VECTOR );
	}

	cout << "      ";
	for( c = 0 ; c < width ; c++ ) 
		cout << setw(7) << c;
	cout << endl;
	for( l = 0 ; l < height ; l++ ) {
	
		cout << setw(6) << l;
		for( c = 0 ; c < width ; c++ ) 
			cout << setw(7) << mat[ ELEM(c,l,width) ];
		cout << endl;

	}
	
}

/**************************************************************************************************/

__host__ bool compareMatrices( int *matA, int * matB, int width, int height ) {

	int w, h;

	for( h = 0 ; h < height ; h++ ) 
	
		for( w = 0 ; w < width ; w++ ) 

			if( matA[ ELEM(w,h,width) ] != matB[ ELEM(w,h,width) ] )
				return false;

	return true;

}

/**************************************************************************************************/

__host__ void multMatricesCPU( int *matA, int * matB, int *matC, int width, int height ) {

	int l, c, k, sum;

	for( l = 0 ; l < height ; l++ ) {
	
		for( c = 0 ; c < width ; c++ ) {

			sum = 0;

			for( k = 0 ; k < width ; k++ )

				sum += matA[ ELEM( l, k, width ) ] * matB[ ELEM( k, c, width ) ];
				
			matC[ ELEM( l, c, width ) ] = sum;

		}

	}

}


/**************************************************************************************************/
__global__ void multMatricesGPU( int n, int *matA, int *matB, int *matC ) {

	int c = threadIdx.x;
	int l = threadIdx.y;

	if( l < n && c < n ) {
		
		int sum = 0;
		
		for( int k = 0 ; k < n ; k++ ) {
			
			sum += matA[ ELEM( l, k, n ) ]*matB[ ELEM( k, c, n ) ];
			
		}
		
		matC[ ELEM( l, c, n ) ] = sum;

	}
	
}


/**************************************************************************************************/
__host__ int main( int argc, char *argv[] ) {

	int blSizeX, blSizeY;
	double start_time, cpu_mult_time, gpu_mult_time, copy_to_time, copy_from_time;
	
	// Neste exemplo, consideramos apenas matrizes quadradas
	// Ou seja, o parametro N de entrada indicara uma matrix NxN

	// Trata parâmetros de entrada
	int h_numElem = 3;

	if( argc == 2 ) {

		h_numElem = atoi( argv[ 1 ] );

		if( h_numElem < 1 ) {
			erro( ERROR_INVALID_PARMS );
		}

	}
	cout << "Multiplicar duas matrizes " << h_numElem << "x" << h_numElem << endl;

	// Gera vetorA e vetorB
	int *h_matA = createMatrix( h_numElem, h_numElem );
	int *h_matB = createMatrix( h_numElem, h_numElem );
	// Alocar matriz resultado
	int *h_matC = (int*)malloc( h_numElem*h_numElem*sizeof( int ) );
	if( !h_matC ) {
		erro( ERROR_ALOCATING_MATRIX );
	}
	int *h_gpuResp = (int*)malloc( h_numElem*h_numElem*sizeof( int ) );
	if( !h_gpuResp ) {
		erro( ERROR_ALOCATING_MATRIX );
	}


	// Calcula multiplicacao na cpu
	start_time = get_clock_msec();
	multMatricesCPU( h_matA, h_matB, h_matC, h_numElem, h_numElem );
	cpu_mult_time = get_clock_msec() - start_time;

	
	// Imprime vetorA
	cout << "Matrix A -----------------------------------------------------" << endl;
	printMatrix( h_matA, h_numElem, h_numElem );

	// Imprime vetorB
	cout << "Matrix B -----------------------------------------------------" << endl;
	printMatrix( h_matB, h_numElem, h_numElem );
	

	// Imprime resultado CPU
	cout << "Matrix Resultado CPU -----------------------------------------" << endl;
	printMatrix( h_matC, h_numElem, h_numElem );
	

	// Aloca memória no device e copia vetorA e vetorB para lá
	int* d_matA;
	int* d_matB;
	int* d_matC;
	cudaMalloc( (void**)&d_matA, h_numElem*h_numElem*sizeof( int ) );
	cudaMalloc( (void**)&d_matB, h_numElem*h_numElem*sizeof( int ) );
	cudaMalloc( (void**)&d_matC, h_numElem*h_numElem*sizeof( int ) );

	start_time = get_clock_msec();
	cudaMemcpy( d_matA, h_matA, h_numElem*h_numElem*sizeof( int ), cudaMemcpyHostToDevice );
	cudaMemcpy( d_matB, h_matB, h_numElem*h_numElem*sizeof( int ), cudaMemcpyHostToDevice );
	copy_to_time = get_clock_msec() - start_time;

	// Calcula dimensoes da grid e dos blocos
	blSizeX = blSizeY = 16;
	dim3 blockSize( blSizeX, blSizeY, 1 );
	int threadsPorBloco = blockSize.x * blockSize.y;

	int numeroDeBlocos = (h_numElem*h_numElem) / threadsPorBloco + 
	                     ( (h_numElem*h_numElem) % threadsPorBloco == 0 ? 0 : 1 );
	dim3 gridSize( numeroDeBlocos, 1, 1 );


	// Chama SomarVetoresGPU
	start_time = get_clock_msec();
	multMatricesGPU<<< gridSize, blockSize >>>( h_numElem, d_matA, d_matB, d_matC );
	cudaThreadSynchronize();
	gpu_mult_time = get_clock_msec() - start_time;

	// Copia o resultado de volta para o host
	start_time = get_clock_msec();
	cudaMemcpy( h_gpuResp, d_matC, h_numElem*h_numElem*sizeof( int ), cudaMemcpyDeviceToHost );
	copy_from_time = get_clock_msec() - start_time;

	// Imprime resultado GPU
	cout << "Matrix Resultado GPU -----------------------------------------" << endl;
	printMatrix( h_gpuResp, h_numElem, h_numElem );
	

	cout << "--------------------------------------------------------------" << endl;
	if( compareMatrices( h_matC, h_gpuResp, h_numElem, h_numElem ) )
		cout << "Resultado CORRETO  :-)" << endl;
	else
		cout << "Resultado INCORRETO!!!!!!!!!!!!!!!!!!!  :-(" << endl;



	// Imprime tempos
	cout << "--------------------------------------------------------------" << endl;
	cout << "Informacoes da execucao..." << endl;
	cout << "--------------------------------------------------------------" << endl;
	cout << "Dimensoes das Matrizes: " << h_numElem << "x" << h_numElem << endl;
	cout << "\tNumero de Blocos : " << numeroDeBlocos << endl;
	cout << "\tThreads por Bloco: " << threadsPorBloco << endl;
	cout << "Tempos de execucao: " << endl;
	cout << "\tCopia CPU->GPU das matrizes A e B: " << copy_to_time << endl;
	cout << "\tGPU multiplicacao: " << gpu_mult_time << endl;
	cout << "\tCPU multiplicacao: " << cpu_mult_time << endl;
	cout << "\tCopia GPU->CPU da matriz resultado: " << copy_from_time << endl;


	// Libera memória do device
	cudaFree( d_matA );
	cudaFree( d_matB );
	cudaFree( d_matC );
	
	// Libera memória do host
	free( h_matA );
	free( h_matB );
	free( h_matC );
	free( h_gpuResp );

	cout << "--------------------------------------------------------------" << endl;

	return 0;

}
