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

#define ELEM( l, c, WID ) ((c)+(l)*(WID))


/***************************************************************************************************
	Functions
***************************************************************************************************/

using namespace std;


/**************************************************************************************************/

void erro( const char tipoDeErro[] ) {

	fprintf( stderr, "%s\n", tipoDeErro );
	exit(0);

}


/**************************************************************************************************/

int *createMatrix( int width, int height ) {

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

void printMatrix( int *mat, int width, int height ) {

	int w, h;
	
	if( !mat ) {
		erro( ERROR_NULL_VECTOR );
	}

	cout << "      ";
	for( w = 0 ; w < width ; w++ ) 
		cout << setw(7) << w;
	cout << endl;
	for( h = 0 ; h < height ; h++ ) {
	
		cout << setw(6) << h;
		for( w = 0 ; w < width ; w++ ) 
			cout << setw(7) << mat[ ELEM(w,h,width) ];
		cout << endl;

	}
	
}

/**************************************************************************************************/

bool compareMatrices( int *matA, int * matB, int width, int height ) {

	int w, h;

	for( h = 0 ; h < height ; h++ ) 
	
		for( w = 0 ; w < width ; w++ ) 

			if( matA[ ELEM(w,h,width) ] != matB[ ELEM(w,h,width) ] )
				return false;

	return true;

}

/**************************************************************************************************/

void multMatricesCPU( int *matA, int * matB, int *matC, int width, int height ) {

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
int main( int argc, char *argv[] ) {

	double start_time, cpu_mult_time;
	
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
	
	// Imprime tempos
	cout << "--------------------------------------------------------------" << endl;
	cout << "Informacoes da execucao..." << endl;
	cout << "--------------------------------------------------------------" << endl;
	cout << "Dimensoes das Matrizes: " << h_numElem << "x" << h_numElem << endl;
	cout << "Tempo de execucao: " << endl;
	cout << "\tCPU multiplicacao: " << cpu_mult_time << endl;

	// Libera memória do host
	free( h_matA );
	free( h_matB );
	free( h_matC );

	cout << "--------------------------------------------------------------" << endl;

	return 0;

}
