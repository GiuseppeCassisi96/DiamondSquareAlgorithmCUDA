#include <iostream>

#include "DSSequential/DiamondSquareSEQ.h"
#include "CUDAProgram/API/DSParallel/DiamondSquarePAR.h"
#include "GeneratePPMImage.h"
#include "cuda_runtime.h"
//MAX dim for the problem is 12 (2^12) 
//rand value must ranged between 100.0f and 200.0f
//TODO Measure the execution time of the sequential method and parallel too
//TODO Use Nsight Compute to measure the occupancy  
int main()
{
	std::cout << "START SEQ EXE\n";
	DiamondSquareSEQ DSSEQ (12, 0.0f, 255.0f, 300.0f);
	DSSEQ.InitializationDiamondSquare();
	DSSEQ.RunDiamondSquare();
	std::cout << "STOP SEQ EXE\n";

	std::cout << "START PARALLEL EXE\n";
	DiamondSquarePAR DSPAR(12, 0.0f, 255.0f, 300.0f);
	DSPAR.InitializationDS();
	DSPAR.RunDiamondSquare();
	std::cout << "STOP PARALLEL EXE\n";

	/*DSPAR.PrintMap();*/
	std::cout << "CREATE IMAGE\n";
	PPMImage image(255.0f, DSPAR.HeightMapSize, DSPAR.HeightMap);
	image.Generation();
	std::cout << "FINISH\n";
	//Free CPU memory
	free(DSPAR.HeightMap);
	return 0;
}