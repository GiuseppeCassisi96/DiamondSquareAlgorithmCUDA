#include <iostream>

#include "DSSequential/DiamondSquareSEQ.h"
#include "CUDAProgram/API/DSParallel/DiamondSquarePAR.h"
#include "GeneratePPMImage.h"
#include "cuda_runtime.h"
//MAX dim for the problem is 12 (2^12) only 4k images
int main()
{
	float timeCPU = 0.0f, timeGPU = 0.0f;
	cudaEvent_t startCPU, stopCPU, startGPU, stopGPU;
	cudaEventCreate(&startCPU);
	cudaEventCreate(&stopCPU);
	cudaEventCreate(&startGPU);
	cudaEventCreate(&stopGPU);

	cudaEventRecord(startCPU);
	DiamondSquareSEQ DSSEQ (10, 0.0f, 255.0f, 200.0f);
	DSSEQ.InitializationDiamondSquare();
	DSSEQ.RunDiamondSquare();
	cudaEventRecord(stopCPU);
	cudaEventSynchronize(stopCPU);
	cudaEventElapsedTime(&timeCPU, startCPU, stopCPU);

	cudaEventRecord(startGPU);
	DiamondSquarePAR DSPAR(10, 0.0f, 255.0f, 200.0f);
	DSPAR.InitializationDS();
	DSPAR.RunDiamondSquare();
	cudaEventRecord(stopGPU);
	cudaEventSynchronize(stopGPU);
	cudaEventElapsedTime(&timeGPU, startGPU, stopGPU);
	std::cout << "TIME CPU: " << timeCPU << "\n";
	std::cout << "TIME GPU: " << timeGPU << "\n";
	std::cout << "Speedup: " << timeCPU / timeGPU << "\n";

	/*DSPAR.PrintMap();*/
	std::cout << "CREATE IMAGE\n";
	PPMImage image(255.0f, DSPAR.HeightMapSize, DSPAR.HeightMap);
	image.Generation();
	std::cout << "FINISH\n";
	//Free CPU memory
	free(DSPAR.HeightMap);
	return 0;
}