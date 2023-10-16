
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//for __syncthreads
#ifndef __CUDACC__
#define __CUDACC__
#endif // !__CUDACC__
#include <device_functions.h>
#include <iostream>
#include <cstdlib>
#include "CUDAProgram/API/DSParallel/DiamondSquarePAR.h"
#include "CUDAProgram/API/Utils/RandNumGenDevice.h"
#include "CPUProgram/API/Utils/RandNumberGenerator.h"


// SETTINGS
#define SQUARE_BLOCK_X_SIZE		16  //8 or 16
#define MAX_BLOCK_SIZE			32 //16 or 32

#define cudaCheckError() {                                          \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
   exit(0); \
 }                                                                 \
}



int blockSizeDiamond, blockXSizeSquare, blockYSizeSquare;
int gridSizeDiamond, gridSizeSquare;
RandNumberGenerator generator;

//DEVICE FUNCTIONS
__device__ float clamp(float x, float min_val, float max_val)
{
	return fminf(fmaxf(x, min_val), max_val);
}


__global__ void KERNEL_InitCorners(float* HeightMap, int HeightMapSize, int chunkSize,
	float minHeightValue, float maxHeightValue, curandState* ranGenStates)
{
	unsigned const int idx = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned const int idy = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned const int x = idx * chunkSize;
	unsigned const int y = idy * chunkSize;
	unsigned const int index = y * HeightMapSize + x;
	if(index < HeightMapSize * HeightMapSize)
	{
		float value = GenerateFloatInRangeGPU(ranGenStates, minHeightValue, maxHeightValue, index);
		value = clamp(value, minHeightValue, maxHeightValue);
		HeightMap[index] = value;
	}
}

__global__ void KERNEL_DiamondStep(int chunkSize, float* HeightMap, int HeightMapSize, 
	float randMagnitude, int half, curandState* ranGenStates,
	float minHeightValue, float maxHeightValue)
{
	unsigned const int x = (blockDim.x * blockIdx.x + threadIdx.x) * chunkSize;
	unsigned const int y = (blockDim.y * blockIdx.y + threadIdx.y) * chunkSize;
	if (x > HeightMapSize || y > HeightMapSize)
	{
		return;
	}
	float value = HeightMap[y * HeightMapSize + x] + HeightMap[y * HeightMapSize + (x + chunkSize)] +
		HeightMap[(y + chunkSize) * HeightMapSize + x] + 
		HeightMap[(y + chunkSize) * HeightMapSize + (x + chunkSize)];
	value /= 4.0f;
	value += GenerateFloatGPU(ranGenStates, randMagnitude, (blockDim.x * blockIdx.x + threadIdx.x) *
		HeightMapSize + (blockDim.y * blockIdx.y + threadIdx.y));
	value = clamp(value, minHeightValue, maxHeightValue);
	HeightMap[(y + half) * HeightMapSize + (x + half)] = value;
	
}

__global__ void KERNEL_SquareStep(int chunkSize, float* HeightMap, int HeightMapSize,
	float randMagnitude, int half, curandState* ranGenStates, float minHeight, float maxHeight)
{
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned const  int x = (blockIdx.x * blockDim.x + threadIdx.x) * chunkSize * (y % 2 == 0)
		+ y * half * (y % 2 != 0);
	y = (y * half + half) * (y % 2 == 0)
		+ (blockIdx.x * blockDim.x + threadIdx.x) * chunkSize * (y % 2 != 0);
	
	if (x > HeightMapSize || y > HeightMapSize)
	{
		return;
	}
	float value = 0.0f;

	//FIRST ELEMENT
	int currentIndex = static_cast<int>((x - half) * HeightMapSize + y);
	if(currentIndex >= 0)
	{
		value += HeightMap[currentIndex];
	}
	else
	{
		value += GenerateFloatInRangeGPU(ranGenStates, minHeight, maxHeight, 
			(blockIdx.y * blockDim.y + threadIdx.y) * HeightMapSize + (blockIdx.x * blockDim.x + threadIdx.x));
	}

	//SECOND ELEMENT
	currentIndex = static_cast<int>(x * HeightMapSize + (y - half));
	if (currentIndex >= 0)
	{
		value += HeightMap[currentIndex];
	}
	else
	{
		value += GenerateFloatInRangeGPU(ranGenStates, minHeight, maxHeight, 
			(blockIdx.y * blockDim.y + threadIdx.y) * HeightMapSize + (blockIdx.x * blockDim.x + threadIdx.x));
	}

	//THIRD ELEMENT
	currentIndex = static_cast<int>(x * HeightMapSize + (y + half));
	if (currentIndex < HeightMapSize * HeightMapSize)
	{
		value += HeightMap[currentIndex];
	}
	else
	{
		value += GenerateFloatInRangeGPU(ranGenStates, minHeight, maxHeight, 
			(blockIdx.y * blockDim.y + threadIdx.y) * HeightMapSize + (blockIdx.x * blockDim.x + threadIdx.x));
	}

	//FOURTH ELEMENT
	currentIndex = static_cast<int>((x + half) * HeightMapSize + y);
	if (currentIndex < HeightMapSize * HeightMapSize)
	{
		value += HeightMap[currentIndex];
	}
	else
	{
		value += GenerateFloatInRangeGPU(ranGenStates, minHeight, maxHeight, 
			(blockIdx.y * blockDim.y + threadIdx.y) * HeightMapSize + (blockIdx.x * blockDim.x + threadIdx.x));
	}

	//FINAL COMPUTATION
	value /= 4.0f;
	value += GenerateFloatGPU(ranGenStates, randMagnitude, 
		(blockIdx.y * blockDim.y + threadIdx.y) * HeightMapSize + (blockIdx.x * blockDim.x + threadIdx.x));
	value = clamp(value, minHeight, maxHeight);
	HeightMap[x * HeightMapSize + y] = value;
}


//HOST FUNCTIONS
void DiamondSquarePAR::ComputeBlockGridSizes()
{
	//It can be 1 or 32 at maximum. The size is 2^k until it reaches the max block size 
	blockSizeDiamond = TwoKElements <= MAX_BLOCK_SIZE ? TwoKElements : MAX_BLOCK_SIZE;
	/* The size is 2^k + 1 until it reaches the max X dim for the block, in this case will be always
	* the max X dim for the block*/
	blockXSizeSquare = TwoKElements < SQUARE_BLOCK_X_SIZE ? blockSizeDiamond + 1 : SQUARE_BLOCK_X_SIZE;
	//The size is 2^(k+1) until it reaches the X dim for the block, in this case the dim will be always 2^k
	blockYSizeSquare = TwoKElements <= SQUARE_BLOCK_X_SIZE ? TwoKElements * 2 : blockSizeDiamond;

	/*With this formula every time my TwoKElements over a multiple of 32 (32, 64, 96, ecc) I increment the
	* grid size exponentially (powers of 2)*/
	gridSizeDiamond = (TwoKElements + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE;
	/*With this formula every time the double of TwoKElements over a multiple of 32 (32, 64, 96, ecc)
	 *I increment the grid size exponentially (powers of 2)*/
	gridSizeSquare = (TwoKElements * 2 + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE;


}


DiamondSquarePAR::DiamondSquarePAR(int NSize, float minHeightValue, float maxHeightValue, float randomValue) : N(NSize), minHeightValue(minHeightValue),
maxHeightValue(maxHeightValue), randomMagnitude(randomValue)
{
	//Height map size computation 
	HeightMapSize = static_cast<int>(std::pow(2, N) + 1);
	totalSize = HeightMapSize * HeightMapSize;

	//Bytesize computation
	byteSize =  static_cast<int>(sizeof(float) * totalSize);
	const int byteSizeForRandom = static_cast<int>(sizeof(curandState) * totalSize);

	//CPU memory allocation;
	HeightMap = static_cast<float*>(malloc(byteSize));

	//GPU memory allocation and setting
	cudaMalloc((void**)&HeightMapGPU, byteSize);
	cudaMalloc((void**)&states, byteSizeForRandom);
	cudaMemset(HeightMapGPU, 0.0f, byteSize);

	//chunkSize, half and threadAmount computation 
	chunkSize = HeightMapSize - 1;
	half = chunkSize / 2;
	TwoKElements = (HeightMapSize - 1) / chunkSize;
}

void DiamondSquarePAR::InitializationDS()
{
	//InitCurand kernel threads configuration
	constexpr int xDimBlock = MAX_BLOCK_SIZE;
	constexpr int yDimBlock = MAX_BLOCK_SIZE;
	const int xDimGrid = static_cast<int>(ceil(static_cast<float>(HeightMapSize) / static_cast<float>(xDimBlock)));
	const int yDimGrid = static_cast<int>(ceil(static_cast<float>(HeightMapSize) / static_cast<float>(yDimBlock)));
	dim3 randblock_dim(xDimBlock, yDimBlock);
	dim3 randgrid_dim(xDimGrid, yDimGrid);

	//Rand seed generation
	const int randSeed = static_cast<int>( generator.GenerateFloat(100.0f));

	//InitCurand kernel execution
	InitCurand<<<randgrid_dim, randblock_dim>>> (states, HeightMapSize, randSeed);
	cudaDeviceSynchronize();
	cudaCheckError()

	//InitCorners kernel execution
	KERNEL_InitCorners <<<2, 2>>> (HeightMapGPU, HeightMapSize, chunkSize,
		minHeightValue, maxHeightValue, states);
	cudaCheckError()
	cudaDeviceSynchronize();
}

void DiamondSquarePAR::DiamondStep()
{
	dim3 block_dim(blockSizeDiamond, blockSizeDiamond);
	dim3 grid_dim(gridSizeDiamond, gridSizeDiamond);
	KERNEL_DiamondStep <<<grid_dim, block_dim >>> (chunkSize, HeightMapGPU,
		HeightMapSize, randomMagnitude, half, states, minHeightValue, maxHeightValue);
	cudaCheckError()
}

void DiamondSquarePAR::SquareStep()
{
	dim3 block_dim(blockXSizeSquare, blockYSizeSquare);
	dim3 grid_dim(gridSizeSquare, gridSizeSquare);
	KERNEL_SquareStep <<<grid_dim, block_dim >>> (chunkSize, HeightMapGPU,
		HeightMapSize, randomMagnitude, half, states, minHeightValue, maxHeightValue);
	cudaCheckError()
}
void DiamondSquarePAR::RunDiamondSquare()
{
	while (chunkSize > 1)
	{
		ComputeBlockGridSizes();
		DiamondStep();
		cudaDeviceSynchronize();
		SquareStep();
		cudaDeviceSynchronize();
		randomMagnitude /= 2.0f;
		chunkSize /= 2;
		half = chunkSize / 2;
		algoStep++;
		TwoKElements *= 2;
	}
	cudaMemcpy(HeightMap, HeightMapGPU, byteSize, cudaMemcpyDeviceToHost);
	//Free memories
	cudaFree(states);
	cudaFree(HeightMapGPU);
}

void DiamondSquarePAR::PrintMap()
{
	for (int i = 0; i < HeightMapSize; i++)
	{
		for (int j = 0; j < HeightMapSize; j++)
		{
			if (HeightMap[i * HeightMapSize + j] != 0.0f)
			{
				std::cout << HeightMap[i * HeightMapSize + j] << " ";
				continue;
			}
			std::cout << "0" << " ";
		}
		std::cout << "\n";
	}
}





