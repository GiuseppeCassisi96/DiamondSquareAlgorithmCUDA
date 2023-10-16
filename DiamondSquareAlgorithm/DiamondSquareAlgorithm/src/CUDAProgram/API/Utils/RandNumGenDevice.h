#pragma once
//for __syncthreads
#ifndef __CUDACC__
#define __CUDACC__
#endif // !__CUDACC__
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"
const int seed = 10;
curandState* states;
__device__ float GenerateFloatGPU(curandState* randGenstates, float randMagnitude, unsigned int index)
{
	curandState state = randGenstates[index];
	float randValue = (-1.0f + curand_uniform(&state) * 2.0f) * randMagnitude;
	randGenstates[index] = state;
	return randValue;
}

__device__ float GenerateFloatInRangeGPU(curandState* randGenstates, float min ,float max, unsigned int index)
{
	curandState state = randGenstates[index];
	float randValue = min + curand_uniform(&state) * (max - min);
	randGenstates[index] = state;
	return randValue;
}

__global__ void InitCurand(curandState* state, int HeightMapSize, int randSeed)
{
	unsigned const int idx = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned const int idy = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned const int index = idy * HeightMapSize + idx;
	if(index >= HeightMapSize * HeightMapSize)
	{
		return;
	}
	curand_init(randSeed + index, 0, 0, &state[index]);
}

