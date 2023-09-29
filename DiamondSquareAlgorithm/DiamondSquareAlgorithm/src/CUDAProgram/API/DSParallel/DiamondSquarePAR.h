#pragma once

class DiamondSquarePAR
{
public:
	int N = 0;
	int HeightMapSize;
	float minHeightValue, maxHeightValue;
	float randomMagnitude;
	DiamondSquarePAR(int NSize, float minHeightValue,
		float maxHeightValue, float randomValue);
	void InitializationDS();
	void RunDiamondSquare();
	void PrintMap();
	float* HeightMap, * HeightMapGPU;
	int byteSize;
private:
	
	//'chunkSize' refers to dimension of the square and the diamond at each iteration 
	int chunkSize;
	//Is simple the current chunkSize divide by two
	int half;
	unsigned int totalSize;
	int algoStep = 0;
	int threadAmount;
	void DiamondStep();
	void SquareStep();
	void  ComputeBlockGridSizes();

};
