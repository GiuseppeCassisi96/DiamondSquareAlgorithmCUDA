#pragma once
#include <vector>
#include "Utils/RandNumberGenerator.h"
class DiamondSquareSEQ
{
public:
	int N = 0;
	int HeightMapSize;
	float minHeightValue, maxHeightValue;
	float randomMagnitude;
	DiamondSquareSEQ(int NSize, float minHeightValue,
	float maxHeightValue, float randomValue);
	void PrintMap();
	void RunDiamondSquare();
	void InitializationDiamondSquare();
	std::vector<float> GetHeightMapData();
	
private:
	std::vector<float> HeightMap;
	RandNumberGenerator randGen;
	//'chunkSize' refers to dimension of the square and the diamond at each iteration 
	int chunkSize;
	//Is simple the current chunkSize divide by two
	int half;
	void SquareStep();
	void DiamondStep();
};




