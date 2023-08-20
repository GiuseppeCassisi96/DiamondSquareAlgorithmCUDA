#include "DSSequential/DiamondSquareSEQ.h"
#include "iostream"
#include "algorithm"

DiamondSquareSEQ::DiamondSquareSEQ(int NSize, float minHeightValue,
	float maxHeightValue, float randomValue) : N(NSize), minHeightValue(minHeightValue),
maxHeightValue(maxHeightValue), randomMagnitude(randomValue)
{
	//Height map size computation 
	HeightMapSize = static_cast<int>(std::pow(2, N) + 1);
	HeightMap.reserve(static_cast<int>(HeightMapSize * HeightMapSize));
}

void DiamondSquareSEQ::PrintMap()
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

std::vector<float> DiamondSquareSEQ::GetHeightMapData()
{
	return HeightMap;
}

void DiamondSquareSEQ::RunDiamondSquare()
{
	while (chunkSize > 1)
	{
		half = chunkSize / 2;
		DiamondStep();
		SquareStep();
		chunkSize /= 2;
		randomMagnitude/= 2.0f;
	}
}

void DiamondSquareSEQ::InitializationDiamondSquare()
{
	//Filling of the height map 
	for (int i = 0; i < HeightMapSize; i++)
	{
		for (int j = 0; j < HeightMapSize; j++)
		{
			HeightMap.emplace_back(0.0f);
		}
	}

	chunkSize = HeightMapSize - 1;
	half = chunkSize / 2;

	//Four corner value computation
	for (int i = 0; i < HeightMapSize; i += chunkSize) 
	{
		for (int j = 0; j < HeightMapSize; j += chunkSize) 
		{
			HeightMap[i * HeightMapSize + j] = std::floor(randGen.GenerateFloatInARange(
					minHeightValue, maxHeightValue));
		}
	}

}


void DiamondSquareSEQ::DiamondStep()
{
	for(int i = 0; i < HeightMapSize - 1; i += chunkSize)
	{
		for(int j = 0; j < HeightMapSize - 1; j += chunkSize)
		{
			float value = 0.0f;
			const float randValue = randGen.GenerateFloat(randomMagnitude);

			//Average computation
			value += HeightMap[i * HeightMapSize + j];
			value += HeightMap[i * HeightMapSize + (j + chunkSize)];
			value += HeightMap[(i + chunkSize) * HeightMapSize + j];
			value += HeightMap[(i + chunkSize) * HeightMapSize + (j + chunkSize)];
			value /= 4.0f;

			value += randValue;

			value = std::clamp(value, minHeightValue, maxHeightValue);
			HeightMap[(i + half) * HeightMapSize + (j + half)] = value;
		}
	}
}

void DiamondSquareSEQ::SquareStep()
{
	for (int i = 0; i < HeightMapSize; i += half)
	{
		for (int j = (i + half) % chunkSize; j < HeightMapSize; j += chunkSize)
		{
			float value = 0.0f;
			const float randValue = randGen.GenerateFloat(randomMagnitude);
			int index = (i - half) * HeightMapSize + j;
			if(index >= 0)
			{
				value += HeightMap[index];
			}
			else
			{
				value += randGen.GenerateFloatInARange(minHeightValue, maxHeightValue);
			}

			index = i * HeightMapSize + (j - half);
			if (index >= 0)
			{
				value += HeightMap[index];
			}
			else
			{
				value += randGen.GenerateFloatInARange(minHeightValue, maxHeightValue);
			}

			index = i * HeightMapSize + (j + half);
			if (index < static_cast<int>(HeightMap.size()))
			{
				value += HeightMap[index];
			}
			else
			{
				value += randGen.GenerateFloatInARange(minHeightValue, maxHeightValue);
			}

			index = (i + half) * HeightMapSize + j;
			if (index < static_cast<int>(HeightMap.size()))
			{
				value += HeightMap[index];
			}
			else
			{
				value += randGen.GenerateFloatInARange(minHeightValue, maxHeightValue);
			}

			value /= 4.0f;
			value += randValue;
			value = std::clamp(value, minHeightValue, maxHeightValue);
			HeightMap[i * HeightMapSize + j] = value;
		}
	}
}


