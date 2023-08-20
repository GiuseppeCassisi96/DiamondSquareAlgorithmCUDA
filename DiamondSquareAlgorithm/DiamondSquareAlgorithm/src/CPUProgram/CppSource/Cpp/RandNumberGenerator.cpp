#include "Utils/RandNumberGenerator.h"


float RandNumberGenerator::GenerateFloat(float randSeed)
{
	std::uniform_real_distribution<float> randfloatdist (-randSeed, randSeed);
	return randfloatdist(APPgen);
}

float RandNumberGenerator::GenerateFloatInARange(float minLimit, float maxLimit)
{
	std::uniform_real_distribution<float> randfloatdist(minLimit, maxLimit);
	return randfloatdist(APPgen);
}
