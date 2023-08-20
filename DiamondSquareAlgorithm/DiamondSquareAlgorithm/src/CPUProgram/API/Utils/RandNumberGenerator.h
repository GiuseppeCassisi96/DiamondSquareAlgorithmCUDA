#pragma once
#include <random>
static std::random_device rd;
static std::mt19937 APPgen (rd());
class RandNumberGenerator
{
public:
	RandNumberGenerator() = default;
	float GenerateFloat(float randSeed);
	float GenerateFloatInARange(float minLimit, float maxLimit);
	
};
