#pragma once
#include <fstream>
#include <vector>

class PPMImage
{
public:
	std::ofstream image;
	float* heightMapData;
	std::vector<float> HeightMapDataVector;
	float maxPixelValue;
	int size;
	PPMImage(float maxPixelValue, int size, float* heightMapData, std::string filePath = "Image.ppm");
	PPMImage(float maxPixelValue, int size, std::vector<float> heightMapData, std::string filePath = "Image.ppm");
	void Generation();
	void GenerationUsingVector();
};
