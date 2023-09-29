#pragma once
#include <fstream>
#include <vector>

class PPMImage
{
public:
	std::ofstream image;
	float* heightMapData;
	float maxPixelValue;
	int size;
	PPMImage(float maxPixelValue, int size, float* heightMapData, std::string filePath = "Image.ppm");
	void Generation();
};
