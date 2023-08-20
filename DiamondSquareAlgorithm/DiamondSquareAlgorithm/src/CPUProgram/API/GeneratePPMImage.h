#pragma once
#include <fstream>
#include <vector>

class PPMImage
{
public:
	std::ofstream image;
	std::vector<float> heightMapData;
	float maxPixelValue;
	int size;
	PPMImage(float maxPixelValue, int size, std::vector<float> heightMapData, std::string filePath = "Image.ppm");
	void Generation();
};
