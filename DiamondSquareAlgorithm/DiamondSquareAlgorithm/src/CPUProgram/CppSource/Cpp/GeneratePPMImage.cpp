#include "GeneratePPMImage.h"

#include <iomanip>

PPMImage::PPMImage(float maxPixelValue, int size, float* heightMapData, std::string filePath) :
heightMapData(heightMapData), size(size), maxPixelValue(maxPixelValue)
{
	image.open(filePath);
}



void PPMImage::Generation()
{
	if (image.is_open())
	{
		//place header info
		image << "P3\n";
		image << size << " " << size << "\n";
		image << maxPixelValue << "\n";
		//place RGB values
		for (int i = 0; i < size; i++)
		{
			for (int j = 0; j < size; j++)
			{
				const int heightValue = static_cast<int>(heightMapData[i * size + j]);
				image << heightValue << " " << heightValue << " " << heightValue ;
				image << "  ";
			}
			image << "\n";
		}
	}
	image.close();
}
