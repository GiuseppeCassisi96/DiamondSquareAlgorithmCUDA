#include "DSSequential/DiamondSquareSEQ.h"
#include "GeneratePPMImage.h"

//rand value must ranged between 100.0f and 200.0f
int main()
{
	DiamondSquareSEQ DSSEQ (9, 0.0f, 255.0f, 300.0f);
	DSSEQ.InitializationDiamondSquare();
	DSSEQ.RunDiamondSquare();

	PPMImage image(255.0f, DSSEQ.HeightMapSize, DSSEQ.GetHeightMapData());
	image.Generation();
	return 0;
}