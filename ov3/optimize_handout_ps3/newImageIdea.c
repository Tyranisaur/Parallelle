#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include "ppm.h"

// Image from:
// http://7-themes.com/6971875-funny-flowers-pictures.html

typedef struct {
	double red,green,blue;
} AccuratePixel;

typedef struct {
	double x, y;
	AccuratePixel *data;
} AccurateImage;

// Convert ppm to high precision format.
AccurateImage *convertImageToNewFormat(PPMImage *image) {
	// Make a copy
	AccurateImage *imageAccurate;
	imageAccurate = (AccurateImage *)malloc(sizeof(AccurateImage));
	imageAccurate->data = (AccuratePixel*)malloc(image->x * image->y * sizeof(AccuratePixel));
	for(int i = 0; i < image->x * image->y; i++) {
		imageAccurate->data[i].red   =  (double)image->data[i].red;
		imageAccurate->data[i].green =  (double)image->data[i].green;
		imageAccurate->data[i].blue  =  (double)image->data[i].blue;
	}
	imageAccurate->x = image->x;
	imageAccurate->y = image->y;

	return imageAccurate;
}

// Perform the new idea:
void performNewIdeaIteration(AccurateImage *imageOut, AccurateImage *imageIn, int size)
{
	AccurateImage* temp;
	int width = imageIn->x;
	int height = imageIn->y;
	int inOffset, outOffset;
	double sumR, sumG, sumB;
	int countIncluded;
	//Do all five iterations inside function
	for(int i = 0; i < 5; i++)
	{
		outOffset = 0;
		// Iterate over each pixel
		for(int senterY = 0; senterY < height; senterY++)
		{
			for(int senterX = 0; senterX < width; senterX++)
			{
				// For each pixel we compute the magic numbers
				sumR = 0.0;
				sumG = 0.0;
				sumB = 0.0;
				countIncluded = 0;
				//Iterate through filter mask
				for(int y = -size; y <= size; y++)
				{
					int currentY = senterY + y;
					//Check height bounds before doing anything width wise
					if(currentY < 0)
						continue;
					if(currentY >= height)
						continue;

					for(int x = -size; x <= size; x++)
					{
						int currentX = senterX + x;

						// Check if we are outside the bounds width wise
						if(currentX < 0)
							continue;
						if(currentX >= imageIn->x)
							continue;

						// Now we can begin
						inOffset = currentY * width + currentX;

						sumR += imageIn->data[inOffset].red;
						sumG += imageIn->data[inOffset].green;
						sumB += imageIn->data[inOffset].blue;

						// Keep track of how many values we have included
						countIncluded++;
					}

				}


				// Update the output image
				imageOut->data[outOffset].red = sumR/countIncluded;
				imageOut->data[outOffset].green = sumG/countIncluded;
				imageOut->data[outOffset].blue = sumB/countIncluded;
				outOffset++;
			}

		}
		//Do trickery with argument variables
		temp = imageIn;
		imageIn = imageOut;
		imageOut = temp;

	}
}


// Perform the final step, and return it as ppm.
PPMImage * performNewIdeaFinalization(AccurateImage *imageInSmall, AccurateImage *imageInLarge) {
	PPMImage *imageOut;
	imageOut = (PPMImage *)malloc(sizeof(PPMImage));
	imageOut->data = (PPMPixel*)malloc(imageInSmall->x * imageInSmall->y * sizeof(PPMPixel));

	imageOut->x = imageInSmall->x;
	imageOut->y = imageInSmall->y;

	for(int i = 0; i < imageInSmall->x * imageInSmall->y; i++) {
		double value = (imageInLarge->data[i].red - imageInSmall->data[i].red);
		if(value > 255)
			imageOut->data[i].red = 255;
		else if (value < -1.0) {
			value = 257.0+value;
			if(value > 255)
				imageOut->data[i].red = 255;
			else
				imageOut->data[i].red = floor(value);
		} else if (value > -1.0 && value < 0.0) {
			imageOut->data[i].red = 0;
		} else {
			imageOut->data[i].red = floor(value);
		}

		value = (imageInLarge->data[i].green - imageInSmall->data[i].green);
		if(value > 255)
			imageOut->data[i].green = 255;
		else if (value < -1.0) {
			value = 257.0+value;
			if(value > 255)
				imageOut->data[i].green = 255;
			else
				imageOut->data[i].green = floor(value);
		} else if (value > -1.0 && value < 0.0) {
			imageOut->data[i].green = 0;
		} else {
			imageOut->data[i].green = floor(value);
		}

		value = (imageInLarge->data[i].blue - imageInSmall->data[i].blue);
		if(value > 255)
			imageOut->data[i].blue = 255;
		else if (value < -1.0) {
			value = 257.0+value;
			if(value > 255)
				imageOut->data[i].blue = 255;
			else
				imageOut->data[i].blue = floor(value);
		} else if (value > -1.0 && value < 0.0) {
			imageOut->data[i].blue = 0;
		} else {
			imageOut->data[i].blue = floor(value);
		}
	}


	return imageOut;
}


int main(int argc, char** argv) {

	PPMImage *image;

	if(argc > 1) {
		image = readPPM("flower.ppm");
	} else {
		image = readStreamPPM(stdin);
	}
	// Process the tiny case:
	int size = 2;
	//Allocate memory
	AccurateImage *imageAccurate1_tiny = convertImageToNewFormat(image);
	AccurateImage *imageAccurate2_tiny = convertImageToNewFormat(image);
	//Do the computation
	performNewIdeaIteration(imageAccurate2_tiny, imageAccurate1_tiny, size);
	//Free memory no longer in use
	free(imageAccurate1_tiny->data);
	free(imageAccurate1_tiny);

	// Process the small case:
	size = 3;
	AccurateImage *imageAccurate1_small = convertImageToNewFormat(image);
	AccurateImage *imageAccurate2_small = convertImageToNewFormat(image);
	performNewIdeaIteration(imageAccurate2_small, imageAccurate1_small, size);
	free(imageAccurate1_small->data);
	free(imageAccurate1_small);
	//Finalize small case
	PPMImage *final_tiny = performNewIdeaFinalization(imageAccurate2_tiny,  imageAccurate2_small);
	//This is when tiny data is no longer in use
	free(imageAccurate2_tiny->data);
	free(imageAccurate2_tiny);

	// Process the medium case:
	size = 5;
	AccurateImage *imageAccurate1_medium = convertImageToNewFormat(image);
	AccurateImage *imageAccurate2_medium = convertImageToNewFormat(image);
	performNewIdeaIteration(imageAccurate2_medium, imageAccurate1_medium, size);
	free(imageAccurate1_medium->data);
	free(imageAccurate1_medium);
	PPMImage *final_small = performNewIdeaFinalization(imageAccurate2_small,  imageAccurate2_medium);
	free(imageAccurate2_small->data);
	free(imageAccurate2_small);

	//Process the large case:
	size = 8;
	AccurateImage *imageAccurate1_large = convertImageToNewFormat(image);
	AccurateImage *imageAccurate2_large = convertImageToNewFormat(image);
	free(image->data);
	free(image);
	performNewIdeaIteration(imageAccurate2_large, imageAccurate1_large, size);
	free(imageAccurate1_large->data);
	free(imageAccurate1_large);
	PPMImage *final_medium = performNewIdeaFinalization(imageAccurate2_medium,  imageAccurate2_large);
	free(imageAccurate2_medium->data);
	free(imageAccurate2_medium);
	free(imageAccurate2_large->data);
	free(imageAccurate2_large);

	if(argc > 1) {
		writePPM("flower_tiny.ppm", final_tiny);
		writePPM("flower_small.ppm", final_small);
		writePPM("flower_medium.ppm", final_medium);
	} else {
		writeStreamPPM(stdout, final_tiny);
		writeStreamPPM(stdout, final_small);
		writeStreamPPM(stdout, final_medium);
	}
	free(final_tiny->data);
	free(final_tiny);
	free(final_small->data);
	free(final_small);
	free(final_medium->data);
	free(final_medium);

	return 0;
}

