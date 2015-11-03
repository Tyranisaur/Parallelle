#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include "ppmCU.h"

// Image from:
// http://7-themes.com/6971875-funny-flowers-pictures.html

typedef struct {
     float red,green,blue;
} AccuratePixel;

typedef struct {
     int x, y;
     AccuratePixel *data;
} AccurateImage;

__global__ void performNewIdeaIterationGPU(AccurateImage * output, AccurateImage * input, int * size )
 {
	int senterY = blockIdx.x;
	int senterX = threadIdx.x;
	
	float sumR = 0;
	float sumG = 0;
	float sumB = 0;
	int countIncluded = 0;
	for(int x = -(*size); x <= *size; x++) {
	
		for(int y = -(*size); y <= *size; y++) {
			int currentX = senterX + x;
			int currentY = senterY + y;
			
			// Check if we are outside the bounds
			if(currentX < 0)
				continue;
			if(currentX >= input->x)
				continue;
			if(currentY < 0)
				continue;
			if(currentY >= input->y)
				continue;
			
			// Now we can begin
			int numberOfValuesInEachRow = input->x; 
			int offsetOfThePixel = (numberOfValuesInEachRow * currentY + currentX);
			sumR += input->data[offsetOfThePixel].red;
			sumG += input->data[offsetOfThePixel].green;
			sumB += input->data[offsetOfThePixel].blue;
		
			// Keep track of how many values we have included
			countIncluded++;
		}
	
	}
		
	// Now we compute the final value for all colours
	float valueR = sumR / countIncluded;
	float valueG = sumG / countIncluded;
	float valueB = sumB / countIncluded;
	
	// Update the output image
	int numberOfValuesInEachRow = output->x; // R, G and B
	int offsetOfThePixel = (numberOfValuesInEachRow * senterY + senterX);
	output->data[offsetOfThePixel].red = valueR;
	output->data[offsetOfThePixel].green = valueG;
	output->data[offsetOfThePixel].blue = valueB;
 
 
 }

// Finalization function assumes allocated pointers
__global__ void performNewIdeaFinalizationGPU( AccuarateImage * smallImage, AccurateImage * bigImage, PPMImage * outputImage)
{
	int index = blockIdx.x * blockDim-x + threadIdx.x;

	float value = (bigImage->data[index].red - smallImage->data[index].red);
		if(value > 255.0f)
			outputImage->data[index].red = 255;
		else if (value < -1.0f) {
			value = 257.0f+value;
			if(value > 255.0f)
				outputImage->data[index].red = 255;
			else
				outputImage->data[index].red = floorf(value);
		} else if (value > -1.0f && value < 0.0f) {
			outputImage->data[index].red = 0;
		} else {
			outputImage->data[index].red = floorf(value);
		}
		
		value = (bigImage->data[index].green - smallImage->data[index].green);
		if(value > 255.0f)
			outputImage->data[index].green = 255;
		else if (value < -1.0f) {
			value = 257.0f+value;
			if(value > 255.0f)
				outputImage->data[index].green = 255;
			else
				outputImage->data[index].green = floorf(value);
		} else if (value > -1.0f && value < 0.0f) {
			outputImage->data[index].green = 0;
		} else {
			outputImage->data[index].green = floorf(value);
		}
		
		value = (bigImage->data[index].blue - smallImage->data[index].blue);
		if(value > 255.0f)
			outputImage->data[index].blue = 255;
		else if (value < -1.0f) {
			value = 257.0f+value;
			if(value > 255.0f)
				outputImage->data[index].blue = 255;
			else
				outputImage->data[index].blue = floorf(value);
		} else if (value > -1.0f && value < 0.0f) {
			outputImage->data[index].blue = 0;
		} else {
			outputImage->data[index].blue = floorf(value);
		}
}

//conversion function takes in allocated pointers and fills in output pointer
__global__ void convertImageToNewFormatGPU( PPMImage * inputImage, AccurateImage * outputImage )
{
	int index = blckIdx.x * blockDim.x + threadIdx.x;
	outputImage->data[index].red   = (float) inputImage->data[index].red;
	outputImage->data[index].green = (float) inputImage->data[index].green;
	outputImage->data[index].blue  = (float) inputImage->data[index].blue;
}




int main(int argc, char** argv) {
	
	PPMImage *image;
	PPMIMage * gpuImage, gpuOutImage;
	AccurateImage * gpuUnchanged, gpuSmall, gpuBig, gpuBuffer;
	int* gpuFilter;
    int* filter;
	if(argc > 1) {
		image = readPPM("flower.ppm");
	} else {
		image = readStreamPPM(stdin);
	}
	int x, y;
	x = image->x;
	y = image->y;

	cudaMalloc((void**) &gpuImage, sizeof(PPMImage);
	cudaMemcpy(gpuImage, image, sizeof(PPMImage), cudaMemcpyHostToDevice);
	cudaMalloc((void**) &(gpuImage->data), sizeof(PPMPixel) * x * y);
	cudaMemcpy(gpuImage->data, image->data, sizeof(PPMPixel) * x * y, cudaMemcpyHostToDevice);
	
	cudaMalloc((void**) &gpuUnchanged, sizeof(AccurateImage));
	cudaMalloc((void**) &(gpuUnchanged->data), sizeof(AccuratePixel) * x * y);
	
	convertImageToNewFormatGPU<<<y, x>>>(gpuImage, gpuUnchanged);
	
	
	cudaMalloc((void**) &gpuBuffer, sizeof(AccurateImage));
	cudaMemcpy((void*) &(gpuBuffer->x), &x, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy((void*) &(gpuBuffer->y), &y, sizeof(int), cudaMemcpyHostToDevice);
	cudaMalloc((void**) &(gpuBuffer->data), sizeof(AccuratePixel) * x * y);
	
	cudaMalloc((void**) &gpuSmall, sizeof(AccurateImage));
	cudaMemcpy((void*) &(gpuSmall->x), &x, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy((void*) &(gpuSmall->y), &y, sizeof(int), cudaMemcpyHostToDevice);
	cudaMalloc((void**) &(gpuSmall->data), sizeof(AccuratePixel) * x * y);
	
	cudaMalloc((void**) &gpuBig, sizeof(AccurateImage));
	cudaMemcpy((void*) &(gpuBig->x), &x, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy((void*) &(gpuBig->y), &y, sizeof(int), cudaMemcpyHostToDevice);
	cudaMalloc((void**) &(gpuBig->data), sizeof(AccuratePixel) * x * y);
	
	cudaMalloc((void**) &gpuOutImage, sizeof(PPMImage));
	cudaMemcpy((void*) &(gpuOutImage->x), &x, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy((void*) &(gpuOutImage->y), &y, sizeof(int), cudaMemcpyHostToDevice);
	cudaMalloc((void**) &(gpuOutImage->data), sizeof(PPMImage) * x * y);

	cudaMalloc((void**) &gpuFilter, sizeof(int));
	filter* = 2;
	cudaMemcpy((void**)&gpuFilter, &filter, sizeof(int), cudaMemcpyHostToDevice);

	performNewIdeaIterationGPU<<<y, x>>>(gpuSmall, gpuUnchanged, gpuFilter);
	performNewIdeaIterationGPU<<<y, x>>>(gpuBuffer, gpuSmall, gpuFilter);
	performNewIdeaIterationGPU<<<y, x>>>(gpuSmall, gpuBuffer, gpuFilter);
	performNewIdeaIterationGPU<<<y, x>>>(gpuBuffer, gpuSmall, gpuFilter);
	performNewIdeaIterationGPU<<<y, x>>>(gpuSmall, gpuBuffer, gpuFilter);
	
	filter* = 3;
	cudaMemcpy((void**)&gpuFilter, &filter, sizeof(int), cudaMemcpyHostToDevice);
	
	performNewIdeaIterationGPU<<<y, x>>>(gpuBig, gpuUnchanged, gpuFilter);
	performNewIdeaIterationGPU<<<y, x>>>(gpuBuffer, gpuBig, gpuFilter);
	performNewIdeaIterationGPU<<<y, x>>>(gpuBig, gpuBuffer, gpuFilter);
	performNewIdeaIterationGPU<<<y, x>>>(gpuBuffer, gpuBig, gpuFilter);
	performNewIdeaIterationGPU<<<y, x>>>(gpuBig, gpuBuffer, gpuFilter);
	
	
	performNewIdeaFinalizationGPU<<<y, x>>>(gpuSmall, gpuBig, gpuOutImage);
	cudaMemcpy(image->data, gpuOutImage->data, sizeof(PPMPixel) * x * y, cudaMemcpyDeviceToHost);

	if(argc > 1) {
		writePPM("flower_tiny.ppm", image);
	} else {
		writeStreamPPM(stdout, image);
	}

	filter* = 5;
	cudaMemcpy((void**)&gpuFilter, &filter, sizeof(int), cudaMemcpyHostToDevice);
	
	performNewIdeaIterationGPU<<<y, x>>>(gpuSmall, gpuUnchanged, gpuFilter);
	performNewIdeaIterationGPU<<<y, x>>>(gpuBuffer, gpuSmall, gpuFilter);
	performNewIdeaIterationGPU<<<y, x>>>(gpuSmall, gpuBuffer, gpuFilter);
	performNewIdeaIterationGPU<<<y, x>>>(gpuBuffer, gpuSmall, gpuFilter);
	performNewIdeaIterationGPU<<<y, x>>>(gpuSmall, gpuBuffer, gpuFilter);
	
	performNewIdeaFinalizationGPU<<<y, x>>>(gpuBig, gpuSmall, gpuOutImage);
	cudaMemcpy(image->data, gpuOutImage->data, sizeof(PPMPixel) * x * y, cudaMemcpyDeviceToHost);

	if(argc > 1) {
		writePPM("flower_small.ppm", image);
	} else {
		writeStreamPPM(stdout, image);
	}
	
	filter* = 8;
	cudaMemcpy((void**)&gpuFilter, &filter, sizeof(int), cudaMemcpyHostToDevice);

	performNewIdeaIterationGPU<<<y, x>>>(gpuBig, gpuUnchanged, gpuFilter);
	performNewIdeaIterationGPU<<<y, x>>>(gpuBuffer, gpuBig, gpuFilter);
	performNewIdeaIterationGPU<<<y, x>>>(gpuBig, gpuBuffer, gpuFilter);
	performNewIdeaIterationGPU<<<y, x>>>(gpuBuffer, gpuBig, gpuFilter);
	performNewIdeaIterationGPU<<<y, x>>>(gpuBig, gpuBuffer, gpuFilter);

	performNewIdeaFinalizationGPU<<<y, x>>>(gpuSmall, gpuBig, gpuOutImage);
	cudaMemcpy(image->data, gpuOutImage->data, sizeof(PPMPixel) * x * y, cudaMemcpyDeviceToHost);
	
	if(argc > 1) {
		writePPM("flower_medium.ppm", image);
	} else {
		writeStreamPPM(stdout, image);
	}

	cudaFree(gpuFilter);
	cudaFree(gpuUnchanged->data);
	cudaFree(gpuSmall->data);
	cudaFree(gpuBig->data);
	cudaFree(gpuBuffer->data);
	cudaFree(gpuUnchanged);
	cudaFree(gpuSmall);
	cudaFree(gpuBig);
	cudaFree(gpuBuffer);
	
	
	free(image->data);
	free(image);
	
	return 0;
}

