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
			  		  
	int senterY = blockIdx.y
				+ blockIdx.z * gridDim.y;
	
	int senterX = threadIdx.x
				+ blockIdx.x * blockDim.x;
	
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
__global__ void performNewIdeaFinalizationGPU( AccurateImage * smallImage, AccurateImage * bigImage, PPMImage * outputImage)
{
	int index =  threadIdx.x
			  +  blockIdx.x * blockDim.x
			  +  blockIdx.y * blockDim.x * gridDim.x 
			  +  blockIdx.z * blockDim.x * gridDim.x * gridDim.y;

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
	int index =  threadIdx.x
			  +  blockIdx.x * blockDim.x
			  +  blockIdx.y * blockDim.x * gridDim.x 
			  +  blockIdx.z * blockDim.x * gridDim.x * gridDim.y;
			  
	outputImage->data[index].red   = (float) inputImage->data[index].red;
	outputImage->data[index].green = (float) inputImage->data[index].green;
	outputImage->data[index].blue  = (float) inputImage->data[index].blue;
}




int main(int argc, char** argv) {
	
	PPMImage *image;
	PPMImage * gpuImage, *gpuOutImage;
	AccurateImage * gpuUnchanged, *gpuSmall, *gpuBig, *gpuBuffer;
	PPMPixel * ppmPixelPtr;
	AccuratePixel * accuratePixelPtr;
	dim3 gridBlock;
	gridBlock.x = 60;
	gridBlock.y = 40;
	gridBlock.z = 30;
	int* gpuFilter;
    int * filter = (int*)malloc(sizeof(int));
	if(argc > 1) {
		image = readPPM("flower.ppm");
	} else {
		image = readStreamPPM(stdin);
	}
	//Image dimmensions are constant, store these here instead of looking up in some struct every time
	int x, y;
	x = image->x;
	y = image->y;

	//Allocate struct for on device
	cudaMalloc((void**) &gpuImage, sizeof(PPMImage));
	
	//Allocate memory for array on device
	cudaMalloc((void**) &ppmPixelPtr, sizeof(PPMPixel) * x * y);

	//Copy address of allocated array to device
	cudaMemcpy(&(gpuImage->data), &ppmPixelPtr, sizeof(PPMPixel*), cudaMemcpyHostToDevice);

	//Copy array to address previously copied
	cudaMemcpy(ppmPixelPtr, image->data, sizeof(PPMPixel) * x * y, cudaMemcpyHostToDevice);

	//Copy int values
	cudaMemcpy(&(gpuImage->x), &x, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(&(gpuImage->y), &y, sizeof(int), cudaMemcpyHostToDevice);

	//Allocate and copy values for unchanged struct with new format
	cudaMalloc((void**) &gpuUnchanged, sizeof(AccurateImage));
	cudaMalloc((void**) &(accuratePixelPtr), sizeof(AccuratePixel) * x * y);
	cudaMemcpy(&(gpuUnchanged->data), &accuratePixelPtr, sizeof(AccuratePixel*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(gpuUnchanged->y), &y, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(&(gpuUnchanged->x), &x, sizeof(int), cudaMemcpyHostToDevice);
	
	//Call kernel to fill in the values of unchanged struct
	convertImageToNewFormatGPU<<<gridBlock, 32>>>(gpuImage, gpuUnchanged);
	
	
	
	cudaMalloc((void**) &gpuBuffer, sizeof(AccurateImage));
	cudaMalloc((void**) &(accuratePixelPtr), sizeof(AccuratePixel) * x * y);
	cudaMemcpy(&(gpuBuffer->data), &accuratePixelPtr, sizeof(AccuratePixel*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(gpuBuffer->y), &y, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(&(gpuBuffer->x), &x, sizeof(int), cudaMemcpyHostToDevice);
	
	
	cudaMalloc((void**) &gpuSmall, sizeof(AccurateImage));
	cudaMalloc((void**) &(accuratePixelPtr), sizeof(AccuratePixel) * x * y);
	cudaMemcpy(&(gpuSmall->data), &accuratePixelPtr, sizeof(AccuratePixel*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(gpuSmall->y), &y, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(&(gpuSmall->x), &x, sizeof(int), cudaMemcpyHostToDevice);
	
	
	cudaMalloc((void**) &gpuBig, sizeof(AccurateImage));
	cudaMalloc((void**) &(accuratePixelPtr), sizeof(AccuratePixel) * x * y);
	cudaMemcpy(&(gpuBig->data), &accuratePixelPtr, sizeof(AccuratePixel*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(gpuBig->y), &y, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(&(gpuBig->x), &x, sizeof(int), cudaMemcpyHostToDevice);
	
	
	cudaMalloc((void**) &gpuOutImage, sizeof(PPMImage));
	cudaMalloc((void**) &(ppmPixelPtr), sizeof(PPMPixel) * x * y);
	cudaMemcpy(&(gpuOutImage->data), &ppmPixelPtr, sizeof(PPMPixel*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(gpuOutImage->y), &y, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(&(gpuOutImage->x), &x, sizeof(int), cudaMemcpyHostToDevice);
	
	
	cudaMalloc((void**) &gpuFilter, sizeof(int));
	*filter = 2;
	cudaMemcpy(gpuFilter, filter, sizeof(int), cudaMemcpyHostToDevice);

	performNewIdeaIterationGPU<<<gridBlock, 32>>>(gpuSmall, gpuUnchanged, gpuFilter);
	performNewIdeaIterationGPU<<<gridBlock, 32>>>(gpuBuffer, gpuSmall, gpuFilter);
	performNewIdeaIterationGPU<<<gridBlock, 32>>>(gpuSmall, gpuBuffer, gpuFilter);
	performNewIdeaIterationGPU<<<gridBlock, 32>>>(gpuBuffer, gpuSmall, gpuFilter);
	performNewIdeaIterationGPU<<<gridBlock, 32>>>(gpuSmall, gpuBuffer, gpuFilter);
	
	*filter = 3;
	cudaMemcpy(gpuFilter, filter, sizeof(int), cudaMemcpyHostToDevice);
	
	performNewIdeaIterationGPU<<<gridBlock, 32>>>(gpuBig, gpuUnchanged, gpuFilter);
	performNewIdeaIterationGPU<<<gridBlock, 32>>>(gpuBuffer, gpuBig, gpuFilter);
	performNewIdeaIterationGPU<<<gridBlock, 32>>>(gpuBig, gpuBuffer, gpuFilter);
	performNewIdeaIterationGPU<<<gridBlock, 32>>>(gpuBuffer, gpuBig, gpuFilter);
	performNewIdeaIterationGPU<<<gridBlock, 32>>>(gpuBig, gpuBuffer, gpuFilter);
	
	
	performNewIdeaFinalizationGPU<<<gridBlock, 32>>>(gpuSmall, gpuBig, gpuOutImage);
	printf("this is supposed to happen\n");
	//TODO ------------------------------------------------------------------------------------------------
	//TODO ------------------------------------------------------------------------------------------------
	//TODO ------------------------------------------------------------------------------------------------
	cudaMemcpy(image->data, gpuOutImage->data, sizeof(PPMPixel) * x * y, cudaMemcpyDeviceToHost);
	printf("this is not supposed to happen\n");

	if(argc > 1) {
		writePPM("flower_tiny.ppm", image);
	} else {
		writeStreamPPM(stdout, image);
	}

	*filter = 5;
	cudaMemcpy(gpuFilter, filter, sizeof(int), cudaMemcpyHostToDevice);
	
	performNewIdeaIterationGPU<<<gridBlock, 32>>>(gpuSmall, gpuUnchanged, gpuFilter);
	performNewIdeaIterationGPU<<<gridBlock, 32>>>(gpuBuffer, gpuSmall, gpuFilter);
	performNewIdeaIterationGPU<<<gridBlock, 32>>>(gpuSmall, gpuBuffer, gpuFilter);
	performNewIdeaIterationGPU<<<gridBlock, 32>>>(gpuBuffer, gpuSmall, gpuFilter);
	performNewIdeaIterationGPU<<<gridBlock, 32>>>(gpuSmall, gpuBuffer, gpuFilter);
	
	performNewIdeaFinalizationGPU<<<gridBlock, 32>>>(gpuBig, gpuSmall, gpuOutImage);
	cudaMemcpy(image->data, gpuOutImage->data, sizeof(PPMPixel) * x * y, cudaMemcpyDeviceToHost);

	if(argc > 1) {
		writePPM("flower_small.ppm", image);
	} else {
		writeStreamPPM(stdout, image);
	}
	
	*filter = 8;
	cudaMemcpy(gpuFilter, filter, sizeof(int), cudaMemcpyHostToDevice);

	performNewIdeaIterationGPU<<<gridBlock, 32>>>(gpuBig, gpuUnchanged, gpuFilter);
	performNewIdeaIterationGPU<<<gridBlock, 32>>>(gpuBuffer, gpuBig, gpuFilter);
	performNewIdeaIterationGPU<<<gridBlock, 32>>>(gpuBig, gpuBuffer, gpuFilter);
	performNewIdeaIterationGPU<<<gridBlock, 32>>>(gpuBuffer, gpuBig, gpuFilter);
	performNewIdeaIterationGPU<<<gridBlock, 32>>>(gpuBig, gpuBuffer, gpuFilter);

	performNewIdeaFinalizationGPU<<<gridBlock, 32>>>(gpuSmall, gpuBig, gpuOutImage);
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
	free(filter);
	return 0;
}

