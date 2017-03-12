/************************************************************************/
// The purpose of this file is to provide a GPU implementation of the 
// heat transfer simulation using MATLAB.
//
// Author: Jason Lowden
// Date: October 20, 2013
//
// File: KMeans.h
/************************************************************************/
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <texture_fetch_functions.h>
#include <cuda_texture_types.h>
#include <iostream>
#include "HeatTransfer.h"

texture<float, cudaTextureType2D, cudaReadModeElementType> heatTexture;

__global__ void UpdateHeatMapKernel(float *OutputData, int size, float heatSpeed)
{
	int x=threadIdx.x + blockIdx.x*blockDim.x;
	int y=threadIdx.y + blockIdx.y*blockDim.y;
	int tID = x+(y*size);
	float t_center, t_left, t_right, t_bottom, t_top;
	if(x>0 && x<(size-1) && y>0 && y<(size-1))
	{
		t_top= tex2D(heatTexture,x,y-1);
		t_left= tex2D(heatTexture,x-1,y);
		t_center= tex2D(heatTexture,x,y);
		t_right= tex2D(heatTexture,x+1,y);
		t_bottom= tex2D(heatTexture,x,y+1); 
		OutputData[tID] = t_center + ((t_top + t_left + t_right + t_bottom - (4 * t_center)) * heatSpeed);
	}

}

bool UpdateHeatMap(float* dataIn, float* dataOut, int size, float heatSpeed, int numIterations)
{
	cudaError_t status;
	float *OutputData;
	cudaArray_t InputData;
	int bytes = size * size * sizeof(float);
	cudaMalloc((void**) &OutputData, bytes);
	cudaMemcpy(OutputData, dataIn, (size*size*sizeof(float)), cudaMemcpyHostToDevice);
	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();

	cudaMallocArray (&InputData, &desc, size, size, 0);	cudaMemcpyToArray (InputData, 0, 0, dataIn, (size*size*sizeof(float)),  cudaMemcpyHostToDevice);	cudaBindTextureToArray (&heatTexture, InputData, &desc);	for(int i=0;i<numIterations;i++)	{		dim3 dimBlock(16, 16); 
		dim3 dimGrid((int)ceil((float)size/16), (int)ceil((float)size/16));		UpdateHeatMapKernel<<<dimGrid, dimBlock>>>(OutputData, size , heatSpeed);		cudaThreadSynchronize();
		// Check for errors
		status = cudaGetLastError();
		if (status != cudaSuccess) 
		{
			std::cout << "Kernel failed: " << cudaGetErrorString(status) << std::endl;
			cudaUnbindTexture (&heatTexture);
			cudaFree(OutputData);
			return false;
		}		cudaUnbindTexture (&heatTexture);		cudaMemcpyToArray (InputData, 0, 0, OutputData, (size*size*sizeof(float)),  cudaMemcpyDeviceToDevice);		cudaBindTextureToArray (&heatTexture, InputData, &desc);	}	cudaMemcpy(dataOut, OutputData, (size*size*sizeof(float)), cudaMemcpyDeviceToHost);
	cudaFree(OutputData);
	cudaUnbindTexture (&heatTexture);
	cudaFreeArray(InputData);
	return true;
}