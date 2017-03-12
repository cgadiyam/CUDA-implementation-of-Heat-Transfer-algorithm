/************************************************************************/
// The purpose of this file is to provide a heat transfer simulation for
// use with MATLAB.
//
// Author: Jason Lowden
// Date: October 20, 2013
//
// File: mexFunction.cpp
/************************************************************************/

#include <mex.h>

#include "..\HeatTransferCUDALib\HeatTransfer.h"

EXTERN_C void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
{
	//Check the arguments to make sure that the left and right parameters
	//were provided.
	if( nrhs == 3 )
	{
		if( nlhs == 1 )
		{
			//We will only be producing one output, which is the updated
			//heat map. Regardless of the number of inputs, we will only
			//be using the first one.

			//There are a total of 3 arguments that have to be passed.
			//They are listed as follows:
			//1.) The input heat array - the grid of temperatures. We can get the size of the matrix from the object.
			//2.) The "heat speed", which will be used for the simulation.
			//3.) The number of iterations that will need to be performed.

			//Do some error checking, just make sure that both dimensions
			//are the same for the input matrix.
			//m is the number of rows; n is the number of columns.
			int m = (int)mxGetM( prhs[0] );
			int n = (int)mxGetN( prhs[0] );

			//Check to make sure that we are only using a square matrix.
			if( m == n )
			{
				//Get the input matrix pointer to access the data.
				float* inputPtr = (float*)mxGetData(prhs[0]);

				//Get the values for the heat speed and the number of iterations.
				//By default, Matlab provides everything as a double precision value.
				double* heatSpeed = mxGetPr(prhs[1]);
				double* numIterations = mxGetPr(prhs[2]);

				//Create the output matrix.
				plhs[0] = mxCreateNumericMatrix(m,n,mxSINGLE_CLASS,mxREAL);
				//Get the pointer to work with the data.
				float* outputPtr = (float*)mxGetData(plhs[0]);

				//At this point, we have everything that is needed to call
				//the function to update the transfer map.
				//Determine if we are going to use texture memory or not.
				bool result = UpdateHeatMap(inputPtr,outputPtr,m,(float)*heatSpeed,(int)(*numIterations));
				if( !result )
				{
					mexErrMsgTxt("There was an error in the setup of the kernel.");
				}
			}
			else
			{
				mexErrMsgTxt("The input matrix must be a square matrix.");
			}
		}
		else
		{
			mexErrMsgTxt("Only one output variable can be provided.");
		}
	}
	else
	{
		mexErrMsgTxt("Right hand arguments must be provided.\nUsage: HeatTransferLab(heat_array,heatSpeed,numIterations,textureMemory)");
	}
}