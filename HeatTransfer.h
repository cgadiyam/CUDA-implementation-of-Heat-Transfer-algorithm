/************************************************************************/
// The purpose of this file is to provide a common header for the heat
// transfer simulation with the MEX/MATLAB files.
//
// Author: Jason Lowden
// Date: October 20, 2013
//
// File: KMeans.h
/************************************************************************/

#ifndef __K_MEANS_H__
#define __K_MEANS_H__

bool UpdateHeatMap(float* dataIn, float* dataOut, int size, float heatSpeed, int numIterations);

#endif