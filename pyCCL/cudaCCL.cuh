#ifndef CUDACCL_H
#define CUDACCL_H

#include <opencv2/core.hpp>

void processCCL(char* d_img, int numRows, int numCols, int* numComponents, int* boundingBoxes);

#endif