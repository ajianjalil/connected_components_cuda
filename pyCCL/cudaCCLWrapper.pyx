import numpy as np
cimport numpy as np # for np.ndarray
from libcpp.string cimport string
from libc.string cimport memcpy
from libcpp.map cimport map
from libcpp.vector cimport vector
from cython.operator cimport dereference as deref
from libc.stdlib cimport malloc, free
import cupy as cp
'''
cdef extern from "opencv2/core/core.hpp":
  cdef int  CV_WINDOW_AUTOSIZE
  cdef int CV_8UC1

cdef extern from "opencv2/core/core.hpp" namespace "cv":
  cdef cppclass Mat:
    Mat() except +
    void create(int, int, int)
    void* data

cdef extern from "opencv2/highgui/highgui.hpp" namespace "cv":
  void namedWindow(const string, int flag)
  void imshow(const string, Mat)
  int  waitKey(int delay)


cdef void ary2cvMat(np.ndarray ary, Mat& out):
  cdef np.ndarray[np.uint8_t, ndim=2, mode = 'c'] np_buff = np.ascontiguousarray(ary, dtype = np.uint8)
  cdef unsigned char* im_buff = <unsigned char*> np_buff.data
  cdef int r = ary.shape[0]
  cdef int c = ary.shape[1]
  out.create(r, c, CV_8UC1)
  memcpy(out.data, im_buff, r*c)
'''

cdef extern from "cudaCCL.cuh":
    void processCCL(char* image, int numRows, int numCols, int* numComponents, int* boundingBoxes)

cdef processCCLWrapper(size_t image, int numRows, int numCols, int* numComponents, size_t boundingBoxes):
  processCCL(<char*> image,numRows,numCols, <int*> numComponents, <int*> boundingBoxes)

def PyprocessCCL(image,width,height,boxes):

    # cdef int* boundingBoxes = <int*>malloc(width * height * 4 * sizeof(int))
    cdef int numComponents

    img_ptr = image.data.ptr
    boundingBoxes = boxes.data.ptr
    processCCLWrapper(img_ptr, width, height, &numComponents , boundingBoxes)

    # Convert the bounding boxes to a NumPy array
    # cdef np.ndarray[np.int32_t, ndim=2] result = np.zeros((numComponents, 4), dtype=np.int32)
    # for i in range(numComponents):
    #     result[i,0] = boundingBoxes[4*i]
    #     result[i,1] = boundingBoxes[4*i+1]
    #     result[i,2] = boundingBoxes[4*i+2]
    #     result[i,3] = boundingBoxes[4*i+3]
    # free(boundingBoxes)

    # return result