#ifdef USE_CCA

#ifndef INCLUDE_CCA_CU_H_
#define INCLUDE_CCA_CU_H_

#include <cuda_runtime.h>
// #include <opencv2/opencv.hpp>

void convertBGR2RGBfloat(void *src, void *dst, int width, int height, cudaStream_t stream);
void convertBGR2BGRfloat(void *src, void *dst, int width, int height, cudaStream_t stream);
void imageSplit(const void *src, float *dst, int width, int height, cudaStream_t stream);
void imageNormalization(void *ptr, int width, int height, float *mean, float *scale, cudaStream_t stream);
// void convertGpuMatBGR2RGBfloat(cv::cuda::GpuMat src, void *dst, int width, int height, cudaStream_t stream);

void convertGray2Grayfloat(void *src, void *dst, int width, int height, cudaStream_t stream);
void GraySplit(const void *src, float *dst, int width, int height, cudaStream_t stream);
void GrayNormalization(void *ptr, int width, int height, float *mean, float *scale, cudaStream_t stream);
void ResizeGPU(const unsigned char* src, int srcWidth, int srcHeight, unsigned char* dst, int dstWidth, int dstHeight, cudaStream_t stream);
void Rgb2gray(uchar3 *dataIn, unsigned char *dataOut, int imgHeight, int imgWidth, cudaStream_t stream);

#endif // INCLUDE_CCA_CU_H_
#endif // (defined USE_CCA)