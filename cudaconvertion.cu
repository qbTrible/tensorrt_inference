#ifdef USE_CCA
#include "cudaconvertion.h"

__global__ void convertBGR2RGBfloatKernel(uchar3 *src, float3 *dst, int width, int height)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= width || y >= height) {
        return;
    }

    uchar3 color = src[y * width + x];
    dst[y * width + x] = make_float3(color.z, color.y, color.x);
}

void convertBGR2RGBfloat(void *src, void *dst, int width, int height, cudaStream_t stream)
{
    dim3 grids((width + 31) / 32, (height + 7) / 8);
    dim3 blocks(32, 8);
    convertBGR2RGBfloatKernel<<<grids, blocks, 0, stream>>>((uchar3 *)src, (float3 *)dst, width, height);
    // cudaDeviceSynchronize();
}

__global__ void convertGray2GrayfloatKernel(uchar1 *src, float1 *dst, int width, int height)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= width || y >= height) {
        return;
    }

    uchar1 color = src[y * width + x];
    dst[y * width + x] = make_float1(color.x);
}

void convertGray2Grayfloat(void *src, void *dst, int width, int height, cudaStream_t stream)
{
    dim3 grids((width + 31) / 32, (height + 7) / 8);
    dim3 blocks(32, 8);
    convertGray2GrayfloatKernel<<<grids, blocks, 0, stream>>>((uchar1 *)src, (float1 *)dst, width, height);
    // cudaDeviceSynchronize();
}

// __global__ void convertGpuMatBGR2RGBfloatKernel(cv::cuda::PtrStepSz<uchar3> src, float3 *dst, int width, int height)
// {
//     int x = threadIdx.x + blockIdx.x * blockDim.x;
//     int y = threadIdx.y + blockIdx.y * blockDim.y;
//     if (x >= width || y >= height) {
//         return;
//     }
    
//     uchar3 color = src(y, x);
//     dst[y * width + x] = make_float3(color.z, color.y, color.x);
// }

// void convertGpuMatBGR2RGBfloat(cv::cuda::GpuMat src, void *dst, int width, int height, cudaStream_t stream)
// {
//     dim3 blocks(32, 8);
//     dim3 grids((width + blocks.x - 1) / blocks.y, (height + blocks.y - 1) / blocks.y); 
//     convertGpuMatBGR2RGBfloatKernel<<<grids, blocks, 0, stream>>>(src, (float3 *)dst, width, height);
// }

__global__ void convertBGR2BGRfloatKernel(uchar3 *src, float3 *dst, int width, int height)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= width || y >= height) {
        return;
    }

    uchar3 color = src[y * width + x];
    dst[y * width + x] = make_float3(color.x, color.y, color.z);
}

void convertBGR2BGRfloat(void *src, void *dst, int width, int height, cudaStream_t stream)
{
    dim3 grids((width + 31) / 32, (height + 7) / 8);
    dim3 blocks(32, 8);
    convertBGR2RGBfloatKernel<<<grids, blocks, 0, stream>>>((uchar3 *)src, (float3 *)dst, width, height);
    // cudaDeviceSynchronize();
}

// ---------------------
// #### NORMALIZATION ####
// ---------------------
__global__ void imageNormalizationKernel(float3 *ptr, int width, int height, float3 im_mean, float3 im_scale)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= width || y >= height) {
        return;
    }
    float3 color = ptr[y * width + x];
    color.x = (color.x - im_mean.x) * im_scale.x;
    color.y = (color.y - im_mean.y) * im_scale.y;
    color.z = (color.z - im_mean.z) * im_scale.z;

    ptr[y * width + x] = make_float3(color.x, color.y, color.z);
}

void imageNormalization(void *ptr, int width, int height, float *mean, float *scale, cudaStream_t stream)
{
    float3 im_mean = make_float3(mean[0],mean[1],mean[2]);
    float3 im_scale = make_float3(scale[0],scale[1],scale[2]);

    dim3 grids((width + 31) / 32, (height + 31) / 32);
    dim3 blocks(32, 32);
    imageNormalizationKernel<<<grids, blocks, 0, stream>>>((float3 *)ptr, width, height, im_mean, im_scale);
    // cudaDeviceSynchronize();
}

__global__ void GrayNormalizationKernel(float1 *ptr, int width, int height, float1 im_mean, float1 im_scale)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= width || y >= height) {
        return;
    }
    float1 color = ptr[y * width + x];
    color.x = (color.x - im_mean.x) * im_scale.x;

    ptr[y * width + x] = make_float1(color.x);
}

void GrayNormalization(void *ptr, int width, int height, float *mean, float *scale, cudaStream_t stream)
{
    float1 im_mean = make_float1(mean[0]);
    float1 im_scale = make_float1(scale[0]);

    dim3 grids((width + 31) / 32, (height + 31) / 32);
    dim3 blocks(32, 32);
    GrayNormalizationKernel<<<grids, blocks, 0, stream>>>((float1 *)ptr, width, height, im_mean, im_scale);
    // cudaDeviceSynchronize();
}

// ---------------------
// #### SPLIT ####
// ---------------------
__global__ void imageSplitKernel(float3 *ptr, float *dst, int width, int height)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= width || y >= height) {
        return;
    }

    float3 color = ptr[y * width + x];

    dst[y * width + x] = color.x;
    dst[y * width + x + width * height] = color.y;
    dst[y * width + x + width * height * 2] = color.z;
}

void imageSplit(const void *src, float *dst, int width, int height, cudaStream_t stream)
{
    dim3 grids((width + 31) / 32, (height + 7) / 8);
    dim3 blocks(32, 8);
    imageSplitKernel<<<grids, blocks, 0, stream>>>((float3 *)src, (float *)dst, width, height);
    // cudaDeviceSynchronize();
}

__global__ void GraySplitKernel(float1 *ptr, float *dst, int width, int height)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= width || y >= height) {
        return;
    }

    float1 color = ptr[y * width + x];

    dst[y * width + x] = color.x;
}

void GraySplit(const void *src, float *dst, int width, int height, cudaStream_t stream)
{
    dim3 grids((width + 31) / 32, (height + 7) / 8);
    dim3 blocks(32, 8);
    GraySplitKernel<<<grids, blocks, 0, stream>>>((float1 *)src, (float *)dst, width, height);
    // cudaDeviceSynchronize();
}

__global__ void ResizeGPUKernel(const unsigned char* src, int srcWidth, int srcHeight, unsigned char* dst, int dstWidth, int dstHeight)
{
	//核函数会在每个thread上运行，这里求的x、y是当前thread的坐标，同时也代表当前要处理的像素的坐标
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int x = blockIdx.x * blockDim.x + threadIdx.x;	
	if (x >= dstWidth || y >= dstHeight) return;
	//以指针的形式操作图像，outPosition是指目标图像素在内存中的位置 
	int outPosition = y * dstWidth + x;		
	//求取对应原图的像素点，srcPosition是指原图像素在内存中的位置 
	int srcX = x * srcWidth / dstWidth;		//如果出现浮点数，这里就会向下取整，以此来表示最近邻
	int srcY = y * srcHeight / dstHeight;	//（如果不喜欢向下取整，也可以选择四舍五入）
	int srcPosition = srcY * srcWidth + srcX;		
	//为目标图像素赋值。RGB三通道，在内存中的位置是挨着的。
	dst[outPosition * 3 + 0] = src[srcPosition * 3 + 0];		
	dst[outPosition * 3 + 1] = src[srcPosition * 3 + 1];
	dst[outPosition * 3 + 2] = src[srcPosition * 3 + 2];
}

void ResizeGPU(const unsigned char* src, int srcWidth, int srcHeight, unsigned char* dst, int dstWidth, int dstHeight, cudaStream_t stream)
{
    dim3 grids((dstWidth + 31) / 32, (dstHeight + 31) / 32);
    dim3 blocks(32, 32);
	//调用核函数，重点关注blocks与threads的设置，这样设置是为了让thread的坐标代表目标图像素的坐标
	ResizeGPUKernel<<<grids, blocks, 0, stream>>>(src, srcWidth, srcHeight, dst, dstWidth, dstHeight);
    // cudaDeviceSynchronize();
}

//输入图像为BGR图，将其转化为gray图
__global__ void Rgb2grayKernel(uchar3 *dataIn, unsigned char *dataOut, int imgHeight, int imgWidth)
{
    //图片二维扫描，分别有x方向，y方向的像素点
    int xIndex = threadIdx.x + blockIdx.x * blockDim.x;	//表示x方向上的ID
    int yIndex = threadIdx.y + blockIdx.y * blockDim.y;	//表示y方向上的ID
    //灰度变换操作
    if (xIndex < imgWidth && yIndex < imgHeight)
    {
        uchar3 rgb = dataIn[yIndex * imgWidth + xIndex];
        dataOut[yIndex * imgWidth + xIndex] = 0.299f * rgb.x + 0.587f * rgb.y + 0.114f * rgb.z;
    }
}

void Rgb2gray(uchar3 *dataIn, unsigned char *dataOut, int imgHeight, int imgWidth, cudaStream_t stream)
{
    dim3 grids((imgWidth + 31) / 32, (imgHeight + 31) / 32);
    dim3 blocks(32, 32);
	//调用核函数，重点关注blocks与threads的设置，这样设置是为了让thread的坐标代表目标图像素的坐标
	Rgb2grayKernel<<<grids, blocks, 0, stream>>>(dataIn, dataOut, imgHeight, imgWidth);
    // cudaDeviceSynchronize();
}

#endif // (defined USE_CCA)