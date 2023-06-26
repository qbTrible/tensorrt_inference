#include <iostream>
#include <fstream>
#include <sstream>
#include <thread>
#include <functional>
#include <numeric>
#include "trt_inference.h"

inline int64_t volume(const nvinfer1::Dims& d)
{
    return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}

inline unsigned int getElementSize(nvinfer1::DataType t)
{
    switch (t)
    {
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF: return 2;
        case nvinfer1::DataType::kINT8: return 1;
    }
    throw std::runtime_error("Invalid DataType.");
    return 0;
}

inline void* safeCudaMalloc(size_t memSize)
{
    void* deviceMem = nullptr;
    cudaError_t rc = cudaMalloc(&deviceMem, memSize);
    // cudaMallocManaged(&deviceMem, memSize);
    if (rc != cudaSuccess)
    {
        printf("Could not allocate memory: %d\n", rc);
        exit(1);
    }
    // cudaDeviceSynchronize();
    return deviceMem;
}

TensorRTEngine::TensorRTEngine(const std::string &modelFile,
                      const std::vector<std::string> &_outputname,
                      const std::string &deviceForInference,
                      int maxBatch)
{
    this->outputname = _outputname;
    int count;
    cudaGetDeviceCount(&count);
    gpuid = std::stoi(deviceForInference.substr(deviceForInference.rfind(":") + 1,std::string::npos));
    std::cout << "GPU nums: " << count << ", Engine gpuid: " << gpuid << std::endl;

    if (gpuid < count) { 
        cudaSetDevice(gpuid);
    } else {
        std::cout << "Cannot find GPU " << gpuid << ", please check config file." << std::endl;
    }

    // 加载 trt 模型文件，生成 ICudaEngine
    engine = TensorRTEngineHandle::get_engine(modelFile, outputname, deviceForInference, maxBatch);
    maxbatchSize = engine->getMaxBatchSize();
    if(!engine) {
        std::cout << "Empty Engine, please check model path: " << modelFile << std::endl;
        return;
    }
    // 创建engine上下文
    context = engine->createExecutionContext();
    int nbBindings = engine->getNbBindings();
    // 输入可能有多个，设置显示输入形状
    for (int i=0; i<nbBindings; i++)
    {
        if (engine->bindingIsInput(i))
        {
            InputCount ++;
            nvinfer1::Dims inputDims = context->getBindingDimensions(i);
            if (!engine->hasImplicitBatchDimension() && engine->getNbOptimizationProfiles() > 0){
                const bool isDynamicInput = std::any_of(inputDims.d, inputDims.d + inputDims.nbDims, [](int dim){ return dim == -1; });
                if (isDynamicInput){
                    context->setBindingDimensions(i, nvinfer1::Dims4(maxBatch, inputDims.d[1], inputDims.d[2], inputDims.d[3]));
                    if (!context->allInputDimensionsSpecified())
                    {
                        std::cout << "InputDimensionsSpecified Failed!" << std::endl;
                        return;
                    }
                }
                if (maxBatch > maxbatchSize)
                {
                    std::cout << "Invalid Batch set: " << maxBatch << ", maxBatchSize: " << maxbatchSize << std::endl;
                    return;
                }
                maxbatchSize = maxBatch;
            }
            inputDims = context->getBindingDimensions(i);
            InputDims.push_back(inputDims);
        }
    }
    CudaBuffer.resize(nbBindings);
    BindBufferSize.resize(nbBindings);
    for (int i = 0; i < nbBindings; ++i)
    {
        nvinfer1::Dims dims = context->getBindingDimensions(i);
        nvinfer1::DataType dtype = engine->getBindingDataType(i);
        int64_t totalSize;
        if(!engine->hasImplicitBatchDimension()){
            totalSize = volume(dims) * getElementSize(dtype);
        }
        else
        {
            totalSize = volume(dims) * maxbatchSize * getElementSize(dtype);
        }
        BindBufferSize[i] = totalSize;
        CudaBuffer[i] = safeCudaMalloc(totalSize);
        BindDims.push_back(dims);
    }
    OutputCpuBuffer.resize(outputname.size());
    for(int i = 0;i < outputname.size(); i++){
        int outputIndex = engine->getBindingIndex(outputname[i].c_str());
        OutputCpuBuffer[i] = malloc(BindBufferSize[outputIndex]);
    }   

#if (defined USE_CCA)
    img_gpu_data8u.resize(InputCount);
    img_gpu_data32f.resize(InputCount);
    for (int i=0; i < InputDims.size(); i++)
    {
        int maxResize = volume(InputDims[i]);
        img_gpu_data8u[i].data = safeCudaMalloc(maxResize);
        img_gpu_data32f[i].data = safeCudaMalloc(maxResize * sizeof(float));
    } 
#endif
    cudaStreamCreate(&stream);
}

TensorRTEngine::~TensorRTEngine() {
    if(!context) {
        delete context;
        // context->destroy();  // Deprecated in TRT 8.0. Superseded by delete.
    }
    cudaStreamDestroy(stream);
    for(auto& item : OutputCpuBuffer) {
        free(item);
        // cudaFree(item);
    }
#if (defined USE_CCA)
    for (int i = 0; i < InputCount; i++)
    {
        if(img_gpu_data8u[i].data) {
        cudaFree(img_gpu_data8u[i].data);
        }
        if(img_gpu_data32f[i].data) {
            cudaFree(img_gpu_data32f[i].data);
        }
    }   
    img_gpu_data8u.clear();
    img_gpu_data32f.clear();
#endif
    for(auto& item : CudaBuffer) {
        cudaFree(item);
    }
}

void TensorRTEngine::SetRGB(int _RGBmodel)
{
    RGBmodel = _RGBmodel;
}

void TensorRTEngine::SetMean(float b, float g, float r)
{
    if (RGBmodel) image_mean = {r, g, b};
    else image_mean = {b, g, r};
}

void TensorRTEngine::SetScale(float b, float g, float r)
{
    if (RGBmodel) image_scale = {r, g, b};
    else image_scale = {b, g, r};
}

void TensorRTEngine::cvImageTocudaMem(std::vector<std::vector<cv::Mat>>& imgs_, std::vector<nvinfer1::Dims> dimensions)
{
    assert(imgs_.size() == dimensions.size());
    std::vector<int> channels(InputCount);
    std::vector<int> height(InputCount);
    std::vector<int> width(InputCount);
    for (int i = 0; i < InputCount; i++)
    {
        if(!engine->hasImplicitBatchDimension()){
            channels[i] = dimensions[i].d[1];
            height[i] = dimensions[i].d[2];
            width[i] = dimensions[i].d[3];
        }else{
            channels[i] = dimensions[i].d[0];
            height[i] = dimensions[i].d[1];
            width[i] = dimensions[i].d[2];
        }
    }
    std::vector<float *> inputData(InputCount);
    for (size_t j = 0; j < InputCount; j++)
    {
        inputData[j] = (float*)CudaBuffer[j];
    }
    
#ifdef USE_CCA
    for(size_t i = 0; i < imgs_[0].size(); i++) {
        for (size_t j = 0; j < InputCount; j++)
        {
            void *temp_img = safeCudaMalloc(imgs_[j][i].rows * imgs_[j][i].cols * imgs_[j][i].channels());
            if (channels[j] == 1 && imgs_[j][i].channels() != 1) 
            {
                // Rgb2gray((uchar3 *)imgs_[j][i].data, (unsigned char *)temp_img, imgs_[j][i].rows, imgs_[j][i].cols, stream);
                cv::cvtColor(imgs_[j][i], imgs_[j][i], cv::COLOR_BGR2GRAY);
            }
            cudaMemcpyAsync(temp_img, imgs_[j][i].data, imgs_[j][i].rows * imgs_[j][i].cols * imgs_[j][i].channels(), cudaMemcpyHostToDevice, stream);
            if (imgs_[j][i].cols != width[j] || imgs_[j][i].rows || height[j])
            {
                ResizeGPU((uint8_t *)temp_img, imgs_[j][i].cols, imgs_[j][i].rows, (uint8_t *)img_gpu_data8u[j].data, width[j], height[j], stream);
                // cv::resize(imgs_[j][i], resized, cv::Size(width[j], height[j]));
            }
            else
            {
                img_gpu_data8u[j].data = temp_img;
            }
 
            if (channels[j] == 3){
                if (RGBmodel) convertBGR2RGBfloat(img_gpu_data8u[j].data, img_gpu_data32f[j].data, width[j], height[j], stream);
                else convertBGR2BGRfloat(img_gpu_data8u[j].data, img_gpu_data32f[j].data, width[j], height[j], stream);
                imageNormalization(img_gpu_data32f[j].data, width[j], height[j], &image_mean[0], &image_scale[0], stream);
                imageSplit(img_gpu_data32f[j].data, inputData[j], width[j], height[j], stream);
            }
            else if (channels[j] == 1){
                convertGray2Grayfloat(img_gpu_data8u[j].data, img_gpu_data32f[j].data, width[j], height[j], stream);
                GrayNormalization(img_gpu_data32f[j].data, width[j], height[j], &image_mean[0], &image_scale[0], stream);
                GraySplit(img_gpu_data32f[j].data, inputData[j], width[j], height[j], stream);
            }
            else{
                std::cout << "Channels == " << channels[j] << " not support!" << std::endl;
                return;
            }
            inputData[j] += width[j] * height[j] * channels[j];
        }      
    }
#else
    for(size_t j = 0; j < imgs_[0].size(); j++) {
        for (size_t i = 0; i < InputCount; i++)
        {
            cv::Mat blob, resized;
            if (imgs_[i][j].cols != width[i] || imgs_[i][j].rows || height[i])
            {
                cv::resize(imgs_[i][j], resized, cv::Size(width[i], height[i]));
            }
            else
            {
                resized = imgs_[i][j];
            }
            if (RGBmodel){
                cv::dnn::blobFromImage(imgs_[i][j], blob, image_scale[0], cv::Size(width[i], height[i]), cv::Scalar(image_mean[0],image_mean[1],image_mean[2]), true, false, CV_32F);
            }
            else{
                cv::dnn::blobFromImage(imgs_[i][j], blob, image_scale[0], cv::Size(width[i], height[i]), cv::Scalar(image_mean[0],image_mean[1],image_mean[2]), false, false, CV_32F);
            }        
            cudaMemcpyAsync(inputData[i], blob.data, channels[i] * width[i] * height[i] * sizeof(float), cudaMemcpyHostToDevice, stream);
            // blob.data = inputData[i];
            inputData[i] += width[i] * height[i] * channels[i];
        }
    }
#endif //(defined USE_CCA)
}

void TensorRTEngine::fill_batch_result(int start, int cnt, std::vector<IERlt>& rlts)
{
     for (int j = 0; j < cnt; j++){
        int outputname_id = 0;
        for(auto &s : outputname) {
            int outputIndex = engine->getBindingIndex(s.c_str());
            nvinfer1::DataType dtype = engine->getBindingDataType(outputIndex);
            if(!engine->hasImplicitBatchDimension()){
                if (BindDims[outputIndex].nbDims == 1){
                    rlts[j+start][outputname_id].height = 1;
                    rlts[j+start][outputname_id].width = 1;
                    rlts[j+start][outputname_id].channel = 1;

                    std::cout << outputIndex << " BindDims Size: " << BindDims[outputIndex].d[0] << std::endl;
                }
                else if (BindDims[outputIndex].nbDims == 2){
                    rlts[j+start][outputname_id].height = 1;
                    rlts[j+start][outputname_id].width = BindDims[outputIndex].d[1];
                    rlts[j+start][outputname_id].channel = 1;
                    std::cout << outputIndex << " BindDims Size: " << BindDims[outputIndex].d[1] << std::endl;
                }
                else if (BindDims[outputIndex].nbDims == 3){
                    rlts[j+start][outputname_id].height = BindDims[outputIndex].d[1];
                    rlts[j+start][outputname_id].width = BindDims[outputIndex].d[2];
                    rlts[j+start][outputname_id].channel = 1;
                    std::cout << outputIndex << " BindDims Size: " << BindDims[outputIndex].d[1] << ", " << BindDims[outputIndex].d[2] << std::endl;
                }
                else{
                    rlts[j+start][outputname_id].height = BindDims[outputIndex].d[2];
                    rlts[j+start][outputname_id].width = BindDims[outputIndex].d[3];
                    rlts[j+start][outputname_id].channel = BindDims[outputIndex].d[1];
                    std::cout << outputIndex << " BindDims Size: " << BindDims[outputIndex].d[1] << ", "
                            << BindDims[outputIndex].d[2] << ", " << BindDims[outputIndex].d[3] << std::endl;
                }
            }
            else{
                if (BindDims[outputIndex].nbDims == 1){
                    rlts[j+start][outputname_id].height = 1;
                    rlts[j+start][outputname_id].width = BindDims[outputIndex].d[0];
                    rlts[j+start][outputname_id].channel = 1;
                    std::cout << outputIndex << " BindDims Size: " << BindDims[outputIndex].d[0] << std::endl;
                }
                else if (BindDims[outputIndex].nbDims == 2){
                    rlts[j+start][outputname_id].height = BindDims[outputIndex].d[0];
                    rlts[j+start][outputname_id].width = BindDims[outputIndex].d[1];
                    rlts[j+start][outputname_id].channel = 1;
                    std::cout << outputIndex << " BindDims Size: " << BindDims[outputIndex].d[0] << ", " << BindDims[outputIndex].d[1] << std::endl;
                }
                else if (BindDims[outputIndex].nbDims == 3){
                    rlts[j+start][outputname_id].height = BindDims[outputIndex].d[1];
                    rlts[j+start][outputname_id].width = BindDims[outputIndex].d[2];
                    rlts[j+start][outputname_id].channel = BindDims[outputIndex].d[0];
                    std::cout << outputIndex << " BindDims Size: " << BindDims[outputIndex].d[0] << ", "
                            << BindDims[outputIndex].d[1] << ", " << BindDims[outputIndex].d[2] << std::endl;
                }
                else{
                    rlts[j+start][outputname_id].height = BindDims[outputIndex].d[2];
                    rlts[j+start][outputname_id].width = BindDims[outputIndex].d[3];
                    rlts[j+start][outputname_id].channel = BindDims[outputIndex].d[1];
                    std::cout << outputIndex << " BindDims Size: " << BindDims[outputIndex].d[0] << ", " 
                    << BindDims[outputIndex].d[1] << ", " << BindDims[outputIndex].d[2] << ", " << BindDims[outputIndex].d[3] << std::endl;
                }
            }
            auto size = BindBufferSize[outputIndex]/maxbatchSize;
            if (dtype == nvinfer1::DataType::kINT32){
                rlts[j+start][outputname_id].intdata.reset(new int[size/sizeof(int)], array_deleter<int>());
                memcpy(rlts[j+start][outputname_id].intdata.get(), (char*)OutputCpuBuffer[outputname_id] + j * size, size);
            }
            else{
                rlts[j+start][outputname_id].data.reset(new float[size/sizeof(float)], array_deleter<float>());
                memcpy(rlts[j+start][outputname_id].data.get(), (char*)OutputCpuBuffer[outputname_id] + j * size, size);
            }
            outputname_id++;
        }
    }
}

void TensorRTEngine::inference(std::vector<std::vector<cv::Mat>>& imgs, std::vector<IERlt>& rlts)
{
    rlts.resize(imgs.size());
    cudaSetDevice(gpuid);
    const int inputBatch = imgs[0].size();
    std::cout << "imgs batches: " << inputBatch << std::endl;
    
    std::vector<std::vector<cv::Mat>> imgs_(InputCount);
    for(size_t k = 0; k < inputBatch; k++){
        rlts[k].resize(outputname.size());
        for (size_t j = 0; j < InputCount; j++)
        {
            imgs_[j].push_back(imgs[j][k]);
        } 
        if(imgs_[0].size() == maxbatchSize || k == inputBatch -1){
            cvImageTocudaMem(imgs_, InputDims);
            if(!engine->hasImplicitBatchDimension()){
                context->enqueueV2(&CudaBuffer[0], stream, nullptr);
            }
            else{
                context->enqueue(imgs_.size(), &CudaBuffer[0], stream, nullptr);
            }        
            for(int i = 0;i < outputname.size(); i++){
                int outputIndex = engine->getBindingIndex(outputname[i].c_str());
                cudaMemcpyAsync(OutputCpuBuffer[i], CudaBuffer[outputIndex], BindBufferSize[outputIndex]/maxbatchSize*(k%maxbatchSize + 1), cudaMemcpyDeviceToHost, stream);
            }
            fill_batch_result(k/maxbatchSize*maxbatchSize, k%maxbatchSize+1, rlts);
            imgs_.clear();
        }       
    }
    cudaStreamSynchronize(stream);
    std::cout << "inference finished!" << std::endl;
}
