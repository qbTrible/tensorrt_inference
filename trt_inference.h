#ifndef INCLUDE_TRT_INFERENCE_H_
#define INCLUDE_TRT_INFERENCE_H_

#include <unistd.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <NvInferPlugin.h>

#ifdef USE_CCA
#include "cudaconvertion.h"
#endif

using namespace std;

#define _CS(str) (str).c_str()  

static bool check_file(const string& file)
{
    int ret = 0;
#ifdef _MSC_VER
    ret = _access(_CS(file),0);
#else
    ret = access(_CS(file),F_OK); // F_OK ???????
#endif
    return ret != -1;
}

template< typename T >
struct array_deleter
{
  void operator ()( T const * p)
  { 
    delete[] p; 
  }
};
// 单个结果
typedef struct inference_result {
    int height;
    int width;
    int channel;
    std::shared_ptr<float> data;
    std::shared_ptr<int> intdata;

public:
    int elemCount() {
        int count = 1;
        if (height > 0) {
            count *= height;
        }
        if (width > 0) {
            count *= width;
        }
        if(channel > 0) {
            count *= channel;
        }
        return count;
    }

    void copyToFloatVector(std::vector<float> &probs) {
        float *attr = data.get();
        int *intattr;
        if (attr == NULL) {
            intattr = intdata.get();
            for (int i = 0; i < elemCount(); i++) {
                probs.push_back((float)intattr[i]);
            }
        }
        else {
            for (int i = 0; i < elemCount(); i++) {
                probs.push_back(attr[i]);
            }
        }
    }
} InferRlt;
// 多个输出结果
typedef std::vector<InferRlt> IERlt;

//1：创建.logger
class Logger : public  nvinfer1::ILogger
{
public:
	void log(Severity severity,const char* msg) noexcept override
	{
		if(severity != Severity::kINFO)
			std::cout << msg << std::endl;	
	}
};

class RuntimeWarp {
public:
    RuntimeWarp() {
        runtime = nvinfer1::createInferRuntime(gLogger);
        //initLibNvInferPlugins(&gLogger, "");
    }
    ~RuntimeWarp() {
        if(runtime)
            delete runtime;
            // runtime->destroy();
    }
public:
    Logger gLogger;
    nvinfer1::IRuntime *runtime;
};
static RuntimeWarp rt;

/**
 * @brief 生成 nvinfer1::ICudaEngine
 * 
 */
class EngineWarp {
public:
    EngineWarp(const std::string& model_path) {
        std::string model;
        if (decode_model(model_path, model)) {
            initLibNvInferPlugins(nullptr, "");
            engine = rt.runtime->deserializeCudaEngine((void*)model.data(), model.size());

            for(int i = 0; i < engine->getNbBindings(); i++) {
                auto dim = engine->getBindingDimensions(i);
                std::string dim_str = "[";
                for (int j = 0; j < dim.nbDims; j++) {
                    dim_str += std::to_string(dim.d[j]) + ",";
                }
                dim_str.pop_back();
                dim_str += "]";
                inout_infos += (std::to_string(i) + "_" + (engine->bindingIsInput(i) ? "Input": "Output") + "_" + engine->getBindingName(i) + "_" + dim_str +";");
            }
            
        }
    }
    ~EngineWarp() {
        delete engine;
        // engine->destroy();
    }
    nvinfer1::ICudaEngine* engine_get(std::string& engine_id) {
        return engine;
    }

    /**
     * @brief 校验外部指定的输出与网络的输出是否一致
     *      TensorRT 引擎在运动阶段无法动态调整输出层
     * 
     * @param outputname 
     * @return true 
     * @return false 
     */
    bool checkOutputs(const std::string& model_path, const std::vector<std::string> &outputname) {
        if(engine) {
            std::cout << "TensorRT Engine [" << engine->getName() << "]: " << model_path << " has " << engine->getNbBindings() << " inputs and outputs;"
                << "\n\tmax_batch_size = " << engine->getMaxBatchSize() << "; implicit_batch = " << engine->hasImplicitBatchDimension()
                << "\n\t" << inout_infos << std::endl;
            
            for(auto n : outputname) {
                int idx = engine->getBindingIndex(n.c_str());
                if(idx < 0) {
                    std::cout << "TensorRT Engine [" << engine->getName() << "] Output Layer Not Found: " << n << std::endl;
                }
            }
        }
        else {
            std::cout << "model load failured" << std::endl;
        }
    }

public:
    //PluginFactory pluginFactory;
    nvinfer1::ICudaEngine *engine;
    std::string inout_infos;

private:
    bool decode_model(const std::string& model_path, std::string& model){
        std::ifstream planFile(model_path);
        if (!planFile.is_open())
        {
            std::cout << "Can't Open Model File:" << model_path << std::endl;
            return false;
        }
        std::stringstream planBuffer;
        planBuffer << planFile.rdbuf();
        model = planBuffer.str();
        return true;
    }
};

/**
 * @brief nvinfer1::ICudaEngine 对象管理，防止在同一张 GPU 卡上创建多个相同的引擎
 *  即每张卡同样的引擎只保留一个; TODO: 这样是否合理呢?
 * 
 */
class TensorRTEngineHandle {
public:
    typedef std::map<std::string, std::shared_ptr<EngineWarp> > tensorRTEngineMap;
    static tensorRTEngineMap& registery(){
        static tensorRTEngineMap* tensorrt_engine_register = new tensorRTEngineMap();
        return *(tensorrt_engine_register);
    }
    
    static nvinfer1::ICudaEngine* get_engine(const std::string& model_path,
                                             const std::vector<std::string> &outputname,
                                             const std::string &deviceForInference,
                                             int maxBatch
                                             )
    {
        static int model_cnt = 0;
        tensorRTEngineMap& registe = registery();
        // build engine id: 一个模型一个engine
        std::string engine_id = deviceForInference;
        size_t name_pos = model_path.rfind('/');
        engine_id += "-" + model_path.substr(name_pos + 1);
        if(!check_file(model_path)) {
            return NULL;
        }
        if(registe.count(engine_id) == 0) {
#ifdef DEV_STAGE
            std::cout << "Load new engine: " << engine_id << std::endl;
#endif
            registe[engine_id].reset(new EngineWarp(model_path));
            registe[engine_id]->checkOutputs(model_path, outputname);
        }
        //std::cout << engine_id << ", Engine Count: " << registe.size() << ", Model Count: " << ++model_cnt << std::endl;;
        return registe[engine_id]->engine;
    }
};

class TensorRTEngine
{
public:
    
    TensorRTEngine(const std::string &modelFile,
                      const std::vector<std::string> &_outputname,
                      const std::string &deviceForInference,
                      int maxBatch);
    virtual void inference(std::vector<std::vector<cv::Mat>>& imgs, std::vector<IERlt>& rlts);
    virtual void SetRGB(int _RGBmodel);
    virtual void SetMean(float b, float g, float r);
    virtual void SetScale(float b, float g, float r);

    std::vector<float>& GetMean() {return image_mean;}
    std::vector<float>& GetScale() {return image_scale;}
    int getBindingIndex(const char* name) {return engine->getBindingIndex(name);}
    std::vector<std::string>& getOutputname() {return outputname;}
    virtual ~TensorRTEngine();

private:
    void cvImageTocudaMem(std::vector<std::vector<cv::Mat>>& imgs_, std::vector<nvinfer1::Dims> dimensions);
    void fill_batch_result(int start, int cnt, std::vector<IERlt>& rlts);

private:
//    nvinfer1::IRuntime *runtime;
    nvinfer1::ICudaEngine *engine;
    nvinfer1::IExecutionContext *context;
    size_t maxbatchSize {0};
    size_t InputCount {0};
    size_t gpuid {0};
    std::vector<std::string> outputname;
    std::vector<float> image_mean = {0, 0, 0};
    std::vector<float> image_scale = {1, 1, 1};
    int RGBmodel {0};
    
public:
    std::vector<nvinfer1::Dims> BindDims;
    std::vector<int64_t> BindBufferSize;
    bool RGBScaleSame = true;
    // float *inputDataHost;
    std::vector<void*> CudaBuffer;
    std::vector<void*> OutputCpuBuffer;
    std::vector<nvinfer1::Dims> InputDims;
    cudaStream_t stream;
#ifdef USE_CCA
    typedef struct GPUImg {
    void *data;
    int width;
    int height;
    int channel;
    } GPUImg;

    std::vector<GPUImg> img_gpu_data8u;
    std::vector<GPUImg> img_gpu_data32f;
#endif
};

#endif // INCLUDE_TRT_INFERENCE_H_
