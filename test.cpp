// test
#include <opencv2/opencv.hpp>
#include "trt_inference.h"


#include <limits>

int main()
{
    std::string model_file = "/home/linux/project/quanbo/onnx2trt/dynamic_resnet50.trt";
    std::vector<std::string> outputname = {"output"};
    std::string device = "0";
    int maxBatch = 2;
    TensorRTEngine *trt_engine = new TensorRTEngine(model_file, outputname, device, maxBatch);

    std::vector<std::vector<cv::Mat>> imgs(1);
    cv::Mat img = cv::imread("../cat.jpg");
    std::cout << "width: " << img.cols << ", height: " << img.rows << std::endl;
    imgs[0].push_back(img);
    trt_engine->SetMean(123.675, 116.28, 103.53);
    trt_engine->SetScale(0.017124, 0.017507, 0.0174292);
    std::vector<IERlt> rlts;
    trt_engine->inference(imgs, rlts);
    float *data = rlts[0][0].data.get();
    int max_index = 0;
    float max_value = FLT_MIN;
    for (int i=0; i<1000; i++)
    {
        float value = data[i];
        if (value > max_value)
        {
            max_value = value;
            max_index = i;
        }
    }
    std::cout << "max_index: " << max_index << ", max_value: " << max_value << std::endl;

    delete trt_engine;
}