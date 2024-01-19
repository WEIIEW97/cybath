#include "onnxruntimeEngine.h"


class ortPathSegGPU {


    public:
        ortPathSegGPU();

        ~ortPathSegGPU();

        int processMask(cv::Mat src, cv::Mat& finalmask);
    
    private:
        OnnxRuntimeEngine* ortengine_road;
        OnnxRuntimeEngine* ortengine_line;

        void *outbuffer_road_border;
        void* outbuffer_line;
};