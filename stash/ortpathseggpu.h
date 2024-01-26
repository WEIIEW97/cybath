#include "../src/cnnseg/onnxruntimeEngine.h"

class ortPathSegGPU {
public:
  const char* road_onnx_model_path_;
  const char* line_onnx_model_path_;

public:
  ortPathSegGPU(const std::string& road_onnx_model_path,
                const std::string& line_onnx_model_path);
  ~ortPathSegGPU();
  int processMask(cv::Mat src, cv::Mat& finalmask);

private:
  OnnxRuntimeEngine* ortengine_road;
  OnnxRuntimeEngine* ortengine_line;
  void* outbuffer_road_border;
  void* outbuffer_line;
};