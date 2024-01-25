#ifndef FORESTTASKDEFINITION

#define FORESTTASKDEFINITION

#include "onnxruntimeEngine.h"

class ForestTask {
public:
#ifdef _WIN32
  ForestTask(const wchar_t* border_model_path);
#else
  explicit ForestTask(const char* border_model_path);
#endif

  ~ForestTask();

  int processMask(cv::Mat src, cv::Mat& finalmask);

private:
  OnnxRuntimeEngine* ortengine_border;

  void* outbuffer_border;
};

int mainForestTask();

#endif // FORESTTASKDEFINITION