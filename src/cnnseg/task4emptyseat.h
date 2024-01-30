#include "onnxruntimeEngine.h"
#include "task1forest.h"
#include <vector>

class EmptyseatTask {

public:
#ifdef _WIN32
  EmptyseatTask(const wchar_t* coco_model_path,
                const wchar_t* furnitures_model_path,
                const wchar_t* border_model_path);
#else
  EmptyseatTask(const char* coco_model_path, const char* furnitures_model_path,
                const char* border_model_path);
#endif

  ~EmptyseatTask();

  int findChair(cv::Mat src, bool available[6], bool& hasCabinet,
                std::vector<int>& position, cv::Mat& bordermask,
                cv::Mat& visimg);

private:
  OnnxRuntimeEngine* ortengine_coco;
  OnnxRuntimeEngine* ortengine_furn;

  // initialize onnxruntime engine
  ForestTask* foresttask;
};