#include "onnxruntimeEngine.h"
#include "task1forest.h"
#include <vector>

class TabletnaviTask {

public:
#ifdef _WIN32
  TabletnaviTask(const wchar_t* tablet_model_path,
                 const wchar_t* border_model_path);
#else
  TabletnaviTask(const char* tablet_model_path, const char* border_model_path);
#endif

  ~TabletnaviTask();

  int findObject(cv::Mat src, bool& hasCabinet, std::vector<int>& cabinPos,
                 bool& hasTab, std::vector<int>& tabPos, cv::Mat& bordermask,
                 cv::Mat& visimg);

private:
  OnnxRuntimeEngine* ortengine_tablet;

  // initialize onnxruntime engine
  ForestTask* foresttask;
};

int mainTabletnaviTask();