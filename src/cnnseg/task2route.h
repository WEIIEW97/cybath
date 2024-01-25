#include "onnxruntimeEngine.h"

class RouteTask {

public:
#ifdef _WIN32
  RouteTask(const wchar_t* road_model_path, const wchar_t* border_model_path,
            const wchar_t* line_model_path);
#else
  RouteTask(const char* road_model_path, const char* border_model_path,
            const char* line_model_path);
#endif

  ~RouteTask();

  int processMask(cv::Mat src, cv::Mat& finalmask, bool& hasShoe,
                  std::vector<int>& shoePos, cv::Mat& visimg);

private:
  OnnxRuntimeEngine* ortengine_road;
  OnnxRuntimeEngine* ortengine_border;
  OnnxRuntimeEngine* ortengine_line;

  OnnxRuntimeEngine* ortengine_shoe;

  void* outbuffer_road;
  void* outbuffer_border;
  void* outbuffer_line;
};

int mainRouteTask();