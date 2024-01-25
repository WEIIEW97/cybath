#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/dnn.hpp>
#include <vector>
#include "fastdeploy/vision.h"
#include "onnxruntimeEngine.h"

#define PADLEOCRENGINE_SUCCESS                 0
#define PADLEOCRENGINE_NO_STAT                 1
#define PADLEOCRENGINE_FILE_ERROR              2
#define PADLEOCRENGINE_PATH_ERROR              3
#define PADLEOCRENGINE_DET_INIT_ERROR          4
#define PADLEOCRENGINE_CLS_INIT_ERROR          5
#define PADLEOCRENGINE_REC_INIT_ERROR          6
#define PADLEOCRENGINE_INFERENGINE_INIT_ERROR  7
#define PADLEOCRENGINE_PREDICT_ERROR           8
#define PADLEOCRENGINE_ORDER_NOT_FOUND_ERROR   9
#define PADLEOCRENGINE_ORDER_NOT_IN_MENU_ERROR 10
#define PADLEOCRENGINE_NO_VALID_TEXTBOX_ERROR  11

#define DEBUG_ENABLE 1

class TouchscreenTask {
public:
  TouchscreenTask(const char* fingertip_model_path,
                  const std::string det_model_dir,
                  const std::string cls_model_dir,
                  const std::string rec_model_dir,
                  const std::string rec_label_file);

  ~TouchscreenTask();

  int doOCR(cv::Mat& src, cv::Mat& vis_im, std::string& orderstr,
            std::string& targetstr, std::array<int, 8>& targetbox,
            bool& findFingertip, BoundingBox& fingertip);

private:
  OnnxRuntimeEngine* ortengine_fingertip;

  fastdeploy::vision::ocr::DBDetector* det_model;
  fastdeploy::vision::ocr::Classifier* cls_model;
  fastdeploy::vision::ocr::Recognizer* rec_model;
  fastdeploy::pipeline::PPOCRv3* ppocr_v3;
  // std::vector<std::string> menutitems =
  // {"hamburger","cake","sausages","waffles","tomatosoup","chicken","spaghetti","pizza","steak","yogurt","croissant","fruitsalad","apple","mashedpotatoes","bread","lasagna","banana","pancakes","baconandeggs","icecream","broccoll","cupcake","salad","sandwich","carrot"};
};

int mainTouchscreenTask();