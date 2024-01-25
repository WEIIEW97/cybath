#pragma once

#ifndef ONNXRUNTIMEENGINE_H
#define ONNXRUNTIMEENGINE_H

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <cassert>
#include <vector>
#include <fstream>
#include <numeric>
#include <string>

#define ONNXRUNTIMEENGINE_SUCCESS    0
#define ONNXRUNTIMEENGINE_NO_STAT    1
#define ONNXRUNTIMEENGINE_FILE_ERROR 2
#define ONNXRUNTIMEENGINE_PATH_ERROR 3
#define ONNXRUNTIMEENGINE_NO_MASK    4

using namespace cv;

typedef struct {
  float x1;
  float y1;
  float x2;
  float y2;
  float score;
  long long label; // int64
} BoundingBox;

class OnnxRuntimeEngine {
public:
#ifdef _WIN32
  OnnxRuntimeEngine(const wchar_t* model_path);
#else
  OnnxRuntimeEngine(const char* model_path);
#endif

  OrtApi* getOrtApi() { return g_ort_rt; }
  OrtSession* getOrtSession() { return session; }
  OrtAllocator* getOrtAllocator() { return allocator; }
  int processSeg(Mat& img, Size targetSize, int& padw, int& padh,
                 void* output_buffer);
  int segmentationPost(void* output_buffer, Mat& mask, int exrows, int excols,
                       int expadw, int expadh, int rawrows, int rawcols,
                       int64_t objlabel = 1);

  int processDet(Mat& img, Size targetSize, long long maxclassid,
                 float threshold, std::vector<BoundingBox>& outboxes);

  int drawboxsave(Mat& src, int x1, int y1, int x2, int y2,
                  cv::Scalar color = cv::Scalar(0, 0, 255));

private:
  OrtApi* g_ort_rt;
  OrtSession* session;
  OrtAllocator* allocator;

  size_t num_input_nodes;
  std::vector<const char*> input_node_names;
  std::vector<std::vector<int64_t>> input_node_dims;
  std::vector<ONNXTensorElementDataType> input_types;
  std::vector<OrtValue*> input_tensors;
  size_t num_output_nodes;
  std::vector<const char*> output_node_names;
  std::vector<std::vector<int64_t>> output_node_dims;
  std::vector<OrtValue*> output_tensors;

  bool CheckStatus(const OrtApi* g_ort, OrtStatus* status);
  int paddingResize(cv::InputArray& src, cv::OutputArray& dst, float& fratio,
                    int& pad_w, int& pad_h, cv::Size size, double fx = 0,
                    double fy = 0, int interpolation = cv::INTER_LINEAR,
                    cv::Scalar padval = cv::Scalar(0, 0, 0));
  int unpaddingResize(Mat& src, Mat& dst, int pad_w, int pad_h, int raw_w,
                      int raw_h, double fx = 0, double fy = 0,
                      int interpolation = cv::INTER_LINEAR);
  int preProcess(Mat& image, Mat& image_blob);

  int classificationPost(void* output_buffer, size_t output_data_size);
  int detectionPost(void* dets_buffer, void* labels_buffer,
                    std::vector<BoundingBox>& outboxes, float fratio,
                    int outboxnum, long long maxclassid, int rawcols,
                    int rawrows, float threshold = 0.5);
};

int FilterNoisyArea(Mat& mask, float min_mask_area_ratio, Mat& outmask);

#endif // ONNXRUNTIMEENGINE_H