#include "onnxruntimeEngine.h"
#include "ortpathseggpu.h"
#ifndef _WIN32
#include <dirent.h>
#endif

#include <stdio.h>
#include <iostream>

using namespace cv;
using namespace std;

typedef struct {
  int label;
  int area;
} LabelArea;

bool labelareacmp(LabelArea x, LabelArea y) { return x.area > y.area; }

int FilterNoisyArea(Mat& mask, float min_mask_area_ratio, Mat& outmask) {
  // find connectedcomponents
  Mat labels, stats, centroids;
  int num_labels =
      connectedComponentsWithStats(mask, labels, stats, centroids, 8);

  if (num_labels < 1) {
    fprintf(stderr, "no object!\n");
    return ONNXRUNTIMEENGINE_NO_MASK;
  }

  // label in labellist is ordered by area
  std::vector<LabelArea> labellist((int64)num_labels - 1);
  for (int i = 1; i < num_labels; i++) {
    labellist[(int64)i - 1].label = i;
    labellist[(int64)i - 1].area = stats.at<int>(i, CC_STAT_AREA);
  }

  sort(labellist.begin(), labellist.end(), labelareacmp);

  int maskW = mask.cols;
  int maskH = mask.rows;
  Mat image_filtered = Mat::zeros(maskH, maskW, CV_8UC1);

  int min_mask_area = int(min_mask_area_ratio * maskH * maskW);
  // printf("min_mask_area: %d\n", min_mask_area);

  // filter area smaller than min_mask_area
  std::vector<Vec3b> colors(num_labels);
  for (int i = 0; i < num_labels; i++)
    colors[i] = Vec3b(0, 0, 0);

  for (int i = 1; i < num_labels; i++) {
    int label = labellist[(int64)i - 1].label;

    int maskArea = stats.at<int>(label, CC_STAT_AREA);

    // printf("maskArea: %d\n", maskArea);

    if (maskArea > min_mask_area)
      colors[label] = Vec3b(255, 255, 255);
    else
      colors[label] = Vec3b(0, 0, 0);

    // label in labellist is ordered by area
    // so here only process the largest area
    break;
  }

  for (int y = 0; y < image_filtered.rows; y++) {
    for (int x = 0; x < image_filtered.cols; x++) {
      int label = labels.at<int>(y, x);
      // CV_Assert(0 <= label && label <= num_labels);
      if (0 > label || label > num_labels) {
        fprintf(stderr, "label exceed [%d, %d]\n", 0, num_labels);
        return ONNXRUNTIMEENGINE_NO_MASK;
      }

      image_filtered.at<uchar>(y, x) = uchar(colors[label][0]);
    }
  }

  // imwrite("image_filtered.jpg", image_filtered);
  bitwise_and(mask, mask, outmask, mask = image_filtered);

  return ONNXRUNTIMEENGINE_SUCCESS;
}

// out_linemask
int labelLineType(Mat& out_linemask, Mat& out_linetypemask) {
  out_linetypemask = Mat::zeros(out_linemask.rows, out_linemask.cols, CV_8UC1);

  // get contours
  vector<vector<Point>> contours;
  findContours(out_linemask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

  // iterate each contour
  for (int ii = 0; ii < contours.size(); ii++) {
    // find convelHull
    vector<int> hull;
    convexHull(contours[ii], hull, false, false);
    cv::RotatedRect mar = minAreaRect(Mat(contours[ii]));

    // minAreaRect area
    double mar_area = mar.size.area();

    // convexHull area
    vector<Point> points = contours[ii];
    double convex_area = 0;
    for (int i = 0; i < hull.size(); i++) {
      int j = (i + 1) % hull.size();
      convex_area += points[hull[i]].x * points[hull[j]].y -
                     points[hull[j]].x * points[hull[i]].y;
    }
    double cont_area = contourArea(points);

    // check hull points
    if (convex_area / cont_area < 6.0)
      if (mar_area / cont_area < 5.0) {
        // line
        drawContours(out_linetypemask, contours, ii, 128, -1);
      } else {
        // v
        drawContours(out_linetypemask, contours, ii, 255, -1);
      }
  }

  return ONNXRUNTIMEENGINE_SUCCESS;
}

// mask: road border: objlabel: 128->5 road, 25->2 black, 75->1 white, 192->4
// gapline , 255->3 startstop v line
//
int replaceMaskLabel(Mat& mask, Mat& out_newlabelmask) {
  //
  uchar scaleForDebug = 50; // 1;

  out_newlabelmask = Mat::zeros(mask.rows, mask.cols, CV_8UC1);

  // replace
  for (int i = 0; i < mask.rows; i++) {
    for (int j = 0; j < mask.cols; j++) {
      uchar pixel = mask.at<uchar>(i, j);
      if (pixel < 10)
        // background
        out_newlabelmask.at<uchar>(i, j) =
            static_cast<uchar>(0 * scaleForDebug);
      else if (pixel < 50)
        // black line
        out_newlabelmask.at<uchar>(i, j) =
            static_cast<uchar>(2 * scaleForDebug);
      else if (pixel < 100)
        // white line
        out_newlabelmask.at<uchar>(i, j) =
            static_cast<uchar>(1 * scaleForDebug);
      else if (pixel < 150)
        // road
        out_newlabelmask.at<uchar>(i, j) =
            static_cast<uchar>(5 * scaleForDebug);
      else if (pixel < 220)
        // gapline
        out_newlabelmask.at<uchar>(i, j) =
            static_cast<uchar>(4 * scaleForDebug);
      else
        // startstopline
        out_newlabelmask.at<uchar>(i, j) =
            static_cast<uchar>(3 * scaleForDebug);
    }
  }

  return ONNXRUNTIMEENGINE_SUCCESS;
}

#ifdef _WIN32

ortPathSegGPU::ortPathSegGPU() {
  const wchar_t* road_model_path =
      L"C:/Users/Administrator/Desktop/end2end_ocrnet_road_border.onnx";
  const wchar_t* line_model_path =
      L"C:/Users/Administrator/Desktop/end2end_ocrnet_line.onnx";

  // initialize onnxruntime engine
  this->ortengine_road = new OnnxRuntimeEngine(road_model_path);
  this->ortengine_line = new OnnxRuntimeEngine(line_model_path);

  this->outbuffer_road_border = (void*)malloc(512 * 512 * sizeof(int64_t));
  this->outbuffer_line = (void*)malloc(512 * 512 * sizeof(int64_t));
};

#else
ortPathSegGPU::ortPathSegGPU(const std::string& road_onnx_model_path,
                             const std::string& line_onnx_model_path) {
  //  const char* road_model_path =
  //      "/home/nvp/codes/cybath/models/end2end_ocrnet_road_border.onnx";
  //  const char* line_model_path =
  //      "/home/nvp/codes/cybath/models/end2end_ocrnet_line.onnx";
  road_onnx_model_path_ = road_onnx_model_path.c_str();
  line_onnx_model_path_ = line_onnx_model_path.c_str();

  // initialize onnxruntime engine
  this->ortengine_road = new OnnxRuntimeEngine(road_onnx_model_path_);
  this->ortengine_line = new OnnxRuntimeEngine(line_onnx_model_path_);

  this->outbuffer_road_border = (void*)malloc(512 * 512 * sizeof(int64_t));
  this->outbuffer_line = (void*)malloc(512 * 512 * sizeof(int64_t));
};
#endif

ortPathSegGPU::~ortPathSegGPU() {
  free(outbuffer_road_border);
  free(outbuffer_line);

  delete (ortengine_road);
  delete (ortengine_line);
}

int ortPathSegGPU::processMask(cv::Mat src, cv::Mat& finalmask) {

  int res;
  float min_mask_area_ratio = 0.03;
  clock_t startTime, endTime;

  cv::Mat final_roadmask, final_blackmask, final_whitemask, final_linemask,
      out_roadmask, out_linemask, out_linetypemask, newlabel_finalmask;
  memset(this->outbuffer_road_border, 0, 512 * 512 * sizeof(int64_t));
  memset(this->outbuffer_line, 0, 512 * 512 * sizeof(int64_t));
  int padw, padh;

  if (src.empty()) // check image is valid
  {
    fprintf(stderr, "src image is empty\n");
    return ONNXRUNTIMEENGINE_FILE_ERROR;
  }

  startTime = clock();

  // road + border
  res = ortengine_road->processSeg(src, Size(512, 512), padw, padh,
                                   this->outbuffer_road_border);

  if (res != ONNXRUNTIMEENGINE_SUCCESS) {
    fprintf(stderr, "processSeg error\n");
    return res;
  }

  // road border: objlabel: 1 road, 2 black, 3 white
  // road line: 1 line
  res = ortengine_road->segmentationPost(this->outbuffer_road_border,
                                         final_roadmask, 512, 512, padw, padh,
                                         src.rows, src.cols, 1);

  if (res != ONNXRUNTIMEENGINE_SUCCESS) {
    fprintf(stderr, "processSeg error\n");
    return res;
  }

  // remove small areas in road segmentation result
  FilterNoisyArea(final_roadmask, min_mask_area_ratio, out_roadmask);

  // imwrite("./out_roadmask.jpg", out_roadmask);

  res = ortengine_road->segmentationPost(this->outbuffer_road_border,
                                         final_blackmask, 512, 512, padw, padh,
                                         src.rows, src.cols, 2);

  if (res != ONNXRUNTIMEENGINE_SUCCESS) {
    fprintf(stderr, "processSeg error\n");
    return res;
  }

  // imwrite("./out_blackmask.jpg", final_blackmask);

  res = ortengine_road->segmentationPost(this->outbuffer_road_border,
                                         final_whitemask, 512, 512, padw, padh,
                                         src.rows, src.cols, 3);

  if (res != ONNXRUNTIMEENGINE_SUCCESS) {
    fprintf(stderr, "processSeg error\n");
    return res;
  }

  // imwrite("./out_whitemask.jpg", final_whitemask);

  // line
  res = ortengine_line->processSeg(src, Size(512, 512), padw, padh,
                                   this->outbuffer_line);

  if (res != ONNXRUNTIMEENGINE_SUCCESS) {
    fprintf(stderr, "processSeg error\n");
    return res;
  }

  res = ortengine_road->segmentationPost(this->outbuffer_line, final_linemask,
                                         512, 512, padw, padh, src.rows,
                                         src.cols, 1);

  if (res != ONNXRUNTIMEENGINE_SUCCESS) {
    fprintf(stderr, "processSeg error\n");
    return res;
  }

  // add final_linemask
  bitwise_and(final_linemask, final_linemask, out_linemask, out_roadmask);

  // label out_linemask to startstopline and gapline
  res = labelLineType(out_linemask, out_linetypemask);
  if (res != ONNXRUNTIMEENGINE_SUCCESS) {
    fprintf(stderr, "processSeg error\n");
    return res;
  }

  cv::addWeighted(out_roadmask, 0.5, out_linetypemask, 0.5, 0, finalmask);
  cv::addWeighted(finalmask, 1.0, final_blackmask, 0.1, 0, finalmask);
  cv::addWeighted(finalmask, 1.0, final_whitemask, 0.3, 0, finalmask);

  // replace label
  res = replaceMaskLabel(finalmask, newlabel_finalmask);
  if (res != ONNXRUNTIMEENGINE_SUCCESS) {
    fprintf(stderr, "processSeg error\n");
    return res;
  }

  // imwrite("./newlabel_finalmask.jpg", newlabel_finalmask);

  finalmask = newlabel_finalmask;

  endTime = clock();
  // cout << i << ": " << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s"
  // << endl;

  return ONNXRUNTIMEENGINE_SUCCESS;
}